from __future__ import print_function, division
import argparse
import torch
import os, sys
import numpy as np
import joblib
import smplx
import trimesh
import h5py

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.smplify import SMPLify3D
from src.config import *

# parsing argument
parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=1, help="input batch size")
parser.add_argument("--num_smplify_iters", type=int, default=100, help="num of smplify iters")
parser.add_argument("--cuda", action="store_true", help="enables cuda (default: CPU)")
parser.add_argument("--gpu_ids", type=int, default=0, help="choose gpu ids")
parser.add_argument("--num_joints", type=int, default=22, help="joint number")
parser.add_argument("--joint_category", type=str, default="AMASS", help="use correspondence")
parser.add_argument("--fix_foot", type=str, default="False", help="fix foot or not")
parser.add_argument("--data_folder", type=str, default="./demo/demo_data/", help="data in the folder")
parser.add_argument(
    "--save_folder",
    type=str,
    default="./demo/demo_results/",
    help="results save folder",
)
parser.add_argument("--files", type=str, default="test_motion.npz", help="files use")
opt = parser.parse_args()
print(opt)

# Load predefined models
device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")
print(SMPL_MODEL_DIR)
smplmodel = smplx.create(
    SMPL_MODEL_DIR,
    model_type="smpl",
    gender="neutral",
    ext="pkl",
    batch_size=opt.batchSize,
).to(device)

# Load mean pose
smpl_mean_file = SMPL_MEAN_FILE
file = h5py.File(smpl_mean_file, "r")
init_mean_pose = torch.from_numpy(file["pose"][:]).unsqueeze(0).float()
init_mean_shape = torch.from_numpy(file["shape"][:]).unsqueeze(0).float()
cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).to(device)

# Initialize tensors
pred_pose = torch.zeros(opt.batchSize, 72).to(device)
pred_betas = torch.zeros(opt.batchSize, 10).to(device)
pred_cam_t = torch.zeros(opt.batchSize, 3).to(device)
keypoints_3d = torch.zeros(opt.batchSize, opt.num_joints, 3).to(device)

# Initialize SMPLify
smplify = SMPLify3D(
    smplxmodel=smplmodel,
    batch_size=opt.batchSize,
    joints_category=opt.joint_category,
    num_iters=opt.num_smplify_iters,
    device=device,
)

# Load data
purename = os.path.splitext(opt.files)[0]
full_path = os.path.join(opt.data_folder, opt.files)

npz_data = np.load(full_path)
data = npz_data["joints"]  # (T, 22, 3) - root-centered
pelvis_traj = npz_data["pelvis_traj"]  # (T, 3) - world-space trajectory
fps = float(npz_data["fps"])

print(f"✓ Loaded: {data.shape[0]} frames at {fps} FPS")
print(f"  Pelvis trajectory: {pelvis_traj[0]} → {pelvis_traj[-1]}")
print(f"  Distance: {np.linalg.norm(np.diff(pelvis_traj, axis=0), axis=1).sum():.2f}m\n")

# Create output directory
dir_save = os.path.join(opt.save_folder, purename)
os.makedirs(dir_save, exist_ok=True)

# Process each frame
num_seqs = data.shape[0]

for idx in range(num_seqs):
    print(f"Frame {idx}/{num_seqs-1}")

    # Get root-centered joints and add world-space translation
    joints3d_centered = data[idx]  # (22, 3) - pelvis at origin
    world_root = pelvis_traj[idx]  # (3,) - world position
    joints3d_world = joints3d_centered + world_root

    keypoints_3d[0, :, :] = torch.Tensor(joints3d_world).to(device).float()

    # Initialize or load previous frame parameters
    if idx == 0:
        pred_betas[0, :] = init_mean_shape
        pred_pose[0, :] = init_mean_pose
        pred_cam_t[0, :] = torch.Tensor(world_root).to(device)
    else:
        prev_param = joblib.load(os.path.join(dir_save, f"{idx-1:04d}.pkl"))
        pred_betas[0, :] = torch.from_numpy(prev_param["beta"]).float()
        pred_pose[0, :] = torch.from_numpy(prev_param["pose"]).float()
        pred_cam_t[0, :] = torch.Tensor(world_root).to(device)

    # Joint confidence weights
    confidence_input = torch.ones(opt.num_joints)
    if opt.fix_foot == "True":
        confidence_input[7] = 1.5  # Left ankle
        confidence_input[8] = 1.5  # Right ankle
        confidence_input[10] = 1.5  # Left foot
        confidence_input[11] = 1.5  # Right foot

    # Fit SMPL to joints
    (
        new_opt_vertices,
        new_opt_joints,
        new_opt_pose,
        new_opt_betas,
        new_opt_cam_t,
        new_opt_joint_loss,
    ) = smplify(
        pred_pose.detach(),
        pred_betas.detach(),
        pred_cam_t.detach(),
        keypoints_3d,
        conf_3d=confidence_input.to(device),
        seq_ind=idx,
    )

    # Generate and save mesh
    outputp = smplmodel(
        betas=new_opt_betas,
        global_orient=new_opt_pose[:, :3],
        body_pose=new_opt_pose[:, 3:],
        transl=new_opt_cam_t,
        return_verts=True,
    )
    mesh = trimesh.Trimesh(
        vertices=outputp.vertices.detach().cpu().numpy().squeeze(),
        faces=smplmodel.faces,
        process=False,
    )
    mesh.export(os.path.join(dir_save, f"{idx:04d}.ply"))

    # Save parameters
    param = {
        "beta": new_opt_betas.detach().cpu().numpy(),
        "pose": new_opt_pose.detach().cpu().numpy(),
        "cam": new_opt_cam_t.detach().cpu().numpy(),
        "root": new_opt_cam_t.detach().cpu().numpy().squeeze(),
    }
    joblib.dump(param, os.path.join(dir_save, f"{idx:04d}.pkl"), compress=3)

print(f"\n✓ Saved {num_seqs} frames to {dir_save}")
