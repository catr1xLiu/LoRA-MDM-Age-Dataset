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
parser.add_argument("--batch_size", type=int, default=32, help="input batch size")
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
parser.add_argument("--files", type=str, default="test_motion.npy", help="files use")
opt = parser.parse_args()
print(opt)


def _to_time_joint_xyz(motion, num_joints):
    motion = np.asarray(motion)

    if motion.ndim == 3:
        if motion.shape[1:] == (num_joints, 3):
            return motion.astype(np.float32)
        if motion.shape[:2] == (num_joints, 3):
            return np.transpose(motion, (2, 0, 1)).astype(np.float32)
        raise ValueError(f"Unsupported 3D motion shape: {motion.shape}")

    if motion.ndim == 4:
        if motion.shape[1:3] == (num_joints, 3):  # (N, 22, 3, T)
            return np.transpose(motion[0], (2, 0, 1)).astype(np.float32)
        if motion.shape[2:] == (num_joints, 3):  # (N, T, 22, 3)
            return motion[0].astype(np.float32)
        raise ValueError(f"Unsupported 4D motion shape: {motion.shape}")

    raise ValueError(f"Unsupported motion rank: {motion.ndim} for shape {motion.shape}")


def load_motion_file(full_path, num_joints):
    loaded = np.load(full_path, allow_pickle=True)

    data = None
    pelvis_traj = None
    fps = 20.0
    source_fmt = None

    if isinstance(loaded, np.lib.npyio.NpzFile):
        source_fmt = "npz"
        if "joints" in loaded.files:
            data = np.asarray(loaded["joints"], dtype=np.float32)
            pelvis_traj = np.asarray(loaded["pelvis_traj"], dtype=np.float32) if "pelvis_traj" in loaded.files else None
            if "fps" in loaded.files:
                fps = float(loaded["fps"])
        elif "motion" in loaded.files:
            data = _to_time_joint_xyz(loaded["motion"], num_joints)
            if "fps" in loaded.files:
                fps = float(loaded["fps"])
        else:
            raise ValueError(f"NPZ missing supported keys. Found: {list(loaded.files)}")
        loaded.close()

    elif isinstance(loaded, np.ndarray):
        if loaded.dtype == object:
            source_fmt = "pickle-npy"
            payload = loaded.item() if loaded.shape == () else loaded.flat[0]
            if not isinstance(payload, dict):
                raise ValueError(f"Pickle npy payload must be dict, got {type(payload)}")
            if "motion" not in payload:
                raise ValueError(f"Pickle dict missing 'motion'. Found keys: {list(payload.keys())}")

            data = _to_time_joint_xyz(payload["motion"], num_joints)

            lengths = payload.get("lengths")
            if lengths is not None:
                seq_len = int(np.asarray(lengths).reshape(-1)[0])
                data = data[:seq_len]

            if "pelvis_traj" in payload:
                pelvis_traj_raw = np.asarray(payload["pelvis_traj"], dtype=np.float32)
                if pelvis_traj_raw.ndim == 3:
                    pelvis_traj = pelvis_traj_raw[0]
                elif pelvis_traj_raw.ndim == 2:
                    pelvis_traj = pelvis_traj_raw

            if "fps" in payload:
                fps = float(payload["fps"])

        else:
            source_fmt = "plain-npy"
            data = _to_time_joint_xyz(loaded, num_joints)

    else:
        raise ValueError(f"Unsupported loaded file type: {type(loaded)}")

    if pelvis_traj is None:
        pelvis_traj = np.zeros((data.shape[0], 3), dtype=np.float32)
        print("Warning: No pelvis trajectory found. Defaulting to zeros.")

    pelvis_traj = np.asarray(pelvis_traj, dtype=np.float32)
    if pelvis_traj.shape[0] < data.shape[0]:
        pad = np.repeat(pelvis_traj[-1:], data.shape[0] - pelvis_traj.shape[0], axis=0)
        pelvis_traj = np.concatenate([pelvis_traj, pad], axis=0)
    elif pelvis_traj.shape[0] > data.shape[0]:
        pelvis_traj = pelvis_traj[: data.shape[0]]

    if pelvis_traj.ndim != 2 or pelvis_traj.shape[1] != 3:
        raise ValueError(f"Invalid pelvis_traj shape: {pelvis_traj.shape}")

    return data, pelvis_traj, fps, source_fmt


# Load predefined models
device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")
print(SMPL_MODEL_DIR)
smplmodel = smplx.create(
    SMPL_MODEL_DIR,
    model_type="smpl",
    gender="neutral",
    ext="pkl",
    batch_size=1,
).to(device)

# Load mean pose
smpl_mean_file = SMPL_MEAN_FILE
file = h5py.File(smpl_mean_file, "r")
init_mean_pose = torch.from_numpy(file["pose"][:]).unsqueeze(0).float()
init_mean_shape = torch.from_numpy(file["shape"][:]).unsqueeze(0).float()

# Initialize SMPLify
smplify = SMPLify3D(
    smplxmodel=smplmodel,
    batch_size=opt.batch_size,
    joints_category=opt.joint_category,
    num_iters=opt.num_smplify_iters,
    use_lbfgs=True,
    device=device,
)

# Load data
purename = os.path.splitext(opt.files)[0]
full_path = os.path.join(opt.data_folder, opt.files)

data, pelvis_traj, fps, source_fmt = load_motion_file(full_path, opt.num_joints)

print(f"✓ Loaded ({source_fmt}): {data.shape[0]} frames at {fps} FPS")
print(f"  Pelvis trajectory: {pelvis_traj[0]} → {pelvis_traj[-1]}")
print(f"  Distance: {np.linalg.norm(np.diff(pelvis_traj, axis=0), axis=1).sum():.2f}m\n")

# Create output directory
dir_save = os.path.join(opt.save_folder, purename)
os.makedirs(dir_save, exist_ok=True)

# Joint confidence weights (constant across all frames)
confidence_input = torch.ones(opt.num_joints)
if opt.fix_foot == "True":
    confidence_input[7] = 1.5  # Left ankle
    confidence_input[8] = 1.5  # Right ankle
    confidence_input[10] = 1.5  # Left foot
    confidence_input[11] = 1.5  # Right foot
conf_batch = confidence_input.to(device)

# Process frames in batches
num_seqs = data.shape[0]
prev_last_pose = None
prev_last_betas = None

for batch_start in range(0, num_seqs, opt.batch_size):
    batch_end = min(batch_start + opt.batch_size, num_seqs)
    actual_batch_size = batch_end - batch_start
    batch_idx = batch_start // opt.batch_size
    print(f"Batch {batch_idx}: frames {batch_start}–{batch_end - 1}")

    # Build batched joint tensor [B, num_joints, 3]
    j3d_batch = torch.stack(
        [torch.tensor(data[i] + pelvis_traj[i], dtype=torch.float32) for i in range(batch_start, batch_end)]
    ).to(device)

    # Initialize pose/betas from previous batch's last frame, or mean on first batch
    if batch_start == 0:
        init_pose_batch = init_mean_pose.expand(actual_batch_size, -1).clone().to(device)
        init_betas_batch = init_mean_shape.expand(actual_batch_size, -1).clone().to(device)
    else:
        assert prev_last_pose is not None and prev_last_betas is not None
        init_pose_batch = prev_last_pose.expand(actual_batch_size, -1).clone()
        init_betas_batch = prev_last_betas.expand(actual_batch_size, -1).clone()

    init_cam_t_batch = torch.tensor(pelvis_traj[batch_start:batch_end], dtype=torch.float32).to(device)  # [B, 3]

    # Phase 1: parallel initial fit across the batch
    (new_vertices, new_joints, new_pose, new_betas, new_cam_t, _) = smplify(
        init_pose_batch,
        init_betas_batch,
        init_cam_t_batch,
        j3d_batch,
        conf_3d=conf_batch,
        seq_ind=0 if batch_start == 0 else 1,
    )

    # Phase 2: temporal refinement — anchor first frame of batch to last frame of previous batch
    PHASE2_ENABLED = True
    if PHASE2_ENABLED and batch_start > 0:
        preserve_for_phase2 = new_pose[:, 3:].clone()  # [B, 69] body pose
        if prev_last_pose is not None:
            preserve_for_phase2[0] = prev_last_pose[0, 3:]

        (new_vertices, new_joints, new_pose, new_betas, new_cam_t, _) = smplify(
            new_pose,
            new_betas,
            new_cam_t,
            j3d_batch,
            conf_3d=conf_batch,
            seq_ind=1,
            num_iters_override=opt.num_smplify_iters // 5,
            preserve_pose_override=preserve_for_phase2,
            pose_preserve_weight=50.0 if prev_last_pose is not None else 0.0,
        )

    prev_last_pose = new_pose[-1:].detach()
    prev_last_betas = new_betas[-1:].detach()

    # Save per-frame results
    for i, frame_idx in enumerate(range(batch_start, batch_end)):
        outputp = smplmodel(
            betas=new_betas[i : i + 1],
            global_orient=new_pose[i : i + 1, :3],
            body_pose=new_pose[i : i + 1, 3:],
            transl=new_cam_t[i : i + 1],
            return_verts=True,
        )
        mesh = trimesh.Trimesh(
            vertices=outputp.vertices.detach().cpu().numpy().squeeze(),
            faces=smplmodel.faces,
            process=False,
        )
        mesh.export(os.path.join(dir_save, f"{frame_idx:04d}.ply"))

        param = {
            "beta": new_betas[i : i + 1].detach().cpu().numpy(),
            "pose": new_pose[i : i + 1].detach().cpu().numpy(),
            "cam": new_cam_t[i : i + 1].detach().cpu().numpy(),
            "root": new_cam_t[i : i + 1].detach().cpu().numpy().squeeze(),
        }
        joblib.dump(param, os.path.join(dir_save, f"{frame_idx:04d}.pkl"), compress=3)

print(f"\n✓ Saved {num_seqs} frames to {dir_save}")
