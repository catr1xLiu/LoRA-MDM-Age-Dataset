#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, math, argparse, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d


def vicon_to_smpl_coords(points, vicon_up="Z", vicon_forward="Y"):
    if vicon_up == "Z" and vicon_forward == "Y":
        rot = R.from_euler("x", -90, degrees=True).as_matrix()
    elif vicon_up == "Z" and vicon_forward == "X":
        rot = R.from_euler("xy", [-90, -90], degrees=True).as_matrix()
    elif vicon_up == "Y" and vicon_forward == "Z":
        rot = np.eye(3)
    elif vicon_up == "Y" and vicon_forward == "X":
        rot = R.from_euler("y", 90, degrees=True).as_matrix()
    else:
        raise ValueError(f"Unsupported Vicon convention: up={vicon_up}, forward={vicon_forward}")

    original_shape = points.shape
    points_flat = points.reshape(-1, 3)
    transformed = (rot @ points_flat.T).T
    return transformed.reshape(original_shape)


SMPL_JOINTS = [
    "pelvis",
    "L_hip",
    "R_hip",
    "spine1",
    "L_knee",
    "R_knee",
    "spine2",
    "L_ankle",
    "R_ankle",
    "spine3",
    "L_foot",
    "R_foot",
    "neck",
    "L_collar",
    "R_collar",
    "head",
    "L_shoulder",
    "R_shoulder",
    "L_elbow",
    "R_elbow",
    "L_wrist",
    "R_wrist",
    "L_hand",
    "R_hand",
]
J = {n: i for i, n in enumerate(SMPL_JOINTS)}

MARKER_TO_JOINT = {
    "LFHD": J["head"],
    "RFHD": J["head"],
    "LBHD": J["head"],
    "RBHD": J["head"],
    "C7": J["neck"],
    "T10": J["spine2"],
    "CLAV": J["spine3"],
    "STRN": J["spine3"],
    "LASI": J["pelvis"],
    "RASI": J["pelvis"],
    "SACR": J["pelvis"],
    "LSHO": J["L_shoulder"],
    "RSHO": J["R_shoulder"],
    "LELB": J["L_elbow"],
    "RELB": J["R_elbow"],
    "LWRA": J["L_wrist"],
    "LWRB": J["L_wrist"],
    "RWRA": J["R_wrist"],
    "RWRB": J["R_wrist"],
    "LKNE": J["L_knee"],
    "RKNE": J["R_knee"],
    "LANK": J["L_ankle"],
    "RANK": J["R_ankle"],
    "LHEE": J["L_ankle"],
    "RHEE": J["R_ankle"],
    "LTOE": J["L_foot"],
    "RTOE": J["R_foot"],
    "LMT5": J["L_foot"],
    "RMT5": J["R_foot"],
}


def rotation_matrix_to_axis_angle(R_mat):
    eps = 1e-8
    cos = (np.trace(R_mat) - 1.0) * 0.5
    cos = np.clip(cos, -1.0, 1.0)
    angle = np.arccos(cos)

    if angle < 1e-6:
        return np.zeros(3, dtype=np.float32)

    rx = R_mat[2, 1] - R_mat[1, 2]
    ry = R_mat[0, 2] - R_mat[2, 0]
    rz = R_mat[1, 0] - R_mat[0, 1]
    axis = np.array([rx, ry, rz], dtype=np.float64)
    axis /= 2.0 * np.sin(angle) + eps

    return (axis * angle).astype(np.float32)


def procrustes_align(source, target):
    """Rigid Kabsch alignment (rotation + translation, no scale)."""
    valid = ~np.isnan(target).any(axis=1)
    if valid.sum() < 3:
        return np.eye(3), np.zeros(3)

    src = source[valid]
    tgt = target[valid]

    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean

    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T

    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T

    t = tgt_mean - (R_mat @ src_mean)

    return R_mat, t


def linear_interp_nans(arr):
    x = arr.copy()
    T = x.shape[0]
    for d in range(3):
        v = x[:, d]
        nans = np.isnan(v)
        if nans.all():
            x[:, d] = 0.0
        elif nans.any():
            idx = np.arange(T)
            v[nans] = np.interp(idx[nans], idx[~nans], v[~nans])
            x[:, d] = v
    return x


def estimate_travel_direction(transl, fps=100.0):
    T = transl.shape[0]
    velocity = np.zeros_like(transl)
    velocity[:-1] = transl[1:] - transl[:-1]
    velocity[-1] = velocity[-2]

    for d in range(3):
        velocity[:, d] = gaussian_filter1d(velocity[:, d], sigma=5)

    velocity_xz = velocity.copy()
    velocity_xz[:, 1] = 0

    speed_per_frame = np.linalg.norm(velocity_xz, axis=1)
    speed_mps = speed_per_frame * fps
    norms = np.maximum(speed_per_frame, 1e-6)
    direction = velocity_xz / norms[:, np.newaxis]

    slow_mask = speed_mps < 0.05
    if not slow_mask.all():
        avg_direction = direction[~slow_mask].mean(axis=0)
        avg_direction /= np.linalg.norm(avg_direction) + 1e-6
        direction[slow_mask] = avg_direction

    return direction, speed_mps


def smooth_poses_gaussian(poses, sigma=2):
    smoothed = poses.copy()
    for d in range(poses.shape[1]):
        smoothed[:, d] = gaussian_filter1d(poses[:, d], sigma=sigma)
    return smoothed


def detect_outlier_frames(poses, threshold=2.0):
    T = poses.shape[0]
    if T < 3:
        return []

    diffs = np.linalg.norm(poses[1:] - poses[:-1], axis=1)

    median_diff = np.median(diffs)
    mad = np.median(np.abs(diffs - median_diff))

    threshold_val = median_diff + threshold * mad * 1.4826

    outlier_indices = []
    for i in range(len(diffs)):
        if diffs[i] > threshold_val:
            outlier_indices.append(i + 1)

    if T > 20:
        reference_pose = poses[10:20].mean(axis=0)
        for i in range(min(5, T)):
            dist = np.linalg.norm(poses[i] - reference_pose)
            if dist > threshold_val * 2:
                if i not in outlier_indices:
                    outlier_indices.append(i)

    return sorted(outlier_indices)


def sanitize(name: str):
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def load_markers_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    M = d["marker_data"]
    names = [str(x) for x in d["marker_names"].tolist()]
    fps = float(d["frame_rate"])
    return M, names, fps


def subset_markers(M, names):
    keep_indices = []
    joint_ids = []
    used_names = []

    for i, nm in enumerate(names):
        key = nm.strip().upper()
        if key in MARKER_TO_JOINT:
            keep_indices.append(i)
            joint_ids.append(MARKER_TO_JOINT[key])
            used_names.append(key)

    if len(keep_indices) < 12:
        raise RuntimeError(f"Too few markers found ({len(keep_indices)})")

    return M[:, keep_indices, :], used_names, np.array(joint_ids, dtype=np.int64)


def prep_markers(M_sub, drop_thresh=0.6):
    T, K, _ = M_sub.shape
    valid = ~np.isnan(M_sub).any(axis=2)
    keep_frames = valid.sum(axis=1) >= math.ceil(drop_thresh * K)
    M2 = M_sub[keep_frames].copy()

    for k in range(M2.shape[1]):
        M2[:, k, :] = linear_interp_nans(M2[:, k, :])

    return M2, keep_frames


def decimate_indices(T, src_fps, dst_fps):
    if src_fps == dst_fps:
        return np.arange(T)
    step = src_fps / dst_fps
    return np.clip(np.round(np.arange(0, T, step)).astype(int), 0, T - 1)


class SMPLFitterV3(nn.Module):

    def __init__(self, smpl_layer, T, device="cpu", init_betas=None):
        super().__init__()
        self.smpl = smpl_layer
        self.device = device
        self.T = T

        self.global_orient = nn.Parameter(torch.zeros(T, 3, device=device))
        self.transl = nn.Parameter(torch.zeros(T, 3, device=device))
        self.body_pose = nn.Parameter(torch.zeros(T, 69, device=device))

        betas_init = (
            torch.zeros(10, device=device) if init_betas is None else torch.as_tensor(init_betas, device=device).float()
        )
        self.betas = nn.Parameter(betas_init)

    def forward(self):
        out = self.smpl(
            global_orient=self.global_orient,
            body_pose=self.body_pose,
            betas=self.betas.unsqueeze(0).expand(self.T, -1),
            transl=self.transl,
            pose2rot=True,
        )

        joints = out.joints[:, :24, :]
        return joints


def build_smpl(model_dir: str, gender: str, device: str):
    import smplx

    smpl = smplx.create(model_path=model_dir, model_type="smpl", gender=gender, use_pca=False, num_betas=10).to(device)
    smpl.eval()
    for p in smpl.parameters():
        p.requires_grad = False
    return smpl


def estimate_initial_pose(markers_np, used_names, smpl):
    from collections import defaultdict

    device = next(smpl.parameters()).device
    with torch.no_grad():
        out = smpl(
            global_orient=torch.zeros(1, 3, device=device),
            body_pose=torch.zeros(1, 69, device=device),
            betas=torch.zeros(1, 10, device=device),
            transl=torch.zeros(1, 3, device=device),
        )
        tpose_joints = out.joints[0].cpu().numpy()

    T = markers_np.shape[0]
    name2idx = {n: i for i, n in enumerate(used_names)}

    marker_joint_map = {
        "LASI": J["pelvis"],
        "RASI": J["pelvis"],
        "SACR": J["pelvis"],
        "C7": J["neck"],
        "CLAV": J["spine3"],
        "LSHO": J["L_shoulder"],
        "RSHO": J["R_shoulder"],
        "LELB": J["L_elbow"],
        "RELB": J["R_elbow"],
        "LKNE": J["L_knee"],
        "RKNE": J["R_knee"],
        "LANK": J["L_ankle"],
        "RANK": J["R_ankle"],
    }

    # Group markers by their target joint to avoid overweighting joints with multiple markers
    joint_to_marker_idxs = defaultdict(list)
    for m, jid in marker_joint_map.items():
        if m in name2idx:
            joint_to_marker_idxs[jid].append(name2idx[m])

    joint_ids = sorted(joint_to_marker_idxs.keys())
    if len(joint_ids) < 6:
        pelvis_markers = [m for m in ["LASI", "RASI", "SACR"] if m in name2idx]
        if pelvis_markers:
            pelvis_pos = np.mean([markers_np[:, name2idx[m], :] for m in pelvis_markers], axis=0)
        else:
            pelvis_pos = np.mean(markers_np, axis=1)
        return np.zeros(3, dtype=np.float32), pelvis_pos.astype(np.float32)

    mid_frame = min(T // 2, T - 1)
    # Each joint appears only once; markers mapping to the same joint are averaged
    src_pts = np.array([tpose_joints[jid] for jid in joint_ids])
    tgt_pts = np.array([markers_np[mid_frame, joint_to_marker_idxs[jid], :].mean(axis=0) for jid in joint_ids])

    R_init, t_init = procrustes_align(src_pts, tgt_pts)
    aa_init = rotation_matrix_to_axis_angle(R_init)

    transl_init = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        tgt_pts_t = np.array([markers_np[t, joint_to_marker_idxs[jid], :].mean(axis=0) for jid in joint_ids])
        _, t_t = procrustes_align(src_pts, tgt_pts_t)
        transl_init[t] = t_t

    return aa_init.astype(np.float32), transl_init.astype(np.float32)


def fit_sequence_single_pass(
    fitter,
    markers_torch,
    targets_idx,
    travel_direction_torch,
    travel_speed_torch,
    device,
    iters,
    w_marker,
    w_pose,
    w_betas,
    w_transl_vel,
    w_transl_acc,
    w_pose_vel,
    w_pose_acc,
    w_spine_vel,
    w_spine_acc,
    w_foot_orient,
    w_foot_smooth,
    verbose=True,
):
    T = markers_torch.shape[0]

    opt = Adam(
        [
            {"params": [fitter.global_orient, fitter.transl], "lr": 3e-2},
            {"params": [fitter.body_pose], "lr": 2e-2},
            {"params": [fitter.betas], "lr": 1e-3},
        ]
    )

    huber = nn.SmoothL1Loss(beta=0.01, reduction="none")

    spine_joint_ids_body = [J["spine1"] - 1, J["spine2"] - 1, J["spine3"] - 1, J["neck"] - 1]
    spine_pose_indices = []
    for jid in spine_joint_ids_body:
        spine_pose_indices.extend([jid * 3, jid * 3 + 1, jid * 3 + 2])
    spine_pose_indices = torch.tensor(spine_pose_indices, device=device, dtype=torch.long)

    foot_joint_ids_body = [J["L_ankle"] - 1, J["R_ankle"] - 1, J["L_foot"] - 1, J["R_foot"] - 1]
    foot_pose_indices = []
    for jid in foot_joint_ids_body:
        foot_pose_indices.extend([jid * 3, jid * 3 + 1, jid * 3 + 2])
    foot_pose_indices = torch.tensor(foot_pose_indices, device=device, dtype=torch.long)

    for it in range(iters):
        opt.zero_grad()

        joints = fitter()
        pred_markers = joints[:, targets_idx, :]

        per_elem = huber(pred_markers, markers_torch)
        marker_loss = per_elem.mean()

        pose_l2 = (fitter.body_pose**2).mean()
        betas_l2 = (fitter.betas**2).mean()

        transl_vel = (fitter.transl[1:] - fitter.transl[:-1]).pow(2).mean()
        transl_acc = torch.tensor(0.0, device=device)
        if T > 2:
            transl_acc = (fitter.transl[2:] - 2 * fitter.transl[1:-1] + fitter.transl[:-2]).pow(2).mean()

        pose_vel = (fitter.body_pose[1:] - fitter.body_pose[:-1]).pow(2).mean()
        orient_vel = (fitter.global_orient[1:] - fitter.global_orient[:-1]).pow(2).mean()

        pose_acc = torch.tensor(0.0, device=device)
        orient_acc = torch.tensor(0.0, device=device)
        if T > 2:
            pose_acc = (fitter.body_pose[2:] - 2 * fitter.body_pose[1:-1] + fitter.body_pose[:-2]).pow(2).mean()
            orient_acc = (
                (fitter.global_orient[2:] - 2 * fitter.global_orient[1:-1] + fitter.global_orient[:-2]).pow(2).mean()
            )

        spine_pose = fitter.body_pose[:, spine_pose_indices]
        spine_vel = (spine_pose[1:] - spine_pose[:-1]).pow(2).mean()
        spine_acc = torch.tensor(0.0, device=device)
        if T > 2:
            spine_acc = (spine_pose[2:] - 2 * spine_pose[1:-1] + spine_pose[:-2]).pow(2).mean()

        foot_pose = fitter.body_pose[:, foot_pose_indices]
        foot_vel = (foot_pose[1:] - foot_pose[:-1]).pow(2).mean()
        foot_acc = torch.tensor(0.0, device=device)
        if T > 2:
            foot_acc = (foot_pose[2:] - 2 * foot_pose[1:-1] + foot_pose[:-2]).pow(2).mean()

        l_ankle_pos = joints[:, J["L_ankle"], :]
        r_ankle_pos = joints[:, J["R_ankle"], :]
        l_foot_pos = joints[:, J["L_foot"], :]
        r_foot_pos = joints[:, J["R_foot"], :]

        l_foot_dir = l_foot_pos - l_ankle_pos
        r_foot_dir = r_foot_pos - r_ankle_pos

        l_foot_dir_xz = l_foot_dir.clone()
        l_foot_dir_xz[:, 1] = 0
        r_foot_dir_xz = r_foot_dir.clone()
        r_foot_dir_xz[:, 1] = 0

        l_foot_dir_xz = l_foot_dir_xz / (torch.norm(l_foot_dir_xz, dim=1, keepdim=True) + 1e-6)
        r_foot_dir_xz = r_foot_dir_xz / (torch.norm(r_foot_dir_xz, dim=1, keepdim=True) + 1e-6)

        l_dot = (l_foot_dir_xz * travel_direction_torch).sum(dim=1)
        r_dot = (r_foot_dir_xz * travel_direction_torch).sum(dim=1)

        l_hip_pos = joints[:, J["L_hip"], :]
        r_hip_pos = joints[:, J["R_hip"], :]
        pelvis_pos = joints[:, J["pelvis"], :]
        spine1_pos = joints[:, J["spine1"], :]

        hip_lateral = r_hip_pos - l_hip_pos
        hip_lateral[:, 1] = 0
        spine_up = spine1_pos - pelvis_pos

        body_forward = torch.cross(hip_lateral, spine_up, dim=1)
        body_forward[:, 1] = 0
        body_forward = body_forward / (torch.norm(body_forward, dim=1, keepdim=True) + 1e-6)

        body_travel_dot = (body_forward * travel_direction_torch).sum(dim=1)

        movement_mask = travel_speed_torch > 0.05
        forward_facing_mask = body_travel_dot > 0.5
        apply_foot_prior_mask = movement_mask & forward_facing_mask

        foot_orient_loss = torch.tensor(0.0, device=device)
        if apply_foot_prior_mask.any():
            l_orient_loss = ((1 - l_dot[apply_foot_prior_mask]) ** 2).mean()
            r_orient_loss = ((1 - r_dot[apply_foot_prior_mask]) ** 2).mean()
            foot_orient_loss = (l_orient_loss + r_orient_loss) / 2

        loss = (
            w_marker * marker_loss
            + w_pose * pose_l2
            + w_betas * betas_l2
            + w_transl_vel * transl_vel
            + w_transl_acc * transl_acc
            + w_pose_vel * (pose_vel + orient_vel)
            + w_pose_acc * (pose_acc + orient_acc)
            + w_spine_vel * spine_vel
            + w_spine_acc * spine_acc
            + w_foot_smooth * (foot_vel + foot_acc)
            + w_foot_orient * foot_orient_loss
        )

        loss.backward()
        opt.step()

        if verbose and ((it + 1) % 100 == 0 or it == 0):
            print(f"    iter {it+1:03d}: loss={loss.item():.5f}  marker={marker_loss.item():.5f}")

    return marker_loss.item()


def fit_sequence(
    smpl,
    markers_torch,
    marker_joint_idx,
    used_names,
    init_betas,
    device,
    iters=400,
    fps=100.0,
    w_marker=1.0,
    w_pose=1e-3,
    w_betas=5e-4,
    w_transl_vel=1e-2,
    w_transl_acc=1e-3,
    w_pose_vel=0.02,
    w_pose_acc=0.005,
    w_spine_vel=0.05,
    w_spine_acc=0.01,
    w_foot_orient=0.01,
    w_foot_smooth=0.03,
    two_pass=True,
    smooth_sigma=1.5,
):
    T, K, _ = markers_torch.shape

    fitter = SMPLFitterV3(smpl, T, device, init_betas).to(device)

    mk_np = markers_torch.detach().cpu().numpy()
    aa_init, transl_init = estimate_initial_pose(mk_np, used_names, smpl)

    fitter.global_orient.data[:] = torch.from_numpy(np.tile(aa_init[None, :], (T, 1))).to(device)
    fitter.transl.data[:] = torch.from_numpy(transl_init).to(device)

    travel_direction, travel_speed = estimate_travel_direction(transl_init, fps)
    travel_direction_torch = torch.from_numpy(travel_direction).float().to(device)
    travel_speed_torch = torch.from_numpy(travel_speed).float().to(device)

    def update_travel_from_current_transl():
        transl_now = fitter.transl.detach().cpu().numpy()
        td, sp = estimate_travel_direction(transl_now, fps)
        return (torch.from_numpy(td).float().to(device), torch.from_numpy(sp).float().to(device))

    targets_idx = torch.as_tensor(marker_joint_idx, device=device, dtype=torch.long)

    print("  Pass 1: Initial fitting...")
    fit_sequence_single_pass(
        fitter,
        markers_torch,
        targets_idx,
        travel_direction_torch,
        travel_speed_torch,
        device,
        iters,
        w_marker,
        w_pose,
        w_betas,
        w_transl_vel,
        w_transl_acc,
        w_pose_vel,
        w_pose_acc,
        w_spine_vel,
        w_spine_acc,
        w_foot_orient,
        w_foot_smooth,
        verbose=True,
    )

    travel_direction_torch, travel_speed_torch = update_travel_from_current_transl()

    if two_pass and T > 10:
        body_pose_np = fitter.body_pose.detach().cpu().numpy()
        global_orient_np = fitter.global_orient.detach().cpu().numpy()
        full_pose = np.concatenate([global_orient_np, body_pose_np], axis=1)

        outlier_frames = detect_outlier_frames(full_pose, threshold=2.5)

        edge_frames = list(range(min(5, T))) + list(range(max(0, T - 5), T))
        frames_to_check = sorted(set(outlier_frames + edge_frames))

        if frames_to_check:
            print(f"  Pass 2: Re-fitting frames {frames_to_check[:10]}{'...' if len(frames_to_check) > 10 else ''}")

            mid = T // 2
            reference_frame = mid
            while reference_frame in outlier_frames and reference_frame < T - 1:
                reference_frame += 1
            if reference_frame in outlier_frames:
                reference_frame = mid
                while reference_frame in outlier_frames and reference_frame > 0:
                    reference_frame -= 1

            with torch.no_grad():
                ref_pose = fitter.body_pose[reference_frame].clone()
                ref_orient = fitter.global_orient[reference_frame].clone()

                for frame_idx in frames_to_check:
                    if frame_idx == reference_frame:
                        continue

                    dist = abs(frame_idx - reference_frame)

                    if frame_idx < 5:
                        blend = 0.9
                    else:
                        blend = max(0.3, 1.0 - dist / 20.0)

                    fitter.body_pose.data[frame_idx] = blend * ref_pose + (1 - blend) * fitter.body_pose.data[frame_idx]
                    fitter.global_orient.data[frame_idx] = (
                        blend * ref_orient + (1 - blend) * fitter.global_orient.data[frame_idx]
                    )

            fit_sequence_single_pass(
                fitter,
                markers_torch,
                targets_idx,
                travel_direction_torch,
                travel_speed_torch,
                device,
                iters // 2,
                w_marker,
                w_pose,
                w_betas,
                w_transl_vel * 2,
                w_transl_acc * 2,
                w_pose_vel * 3,
                w_pose_acc * 3,
                w_spine_vel * 2,
                w_spine_acc * 2,
                w_foot_orient,
                w_foot_smooth * 2,
                verbose=True,
            )

            travel_direction_torch, travel_speed_torch = update_travel_from_current_transl()

    if smooth_sigma > 0:
        with torch.no_grad():
            body_pose_np = fitter.body_pose.detach().cpu().numpy()
            global_orient_np = fitter.global_orient.detach().cpu().numpy()
            transl_np = fitter.transl.detach().cpu().numpy()

            body_pose_smooth = smooth_poses_gaussian(body_pose_np, sigma=smooth_sigma)
            global_orient_smooth = smooth_poses_gaussian(global_orient_np, sigma=smooth_sigma)
            transl_smooth = smooth_poses_gaussian(transl_np, sigma=smooth_sigma)

            fitter.body_pose.data[:] = torch.from_numpy(body_pose_smooth).to(device)
            fitter.global_orient.data[:] = torch.from_numpy(global_orient_smooth).to(device)
            fitter.transl.data[:] = torch.from_numpy(transl_smooth).to(device)

    print("  Final pass: Refinement...")
    final_marker_loss = fit_sequence_single_pass(
        fitter,
        markers_torch,
        targets_idx,
        travel_direction_torch,
        travel_speed_torch,
        device,
        iters // 4,
        w_marker * 2,
        w_pose,
        w_betas,
        w_transl_vel,
        w_transl_acc,
        w_pose_vel,
        w_pose_acc,
        w_spine_vel,
        w_spine_acc,
        w_foot_orient,
        w_foot_smooth,
        verbose=True,
    )

    with torch.no_grad():
        joints = fitter()

    result = {
        "poses": fitter.body_pose.detach().cpu().numpy(),
        "global_orient": fitter.global_orient.detach().cpu().numpy(),
        "trans": fitter.transl.detach().cpu().numpy(),
        "betas": fitter.betas.detach().cpu().numpy(),
        "joints": joints.detach().cpu().numpy(),
        "final_marker_loss": final_marker_loss,
    }

    return result


def pick_gender(subject_meta):
    g = subject_meta.get("gender", "neutral").lower()
    return g if g in ("male", "female") else "neutral"


def subject_cache_paths(out_root, subject_id):
    subj_dir = Path(out_root) / subject_id
    subj_dir.mkdir(parents=True, exist_ok=True)
    return subj_dir, subj_dir / "betas.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed_dir", required=True, help="Path to processed markers (typically data/processed_markers_all_2/)"
    )
    ap.add_argument(
        "--models_dir",
        required=True,
        help="Path to SMPL models directory (typically data/smpl/ - see DATA.MD for download)",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for fitted SMPL parameters (typically data/fitted_smpl_all_3/)",
    )
    ap.add_argument("--subject", default=None)
    ap.add_argument("--trial", default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--vicon_up", default="Z", choices=["X", "Y", "Z"])
    ap.add_argument("--vicon_forward", default="Y", choices=["X", "Y", "Z"])
    ap.add_argument("--w_pose_vel", type=float, default=0.02)
    ap.add_argument("--w_pose_acc", type=float, default=0.005)
    ap.add_argument("--w_spine_vel", type=float, default=0.05)
    ap.add_argument("--w_spine_acc", type=float, default=0.01)
    ap.add_argument("--w_foot_orient", type=float, default=0.01)
    ap.add_argument("--w_foot_smooth", type=float, default=0.03)
    ap.add_argument("--no_two_pass", action="store_true")
    ap.add_argument("--smooth_sigma", type=float, default=1.5)
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    subjects = [args.subject] if args.subject else sorted([p.name for p in processed.iterdir() if p.is_dir()])
    print(f"Found {len(subjects)} subjects")

    for subj in subjects:
        subj_dir = processed / subj
        npz_files = sorted(glob.glob(str(subj_dir / "*_markers_positions.npz")))

        if not npz_files:
            print(f"[{subj}] no marker files found — skipping")
            continue

        subj_out_dir, betas_path = subject_cache_paths(out_root, subj)

        meta_files = sorted(glob.glob(str(subj_dir / "*_metadata.json")))
        gender = "neutral"
        if meta_files:
            with open(meta_files[0], "r") as f:
                subj_meta = json.load(f)
            gender = pick_gender(subj_meta)
        print(f"[{subj}] gender={gender}")

        device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
        smpl = build_smpl(args.models_dir, gender, device)

        init_betas = None
        if betas_path.exists():
            init_betas = np.load(betas_path)
            print(f"[{subj}] loaded cached betas")

        if args.trial:
            npz_files = [p for p in npz_files if Path(p).stem.startswith(args.trial)]
            if not npz_files:
                print(f"[{subj}] trial {args.trial} not found")
                continue

        if init_betas is None:
            first_npz = npz_files[0]
            M, names, fps = load_markers_npz(first_npz)
            M = vicon_to_smpl_coords(M, args.vicon_up, args.vicon_forward)
            M_sub, used_names, marker_ids = subset_markers(M, names)
            M_sub, keep = prep_markers(M_sub)
            T = M_sub.shape[0]

            if T > 220:
                sel = decimate_indices(T, src_fps=fps, dst_fps=min(fps, 20))
                M_sub = M_sub[sel]

            markers_torch = torch.from_numpy(M_sub).float().to(device)
            print(f"[{subj}] fitting betas on {markers_torch.shape[0]} frames")

            result = fit_sequence(
                smpl,
                markers_torch,
                marker_ids,
                used_names,
                init_betas=None,
                device=device,
                fps=fps,
                iters=min(args.iters * 2, 1000),
                w_pose_vel=args.w_pose_vel,
                w_pose_acc=args.w_pose_acc,
                w_spine_vel=args.w_spine_vel,
                w_spine_acc=args.w_spine_acc,
                w_foot_orient=args.w_foot_orient,
                w_foot_smooth=args.w_foot_smooth,
                two_pass=not args.no_two_pass,
                smooth_sigma=args.smooth_sigma,
            )

            np.save(betas_path, result["betas"])
            init_betas = result["betas"]
            print(f"[{subj}] saved betas")

        for npz_path in npz_files:
            trial_name = Path(npz_path).stem.replace("_markers_positions", "")
            trial_safe = sanitize(trial_name)
            out_trial = subj_out_dir / f"{trial_safe}_smpl_params.npz"
            report_path = subj_out_dir / f"{trial_safe}_smpl_metadata.json"

            if out_trial.exists():
                print(f"[{subj}] {trial_name}: already fitted")
                continue

            M, names, fps = load_markers_npz(npz_path)
            M = vicon_to_smpl_coords(M, args.vicon_up, args.vicon_forward)
            M_sub, used_names, marker_ids = subset_markers(M, names)
            M_sub, keep = prep_markers(M_sub)
            markers_torch = torch.from_numpy(M_sub).float().to(device)

            print(f"[{subj}] fitting {trial_name}: frames={M_sub.shape[0]} markers={len(used_names)}")

            result = fit_sequence(
                smpl,
                markers_torch,
                marker_ids,
                used_names,
                init_betas=init_betas,
                device=device,
                fps=fps,
                iters=args.iters,
                w_pose_vel=args.w_pose_vel,
                w_pose_acc=args.w_pose_acc,
                w_spine_vel=args.w_spine_vel,
                w_spine_acc=args.w_spine_acc,
                w_foot_orient=args.w_foot_orient,
                w_foot_smooth=args.w_foot_smooth,
                two_pass=not args.no_two_pass,
                smooth_sigma=args.smooth_sigma,
            )

            poses72 = np.concatenate([result["global_orient"], result["poses"]], axis=1)
            np.savez(
                out_trial,
                poses=poses72,
                trans=result["trans"],
                betas=result["betas"],
                gender=gender,
                subject_id=subj,
                trial_name=trial_name,
                fps=fps,
                n_frames=result["trans"].shape[0],
                joints=result["joints"],
            )

            report = {
                "subject_id": subj,
                "trial_name": trial_name,
                "gender": gender,
                "frames_kept": int(markers_torch.shape[0]),
                "fps": fps,
                "markers_used": used_names,
                "final_marker_loss": result["final_marker_loss"],
                "settings": {
                    "vicon_up": args.vicon_up,
                    "vicon_forward": args.vicon_forward,
                    "two_pass": not args.no_two_pass,
                    "smooth_sigma": args.smooth_sigma,
                },
            }
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            print(f"[{subj}] wrote {out_trial.name}")


if __name__ == "__main__":
    main()
