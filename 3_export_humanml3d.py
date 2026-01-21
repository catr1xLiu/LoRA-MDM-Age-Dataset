#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert fitted SMPL params to HumanML3D-style 22 SMPL joints at 20 FPS.
- Select first 22 SMPL joints
- Canonicalize: root-centered; rotate so Up=+Y and Forward=+Z (estimated)
- Resample to 20 FPS
"""
import os, glob, json, argparse
from pathlib import Path
import numpy as np
import torch

# same 24-joint order as fitter
SMPL_24 = [
    "pelvis","L_hip","R_hip","spine1","L_knee","R_knee","spine2","L_ankle","R_ankle",
    "spine3","L_foot","R_foot","neck","L_collar","R_collar","head",
    "L_shoulder","R_shoulder","L_elbow","R_elbow","L_wrist","R_wrist","L_hand","R_hand"
]

def sanitize(s: str) -> str:
    return s.replace(" ", "_").replace("(", "").replace(")", "")

def decimate_indices(T, src_fps, dst_fps):
    if src_fps == dst_fps: return np.arange(T)
    step = src_fps / dst_fps
    return np.clip(np.round(np.arange(0, T, step)).astype(int), 0, T-1)

def estimate_axes(j):  # (T,24,3)
    # Up: pelvis->neck averaged direction
    up = j[:,12,:] - j[:,0,:]     # neck - pelvis
    up = up / (np.linalg.norm(up, axis=1, keepdims=True)+1e-8)
    # Left-right across hips (R_hip->L_hip)
    lr = j[:,1,:] - j[:,2,:]
    lr = lr / (np.linalg.norm(lr, axis=1, keepdims=True)+1e-8)
    # Forward = Up x LeftRight  (right-handed)
    fwd = np.cross(up, lr)
    fwd = fwd / (np.linalg.norm(fwd, axis=1, keepdims=True)+1e-8)
    # Orthonormalize
    lr = np.cross(fwd, up)
    lr = lr / (np.linalg.norm(lr, axis=1, keepdims=True)+1e-8)
    return up, fwd, lr

def rot_to_align(u_src, f_src):
    # Build mean frame
    u = u_src.mean(axis=0); u /= (np.linalg.norm(u)+1e-8)
    f = f_src.mean(axis=0); f /= (np.linalg.norm(f)+1e-8)
    l = np.cross(f, u); l /= (np.linalg.norm(l)+1e-8)
    R_src = np.stack([l, u, f], axis=1)  # columns: X(lr), Y(up), Z(fwd)

    # Target frame: +Y up, +Z fwd, +X left-right
    R_tgt = np.eye(3)
    # Columns are [X,Y,Z] = [lr, up, fwd] with +Y up and +Z fwd already

    # Rotation mapping src->tgt: R = R_tgt * R_src^T
    R = R_tgt @ R_src.T
    return R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits_dir", required=True,
                    help="Path to fitted SMPL parameters (typically data/fitted_smpl_all_3/)")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for HumanML3D format joints (typically data/humanml3d_joints_4/)")
    ap.add_argument("--dst_fps", type=int, default=20)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--trial", default=None)
    args = ap.parse_args()

    fits = Path(args.fits_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    subj_dirs = [fits/args.subject] if args.subject else sorted([p for p in fits.iterdir() if p.is_dir()])

    for subj_dir in subj_dirs:
        subj = subj_dir.name
        out_subj = out_root / subj
        out_subj.mkdir(parents=True, exist_ok=True)

        npz_files = sorted(subj_dir.glob("*_smpl_params.npz"))
        if args.trial:
            npz_files = [p for p in npz_files if p.stem.startswith(args.trial)]
        if not npz_files: 
            print(f"[{subj}] no fitted trials"); 
            continue

        for fpath in npz_files:
            d = np.load(fpath, allow_pickle=True)
            joints = d["joints"]        # (T,24,3)
            fps = float(d["fps"])
            trial = str(d["trial_name"]) if "trial_name" in d else fpath.stem.replace("_smpl_params","")
            # estimate orientation & align
            up, fwd, lr = estimate_axes(joints)
            R = rot_to_align(up, fwd)     # (3,3)
            j = joints @ R.T              # (T,24,3)
            # root-center
            j = j - j[:,[0],:]            # pelvis at origin
            # select first 22 joints (HumanML3D uses first 22 SMPL joints)
            j22 = j[:, :22, :]            # (T,22,3)

            # resample to 20 FPS
            idx = decimate_indices(j22.shape[0], src_fps=fps, dst_fps=args.dst_fps)
            j22_20 = j22[idx]

            # pelvis trajectory in the canonical frame (before root-centering)
            pelvis_world = joints[:, 0, :]              # original coords, before rotation / centering
            pelvis_world_canon = pelvis_world @ R.T
            pelvis_20 = pelvis_world_canon[idx]

            # pull per-trial metadata written by the fitter
            meta_path = fpath.with_name(
                fpath.stem.replace("_smpl_params", "_smpl_metadata") + ".json"
            )
            age = sex = height_m = mass_kg = None
            if meta_path.exists():
                with open(meta_path, "r") as fh:
                    meta = json.load(fh)
                age = meta.get("age")
                sex = meta.get("gender")
                height_m = meta.get("height_m")
                mass_kg = meta.get("body_mass_kg")

            trial_raw = str(d["trial_name"]) if "trial_name" in d else fpath.stem.replace("_smpl_params","")
            trial = sanitize(trial_raw)

            out_file = out_subj / f"{trial}_humanml3d_22joints.npz"
            np.savez(
                out_file,
                joints=j22_20.astype(np.float32),  # (T,22,3)
                fps=args.dst_fps,
                subject_id=subj,
                trial_name=trial_raw,        # keep original
                trial_name_sanitized=trial,
                age=age,
                sex=sex,
                height_m=height_m,
                body_mass_kg=mass_kg,
                # canonicalization info
                R=R.astype(np.float32),                # 3x3: original -> canonical ( +Y up, +Z fwd, +X left )
                canon_axes=np.stack([lr.mean(0), up.mean(0), fwd.mean(0)], 0).astype(np.float32),
                pelvis_traj=pelvis_20.astype(np.float32)  # (T,3) pelvis path in canonical frame
            )
            print(f"[{subj}] wrote {out_file.name} (orig='{trial_raw}') (T={j22_20.shape[0]})")

if __name__ == "__main__":
    main()
