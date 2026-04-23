#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert fitted SMPL params to HumanML3D-style 22 SMPL joints at 20 FPS.
- Select first 22 SMPL joints
- Canonicalize: rotate so Up=+Y and Forward=+Z (estimated); global root position preserved
- Resample to 20 FPS
"""

import os, glob, json, argparse
from pathlib import Path
import numpy as np

# same 24-joint order as fitter
SMPL_24 = [
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


def sanitize(s: str) -> str:
    return s.replace(" ", "_").replace("(", "").replace(")", "")


def decimate_indices(T, src_fps, dst_fps):
    if src_fps == dst_fps:
        return np.arange(T)
    step = src_fps / dst_fps
    return np.clip(np.round(np.arange(0, T, step)).astype(int), 0, T - 1)


def normalize(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-8)


def estimate_axes(j, trim=10):
    """Estimate body-frame axes from joint positions.

    Up is hardcoded to world +Y — step 2 guarantees this via vicon_to_smpl_coords.

    Forward is estimated from the pelvis trajectory projected onto the XZ plane,
    using the trimmed clip (first and last `trim` frames dropped) to avoid edge
    noise from T-pose settling or marker dropout. This handles subjects walking
    in any direction in the capture volume, which VC does not constrain.

    Left is cross(up, fwd), keeping all three axes exactly orthonormal.

    Args:
        j:    SMPL joints (T, 24, 3) in world space with +Y up (post step 2).
        trim: frames to drop from each end when estimating the walking direction.

    Returns:
        up:   (T, 3) constant [0, 1, 0] — world vertical.
        fwd:  (T, 3) constant unit vector in XZ — pelvis walking direction.
        left: (T, 3) constant unit vector in XZ — 90° left of fwd.
    """
    T = j.shape[0]

    # Up: world +Y, constant
    up = np.zeros((T, 3), dtype=np.float64)
    up[:, 1] = 1.0

    # Forward: pelvis displacement over trimmed clip, projected onto XZ
    t0 = min(trim, T // 4)          # graceful fallback for very short clips
    t1 = max(T - trim, 3 * T // 4)
    pelvis_disp = j[t1, 0, :] - j[t0, 0, :]   # joint 0 = pelvis
    pelvis_disp[1] = 0.0                        # project onto XZ
    fwd_vec = normalize(pelvis_disp[None, :])[0]

    # Fallback: if pelvis barely moved (< 1 cm), use hip vector
    if np.linalg.norm(pelvis_disp) < 0.01:
        hip = j[:, 1, :] - j[:, 2, :]   # L_hip - R_hip
        hip[:, 1] = 0.0
        hip_mean = hip.mean(axis=0)
        # fwd is 90° from lateral hip axis
        fwd_vec = normalize(np.cross(normalize(hip_mean[None, :])[0], up[0]))[None, :][0]

    fwd = np.tile(fwd_vec, (T, 1))

    # Left: cross(up, fwd) — orthonormal by construction
    left_vec = normalize(np.cross(up[0], fwd_vec)[None, :])[0]
    left = np.tile(left_vec, (T, 1))

    return up, fwd, left


def rot_to_align(up, fwd, left):
    """Return rotation R such that R @ v maps world vectors to canonical frame.

    Canonical frame: +X = body-left, +Y = body-up, +Z = body-forward.

    Args:
        up:  (T, 3) per-frame up vectors from estimate_axes.
        fwd: (T, 3) per-frame forward vectors from estimate_axes.
        left: (T, 3) per-frame left vectors from estimate_axes.
    Returns:
        R: (3, 3) rotation matrix (world → canonical).
    """
    # Average over all frames and re-orthonormalize
    u = normalize(up.mean(axis=0), axis=-1).squeeze()
    f = normalize(fwd.mean(axis=0), axis=-1).squeeze()
    l = normalize(left.mean(axis=0), axis=-1).squeeze()  # Up x Forward = Left

    # Source frame: columns are the three body axes expressed in world space
    R_src = np.stack([l, u, f], axis=1)  # shape (3, 3), columns = [left, up, fwd]

    # Target frame is the identity (canonical axes are the standard basis),
    # so R = R_tgt @ R_src.T = R_src.T maps world → canonical.
    return R_src.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fits_dir",
        required=True,
        help="Path to fitted SMPL parameters (typically data/fitted_smpl_all_3/)",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for HumanML3D format joints (typically data/humanml3d_joints_4/)",
    )
    ap.add_argument("--dst_fps", type=int, default=20)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--trial", default=None)
    args = ap.parse_args()

    fits = Path(args.fits_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    subj_dirs = [fits / args.subject] if args.subject else sorted([p for p in fits.iterdir() if p.is_dir()])

    for subj_dir in subj_dirs:
        subj = subj_dir.name
        out_subj = out_root / subj
        out_subj.mkdir(parents=True, exist_ok=True)

        npz_files = sorted(subj_dir.glob("*_smpl_params.npz"))
        if args.trial:
            npz_files = [p for p in npz_files if p.stem.startswith(args.trial)]
        if not npz_files:
            print(f"[{subj}] no fitted trials")
            continue

        for fpath in npz_files:
            d = np.load(fpath, allow_pickle=True)
            joints = d["joints"]  # (T,24,3)
            fps = float(d["fps"])
            trial = str(d["trial_name"]) if "trial_name" in d else fpath.stem.replace("_smpl_params", "")
            # estimate orientation & align
            up, fwd, left = estimate_axes(joints)
            R = rot_to_align(up, fwd, left)  # (3,3)
            j = joints @ R.T  # (T,24,3)
            # select first 22 joints (HumanML3D uses first 22 SMPL joints)
            j22 = j[:, :22, :]  # (T,22,3)

            # resample to 20 FPS
            idx = decimate_indices(j22.shape[0], src_fps=fps, dst_fps=args.dst_fps)
            j22_20 = j22[idx]

            # pull per-trial metadata written by the fitter
            meta_path = fpath.with_name(fpath.stem.replace("_smpl_params", "_smpl_metadata") + ".json")
            age = sex = height_m = mass_kg = None
            if meta_path.exists():
                with open(meta_path, "r") as fh:
                    meta = json.load(fh)
                age = meta.get("age")
                sex = meta.get("gender")
                height_m = meta.get("height_m")
                mass_kg = meta.get("body_mass_kg")

            trial_raw = str(d["trial_name"]) if "trial_name" in d else fpath.stem.replace("_smpl_params", "")
            trial = sanitize(trial_raw)

            out_file = out_subj / f"{trial}_humanml3d_22joints.npz"
            np.savez(
                out_file,
                joints=j22_20.astype(np.float32),  # (T,22,3)
                fps=args.dst_fps,
                subject_id=subj,
                trial_name=trial_raw,  # keep original
                trial_name_sanitized=trial,
                age=age,
                sex=sex,
                height_m=height_m,
                body_mass_kg=mass_kg,
                # canonicalization info
                R=R.astype(np.float32),  # 3x3: original -> canonical ( +Y up, +Z fwd, +X left )
                canon_axes=np.stack([left.mean(0), up.mean(0), fwd.mean(0)], 0).astype(np.float32),
            )
            print(f"[{subj}] wrote {out_file.name} (orig='{trial_raw}') (T={j22_20.shape[0]})")


if __name__ == "__main__":
    main()
