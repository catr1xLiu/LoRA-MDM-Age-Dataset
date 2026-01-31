#!/usr/bin/env python3
"""
SMPL fitting for Van Criekinge dataset using SOMA MoSh++.

This script uses soma.amass.mosh_manual (alias for moshpp) to fit SMPL
body models to Vicon marker data from the Van Criekinge dataset.

    Usage:
        podman run --rm \
            -v $(pwd):/project \
            -v $(pwd)/data:/data \
            localhost/soma:latest \
            /bin/bash -lc ". /root/miniconda3/etc/profile.d/conda.sh && conda activate soma && python /project/2_fit_smpl_markers_moshpp.py \
                --c3d_path /data/van_criekinge_unprocessed_1/able_bodied/SUBJ01/SUBJ1_1.c3d \
                --out_dir /data/fitted_smpl_all_3_moshpp \
                --support_dir /data/smpl_mosh_fixed"

Output structure:
    {out_dir}/
    └── {subject_id}/
        ├── betas.npy                          # Body shape (10,)
        └── {trial_name}_smpl_params.npz      # Motion data + params (SMPL format)
"""

import os
import os.path as osp
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from loguru import logger


def sanitize(name: str) -> str:
    """Sanitize trial name for file system compatibility."""
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_")


def create_trial_name(c3d_path: str) -> str:
    """Create standardized trial name from C3D file path."""
    stem = Path(c3d_path).stem
    return sanitize(stem)


def run_mosh_on_c3d(
    c3d_path: str,
    output_dir: str,
    support_dir: str,
    gender: str = "neutral",
    subject_id: str = "SUBJ01",
) -> dict:
    """
    Run SOMA MoSh++ (soma.manual / moshpp) on a single C3D file.

    Args:
        c3d_path: Path to input C3D file
        output_dir: Base output directory for results
        support_dir: SOMA support directory (contains SMPL models, etc.)
        gender: Subject gender ('male', 'female', 'neutral')
        subject_id: Subject identifier

    Returns:
        dict with output file paths and metadata
    """
    from soma.amass.mosh_manual import mosh_manual

    trial_name = create_trial_name(c3d_path)

    logger.info(f"Processing: {c3d_path}")
    logger.info(f"  Subject: {subject_id}, Gender: {gender}")

    os.makedirs(output_dir, exist_ok=True)

    c3d_dir = osp.dirname(c3d_path)
    settings_path = osp.join(c3d_dir, "settings.json")

    if not osp.exists(settings_path):
        logger.info(f"Creating settings.json with gender={gender}")
        settings = {"gender": gender}
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        logger.info(f"Saved settings to: {settings_path}")

    work_dir = osp.join(output_dir, "mosh_work")
    os.makedirs(work_dir, exist_ok=True)

    logger.info("Running SOMA MoSh++ via soma.amass.mosh_manual...")

    mosh_manual(
        mocap_fnames=[c3d_path],
        mosh_cfg={
            "moshpp.verbosity": 1,
            "dirs.work_base_dir": work_dir,
            "dirs.support_base_dir": support_dir,
            "surface_model.type": "smpl",
            "opt_settings.weights_type": "smpl",
        },
        render_cfg=None,
        parallel_cfg={
            "pool_size": 1,
            "max_num_jobs": 1,
            "randomly_run_jobs": False,
        },
        run_tasks=["mosh"],
    )

    logger.info(f"MoSh++ complete. Checking output...")

    mosh_output_dir = osp.join(work_dir, "mosh_results", f"{subject_id}_{trial_name}")
    npz_files = (
        list(Path(mosh_output_dir).glob("*.npz")) if osp.exists(mosh_output_dir) else []
    )

    if npz_files:
        logger.info(f"Found MoSh++ output: {npz_files[0]}")
        return convert_soma_output(
            str(npz_files[0]), output_dir, subject_id, trial_name, gender, c3d_path
        )
    else:
        logger.error(f"No MoSh++ output found in {mosh_output_dir}")
        return None


def convert_soma_output(
    mosh_npz_path: str,
    output_dir: str,
    subject_id: str,
    trial_name: str,
    gender: str,
    c3d_path: str,
) -> dict:
    """
    Convert SOMA MoSh++ output to our expected npz format.

    SOMA moshpp outputs SMPL format: global_orient, body_pose, betas, trans, etc.
    SMPL has 24 joints and 72 pose parameters (69 body + 3 global orientation).
    We preserve the full SMPL format for downstream processing.
    """
    logger.info(f"Converting SOMA output: {mosh_npz_path}")

    mosh_data = np.load(mosh_npz_path)

    logger.info(f"MoSh++ output keys: {list(mosh_data.keys())}")

    subject_out_dir = osp.join(output_dir, subject_id)
    os.makedirs(subject_out_dir, exist_ok=True)

    betas_path = osp.join(subject_out_dir, "betas.npy")
    trial_path = osp.join(subject_out_dir, f"{trial_name}_smpl_params.npz")

    if "betas" in mosh_data:
        final_betas = mosh_data["betas"]
        if isinstance(final_betas, torch.Tensor):
            final_betas = final_betas.detach().cpu().numpy()
        if final_betas.ndim == 1:
            final_betas = final_betas[:10]
        np.save(betas_path, final_betas)
        logger.info(f"Saved betas to: {betas_path}")
    else:
        final_betas = np.zeros(10)
        np.save(betas_path, final_betas)
        logger.warning("No betas in mosh output, using zeros")

    if "global_orient" in mosh_data and "body_pose" in mosh_data:
        global_orient = mosh_data["global_orient"]
        body_pose = mosh_data["body_pose"]
        if isinstance(global_orient, torch.Tensor):
            global_orient = global_orient.detach().cpu().numpy()
        if isinstance(body_pose, torch.Tensor):
            body_pose = body_pose.detach().cpu().numpy()
        final_poses = np.concatenate([global_orient, body_pose], axis=-1)
    elif "poses" in mosh_data:
        final_poses = mosh_data["poses"]
        if isinstance(final_poses, torch.Tensor):
            final_poses = final_poses.detach().cpu().numpy()
    else:
        logger.error("No pose data found in mosh output!")
        final_poses = np.zeros((337, 72))

    if "trans" in mosh_data:
        final_trans = mosh_data["trans"]
        if isinstance(final_trans, torch.Tensor):
            final_trans = final_trans.detach().cpu().numpy()
    elif "transl" in mosh_data:
        final_trans = mosh_data["transl"]
        if isinstance(final_trans, torch.Tensor):
            final_trans = final_trans.detach().cpu().numpy()
    else:
        logger.warning("No translation in mosh output, using zeros")
        final_trans = np.zeros((final_poses.shape[0], 3))

    n_frames = len(final_poses)
    fps = float(mosh_data.get("fps", 100.0))

    if "joints" in mosh_data:
        final_joints = mosh_data["joints"]
        if isinstance(final_joints, torch.Tensor):
            final_joints = final_joints.detach().cpu().numpy()
    else:
        # SMPL has 24 joints (23 body + 1 root)
        final_joints = np.zeros((n_frames, 24, 3))

    np.savez(
        trial_path,
        poses=final_poses,
        trans=final_trans,
        betas=final_betas,
        gender=gender,
        subject_id=subject_id,
        trial_name=trial_name,
        fps=fps,
        n_frames=n_frames,
        joints=final_joints,
    )
    logger.info(f"Saved SMPL parameters to: {trial_path}")

    metadata = {
        "subject_id": subject_id,
        "trial_name": trial_name,
        "gender": gender,
        "n_frames": n_frames,
        "fps": fps,
        "marker_set": "plug_in_gait",
        "input_c3d": c3d_path,
        "model_type": "smpl",
        "n_joints": 24,
        "n_pose_params": final_poses.shape[-1] if final_poses.ndim > 1 else 72,
    }
    meta_path = osp.join(subject_out_dir, f"{trial_name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {meta_path}")

    return {
        "betas_path": betas_path,
        "trial_path": trial_path,
        "metadata_path": meta_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit SMPL to Van Criekinge markers using SOMA MoSh++",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in SOMA docker with SMPL models
  python 2_fit_smpl_markers_moshpp.py --c3d_path data/van_criekinge_unprocessed_1/able_bodied/SUBJ01/SUBJ1_1.c3d --out_dir data/fitted_smpl_all_3_moshpp --support_dir /data/smpl_mosh_fixed --gender neutral
        """,
    )

    parser.add_argument(
        "--c3d_path", type=str, required=True, help="Path to C3D file to process"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for fitted SMPL parameters",
    )
    parser.add_argument(
        "--support_dir",
        type=str,
        default=None,
        help="SOMA support directory (contains SMPL models, default: /data/smpl_mosh_fixed)",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["male", "female", "neutral"],
        help="Subject gender (default: neutral)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Subject ID (default: extracted from path)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (default: cuda)",
    )

    args = parser.parse_args()

    if not osp.exists(args.c3d_path):
        raise FileNotFoundError(f"C3D file not found: {args.c3d_path}")

    if args.support_dir is None:
        # Default to smpl models directory
        args.support_dir = "/data/smpl_mosh_fixed"

    if not osp.exists(args.support_dir):
        logger.warning(f"Support dir not found: {args.support_dir}")
        logger.info("Attempting to use fallback: /data/smpl_mosh_fixed")
        args.support_dir = "/data/smpl_mosh_fixed"

    os.makedirs(args.out_dir, exist_ok=True)

    subject_id = args.subject
    if subject_id is None:
        subject_id = Path(args.c3d_path).parent.name.upper()

    result = run_mosh_on_c3d(
        c3d_path=args.c3d_path,
        output_dir=args.out_dir,
        support_dir=args.support_dir,
        gender=args.gender,
        subject_id=subject_id,
    )

    if result:
        logger.success(f"Done! Output: {result['trial_path']}")
    else:
        logger.error("MoSh++ processing failed!")


if __name__ == "__main__":
    main()
