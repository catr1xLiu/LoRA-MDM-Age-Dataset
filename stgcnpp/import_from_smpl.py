"""
Convert SMPL-fitted Van Criekinge data to NTU-25 skeleton format.

Uses SMPL mesh VERTICES (not joints) mapped to NTU joints via NTU_25_MARKERS.

Usage:
    python import_from_smpl.py
    python import_from_smpl.py --source-dir ../data/fitted_smpl_all_3_tuned --output-path data/custom_vc.pkl

Output:
    stgcnpp/data/vc_ntu25.pkl - PYSKL-format pickle with age group labels
"""

import argparse
import pickle
import random
import re
from pathlib import Path

import numpy as np
import smplx
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# fmt: off
# NTU joint index to SMPL vertex ID mapping from explore_smpl_vertices.py
# Note: Original uses 1-based keys (for display), converted to 0-based (for array indexing)
# NTU_25_MARKERS[joint_index] = smpl_vertex_id
# Fig. 1: 25 body joints - (1) base of spine, (2) middle of spine, (3) neck, (4) head,
#         (5-8) left arm, (9-12) right arm, (13-16) left leg, (17-20) right leg,
#         (21) spine, (22) tip of left hand, (23) left thumb, (24) tip of right hand, (25) right thumb
NTU_25_MARKERS = {
    # Spine and head (indices 0-3)
    0: 1807,  # (1) SpineBase - base of spine
    1: 3511,  # (2) SpineMid - middle of spine
    2: 3069,  # (3) Neck
    3: 336,   # (4) Head

    # Left arm (indices 4-8): shoulder, elbow, wrist, hand
    4: 1291, 5: 1573, 6: 1923, 7: 2226,  # (5) LeftShoulder, (6) LeftElbow, (7) LeftWrist, (8) LeftHand

    # Right arm (indices 9-12): shoulder, elbow, wrist, hand
    8: 4773, 9: 5044, 10: 5385, 11: 5688,  # (9) RightShoulder, (10) RightElbow, (11) RightWrist, (12) RightHand

    # Left leg (indices 13-16): hip, knee, ankle, foot
    12: 1801, 13: 1046, 14: 3321, 15: 3366,  # (13) HipLeft, (14) LeftKnee, (15) LeftAnkle, (16) LeftFoot

    # Right leg (indices 17-20): hip, knee, ankle, foot
    16: 5263, 17: 4530, 18: 6721, 19: 6766,  # (17) HipRight, (18) RightKnee, (19) RightAnkle, (20) RightFoot

    # Additional joints (indices 21-24): spine, hand tips, thumbs
    20: 3495,  # (21) Spine - spine
    21: 2297,  # (22) Spine2 - tip of left hand
    22: 2710,  # (23) Spine3 - left thumb
    23: 5758,  # (24) Thorax - tip of right hand
    24: 6170,  # (25) RightEye - right thumb
}
# fmt: on

NTU_JOINT_COUNT = 25


def load_smpl_model(models_dir: Path, gender: str = "neutral"):
    """Load SMPL model for vertex generation on GPU."""
    model = smplx.create(
        str(models_dir),
        model_type="smpl",
        gender=gender,
        ext="pkl",
        batch_size=32,
        use_hands=False,
        use_face=False,
    ).to(device)
    return model


def get_smpl_vertices(
    model: smplx.SMPL, poses: np.ndarray, trans: np.ndarray, betas: np.ndarray
) -> np.ndarray:
    """Generate SMPL mesh vertices from pose parameters using GPU batch processing.

    Args:
        model: SMPL model (on GPU)
        poses: (T, 72) SMPL pose parameters
        trans: (T, 3) translation
        betas: (10,) shape parameters

    Returns:
        (T, 10475, 3) SMPL mesh vertices
    """
    T = poses.shape[0]
    batch_size = 32

    betas_tensor = (
        torch.tensor(betas, dtype=torch.float32)
        .to(device)
        .unsqueeze(0)
        .repeat(min(batch_size, T), 1)
    )

    all_vertices = []
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        batch_T = end - start

        poses_batch = torch.tensor(poses[start:end], dtype=torch.float32).to(device)
        trans_batch = torch.tensor(trans[start:end], dtype=torch.float32).to(device)

        betas_batch = betas_tensor[:batch_T]

        output = model(
            betas=betas_batch,
            body_pose=poses_batch[:, 3:],
            global_orient=poses_batch[:, :3],
            transl=trans_batch,
        )

        verts = output.vertices.detach().cpu().numpy()
        all_vertices.append(verts)

    return np.concatenate(all_vertices, axis=0)


def extract_ntu_joints_from_vertices(vertices: np.ndarray) -> np.ndarray:
    """Extract NTU-25 joint positions from SMPL mesh vertices.

    Uses vertex IDs from NTU_25_MARKERS mapping with GPU acceleration.

    Args:
        vertices: (T, V, 3) SMPL mesh vertices

    Returns:
        (T, 25, 3) NTU-25 joint positions
    """
    vertex_indices = torch.tensor(
        list(NTU_25_MARKERS.values()), dtype=torch.long, device=device
    )

    vertices_tensor = torch.tensor(vertices, dtype=torch.float32).to(device)

    ntu_joints = vertices_tensor[:, vertex_indices, :].cpu().numpy()

    return ntu_joints.astype(np.float32)


def load_smpl_params(npz_path: Path) -> dict:
    """Load SMPL parameters from npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "poses": data["poses"],
        "trans": data["trans"],
        "betas": data["betas"],
        "gender": str(data["gender"]),
        "subject_id": str(data["subject_id"]),
        "age": int(data["age"]),
        "trial_name": str(data["trial_name"]),
    }


def smpl_to_ntu25(params: dict, model: smplx.SMPL) -> np.ndarray:
    """Convert SMPL parameters to NTU-25 joint positions using vertices.

    Trims FRAME_TRIM frames from the start and end to remove instrumental error.
    """
    poses = params["poses"]
    trans = params["trans"]

    # Trim noisy boundary frames
    if poses.shape[0] > 2 * FRAME_TRIM:
        poses = poses[FRAME_TRIM:-FRAME_TRIM]
        trans = trans[FRAME_TRIM:-FRAME_TRIM]

    vertices = get_smpl_vertices(model, poses, trans, params["betas"])
    ntu_joints = extract_ntu_joints_from_vertices(vertices)
    return ntu_joints


def get_age_group(age: int) -> int:
    """Assign age group label."""
    if age < 40:
        return 0
    elif age < 65:
        return 1
    else:
        return 2


FRAME_TRIM = 5  # frames to trim from start and end (instrumental error)


_TRIAL_RE = re.compile(r"^SUBJ\d+_(\d+)_smpl_params$")


def get_trial_number(npz_path: Path) -> int | None:
    """Extract trial number from filename, e.g. SUBJ92_2_smpl_params.npz -> 2.

    Returns None if the filename does not match the expected pattern.
    """
    m = _TRIAL_RE.match(npz_path.stem)
    return int(m.group(1)) if m else None


def collect_clips(fitted_dir: Path) -> tuple[list[dict], int]:
    """Collect all SMPL-fitted clips from the directory.

    Skips trial 0 (T-pose) for every subject.

    Returns:
        (clips, n_discarded) where n_discarded counts trial-0 files skipped.
    """
    clips = []
    n_discarded = 0

    for subject_dir in sorted(fitted_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        for npz_file in sorted(subject_dir.glob("*_smpl_params.npz")):
            trial_num = get_trial_number(npz_file)

            if trial_num is None:
                print(f"  [SKIP] Unrecognised filename: {npz_file.name}")
                n_discarded += 1
                continue

            if trial_num == 0:
                n_discarded += 1
                continue

            try:
                params = load_smpl_params(npz_file)
                clips.append(
                    {
                        "subject_id": params["subject_id"],
                        "trial_name": params["trial_name"],
                        "npz_path": str(npz_file),
                        "age": params["age"],
                        "gender": params["gender"],
                        "params": params,
                    }
                )
            except Exception as e:
                print(f"Warning: Failed to load {npz_file}: {e}")
                continue

    return clips, n_discarded


def extract_subject_number(subject_id: str) -> int:
    """Extract numeric subject ID from string like 'SUBJ60' -> 60."""
    match = re.search(r"\d+", subject_id)
    return int(match.group()) if match else 0


def create_train_val_split(
    clips: list[dict], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list, list]:
    """Create strict subject-level train/val split using round-robin.

    Subjects are sorted numerically (SUBJ1, SUBJ2, etc.) which preserves
    rough age ordering (older subjects have lower numbers).

    For every 4 subjects: first 3 go to train, 4th goes to val.
    This maintains approximately 75% train / 25% val split while ensuring
    all clips from a subject are in the same split.
    """
    # Extract unique subject IDs and sort numerically
    subject_ids = sorted(
        set(c["subject_id"] for c in clips), key=extract_subject_number
    )

    train_subjects = set()
    val_subjects = set()

    # Round-robin: for every 4 subjects, assign 3 to train, 1 to val
    # This gives ~75/25 split which is close to the requested 80/20
    for i, subject_id in enumerate(subject_ids):
        if (i % 4) == 3:  # Every 4th subject (indices 3, 7, 11, ...)
            val_subjects.add(subject_id)
        else:
            train_subjects.add(subject_id)

    train_clips = [c for c in clips if c["subject_id"] in train_subjects]
    val_clips = [c for c in clips if c["subject_id"] in val_subjects]

    # Log the split distribution
    print(f"  Split distribution:")
    print(
        f"    Train subjects: {len(train_subjects)} ({len(train_subjects) / len(subject_ids) * 100:.1f}%)"
    )
    print(
        f"    Val subjects: {len(val_subjects)} ({len(val_subjects) / len(subject_ids) * 100:.1f}%)"
    )

    return train_clips, val_clips


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Convert SMPL-fitted Van Criekinge data to NTU-25 PYSKL format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=project_dir / "data" / "fitted_smpl_all_3_tuned",
        help="Directory containing subject folders with *_smpl_params.npz files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=script_dir / "data" / "vc_ntu25.pkl",
        help="Full output path, including the .pkl filename.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    fitted_dir = args.source_dir.expanduser().resolve()
    smpl_dir = project_dir / "data" / "smpl"
    output_path = args.output_path.expanduser().resolve()

    print("=" * 60)
    print("SMPL Vertices to NTU-25 Conversion")
    print("=" * 60)
    print(f"Input: {fitted_dir}")
    print(f"SMPL models: {smpl_dir}")
    print(f"Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nLoading SMPL models...")
    model_neutral = load_smpl_model(smpl_dir, "neutral")
    model_male = load_smpl_model(smpl_dir, "male")
    model_female = load_smpl_model(smpl_dir, "female")
    print("  Loaded neutral, male, female models")

    print("\nCollecting clips...")
    clips, n_discarded = collect_clips(fitted_dir)
    print(f"  Found {len(clips)} clips ({n_discarded} trial-0 T-pose clips discarded)")

    subject_counts = {}
    age_dist = {0: 0, 1: 0, 2: 0}
    for c in clips:
        sid = c["subject_id"]
        subject_counts[sid] = subject_counts.get(sid, 0) + 1
        age_dist[get_age_group(c["age"])] += 1

    print(f"  Unique subjects: {len(subject_counts)}")
    print(
        f"  Age dist: Young={age_dist[0]}, Adult={age_dist[1]}, Elderly={age_dist[2]}"
    )

    print("\nCreating train/val split (80/20)...")
    train_clips, val_clips = create_train_val_split(clips, val_ratio=0.2)
    print(f"  Train: {len(train_clips)}, Val: {len(val_clips)}")

    def clips_to_annotations(clips_list: list[dict]) -> list[dict]:
        annotations = []
        for i, clip in enumerate(clips_list):
            if (i + 1) % 50 == 0:
                print(f"    Processing {i + 1}/{len(clips_list)}...")

            gender = clip["gender"]
            if gender == "male":
                model = model_male
            elif gender == "female":
                model = model_female
            else:
                model = model_neutral

            try:
                ntu_joints = smpl_to_ntu25(clip["params"], model)
                T = ntu_joints.shape[0]

                frame_dir = f"{clip['subject_id']}_{clip['trial_name']}"
                label = get_age_group(clip["age"])

                keypoint = ntu_joints[np.newaxis, :, :, :]

                annotations.append(
                    {
                        "frame_dir": frame_dir,
                        "label": label,
                        "keypoint": keypoint,
                        "keypoint_score": np.ones((1, T, 25), dtype=np.float32),
                        "total_frames": T,
                        "img_shape": (1080, 1920),
                    }
                )
            except Exception as e:
                print(f"    Warning: Failed {clip['npz_path']}: {e}")
                continue

        return annotations

    print("\nConverting training clips...")
    train_annotations = clips_to_annotations(train_clips)
    print(f"  Train: {len(train_annotations)}")

    print("\nConverting validation clips...")
    val_annotations = clips_to_annotations(val_clips)
    print(f"  Val: {len(val_annotations)}")

    train_ids = [ann["frame_dir"] for ann in train_annotations]
    val_ids = [ann["frame_dir"] for ann in val_annotations]

    data = {
        "split": {"train": train_ids, "val": val_ids},
        "annotations": train_annotations + val_annotations,
    }

    print(f"\nTotal: {len(data['annotations'])} annotations")

    print(f"\nSaving to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    sample = data["annotations"][0]
    print(f"\nSample: {sample['frame_dir']}")
    print(f"  keypoint shape: {sample['keypoint'].shape}")
    print(f"  Joint 0 (SpineBase): {sample['keypoint'][0, 0, 0, :]}")
    print(f"  Joint 24 (RightEye): {sample['keypoint'][0, 0, 24, :]}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
