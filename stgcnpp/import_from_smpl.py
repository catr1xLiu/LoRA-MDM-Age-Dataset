"""
Convert SMPL-fitted Van Criekinge data to NTU-25 skeleton format.

Uses SMPL mesh VERTICES (not joints) mapped to NTU joints via NTU_25_MARKERS.

Usage:
    python import_from_smpl.py

Output:
    stgcnpp/data/vc_ntu25.pkl - PYSKL-format pickle with age group labels
"""

import pickle
import random
from pathlib import Path

import numpy as np
import torch
import smplx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# NTU joint index to SMPL vertex ID mapping from explore_smpl_vertices.py
# NTU_25_MARKERS[joint_index] = smpl_vertex_id
NTU_25_MARKERS = {
    0: 1807,  # SpineBase
    1: 3511,  # SpineMid
    2: 3069,  # Neck
    3: 336,  # Head
    4: 1291,  # LeftShoulder
    5: 1573,  # LeftElbow
    6: 1923,  # LeftWrist
    7: 2226,  # LeftHand
    8: 4773,  # RightShoulder
    9: 5044,  # RightElbow
    10: 5385,  # RightWrist
    11: 5688,  # RightHand
    12: 1801,  # HipLeft
    13: 1046,  # LeftKnee
    14: 3321,  # LeftAnkle
    15: 3366,  # LeftFoot
    16: 5263,  # HipRight
    17: 4530,  # RightKnee
    18: 6721,  # RightAnkle
    19: 6766,  # RightFoot
    20: 3495,  # Spine1
    21: 2297,  # Spine2
    22: 5758,  # Thorax
    23: 2710,  # Nose
    24: 6170,  # RightEye
}

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
    """Convert SMPL parameters to NTU-25 joint positions using vertices."""
    vertices = get_smpl_vertices(
        model, params["poses"], params["trans"], params["betas"]
    )
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


def collect_clips(fitted_dir: Path) -> list[dict]:
    """Collect all SMPL-fitted clips from the directory."""
    clips = []

    for subject_dir in sorted(fitted_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name

        for npz_file in subject_dir.glob("*_smpl_params.npz"):
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

    return clips


def create_train_val_split(
    clips: list[dict], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list, list]:
    """Create subject-level train/val split."""
    random.seed(seed)
    subject_ids = sorted(set(c["subject_id"] for c in clips))
    random.shuffle(subject_ids)

    n_val = int(len(subject_ids) * val_ratio)
    val_subjects = set(subject_ids[:n_val])
    train_subjects = set(subject_ids[n_val:])

    train_clips = [c for c in clips if c["subject_id"] in train_subjects]
    val_clips = [c for c in clips if c["subject_id"] in val_subjects]

    return train_clips, val_clips


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    fitted_dir = project_dir / "data" / "fitted_smpl_all_3_tuned"
    smpl_dir = project_dir / "data" / "smpl"
    output_path = script_dir / "data" / "vc_ntu25.pkl"

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
    clips = collect_clips(fitted_dir)
    print(f"  Found {len(clips)} clips")

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
