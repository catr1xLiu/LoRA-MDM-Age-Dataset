"""
NTU RGB+D skeleton dataset loader for ST-GCN++ inference.

Data format
-----------
The NTU 3-D skeleton pickle file (``ntu120_3danno.pkl``) is a dict with two
top-level keys:

    'split'       : dict mapping split names to lists of sample IDs
                    (e.g. 'xsub_train', 'xsub_val')
    'annotations' : list of per-sample dicts, each containing:
                        'frame_dir'    : str  — unique sample identifier
                        'total_frames' : int  — frame count
                        'label'        : int  — action class index (0-based)
                        'keypoint'     : ndarray (M, T, 25, 3) float32

Preprocessing pipeline (matches PYSKL ST-GCN++ NTU-120 3D test pipeline)
--------------------------------------------------------------------------
1. PreNormalize3D  — translate + rotate skeleton into a canonical pose
2. GenSkeFeat      — select 'joint' or 'bone' modality features
3. UniformSample   — sample 100 frames, optionally across multiple clips
4. PoseDecode      — index keypoints by sampled frame indices
5. FormatGCNInput  — pad/trim to exactly 2 persons, reshape to (C, M, T, V, C)

Reference: PYSKL pyskl/datasets/pipelines/pose_related.py
                  pyskl/datasets/pipelines/sampling.py
"""

import pickle

import numpy as np
import torch
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Skeleton joint pairs for computing bone (edge-vector) features
# NTU RGB+D: bone[v1] = joint[v1] - joint[v2]
# ---------------------------------------------------------------------------
# fmt: off
_NTU_BONE_PAIRS = (
    (0,  1),  (1, 20),  (2, 20),  (3,  2),
    (4, 20),  (5,  4),  (6,  5),  (7,  6),
    (8, 20),  (9,  8),  (10, 9),  (11, 10),
    (12, 0),  (13, 12), (14, 13), (15, 14),
    (16, 0),  (17, 16), (18, 17), (19, 18),
    (21, 22), (20, 20), (22,  7), (23, 24),
    (24, 11),
)
# fmt: on

# Seed used during test-time sampling (matches PYSKL default)
_TEST_SEED = 255


# ---------------------------------------------------------------------------
# Preprocessing transforms
# ---------------------------------------------------------------------------


def _normalize(vector: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, float]:
    norm: float = float(np.linalg.norm(vector))
    if norm < tol:
        return np.zeros_like(vector), 0.0
    return vector / norm, norm


def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    norm_ax, magnitude = _normalize(axis)
    if magnitude == 0.0 or abs(theta) < 1e-6:
        return np.eye(3, dtype=np.float32)
    rotation = Rotation.from_rotvec(norm_ax * theta)
    return rotation.as_matrix().astype(np.float32)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in radians between two 3-D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_angle = np.dot(v1 / n1, v2 / n2)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def pre_normalize_3d(keypoint: np.ndarray) -> np.ndarray:
    """Align an NTU RGB+D 3-D skeleton sequence into a canonical coordinate frame.

    Three steps (matching PYSKL's PreNormalize3D):

    1. **Person selection**: keep the person whose non-zero frames are more
       numerous as person-0 (primary actor).

    2. **Center translation**: translate all joints so that the spine joint
       (joint index 1, 0-based) of person-0 at the first valid frame is at
       the origin.  Zero-padded frames (all joints zero) are left at zero.

    3. **Spine alignment** (zaxis=[0,1]): rotate so that the vector from
       hip-center (joint 0) to spine (joint 1) points along +Z.

    4. **Shoulder alignment** (xaxis=[8,4]): rotate so that the vector from
       right-shoulder (joint 8) to left-shoulder (joint 4) points along +X.

    Args:
        keypoint: (M, T, 25, 3) skeleton array, float32.

    Returns:
        Aligned keypoint array of shape (M', T', 25, 3) where T' ≤ T
        (leading/trailing all-zero frames are stripped for the primary person).
    """
    M, T, V, C = keypoint.shape
    assert C == 3 and V == 25, f"Expected (M, T, 25, 3), got {keypoint.shape}"

    # --- Step 1: person selection ---
    # Find non-zero frame indices for each person
    valid0 = [t for t in range(T) if not np.allclose(keypoint[0, t], 0)]

    if M == 2:
        valid1 = [t for t in range(T) if not np.allclose(keypoint[1, t], 0)]
        # If person 1 has more valid frames, swap so person 0 is the primary actor
        if len(valid1) > len(valid0):
            keypoint = keypoint[[1, 0]]
            valid0 = valid1

    if len(valid0) == 0:
        # Degenerate sample — return as-is
        return keypoint

    # Trim to the valid frame range of the primary person
    keypoint = keypoint[:, np.array(valid0)]

    # --- Step 2: center translation ---
    # Reference: spine joint (index 1) of person 0 at the first valid frame
    center = keypoint[0, 0, 1].copy()  # shape (3,)

    # A mask of shape (M, T', V, 1) that is 1 for non-zero frames
    non_zero_mask = (keypoint != 0).any(axis=-1, keepdims=True)  # (M, T', V, 1)
    keypoint = (keypoint - center) * non_zero_mask

    # --- Step 3: spine alignment along +Z (zaxis = [joint_0, joint_1]) ---
    joint_bottom = keypoint[0, 0, 0].copy()  # hip center
    joint_top = keypoint[0, 0, 1].copy()  # spine

    spine_vec = joint_top - joint_bottom
    z_axis = np.array([0, 0, 1], dtype=np.float32)
    rot_axis = np.cross(spine_vec, z_axis)
    angle_z = _angle_between(spine_vec, z_axis)
    R_z = _rotation_matrix(rot_axis, angle_z)

    # Apply rotation: (M, T', V, 3) @ (3, 3)^T using einsum
    keypoint = np.einsum("mtvd,kd->mtvk", keypoint, R_z)

    # --- Step 4: shoulder alignment along +X (xaxis = [joint_8, joint_4]) ---
    joint_r_shoulder = keypoint[0, 0, 8].copy()  # right shoulder
    joint_l_shoulder = keypoint[0, 0, 4].copy()  # left shoulder

    shoulder_vec = joint_r_shoulder - joint_l_shoulder
    x_axis = np.array([1, 0, 0], dtype=np.float32)
    rot_axis_x = np.cross(shoulder_vec, x_axis)
    angle_x = _angle_between(shoulder_vec, x_axis)
    R_x = _rotation_matrix(rot_axis_x, angle_x)

    keypoint = np.einsum("mtvd,kd->mtvk", keypoint, R_x)

    return keypoint.astype(np.float32)


def joint_to_bone(keypoint: np.ndarray) -> np.ndarray:
    """Compute bone (edge-vector) features for the NTU skeleton.

    bone[v1] = joint[v1] − joint[v2]  for each (v1, v2) pair.

    Args:
        keypoint: (M, T, 25, 3) joint coordinate array.

    Returns:
        (M, T, 25, 3) bone feature array.
    """
    M, T, V, C = keypoint.shape
    bone = np.zeros_like(keypoint)
    for v1, v2 in _NTU_BONE_PAIRS:
        bone[:, :, v1, :] = keypoint[:, :, v1, :] - keypoint[:, :, v2, :]
    return bone


def uniform_sample_frames(
    total_frames: int,
    clip_len: int,
    num_clips: int,
    *,
    test_mode: bool,
    seed: int = _TEST_SEED,
) -> np.ndarray:
    """Uniformly sample frame indices from a skeleton sequence.

    Training: one stochastic clip (random offset within each equal-length bin).
    Testing:  ``num_clips`` deterministic clips (fixed seed=255).

    The logic mirrors PYSKL's UniformSampleFrames exactly, including the
    p_interval=1 (no temporal sub-sampling) used by ST-GCN++.

    Args:
        total_frames: Length of the source sequence.
        clip_len:     Number of frames to sample per clip.
        num_clips:    Number of clips (1 for train/val, 10 for test).
        test_mode:    If True use a fixed random seed for reproducibility.
        seed:         RNG seed used in test mode.

    Returns:
        1-D integer array of shape (num_clips * clip_len,).
    """
    if test_mode:
        np.random.seed(seed)

    all_inds = []

    for _ in range(num_clips):
        n = total_frames  # each clip samples from the full sequence

        if n < clip_len:
            # Short video: tile starting from a random/deterministic offset
            start = 0 if test_mode else np.random.randint(0, n)
            inds = np.arange(start, start + clip_len)

        elif clip_len <= n < 2 * clip_len:
            # Medium video: jitter a uniform grid
            basic = np.arange(clip_len)
            extras = np.random.choice(clip_len + 1, n - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[extras] = 1
            inds = basic + np.cumsum(offset)[:-1]

        else:
            # Long video: sample one frame from each equal-width bin
            bids = np.array([i * n // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            inds = bst + np.random.randint(bsize)

        all_inds.append(inds)

    indices = np.concatenate(all_inds)
    return np.mod(indices, total_frames).astype(np.int32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class NTUDataset(Dataset):
    """NTU RGB+D skeleton dataset for ST-GCN++ inference and evaluation.

    Loads pre-processed pickle annotations and applies the full preprocessing
    pipeline described in the PYSKL config.

    Args:
        pkl_path:   Path to ``ntu120_3danno.pkl``.
        split:      Name of the data split (e.g. ``'xsub_val'``).
        modality:   ``'joint'`` for raw joint coordinates, ``'bone'`` for
                    edge-vector features.
        clip_len:   Number of frames per clip. Default: 100.
        num_clips:  Number of clips per sample (1 for val, 10 for test).
        test_mode:  If True, apply deterministic test-time sampling.
        num_person: Maximum number of persons per clip. Default: 2.
    """

    def __init__(
        self,
        pkl_path: str,
        split: str,
        modality: str = "joint",
        clip_len: int = 100,
        num_clips: int = 10,
        test_mode: bool = True,
        num_person: int = 2,
        max_samples: int | None = None,
    ) -> None:
        assert modality in ("joint", "bone"), (
            f"modality must be 'joint' or 'bone', got '{modality}'"
        )

        self.modality = modality
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.num_person = num_person

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Build a lookup from frame_dir to annotation dict
        ann_by_id = {ann["frame_dir"]: ann for ann in data["annotations"]}

        # Filter to the requested split
        self.samples = [
            ann_by_id[sid] for sid in data["split"][split] if sid in ann_by_id
        ]

        if len(self.samples) == 0:
            raise ValueError(
                f"Split '{split}' is empty or not found. "
                f"Available splits: {list(data['split'].keys())}"
            )

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return a (keypoint, label) pair.

        keypoint shape: (num_clips, num_person, clip_len, V, C)
                        e.g. (10, 2, 100, 25, 3)
        """
        ann = self.samples[idx]
        keypoint = ann["keypoint"].astype(np.float32)  # (M, T, 25, 3)
        label = int(ann["label"])
        total_frames = ann["total_frames"]

        # --- Step 1: 3-D normalisation ---
        keypoint = pre_normalize_3d(keypoint)
        total_frames = keypoint.shape[1]  # may shrink after stripping zero frames

        # --- Step 2: modality features ---
        if self.modality == "bone":
            keypoint = joint_to_bone(keypoint)

        # --- Step 3: temporal sampling ---
        frame_inds = uniform_sample_frames(
            total_frames,
            self.clip_len,
            self.num_clips,
            test_mode=self.test_mode,
        )

        # --- Step 4: pose decode (index keypoints by frame_inds) ---
        keypoint = keypoint[:, frame_inds, :, :]  # (M, num_clips*clip_len, 25, 3)

        # --- Step 5: format for GCN input ---
        keypoint = self._format_gcn_input(keypoint)

        return torch.from_numpy(keypoint), label

    def _format_gcn_input(self, keypoint: np.ndarray) -> np.ndarray:
        """Reshape and pad the keypoint array to (num_clips, M, clip_len, V, C).

        Persons are zero-padded to ``num_person`` if there are fewer.
        If there are more, only the first ``num_person`` are kept.
        """
        M_actual = keypoint.shape[0]

        if M_actual < self.num_person:
            # Zero-pad extra person slots
            pad = np.zeros(
                (self.num_person - M_actual, *keypoint.shape[1:]),
                dtype=keypoint.dtype,
            )
            keypoint = np.concatenate([keypoint, pad], axis=0)
        elif M_actual > self.num_person:
            keypoint = keypoint[: self.num_person]

        M, T_total, V, C = keypoint.shape  # T_total = num_clips * clip_len

        # Reshape to (M, num_clips, clip_len, V, C) then transpose to
        # (num_clips, M, clip_len, V, C) to match the model's expected layout.
        keypoint = keypoint.reshape(M, self.num_clips, self.clip_len, V, C).transpose(
            1, 0, 2, 3, 4
        )
        return np.ascontiguousarray(keypoint)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def build_dataloader(
    pkl_path: str,
    split: str,
    modality: str = "joint",
    clip_len: int = 100,
    num_clips: int = 10,
    batch_size: int = 16,
    num_workers: int = 4,
    max_samples: int | None = None,
) -> DataLoader:
    """Convenience function: build a test-mode DataLoader.

    Args:
        pkl_path:    Path to the pickle annotation file.
        split:       Split name (e.g. ``'xsub_val'``).
        modality:    ``'joint'`` or ``'bone'``.
        clip_len:    Frames per clip. Default: 100.
        num_clips:   Clips per sample. Default: 10.
        batch_size:  Samples per mini-batch. Default: 16.
        num_workers: Parallel data-loading workers. Default: 4.
        max_samples: If set, evaluate only the first N samples.

    Returns:
        A configured ``DataLoader`` ready for iteration.
    """
    dataset = NTUDataset(
        pkl_path=pkl_path,
        split=split,
        modality=modality,
        clip_len=clip_len,
        num_clips=num_clips,
        test_mode=True,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
