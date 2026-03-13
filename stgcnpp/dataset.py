"""Custom skeleton action dataset.

Expected data format (a .pkl file containing a list of dicts):
    [
      {
        'keypoint':  np.ndarray of shape (M, T, V, C),  # persons, frames, joints, coords
        'label':     int,
      },
      ...
    ]

M = number of persons (padded to max_person with zeros)
T = number of frames  (padded / sampled to clip_len)
V = 25  (NTU RGB+D joints)
C = 3   (x, y, z)
"""

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import _angle_between, _rotation_matrix


def normalize_skeleton(keypoint: np.ndarray) -> np.ndarray:
    """Reproduce pyskl PreNormalize3D.

    Input:  np.ndarray (M, T, V, C) — raw keypoints from the annotation pkl
    Output: np.ndarray (M, T', V, C) — centred, aligned, zero-frames stripped

    Steps:
      1. Strips all-zero frames; if 2 persons, keeps the one with more valid
         frames as person 0.
      2. Centres on joint 1 (SpineMid) of person 0, frame 0.
      3. Aligns spine (joint 0→1) to Z-axis [0, 0, 1].
      4. Aligns shoulders (joint 8→4) to X-axis [1, 0, 0].
    """
    keypoint = keypoint.copy().astype(np.float32)
    M, T, V, C = keypoint.shape

    if keypoint.sum() == 0:
        return keypoint.astype(np.float32)

    # 1. Strip all-zero frames; keep person with most valid frames as index 0
    index0 = [i for i in range(T) if not np.allclose(keypoint[0, i], 0, atol=1e-5)]
    if M == 2:
        index1 = [i for i in range(T) if not np.allclose(keypoint[1, i], 0, atol=1e-5)]
        if len(index0) < len(index1):
            keypoint = keypoint[:, index1]
            keypoint = keypoint[[1, 0]]
        else:
            keypoint = keypoint[:, index0]
    else:
        keypoint = keypoint[:, index0]

    # 2. Centre on joint 1 (SpineMid) of person 0, frame 0
    main_body_center = keypoint[0, 0, 1].copy()
    mask = ((keypoint != 0).sum(-1) > 0)[..., None]
    keypoint = (keypoint - main_body_center) * mask

    # 3. Rotate spine vector (joint 0 → joint 1) onto Z-axis
    spine = keypoint[0, 0, 1] - keypoint[0, 0, 0]
    axis_z = np.cross(spine, [0, 0, 1])
    angle_z = _angle_between(spine, [0, 0, 1])
    rot_z = _rotation_matrix(axis_z, angle_z)
    keypoint = np.einsum("mtvc,kc->mtvk", keypoint, rot_z)

    # 4. Rotate shoulder vector (joint 8 → joint 4) onto X-axis
    shoulder = keypoint[0, 0, 8] - keypoint[0, 0, 4]
    axis_x = np.cross(shoulder, [1, 0, 0])
    angle_x = _angle_between(shoulder, [1, 0, 0])
    rot_x = _rotation_matrix(axis_x, angle_x)
    keypoint = np.einsum("mtvc,kc->mtvk", keypoint, rot_x)

    return keypoint.astype(np.float32)


class SkeletonDataset(Dataset):
    """Skeleton action recognition dataset.

    Args:
        ann_file:   path to a pickle annotation file (list of dicts, see above).
        clip_len:   number of frames to sample per clip.
        max_person: maximum number of persons; extras are cropped, missing are zero-padded.
        split:      optional string key; if provided the annotation file is expected to be a
                    dict with split names as keys (compatible with NTU-style .pkl files).
        normalize:  if True, apply skeleton normalization (centering + rotation alignment).
    """

    def __init__(
        self,
        ann_file: str,
        clip_len: int = 100,
        max_person: int = 2,
        split: str | None = None,
        normalize: bool = True,
    ):
        with open(ann_file, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "annotations" in data:
            if split is not None and "split" in data:
                frame_dirs = data["split"].get(split, None)
                if frame_dirs is not None:
                    frame_to_idx = {
                        ann["frame_dir"]: i for i, ann in enumerate(data["annotations"])
                    }
                    self.samples = [
                        data["annotations"][frame_to_idx[fd]] for fd in frame_dirs
                    ]
                else:
                    self.samples = data["annotations"]
            else:
                self.samples = data["annotations"]
        else:
            self.samples = data
        self.clip_len = clip_len
        self.max_person = max_person
        self.normalize = normalize

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, kp: np.ndarray) -> np.ndarray:
        """Uniformly sample clip_len frames — matches pyskl val pipeline."""
        T = kp.shape[1]
        if T == self.clip_len:
            return kp

        if T < self.clip_len:
            # SHORT CLIPS: loop the sequence (pyskl behavior)
            # NOT linspace which duplicates individual frames
            inds = np.mod(np.arange(self.clip_len), T)
        elif T < 2 * self.clip_len:
            # MEDIUM CLIPS: segment-based with gap insertion
            np.random.seed(255)
            # consume the same RNG draws pyskl does before this branch
            np.random.rand()  # ratio (unused, =1.0)
            np.random.randint(1)  # off (unused, =0)
            basic = np.arange(self.clip_len)
            extra = np.random.choice(
                self.clip_len + 1, T - self.clip_len, replace=False
            )
            offset = np.zeros(self.clip_len + 1, dtype=np.int64)
            offset[extra] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            # LONG CLIPS: segment-based with seeded jitter
            np.random.seed(255)
            np.random.rand()
            np.random.randint(1)
            bids = np.array([i * T // self.clip_len for i in range(self.clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[: self.clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset

        return kp[:, inds]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        kp = sample["keypoint"].astype(np.float32)  # (M, T, V, C) or (T, V, C)
        label = int(sample["label"])

        if kp.ndim == 3:  # (T, V, C) → (1, T, V, C)
            kp = kp[np.newaxis]

        if self.normalize:
            kp = normalize_skeleton(kp)  # may change T (strips zero frames)

        # Pad / crop persons
        M = kp.shape[0]
        if M < self.max_person:
            pad = np.zeros((self.max_person - M, *kp.shape[1:]), dtype=np.float32)
            kp = np.concatenate([kp, pad], axis=0)
        else:
            kp = kp[: self.max_person]

        kp = self._sample_frames(kp)  # (max_person, clip_len, V, C)
        return torch.from_numpy(kp), label
