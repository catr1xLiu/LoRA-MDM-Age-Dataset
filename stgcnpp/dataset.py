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


class SkeletonDataset(Dataset):
    """Skeleton action recognition dataset.

    Args:
        ann_file:   path to a pickle annotation file (list of dicts, see above).
        clip_len:   number of frames to sample per clip.
        max_person: maximum number of persons; extras are cropped, missing are zero-padded.
        split:      optional string key; if provided the annotation file is expected to be a
                    dict with split names as keys (compatible with NTU-style .pkl files).
    """

    def __init__(self, ann_file: str, clip_len: int = 100,
                 max_person: int = 2, split: str | None = None):
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)

        if split is not None and isinstance(data, dict):
            data = data[split]

        self.samples = data
        self.clip_len = clip_len
        self.max_person = max_person

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, kp: np.ndarray) -> np.ndarray:
        """Uniformly sample clip_len frames from (T, V, C) or (M, T, V, C)."""
        T = kp.shape[-3]
        if T == self.clip_len:
            return kp
        indices = np.linspace(0, T - 1, self.clip_len, dtype=int)
        return kp[..., indices, :, :]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        kp = sample['keypoint'].astype(np.float32)   # (M, T, V, C) or (T, V, C)
        label = int(sample['label'])

        if kp.ndim == 3:          # (T, V, C) → (1, T, V, C)
            kp = kp[np.newaxis]

        # Pad / crop persons
        M = kp.shape[0]
        if M < self.max_person:
            pad = np.zeros((self.max_person - M, *kp.shape[1:]), dtype=np.float32)
            kp = np.concatenate([kp, pad], axis=0)
        else:
            kp = kp[:self.max_person]

        kp = self._sample_frames(kp)         # (M, clip_len, V, C)
        return torch.from_numpy(kp), label
