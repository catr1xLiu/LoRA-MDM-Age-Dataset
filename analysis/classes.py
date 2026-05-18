"""
Core data and abstract classes for gait analysis.

MotionClip is the single data container used throughout the pipeline.
BaseMetric is the interface every scalar metric must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from analysis.constants import N_GAIT_PTS


@dataclass
class MotionClip:
    """
    A single motion clip with aligned joints, timestamps, and stride segments.

    Attributes:
        joints:     (T, 22, 3) joint positions; forward motion along +X, vertical +Y.
        timestamps: (T,) seconds elapsed from clip start.
        strides:    List of (t_start, t_end) frame-index pairs, one per detected
                    gait cycle (heel-strike → next ipsilateral heel-strike).
        subject_id: Unique clip / subject identifier.
        source:     "dataset" or "generated".
        age:        Subject age in years.
        age_group:  Discretised bin — "young", "mid", "old", or "unknown".
        sex:        "M", "F", or "?" when unknown.
        condition:  Clinical label (e.g. "able_bodied", "stroke").
    """

    joints:     np.ndarray              # (T, 22, 3)
    timestamps: np.ndarray              # (T,) seconds
    strides:    list[tuple[int, int]]   # [(t0, t1), ...]
    subject_id: str
    source:     str
    age:        float
    age_group:  str
    sex:        str = "?"
    condition:  str = "unknown"

    # ── Convenience properties ──────────────────────────────────────────────

    @property
    def n_frames(self) -> int:
        return int(self.joints.shape[0])

    @property
    def n_strides(self) -> int:
        return len(self.strides)

    @property
    def duration(self) -> float:
        """Total clip duration in seconds."""
        return float(self.timestamps[-1] - self.timestamps[0])

    # ── Per-stride access ───────────────────────────────────────────────────

    def get_stride(self, i: int) -> np.ndarray:
        """
        Joint positions for stride i.

        Args:
            i: Index into self.strides.

        Returns:
            (L, 22, 3) where L = t1 − t0 + 1.
        """
        t0, t1 = self.strides[i]
        return self.joints[t0 : t1 + 1]

    def stride_duration(self, i: int) -> float:
        """Duration of stride i in seconds."""
        t0, t1 = self.strides[i]
        return float(self.timestamps[t1] - self.timestamps[t0])

    def stride_length(self, i: int) -> float:
        """Pelvis displacement along +X for stride i (metres)."""
        t0, t1 = self.strides[i]
        return float(abs(self.joints[t1, 0, 0] - self.joints[t0, 0, 0]))

    # ── Time normalisation ──────────────────────────────────────────────────

    def normalize_stride(self, i: int, n_pts: int = N_GAIT_PTS) -> np.ndarray:
        """
        Resample stride i to a fixed number of time points via linear interpolation.

        Args:
            i:     Index into self.strides.
            n_pts: Output time points (default N_GAIT_PTS = 101, i.e. 0 %–100 %).

        Returns:
            (n_pts, 22, 3) resampled joint array.
        """
        seg   = self.get_stride(i)
        src_t = np.linspace(0.0, 1.0, len(seg))
        tgt_t = np.linspace(0.0, 1.0, n_pts)
        return (
            np.stack(
                [np.interp(tgt_t, src_t, seg[:, j, ax])
                 for j in range(22) for ax in range(3)]
            )
            .reshape(22, 3, n_pts)
            .transpose(2, 0, 1)             # (n_pts, 22, 3)
        )

    def normalized_gait_cycle(self, n_pts: int = N_GAIT_PTS) -> Optional[np.ndarray]:
        """
        Stack all strides into a time-normalised array.

        Returns:
            (n_strides, n_pts, 22, 3), or None when no strides exist.
        """
        if not self.strides:
            return None
        return np.stack([self.normalize_stride(i, n_pts) for i in range(self.n_strides)])


# ── Abstract metric base ────────────────────────────────────────────────────────

class BaseMetric(ABC):
    """Interface for a single-clip scalar gait metric."""

    title: str   # human-readable name (plot titles, axis labels)
    unit:  str   # unit string, e.g. "m/s", "°", "%"

    @abstractmethod
    def __call__(self, clip: MotionClip) -> float:
        """
        Compute the metric for a single clip.

        Args:
            clip: A MotionClip with pre-detected strides.

        Returns:
            Scalar float, or np.nan if the clip is unsuitable.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(title={self.title!r}, unit={self.unit!r})"
