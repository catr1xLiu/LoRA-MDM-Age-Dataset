from typing import Optional

import numpy as np

from analysis.classes import MotionClip
from analysis.constants import FPS


class VelocityMap:
    """
    Mean joint speed (m/s) over the normalised gait cycle.

    Returns (N_GAIT_PTS, 22) — not a scalar, so not a BaseMetric subclass.
    Use plot_velocity_heatmap_grid from analysis.plots to visualise.
    """

    def __call__(self, clip: MotionClip) -> Optional[np.ndarray]:
        """
        Args:
            clip: MotionClip with pre-detected strides.

        Returns:
            (N_GAIT_PTS, 22) mean joint speed array, or None if no strides exist.
        """
        cycles = clip.normalized_gait_cycle()   # (n, N_GAIT_PTS, 22, 3) or None
        if cycles is None:
            return None

        vel   = np.diff(cycles, axis=1) * FPS              # (n, N-1, 22, 3)
        speed = np.linalg.norm(vel, axis=-1)               # (n, N-1, 22)
        speed = np.concatenate([speed, speed[:, -1:]], axis=1)  # pad to (n, N, 22)
        return np.mean(speed, axis=0)                      # (N_GAIT_PTS, 22)
