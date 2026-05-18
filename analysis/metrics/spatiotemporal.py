import numpy as np

from analysis.classes import BaseMetric, MotionClip
from analysis.constants import MIN_CLIP_FRAMES, PELVIS


class WalkingSpeed(BaseMetric):
    """Mean forward speed derived from total pelvis displacement over the clip."""
    title = "Walking Speed"
    unit  = "m/s"

    def __call__(self, clip: MotionClip) -> float:
        """
        Args:
            clip: MotionClip with forward axis +X.

        Returns:
            Average speed in m/s, or np.nan if the clip is too short.
        """
        if clip.n_frames < MIN_CLIP_FRAMES:
            return np.nan
        dist = abs(clip.joints[-1, PELVIS, 0] - clip.joints[0, PELVIS, 0])
        return float(dist / clip.duration)


class StrideLength(BaseMetric):
    """Mean pelvis displacement along +X per stride."""
    title = "Stride Length"
    unit  = "m"

    def __call__(self, clip: MotionClip) -> float:
        """
        Args:
            clip: MotionClip with at least one detected stride.

        Returns:
            Mean stride length in metres, or np.nan if no strides.
        """
        if not clip.strides:
            return np.nan
        return float(np.mean([clip.stride_length(i) for i in range(clip.n_strides)]))


class StrideTime(BaseMetric):
    """Mean duration of a complete stride cycle."""
    title = "Stride Time"
    unit  = "s"

    def __call__(self, clip: MotionClip) -> float:
        """
        Args:
            clip: MotionClip with at least one detected stride.

        Returns:
            Mean stride time in seconds, or np.nan if no strides.
        """
        if not clip.strides:
            return np.nan
        return float(np.mean([clip.stride_duration(i) for i in range(clip.n_strides)]))


class Cadence(BaseMetric):
    """Number of strides per minute."""
    title = "Cadence"
    unit  = "strides/min"

    def __call__(self, clip: MotionClip) -> float:
        """
        Args:
            clip: MotionClip with at least one detected stride.

        Returns:
            Cadence in strides/min, or np.nan if no strides.
        """
        if not clip.strides:
            return np.nan
        mean_stride_time = np.mean([clip.stride_duration(i) for i in range(clip.n_strides)])
        return float(60.0 / mean_stride_time)
