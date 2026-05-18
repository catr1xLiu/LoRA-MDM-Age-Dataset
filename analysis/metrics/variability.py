import numpy as np

from analysis.classes import BaseMetric, MotionClip


class StrideTimeCV(BaseMetric):
    """Coefficient of variation of stride time — temporal gait variability."""
    title = "Stride Time CV"
    unit  = "%"

    def __call__(self, clip: MotionClip) -> float:
        """
        Args:
            clip: MotionClip; requires at least 3 strides for a meaningful CV.

        Returns:
            CV of stride times as a percentage, or np.nan if fewer than 3 strides.
        """
        if clip.n_strides < 3:
            return np.nan
        times = np.array([clip.stride_duration(i) for i in range(clip.n_strides)])
        return float(np.std(times) / np.mean(times) * 100)


class StrideLengthCV(BaseMetric):
    """Coefficient of variation of stride length — spatial gait variability."""
    title = "Stride Length CV"
    unit  = "%"

    def __call__(self, clip: MotionClip) -> float:
        """
        Args:
            clip: MotionClip; requires at least 3 strides for a meaningful CV.

        Returns:
            CV of stride lengths as a percentage, or np.nan if fewer than 3 strides.
        """
        if clip.n_strides < 3:
            return np.nan
        lengths = np.array([clip.stride_length(i) for i in range(clip.n_strides)])
        return float(np.std(lengths) / np.mean(lengths) * 100)
