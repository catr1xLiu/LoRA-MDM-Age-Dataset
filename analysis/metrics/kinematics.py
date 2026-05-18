import numpy as np

from analysis.classes import BaseMetric, MotionClip
from analysis.constants import L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, SPINE1


def _joint_angle_deg(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> np.ndarray:
    """
    Included angle at vertex p2 formed by the rays p1→p2 and p3→p2.

    Args:
        p1, p2, p3: Arrays of shape (..., 3).

    Returns:
        Angles in degrees, shape (...).
    """
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    cos_a = np.sum(v1 * v2, axis=-1) / (n1[..., 0] * n2[..., 0] + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


class KneeROM(BaseMetric):
    """Peak-to-peak knee flexion/extension range across both legs."""
    title = "Knee ROM"
    unit  = "°"

    def __call__(self, clip: MotionClip) -> float:
        """
        Args:
            clip: MotionClip with at least 5 frames.

        Returns:
            Bilateral knee ROM in degrees, or np.nan if clip is too short.
        """
        if clip.n_frames < 5:
            return np.nan
        j = clip.joints
        l = _joint_angle_deg(j[:, L_HIP], j[:, L_KNEE], j[:, L_ANKLE])
        r = _joint_angle_deg(j[:, R_HIP], j[:, R_KNEE], j[:, R_ANKLE])
        return float(np.ptp(np.concatenate([l, r])))


class HipROM(BaseMetric):
    """Peak-to-peak hip flexion/extension range across both legs."""
    title = "Hip ROM"
    unit  = "°"

    def __call__(self, clip: MotionClip) -> float:
        """
        Args:
            clip: MotionClip with at least 5 frames.

        Returns:
            Bilateral hip ROM in degrees, or np.nan if clip is too short.
        """
        if clip.n_frames < 5:
            return np.nan
        j = clip.joints
        l = _joint_angle_deg(j[:, SPINE1], j[:, L_HIP], j[:, L_KNEE])
        r = _joint_angle_deg(j[:, SPINE1], j[:, R_HIP], j[:, R_KNEE])
        return float(np.ptp(np.concatenate([l, r])))
