"""
Gait cycle line plots for speed and angular velocity.

Kept separate from results.py because it operates on per-stride kinematics
rather than per-clip scalar metrics.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.classes import MotionClip
from analysis.constants import (
    AGE_BINS, FPS, GROUP_COLORS,
    L_HIP, R_HIP,
    L_FOOT, R_FOOT, L_ANKLE, R_ANKLE, L_KNEE, R_KNEE,
    N_GAIT_PTS,
)

# ── Panel definitions ──────────────────────────────────────────────────────────

_GAIT_PANELS: list[tuple[int, str]] = [
    (L_FOOT,  "Left Foot"),
    (L_ANKLE, "Left Ankle"),
    (L_KNEE,  "Left Knee"),
    (R_FOOT,  "Right Foot"),
    (R_ANKLE, "Right Ankle"),
    (R_KNEE,  "Right Knee"),
]
_PANEL_JOINTS: list[int] = [j for j, _ in _GAIT_PANELS]


# ── Angle helpers ──────────────────────────────────────────────────────────────

def _angle_at_vertex(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Included angle at p2 in radians; inputs (T, 3)."""
    v1 = p1 - p2; v2 = p3 - p2
    cos_a = np.sum(v1 * v2, axis=-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
    )
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


def _joint_angle_series(joints: np.ndarray, j: int) -> np.ndarray:
    """
    Angle time series (radians) for a panel joint over a stride segment.

    - Knee:  hip–knee–ankle included angle
    - Ankle: knee–ankle–foot included angle
    - Foot:  sagittal-plane pitch of ankle→foot segment from +Y (vertical)
    """
    if j == L_KNEE:
        return _angle_at_vertex(joints[:, L_HIP],  joints[:, L_KNEE],  joints[:, L_ANKLE])
    if j == R_KNEE:
        return _angle_at_vertex(joints[:, R_HIP],  joints[:, R_KNEE],  joints[:, R_ANKLE])
    if j == L_ANKLE:
        return _angle_at_vertex(joints[:, L_KNEE], joints[:, L_ANKLE], joints[:, L_FOOT])
    if j == R_ANKLE:
        return _angle_at_vertex(joints[:, R_KNEE], joints[:, R_ANKLE], joints[:, R_FOOT])
    foot_idx, ankle_idx = (L_FOOT, L_ANKLE) if j == L_FOOT else (R_FOOT, R_ANKLE)
    seg = joints[:, foot_idx] - joints[:, ankle_idx]
    return np.arctan2(seg[:, 0], seg[:, 1])   # sagittal pitch from +Y toward +X


def _resample1d(signal: np.ndarray, n_out: int) -> np.ndarray:
    return np.interp(
        np.linspace(0.0, 1.0, n_out),
        np.linspace(0.0, 1.0, len(signal)),
        signal,
    )


# ── Per-clip curve computation ─────────────────────────────────────────────────

def _clip_speed_curves(clip: MotionClip) -> Optional[np.ndarray]:
    """
    (N_GAIT_PTS, 6) mean joint speed (m/s) over the gait panels, averaged over strides.

    Speed is computed at native FPS within each stride, then resampled to the
    normalised gait cycle axis — giving true m/s values.
    """
    if not clip.strides or not clip.has_valid_stride_order:
        return None
    stride_curves = []
    for t0, t1, _ in clip.strides:
        seg = clip.joints[t0 : t1 + 1]                               # (L, 22, 3)
        spd = np.linalg.norm(np.diff(seg, axis=0) * FPS, axis=-1)    # (L-1, 22) m/s
        spd = np.concatenate([spd, spd[-1:]], axis=0)                 # (L, 22) pad last
        stride_curves.append(np.stack(
            [_resample1d(spd[:, j], N_GAIT_PTS) for j in _PANEL_JOINTS], axis=1
        ))   # (N_GAIT_PTS, 6)
    return np.mean(stride_curves, axis=0)                             # (N_GAIT_PTS, 6)


def _clip_angular_vel_curves(clip: MotionClip) -> Optional[np.ndarray]:
    """
    (N_GAIT_PTS, 6) mean joint angular velocity magnitude (rad/s), averaged over strides.

    Each joint angle is computed at native FPS, differentiated and absolute-valued
    to get magnitude, then resampled to the normalised gait cycle axis.
    """
    if not clip.strides or not clip.has_valid_stride_order:
        return None
    stride_curves = []
    for t0, t1, _ in clip.strides:
        seg = clip.joints[t0 : t1 + 1]            # (L, 22, 3)
        panel_curves = []
        for j in _PANEL_JOINTS:
            angles  = _joint_angle_series(seg, j)                    # (L,) rad
            ang_vel = np.abs(np.diff(angles)) * FPS                  # (L-1,) rad/s magnitude
            ang_vel = np.concatenate([ang_vel, ang_vel[-1:]])        # (L,) pad last
            panel_curves.append(_resample1d(ang_vel, N_GAIT_PTS))
        stride_curves.append(np.stack(panel_curves, axis=1))        # (N_GAIT_PTS, 6)
    return np.mean(stride_curves, axis=0)                            # (N_GAIT_PTS, 6)


# ── Shared plot renderer ───────────────────────────────────────────────────────

def _plot_gait_cycle_by_age(
    clip_curves: list[tuple[MotionClip, np.ndarray]],
    ylabel: str,
    out_path: Path,
    title: str,
) -> None:
    """2×3 median + Q1–Q3 band over gait cycle; one panel per entry in _GAIT_PANELS."""
    from analysis.results import _ax_style

    pct = np.linspace(0, 100, N_GAIT_PTS)
    fig, axes = plt.subplots(2, 3, figsize=(18, 6), squeeze=False)

    for pi, (_, pname) in enumerate(_GAIT_PANELS):
        ax = axes[pi // 3][pi % 3]
        for gname, (lo, hi) in AGE_BINS.items():
            color  = GROUP_COLORS[gname]
            curves = np.array([
                curve[:, pi] for c, curve in clip_curves if lo <= c.age < hi
            ])
            if len(curves) < 3:
                continue
            ax.plot(pct, np.median(curves, axis=0), color=color, lw=2, label=gname.capitalize())
            ax.fill_between(pct,
                            np.percentile(curves, 25, axis=0),
                            np.percentile(curves, 75, axis=0),
                            color=color, alpha=0.18)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Gait Cycle (%)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(pname, fontsize=10, fontweight="bold")
        _ax_style(ax)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10, framealpha=0.9)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Public API ─────────────────────────────────────────────────────────────────

def plot_gait_cycle_speed_by_age(
    clips: list[MotionClip],
    out_path: Path,
    title: str = "Joint Speed over Gait Cycle by Age Group",
) -> None:
    """
    Median joint speed (m/s) over the normalised gait cycle for feet, ankles, and knees.

    Speed is computed at native FPS per stride then resampled to the gait cycle axis.
    """
    curves = [(c, cv) for c in clips if (cv := _clip_speed_curves(c)) is not None]
    _plot_gait_cycle_by_age(curves, "Joint Speed (m/s)", out_path, title)


def plot_gait_cycle_angular_vel_by_age(
    clips: list[MotionClip],
    out_path: Path,
    title: str = "Joint Angular Velocity over Gait Cycle by Age Group",
) -> None:
    """
    Median joint angular velocity magnitude (rad/s) over the normalised gait cycle.

    Angle definitions — Knee: hip–knee–ankle; Ankle: knee–ankle–foot;
    Foot: sagittal pitch of ankle→foot segment from vertical.
    Angular velocity computed at native FPS per stride, magnitude taken, then resampled.
    """
    curves = [(c, cv) for c in clips if (cv := _clip_angular_vel_curves(c)) is not None]
    _plot_gait_cycle_by_age(curves, "Angular Velocity (rad/s)", out_path, title)
