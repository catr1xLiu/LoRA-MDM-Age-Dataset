"""
Data loading and pre-processing for gait analysis.

Produces MotionClip objects with:
  - joints axis-aligned so forward motion is along +X
  - timestamps in seconds
  - strides detected from bilateral foot contacts
"""

# ── Class declarations ─────────────────────────────────────────────────────────
# Imported here so readers of this module see the data contract up front.
from analysis.classes import MotionClip, BaseMetric  # noqa: F401

# ── Standard imports ───────────────────────────────────────────────────────────
import glob
import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np

from analysis.constants import (
    DT, FPS, PELVIS, L_FOOT, R_FOOT,
    AGE_BINS, GROUP_REPRESENTATIVE_AGE,
)

_ROOT = Path(__file__).parent.parent
_DATA = _ROOT / "data"

# ── Contact detection thresholds ───────────────────────────────────────────────
_HEIGHT_THRESH = 0.05   # max foot height above per-clip floor to count as contact (m)
_VEL_THRESH    = 1.0    # max foot speed to count as contact (m/s)
_MIN_CONTACT   = 3      # minimum consecutive contact frames for a valid heel strike
_MIN_STRIDE_T  = 0.4    # minimum valid stride duration (s)
_MAX_STRIDE_T  = 3.0    # maximum valid stride duration (s)
_MIN_STRIDE_L  = 0.1    # minimum valid stride length along +X (m)
_MAX_STRIDE_L  = 3.0    # maximum valid stride length along +X (m)


# ── Axis alignment ─────────────────────────────────────────────────────────────

def _align_forward_axis(joints: np.ndarray) -> np.ndarray:
    """
    Reorder and flip axes so the primary walking direction is along +X.

    Compares the peak-to-peak pelvis range on X vs Z and swaps them if Z
    dominates.  Then flips the sign if net displacement is negative.

    Args:
        joints: (T, 22, 3) in any axis convention.

    Returns:
        (T, 22, 3) with forward motion along +X, vertical unchanged on Y.
    """
    pelvis = joints[:, PELVIS]
    out    = joints.copy()
    if np.ptp(pelvis[:, 2]) > np.ptp(pelvis[:, 0]):
        out = out[:, :, [2, 1, 0]]              # swap X ↔ Z
    if out[-1, PELVIS, 0] < out[0, PELVIS, 0]:
        out[:, :, 0] = -out[:, :, 0]           # ensure net +X displacement
    return out


# ── Stride detection ───────────────────────────────────────────────────────────

def _heel_strikes(joints: np.ndarray, foot_idx: int) -> np.ndarray:
    """
    Rising-edge contact frames for one foot using relative height + velocity.

    Floor is estimated as the 2nd-percentile of foot Y, making detection
    robust across clips where the global Y origin varies.

    Args:
        joints:   (T, 22, 3) joint array (forward already on +X).
        foot_idx: Joint index for the toe/foot to track.

    Returns:
        Integer array of heel-strike frame indices.
    """
    pos   = joints[:, foot_idx]                                     # (T, 3)
    floor = np.percentile(pos[:, 1], 2)
    rel_h = pos[:, 1] - floor
    speed = np.linalg.norm(
        np.diff(pos, axis=0, prepend=pos[:1]) * FPS, axis=-1
    )
    contact = (rel_h < _HEIGHT_THRESH) & (speed < _VEL_THRESH)

    padded = np.concatenate([[False], contact, [False]])
    starts = np.where(~padded[:-1] &  padded[1:])[0]
    ends   = np.where( padded[:-1] & ~padded[1:])[0]
    return np.array(
        [s for s, e in zip(starts, ends) if e - s >= _MIN_CONTACT], dtype=int
    )


def detect_strides(joints: np.ndarray) -> list[tuple[int, int, str]]:
    """
    Detect all valid stride segments from bilateral heel-strike events.

    A stride spans one heel-strike to the next on the same foot (complete
    gait cycle).  Pairs are sanity-filtered by duration and pelvis
    displacement along the forward axis (+X).

    Args:
        joints: (T, 22, 3) with forward motion already aligned to +X.

    Returns:
        Time-sorted list of (t_start, t_end, foot) where foot is "L" or "R".
    """
    strides: list[tuple[int, int, str]] = []
    for foot, label in ((L_FOOT, "L"), (R_FOOT, "R")):
        strikes = _heel_strikes(joints, foot)
        for t0, t1 in zip(strikes[:-1], strikes[1:]):
            dt = (int(t1) - int(t0)) * DT
            dl = abs(joints[int(t1), PELVIS, 0] - joints[int(t0), PELVIS, 0])
            if _MIN_STRIDE_T <= dt <= _MAX_STRIDE_T and _MIN_STRIDE_L <= dl <= _MAX_STRIDE_L:
                strides.append((int(t0), int(t1), label))
    return sorted(strides)


# ── Metadata helper ────────────────────────────────────────────────────────────

def _load_metadata(path: Path) -> dict[str, dict]:
    spec = importlib.util.spec_from_file_location("metadata", path)
    assert spec is not None and spec.loader is not None, f"Cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return {**mod.create_able_bodied_metadata(), **mod.create_stroke_metadata()}


def _age_to_group(age: float) -> str:
    for group, (lo, hi) in AGE_BINS.items():
        if lo <= age < hi:
            return group
    return "unknown"


# ── Public loaders ─────────────────────────────────────────────────────────────

def load_dataset_clips(
    joints_dir: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    min_frames: int = 20,
) -> list[MotionClip]:
    """
    Load ground-truth motion clips as axis-aligned MotionClip objects.

    Each .npy file is a single clip of shape (T, 22, 3).  Filename stem must
    start with a subject ID that matches an entry in metadata.py.
    Strides are detected automatically via bilateral foot contacts.

    Args:
        joints_dir:    Directory of per-clip .npy files.
                       Defaults to data/humanml3d_new_joints_6.
        metadata_path: Path to metadata.py.
                       Defaults to data/van_criekinge_unprocessed_1/metadata.py.
        min_frames:    Discard clips shorter than this many frames.

    Returns:
        List of MotionClip objects sorted by filename.
    """
    joints_dir    = joints_dir    or (_DATA / "humanml3d_new_joints_6")
    metadata_path = metadata_path or (_DATA / "van_criekinge_unprocessed_1" / "metadata.py")

    metadata = _load_metadata(metadata_path)

    clips: list[MotionClip] = []
    for f in sorted(glob.glob(str(joints_dir / "*.npy"))):
        sid  = Path(f).stem.split("_")[0]
        meta = metadata.get(sid)
        if meta is None:
            continue
        age = meta.get("age")
        if age is None:
            continue

        raw = np.load(f)                         # (T, 22, 3)
        if raw.shape[0] < min_frames:
            continue

        joints = _align_forward_axis(raw)
        clips.append(MotionClip(
            joints     = joints,
            timestamps = np.arange(joints.shape[0], dtype=float) * DT,
            strides    = detect_strides(joints),
            subject_id = Path(f).stem,
            source     = "dataset",
            age        = float(age),
            age_group  = _age_to_group(float(age)),
            sex        = meta.get("sex", "?"),
            condition  = meta.get("condition", "unknown"),
        ))

    return clips


def load_generated_clips(
    gen_dir: Optional[Path] = None,
    motion_name: str = "motion1_walk_forward",
    max_batches: int = 16,
    min_frames: int = 20,
) -> list[MotionClip]:
    """
    Load LoRA-generated motion clips for all age groups.

    Each batch file is a dict with:
        "motion":  (B, 22, 3, T) float32 — joint positions
        "lengths": (B,)          int     — valid frame count per sample

    Args:
        gen_dir:      Root of generated clips. Defaults to data/generated_clips_humanml3d.
        motion_name:  Motion-type subdirectory (e.g. "motion1_walk_forward").
        max_batches:  Max batch files to read per age group.
        min_frames:   Discard clips shorter than this many frames.

    Returns:
        List of MotionClip objects.  Age is set to the group midpoint.
    """
    gen_dir    = gen_dir or (_DATA / "generated_clips_humanml3d")
    motion_dir = gen_dir / motion_name

    clips: list[MotionClip] = []
    for ag in AGE_BINS:
        batch_files = sorted(
            glob.glob(str(motion_dir / ag / "batch_*.npy"))
        )[:max_batches]

        for bf in batch_files:
            data    = np.load(bf, allow_pickle=True).item()
            motion  = data["motion"]     # (B, 22, 3, T)
            lengths = data["lengths"]    # (B,)

            for i in range(motion.shape[0]):
                t   = int(lengths[i])
                raw = motion[i].transpose(2, 0, 1)[:t]   # (T, 22, 3)
                if raw.shape[0] < min_frames:
                    continue

                joints = _align_forward_axis(raw)
                clips.append(MotionClip(
                    joints     = joints,
                    timestamps = np.arange(joints.shape[0], dtype=float) * DT,
                    strides    = detect_strides(joints),
                    subject_id = f"{ag}_{Path(bf).stem}_{i}",
                    source     = "generated",
                    age        = GROUP_REPRESENTATIVE_AGE[ag],
                    age_group  = ag,
                ))

    return clips
