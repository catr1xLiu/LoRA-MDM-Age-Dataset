#!/usr/bin/env python3
"""
Stride curve visualisation.

For a sample of healthy clips (n_strides >= 2) plots left and right foot
marker height over time with detected heel-strike events overlaid.
Saves one figure per clip to output/stride_curves/.
"""

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.constants import AGE_BINS, DT, L_FOOT, R_FOOT
from analysis.data import (
    _DATA,
    _align_forward_axis,
    _heel_strikes,
    _load_metadata,
    detect_strides,
)

OUT = Path("output/stride_curves")
OUT.mkdir(parents=True, exist_ok=True)

CLIPS_PER_GROUP = 3   # how many healthy clips to sample per age group


def load_clip(path: Path) -> np.ndarray:
    return _align_forward_axis(np.load(path))


def plot_clip(
    joints: np.ndarray,
    subject_id: str,
    clip_name: str,
    age: float,
    out_path: Path,
) -> None:
    T          = joints.shape[0]
    time       = np.arange(T) * DT

    l_pos      = joints[:, L_FOOT, 1]     # Y (vertical)
    r_pos      = joints[:, R_FOOT, 1]

    l_floor    = np.percentile(l_pos, 2)
    r_floor    = np.percentile(r_pos, 2)

    l_strikes  = _heel_strikes(joints, L_FOOT)
    r_strikes  = _heel_strikes(joints, R_FOOT)
    strides    = detect_strides(joints)

    fig, ax = plt.subplots(figsize=(12, 4))

    # Foot height curves
    ax.plot(time, l_pos, color="#1565C0", lw=1.5, label="Left foot height")
    ax.plot(time, r_pos, color="#C62828", lw=1.5, label="Right foot height", alpha=0.85)

    # Floor estimates
    ax.axhline(l_floor, color="#1565C0", lw=0.8, ls="--", alpha=0.5, label=f"L floor ({l_floor:.3f} m)")
    ax.axhline(r_floor, color="#C62828", lw=0.8, ls="--", alpha=0.5, label=f"R floor ({r_floor:.3f} m)")

    # Contact threshold bands
    ax.axhline(l_floor + 0.05, color="#1565C0", lw=0.6, ls=":", alpha=0.35)
    ax.axhline(r_floor + 0.05, color="#C62828", lw=0.6, ls=":", alpha=0.35)

    # Heel strikes
    for f in l_strikes:
        ax.axvline(time[f], color="#1565C0", lw=1.2, alpha=0.7,
                   label="L strike" if f == l_strikes[0] else None)
        ax.text(time[f], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.25,
                "L", fontsize=7, color="#1565C0", ha="center", va="bottom")

    for f in r_strikes:
        ax.axvline(time[f], color="#C62828", lw=1.2, alpha=0.7,
                   label="R strike" if f == r_strikes[0] else None)
        ax.text(time[f], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.25,
                "R", fontsize=7, color="#C62828", ha="center", va="top")

    # Shade validated strides
    for i, (t0, t1) in enumerate(strides):
        ax.axvspan(time[t0], time[t1], alpha=0.07, color="green",
                   label="Validated stride" if i == 0 else None)

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Foot height (m)", fontsize=10)
    ax.set_title(
        f"{clip_name}  |  Age {age:.0f}y  |  "
        f"L strikes: {len(l_strikes)}   R strikes: {len(r_strikes)}   "
        f"Strides: {len(strides)}",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=7.5, ncol=4, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def main() -> None:
    joints_dir    = _DATA / "humanml3d_new_joints_6"
    metadata_path = _DATA / "van_criekinge_unprocessed_1" / "metadata.py"
    metadata      = _load_metadata(metadata_path)

    # Group files by age group, filter to healthy clips
    by_group: dict[str, list[tuple[Path, str, float]]] = {g: [] for g in AGE_BINS}

    for f in sorted(glob.glob(str(joints_dir / "*.npy"))):
        fp  = Path(f)
        sid = fp.stem.split("_")[0]
        if sid not in metadata:
            continue
        age = metadata[sid].get("age")
        if age is None:
            continue

        joints = load_clip(fp)
        if len(detect_strides(joints)) < 2:
            continue

        for gname, (lo, hi) in AGE_BINS.items():
            if lo <= float(age) < hi:
                by_group[gname].append((fp, sid, float(age)))
                break

    print("Generating stride curves...")
    for gname, candidates in by_group.items():
        sample = candidates[:CLIPS_PER_GROUP]
        for fp, sid, age in sample:
            joints = load_clip(fp)
            plot_clip(
                joints, sid, fp.stem, age,
                OUT / f"{fp.stem}.png",
            )

    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
