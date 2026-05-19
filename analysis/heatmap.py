"""
Velocity heatmap plotting for gait analysis.

Kept separate from results.py because it visualises the full 2-D
(joint × gait-cycle) speed map rather than per-metric distributions.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.classes import MotionClip
from analysis.constants import HEATMAP_LABELS, HEATMAP_ORDER
from analysis.metrics.velocity_map import VelocityMap


def plot_velocity_heatmap_grid(
    clips_by_label: list[tuple[str, list[MotionClip]]],
    out_path: Path,
    title: str = "Mean Joint Speed over Gait Cycle",
) -> None:
    """
    Grid of velocity heatmaps, one column per (label, clips) pair.

    All panels share the same colour scale (98th-percentile maximum).

    Args:
        clips_by_label: List of (column_title, clips) pairs.
        out_path:       Output PNG file path.
        title:          Figure-level title.
    """
    vm_fn = VelocityMap()
    panels: list[tuple[str, Optional[np.ndarray]]] = []
    for label, clips in clips_by_label:
        vmaps = [vm for c in clips if (vm := vm_fn(c)) is not None]
        panels.append((label, np.mean(vmaps, axis=0) if vmaps else None))

    valid = [(lbl, vm) for lbl, vm in panels if vm is not None]
    if not valid:
        print(f"  Skipped {out_path.name}: no velocity maps available")
        return

    vmax = max(float(np.percentile(vm, 98)) for _, vm in valid)
    n    = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(11 * n, 8), sharey=True, squeeze=False)
    im = None
    for ax, (label, vm) in zip(axes[0], panels):
        if vm is None:
            ax.set_visible(False)
            continue
        data = vm[:, HEATMAP_ORDER].T
        im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r",
                       extent=[0, 100, -0.5, 21.5], origin="lower", vmin=0, vmax=vmax)
        ax.set_yticks(range(22))
        ax.set_yticklabels(HEATMAP_LABELS, fontsize=7.5)
        ax.set_xticks(range(0, 101, 10))
        ax.set_xticklabels([f"{x}%" for x in range(0, 101, 10)], fontsize=7.5, rotation=45)
        ax.set_xlabel("Normalized Gait Cycle (%)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")

    if im is not None:
        fig.colorbar(im, ax=list(axes[0]), shrink=0.55, label="Joint Speed (m/s)")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
