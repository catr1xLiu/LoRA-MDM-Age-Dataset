"""
Statistical computation and plotting for gait analysis results.

Statistics functions return MetricStats namedtuples.
Plot functions write PNG files to caller-specified paths.
save_summary_csv writes a mean ± SD table across sources and age groups.
"""

import csv
import warnings
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress

from analysis.classes import BaseMetric, MotionClip
from analysis.constants import (
    AGE_BINS, GROUP_COLORS,
    HEATMAP_LABELS, HEATMAP_ORDER,
)
from analysis.metrics.velocity_map import VelocityMap

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════

class MetricStats(NamedTuple):
    mean:   float
    std:    float
    values: list[float]   # all raw values including np.nan
    n:      int           # count of non-nan values


def compute_stats(
    clips: list[MotionClip],
    metric: BaseMetric,
) -> MetricStats:
    """
    Compute distribution statistics of a scalar metric across a list of clips.

    Args:
        clips:  MotionClip objects to evaluate.
        metric: Callable BaseMetric instance.

    Returns:
        MetricStats with mean, std, all raw values (including nan), and valid count.
    """
    values = [metric(c) for c in clips]
    valid  = [v for v in values if not np.isnan(v)]
    mean   = float(np.mean(valid)) if valid else np.nan
    std    = float(np.std(valid))  if len(valid) > 1 else np.nan
    return MetricStats(mean=mean, std=std, values=values, n=len(valid))


def compute_group_stats(
    clips: list[MotionClip],
    metric: BaseMetric,
    groups: list[str] | None = None,
) -> dict[str, MetricStats]:
    """
    Compute per-age-group distribution statistics for a single metric.

    Args:
        clips:  MotionClip objects.
        metric: Callable BaseMetric instance.
        groups: Age group keys to include; defaults to all groups present in clips.

    Returns:
        Dict mapping age_group → MetricStats.
    """
    by_group: dict[str, list[MotionClip]] = defaultdict(list)
    for c in clips:
        by_group[c.age_group].append(c)

    target = groups or sorted(by_group.keys())
    return {g: compute_stats(by_group[g], metric) for g in target if g in by_group}


# ══════════════════════════════════════════════════════════════════════════════
# Shared plot helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ax_style(ax: matplotlib.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)


def _ylabel(metric: BaseMetric) -> str:
    return f"{metric.title} ({metric.unit})"


# ══════════════════════════════════════════════════════════════════════════════
# Plot functions
# ══════════════════════════════════════════════════════════════════════════════

def plot_scatter_grid(
    clips: list[MotionClip],
    metrics: list[BaseMetric],
    out_path: Path,
    ncols: int = 4,
    suptitle: str = "Gait Metrics vs Age",
    color: str = "#1565C0",
) -> None:
    """
    Grid of age-scatter plots with linear regression, one panel per metric.

    Args:
        clips:    MotionClip list (typically the ground-truth dataset).
        metrics:  Scalar metric callables.
        out_path: Output PNG file path.
        ncols:    Number of subplot columns.
        suptitle: Figure-level title.
        color:    Scatter point and regression line colour.
    """
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4), squeeze=False)
    axes_flat = axes.flatten()
    ages = np.array([c.age for c in clips], dtype=float)

    for ax, m in zip(axes_flat, metrics):
        vals = np.array([m(c) for c in clips], dtype=float)
        mask = ~np.isnan(vals)
        ax.scatter(ages[mask], vals[mask], alpha=0.45, s=18, color=color, zorder=2)
        if mask.sum() >= 5:
            slope, intercept, r, p, _ = linregress(ages[mask], vals[mask])
            xl = np.array([ages[mask].min(), ages[mask].max()])
            ax.plot(xl, slope * xl + intercept, color=color, lw=2, zorder=3)
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            ax.text(0.97, 0.06, f"r={r:.2f}{star}", transform=ax.transAxes,
                    ha="right", fontsize=8, color=color)
        ax.set_xlabel("Age (years)", fontsize=9)
        ax.set_ylabel(_ylabel(m), fontsize=9)
        ax.set_title(m.title, fontsize=10, fontweight="bold")
        _ax_style(ax)

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_violin_by_age_group(
    clips: list[MotionClip],
    metrics: list[BaseMetric],
    out_path: Path,
    suptitle: str = "Gait Metric Distribution by Age Group",
) -> None:
    """
    Row of violin plots, one panel per metric, split by age group.

    Args:
        clips:    MotionClip list.
        metrics:  Scalar metric callables.
        out_path: Output PNG file path.
        suptitle: Figure-level title.
    """
    color_list = [GROUP_COLORS[g] for g in AGE_BINS]
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(len(metrics) * 3.5 + 1, 5), squeeze=False)

    for ax, m in zip(axes[0], metrics):
        groups_data: list[list[float]] = []
        labels: list[str] = []
        for (gname, (lo, hi)), col in zip(AGE_BINS.items(), color_list):
            vals = [v for c in clips if lo <= c.age < hi for v in [m(c)] if not np.isnan(v)]
            if vals:
                groups_data.append(vals)
                labels.append(f"{gname.capitalize()}\n(n={len(vals)})")

        if groups_data:
            parts = ax.violinplot(groups_data, positions=range(len(groups_data)),
                                  showmedians=True)
            for pc, col in zip(parts["bodies"], color_list):
                pc.set_facecolor(col)
                pc.set_alpha(0.7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)

        ax.set_ylabel(_ylabel(m), fontsize=9)
        ax.set_title(m.title, fontsize=10, fontweight="bold")
        _ax_style(ax)

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_group_comparison_bars(
    dataset_clips: list[MotionClip],
    generated_clips: list[MotionClip],
    metrics: list[BaseMetric],
    out_path: Path,
    ncols: int = 4,
    suptitle: str = "Dataset vs Generated: Gait Metrics by Age Group",
) -> None:
    """
    Side-by-side bar charts comparing dataset and generated clips per metric.

    Dataset bars are blue shades, generated bars are green shades.

    Args:
        dataset_clips:   Ground-truth clips.
        generated_clips: LoRA-generated clips.
        metrics:         Scalar metric callables.
        out_path:        Output PNG file path.
        ncols:           Number of subplot columns.
        suptitle:        Figure-level title.
    """
    x = np.arange(len(AGE_BINS))
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4), squeeze=False)
    axes_flat = axes.flatten()

    ds_colors = ["#BBDEFB", "#90CAF9", "#42A5F5"]
    gn_colors = ["#C8E6C9", "#A5D6A7", "#66BB6A"]

    for ax, m in zip(axes_flat, metrics):
        ds_m, ds_s, gn_m, gn_s = [], [], [], []
        for gname, (lo, hi) in AGE_BINS.items():
            ds = compute_stats([c for c in dataset_clips   if lo <= c.age < hi], m)
            gn = compute_stats([c for c in generated_clips if c.age_group == gname], m)
            ds_m.append(ds.mean); ds_s.append(ds.std)
            gn_m.append(gn.mean); gn_s.append(gn.std)

        w = 0.35
        ax.bar(x - w / 2, ds_m, w, yerr=ds_s, capsize=4,
               color=ds_colors, alpha=0.9, label="Dataset")
        ax.bar(x + w / 2, gn_m, w, yerr=gn_s, capsize=4,
               color=gn_colors, alpha=0.9, label="Generated")
        ax.set_xticks(x)
        ax.set_xticklabels([g.capitalize() for g in AGE_BINS], fontsize=10)
        ax.set_ylabel(_ylabel(m), fontsize=9)
        ax.set_title(m.title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        _ax_style(ax)

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_violin_comparison(
    dataset_clips: list[MotionClip],
    generated_clips: list[MotionClip],
    metrics: list[BaseMetric],
    out_path: Path,
    suptitle: str = "Dataset vs Generated — Gait Metric Distributions by Age Group",
) -> None:
    """
    Paired violin plots: dataset (opaque, left) and generated (translucent, right)
    per age group, sharing the same group colour.

    Args:
        dataset_clips:   Ground-truth clips.
        generated_clips: LoRA-generated clips.
        metrics:         Scalar metric callables.
        out_path:        Output PNG file path.
        suptitle:        Figure-level title.
    """
    W      = 0.28
    OFFSET = 0.22
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(len(metrics) * 4.5, 6), squeeze=False)

    for ax, m in zip(axes[0], metrics):
        xtick_pos: list[int] = []
        xtick_labels: list[str] = []

        for gi, (gname, (lo, hi)) in enumerate(AGE_BINS.items()):
            color   = GROUP_COLORS[gname]
            ds_vals = [v for c in dataset_clips   if lo <= c.age < hi
                       for v in [m(c)] if not np.isnan(v)]
            gn_vals = [v for c in generated_clips if c.age_group == gname
                       for v in [m(c)] if not np.isnan(v)]
            xtick_pos.append(gi)
            xtick_labels.append(gname.capitalize())

            for data, xpos, alpha in [
                (ds_vals, gi - OFFSET, 0.80),
                (gn_vals, gi + OFFSET, 0.30),
            ]:
                if len(data) < 3:
                    continue
                parts = ax.violinplot(data, positions=[xpos], widths=W * 2,
                                      showmedians=True, showextrema=True)
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(alpha)
                    pc.set_edgecolor(color if alpha > 0.5 else "black")
                    pc.set_linewidth(1.2)
                for pname in ("cmins", "cmaxes", "cbars"):
                    if pname in parts:
                        parts[pname].set_edgecolor(color if alpha > 0.5 else "black")
                        parts[pname].set_linewidth(1.0)
                if "cmedians" in parts:
                    parts["cmedians"].set_edgecolor("black")
                    parts["cmedians"].set_linewidth(2.5)
                    parts["cmedians"].set_zorder(5)

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, fontsize=10)
        ax.set_ylabel(_ylabel(m), fontsize=10)
        ax.set_title(m.title, fontsize=11, fontweight="bold")
        _ax_style(ax)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#888888", alpha=0.80, label="Dataset"),
        Patch(facecolor="#888888", alpha=0.30, edgecolor="black", label="Generated"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.9)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_correlation_matrix(
    clips: list[MotionClip],
    metrics: list[BaseMetric],
    out_path: Path,
    include_age: bool = True,
    title: str = "Gait Metric Correlation Matrix",
) -> None:
    """
    Pearson correlation heatmap across all metrics (and optionally age).

    Args:
        clips:       MotionClip list.
        metrics:     Scalar metric callables.
        out_path:    Output PNG file path.
        include_age: Prepend subject age as the first variable when True.
        title:       Plot title.
    """
    labels = (["Age"] if include_age else []) + [m.title for m in metrics]
    n      = len(labels)
    data   = np.full((len(clips), n), np.nan)

    for i, c in enumerate(clips):
        offset = 0
        if include_age:
            data[i, 0] = c.age
            offset = 1
        for j, m in enumerate(metrics):
            data[i, j + offset] = m(c)

    corr = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            mask = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
            if mask.sum() >= 5:
                corr[i, j] = float(np.corrcoef(data[mask, i], data[mask, j])[0, 1])

    fig, ax = plt.subplots(figsize=(max(7, n), max(6, n - 1)))
    sns.heatmap(corr, mask=np.isnan(corr), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels,
                ax=ax, square=True, linewidths=0.5,
                cbar_kws={"label": "Pearson r"})
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


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

    valid = [(l, vm) for l, vm in panels if vm is not None]
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
        data = vm[:, HEATMAP_ORDER].T           # (22, N_GAIT_PTS)
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


def save_summary_csv(
    dataset_clips: list[MotionClip],
    generated_clips: list[MotionClip],
    metrics: list[BaseMetric],
    out_path: Path,
) -> None:
    """
    Write a CSV table of mean ± SD for each metric, source, and age group.

    Args:
        dataset_clips:   Ground-truth clips.
        generated_clips: LoRA-generated clips.
        metrics:         Scalar metric callables.
        out_path:        Output CSV file path.
    """
    rows: list[dict] = []
    for source, clips in [("Dataset", dataset_clips), ("Generated", generated_clips)]:
        for gname, (lo, hi) in AGE_BINS.items():
            group_clips = [c for c in clips if lo <= c.age < hi]
            row: dict = {"source": source, "group": gname, "n": len(group_clips)}
            for m in metrics:
                s   = compute_stats(group_clips, m)
                key = m.title.lower().replace(" ", "_")
                row[f"{key}_mean"] = f"{s.mean:.3f}" if not np.isnan(s.mean) else "N/A"
                row[f"{key}_sd"]   = f"{s.std:.3f}"  if not np.isnan(s.std)  else "N/A"
            rows.append(row)

    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved: {out_path.name}")
