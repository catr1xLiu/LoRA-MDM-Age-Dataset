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
    AGE_BINS, FPS, GROUP_COLORS,
    L_HIP, R_HIP,
    L_FOOT, R_FOOT, L_ANKLE, R_ANKLE, L_KNEE, R_KNEE,
    N_GAIT_PTS,
)

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


# ── Gait cycle line-plot helpers ───────────────────────────────────────────────

_GAIT_PANELS: list[tuple[int, str]] = [
    (L_FOOT,  "Left Foot"),
    (L_ANKLE, "Left Ankle"),
    (L_KNEE,  "Left Knee"),
    (R_FOOT,  "Right Foot"),
    (R_ANKLE, "Right Ankle"),
    (R_KNEE,  "Right Knee"),
]
_PANEL_JOINTS: list[int] = [j for j, _ in _GAIT_PANELS]


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


def _clip_speed_curves(clip: MotionClip) -> Optional[np.ndarray]:
    """
    (N_GAIT_PTS, 6) mean joint speed (m/s) over the gait panels, averaged over strides.

    Speed is computed at native FPS within each stride segment, then resampled
    to the normalised gait cycle axis — giving true m/s, not a per-step artefact.
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
    (N_GAIT_PTS, 6) mean joint angular velocity (rad/s), averaged over strides.

    Each joint angle is computed at native FPS, differentiated to rad/s, then
    resampled to the normalised gait cycle axis.
    """
    if not clip.strides or not clip.has_valid_stride_order:
        return None
    stride_curves = []
    for t0, t1, _ in clip.strides:
        seg = clip.joints[t0 : t1 + 1]            # (L, 22, 3)
        panel_curves = []
        for j in _PANEL_JOINTS:
            angles  = _joint_angle_series(seg, j)               # (L,) rad
            ang_vel = np.abs(np.diff(angles)) * FPS              # (L-1,) rad/s magnitude
            ang_vel = np.concatenate([ang_vel, ang_vel[-1:]])   # (L,) pad last
            panel_curves.append(_resample1d(ang_vel, N_GAIT_PTS))
        stride_curves.append(np.stack(panel_curves, axis=1))   # (N_GAIT_PTS, 6)
    return np.mean(stride_curves, axis=0)                       # (N_GAIT_PTS, 6)


def _plot_gait_cycle_by_age(
    clip_curves: list[tuple[MotionClip, np.ndarray]],
    ylabel: str,
    out_path: Path,
    title: str,
) -> None:
    """2×3 median + Q1–Q3 band over gait cycle; one panel per entry in _GAIT_PANELS."""
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
    Median joint angular velocity (rad/s) over the normalised gait cycle.

    Angle definitions — Knee: hip–knee–ankle; Ankle: knee–ankle–foot;
    Foot: sagittal pitch of ankle→foot segment from vertical.
    Angular velocity computed at native FPS per stride then resampled.
    """
    curves = [(c, cv) for c in clips if (cv := _clip_angular_vel_curves(c)) is not None]
    _plot_gait_cycle_by_age(curves, "Angular Velocity (rad/s)", out_path, title)


def plot_combined_violin(
    dataset_clips: list[MotionClip],
    generated_clips: list[MotionClip],
    metrics: list[BaseMetric],
    out_path: Path,
) -> None:
    """
    2×4 grid of violin plots; each panel compares dataset (opaque) and
    generated (translucent, hatched) per age group for one metric.

    Args:
        dataset_clips:   Ground-truth clips.
        generated_clips: LoRA-generated clips.
        metrics:         8 scalar metric callables.
        out_path:        Output PNG file path.
    """
    from matplotlib.patches import Patch

    ncols, nrows = 4, 2
    W, OFFSET = 0.25, 0.20
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 5), squeeze=False)
    axes_flat = axes.flatten()

    for ax, m in zip(axes_flat, metrics):
        for gi, (gname, (lo, hi)) in enumerate(AGE_BINS.items()):
            color   = GROUP_COLORS[gname]
            ds_vals = [v for c in dataset_clips   if lo <= c.age < hi
                       for v in [m(c)] if not np.isnan(v)]
            gn_vals = [v for c in generated_clips if c.age_group == gname
                       for v in [m(c)] if not np.isnan(v)]

            for data, xpos, alpha, ec in [
                (ds_vals, gi - OFFSET, 0.80, color),
                (gn_vals, gi + OFFSET, 0.35, "black"),
            ]:
                if len(data) < 3:
                    continue
                parts = ax.violinplot(data, positions=[xpos], widths=W * 2,
                                     showmedians=True, showextrema=True)
                for pc in parts["bodies"]:
                    pc.set_facecolor(color); pc.set_alpha(alpha)
                    pc.set_edgecolor(ec);    pc.set_linewidth(1.2)
                for k in ("cmins", "cmaxes", "cbars"):
                    if k in parts:
                        parts[k].set_edgecolor(ec)
                if "cmedians" in parts:
                    parts["cmedians"].set_edgecolor("black")
                    parts["cmedians"].set_linewidth(2.0)
                    parts["cmedians"].set_zorder(5)

        ax.set_xticks(range(len(AGE_BINS)))
        ax.set_xticklabels([g.capitalize() for g in AGE_BINS], fontsize=10)
        ax.set_ylabel(_ylabel(m), fontsize=9)
        ax.set_title(m.title, fontsize=10, fontweight="bold")
        _ax_style(ax)

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    handles = [
        Patch(facecolor="#888888", alpha=0.80, label="Dataset"),
        Patch(facecolor="#888888", alpha=0.35, edgecolor="black", label="Generated"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.9)
    fig.suptitle("Dataset vs Generated — Gait Metric Distributions by Age Group",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_combined_bars(
    dataset_clips: list[MotionClip],
    generated_clips: list[MotionClip],
    metrics: list[BaseMetric],
    out_path: Path,
) -> None:
    """
    2×4 grid of bar charts; each panel shows mean ± SD for dataset (solid) and
    generated (hatched) per age group for one metric.

    Args:
        dataset_clips:   Ground-truth clips.
        generated_clips: LoRA-generated clips.
        metrics:         8 scalar metric callables.
        out_path:        Output PNG file path.
    """
    from matplotlib.patches import Patch

    ncols, nrows = 4, 2
    x = np.arange(len(AGE_BINS))
    w = 0.35
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4), squeeze=False)
    axes_flat = axes.flatten()

    for ax, m in zip(axes_flat, metrics):
        colors = [GROUP_COLORS[g] for g in AGE_BINS]
        ds_m, ds_s, gn_m, gn_s = [], [], [], []
        for gname, (lo, hi) in AGE_BINS.items():
            ds = compute_stats([c for c in dataset_clips   if lo <= c.age < hi], m)
            gn = compute_stats([c for c in generated_clips if c.age_group == gname], m)
            ds_m.append(ds.mean); ds_s.append(0.0 if np.isnan(ds.std) else ds.std)
            gn_m.append(gn.mean); gn_s.append(0.0 if np.isnan(gn.std) else gn.std)

        ax.bar(x - w / 2, ds_m, w, yerr=ds_s, capsize=4,
               color=colors, alpha=0.85, label="Dataset")
        gn_bars = ax.bar(x + w / 2, gn_m, w, yerr=gn_s, capsize=4,
                         color=colors, alpha=0.35, label="Generated")
        for bar in gn_bars:
            bar.set_edgecolor("black"); bar.set_linewidth(0.8); bar.set_hatch("///")

        ax.set_xticks(x)
        ax.set_xticklabels([g.capitalize() for g in AGE_BINS], fontsize=10)
        ax.set_ylabel(_ylabel(m), fontsize=9)
        ax.set_title(m.title, fontsize=10, fontweight="bold")
        _ax_style(ax)

    for ax in axes_flat[len(metrics):]:
        ax.set_visible(False)

    handles = [
        Patch(facecolor="#888888", alpha=0.85, label="Dataset"),
        Patch(facecolor="#888888", alpha=0.35, edgecolor="black", hatch="///", label="Generated"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.9)
    fig.suptitle("Dataset vs Generated — Gait Metrics by Age Group",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _clip_velocity_percentiles(
    clip: MotionClip,
) -> Optional[tuple[float, float, float]]:
    """
    Compute (Q1, Median, Q3) of Cartesian joint speed across all gait-cycle frames.

    Joints are evaluated over the time-normalised gait cycle so each clip
    contributes equal weight regardless of the number of strides.

    Args:
        clip: MotionClip with pre-detected strides.

    Returns:
        (Q1, Median, Q3) in m/s, or None if no gait cycles are available.
    """
    cycles = clip.normalized_gait_cycle()          # (n, N_GAIT_PTS, 22, 3) or None
    if cycles is None:
        return None
    speed = np.linalg.norm(np.diff(cycles, axis=1) * FPS, axis=-1)  # (n, N-1, 22)
    flat  = speed.ravel()
    return (float(np.percentile(flat, 25)),
            float(np.percentile(flat, 50)),
            float(np.percentile(flat, 75)))


def plot_joint_velocity_percentiles(
    dataset_clips: list[MotionClip],
    generated_clips: list[MotionClip],
    out_path: Path,
) -> None:
    """
    Violin distribution of per-clip joint-speed Q1, Median, and Q3 over gait cycles.

    For each clip the Q1, Median, and Q3 of all Cartesian joint speeds across
    all normalised gait-cycle frames are treated as three independent statistics.
    The resulting distributions are compared between dataset and generated clips,
    broken down by age group.

    Args:
        dataset_clips:   Ground-truth clips.
        generated_clips: LoRA-generated clips.
        out_path:        Output PNG file path.
    """
    from matplotlib.patches import Patch

    stat_labels = ["Q1 (25th pct)", "Median (50th pct)", "Q3 (75th pct)"]
    W, OFFSET = 0.25, 0.20
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)

    for si, (ax, stat_label) in enumerate(zip(axes[0], stat_labels)):
        for gi, (gname, (lo, hi)) in enumerate(AGE_BINS.items()):
            color = GROUP_COLORS[gname]

            ds_vals: list[float] = []
            for c in dataset_clips:
                if lo <= c.age < hi:
                    p = _clip_velocity_percentiles(c)
                    if p is not None:
                        ds_vals.append(p[si])

            gn_vals: list[float] = []
            for c in generated_clips:
                if c.age_group == gname:
                    p = _clip_velocity_percentiles(c)
                    if p is not None:
                        gn_vals.append(p[si])

            for data, xpos, alpha, ec in [
                (ds_vals, gi - OFFSET, 0.80, color),
                (gn_vals, gi + OFFSET, 0.35, "black"),
            ]:
                if len(data) < 3:
                    continue
                parts = ax.violinplot(data, positions=[xpos], widths=W * 2,
                                     showmedians=True, showextrema=True)
                for pc in parts["bodies"]:
                    pc.set_facecolor(color); pc.set_alpha(alpha)
                    pc.set_edgecolor(ec);    pc.set_linewidth(1.2)
                for k in ("cmins", "cmaxes", "cbars"):
                    if k in parts:
                        parts[k].set_edgecolor(ec)
                if "cmedians" in parts:
                    parts["cmedians"].set_edgecolor("black")
                    parts["cmedians"].set_linewidth(2.0)
                    parts["cmedians"].set_zorder(5)

        ax.set_xticks(range(len(AGE_BINS)))
        ax.set_xticklabels([g.capitalize() for g in AGE_BINS], fontsize=10)
        ax.set_ylabel("Joint Speed (m/s)", fontsize=9)
        ax.set_title(stat_label, fontsize=11, fontweight="bold")
        _ax_style(ax)

    handles = [
        Patch(facecolor="#888888", alpha=0.80, label="Dataset"),
        Patch(facecolor="#888888", alpha=0.35, edgecolor="black", label="Generated"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.9)
    fig.suptitle("Joint Speed Percentile Distributions over Gait Cycle",
                 fontsize=13, fontweight="bold")
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
