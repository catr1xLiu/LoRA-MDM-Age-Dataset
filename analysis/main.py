#!/usr/bin/env python3
"""
Gait analysis entry point.

Loads ground-truth dataset and LoRA-generated clips, runs all scalar metrics,
and writes figures + a summary CSV to output/ (git-ignored).

Usage (from repo root):
    python -m analysis.main
"""

from pathlib import Path

from analysis.constants import AGE_BINS
from analysis.data import load_dataset_clips, load_generated_clips
from analysis.metrics import ALL_SCALAR_METRICS
from analysis.metrics.kinematics import HipROM
from analysis.metrics.spatiotemporal import WalkingSpeed
from analysis.metrics.variability import StrideTimeCV
from analysis.results import (
    plot_correlation_matrix,
    plot_group_comparison_bars,
    plot_scatter_grid,
    plot_velocity_heatmap_grid,
    plot_violin_by_age_group,
    plot_violin_comparison,
    save_summary_csv,
)

_REPO = Path(__file__).parent.parent
OUT   = _REPO / "output"
for _sub in ("dataset", "generated", "combined"):
    (OUT / _sub).mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading dataset clips...")
    dataset = load_dataset_clips()
    print(f"  {len(dataset)} clips loaded")
    for gname, (lo, hi) in AGE_BINS.items():
        print(f"  {gname}: {sum(1 for c in dataset if lo <= c.age < hi)}")

    print("\nLoading generated clips...")
    generated = load_generated_clips()
    print(f"  {len(generated)} clips loaded")
    for ag in AGE_BINS:
        print(f"  {ag}: {sum(1 for c in generated if c.age_group == ag)}")

    metrics = ALL_SCALAR_METRICS

    # ── Dataset plots ──────────────────────────────────────────────────────────
    print("\nGenerating dataset plots...")

    plot_scatter_grid(
        dataset, metrics,
        OUT / "dataset" / "age_vs_metrics_scatter.png",
        suptitle="Gait Metrics vs Age — Ground-Truth Dataset",
    )
    plot_violin_by_age_group(
        dataset, metrics,
        OUT / "dataset" / "violin_by_age_group.png",
        suptitle="Gait Metric Distribution by Age Group — Dataset",
    )
    plot_correlation_matrix(
        dataset, metrics,
        OUT / "dataset" / "correlation_matrix.png",
        title="Gait Metric Correlation Matrix — Dataset",
    )
    plot_velocity_heatmap_grid(
        [(f"{g.capitalize()} ({lo}–{hi}y)", [c for c in dataset if lo <= c.age < hi])
         for g, (lo, hi) in AGE_BINS.items()],
        OUT / "dataset" / "velocity_heatmap_by_age_group.png",
        title="Mean Joint Speed Heatmap by Age Group — Dataset",
    )

    # ── Generated plots ────────────────────────────────────────────────────────
    print("\nGenerating plots for generated clips...")

    plot_violin_by_age_group(
        generated, metrics,
        OUT / "generated" / "violin_by_age_group.png",
        suptitle="Gait Metric Distribution by Age Group — Generated",
    )
    plot_velocity_heatmap_grid(
        [(f"{g.capitalize()} Age Group", [c for c in generated if c.age_group == g])
         for g in AGE_BINS],
        OUT / "generated" / "velocity_heatmap_by_age_group.png",
        title="Mean Joint Speed Heatmap by Age Group — Generated",
    )

    # ── Combined plots ─────────────────────────────────────────────────────────
    print("\nGenerating combined plots...")

    plot_group_comparison_bars(
        dataset, generated, metrics,
        OUT / "combined" / "dataset_vs_generated_bars.png",
    )
    plot_violin_comparison(
        dataset, generated,
        [WalkingSpeed(), StrideTimeCV(), HipROM()],
        OUT / "combined" / "violin_comparison.png",
    )
    save_summary_csv(
        dataset, generated, metrics,
        OUT / "combined" / "summary_table.csv",
    )

    print(f"\nAll results in: {OUT}")


if __name__ == "__main__":
    main()
