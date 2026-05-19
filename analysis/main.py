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
from analysis.gait_cycle import (
    plot_gait_cycle_angular_vel_by_age,
    plot_gait_cycle_speed_by_age,
)
from analysis.results import (
    plot_combined_bars,
    plot_combined_violin,
    plot_correlation_matrix,
    plot_joint_velocity_percentiles,
    plot_scatter_grid,
    save_summary_csv,
)

_REPO = Path(__file__).parent.parent
OUT   = _REPO / "output"
for _sub in ("dataset", "generated", "combined"):
    (OUT / _sub).mkdir(parents=True, exist_ok=True)


def _is_init_clip(clip_id: str) -> bool:
    """Return True when the first numeric token in the stem is 0."""
    for part in clip_id.split("_"):
        if part.isdigit():
            return part == "0"
    return False


def _print_rejection_summary(clips: list, label: str) -> None:
    rejected = [
        c for c in clips
        if (c.n_strides < 1 or not c.has_valid_stride_order)
        and not _is_init_clip(c.subject_id)
    ]
    if not rejected:
        return
    print(f"\n  Rejected {label} clips (bad stride quality):")
    for c in rejected:
        order = "".join(s[2] for s in c.strides) if c.strides else "(none)"
        print(f"    {c.subject_id:<40}  strides={c.n_strides}  order={order}")


def main() -> None:
    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading dataset clips...")
    dataset = load_dataset_clips()
    print(f"  {len(dataset)} clips loaded")
    for gname, (lo, hi) in AGE_BINS.items():
        print(f"  {gname}: {sum(1 for c in dataset if lo <= c.age < hi)}")
    _print_rejection_summary(dataset, "dataset")

    print("\nLoading generated clips...")
    generated = load_generated_clips()
    print(f"  {len(generated)} clips loaded")
    for ag in AGE_BINS:
        print(f"  {ag}: {sum(1 for c in generated if c.age_group == ag)}")
    _print_rejection_summary(generated, "generated")

    metrics = ALL_SCALAR_METRICS

    # ── Dataset plots ──────────────────────────────────────────────────────────
    print("\nGenerating dataset plots...")

    plot_scatter_grid(
        dataset, metrics,
        OUT / "dataset" / "age_regression.png",
        suptitle="Gait Metrics vs Age — Ground-Truth Dataset",
    )
    plot_correlation_matrix(
        dataset, metrics,
        OUT / "dataset" / "correlation_matrix.png",
        title="Gait Metric Correlation Matrix — Dataset",
    )
    plot_gait_cycle_speed_by_age(
        dataset,
        OUT / "dataset" / "gait_cycle_speed.png",
        title="Joint Speed over Gait Cycle by Age Group — Dataset",
    )
    plot_gait_cycle_angular_vel_by_age(
        dataset,
        OUT / "dataset" / "gait_cycle_angular_vel.png",
        title="Joint Angular Velocity over Gait Cycle by Age Group — Dataset",
    )

    # ── Generated plots ────────────────────────────────────────────────────────
    print("\nGenerating plots for generated clips...")

    plot_gait_cycle_speed_by_age(
        generated,
        OUT / "generated" / "gait_cycle_speed.png",
        title="Joint Speed over Gait Cycle by Age Group — Generated",
    )
    plot_gait_cycle_angular_vel_by_age(
        generated,
        OUT / "generated" / "gait_cycle_angular_vel.png",
        title="Joint Angular Velocity over Gait Cycle by Age Group — Generated",
    )

    # ── Combined plots ─────────────────────────────────────────────────────────
    print("\nGenerating combined plots...")

    plot_combined_violin(
        dataset, generated, metrics,
        OUT / "combined" / "combined_violin.png",
    )
    plot_combined_bars(
        dataset, generated, metrics,
        OUT / "combined" / "combined_bar.png",
    )
    plot_joint_velocity_percentiles(
        dataset, generated,
        OUT / "combined" / "joint_velocity_percentiles.png",
    )
    save_summary_csv(
        dataset, generated, metrics,
        OUT / "combined" / "summary_table.csv",
    )

    print(f"\nAll results in: {OUT}")


if __name__ == "__main__":
    main()
