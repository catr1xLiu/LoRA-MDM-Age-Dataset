#!/usr/bin/env python3
"""
Stride detection diagnostic.

For every dataset clip prints the raw heel-strike sequence (L/R frame indices),
the interleaved step order, validated strides, and flags clips with fewer than
2 strides as potentially problematic.

Output is written to stdout and mirrored to output/stride_diagnostic.txt.
"""

from pathlib import Path

import numpy as np

from analysis.constants import AGE_BINS, DT, L_FOOT, R_FOOT
from analysis.data import (
    _DATA,
    _align_forward_axis,
    _heel_strikes,
    _load_metadata,
    detect_strides,
)

import glob

OUT = Path("output")
OUT.mkdir(exist_ok=True)

_MIN_EXPECTED_STRIDES = 2


def analyse_clip(
    joints: np.ndarray,
    subject_id: str,
    clip_name: str,
    age: float,
) -> dict:
    T = joints.shape[0]
    l_strikes = _heel_strikes(joints, L_FOOT)
    r_strikes = _heel_strikes(joints, R_FOOT)
    strides   = detect_strides(joints)

    # Interleaved step order
    events = (
        [(f, "L") for f in l_strikes] +
        [(f, "R") for f in r_strikes]
    )
    events.sort(key=lambda x: x[0])
    step_order = "  ".join(f"{side}@{f}" for f, side in events)

    return {
        "subject_id": subject_id,
        "clip":       clip_name,
        "age":        age,
        "frames":     T,
        "duration_s": round(T * DT, 2),
        "l_strikes":  len(l_strikes),
        "r_strikes":  len(r_strikes),
        "n_strides":  len(strides),
        "step_order": step_order if step_order else "(none)",
        "strides":    strides,
    }


def main() -> None:
    joints_dir    = _DATA / "humanml3d_new_joints_6"
    metadata_path = _DATA / "van_criekinge_unprocessed_1" / "metadata.py"
    metadata      = _load_metadata(metadata_path)

    files = sorted(glob.glob(str(joints_dir / "*.npy")))

    results = []
    for f in files:
        fp  = Path(f)
        sid = fp.stem.split("_")[0]
        if sid not in metadata:
            continue
        age = metadata[sid].get("age")
        if age is None:
            continue
        raw    = np.load(f)
        joints = _align_forward_axis(raw)
        results.append(analyse_clip(joints, sid, fp.stem, float(age)))

    # ── Summary counts ──────────────────────────────────────────────────────
    total         = len(results)
    zero_strides  = [r for r in results if r["n_strides"] == 0]
    one_stride    = [r for r in results if r["n_strides"] == 1]
    healthy       = [r for r in results if r["n_strides"] >= _MIN_EXPECTED_STRIDES]

    lines = []
    def p(*args):
        line = " ".join(str(a) for a in args)
        print(line)
        lines.append(line)

    p("=" * 90)
    p(f"STRIDE DETECTION DIAGNOSTIC   ({total} clips, {len(zero_strides)} with 0 strides, "
      f"{len(one_stride)} with 1 stride, {len(healthy)} healthy (≥{_MIN_EXPECTED_STRIDES}))")
    p("=" * 90)

    # ── Per-clip table ──────────────────────────────────────────────────────
    p(f"\n{'CLIP':<35} {'AGE':>4}  {'FRM':>4}  {'DUR':>5}  {'L':>3}  {'R':>3}  {'STR':>3}  STEP ORDER")
    p("-" * 90)
    for r in results:
        flag = "  <<<" if r["n_strides"] < _MIN_EXPECTED_STRIDES else ""
        p(f"{r['clip']:<35} {r['age']:>4.0f}  {r['frames']:>4}  {r['duration_s']:>5.1f}s"
          f"  {r['l_strikes']:>3}  {r['r_strikes']:>3}  {r['n_strides']:>3}"
          f"  {r['step_order']}{flag}")

    # ── Problem clips grouped by age group ──────────────────────────────────
    p("\n" + "=" * 90)
    p("PROBLEMATIC CLIPS (< 2 strides) BY AGE GROUP")
    p("=" * 90)
    for gname, (lo, hi) in AGE_BINS.items():
        bad = [r for r in results if lo <= r["age"] < hi and r["n_strides"] < _MIN_EXPECTED_STRIDES]
        p(f"\n  {gname.upper()} ({lo}–{hi}y) — {len(bad)} problematic clips:")
        for r in bad:
            p(f"    {r['clip']:<35}  strides={r['n_strides']}  "
              f"L={r['l_strikes']} R={r['r_strikes']}  steps: {r['step_order']}")

    # ── Stride count histogram ──────────────────────────────────────────────
    p("\n" + "=" * 90)
    p("STRIDE COUNT DISTRIBUTION")
    p("=" * 90)
    max_n = max(r["n_strides"] for r in results)
    for n in range(max_n + 1):
        count = sum(1 for r in results if r["n_strides"] == n)
        bar   = "█" * count
        p(f"  {n:>2} strides: {count:>4}  {bar}")

    out_path = OUT / "stride_diagnostic.txt"
    out_path.write_text("\n".join(lines))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
