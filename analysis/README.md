# analysis/

Gait analysis pipeline for the LoRA-MDM age dataset.  
Compares spatiotemporal parameters, gait variability, and joint ROM between the Van Criekinge ground-truth dataset and LoRA-generated motion clips.

All outputs (figures, CSV) are written to `output/` at the repo root (git-ignored).

---

## Directory structure

```
analysis/
├── classes.py        — MotionClip dataclass and BaseMetric ABC
├── constants.py      — joint indices, FPS, age bins, colours
├── data.py           — axis alignment, stride detection, data loaders
├── metrics/
│   ├── spatiotemporal.py  — WalkingSpeed, StrideLength, StrideTime, Cadence
│   ├── variability.py     — StrideTimeCV, StrideLengthCV
│   ├── kinematics.py      — KneeROM, HipROM
│   └── velocity_map.py    — VelocityMap (non-scalar, used by results.py)
├── results.py        — compute_stats, compute_group_stats, all plot functions, save_summary_csv
├── main.py           — entry point; run with `python -m analysis.main`
└── tests/
    ├── diagnose_strides.py    — per-clip stride count and step-order table
    └── plot_stride_curves.py  — foot height curves with detected strikes overlaid
```

---

## Data

| Source | Path | Format |
|---|---|---|
| Ground-truth | `data/humanml3d_new_joints_6/*.npy` | `(T, 22, 3)` joint positions |
| Generated | `data/generated_clips_humanml3d/<motion>/<group>/batch_*.npy` | dict with `motion (B,22,3,T)` and `lengths (B,)` |
| Metadata | `data/van_criekinge_unprocessed_1/metadata.py` | per-subject age, sex, condition |

All clips are axis-aligned on load so forward motion is along **+X**, vertical is **+Y**.

---

## Design

### MotionClip

The central data object. Created by the loaders in `data.py` and passed to every metric and plot function.

```python
@dataclass
class MotionClip:
    joints:     np.ndarray              # (T, 22, 3)
    timestamps: np.ndarray              # (T,) seconds
    strides:    list[tuple[int, int]]   # [(t_start, t_end), ...] detected gait cycles
    subject_id: str
    source:     str                     # "dataset" | "generated"
    age:        float
    age_group:  str                     # "young" | "mid" | "old"
    sex:        str
    condition:  str
```

Helper methods on the clip: `get_stride(i)`, `stride_duration(i)`, `stride_length(i)`, `normalize_stride(i, n_pts)`, `normalized_gait_cycle(n_pts)`.

### Metrics

Each metric is a stateless callable class implementing `BaseMetric`:

```python
class BaseMetric(ABC):
    title: str
    unit:  str
    def __call__(self, clip: MotionClip) -> float: ...
```

`VelocityMap` is the only non-scalar metric (returns `(N_GAIT_PTS, 22)`); it is handled separately by `plot_velocity_heatmap_grid`.

### Statistics and plotting

`results.py` provides:
- `compute_stats(clips, metric) → MetricStats(mean, std, values, n)`
- `compute_group_stats(clips, metric) → dict[group, MetricStats]`
- Unified plot functions that accept `list[MotionClip]` and `list[BaseMetric]`

### Running

```bash
# from repo root
python -m analysis.main

# diagnostic scripts
python analysis/tests/diagnose_strides.py
python analysis/tests/plot_stride_curves.py
```

---

## Known issues — stride detection

### Method

Strides are detected in `data.py` via `_heel_strikes` + `detect_strides`:

1. **Contact detection** — a foot frame is "in contact" if:
   - height above per-clip 2nd-percentile floor < **5 cm**, and
   - foot speed < **1.0 m/s**
2. **Heel strikes** — rising edges (False → True) of the contact signal, requiring ≥ 3 consecutive contact frames.
3. **Stride filtering** — consecutive ipsilateral strikes are accepted if stride duration is 0.4–3.0 s and pelvis displacement is 0.1–3.0 m.

### Problem: frame-0 false positives

When a clip starts with the subject already standing (both feet on the ground), the contact signal is `True` from frame 0.  The rising edge is captured at frame 0, producing a heel strike that is not a real landing event.

- In **short clips** (~0.8–1.2 s, always the `_0` segment per subject) this is the only detected strike per foot, so no valid stride pair is formed → **0 strides**.
- In **longer clips** the frame-0 strike is still spurious but subsequent real strikes produce valid pairs, so detection is otherwise correct.

### Scale of the problem (588 dataset clips)

| Strides | Clips |
|---|---|
| 0 | 160 (27 %) |
| 1 | 14 |
| ≥ 2 (healthy) | 414 |

Nearly all 160 zero-stride clips are the `_0` initialization segment, identifiable by `L@0 R@0` as the only detected events and duration < ~1.5 s.

### Impact on metrics

`WalkingSpeed`, `KneeROM`, and `HipROM` do not require stride detection, so they return a value for every clip including the zero-stride ones.  Zero-stride clips tend to have near-zero pelvis displacement (speed → 0 m/s) and standing-posture joint angles (ROM → low), pulling the lower tail of every distribution down compared to the previous analysis which aggregated per-subject and gated on stride presence.

### Proposed fix

Gate all metrics on `clip.n_strides >= 1` so that clips containing no detected walking are excluded from every metric — equivalent to the implicit filter in the previous per-subject aggregation.  The stride detection algorithm itself is well-tuned for in-clip walking; the issue is purely at clip boundaries.
