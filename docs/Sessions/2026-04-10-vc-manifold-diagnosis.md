# VanCriekinge Manifold Diagnosis — 2026-04-10

## Context

Follow-up to the [normalization investigation](2026-04-10-normalization-investigation.md).
After fixing the 100STYLE normalization bug, LoRA adapters trained on 100STYLE are
clean, but adapters trained on VanCriekinge (VC) still produce "on drugs" jittery
motion at inference — walking forward in `sks/hta/oue` style turns into dancing-like
random movement.

Normalization was already ruled out (VC has always used HumanML3D stats).
The training loss also tells a clear story: `AgeEncoder` (VC) converges to
`loss=0.321`, more than **2× higher** than either `Old_Retrained_humanml_normalized`
(0.156) or `Old_Retrained_style_normalized` (0.179), despite identical hyperparameters.
Something about the VC data itself is harder for the base MDM to fit.

This session runs a systematic diagnostic comparing the VC 263-dim motion files
against the HumanML3D training distribution and 100STYLE-SMPL (forward walking
only), applying the **same sample filtering** `VanCriekingeDataset` uses
(`trial_index != 0`, `length >= 40`).

Diagnostic script: [`diagnose_vc_manifold.py`](diagnose_vc_manifold.py)
Output plots: [`assets/2026-04-10-vc-manifold/`](assets/2026-04-10-vc-manifold/)

Sample counts after filtering:
- VanCriekinge: **426 clips**  (138 T-pose skipped, 24 short skipped)
- HumanML3D 'walking forward' text subset: 500 clips  (full: 1500 for per-dim stats)
- 100STYLE-SMPL FW (forward walking motion type): 1500 clips

---

## TL;DR — the single most suspicious finding

**VC characters systematically float upward across each clip.** The stored
pelvis Y (dim 3) is not a noisy bounded value like it is in HumanML3D; it's
monotonically increasing in ~half the clips. This is physically impossible for
level-ground walking and is a clear artifact of a broken root-trajectory / floor-
contact step in whatever script converted the VC marker data into the
HumanML3D 263-dim format.

![Root Y drift](assets/2026-04-10-vc-manifold/10_root_y_drift.png)

![Feet levitation](assets/2026-04-10-vc-manifold/11_feet_levitation.png)

- **43.2%** of VC clips (184/426) have the pelvis rising by **> 10 cm** across the clip.
- **12.7%** (54/426) have the pelvis rising by **> 30 cm**.
- Worst clip: `SUBJ44_SUBJ44_5` — pelvis ascends from **0.89 m to 2.06 m in 61 frames (~3 seconds)**. Reconstructed feet rise from y=0 to y=1.5 m simultaneously — the character literally levitates above head-height while "walking".
- HumanML3D 'walking forward' comparator: only **4.2%** of clips have drift > 10 cm, and median drift is **−0.005 m** (≈ 0).

This alone is enough to explain the "on drugs" output. The LoRA is being trained
on data where the only consistent signal (beyond "walking forward") is
**"move the pelvis upward across the clip"**. With rank-5 capacity and a prior-
preservation loss anchoring the base model's natural flat-ground prior, the
adapter cannot learn a coherent style — it learns a warped spatial drift that
inference then reproduces as incoherent, noise-like motion. That's the root
cause. Everything else below is secondary evidence.

---

## 1. Distributional mismatch in HumanML3D-normalised space

Per-section mean and std after normalising all motions with HumanML3D Mean/Std:

```
                          VC           HumanML3D walking      100STYLE FW
section            mean     std        mean     std           mean     std
root_rot_vel      +0.02    0.74       -0.01    1.18          +0.00    1.04
root_lin_vel_xz   +1.50    2.19       +0.29    1.23          +0.34    0.67
root_y            +0.69    0.82       +0.24    0.97          +0.18    0.18
ric_data          +0.25    0.98       +0.05    1.05          +0.07    0.72
rot_data          +0.11    0.59       -0.02    1.19          -0.20    1.66
local_vel         +0.88    2.14       +0.17    1.20          +0.20    0.68
foot_contact      -0.78    1.40       -0.26    1.21          -0.24    1.19
```

**root_lin_vel_xz** and **local_vel** are the two sections where VC departs most
dramatically from both other datasets — not just shifted, but with **~2× wider
spread**. root_y is also clearly shifted (+0.69 vs +0.24). `foot_contact` is
also unusually negative (−0.78 vs −0.26) which is consistent with the floor-
contact flags being off because the feet are rarely actually touching the floor.

![Per-dim mean](assets/2026-04-10-vc-manifold/01_perdim_mean.png)

The per-dim mean plot makes the anomaly unmissable: in the `local_vel` block
(dims 193–259), VC (red) has a striking sawtooth pattern where every 3rd dim
(the Z component) sits at **+2.5** while the other two (X, Y) sit near 0. A
uniform +2.5 offset across every joint's Z velocity is a fingerprint of
common-mode root translation — every joint is moving forward together by the
same extra amount per frame, because they're all rigidly attached to the
pelvis and the pelvis is moving forward.

![Per-section histograms](assets/2026-04-10-vc-manifold/03_section_histograms.png)

The section histograms confirm: VC's `root_lin_vel_xz` and `local_vel`
distributions have heavy right tails around +2.5, while HumanML3D walking and
100STYLE FW are both sharply centred near 0.

## 2. The uniform +2.5 offset is **not** a local_vel computation bug

Internal consistency check: I verified the stored `local_vel` block matches the
stored `root_lin_vel` + `ric_data` deltas. For every VC clip tested:

```
SUBJ01_SUBJ1_1:  head stored lv_z mean = +0.0614   expected = +0.0614  ✓
                 pelvis stored lv_z     = +0.0609   root_delta_z = +0.0609  ✓
SUBJ01_SUBJ1_2:  head stored lv_z mean = +0.0608   expected = +0.0608  ✓
...
```

The 263-dim vector is internally self-consistent. So the local_vel block was
computed correctly **from its inputs** — the bug is upstream, in the root
trajectory itself.

![Per-joint local_vel mean](assets/2026-04-10-vc-manifold/04_per_joint_local_vel_mean.png)

The per-joint bar chart confirms this: every joint (pelvis through wrist) shows
the **same** +2.5 Z-offset in VC. It's not "some joints have weird velocities",
it's "the whole body is translating uniformly, 3× faster than HumanML3D walking
samples".

## 3. Per-frame root displacement is ~3× HumanML3D walking

```
VanCriekinge       median |root_lin_vel_xz| = 0.0630 m/frame  →  1.26 m/s at 20 fps
HumanML3D walking  median                   = 0.0215 m/frame  →  0.43 m/s at 20 fps
100STYLE FW        median                   = 0.0211 m/frame  →  0.42 m/s at 20 fps
```

![Walking speed distribution](assets/2026-04-10-vc-manifold/09_walking_speed_distribution.png)

**Important caveat:** VC's 1.22 m/s median walking speed is actually *correct*
for healthy adult walking (the canonical figure is 1.2–1.4 m/s). I spot-checked
`SUBJ100`'s metadata and found:

```json
"population": "able_bodied",
"walking_speeds": { "left_speed": 1.3257, "right_speed": 1.3232 }
```

which matches the stored velocity almost exactly. So VC data is **not on the
wrong fps** and **not in the wrong units** — it's genuinely capturing healthy
walking at realistic speed.

What's "wrong" is that HumanML3D's `walking forward` text subset is dominated
by hesitant/stylised/stop-start walking with a median speed of only **0.43
m/s**. And in the **full** HumanML3D training distribution (all clips, not just
walking text), VC's 0.063 m/frame displacement sits at the **97th percentile**:

```
HumanML3D full-training dim 2 (root_lin_vel_z):
  mean=+0.009  std=0.022  p90=+0.040
  fraction of frames above 0.061 (VC median): 3.04%
```

So VC walking is a 3σ outlier for the base MDM. At inference, when the adapter
is asked for "walking in sks style", the base model's conditional distribution
for the word "walking" produces slow motion, and the LoRA has to push it into a
region the base model associates with running/sprinting. With rank-5 capacity
and prior preservation fighting back, the pushed output diverges into noise.

![Raw root velocity histogram](assets/2026-04-10-vc-manifold/08_raw_root_vel_hist.png)

## 4. Sample trajectories show the drift visually

![Sample root trajectories](assets/2026-04-10-vc-manifold/05_sample_root_trajectories.png)

Top row, left: VC's raw `root_y` trajectories are monotonically rising ramps.
Top row, middle/right: HumanML3D and 100STYLE root_y oscillates around a flat
baseline. This is the most diagnostic plot in the set.

## 5. Per-dim std is suppressed everywhere *except* in velocity sections

![Per-dim std](assets/2026-04-10-vc-manifold/02_perdim_std.png)

VC is *less* varied than HumanML3D on `ric_data` and `rot_data` (because it's
just walking, not 100 diverse actions) but *more* varied on `root_lin_vel_xz`
and `local_vel` (because of inter-subject speed variability plus the drift
artifact). This combination is bad for LoRA: the adapter has high signal
magnitude in exactly the dimensions it should leave alone.

![Velocity magnitude distribution](assets/2026-04-10-vc-manifold/06_velocity_magnitude_dist.png)

---

## Ranked list of suspicious findings

| # | Finding | Severity | Notes |
|---|---|---|---|
| **1** | **Pelvis Y drifts upward across clips** (43% of VC clips drift > 10 cm; worst = +1.13 m) | **CRITICAL** | Physically impossible; points to a broken root extraction or missing per-frame floor anchoring in the VC conversion. Directly explains "on drugs" output. |
| 2 | Walking speed = 97th-percentile of HumanML3D full-train distribution | HIGH | Real walking speed is numerically correct but the base model has seen very few "walking"-captioned clips moving this fast → rank-5 LoRA + prior loss can't bridge. |
| 3 | `foot_contact` section sits at mean −0.78 vs HumanML3D's −0.26 | MED | Consistent with feet rarely actually touching the floor in the reconstructed skeleton — downstream effect of finding #1. |
| 4 | Per-dim std ~2× HumanML3D on velocity sections | MED | Partially attributable to inter-subject variability (ages 20–95, mixed populations), partially to the drift artifact. |
| 5 | Worst-case clips have physically impossible states (legs compress from 0.89 m to 0.56 m of span) | HIGH | Some clips are corrupted end-to-end, not just drifting — should be filtered out of training regardless of the root cause. |
| 6 | VC median root_y = 0.99 m vs HumanML3D 0.94 m (+5 cm) | LOW | On its own this is small, but it's the *median* — the *max* is 2.06 m because of drift. |

---

## The most issue-looking thing

**It is finding #1: the systematic upward drift of `root_y` across VC clips.**

This is not a distributional mismatch, it's a **data corruption fingerprint**.
The VC → HumanML3D converter is either:

1. **Missing a per-frame floor-contact re-anchoring step.** HumanML3D's
   `process_file` only grounds the lowest foot *once* per clip (a single
   translation). Any internal drift in the raw root trajectory passes through
   unchanged. If VC raw data was captured in a world coordinate system where
   the subject walked across a capture volume and the marker-based root
   estimate had a small monotonic bias (e.g., SACR marker drift, or
   accumulation from inertial integration), that bias stays in the output.

2. **Broken root extraction from marker data.** The metadata lists markers
   (`LASI`, `RASI`, `SACR`, `PELO`, etc.). If the converter estimates the SMPL
   pelvis from these markers and one of them has tracking loss or occlusion,
   the estimated pelvis Y can drift progressively. The fact that some clips
   drift by **+1.1 m** in 3 seconds strongly suggests tracking failure in a
   subset of clips, not a uniform per-clip bias.

3. **Accumulated integration error.** If the converter reconstructs pelvis
   position by integrating velocity (rather than using the raw marker-derived
   position), a small systematic bias in velocity compounds into large position
   drift — exactly what we observe.

### What I'd do before retraining any VC LoRA

1. **Filter out drifting clips.** As a sanity measure, reject any clip where
   `abs(root_y[-1] - root_y[0]) > 0.05` *or* where the recovered lowest foot Y
   exceeds 0.1 m at any point. This will drop ~40% of the current 426 clips
   but the remainder should at least be trainable. Re-run `AgeEncoder` on the
   filtered subset first to confirm the LoRA output becomes usable.
2. **Then** investigate the conversion pipeline proper — pull up whatever
   script produced `*_humanml3d_22joints.npy` and check the pelvis-extraction
   and floor-anchoring steps. The fix probably belongs upstream, not in the
   dataset loader.
3. Separately (lower priority): once the drift is fixed, the walking-speed
   mismatch (finding #2) may still cause training friction. If so, increase
   `lora_rank` from 5 to 16–32, or lower `lambda_prior_preserv` from 1.0 to
   ~0.3, to give the adapter enough capacity to bridge the distribution gap.

---

## Generated artifacts

| File | Description |
|---|---|
| `01_perdim_mean.png` | Per-dim mean across 263 dims, VC vs HML walking vs 100STYLE FW. Shows the local_vel sawtooth pattern. |
| `02_perdim_std.png` | Per-dim std across 263 dims. |
| `03_section_histograms.png` | Histograms of normalised values per HumanML3D feature section. |
| `04_per_joint_local_vel_mean.png` | Per-joint (22 joints × 3 axes) bar chart of local_vel mean. Every joint's Z component is pinned at +2.5 for VC. |
| `05_sample_root_trajectories.png` | Raw `root_y` and `|root_lin_vel_xz|` trajectories for 8 sample clips per dataset. |
| `06_velocity_magnitude_dist.png` | Distribution of per-frame mean joint-velocity magnitude. |
| `07_raw_root_y_hist.png` | Histogram of raw `root_y` values. |
| `08_raw_root_vel_hist.png` | Histogram of raw `|root_lin_vel_xz|` per frame. |
| `09_walking_speed_distribution.png` | Per-clip mean forward speed (m/s at 20 fps), VC vs HML walking vs HML running. |
| `10_root_y_drift.png` | **Per-clip net root_y drift distribution and anchored root_y trajectories.** |
| `11_feet_levitation.png` | **Reconstructed lowest-foot Y over time — VC feet levitate, HumanML3D feet stay on the floor.** |

Plots 10 and 11 are the key evidence; plots 1, 4, 5, and 9 provide the
supporting context.
