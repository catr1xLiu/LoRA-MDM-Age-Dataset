# Batched SMPL Optimization — Session Notes

**Date:** 2026-04-05  
**Script:** `visualize_joints/fit_seq.py` + `visualize_joints/src/smplify.py`  
**Goal:** Speed up the sequential per-frame SMPL fitting pipeline by exploiting GPU parallelism across frames.

---

## Background

The pipeline fits an SMPL body model to 22-joint 3D motion sequences using SMPLify-3D. The original code optimizes one frame at a time sequentially with LBFGS. Each frame is warm-started from the previous frame's result, which provides both good initialization and implicit temporal smoothness (via `pose_preserve_weight` in the loss).

---

## 1. Initial Parallelization Attempt

The first approach grouped frames into batches and optimized all frames in a batch simultaneously in a single SMPL forward pass.

**Key changes:**
- Renamed `--batchSize` → `--batch_size` (default 32)
- Stacked `j3d`, `body_pose`, `betas`, `camera_translation` into `[B, ...]` tensors
- Switched to Adam (LBFGS cannot parallelize across independent problems efficiently due to its sequential line search)
- Two-phase design:
  - **Phase 1:** 100 Adam iterations per batch, all frames in parallel, no temporal constraint
  - **Phase 2:** Temporal refinement with `pose_preserve_weight` anchoring batch boundaries

**Bugs fixed along the way:**
- `camera_t [B, 3]` + `model_joints [B, 45, 3]` broadcasting silently worked at B=1 but crashed at B>1 — fixed with `.unsqueeze(1)`
- Same issue in `body_fitting_loss_3d`
- `joint3d_loss [B, 22]` added to per-sample `[B]` losses — fixed with `.sum(dim=-1)`
- smplx model created with `batch_size=opt.batch_size` caused last partial batch to fail against internal `transl` buffer — fixed by using `batch_size=1` at model creation
- Adam path in `smplify.__call__` was not passing `pose_preserve_weight` to the loss (bug inherited from original code, default was 0.0 silently)
- Adam Stage 1 was double-indexing joints before passing to `camera_fitting_loss_3d`, which re-indexes internally — fixed

---

## 2. Results: Significant Jitter

Visually, the batched Adam output was noticeably more jittery than the original sequential LBFGS:

- Head position bouncing up and down between frames
- Body orientation flickering
- Limb poses inconsistent across adjacent frames

Two root causes were identified:

**A. Adam vs LBFGS convergence quality.** Adam requires many more iterations than LBFGS to reach the same solution quality on small, smooth problems like SMPL fitting. LBFGS uses curvature information and finds accurate solutions in fewer steps.

**B. Warm-start quality degraded.** The original code initialized each frame from its neighbor's result — a very close starting point. The batched approach initializes all frames from the same pose (previous batch's last frame), so frames near the end of the batch start far from their optimal solution and have less effective optimization budget.

---

## 3. Jacobi-Style Phase 2 (Second Attempt)

To address jitter, a Jacobi-style temporal refinement pass was added as phase 2.

**Concept:** Each "macro iteration" updates all frames in parallel. Frame `n` uses frame `n-1`'s result from the *previous* macro iteration as its `preserve_pose` anchor. Because each frame's anchor is from the prior iteration (already detached), all frames are independent within a macro iteration and can be batched.

```
Macro iter 0:  [F0_ph1, F1_ph1, F2_ph1, F3_ph1]   ← phase 1 results, anchors initialized
Macro iter 1:  [F0_1,   F1_1,   F2_1,   F3_1]      ← F1_1 anchored to F0_ph1, etc.
Macro iter 2:  [F0_2,   F1_2,   F2_2,   F3_2]      ← F1_2 anchored to F0_1, etc.
```

**Implementation:**
- `skip_stage1=True` parameter added to `smplify.__call__` to bypass camera re-initialization in phase 2
- 20 macro iterations × 1 Adam step each = 20 total body optimization steps per frame in phase 2
- `preserve_poses[0]` stays fixed to the previous batch's last frame throughout all macro iterations

**Results:** Slightly smoother than the pure parallel approach, but still noticeably jittier than the original sequential LBFGS output.

---

## 4. Failure Analysis

Several fundamental problems with the Jacobi approach were identified:

### 4.1 Weight imbalance makes temporal smoothing nearly ineffective
- Joint loss weight: `600²` per joint
- Preserve loss weight: `5²` total
- Ratio: `25 / 360,000 ≈ 0.007%` — the smoothness gradient is ~14,000× weaker than the joint gradient
- Adam adapts per-parameter learning rates via its second moment, which partially compensates, but the optimizer still converges to essentially the same solution as phase 1

### 4.2 Global orientation is not smoothed
`preserve_pose` is derived from `init_pose[:, 3:]` (body pose, 69 dims). `global_orient` (3 dims, root rotation) is optimized in stage 2 but **not included in the preserve loss**. Root rotation jitter is often the most visually obvious artifact and is entirely unaddressed.

### 4.3 Jacobi propagation is too slow through large batches
Temporal smoothing information from the batch boundary propagates one frame per macro iteration. With a batch of 32 frames and 20 macro iterations, only the first 20 frames can receive any boundary influence. Frames 20–31 are essentially unaffected.

### 4.4 Adam optimizer state lost between macro iterations
Each macro iteration creates a fresh Adam optimizer. With `num_iters_override=1`, there is no accumulated momentum — it degenerates to 20 independent vanilla gradient steps with adaptive learning rates, not 20 steps of Adam.

### 4.5 Phase 1 warm-start quality
All frames in a batch start from the same initialization. For a 32-frame batch covering varied motion, later frames are poorly initialized and do not converge as well as frames near the start of the batch.

---

## 5. Control Experiments (git worktree `d4876bc`)

A worktree was created from commit `d4876bc` (sequential code with working Adam) to run controlled comparisons. The worktree is at `/tmp/fit_seq_control`.

**Experiments run:**

| Run | Optimizer | Iters | ppw | Notes |
|-----|-----------|-------|-----|-------|
| `results_control_adam_ppw50` | Adam | 160 | 50 | Single-phase, sequential |
| `results_control_ppw50` | LBFGS | 80 | 50 | Single-phase, sequential |
| `results_adam_2step` | Adam | 90 + 10 | 0 → 50 | Two-phase sequential |

**Two-phase Adam timing (68 frames, GPU):**
- Phase 1 (90 iters, ppw=0): **36.2s** total — 0.532s/frame  
- Phase 2 (10 iters, ppw=50): **9.2s** total — 0.135s/frame  
- Total: **45.4s** — 0.668s/frame

Note: Phase 2 overhead is ~25% of phase 1 despite only 10/90 iterations, because `skip_stage1` was not used — stage 1 camera re-optimization runs again unnecessarily.

**Observations:**
- At ppw=50 the temporal smoothing is visually more effective than ppw=5 (the original default)
- LBFGS at 80 iterations still produces visibly smoother results than Adam at 160 iterations
- Two-phase Adam shows some jitter reduction vs single-phase, but not conclusively better

---

## 6. Suggested Next Steps

### 6.1 Parallel initial optimization + short sequential refinement

Best of both worlds: fast parallel phase 1 followed by a short sequential phase 2 that actually propagates smoothness correctly.

**Phase 1:** Run all frames in a batch in parallel with Adam for ~90 iterations, no temporal constraint. GPU-efficient, fast.

**Phase 2:** Run a sequential pass through the batch (frame by frame) for ~10 iterations each with `pose_preserve_weight=50` and `skip_stage1=True`. Each frame uses the actual current optimized result of the previous frame as its anchor — not the Jacobi approximation from a prior iteration. This propagates temporal information in a single forward pass through the batch.

```python
# Phase 2: sequential within-batch refinement
for i in range(actual_batch_size):
    prev_pose = phase1_poses[i-1] if i > 0 else prev_batch_last_pose
    phase2_poses[i] = smplify(
        phase1_poses[i], ...,
        num_iters_override=10,
        preserve_pose_override=prev_pose[:, 3:],
        pose_preserve_weight=50.0,
        skip_stage1=True,
    )
```

This costs `batch_size × 10` sequential Adam steps but each step is tiny (single frame, skip_stage1 avoids 20 camera steps), making phase 2 very fast.

### 6.2 Include `global_orient` in preserve loss

Extend `preserve_pose` to cover the full 72-dim pose (including global orientation) or add a separate term for it in `body_fitting_loss_3d`. Root rotation smoothing is critical for eliminating the head-bobbing artifact.

### 6.3 Increase `pose_preserve_weight` significantly

The current weight ratio (5 vs 600) is far too imbalanced. For temporal refinement passes, a value of 50–200 is needed to have any meaningful effect against the joint loss.

### 6.4 `skip_stage1=True` in all phase 2 calls

Stage 1 (camera + orientation, 20 Adam steps) should be skipped in any refinement pass where the camera is already calibrated from phase 1. Currently it runs redundantly in the two-phase sequential experiment, adding ~13% overhead per frame.
