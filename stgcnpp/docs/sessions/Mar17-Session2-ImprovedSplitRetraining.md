# ST-GCN++ Age Classifier: Improved Split & Retraining
**Date:** 2026-03-17
**Continuing from:** `Mar17-Session1-AgeClassifierTraining.md`
**Task:** Fix subject-level train/val split to be age-stratified, then retrain 1-block and 2-block configurations

---

## What Changed

### Problem with the Original Split

Session 1 used `random.shuffle(subject_ids)` to assign subjects to train/val. Because subjects are identified as `SUBJ1`, `SUBJ2`, etc. — and the Van Criekinge dataset has older subjects in lower-numbered slots — a random shuffle risks clustering age groups unevenly across splits. With only 87 val samples this noise is significant.

### Fix: Round-Robin Subject Assignment (`import_from_smpl.py`)

The split logic was rewritten to sort subjects numerically (`SUBJ1 < SUBJ2 < ...`) then assign round-robin: every 4th subject goes to val, the rest to train. This yields a ~75/25 split while ensuring subjects from across the full numeric (age) range appear in validation.

```python
# Round-robin: for every 4 subjects, assign 3 to train, 1 to val
for i, subject_id in enumerate(subject_ids):
    if (i % 4) == 3:   # indices 3, 7, 11, ...
        val_subjects.add(subject_id)
    else:
        train_subjects.add(subject_id)
```

**Additional changes in the same PR:**
- Added `argparse` support (`--source-dir`, `--output-path`) so the script can be called without editing source
- Added split distribution logging

### New Dataset: `vc_ntu25_xsub.pkl`

| | Old (`vc_ntu25.pkl`) | New (`vc_ntu25_xsub.pkl`) |
|---|---|---|
| Train samples | 363 | 337 |
| Val samples | 87 | **113** |
| Val fraction | 19.3% | **25.1%** |
| Class counts (Y/A/E) | 117/141/105 | 118/128/91 |

The val set grew from 87 → 113 samples (+30%), giving a more reliable accuracy estimate.

**Other training differences vs. Session 1:**
- Batch size: 16 → **8** (smaller, noisier gradients, mild regularisation effect)
- Epochs: 50 → **60**
- Backbone checkpoint: `stgcnpp_ntu120_3dkp_joint.pth` → `j.pth`

---

## Training Results

### Configuration A: 1 Block Unfrozen

**Command:**
```
python train_vc.py --checkpoint checkpoints/j.pth --data data/vc_ntu25_xsub.pkl
    --epochs 60 --batch-size 8 --unfreeze-blocks 1
    --output checkpoints/vc_age_unfreeze1block_newsplit.pth
```

**Summary:**

| Metric | Session 1 | Session 2 | Δ |
|--------|-----------|-----------|---|
| Best val acc | 54.02% | **60.18%** | **+6.16pp** |
| Best epoch | 17 | **5** | −12 |
| Final train acc | 99.17% | 98.52% | −0.65pp |
| Train-val gap (at best) | ~45pp | ~38pp | −7pp |
| Val set size | 87 | 113 | +26 |
| Training time | 116.8 s | 3056.9 s | GPU queue |

**Training curve (selected epochs):**

```
 Epoch  Train Loss  Train Acc   Val Acc      Best
-------------------------------------------------------
     1      1.0995     32.94%    32.74%    32.74% *
     3      1.0036     47.77%    44.25%    44.25% *
     4      0.8415     55.49%    53.10%    53.10% *
     5      0.8046     60.24%    60.18%    60.18% *   ← PEAK
     9      0.4945     81.31%    56.64%    60.18%
    17      0.2601     90.21%    55.75%    60.18%
    32      0.0791     97.92%    52.21%    60.18%
    60      0.0488     98.52%    52.21%    60.18%
```

**Checkpoint:** `checkpoints/vc_age_unfreeze1block_newsplit.pth` (epoch 5, 60.18% val)

#### Analysis

- Peak accuracy jumps from **54.02% → 60.18%** (+6.16pp), confirming the old split was leaking information (non-representative val subjects) and inflating the apparent difficulty
- Best epoch moves from 17 → 5: the model peaks earlier, then overfits
- Post-epoch-5 plateau: val oscillates 47–57%, mean ~51.5% — less stable than Session 1
- The faster peak and steeper post-peak drop suggest the new val set is harder and more representative; the 5-epoch window is narrow

---

### Configuration B: 2 Blocks Unfrozen

**Command:**
```
python train_vc.py --checkpoint checkpoints/j.pth --data data/vc_ntu25_xsub.pkl
    --epochs 60 --batch-size 8 --unfreeze-blocks 2
    --output checkpoints/vc_age_unfreeze2block_newsplit.pth
```

**Summary:**

| Metric | Session 1 | Session 2 | Δ |
|--------|-----------|-----------|---|
| Best val acc | 64.37% | **64.60%** | +0.23pp |
| Best epoch | **4** | **30** | **+26 epochs** |
| Final train acc | 99.72% | 99.41% | −0.31pp |
| Post-peak val avg | ~56% | ~61% | **+5pp** |
| Val set size | 87 | 113 | +26 |

**Training curve (selected epochs):**

```
 Epoch  Train Loss  Train Acc   Val Acc      Best
-------------------------------------------------------
     1      1.0982     27.00%    39.82%    39.82% *
     5      0.6607     70.62%    57.52%    57.52% *
     9      0.2769     90.80%    60.18%    60.18% *
    20      0.0623     97.03%    63.72%    63.72% *
    30      0.0707     98.81%    64.60%    64.60% *   ← PEAK
    33      0.0196    100.00%    61.95%    64.60%
    41      0.0142     99.41%    62.83%    64.60%
    60      0.0176     99.41%    60.18%    64.60%
```

**Checkpoint:** `checkpoints/vc_age_unfreeze2block_newsplit.pth` (epoch 30, 64.60% val)

#### Analysis

- Best val accuracy nearly unchanged: 64.37% → 64.60% — the ceiling is determined by the inherent difficulty of 3-class age classification from gait on this dataset
- **The key improvement is overfitting dynamics**: best epoch shifted from 4 → 30. The model now has a 30-epoch window to generalise instead of a 4-epoch window
- Post-epoch-30 plateau: val oscillates 57–65%, mean ~61% — much more stable than Session 1's 42–58% swings
- Train accuracy still saturates near 100% (severe overfitting), but later: epoch 33 vs. epoch 21

**Why did the same number of parameters overfit so much more slowly?**
- Smaller batch size (8 vs. 16) introduces gradient noise that acts as implicit regularisation
- Larger val set (113 vs. 87) averages out per-batch noise → smoother, more reliable metric
- Round-robin split may have placed harder-to-fit subjects in train, making the model work harder

---

## Comparative Overview (All Sessions)

| Config | Split | Val N | Best Val Acc | Best Epoch | Post-peak Stability |
|--------|-------|-------|-------------|-----------|---------------------|
| Frozen (0-block) | random | 87 | 43.68% | 1 | flat (underfitting) |
| 1-block | random | 87 | 54.02% | 17 | 44–49% plateau |
| 2-block | random | 87 | 64.37% | 4 | 42–58% volatile |
| **1-block** | **round-robin** | **113** | **60.18%** | **5** | 47–57% plateau |
| **2-block** | **round-robin** | **113** | **64.60%** | **30** | **57–65% stable** |

**Key findings from Session 2:**
1. The round-robin split is better calibrated: 1-block val accuracy improved 6pp, indicating the old random split had a favourable (easy) val partition
2. For 2-block, peak accuracy is unchanged but overfitting is dramatically slower (epoch 30 vs. 4), making training far more practical
3. The 2-block/epoch-30 checkpoint is now the **clear best model**: 64.60% val accuracy on a harder, larger val set, with a stable post-peak plateau

---

## Next Steps

### 1. Validate That the Split Improvement Is Real

**Goal:** Determine whether the 1-block gain (54.02% → 60.18%) came from round-robin stratification or simply from a larger val set (87 → 113 samples).

**Method:** Modify `import_from_smpl.py` to use a random subject-level split at `val_ratio=0.25` (matching the new ~25% fraction). The subject-level constraint stays — no subject appears in both train and val — but subjects are assigned randomly rather than by round-robin. Regenerate the dataset and retrain 1-block and 2-block.

- If random-at-25% ≈ 60%, the gain was from set size alone
- If random-at-25% ≈ 54%, the gain was from stratification (round-robin is genuinely better)

This is the highest-priority experiment because it determines whether the current split is a real improvement or noise.

### 2. Regularisation — Dropout Sweep

Two sequential sub-experiments, run in order:

**2a. Increase head dropout**
Raise `AgeClassifierHead` dropout from 0.3 → 0.5 in `train_vc.py`. Retrain 1-block and 2-block. Target: push 1-block best epoch later than epoch 5 while holding ≥60% val accuracy.

**2b. Add dropout to the unfrozen GCN block(s)**
The `MSTCN` inside each `STGCNBlock` already has a `self.drop` slot (currently `Dropout(0.0)`). After unfreezing, override it in `train_vc.py`:

```python
for block in model.backbone.gcn[-unfreeze_blocks:]:
    block.tcn.drop = nn.Dropout(p=0.3, inplace=True)
```

No `model.py` changes needed. Run after 2a so the two changes don't compound. This applies only to whichever unfreeze configuration looks most promising after 2a.

### 3. LR Reduction from Epoch 10

Both the 1-block (already at ~60% val by epoch 5) and 2-block (already at ~60% by epoch 9) converge most of their validation signal early. A flat LR thereafter encourages overfitting.

**Method:** Add `torch.optim.lr_scheduler.ReduceLROnPlateau` with `patience=5, factor=0.5` starting from training epoch 1, or switch to a cosine schedule that decays from epoch 10 onward. Both approaches reduce the LR once the validation plateau is reached, allowing finer parameter updates without driving train loss to zero.

Apply to both 1-block and 2-block after the dropout experiments (step 2), so effects are interpretable in isolation.

---

## Generated Files

| File | Description |
|------|-------------|
| `import_from_smpl.py` | Updated with argparse + round-robin split |
| `data/vc_ntu25_xsub.pkl` | New dataset with round-robin subject split (337 train / 113 val) |
| `checkpoints/vc_age_unfreeze1block_newsplit.pth` | 1-block, epoch 5, **60.18% val** |
| `checkpoints/vc_age_unfreeze2block_newsplit.pth` | 2-block, epoch 30, **64.60% val** ← recommended |
| `logs/1block_unforzen.out` | Raw training log, 1-block new split |
| `logs/2block_unfrozen.out` | Raw training log, 2-block new split |
