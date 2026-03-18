# ST-GCN++ Age Classifier: Per-Class Accuracy Analysis
**Date:** 2026-03-17
**Continuing from:** `Mar17-Session2-ImprovedSplitRetraining.md`
**Task:** Analyse per-class accuracy and rough accuracy before moving to next training experiments

---

## New Tool: `batch_age_inference.py`

A dedicated inference script was written for the age classification task, separate from the existing `batch_inference.py` (which is for NTU-120 action recognition).

### What it does

- Loads a fine-tuned age classifier checkpoint saved by `train_vc.py`
- Reports **per-class accuracy** (Young / Adult / Elderly) and a **confusion matrix**
- Reports **rough accuracy**: samples within ±3 years of an age boundary may be classified as either adjacent group and still count as correct

### Rough accuracy definition

Age group boundaries: Young < 40, Adult 40–64, Elderly ≥ 65.

| Tolerance zone | True label | Acceptable preds |
|---|---|---|
| age 37–39 | Young | Young **or** Adult |
| age 40–42 | Adult | Adult **or** Young |
| age 62–64 | Adult | Adult **or** Elderly |
| age 65–67 | Elderly | Elderly **or** Adult |

Age values are looked up from the original SMPL json files (`subject_metadata.age`), matched to each sample via its `frame_dir` subject ID prefix.

Usage:
```bash
uv run python batch_age_inference.py \
    --checkpoint checkpoints/vc_age_unfreeze2block_newsplit.pth \
    --data       data/vc_ntu25_xsub.pkl \
    --smpl-dir   ../data/fitted_smpl_all_3_tuned
```

---

## Results

### 1-Block Model (`vc_age_unfreeze1block_newsplit.pth`, epoch 5, 60.18% val)

**Per-class accuracy:**

| Class | Correct | Total | Accuracy |
|---|---|---|---|
| Young (<40) | 24 | 37 | 64.86% |
| Adult (40–64) | 30 | 44 | 68.18% |
| Elderly (≥65) | 14 | 32 | **43.75%** |
| **Overall** | **68** | **113** | **60.18%** |

**Confusion matrix (rows = true, cols = pred):**

```
               Young  Adult  Elderly
Young  (<40)     24     12        1
Adult  (40-64)    9     30        5
Elderly (≥65)    10      8       14
```

**Rough accuracy (±3 yr):**

| Class | Rough OK | Total | Rough Acc |
|---|---|---|---|
| Young (<40) | 24 | 37 | 64.86% |
| Adult (40–64) | 30 | 44 | 68.18% |
| Elderly (≥65) | 17 | 32 | **53.12%** |
| **Overall** | **71** | **113** | **62.83%** |

Boundary zone: Near-40 3/3 (100%), Near-65 **4/7 (57.1%)**

---

### 2-Block Model (`vc_age_unfreeze2block_newsplit.pth`, epoch 30, 64.60% val)

**Per-class accuracy:**

| Class | Correct | Total | Accuracy |
|---|---|---|---|
| Young (<40) | 27 | 37 | 72.97% |
| Adult (40–64) | 24 | 44 | **54.55%** |
| Elderly (≥65) | 22 | 32 | 68.75% |
| **Overall** | **73** | **113** | **64.60%** |

**Confusion matrix (rows = true, cols = pred):**

```
               Young  Adult  Elderly
Young  (<40)     27      5        5
Adult  (40-64)    7     24       13
Elderly (≥65)     6      4       22
```

**Rough accuracy (±3 yr):**

| Class | Rough OK | Total | Rough Acc |
|---|---|---|---|
| Young (<40) | 27 | 37 | 72.97% |
| Adult (40–64) | 24 | 44 | 54.55% |
| Elderly (≥65) | 24 | 32 | **75.00%** |
| **Overall** | **75** | **113** | **66.37%** |

Boundary zone: Near-40 3/3 (100%), Near-65 **7/7 (100.0%)**

---

## Comparative Summary

| Metric | 1-block | 2-block | Δ |
|--------|---------|---------|---|
| Overall acc | 60.18% | **64.60%** | +4.42pp |
| Young acc | 64.86% | **72.97%** | +8.11pp |
| Adult acc | **68.18%** | 54.55% | −13.63pp |
| Elderly acc | 43.75% | **68.75%** | +25pp |
| Rough overall | 62.83% | **66.37%** | +3.54pp |
| Near-65 rough | 4/7 (57%) | **7/7 (100%)** | perfect |

---

## Analysis

### 1-block failure mode: Elderly collapse
The 1-block model has severe Elderly under-recognition (43.75%). Its confusion matrix shows 10 Elderly samples classified as Young and 8 as Adult — nearly as many incorrect as correct. The model may not have enough capacity to distinguish elderly gait patterns, so it defaults to the majority classes (Adult/Young). This is a capacity issue, not a data issue.

### 2-block failure mode: Adult fragmentation
The 2-block model's weakness is the Adult class (54.55%). It pushes 13 Adult samples into Elderly and 7 into Young. The Adult group spans 25 years (40–64) — a wide range that contains subjects at very different functional ages. Some 60-year-old adults walk with elderly-like gait, and some 40-year-olds with young-like gait.

The 2-block model essentially learns **a bimodal gait polarisation**: it excels at recognising extreme gait patterns (Young 73%, Elderly 69%) but struggles with the ambiguous middle group. This is arguably a **better representation** of the biological reality than the 1-block model's flat Adult-majority bias.

### Rough accuracy reveals near-65 boundary quality
The most revealing metric is the near-65 boundary zone:
- 1-block: 4/7 (57%) — misses 3 borderline Elderly subjects
- 2-block: 7/7 (100%) — all borderline Elderly subjects classified as Elderly or Adult

This means the 2-block model has correctly identified the gait characteristics that distinguish elderly from younger subjects near the clinical threshold. The adult fragmentation may be a worthwhile trade-off for this boundary sensitivity.

### Rough accuracy delta is modest
Rough accuracy gains over standard accuracy are +2.65pp (1-block) and +1.77pp (2-block). The dataset has few true boundary subjects (only 3 near-40, 7 near-65 in the val set), so rough accuracy is a good diagnostic for edge-case quality but not a strong differentiator at this dataset size.

---

## Implications for Next Steps

The adult fragmentation in the 2-block model is the primary target for improvement. The planned experiments remain valid and prioritised as follows:

### Most impactful: regularisation to reduce Adult→Elderly over-prediction
The 13 Adult samples classified as Elderly suggest the model is biased toward elderly gait features in ambiguous cases. Dropout experiments (Session 2 plan, step 2a/2b) may reduce this by slowing the model's convergence to a bimodal solution.

### Stratified class weighting may help Adult recall
Currently, class weights are computed by inverse frequency from the training split. Given that Adult is the largest class (≈128 samples) it has the smallest weight, which may contribute to Adult being "crowded out" by the extremes. Increasing the Adult weight explicitly (or running experiments with equal weights) could improve Adult recall at the cost of some Young/Elderly accuracy.

### Confusion-aware evaluation
For the LoRA-MDM downstream task (conditioning motion generation on age), the 2-block bimodal failure mode may be tolerable: Young and Elderly conditioning signals are the most important for motion diversity, while Adult is an intermediate. The near-65 boundary precision (7/7 rough correct) is particularly useful.

---

## Generated Files

| File | Description |
|------|-------------|
| `batch_age_inference.py` | Per-class + rough accuracy inference script |
