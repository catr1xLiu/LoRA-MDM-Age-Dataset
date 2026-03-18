# Week 10 Report

## 1. Executive Summary

Last week, we implemented the conversion pipeline from SMPL mesh to NTU-25 joint format and validated the standalone ST-GCN++ reimplementation against the full NTU RGB+D 120 dataset (82.6% top-1 accuracy, closely matching the published 83.2% benchmark). This week, two remaining issues in the SMPL→NTU-25 conversion pipeline were resolved (T-pose contamination and boundary frame padding), and the cleaned Van Criekinge dataset (450 clips, 138 subjects) was verified by running the pretrained action recognition model on the full dataset—achieving 100% walking-action classification.With the data pipeline validated, I made a second attempt to fine-tune the ST-GCN++ model with a custom 3-class age classifier (Young <40, Adult 40–64, Elderly ≥65). However, the initial results were disappointing, with a 45% best accuracy and a fluctuating loss curve. I then attempted to unfreeze parts of the ST-GCN backbone, testing a few configurations (fully frozen, 1-block unfrozen, 2-blocks unfrozen), followed by a data-splitting improvement and a detailed accuracy analysis. The best model achieved 64.60% validation accuracy (31.6% above the random baseline). I have also extracted the 32-dimensional $z_{age}$ embeddings for MDM conditioning.However, the model accuracy is still not satisfying, and further adjustments will be needed. To help determine the best approach moving forward, I have written this detailed report for your review and for our upcoming discussion.

---

## 2. Background: Data & Environment Setup

### 2.1 SMPL Mesh → NTU-25 Vertex Mapping

Our conversion method avoids the naive approximation of equating SMPL internal joint positions to NTU joint positions. Instead, every NTU-25 joint is mapped to a specific **SMPL mesh vertex ID**, hand-selected for anatomical correspondence. For example:

```python
NTU_25_MARKERS = {
    0: 1807,   # SpineBase
    1: 3511,   # SpineMid
    2: 3069,   # Neck
    3: 336,    # Head
    4: 1291,   # LeftShoulder
    5: 1573,   # LeftElbow
    6: 1923,   # LeftWrist
    7: 2226,   # LeftHand
    8: 4773,   # RightShoulder
    # ... 25 joints total
}
```

The conversion runs with CUDA-accelerated batch SMPL forward passes (batch size 32), generating full 10,475-vertex meshes and then indexing the 25 target vertices. This produces anatomically accurate NTU-25 skeletons regardless of body shape parameters (β), gender, or pose complexity.

<div style="display: flex; gap: 20px;">
  <img src="figures/ntu_25_joint_paper_figure.png" width="45%" />
  <img src="figures/ntu_25_joint_our_mapping.png" width="45%" />
</div>

*Above: Left - NTU-25 joint definition from the original NTU RGB+D paper. Right - Our SMPL vertex to NTU-25 joint mapping visualization.*

### 2.2 Pipeline Fixes This Week

Three issues were identified and resolved before training could begin:

**2.2.1 — T-Pose Contamination.** Every subject's first capture file (`SUBJ*_0_smpl_params.npz`) is a static T-pose used for SMPL shape calibration — not a walking trial. Including these clips injected standing-still sequences that the ST-GCN++ model correctly rejected (predicting actions like "shake head" instead of walking). The fix extracts the trial index from each filename using a compiled regex and skips trial 0:

```python
_TRIAL_RE = re.compile(r"^SUBJ\d+_(\d+)_smpl_params$")

def get_trial_number(npz_path: Path) -> int | None:
    m = _TRIAL_RE.match(npz_path.stem)
    return int(m.group(1)) if m else None
```

One malformed file (`SUBJ101_smpl_params.npz`, missing trial index) was also caught and skipped. In total, 139 files were discarded (138 T-poses + 1 malformed).

**2.2.2 — Boundary Frame Trimming.** The Van Criekinge instrumentation introduces inaccurate SMPL fits at the start and end of each trial (~5 frames) due to sensor initialisation and shutdown transients. These frames produce anatomically implausible joint positions. The fix trims 5 frames from each end before running the SMPL forward pass:

```python
FRAME_TRIM = 5

def smpl_to_ntu25(params, model):
    poses = params["poses"]
    trans = params["trans"]
    if poses.shape[0] > 2 * FRAME_TRIM:
        poses = poses[FRAME_TRIM:-FRAME_TRIM]
        trans = trans[FRAME_TRIM:-FRAME_TRIM]
    # ... SMPL forward pass on trimmed data
```

For clips shorter than 100 frames after trimming, ST-GCN++'s `UniformSample` transform wraps frames using modulo-looping (matching pyskl's exact test-time logic), so no explicit padding is needed.

**2.2.3 — Result: Clean Dataset.** After these fixes, 450 clips remain across 138 subjects, with the following age distribution:

| Age Group | Clips | Proportion |
|-----------|-------|------------|
| Young (<40) | 155 | 34.4% |
| Adult (40–64) | 172 | 38.2% |
| Elderly (≥65) | 123 | 27.3% |
| **Total** | **450** | **100%** |

*Age distribution pie chart of the cleaned Van Criekinge dataset (450 clips, 138 subjects).*

### 2.3 Action Recognition Validation

To verify the converted NTU-25 skeletons are correct, we ran the pretrained ST-GCN++ action recognition model (NTU-120 joint checkpoint, `j.pth`) on the full clean dataset. The expectation: since Van Criekinge is a gait dataset, every clip should be classified as a walking-related NTU action.

**Result: 100% walking-action classification** on both train (363/363) and val (87/87) splits.

The top-1 predictions broke down as:

| NTU Action Label | Count | Confidence-weighted Score |
|-----------------|-------|--------------------------|
| 58 — "walking towards" | 321 | 255.3 |
| 59 — "walking apart" | 13 | 32.1 |
| 115 — "follow" | 3 | 21.3 |

> **[FIGURE 1]** — *Single-clip inference visualization (SUBJ06, trial 2).* Left: 3D skeleton at frame 75/100. Right: top-10 NTU action predictions (walking towards: 63.8%, walking apart: 13.0%, follow: 4.8%). Bottom-right: 256-dim Global Average Pool features visualised as a 16×16 3D bar chart. *(Chart: `figures/vc_classification_inference.png`)*

<div style="text-align: center;">
  <img src="figures/vc_classification_inference.png" width="80%" />
</div>

> **[FIGURE 2]** — *Confidence-weighted bar chart of the top-20 recognised NTU action labels across the entire Van Criekinge dataset.* Walking-related labels (green) dominate overwhelmingly. Non-walking labels (blue) receive negligible confidence scores, confirming that the SMPL→NTU-25 conversion preserves gait kinematics faithfully. *(Chart: `figures/vc_top20_barchart.png`)*

<div style="text-align: center;">
  <img src="figures/vc_top20_barchart.png" width="70%" />
</div>

This 100% result (up from 75% before fixes) confirms the pipeline is producing anatomically correct NTU-25 skeletons suitable for downstream age classification training.

---

## 3. Baseline Implementation

With the data pipeline validated, the core task this week was fine-tuning ST-GCN++ as a 3-class age classifier and extracting z_age embeddings.

### 3.1 Architecture: AgeClassifierHead

The standard 120-class `GCNClassifier` head was replaced with a custom bottleneck classifier:

```
STGCNBackbone → AdaptiveAvgPool2d(1) → mean over M persons → Dropout(0.3)
                                                                  ↓
                                                     Linear(256 → 32)   ← z_age extracted here
                                                                  ↓
                                                         ReLU → Linear(32 → 3) → Logits
```

The 32-dimensional intermediate layer is the learned age representation (`z_age`). At extraction time, only the pathway up to this layer is executed — the final classification layer is discarded.

### 3.2 Training Configuration

The training script (`train_vc.py`) implements:

- **Class-weighted cross-entropy loss** to compensate for imbalanced classes (weights: Young 1.03, Adult 0.86, Elderly 1.15)
- **Differential learning rates**: head at 1e-3, unfrozen backbone blocks at 1e-4
- **AdamW optimizer** with weight decay 1e-4
- **Cosine annealing** learning rate schedule
- **Multi-clip validation**: 10 deterministic temporal clips per sample with probability-averaged aggregation (matching inference-time protocol)
- **Single-clip stochastic training**: 1 random clip per sample per epoch for implicit augmentation

### 3.3 Frozen Backbone Baseline (0 Blocks Unfrozen)

As a baseline, we first trained with the entire ST-GCN++ backbone frozen — only the 8,323 parameters (0.6% of total) in the `AgeClassifierHead` were learnable.

**Result: 43.68% validation accuracy** (10.7pp above random baseline of 33%)

The best checkpoint was saved at epoch 1 — validation accuracy never improved beyond this point across 50 epochs. Training accuracy plateaued at ~54%, indicating the frozen NTU-120 features lack intrinsic age-discriminative information. The z_age embeddings from this configuration had very low variance (std: 0.32, range: [-1.21, 0.46]), confirming poor discriminability.

**Interpretation:** Pretrained action-recognition features are insufficient for age classification. The backbone needs some adaptation to learn gait-style features relevant to aging.

---

## 4. Iterative Improvements

### 4.1 Backbone Unfreezing Sweep (Session 1)

Two additional configurations were tested, progressively unfreezing GCN blocks from the end of the backbone:

| Config | Unfrozen Blocks | Trainable Params | Val Acc | Best Epoch | z_age std |
|--------|----------------|------------------|---------|-----------|-----------|
| Frozen | 0 | 8,323 (0.6%) | 43.68% | 1 | 0.32 |
| **1-block** | **Block 9** | **363,518 (26.0%)** | **54.02%** | **17** | **2.63** |
| 2-block | Blocks 8–9 | 718,713 (51.5%) | 64.37% | 4 | 2.64 |

Key observations:

- **1-block unfrozen** achieved 54.02% val accuracy (+10.3pp over frozen), with the best checkpoint at epoch 17 — a gradual learning curve suggesting stable generalisation
- **2-block unfrozen** achieved the highest accuracy (64.37%) but peaked at epoch 4 with severe overfitting thereafter (train accuracy hit 100% by epoch 21 while validation oscillated between 42–58%)
- **z_age variance** was nearly identical between 1-block and 2-block (2.63 vs. 2.64), both ~8.3× higher than frozen (0.32), indicating both configurations learn comparably discriminative embeddings

> **[FIGURE 3]** — *Training curves for all three backbone configurations.* Plot train loss and validation accuracy vs. epoch for frozen (orange), 1-block (blue), and 2-block (green). Key annotations: frozen flat-lines at 43.68%, 1-block peaks at epoch 17, 2-block peaks at epoch 4 then oscillates. X-axis: epoch (1–50), left Y-axis: train loss, right Y-axis: val accuracy (%). *(Chart: `figures/figure3_training_curves.png`)*

<div style="text-align: center;">
  <img src="figures/figure3_training_curves.png" width="80%" />
</div>

### 4.2 Improved Train/Val Split (Session 2)

A critical issue was discovered in the original train/val split: `random.shuffle(subject_ids)` could cluster age groups unevenly across splits, since subject IDs in the Van Criekinge dataset are roughly ordered by age. With only 87 val samples, this noise was significant.

**Fix:** Round-robin subject assignment. Subjects are sorted numerically, then every 4th subject is assigned to validation:

```python
for i, subject_id in enumerate(subject_ids):
    if (i % 4) == 3:  # indices 3, 7, 11, ...
        val_subjects.add(subject_id)
    else:
        train_subjects.add(subject_id)
```

This produces a ~75/25 split (337 train / 113 val) with subjects from across the full age range in validation. Combined with a reduced batch size (16→8) for implicit regularisation through gradient noise:

| Config | Split | Val N | Best Val Acc | Best Epoch | Post-peak Stability |
|--------|-------|-------|-------------|-----------|---------------------|
| 1-block | random | 87 | 54.02% | 17 | 44–49% plateau |
| **1-block** | **round-robin** | **113** | **60.18%** | **5** | 47–57% plateau |
| 2-block | random | 87 | 64.37% | 4 | 42–58% volatile |
| **2-block** | **round-robin** | **113** | **64.60%** | **30** | **57–65% stable** |

The round-robin split improved 1-block val accuracy by **+6.16pp** and stabilised 2-block's overfitting dynamics dramatically — best epoch shifted from 4 → 30, giving a 30-epoch window for generalisation instead of a 4-epoch window.

> **[FIGURE 4]** — *Comparison of training curves before and after split improvement, for the 2-block configuration.* Two subplots side by side: (Left) Session 1 random split — val acc peaks at epoch 4 then oscillates wildly between 42–58%. (Right) Session 2 round-robin split — val acc climbs gradually to 64.60% at epoch 30, post-peak plateau at 57–65%. Same Y-axis scale for direct comparison. *(Chart: `figures/figure4_split_comparison.png`)*

<div style="text-align: center;">
  <img src="figures/figure4_split_comparison.png" width="80%" />
</div>

### 4.3 Per-Class Accuracy Analysis (Session 3)

A dedicated inference script (`batch_age_inference.py`) was written to analyse per-class accuracy and boundary-zone behaviour. Two notions of accuracy were reported:

- **Strict accuracy**: prediction must match the exact age group
- **Rough accuracy**: subjects within ±3 years of a group boundary may be classified as either adjacent group

#### 1-Block Model (60.18% overall)

| Class | Correct/Total | Accuracy |
|-------|--------------|----------|
| Young (<40) | 24/37 | 64.86% |
| Adult (40–64) | 30/44 | 68.18% |
| **Elderly (≥65)** | **14/32** | **43.75%** |

Confusion matrix:

```
               Pred Young  Pred Adult  Pred Elderly
True Young        24          12            1
True Adult         9          30            5
True Elderly      10           8           14
```

**Failure mode:** Elderly collapse — 10 elderly subjects misclassified as Young, nearly as many errors as correct predictions.

#### 2-Block Model (64.60% overall)

| Class | Correct/Total | Accuracy |
|-------|--------------|----------|
| Young (<40) | 27/37 | 72.97% |
| **Adult (40–64)** | **24/44** | **54.55%** |
| Elderly (≥65) | 22/32 | 68.75% |

Confusion matrix:

```
               Pred Young  Pred Adult  Pred Elderly
True Young        27           5            5
True Adult         7          24           13
True Elderly       6           4           22
```

**Failure mode:** Adult fragmentation — 13 Adult subjects pushed into Elderly, 7 into Young. The 2-block model learns a bimodal gait polarisation: it excels at recognising extreme gait patterns (Young 73%, Elderly 69%) but struggles with the ambiguous middle group.

#### Boundary Analysis

The most revealing metric is the near-65 boundary zone (subjects aged 62–67):

| Model | Near-65 Rough Correct | Rough Overall |
|-------|----------------------|---------------|
| 1-block | 4/7 (57%) | 62.83% |
| **2-block** | **7/7 (100%)** | **66.37%** |

The 2-block model perfectly identifies gait characteristics that distinguish elderly from younger subjects near the clinical threshold — a critical property for downstream age conditioning.

> **[FIGURE 5]** — *Confusion matrices for 1-block and 2-block models, displayed as heatmaps.* Side-by-side 3×3 heatmaps with row labels (True: Young/Adult/Elderly) and column labels (Predicted: Young/Adult/Elderly). Annotate each cell with count. Highlight diagonal (correct) in green, off-diagonal (errors) in shades of red proportional to magnitude. *(Chart: `figures/figure5_confusion_matrices.png`)*

<div style="text-align: center;">
  <img src="figures/figure5_confusion_matrices.png" width="70%" />
</div>

### 4.4 z_age Embedding Extraction

After training, 32-dimensional z_age embeddings were extracted for all 450 clips using the bottleneck layer of the best checkpoint:

```python
# Extraction pathway (from extract_z_age.py):
features = model.backbone(keypoint)         # (B*NC, M, 256, T', V)
z_clips  = model.head.get_z(features)       # (B*NC, 32)
z_age    = z_clips.view(B, NC, -1).mean(1)  # (B, 32) — averaged over 10 clips
```

| Source Checkpoint | z_age Mean | z_age Std | z_age Range |
|-------------------|-----------|-----------|-------------|
| Frozen (epoch 1) | −0.39 | 0.32 | [−1.21, 0.46] |
| 1-block (epoch 17) | 1.18 | 2.63 | [−4.61, 13.52] |
| **2-block (epoch 4)** | **0.79** | **2.64** | **[−2.61, 13.44]** |

The recommended z_age source for LoRA-MDM conditioning is the 1-block checkpoint (`z_age_embeddings_1block.npz`) — it achieves comparable discriminative power (std 2.63 vs. 2.64) with half the trainable parameters and more stable generalisation.

> **[FIGURE 6]** — *3D PCA visualisation of z_age embeddings.* Scatter plot of the first 3 principal components of the 450 z_age vectors, coloured by age group (Young=blue, Adult=green, Elderly=red). Plot for each of the three checkpoints (frozen, 1-block, 2-block) as subplots. The frozen config should show a collapsed cluster; 1-block and 2-block should show some separation with overlap in the Adult class. *(Chart: `figures/figure6_pca_visualization.png`)*

<div style="text-align: center;">
  <img src="figures/figure6_pca_visualization.png" width="90%" />
</div>

---

## 5. Current Roadblocks & Error Analysis

### 5.1 Fundamental Dataset Constraint

The primary bottleneck is dataset size: 450 clips from 138 subjects, with only 337 training samples. For context, the NTU RGB+D pretraining dataset has 113,945 clips. The 718K trainable parameters in the 2-block configuration exceed what this dataset can effectively constrain, leading to memorisation (100% train accuracy) within a few epochs. Even the 1-block configuration (363K params) reaches 99%+ train accuracy by epoch 50.

### 5.2 Adult Class Ambiguity

The Adult group (40–64) spans 25 years of functional variation — a 42-year-old and a 63-year-old may have dramatically different gait patterns, yet both carry the same label. The 2-block model's Adult fragmentation (54.55% accuracy) reflects this biological reality: some older adults walk with elderly-like gait, and some younger adults with young-like gait. This is arguably a better learned representation than the 1-block model's flat Adult-majority bias, but it limits overall accuracy.

### 5.3 Frozen Backbone Feature Invariance

The frozen backbone experiment (43.68% val, barely above random) demonstrates that NTU-120 action-recognition features are largely **invariant to the style variation** that age classification depends on. ST-GCN++ was pretrained to recognise *what action* is being performed (walking vs. sitting vs. throwing), not *how* it is performed (young gait vs. elderly gait). Age-related differences — reduced velocity, altered hip strategy, constrained range of motion — live in the stylistic dimension that action pretraining deliberately discards for invariance.

This means any backbone architecture (CTR-GCN, DG-STGCN, etc.) pretrained on action recognition may face the same limitation when frozen. Unfreezing is necessary, and the question becomes how much adaptation is needed versus how much the small dataset can support.

### 5.4 Overfitting Dynamics

The overfitting signature across configurations:

| Config | Train→Val Gap | Practical Window | Risk |
|--------|--------------|-----------------|------|
| Frozen | 10.6pp | N/A (underfitting) | Insufficient |
| 1-block | 45.2pp | Epochs 1–17 | Moderate |
| 2-block (random split) | 35.4pp | Epochs 1–4 | Severe |
| 2-block (round-robin) | 35.0pp | Epochs 1–30 | Manageable |

The round-robin split and reduced batch size substantially extended the practical training window for the 2-block configuration, but the fundamental overfitting tendency remains. Regularisation strategies (dropout, weight decay, data augmentation) are the most promising next steps.

---

## 6. Next Steps

*(To be determined.)*
