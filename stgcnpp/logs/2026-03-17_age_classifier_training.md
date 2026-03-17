# Age Classifier Training on Van Criekinge Dataset

**Date:** 2026-03-17
**Cluster:** NRC Trixie (trixie-cn109)
**GPU:** Tesla V100-SXM2-32GB
**Author:** Yiren Liu, Claude (Opus 4.6)

---

## Objective

Train a 3-class age classifier (Young / Adult / Elderly) on the Van Criekinge gait dataset using an ST-GCN++ backbone pretrained on NTU-120. The trained classifier's 32-dim bottleneck layer serves as the z_age embedding for conditioning LoRA-MDM.

## Setup

### Data

- **Source:** `data/vc_ntu25.pkl` — 450 clips (363 train / 87 val), 138 subjects
- **Label distribution (train):** Young (<40): 117, Adult (40-64): 141, Elderly (>=65): 105
- **Class weights:** [1.034, 0.858, 1.152] (inverse frequency)

### Architecture

- **Backbone:** ST-GCN++ (10 GCN blocks), pretrained on NTU-120 3D keypoints (joint modality)
- **Head:** `AgeClassifierHead(256 -> 32 -> 3)` — bottleneck design where the 32-dim intermediate layer is the z_age embedding
- **Loss:** Class-weighted CrossEntropyLoss

### Optimizer

- AdamW with cosine annealing over 50 epochs
- Head LR: 1e-3, backbone LR: 1e-4 (when applicable)
- Weight decay: 1e-4

---

## Run 1: Backbone Partially Unfrozen (2 GCN Blocks)

**Job ID:** 182894
**Config:** `--unfreeze-blocks 2` (default)
**Trainable params:** 718,713 / 1,396,047 (51.5%)
**Output checkpoint:** `checkpoints/vc_age_best.pth`
**Output embeddings:** `data/z_age_embeddings.npz`

### Training Log

```
 Epoch  Train Loss  Train Acc   Val Acc      Best
-------------------------------------------------------
     1      1.0978     33.61%    48.28%    48.28% *
     2      1.0573     46.56%    45.98%    48.28%
     3      0.8949     58.95%    45.98%    48.28%
     4      0.6585     71.35%    64.37%    64.37% *
     5      0.4770     82.92%    48.28%    64.37%
     6      0.3414     88.71%    42.53%    64.37%
     7      0.2414     92.56%    63.22%    64.37%
     8      0.1578     96.42%    45.98%    64.37%
     9      0.1793     94.77%    55.17%    64.37%
    10      0.1111     95.59%    49.43%    64.37%
    11      0.2108     92.84%    52.87%    64.37%
    12      0.1547     94.77%    49.43%    64.37%
    13      0.1303     95.32%    57.47%    64.37%
    14      0.0835     97.80%    51.72%    64.37%
    15      0.0551     98.90%    51.72%    64.37%
    16      0.0664     98.35%    58.62%    64.37%
    17      0.0607     98.62%    55.17%    64.37%
    18      0.0389     98.90%    55.17%    64.37%
    19      0.0329     99.17%    51.72%    64.37%
    20      0.0424     99.17%    57.47%    64.37%
    21      0.0279    100.00%    56.32%    64.37%
    22      0.0325     99.17%    57.47%    64.37%
    23      0.0311     99.17%    54.02%    64.37%
    24      0.0511     98.90%    57.47%    64.37%
    25      0.0347     99.17%    57.47%    64.37%
    26      0.0294     99.45%    56.32%    64.37%
    27      0.0213     99.45%    42.53%    64.37%
    28      0.0139     99.72%    57.47%    64.37%
    29      0.0314     98.62%    52.87%    64.37%
    30      0.0134     99.72%    55.17%    64.37%
    31      0.0173     99.72%    55.17%    64.37%
    32      0.0415     99.45%    55.17%    64.37%
    33      0.0056    100.00%    56.32%    64.37%
    34      0.0113    100.00%    57.47%    64.37%
    35      0.0061    100.00%    57.47%    64.37%
    36      0.0132     99.72%    54.02%    64.37%
    37      0.0147    100.00%    56.32%    64.37%
    38      0.0086    100.00%    56.32%    64.37%
    39      0.0053    100.00%    56.32%    64.37%
    40      0.0073    100.00%    58.62%    64.37%
    41      0.0078    100.00%    58.62%    64.37%
    42      0.0034    100.00%    58.62%    64.37%
    43      0.0080    100.00%    56.32%    64.37%
    44      0.0160     99.45%    57.47%    64.37%
    45      0.0037    100.00%    56.32%    64.37%
    46      0.0095     99.72%    55.17%    64.37%
    47      0.0061    100.00%    55.17%    64.37%
    48      0.0051    100.00%    57.47%    64.37%
    49      0.0107     99.72%    57.47%    64.37%
    50      0.0099     99.72%    56.32%    64.37%
```

**Best val accuracy: 64.37% (epoch 4)**
**Wall time: 124.1s**

### z_age Embeddings

```
z_age shape : (450, 32)  (dtype: float32)
Label dist  : Young=155, Adult=172, Elderly=123
z_age mean  : 0.7931  std: 2.6430
z_age range : [-2.6147, 13.4431]
```

---

## Run 2: Backbone Fully Frozen (0 GCN Blocks)

**Job ID:** 182898
**Config:** `--unfreeze-blocks 0`
**Trainable params:** 8,323 / 1,396,047 (0.6%)
**Output checkpoint:** `checkpoints/vc_age_frozen.pth`
**Output embeddings:** `data/z_age_embeddings_frozen.npz`

### Training Log

```
 Epoch  Train Loss  Train Acc   Val Acc      Best
-------------------------------------------------------
     1      1.1002     29.48%    43.68%    43.68% *
     2      1.0984     37.47%    41.38%    43.68%
     3      1.0979     37.19%    40.23%    43.68%
     4      1.0951     37.47%    41.38%    43.68%
     5      1.0899     43.25%    35.63%    43.68%
     6      1.0858     35.81%    28.74%    43.68%
     7      1.0863     39.67%    43.68%    43.68%
     8      1.0688     45.18%    40.23%    43.68%
     9      1.0573     47.38%    37.93%    43.68%
    10      1.0489     49.86%    39.08%    43.68%
    11      1.0386     49.04%    26.44%    43.68%
    12      1.0355     46.56%    36.78%    43.68%
    13      1.0315     43.80%    27.59%    43.68%
    14      1.0209     50.41%    27.59%    43.68%
    15      1.0321     50.96%    29.89%    43.68%
    16      1.0054     53.17%    35.63%    43.68%
    17      0.9964     53.44%    31.03%    43.68%
    18      1.0112     47.11%    32.18%    43.68%
    19      1.0042     47.93%    29.89%    43.68%
    20      0.9974     54.27%    32.18%    43.68%
    21      0.9934     50.69%    34.48%    43.68%
    22      0.9804     52.34%    32.18%    43.68%
    23      0.9817     51.24%    33.33%    43.68%
    24      0.9880     53.99%    35.63%    43.68%
    25      0.9789     53.17%    35.63%    43.68%
    26      0.9497     55.65%    34.48%    43.68%
    27      0.9476     56.47%    35.63%    43.68%
    28      0.9671     53.44%    35.63%    43.68%
    29      0.9513     55.37%    36.78%    43.68%
    30      0.9548     54.27%    33.33%    43.68%
    31      0.9733     55.10%    34.48%    43.68%
    32      0.9797     52.89%    35.63%    43.68%
    33      0.9562     56.20%    37.93%    43.68%
    34      0.9651     53.44%    35.63%    43.68%
    35      0.9836     52.07%    36.78%    43.68%
    36      0.9582     55.10%    33.33%    43.68%
    37      0.9869     50.96%    35.63%    43.68%
    38      0.9541     52.62%    36.78%    43.68%
    39      0.9775     54.27%    35.63%    43.68%
    40      0.9446     53.17%    34.48%    43.68%
    41      0.9570     55.37%    36.78%    43.68%
    42      0.9608     53.44%    35.63%    43.68%
    43      0.9571     56.75%    35.63%    43.68%
    44      0.9634     51.24%    36.78%    43.68%
    45      0.9708     50.69%    35.63%    43.68%
    46      0.9538     55.10%    34.48%    43.68%
    47      0.9660     53.44%    35.63%    43.68%
    48      0.9886     49.86%    35.63%    43.68%
    49      0.9558     52.62%    35.63%    43.68%
    50      0.9489     54.27%    35.63%    43.68%
```

**Best val accuracy: 43.68% (epoch 1)**
**Wall time: 110.2s**

### z_age Embeddings

```
z_age shape : (450, 32)  (dtype: float32)
Label dist  : Young=155, Adult=172, Elderly=123
z_age mean  : -0.3901  std: 0.3173
z_age range : [-1.2141, 0.4630]
```

---

## Comparative Analysis

| Metric | Run 1 (2 blocks unfrozen) | Run 2 (fully frozen) |
|---|---|---|
| Trainable params | 718,713 (51.5%) | 8,323 (0.6%) |
| Best val accuracy | **64.37%** (epoch 4) | 43.68% (epoch 1) |
| Final train accuracy | ~100% (epoch 21+) | ~55% (epoch 50) |
| Train-val gap | ~36 pp (severe overfit) | ~20 pp |
| z_age std | 2.643 (high spread) | 0.317 (low spread) |
| z_age range | [-2.61, 13.44] | [-1.21, 0.46] |

### Key Observations

1. **Frozen backbone is insufficient.** 43.68% val accuracy is barely above chance (33.3% for 3 classes). The NTU-120 action recognition features do not encode age-related gait patterns — the head alone cannot learn a useful mapping from these frozen features.

2. **Unfreezing 2 blocks works, but overfits severely.** Val accuracy peaks at epoch 4 (64.37%) then fluctuates around 55-58% while train accuracy reaches 100%. The model memorises the training set almost immediately. This is expected: 363 training samples is very small for 718K trainable parameters.

3. **Best checkpoint is saved early (epoch 4).** The early-stopping mechanism is critical here — without it, we'd be using a heavily overfit model.

4. **z_age quality correlates with classifier quality.** The 2-block model produces embeddings with much larger dynamic range (std=2.64 vs 0.32), suggesting more discriminative features. However, given the overfitting, these embeddings may capture training-set-specific patterns rather than generalisable age signals.

5. **The fundamental bottleneck is dataset size.** With only 363 training samples spread across 3 classes, there is very limited capacity for the model to learn generalisable age features. Standard regularisation (dropout=0.3, weight decay=1e-4, cosine annealing) is not sufficient.

---

## Files Produced

| File | Description |
|---|---|
| `stgcnpp/train_vc.py` | Training script with `--unfreeze-blocks` parameter |
| `stgcnpp/extract_z_age.py` | z_age embedding extraction script |
| `stgcnpp/submit_train_vc.sh` | SBATCH job submission script |
| `stgcnpp/stgcnpp/model.py` | Added `AgeClassifierHead` class |
| `stgcnpp/stgcnpp/__init__.py` | Exported `AgeClassifierHead` |
| `checkpoints/vc_age_best.pth` | Best checkpoint (2 blocks, epoch 4, 64.37%) |
| `checkpoints/vc_age_frozen.pth` | Best checkpoint (frozen, epoch 1, 43.68%) |
| `data/z_age_embeddings.npz` | z_age from 2-block model |
| `data/z_age_embeddings_frozen.npz` | z_age from frozen model |
