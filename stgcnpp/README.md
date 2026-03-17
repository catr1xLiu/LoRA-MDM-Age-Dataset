# ST-GCN++ — Standalone PyTorch Implementation

A clean, dependency-minimal re-implementation of **ST-GCN++** for skeleton-based action recognition, targeting the **NTU RGB+D 120** dataset with official 3-D skeleton annotations.

This project is a faithful reimplementation of the model described in:
> **PYSKL: Towards Good Practices for Skeleton Action Recognition**
> Haodong Duan, Jiaqi Wang, Kai Chen, Dahua Lin — ACM MM 2022
> https://arxiv.org/abs/2205.09443

The original PYSKL codebase depends on the OpenMMLab ecosystem (mmcv, mmaction2) which is no longer actively maintained and is difficult to run in modern environments. This project eliminates all such dependencies — only **PyTorch**, **NumPy**, **SciPy**, and **tqdm** are required.

---

## Results

### NTU RGB+D 120 (X-Subject split, official 3-D skeletons)

| Modality | Top-1 |
|----------|-------|
| Joint    | 83.2% |
| Bone     | 85.6% |

### Van Criekinge Age Classification

When fine-tuned on the Van Criekinge gait dataset for 3-class age classification (Young <40, Adult 40-64, Elderly ≥65):

| Unfreeze Blocks | Train Acc | Val Acc | Best Epoch | Trainable Params | z_age std |
|-----------------|-----------|---------|------------|------------------|-----------|
| 0 (frozen)      | 54.3%     | 43.68%  | epoch 1    | 0.6% (8K)        | 0.32      |
| 2               | 100%      | 64.37%  | epoch 4    | 51.5% (718K)     | 2.64      |
| 1 *(testing)*   | *TBD*     | *TBD*   | *TBD*      | 25% (~350K)      | *TBD*     |

**Key observations**:
- **Frozen backbone (0 blocks)**: Poor discrimination — pretrained NTU-120 features don't encode age well
- **2-block unfrozen**: Best accuracy but severe overfitting (train 100%, val 64%) due to small dataset (363 train samples)
- **Trade-off**: 1-block unfrozen being tested as middle ground
- **z_age quality**: Larger std with more unfrozen blocks indicates more discriminative features

Used as feature extractor (`z_age` embedding) for LoRA-MDM age-conditioned motion generation.

---

## Project layout

```
stgcnpp/
├── stgcnpp/
│   ├── __init__.py      re-exports the public API
│   ├── graph.py         NTU skeleton graph construction (adjacency matrix)
│   ├── model.py         Backbone (STGCNBackbone) + head (GCNClassifier)
│   └── dataset.py       Dataset, all preprocessing transforms, DataLoader
├── inference.py         Single-clip inference with visualization
├── batch_inference.py   Batch evaluation script
├── import_from_smpl.py  SMPL → NTU-25 conversion (with CUDA acceleration)
├── train_vc.py          Fine-tune for 3-class age classification
├── extract_z_age.py     Extract 32-dim bottleneck embeddings
├── data/
│   ├── ntu120_3danno.pkl   Original NTU dataset
│   └── vc_ntu25.pkl        Van Criekinge dataset in NTU-25 format
├── checkpoints/
│   ├── j.pth            Joint modality pretrained (NTU-120)
│   ├── b.pth            Bone modality pretrained (NTU-120)
│   └── vc_age_best.pth  Fine-tuned age classifier
└── pyproject.toml       uv / Python 3.14 project definition
```

---

## Architecture

### Graph (`graph.py`)

The NTU RGB+D skeleton is modelled as a directed graph with **25 joints** and **three adjacency subsets** (the "spatial" partition strategy):

- `A[0]` — self-links (identity)
- `A[1]` — inward edges (child → parent), degree-normalised
- `A[2]` — outward edges (parent → child), degree-normalised

The resulting tensor `A` has shape `(3, 25, 25)` and is used to initialise the learnable adjacency parameter in every `UnitGCN` layer.

### Backbone — `STGCNBackbone` (`model.py`)

Ten stacked `STGCNBlock` layers, each consisting of:

1. **UnitGCN** — adaptive graph convolution
   - `adaptive='init'`: the adjacency matrix is a fully learnable `nn.Parameter`
   - `conv_pos='pre'`: channel mixing happens before spatial aggregation
   - Internal GCN residual (`with_res=True`)

2. **MSTCN** — multi-scale temporal convolution
   Six parallel branches: `(3,d=1) (3,d=2) (3,d=3) (3,d=4) MaxPool 1×1`
   Outputs are concatenated and fused by a BN→ReLU→1×1 transform.

3. **Outer residual** — identity or `UnitTCN(k=1)` when dimensions change.

Channel schedule (`base=64`, `ratio=2`):

| Stages | Channels | Stride |
|--------|----------|--------|
| 1–4    | 64       | 1      |
| 5      | 64→128   | 2      |
| 6–7    | 128      | 1      |
| 8      | 128→256  | 2      |
| 9–10   | 256      | 1      |

Input:  `(N, M=2, T=100, V=25, C=3)`
Output: `(N, M=2, 256, 25, 25)`

A **data batch-norm** layer `BatchNorm1d(V×C = 75)` is applied first to standardise raw skeleton coordinates.

### Head — `GCNClassifier` (`model.py`)

```
AdaptiveAvgPool2d(1)  →  mean over M persons  →  Linear(256, 120)
```

No dropout (as in the original PYSKL configuration).

### Full model — `STGCNpp`

Wraps `STGCNBackbone` + `GCNClassifier` as a single `nn.Module` for convenient checkpoint loading.

### Dataset & preprocessing (`dataset.py`)

The test pipeline applied to each sample:

1. **PreNormalize3D** — canonical alignment:
   - Select the primary actor (person with more non-zero frames)
   - Translate so that spine joint (joint 1) is at the origin
   - Rotate spine vector onto the +Z axis
   - Rotate shoulder vector onto the +X axis

2. **JointToBone** *(bone modality only)* — `bone[v] = joint[v] − joint[parent(v)]`

3. **UniformSample** — sample 100 frames per clip over 10 clips (deterministic, seed=255)

4. **PoseDecode** — index the keypoint array with sampled frame indices

5. **FormatGCNInput** — zero-pad to 2 persons, reshape to `(num_clips, 2, 100, 25, 3)`

The DataLoader uses `num_workers=4` parallel workers for pre-processing.

---

## Van Criekinge Integration

### Purpose

Convert SMPL-fitted Van Criekinge gait data to NTU-25 format for:
1. Fine-tuning ST-GCN++ as an age classifier (3 classes: Young/Adult/Elderly)
2. Extracting `z_age` embeddings (32-dim bottleneck) for LoRA-MDM conditioning

### SMPL to NTU-25 Conversion

**Critical**: Uses **mesh vertices** (surface points), not SMPL **joints** (internal skeletal points).

The mapping uses `NTU_25_MARKERS` from `explore_smpl_vertices.py` which maps NTU joint indices to SMPL vertex IDs:

```python
NTU_25_MARKERS = {
    0: 1807,  # SpineBase
    1: 3511,  # SpineMid
    2: 3069,  # Neck
    # ... (25 joints total)
}
```

This ensures correct anatomical positioning for skeleton-based models.

### Conversion Script

```bash
# Run with conda environment (requires smplx with chumpy)
conda run -n mdm-data-pipeline python import_from_smpl.py
```

**Features**:
- CUDA-accelerated vertex generation (batch processing)
- GPU tensor operations for vertex indexing
- Subject-level train/val split (80/20) to prevent data leakage

**Output**: `data/vc_ntu25.pkl` — PYSKL-format dataset with 588 clips across 138 subjects

---

## Setup

Install Python 3.14 and [uv](https://github.com/astral-sh/uv), then:

```bash
cd stgcnpp
uv sync
```

For SMPL conversion (requires smplx with chumpy):
```bash
conda activate mdm-data-pipeline
```

---

## Download

### Dataset

| File | Link |
|------|------|
| `ntu120_3danno.pkl` | https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl |

Place the file at `data/ntu120_3danno.pkl`.

### Pre-trained checkpoints (NTU-120, X-Subject, Official 3-D Skeleton)

| Modality | Reported Top-1 | Link |
|----------|---------------|------|
| Joint    | 83.2%         | http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/j.pth |
| Bone     | 85.6%         | http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/b.pth |

Place the files at `checkpoints/j.pth` and `checkpoints/b.pth`.

```bash
# One-liner downloads
wget -P checkpoints http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/j.pth
wget -P checkpoints http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/b.pth
wget -P data        https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl
```

---

## Running Inference

### NTU Dataset

```bash
# Joint model — X-Subject validation split
uv run python inference.py \
    --data       data/ntu120_3danno.pkl \
    --checkpoint checkpoints/j.pth \
    --modality   joint \
    --split      xsub_val

# Bone model
uv run python inference.py \
    --data       data/ntu120_3danno.pkl \
    --checkpoint checkpoints/b.pth \
    --modality   bone \
    --split      xsub_val
```

### Van Criekinge Dataset

```bash
# Test on VC data with pretrained model (should predict walking)
uv run python inference.py \
    --data       data/vc_ntu25.pkl \
    --checkpoint checkpoints/j.pth \
    --modality   joint \
    --split      val \
    --clip       0
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/ntu120_3danno.pkl` | Path to annotation pickle |
| `--checkpoint` | *(required)* | Path to .pth checkpoint |
| `--modality` | `joint` | `joint` or `bone` |
| `--split` | `xsub_val` | Dataset split name |
| `--num-clips` | `10` | Temporal clips per sample |
| `--clip-len` | `100` | Frames per clip |
| `--batch-size` | `16` | Samples per batch |
| `--num-workers` | `4` | DataLoader worker count |
| `--device` | `cuda` | `cuda` or `cpu` |

---

## Fine-tuning for Age Classification

Fine-tune ST-GCN++ to classify age groups (Young <40, Adult 40-64, Elderly ≥65) on Van Criekinge data and extract 32-dim age embeddings (`z_age`) for LoRA-MDM conditioning.

**Key insight from empirical testing (March 2026):**
- **Frozen backbone** (0 blocks): 43.68% val acc — pretrained NTU-120 features alone don't encode age
- **2 blocks unfrozen**: 64.37% val acc — learns age signal but overfits severely on small dataset (363 train clips)
- **Recommendation**: Use 1–2 unfrozen blocks depending on your regularization budget; save checkpoint at epoch 4 (best generalization)

### Architecture: `AgeClassifierHead`

A bottleneck classifier head replaces the 120-class `GCNClassifier`:

```
Input (N, M, C, T, V)  →  STGCNBackbone  →  Pool & Average  →  Dropout(256)
                                                                       ↓
                                                     Linear(256 → z_dim=32)  [z_age extracted here]
                                                                       ↓
                                                               ReLU  →  Linear(32 → 3)  →  Logits
```

The 32-dim intermediate layer (`z_age`) is the learned age representation, extracted without running the final classification layer.

**Parameters:**
- Input channels: 256 (from ST-GCN++ backbone)
- Bottleneck dim (z_age): 32
- Output classes: 3 (Young, Adult, Elderly)
- Dropout: 0.3 (applied before bottleneck)

### Training Script (`train_vc.py`)

Fine-tunes the age classifier on Van Criekinge with configurable backbone adaptation.

**Usage:**

```bash
# Frozen backbone (only head learns, minimal overfitting but limited accuracy)
python train_vc.py \
    --checkpoint checkpoints/stgcnpp_ntu120_3dkp_joint.pth \
    --data data/vc_ntu25.pkl \
    --epochs 50 \
    --batch-size 16 \
    --unfreeze-blocks 0 \
    --output checkpoints/vc_age_frozen.pth

# Unfreeze 1 GCN block (mid-ground: ~25% trainable params)
python train_vc.py \
    --checkpoint checkpoints/stgcnpp_ntu120_3dkp_joint.pth \
    --data data/vc_ntu25.pkl \
    --epochs 50 \
    --batch-size 16 \
    --unfreeze-blocks 1 \
    --output checkpoints/vc_age_1block.pth

# Unfreeze 2 GCN blocks (aggressive adaptation, ~51.5% trainable params, severe overfitting)
python train_vc.py \
    --checkpoint checkpoints/stgcnpp_ntu120_3dkp_joint.pth \
    --data data/vc_ntu25.pkl \
    --epochs 50 \
    --batch-size 16 \
    --unfreeze-blocks 2 \
    --output checkpoints/vc_age_2block.pth
```

**CLI Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(required)* | Pretrained NTU-120 checkpoint (e.g., `checkpoints/stgcnpp_ntu120_3dkp_joint.pth`) |
| `--data` | `data/vc_ntu25.pkl` | Van Criekinge NTU-25 pickle (450 clips: 363 train, 87 val) |
| `--epochs` | `50` | Maximum epochs (stops early when validation acc plateaus) |
| `--batch-size` | `16` | Batch size (16 works well for V100 with 32 GB) |
| `--num-workers` | `8` | Parallel data loading workers |
| `--unfreeze-blocks` | `0` | GCN blocks to unfreeze (0–10 blocks available) |
| `--output` | `checkpoints/vc_age.pth` | Checkpoint save path |
| `--device` | `cuda` | Device: `cuda` or `cpu` |

**Key Training Features:**
- **Class-weighted loss**: Compensates for imbalanced classes (Young 117, Adult 141, Elderly 105)
- **Early stopping**: Saves best checkpoint; continues training but checkpoint doesn't improve
- **Trainable parameter scaling**:
  - `--unfreeze-blocks 0`: 8,323 params (0.6% of 1.4M)
  - `--unfreeze-blocks 1`: ~350K params (25%)
  - `--unfreeze-blocks 2`: 718,713 params (51.5%)
- **Output**: Checkpoint file (`.pth`) containing best model weights + metadata (epoch, val_acc)

### Running on Trixie HPC Cluster

**Submit a complete training + extraction job:**

```bash
# Edit submit_train_vc.sh to set the desired --unfreeze-blocks parameter, then:
sbatch submit_train_vc.sh
```

The script (`submit_train_vc.sh`) automatically runs both steps:
1. Fine-tune the age classifier (50 epochs)
2. Extract z_age embeddings from the best checkpoint

**Trixie SBATCH configuration** (in `submit_train_vc.sh`):
```bash
#SBATCH --partition=TrixieMain
#SBATCH --account=jpn-302
#SBATCH --time=01:00:00           # 1 hour (sufficient for training + extraction)
#SBATCH --cpus-per-task=8         # 8 CPU cores for data loading
#SBATCH --mem=32GB                # 32 GB RAM (peak ~15 GB)
#SBATCH --gres=gpu:1              # 1 × Tesla V100 (32 GB)
```

**Monitor job status:**
```bash
# Check queue status
squeue --job <job_id>

# Watch training progress (live tail)
tail -f /gpfs/projects/AIP/jpn-302/LoRA-MDM-Age-Dataset/logs/vc_train_<job_id>.out

# After completion, review results
cat /gpfs/projects/AIP/jpn-302/LoRA-MDM-Age-Dataset/logs/vc_train_<job_id>.out
```

**Example output:**
```
Best val accuracy: 64.37%
Checkpoint saved : checkpoints/vc_age_best.pth
Saved 450 embeddings to: data/z_age_embeddings.npz
  z_age shape : (450, 32)  (dtype: float32)
  Label dist  : Young=155, Adult=172, Elderly=123
  z_age mean  : 0.7931  std: 2.6430
```

### Extraction Script (`extract_z_age.py`)

Extracts the 32-dim bottleneck embeddings (`z_age`) from a trained checkpoint.

**Usage:**

```bash
# Extract z_age from a trained checkpoint
python extract_z_age.py \
    --checkpoint checkpoints/vc_age_2block.pth \
    --data data/vc_ntu25.pkl \
    --output data/z_age_embeddings.npz \
    --batch-size 16 \
    --num-workers 8
```

**CLI Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `checkpoints/vc_age_best.pth` | Fine-tuned checkpoint from `train_vc.py` |
| `--data` | `data/vc_ntu25.pkl` | Van Criekinge NTU-25 pickle |
| `--output` | `data/z_age_embeddings.npz` | Output .npz file path |
| `--batch-size` | `16` | Batch size |
| `--num-workers` | `4` | Parallel data loading workers |
| `--device` | `cuda` | Device: `cuda` or `cpu` |

**Output File (`data/z_age_embeddings.npz`):**

A NumPy compressed archive containing:
```python
embeddings = np.load('data/z_age_embeddings.npz')
embeddings['clip_ids']  # (450,) str — unique clip identifiers
embeddings['z_age']     # (450, 32) float32 — age embeddings
embeddings['labels']    # (450,) int64 — age class (0=Young, 1=Adult, 2=Elderly)
embeddings['split']     # (450,) str — "train" or "val"
```

**Usage in LoRA-MDM:**
```python
import numpy as np

embeddings = np.load('data/z_age_embeddings.npz')
z_age = embeddings['z_age']                  # (450, 32)
labels = embeddings['labels']                # (450,)
split = embeddings['split']                  # (450,)

# Separate by age group
young_idx = labels == 0
adult_idx = labels == 1
elderly_idx = labels == 2

print(f"Young: {young_idx.sum()}, Adult: {adult_idx.sum()}, Elderly: {elderly_idx.sum()}")
# Output: Young: 155, Adult: 172, Elderly: 123
```

### Recommended Workflow

1. **Local exploration** (optional, on any machine with GPU):
   ```bash
   # Test with frozen backbone first (small memory, no overfitting risk)
   python train_vc.py --unfreeze-blocks 0 --epochs 10 --output test_ckpt.pth
   ```

2. **Submit to Trixie** (full 50-epoch training):
   ```bash
   # Edit submit_train_vc.sh to use --unfreeze-blocks 1 or 2
   sbatch submit_train_vc.sh
   # Job ID: <printed on submission>
   ```

3. **Monitor and extract**:
   ```bash
   # Watch training in real-time
   tail -f logs/vc_train_<job_id>.out

   # After job completes, both checkpoint + z_age embeddings are saved
   ls checkpoints/vc_age_*.pth data/z_age_embeddings*.npz
   ```

---

## Troubleshooting

**Accuracy is significantly below ~83%**
The most common causes:
- Wrong `--modality` flag (using `bone` checkpoint with `--modality joint` or vice versa)
- Using the wrong split (train vs. val)
- Checkpoint key mismatch — check the WARN lines printed during loading; there should be zero missing keys

**Joint 24 (or any joint) stuck in visualization**
- Ensure `import_from_smpl.py` uses vertex mapping (not joint mapping)
- Run with CUDA for correct tensor operations

**CUDA out of memory**
Reduce `--batch-size`. With an 8 GB GPU and the default settings, peak VRAM usage is approximately 3–4 GB.

**Slow data loading**
Increase `--num-workers` up to the number of CPU cores.

**smplx/chumpy import error**
Use conda environment with required dependencies:
```bash
conda activate mdm-data-pipeline
```