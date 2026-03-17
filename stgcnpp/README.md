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
- Validation accuracy: ~48% (baseline random = 33%)
- Used as feature extractor for LoRA-MDM age-conditioned motion generation

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

### Training

```bash
conda run -n mdm-data-pipeline python train_vc.py \
    --checkpoint checkpoints/b.pth \
    --epochs 50 \
    --batch-size 16
```

**Configuration**:
- Load pretrained bone model (`b.pth`)
- Freeze STGCNBackbone entirely
- Replace head: `Linear(256 → 3)` for Young/Adult/Elderly
- Optimizer: AdamW, lr=1e-3 (head only)
- Save best checkpoint to `checkpoints/vc_age_best.pth`

### Extracting z_age Embeddings

```bash
conda run -n mdm-data-pipeline python extract_z_age.py \
    --checkpoint checkpoints/vc_age_best.pth \
    --data data/vc_ntu25.pkl
```

**Output**: `data/z_age_embeddings.npz` with:
- `clip_ids`: list of sample IDs
- `z_age`: (N, 32) bottleneck embeddings
- `labels`: (N,) age group labels
- `ages`: (N,) actual ages

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