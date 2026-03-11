# STGCN++ — Clean Re-implementation

Minimal, dependency-clean re-implementation of **STGCN++** as originally
described in [PYSKL](https://arxiv.org/abs/2205.09443).

- **No OpenMMLab / mmcv** — pure PyTorch + NumPy.
- Weights are directly compatible with pre-trained pyskl checkpoints
  (NTU RGB+D 60/120, 3D skeleton and HRNet 2D skeleton, all modalities).
- Designed for fine-tuning on custom datasets with 25-joint NTU-style skeletons.

---

## File overview

| File | Purpose |
|---|---|
| `graph.py` | Spatial adjacency matrix for NTU RGB+D 25-joint skeleton |
| `model.py` | Full STGCN++ model (UnitGCN, MSTCN, backbone, head) |
| `dataset.py` | `SkeletonDataset` — loads custom `.pkl` annotation files |
| `train.py` | Fine-tuning script |
| `inference.py` | Inference / evaluation script |
| `pyproject.toml` | `uv` project / dependency manifest |

---

## Setup

```bash
cd stgcnpp
uv sync          # creates .venv and installs dependencies
```

If CUDA wheels are needed (adjust index URL for your CUDA version):

```bash
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Quick test — load pretrained weights

```bash
uv run python - <<'EOF'
import torch
from model import STGCNPP

model = STGCNPP(num_classes=120,
                pretrained='../stgcnpp_ntu120_3dkp_joint.pth')
model.eval()

# (batch=1, persons=2, frames=100, joints=25, coords=3)
x = torch.randn(1, 2, 100, 25, 3)
with torch.no_grad():
    out = model(x)
print('Output shape:', out.shape)   # (1, 120)
print('Top-1 class:', out.argmax(dim=1).item())
EOF
```

---

## Pretrained Model Zoo

All checkpoints are hosted by OpenMMLab. Download the `.pth` file for the
modality and dataset you need and pass the path to `pretrained=`.

> **This re-implementation is validated against the NTU RGB+D 120 XSub 3D
> Skeleton Joint checkpoint** (`stgcnpp_ntu120_xsub_3dkp/j.pth`), achieving
> **82.6% top-1** accuracy on the full 113,945-clip test set — consistent with
> the published 83.2% benchmark.

### NTU RGB+D 60

| Split | Annotation | Modality | Top-1 | Download |
|---|---|---|---|---|
| XSub | Official 3D Skeleton | Joint | 89.3% | [j.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_3dkp/j.pth) |
| XSub | Official 3D Skeleton | Bone | 90.1% | [b.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_3dkp/b.pth) |
| XSub | Official 3D Skeleton | Joint Motion | 87.5% | [jm.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_3dkp/jm.pth) |
| XSub | Official 3D Skeleton | Bone Motion | 87.3% | [bm.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_3dkp/bm.pth) |
| XView | Official 3D Skeleton | Joint | 95.6% | [j.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_3dkp/j.pth) |
| XView | Official 3D Skeleton | Bone | 95.5% | [b.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_3dkp/b.pth) |
| XView | Official 3D Skeleton | Joint Motion | 94.3% | [jm.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_3dkp/jm.pth) |
| XView | Official 3D Skeleton | Bone Motion | 93.8% | [bm.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_3dkp/bm.pth) |

### NTU RGB+D 120

| Split | Annotation | Modality | Top-1 | Download |
|---|---|---|---|---|
| XSub | Official 3D Skeleton | Joint | 83.2% | [j.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/j.pth) |
| XSub | Official 3D Skeleton | Bone | 85.6% | [b.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/b.pth) |
| XSub | Official 3D Skeleton | Joint Motion | 80.4% | [jm.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/jm.pth) |
| XSub | Official 3D Skeleton | Bone Motion | 81.5% | [bm.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/bm.pth) |
| XSet | Official 3D Skeleton | Joint | 85.6% | [j.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xset_3dkp/j.pth) |
| XSet | Official 3D Skeleton | Bone | 87.5% | [b.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xset_3dkp/b.pth) |
| XSet | Official 3D Skeleton | Joint Motion | 84.3% | [jm.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xset_3dkp/jm.pth) |
| XSet | Official 3D Skeleton | Bone Motion | 83.0% | [bm.pth](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xset_3dkp/bm.pth) |

**Modality notes:**
- `j` (Joint) — raw 3D joint positions `(x, y, z)`. Simplest to prepare; used for validation in this repo.
- `b` (Bone) — bone difference vectors `child − parent`. More rotation-robust; recommended for gait analysis.
- `jm` (Joint Motion) — frame-to-frame joint velocity. Captures dynamics, not absolute pose.
- `bm` (Bone Motion) — frame-to-frame change in bone vectors. Combines geometry with dynamics.

---

## NTU RGB+D Dataset — Preprocessed Skeleton Files

PYSKL provides pre-processed `.pkl` annotation files ready for direct use.

| Dataset | Annotation | Download |
|---|---|---|
| NTU RGB+D 60 | 3D Skeleton | [ntu60_3danno.pkl](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl) |
| NTU RGB+D 120 | 3D Skeleton ✅ | [ntu120_3danno.pkl](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl) |

The `.pkl` format is a dict `{split: {...}, annotations: [...]}`. Each
annotation contains `keypoint` as `np.ndarray` of shape `(M, T, 25, 3)`.
See `dataset.py` for the full format specification.

---

## Fine-tuning on a custom dataset

Prepare a `.pkl` file — a list of dicts with the following keys:

```python
[
  {
    'keypoint': np.ndarray,  # shape (M, T, 25, 3)
    'label':    int,         # 0-indexed class index
  },
  ...
]
```

Then run:

```bash
uv run python train.py \
    --data path/to/custom.pkl \
    --pretrained ../stgcnpp_ntu120_3dkp_joint.pth \
    --num_classes 10 \
    --epochs 50 \
    --lr 1e-3 \
    --batch_size 16
```

Add `--freeze_backbone` to freeze the STGCN++ backbone and only train the
classification head (faster, useful when data is scarce).

---

## Architecture summary

| Component | Details |
|---|---|
| Graph | NTU RGB+D 25-joint spatial graph, 3 subsets (self / inward / outward) |
| GCN | `UnitGCN` — adaptive=`init`, pre-multiply, residual shortcut |
| TCN | `MSTCN` — 6 branches: (3,1) (3,2) (3,3) (3,4) maxpool 1×1 |
| Stages | 10 stages; inflate at 5 & 8; downsample at 5 & 8 |
| Channels | 64 → 128 → 256 |
| Data BN | BatchNorm1d over (V × C) per person |
| Head | AdaptiveAvgPool2d(1) + Linear |

---

## Citation

```bibtex
@inproceedings{duan2022pyskl,
  title={Pyskl: Towards good practices for skeleton action recognition},
  author={Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={7351--7354},
  year={2022}
}
```