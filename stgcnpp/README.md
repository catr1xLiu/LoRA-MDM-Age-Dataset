# STGCN++ — Clean Re-implementation

Minimal, dependency-clean re-implementation of **STGCN++** as originally
described in [PYSKL](https://arxiv.org/abs/2205.09443).

- **No OpenMMLab / mmcv** — pure PyTorch + NumPy.
- Weights are directly compatible with the pre-trained pyskl checkpoint
  `stgcnpp_ntu120_3dkp_joint.pth` (NTU RGB+D 120, 3D skeleton, Joint modality).
- Designed for fine-tuning on custom datasets with 25-joint NTU-style skeletons.

---

## File overview

| File | Purpose |
|---|---|
| `graph.py` | Spatial adjacency matrix for NTU RGB+D 25-joint skeleton |
| `model.py` | Full STGCN++ model (UnitGCN, MSTCN, backbone, head) |
| `dataset.py` | `SkeletonDataset` — loads custom `.pkl` annotation files |
| `train.py` | Fine-tuning script |
| `inference.py` | Inference / evaluation script — (TODO) read NTU120 dataset, run model inference, report accuracy |
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

---

## TODO

- `inference.py` — to be implemented: read the NTU120 `.pkl` dataset, run the STGCN++ model in inference mode, and report accuracy metrics (top-1/top-5) across the test split.
