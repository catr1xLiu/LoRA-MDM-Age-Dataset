"""
Extract 32-dim z_age bottleneck embeddings from the fine-tuned age classifier.

For every clip in both the train and val splits, the AgeClassifierHead bottleneck
(the 32-dim layer before the final linear) is extracted and saved as a numpy array.
These embeddings are used to condition the LoRA-MDM for age-aware motion generation.

Usage
-----
    python extract_z_age.py \
        --checkpoint checkpoints/vc_age_best.pth \
        --data       data/vc_ntu25.pkl

Output
------
    data/z_age_embeddings.npz  containing:
        clip_ids  (N,)    str   — unique clip identifier (frame_dir)
        z_age     (N, 32) f32   — bottleneck embeddings
        labels    (N,)    i64   — age group (0=Young, 1=Adult, 2=Elderly)
        split     (N,)    str   — "train" or "val"
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from stgcnpp import AgeClassifierHead, NTUDataset, STGCNpp

AGE_GROUPS = ["Young (<40)", "Adult (40-64)", "Elderly (≥65)"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_age_model(ckpt_path: str, device: torch.device) -> STGCNpp:
    """Load the fine-tuned age classifier from a train_vc.py checkpoint."""
    model = STGCNpp(in_channels=3, num_classes=120)
    model.head = AgeClassifierHead(in_channels=256, z_dim=32, num_classes=3, dropout=0.3)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected[:5]}")

    val_acc = ckpt.get("val_acc", None)
    epoch = ckpt.get("epoch", None)
    if val_acc is not None:
        print(f"  Checkpoint: epoch {epoch}, val_acc={100*val_acc:.2f}%")

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_split(
    model: STGCNpp,
    pkl_path: str,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Extract z_age embeddings for one data split.

    Returns:
        clip_ids : list of str, length N
        z_age    : (N, 32) float32
        labels   : (N,) int64
    """
    dataset = NTUDataset(
        pkl_path=pkl_path,
        split=split,
        modality="joint",
        clip_len=100,
        num_clips=10,
        test_mode=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    all_z: list[np.ndarray] = []
    all_labels: list[int] = []

    for keypoint, labels in tqdm(loader, desc=f"  {split:5s}", unit="batch"):
        B, NC, M, T, V, C = keypoint.shape
        keypoint = keypoint.view(B * NC, M, T, V, C).to(device, non_blocking=True)

        # Run backbone, then extract z from head without final classification
        features = model.backbone(keypoint)             # (B*NC, M, 256, T', V)
        z_clips = model.head.get_z(features)            # (B*NC, 32)

        # Average z_age over the 10 clips per sample
        z_clips = z_clips.view(B, NC, -1).mean(dim=1)  # (B, 32)

        all_z.append(z_clips.cpu().numpy())
        all_labels.extend(labels.tolist())

    clip_ids = [ann["frame_dir"] for ann in dataset.samples]
    z_age = np.concatenate(all_z, axis=0).astype(np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    return clip_ids, z_age, labels_arr


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 32-dim z_age embeddings from the fine-tuned age classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/vc_age_best.pth",
        help="Path to fine-tuned checkpoint from train_vc.py",
    )
    parser.add_argument(
        "--data",
        default="data/vc_ntu25.pkl",
        help="Path to the Van Criekinge NTU-25 pickle",
    )
    parser.add_argument(
        "--output",
        default="data/z_age_embeddings.npz",
        help="Output path for z_age embeddings",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("z_age Embedding Extraction")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data       : {args.data}")
    print(f"Output     : {args.output}")

    # --- Load model ---
    print("\nLoading model...")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = load_age_model(str(ckpt_path), device)

    # --- Extract embeddings for both splits ---
    all_clip_ids: list[str] = []
    all_z: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_splits: list[str] = []

    for split in ("train", "val"):
        print(f"\nExtracting {split} split...")
        clip_ids, z_age, labels = extract_split(
            model, args.data, split, device, args.batch_size, args.num_workers
        )
        all_clip_ids.extend(clip_ids)
        all_z.append(z_age)
        all_labels.append(labels)
        all_splits.extend([split] * len(clip_ids))

        # Per-split summary
        for g, name in enumerate(AGE_GROUPS):
            n = int((labels == g).sum())
            print(f"  {name}: {n} clips")

    z_age_all = np.concatenate(all_z, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)
    split_all = np.array(all_splits)
    clip_ids_all = np.array(all_clip_ids)

    # --- Save ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        clip_ids=clip_ids_all,
        z_age=z_age_all,
        labels=labels_all,
        split=split_all,
    )

    print(f"\nSaved {len(clip_ids_all)} embeddings to: {output_path}")
    print(f"  z_age shape : {z_age_all.shape}  (dtype: {z_age_all.dtype})")
    print(f"  Label dist  : Young={int((labels_all==0).sum())}, "
          f"Adult={int((labels_all==1).sum())}, Elderly={int((labels_all==2).sum())}")
    print(f"  z_age mean  : {z_age_all.mean():.4f}  std: {z_age_all.std():.4f}")
    print(f"  z_age range : [{z_age_all.min():.4f}, {z_age_all.max():.4f}]")


if __name__ == "__main__":
    main()
