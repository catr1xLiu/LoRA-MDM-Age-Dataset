"""
ST-GCN++ inference script.

Loads a pre-trained checkpoint and evaluates Top-1 accuracy on any NTU
RGB+D split (e.g. xsub_val, xset_val).

Usage
-----
    uv run python infer.py \\
        --data       data/ntu120_3danno.pkl \\
        --checkpoint checkpoints/j.pth \\
        --modality   joint \\
        --split      xsub_val

Multi-clip test (default: 10 clips per sample)
    Scores from 10 random temporal crops are aggregated by:
        1. softmax over the class dimension for each clip
        2. mean across the 10 clips
    This 'prob' averaging scheme exactly matches the PYSKL evaluation protocol.

Checkpoint format
-----------------
PYSKL saves checkpoints with mmcv which wraps the state dict:
    {
        'state_dict': { 'backbone.*': ..., 'cls_head.*': ... },
        'meta':       { ... },
    }
This script reads 'state_dict' and remaps the key prefixes:
    backbone.  →  backbone.
    cls_head.  →  head.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from stgcnpp import STGCNpp, build_dataloader


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _remap_key(key: str) -> str | None:
    """Map a PYSKL checkpoint key to our model's key.

    PYSKL key prefix   →   our model prefix
    ─────────────────────────────────────────
    backbone.*         →   backbone.*    (unchanged)
    cls_head.*         →   head.*

    Returns None if the key should be skipped.
    """
    if key.startswith("backbone."):
        return key
    if key.startswith("cls_head."):
        return "head." + key[len("cls_head."):]
    return None


def load_checkpoint(model: STGCNpp, ckpt_path: str, device: torch.device) -> None:
    """Load a PYSKL-format checkpoint into our model.

    Args:
        model:     The STGCNpp model instance (weights will be mutated).
        ckpt_path: Path to the .pth checkpoint file.
        device:    Target device for the tensors.
    """
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)

    # mmcv wraps the state dict under the key 'state_dict'
    state_dict = raw.get("state_dict", raw)

    remapped: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for k, v in state_dict.items():
        new_k = _remap_key(k)
        if new_k is None:
            skipped.append(k)
        else:
            remapped[new_k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)

    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if skipped:
        print(f"  [INFO] Skipped {len(skipped)} keys not belonging to backbone/head")


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: STGCNpp,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_clips: int,
) -> float:
    """Run inference and compute Top-1 accuracy.

    Each sample is represented by ``num_clips`` temporal clips.  The per-clip
    logits are converted to probabilities (softmax) and averaged before taking
    the argmax.

    Args:
        model:      The STGCNpp model in eval mode.
        dataloader: Test DataLoader yielding (keypoint, label) batches.
                    keypoint shape: (B, num_clips, M, T, V, C)
        device:     Inference device.
        num_clips:  Number of clips per sample used during sampling.

    Returns:
        Top-1 accuracy as a float in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0

    for keypoint, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
        # keypoint: (B, num_clips, M, T, V, C)
        B, NC, M, T, V, C = keypoint.shape
        assert NC == num_clips, f"Expected {num_clips} clips, got {NC}"

        keypoint = keypoint.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)

        # Fold clips into the batch dimension: (B*NC, M, T, V, C)
        keypoint = keypoint.view(B * NC, M, T, V, C)

        # Forward pass: (B*NC, num_classes)
        logits = model(keypoint)

        # Restore clip dimension and aggregate
        # (B*NC, num_classes) → (B, NC, num_classes)
        logits = logits.view(B, NC, -1)

        # 'prob' averaging: softmax per clip, then mean across clips
        probs = F.softmax(logits, dim=2).mean(dim=1)   # (B, num_classes)

        preds = probs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += B

    return correct / total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ST-GCN++ inference on NTU RGB+D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="data/ntu120_3danno.pkl",
        help="Path to the NTU 3-D skeleton pickle file",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the pre-trained .pth checkpoint (joint or bone model)",
    )
    parser.add_argument(
        "--modality",
        choices=["joint", "bone"],
        default="joint",
        help="Input modality: 'joint' (raw coordinates) or 'bone' (edge vectors)",
    )
    parser.add_argument(
        "--split",
        default="xsub_val",
        help="Dataset split to evaluate (e.g. xsub_val, xset_val)",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=10,
        help="Number of temporal clips per sample for test-time augmentation",
    )
    parser.add_argument(
        "--clip-len",
        type=int,
        default=100,
        help="Number of frames per clip",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size (number of samples, not clips)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes for parallel pre-processing",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples (useful for quick checks)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    print(f"Device  : {device}")
    print(f"Split   : {args.split}")
    print(f"Modality: {args.modality}")
    print(f"Clips   : {args.num_clips} × {args.clip_len} frames")
    if args.max_samples:
        print(f"Max     : first {args.max_samples:,} samples")

    # --- Dataset ---
    print("\nBuilding dataset...")
    dataloader = build_dataloader(
        pkl_path=args.data,
        split=args.split,
        modality=args.modality,
        clip_len=args.clip_len,
        num_clips=args.num_clips,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    print(f"  {len(dataloader.dataset):,} samples  |  {len(dataloader):,} batches")

    # --- Model ---
    print("\nBuilding model...")
    model = STGCNpp(in_channels=3, num_classes=120)
    model.to(device)

    # --- Checkpoint ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"\nLoading checkpoint: {ckpt_path}")
    load_checkpoint(model, str(ckpt_path), device)

    # --- Inference ---
    print()
    t0 = time.perf_counter()
    top1 = evaluate(model, dataloader, device, args.num_clips)
    elapsed = time.perf_counter() - t0

    print(f"\nTop-1 accuracy : {top1 * 100:.2f}%")
    print(f"Elapsed        : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
