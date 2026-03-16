"""
ST-GCN++ single clip inference script.

Loads a pre-trained checkpoint and runs inference on a single clip from the
NTU RGB+D dataset.

Usage
-----
    uv run python inference.py \\
        --data       data/ntu120_3danno.pkl \\
        --checkpoint checkpoints/j.pth \\
        --index      0

Output
------
    Displays prediction results in the terminal including:
    - Sample information (index, frame_dir, label)
    - Top-5 predictions with probabilities
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from stgcnpp import STGCNpp, build_dataloader


def _remap_key(key: str) -> str | None:
    """Map a PYSKL checkpoint key to our model's key."""
    if key.startswith("backbone."):
        return key
    if key.startswith("cls_head."):
        return "head." + key[len("cls_head.") :]
    return None


def load_checkpoint(model: STGCNpp, ckpt_path: str, device: torch.device) -> None:
    """Load a PYSKL-format checkpoint into our model."""
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
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
        print(
            f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        print(
            f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    if skipped:
        print(f"  [INFO] Skipped {len(skipped)} keys not belonging to backbone/head")


def load_annotation(pkl_path: str) -> dict:
    """Load the NTU annotation pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_sample_info(annotation: dict, split: str, index: int) -> dict:
    """Get sample information by index from a split."""
    split_data = annotation["split"].get(split, [])
    if index >= len(split_data):
        raise IndexError(
            f"Index {index} out of range for split '{split}' (length {len(split_data)})"
        )

    frame_dir = split_data[index]
    ann_by_id = {ann["frame_dir"]: ann for ann in annotation["annotations"]}
    sample = ann_by_id[frame_dir]

    return {
        "index": index,
        "frame_dir": frame_dir,
        "label": sample["label"],
        "total_frames": sample["total_frames"],
    }


@torch.no_grad()
def inference_single(
    model: STGCNpp,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    sample_index: int,
) -> tuple[int, torch.Tensor]:
    """Run inference on a single sample.

    Args:
        model:        The STGCNpp model in eval mode.
        dataloader:   Test DataLoader yielding (keypoint, label) batches.
        device:      Inference device.
        sample_index: Index of the sample to inference on.

    Returns:
        Tuple of (true_label, logits_tensor)
    """
    model.eval()

    dataset = dataloader.dataset

    if sample_index >= len(dataset):
        raise IndexError(
            f"Sample index {sample_index} out of range (dataset has {len(dataset)} samples)"
        )

    keypoint, label = dataset[sample_index]

    keypoint = keypoint.unsqueeze(0).to(device, non_blocking=True)

    B, NC, M, T, V, C = keypoint.shape

    keypoint = keypoint.view(B * NC, M, T, V, C)

    logits = model(keypoint)

    probs = F.softmax(logits, dim=1)

    return label, probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ST-GCN++ single clip inference on NTU RGB+D",
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
        help="Path to the pre-trained .pth checkpoint",
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
        "--index",
        type=int,
        default=0,
        help="Index of the sample to inference on",
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
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("=" * 60)
    print("ST-GCN++ Single Clip Inference")
    print("=" * 60)
    print(f"Device    : {device}")
    print(f"Data      : {args.data}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split     : {args.split}")
    print(f"Modality  : {args.modality}")
    print(f"Sample Idx: {args.index}")
    print(f"Clips     : {args.num_clips} × {args.clip_len} frames")
    print("=" * 60)

    print("\nLoading annotation...")
    annotation = load_annotation(args.data)

    sample_info = get_sample_info(annotation, args.split, args.index)
    print(f"\nSample Info:")
    print(f"  Index       : {sample_info['index']}")
    print(f"  Frame Dir   : {sample_info['frame_dir']}")
    print(f"  True Label  : {sample_info['label']}")
    print(f"  Total Frames: {sample_info['total_frames']}")

    print("\nBuilding dataloader...")
    dataloader = build_dataloader(
        pkl_path=args.data,
        split=args.split,
        modality=args.modality,
        clip_len=args.clip_len,
        num_clips=args.num_clips,
        batch_size=1,
        num_workers=0,
        max_samples=args.index + 1,
    )
    print(f"  Dataset size: {len(dataloader.dataset)} samples")

    print("\nBuilding model...")
    model = STGCNpp(in_channels=3, num_classes=120)
    model.to(device)

    print(f"\nLoading checkpoint: {ckpt_path}")
    load_checkpoint(model, str(ckpt_path), device)

    print("\nRunning inference...")
    true_label, probs = inference_single(model, dataloader, device, args.index)

    num_classes = probs.shape[1]
    top5_probs, top5_indices = probs[0].topk(5)

    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"\nTrue Label: {true_label}")
    print("\nTop-5 Predictions:")
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices), 1):
        marker = " <-- TRUE LABEL" if idx.item() == true_label else ""
        print(
            f"  {i}. Class {idx.item():>3} | Probability: {prob.item() * 100:>6.2f}%{marker}"
        )
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
