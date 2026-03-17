"""
ST-GCN++ inference script.

Loads a pre-trained checkpoint and evaluates Top-1 accuracy on any NTU
RGB+D split (e.g. xsub_val, xset_val), or runs inference on Van Criekinge dataset.

Usage
-----
    # NTU RGB+D dataset
    uv run python batch_inference.py \
        --data       data/ntu120_3danno.pkl \
        --checkpoint checkpoints/j.pth \
        --modality   joint \
        --split      xsub_val

    # Van Criekinge dataset (auto-detected from available splits)
    uv run python batch_inference.py \
        --data       data/vc_ntu25.pkl \
        --checkpoint checkpoints/j.pth \
        --modality   joint

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
import pickle
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from stgcnpp import STGCNpp, build_dataloader


# NTU-120 action labels that correspond to walking-related actions
# fmt: off
# NTU-120 walking-related action labels (used to validate gait recognition)
# 58: walking towards (one person walks towards camera)
# 59: walking apart (one person walks away from camera)
# 115: follow (one person follows another) - walking behavior
NTU_WALKING_LABELS = {58, 59, 115}
# fmt: on

# fmt: off
NTU_ACTION_NAMES = {
    0: "drink water",           1: "eat meal",              2: "brush teeth",
    3: "brush hair",            4: "drop",                  5: "pick up",
    6: "throw",                 7: "sit down",              8: "stand up",
    9: "clapping",              10: "reading",              11: "writing",
    12: "tear up paper",        13: "put on jacket",        14: "take off jacket",
    15: "put on a shoe",        16: "take off a shoe",      17: "put on glasses",
    18: "take off glasses",     19: "put on a hat/cap",     20: "take off a hat/cap",
    21: "cheer up",             22: "hand waving",          23: "kicking something",
    24: "reach into pocket",    25: "hopping",              26: "jump up",
    27: "phone call",           28: "play with phone",      29: "type on keyboard",
    30: "point to something",   31: "taking a selfie",      32: "check time (watch)",
    33: "rub two hands",        34: "nod head/bow",         35: "shake head",
    36: "wipe face",            37: "salute",               38: "put palms together",
    39: "cross hands in front",
    40: "sneeze/cough",         41: "staggering",           42: "falling down",
    43: "headache",             44: "chest pain",           45: "back pain",
    46: "neck pain",            47: "nausea/vomiting",      48: "fan self",
    49: "punch/slap",           50: "kicking",              51: "pushing",
    52: "pat on back",          53: "point finger",         54: "hugging",
    55: "giving object",        56: "touch pocket",         57: "shaking hands",
    58: "walking towards",      59: "walking apart",
    60: "put on headphone",     61: "take off headphone",   62: "shoot at basket",
    63: "bounce ball",          64: "tennis bat swing",     65: "juggle table tennis ball",
    66: "hush",                 67: "flick hair",           68: "thumb up",
    69: "thumb down",           70: "make OK sign",         71: "make victory sign",
    72: "staple book",          73: "counting money",       74: "cutting nails",
    75: "cutting paper",        76: "snap fingers",         77: "open bottle",
    78: "sniff/smell",          79: "squat down",           80: "toss a coin",
    81: "fold paper",           82: "ball up paper",        83: "play magic cube",
    84: "apply cream on face",  85: "apply cream on hand",  86: "put on bag",
    87: "take off bag",         88: "put object into bag",  89: "take object out of bag",
    90: "open a box",           91: "move heavy objects",   92: "shake fist",
    93: "throw up cap/hat",     94: "capitulate",           95: "cross arms",
    96: "arm circles",          97: "arm swings",           98: "run on the spot",
    99: "butt kicks",           100: "cross toe touch",    101: "side kick",
    102: "yawn",                103: "stretch oneself",     104: "blow nose",
    105: "hit with object",     106: "wield knife",         107: "knock over",
    108: "grab stuff",          109: "shoot with gun",      110: "step on foot",
    111: "high-five",           112: "cheers and drink",    113: "carry object",
    114: "take a photo",        115: "follow",              116: "whisper",
    117: "exchange things",     118: "support somebody",    119: "rock-paper-scissors",
}
# fmt: on


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
        return "head." + key[len("cls_head.") :]
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
        print(
            f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        print(
            f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
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
        labels = labels.to(device, non_blocking=True)

        # Fold clips into the batch dimension: (B*NC, M, T, V, C)
        keypoint = keypoint.view(B * NC, M, T, V, C)

        # Forward pass: (B*NC, num_classes)
        logits = model(keypoint)

        # Restore clip dimension and aggregate
        # (B*NC, num_classes) → (B, NC, num_classes)
        logits = logits.view(B, NC, -1)

        # 'prob' averaging: softmax per clip, then mean across clips
        probs = F.softmax(logits, dim=2).mean(dim=1)  # (B, num_classes)

        preds = probs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += B

    return correct / total


@torch.no_grad()
def evaluate_with_predictions(
    model: STGCNpp,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_clips: int,
) -> tuple[list[int], list[int], list[float]]:
    """Run inference and return predictions with probabilities.

    Args:
        model:      The STGCNpp model in eval mode.
        dataloader: DataLoader yielding (keypoint, label) batches.
        device:     Inference device.
        num_clips:  Number of clips per sample.

    Returns:
        (top1_predictions, labels, top1_confidences, all_top10_predictions)
        - top1_predictions: list of top-1 predicted labels
        - labels: list of ground truth labels
        - top1_confidences: list of top-1 confidence scores
        - all_top10_predictions: list of lists, each containing top10 labels for each sample
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    all_top10 = []
    all_top10_conf = []

    for keypoint, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
        B, NC, M, T, V, C = keypoint.shape
        assert NC == num_clips, f"Expected {num_clips} clips, got {NC}"

        keypoint = keypoint.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        keypoint = keypoint.view(B * NC, M, T, V, C)
        logits = model(keypoint)
        logits = logits.view(B, NC, -1)

        probs = F.softmax(logits, dim=2).mean(dim=1)

        # Get top-1 predictions and confidences
        confidences, preds = probs.max(dim=1)

        # Get top-10 predictions for each sample
        top10_probs, top10_preds = probs.topk(k=10, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_confidences.extend(confidences.cpu().tolist())
        all_top10.extend(top10_preds.cpu().tolist())
        all_top10_conf.extend(top10_probs.cpu().tolist())

    return all_preds, all_labels, all_confidences, all_top10, all_top10_conf


def generate_top20_barchart(
    all_top10: list[list[int]],
    all_top10_conf: list[list[float]],
    output_path: str,
    walking_labels: set,
) -> None:
    """Generate a horizontal bar chart of the top 20 most recognized results.

    Weighted by confidence scores from top10 predictions.

    Args:
        all_top10: List of lists, each containing top10 predicted labels for each sample
        all_top10_conf: List of lists, confidence scores for each top10 prediction
        output_path: Path to save the bar chart image
        walking_labels: Set of walking-related NTU action labels for marking
    """
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")

    label_weighted_counts: dict[int, float] = {}
    for top10_list, top10_conf_list in zip(all_top10, all_top10_conf):
        for label, conf in zip(top10_list, top10_conf_list):
            label_weighted_counts[label] = label_weighted_counts.get(label, 0.0) + conf

    top20_labels = sorted(
        label_weighted_counts.items(), key=lambda x: x[1], reverse=True
    )[:20]

    y_labels = [
        f"{label} ({NTU_ACTION_NAMES.get(label, 'unknown')})"
        for label, _ in top20_labels
    ]
    counts = [count for _, count in top20_labels]
    labels = [label for label, _ in top20_labels]

    walking_color = "#27ae60"
    other_color = "#3498db"
    colors = [
        walking_color if label in walking_labels else other_color for label in labels
    ]

    fig, ax = plt.subplots(figsize=(14, 10))
    y_positions = range(len(y_labels) - 1, -1, -1)
    bars = ax.barh(
        y_positions, counts, color=colors, edgecolor="white", linewidth=0.8, height=0.8
    )

    ax.set_xlabel(
        "Confidence-weighted Count (top10 predictions)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("NTU Action Label", fontsize=12, fontweight="bold")
    ax.set_title(
        "Top 20 Most Recognized NTU Action Labels\n(Van Criekinge Gait Dataset)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{count:.1f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=walking_color, edgecolor="white", label="Walking-related"),
        Patch(facecolor=other_color, edgecolor="white", label="Other actions"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[INFO] Bar chart saved to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ST-GCN++ inference on NTU RGB+D or Van Criekinge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="data/ntu120_3danno.pkl",
        help="Path to the skeleton pickle file",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the pre-trained .pth checkpoint (joint or bone model)",
    )
    parser.add_argument(
        "--task",
        choices=["action", "age"],
        default="action",
        help="Task type: 'action' for action recognition (output accuracy), 'age' for age classification (ground truth is age, check if predictions are walking-related actions)",
    )
    parser.add_argument(
        "--modality",
        choices=["joint", "bone"],
        default="joint",
        help="Input modality: 'joint' (raw coordinates) or 'bone' (edge vectors)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to evaluate (e.g. xsub_val, xset_val, val). Auto-detected based on data file if not specified.",
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
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the bar chart (only for age task)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-detect split if not specified
    split = args.split
    if split is None:
        with open(args.data, "rb") as f:
            data = pickle.load(f)
        available_splits = list(data.get("split", {}).keys())
        if "val" in available_splits:
            split = "val"
        elif "xsub_val" in available_splits:
            split = "xsub_val"
        else:
            split = available_splits[0] if available_splits else "xsub_val"
        print(f"[INFO] Auto-detected split: {split} (available: {available_splits})")

    device = torch.device(args.device)
    print(f"Device       : {device}")
    print(f"Task         : {args.task}")
    print(f"Split        : {split}")
    print(f"Modality     : {args.modality}")
    print(f"Clips        : {args.num_clips} × {args.clip_len} frames")
    if args.max_samples:
        print(f"Max          : first {args.max_samples:,} samples")

    # --- Dataset ---
    print("\nBuilding dataset...")
    dataloader = build_dataloader(
        pkl_path=args.data,
        split=split,
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

    if args.task == "age":
        # For age classification task: ground truth is age group, check if
        # predictions are walking-related actions (expected for gait data)
        preds, labels, confidences, all_top10, all_top10_conf = (
            evaluate_with_predictions(model, dataloader, device, args.num_clips)
        )

        walking_preds = sum(1 for p in preds if p in NTU_WALKING_LABELS)
        total = len(preds)
        walking_pct = walking_preds / total * 100

        print(f"\n--- Age Classification Task Results ---")
        print(f"Total samples     : {total}")
        print(f"Walking-related  : {walking_preds} ({walking_pct:.1f}%)")

        from collections import Counter

        pred_counts = Counter(preds)
        print(f"\nTop 10 predicted NTU action labels (top-1):")
        for label, count in pred_counts.most_common(10):
            walking_tag = " [WALKING]" if label in NTU_WALKING_LABELS else ""
            print(f"  Label {label:3d}: {count:3d}{walking_tag}")

        # Generate bar chart if output path is specified
        if args.output:
            generate_top20_barchart(
                all_top10, all_top10_conf, args.output, NTU_WALKING_LABELS
            )
    else:
        # For action recognition task: compute accuracy
        top1 = evaluate(model, dataloader, device, args.num_clips)
        print(f"\nTop-1 accuracy : {top1 * 100:.2f}%")

    elapsed = time.perf_counter() - t0
    print(f"Elapsed        : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
