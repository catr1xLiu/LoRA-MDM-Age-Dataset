"""
ST-GCN++ age classification inference script.

Loads a fine-tuned age classifier checkpoint and evaluates per-class accuracy
on the Van Criekinge dataset validation split.

Reports
-------
- Overall accuracy
- Per-class accuracy (Young / Adult / Elderly)
- Confusion matrix
- Rough accuracy: samples within 3 years of an age group boundary
  may be classified as either adjacent group and still count as correct.

Age group boundaries
--------------------
    Young   (0): age <  40
    Adult   (1): age 40–64
    Elderly (2): age >= 65

Rough accuracy tolerance zones (±3 years from each boundary):
    age 37–39  → Young or Adult both count as correct
    age 40–42  → Adult or Young both count as correct
    age 62–64  → Adult or Elderly both count as correct
    age 65–67  → Elderly or Adult both count as correct

Usage
-----
    uv run python batch_age_inference.py \\
        --checkpoint checkpoints/vc_age_unfreeze2block_newsplit.pth \\
        --data       data/vc_ntu25_xsub.pkl

    # With rough accuracy (requires access to original SMPL json files):
    uv run python batch_age_inference.py \\
        --checkpoint checkpoints/vc_age_unfreeze2block_newsplit.pth \\
        --data       data/vc_ntu25_xsub.pkl \\
        --smpl-dir   ../data/fitted_smpl_all_3_tuned
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from stgcnpp import AgeClassifierHead, NTUDataset, STGCNpp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGE_GROUPS = {0: "Young  (<40) ", 1: "Adult  (40-64)", 2: "Elderly (≥65)"}
AGE_BOUNDARIES = [40, 65]  # class 0→1, class 1→2
ROUGH_TOLERANCE = 3  # years


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_age_checkpoint(model: STGCNpp, ckpt_path: str, device: torch.device) -> dict:
    """Load a fine-tuned age classifier checkpoint saved by train_vc.py.

    The checkpoint format is::

        {
            'state_dict': model.state_dict(),   # backbone.* + head.*
            'epoch':      int,
            'val_acc':    float,
        }

    The model's head must already be replaced with AgeClassifierHead before
    calling this function.

    Args:
        model:     STGCNpp with AgeClassifierHead already in place.
        ckpt_path: Path to the .pth checkpoint.
        device:    Target device.

    Returns:
        The raw checkpoint dict (contains 'epoch' and 'val_acc').
    """
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = raw.get("state_dict", raw)

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"  [WARN] Missing keys  : {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    return raw


# ---------------------------------------------------------------------------
# Age lookup from SMPL json files
# ---------------------------------------------------------------------------


def build_subject_age_map(smpl_dir: Path) -> dict[str, int]:
    """Build a subject_id → age mapping from SMPL fit json files.

    Each subject sub-directory contains one or more *_smpl_params.json files.
    The age is stored in ``subject_metadata.age``.  Only one file per subject
    is needed; we take the first one found.

    Args:
        smpl_dir: Root directory containing per-subject sub-directories
                  (e.g. ``../data/fitted_smpl_all_3_tuned``).

    Returns:
        Dict mapping subject IDs (e.g. ``'SUBJ01'``) to integer ages.
    """
    age_map: dict[str, int] = {}
    for subject_dir in sorted(smpl_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        for json_file in sorted(subject_dir.glob("*_smpl_params.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                subject_id = data["subject_id"]
                age = data["subject_metadata"]["age"]
                age_map[subject_id] = age
                break  # one file per subject is enough
            except (KeyError, json.JSONDecodeError):
                continue
    return age_map


_FRAME_DIR_RE = re.compile(r"^(SUBJ\d+)_")


def extract_subject_id(frame_dir: str) -> str | None:
    """Extract subject ID (e.g. 'SUBJ01') from a frame_dir string.

    frame_dir format: ``{subject_id}_{trial_name}``  e.g. ``SUBJ01_SUBJ1_2``
    """
    m = _FRAME_DIR_RE.match(frame_dir)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Rough accuracy helpers
# ---------------------------------------------------------------------------


def is_rough_correct(
    pred: int,
    true_label: int,
    age: int,
    tolerance: int = ROUGH_TOLERANCE,
) -> bool:
    """Return True if pred is 'roughly correct' given the subject's actual age.

    Standard correct:  pred == true_label
    Rough correct:     pred is an adjacent class AND age falls within
                       `tolerance` years of the shared boundary.

    Args:
        pred:       Predicted age group (0/1/2).
        true_label: Ground-truth age group (0/1/2).
        age:        Subject's actual age in years.
        tolerance:  Years from boundary that permit an adjacent-class guess.

    Returns:
        True if the prediction is standard or roughly correct.
    """
    if pred == true_label:
        return True

    # Boundary at 40 (Young ↔ Adult)
    if abs(pred - true_label) == 1 and min(pred, true_label) == 0:
        # Either Young predicted as Adult OR Adult predicted as Young
        return abs(age - AGE_BOUNDARIES[0]) < tolerance

    # Boundary at 65 (Adult ↔ Elderly)
    if abs(pred - true_label) == 1 and min(pred, true_label) == 1:
        # Either Adult predicted as Elderly OR Elderly predicted as Adult
        return abs(age - AGE_BOUNDARIES[1]) < tolerance

    # Wrong by two classes (Young ↔ Elderly) — never rough correct
    return False


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference(
    model: STGCNpp,
    dataset: NTUDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    num_clips: int,
) -> tuple[list[int], list[int]]:
    """Run multi-clip inference over the dataset.

    Args:
        model:       STGCNpp with AgeClassifierHead, in eval mode.
        dataset:     NTUDataset (test_mode=True).
        device:      Inference device.
        batch_size:  Samples per batch.
        num_workers: DataLoader workers.
        num_clips:   Clips per sample (must match dataset).

    Returns:
        (all_preds, all_labels) — integer lists of length len(dataset).
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for keypoint, labels in tqdm(loader, desc="Evaluating", unit="batch"):
        B, NC, M, T, V, C = keypoint.shape
        assert NC == num_clips, f"Expected {num_clips} clips per sample, got {NC}"

        keypoint = keypoint.view(B * NC, M, T, V, C).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(keypoint).view(B, NC, -1)             # (B, NC, 3)
        probs = F.softmax(logits, dim=2).mean(dim=1)         # (B, 3)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_per_class_accuracy(
    preds: list[int],
    labels: list[int],
    num_classes: int = 3,
) -> None:
    """Print per-class accuracy and a confusion matrix."""
    # --- Per-class accuracy ---
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for p, t in zip(preds, labels):
        class_total[t] += 1
        if p == t:
            class_correct[t] += 1

    print("\n--- Per-Class Accuracy ---")
    print(f"  {'Class':<20}  {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
    print("  " + "-" * 46)
    for c in range(num_classes):
        acc = 100 * class_correct[c] / class_total[c] if class_total[c] > 0 else float("nan")
        print(f"  {AGE_GROUPS[c]:<20}  {class_correct[c]:>7}  {class_total[c]:>5}  {acc:>7.2f}%")
    overall_acc = 100 * sum(class_correct) / len(labels)
    print("  " + "-" * 46)
    print(f"  {'Overall':<20}  {sum(class_correct):>7}  {len(labels):>5}  {overall_acc:>7.2f}%")

    # --- Confusion matrix ---
    conf = [[0] * num_classes for _ in range(num_classes)]
    for p, t in zip(preds, labels):
        conf[t][p] += 1

    print("\n--- Confusion Matrix (rows=true, cols=pred) ---")
    short_names = ["Young", "Adult", "Elderly"]
    header = f"  {'':15}" + "".join(f"{n:>9}" for n in short_names)
    print(header)
    print("  " + "-" * (15 + 9 * num_classes))
    for t in range(num_classes):
        row = f"  {AGE_GROUPS[t]:<15}" + "".join(f"{conf[t][p]:>9}" for p in range(num_classes))
        print(row)


def print_rough_accuracy(
    preds: list[int],
    labels: list[int],
    frame_dirs: list[str],
    age_map: dict[str, int],
    tolerance: int = ROUGH_TOLERANCE,
    num_classes: int = 3,
) -> None:
    """Print rough accuracy statistics.

    Samples whose actual age is within `tolerance` years of a class boundary
    may be classified as either adjacent class and still count as correct.
    Samples whose subject is not in `age_map` are excluded from rough accuracy.
    """
    rough_correct = 0
    rough_total = 0
    missing_age = 0
    boundary_zone_totals = [0, 0]  # [boundary@40, boundary@65]
    boundary_zone_correct = [0, 0]
    # per-class rough stats
    class_rough_correct = [0] * num_classes
    class_rough_total = [0] * num_classes

    for pred, true_label, frame_dir in zip(preds, labels, frame_dirs):
        subject_id = extract_subject_id(frame_dir)
        age = age_map.get(subject_id) if subject_id else None

        if age is None:
            missing_age += 1
            continue

        rough_total += 1
        class_rough_total[true_label] += 1
        ok = is_rough_correct(pred, true_label, age, tolerance)
        if ok:
            rough_correct += 1
            class_rough_correct[true_label] += 1

        # Track boundary zone samples independently
        if abs(age - AGE_BOUNDARIES[0]) < tolerance:
            boundary_zone_totals[0] += 1
            if ok:
                boundary_zone_correct[0] += 1
        if abs(age - AGE_BOUNDARIES[1]) < tolerance:
            boundary_zone_totals[1] += 1
            if ok:
                boundary_zone_correct[1] += 1

    if rough_total == 0:
        print("\n[WARN] No samples with known age — rough accuracy unavailable.")
        return

    overall_rough = 100 * rough_correct / rough_total

    print(f"\n--- Rough Accuracy (±{tolerance} yr from boundary) ---")
    if missing_age:
        print(f"  [NOTE] {missing_age} sample(s) excluded (subject age unknown)")
    print(f"  {'Class':<20}  {'Rough OK':>8}  {'Total':>5}  {'Rough Acc':>9}")
    print("  " + "-" * 48)
    for c in range(num_classes):
        if class_rough_total[c] > 0:
            acc = 100 * class_rough_correct[c] / class_rough_total[c]
        else:
            acc = float("nan")
        print(f"  {AGE_GROUPS[c]:<20}  {class_rough_correct[c]:>8}  {class_rough_total[c]:>5}  {acc:>8.2f}%")
    print("  " + "-" * 48)
    print(f"  {'Overall':<20}  {rough_correct:>8}  {rough_total:>5}  {overall_rough:>8.2f}%")

    # Boundary zone breakdown
    bnames = [f"Near-40 (age {40 - tolerance}–{40 + tolerance - 1})",
              f"Near-65 (age {65 - tolerance}–{65 + tolerance - 1})"]
    print(f"\n  Boundary zone breakdown:")
    for i, name in enumerate(bnames):
        if boundary_zone_totals[i] > 0:
            acc = 100 * boundary_zone_correct[i] / boundary_zone_totals[i]
            print(f"    {name}: {boundary_zone_correct[i]}/{boundary_zone_totals[i]} ({acc:.1f}%)")
        else:
            print(f"    {name}: no samples in val set")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ST-GCN++ age classification inference on Van Criekinge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to age classifier .pth checkpoint (saved by train_vc.py)",
    )
    parser.add_argument(
        "--data",
        default="data/vc_ntu25_xsub.pkl",
        help="Path to the Van Criekinge NTU-25 pickle",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--smpl-dir",
        type=Path,
        default=None,
        help=(
            "Path to fitted SMPL directory (e.g. ../data/fitted_smpl_all_3_tuned). "
            "Required for rough accuracy; if omitted, rough accuracy is skipped."
        ),
    )
    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=2,
        help="Number of GCN blocks unfrozen during training (informational only)",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=10,
        help="Clips per sample for test-time aggregation",
    )
    parser.add_argument(
        "--clip-len",
        type=int,
        default=100,
        help="Frames per clip",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size (samples, not clips)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes",
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

    print("=" * 60)
    print("ST-GCN++ Age Classifier Evaluation")
    print("=" * 60)
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Data         : {args.data}")
    print(f"Split        : {args.split}")
    print(f"Device       : {device}")
    print(f"Clips        : {args.num_clips} × {args.clip_len} frames")

    # --- Dataset ---
    print("\nBuilding dataset...")
    dataset = NTUDataset(
        pkl_path=args.data,
        split=args.split,
        modality="joint",
        clip_len=args.clip_len,
        num_clips=args.num_clips,
        test_mode=True,
    )
    print(f"  {len(dataset):,} samples in '{args.split}' split")

    # Collect frame_dirs in dataset order (needed for age lookup)
    frame_dirs = [s["frame_dir"] for s in dataset.samples]

    # --- Model ---
    print("\nBuilding model...")
    model = STGCNpp(in_channels=3, num_classes=120)
    model.head = AgeClassifierHead(in_channels=256, z_dim=32, num_classes=3, dropout=0.3)
    model.to(device)

    # --- Checkpoint ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path}")
    meta = load_age_checkpoint(model, str(ckpt_path), device)
    saved_epoch = meta.get("epoch", "?")
    saved_val_acc = meta.get("val_acc", None)
    if saved_val_acc is not None:
        print(f"  Saved at epoch {saved_epoch}, val_acc = {100 * saved_val_acc:.2f}%")

    # --- Inference ---
    print("\nRunning inference...")
    t0 = time.perf_counter()
    preds, labels = run_inference(
        model, dataset, device, args.batch_size, args.num_workers, args.num_clips
    )
    elapsed = time.perf_counter() - t0
    print(f"Elapsed: {elapsed:.1f}s")

    # --- Standard accuracy ---
    print_per_class_accuracy(preds, labels)

    # --- Rough accuracy ---
    if args.smpl_dir is not None:
        smpl_dir = args.smpl_dir.expanduser().resolve()
        if smpl_dir.exists():
            print(f"\nBuilding subject age map from: {smpl_dir}")
            age_map = build_subject_age_map(smpl_dir)
            print(f"  Found ages for {len(age_map)} subjects")
            print_rough_accuracy(preds, labels, frame_dirs, age_map)
        else:
            print(f"\n[WARN] --smpl-dir not found: {smpl_dir}. Rough accuracy skipped.")
    else:
        print("\n[INFO] Rough accuracy skipped (pass --smpl-dir to enable).")


if __name__ == "__main__":
    main()
