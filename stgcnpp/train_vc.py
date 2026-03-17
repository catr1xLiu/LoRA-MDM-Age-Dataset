"""
Fine-tune ST-GCN++ for 3-class age classification on the Van Criekinge dataset.

Strategy
--------
* Load pretrained NTU-120 bone backbone (stgcnpp_ntu120_3dkp_joint.pth).
* Replace the 120-class head with AgeClassifierHead (256→32→3 bottleneck).
* Freeze the entire backbone, then unfreeze the last 2 GCN blocks.
* Train with AdamW, class-weighted cross-entropy, 50 epochs.
* Save the best-val-accuracy checkpoint to checkpoints/vc_age_best.pth.

Usage
-----
    python train_vc.py \
        --checkpoint checkpoints/stgcnpp_ntu120_3dkp_joint.pth \
        --epochs 50

    # Quick smoke-test (5 epochs, small batch):
    python train_vc.py --epochs 5 --batch-size 8
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from stgcnpp import AgeClassifierHead, NTUDataset, STGCNpp

# ---------------------------------------------------------------------------
# Age group names (for reporting)
# ---------------------------------------------------------------------------
AGE_GROUPS = {0: "Young (<40)", 1: "Adult (40-64)", 2: "Elderly (≥65)"}

# ---------------------------------------------------------------------------
# Checkpoint loading (PYSKL format → our model)
# ---------------------------------------------------------------------------


def _remap_key(key: str) -> str | None:
    if key.startswith("backbone."):
        return key
    if key.startswith("cls_head."):
        return "head." + key[len("cls_head."):]
    return None


def load_pretrained_backbone(model: STGCNpp, ckpt_path: str, device: torch.device) -> None:
    """Load a PYSKL-format NTU-120 checkpoint into the model.

    Only backbone weights are loaded; the head is intentionally left at its
    random initialisation (it will be replaced immediately after).
    """
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = raw.get("state_dict", raw)

    remapped: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = _remap_key(k)
        if new_k is not None:
            remapped[new_k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    backbone_missing = [k for k in missing if k.startswith("backbone.")]
    if backbone_missing:
        print(f"  [WARN] Missing backbone keys: {backbone_missing[:5]}")
    print(f"  Loaded {len(remapped)} keys  |  {len(missing)} missing  |  {len(unexpected)} unexpected")


def save_checkpoint(model: STGCNpp, epoch: int, val_acc: float, path: Path) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_train_loader(
    pkl_path: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Single-clip stochastic training dataloader."""
    dataset = NTUDataset(
        pkl_path=pkl_path,
        split="train",
        modality="joint",
        clip_len=100,
        num_clips=1,
        test_mode=False,  # stochastic sampling
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def build_val_loader(
    pkl_path: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """10-clip deterministic validation dataloader (matches inference eval)."""
    dataset = NTUDataset(
        pkl_path=pkl_path,
        split="val",
        modality="joint",
        clip_len=100,
        num_clips=10,
        test_mode=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def compute_class_weights(dataset: NTUDataset, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training split."""
    labels = [int(ann["label"]) for ann in dataset.samples]
    counts = torch.bincount(torch.tensor(labels), minlength=3).float()
    weights = counts.sum() / (3.0 * counts)
    print(f"  Class counts : {counts.int().tolist()}  (Young / Adult / Elderly)")
    print(f"  Class weights: {weights.tolist()}")
    return weights.to(device)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch.

    Returns:
        (mean_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for keypoint, labels in tqdm(loader, desc="  Train", leave=False, unit="batch"):
        # keypoint: (B, num_clips=1, M, T, V, C)
        B, NC, M, T, V, C = keypoint.shape
        keypoint = keypoint.view(B * NC, M, T, V, C).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(keypoint)          # (B, 3)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += B

    return total_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Multi-clip validation (10 clips per sample, prob-average aggregation)."""
    model.eval()
    correct = 0
    total = 0

    for keypoint, labels in tqdm(loader, desc="  Val  ", leave=False, unit="batch"):
        B, NC, M, T, V, C = keypoint.shape
        keypoint = keypoint.view(B * NC, M, T, V, C).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(keypoint).view(B, NC, -1)       # (B, NC, 3)
        probs = F.softmax(logits, dim=2).mean(dim=1)   # (B, 3)
        correct += (probs.argmax(dim=1) == labels).sum().item()
        total += B

    return correct / total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune ST-GCN++ for 3-class age classification on Van Criekinge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/stgcnpp_ntu120_3dkp_joint.pth",
        help="Pretrained NTU-120 checkpoint to initialise the backbone",
    )
    parser.add_argument(
        "--data",
        default="data/vc_ntu25.pkl",
        help="Path to the Van Criekinge NTU-25 pickle",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Number of GCN blocks to unfreeze from the END of the backbone "
            "(0 = fully frozen backbone, train head only; max = 10). "
            "STGCNBackbone has 10 GCN blocks total (backbone.gcn[0..9]): "
            "blocks 0-3 = 64ch, block 4 = 64→128ch, blocks 5-6 = 128ch, "
            "block 7 = 128→256ch, blocks 8-9 = 256ch."
        ),
    )
    parser.add_argument(
        "--output",
        default="checkpoints/vc_age_best.pth",
        help="Where to save the best checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("ST-GCN++ Age Classifier Fine-tuning")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Data       : {args.data}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Unfreeze   : {args.unfreeze_blocks} GCN block(s)")
    print(f"Output     : {args.output}")

    # --- Dataloaders ---
    print("\nBuilding dataloaders...")
    train_loader = build_train_loader(args.data, args.batch_size, args.num_workers)
    val_loader = build_val_loader(args.data, args.batch_size, args.num_workers)
    print(f"  Train: {len(train_loader.dataset):,} samples  |  Val: {len(val_loader.dataset):,} samples")

    # --- Class weights ---
    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_loader.dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Model ---
    print("\nBuilding model...")
    model = STGCNpp(in_channels=3, num_classes=120)
    model.to(device)

    # Load pretrained backbone
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"  Loading pretrained weights from: {ckpt_path}")
    load_pretrained_backbone(model, str(ckpt_path), device)

    # Replace head with age bottleneck head
    model.head = AgeClassifierHead(in_channels=256, z_dim=32, num_classes=3, dropout=0.3)
    model.head.to(device)
    print(f"  Replaced head with AgeClassifierHead(256→32→3)")

    # Freeze backbone, then unfreeze the last N GCN blocks
    for p in model.backbone.parameters():
        p.requires_grad = False
    if args.unfreeze_blocks > 0:
        for block in model.backbone.gcn[-args.unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # --- Optimizer ---
    # Collect unfrozen backbone params (may be empty if unfreeze_blocks=0)
    if args.unfreeze_blocks > 0:
        backbone_params = [
            p for block in model.backbone.gcn[-args.unfreeze_blocks:]
            for p in block.parameters()
        ]
    else:
        backbone_params = []
    head_params = list(model.head.parameters())

    param_groups = [{"params": head_params, "lr": 1e-3, "weight_decay": 1e-4}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": 1e-4, "weight_decay": 1e-4})
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Training loop ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    t0 = time.perf_counter()

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Acc':>8}  {'Best':>8}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, val_loader, device)
        scheduler.step()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            save_checkpoint(model, epoch, val_acc, output_path)

        best_marker = " *" if is_best else ""
        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {100*train_acc:>8.2f}%  {100*val_acc:>7.2f}%  {100*best_val_acc:>7.2f}%{best_marker}"
        )

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Best val accuracy: {100*best_val_acc:.2f}%")
    print(f"Checkpoint saved : {output_path}")


if __name__ == "__main__":
    main()
