"""Fine-tuning script for STGCN++ on a custom dataset.

Usage:
    uv run python train.py \\
        --data path/to/custom.pkl \\
        --pretrained ../stgcnpp_ntu120_3dkp_joint.pth \\
        --num_classes 10 \\
        --epochs 50 \\
        --lr 0.001
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import STGCNPP
from dataset import SkeletonDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to annotation .pkl file')
    p.add_argument('--pretrained', default=None, help='Path to pretrained .pth')
    p.add_argument('--num_classes', type=int, required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--clip_len', type=int, default=100)
    p.add_argument('--freeze_backbone', action='store_true',
                   help='Freeze backbone and only train the head')
    p.add_argument('--val_split', type=float, default=0.1,
                   help='Fraction of data to use as validation')
    p.add_argument('--save', default='best.pth', help='Output checkpoint path')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---- Dataset ----
    dataset = SkeletonDataset(args.data, clip_len=args.clip_len)
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=4, pin_memory=True)

    # ---- Model ----
    # Load pretrained with original 120 classes, then replace head
    model = STGCNPP(num_classes=120, pretrained=args.pretrained)
    model.reset_head(args.num_classes)
    if args.freeze_backbone:
        model.freeze_backbone()
    model = model.to(device)

    # ---- Optimiser ----
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0)
    criterion = nn.CrossEntropyLoss()

    # ---- Training loop ----
    best_acc = 0.
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}',
                         leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch:3d} | loss {avg_loss:.4f} | val_acc {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save)
            print(f'  Saved best model ({best_acc:.4f}) → {args.save}')

    print(f'\nTraining done. Best val accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    main()
