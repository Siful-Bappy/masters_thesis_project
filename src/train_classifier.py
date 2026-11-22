"""
Train MobileNetV1-1.25 for 52-class playing-card classification.

Run:
    # original (clean) dataset — 60 epochs
    python src/train_classifier.py

    # compressed dataset at a specific CRF — 50 epochs
    python src/train_classifier.py --crf 43

    # force re-train even if model already exists
    python src/train_classifier.py --crf 43 --force

Output:
    models/mobilenet_classifier.pt      ← original (crf=0)
    models/mobilenet_crf18.pt           ← trained on CRF-18 data
    models/mobilenet_crf23.pt           ...
    models/mobilenet_crf28.pt
    models/mobilenet_crf35.pt
    models/mobilenet_crf43.pt
    models/mobilenet_crf51.pt
    models/history_crf{N}.csv           ← per-epoch training log
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT       = Path(__file__).parent.parent
IMG_DATA   = ROOT / "dataset" / "image_data"
MODELS_DIR = ROOT / "models"

import sys
sys.path.insert(0, str(ROOT))
from src.models.mobilenet import MobileNetV1

# ── hyper-parameters ──────────────────────────────────────────────────────────
IMG_SIZE     = 128
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 5e-4
NUM_WORKERS  = 4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05),
                                scale=(0.9, 1.1), shear=5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = seen = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        seen       += imgs.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = seen = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        seen       += imgs.size(0)
    return total_loss / seen, correct / seen


def get_paths(crf: int):
    """Return (train_dir, val_dir, save_best, save_last, history_csv)."""
    if crf == 0:
        data_dir  = IMG_DATA / "card_datset"
        model_stem = "mobilenet_classifier"
    else:
        data_dir  = IMG_DATA / f"card_datset_crf{crf}"
        model_stem = f"mobilenet_crf{crf}"

    return (
        data_dir / "train",
        data_dir / "valid",
        MODELS_DIR / f"{model_stem}.pt",
        MODELS_DIR / f"{model_stem}_last.pt",
        MODELS_DIR / f"history_{model_stem}.csv",
    )


def train(crf: int, epochs: int, force: bool):
    MODELS_DIR.mkdir(exist_ok=True)

    train_dir, val_dir, save_best, save_last, history_csv = get_paths(crf)

    label = "original" if crf == 0 else f"CRF {crf}"
    print(f"\n{'='*55}")
    print(f"Training on: {label}  |  epochs={epochs}")
    print(f"Train dir  : {train_dir}")
    print(f"Val dir    : {val_dir}")
    print(f"Save best  : {save_best}")
    print(f"{'='*55}")

    if save_best.exists() and not force:
        print(f"[skip] {save_best.name} already exists. Use --force to re-train.")
        return

    if not train_dir.exists():
        print(f"[error] {train_dir} not found.")
        if crf != 0:
            print(f"  Run: python src/compress_crops.py --crfs {crf}")
        return

    train_tf, val_tf = build_transforms()

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    num_classes = len(train_ds.classes)
    print(f"Classes     : {num_classes}")
    print(f"Train images: {len(train_ds):,}")
    print(f"Val images  : {len(val_ds):,}")
    print(f"Device      : {DEVICE}")
    print("-" * 55)

    if crf == 0:
        class_map = {"class_to_idx": train_ds.class_to_idx,
                     "idx_to_class": {v: k for k, v in train_ds.class_to_idx.items()}}
        (ROOT / "data" / "classifier_class_map.json").write_text(
            json.dumps(class_map, indent=2))

    model     = MobileNetV1(num_classes=num_classes, alpha=1.25).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    history      = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          criterion, optimizer, DEVICE)
        va_loss, va_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                            va_loss=va_loss, va_acc=va_acc))

        star = " *" if va_acc > best_val_acc else ""
        print(f"Epoch {epoch:3d}/{epochs}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"va_loss={va_loss:.4f}  va_acc={va_acc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"t={elapsed:.1f}s{star}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "epoch":        epoch,
                "crf":          crf,
                "model_state":  model.state_dict(),
                "val_acc":      va_acc,
                "class_to_idx": train_ds.class_to_idx,
                "num_classes":  num_classes,
                "img_size":     IMG_SIZE,
                "alpha":        1.25,
            }, save_best)

    torch.save({
        "epoch":        epochs,
        "crf":          crf,
        "model_state":  model.state_dict(),
        "val_acc":      va_acc,
        "class_to_idx": train_ds.class_to_idx,
        "num_classes":  num_classes,
        "img_size":     IMG_SIZE,
        "alpha":        1.25,
    }, save_last)

    import csv
    with open(history_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","tr_loss","tr_acc","va_loss","va_acc"])
        writer.writeheader()
        writer.writerows(history)

    print(f"\nBest val accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)")
    print(f"Best checkpoint   → {save_best}")


def main():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV1-1.25 card classifier")
    parser.add_argument(
        "--crf", type=int, default=0,
        help="CRF level to train on: 0=original, 18/23/28/35/43/51=compressed")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: 20)")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-train even if model already exists")
    args = parser.parse_args()

    epochs = args.epochs or 20
    train(crf=args.crf, epochs=epochs, force=args.force)


if __name__ == "__main__":
    main()
