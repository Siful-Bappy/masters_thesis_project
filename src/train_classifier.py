"""
Train MobileNetV1-1.25 for 52-class playing-card classification.

Run:
    conda activate venv_masters_thesis_project
    python src/train_classifier.py

Output:
    models/mobilenet_classifier.pt   ← best checkpoint
    models/mobilenet_last.pt         ← last epoch checkpoint

Training details:
    - Input size  : 128 × 128
    - Backbone    : MobileNetV1, alpha=1.25, fdim=1280
    - Classes     : 52 (13 ranks × 4 suits, no card-back)
    - Optimiser   : AdamW, lr=1e-3, cosine schedule
    - Epochs      : 60
    - Augmentation: colour jitter, random affine, horizontal flip disabled
                    (flipping breaks suit/rank orientation)
"""

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
CARD_DIR   = ROOT / "dataset" / "image_data" / "card_datset"
MODELS_DIR = ROOT / "models"
SAVE_BEST  = MODELS_DIR / "mobilenet_classifier.pt"
SAVE_LAST  = MODELS_DIR / "mobilenet_last.pt"

import sys
sys.path.insert(0, str(ROOT))
from src.models.mobilenet import MobileNetV1

# ── hyper-parameters ──────────────────────────────────────────────────────────
IMG_SIZE   = 128
BATCH_SIZE = 128
EPOCHS     = 60
LR         = 1e-3
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


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    if SAVE_BEST.exists():
        print(f"[skip] Model already exists at {SAVE_BEST}")
        print("       Delete it to re-train.")
        return

    train_tf, val_tf = build_transforms()

    train_ds = datasets.ImageFolder(str(CARD_DIR / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(CARD_DIR / "valid"), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    num_classes = len(train_ds.classes)
    print(f"Classes     : {num_classes}")
    print(f"Train images: {len(train_ds):,}")
    print(f"Val images  : {len(val_ds):,}")
    print(f"Device      : {DEVICE}")
    print("-" * 50)

    # save class→idx mapping produced by ImageFolder (alphabetical sort)
    class_map = {"class_to_idx": train_ds.class_to_idx,
                 "idx_to_class": {v: k for k, v in train_ds.class_to_idx.items()}}
    (ROOT / "data" / "classifier_class_map.json").write_text(
        json.dumps(class_map, indent=2))

    model     = MobileNetV1(num_classes=num_classes, alpha=1.25).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          criterion, optimizer, DEVICE)
        va_loss, va_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                            va_loss=va_loss, va_acc=va_acc))

        star = " *" if va_acc > best_val_acc else ""
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"va_loss={va_loss:.4f}  va_acc={va_acc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"t={elapsed:.1f}s{star}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": va_acc,
                "class_to_idx": train_ds.class_to_idx,
                "num_classes": num_classes,
                "img_size": IMG_SIZE,
                "alpha": 1.25,
            }, SAVE_BEST)

    torch.save({
        "epoch": EPOCHS,
        "model_state": model.state_dict(),
        "val_acc": va_acc,
        "class_to_idx": train_ds.class_to_idx,
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
        "alpha": 1.25,
    }, SAVE_LAST)

    # save training history
    import pandas as pd
    pd.DataFrame(history).to_csv(ROOT / "models" / "classifier_history.csv",
                                 index=False)

    print(f"\nBest val accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Best checkpoint   → {SAVE_BEST}")
    print(f"Last checkpoint   → {SAVE_LAST}")


if __name__ == "__main__":
    main()
