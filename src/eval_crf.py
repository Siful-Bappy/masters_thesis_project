"""
Evaluate MobileNet classifier accuracy across H.264 CRF levels.

Loads the trained model (models/mobilenet_classifier.pt) and tests it
on the original dataset + all compressed CRF versions, then produces:

    results/accuracy_vs_crf.csv    ← per-CRF accuracy numbers
    results/accuracy_vs_crf.png    ← plot (matches KICS paper Fig style)

Run:
    conda activate venv_masters_thesis_project
    python src/eval_crf.py
"""

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

ROOT      = Path(__file__).parent.parent
IMG_DATA  = ROOT / "dataset" / "image_data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT))
from src.models.mobilenet import MobileNetV1

IMG_SIZE    = 128
BATCH_SIZE  = 256
NUM_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

# CRF=0 label = "original" (the uncompressed training data)
CRF_LABELS = [0, 18, 23, 28, 35, 43, 51]


def load_model(ckpt_path: Path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MobileNetV1(num_classes=ckpt["num_classes"],
                        alpha=ckpt.get("alpha", 1.25))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["class_to_idx"]


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


@torch.no_grad()
def evaluate_dataset(model, data_dir: Path, class_to_idx: dict, device):
    """Return (top1_acc, top5_acc, per_class_acc)."""
    if not data_dir.exists():
        return None, None, {}

    ds = datasets.ImageFolder(str(data_dir), transform=get_val_transform())

    # align class indices with the training mapping
    # (ImageFolder sorts alphabetically, should match, but verify)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    correct_top1 = correct_top5 = total = 0
    per_class_correct = {}
    per_class_total   = {}

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1)

        # top-1
        pred1  = probs.argmax(dim=1)
        correct_top1 += (pred1 == labels).sum().item()

        # top-5
        top5   = probs.topk(5, dim=1).indices
        correct_top5 += sum(
            labels[i].item() in top5[i].tolist()
            for i in range(labels.size(0)))

        total += labels.size(0)

        # per-class
        for pred, lbl in zip(pred1.cpu().tolist(), labels.cpu().tolist()):
            per_class_total[lbl]   = per_class_total.get(lbl, 0) + 1
            per_class_correct[lbl] = per_class_correct.get(lbl, 0) + (pred == lbl)

    top1_acc = correct_top1 / total if total > 0 else 0
    top5_acc = correct_top5 / total if total > 0 else 0
    per_cls  = {k: per_class_correct.get(k, 0) / per_class_total[k]
                for k in per_class_total}
    return top1_acc, top5_acc, per_cls


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    ckpt_path = MODELS_DIR / "mobilenet_classifier.pt"
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}\n"
                 "Run: python src/train_classifier.py")

    print(f"Loading model: {ckpt_path}")
    model, class_to_idx = load_model(ckpt_path, DEVICE)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"Device: {DEVICE}")
    print("-" * 55)

    rows = []
    for crf in CRF_LABELS:
        if crf == 0:
            val_dir = IMG_DATA / "card_datset" / "valid"
            label   = "original (CRF≈0)"
        else:
            val_dir = IMG_DATA / f"card_datset_crf{crf}" / "valid"
            label   = f"CRF {crf}"

        print(f"Evaluating {label:<20} ...", end=" ", flush=True)

        if not val_dir.exists():
            print(f"SKIP — {val_dir.name} not found")
            print("  Run: python src/compress_crops.py")
            rows.append({"crf": crf, "label": label,
                         "top1_acc": None, "top5_acc": None})
            continue

        top1, top5, per_cls = evaluate_dataset(
            model, val_dir, class_to_idx, DEVICE)

        print(f"Top-1: {top1*100:5.2f}%   Top-5: {top5*100:5.2f}%")
        rows.append({
            "crf":      crf,
            "label":    label,
            "top1_acc": round(top1 * 100, 2),
            "top5_acc": round(top5 * 100, 2),
        })

    # ── save CSV ──────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "accuracy_vs_crf.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crf", "label", "top1_acc", "top5_acc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {csv_path}")

    # ── plot ──────────────────────────────────────────────────────────────
    df_plot = [r for r in rows if r["top1_acc"] is not None]
    if len(df_plot) < 2 or not HAS_MPL:
        if not HAS_MPL:
            print("matplotlib not installed — skipping plot (pip install matplotlib)")
        else:
            print("Not enough data to plot. Run compress_crops.py first.")
        return

    crfs   = [r["crf"]      for r in df_plot]
    top1s  = [r["top1_acc"] for r in df_plot]
    top5s  = [r["top5_acc"] for r in df_plot]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(crfs, top1s, "o-", color="#1565C0", linewidth=2,
            markersize=7, label="Top-1 Accuracy")
    ax.plot(crfs, top5s, "s--", color="#42A5F5", linewidth=1.5,
            markersize=6, label="Top-5 Accuracy")

    for crf, t1 in zip(crfs, top1s):
        ax.annotate(f"{t1:.1f}%", xy=(crf, t1),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, color="#1565C0")

    ax.set_xlabel("H.264 CRF (0 = near-lossless, 51 = maximum compression)",
                  fontsize=11)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=11)
    ax.set_title("MobileNetV1-1.25 Card Classification Accuracy vs H.264 CRF\n"
                 "(52 classes, trained on uncompressed crops, evaluated at each CRF)",
                 fontsize=11)
    ax.set_xticks(crfs)
    ax.set_xticklabels(["0\n(ref)"] + [str(c) for c in crfs[1:]])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()

    plot_path = RESULTS_DIR / "accuracy_vs_crf.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved     → {plot_path}")

    # ── print table ───────────────────────────────────────────────────────
    print(f"\n{'='*45}")
    print(f"{'CRF':<12} {'Top-1 Acc':>10}  {'Top-5 Acc':>10}")
    print("-" * 45)
    for row in rows:
        t1 = f"{row['top1_acc']:.2f}%" if row["top1_acc"] is not None else "N/A"
        t5 = f"{row['top5_acc']:.2f}%" if row["top5_acc"] is not None else "N/A"
        print(f"  {row['label']:<14} {t1:>10}  {t5:>10}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
