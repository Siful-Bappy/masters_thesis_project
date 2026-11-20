"""
Train YOLOv11n for 3-class baccarat table detection: card, hand, chip.

Run:
    conda activate venv_masters_thesis_project
    python src/train_detector.py

Output:
    models/yolo11n_cards.pt   ← best checkpoint (copied here after training)
    runs/detect/yolo_cards/   ← full ultralytics training artifacts
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT       = Path(__file__).parent.parent
DATA_YAML  = ROOT / "data" / "data_detector.yaml"
MODELS_DIR = ROOT / "models"
SAVE_PATH  = MODELS_DIR / "yolo11n_cards.pt"

TRAIN_CFG = dict(
    data        = str(DATA_YAML),
    epochs      = 100,
    imgsz       = 640,
    batch       = 16,
    device      = 0,            # GPU 0
    workers     = 4,
    patience    = 20,           # early stop if no improvement for 20 epochs
    project     = str(ROOT / "runs" / "detect"),
    name        = "yolo_cards",
    exist_ok    = True,
    pretrained  = True,
    optimizer   = "AdamW",
    lr0         = 1e-3,
    weight_decay= 5e-4,
    warmup_epochs = 3,
    close_mosaic  = 10,         # disable mosaic augmentation last 10 epochs
    save_period   = 10,
    val           = True,
    plots         = True,
)


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    if SAVE_PATH.exists():
        print(f"[skip] Model already exists at {SAVE_PATH}")
        print("       Delete it to re-train.")
        return

    print("Loading pretrained YOLOv11n …")
    model = YOLO("yolo11n.pt")   # downloads ~6 MB from ultralytics hub

    print(f"Training on: {DATA_YAML}")
    print(f"Classes: card (0), hand (1), chip (2)")
    print("-" * 50)

    results = model.train(**TRAIN_CFG)

    # copy best weights to models/
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, SAVE_PATH)
        print(f"\nBest weights saved → {SAVE_PATH}")
    else:
        print(f"\nWARNING: best.pt not found at {best}")

    # print final metrics
    print("\n── Validation metrics ──")
    metrics = model.val()
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    for i, name in enumerate(["card", "hand", "chip"]):
        print(f"  AP50 {name:<5}: {metrics.box.ap50[i]:.4f}")


if __name__ == "__main__":
    main()
