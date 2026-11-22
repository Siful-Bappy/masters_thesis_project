"""
Train one MobileNetV1-1.25 model per CRF level (20 epochs each).

Trains 7 models sequentially:
    mobilenet_classifier.pt   ← original (crf=0)
    mobilenet_crf18.pt
    mobilenet_crf23.pt
    mobilenet_crf28.pt
    mobilenet_crf35.pt
    mobilenet_crf43.pt
    mobilenet_crf51.pt

Each model is trained on images compressed at that CRF level and validated
on the same CRF's validation set.

Run:
    conda activate venv_masters_thesis_project
    python src/train_all_crfs.py

    # force re-train everything:
    python src/train_all_crfs.py --force

    # train specific CRFs only:
    python src/train_all_crfs.py --crfs 35 43 51
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.train_classifier import train

CRFS   = [0, 18, 23, 28, 35, 43, 51]
EPOCHS = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-train even if model already exists")
    parser.add_argument("--crfs", nargs="+", type=int, default=CRFS,
                        help="CRF levels to train (default: 18 23 28 35 43 51)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Epochs per model (default: {EPOCHS})")
    args = parser.parse_args()

    total_start = time.time()
    results = []

    for crf in args.crfs:
        t0 = time.time()
        train(crf=crf, epochs=args.epochs, force=args.force)
        elapsed = time.time() - t0

        stem = "mobilenet_classifier" if crf == 0 else f"mobilenet_crf{crf}"
        model_path = ROOT / "models" / f"{stem}.pt"
        if model_path.exists():
            import torch
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            val_acc = ckpt.get("val_acc", 0.0)
            results.append({"crf": crf, "val_acc": val_acc,
                            "time_min": elapsed / 60})
        else:
            results.append({"crf": crf, "val_acc": None,
                            "time_min": elapsed / 60})

    total_elapsed = time.time() - total_start

    print(f"\n{'='*55}")
    print(f"ALL DONE  —  total time: {total_elapsed/60:.1f} min")
    print(f"{'='*55}")
    print(f"{'CRF':<8} {'Best Val Acc':>14}  {'Time (min)':>12}")
    print("-" * 40)
    for r in results:
        acc_str = f"{r['val_acc']*100:.2f}%" if r["val_acc"] else "skipped"
        print(f"  {r['crf']:<6} {acc_str:>14}  {r['time_min']:>10.1f}")
    print(f"\nModels saved in: {ROOT / 'models'}/")


if __name__ == "__main__":
    main()
