"""
Compress card crop images at H.264 CRF levels for classifier robustness study.

Creates parallel compressed dataset directories:
    dataset/image_data/card_datset_crf18/ ... card_datset_crf51/

Each image is:
  1. Resized to 128×128 (the resolution MobileNet actually trains on)
  2. Independently H.264-compressed at the given CRF (GOP=1 → pure I-frame)
  3. Saved back as JPEG

Using 128×128 as the compression resolution is CORRECT because:
  - This is what MobileNet sees at both train and inference time
  - It avoids variable-size batching issues in ffmpeg
  - Compression artifacts are measured at the exact scale the model uses

All images in a split are batched into ONE video per CRF → ONE encode+decode
call per split per CRF = very fast (2 splits × 6 CRFs = 12 total ffmpeg calls).

Run:
    conda activate venv_masters_thesis_project
    python src/compress_crops.py              # all CRFs, all classes
    python src/compress_crops.py --crfs 43   # single CRF test
"""

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT      = Path(__file__).parent.parent
CARD_DIR  = ROOT / "dataset" / "image_data" / "card_datset"
OUT_BASE  = ROOT / "dataset" / "image_data"
SPLITS    = ["train", "valid"]
ALL_CRFS  = [18, 23, 28, 35, 43, 51]
IMG_SIZE  = 128   # resize before compression — matches training resolution


def collect_images(split: str) -> list[tuple[Path, str, str]]:
    """Return list of (image_path, class_name, filename) sorted by class."""
    items = []
    split_dir = CARD_DIR / split
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")):
            items.append((img, cls_dir.name, img.name))
    return items


def compress_split(items: list, out_split_dir: Path, crf: int,
                   skip_existing: bool) -> int:
    """
    Compress all images in `items` at `crf` and save to out_split_dir.

    Batches the entire split into one video:
      - resize all to IMG_SIZE × IMG_SIZE
      - encode with GOP=1 (all I-frames) at given CRF
      - decode all frames back and save to per-class folders

    Returns number of images written.
    """
    # filter to only images that need processing
    to_process = []
    for img_path, cls_name, fname in items:
        dst = out_split_dir / cls_name / fname
        if skip_existing and dst.exists():
            continue
        to_process.append((img_path, cls_name, fname))

    if not to_process:
        return 0

    # create output class folders
    for _, cls_name, _ in to_process:
        (out_split_dir / cls_name).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path  = Path(tmp)
        tmp_video = tmp_path / "batch.mp4"
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        # ── step 1: write concat list (all images scaled to IMG_SIZE) ─────
        concat_txt = tmp_path / "concat.txt"
        with open(concat_txt, "w") as f:
            for img_path, _, _ in to_process:
                f.write(f"file '{img_path}'\n")
                f.write("duration 1\n")
            # repeat last image to flush ffmpeg's concat buffer
            f.write(f"file '{to_process[-1][0]}'\n")

        # ── step 2: encode — scale to IMG_SIZE, all I-frames ──────────────
        encode_cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-vf", f"scale={IMG_SIZE}:{IMG_SIZE}",   # uniform size
            "-c:v", "libx264",
            "-crf",          str(crf),
            "-g",            "1",    # GOP=1 → every frame is I-frame
            "-keyint_min",   "1",
            "-sc_threshold", "0",
            "-preset",       "medium",
            str(tmp_video),
        ]
        r = subprocess.run(encode_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"\n  ENCODE ERROR (crf={crf}): {r.stderr[-300:]}")
            return 0

        # ── step 3: decode all frames ──────────────────────────────────────
        decode_cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(tmp_video),
            "-vframes", str(len(to_process)),
            "-q:v", "1",   # near-lossless JPEG output
            str(frame_dir / "frame_%06d.jpg"),
        ]
        r = subprocess.run(decode_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"\n  DECODE ERROR (crf={crf}): {r.stderr[-300:]}")
            return 0

        # ── step 4: copy decoded frames to correct class folders ───────────
        decoded = sorted(frame_dir.glob("frame_*.jpg"))
        if len(decoded) != len(to_process):
            print(f"\n  WARNING: expected {len(to_process)} frames, "
                  f"got {len(decoded)}")

        written = 0
        for dec_path, (_, cls_name, fname) in zip(decoded, to_process):
            shutil.copy(dec_path, out_split_dir / cls_name / fname)
            written += 1

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Compress card crops at H.264 CRF levels")
    parser.add_argument(
        "--crfs", nargs="+", type=int, default=ALL_CRFS,
        help="CRF values (default: 18 23 28 35 43 51)")
    parser.add_argument(
        "--splits", nargs="+", default=SPLITS,
        help="Splits to process (default: train valid)")
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip images that already exist (default: True)")
    args = parser.parse_args()

    print(f"Compress resolution : {IMG_SIZE}×{IMG_SIZE}  (matches training size)")
    print(f"CRFs                : {args.crfs}")
    print(f"Splits              : {args.splits}")
    print(f"Skip existing       : {args.skip_existing}")
    print("-" * 55)

    # remove any previous partial run from old approach
    for crf in args.crfs:
        old_dir = OUT_BASE / f"card_datset_crf{crf}"
        if old_dir.exists():
            # check if images are wrong size (old bug: all same size per class)
            sample = list((old_dir / "train").rglob("*.jpg"))
            if sample:
                from PIL import Image as _PIL
                try:
                    sz = _PIL.open(sample[0]).size
                    # old approach produced ~72×90; new produces 128×128
                    if sz != (IMG_SIZE, IMG_SIZE) and not args.skip_existing:
                        print(f"  Removing old compressed dataset: {old_dir.name}")
                        shutil.rmtree(old_dir)
                except Exception:
                    pass

    total_start = time.time()
    summary = []

    for crf in args.crfs:
        out_dir = OUT_BASE / f"card_datset_crf{crf}"
        print(f"\n── CRF {crf}  →  {out_dir.name} ──────────────────────")

        crf_total = 0
        for split in args.splits:
            items = collect_images(split)
            t0 = time.time()
            print(f"  {split}: {len(items):,} images  ", end="", flush=True)

            n = compress_split(items, out_dir / split, crf, args.skip_existing)

            elapsed = time.time() - t0
            print(f"→ {n:,} written  ({elapsed:.0f}s)")
            crf_total += n

        summary.append({"crf": crf, "total": crf_total,
                         "dir": f"card_datset_crf{crf}"})

    total_elapsed = time.time() - total_start
    print(f"\n{'='*55}")
    print(f"DONE  —  total time: {total_elapsed/60:.1f} min")
    print(f"{'='*55}")
    print(f"{'CRF':<8} {'Images written':>16}  Output dir")
    print("-" * 55)
    for row in summary:
        print(f"  {row['crf']:<6} {row['total']:>16,}  {row['dir']}")
    print(f"\nAll outputs in: {OUT_BASE}")
    print("\nNext step — measure accuracy:")
    print("  python src/eval_crf.py")


if __name__ == "__main__":
    main()
