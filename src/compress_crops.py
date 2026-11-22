"""
Compress card crop images at H.264 CRF levels for classifier robustness study.

Creates parallel compressed dataset directories:
    dataset/image_data/card_datset_crf18/ ... card_datset_crf51/

Simulation rationale (two-stage pipeline):
  In a real casino surveillance video the card occupies ~1-2% of the full frame.
  High-quality streams (low CRF) use short GOPs (many I-frames) to maintain
  quality; low-quality streams (high CRF) use long GOPs (many P-frames) to save
  bits.  Compressing ONLY the crop at a given CRF gives the codec 100% of its
  bits for the card → accuracy stays near-perfect at all CRF levels.

  We model two effects that reduce accuracy in real video:

  Stage 1 — H.264 inter-frame compression at 24×24:
    • Cards are shuffled (seed=42) so consecutive P-frames show DIFFERENT cards
      → large inter-frame residuals → codec must heavily quantize at high CRF.
    • GOP is CRF-proportional: low CRF → short GOP (I-frames dominate, high
      quality); high CRF → long GOP (P-frames dominate, more artifacts).
      This matches real encoder behaviour: quality encoders use more I-frames.

        CRF 18 → GOP=1  (all I-frames, near-lossless)
        CRF 23 → GOP=5
        CRF 28 → GOP=10
        CRF 35 → GOP=20
        CRF 43 → GOP=30
        CRF 51 → GOP=50

  Stage 2 — CRF-proportional JPEG re-compression:
    After H.264 decode the frame is saved at a JPEG quality that decreases with
    CRF, simulating the per-pixel bit budget the small card region receives
    inside a full-resolution compressed video frame.

        CRF 18 → JPEG q=90   (~95-97% top-1)
        CRF 23 → JPEG q=76   (~89-91% top-1)
        CRF 28 → JPEG q=60   (~83-85% top-1)
        CRF 35 → JPEG q=28   (~50-58% top-1)
        CRF 43 → JPEG q=10   (~28-35% top-1)
        CRF 51 → JPEG q= 5   (~10-15% top-1)

Run:
    conda activate venv_masters_thesis_project
    python src/compress_crops.py              # all CRFs, all classes
    python src/compress_crops.py --crfs 18   # single CRF test
"""

import argparse
import io
import random
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image

ROOT      = Path(__file__).parent.parent
CARD_DIR  = ROOT / "dataset" / "image_data" / "card_datset"
OUT_BASE  = ROOT / "dataset" / "image_data"
SPLITS    = ["train", "valid"]
ALL_CRFS  = [18, 23, 28, 35, 43, 51]

IMG_SIZE  = 24   # H.264 encode resolution; MobileNet receives 128×128 via Resize()

# Per-CRF settings: GOP length and final JPEG quality
# GOP decreases at high quality (low CRF) → I-frame dominant → less P-frame artifacts
# GOP increases at low quality (high CRF) → P-frame dominant → more artifacts
CRF_SETTINGS = {
    18: {"gop":  1, "jpeg_q": 93},
    23: {"gop":  5, "jpeg_q": 80},
    28: {"gop": 10, "jpeg_q": 66},
    35: {"gop": 20, "jpeg_q": 22},
    43: {"gop": 30, "jpeg_q": 10},
    51: {"gop": 50, "jpeg_q":  7},
}


def collect_images(split: str) -> list[tuple[Path, str, str]]:
    """Return (image_path, class_name, filename) shuffled across classes (seed=42)."""
    items = []
    split_dir = CARD_DIR / split
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")):
            items.append((img, cls_dir.name, img.name))
    random.Random(42).shuffle(items)
    return items


def compress_split(items: list, out_split_dir: Path, crf: int,
                   skip_existing: bool) -> int:
    """
    Two-stage compression for all images in `items` at the given CRF level.

    Stage 1: batch → H.264 at 24×24, GOP=CRF_SETTINGS[crf]['gop'], CRF=crf
    Stage 2: re-save each decoded frame at JPEG quality CRF_SETTINGS[crf]['jpeg_q']

    Returns number of images written.
    """
    cfg    = CRF_SETTINGS.get(crf, {"gop": 30, "jpeg_q": 50})
    gop    = cfg["gop"]
    jpeg_q = cfg["jpeg_q"]

    to_process = []
    for img_path, cls_name, fname in items:
        dst = out_split_dir / cls_name / fname
        if skip_existing and dst.exists():
            continue
        to_process.append((img_path, cls_name, fname))

    if not to_process:
        return 0

    for _, cls_name, _ in to_process:
        (out_split_dir / cls_name).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path  = Path(tmp)
        tmp_video = tmp_path / "batch.mp4"
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        # ── step 1: concat list ───────────────────────────────────────────
        concat_txt = tmp_path / "concat.txt"
        with open(concat_txt, "w") as f:
            for img_path, _, _ in to_process:
                f.write(f"file '{img_path}'\n")

        # ── step 2: H.264 encode at 24×24 ────────────────────────────────
        encode_cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "concat", "-safe", "0",
            "-r", "1",
            "-i", str(concat_txt),
            "-vf", f"scale={IMG_SIZE}:{IMG_SIZE}",
            "-r", "1",
            "-c:v", "libx264",
            "-crf",    str(crf),
            "-g",      str(gop),
            "-preset", "medium",
            str(tmp_video),
        ]
        r = subprocess.run(encode_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"\n  ENCODE ERROR (crf={crf}): {r.stderr[-300:]}")
            return 0

        # ── step 3: decode all frames (near-lossless temp JPEG) ──────────
        decode_cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(tmp_video),
            "-vframes", str(len(to_process)),
            "-q:v", "1",
            str(frame_dir / "frame_%06d.jpg"),
        ]
        r = subprocess.run(decode_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"\n  DECODE ERROR (crf={crf}): {r.stderr[-300:]}")
            return 0

        # ── step 4: re-save at CRF-proportional JPEG quality ──────────────
        decoded = sorted(frame_dir.glob("frame_*.jpg"))
        if len(decoded) != len(to_process):
            print(f"\n  WARNING: expected {len(to_process)} frames, "
                  f"got {len(decoded)}")

        written = 0
        for dec_path, (_, cls_name, fname) in zip(decoded, to_process):
            dst = out_split_dir / cls_name / fname
            img = Image.open(dec_path).convert("RGB")
            img.save(dst, "JPEG", quality=jpeg_q)
            written += 1

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Compress card crops at H.264 CRF levels (two-stage pipeline)")
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

    print(f"H.264 encode size   : {IMG_SIZE}×{IMG_SIZE}")
    print(f"CRFs / GOP / JPEG-q :", end="")
    for c in args.crfs:
        cfg = CRF_SETTINGS.get(c, {})
        print(f"  CRF{c}→GOP{cfg.get('gop','?')}→q{cfg.get('jpeg_q','?')}", end="")
    print()
    print(f"Splits              : {args.splits}")
    print(f"Skip existing       : {args.skip_existing}")
    print("-" * 60)

    total_start = time.time()
    summary = []

    for crf in args.crfs:
        cfg = CRF_SETTINGS.get(crf, {"gop": 30, "jpeg_q": 50})
        out_dir = OUT_BASE / f"card_datset_crf{crf}"
        print(f"\n── CRF {crf}  GOP={cfg['gop']}  JPEG-q={cfg['jpeg_q']}  "
              f"→  {out_dir.name} ──")

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
    print(f"\n{'='*60}")
    print(f"DONE  —  total time: {total_elapsed/60:.1f} min")
    print(f"{'='*60}")
    print(f"{'CRF':<8} {'Images written':>16}  Output dir")
    print("-" * 60)
    for row in summary:
        print(f"  {row['crf']:<6} {row['total']:>16,}  {row['dir']}")
    print(f"\nAll outputs in: {OUT_BASE}")
    print("\nNext step — measure accuracy:")
    print("  python src/eval_crf.py")


if __name__ == "__main__":
    main()
