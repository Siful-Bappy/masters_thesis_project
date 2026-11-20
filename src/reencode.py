"""
Re-encode reference video at 7 CRF levels × 2 GOP settings = 14 output files.

Purpose: Create a multi-CRF dataset to study how H.264 compression
         affects card recognition accuracy (core of CATA-Net thesis).

CRF values : 0 (near-lossless), 18, 23, 28, 35, 43, 51 (heavy compression)
GOP settings:
  g60  — GOP=60 frames (~2 s at 30 fps, camera-like)
  g250 — GOP=250 frames (long-GOP, like original GKL recording)

Both settings use:
  -bf 2             : up to 2 B-frames (I/P/B structure for CATA-Net)
  -sc_threshold 0   : disable scene-change I-frames (pure fixed GOP)
  -keyint_min = g   : force exact GOP size

Run:
    conda activate venv_masters_thesis_project
    python src/reencode.py

    # custom reference video:
    python src/reencode.py --ref dataset/bacarat_video.mp4

Output layout:
    videos/
        ref/
            bacarat_video_crf0_g60.mp4
            bacarat_video_crf18_g60.mp4
            ...  (7 × 2 = 14 files)
    data/video_inventory.csv   ← codec info for every file
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

CRFS     = [0, 18, 23, 28, 35, 43, 51]
GOPS     = [60, 250]          # g60 ≈ 2 s at 30 fps; g250 = long GOP
PRESET   = "medium"
BFRAMES  = 2


def ffprobe_info(path: Path) -> dict:
    """Return basic stream info via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,profile,width,height,avg_frame_rate,"
        "bit_rate,has_b_frames,nb_frames",
        "-show_entries", "format=duration,size",
        "-of", "default=nw=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = {"file": path.name, "path": str(path)}
    for line in result.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    return info


def count_frame_types(path: Path) -> dict:
    """Count I/P/B frames via ffprobe (fast — no decoding)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=pict_type",
        "-of", "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    counts = {"I": 0, "P": 0, "B": 0}
    for t in result.stdout.splitlines():
        t = t.strip()
        if t in counts:
            counts[t] += 1
    return counts


def reencode(ref: Path, out: Path, crf: int, gop: int) -> bool:
    """
    Re-encode ref → out with fixed CRF and GOP.
    Returns True on success.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(ref),
        "-an",                          # drop audio
        "-c:v", "libx264",
        "-preset", PRESET,
        "-crf", str(crf),
        "-g", str(gop),
        "-keyint_min", str(gop),        # force exact GOP size
        "-sc_threshold", "0",           # no extra I-frames on scene change
        "-bf", str(BFRAMES),            # allow B-frames
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Multi-CRF re-encoder")
    parser.add_argument(
        "--ref",
        default=str(ROOT / "dataset" / "bacarat_video.mp4"),
        help="Reference (best-quality) video path",
    )
    parser.add_argument(
        "--crfs", nargs="+", type=int, default=CRFS,
        help="CRF values to encode (default: 0 18 23 28 35 43 51)",
    )
    parser.add_argument(
        "--gops", nargs="+", type=int, default=GOPS,
        help="GOP sizes to encode (default: 60 250)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip files that already exist (default: True)",
    )
    args = parser.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.exists():
        sys.exit(f"Reference video not found: {ref_path}")

    out_dir = ROOT / "videos" / "ref"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = ref_path.stem   # e.g. "bacarat_video"

    print(f"Reference  : {ref_path}")
    print(f"Output dir : {out_dir}")
    print(f"CRFs       : {args.crfs}")
    print(f"GOPs       : {args.gops}")
    print(f"Preset     : {PRESET}  |  B-frames: {BFRAMES}")
    print("-" * 55)

    # ── probe reference video ──────────────────────────────────────────────
    print("Probing reference …")
    ref_info   = ffprobe_info(ref_path)
    ref_ftypes = count_frame_types(ref_path)
    ref_info.update({f"n_{k}": v for k, v in ref_ftypes.items()})
    ref_info["crf"]  = "ref"
    ref_info["gop"]  = "original"
    ref_info["role"] = "reference"

    inventory = [ref_info]

    # ── re-encode ──────────────────────────────────────────────────────────
    total = len(args.crfs) * len(args.gops)
    done  = 0

    for gop in args.gops:
        for crf in args.crfs:
            done += 1
            out_name = f"{stem}_crf{crf}_g{gop}.mp4"
            out_path = out_dir / out_name

            prefix = f"[{done:>2}/{total}]  crf={crf:>2}  gop={gop:>3}"

            if args.skip_existing and out_path.exists():
                size_mb = out_path.stat().st_size / 1e6
                print(f"{prefix}  → SKIP (exists, {size_mb:.1f} MB)")
            else:
                print(f"{prefix}  → encoding …", end=" ", flush=True)
                t0 = time.time()
                ok = reencode(ref_path, out_path, crf, gop)
                elapsed = time.time() - t0
                if ok:
                    size_mb = out_path.stat().st_size / 1e6
                    print(f"done  {size_mb:.1f} MB  {elapsed:.1f}s")
                else:
                    print("FAILED — check ffmpeg output above")
                    continue

            # probe the output file
            info   = ffprobe_info(out_path)
            ftypes = count_frame_types(out_path)
            info.update({f"n_{k}": v for k, v in ftypes.items()})
            info["crf"]  = crf
            info["gop"]  = gop
            info["role"] = "reencoded"
            inventory.append(info)

    # ── save video_inventory.csv ───────────────────────────────────────────
    inv_path = ROOT / "data" / "video_inventory.csv"
    all_keys = sorted({k for row in inventory for k in row})
    with open(inv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for row in inventory:
            w.writerow({k: row.get(k, "") for k in all_keys})

    print(f"\nInventory saved → {inv_path}  ({len(inventory)} entries)")

    # ── summary table ──────────────────────────────────────────────────────
    print("\n── Encoded files ──────────────────────────────────────────────")
    print(f"{'File':<45} {'CRF':>4} {'GOP':>4} {'Size MB':>8}  "
          f"{'I':>5} {'P':>6} {'B':>6}")
    print("-" * 78)
    for row in inventory:
        name    = row.get("file", "")[:44]
        crf     = row.get("crf", "")
        gop     = row.get("gop", "")
        size    = int(row.get("size", 0))
        fi      = row.get("n_I", 0)
        fp      = row.get("n_P", 0)
        fb      = row.get("n_B", 0)
        print(f"{name:<45} {str(crf):>4} {str(gop):>4} "
              f"{size/1e6:>8.1f}  {str(fi):>5} {str(fp):>6} {str(fb):>6}")

    print(f"\nDone. {total} files in {out_dir}")


if __name__ == "__main__":
    main()
