"""
Dataset Analysis Report for CATA-Net Thesis Project
Compression-Robust Playing-Card Recognition for AI-Based Baccarat Table Monitoring

Author : B. M. S. Islam
Advisor: Prof. Heejune Ahn  |  SeoulTech

Run:
    conda activate venv_masters_thesis_project
    python data_analysis.py

Output:
    dataset_report/
        01_summary.txt          -- text summary (paste into email/report)
        02_class_counts.png     -- bar chart: images per class per split
        03_augmentation.png     -- bar chart: original vs augmented breakdown
        04_sample_images.png    -- 5×5 grid of sample card crops (one per class)
        05_video_info.png       -- table + frame-type pie charts for each video
"""

import os, sys, re, subprocess, textwrap
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import av

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
CARD_DIR    = DATASET_DIR / "image_data" / "card_datset"
YOLO_DIR    = DATASET_DIR / "image_data" / "yolo_dataset"
VIDEO_DIR   = DATASET_DIR / "live_test_video"
MAIN_VIDEO  = DATASET_DIR / "bacarat_video.mp4"
OUT_DIR     = ROOT / "dataset_report"
OUT_DIR.mkdir(exist_ok=True)

SPLITS      = ["train", "valid"]

# prefix categories used during data collection
PREFIX_ORIG  = re.compile(r"^(?!aug_|topup_|low_|unk_)")
PREFIX_AUG   = re.compile(r"^aug_")
PREFIX_TOPUP = re.compile(r"^topup_")
PREFIX_LOW   = re.compile(r"^low_")
PREFIX_UNK   = re.compile(r"^unk_")

# ── helpers ───────────────────────────────────────────────────────────────────

def list_images(folder: Path):
    return sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))


def classify_prefix(name: str) -> str:
    if PREFIX_AUG.match(name):   return "aug"
    if PREFIX_TOPUP.match(name): return "topup"
    if PREFIX_LOW.match(name):   return "low_quality"
    if PREFIX_UNK.match(name):   return "unknown"
    return "original"


def video_stats(path: Path) -> dict:
    """Return codec/resolution/fps/duration/frame-type counts via PyAV."""
    stats = {"path": path, "name": path.name, "exists": path.exists()}
    if not path.exists():
        return stats
    try:
        container = av.open(str(path))
        vs = container.streams.video[0]
        dur = float(container.duration) / av.time_base if container.duration else 0
        fps_rat = vs.average_rate
        fps = float(fps_rat) if fps_rat else 0.0
        stats.update({
            "codec":      vs.codec_context.name,
            "profile":    getattr(vs.codec_context, "profile", "N/A"),
            "width":      vs.width,
            "height":     vs.height,
            "fps":        round(fps, 2),
            "duration_s": round(dur, 1),
            "nb_frames":  vs.frames,
            "has_bframes": vs.codec_context.has_b_frames,
        })
        container.close()
    except Exception as e:
        stats["error"] = str(e)
        return stats

    # frame type counts via ffprobe (faster than decoding)
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "frame=pict_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=120
        )
        lines  = result.stdout.strip().split("\n")
        counts = defaultdict(int)
        for t in lines:
            counts[t.strip()] += 1
        stats["frame_counts"] = dict(counts)
    except Exception:
        stats["frame_counts"] = {}
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# 1. COLLECT CARD DATASET STATS
# ══════════════════════════════════════════════════════════════════════════════
print("Analysing card dataset …")

# per split → per class → per prefix → count
data = {}
for split in SPLITS:
    split_dir = CARD_DIR / split
    if not split_dir.exists():
        data[split] = {}
        continue
    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    class_data = {}
    for cls in classes:
        imgs = list_images(split_dir / cls)
        prefix_counts = defaultdict(int)
        for img in imgs:
            prefix_counts[classify_prefix(img.name)] += 1
        class_data[cls] = {
            "total": len(imgs),
            "original":    prefix_counts["original"],
            "aug":         prefix_counts["aug"],
            "topup":       prefix_counts["topup"],
            "low_quality": prefix_counts["low_quality"],
            "unknown":     prefix_counts["unknown"],
        }
    data[split] = class_data

# flatten to DataFrames
dfs = {}
for split in SPLITS:
    if data[split]:
        dfs[split] = pd.DataFrame(data[split]).T
        dfs[split].index.name = "class"
        dfs[split] = dfs[split].sort_index()

# YOLO dataset
yolo_counts = {}
for split in ["train", "valid"]:
    img_dir = YOLO_DIR / split / "images"
    lbl_dir = YOLO_DIR / split / "labels"
    n_img = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
    n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
    # count per class-id
    cls_boxes = defaultdict(int)
    if lbl_dir.exists():
        for lf in lbl_dir.glob("*.txt"):
            for line in lf.read_text().splitlines():
                if line.strip():
                    cls_boxes[int(line.split()[0])] += 1
    yolo_counts[split] = {"images": n_img, "labels": n_lbl, "cls_boxes": dict(cls_boxes)}

# ══════════════════════════════════════════════════════════════════════════════
# 2. VIDEO STATS
# ══════════════════════════════════════════════════════════════════════════════
print("Analysing videos …")

video_paths = [MAIN_VIDEO] + sorted(VIDEO_DIR.glob("*.mp4"))
vstats = [video_stats(p) for p in video_paths]


# ══════════════════════════════════════════════════════════════════════════════
# 3. TEXT SUMMARY  →  01_summary.txt
# ══════════════════════════════════════════════════════════════════════════════
print("Writing text summary …")

lines = []
sep = "=" * 72

def h(title):
    lines.append(sep)
    lines.append(f"  {title}")
    lines.append(sep)

lines.append("DATASET ANALYSIS REPORT")
lines.append("CATA-Net Thesis — Compression-Robust Card Recognition")
lines.append("Author: B. M. S. Islam  |  Advisor: Prof. Heejune Ahn  |  SeoulTech")
lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
lines.append("")

# ── Card classification dataset
h("SECTION 1 — CARD CLASSIFICATION DATASET  (card_datset)")
lines.append("")
lines.append("Purpose : Train the per-frame card classifier (MobileNet backbone).")
lines.append("Location: dataset/image_data/card_datset/")
lines.append("")

for split in SPLITS:
    df = dfs.get(split)
    if df is None or df.empty:
        lines.append(f"  [{split}]  — not found")
        continue
    total = int(df["total"].sum())
    n_cls = len(df)
    lines.append(f"  [{split.upper()}]")
    lines.append(f"    Classes  : {n_cls}")
    lines.append(f"    Images   : {total:,}")
    lines.append(f"    Min/class: {int(df['total'].min()):,}  ({df['total'].idxmin()})")
    lines.append(f"    Max/class: {int(df['total'].max()):,}  ({df['total'].idxmax()})")
    lines.append(f"    Mean/class: {df['total'].mean():.1f}")
    lines.append("")
    orig_total = int(df["original"].sum())
    aug_total  = int(df["aug"].sum())
    top_total  = int(df["topup"].sum())
    low_total  = int(df["low_quality"].sum())
    lines.append(f"    Image origin breakdown:")
    lines.append(f"      Original (real crops)  : {orig_total:>6,}  ({100*orig_total/total:.1f}%)")
    lines.append(f"      aug_N_ (augmented)     : {aug_total:>6,}  ({100*aug_total/total:.1f}%)")
    lines.append(f"      topup_N_ (extra-aug)   : {top_total:>6,}  ({100*top_total/total:.1f}%)")
    lines.append(f"      low_ (low-quality)     : {low_total:>6,}  ({100*low_total/total:.1f}%)")
    lines.append("")

lines.append("  NOTE: Classes are named {rank}_{suit}.")
lines.append("        rank  : 2 3 4 5 6 7 8 9 10 a j q k  (13 ranks)")
lines.append("        suit  : c (clubs) d (diamonds) h (hearts) s (spades)  (4 suits)")
lines.append("        Total : 13 × 4 = 52 classes  — card-back class NOT present.")
lines.append("        Image size: variable, approximately 62–86 px wide × 74–106 px tall.")
lines.append("")
lines.append("  IMPORTANT — Data quality concern:")
lines.append("    Most classes have only 3–15 real (non-augmented) crops.")
lines.append("    The vast majority of images are topup_/aug_ copies of those few originals.")
lines.append("    A naive random split is therefore LEAKY — augmented copies of the same")
lines.append("    original appear in both train and valid, inflating reported accuracy.")
lines.append("    For CATA-Net we will rebuild the dataset from raw video tracks to avoid this.")
lines.append("")

lines.append("  PER-CLASS COUNTS — TRAIN (sorted by total)")
lines.append("  " + "-" * 46)
lines.append(f"  {'Class':<8} {'Total':>7} {'Orig':>6} {'Aug':>6} {'Topup':>7} {'Low':>5}")
lines.append("  " + "-" * 46)
if "train" in dfs:
    for cls, row in dfs["train"].sort_values("total").iterrows():
        lines.append(f"  {cls:<8} {int(row.total):>7,} {int(row.original):>6,} "
                     f"{int(row.aug):>6,} {int(row.topup):>7,} {int(row.low_quality):>5,}")
lines.append("")

# ── YOLO detection dataset
h("SECTION 2 — YOLO DETECTION DATASET  (yolo_dataset)")
lines.append("")
lines.append("Purpose : Train / evaluate YOLOv11n object detector (card, hand, chip).")
lines.append("Location: dataset/image_data/yolo_dataset/")
lines.append("Format  : YOLO (class cx cy w h, normalised), 1920×1080 frames.")
lines.append("")
lines.append("  Classes (by bounding-box count in train labels):")
lines.append("    ID 0 — card   (foreground object of interest)")
lines.append("    ID 1 — hand   (dealer / player hand)")
lines.append("    ID 2 — chip   (betting chips, most frequent)")
lines.append("")
for split in ["train", "valid"]:
    yc = yolo_counts[split]
    lines.append(f"  [{split.upper()}]")
    lines.append(f"    Images : {yc['images']:,}")
    lines.append(f"    Labels : {yc['labels']:,}")
    for cid, cnt in sorted(yc["cls_boxes"].items()):
        name = {0:"card", 1:"hand", 2:"chip"}.get(cid, f"class_{cid}")
        lines.append(f"    Boxes  class {cid} ({name}): {cnt:,}")
    lines.append("")

# ── Videos
h("SECTION 3 — VIDEO FILES")
lines.append("")
lines.append("  All videos: H.264, 1920×1080.")
lines.append("  'has_bframes' — whether B-frames appear in the stream.")
lines.append("  B-frames matter for CATA-Net: they create bidirectional temporal redundancy.")
lines.append("")
lines.append(f"  {'File':<28} {'Dur(s)':>7} {'FPS':>6} {'Frames':>7} "
             f"{'B-fr':>5}  {'I':>5} {'P':>6} {'B':>6}")
lines.append("  " + "-" * 78)
for vs in vstats:
    if not vs.get("exists") or "error" in vs:
        lines.append(f"  {vs['name']:<28}  NOT FOUND or ERROR")
        continue
    fc = vs.get("frame_counts", {})
    fi = fc.get("I", 0)
    fp = fc.get("P", 0)
    fb = fc.get("B", 0)
    bflag = "YES" if vs.get("has_bframes") else "no"
    loc = "dataset/" if vs["path"].parent == DATASET_DIR else "live_test_video/"
    name_str = f"{loc}{vs['name']}"
    lines.append(f"  {name_str:<36} {vs['duration_s']:>7.1f} {vs['fps']:>6.2f} "
                 f"{vs['nb_frames']:>7,} {bflag:>5}  {fi:>5,} {fp:>6,} {fb:>6,}")

lines.append("")
lines.append("  KEY OBSERVATIONS:")
lines.append("  1. bacarat_video.mp4 (main dataset video): NO B-frames (I+P only).")
lines.append("     GOP ~30 frames.  Duration: ~4 min.")
lines.append("  2. live_test_video/bacarat.mp4: HAS B-frames (I+P+B structure).")
lines.append("     This is the most relevant video for CATA-Net experiments.")
lines.append("     Duration: ~6.7 min.")
lines.append("  3. live_test_video/chip.mp4: very few I-frames (40 in 330 s = ~1 per 8 s).")
lines.append("     Extreme long-GOP — card crops here would be heavily P-frame dominated.")
lines.append("  4. Total baccarat gameplay video available: ~11 minutes combined.")
lines.append("     Thesis plan target is ~5–6 hours for a full track dataset.")
lines.append("     Additional footage from GKL/ITICO is required.")
lines.append("")

# ── Summary gaps
h("SECTION 4 — GAPS vs THESIS PLAN")
lines.append("")
gaps = [
    ("Card back class (53rd class)", "MISSING — only 52 classes present in image data"),
    ("Test split",                   "MISSING — only train + valid splits exist"),
    ("gkl_mask1.avi",               "NOT FOUND — closest is bacarat_video.mp4 (4 min)"),
    ("yolo11n_cards.pt weights",    "NOT FOUND in source_code/"),
    ("Multiple sessions for split", "Only 1 main video — session-level split not yet possible"),
    ("~5-6 h of gameplay video",    "Only ~11 min available; GKL/ITICO data request needed"),
    ("frame-type metadata in crops","MISSING — existing crops cannot be used for CATA-Net training"),
]
for item, status in gaps:
    lines.append(f"  {'[ ' + ('OK' if 'NOT' not in status and 'MISSING' not in status else '!!') + ' ]'} "
                 f"{item}")
    lines.append(f"         → {status}")
    lines.append("")

lines.append(sep)
lines.append("END OF REPORT")
lines.append(sep)

summary_text = "\n".join(lines)
(OUT_DIR / "01_summary.txt").write_text(summary_text)
print(summary_text)


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIGURE 1 — Class counts bar chart  →  02_class_counts.png
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlotting class counts …")

fig, axes = plt.subplots(2, 1, figsize=(18, 12))
fig.suptitle("Card Classification Dataset — Images per Class",
             fontsize=14, fontweight="bold", y=0.98)

colors = {"train": "#2196F3", "valid": "#4CAF50"}

for ax, split in zip(axes, SPLITS):
    df = dfs.get(split)
    if df is None or df.empty:
        ax.text(0.5, 0.5, f"{split} split not found", ha="center", va="center")
        continue
    classes = df.index.tolist()
    totals  = df["total"].values
    x = np.arange(len(classes))

    bars = ax.bar(x, totals, color=colors[split], alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of images", fontsize=10)
    ax.set_title(f"{split.upper()} split  —  {len(classes)} classes, "
                 f"{int(totals.sum()):,} total images", fontsize=11)
    ax.axhline(totals.mean(), color="red", linestyle="--", linewidth=1,
               label=f"Mean = {totals.mean():.0f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, totals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(int(val)), ha="center", va="bottom", fontsize=6)

plt.tight_layout()
fig.savefig(OUT_DIR / "02_class_counts.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 02_class_counts.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 2 — Augmentation breakdown  →  03_augmentation.png
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting augmentation breakdown …")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Train Split — Original vs Augmented Images per Class",
             fontsize=13, fontweight="bold")

cats   = ["original", "aug", "topup", "low_quality"]
clrs   = ["#1565C0", "#42A5F5", "#90CAF9", "#FF8F00"]
labels = ["Original crop", "aug_N_", "topup_N_", "low_quality"]

df_tr = dfs.get("train")
if df_tr is not None:
    df_sorted = df_tr.sort_values("original")
    classes   = df_sorted.index.tolist()
    x = np.arange(len(classes))
    bottom = np.zeros(len(classes))
    ax = axes[0]
    for cat, clr, lbl in zip(cats, clrs, labels):
        vals = df_sorted[cat].values.astype(float)
        ax.bar(x, vals, bottom=bottom, color=clr, label=lbl, alpha=0.9)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Images")
    ax.set_title("Stacked: original vs augmented (sorted by original count)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Pie chart of totals
    ax2 = axes[1]
    totals_by_cat = [int(df_tr[c].sum()) for c in cats]
    wedge_colors  = clrs
    explode = [0.05, 0, 0, 0]
    wedges, texts, autotexts = ax2.pie(
        totals_by_cat, labels=labels, colors=wedge_colors,
        autopct="%1.1f%%", startangle=140, explode=explode,
        textprops={"fontsize": 10}
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax2.set_title(f"Train split composition\n"
                  f"(Total: {sum(totals_by_cat):,} images)", fontsize=11)

plt.tight_layout()
fig.savefig(OUT_DIR / "03_augmentation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 03_augmentation.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 3 — Sample images grid  →  04_sample_images.png
# ══════════════════════════════════════════════════════════════════════════════
print("Building sample image grid …")

RANK_ORDER = ["a", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k"]
SUIT_ORDER = ["c", "d", "h", "s"]
SUIT_NAMES = {"c": "Clubs", "d": "Diamonds", "h": "Hearts", "s": "Spades"}

fig, axes = plt.subplots(4, 13, figsize=(22, 8))
fig.suptitle("Sample Card Crops — One per Class (from train split)\n"
             "Suits (rows): Clubs, Diamonds, Hearts, Spades  |  "
             "Ranks (columns): A 2 3 4 5 6 7 8 9 10 J Q K",
             fontsize=11, fontweight="bold")

TARGET = 64  # display size

for si, suit in enumerate(SUIT_ORDER):
    for ri, rank in enumerate(RANK_ORDER):
        cls  = f"{rank}_{suit}"
        ax   = axes[si][ri]
        ax.axis("off")
        cls_dir = CARD_DIR / "train" / cls
        if not cls_dir.exists():
            ax.set_title(cls, fontsize=6, color="red")
            continue
        # prefer an original crop; fall back to any image
        imgs = sorted(cls_dir.glob("*.jpg"))
        orig = [p for p in imgs if classify_prefix(p.name) == "original"]
        pick = orig[0] if orig else (imgs[0] if imgs else None)
        if pick is None:
            ax.set_title(cls, fontsize=6, color="gray")
            continue
        img = Image.open(pick).convert("RGB")
        img_resized = img.resize((TARGET, TARGET), Image.LANCZOS)
        ax.imshow(np.array(img_resized))
        # label: rank on top, suit colour below
        suit_color = {"c": "#1a1a1a", "d": "#c0392b", "h": "#c0392b", "s": "#1a1a1a"}[suit]
        ax.set_title(cls.upper(), fontsize=7, color=suit_color, fontweight="bold", pad=2)

        # draw border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#cccccc")

plt.subplots_adjust(wspace=0.1, hspace=0.3)
fig.savefig(OUT_DIR / "04_sample_images.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 04_sample_images.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURE 4 — Video info table + frame-type pies  →  05_video_info.png
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting video info …")

valid_vs = [v for v in vstats if v.get("exists") and "error" not in v]
n_videos = len(valid_vs)

fig = plt.figure(figsize=(20, 4 + 3.5 * n_videos))
fig.suptitle("Video File Analysis", fontsize=14, fontweight="bold", y=0.99)

outer = gridspec.GridSpec(n_videos + 1, 1, hspace=0.6)

# Top: summary table
ax_tbl = fig.add_subplot(outer[0])
ax_tbl.axis("off")
col_labels = ["File", "Duration", "FPS", "Total Frames", "Has B-frames",
              "I-frames", "P-frames", "B-frames", "GOP (avg)"]
rows_tbl = []
for vs in valid_vs:
    fc   = vs.get("frame_counts", {})
    fi   = fc.get("I", 0)
    fp   = fc.get("P", 0)
    fb   = fc.get("B", 0)
    gop  = round((fp + fb) / fi, 1) if fi > 0 else "N/A"
    loc  = "dataset/" if vs["path"].parent == DATASET_DIR else "live_test/"
    rows_tbl.append([
        f"{loc}{vs['name']}",
        f"{vs['duration_s']} s  ({vs['duration_s']/60:.1f} min)",
        f"{vs['fps']}",
        f"{vs['nb_frames']:,}",
        "YES ✓" if vs.get("has_bframes") else "no",
        f"{fi:,}", f"{fp:,}", f"{fb:,}", str(gop),
    ])
tbl = ax_tbl.table(
    cellText=rows_tbl, colLabels=col_labels,
    cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#E3F2FD")
    cell.set_edgecolor("#cccccc")

# Per-video: frame-type pie + info text
pie_colors = {"I": "#1565C0", "P": "#42A5F5", "B": "#FF8F00"}

for idx, vs in enumerate(valid_vs):
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[idx + 1],
                                             width_ratios=[1, 2], wspace=0.3)
    ax_pie  = fig.add_subplot(inner[0])
    ax_info = fig.add_subplot(inner[1])

    fc = vs.get("frame_counts", {})
    fi = fc.get("I", 0)
    fp = fc.get("P", 0)
    fb = fc.get("B", 0)
    total_f = fi + fp + fb

    sizes  = [v for v in [fi, fp, fb] if v > 0]
    lbls   = [k for k, v in [("I", fi), ("P", fp), ("B", fb)] if v > 0]
    clrs   = [pie_colors[k] for k in lbls]

    if sizes:
        ax_pie.pie(sizes, labels=lbls, colors=clrs, autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize": 9})
    ax_pie.set_title(vs["name"], fontsize=9, fontweight="bold")

    gop_avg = round((fp + fb) / fi, 1) if fi > 0 else "N/A"
    info_lines = [
        f"Codec     : {vs.get('codec','?').upper()}  ({vs.get('profile','?')})",
        f"Resolution: {vs.get('width','?')} × {vs.get('height','?')} px",
        f"Duration  : {vs['duration_s']} s  ({vs['duration_s']/60:.1f} min)",
        f"FPS       : {vs['fps']}",
        f"Total frames: {vs['nb_frames']:,}",
        f"B-frames  : {'YES — I+P+B structure' if vs.get('has_bframes') else 'No — I+P only'}",
        f"I-frames  : {fi:,}  ({100*fi/total_f:.1f}%)" if total_f else "I-frames: N/A",
        f"P-frames  : {fp:,}  ({100*fp/total_f:.1f}%)" if total_f else "",
        f"B-frames  : {fb:,}  ({100*fb/total_f:.1f}%)" if (total_f and fb) else "",
        f"Avg GOP length: {gop_avg} frames",
    ]
    ax_info.axis("off")
    ax_info.text(0.02, 0.95, "\n".join([l for l in info_lines if l]),
                 transform=ax_info.transAxes,
                 fontsize=9, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", alpha=0.8))

fig.savefig(OUT_DIR / "05_video_info.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved 05_video_info.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. DONE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"All outputs saved to:  {OUT_DIR}/")
print(f"{'='*60}")
print("  01_summary.txt       — full text report")
print("  02_class_counts.png  — images per class (train + valid)")
print("  03_augmentation.png  — original vs augmented breakdown")
print("  04_sample_images.png — 52-class sample image grid")
print("  05_video_info.png    — video stats + frame-type charts")
print(f"{'='*60}\n")
