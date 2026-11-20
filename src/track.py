"""
YOLO + ByteTrack + MobileNet inference pipeline on video.

Detects cards/hands/chips with YOLOv11n, tracks them with ByteTrack,
and classifies every card crop with the trained MobileNetV1-1.25.

Run:
    conda activate venv_masters_thesis_project
    python src/track.py --video dataset/bacarat_video.mp4

Output:
    data/detections.csv   ← one row per detection per frame
    data/tracks.csv       ← one row per unique track (track-level summary)
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).parent.parent

import sys
sys.path.insert(0, str(ROOT))
from src.models.mobilenet import MobileNetV1

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("ultralytics not installed. Run: pip install ultralytics")

# ── constants ─────────────────────────────────────────────────────────────────
YOLO_WEIGHTS = ROOT / "models" / "yolo11n_cards.pt"
CLS_WEIGHTS  = ROOT / "models" / "mobilenet_classifier.pt"
CLASS_MAP    = ROOT / "data"   / "classifier_class_map.json"

YOLO_CLASSES = {0: "card", 1: "hand", 2: "chip"}
CARD_CLS_ID  = 0

IMG_SIZE     = 128
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESH  = 0.25
IOU_THRESH   = 0.45

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def load_classifier(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MobileNetV1(num_classes=ckpt["num_classes"], alpha=ckpt.get("alpha", 1.25))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    return model, idx_to_class


def build_crop_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


@torch.no_grad()
def classify_crops(model, crops_bgr: list, transform, device):
    """Classify a list of BGR numpy crops; return (class_name, confidence) per crop."""
    if not crops_bgr:
        return []
    tensors = []
    for crop in crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(transform(rgb))
    batch  = torch.stack(tensors).to(device)
    logits = model(batch)
    probs  = F.softmax(logits, dim=1)
    confs, idxs = probs.max(dim=1)
    return list(zip(idxs.cpu().tolist(), confs.cpu().tolist()))


def run(video_path: Path, yolo_weights: Path, cls_weights: Path,
        out_det: Path, out_tracks: Path, show: bool = False):

    print(f"Video          : {video_path}")
    print(f"YOLO weights   : {yolo_weights}")
    print(f"Classifier     : {cls_weights}")
    print(f"Device         : {DEVICE}")
    print("-" * 55)

    # ── load models ───────────────────────────────────────────────────────────
    yolo_model = YOLO(str(yolo_weights))
    cls_model, idx_to_class = load_classifier(cls_weights, DEVICE)
    crop_tf = build_crop_transform()

    # ── open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # ── tracking + classification ──────────────────────────────────────────────
    det_rows   = []   # one row per detection per frame
    track_seen = {}   # track_id → {label, start_frame, end_frame, frames, confs}

    tracker_results = yolo_model.track(
        source      = str(video_path),
        tracker     = "bytetrack.yaml",
        conf        = CONF_THRESH,
        iou         = IOU_THRESH,
        stream      = True,
        persist     = True,
        verbose     = False,
    )

    frame_idx = 0
    for result in tracker_results:
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx:>5}/{total_frames}  "
                  f"tracks active: {len(track_seen)}")

        if result.boxes is None or result.boxes.id is None:
            continue

        boxes   = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().int().numpy()
        confs   = result.boxes.conf.cpu().numpy()
        track_ids = result.boxes.id.cpu().int().numpy()
        orig_img  = result.orig_img   # BGR numpy HWC

        # collect card crops for batch classification
        card_indices = []
        card_crops   = []
        for i, (box, cls_id, conf, tid) in enumerate(
                zip(boxes, cls_ids, confs, track_ids)):
            if int(cls_id) == CARD_CLS_ID:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_img.shape[1], x2), min(orig_img.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    card_crops.append(orig_img[y1:y2, x1:x2])
                    card_indices.append(i)

        # batch classify all card crops in this frame
        cls_results = classify_crops(cls_model, card_crops, crop_tf, DEVICE)
        card_cls_map = {card_indices[k]: cls_results[k]
                        for k in range(len(card_indices))}

        # write detections row + update track summaries
        for i, (box, cls_id, det_conf, tid) in enumerate(
                zip(boxes, cls_ids, confs, track_ids)):
            x1, y1, x2, y2 = box
            tid = int(tid)
            cls_name = YOLO_CLASSES.get(int(cls_id), "unknown")

            card_label = ""
            card_conf  = 0.0
            if i in card_cls_map:
                card_idx, card_conf = card_cls_map[i]
                card_label = idx_to_class.get(card_idx, "unknown")

            det_rows.append({
                "frame":      frame_idx,
                "track_id":   tid,
                "yolo_class": cls_name,
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1),
                "yolo_conf":  round(float(det_conf), 3),
                "card_label": card_label,
                "card_conf":  round(float(card_conf), 3),
            })

            # update track-level summary (card tracks only)
            if int(cls_id) == CARD_CLS_ID and card_label:
                if tid not in track_seen:
                    track_seen[tid] = {
                        "start_frame": frame_idx,
                        "end_frame":   frame_idx,
                        "labels": [],
                        "confs":  [],
                    }
                track_seen[tid]["end_frame"] = frame_idx
                track_seen[tid]["labels"].append(card_label)
                track_seen[tid]["confs"].append(card_conf)

    # ── save detections.csv ───────────────────────────────────────────────────
    out_det.parent.mkdir(exist_ok=True)
    det_fields = ["frame","track_id","yolo_class","x1","y1","x2","y2",
                  "yolo_conf","card_label","card_conf"]
    with open(out_det, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=det_fields)
        w.writeheader()
        w.writerows(det_rows)
    print(f"\nDetections saved → {out_det}  ({len(det_rows):,} rows)")

    # ── build & save tracks.csv ───────────────────────────────────────────────
    # majority-vote label per track, weighted by confidence
    from collections import Counter
    track_rows = []
    for tid, info in sorted(track_seen.items()):
        # weighted majority vote: pick label with highest sum of confidences
        label_conf = {}
        for lbl, cf in zip(info["labels"], info["confs"]):
            label_conf[lbl] = label_conf.get(lbl, 0.0) + cf
        best_label = max(label_conf, key=label_conf.get)
        avg_conf   = sum(info["confs"]) / len(info["confs"]) if info["confs"] else 0
        rank       = best_label.split("_")[0] if "_" in best_label else ""
        suit       = best_label.split("_")[1] if "_" in best_label else ""
        track_rows.append({
            "track_id":    tid,
            "label":       best_label,
            "rank":        rank,
            "suit":        suit,
            "start_frame": info["start_frame"],
            "end_frame":   info["end_frame"],
            "n_frames":    len(info["labels"]),
            "avg_conf":    round(avg_conf, 3),
        })

    with open(out_tracks, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(track_rows[0].keys()) if track_rows else [])
        w.writeheader()
        w.writerows(track_rows)
    print(f"Tracks    saved → {out_tracks}  ({len(track_rows)} card tracks)")
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv11n + ByteTrack + MobileNet card pipeline")
    parser.add_argument("--video",  default="dataset/bacarat_video.mp4")
    parser.add_argument("--yolo",   default=str(YOLO_WEIGHTS))
    parser.add_argument("--cls",    default=str(CLS_WEIGHTS))
    parser.add_argument("--out_det",    default="data/detections.csv")
    parser.add_argument("--out_tracks", default="data/tracks.csv")
    args = parser.parse_args()

    video_path = ROOT / args.video
    yolo_w     = Path(args.yolo)
    cls_w      = Path(args.cls)
    out_det    = ROOT / args.out_det
    out_tracks = ROOT / args.out_tracks

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not yolo_w.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {yolo_w}\n"
            "Run: python src/train_detector.py")
    if not cls_w.exists():
        raise FileNotFoundError(
            f"Classifier weights not found: {cls_w}\n"
            "Run: python src/train_classifier.py")

    run(video_path, yolo_w, cls_w, out_det, out_tracks)


if __name__ == "__main__":
    main()
