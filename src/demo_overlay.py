"""
Demo overlay: runs YOLOv11n + ByteTrack + MobileNet on a video and writes
an annotated output video showing:

  • Bounding boxes — colour-coded by class (card=blue, hand=green, chip=yellow)
  • Track ID for every tracked object
  • Card classification label + confidence (top-5 per crop)
  • Frame type (I / P / B) in the top-left corner
  • Frame index and FPS counter

Run:
    conda activate venv_masters_thesis_project
    python src/demo_overlay.py --video dataset/bacarat_video.mp4

Output:
    dataset_report/demo_<video_name>.mp4
"""

import argparse
import time
from collections import deque
from pathlib import Path

import av
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
    raise SystemExit("ultralytics not installed.")

# ── visual style ──────────────────────────────────────────────────────────────
BOX_COLORS = {
    "card": (220,  80,  20),   # blue-ish  (BGR)
    "hand": ( 40, 180,  40),   # green
    "chip": ( 20, 200, 220),   # yellow
}
TEXT_COLOR   = (255, 255, 255)
FTYPE_COLORS = {"I": (0, 200, 0), "P": (0, 165, 255), "B": (0, 0, 255)}
FONT         = cv2.FONT_HERSHEY_SIMPLEX

YOLO_CLASSES = {0: "card", 1: "hand", 2: "chip"}
CARD_CLS_ID  = 0
IMG_SIZE     = 128
CONF_THRESH  = 0.25
IOU_THRESH   = 0.45
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN         = [0.485, 0.456, 0.406]
STD          = [0.229, 0.224, 0.225]


def load_classifier(ckpt_path: Path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MobileNetV1(num_classes=ckpt["num_classes"],
                         alpha=ckpt.get("alpha", 1.25))
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
def classify_crop(model, crop_bgr, transform, device):
    """Return (top1_label_idx, top1_conf, top5_logits)."""
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = F.softmax(logits, dim=1)[0]
    top1_conf, top1_idx = probs.max(0)
    return int(top1_idx), float(top1_conf)


def get_frame_types(video_path: Path) -> dict:
    """Pre-read all frame types from the video using ffprobe via PyAV."""
    print("Reading frame types …", end=" ", flush=True)
    ftypes = {}
    container = av.open(str(video_path))
    stream    = container.streams.video[0]
    stream.codec_context.skip_frame = "NONKEY"  # only keyframes for speed
    # We need all frames, so reset
    stream.codec_context.skip_frame = "DEFAULT"
    # PyAV returns pict_type as an enum or int depending on version
    _PICT_INT = {1: "I", 2: "P", 3: "B"}
    gop_idx, gop_pos = -1, 0
    for i, frame in enumerate(container.decode(stream)):
        pt = frame.pict_type
        t  = pt.name if hasattr(pt, "name") else _PICT_INT.get(int(pt), "?")
        if t == "I":
            gop_idx += 1
            gop_pos  = 0
        else:
            gop_pos += 1
        ftypes[i] = {"type": t, "gop_idx": gop_idx, "gop_pos": gop_pos}
    container.close()
    print(f"done ({len(ftypes)} frames)")
    return ftypes


def draw_box(img, x1, y1, x2, y2, color, label_text, thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label_text, FONT, 0.45, 1)
    # background rect for text
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label_text, (x1 + 2, y1 - 3),
                FONT, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_hud(img, frame_idx: int, ftype_info: dict, fps: float):
    h, w = img.shape[:2]
    ftype = ftype_info.get("type", "?")
    gop_i = ftype_info.get("gop_idx", -1)
    gop_p = ftype_info.get("gop_pos", -1)

    color = FTYPE_COLORS.get(ftype, (200, 200, 200))

    # frame type badge (top-left)
    cv2.rectangle(img, (8, 8), (160, 70), (30, 30, 30), -1)
    cv2.putText(img, f"Frame: {frame_idx}", (12, 26),
                FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"Type : ", (12, 46),
                FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, ftype, (70, 46),
                FONT, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(img, f"GOP  : {gop_i}  pos={gop_p}", (12, 64),
                FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # FPS (top-right)
    fps_txt = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_txt, FONT, 0.55, 1)
    cv2.putText(img, fps_txt, (w - tw - 10, 26),
                FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def run(video_path: Path, yolo_weights: Path, cls_weights: Path,
        out_path: Path, max_frames: int = 0):

    print(f"Video      : {video_path}")
    print(f"YOLO       : {yolo_weights}")
    print(f"Classifier : {cls_weights}")
    print(f"Output     : {out_path}")
    print(f"Device     : {DEVICE}")
    print("-" * 55)

    # pre-load frame types
    frame_types = get_frame_types(video_path)

    # load models
    yolo_model = YOLO(str(yolo_weights))
    cls_model, idx_to_class = load_classifier(cls_weights, DEVICE)
    crop_tf = build_crop_transform()

    # open video for dimensions
    cap = cv2.VideoCapture(str(video_path))
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    SRC_FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    out_path.parent.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, SRC_FPS, (W, H))

    fps_buf = deque(maxlen=30)
    frame_idx = 0
    t_prev    = time.perf_counter()

    tracker_results = yolo_model.track(
        source   = str(video_path),
        tracker  = "bytetrack.yaml",
        conf     = CONF_THRESH,
        iou      = IOU_THRESH,
        stream   = True,
        persist  = True,
        verbose  = False,
    )

    for result in tracker_results:
        if max_frames and frame_idx >= max_frames:
            break

        frame = result.orig_img.copy()   # BGR HWC numpy
        ftype_info = frame_types.get(frame_idx, {"type": "?", "gop_idx": -1, "gop_pos": -1})

        if result.boxes is not None and result.boxes.id is not None:
            boxes     = result.boxes.xyxy.cpu().numpy()
            cls_ids   = result.boxes.cls.cpu().int().numpy()
            det_confs = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.cpu().int().numpy()

            for box, cls_id, det_conf, tid in zip(
                    boxes, cls_ids, det_confs, track_ids):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                cls_name = YOLO_CLASSES.get(int(cls_id), "?")
                color    = BOX_COLORS.get(cls_name, (200, 200, 200))

                if int(cls_id) == CARD_CLS_ID and x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    card_idx, card_conf = classify_crop(
                        cls_model, crop, crop_tf, DEVICE)
                    card_lbl = idx_to_class.get(card_idx, "?")
                    label_txt = (f"[{int(tid)}] {card_lbl.upper()} "
                                 f"{card_conf*100:.0f}%")
                else:
                    label_txt = f"[{int(tid)}] {cls_name}  {det_conf:.2f}"

                draw_box(frame, x1, y1, x2, y2, color, label_txt)

        # FPS calculation
        t_now = time.perf_counter()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        display_fps = sum(fps_buf) / len(fps_buf)

        draw_hud(frame, frame_idx, ftype_info, display_fps)
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  {frame_idx}/{total}  FPS={display_fps:.1f}")

    writer.release()
    print(f"\nOutput saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Demo overlay video")
    parser.add_argument("--video",   default="dataset/bacarat_video.mp4")
    parser.add_argument("--yolo",    default=str(ROOT / "models" / "yolo11n_cards.pt"))
    parser.add_argument("--cls",     default=str(ROOT / "models" / "mobilenet_classifier.pt"))
    parser.add_argument("--out",     default="")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Process only first N frames (0 = all)")
    args = parser.parse_args()

    video_path = ROOT / args.video
    yolo_w     = Path(args.yolo)
    cls_w      = Path(args.cls)
    out_path   = Path(args.out) if args.out else \
                 ROOT / "dataset_report" / f"demo_{video_path.stem}.mp4"

    for p, label in [(video_path, "Video"), (yolo_w, "YOLO weights"),
                     (cls_w, "Classifier weights")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    run(video_path, yolo_w, cls_w, out_path, args.max_frames)


if __name__ == "__main__":
    main()
