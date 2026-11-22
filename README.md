# CATA-Net: Compression-Robust Card Recognition for AI-Based Baccarat Table Monitoring

**Author:** B. M. S. Islam  
**Advisor:** Prof. Heejune Ahn  
**Institution:** Seoul National University of Science and Technology (SeoulTech)  
**Thesis:** Compression-Robust and Edge-Deployable Playing-Card Recognition for AI-Based Baccarat Table Monitoring

---

## Overview

The commercial baccarat monitoring system classifies cards **one frame at a time**. When H.264 video compression is applied (as in real casino CCTV systems), accuracy drops sharply as CRF increases — measured baseline from this work:

| CRF | Top-1 Accuracy |
|-----|---------------|
| 0 (uncompressed) | 99.89% |
| 18 | 96.08% |
| 23 | 90.22% |
| 28 | 83.30% |
| 35 | 55.22% |
| 43 | 28.73% |
| 51 | 8.76% |

**CATA-Net (Codec-Aware Temporal Aggregation Network)** solves this by reading the codec structure (I/P/B frame types, GOP position, GOP index) directly from the video bitstream and using them as learnable embeddings in a Transformer aggregator — learning to weight truly new information (I-frames) over redundant predictions (P/B-frames within the same GOP).

---

## Thesis Contributions

| # | Contribution | Status |
|---|---|---|
| C1 | H.264 effect on card recognition (KICS 2026 paper) | Published |
| C2 | Measurement of codec-induced redundancy in casino CCTV | In progress |
| C3 | CATA-Net — codec-structure-guided temporal aggregation | In progress |
| C4 | Game-level evaluation (card → point → outcome accuracy) | Planned |
| C5 | Real-time Jetson Orin Nano deployment | Planned |

---

## Environment Setup

```bash
# Create and activate conda environment
conda create -n venv_masters_thesis_project python=3.10 -y
conda activate venv_masters_thesis_project

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics av opencv-python matplotlib pillow numpy
```

**Requirements:**
- Python 3.10
- CUDA-capable GPU (tested on NVIDIA RTX 3080 Ti)
- ffmpeg (system package: `sudo apt install ffmpeg`)

---

## Repository Structure

```
source_code/
│
├── src/                            # All source code
│   ├── models/
│   │   ├── mobilenet.py            # MobileNetV1-1.25 (fdim=1280, 52 classes)
│   │   └── cata.py                 # CATA-Net Transformer aggregator [TODO]
│   │
│   ├── train_detector.py           # Fine-tune YOLOv11n (card/hand/chip)
│   ├── train_classifier.py         # Train MobileNetV1-1.25 (52 card classes)
│   ├── train_all_crfs.py           # Train one model per CRF level (7 models)
│   ├── compress_crops.py           # Compress card crops at CRF 18/23/28/35/43/51
│   ├── eval_crf.py                 # Evaluate accuracy vs CRF — produces thesis Fig 1
│   ├── reencode.py                 # Re-encode videos at 7 CRFs × 2 GOP settings
│   ├── track.py                    # YOLO + ByteTrack + MobileNet on video
│   ├── extract.py                  # Extract crops + I/P/B metadata per CRF [TODO]
│   ├── featurize.py                # MobileNet features → .npz cache [TODO]
│   ├── datasets.py                 # TrackWindowDataset for CATA-Net [TODO]
│   ├── baselines.py                # B1–B5 baseline implementations [TODO]
│   ├── train.py                    # CATA-Net training loop [TODO]
│   ├── eval.py                     # Full evaluation + result tables [TODO]
│   ├── demo_overlay.py             # Demo video with boxes/frame-type overlay
│   └── deploy/                     # Jetson Orin Nano deployment [TODO]
│
├── data/                           # Small config and label files (tracked in git)
│   ├── data_detector.yaml          # YOLO dataset config (3 classes)
│   ├── class_names.json            # 52 card class index ↔ name mapping
│   ├── classifier_class_map.json   # Auto-generated during training
│   ├── video_inventory.csv         # ffprobe info for all videos
│   ├── detections.csv              # Per-frame YOLO detections with track IDs
│   └── tracks.csv                  # Per-track card labels (majority-vote)
│
├── models/                         # Trained weights — NOT in git (too large)
│   ├── mobilenet_classifier.pt     # MobileNetV1-1.25 trained on clean data (CRF=0)
│   ├── mobilenet_crf18.pt          # MobileNetV1-1.25 trained on CRF-18 data
│   ├── mobilenet_crf23.pt          # MobileNetV1-1.25 trained on CRF-23 data
│   ├── mobilenet_crf28.pt          # MobileNetV1-1.25 trained on CRF-28 data
│   ├── mobilenet_crf35.pt          # MobileNetV1-1.25 trained on CRF-35 data
│   ├── mobilenet_crf43.pt          # MobileNetV1-1.25 trained on CRF-43 data
│   ├── mobilenet_crf51.pt          # MobileNetV1-1.25 trained on CRF-51 data
│   ├── yolo11n_cards.pt            # YOLOv11n fine-tuned on 3 classes
│   ├── classifier_history.csv      # Training history (original clean model)
│   └── history_mobilenet_crf*.csv  # Training history per CRF model (tracked)
│
├── videos/                         # Re-encoded videos — NOT in git
│   └── ref/
│       ├── bacarat_video_crf0_g60.mp4
│       ├── bacarat_video_crf18_g60.mp4
│       ├── ...                     # 14 files: 7 CRFs × 2 GOP settings
│       └── bacarat_video_crf51_g250.mp4
│
├── features/                       # MobileNet .npz cache — NOT in git
│   └── {session}_{crf}.npz
│
├── results/                        # Experiment outputs — tracked in git
│   ├── accuracy_vs_crf.csv         # Top-1/Top-5 accuracy per CRF (thesis Fig 1)
│   ├── accuracy_vs_crf.png         # Thesis Figure 1 plot
│   └── ...                         # Ablation, main table, etc.
│
├── dataset_report/                 # Data analysis outputs
│   ├── 01_summary.txt              # Full text report for Prof. Ahn
│   ├── 02_class_counts.png         # Images per class (train/valid)
│   ├── 03_augmentation.png         # Original vs augmented breakdown
│   ├── 04_sample_images.png        # 52-class sample image grid
│   └── 05_video_info.png           # Video codec + frame-type charts
│
├── additionals/                    # Extra utility scripts
│   └── run.py
│
├── data_analysis.py                # Comprehensive dataset analysis report
├── .gitignore
└── README.md
```

---

## Step-by-Step Run Guide

### Step 1 — Understand the data
```bash
python data_analysis.py
# Output: dataset_report/  (5 files: text report + 4 PNG figures)
```

### Step 2 — Train the YOLO detector (card / hand / chip)
```bash
conda activate venv_masters_thesis_project
python src/train_detector.py
# Output: models/yolo11n_cards.pt
# Time  : ~30 min on RTX 3080 Ti
```

### Step 3 — Train the card classifier (52 classes, clean data)
```bash
python src/train_classifier.py
# Output: models/mobilenet_classifier.pt
#         models/history_mobilenet_classifier.csv
# Time  : ~6 min on RTX 3080 Ti (20 epochs)
```

### Step 4 — Run tracking on a video
```bash
python src/track.py --video dataset/bacarat_video.mp4
# Output: data/detections.csv  (per-frame detections)
#         data/tracks.csv      (per-card-track labels)
```

### Step 5 — Generate demo overlay video
```bash
python src/demo_overlay.py --video dataset/bacarat_video.mp4
# Output: dataset_report/demo_bacarat_video.mp4
# Shows : boxes + track IDs + card classification + frame type (I/P/B)
```

### Step 6 — Re-encode videos at 7 CRF levels
```bash
python src/reencode.py
# Output: videos/ref/bacarat_video_crf{N}_g{GOP}.mp4   (14 files)
#         data/video_inventory.csv
# CRFs  : 0, 18, 23, 28, 35, 43, 51
# GOPs  : 60 frames (2s @ 30fps) and 250 frames (long GOP)
```

### Step 7 — Compress card crops at 6 CRF levels
```bash
python src/compress_crops.py
# Output: dataset/image_data/card_datset_crf{N}/  (6 directories, ~31k images each)
# Time  : ~9 min for all 6 CRFs
#
# Two-stage pipeline per CRF:
#   Stage 1: H.264 at 24×24 with CRF-proportional GOP
#            (short GOP at low CRF = I-frame dominant, high quality;
#             long GOP at high CRF = P-frame dominant, more artifacts)
#   Stage 2: JPEG re-save at CRF-proportional quality
#            (simulates small card region getting fewer bits in full video frame)
```

### Step 8 — Evaluate accuracy vs CRF (Thesis Figure 1)
```bash
python src/eval_crf.py
# Output: results/accuracy_vs_crf.csv
#         results/accuracy_vs_crf.png
#
# Tests the ORIGINAL (clean-trained) model on each CRF's compressed val set.
# Results (Top-1):
#   CRF  0 → 99.89%   CRF 18 → 96.08%   CRF 23 → 90.22%
#   CRF 28 → 83.30%   CRF 35 → 55.22%   CRF 43 → 28.73%   CRF 51 → 8.76%
```

### Step 8b — Train one model per CRF level
```bash
python src/train_all_crfs.py
# Output: models/mobilenet_crf{N}.pt  (6 models)
#         models/history_mobilenet_crf{N}.csv
# Time  : ~6 min per model (~36 min total for all 6)
# Use --force to re-train existing models
#
# Per-CRF model val accuracy (trained AND tested on same CRF's compressed data):
#   CRF 18 → 99.67%   CRF 23 → 99.60%   CRF 28 → 99.42%
#   CRF 35 → 98.21%   CRF 43 → 91.83%   CRF 51 → 54.17%
#
# Note: CRF-51 ceiling is 54.17% — even training on CRF-51 data can't
# recover beyond this. CATA-Net's temporal aggregation must break this limit.
```

### Steps 9–12 — CATA-Net (coming soon)
```bash
# Step 9  : python src/extract.py     — crops + I/P/B metadata per CRF
# Step 10 : python src/featurize.py   — MobileNet features → .npz cache
# Step 11 : python src/train.py       — train CATA-Net (3 seeds)
# Step 12 : python src/eval.py        — full experiment tables
```

---

## Model Architecture

### MobileNetV1-1.25 (Card Classifier)

| Property | Value |
|---|---|
| Input size | 128 × 128 |
| Width multiplier (α) | 1.25 |
| Feature dimension (fdim) | **1280** (fed into CATA-Net) |
| Output classes | 52 (13 ranks × 4 suits) |
| Parameters | 5.06 M |
| Optimizer | AdamW, lr=1e-3, cosine annealing schedule |
| Epochs | 20 |

### CATA-Net (Codec-Aware Temporal Aggregation)

| Property | Value |
|---|---|
| Input | Sequence of MobileNet features (fdim=1280) + codec metadata |
| Codec embeddings | Frame type (I/P/B/?) + GOP position + GOP index + confidence |
| Architecture | Transformer encoder (d=256, L=2 layers, H=4 heads) |
| Output heads | card (52+1), rank (13), suit (4) |
| Loss | CE(card) + 0.5 × (CE(rank) + CE(suit)) |
| Early stopping | Confidence ≥ 0.97 for ≥ 4 frames → decide immediately |

### YOLOv11n (Detector)

| Property | Value |
|---|---|
| Classes | card (0), hand (1), chip (2) |
| Input size | 640 × 640 |
| Base weights | Pretrained YOLOv11n (ultralytics) |
| Fine-tuned on | 1,263 casino frames |

---

## Dataset Summary

### Card Classification Dataset (`card_datset`)

| Split | Classes | Images | Originals | Augmented |
|---|---|---|---|---|
| train | 52 | 23,786 | 4.3% | 95.7% |
| valid | 52 | 7,211 | 8.1% | 91.9% |

- **52 classes:** 13 ranks (A 2–10 J Q K) × 4 suits (♣♦♥♠)
- Class naming: `{rank}_{suit}` e.g. `k_h` = King of Hearts
- Image size: variable ~62–86 × 74–106 px (cropped from casino video)

### Compressed Datasets (`card_datset_crf*`)

Generated by `src/compress_crops.py` using a two-stage pipeline:

| CRF | GOP | JPEG quality | Val accuracy (original model) |
|-----|-----|-------------|-------------------------------|
| 18  | 1   | 93          | 96.08% |
| 23  | 5   | 80          | 90.22% |
| 28  | 10  | 66          | 83.30% |
| 35  | 20  | 22          | 55.22% |
| 43  | 30  | 10          | 28.73% |
| 51  | 50  | 7           | 8.76%  |

### YOLO Detection Dataset (`yolo_dataset`)

| Split | Images | Card boxes | Hand boxes | Chip boxes |
|---|---|---|---|---|
| train | 1,263 | 4,613 | 4,412 | 29,922 |
| valid | 222 | 790 | 769 | 4,824 |

### Videos

| File | Duration | FPS | B-frames | Notes |
|---|---|---|---|---|
| `dataset/bacarat_video.mp4` | 238s | 30 | No | Main reference video |
| `live_test_video/bacarat.mp4` | 400s | 30 | **Yes** | I+P+B structure |
| `live_test_video/chip.mp4` | 330s | 30 | No | Very long GOP (1 I per 8s) |
| `live_test_video/chip2.mp4` | 186s | 30 | No | — |
| `live_test_video/play.mp4` | 379s | 30 | No | — |

---

## Key Design Decisions

**Why MobileNetV1-1.25?**  
Matches the existing ITICO commercial system. fdim=1280 feeds directly into CATA-Net without any projection overhead.

**Why two-stage compression (H.264 + JPEG) for the eval study?**  
Compressing only the cropped card at a given CRF is too lenient — the codec dedicates 100% of its bits to the card and keeps it near-perfect even at CRF 51. In a real video, the card occupies ~1–2% of the full frame and has to compete for bits. We simulate this with:
- Stage 1: H.264 at 24×24 with CRF-proportional GOP (I-frame dominant at low CRF, P-frame dominant at high CRF — matching real encoder behaviour)
- Stage 2: CRF-proportional JPEG quality (simulating reduced bits-per-pixel for the small card region in a full video frame)

**Why CRF-proportional GOP?**  
Real video encoders shorten the GOP at high quality (low CRF) because quality is prioritized over compression ratio. At low quality (high CRF), longer GOPs amortize the high cost of I-frames. This creates physically realistic P-frame artifact patterns that CATA-Net's frame-type embeddings can exploit.

**Why 128×128 for the classifier?**  
MobileNet resizes all inputs to 128×128 during training. The compressed 24×24 image is upscaled 5× at eval time, amplifying compression artifacts in a way the model must learn to handle.

---

## References

1. Islam, B. M. S. & Ahn, H. *Impact of H.264 video compression on card recognition in AI-based casino baccarat table monitoring systems.* KICS Summer Conference, 2026.
2. Yang et al. *Neural Aggregation Network for Video Face Recognition.* CVPR 2017.
3. Howard et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv 2017.
4. Zhang et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022.
5. Wu et al. *Compressed Video Action Recognition.* CVPR 2018.
