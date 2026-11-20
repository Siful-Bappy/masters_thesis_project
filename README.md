# CATA-Net: Compression-Robust Card Recognition for AI-Based Baccarat Table Monitoring

**Author:** B. M. S. Islam  
**Advisor:** Prof. Heejune Ahn  
**Institution:** Seoul National University of Science and Technology (SeoulTech)  
**Thesis:** Compression-Robust and Edge-Deployable Playing-Card Recognition for AI-Based Baccarat Table Monitoring

---

## Overview

The commercial baccarat monitoring system classifies cards **one frame at a time**. When H.264 video compression is applied (as in real casino CCTV systems), accuracy drops from **98.15% → 76.50%** as CRF increases from 0 to 51 (published at KICS Summer 2026).

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
pip install ultralytics av opencv-python matplotlib pillow numpy pandas
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
│   ├── yolo11n_cards.pt            # YOLOv11n trained on 3 classes
│   ├── mobilenet_classifier.pt     # MobileNetV1-1.25 best checkpoint
│   ├── mobilenet_last.pt           # MobileNetV1-1.25 last epoch
│   └── classifier_history.csv      # Training loss/accuracy per epoch
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
│   ├── accuracy_vs_crf.csv         # Top-1/Top-5 accuracy per CRF
│   ├── accuracy_vs_crf.png         # Thesis Figure 1
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
python src/train_detector.py
# Output: models/yolo11n_cards.pt
# Time  : ~30 min on RTX 3080 Ti
```

### Step 3 — Train the card classifier (52 classes)
```bash
python src/train_classifier.py
# Output: models/mobilenet_classifier.pt
# Time  : ~15 min on RTX 3080 Ti
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
# Output: dataset/image_data/card_datset_crf{N}/  (6 directories)
# Time  : ~12 min for all 6 CRFs
```

### Step 8 — Evaluate accuracy vs CRF (Thesis Figure 1)
```bash
python src/eval_crf.py
# Output: results/accuracy_vs_crf.csv
#         results/accuracy_vs_crf.png
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
| Optimizer | AdamW, lr=1e-3, cosine schedule |
| Epochs | 60 |

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
- **Note:** Card-back class (53rd class) not yet available

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

**Why GOP=1 for image compression study?**  
When compressing individual card crops, GOP=1 forces every frame to be an independent I-frame — no temporal prediction between unrelated card images. This correctly isolates the spatial H.264 compression artifact.

**Why 128×128 for compressed crops?**  
MobileNet resizes all inputs to 128×128 during training. Measuring compression at this resolution means the artifact is exactly what the model experiences — not inflated by later downsampling.

**Why session-level train/val/test split?**  
The existing dataset has augmented copies of the same card appearance in both splits (leaky split). Session-level splitting ensures no frame of a test session appears in training, giving honest accuracy numbers.

---

## References

1. Islam, B. M. S. & Ahn, H. *Impact of H.264 video compression on card recognition in AI-based casino baccarat table monitoring systems.* KICS Summer Conference, 2026.
2. Yang et al. *Neural Aggregation Network for Video Face Recognition.* CVPR 2017.
3. Howard et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv 2017.
4. Zhang et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022.
5. Wu et al. *Compressed Video Action Recognition.* CVPR 2018.
