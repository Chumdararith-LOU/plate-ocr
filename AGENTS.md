# AGENTS.md

## Dataset Analysis

Two sub-datasets, two-stage pipeline:

### 1. `DataSet/train/` — Detection (YOLO format)
- **1,962 images** (jpg) + 1,963 label files (one mismatch to fix)
- Labels are **YOLO polygon format**: `class_id x1 y1 x2 y2 x3 y3 x4 y4 x5 y5`
- **21 classes** (IDs 1–27, with gaps) — likely individual Khmer characters/digits on plates
- Class distribution is imbalanced

### 2. `DataSet/khmer_dataset/` — Recognition (OCR)
- **2,860 images** + `labels.csv` (`filename,text`)
- Text labels like `test_licence_plate_stung_treng_2` — plate text encoded with province prefix
- Covers provinces/categories: phnom_penh, svay_rieng, siem_reap, prey_veng, kampong_cham, cambodia, kampong_speu, kratie, battambang, rcaf, kampong_thom, banteay_meanchey, pursat, police, kandal, oddar_meanchey, takeo, tboung_khmum, kampong_chhnang, kampot, koh_kong, state, preah_vihear, pailin, stung_treng, sihanoukville, ...
- Includes train/valid splits encoded in label prefixes

## Proposed Plan

**Architecture: 2-stage pipeline**
1. **Detection** — YOLOv8/YOLO11 (Ultralytics) trained on `train/` to detect plates + characters (21 classes, polygon boxes)
2. **Recognition** — read character sequence from cropped plates using `khmer_dataset/` (image→text), e.g. CRNN+CTC or sequence classification

**Workspace structure**
```
Plate number detection/
├── data/                  # gitignored; dataset lives here, synced separately
├── src/
│   ├── prepare_data.py    # split train/val, validate labels, fix class ID gaps
│   ├── train_detect.py    # YOLO training config
│   ├── train_ocr.py
│   └── inference.py       # full pipeline: detect → crop → read
├── configs/
├── runs/                  # gitignored (training outputs)
├── requirements.txt
└── .gitignore
```

**Key decisions**
- **Don't push DataSet via git** (136M, and GitHub blocks files >100MB). Use `.gitignore` + `rsync`/`scp` to server, or Git LFS if you prefer.
- Local MacBook: prototyping, data prep, tiny smoke-train. Server (RTX 4090): real training.
- Data prep first: verify image↔label pairing (1962 vs 1963 mismatch), remap sparse class IDs (1–27 with gaps) to contiguous 0–20, create train/val split + YOLO `data.yaml`.

## Commands
- Sync code to server via git/GitHub; sync dataset separately via `rsync`/`scp`.
- Run lint/typecheck commands if provided before committing.
