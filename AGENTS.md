# AGENTS.md

## Workflow Rules (IMPORTANT — follow these)

1. **Write code on the MacBook, train on the RTX 4090 server.** The Mac is for
   writing code, data prep, and debugging only. All real training runs happen on
   the server. Never propose or start a full training run on the Mac.
2. **Local smoke tests only.** The Mac may run tiny sanity checks
   (e.g. `--epochs 1 --limit 64`) to verify a script works before pushing.
3. **Code sync = git + GitHub.** Commit working, smoke-tested code on the Mac,
   push to GitHub, then `git pull` on the server and train there.
4. **Data sync ≠ git.** `DataSet/` is ~136 MB and GitHub blocks files >100 MB.
   `data/`, `DataSet/`, and `runs/` are gitignored. Sync datasets to the server
   separately with `rsync`/`scp` (or Git LFS if preferred).
5. **Before committing:** check `git status` and `git diff`, stage only intended
   files, never commit secrets, and run lint/typecheck commands if provided.
6. **Server-side run pattern:** pull latest code, confirm the dataset is present
   at the expected path, launch training, and report metrics back (the weights
   in `runs/` stay gitignored — copy checkpoints back manually if needed).

## Project

Cambodian licence-plate recognition, 2-stage pipeline:
1. **Detection** — YOLO trained on `DataSet/train/` (21 character classes, polygon boxes)
2. **Recognition** — read the province text from the cropped plate

## Dataset Analysis

Two sub-datasets:

### 1. `DataSet/train/` — Detection (YOLO format)
- **1,962 images** (jpg) + 1,963 label files (one mismatch to fix)
- Labels are **YOLO polygon format**: `class_id x1 y1 x2 y2 x3 y3 x4 y4 x5 y5`
- **21 classes** (IDs 1–27, with gaps) — Khmer characters/digits on plates
- Class distribution is imbalanced

### 2. `DataSet/khmer_dataset/` — Recognition (OCR)
- **2,860 images** + `labels.csv` (`filename,text`)
- Text labels like `test_licence_plate_stung_treng_2` — plate text encoded with
  province prefix; parsed via `" ".join(text.split("_")[3:-1])`
- 28 classes incl. provinces and special categories (cambodia, police, rcaf, state)
- train/val/test splits encoded in label prefixes → `data/ocr/{train,val,test}.csv`
- Class imbalance: phnom penh 243 vs mondulkiri 6 (train split)

**Workspace structure**
```
Plate number detection/
├── data/                     # gitignored; dataset synced separately
├── src/
│   ├── prepare_data.py       # split train/val, validate labels, fix class ID gaps
│   ├── train_detect.py       # YOLO training
│   ├── train_ocr.py          # CRNN + CTC baseline
│   ├── train_classifier.py   # 28-class ResNet18 classifier (current approach)
│   ├── audit_labels.py       # flags likely mislabels via model-vs-label disagreement
│   ├── review_labels.py      # headless review + contact sheets for flagged images
│   └── inference.py          # full pipeline: detect → crop → read
├── configs/
├── runs/                     # gitignored (training outputs)
├── requirements.txt
└── .gitignore
```

## Status & Decisions

- **CRNN+CTC baseline** (`src/train_ocr.py`): plateaued; most errors occur on
  tiny/blurry crops where character-level decoding fails.
- **Label audit complete** (`audit_labels.py` + manual review of flagged crops):
  only 2 label issues found —
  - fix `plate_2839` → `rcaf` (conf 0.996, clearly an RCAF red plate)
  - drop `plate_2554` (ROYAL PALACE plate, no matching class)
  - all other flags were model errors, not mislabels.
  **Conclusion: the accuracy ceiling is image resolution, not label quality.**
- **Current approach: 28-class classifier** (`src/train_classifier.py`) —
  ResNet18 (ImageNet-pretrained) fine-tuned to classify the province text as a
  whole, instead of CTC character-by-character decoding. 64×160 input,
  WeightedRandomSampler for imbalance, AdamW + cosine LR, best checkpoint by
  val acc → `runs/ocr/plate_classifier/best.pt`, per-class `test_report.json`.
  Smoke-tested on the Mac (1 epoch, 64 samples); full training happens on the
  RTX 4090 server.

## Commands

```bash
# Mac — smoke test only
.venv/bin/python src/train_classifier.py --epochs 1 --limit 64

# git — sync code
git add -p && git commit -m "..." && git push        # Mac
git pull                                              # server

# dataset — sync separately (never via git)
rsync -av data/ocr/ user@server:<repo>/data/ocr/

# server — real training
python src/train_classifier.py
```
