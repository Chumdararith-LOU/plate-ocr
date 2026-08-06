#!/usr/bin/env python3
"""Train the plate-text recognition model (CRNN + CTC) on data/ocr/.

Targets are province strings parsed from labels, e.g.
    train_licence_plate_banteay_meanchey_73 -> "banteay meanchey"

Smoke test:  python src/train_ocr.py --epochs 1 --limit 64
Full run:    python src/train_ocr.py
"""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = ROOT / "data" / "ocr"
RUN_DIR = ROOT / "runs" / "ocr" / "plate_ocr"


def parse_target(text: str) -> str:
    parts = text.split("_")
    province = parts[3:-1]
    return " ".join(province)


def build_vocab(targets) -> str:
    chars = sorted({c for t in targets for c in t})
    return "".join(chars)


def encode(text: str, vocab: str):
    return [vocab.index(c) for c in text]


def decode(ids, vocab: str) -> str:
    blank = len(vocab)
    out, prev = [], blank
    for i in ids:
        if i != blank and i != prev:
            out.append(vocab[i])
        prev = i
    return "".join(out)


class OcrDataset(Dataset):
    def __init__(self, csv_path: Path, vocab: str, tfm, limit=None):
        self.rows, self.tfm, self.vocab = [], tfm, vocab
        self.img_dir = OCR_DIR / csv_path.stem
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                self.rows.append((row["filename"], parse_target(row["text"])))
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fname, text = self.rows[idx]
        img = Image.open(self.img_dir / fname).convert("RGB")
        return self.tfm(img), torch.tensor(encode(text, self.vocab), dtype=torch.long), text


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = torch.cat([b[1] for b in batch])
    texts = [b[2] for b in batch]
    return imgs, targets, texts


class CRNN(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(512, vocab_size + 1)

    def forward(self, x):
        x = self.cnn(x).squeeze(2).permute(2, 0, 1)
        x, _ = self.rnn(x)
        return self.fc(x)


def run_epoch(model, loader, device, train: bool, opt=None):
    model.train(train)
    loss_fn = nn.CTCLoss(zero_infinity=True)
    total, correct, n_chars, n_err, total_loss = 0, 0, 0, 0, 0.0
    vocab_len = model.fc.out_features - 1
    vocab = loader.dataset.vocab
    for imgs, targets, texts in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        logits = model(imgs)
        log_probs = logits.log_softmax(2)
        target_lengths = torch.tensor([len(t) for t in texts], device=device)
        input_lengths = torch.full((len(texts),), log_probs.size(0), device=device)
        loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        total_loss += loss.item() * len(texts)
        ids = logits.argmax(2).permute(1, 0).cpu().tolist()
        for pred_ids, text in zip(ids, texts):
            pred = decode(pred_ids, vocab)
            correct += pred == text
            n_chars += len(text)
            n_err += sum(a != b for a, b in zip(pred, text)) + abs(len(pred) - len(text))
        total += len(texts)
    return total_loss / total, correct / total, n_err / max(n_chars, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap samples (smoke test)")
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")

    with open(OCR_DIR / "train.csv", newline="") as f:
        targets = [parse_target(r["text"]) for r in csv.DictReader(f)]
    vocab = build_vocab(targets)

    train_tfm = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.RandomPerspective(0.1, p=0.3),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = OcrDataset(OCR_DIR / "train.csv", vocab, train_tfm, args.limit)
    val_ds = OcrDataset(OCR_DIR / "val.csv", vocab, val_tfm, args.limit)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers,
                        collate_fn=collate)

    model = CRNN(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "vocab.json").write_text(json.dumps({"vocab": vocab}))
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(model, train_dl, device, True, opt)
        val_loss, acc, cer = run_epoch(model, val_dl, device, False)
        sched.step()
        print(f"epoch {epoch}/{args.epochs} train_loss={tr_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={acc:.4f} val_cer={cer:.4f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "vocab": vocab,
                        "height": args.height, "width": args.width},
                       RUN_DIR / "best.pt")
    print(f"best val_acc={best_acc:.4f} weights: {RUN_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
