#!/usr/bin/env python3
"""Train the plate-province classifier (ResNet18, ImageNet-pretrained) on data/ocr/.

Unlike the CRNN+CTC model, this treats each plate's province text as one of
28 classes and learns holistic visual patterns — much more robust on the
tiny/blurry crops where CTC struggles.

Targets are parsed from labels, e.g.
    train_licence_plate_banteay_meanchey_73 -> "banteay meanchey"

Smoke test:  python src/train_classifier.py --epochs 1 --limit 64
Full run:    python src/train_classifier.py
"""
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = ROOT / "data" / "ocr"
RUN_DIR = ROOT / "runs" / "ocr" / "plate_classifier"


def parse_target(text: str) -> str:
    parts = text.split("_")
    province = parts[3:-1]
    return " ".join(province)


class PlateDataset(Dataset):
    def __init__(self, csv_path: Path, class_to_idx: dict, tfm, limit=None):
        self.rows, self.tfm, self.class_to_idx = [], tfm, class_to_idx
        self.img_dir = OCR_DIR / csv_path.stem
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                cls = parse_target(row["text"])
                if cls in class_to_idx:
                    self.rows.append((row["filename"], cls))
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fname, cls = self.rows[idx]
        img = Image.open(self.img_dir / fname).convert("RGB")
        return self.tfm(img), self.class_to_idx[cls], cls


def build_model(num_classes: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (loss, accuracy, per-class [correct, total])."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    per_class = {}
    for imgs, labels, names in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * len(names)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += len(names)
        for pred, label, name in zip(preds.tolist(), labels.tolist(), names):
            c = per_class.setdefault(name, [0, 0])
            c[1] += 1
            c[0] += int(pred == label)
    return total_loss / max(total, 1), correct / max(total, 1), per_class


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap samples (smoke test)")
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")

    # classes come from the training split only, sorted for determinism
    with open(OCR_DIR / "train.csv", newline="") as f:
        classes = sorted({parse_target(r["text"]) for r in csv.DictReader(f)})
    class_to_idx = {c: i for i, c in enumerate(classes)}

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

    train_ds = PlateDataset(OCR_DIR / "train.csv", class_to_idx, train_tfm, args.limit)
    val_ds = PlateDataset(OCR_DIR / "val.csv", class_to_idx, val_tfm, args.limit)
    counts = Counter(cls for _, cls in train_ds.rows)
    weights = [1.0 / counts[cls] for _, cls in train_ds.rows]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                          num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers)

    model = build_model(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "classes.json").write_text(json.dumps({"classes": classes}))
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, total_loss = 0, 0, 0.0
        for imgs, labels, _ in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        sched.step()
        val_loss, acc, _ = evaluate(model, val_dl, device)
        print(f"epoch {epoch}/{args.epochs} "
              f"train_loss={total_loss / max(total, 1):.4f} "
              f"train_acc={correct / max(total, 1):.4f} "
              f"val_loss={val_loss:.4f} val_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "classes": classes,
                        "height": args.height, "width": args.width},
                       RUN_DIR / "best.pt")
    print(f"best val_acc={best_acc:.4f} weights: {RUN_DIR / 'best.pt'}")

    # final test-set report with the best checkpoint
    test_path = OCR_DIR / "test.csv"
    if test_path.exists():
        ckpt = torch.load(RUN_DIR / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        test_ds = PlateDataset(test_path, class_to_idx, val_tfm)
        test_dl = DataLoader(test_ds, batch_size=args.batch, num_workers=args.workers)
        _, acc, per_class = evaluate(model, test_dl, device)
        print(f"test_acc={acc:.4f} ({len(test_ds)} samples)")
        report = {name: {"acc": c[0] / c[1], "n": c[1]}
                  for name, c in sorted(per_class.items())}
        (RUN_DIR / "test_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
