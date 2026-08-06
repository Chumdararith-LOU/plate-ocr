#!/usr/bin/env python3
"""Evaluate the OCR model on a split (default: data/ocr/test.csv).

Usage:  python src/evaluate.py [--weights runs/ocr/plate_ocr/best.pt]
"""
import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from train_ocr import CRNN, OCR_DIR, OcrDataset, collate, decode, parse_target

ROOT = Path(__file__).resolve().parent.parent


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(ROOT / "runs" / "ocr" / "plate_ocr" / "best.pt"))
    ap.add_argument("--split", default="test", help="test | val | train")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--show-errors", type=int, default=10)
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location=device)
    vocab = ckpt["vocab"]
    model = CRNN(len(vocab)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((ckpt["height"], ckpt["width"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = OcrDataset(OCR_DIR / f"{args.split}.csv", vocab, tfm)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                    collate_fn=collate)

    total, correct, n_chars, n_err = 0, 0, 0, 0
    per_class = {}
    errors = []
    with torch.no_grad():
        for imgs, _, texts in dl:
            logits = model(imgs.to(device))
            ids_batch = logits.argmax(2).permute(1, 0).cpu().tolist()
            for ids, text in zip(ids_batch, texts):
                pred = decode(ids, vocab)
                correct += pred == text
                n_chars += len(text)
                n_err += levenshtein(pred, text)
                c = per_class.setdefault(text, [0, 0])
                c[0] += pred == text
                c[1] += 1
                if pred != text and len(errors) < args.show_errors:
                    errors.append((text, pred))
            total += len(texts)

    print(f"split={args.split} samples={total} "
          f"exact_acc={correct / total:.4f} cer={n_err / max(n_chars, 1):.4f}")
    print("\nper-class accuracy:")
    for text, (ok, n) in sorted(per_class.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {text:<20} {ok}/{n}  {ok / n:.2f}")
    if errors:
        print("\nsample errors (target -> pred):")
        for t, p in errors:
            print(f"  {t} -> {p}")


if __name__ == "__main__":
    main()
