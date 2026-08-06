#!/usr/bin/env python3
"""Audit labels for likely mislabels using the trained OCR model.

Runs the model over a split and flags samples where the model disagrees
with the label with high confidence. Review the flagged images manually.

Usage:  python src/audit_labels.py [--split train] [--weights runs/ocr/plate_ocr/best.pt]
"""
import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from train_ocr import CRNN, OCR_DIR, OcrDataset, collate

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=str(ROOT / "runs" / "ocr" / "plate_ocr" / "best.pt"))
    ap.add_argument("--split", default="train", help="train | val | test")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--top", type=int, default=50, help="show this many most suspicious")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="flag disagreements with confidence above this")
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location=device)
    vocab = ckpt["vocab"]
    blank = len(vocab)
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
                    collate_fn=collate, shuffle=False)

    def greedy(ids):
        out, prev = [], blank
        for i in ids:
            if i != blank and i != prev:
                out.append(vocab[i])
            prev = i
        return "".join(out)

    flagged, confusion = [], Counter()
    idx = 0
    with torch.no_grad():
        for imgs, _, texts in dl:
            probs = model(imgs.to(device)).softmax(2)
            for b, text in enumerate(texts):
                p = probs[b]
                ids = p.argmax(1).cpu().tolist()
                pred = greedy(ids)
                conf = p.max(1).values.mean().item()
                fname = ds.rows[idx][0]
                idx += 1
                confusion[(text, pred)] += 1
                if pred != text and conf >= args.min_conf:
                    flagged.append((conf, text, pred, fname))
    flagged.sort(reverse=True)

    print(f"split={args.split} samples={idx} flagged={len(flagged)}")
    print("\nmost suspicious (high-confidence disagreements):")
    for conf, text, pred, fname in flagged[:args.top]:
        print(f"  conf={conf:.2f}  {text:<20} -> {pred:<20}  {fname}")

    print("\ntop confusion pairs (target -> pred):")
    for (t, p), n in confusion.most_common(20):
        if t != p:
            print(f"  {t:<20} -> {p:<20}  x{n}")

    out = OCR_DIR / f"audit_{args.split}.csv"
    with open(out, "w") as f:
        f.write("filename,label,pred,conf\n")
        for conf, text, pred, fname in flagged:
            f.write(f"{fname},{text},{pred},{conf:.3f}\n")
    print(f"\nfull list: {out}")


if __name__ == "__main__":
    main()
