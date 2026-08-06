#!/usr/bin/env python3
"""Interactively review flagged images from audit_labels.py and fix the CSV.

Shows each flagged crop. Keys:
    Enter/Space  keep the label
    c            correct the label to the model's prediction
    d            drop the sample from the split
    q            stop and save

Usage:  python src/review_labels.py --split val [--min-conf 0.95] [--retrain]
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = ROOT / "data" / "ocr"


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def corrected_text(text: str, pred: str) -> str:
    parts = text.split("_")
    return "_".join([parts[0], "licence", "plate",
                     pred.replace(" ", "_"), parts[-1]])


def render(img, fname, label, pred, conf):
    h, w = img.shape[:2]
    scale = max(240 / h, 640 / w, 1.0)
    disp = cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_CUBIC)
    header = np.full((100, max(disp.shape[1], 700), 3), 40, np.uint8)
    if disp.shape[1] < header.shape[1]:
        disp = cv2.copyMakeBorder(disp, 0, 0, 0, header.shape[1] - disp.shape[1],
                                  cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(header, f"{fname}  conf={conf}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(header, f"label: {label}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(header, f"pred:  {pred}", (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return np.vstack([header, disp])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", help="train | val | test")
    ap.add_argument("--audit", default=None,
                    help="audit CSV (default: data/ocr/audit_<split>.csv)")
    ap.add_argument("--min-conf", type=float, default=0.9,
                    help="only review flags with confidence above this")
    ap.add_argument("--max", type=int, default=None, help="review at most N flags")
    ap.add_argument("--dry-run", action="store_true",
                    help="list flags without showing images")
    ap.add_argument("--retrain", action="store_true",
                    help="run train_ocr.py after saving corrections")
    args = ap.parse_args()

    audit_path = Path(args.audit) if args.audit else OCR_DIR / f"audit_{args.split}.csv"
    flags = [r for r in load_csv(audit_path) if float(r["conf"]) >= args.min_conf]
    flags.sort(key=lambda r: float(r["conf"]), reverse=True)
    if args.max:
        flags = flags[:args.max]
    print(f"{len(flags)} flags with conf >= {args.min_conf}")

    if args.dry_run:
        for fl in flags:
            print(f"  {fl['conf']}  {fl['label']:<20} -> {fl['pred']:<20}  {fl['filename']}")
        return

    split_csv = OCR_DIR / f"{args.split}.csv"
    rows = load_csv(split_csv)
    by_fname = {r["filename"]: r for r in rows}
    img_dir = OCR_DIR / args.split
    drop = set()
    kept = fixed = 0

    for i, fl in enumerate(flags, 1):
        fname, label, pred, conf = fl["filename"], fl["label"], fl["pred"], fl["conf"]
        if fname not in by_fname or fname in drop:
            continue
        img = cv2.imread(str(img_dir / fname))
        if img is None:
            print(f"missing image: {fname}")
            continue
        cv2.imshow(f"[{i}/{len(flags)}]  keep=Enter  fix=c  drop=d  quit=q",
                   render(img, fname, label, pred, conf))
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord("q"):
            break
        if key == ord("c"):
            by_fname[fname]["text"] = corrected_text(by_fname[fname]["text"], pred)
            fixed += 1
            print(f"fixed   {fname}: {label} -> {pred}")
        elif key == ord("d"):
            drop.add(fname)
            print(f"dropped {fname}")
        else:
            kept += 1

    out_rows = [r for r in rows if r["filename"] not in drop]
    with open(split_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "text"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nkept={kept} fixed={fixed} dropped={len(drop)}  ->  {split_csv}")

    if args.retrain:
        subprocess.run([sys.executable, "src/train_ocr.py"], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
