#!/usr/bin/env python3
"""Single-command pipeline: prepare data -> train detector -> train OCR -> inference.

Usage:  python src/run_all.py --test-image path/to/image.jpg
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--test-image", default=None,
                    help="run inference on this image after training")
    ap.add_argument("--skip-prepare", action="store_true",
                    help="skip data prep if already done")
    args = ap.parse_args()

    py = sys.executable
    if not args.skip_prepare:
        run([py, "src/prepare_data.py"])
    run([py, "src/train_detect.py", "--epochs", str(args.epochs),
         "--imgsz", str(args.imgsz), "--batch", str(args.batch)])
    run([py, "src/train_ocr.py", "--epochs", str(args.epochs)])
    if args.test_image:
        run([py, "src/inference.py", "--source", args.test_image])
    print("\nAll done. Weights: runs/detect/plate/weights/best.pt, "
          "runs/ocr/plate_ocr/best.pt")


if __name__ == "__main__":
    main()
