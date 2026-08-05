#!/usr/bin/env python3
"""Train the plate/character YOLO detector.

Smoke test on laptop:  python src/train_detect.py --epochs 2 --imgsz 320
Full run on server:    python src/train_detect.py --epochs 100 --model yolo11n.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data" / "detector" / "data.yaml"))
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, 'cpu' to force CPU")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs" / "detect"),
        name="plate",
        exist_ok=True,
        resume=args.resume,
    )
    print(f"best weights: {ROOT / 'runs' / 'detect' / 'plate' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
