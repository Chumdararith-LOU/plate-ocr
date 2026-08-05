#!/usr/bin/env python3
"""Full pipeline: detect plate characters with YOLO, crop the plate region,
read its text with the CRNN.

Usage:  python src/inference.py --source path/to/image_or_dir
"""
import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from train_ocr import CRNN, decode

ROOT = Path(__file__).resolve().parent.parent
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_ocr(weights: Path, device: str):
    ckpt = torch.load(weights, map_location=device)
    vocab = ckpt["vocab"]
    model = CRNN(len(vocab)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tfm = transforms.Compose([
        transforms.Resize((ckpt["height"], ckpt["width"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, vocab, tfm


def read_text(model, vocab, tfm, crop: Image.Image, device: str) -> str:
    with torch.no_grad():
        logits = model(tfm(crop).unsqueeze(0).to(device))
    ids = logits.argmax(2).permute(1, 0).cpu().tolist()[0]
    return decode(ids, vocab)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="image file or directory")
    ap.add_argument("--detect-weights",
                    default=str(ROOT / "runs" / "detect" / "plate" / "weights" / "best.pt"))
    ap.add_argument("--ocr-weights",
                    default=str(ROOT / "runs" / "ocr" / "plate_ocr" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pad", type=float, default=0.1,
                    help="fractional padding around the union box")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-dir", default=str(ROOT / "runs" / "inference"))
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else
                             "cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.detect_weights)
    ocr_model, vocab, tfm = load_ocr(Path(args.ocr_weights), device)

    source = Path(args.source)
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in EXTS) \
        if source.is_dir() else [source]

    for path in paths:
        results = yolo(path, conf=args.conf, device=device, verbose=False)[0]
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        text = None
        if results.boxes is not None and len(results.boxes):
            for b in results.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            x1, y1, x2, y2 = results.boxes.xyxy.cpu().numpy().min(axis=0)[:2].tolist() + \
                results.boxes.xyxy.cpu().numpy().max(axis=0)[2:].tolist()
            pw, ph = (x2 - x1) * args.pad, (y2 - y1) * args.pad
            x1 = max(0, int(x1 - pw)); y1 = max(0, int(y1 - ph))
            x2 = min(w, int(x2 + pw)); y2 = min(h, int(y2 + ph))
            crop = Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            text = read_text(ocr_model, vocab, tfm, crop, device)
            cv2.putText(img, text, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        out = save_dir / f"result_{path.name}"
        cv2.imwrite(str(out), img)
        print(f"{path.name}: {text if text else 'no plate detected'}")


if __name__ == "__main__":
    main()
