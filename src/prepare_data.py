#!/usr/bin/env python3
"""Validate, remap, and split both datasets into training-ready layouts.

Outputs:
    data/detector/{images,labels}/{train,val}/ + data/detector/data.yaml
    data/ocr/{train,val,test}/ + data/ocr/{train,val,test}.csv
    configs/class_map.json
"""
import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DET_SRC = ROOT / "DataSet" / "train"
OCR_SRC = ROOT / "DataSet" / "khmer_dataset"
DET_OUT = ROOT / "data" / "detector"
OCR_OUT = ROOT / "data" / "ocr"
CONFIGS = ROOT / "configs"


def parse_label_file(path: Path):
    """Return (rows, problems). Each row is (class_id, [coords])."""
    rows, problems = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5 or (len(parts) - 1) % 2 != 0:
            problems.append(f"{path.name}:{i} bad field count ({len(parts)})")
            continue
        try:
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            problems.append(f"{path.name}:{i} non-numeric values")
            continue
        bad = [v for v in coords if not 0.0 <= v <= 1.0]
        if bad:
            problems.append(f"{path.name}:{i} coords out of [0,1]: {bad[:3]}")
            continue
        rows.append((cls, coords))
    return rows, problems


def validate_detection():
    images = {p.stem: p for p in DET_SRC.glob("images/*.jpg")}
    labels = {p.stem: p for p in DET_SRC.glob("labels/*.txt")}
    orphan_labels = sorted(set(labels) - set(images))
    orphan_images = sorted(set(images) - set(labels))
    print(f"[detect] images={len(images)} labels={len(labels)}")
    for s in orphan_labels:
        print(f"[detect] label without image: {s}.txt")
    for s in orphan_images:
        print(f"[detect] image without label: {s}.jpg")

    samples, problems, class_counts = {}, [], Counter()
    for stem, lpath in labels.items():
        if stem not in images:
            continue
        rows, probs = parse_label_file(lpath)
        problems.extend(probs)
        samples[stem] = rows
        class_counts.update(c for c, _ in rows)

    n_neg = sum(1 for r in samples.values() if not r)
    print(f"[detect] valid samples={len(samples)} ({n_neg} negatives), boxes={sum(class_counts.values())}")
    print(f"[detect] class id counts: {dict(sorted(class_counts.items()))}")
    if problems:
        print(f"[detect] {len(problems)} label problems (first 10):")
        for p in problems[:10]:
            print(f"  {p}")
    return samples, class_counts


def build_detector(samples, class_counts, seed, val_frac):
    old_ids = sorted(class_counts)
    id_map = {old: new for new, old in enumerate(old_ids)}
    CONFIGS.mkdir(exist_ok=True)
    (CONFIGS / "class_map.json").write_text(json.dumps(
        {str(k): v for k, v in id_map.items()}, indent=2))
    print(f"[detect] class remap {old_ids} -> 0..{len(old_ids)-1} saved to configs/class_map.json")

    stems = sorted(samples)
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val = int(len(stems) * val_frac)
    split = {"val": set(stems[:n_val]), "train": set(stems[n_val:])}

    shutil.rmtree(DET_OUT, ignore_errors=True)
    for part in ("train", "val"):
        (DET_OUT / "images" / part).mkdir(parents=True)
        (DET_OUT / "labels" / part).mkdir(parents=True)

    for part, stems_in_part in split.items():
        for stem in stems_in_part:
            shutil.copy2(DET_SRC / "images" / f"{stem}.jpg",
                         DET_OUT / "images" / part / f"{stem}.jpg")
            lines = []
            for cls, coords in samples[stem]:
                lines.append(" ".join([str(id_map[cls])] + [f"{v:.6f}" for v in coords]))
            text = "\n".join(lines) + "\n" if lines else ""
            (DET_OUT / "labels" / part / f"{stem}.txt").write_text(text)
        print(f"[detect] {part}: {len(stems_in_part)} images")

    names = "\n".join(f"  {i}: class_{i}" for i in range(len(id_map)))
    yaml_text = (
        f"path: {DET_OUT}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(id_map)}\n"
        f"names:\n{names}\n"
    )
    (DET_OUT / "data.yaml").write_text(yaml_text)
    print(f"[detect] wrote {DET_OUT / 'data.yaml'}")


def build_ocr():
    shutil.rmtree(OCR_OUT, ignore_errors=True)
    counts = Counter()
    rows_by_part = {"train": [], "val": [], "test": []}
    with open(OCR_SRC / "labels.csv", newline="") as f:
        for row in csv.DictReader(f):
            fname, text = row["filename"], row["text"]
            part = text.split("_", 1)[0]
            if part == "valid":
                part = "val"
            if part not in rows_by_part:
                print(f"[ocr] unknown split prefix '{part}' in {fname}")
                continue
            src = OCR_SRC / "images" / fname
            if not src.exists():
                print(f"[ocr] missing image: {fname}")
                continue
            dst_dir = OCR_OUT / part
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / fname)
            rows_by_part[part].append((fname, text))
            counts[part] += 1

    for part, rows in rows_by_part.items():
        if not rows:
            continue
        with open(OCR_OUT / f"{part}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "text"])
            w.writerows(rows)
        print(f"[ocr] {part}: {counts[part]} images -> {OCR_OUT / f'{part}.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--skip-ocr", action="store_true")
    args = ap.parse_args()

    samples, class_counts = validate_detection()
    build_detector(samples, class_counts, args.seed, args.val_frac)
    if not args.skip_ocr:
        build_ocr()


if __name__ == "__main__":
    main()
