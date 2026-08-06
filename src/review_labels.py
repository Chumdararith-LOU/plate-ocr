#!/usr/bin/env python3
"""Review flagged images from audit_labels.py and fix the CSV.

Two ways to view the flagged images:
    --html out.html   generate a self-contained web page with all flagged crops
    (default)         terminal prompts; open each image yourself (e.g. in VS Code)

Answers:  Enter=keep  c=accept model prediction  d=drop sample  q=save and quit

Usage:  python src/review_labels.py --split val [--min-conf 0.95] [--retrain]
"""
import argparse
import base64
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OCR_DIR = ROOT / "data" / "ocr"


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def corrected_text(text: str, pred: str) -> str:
    parts = text.split("_")
    return "_".join([parts[0], "licence", "plate",
                     pred.replace(" ", "_"), parts[-1]])


def write_html(flags, img_dir, out_path):
    cards = []
    for fl in flags:
        p = img_dir / fl["filename"]
        if not p.exists():
            continue
        b64 = base64.b64encode(p.read_bytes()).decode()
        cards.append(f"""
        <div class="card">
          <div><b>{fl['filename']}</b> conf={fl['conf']}</div>
          <div class="label">label: {fl['label']}</div>
          <div class="pred">pred:  {fl['pred']}</div>
          <img src="data:image/jpeg;base64,{b64}">
        </div>""")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>flagged labels</title><style>
body {{ font-family: monospace; background: #222; color: #eee; }}
.card {{ display: inline-block; margin: 10px; padding: 10px; background: #333;
        border: 1px solid #555; vertical-align: top; }}
.card img {{ max-width: 480px; image-rendering: pixelated;
            display: block; margin-top: 6px; }}
.label {{ color: #7f7; }} .pred {{ color: #f77; }}
</style></head><body>
<h1>{len(flags)} flagged samples</h1>{''.join(cards)}
</body></html>"""
    Path(out_path).write_text(html)
    print(f"wrote {out_path} -- open it in a browser to inspect the images")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", help="train | val | test")
    ap.add_argument("--audit", default=None,
                    help="audit CSV (default: data/ocr/audit_<split>.csv)")
    ap.add_argument("--min-conf", type=float, default=0.9,
                    help="only review flags with confidence above this")
    ap.add_argument("--max", type=int, default=None, help="review at most N flags")
    ap.add_argument("--html", default=None,
                    help="write an HTML contact sheet and exit (no editing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list flags without prompting")
    ap.add_argument("--retrain", action="store_true",
                    help="run train_ocr.py after saving corrections")
    args = ap.parse_args()

    audit_path = Path(args.audit) if args.audit else OCR_DIR / f"audit_{args.split}.csv"
    flags = [r for r in load_csv(audit_path) if float(r["conf"]) >= args.min_conf]
    flags.sort(key=lambda r: float(r["conf"]), reverse=True)
    if args.max:
        flags = flags[:args.max]
    print(f"{len(flags)} flags with conf >= {args.min_conf}")

    img_dir = OCR_DIR / args.split
    if args.html:
        write_html(flags, img_dir, args.html)
        return
    if args.dry_run:
        for fl in flags:
            print(f"  {fl['conf']}  {fl['label']:<20} -> {fl['pred']:<20}  {fl['filename']}")
        return

    split_csv = OCR_DIR / f"{args.split}.csv"
    rows = load_csv(split_csv)
    by_fname = {r["filename"]: r for r in rows}
    drop = set()
    kept = fixed = 0

    for i, fl in enumerate(flags, 1):
        fname, label, pred, conf = fl["filename"], fl["label"], fl["pred"], fl["conf"]
        if fname not in by_fname or fname in drop:
            continue
        if not (img_dir / fname).exists():
            print(f"missing image: {fname}")
            continue
        print(f"\n[{i}/{len(flags)}] {fname}  conf={conf}")
        print(f"  label: {label}")
        print(f"  pred:  {pred}")
        print(f"  image: {img_dir / fname}")
        while True:
            key = input("  keep=Enter  fix=c  drop=d  quit=q > ").strip().lower()
            if key in ("", "c", "d", "q"):
                break
        if key == "q":
            break
        if key == "c":
            by_fname[fname]["text"] = corrected_text(by_fname[fname]["text"], pred)
            fixed += 1
            print(f"  fixed   {fname}: {label} -> {pred}")
        elif key == "d":
            drop.add(fname)
            print(f"  dropped {fname}")
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
