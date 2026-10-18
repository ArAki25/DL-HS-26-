"""Komplette Pipeline ohne YOLO auf einem neuen Mikroskopbild:
klassische Segmentierung findet die Zellen, das trainierte Zell-CNN klassifiziert sie.

    python scripts/predict.py <bild.png> runs/<name>/best.pt [--out ergebnis.png]

Gibt Zellzahl pro Klasse und Parasitämie (Anteil infizierter Zellen) aus und speichert
das Bild mit eingezeichneten Boxen.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import STAGES, cell_transform, crop_box  # noqa: E402
from src.engine import get_device  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.segmentation import detect_cells  # noqa: E402

COLORS = {"rbc": "#4c9be8", "uninfected": "#4c9be8", "leukocyte": "#9b59b6",
          "ring": "#e67e22", "trophozoite": "#e74c3c", "schizont": "#c0392b",
          "gametocyte": "#f1c40f", "infected": "#e74c3c"}


@torch.no_grad()
def classify_cells(image, boxes, model, ckpt, device, batch_size=256):
    transform = cell_transform(ckpt["img_size"])
    crops = [transform(crop_box(image, *b)) for b in boxes]
    probs = []
    for i in range(0, len(crops), batch_size):
        probs.append(torch.softmax(model(torch.stack(crops[i:i + batch_size]).to(device)), 1).cpu())
    return torch.cat(probs).numpy() if probs else np.zeros((0, len(ckpt["class_names"])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("checkpoint", type=Path, help="Zell-Modell (task stage oder binary)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cell-diameter", type=int, default=100)
    args = p.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    if ckpt["task"] == "smear":
        sys.exit("Für predict.py ein Zell-Modell (--task stage/binary) verwenden.")
    names = ckpt["class_names"]

    image = Image.open(args.image).convert("RGB")
    boxes = detect_cells(np.asarray(image), cell_diameter=args.cell_diameter)
    probs = classify_cells(image, boxes, model, ckpt, device)
    preds = [names[i] for i in probs.argmax(1)]

    counts = Counter(preds)
    infected = sum(counts[c] for c in (*STAGES, "infected"))
    print(f"{len(boxes)} Zellen gefunden")
    for name in names:
        print(f"  {name:12s} {counts[name]:5d}")
    if boxes.size:
        print(f"Parasitämie: {infected / len(boxes):.1%}")

    draw = ImageDraw.Draw(image)
    for (r0, c0, r1, c1), pred in zip(boxes, preds):
        draw.rectangle((c0, r0, c1, r1), outline=COLORS.get(pred, "white"), width=3)
        if pred not in ("rbc", "uninfected"):
            draw.text((c0 + 4, r0 + 2), pred, fill=COLORS.get(pred, "white"))
    out = args.out or args.image.with_name(f"{args.image.stem}_pred.png")
    image.save(out)
    print(f"Gespeichert: {out}")


if __name__ == "__main__":
    main()
