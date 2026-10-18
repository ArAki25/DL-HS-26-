"""Wertet ein trainiertes Modell auf val oder test aus.

    python scripts/evaluate.py runs/<name>/best.pt --split test
    python scripts/evaluate.py runs/<name>/best.pt --fraction 0.1   # schneller Test

Schreibt metrics_<split>.json und confusion_<split>.png neben den Checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (CellDataset, SmearDataset, cell_transform, load_cells,  # noqa: E402
                      load_smears, smear_transform, subsample_images)
from src.engine import compute_metrics, get_device, predict  # noqa: E402
from src.models import load_checkpoint  # noqa: E402


def plot_confusion(cm, names, out_path):
    fig, ax = plt.subplots(figsize=(1 + 0.9 * len(names), 1 + 0.9 * len(names)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Vorhersage")
    ax.set_ylabel("Wahr")
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--fraction", type=float, default=1.0,
                   help="nur Anteil der Bilder auswerten (schneller Test), Default: alle")
    args = p.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    task, size, names = ckpt["task"], ckpt["img_size"], ckpt["class_names"]
    df = subsample_images((load_smears if task == "smear" else load_cells)(args.split), args.fraction)
    if task == "smear":
        ds = SmearDataset(df, smear_transform(size))
    else:
        ds = CellDataset(df, task, cell_transform(size))

    multilabel = task == "smear"
    probs, targets = predict(model, DataLoader(ds, args.batch_size), device, multilabel)
    metrics = compute_metrics(probs, targets, names, multilabel)
    out = args.checkpoint.parent
    suffix = args.split if args.fraction >= 1 else f"{args.split}_{args.fraction:g}"
    (out / f"metrics_{suffix}.json").write_text(json.dumps(metrics, indent=2))
    if "confusion_matrix" in metrics:
        plot_confusion(metrics["confusion_matrix"], names, out / f"confusion_{suffix}.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
