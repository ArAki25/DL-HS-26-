"""Wie gut findet die klassische Segmentierung (src/segmentation.py) die annotierten Zellen?

    python scripts/evaluate_segmentation.py --split val
    python scripts/evaluate_segmentation.py --split val --show 3    # Beispiele plotten

Eine gefundene Box zählt als Treffer, wenn sie eine annotierte Box mit IoU >= 0.5 überdeckt.
Wichtig ist vor allem der Recall bei infizierten Zellen: was die Segmentierung verpasst,
kann kein Klassifikator mehr retten.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import STAGES, assign_splits, load_annotations  # noqa: E402
from src.segmentation import detect_cells, match_boxes  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--cell-diameter", type=int, default=None,
                   help="Default: pro Bild automatisch geschätzt")
    p.add_argument("--limit", type=int, default=None, help="nur die ersten N Bilder")
    p.add_argument("--show", type=int, default=0, help="N Beispielbilder nach docs/ speichern")
    args = p.parse_args()

    ann = load_annotations()
    ann["split"] = ann["image_id"].map(assign_splits(ann))
    ann = ann[(ann.split == args.split) & (ann.category != "difficult")]
    groups = list(ann.groupby("image_id"))[:args.limit]

    tp = fp = fn = 0
    inf_found = inf_total = 0
    for k, (image_id, objs) in enumerate(tqdm(groups, desc="Bilder")):
        image = np.asarray(Image.open(objs.image_path.iloc[0]).convert("RGB"))
        gt = objs[["r0", "c0", "r1", "c1"]].to_numpy()
        pred = detect_cells(image, cell_diameter=args.cell_diameter)
        pairs, t, f, n = match_boxes(pred, gt, args.iou)
        tp, fp, fn = tp + t, fp + f, fn + n

        infected = objs.category.isin(STAGES).to_numpy()
        inf_total += infected.sum()
        inf_found += sum(infected[g] for _, g in pairs)

        if k < args.show:
            fig, ax = plt.subplots(figsize=(10, 7.5))
            ax.imshow(image)
            for boxes, color in ((gt, "lime"), (pred, "red")):
                for r0, c0, r1, c1 in boxes:
                    ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec=color, lw=1))
            ax.set_title(f"{image_id}: grün = Annotation, rot = Segmentierung")
            ax.axis("off")
            out = ROOT / "docs" / f"segmentation_{image_id}.png"
            fig.savefig(out, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"Gespeichert: {out}")

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"\n{len(groups)} Bilder, IoU >= {args.iou}")
    print(f"Precision {precision:.3f}  Recall {recall:.3f}  "
          f"F1 {2 * precision * recall / max(precision + recall, 1e-9):.3f}")
    print(f"Recall infizierte Zellen: {inf_found / max(inf_total, 1):.3f} ({inf_found}/{inf_total})")


if __name__ == "__main__":
    main()
