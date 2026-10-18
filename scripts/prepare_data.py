"""Bereitet die Rohdaten für die Klassifikation vor (einmalig, dauert einige Minuten).

    python scripts/prepare_data.py

Erzeugt in data/processed/:
    cells/<kategorie>/<image_id>_<nr>.png   jede annotierte Zelle ausgeschnitten
    cells.csv                               path, image_id, split, category, height, width
    smears/<image_id>.png                   ganzes Bild, längste Seite --smear-size px
    smears.csv                              path, image_id, split, n_cells, n_infected,
                                            infected, ring, trophozoite, schizont, gametocyte

Die Zellen vorab auszuschneiden spart beim Training das wiederholte Laden der
1600 x 1200 px grossen Originalbilder.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import PROCESSED_DIR, RAW_DIR, STAGES, assign_splits, crop_box, load_annotations  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=RAW_DIR, type=Path)
    parser.add_argument("--margin", default=0.15, type=float, help="Rand um jede Box (Anteil)")
    parser.add_argument("--smear-size", default=800, type=int, help="längste Seite der Ausstriche")
    args = parser.parse_args()

    ann = load_annotations(args.raw_dir)
    split = assign_splits(ann)
    ann["split"] = ann["image_id"].map(split)

    cells, smears = [], []
    for image_id, objs in tqdm(ann.groupby("image_id", sort=True), desc="Bilder"):
        image = Image.open(objs["image_path"].iloc[0]).convert("RGB")
        s = objs["split"].iloc[0]

        for k, o in enumerate(objs.itertuples()):
            rel = Path("cells") / o.category / f"{image_id}_{k:03d}.png"
            (PROCESSED_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
            crop_box(image, o.r0, o.c0, o.r1, o.c1, args.margin).save(PROCESSED_DIR / rel)
            cells.append({"path": rel.as_posix(), "image_id": image_id, "split": s,
                          "category": o.category, "height": o.r1 - o.r0, "width": o.c1 - o.c0})

        rel = Path("smears") / f"{image_id}.png"
        (PROCESSED_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
        small = image.copy()
        small.thumbnail((args.smear_size, args.smear_size))
        small.save(PROCESSED_DIR / rel)
        cats = set(objs["category"])
        row = {"path": rel.as_posix(), "image_id": image_id, "split": s,
               "n_cells": len(objs), "n_infected": int(objs["category"].isin(STAGES).sum())}
        row["infected"] = int(row["n_infected"] > 0)
        row.update({st: int(st in cats) for st in STAGES})
        smears.append(row)

    cells, smears = pd.DataFrame(cells), pd.DataFrame(smears)
    cells.to_csv(PROCESSED_DIR / "cells.csv", index=False)
    smears.to_csv(PROCESSED_DIR / "smears.csv", index=False)

    print("\nZellen pro Kategorie und Split:")
    print(pd.crosstab(cells["category"], cells["split"], margins=True))
    print("\nAusstriche: Anteil Bilder mit Stadium pro Split:")
    print(smears.groupby("split")[["infected", *STAGES]].mean().round(3))
    print(f"\nFertig: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
