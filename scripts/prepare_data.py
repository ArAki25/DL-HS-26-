"""Bereitet die Rohdaten für die Klassifikation vor (einmalig, ca. 10-20 Minuten).

    python scripts/prepare_data.py                      # mit Farbnormalisierung (Standard)
    python scripts/prepare_data.py --normalize none     # ohne, für den Vergleich im Bericht

Gegen Domain Shift (andere Kamera / Färbung im Test-Set) wird jedes Bild vorher
angeglichen, siehe src/stain.py:
    --normalize stain         Weissabgleich + Zellfarbe auf Referenz (Standard)
    --normalize whitebalance  nur Hintergrund auf Weiss
    --normalize none          Rohbilder

Erzeugt in data/processed/:
    normalization.json                      Methode + Referenzfarbe (auch für predict.py)
    cells/<kategorie>/<image_id>_<nr>.png   jede annotierte Zelle ausgeschnitten
    cells.csv                               path, image_id, split, category, height, width
    smears/<image_id>.png                   ganzes Bild, skaliert auf gleiche Zellgrösse
    smears.csv                              path, image_id, split, cell_diameter, n_cells,
                                            n_infected, infected, ring, trophozoite, ...

Wer die Methode wechselt, muss danach neu trainieren.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (NORMALIZATION_FILE, PROCESSED_DIR, RAW_DIR, STAGES,  # noqa: E402
                      assign_splits, crop_box, load_annotations)
from src.segmentation import estimate_cell_diameter  # noqa: E402
from src.stain import METHODS, StainNormalizer  # noqa: E402


def process_image(job):
    """Ein Mikroskopbild: normalisieren, Zellen ausschneiden, verkleinerte Version speichern."""
    image_id, objs, normalizer, margin, smear_cell_px = job
    image = normalizer(Image.open(objs[0]["image_path"]))
    split = objs[0]["split"]

    cells = []
    for k, o in enumerate(objs):
        rel = Path("cells") / o["category"] / f"{image_id}_{k:03d}.png"
        (PROCESSED_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
        crop_box(image, o["r0"], o["c0"], o["r1"], o["c1"], margin).save(PROCESSED_DIR / rel)
        cells.append({"path": rel.as_posix(), "image_id": image_id, "split": split,
                      "category": o["category"], "height": o["r1"] - o["r0"],
                      "width": o["c1"] - o["c0"]})

    # Andere Kamera = andere Vergrösserung. Ganze Bilder so skalieren, dass eine Zelle
    # überall gleich viele Pixel hat; der Durchmesser wird ohne Annotation geschätzt,
    # damit predict.py auf neuen Bildern genau gleich vorgehen kann.
    diameter = estimate_cell_diameter(image) or 110.0
    factor = smear_cell_px / diameter
    small = image.resize((round(image.width * factor), round(image.height * factor)), Image.LANCZOS)
    rel = Path("smears") / f"{image_id}.png"
    (PROCESSED_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
    small.save(PROCESSED_DIR / rel)

    categories = {o["category"] for o in objs}
    n_infected = sum(o["category"] in STAGES for o in objs)
    smear = {"path": rel.as_posix(), "image_id": image_id, "split": split,
             "cell_diameter": round(diameter, 1), "n_cells": len(objs),
             "n_infected": n_infected, "infected": int(n_infected > 0),
             **{st: int(st in categories) for st in STAGES}}
    return cells, smear


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=RAW_DIR, type=Path)
    parser.add_argument("--normalize", choices=METHODS, default="stain")
    parser.add_argument("--margin", default=0.15, type=float, help="Rand um jede Box (Anteil)")
    parser.add_argument("--smear-cell-px", default=54, type=int,
                        help="Zelldurchmesser in den verkleinerten Ausstrichen (54 -> ca. 800 px breit)")
    parser.add_argument("--workers", default=max(1, (os.cpu_count() or 2) - 1), type=int)
    args = parser.parse_args()

    ann = load_annotations(args.raw_dir)
    ann["split"] = ann["image_id"].map(assign_splits(ann))

    # Referenzfarbe nur aus Trainingsbildern (nie aus val/test), verkleinert = schnell
    train_paths = ann.loc[ann.split == "train", "image_path"].drop_duplicates()
    sample = train_paths.sample(min(50, len(train_paths)), random_state=0)
    print(f"Normalisierung '{args.normalize}': Referenz aus {len(sample)} Trainingsbildern ...")
    normalizer = StainNormalizer.fit((Image.open(p).convert("RGB").reduce(4) for p in sample),
                                     args.normalize)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    normalizer.save(NORMALIZATION_FILE)

    jobs = [(image_id, objs.to_dict("records"), normalizer, args.margin, args.smear_cell_px)
            for image_id, objs in ann.groupby("image_id", sort=True)]
    cells, smears = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for c, s in tqdm(pool.map(process_image, jobs, chunksize=4), total=len(jobs), desc="Bilder"):
            cells += c
            smears.append(s)

    cells, smears = pd.DataFrame(cells), pd.DataFrame(smears)
    cells.to_csv(PROCESSED_DIR / "cells.csv", index=False)
    smears.to_csv(PROCESSED_DIR / "smears.csv", index=False)

    print("\nZellen pro Kategorie und Split:")
    print(pd.crosstab(cells["category"], cells["split"], margins=True))
    print("\nAusstriche: Anteil Bilder mit Stadium pro Split:")
    print(smears.groupby("split")[["infected", *STAGES]].mean().round(3))
    print("\nGeschätzter Zelldurchmesser (px) pro Split, vor dem Skalieren:")
    print(smears.groupby("split")["cell_diameter"].describe()[["mean", "min", "max"]].round(1))
    print(f"\nFertig: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
