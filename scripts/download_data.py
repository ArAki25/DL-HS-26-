"""Lädt das NIH Malaria Dataset von Kaggle nach data/raw/ (Windows, Mac, Linux).

    python scripts/download_data.py

Zugangsdaten kommen aus .env (siehe .env.example) oder ~/.kaggle/kaggle.json.
"""
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

DATASET = "iarunava/cell-images-for-detecting-malaria"
ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

if not os.getenv("KAGGLE_KEY") and not (Path.home() / ".kaggle" / "kaggle.json").exists():
    sys.exit("Keine Kaggle-Zugangsdaten gefunden. .env.example nach .env kopieren "
             "und KAGGLE_USERNAME / KAGGLE_KEY ausfüllen.")

# Erst nach load_dotenv importieren: kaggle authentifiziert sich beim Import.
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

dest = ROOT / os.getenv("DATA_DIR", "data/raw")
images = dest / "cell_images"

if (images / "Parasitized").is_dir() and (images / "Uninfected").is_dir():
    print(f"Daten sind schon da: {images}")
    sys.exit(0)

dest.mkdir(parents=True, exist_ok=True)
api = KaggleApi()
api.authenticate()
print(f"Lade {DATASET} (ca. 700 MB) ...")
api.dataset_download_files(DATASET, path=str(dest), unzip=True, quiet=False)

# Das Kaggle-Archiv enthält alle Bilder ein zweites Mal in cell_images/cell_images/.
duplicate = images / "cell_images"
if duplicate.is_dir():
    shutil.rmtree(duplicate)

for cls in ("Parasitized", "Uninfected"):
    n = len(list((images / cls).glob("*.png")))
    print(f"{cls:12s} {n:6d} Bilder")
print(f"Fertig. Bilder liegen in {images}")
