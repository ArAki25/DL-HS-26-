"""Lädt das P. vivax Malaria Dataset (BBBC041) von Kaggle nach data/raw/p_vivax/.

    python scripts/download_data.py

Variante A: ZIP manuell von Kaggle herunterladen ("Download" auf der Datensatz-Seite)
und nach data/raw/ legen. Das Skript entpackt es, ein Kaggle-Key ist dann nicht nötig.
Variante B: automatischer Download, Zugangsdaten aus .env (siehe .env.example)
oder ~/.kaggle/kaggle.json.
"""
import os
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

DATASET = "orvile/p-vivax-malaria-infected-human-blood-smears"
ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")
dest = ROOT / os.getenv("DATA_DIR", "data/raw") / "p_vivax"


def is_ready():
    return bool(list(dest.rglob("training.json")) and list(dest.rglob("test.json")))


def summary():
    for name in ("training.json", "test.json"):
        hits = list(dest.rglob(name))
        print(f"{name:14s} {hits[0] if hits else 'FEHLT'}")
    n = sum(1 for p in dest.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    print(f"{n} Bilder. Weiter mit: python scripts/prepare_data.py")


if is_ready():
    print(f"Daten sind schon da: {dest}")
    sys.exit(0)

# Variante A: manuell heruntergeladenes ZIP in data/raw/ oder data/raw/p_vivax/
zips = sorted({*dest.parent.glob("*.zip"), *dest.glob("*.zip")} if dest.is_dir()
              else dest.parent.glob("*.zip"))
if zips:
    dest.mkdir(parents=True, exist_ok=True)
    for z in zips:
        print(f"Entpacke {z.name} nach {dest} ...")
        with zipfile.ZipFile(z) as archive:
            archive.extractall(dest)
    if not is_ready():
        sys.exit("ZIP entpackt, aber training.json / test.json nicht gefunden. "
                 "Ist es das richtige Kaggle-Dataset?")
    summary()
    print("Das ZIP kann jetzt gelöscht werden.")
    sys.exit(0)

# Variante B: Download über die Kaggle-API
if not os.getenv("KAGGLE_KEY") and not (Path.home() / ".kaggle" / "kaggle.json").exists():
    sys.exit("Kein ZIP in data/raw/ und keine Kaggle-Zugangsdaten gefunden.\n"
             "Entweder das ZIP von Kaggle nach data/raw/ legen oder .env.example nach .env "
             "kopieren und KAGGLE_USERNAME / KAGGLE_KEY ausfüllen.")

# Erst nach load_dotenv importieren: kaggle authentifiziert sich beim Import.
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

dest.mkdir(parents=True, exist_ok=True)
api = KaggleApi()
api.authenticate()
print(f"Lade {DATASET} (ca. 2.3 GB) ...")
api.dataset_download_files(DATASET, path=str(dest), unzip=True, quiet=False)
summary()
