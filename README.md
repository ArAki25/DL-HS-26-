# Malaria-Diagnose aus Blutausstrichen

Deep-Learning-Projekt (FHNW, BSc Business Artificial Intelligence).

**Problem:** Infizierte Blutzellen auf Mikroskopbildern erkennen, für Regionen ohne Fachpersonal.
**DL-Kern:** Zellsegmentierung plus Klassifikation, komprimiert für Smartphone-Mikroskope (Knowledge Distillation).
**Schwierig:** Unterschiedliche Färbungen und Mikroskope verursachen Domain Shift.
**Daten:** [NIH Malaria Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria), ca. 27'000 Zellbilder (`Parasitized` / `Uninfected`).

## Team

| Name | Zuständig für |
|---|---|
| Jhonny | |
| | |
| | |

## Setup (einmalig pro Rechner)

```bash
git clone https://github.com/ArAki25/DL-HS-26-.git
cd DL-HS-26-

# 1. Python-Umgebung (.venv) erstellen und Pakete installieren
bash scripts/setup_venv.sh            # Mac / Linux
# Windows: powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1

# 2. Umgebung aktivieren (bei jeder Arbeitssession)
source .venv/bin/activate             # Windows: .\.venv\Scripts\Activate.ps1

# 3. Kaggle-Zugang eintragen und Daten laden
cp .env.example .env                  # Windows: copy .env.example .env, dann KAGGLE_USERNAME und KAGGLE_KEY ausfüllen
python scripts/download_data.py       # ca. 700 MB nach data/raw/cell_images/
```

Den Kaggle-Key holst du auf kaggle.com unter Settings → API → "Create New Token". Jeder nutzt seinen eigenen Key, die `.env` wird nie committet.

## Daten verwenden

Alle nutzen denselben Split aus [src/dataset.py](src/dataset.py) (70 / 15 / 15, Seed 42). Zellen vom selben Mikroskopbild landen immer im selben Split, damit der Test nicht durch fast identische Zellen geschönt wird.

```python
import sys; sys.path.append("..")     # im Notebook, damit src gefunden wird
from torch.utils.data import DataLoader
from src.dataset import load_splits, MalariaDataset, default_transform

splits = load_splits()
train_ds = MalariaDataset(splits["train"], transform=default_transform(train=True))
val_ds = MalariaDataset(splits["val"])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
```

Label 0 = `Uninfected`, 1 = `Parasitized`. Das Test-Set erst ganz am Schluss für die finale Auswertung anfassen.

## Ordner

```
data/         Rohdaten (lokal, nicht im Repo)
notebooks/    Exploration und Experimente, Dateiname: <kürzel>_<thema>.ipynb
src/          wiederverwendbarer Code (Datasets, Modelle, Training)
docs/         Notizen, Abbildungen, Bericht
scripts/      Setup- und Hilfsskripte
```

## Zusammenarbeit

Siehe [CONTRIBUTING.md](CONTRIBUTING.md). Kurz: nie direkt auf `main` pushen, pro Aufgabe ein Branch, Änderungen per Pull Request mit einem Review.
