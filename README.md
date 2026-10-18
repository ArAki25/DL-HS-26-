# Malaria-Diagnose aus Blutausstrichen (P. vivax)

Deep-Learning-Projekt (FHNW, BSc Business Artificial Intelligence).

**Problem:** Infizierte Blutzellen auf Mikroskopbildern erkennen und das Parasitenstadium bestimmen, für Regionen ohne Fachpersonal.
**DL-Kern:** Eigene CNNs zur Klassifikation von Zellen (Stadium) und ganzen Ausstrichen, Vergleich mit vortrainierten Modellen, kleines Modell für Smartphone-Mikroskope (Knowledge Distillation). Zellen werden ohne Objektdetektor (kein YOLO) gefunden: klassische Segmentierung, danach Klassifikation.
**Schwierig:** Extremes Klassen-Ungleichgewicht (>95 % gesunde Erythrozyten, nur wenige Gametozyten/Schizonten) und Domain Shift: das offizielle Test-Set stammt aus anderen Ausstrichen.
**Daten:** [P. vivax (malaria) infected human blood smears](https://www.kaggle.com/datasets/orvile/p-vivax-malaria-infected-human-blood-smears) (= [BBBC041](https://bbbc.broadinstitute.org/bbbc041), Lizenz CC BY-NC-SA 3.0). 1'328 Giemsa-gefärbte Mikroskopbilder (ca. 1600 x 1200 px), ca. 80'000 Zellen mit Box und Kategorie.

| Kategorie | Bedeutung |
|---|---|
| `rbc` | gesunder Erythrozyt |
| `leukocyte` | weisses Blutkörperchen |
| `ring`, `trophozoite`, `schizont`, `gametocyte` | infiziert, Entwicklungsstadien des Parasiten |
| `difficult` | von Annotatoren unklar, wird standardmässig weggelassen |

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
python scripts/download_data.py       # ca. 2.3 GB nach data/raw/p_vivax/

# 4. Zellen ausschneiden und Bilder verkleinern (einmalig, einige Minuten)
python scripts/prepare_data.py        # -> data/processed/
```

Den Kaggle-Key holst du auf kaggle.com unter Settings → API → "Create New Token". Jeder nutzt seinen eigenen Key, die `.env` wird nie committet.

**Ohne Kaggle-Key:** Das ZIP auf der Kaggle-Seite über "Download" holen, unverändert nach `data/raw/` legen und `python scripts/download_data.py` ausführen. Das Skript entpackt es nach `data/raw/p_vivax/`, danach kann das ZIP gelöscht werden.

## Ansatz

```
Mikroskopbild ──► Annotationen (Boxen) ──► Zellen ausschneiden ──► Zell-CNN ──► Stadium / infiziert
      │                                                                 ▲
      │           (bei neuen Bildern ohne Annotation)                   │
      ├──────► Segmentierung (Otsu + Watershed) ──► Boxen ──────────────┘
      │
      └──────► verkleinertes Gesamtbild ──► Ausstrich-CNN ──► welche Stadien kommen vor (Multi-Label)
```

Drei Klassifikationsaufgaben (`--task`):

| Task | Eingabe | Ziel | Loss |
|---|---|---|---|
| `stage` | Zelle, 96 px | 6 Klassen (rbc, leukocyte, ring, trophozoite, schizont, gametocyte) | Cross-Entropy |
| `binary` | Zelle, 96 px | uninfected / infected | Cross-Entropy |
| `smear` | ganzes Bild, 384 px | Multi-Label über die 4 Stadien | BCE mit `pos_weight` |

Modelle (`--model`): `simplecnn` (eigenes CNN, von Grund auf trainiert, [src/models.py](src/models.py)), `simplecnn-deep` (mehr Blöcke, für ganze Bilder), `resnet18` und `mobilenet_v3_small` als Vergleich (mit `--pretrained` ImageNet-Gewichte).

Hauptmetrik ist **Macro-F1**, nicht Accuracy: ein Modell, das immer "rbc" sagt, hätte schon >95 % Accuracy.

## Training und Auswertung

```bash
python scripts/train.py --task stage --model simplecnn --epochs 20
python scripts/train.py --task binary --model resnet18 --pretrained
python scripts/train.py --task smear --model simplecnn-deep --batch-size 16

python scripts/evaluate.py runs/<name>/best.pt --split val    # Metriken + Konfusionsmatrix
python scripts/evaluate.py runs/<name>/best.pt --split test   # erst ganz am Schluss!

python scripts/evaluate_segmentation.py --split val --show 3  # findet die Segmentierung die Zellen?
python scripts/predict.py <bild.png> runs/<name>/best.pt      # ganze Pipeline auf einem neuen Bild
```

Jeder Lauf landet in `runs/<datum>_<task>_<modell>/` (`best.pt`, `history.json`, `config.json`, `metrics_val.json`). `runs/` ist nicht im Repo.

## Daten in Notebooks verwenden

Alle nutzen denselben Split aus [src/data.py](src/data.py): Test = offizielles `test.json` (120 Bilder), Validierung = 15 % der Bilder aus `training.json` (Seed 42). Zellen vom selben Bild landen immer im selben Split.

```python
import sys; sys.path.append("..")     # im Notebook, damit src gefunden wird
from torch.utils.data import DataLoader
from src.data import load_cells, CellDataset, cell_transform, balanced_sampler

train_ds = CellDataset(load_cells("train"), task="stage", transform=cell_transform(train=True))
val_ds = CellDataset(load_cells("val"), task="stage")
train_loader = DataLoader(train_ds, batch_size=128, sampler=balanced_sampler(train_ds.labels))
```

`load_cells()` und `load_smears()` geben pandas-DataFrames zurück (gut für Exploration), `load_annotations()` die rohen Boxen.

## Ordner

```
data/raw/p_vivax/    Rohdaten von Kaggle (lokal, nicht im Repo)
data/processed/      ausgeschnittene Zellen, verkleinerte Bilder, cells.csv, smears.csv
src/data.py          Annotationen, Split, Datasets, Transforms, Ungleichgewicht
src/models.py        eigenes CNN + torchvision-Modelle
src/engine.py        Trainingsschleife, Metriken
src/segmentation.py  Zellen finden ohne YOLO (Otsu + Watershed)
scripts/             Download, Vorbereitung, Training, Auswertung, Vorhersage
notebooks/           Exploration und Experimente, Dateiname: <kürzel>_<thema>.ipynb
docs/                Notizen, Abbildungen, Bericht
runs/                Trainingsergebnisse (lokal, nicht im Repo)
```

## Zusammenarbeit

Siehe [CONTRIBUTING.md](CONTRIBUTING.md). Kurz: nie direkt auf `main` pushen, pro Aufgabe ein Branch, Änderungen per Pull Request mit einem Review.
