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

# 4. Bilder normalisieren, Zellen ausschneiden (einmalig, ca. 10-20 Minuten)
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

Hauptmetrik ist **Macro-F1**, nicht Accuracy: ein Modell, das immer "rbc" sagt, hätte schon >95 % Accuracy. Auf dem offiziellen Test-Set fehlen Leukozyten ganz, dort deshalb `macro_f1_present` (Mittel nur über vorhandene Klassen) berichten.

## Domain Shift: andere Kamera, andere Färbung

Das offizielle Test-Set stammt aus anderen Ausstrichen und von einer anderen Kamera. Gemessen:

| | Train / Val | Test |
|---|---|---|
| Hintergrund (RGB) | ~237, fast weiss | ~153, grau-gelblich |
| Zellfarbe | kräftig violett | blass rosa |
| Durchmesser Erythrozyt | ~108 px | ~135 px |
| Format | PNG | JPG |

Gegenmassnahmen, auf drei Ebenen:

1. **Normalisierung** ([src/stain.py](src/stain.py), in `prepare_data.py`): Weissabgleich auf den Hintergrund, dann Zellfarbe im LAB-Farbraum auf eine Referenz aus Trainingsbildern (Reinhard, nur Vordergrund). Gleiche Funktion für Training, Test und neue Bilder; die Referenz wird im Checkpoint gespeichert, damit `predict.py` identisch vorgeht.
2. **Augmentation** ([src/augment.py](src/augment.py), `--augment domain`): zufällige Färbung (LAB), JPEG-Kompression, geringere Auflösung, Unschärfe, andere Vergrösserung. Das Netz soll lernen, dass nur die Form der Parasiten zählt.
3. **Skalierung**: Zellbilder werden aus den Boxen ausgeschnitten und auf 96 px gebracht, sind also unabhängig von der Vergrösserung. Für ganze Bilder und die Segmentierung wird der Zelldurchmesser pro Bild geschätzt ([src/segmentation.py](src/segmentation.py)) und das Bild auf gleiche Zellgrösse skaliert.

Ob es hilft, misst die **Lücke zwischen Val (gleiche Kamera) und Test (andere Kamera)**. Für den Bericht als Ablation:

```bash
python scripts/prepare_data.py --normalize none     # bzw. whitebalance / stain (Standard)
python scripts/train.py --task stage --augment basic    # bzw. domain (Standard) / none
python scripts/evaluate.py runs/<name>/best.pt --split test
```

`--normalize` wirkt auf die vorbereiteten Daten, nach einem Wechsel also neu trainieren.

## Training und Auswertung

Vor dem Start fragt `train.py`, wie viele Daten verwendet werden. Das spart beim Ausprobieren viel Zeit:

```
Wie viele Daten sollen verwendet werden? (Train und Val werden gleich verkleinert)
  [1] Schnelltest     5%  ~   51 Bilder, ~  3384 Samples, 3 Epochen vorgeschlagen
  [2] Klein          20%  ...
  [3] Mittel         50%  ...
  [4] Alles         100%  ...
  oder eigenen Prozentwert eingeben, z. B. 10%
Auswahl [1]:
Epochen [3]:
```

Mit `--fraction 0.1 --epochs 5` wird nicht gefragt (z. B. für Skripte oder Server). Die Teilmenge wird nach Bildern gezogen und ist mit gleichem `--seed` immer dieselbe, Läufe bleiben also vergleichbar. Für finale Resultate im Bericht immer `--fraction 1` verwenden.

```bash
python scripts/train.py --task stage --model simplecnn                     # mit Abfrage
python scripts/train.py --task stage --model simplecnn --fraction 0.05 --epochs 3   # Schnelltest ohne Abfrage
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
src/stain.py         Farbnormalisierung gegen Kamera-/Färbungsunterschiede
src/augment.py       Domain-Augmentationen (Färbung, JPEG, Auflösung)
src/models.py        eigenes CNN + torchvision-Modelle
src/engine.py        Trainingsschleife, Metriken
src/segmentation.py  Zellen finden ohne YOLO (Otsu + Watershed), Zellgrösse schätzen
scripts/             Download, Vorbereitung, Training, Auswertung, Vorhersage
notebooks/           Exploration und Experimente, Dateiname: <kürzel>_<thema>.ipynb
docs/                Notizen, Abbildungen, Bericht
runs/                Trainingsergebnisse (lokal, nicht im Repo)
```

## Zusammenarbeit

Siehe [CONTRIBUTING.md](CONTRIBUTING.md). Kurz: nie direkt auf `main` pushen, pro Aufgabe ein Branch, Änderungen per Pull Request mit einem Review.
