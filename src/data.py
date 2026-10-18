"""P. vivax Malaria Dataset (BBBC041, Kaggle: orvile/p-vivax-malaria-infected-human-blood-smears).

Die Rohdaten sind ganze Mikroskopbilder (ca. 1600 x 1200 px) mit einer Bounding-Box und
einer Kategorie pro Zelle (training.json / test.json). Wir verwenden die Boxen nicht für
einen Objektdetektor (kein YOLO), sondern für zwei Klassifikationsaufgaben:

* Zellen: jede Box wird ausgeschnitten und einzeln klassifiziert (Stadium oder infiziert ja/nein)
* Ausstriche: ganzes Bild, Multi-Label "welche Parasitenstadien kommen vor"

Ablauf: scripts/download_data.py -> scripts/prepare_data.py -> scripts/train.py

Beispiel (im Notebook vorher `sys.path.append("..")`, damit `src` gefunden wird):

    from src.data import load_cells, CellDataset, cell_transform

    train_df = load_cells("train")
    train_ds = CellDataset(train_df, task="stage", transform=cell_transform(train=True))
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
RAW_DIR = ROOT / os.getenv("DATA_DIR", "data/raw") / "p_vivax"
PROCESSED_DIR = ROOT / "data" / "processed"

# Kategorie im JSON -> kurzer Name (auch Ordnername der ausgeschnittenen Zellen)
CATEGORIES = {
    "red blood cell": "rbc",
    "leukocyte": "leukocyte",
    "ring": "ring",
    "trophozoite": "trophozoite",
    "schizont": "schizont",
    "gametocyte": "gametocyte",
    "difficult": "difficult",
}
CELL_CLASSES = ("rbc", "leukocyte", "ring", "trophozoite", "schizont", "gametocyte")
STAGES = ("ring", "trophozoite", "schizont", "gametocyte")  # = infizierte Zellen
BINARY_CLASSES = ("uninfected", "infected")

# ImageNet-Statistik, passt auch für vortrainierte torchvision-Modelle
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- Rohdaten

def find_annotation_files(raw_dir=RAW_DIR):
    """{'training': Path, 'test': Path}. Die Ordnertiefe im Kaggle-Archiv ist nicht fix."""
    raw_dir = Path(raw_dir)
    files = {}
    for split in ("training", "test"):
        hits = sorted(raw_dir.rglob(f"{split}.json"))
        if not hits:
            raise FileNotFoundError(f"{split}.json fehlt unter {raw_dir}. "
                                    "Zuerst: python scripts/download_data.py")
        files[split] = hits[0]
    return files


def _resolve_image(json_file, pathname, index):
    """pathname im JSON ist z. B. '/images/<uuid>.png', relativ zum Ordner des JSON."""
    candidate = json_file.parent / pathname.lstrip("/")
    if candidate.exists():
        return candidate
    name = Path(pathname).name
    if not index:  # Fallback: alle Bilder einmal indexieren
        for p in json_file.parent.rglob("*"):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                index[p.name] = p
    if name not in index:
        raise FileNotFoundError(f"Bild {pathname} aus {json_file.name} nicht gefunden")
    return index[name]


def load_annotations(raw_dir=RAW_DIR):
    """Eine Zeile pro annotierter Zelle.

    Spalten: image_id, official_split ('training'/'test'), image_path, img_h, img_w,
    category (kurzer Name aus CATEGORIES), r0, c0, r1, c1 (Box in Pixeln, Zeile/Spalte).
    """
    rows, index = [], {}
    for split, json_file in find_annotation_files(raw_dir).items():
        for entry in json.loads(json_file.read_text()):
            img = entry["image"]
            path = _resolve_image(json_file, img["pathname"], index)
            shape = img.get("shape", {})
            for obj in entry["objects"]:
                box = obj["bounding_box"]
                rows.append({
                    "image_id": path.stem,
                    "official_split": split,
                    "image_path": str(path),
                    "img_h": shape.get("r"),
                    "img_w": shape.get("c"),
                    "category": CATEGORIES.get(obj["category"], obj["category"]),
                    "r0": box["minimum"]["r"], "c0": box["minimum"]["c"],
                    "r1": box["maximum"]["r"], "c1": box["maximum"]["c"],
                })
    return pd.DataFrame(rows)


def assign_splits(annotations, val_size=0.15, seed=42):
    """image_id -> 'train' / 'val' / 'test', für alle im Team identisch (fester Seed).

    Test = offizielles test.json (stammt aus anderen Ausstrichen -> realistischer Test).
    Val wird nach Bild aus training.json gezogen: alle Zellen eines Bildes landen im
    selben Split, sonst sieht das Modell in der Validierung fast dieselben Zellen.
    """
    images = annotations.drop_duplicates("image_id")[["image_id", "official_split"]]
    train_ids = images.loc[images.official_split == "training", "image_id"].to_numpy()
    test_ids = images.loc[images.official_split == "test", "image_id"].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr, va = next(gss.split(train_ids, groups=train_ids))
    split = {i: "train" for i in train_ids[tr]}
    split.update({i: "val" for i in train_ids[va]})
    split.update({i: "test" for i in test_ids})
    return split


def crop_box(image, r0, c0, r1, c1, margin=0.15):
    """Box mit etwas Rand ausschneiden (Kontext hilft bei Ringen am Zellrand)."""
    h, w = r1 - r0, c1 - c0
    pr, pc = int(h * margin), int(w * margin)
    return image.crop((max(c0 - pc, 0), max(r0 - pr, 0),
                       min(c1 + pc, image.width), min(r1 + pr, image.height)))


# --------------------------------------------------------------------------- vorbereitete Daten

def _read_csv(name, split):
    csv = PROCESSED_DIR / name
    if not csv.exists():
        raise FileNotFoundError(f"{csv} fehlt. Zuerst: python scripts/prepare_data.py")
    df = pd.read_csv(csv)
    df["path"] = [str(PROCESSED_DIR / p) for p in df["path"]]
    if split is not None:
        df = df[df.split == split].reset_index(drop=True)
    return df


def load_cells(split=None, include_difficult=False):
    """Ausgeschnittene Zellen (cells.csv). split: 'train' / 'val' / 'test' / None = alle."""
    df = _read_csv("cells.csv", split)
    if not include_difficult:
        df = df[df.category != "difficult"].reset_index(drop=True)
    return df


def load_smears(split=None):
    """Ganze Ausstriche, verkleinert (smears.csv), mit Spalte pro Stadium (0/1)."""
    return _read_csv("smears.csv", split)


def subsample_images(df, fraction, seed=42):
    """Nur einen Teil der Bilder behalten (für schnelle Tests), mit allen ihren Zellen.

    Gezogen wird nach image_id, damit Zellen eines Bildes zusammenbleiben. Mit gleichem
    Seed ist die Auswahl immer dieselbe, Läufe bleiben also vergleichbar.
    """
    if fraction >= 1:
        return df
    ids = np.sort(df["image_id"].unique())
    n = max(1, round(len(ids) * fraction))
    keep = np.random.default_rng(seed).choice(ids, size=n, replace=False)
    return df[df["image_id"].isin(keep)].reset_index(drop=True)


def cell_labels(df, task="stage"):
    """task='stage': Index in CELL_CLASSES, task='binary': 1 = infiziert."""
    if task == "stage":
        lookup = {c: i for i, c in enumerate(CELL_CLASSES)}
        return df["category"].map(lookup).to_numpy()
    if task == "binary":
        return df["category"].isin(STAGES).astype(int).to_numpy()
    raise ValueError(f"Unbekannter task: {task}")


def class_names(task):
    return {"stage": CELL_CLASSES, "binary": BINARY_CLASSES, "smear": STAGES}[task]


# --------------------------------------------------------------------------- Transforms

def cell_transform(img_size=96, train=False):
    """Zellen sind ca. 80-150 px gross. ColorJitter simuliert unterschiedliche Färbungen."""
    steps = [transforms.Resize((img_size, img_size))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.02),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def smear_transform(img_size=384, train=False):
    """Ganze Bilder werden quadratisch skaliert (leichte Verzerrung 4:3 -> 1:1)."""
    steps = [transforms.Resize((img_size, img_size))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.02),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


# --------------------------------------------------------------------------- Datasets

class CellDataset(Dataset):
    """Eine Zelle pro Sample. Label je nach task (siehe cell_labels)."""

    def __init__(self, df, task="stage", transform=None):
        self.paths = df["path"].tolist()
        self.labels = cell_labels(df, task)
        self.transform = transform or cell_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), int(self.labels[idx])


class SmearDataset(Dataset):
    """Ein ganzer Ausstrich pro Sample, Ziel = Multi-Hot-Vektor über STAGES."""

    def __init__(self, df, transform=None):
        self.paths = df["path"].tolist()
        self.targets = df[list(STAGES)].to_numpy(dtype=np.float32)
        self.labels = self.targets  # gleiche Schnittstelle wie CellDataset
        self.transform = transform or smear_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), torch.from_numpy(self.targets[idx])


# --------------------------------------------------------------------------- Ungleichgewicht

def class_weights(labels, num_classes):
    """Inverse Häufigkeit, normiert auf Mittelwert 1 (für CrossEntropyLoss(weight=...))."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def balanced_sampler(labels, power=0.5):
    """Zieht seltene Klassen öfter. power=1: voll ausbalanciert, 0: wie im Datensatz.

    >95 % der Zellen sind gesunde Erythrozyten; power=0.5 ist ein Kompromiss, bei dem
    das Modell seltene Stadien oft sieht, ohne dieselben paar Gametozyten auswendig zu lernen.
    """
    counts = np.bincount(labels)
    weights = (1.0 / np.maximum(counts, 1)) ** power
    return WeightedRandomSampler(weights[labels], num_samples=len(labels), replacement=True)
