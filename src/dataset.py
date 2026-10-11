"""NIH Malaria Dataset: Bilder finden, fester Split, PyTorch-Dataset.

Beispiel (im Notebook vorher `sys.path.append("..")`, damit `src` gefunden wird):

    from src.dataset import load_splits, MalariaDataset, default_transform

    splits = load_splits()
    train_ds = MalariaDataset(splits["train"], transform=default_transform(train=True))
    val_ds = MalariaDataset(splits["val"], transform=default_transform(train=False))
"""
from pathlib import Path

from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "cell_images"
CLASSES = ("Uninfected", "Parasitized")  # Label 0 / 1

# ImageNet-Statistik, passt für vortrainierte torchvision-Modelle
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def list_images(data_dir=DATA_DIR):
    """Alle (Pfad, Label)-Paare, sortiert, damit die Reihenfolge überall gleich ist."""
    data_dir = Path(data_dir)
    samples = []
    for label, cls in enumerate(CLASSES):
        folder = data_dir / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} fehlt. Zuerst: python scripts/download_data.py")
        samples += [(p, label) for p in sorted(folder.glob("*.png"))]
    return samples


def source_image(path):
    """Mikroskopbild, aus dem eine Zelle stammt, z. B. 'C33P1thinF_IMG_20150619_114756a'."""
    return Path(path).stem.split("_cell_")[0]


def load_splits(data_dir=DATA_DIR, val_size=0.15, test_size=0.15, seed=42):
    """Train/Val/Test-Split, für alle im Team identisch (fester Seed).

    Gruppiert wird nach Mikroskopbild: Zellen vom selben Bild landen im selben Split.
    Sonst sieht das Modell im Test fast dieselben Zellen wie im Training und die
    Accuracy ist zu optimistisch.
    """
    samples = list_images(data_dir)
    groups = [source_image(p) for p, _ in samples]

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(outer.split(samples, groups=groups))

    inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    tv_groups = [groups[i] for i in trainval_idx]
    train_rel, val_rel = next(inner.split(trainval_idx, groups=tv_groups))

    return {
        "train": [samples[trainval_idx[i]] for i in train_rel],
        "val": [samples[trainval_idx[i]] for i in val_rel],
        "test": [samples[i] for i in test_idx],
    }


def default_transform(img_size=128, train=False):
    """Zellbilder sind unterschiedlich gross (ca. 100-200 px), daher Resize."""
    steps = [transforms.Resize((img_size, img_size))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


class MalariaDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform or default_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
