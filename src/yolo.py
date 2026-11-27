"""YOLO als Vergleich zur klassischen Segmentierung (ersetzt sie nicht).

Die Hauptpipeline findet Zellen ohne Objektdetektor (src/segmentation.py). Um zu messen,
ob ein gelernter Detektor das besser kann, wird hier ein YOLO-Datensatz aus denselben
Annotationen und mit *denselben* Vorverarbeitungsschritten erzeugt:

* gleicher Split (assign_splits, Seed 42) -> val/test sind identisch zur Hauptpipeline
* gleiche Farbnormalisierung (StainNormalizer aus data/processed/normalization.json)
* gleiche Skalierung auf eine feste Zellgrösse (estimate_cell_diameter, ohne Annotation),
  damit die unterschiedliche Vergrösserung der Test-Kamera (~135 px statt ~108 px)
  für YOLO genauso abgefangen wird wie für die Segmentierung

Zwei Modi:
    mode='cell'   1 Klasse "cell": reiner Lokalisierungs-Vergleich mit der Segmentierung
    mode='stage'  6 Klassen wie --task stage: End-to-End-Detektor (Lokalisierung + Stadium)

    from src.yolo import export_yolo_dataset, YoloDetector
    export_yolo_dataset("data/yolo/cell", mode="cell")
    det = YoloDetector("runs/yolo/<name>/weights/best.pt")
    boxes, conf, cls = det.detect(normalized_pil_image)

Benötigt `pip install -r requirements-yolo.txt` (ultralytics, AGPL-3.0).
"""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .data import (CELL_CLASSES, NORMALIZATION_FILE, ROOT, assign_splits, load_annotations,
                   subsample_images)
from .segmentation import estimate_cell_diameter
from .stain import StainNormalizer

YOLO_DIR = ROOT / "data" / "yolo"
MODES = {"cell": ("cell",), "stage": CELL_CLASSES}
META_FILE = "yolo_meta.json"
FALLBACK_DIAMETER = 110.0  # wie in prepare_data.py, falls die Schätzung scheitert


def load_normalizer(path=NORMALIZATION_FILE):
    """Dieselbe Normalisierung wie für das Zell-CNN; fehlt sie, wird nicht normalisiert."""
    path = Path(path)
    if path.exists():
        return StainNormalizer.load(path)
    print(f"Warnung: {path} fehlt (prepare_data.py nicht gelaufen?), YOLO ohne Normalisierung.")
    return StainNormalizer("none")


def rescale_factor(image, cell_px):
    """Faktor, mit dem das Bild skaliert wird, damit eine Zelle ~cell_px Pixel hat.

    cell_px=None: keine Skalierung (Ablation: lernt YOLO die Vergrösserung selbst?).
    """
    if not cell_px:
        return 1.0
    diameter = estimate_cell_diameter(np.asarray(image)) or FALLBACK_DIAMETER
    return cell_px / diameter


def _resize(image, factor):
    if factor == 1.0:
        return image
    return image.resize((round(image.width * factor), round(image.height * factor)), Image.LANCZOS)


def _yolo_lines(objs, cls_of, w, h, factor):
    """Boxen (r0, c0, r1, c1 im Original) -> 'cls cx cy bw bh', normiert auf [0, 1]."""
    lines = []
    for o in objs:
        k = cls_of(o["category"])
        if k is None:
            continue
        r0, c0, r1, c1 = (np.array([o["r0"], o["c0"], o["r1"], o["c1"]], float) * factor)
        r0, r1 = np.clip([r0, r1], 0, h)
        c0, c1 = np.clip([c0, c1], 0, w)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        lines.append(f"{k} {(c0 + c1) / 2 / w:.6f} {(r0 + r1) / 2 / h:.6f} "
                     f"{(c1 - c0) / w:.6f} {(r1 - r0) / h:.6f}")
    return lines


def export_yolo_dataset(out_dir=None, mode="cell", cell_px=48, normalizer=None,
                        splits=("train", "val", "test"), fraction=1.0, seed=42, raw_dir=None):
    """Schreibt <out_dir>/{images,labels}/<split>/ + data.yaml + yolo_meta.json.

    'difficult'-Zellen: im Modus 'cell' sind es trotzdem Zellen und bleiben als Label drin.
    Im Modus 'stage' gibt es keine Klasse dafür; sie werden weggelassen (in der Auswertung
    werden Treffer auf difficult-Zellen für *alle* Methoden ignoriert, siehe detection_eval).
    """
    names = MODES[mode]
    out_dir = Path(out_dir or YOLO_DIR / mode)
    normalizer = normalizer or load_normalizer()
    if mode == "cell":
        cls_of = lambda c: 0  # noqa: E731
    else:
        lookup = {c: i for i, c in enumerate(names)}
        cls_of = lookup.get

    ann = load_annotations(raw_dir) if raw_dir else load_annotations()
    ann["split"] = ann["image_id"].map(assign_splits(ann))
    if fraction < 1:  # gleiche Teilmengen-Logik wie train.py (nach Bildern, fester Seed)
        ann = subsample_images(ann, fraction, seed)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    stats = {}
    for split in splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        groups = ann[ann.split == split].groupby("image_id", sort=True)
        n_img = n_box = 0
        for image_id, objs in groups:
            image = normalizer(Image.open(objs.image_path.iloc[0]))
            factor = rescale_factor(image, cell_px)
            small = _resize(image, factor)
            small.save(out_dir / "images" / split / f"{image_id}.png")
            lines = _yolo_lines(objs.to_dict("records"), cls_of, small.width, small.height, factor)
            (out_dir / "labels" / split / f"{image_id}.txt").write_text("\n".join(lines))
            n_img, n_box = n_img + 1, n_box + len(lines)
        stats[split] = {"images": n_img, "boxes": n_box}

    yaml = [f"path: {out_dir.resolve().as_posix()}"]
    yaml += [f"{s}: images/{s}" for s in splits]
    yaml += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    (out_dir / "data.yaml").write_text("\n".join(yaml) + "\n")
    meta = {"mode": mode, "names": list(names), "cell_px": cell_px,
            "normalization": normalizer.to_dict(), "fraction": fraction, "seed": seed,
            "stats": stats}
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    return out_dir, meta


class YoloDetector:
    """Trainiertes YOLO + die Vorverarbeitung aus dem Export (yolo_meta.json).

    detect() erwartet ein bereits *normalisiertes* PIL-Bild in Originalgrösse (so wie es
    auch detect_cells bekommt) und gibt Boxen im Format r0, c0, r1, c1 im Original zurück.
    """

    def __init__(self, weights, meta=None, imgsz=None, device=None):
        from ultralytics import YOLO  # erst hier importieren: optionale Abhängigkeit

        weights = Path(weights)
        meta_path = Path(meta) if meta else _find_meta(weights)
        self.meta = json.loads(meta_path.read_text())
        self.names = self.meta["names"]
        self.model = YOLO(str(weights))
        self.imgsz = imgsz or self.meta.get("imgsz", 768)
        self.device = device

    def detect(self, image, conf=0.001, iou=0.6, max_det=1000):
        """-> boxes (N x 4 int), conf (N), cls (N). conf niedrig lassen, gefiltert wird später."""
        factor = rescale_factor(image, self.meta["cell_px"])
        small = _resize(image, factor)
        res = self.model.predict(small, imgsz=self.imgsz, conf=conf, iou=iou, max_det=max_det,
                                 device=self.device, verbose=False)[0]
        xyxy = res.boxes.xyxy.cpu().numpy().reshape(-1, 4) / factor
        boxes = xyxy[:, [1, 0, 3, 2]].round().astype(int)  # x0 y0 x1 y1 -> r0 c0 r1 c1
        return (boxes, res.boxes.conf.cpu().numpy().astype(float),
                res.boxes.cls.cpu().numpy().astype(int))


def _find_meta(weights):
    """yolo_meta.json liegt im Run-Ordner (von train_yolo.py kopiert): runs/yolo/<name>/."""
    for parent in weights.resolve().parents[:3]:
        if (parent / META_FILE).exists():
            return parent / META_FILE
    raise FileNotFoundError(f"{META_FILE} nicht neben {weights} gefunden; mit meta=... angeben.")
