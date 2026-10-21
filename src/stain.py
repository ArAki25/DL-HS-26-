"""Färbung und Kamera angleichen, bevor ein Bild ins Netz geht (Domain Shift).

Gemessen auf BBBC041: Training hat weissen Hintergrund (RGB ~237) und kräftig violette
Zellen, das Test-Set (andere Kamera, andere Ausstriche) grauen Hintergrund (~153) und
blasse, rosa Zellen. Zwei Schritte, für Training, Test und neue Bilder identisch:

1. white_balance: Hintergrund (hellste Pixel) jedes Bildes auf dasselbe Weiss skalieren
2. StainNormalizer: Farbe der Zellen (nicht des Hintergrunds) im LAB-Farbraum auf eine
   Referenz aus den Trainingsbildern bringen (Reinhard et al. 2001, nur Vordergrund,
   mit weichem Übergang und begrenzter Stärke, damit keine Artefakte entstehen)

    norm = StainNormalizer.fit(trainingsbilder)      # einmal, in prepare_data.py
    norm.save(path) / StainNormalizer.load(path)
    image = norm(image)                               # PIL -> PIL
"""
import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab

METHODS = ("stain", "whitebalance", "none")


def white_balance(image, target=235, percentile=80):
    """Jeden Farbkanal so skalieren, dass der Hintergrund `target` erreicht."""
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = a.sum(axis=2)
    background = a[luminance >= np.percentile(luminance, percentile)]
    scale = target / np.maximum(np.median(background, axis=0), 1)
    return Image.fromarray(np.clip(a * scale, 0, 255).astype(np.uint8))


def _lab_and_foreground(image):
    """LAB-Bild und Gewicht 0 (Hintergrund) ... 1 (sicher Zelle) pro Pixel."""
    lab = rgb2lab(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)
    lightness = lab[..., 0]
    background = np.percentile(lightness, 80)
    weight = np.clip((background - 3 - lightness) / 12, 0, 1)
    return lab, weight


def _foreground_stats(image):
    lab, weight = _lab_and_foreground(image)
    fg = lab[weight > 0.5]
    return fg.mean(0), fg.std(0)


class StainNormalizer:
    """method: 'stain' (Weissabgleich + Zellfarbe), 'whitebalance' (nur Hintergrund), 'none'."""

    MAX_SHIFT = np.array([15.0, 10.0, 10.0])  # maximale Verschiebung in L, a, b
    STD_RATIO = (0.67, 1.5)                   # maximale Streckung/Stauchung des Kontrasts

    def __init__(self, method="stain", ref_mean=None, ref_std=None):
        if method not in METHODS:
            raise ValueError(f"method muss einer von {METHODS} sein")
        self.method = method
        self.ref_mean = None if ref_mean is None else np.asarray(ref_mean, dtype=np.float64)
        self.ref_std = None if ref_std is None else np.asarray(ref_std, dtype=np.float64)

    @classmethod
    def fit(cls, images, method="stain"):
        """Referenz = Median der Vordergrund-Statistik über (verkleinerte) Trainingsbilder."""
        if method != "stain":
            return cls(method)
        stats = [_foreground_stats(white_balance(img)) for img in images]
        return cls(method, np.median([s[0] for s in stats], 0), np.median([s[1] for s in stats], 0))

    def __call__(self, image):
        if self.method == "none":
            return image.convert("RGB")
        image = white_balance(image)
        if self.method == "whitebalance":
            return image

        lab, weight = _lab_and_foreground(image)
        fg = lab[weight > 0.5]
        if len(fg) < 100:  # fast leeres Bild: nichts anzupassen
            return image
        mean, std = fg.mean(0), fg.std(0)
        ratio = np.clip(self.ref_std / np.maximum(std, 1e-3), *self.STD_RATIO)
        shift = np.clip(self.ref_mean - mean, -self.MAX_SHIFT, self.MAX_SHIFT)
        normed = (lab - mean) * ratio + mean + shift
        out = lab + weight[..., None] * (normed - lab)
        with warnings.catch_warnings():  # lab2rgb meldet geclippte Werte ausserhalb sRGB
            warnings.simplefilter("ignore")
            rgb = lab2rgb(out)
        return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

    def to_dict(self):
        return {"method": self.method,
                "ref_mean": None if self.ref_mean is None else self.ref_mean.tolist(),
                "ref_std": None if self.ref_std is None else self.ref_std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["method"], d.get("ref_mean"), d.get("ref_std"))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
