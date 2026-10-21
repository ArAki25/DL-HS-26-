"""Augmentationen gegen Domain Shift: das Netz soll Färbung, Kamera und Bildqualität
ignorieren und nur auf die Form der Parasiten schauen.

Jede Klasse simuliert einen konkreten Unterschied zwischen Training und Test:

    StainJitter     andere Färbeintensität / Farbton (zufällig im LAB-Farbraum)
    RandomJPEG      Test-Bilder sind JPG, Training PNG -> Kompressionsartefakte
    RandomDownscale andere Kamera-Auflösung / Optik (Details gehen verloren)
    (in data.py)    RandomResizedCrop für andere Vergrösserung, GaussianBlur für Fokus
"""
import io
import random
import warnings

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab


class StainJitter:
    """Mittelwert und Kontrast jedes LAB-Kanals zufällig verschieben/strecken.

    Gleicher Farbraum wie src/stain.py: die Normalisierung bringt Bilder in die Nähe
    der Referenz, StainJitter sorgt dafür, dass der verbleibende Rest das Netz nicht stört.
    """

    def __init__(self, shift=(8.0, 6.0, 6.0), scale=0.2, p=0.8):
        self.shift, self.scale, self.p = np.asarray(shift), scale, p

    def __call__(self, image):
        if random.random() > self.p:
            return image
        lab = rgb2lab(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)
        mean = lab.reshape(-1, 3).mean(0)
        factor = np.random.uniform(1 - self.scale, 1 + self.scale, 3)
        offset = np.random.uniform(-self.shift, self.shift)
        lab = (lab - mean) * factor + mean + offset
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rgb = lab2rgb(lab)
        return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))


class RandomJPEG:
    def __init__(self, quality=(40, 95), p=0.5):
        self.quality, self.p = quality, p

    def __call__(self, image):
        if random.random() > self.p:
            return image
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=random.randint(*self.quality))
        return Image.open(buffer).convert("RGB")


class RandomDownscale:
    """Verkleinern und wieder vergrössern: simuliert eine Kamera mit weniger Auflösung."""

    def __init__(self, factor=(0.5, 1.0), p=0.3):
        self.factor, self.p = factor, p

    def __call__(self, image):
        if random.random() > self.p:
            return image
        f = random.uniform(*self.factor)
        small = image.resize((max(1, round(image.width * f)), max(1, round(image.height * f))),
                             Image.BILINEAR)
        return small.resize(image.size, Image.BILINEAR)
