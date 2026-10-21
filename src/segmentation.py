"""Zellen ohne Objektdetektor (kein YOLO) finden: klassische Bildverarbeitung.

Idee: Zellen sind dunkler als der Hintergrund -> Otsu-Schwelle -> Löcher füllen
(Erythrozyten sind in der Mitte heller) -> Distanztransformation + Watershed, um sich
berührende Zellen zu trennen -> eine Box pro Region. Die Boxen gehen danach an das
Zell-CNN (siehe scripts/predict.py).

    from src.segmentation import detect_cells, match_boxes
    boxes = detect_cells(np.asarray(Image.open(path).convert("RGB")))
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.transform import rescale


def _cell_mask(image, scale):
    """Binärmaske 'Zelle' auf verkleinertem Graubild."""
    gray = rgb2gray(np.asarray(image))
    if scale != 1:
        gray = rescale(gray, scale, anti_aliasing=True)
    gray = gaussian(gray, sigma=2)
    mask = gray < threshold_otsu(gray)
    mask = ndi.binary_fill_holes(mask)
    return ndi.binary_opening(mask, iterations=2)


def estimate_cell_diameter(image, scale=0.25):
    """Typischer Zelldurchmesser in Pixeln, ohne Annotationen geschätzt.

    Kameras vergrössern unterschiedlich stark (BBBC041: Training ~108 px, Test ~135 px).
    Einzelne, runde Regionen (Solidity > 0.9) sind fast immer genau eine Zelle; deren
    medianer Durchmesser ist robust gegen Zellhaufen. Die Maske verpasst den blassen
    Zellrand, deshalb der Faktor 1.12 (einmal gegen die Trainings-Annotationen kalibriert;
    danach Abweichung Training ~0 %, Test ~-4 %).
    """
    regions = regionprops(ndi.label(_cell_mask(image, scale))[0])
    diameters = [r.equivalent_diameter_area for r in regions if r.solidity > 0.9 and r.area > 20]
    if len(diameters) < 5:
        return None
    return float(np.median(diameters) / scale * 1.12)


def detect_cells(image, scale=0.5, cell_diameter=None, min_area_frac=0.15, max_area_frac=4.0):
    """Boxen (N x 4, Spalten r0, c0, r1, c1) in Koordinaten des Originalbilds.

    cell_diameter: typischer Zelldurchmesser in Pixeln im Originalbild. None = pro Bild
    schätzen (estimate_cell_diameter), damit es mit jeder Kamera/Vergrösserung funktioniert.
    Regionen kleiner/grösser als min/max_area_frac * Fläche einer typischen Zelle fliegen raus.
    """
    cell_diameter = cell_diameter or estimate_cell_diameter(image) or 100
    mask = _cell_mask(image, scale)

    radius = cell_diameter * scale / 2
    distance = ndi.distance_transform_edt(mask)
    regions, _ = ndi.label(mask)
    peaks = peak_local_max(distance, min_distance=max(int(radius * 0.7), 1),
                           labels=regions, exclude_border=False)
    markers = np.zeros_like(regions)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-distance, markers, mask=mask)

    cell_area = np.pi * radius ** 2
    boxes = [r.bbox for r in regionprops(labels)
             if min_area_frac * cell_area <= r.area <= max_area_frac * cell_area]
    return (np.array(boxes, dtype=float).reshape(-1, 4) / scale).round().astype(int)


def box_iou(a, b):
    """IoU-Matrix zwischen Boxen a (N x 4) und b (M x 4), Format r0, c0, r1, c1."""
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None]
    inter_h = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    inter_w = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = inter_h * inter_w
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def match_boxes(pred, gt, iou_threshold=0.5):
    """Greedy-Zuordnung nach IoU. Gibt (Paare [(pred_idx, gt_idx)], tp, fp, fn) zurück."""
    if len(pred) == 0 or len(gt) == 0:
        return [], 0, len(pred), len(gt)
    iou = box_iou(pred, gt)
    pairs, used_p, used_g = [], set(), set()
    for idx in np.argsort(-iou, axis=None):
        p, g = map(int, np.unravel_index(idx, iou.shape))
        if iou[p, g] < iou_threshold:
            break
        if p not in used_p and g not in used_g:
            pairs.append((p, g))
            used_p.add(p)
            used_g.add(g)
    return pairs, len(pairs), len(pred) - len(pairs), len(gt) - len(pairs)
