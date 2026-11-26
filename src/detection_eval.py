"""Gemeinsame Auswertung für alle Zell-Detektions-Pipelines (Segmentierung, YOLO, Hybride).

Damit der Vergleich fair ist, laufen alle Methoden durch exakt denselben Code:

1. Zuordnung: klassen-agnostisch, greedy nach IoU (match_boxes aus src/segmentation.py,
   identisch zu scripts/evaluate_segmentation.py).
2. 'difficult'-Zellen sind keine Ground Truth, aber auch kein Hintergrund: eine sonst
   unzugeordnete Vorhersage, die eine difficult-Box mit IoU >= Schwelle trifft, wird
   ignoriert (wie "crowd" bei COCO). Gilt für jede Methode gleich.
3. Pro Bild eine Detektions-Konfusionsmatrix cm[g, p] mit einer zusätzlichen Zeile/Spalte
   "Hintergrund" (letzter Index): unzugeordnete GT -> cm[g, bg] (FN), unzugeordnete
   Vorhersage -> cm[bg, p] (FP). Alle Metriken (Lokalisierung, pro Klasse, Parasitämie,
   Bild-Ebene) werden daraus berechnet, damit auch das Bootstrap über Bilder billig ist.

    cm = image_confusion(pred_boxes, pred_cls, gt_boxes, gt_cls, ignore_boxes, n_gt, n_pred)
    m = summarize(np.stack(cms), gt_names, pred_names, infected=STAGES)
"""
import numpy as np

from .segmentation import box_iou, match_boxes


def image_confusion(pred_boxes, pred_cls, gt_boxes, gt_cls, ignore_boxes, n_gt, n_pred,
                    iou_threshold=0.5):
    """Konfusionsmatrix (n_gt + 1) x (n_pred + 1) für ein Bild, letzter Index = Hintergrund."""
    pred_boxes = np.asarray(pred_boxes, float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, float).reshape(-1, 4)
    pred_cls, gt_cls = np.asarray(pred_cls, int), np.asarray(gt_cls, int)
    cm = np.zeros((n_gt + 1, n_pred + 1), dtype=np.int64)

    pairs, *_ = match_boxes(pred_boxes, gt_boxes, iou_threshold)
    matched_p = {p for p, _ in pairs}
    matched_g = {g for _, g in pairs}
    for p, g in pairs:
        cm[gt_cls[g], pred_cls[p]] += 1
    for g in range(len(gt_boxes)):
        if g not in matched_g:
            cm[gt_cls[g], n_pred] += 1

    unmatched = np.array([p for p in range(len(pred_boxes)) if p not in matched_p], dtype=int)
    if len(unmatched) and len(ignore_boxes):
        hits_ignore = (box_iou(pred_boxes[unmatched], ignore_boxes) >= iou_threshold).any(1)
        unmatched = unmatched[~hits_ignore]
    for p in unmatched:
        cm[n_gt, pred_cls[p]] += 1
    return cm


def _f1(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall, 2 * precision * recall / max(precision + recall, 1e-12)


def localization_metrics(cms):
    """Klassen-agnostisch: wurde jede Zelle gefunden, egal mit welcher Klasse?"""
    cm = cms.sum(0)
    tp, fp, fn = cm[:-1, :-1].sum(), cm[-1, :-1].sum(), cm[:-1, -1].sum()
    precision, recall, f1 = _f1(tp, fp, fn)
    gt_per_img = cms[:, :-1, :].sum((1, 2))
    pred_per_img = cms[:, :, :-1].sum((1, 2))
    count_err = np.abs(pred_per_img - gt_per_img) / np.maximum(gt_per_img, 1)
    return {"loc_precision": precision, "loc_recall": recall, "loc_f1": f1,
            "count_rel_error": float(count_err.mean())}


def summarize(cms, gt_names, pred_names, infected):
    """Alle Metriken aus gestapelten Bild-Matrizen (N x (K+1) x (P+1)).

    infected: Namen der infizierten Klassen (STAGES oder ('infected',)). Klassen-Metriken
    gibt es nur, wenn die Methode dieselben Klassen vorhersagt wie die GT hat.
    """
    cms = np.asarray(cms)
    m = localization_metrics(cms)
    inf_g = [i for i, n in enumerate(gt_names) if n in infected]
    cm = cms.sum(0)
    m["loc_recall_infected"] = cm[inf_g, :-1].sum() / max(cm[inf_g, :].sum(), 1)
    m["n_images"] = int(len(cms))
    m["n_gt_cells"] = int(cm[:-1].sum())
    m["n_gt_infected"] = int(cm[inf_g].sum())

    if list(pred_names) != list(gt_names):
        return m

    k = len(gt_names)
    f1s, per_class = [], {}
    for c, name in enumerate(gt_names):
        tp = cm[c, c]
        p, r, f1 = _f1(tp, cm[:, c].sum() - tp, cm[c, :].sum() - tp)
        per_class[name] = {"precision": p, "recall": r, "f1": f1, "support": int(cm[c].sum())}
        if cm[c].sum() > 0:  # wie macro_f1_present in engine.py: fehlende Klassen nicht mitteln
            f1s.append(f1)
    m["per_class"] = per_class
    m["macro_f1_present"] = float(np.mean(f1s)) if f1s else 0.0

    # Infiziert vs. nicht infiziert, als Detektionsaufgabe (Box gefunden UND als infiziert erkannt)
    tp = cm[np.ix_(inf_g, inf_g)].sum()
    pred_inf = cm[:, inf_g].sum()
    gt_inf = cm[inf_g, :].sum()
    p, r, f1 = _f1(tp, pred_inf - tp, gt_inf - tp)
    m.update({"infected_precision": p, "infected_recall": r, "infected_f1": f1})

    # Parasitämie pro Bild (Anteil infizierter Zellen), klinisch die eigentliche Zielgrösse
    gt_frac = cms[:, inf_g, :].sum((1, 2)) / np.maximum(cms[:, :k, :].sum((1, 2)), 1)
    pr_frac = cms[:, :, inf_g].sum((1, 2)) / np.maximum(cms[:, :, :k].sum((1, 2)), 1)
    m["parasitemia_mae_pp"] = float(np.abs(pr_frac - gt_frac).mean() * 100)

    # Bild-Ebene: "enthält das Bild mindestens eine infizierte Zelle?"
    gt_pos = cms[:, inf_g, :].sum((1, 2)) > 0
    pr_pos = cms[:, :, inf_g].sum((1, 2)) > 0
    if gt_pos.any():
        m["image_sensitivity"] = float((pr_pos & gt_pos).sum() / gt_pos.sum())
    if (~gt_pos).any():
        m["image_specificity"] = float((~pr_pos & ~gt_pos).sum() / (~gt_pos).sum())
    return m


def average_precision(per_image, thresholds, iou_threshold=0.5):
    """Klassen-agnostische AP aus einer Schwellen-Sweep über die Konfidenz.

    per_image: Liste von (pred_boxes, conf, gt_boxes, ignore_boxes). Gibt (AP, P, R, F1
    je Schwelle) zurück. AP = Fläche unter der monoton gemachten PR-Kurve (VOC all-point).
    """
    curve = []
    for t in thresholds:
        tp = fp = fn = 0
        for boxes, conf, gt, ign in per_image:
            keep = conf >= t
            cm = image_confusion(boxes[keep], np.zeros(keep.sum(), int), gt,
                                 np.zeros(len(gt), int), ign, 1, 1, iou_threshold)
            tp, fp, fn = tp + cm[0, 0], fp + cm[1, 0], fn + cm[0, 1]
        curve.append(_f1(tp, fp, fn))
    curve = np.array(curve)  # (T, 3): precision, recall, f1
    order = np.argsort(curve[:, 1])
    rec = np.concatenate([[0], curve[order, 1], [curve[:, 1].max()]])
    prec = np.concatenate([[1], curve[order, 0], [0]])
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    ap = float(np.sum(np.diff(rec) * prec[1:]))
    return ap, curve


def paired_bootstrap(cms_by_method, metric_fn, baseline, n_boot=2000, seed=0):
    """Paired Bootstrap über Bilder: gleiche Bild-Stichprobe für alle Methoden.

    cms_by_method: {name: (N x ..) Array, gleiche Bildreihenfolge}. metric_fn(cms) -> float.
    Gibt pro Methode Punktwert, 95 %-CI und die Differenz zur Baseline (mit CI und
    p = Anteil Resamples, in denen die Methode nicht besser ist) zurück.
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(cms_by_method.values())))
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = {k: np.array([metric_fn(v[i]) for i in idx]) for k, v in cms_by_method.items()}
    out = {}
    for name, s in samples.items():
        row = {"value": float(metric_fn(cms_by_method[name])),
               "ci95": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]}
        if name != baseline:
            d = s - samples[baseline]
            row["diff_vs_baseline"] = float(row["value"] - metric_fn(cms_by_method[baseline]))
            row["diff_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            row["p_not_better"] = float((d <= 0).mean())
        out[name] = row
    return out
