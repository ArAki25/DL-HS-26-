"""Tests für die gemeinsame Detektions-Auswertung (python -m pytest tests)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection_eval import (average_precision, image_confusion, paired_bootstrap,  # noqa: E402
                                summarize)

NAMES = ("rbc", "ring")
GT = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]])
GT_CLS = np.array([0, 1, 1])


def test_confusion_counts_tp_fp_fn_and_class_errors():
    pred = np.array([[0, 0, 10, 10], [21, 21, 31, 31], [80, 80, 90, 90]])
    cm = image_confusion(pred, [0, 0, 1], GT, GT_CLS, np.zeros((0, 4)), 2, 2)
    # rbc richtig, ring als rbc, 1 ring verpasst, 1 FP als ring
    assert cm.tolist() == [[1, 0, 0], [1, 0, 1], [0, 1, 0]]
    m = summarize(cm[None], NAMES, NAMES, infected=("ring",))
    assert m["loc_recall"] == 2 / 3 and m["loc_precision"] == 2 / 3
    assert m["infected_recall"] == 0.0  # gefunden, aber falsche Klasse zählt nicht
    assert m["loc_recall_infected"] == 0.5


def test_difficult_boxes_are_ignored_not_false_positives():
    pred = np.array([[0, 0, 10, 10], [60, 60, 70, 70]])
    ignore = np.array([[60, 60, 70, 70]], float)
    cm = image_confusion(pred, [0, 0], GT[:1], GT_CLS[:1], ignore, 2, 2)
    assert cm[-1].sum() == 0  # keine FP


def test_agnostic_methods_have_no_class_metrics():
    cm = image_confusion(GT, [0, 0, 0], GT, GT_CLS, np.zeros((0, 4)), 2, 1)
    m = summarize(cm[None], NAMES, ("cell",), infected=("ring",))
    assert m["loc_f1"] == 1.0 and "macro_f1_present" not in m


def test_average_precision_perfect_detector():
    conf = np.array([0.9, 0.8, 0.7])
    ap, curve = average_precision([(GT, conf, GT, np.zeros((0, 4)))], [0.1, 0.5, 0.95])
    assert abs(ap - 1.0) < 1e-9 and curve[0, 2] == 1.0


def test_paired_bootstrap_detects_better_method():
    good = np.stack([image_confusion(GT, [0, 1, 1], GT, GT_CLS, np.zeros((0, 4)), 2, 2)] * 20)
    bad = np.stack([image_confusion(GT[:1], [0], GT, GT_CLS, np.zeros((0, 4)), 2, 2)] * 20)
    f = lambda c: summarize(c, NAMES, NAMES, ("ring",))["macro_f1_present"]  # noqa: E731
    out = paired_bootstrap({"base": bad, "new": good}, f, "base", n_boot=200)
    assert out["new"]["diff_vs_baseline"] > 0 and out["new"]["p_not_better"] == 0.0
