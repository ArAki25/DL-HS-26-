"""Vergleich: klassische Segmentierung (Hauptpipeline) vs. YOLO, mit identischer Auswertung.

    # 1. Lokalisierung: findet YOLO die Zellen besser als Otsu + Watershed?
    python scripts/compare_detection.py --split val --yolo runs/yolo/<cell-run>/weights/best.pt

    # 2. Ganze Pipeline, mit dem bestehenden Zell-CNN
    python scripts/compare_detection.py --split val \\
        --yolo runs/yolo/<cell-run>/weights/best.pt --yolo runs/yolo/<stage-run>/weights/best.pt \\
        --cell-model runs/<stage-run>/best.pt

    # 3. Erst ganz am Schluss: Test (andere Kamera), mit den auf val gewählten Schwellen
    python scripts/compare_detection.py --split test --yolo ... --cell-model ...

Verglichene Pipelines (je nach Argumenten):
    seg                    Otsu + Watershed (src/segmentation.py)            nur Lokalisierung
    yolo:<run>             YOLO-Boxen, Klassen ignoriert                     nur Lokalisierung
    seg+cnn                Hauptpipeline wie scripts/predict.py              Lokalisierung + Klasse
    yolo:<run>+cnn         YOLO-Boxen, Klassifikation durch dasselbe Zell-CNN (isoliert den Detektor)
    yolo:<run> e2e         YOLO im Modus 'stage' klassifiziert selbst        End-to-End

Die YOLO-Konfidenzschwelle wird auf val gewählt (max. Lokalisierungs-F1) und in
operating_points.json gespeichert; auf test wird diese Schwelle verwendet, nie neu optimiert.
Zusätzlich wird für YOLO die AP50 (schwellenunabhängig) berichtet. Unsicherheit: paired
Bootstrap über Bilder (95 %-CI, Differenz zur Hauptpipeline).

Ausgabe in runs/compare/<split>/: metrics.json, summary.md, optional Beispielbilder.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from predict import classify_cells  # noqa: E402  (scripts/predict.py, identische Zell-Klassifikation)
from src.data import (BINARY_CLASSES, CELL_CLASSES, STAGES, assign_splits,  # noqa: E402
                      load_annotations, subsample_images)
from src.detection_eval import (average_precision, image_confusion, paired_bootstrap,  # noqa: E402
                                summarize)
from src.engine import get_device  # noqa: E402
from src.segmentation import detect_cells  # noqa: E402
from src.stain import StainNormalizer  # noqa: E402
from src.yolo import YoloDetector, load_normalizer  # noqa: E402

CONF_GRID = np.round(np.concatenate([np.arange(0.05, 0.5, 0.025), np.arange(0.5, 0.96, 0.05)]), 3)
CLASSIFY_MIN_CONF = CONF_GRID.min()  # darunter wird nicht klassifiziert (spart Zeit)


def load_ground_truth(split, fraction):
    ann = load_annotations()
    ann["split"] = ann["image_id"].map(assign_splits(ann))
    ann = subsample_images(ann[ann.split == split], fraction)
    images = []
    for image_id, objs in ann.groupby("image_id", sort=True):
        normal = objs[objs.category != "difficult"]
        images.append({
            "image_id": image_id,
            "path": objs.image_path.iloc[0],
            "gt_boxes": normal[["r0", "c0", "r1", "c1"]].to_numpy(),
            "gt_stage": normal.category.map({c: i for i, c in enumerate(CELL_CLASSES)}).to_numpy(),
            "ignore": objs.loc[objs.category == "difficult", ["r0", "c0", "r1", "c1"]].to_numpy(float),
        })
    return images


def tag_of(weights):
    """runs/yolo/<name>/weights/best.pt -> <name>"""
    weights = Path(weights)
    return weights.parent.parent.name if weights.parent.name == "weights" else weights.stem


def collect_predictions(images, detectors, cell_model, normalizer, device):
    """Alle Rohvorhersagen einmal berechnen; Schwellen werden danach nur noch gefiltert."""
    timing = {"seg": [], **{t: [] for t in detectors}, "cnn": []}
    for img in tqdm(images, desc="Bilder"):
        normalized = normalizer(Image.open(img["path"]))
        arr = np.asarray(normalized)

        t0 = time.perf_counter()
        img["seg"] = {"boxes": detect_cells(arr)}
        timing["seg"].append(time.perf_counter() - t0)

        for tag, det in detectors.items():
            t0 = time.perf_counter()
            boxes, conf, cls = det.detect(normalized)
            timing[tag].append(time.perf_counter() - t0)
            img[tag] = {"boxes": boxes, "conf": conf, "cls": cls}

        if cell_model is not None:
            model, ckpt = cell_model
            for key in ("seg", *detectors):
                pred = img[key]
                keep = pred.get("conf", np.ones(len(pred["boxes"]))) >= CLASSIFY_MIN_CONF
                t0 = time.perf_counter()
                probs = classify_cells(normalized, pred["boxes"][keep], model, ckpt, device)
                timing["cnn"].append((time.perf_counter() - t0) / max(len(probs), 1))
                cnn_cls = np.full(len(pred["boxes"]), -1)
                cnn_cls[keep] = probs.argmax(1) if len(probs) else []
                pred["cnn_cls"] = cnn_cls
    return {k: float(np.mean(v)) * 1000 for k, v in timing.items() if v}


def build_cms(images, key, gt_names, pred_names, cls_field, conf_thr, iou, gt_map):
    """Gestapelte Bild-Konfusionsmatrizen für eine Pipeline bei fester Schwelle."""
    cms = []
    for img in images:
        pred = img[key]
        keep = np.ones(len(pred["boxes"]), bool)
        if "conf" in pred:
            keep &= pred["conf"] >= conf_thr
        cls = np.zeros(len(pred["boxes"]), int) if cls_field is None else pred[cls_field]
        keep &= cls >= 0
        cms.append(image_confusion(pred["boxes"][keep], cls[keep], img["gt_boxes"],
                                   gt_map[img["gt_stage"]], img["ignore"],
                                   len(gt_names), len(pred_names), iou))
    return np.stack(cms)


def choose_operating_points(images, detectors, iou):
    """Pro YOLO: Konfidenz mit max. Lokalisierungs-F1 (nur auf val aufrufen!) + AP50."""
    out = {}
    for tag in detectors:
        per_image = [(img[tag]["boxes"], img[tag]["conf"], img["gt_boxes"], img["ignore"])
                     for img in images]
        ap, curve = average_precision(per_image, CONF_GRID, iou)
        best = int(np.argmax(curve[:, 2]))
        out[tag] = {"conf": float(CONF_GRID[best]), "ap50": ap, "f1_at_conf": float(curve[best, 2])}
    return out


def fmt(v, pct=True):
    return "–" if v is None else (f"{v * 100:.1f}" if pct else f"{v:.2f}")


def write_summary(results, boot, timing, split, out_dir):
    lines = [f"# Segmentierung vs. YOLO, Split `{split}`", "",
             "Werte in %, 95 %-CI aus paired Bootstrap über Bilder. Δ = Differenz zur "
             "Hauptpipeline (`seg` bzw. `seg+cnn`).", "",
             "## Lokalisierung (klassen-agnostisch)", "",
             "| Pipeline | Precision | Recall | F1 [95 %-CI] | Δ F1 [95 %-CI] | Recall infiziert | "
             "Zählfehler | AP50 | ms/Bild |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        if "+cnn" in name or name.endswith(" e2e"):
            continue
        b = boot["loc_f1"][name]
        diff = (f"{b['diff_vs_baseline'] * 100:+.1f} [{b['diff_ci95'][0] * 100:+.1f}, "
                f"{b['diff_ci95'][1] * 100:+.1f}]" if "diff_vs_baseline" in b else "Baseline")
        lines.append(f"| {name} | {fmt(m['loc_precision'])} | {fmt(m['loc_recall'])} | "
                     f"{fmt(m['loc_f1'])} [{fmt(b['ci95'][0])}, {fmt(b['ci95'][1])}] | {diff} | "
                     f"{fmt(m['loc_recall_infected'])} | {fmt(m['count_rel_error'])} | "
                     f"{fmt(m.get('ap50'))} | {timing.get(name, 0):.0f} |")

    cls_rows = {k: v for k, v in results.items() if "macro_f1_present" in v}
    if cls_rows:
        names = list(next(iter(cls_rows.values()))["per_class"])
        lines += ["", "## Ganze Pipeline (Box gefunden UND Klasse richtig)", "",
                  "| Pipeline | Macro-F1 (vorh.) [95 %-CI] | Δ Macro-F1 | F1 infiziert | "
                  "Recall infiziert | Parasitämie-MAE (pp) | Sens. Bild | "
                  + " | ".join(f"F1 {n}" for n in names) + " |",
                  "|---|---|---|---|---|---|---|" + "---|" * len(names)]
        for name, m in cls_rows.items():
            b = boot["macro_f1_present"].get(name)
            ci = f" [{fmt(b['ci95'][0])}, {fmt(b['ci95'][1])}]" if b else ""
            diff = (f"{b['diff_vs_baseline'] * 100:+.1f} [{b['diff_ci95'][0] * 100:+.1f}, "
                    f"{b['diff_ci95'][1] * 100:+.1f}]" if b and "diff_vs_baseline" in b else "Baseline")
            lines.append(f"| {name} | {fmt(m['macro_f1_present'])}{ci} | {diff} | "
                         f"{fmt(m['infected_f1'])} | {fmt(m['infected_recall'])} | "
                         f"{m['parasitemia_mae_pp']:.2f} | {fmt(m.get('image_sensitivity'))} | "
                         + " | ".join(fmt(m["per_class"][n]["f1"]) for n in names) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def save_examples(images, keys, n, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    for img in images[:n]:
        fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 5.5))
        raw = np.asarray(Image.open(img["path"]).convert("RGB"))
        for ax, (key, thr) in zip(np.atleast_1d(axes), keys.items()):
            ax.imshow(raw)
            pred = img[key]
            keep = pred["conf"] >= thr if "conf" in pred else np.ones(len(pred["boxes"]), bool)
            for boxes, color in ((img["gt_boxes"], "lime"), (pred["boxes"][keep], "red")):
                for r0, c0, r1, c1 in boxes:
                    ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec=color, lw=0.8))
            ax.set_title(f"{key}: grün = Annotation, rot = Vorhersage")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"example_{img['image_id']}.png", dpi=90)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--yolo", type=Path, action="append", default=[], help="best.pt, mehrfach möglich")
    p.add_argument("--cell-model", type=Path, default=None, help="Zell-CNN (task stage oder binary)")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--fraction", type=float, default=1.0, help="Anteil der Bilder (Schnelltest)")
    p.add_argument("--op-file", type=Path, default=None,
                   help="Schwellen von val; Default runs/compare/val/operating_points.json")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--show", type=int, default=0, help="N Beispielbilder speichern")
    p.add_argument("--out", type=Path, default=None, help="Default: runs/compare/<split>")
    args = p.parse_args()

    device = get_device()
    out_dir = args.out or ROOT / "runs" / "compare" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_model = None
    if args.cell_model:
        from src.models import load_checkpoint
        cell_model = load_checkpoint(args.cell_model, device)
        if cell_model[1]["task"] == "smear":
            sys.exit("--cell-model muss ein Zell-Modell sein (task stage oder binary).")
        normalizer = StainNormalizer.from_dict(cell_model[1].get("normalization") or {"method": "none"})
    else:
        normalizer = load_normalizer()

    detectors = {}
    for w in args.yolo:
        det = YoloDetector(w, device=device.type if device.type != "cuda" else 0)
        if det.meta["normalization"] != normalizer.to_dict():
            print(f"Warnung: {w} wurde mit anderer Normalisierung exportiert als das Zell-CNN.")
        detectors[f"yolo:{tag_of(w)}"] = det

    images = load_ground_truth(args.split, args.fraction)
    print(f"{args.split}: {len(images)} Bilder, {sum(len(i['gt_boxes']) for i in images)} Zellen")
    timing = collect_predictions(images, detectors, cell_model, normalizer, device)

    op_file = args.op_file or ROOT / "runs" / "compare" / "val" / "operating_points.json"
    if args.split == "val":
        ops = choose_operating_points(images, detectors, args.iou)
        (out_dir / "operating_points.json").write_text(json.dumps(ops, indent=2))
    elif detectors:
        if not op_file.exists():
            sys.exit(f"{op_file} fehlt: zuerst mit --split val laufen lassen (Schwellen nie auf test wählen).")
        ops = json.loads(op_file.read_text())
        missing = set(detectors) - set(ops)
        if missing:
            sys.exit(f"Keine val-Schwelle für {sorted(missing)} in {op_file}.")
        test_ap = choose_operating_points(images, detectors, args.iou)
        for tag in detectors:  # AP ist schwellenunabhängig, darf auf test berichtet werden
            ops[tag] = {**ops[tag], "ap50": test_ap[tag]["ap50"]}
    else:
        ops = {}

    # Pipelines: (Schlüssel der Vorhersage, Feld mit Klassen, Klassennamen der Vorhersage)
    stage_map = np.arange(len(CELL_CLASSES))
    pipelines = {"seg": ("seg", None, ("cell",), 0.0)}
    pipelines |= {t: (t, None, ("cell",), ops[t]["conf"]) for t in detectors}
    cls_names, gt_map = CELL_CLASSES, stage_map
    if cell_model is not None:
        cls_names = tuple(cell_model[1]["class_names"])
        if cell_model[1]["task"] == "binary":
            gt_map = np.array([int(c in STAGES) for c in CELL_CLASSES])
        pipelines["seg+cnn"] = ("seg", "cnn_cls", cls_names, 0.0)
        pipelines |= {f"{t}+cnn": (t, "cnn_cls", cls_names, ops[t]["conf"]) for t in detectors}
    for t, det in detectors.items():
        if det.meta["mode"] == "stage" and tuple(det.names) == CELL_CLASSES:
            if cell_model is None or cls_names == CELL_CLASSES:
                pipelines[f"{t} e2e"] = (t, "cls", CELL_CLASSES, ops[t]["conf"])

    infected = ("infected",) if cls_names == BINARY_CLASSES else STAGES
    results, cms = {}, {}
    for name, (key, field, pred_names, thr) in pipelines.items():
        gt_names = cls_names if field else CELL_CLASSES
        g_map = gt_map if field else stage_map
        cms[name] = build_cms(images, key, gt_names, pred_names, field, thr, args.iou, g_map)
        results[name] = summarize(cms[name], gt_names, pred_names,
                                  infected if field else STAGES)
        results[name]["conf"] = thr if key != "seg" else None
        if key in ops and field is None:
            results[name]["ap50"] = ops[key]["ap50"]

    # Paired Bootstrap: Lokalisierung gegen 'seg', Klassen-Metrik gegen 'seg+cnn'
    loc_f1 = lambda c: summarize(c, CELL_CLASSES, ("cell",), STAGES)["loc_f1"]  # noqa: E731
    loc_names = [n for n, v in pipelines.items() if v[1] is None]
    boot = {"loc_f1": paired_bootstrap({n: cms[n] for n in loc_names}, loc_f1, "seg", args.bootstrap)}
    cls_pipes = [n for n in results if "macro_f1_present" in results[n]]
    if "seg+cnn" in cls_pipes:
        same = [n for n in cls_pipes if pipelines[n][2] == cls_names]
        mf1 = lambda c: summarize(c, cls_names, cls_names, infected)["macro_f1_present"]  # noqa: E731
        boot["macro_f1_present"] = paired_bootstrap({n: cms[n] for n in same}, mf1, "seg+cnn",
                                                    args.bootstrap)
    else:
        boot["macro_f1_present"] = {}

    timing_by_pipe = {"seg": timing["seg"], **{t: timing[t] for t in detectors}}
    (out_dir / "metrics.json").write_text(json.dumps(
        {"split": args.split, "iou": args.iou, "fraction": args.fraction, "results": results,
         "bootstrap": boot, "ms_per_image": timing, "operating_points": ops,
         "yolo": {t: str(w) for t, w in zip(detectors, args.yolo)},
         "cell_model": str(args.cell_model) if args.cell_model else None},
        indent=2, default=float))
    print("\n" + write_summary(results, boot, timing_by_pipe, args.split, out_dir))
    if "cnn" in timing:
        print(f"\nZell-CNN: {timing['cnn']:.2f} ms pro Zelle ({device})")
    if args.show:
        save_examples(images, {"seg": 0.0, **{t: ops[t]["conf"] for t in detectors}}, args.show, out_dir)
    print(f"\nGespeichert: {out_dir}")


if __name__ == "__main__":
    main()
