"""Trainiert YOLO (Ultralytics) als Vergleichs-Detektor.

    python scripts/train_yolo.py --mode cell                       # COCO-vortrainiert
    python scripts/train_yolo.py --mode cell --scratch             # von Grund auf (fairer zum eigenen CNN)
    python scripts/train_yolo.py --mode stage --epochs 150
    python scripts/train_yolo.py --mode cell --fraction 0.1 --epochs 5 --device cpu   # Schnelltest

Vorher: scripts/prepare_yolo.py --mode <mode>. Ergebnis: runs/yolo/<name>/weights/best.pt
plus yolo_meta.json (Vorverarbeitung). Ultralytics wählt best.pt nach seiner Fitness
(mAP50-95 auf val); den Vergleich mit der Segmentierung macht scripts/compare_detection.py.

Augmentation: Ultralytics-Standard (Mosaic, HSV, Skalierung) plus vertikales Spiegeln, weil
Mikroskopbilder keine Oben/Unten-Richtung haben. HSV ist etwas stärker als Standard, als
Gegenstück zu StainJitter der Hauptpipeline.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.yolo import META_FILE, MODES, YOLO_DIR  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(MODES), default="cell")
    p.add_argument("--data", type=Path, default=None, help="Default: data/yolo/<mode>/data.yaml")
    p.add_argument("--model", default="yolo26n", help="z. B. yolo26n, yolo26s, yolo11n")
    p.add_argument("--scratch", action="store_true", help="ohne COCO-Gewichte (aus .yaml)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--fraction", type=float, default=1.0, help="Anteil der Trainingsbilder")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", default=None, help="z. B. 0, cpu, mps")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    from ultralytics import YOLO

    data = args.data or YOLO_DIR / args.mode / "data.yaml"
    if not data.exists():
        sys.exit(f"{data} fehlt. Zuerst: python scripts/prepare_yolo.py --mode {args.mode}")
    weights = f"{args.model}.yaml" if args.scratch else f"{args.model}.pt"
    name = args.name or (f"{datetime.now():%Y%m%d-%H%M}_{args.mode}_{args.model}"
                         f"{'_scratch' if args.scratch else ''}")
    project = ROOT / "runs" / "yolo"

    model = YOLO(weights)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                fraction=args.fraction, patience=args.patience, device=args.device,
                workers=args.workers, seed=args.seed, deterministic=True,
                project=str(project), name=name, exist_ok=False,
                pretrained=not args.scratch,
                # Zellen sind klein und zahlreich (bis ~200 pro Bild)
                max_det=1000,
                # Mikroskopie: keine Vorzugsrichtung; Färbung variiert (Domain Shift)
                flipud=0.5, fliplr=0.5, degrees=10, hsv_h=0.03, hsv_s=0.6, hsv_v=0.4,
                plots=True)

    run_dir = Path(model.trainer.save_dir)
    meta = json.loads((data.parent / META_FILE).read_text())
    meta.update({"imgsz": args.imgsz, "weights_init": weights, "train_args": vars(args)
                 | {"data": str(data)}})
    (run_dir / META_FILE).write_text(json.dumps(meta, indent=2, default=str))
    shutil.copy(data, run_dir / "data.yaml")
    print(f"\nFertig: {run_dir / 'weights' / 'best.pt'}\n"
          f"Vergleich: python scripts/compare_detection.py --split val "
          f"--yolo {run_dir / 'weights' / 'best.pt'} --cell-model runs/<zell-run>/best.pt")


if __name__ == "__main__":
    main()
