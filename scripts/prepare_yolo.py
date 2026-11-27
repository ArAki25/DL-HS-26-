"""Exportiert die Annotationen als YOLO-Datensatz (nur für den Vergleich, siehe src/yolo.py).

    python scripts/prepare_yolo.py --mode cell     # 1 Klasse: Lokalisierung vs. Segmentierung
    python scripts/prepare_yolo.py --mode stage    # 6 Klassen: End-to-End-Detektor

Voraussetzung: scripts/prepare_data.py ist gelaufen (für normalization.json), damit YOLO
genau dieselbe Farbnormalisierung bekommt wie das Zell-CNN.

Erzeugt data/yolo/<mode>/: images/{train,val,test}, labels/{train,val,test}, data.yaml,
yolo_meta.json (Normalisierung + Zellgrösse, wird von train_yolo.py in den Run kopiert).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.yolo import MODES, YOLO_DIR, export_yolo_dataset  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(MODES), default="cell")
    p.add_argument("--cell-px", type=int, default=48,
                   help="Zelldurchmesser nach dem Skalieren (0 = nicht skalieren, Ablation). "
                        "48 px -> Trainingsbilder ca. 710 x 530, passt zu --imgsz 768")
    p.add_argument("--out", type=Path, default=None, help="Default: data/yolo/<mode>")
    p.add_argument("--fraction", type=float, default=1.0, help="Anteil der Bilder (Schnelltest)")
    args = p.parse_args()

    out, meta = export_yolo_dataset(args.out or YOLO_DIR / args.mode, args.mode,
                                    cell_px=args.cell_px or None, fraction=args.fraction)
    print(json.dumps(meta["stats"], indent=2))
    print(f"Fertig: {out}/data.yaml")


if __name__ == "__main__":
    main()
