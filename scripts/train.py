"""Trainiert einen Klassifikator auf Zellen oder ganzen Ausstrichen.

    python scripts/train.py --task stage  --model simplecnn          # fragt nach Datenmenge
    python scripts/train.py --task binary --model resnet18 --pretrained --fraction 0.2 --epochs 5
    python scripts/train.py --task smear  --model simplecnn-deep --batch-size 16 --fraction 1

Ohne --fraction / --epochs fragt das Skript vor dem Start, wie viel trainiert wird
(Schnelltest, Klein, Mittel, Alles oder eigener Prozentwert). Die Teilmenge wird nach
Bildern gezogen und ist bei gleichem Seed immer dieselbe.

Tasks:
    stage   Zelle -> rbc / leukocyte / ring / trophozoite / schizont / gametocyte
    binary  Zelle -> uninfected / infected
    smear   ganzes Bild -> Multi-Label: welche der 4 Stadien kommen vor

Ergebnis in runs/<name>/: best.pt, history.json, metrics_val.json, config.json.
Das Test-Set erst ganz am Schluss mit scripts/evaluate.py anfassen.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (CellDataset, SmearDataset, balanced_sampler, cell_transform,  # noqa: E402
                      class_names, class_weights, load_cells, load_smears, smear_transform,
                      subsample_images)
from src.engine import compute_metrics, fit, get_device, predict  # noqa: E402
from src.models import build_model, count_parameters  # noqa: E402


# Name, Anteil der Bilder, vorgeschlagene Epochen
PRESETS = [("Schnelltest", 0.05, 3), ("Klein", 0.2, 10), ("Mittel", 0.5, 15), ("Alles", 1.0, 20)]


def ask_fraction(train_df):
    """Menü: wie viel der Daten verwenden? Gibt (Anteil, vorgeschlagene Epochen) zurück."""
    n_img, n_rows = train_df["image_id"].nunique(), len(train_df)
    print("\nWie viele Daten sollen verwendet werden? (Train und Val werden gleich verkleinert)")
    for k, (label, frac, ep) in enumerate(PRESETS, 1):
        print(f"  [{k}] {label:12s} {frac:5.0%}  ~{round(n_img * frac):5d} Bilder, "
              f"~{round(n_rows * frac):6d} Samples, {ep} Epochen vorgeschlagen")
    print("  oder eigenen Prozentwert eingeben, z. B. 10%")
    while True:
        answer = input("Auswahl [1]: ").strip() or "1"
        if answer.isdigit() and 1 <= int(answer) <= len(PRESETS):
            return PRESETS[int(answer) - 1][1:]
        try:
            value = float(answer.rstrip("%"))
        except ValueError:
            value = -1
        if 0 < value <= 100:
            return value / 100, 10
        print(f"Bitte 1-{len(PRESETS)} oder einen Prozentwert wie 10% eingeben.")


def ask_epochs(suggested):
    while True:
        answer = input(f"Epochen [{suggested}]: ").strip()
        if not answer:
            return suggested
        if answer.isdigit() and int(answer) > 0:
            return int(answer)
        print("Bitte eine positive ganze Zahl eingeben.")


def make_datasets(task, img_size, train_df, val_df):
    if task == "smear":
        return (SmearDataset(train_df, smear_transform(img_size, train=True)),
                SmearDataset(val_df, smear_transform(img_size)))
    return (CellDataset(train_df, task, cell_transform(img_size, train=True)),
            CellDataset(val_df, task, cell_transform(img_size)))


def make_criterion(task, train_ds, n_classes, balance):
    if task == "smear":
        # pos_weight = negative / positive pro Stadium (seltene Stadien zählen mehr)
        t = torch.from_numpy(train_ds.targets)
        pos = t.sum(0).clamp(min=1)
        return nn.BCEWithLogitsLoss(pos_weight=(len(t) - pos) / pos)
    if balance == "loss":
        return nn.CrossEntropyLoss(weight=class_weights(train_ds.labels, n_classes))
    return nn.CrossEntropyLoss()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["stage", "binary", "smear"], default="stage")
    p.add_argument("--model", default="simplecnn",
                   choices=["simplecnn", "simplecnn-deep", "resnet18", "mobilenet_v3_small"])
    p.add_argument("--pretrained", action="store_true", help="ImageNet-Gewichte (nur torchvision)")
    p.add_argument("--fraction", type=float, default=None,
                   help="Anteil der Bilder (0-1), z. B. 0.1. Ohne Angabe wird gefragt.")
    p.add_argument("--epochs", type=int, default=None, help="ohne Angabe wird gefragt")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=None, help="Default: 96 (Zellen), 384 (smear)")
    p.add_argument("--balance", choices=["sampler", "loss", "none"], default="sampler",
                   help="Umgang mit Klassen-Ungleichgewicht (nur Zell-Tasks)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default=None, help="Ordnername unter runs/")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    img_size = args.img_size or (384 if args.task == "smear" else 96)
    names = class_names(args.task)
    multilabel = args.task == "smear"
    device = get_device()

    load = load_smears if multilabel else load_cells
    train_df, val_df = load("train"), load("val")
    interactive = sys.stdin.isatty()  # False z. B. bei Jobs auf einem Server: dann nicht fragen
    suggested = 20
    if args.fraction is None:
        args.fraction, suggested = ask_fraction(train_df) if interactive else (1.0, 20)
    if not 0 < args.fraction <= 1:
        sys.exit("--fraction muss zwischen 0 und 1 liegen, z. B. 0.1 für 10 %.")
    if args.epochs is None:
        args.epochs = ask_epochs(suggested) if interactive else suggested
    train_df = subsample_images(train_df, args.fraction, args.seed)
    val_df = subsample_images(val_df, args.fraction, args.seed)
    train_ds, val_ds = make_datasets(args.task, img_size, train_df, val_df)
    sampler = None
    if not multilabel and args.balance == "sampler":
        sampler = balanced_sampler(train_ds.labels)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=sampler is None, sampler=sampler,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, args.batch_size, num_workers=args.workers)

    model = build_model(args.model, len(names), args.pretrained)
    criterion = make_criterion(args.task, train_ds, len(names), args.balance)
    name = args.name or f"{datetime.now():%Y%m%d-%H%M}_{args.task}_{args.model}"
    out_dir = ROOT / "runs" / name
    config = {**vars(args), "img_size": img_size, "class_names": list(names),
              "n_train": len(train_ds), "n_val": len(val_ds), "parameters": count_parameters(model)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\n{args.task} | {args.model} ({config['parameters']:,} Parameter) | {device} | "
          f"{args.fraction:.0%} der Daten: train {len(train_ds)} / val {len(val_ds)} | "
          f"{args.epochs} Epochen | -> {out_dir}")
    fit(model, train_loader, val_loader, criterion, names, epochs=args.epochs, lr=args.lr,
        multilabel=multilabel, device=device, out_dir=out_dir,
        checkpoint_extra={"task": args.task, "model": args.model, "img_size": img_size})

    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["state_dict"])
    metrics = compute_metrics(*predict(model, val_loader, device, multilabel), names, multilabel)
    (out_dir / "metrics_val.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nBestes Modell (Epoche {best['epoch']}), Validierung:")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
