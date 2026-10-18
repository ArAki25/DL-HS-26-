"""Trainiert einen Klassifikator auf Zellen oder ganzen Ausstrichen.

    python scripts/train.py --task stage  --model simplecnn --epochs 20
    python scripts/train.py --task binary --model resnet18 --pretrained
    python scripts/train.py --task smear  --model simplecnn-deep --img-size 384 --batch-size 16

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
                      class_names, class_weights, load_cells, load_smears, smear_transform)
from src.engine import compute_metrics, fit, get_device, predict  # noqa: E402
from src.models import build_model, count_parameters  # noqa: E402


def make_datasets(task, img_size):
    if task == "smear":
        return (SmearDataset(load_smears("train"), smear_transform(img_size, train=True)),
                SmearDataset(load_smears("val"), smear_transform(img_size)))
    return (CellDataset(load_cells("train"), task, cell_transform(img_size, train=True)),
            CellDataset(load_cells("val"), task, cell_transform(img_size)))


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
    p.add_argument("--epochs", type=int, default=20)
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

    train_ds, val_ds = make_datasets(args.task, img_size)
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

    print(f"{args.task} | {args.model} ({config['parameters']:,} Parameter) | {device} | "
          f"train {len(train_ds)} / val {len(val_ds)} | -> {out_dir}")
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
