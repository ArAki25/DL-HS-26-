"""Training, Evaluation und Metriken, für Single-Label (Zellen) und Multi-Label (Ausstriche).

    from src.engine import fit, predict, compute_metrics
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)
from tqdm.auto import tqdm

# Bei kleinen Teilmengen (--fraction) fehlen seltene Klassen in val -> sonst Warnung pro Epoche
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device, multilabel=False):
    """Wahrscheinlichkeiten (N x K) und Ziele für einen ganzen DataLoader."""
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append((torch.sigmoid(logits) if multilabel else torch.softmax(logits, 1)).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def compute_metrics(probs, targets, class_names, multilabel=False):
    """Accuracy allein ist hier wertlos (>95 % gesunde Zellen), deshalb Macro-F1 als Hauptmetrik."""
    if multilabel:
        preds = (probs >= 0.5).astype(int)
        m = {
            "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
            "exact_match": accuracy_score(targets, preds),
            "f1_per_class": dict(zip(class_names, f1_score(targets, preds, average=None,
                                                           zero_division=0).tolist())),
        }
        present = [k for k in range(targets.shape[1]) if 0 < targets[:, k].sum() < len(targets)]
        if present:
            m["macro_auc"] = roc_auc_score(targets[:, present], probs[:, present], average="macro")
        return _to_float(m)

    preds = probs.argmax(1)
    labels = list(range(len(class_names)))
    m = {
        "macro_f1": f1_score(targets, preds, labels=labels, average="macro", zero_division=0),
        "accuracy": accuracy_score(targets, preds),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "f1_per_class": dict(zip(class_names, f1_score(targets, preds, labels=labels, average=None,
                                                       zero_division=0).tolist())),
        "confusion_matrix": confusion_matrix(targets, preds, labels=labels).tolist(),
    }
    # Fehlt eine Klasse im Split ganz (z. B. Leukozyten im offiziellen Test-Set), zieht sie
    # macro_f1 mit F1 = 0 runter. macro_f1_present mittelt nur über vorhandene Klassen.
    present = sorted(set(np.unique(targets).tolist()))
    m["macro_f1_present"] = f1_score(targets, preds, labels=present, average="macro", zero_division=0)
    m["missing_classes"] = [class_names[k] for k in labels if k not in present]
    if len(class_names) == 2 and len(np.unique(targets)) == 2:
        m["roc_auc"] = roc_auc_score(targets, probs[:, 1])
    return _to_float(m)


def _to_float(m):
    return {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in m.items()}


def fit(model, train_loader, val_loader, criterion, class_names, epochs=20, lr=1e-3,
        weight_decay=1e-4, multilabel=False, device=None, out_dir=None, checkpoint_extra=None):
    """Trainiert mit AdamW + Cosine-LR und speichert das beste Modell (Val-Macro-F1).

    Gibt die History (Liste von Dicts pro Epoche) zurück. In out_dir landen
    best.pt und history.json.
    """
    device = device or get_device()
    model.to(device)
    criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total, n = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoche {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
        scheduler.step()

        probs, targets = predict(model, val_loader, device, multilabel)
        val = compute_metrics(probs, targets, class_names, multilabel)
        row = {"epoch": epoch, "train_loss": total / n, "val_macro_f1": val["macro_f1"],
               "lr": scheduler.get_last_lr()[0], "seconds": round(time.time() - start, 1)}
        history.append(row)
        print(f"Epoche {epoch:3d}  loss {row['train_loss']:.4f}  val macro-F1 {val['macro_f1']:.4f}"
              f"  ({row['seconds']} s)")

        if out_dir and val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "class_names": list(class_names), **(checkpoint_extra or {})},
                       out_dir / "best.pt")
        if out_dir:
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history
