"""Modelle: ein eigenes CNN von Grund auf und torchvision-Modelle als Vergleich.

    from src.models import build_model
    model = build_model("simplecnn", num_classes=6)
"""
import torch
from torch import nn
from torchvision import models


def conv_block(c_in, c_out):
    """2x (Conv 3x3 -> BatchNorm -> ReLU), danach halbe Auflösung."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """Eigenes CNN: 4 Conv-Blöcke (32-64-128-256 Kanäle), Global Average Pooling, Dropout.

    Durch das Global Average Pooling funktioniert es mit jeder Eingabegrösse, also für
    Zellen (96 px) und für ganze Ausstriche (384 px). Bei grossen Bildern `depth` erhöhen,
    damit das rezeptive Feld reicht.
    """

    def __init__(self, num_classes, width=32, depth=4, dropout=0.3):
        super().__init__()
        channels = [3] + [width * 2 ** i for i in range(depth)]
        self.features = nn.Sequential(*[conv_block(a, b) for a, b in zip(channels, channels[1:])])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels[-1], num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_model(name, num_classes, pretrained=False):
    """'simplecnn' (eigenes Modell), 'simplecnn-deep' (6 Blöcke, für ganze Bilder),
    'resnet18' oder 'mobilenet_v3_small' (klein, für Smartphone / Distillation)."""
    if name == "simplecnn":
        return SimpleCNN(num_classes)
    if name == "simplecnn-deep":
        return SimpleCNN(num_classes, depth=6)
    if name == "resnet18":
        model = models.resnet18(weights="DEFAULT" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unbekanntes Modell: {name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path, device="cpu"):
    """best.pt aus scripts/train.py laden -> (Modell im eval-Modus, Checkpoint-Dict)."""
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model"], len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
