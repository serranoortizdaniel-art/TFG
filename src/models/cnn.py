"""Arquitecturas de red usadas en el estudio.

La arquitectura es una variable controlada (seccion 3.8): se fija una CNN
sencilla para MNIST/Fashion-MNIST en todos los experimentos comparables, y
una CNN algo mayor para la extension opcional con CIFAR-10.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.datasets.loader import DATASET_INFO


class SmallCNN(nn.Module):
    """CNN sencilla para imagenes 28x28 en escala de grises.

    Dos bloques convolucionales (32 y 64 filtros) con max-pooling y una capa
    oculta totalmente conectada de 128 unidades. Es la arquitectura tipica
    de los trabajos de ataques a FL sobre MNIST/Fashion-MNIST: suficiente
    para aprender la tarea sin encarecer la matriz experimental.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CifarCNN(nn.Module):
    """CNN para CIFAR-10 (32x32 RGB), extension opcional del estudio."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(dataset: str) -> nn.Module:
    """Construye la arquitectura asociada al dataset configurado."""
    info = DATASET_INFO[dataset]
    if dataset in ("mnist", "fashion_mnist"):
        return SmallCNN(info["num_classes"], info["in_channels"])
    if dataset == "cifar10":
        return CifarCNN(info["num_classes"], info["in_channels"])
    raise ValueError(f"Dataset no soportado: {dataset}")
