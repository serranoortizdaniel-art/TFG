"""Entrenamiento local compartido por el baseline centralizado y FL.

Mantener un unico bucle evita que diferencias accidentales de optimizador o
loss contaminen la comparacion entre referencia centralizada y escenarios
federados.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_epochs(model: torch.nn.Module, loader: DataLoader,
                 epochs: int, lr: float, momentum: float,
                 weight_decay: float, device: torch.device) -> float:
    """Entrena el modelo `epochs` epocas con SGD y devuelve la loss media.

    El optimizador se crea localmente en cada llamada: en FL cada cliente
    parte del modelo global de la ronda y no conserva estado del optimizador
    entre rondas (configuracion estandar de FedAvg).
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    total_loss = 0.0
    total_samples = 0
    for _ in range(epochs):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
    return total_loss / max(total_samples, 1)
