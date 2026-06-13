"""Evaluacion del modelo global con las metricas definidas en la metodologia.

Las mismas metricas se usan en el entrenamiento centralizado (fase 1) y en
todos los escenarios federados, siempre sobre el conjunto de prueba
centralizado, comun e identico (seccion 3.13).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader,
             device: torch.device, num_classes: int) -> dict:
    """Evalua el modelo y devuelve todas las metricas del estudio.

    Devuelve un dict con:
      - loss: cross-entropy media sobre el conjunto de prueba
      - accuracy: exactitud global
      - macro_f1: F1 macro (mismo peso a todas las clases)
      - per_class_accuracy: lista con el recall de cada clase
      - confusion_matrix: matriz de confusion (filas = clase real)
    """
    model.eval()
    model.to(device)
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    n = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        n += targets.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    labels = list(range(num_classes))
    cm = confusion_matrix(targets_np, preds, labels=labels)
    # Division segura: si una clase no aparece en un conjunto sintetico de
    # prueba, su accuracy por clase se registra como 0 en lugar de NaN.
    per_class = np.divide(np.diag(cm), cm.sum(axis=1),
                          out=np.zeros(num_classes, dtype=float),
                          where=cm.sum(axis=1) > 0)
    return {
        "loss": total_loss / n,
        "accuracy": float((preds == targets_np).mean()),
        "macro_f1": float(f1_score(targets_np, preds, labels=labels,
                                   average="macro", zero_division=0)),
        "per_class_accuracy": per_class.tolist(),
        "confusion_matrix": cm,
    }
