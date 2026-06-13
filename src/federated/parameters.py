"""Conversion entre modelos de PyTorch y listas de arrays de numpy.

Es el mismo formato de intercambio que usa Flower (NumPyClient): una lista
ordenada de ndarrays, una por entrada del state_dict. Los agregadores y los
ataques de envenenamiento de modelo operan sobre este formato.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch


def get_weights(model: torch.nn.Module) -> list[np.ndarray]:
    """Extrae los pesos del modelo como lista de ndarrays (copia en CPU)."""
    return [param.detach().cpu().numpy().copy()
            for param in model.state_dict().values()]


def set_weights(model: torch.nn.Module, weights: list[np.ndarray]) -> None:
    """Carga una lista de ndarrays en el modelo.

    El orden debe coincidir exactamente con ``model.state_dict()``; ``strict``
    deja fallar pronto si una arquitectura no corresponde al experimento.
    """
    state_dict = OrderedDict(
        (key, torch.tensor(value))
        for key, value in zip(model.state_dict().keys(), weights)
    )
    model.load_state_dict(state_dict, strict=True)


def subtract(a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
    """Resta por capas: a - b (p. ej. delta = w_local - w_global)."""
    return [x - y for x, y in zip(a, b)]


def add(a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
    """Suma por capas: a + b (p. ej. w = w_global + delta)."""
    return [x + y for x, y in zip(a, b)]


def l2_norm(weights: list[np.ndarray]) -> float:
    """Norma L2 del vector aplanado usada para instrumentar actualizaciones."""
    return float(np.sqrt(sum(float((layer ** 2).sum()) for layer in weights)))
