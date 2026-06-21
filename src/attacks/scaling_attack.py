"""Envenenamiento de modelo mediante escalado de la actualizacion.

El cliente malicioso amplifica su actualizacion local por un factor
constante (delta -> factor * delta) para dominar la media ponderada del
servidor. Es la forma mas simple de "model replacement / boosting"
(Bagdasaryan et al., 2020). Variante de alta intensidad del
envenenamiento de modelo evaluada en la matriz experimental.
"""

from __future__ import annotations

import numpy as np

from src.attacks.base_attack import BaseAttack


class ScalingAttack(BaseAttack):
    name = "scaling"
    kind = "model"

    def poison_update(self, delta: list[np.ndarray]) -> list[np.ndarray]:
        """Amplifica la actualizacion local sin tocar el entrenamiento previo."""
        factor = float(self.params.get("scale_factor", 10.0))
        return [factor * layer for layer in delta]
