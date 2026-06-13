"""Envenenamiento de modelo mediante inversion de signo.

El cliente malicioso entrena normalmente sobre sus datos y despues sustituye
su actualizacion delta = w_local - w_global por -gamma * delta antes de
enviarla (seccion 3.8). Con gamma = 1 (valor base) el cliente empuja el
modelo global exactamente en la direccion opuesta a su gradiente local.
"""

from __future__ import annotations

import numpy as np

from src.attacks.base_attack import BaseAttack


class SignFlippingAttack(BaseAttack):
    name = "sign_flipping"
    kind = "model"

    def poison_update(self, delta: list[np.ndarray]) -> list[np.ndarray]:
        """Invierte y escala la actualizacion local capa a capa."""
        gamma = float(self.params.get("gamma", 1.0))
        return [-gamma * layer for layer in delta]
