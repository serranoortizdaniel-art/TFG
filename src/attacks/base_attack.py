"""Interfaz comun de los ataques.

Un ataque encapsula las dos capacidades del adversario del modelo de
amenaza (seccion 3.7):

  1. Envenenamiento de datos: `poison_labels` transforma las etiquetas
     locales del cliente malicioso antes del entrenamiento. Se aplica una
     unica vez, al construir el cliente, porque el adversario controla el
     dataset local durante todo el entrenamiento.

  2. Envenenamiento de modelo: `poison_update` transforma la actualizacion
     local delta = w_local - w_global despues del entrenamiento y antes de
     enviarla al servidor.

Ambos hooks tienen implementacion neutra por defecto, de modo que cada
ataque concreto solo sobreescribe el que le corresponde. Los ataques solo
se instancian para clientes maliciosos: el codigo de los clientes benignos
no contiene ninguna logica de ataque.
"""

from __future__ import annotations

import numpy as np

from src.utils.seed import numpy_rng


class BaseAttack:
    name = "base"
    # Tipo de capacidad usada, util para logging/analisis:
    #   "data" -> envenena datos, "model" -> envenena la actualizacion
    kind = "none"

    def __init__(self, params: dict, num_classes: int, seed: int):
        self.params = params
        self.num_classes = num_classes
        self.rng = numpy_rng(seed)

    def poison_labels(self, labels: np.ndarray) -> np.ndarray:
        """Hook de envenenamiento de datos. Devuelve las etiquetas locales."""
        return labels

    def poison_update(self, delta: list[np.ndarray]) -> list[np.ndarray]:
        """Hook de envenenamiento de modelo. Devuelve la actualizacion."""
        return delta
