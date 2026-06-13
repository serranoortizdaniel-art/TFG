"""Envenenamiento de datos mediante label flipping.

Dos modos configurables (attack_params.flip_mode):

  - "all_to_next": cada etiqueta c pasa a (c + 1) mod C. Es la politica
    adoptada por defecto en toda la matriz experimental (opcion A de la
    metodologia, seccion 3.8): no depende de que clases posea el cliente,
    por lo que la intensidad del veneno es uniforme entre IID y non-IID y
    entre fracciones de maliciosos. Corresponde al label flipping estatico
    usado, por ejemplo, en Fang et al. (USENIX Security 2020).

  - "targeted": flip dirigido clase origen -> clase destino sobre una
    fraccion de las muestras de la clase origen (configuracion complementaria
    de la metodologia, p. ej. T-shirt/top -> Shirt en Fashion-MNIST). En
    non-IID un cliente puede carecer de muestras de la clase origen; esa es
    la razon por la que el modo por defecto de la matriz es "all_to_next".
"""

from __future__ import annotations

import numpy as np

from src.attacks.base_attack import BaseAttack


class LabelFlippingAttack(BaseAttack):
    name = "label_flipping"
    kind = "data"

    def poison_labels(self, labels: np.ndarray) -> np.ndarray:
        """Devuelve una copia de las etiquetas locales envenenadas."""
        # Nunca se modifica el array recibido: cada cliente tiene su copia
        # local y los tests comprueban que el dataset base permanece intacto.
        labels = labels.copy()
        mode = self.params.get("flip_mode", "all_to_next")
        flip_fraction = float(self.params.get("flip_fraction", 1.0))

        if mode == "all_to_next":
            # Politica no dirigida: todas las muestras locales son candidatas.
            candidates = np.arange(len(labels))
        elif mode == "targeted":
            # Politica dirigida: solo se tocan muestras de la clase origen.
            candidates = np.where(labels == int(self.params["source_class"]))[0]
        else:
            raise ValueError(f"flip_mode invalido: {mode}")

        if flip_fraction < 1.0 and len(candidates) > 0:
            # El muestreo usa el generador propio del ataque, derivado de la
            # semilla global y del cliente, para no depender del orden de Ray.
            n_flip = int(round(flip_fraction * len(candidates)))
            candidates = self.rng.choice(candidates, size=n_flip, replace=False)

        if mode == "all_to_next":
            labels[candidates] = (labels[candidates] + 1) % self.num_classes
        else:
            labels[candidates] = int(self.params["target_class"])
        return labels
