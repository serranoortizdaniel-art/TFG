"""Krum (Blanchard et al., 2017).

Para cada cliente i se calcula una puntuacion: la suma de las distancias
euclideas al cuadrado a sus n - f - 2 vecinos mas cercanos (excluyendose a
si mismo). El servidor adopta como modelo global los pesos del cliente con
menor puntuacion. La intuicion es que un cliente malicioso que envia una
actualizacion alejada del grupo mayoritario obtiene una puntuacion alta y
nunca resulta seleccionado.

Garantia teorica: tolera hasta f maliciosos si n > 2f + 2. Con la
configuracion base del estudio (n = 10, f = 3) se cumple 10 > 8, es decir,
el 30% de maliciosos es el caso limite (seccion 3.7).
"""

from __future__ import annotations

import numpy as np

from src.aggregators.base_aggregator import (BaseAggregator, ClientUpdate,
                                             flatten_weights)


class Krum(BaseAggregator):
    name = "krum"

    def __init__(self, num_clients: int, f: int):
        super().__init__(num_clients)
        self.f = int(f)

    def aggregate(self, updates: list[ClientUpdate]) -> list[np.ndarray]:
        """Selecciona el cliente con menor puntuacion Krum."""
        n = len(updates)
        num_neighbors = n - self.f - 2
        if num_neighbors < 1:
            raise ValueError(
                f"Krum requiere n - f - 2 >= 1 (n={n}, f={self.f})")
        vectors = np.stack([flatten_weights(u.weights) for u in updates])
        # Matriz simetrica de distancias euclideas al cuadrado entre clientes.
        # Krum usa distancias al cuadrado; no se toma raiz porque no altera el
        # orden y evita trabajo innecesario.
        sq_dists = np.sum(
            (vectors[:, None, :] - vectors[None, :, :]) ** 2, axis=2)
        scores = np.empty(n)
        for i in range(n):
            others = np.delete(sq_dists[i], i)
            others.sort()
            scores[i] = others[:num_neighbors].sum()
        # np.argmin rompe empates escogiendo el primer indice. La Strategy
        # ordena previamente por client_id para que ese desempate sea estable.
        selected = int(np.argmin(scores))
        self.last_scores = scores          # instrumentacion para analisis
        self.last_selected = updates[selected].client_id
        return [layer.copy() for layer in updates[selected].weights]

    def info(self) -> dict:
        return {"name": self.name, "f": self.f}
