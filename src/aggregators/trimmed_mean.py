"""Trimmed Mean coordinada (Yin et al., 2018).

Para cada parametro del modelo se ordenan los n valores recibidos, se
descartan los b = floor(beta * n) mayores y los b menores, y se promedian
los n - 2b restantes. Con beta >= f/n la media recortada ofrece garantias
frente a f clientes bizantinos. Requiere 2b < n.
"""

from __future__ import annotations

import numpy as np

from src.aggregators.base_aggregator import (BaseAggregator, ClientUpdate,
                                             stack_per_layer)


class TrimmedMean(BaseAggregator):
    name = "trimmed_mean"

    def __init__(self, num_clients: int, beta: float):
        super().__init__(num_clients)
        self.beta = float(beta)

    def aggregate(self, updates: list[ClientUpdate]) -> list[np.ndarray]:
        """Media recortada independiente por coordenada."""
        n = len(updates)
        b = int(np.floor(self.beta * n))
        if 2 * b >= n:
            raise ValueError(
                f"TrimmedMean: beta={self.beta} recorta 2*{b} >= {n} clientes")
        result = []
        for stacked in stack_per_layer(updates):
            # El recorte se hace por coordenada, no descartando clientes
            # completos: un mismo cliente puede ser extremo en unas capas y no
            # en otras, tal y como define la regla coordinada.
            ordered = np.sort(stacked, axis=0)
            if b > 0:
                ordered = ordered[b:n - b]
            result.append(ordered.mean(axis=0))
        return result

    def info(self) -> dict:
        return {"name": self.name, "beta": self.beta}
