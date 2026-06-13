"""Mediana coordinada (Yin et al., 2018).

Para cada parametro del modelo se toma la mediana de los valores enviados
por los clientes. No requiere parametros ni conocer el numero de
maliciosos. Al igual que en la formulacion original, la agregacion no
pondera por numero de muestras.
"""

from __future__ import annotations

import numpy as np

from src.aggregators.base_aggregator import (BaseAggregator, ClientUpdate,
                                             stack_per_layer)


class CoordinateMedian(BaseAggregator):
    name = "median"

    def aggregate(self, updates: list[ClientUpdate]) -> list[np.ndarray]:
        """Mediana independiente por coordenada del modelo."""
        return [np.median(stacked, axis=0)
                for stacked in stack_per_layer(updates)]
