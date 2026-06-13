"""FedAvg: media ponderada por numero de muestras (McMahan et al., 2017).

Es la linea base del estudio: agregacion estandar sin ninguna proteccion
frente a clientes maliciosos.
"""

from __future__ import annotations

import numpy as np

from src.aggregators.base_aggregator import (BaseAggregator, ClientUpdate,
                                             stack_per_layer)


class FedAvg(BaseAggregator):
    name = "fedavg"

    def aggregate(self, updates: list[ClientUpdate]) -> list[np.ndarray]:
        """Media ponderada por tamano de particion local."""
        total = sum(u.num_examples for u in updates)
        coef = np.array([u.num_examples / total for u in updates])
        result = []
        for stacked in stack_per_layer(updates):
            # El reshape hace que cada coeficiente pese todas las coordenadas
            # de su cliente, independientemente de la dimensionalidad de la capa.
            shape = (len(updates),) + (1,) * (stacked.ndim - 1)
            result.append((stacked * coef.reshape(shape)).sum(axis=0))
        return result
