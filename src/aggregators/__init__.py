"""Registro centralizado de estrategias de agregacion.

El resto del codigo pide un agregador por nombre de configuracion YAML; este
modulo mantiene esa correspondencia en un unico sitio para que ampliar la
matriz experimental no disperse condicionales por la simulacion.
"""

from __future__ import annotations

from src.aggregators.base_aggregator import BaseAggregator, ClientUpdate
from src.aggregators.coordinate_median import CoordinateMedian
from src.aggregators.fedavg import FedAvg
from src.aggregators.krum import Krum
from src.aggregators.trimmed_mean import TrimmedMean

AGGREGATORS = {
    "fedavg": FedAvg,
    "median": CoordinateMedian,
    "trimmed_mean": TrimmedMean,
    "krum": Krum,
}


def build_aggregator(name: str, num_clients: int, defense_params: dict) -> BaseAggregator:
    """Instancia el agregador con sus parametros efectivos.

    `defense_params` debe ser el resultado de `resolve_defense_params(cfg)`,
    es decir, los valores de f y beta ya resueltos segun la politica
    (fija conservadora u oracle) descrita en la seccion 3.9.
    """
    if name not in AGGREGATORS:
        raise ValueError(f"Agregador no soportado: {name}")
    if name == "krum":
        return Krum(num_clients=num_clients, f=defense_params["krum_f"])
    if name == "trimmed_mean":
        return TrimmedMean(num_clients=num_clients, beta=defense_params["trim_beta"])
    return AGGREGATORS[name](num_clients=num_clients)
