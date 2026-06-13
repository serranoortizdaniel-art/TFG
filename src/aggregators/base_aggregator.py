"""Interfaz comun de los agregadores.

Cada agregador recibe la lista completa de actualizaciones individuales de
los clientes de la ronda (requisito imprescindible de la metodologia,
seccion 3.3) y produce los nuevos pesos del modelo global.

Los agregadores trabajan directamente sobre los pesos enviados por los
clientes (w_i = w_global + delta_i). Para FedAvg, la mediana coordinada y
Trimmed Mean esto es equivalente a agregar las actualizaciones delta_i y
sumarlas al modelo global, porque w_global es constante en la ronda; para
Krum, seleccionar el w_i de menor puntuacion equivale a seleccionar su
delta_i. Se elige esta formulacion porque es la misma interfaz que expone
una Strategy de Flower (lista de (num_examples, pesos) por cliente).

Importante: el agregador NO recibe informacion de que clientes son
maliciosos. Esa informacion solo existe en el codigo de instrumentacion
(logging de normas), nunca en la defensa.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClientUpdate:
    """Actualizacion observada por el servidor al final de una ronda.

    ``weights`` contiene los pesos locales completos enviados por Flower, no
    solo el delta. Esta representacion mantiene el mismo contrato para todos
    los agregadores y evita conversiones innecesarias dentro de la Strategy.
    """
    client_id: int
    num_examples: int
    weights: list[np.ndarray]  # pesos locales tras el entrenamiento


def stack_per_layer(updates: list[ClientUpdate]) -> list[np.ndarray]:
    """Apila los pesos por capa: una matriz (num_clients, *shape) por capa.

    Todos los clientes deben compartir arquitectura y orden de ``state_dict``;
    si esa precondicion fallara, el experimento dejaria de ser comparable.
    """
    num_layers = len(updates[0].weights)
    return [np.stack([u.weights[layer] for u in updates], axis=0)
            for layer in range(num_layers)]


def flatten_weights(weights: list[np.ndarray]) -> np.ndarray:
    """Concatena todas las capas en un unico vector 1-D.

    Se usa para distancias entre actualizaciones, donde la estructura por capa
    no importa y conviene tratar cada modelo local como un punto del espacio.
    """
    return np.concatenate([layer.ravel() for layer in weights])


class BaseAggregator(ABC):
    name = "base"

    def __init__(self, num_clients: int):
        self.num_clients = num_clients

    @abstractmethod
    def aggregate(self, updates: list[ClientUpdate]) -> list[np.ndarray]:
        """Combina una ronda completa y devuelve los nuevos pesos globales."""

    def info(self) -> dict:
        """Parametros del agregador que se guardan como metadatos trazables."""
        return {"name": self.name}
