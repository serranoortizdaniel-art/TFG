"""Particionado del dataset de entrenamiento entre clientes.

Dos esquemas (seccion 3.10):
  - IID: reparto aleatorio estratificado y aproximadamente equilibrado.
  - non-IID: particionado Dirichlet por clase con parametro alpha, esquema
    estandar en la literatura de FL para controlar la heterogeneidad.

Ambos son deterministas dada una semilla, lo que garantiza que todos los
escenarios comparables (mismo seed) usan exactamente la misma particion.
"""

from __future__ import annotations

import numpy as np


def partition_iid(targets: np.ndarray, num_clients: int,
                  rng: np.random.Generator) -> list[np.ndarray]:
    """Reparto estratificado: cada clase se divide a partes iguales.

    Todos los clientes reciben aproximadamente el mismo numero de muestras
    de cada clase, con asignacion aleatoria de las muestras concretas.
    """
    partitions: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
    for cls in np.unique(targets):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        for client_id, chunk in enumerate(np.array_split(cls_idx, num_clients)):
            partitions[client_id].append(chunk)
    result = []
    for parts in partitions:
        idx = np.concatenate(parts)
        rng.shuffle(idx)
        result.append(idx)
    return result


def partition_dirichlet(targets: np.ndarray, num_clients: int, alpha: float,
                        rng: np.random.Generator, min_size: int = 10,
                        max_retries: int = 100) -> list[np.ndarray]:
    """Particionado heterogeneo mediante distribucion Dirichlet.

    Para cada clase c se muestrea un vector de proporciones
    p_c ~ Dir(alpha, ..., alpha) que determina que fraccion de las muestras
    de esa clase recibe cada cliente. Valores bajos de alpha producen
    particiones mas heterogeneas. Se repite el muestreo hasta que todos los
    clientes tienen al menos `min_size` muestras, para evitar clientes
    vacios que romperian el entrenamiento local.
    """
    num_samples = len(targets)
    classes = np.unique(targets)
    for _ in range(max_retries):
        client_indices: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_idx = np.where(targets == cls)[0]
            rng.shuffle(cls_idx)
            # Se reparte cada clase por separado. Asi el parametro alpha
            # controla heterogeneidad de etiquetas, no solo tamanos totales.
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            # puntos de corte acumulados sobre las muestras de la clase
            cuts = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
            for client_id, chunk in enumerate(np.split(cls_idx, cuts)):
                client_indices[client_id].append(chunk)
        sizes = [sum(len(c) for c in parts) for parts in client_indices]
        if min(sizes) >= min_size:
            result = []
            for parts in client_indices:
                idx = np.concatenate(parts)
                rng.shuffle(idx)
                result.append(idx)
            # Invariante experimental: ninguna muestra se pierde ni se duplica.
            assert sum(len(r) for r in result) == num_samples
            return result
    raise RuntimeError(
        f"No se logro una particion Dirichlet con minimo {min_size} "
        f"muestras por cliente tras {max_retries} intentos (alpha={alpha})"
    )


def build_partitions(targets: np.ndarray, cfg: dict,
                     rng: np.random.Generator) -> list[np.ndarray]:
    """Construye la particion segun la configuracion del experimento."""
    if cfg["partition_type"] == "iid":
        return partition_iid(targets, cfg["num_clients"], rng)
    return partition_dirichlet(targets, cfg["num_clients"],
                               cfg["dirichlet_alpha"], rng,
                               min_size=cfg["min_partition_size"])


def class_distribution(targets: np.ndarray, partitions: list[np.ndarray],
                       num_classes: int) -> np.ndarray:
    """Matriz (num_clients x num_classes) con el numero de muestras por clase.

    Se guarda en cada experimento para documentar la heterogeneidad de la
    particion (tabla/figura de la memoria, seccion 3.10).
    """
    dist = np.zeros((len(partitions), num_classes), dtype=int)
    for client_id, idx in enumerate(partitions):
        values, counts = np.unique(targets[idx], return_counts=True)
        dist[client_id, values] = counts
    return dist
