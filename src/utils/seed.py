"""Control centralizado de aleatoriedad.

La semilla global de cada experimento fija todas las fuentes de aleatoriedad
(torch, numpy, random). Para procesos que necesitan su propia secuencia
reproducible (barajado local de un cliente en una ronda concreta, seleccion
de clientes maliciosos, particionado de datos...) se derivan sub-semillas
deterministas con `derive_seed`, de forma que cada componente es reproducible
de manera independiente y no depende del orden en que se consuma el generador
global.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Fija la semilla de random, numpy y torch (CPU y GPU).

    Con ``deterministic=True`` se desactiva el autotuner de cuDNN y se fuerza
    el uso de algoritmos deterministas en convoluciones. No se activa
    ``torch.use_deterministic_algorithms(True)`` porque algunas operaciones
    no disponen de implementacion determinista en GPU; el determinismo
    obtenido con cuDNN determinista es suficiente para la reproducibilidad
    a nivel de metricas que exige la metodologia (seccion 3.14).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_seed(*parts) -> int:
    """Deriva una sub-semilla determinista de 32 bits a partir de una tupla.

    Ejemplo: ``derive_seed(seed, "client", cid, "round", r)`` produce la
    semilla del barajado local del cliente `cid` en la ronda `r`. Se usa
    SHA-256 para que la derivacion sea estable entre plataformas y versiones
    de Python (`hash()` no lo es).
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "little")


def seeded_generator(seed: int) -> torch.Generator:
    """Generador de torch con semilla fija para barajados locales."""
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def numpy_rng(seed: int) -> np.random.Generator:
    """Generador moderno de numpy con semilla fija e independiente."""
    return np.random.default_rng(int(seed))
