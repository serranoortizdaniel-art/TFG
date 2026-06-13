"""Utilidades del modelo de amenaza federado.

El servidor de instrumentacion conoce que clientes son maliciosos para poder
etiquetar logs y graficas. Los agregadores robustos no reciben esta lista.
"""

from __future__ import annotations

from src.utils.config import num_malicious
from src.utils.seed import derive_seed, numpy_rng


def select_malicious(cfg: dict) -> list[int]:
    """Seleccion determinista (por semilla) de los clientes maliciosos.

    Ordenar los ids hace que los metadatos y las trazas sean estables, aunque
    el muestreo interno de numpy no tenga por que devolverlos ordenados.
    """
    n_mal = num_malicious(cfg)
    if n_mal == 0:
        return []
    rng = numpy_rng(derive_seed(cfg["seed"], "malicious_selection"))
    ids = rng.choice(cfg["num_clients"], size=n_mal, replace=False)
    return sorted(int(i) for i in ids)
