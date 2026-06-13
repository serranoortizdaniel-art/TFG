"""Sistema de configuracion de experimentos.

Cada experimento queda definido integramente por un fichero YAML (seccion
3.14 de la metodologia). Este modulo carga el YAML, lo combina con los
valores por defecto, valida los campos y genera un nombre canonico de
experimento a partir de la configuracion.

La regla de mantenimiento es deliberada: los YAML declaran el escenario
cientifico y este modulo concentra defaults, validaciones y derivaciones. Asi
se evita que un script de lanzamiento cambie silenciosamente el significado de
un experimento.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml

# Valores por defecto de todos los parametros experimentales. Son tambien la
# documentacion ejecutable de la matriz base; cualquier YAML solo especifica lo
# que cambia respecto a esta linea base.
DEFAULTS: dict = {
    # Identificacion y salida
    "experiment_name": None,        # si es None se genera automaticamente
    "output_dir": "results/raw",
    "seed": 42,
    # Dataset
    "dataset": "fashion_mnist",     # mnist | fashion_mnist | cifar10
    "data_dir": "data",
    # Configuracion federada
    "num_clients": 10,
    "num_rounds": 30,
    "local_epochs": 1,
    "batch_size": 64,
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "eval_batch_size": 512,
    # Particionado
    "partition_type": "iid",        # iid | dirichlet
    "dirichlet_alpha": 0.5,
    "min_partition_size": 10,       # tamano minimo por cliente en dirichlet
    # Ataque
    "attack_type": "none",          # none | label_flipping | sign_flipping | scaling
    "malicious_fraction": 0.0,
    "attack_params": {
        # label flipping
        "flip_mode": "all_to_next",  # all_to_next (c -> c+1 mod C) | targeted
        "source_class": 0,
        "target_class": 6,
        "flip_fraction": 1.0,
        # sign flipping
        "gamma": 1.0,
        # scaling
        "scale_factor": 10.0,
    },
    # Agregacion
    "aggregator": "fedavg",         # fedavg | median | trimmed_mean | krum
    "aggregator_params": {
        "policy": "fixed",           # fixed (conservadora) | oracle
        "trim_beta": 0.3,            # beta de TrimmedMean en politica fija
        "krum_f": 3,                 # f de Krum en politica fija
    },
    # Ejecucion
    "device": "cpu",                # cpu | cuda | auto
    "save_model": False,
    "log_update_norms": True,
    # Flower Simulation Runtime
    "flower": {
        # Los clientes entrenan en CPU por defecto para evitar que varios
        # actores Ray compitan por una GPU pequena. La concurrencia se limita
        # a dos actores con los parametros por defecto (4 CPU totales / 2 por
        # cliente), reduciendo el riesgo de OOM en WSL.
        "client_device": "cpu",
        "num_cpus": 2.0,
        "num_gpus": 0.0,
        "ray_init_args": {
            "include_dashboard": False,
            "num_cpus": 4,
        },
    },
}

VALID_DATASETS = ("mnist", "fashion_mnist", "cifar10")
VALID_PARTITIONS = ("iid", "dirichlet")
VALID_ATTACKS = ("none", "label_flipping", "sign_flipping", "scaling")
VALID_AGGREGATORS = ("fedavg", "median", "trimmed_mean", "krum")
VALID_DEVICES = ("cpu", "cuda", "auto")


def _deep_merge(base: dict, override: dict) -> dict:
    """Combina recursivamente `override` sobre `base` sin mutarlos.

    Se usa tanto al cargar YAML como al aplicar overrides de CLI, por lo que
    debe preservar intactos los defaults compartidos entre experimentos.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Carga un YAML de configuracion, aplica defaults y overrides y valida.

    El dict devuelto es la configuracion efectiva que se guarda junto a los
    resultados. Por eso no quedan parametros implicitos fuera de metadata.
    """
    with open(path, "r", encoding="utf-8") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    unknown = set(user_cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Claves desconocidas en {path}: {sorted(unknown)}")
    cfg = _deep_merge(DEFAULTS, user_cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Comprueba rangos y coherencia entre parametros.

    La validacion distingue errores operativos (configuraciones que no pueden
    ejecutarse) de escenarios metodologicos duros pero permitidos, como llevar
    una defensa cerca de su limite teorico.
    """
    if cfg["dataset"] not in VALID_DATASETS:
        raise ValueError(f"dataset invalido: {cfg['dataset']}")
    if cfg["partition_type"] not in VALID_PARTITIONS:
        raise ValueError(f"partition_type invalido: {cfg['partition_type']}")
    if cfg["attack_type"] not in VALID_ATTACKS:
        raise ValueError(f"attack_type invalido: {cfg['attack_type']}")
    if cfg["aggregator"] not in VALID_AGGREGATORS:
        raise ValueError(f"aggregator invalido: {cfg['aggregator']}")
    if cfg["device"] not in VALID_DEVICES:
        raise ValueError(f"device invalido: {cfg['device']}")
    flower = cfg.get("flower", {})
    client_device = flower.get("client_device", "cpu")
    if client_device not in VALID_DEVICES:
        raise ValueError(f"flower.client_device invalido: {client_device}")
    if float(flower.get("num_cpus", 1.0)) <= 0.0:
        raise ValueError("flower.num_cpus debe ser > 0")
    if float(flower.get("num_gpus", 0.0)) < 0.0:
        raise ValueError("flower.num_gpus debe ser >= 0")
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("device=cuda requiere una GPU CUDA disponible")
    if client_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("flower.client_device=cuda requiere una GPU CUDA disponible")
    if not 0.0 <= cfg["malicious_fraction"] <= 1.0:
        raise ValueError("malicious_fraction debe estar en [0, 1]")
    if cfg["num_clients"] < 2:
        raise ValueError("num_clients debe ser >= 2")
    if cfg["attack_type"] != "none" and cfg["malicious_fraction"] == 0.0:
        raise ValueError(
            "attack_type != none requiere malicious_fraction > 0 "
            "(con 0 maliciosos el ataque no tiene efecto y el experimento "
            "duplicaria la linea base)"
        )
    ap = cfg["attack_params"]
    if cfg["attack_type"] == "label_flipping":
        if ap["flip_mode"] not in ("all_to_next", "targeted"):
            raise ValueError(f"flip_mode invalido: {ap['flip_mode']}")
        if not 0.0 < ap["flip_fraction"] <= 1.0:
            raise ValueError("flip_fraction debe estar en (0, 1]")
        if ap["flip_mode"] == "targeted" and ap["source_class"] == ap["target_class"]:
            raise ValueError("source_class y target_class deben ser distintas")
    gp = cfg["aggregator_params"]
    if gp["policy"] not in ("fixed", "oracle"):
        raise ValueError(f"politica de defensa invalida: {gp['policy']}")
    n = cfg["num_clients"]
    if cfg["aggregator"] == "trimmed_mean":
        beta = resolve_defense_params(cfg)["trim_beta"]
        if not 0.0 <= beta < 0.5:
            raise ValueError(f"trim_beta={beta} debe estar en [0, 0.5)")
    if cfg["aggregator"] == "krum":
        f = resolve_defense_params(cfg)["krum_f"]
        # Esta es la cota operativa minima para que Krum pueda calcular al
        # menos un vecino. La cota teorica de robustez se documenta, pero no se
        # bloquea aqui para poder estudiar escenarios de ruptura.
        if n - f - 2 < 1:
            raise ValueError(
                f"Krum requiere n - f - 2 >= 1 (n={n}, f={f}); "
                "reduce f o aumenta num_clients"
            )


def num_malicious(cfg: dict) -> int:
    """Numero de clientes maliciosos resultante de la fraccion configurada.

    El redondeo permite expresar la matriz en porcentajes y convertirlos al
    numero entero de clientes una sola vez, de forma trazable.
    """
    return int(round(cfg["malicious_fraction"] * cfg["num_clients"]))


def resolve_defense_params(cfg: dict) -> dict:
    """Resuelve los parametros efectivos de la defensa segun la politica.

    - fixed:  cota conservadora constante e independiente del ataque real
              (configuracion principal de la metodologia, seccion 3.9).
    - oracle: el servidor conoce la fraccion real de maliciosos y ajusta
              f y beta exactamente (mejor caso para la defensa).
    """
    gp = cfg["aggregator_params"]
    if gp["policy"] == "fixed":
        return {"policy": "fixed", "krum_f": int(gp["krum_f"]),
                "trim_beta": float(gp["trim_beta"])}
    f = num_malicious(cfg)
    return {"policy": "oracle", "krum_f": f,
            "trim_beta": f / cfg["num_clients"]}


def experiment_name(cfg: dict) -> str:
    """Nombre canonico del experimento (sin la semilla, que va en subcarpeta).

    Ejemplo: fashion_mnist_dir0.5_sign_flipping_m20_krum
    """
    if cfg.get("experiment_name"):
        return cfg["experiment_name"]
    if cfg["partition_type"] == "dirichlet":
        part = f"dir{cfg['dirichlet_alpha']}"
    else:
        part = "iid"
    attack = cfg["attack_type"]
    if attack == "none":
        attack_str = "clean"
    else:
        attack_str = f"{attack}_m{int(round(cfg['malicious_fraction'] * 100))}"
    name = f"{cfg['dataset']}_{part}_{attack_str}_{cfg['aggregator']}"
    if cfg["aggregator"] in ("krum", "trimmed_mean") and \
            cfg["aggregator_params"]["policy"] == "oracle":
        name += "_oracle"
    if cfg["num_clients"] != DEFAULTS["num_clients"]:
        name += f"_c{cfg['num_clients']}"
    if cfg["num_rounds"] != DEFAULTS["num_rounds"]:
        name += f"_r{cfg['num_rounds']}"
    return name


def resolve_device(cfg: dict) -> torch.device:
    """Dispositivo del servidor y de la evaluacion centralizada."""
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


def resolve_client_device(cfg: dict) -> torch.device:
    """Dispositivo usado por los clientes virtuales de Flower."""
    device = cfg.get("flower", {}).get("client_device", "cpu")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_config(cfg: dict, path: str | Path) -> None:
    """Guarda la configuracion efectiva (defaults ya aplicados) junto a los resultados."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
