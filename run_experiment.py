"""Ejecuta un experimento federado individual definido por un YAML.

Uso:
    python run_experiment.py --config configs/base_fedavg_iid_fmnist.yaml
    python run_experiment.py --config ... --seed 123
    python run_experiment.py --config ... --override num_rounds=5 attack_params.gamma=2.0

Cualquier parametro de la configuracion puede sobreescribirse desde la
linea de comandos con --override clave=valor (claves anidadas con punto).
La configuracion efectiva completa queda guardada junto a los resultados.
"""

from __future__ import annotations

import argparse

import yaml

from src.federated.simulation import FederatedSimulation
from src.utils.config import load_config


def parse_overrides(pairs: list[str]) -> dict:
    """Convierte ["a.b=1", "c=true"] en un dict anidado con tipos de YAML.

    ``yaml.safe_load`` permite que los overrides de CLI respeten tipos
    numericos, booleanos y listas sin escribir parsers ad hoc.
    """
    overrides: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override invalido (esperado clave=valor): {pair}")
        key, raw_value = pair.split("=", 1)
        value = yaml.safe_load(raw_value)
        node = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return overrides


def main() -> None:
    """Punto de entrada para ejecutar un unico escenario experimental."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="Ruta al fichero YAML de configuracion")
    parser.add_argument("--seed", type=int, default=None,
                        help="Sobreescribe la semilla de la configuracion")
    parser.add_argument("--override", nargs="*", default=[],
                        metavar="CLAVE=VALOR",
                        help="Overrides puntuales (claves anidadas con punto)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suprime el progreso por consola")
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = load_config(args.config, overrides)
    simulation = FederatedSimulation(cfg, verbose=not args.quiet)
    simulation.run()


if __name__ == "__main__":
    main()
