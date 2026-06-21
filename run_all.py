"""Lanzador reanudable de la matriz experimental completa (fase 8).

Genera todas las configuraciones del nucleo experimental (seccion 3.12):

  particiones   : IID y Dirichlet alpha=0.5
  ataques       : sin ataque, label flipping, sign flipping y scaling
  maliciosos    : 10%, 20%, 30% y 40% (0% sin ataque; scaling al 20% y 30%)
  agregadores   : FedAvg, mediana coordinada, Trimmed Mean, Krum
  politicas     : fija (todos los agregadores) y oracle (Krum y Trimmed Mean)
  semillas      : 42, 123, 2024 (3 en el nucleo, seccion 3.11)

y las ejecuta secuencialmente, omitiendo las que ya estan completadas
(marcador `status: completed` en metadata.json), de modo que el proceso
puede interrumpirse y reanudarse sin repetir trabajo.

Uso:
    python run_all.py --dry-run                  # listar el plan sin ejecutar
    python run_all.py                            # ejecutar la matriz completa
    python run_all.py --filter krum              # solo configs cuyo nombre contiene 'krum'
    python run_all.py --seeds 42                 # una unica semilla
    python run_all.py --dataset mnist --rounds 5 # matriz reducida de depuracion
    python run_all.py --device cuda --client-device cuda --client-gpus 1 --client-cpus 4

Ejecutar desde el entorno WSL con Flower instalado:
    ~/.venvs/pfg-flower/bin/python3 run_all.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.federated.simulation import FederatedSimulation
from src.logging.experiment_logger import ExperimentLogger
from src.utils.config import DEFAULTS, _deep_merge, experiment_name

CORE_SEEDS = [42, 123, 2024]
PARTITIONS = [("iid", None), ("dirichlet", 0.5)]
AGGREGATORS = ["fedavg", "median", "trimmed_mean", "krum"]
# Ataques del nucleo: label flipping y sign flipping en cuatro fracciones
# crecientes (hasta el 40%, que rebasa el supuesto K > 2f+2 de Krum), mas el
# scaling como variante de envenenamiento de modelo de intensidad amplificada,
# estudiada en un subconjunto representativo de fracciones (seccion 3.6).
ATTACKS = (
    [("none", 0.0)]
    + [(attack, frac)
       for attack in ("label_flipping", "sign_flipping")
       for frac in (0.1, 0.2, 0.3, 0.4)]
    + [("scaling", frac) for frac in (0.2, 0.3)]
)


def build_matrix(dataset: str, rounds: int | None = None) -> list[dict]:
    """Construye la lista de configuraciones base, todavia sin semilla.

    La semilla se anade despues para que la misma configuracion cientifica
    pueda repetirse de forma controlada y agruparse bajo el mismo nombre.

    Cada agregador se evalua con la politica de defensa fija conservadora
    (seccion 3.9). Krum y Trimmed Mean se evaluan ademas con la politica
    oracle, que ajusta sus parametros a la fraccion real de adversarios y
    permite medir el coste de no conocerla; la mediana coordinada no tiene
    parametros, asi que la politica no la afecta y no se duplica.
    """
    configs = []
    for partition, alpha in PARTITIONS:
        for aggregator in AGGREGATORS:
            policies = ["fixed"]
            if aggregator in ("krum", "trimmed_mean"):
                policies.append("oracle")
            for policy in policies:
                for attack, fraction in ATTACKS:
                    overrides = {
                        "dataset": dataset,
                        "partition_type": partition,
                        "aggregator": aggregator,
                        "attack_type": attack,
                        "malicious_fraction": fraction,
                        "aggregator_params": {"policy": policy},
                    }
                    if alpha is not None:
                        overrides["dirichlet_alpha"] = alpha
                    if rounds is not None:
                        overrides["num_rounds"] = rounds
                    configs.append(_deep_merge(DEFAULTS, overrides))
    return configs


def build_runtime_overrides(args: argparse.Namespace) -> dict:
    """Recoge opciones de ejecucion que no cambian el escenario cientifico.

    Dispositivo y recursos Ray afectan al rendimiento y a la estabilidad, pero
    no deben alterar el nombre canonico ni los factores de la matriz.
    """
    overrides: dict = {}
    flower: dict = {}
    ray_init_args: dict = {}

    if args.device is not None:
        overrides["device"] = args.device
    if args.client_device is not None:
        flower["client_device"] = args.client_device
    if args.client_cpus is not None:
        flower["num_cpus"] = args.client_cpus
    if args.client_gpus is not None:
        flower["num_gpus"] = args.client_gpus
    if args.ray_cpus is not None:
        ray_init_args["num_cpus"] = args.ray_cpus
    if args.ray_dashboard:
        ray_init_args["include_dashboard"] = True

    if ray_init_args:
        flower["ray_init_args"] = ray_init_args
    if flower:
        overrides["flower"] = flower
    return overrides


def main() -> None:
    """Genera el plan, omite ejecuciones completadas y lanza las pendientes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fashion_mnist")
    parser.add_argument("--seeds", type=int, nargs="*", default=CORE_SEEDS)
    parser.add_argument("--rounds", type=int, default=None,
                        help="Sobreescribe num_rounds (para matrices de prueba)")
    parser.add_argument("--filter", default=None,
                        help="Ejecuta solo los experimentos cuyo nombre contiene esta subcadena")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximo de ejecuciones pendientes a lanzar")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra el plan sin ejecutar nada")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"),
                        default=None,
                        help="Dispositivo del servidor/evaluacion centralizada")
    parser.add_argument("--client-device", choices=("cpu", "cuda", "auto"),
                        default=None,
                        help="Dispositivo de los clientes virtuales Flower")
    parser.add_argument("--client-cpus", type=float, default=None,
                        help="CPU Ray reservadas por cliente virtual")
    parser.add_argument("--client-gpus", type=float, default=None,
                        help="GPU Ray reservadas por cliente virtual")
    parser.add_argument("--ray-cpus", type=int, default=None,
                        help="CPU totales expuestas a Ray")
    parser.add_argument("--ray-dashboard", action="store_true",
                        help="Activa el dashboard local de Ray")
    args = parser.parse_args()

    runtime_overrides = build_runtime_overrides(args)
    configs = [
        _deep_merge(cfg, runtime_overrides)
        for cfg in build_matrix(args.dataset, args.rounds)
    ]
    if args.filter:
        configs = [c for c in configs if args.filter in experiment_name(c)]

    # Plan reanudable: cada par (config, semilla) se decide mirando
    # metadata.json, no por presencia de carpetas parciales.
    plan = []
    for cfg in configs:
        for seed in args.seeds:
            run_cfg = _deep_merge(cfg, {"seed": seed})
            run_dir = (Path(run_cfg["output_dir"]) / experiment_name(run_cfg)
                       / f"seed_{seed}")
            done = ExperimentLogger.is_completed(run_dir, required_engine="flower")
            plan.append((run_cfg, seed, done))

    total = len(plan)
    completed = sum(1 for _, _, done in plan if done)
    pending = [(cfg, seed) for cfg, seed, done in plan if not done]
    if args.limit is not None:
        pending = pending[:args.limit]

    print(f"Matriz: {len(configs)} configuraciones x {len(args.seeds)} semillas "
          f"= {total} ejecuciones ({completed} completadas, "
          f"{len(pending)} pendientes a lanzar)")
    if runtime_overrides:
        print(f"Overrides de ejecucion: {runtime_overrides}")
    if args.dry_run:
        for cfg, seed, done in plan:
            estado = "OK " if done else "..."
            print(f"  [{estado}] {experiment_name(cfg)} / seed_{seed}")
        return

    # Log global de la tirada: complementa los metadatos de cada ejecucion y
    # permite estimar tiempos restantes durante sesiones largas.
    log_path = Path("results/run_all_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    log_fh = open(log_path, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_fh)
    if new_log:
        log_writer.writerow(["timestamp", "experiment", "seed", "status",
                             "duration_s", "final_accuracy"])

    durations = []
    failures = 0
    for i, (run_cfg, seed) in enumerate(pending, start=1):
        name = experiment_name(run_cfg)
        print(f"\n=== [{i}/{len(pending)}] {name} / seed_{seed} ===")
        t0 = time.perf_counter()
        try:
            metrics = FederatedSimulation(run_cfg, verbose=True).run()
            duration = time.perf_counter() - t0
            durations.append(duration)
            log_writer.writerow([datetime.now(timezone.utc).isoformat(), name,
                                 seed, "completed", f"{duration:.1f}",
                                 f"{metrics['accuracy']:.4f}"])
        except Exception:
            duration = time.perf_counter() - t0
            failures += 1
            print(f"FALLO en {name} / seed_{seed}:\n{traceback.format_exc()}")
            log_writer.writerow([datetime.now(timezone.utc).isoformat(), name,
                                 seed, "failed", f"{duration:.1f}", ""])
        log_fh.flush()
        if durations:
            mean_d = sum(durations) / len(durations)
            remaining = (len(pending) - i) * mean_d
            print(f"    media por ejecucion: {mean_d/60:.1f} min | "
                  f"ETA restante: {remaining/3600:.2f} h")

    log_fh.close()
    print(f"\nTerminado: {len(pending) - failures} completadas, "
          f"{failures} fallidas. Registro en {log_path}")


if __name__ == "__main__":
    main()
