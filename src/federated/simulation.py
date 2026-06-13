"""Simulacion federada basada en Flower.

Flower Simulation Runtime orquesta las rondas y los clientes virtuales. El
resto del proyecto aporta los elementos experimentales del TFG: particionado
de datos, ataques, agregadores robustos, logging, metricas y semillas.
"""

from __future__ import annotations

import torch

try:
    import ray  # noqa: F401
    from flwr.server import ServerConfig
    from flwr.simulation import start_simulation
except ImportError as exc:  # pragma: no cover - solo se alcanza sin Flower/Ray
    ServerConfig = None
    start_simulation = None
    FLOWER_IMPORT_ERROR = exc
else:
    FLOWER_IMPORT_ERROR = None

from src.datasets.loader import DATASET_INFO, TensorBatchLoader, get_targets, load_datasets
from src.datasets.partitioning import build_partitions, class_distribution
from src.federated.adversary import select_malicious
from src.federated.flower_client import client_fn_factory
from src.federated.flower_strategy import RobustFlowerStrategy
from src.federated.parameters import get_weights
from src.logging.experiment_logger import ExperimentLogger
from src.models.cnn import build_model
from src.utils.config import resolve_defense_params, resolve_device, validate_config
from src.utils.seed import derive_seed, numpy_rng, set_global_seed


class FederatedSimulation:
    """Orquesta un experimento Flower completo desde una configuracion YAML.

    La instancia prepara el estado comun del servidor y deja a Flower la
    ejecucion de rondas. No decide factores de la matriz: esos llegan ya
    resueltos en ``cfg``.
    """

    def __init__(self, cfg: dict, verbose: bool = True):
        if FLOWER_IMPORT_ERROR is not None:
            raise ImportError(
                "Flower Simulation Runtime no esta disponible. Ejecuta el "
                "proyecto desde WSL/Linux e instala: "
                "pip install -r requirements-flower-wsl.txt"
            ) from FLOWER_IMPORT_ERROR
        validate_config(cfg)
        self.cfg = cfg
        self.verbose = verbose
        set_global_seed(cfg["seed"])
        self.device = resolve_device(cfg)
        self.num_classes = DATASET_INFO[cfg["dataset"]]["num_classes"]

        # El servidor carga datos para evaluacion centralizada y para guardar
        # la distribucion de clases. Los clientes crean su propia cache en los
        # actores Ray usando la misma semilla y configuracion.
        train_ds, test_ds = load_datasets(cfg["dataset"], cfg["data_dir"])
        targets = get_targets(train_ds)
        partitions = build_partitions(
            targets, cfg, numpy_rng(derive_seed(cfg["seed"], "partition")))
        self.malicious_ids = select_malicious(cfg)

        self.test_loader = TensorBatchLoader(
            images=test_ds.data.to(self.device),
            labels=torch.as_tensor(test_ds.targets, dtype=torch.long,
                                   device=self.device),
            batch_size=cfg["eval_batch_size"],
            shuffle=False,
        )
        self.model = build_model(cfg["dataset"]).to(self.device)
        # Estos pesos iniciales se fijan despues de set_global_seed, por lo que
        # son reproducibles y compartidos por todos los agregadores comparables.
        initial_weights = get_weights(self.model)

        self.logger = ExperimentLogger(cfg, self.num_classes)
        self.defense_params = resolve_defense_params(cfg)
        dist = class_distribution(targets, partitions, self.num_classes)
        self.logger.save_class_distribution(dist, self.malicious_ids)

        self.strategy = RobustFlowerStrategy(
            cfg=cfg,
            defense_params=self.defense_params,
            logger=self.logger,
            initial_weights=initial_weights,
            eval_model=self.model,
            test_loader=self.test_loader,
            device=self.device,
            num_classes=self.num_classes,
        )
        self.logger.set_info(
            engine="flower",
            malicious_clients=self.malicious_ids,
            num_malicious=len(self.malicious_ids),
            aggregator_info=self.strategy.info(),
            defense_policy=self.defense_params,
            device=str(self.device),
            flower=cfg.get("flower", {}),
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def run(self) -> dict:
        cfg = self.cfg
        self._log(
            f"[{self.logger.name} | seed {cfg['seed']}] "
            f"{cfg['num_clients']} clientes "
            f"({len(self.malicious_ids)} maliciosos: {self.malicious_ids}), "
            f"agregador={cfg['aggregator']}, motor=Flower, "
            f"servidor={self.device}, "
            f"clientes={cfg.get('flower', {}).get('client_device', 'cpu')}"
        )
        flower_cfg = cfg.get("flower", {})
        # Recursos por cliente virtual. Separarlos del dispositivo del servidor
        # permite evaluar en GPU y mantener clientes en CPU, o viceversa.
        client_resources = {
            "num_cpus": float(flower_cfg.get("num_cpus", 1.0)),
            "num_gpus": float(flower_cfg.get("num_gpus", 0.0)),
        }
        ray_init_args = flower_cfg.get("ray_init_args", None)
        if ray_init_args is None:
            # Valor conservador para WSL: suficiente para pruebas y reproducible
            # entre lanzamientos si el YAML no especifica otra cosa.
            ray_init_args = {
                "include_dashboard": False,
                "num_cpus": 4,
            }
        start_simulation(
            client_fn=client_fn_factory(cfg),
            num_clients=cfg["num_clients"],
            config=ServerConfig(num_rounds=cfg["num_rounds"]),
            strategy=self.strategy,
            client_resources=client_resources,
            ray_init_args=ray_init_args,
        )
        if self.strategy.last_metrics is None:
            raise RuntimeError("Flower termino sin metricas de evaluacion")
        self.logger.save_confusion_matrix(
            self.strategy.last_metrics["confusion_matrix"])
        if cfg["save_model"]:
            torch.save(self.model.state_dict(),
                       self.logger.run_dir / "model_final.pt")
        self.logger.finish(self.strategy.last_metrics)
        self._log(
            f"  completado: acc final={self.strategy.last_metrics['accuracy']:.4f} "
            f"-> {self.logger.run_dir}"
        )
        return self.strategy.last_metrics
