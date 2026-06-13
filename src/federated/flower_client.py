"""Cliente Flower del entorno experimental.

Cada instancia es un ``NumPyClient``: recibe los pesos globales como listas
de ndarrays, entrena localmente en PyTorch y devuelve los pesos actualizados.
Los clientes maliciosos se determinan con la misma semilla y aplican los
mismos ataques que el resto del proyecto.

Este modulo es un adaptador: mantiene la API esperada por Flower y delega la
logica experimental en ``LocalClientTrainer`` para que pueda probarse sin
levantar una simulacion.
"""

from __future__ import annotations

from copy import deepcopy

import torch

try:
    import flwr as fl
    from flwr.common import Context
except ImportError as exc:  # pragma: no cover - solo se alcanza sin Flower
    fl = None
    Context = object
    FLOWER_IMPORT_ERROR = exc
else:
    FLOWER_IMPORT_ERROR = None

from src.attacks import build_attack
from src.datasets.loader import ClientDataset, DATASET_INFO, get_targets, load_datasets
from src.datasets.partitioning import build_partitions
from src.federated.adversary import select_malicious
from src.federated.client import LocalClientTrainer
from src.models.cnn import build_model
from src.utils.config import resolve_client_device
from src.utils.seed import derive_seed, numpy_rng


_CLIENT_CACHE: dict[tuple, dict] = {}


def _client_cache_key(cfg: dict, device: torch.device) -> tuple:
    """Clave que separa estados de cliente incompatibles dentro de un actor Ray."""
    return (
        cfg["dataset"],
        cfg["data_dir"],
        cfg["num_clients"],
        cfg["partition_type"],
        cfg["dirichlet_alpha"],
        cfg["min_partition_size"],
        cfg["seed"],
        str(device),
    )


def _client_state(cfg: dict, device: torch.device) -> dict:
    """Carga y particiona datos una vez por actor Ray y configuracion.

    Flower puede crear clientes virtuales repetidamente. La cache evita
    recargar datasets en cada ``client_fn`` sin mezclar ejecuciones: la clave
    incluye semilla, particion, dataset y dispositivo.
    """
    key = _client_cache_key(cfg, device)
    if key not in _CLIENT_CACHE:
        train_ds, _ = load_datasets(cfg["dataset"], cfg["data_dir"])
        train_ds.data = train_ds.data.to(device)
        targets = get_targets(train_ds)
        partitions = build_partitions(
            targets, cfg, numpy_rng(derive_seed(cfg["seed"], "partition")))
        _CLIENT_CACHE[key] = {
            "train_ds": train_ds,
            "partitions": partitions,
            "malicious_ids": select_malicious(cfg),
            "num_classes": DATASET_INFO[cfg["dataset"]]["num_classes"],
        }
    return _CLIENT_CACHE[key]


class FlowerFederatedClient(fl.client.NumPyClient if fl is not None else object):
    """Cliente Flower que delega el entrenamiento en ``LocalClientTrainer``."""

    def __init__(self, client_id: int, cfg: dict):
        if FLOWER_IMPORT_ERROR is not None:
            raise ImportError("Flower no esta instalado") from FLOWER_IMPORT_ERROR
        self.client_id = client_id
        # Cada cliente recibe su propia copia de la configuracion para evitar
        # mutaciones accidentales entre actores Ray.
        self.cfg = deepcopy(cfg)
        self.device = resolve_client_device(self.cfg)

        state = _client_state(self.cfg, self.device)
        self.model = build_model(self.cfg["dataset"]).to(self.device)
        is_malicious = client_id in state["malicious_ids"]
        attack = None
        if is_malicious:
            attack = build_attack(
                self.cfg["attack_type"],
                self.cfg["attack_params"],
                state["num_classes"],
                derive_seed(self.cfg["seed"], "attack", client_id),
            )
        dataset = ClientDataset(state["train_ds"], state["partitions"][client_id])
        self.inner = LocalClientTrainer(
            client_id=client_id,
            dataset=dataset,
            cfg=self.cfg,
            is_malicious=is_malicious,
            attack=attack,
        )

    def get_parameters(self, config):
        """Devuelve los pesos iniciales cuando Flower los solicita."""
        del config
        from src.federated.parameters import get_weights

        return get_weights(self.model)

    def fit(self, parameters, config):
        """Ejecuta una ronda local y devuelve metadatos de instrumentacion."""
        round_num = int(config.get("server_round", 1))
        update, train_loss = self.inner.fit(
            self.model, parameters, round_num, self.device)
        return update.weights, update.num_examples, {
            "client_id": self.client_id,
            "is_malicious": int(self.inner.is_malicious),
            "train_loss": float(train_loss),
        }

    def evaluate(self, parameters, config):
        # La evaluacion principal del TFG es centralizada en el servidor.
        del parameters, config
        return 0.0, self.inner.num_examples(), {}


def client_fn_factory(cfg: dict):
    """Devuelve el ``client_fn`` requerido por Flower Simulation Runtime."""

    def client_fn(context: Context):
        # Flower identifica clientes virtuales con partition-id; el resto del
        # proyecto usa ese valor como client_id estable.
        node_cfg = getattr(context, "node_config", {})
        client_id = int(node_cfg.get("partition-id", 0))
        return FlowerFederatedClient(client_id, cfg).to_client()

    return client_fn
