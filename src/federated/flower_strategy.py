"""Strategy personalizada de Flower para los agregadores del TFG.

Es el punto donde se separan orquestacion y metodologia: Flower entrega
resultados de clientes, y esta Strategy reconstruye actualizaciones,
instrumenta normas, aplica el agregador configurado y evalua el modelo global.
"""

from __future__ import annotations

import time

import numpy as np
import torch

try:
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.strategy import FedAvg
except ImportError as exc:  # pragma: no cover - solo se alcanza sin Flower
    ndarrays_to_parameters = None
    parameters_to_ndarrays = None
    FedAvg = object
    FLOWER_IMPORT_ERROR = exc
else:
    FLOWER_IMPORT_ERROR = None

from src.aggregators import build_aggregator
from src.aggregators.base_aggregator import ClientUpdate
from src.federated.parameters import l2_norm, set_weights, subtract
from src.utils.metrics import evaluate


class RobustFlowerStrategy(FedAvg):
    """Strategy Flower que delega la agregacion en los agregadores propios.

    Flower se encarga de orquestar clientes virtuales y rondas federadas. Esta
    Strategy conserva la parte experimental del TFG: acceso a actualizaciones
    individuales, registro de normas y seleccion del agregador configurado.
    """

    def __init__(
        self,
        cfg: dict,
        defense_params: dict,
        logger,
        initial_weights: list[np.ndarray],
        eval_model: torch.nn.Module,
        test_loader,
        device: torch.device,
        num_classes: int,
    ):
        if FLOWER_IMPORT_ERROR is not None:
            raise ImportError("Flower no esta instalado") from FLOWER_IMPORT_ERROR
        self.cfg = cfg
        self.logger = logger
        self.global_weights = [layer.copy() for layer in initial_weights]
        self.eval_model = eval_model
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.aggregator = build_aggregator(
            cfg["aggregator"], cfg["num_clients"], defense_params)
        self.last_metrics: dict | None = None
        self._round_started_at: dict[int, float] = {}
        self._last_train_loss: float | None = None
        self._last_agg_time: float | None = None

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=cfg["num_clients"],
            min_available_clients=cfg["num_clients"],
            min_evaluate_clients=0,
            accept_failures=False,
            initial_parameters=ndarrays_to_parameters(initial_weights),
            inplace=False,
        )

    def configure_fit(self, server_round, parameters, client_manager):
        """Configura la ronda y pasa el numero de ronda a cada cliente."""
        self._round_started_at[server_round] = time.perf_counter()
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        # Enviamos la ronda al cliente para derivar el barajado local de forma
        # determinista: seed global + cliente + ronda.
        for _, ins in fit_ins:
            ins.config["server_round"] = server_round
        return fit_ins

    def aggregate_fit(self, server_round, results, failures):
        """Recibe todas las actualizaciones individuales y aplica la defensa."""
        if not results:
            return None, {}
        if failures:
            # La matriz experimental asume participacion completa. Aceptar
            # fallos cambiaria el numero efectivo de clientes de una ronda.
            return None, {}

        updates: list[ClientUpdate] = []
        train_losses = []
        malicious_by_client = {}
        for client_proxy, fit_res in results:
            # Ray puede devolver resultados en orden no determinista. El id del
            # cliente viaja en metrics para reconstruir una lista estable.
            metrics = fit_res.metrics or {}
            client_id = int(metrics.get("client_id", getattr(client_proxy, "cid", 0)))
            is_malicious = bool(int(metrics.get("is_malicious", 0)))
            malicious_by_client[client_id] = is_malicious
            if "train_loss" in metrics:
                train_losses.append(float(metrics["train_loss"]))
            updates.append(ClientUpdate(
                client_id=client_id,
                num_examples=fit_res.num_examples,
                weights=parameters_to_ndarrays(fit_res.parameters),
            ))
        updates.sort(key=lambda update: update.client_id)

        if self.cfg["log_update_norms"]:
            for update in updates:
                # La norma se calcula respecto al modelo global anterior. Es
                # instrumentacion, no informacion disponible para la defensa.
                delta_norm = l2_norm(subtract(update.weights, self.global_weights))
                self.logger.norms.log_update(
                    server_round,
                    update.client_id,
                    malicious_by_client.get(update.client_id, False),
                    update.num_examples,
                    delta_norm,
                )

        t_agg = time.perf_counter()
        self.global_weights = self.aggregator.aggregate(updates)
        self._last_agg_time = time.perf_counter() - t_agg
        self._last_train_loss = (
            float(np.mean(train_losses)) if train_losses else None
        )
        return ndarrays_to_parameters(self.global_weights), {
            "aggregator": self.cfg["aggregator"],
        }

    def evaluate(self, server_round, parameters):
        """Evaluacion centralizada del modelo global al final de la ronda."""
        weights = parameters_to_ndarrays(parameters)
        set_weights(self.eval_model, weights)
        metrics = evaluate(
            self.eval_model, self.test_loader, self.device, self.num_classes)
        self.last_metrics = metrics
        round_time = None
        if server_round in self._round_started_at:
            round_time = time.perf_counter() - self._round_started_at[server_round]
        self.logger.metrics.log_round(
            server_round,
            metrics,
            train_loss=self._last_train_loss if server_round > 0 else None,
            round_time=round_time,
            agg_time=self._last_agg_time if server_round > 0 else None,
        )
        return metrics["loss"], {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }

    def info(self) -> dict:
        """Metadatos del agregador usado por la Strategy."""
        return self.aggregator.info()
