"""Entrenamiento local de un cliente federado.

Flower se encarga de la orquestacion federada. Esta clase encapsula la parte
local: dado el modelo global de una ronda, entrena sobre la particion del
cliente y devuelve los pesos resultantes. Un cliente malicioso se diferencia
de uno benigno unicamente en dos hooks (seccion 3.3 de la metodologia):

  - sus etiquetas locales pueden haber sido envenenadas al construirse
    (envenenamiento de datos), y/o
  - su actualizacion delta puede transformarse antes de enviarse
    (envenenamiento de modelo).

El resto del comportamiento (entrenamiento, optimizador y datos de otros
clientes) es identico, conforme al modelo de amenaza (3.7).
"""

from __future__ import annotations

import torch

from src.aggregators.base_aggregator import ClientUpdate
from src.attacks.base_attack import BaseAttack
from src.datasets.loader import ClientDataset, TensorBatchLoader
from src.federated.parameters import add, get_weights, set_weights, subtract
from src.models.training import train_epochs
from src.utils.seed import derive_seed, seeded_generator


class LocalClientTrainer:
    """Entrenador local puro, independiente de la API concreta de Flower."""

    def __init__(self, client_id: int, dataset: ClientDataset, cfg: dict,
                 is_malicious: bool = False, attack: BaseAttack | None = None):
        self.client_id = client_id
        self.dataset = dataset
        self.cfg = cfg
        self.is_malicious = is_malicious
        self.attack = attack
        if self.is_malicious and self.attack is not None:
            # El envenenamiento de datos se aplica una sola vez: el adversario
            # controla el dataset local durante todo el entrenamiento.
            self.dataset.labels = self.attack.poison_labels(self.dataset.labels)

    def num_examples(self) -> int:
        return len(self.dataset)

    def fit(self, model: torch.nn.Module, global_weights: list,
            round_num: int, device: torch.device) -> tuple[ClientUpdate, float]:
        """Ejecuta una ronda local y devuelve la actualizacion y la loss media.

        El barajado de lotes se controla con una semilla derivada de
        (seed global, cliente, ronda), de modo que la ejecucion completa es
        reproducible sea cual sea el orden en que se procesen los clientes.
        """
        loader = TensorBatchLoader(
            images=self.dataset.base.data,
            labels=torch.as_tensor(self.dataset.labels, dtype=torch.long,
                                   device=self.dataset.base.data.device),
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            generator=seeded_generator(
                derive_seed(self.cfg["seed"], "shuffle", self.client_id, round_num)),
            indices=torch.from_numpy(self.dataset.indices),
        )
        # Se carga siempre el modelo global de la ronda antes de entrenar. El
        # cliente no conserva pesos entre rondas fuera de lo que dicta FedAvg.
        set_weights(model, global_weights)
        train_loss = train_epochs(
            model, loader,
            epochs=self.cfg["local_epochs"],
            lr=self.cfg["learning_rate"],
            momentum=self.cfg["momentum"],
            weight_decay=self.cfg["weight_decay"],
            device=device,
        )
        local_weights = get_weights(model)
        if self.is_malicious and self.attack is not None:
            # Envenenamiento de modelo: se manipula la actualizacion
            # delta = w_local - w_global y se reconstruyen los pesos a enviar.
            delta = subtract(local_weights, global_weights)
            delta = self.attack.poison_update(delta)
            local_weights = add(global_weights, delta)
        update = ClientUpdate(client_id=self.client_id,
                              num_examples=self.num_examples(),
                              weights=local_weights)
        return update, train_loss
