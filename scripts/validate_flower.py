"""Validacion tecnica del framework Flower (riesgo identificado en 3.3).

Comprueba, con un problema sintetico minimo, que en la version de Flower
instalada el modo simulacion funciona en esta maquina y que una Strategy
personalizada puede:
  1. acceder a la actualizacion individual de cada cliente,
  2. calcular la norma de cada actualizacion,
  3. identificar que cliente envio cada actualizacion.

Uso:  python scripts/validate_flower.py
"""

import sys

import numpy as np

try:
    import flwr
    from flwr.client import ClientApp, NumPyClient
    from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.simulation import run_simulation
except ImportError as exc:
    print(f"FLOWER NO DISPONIBLE: {exc}")
    sys.exit(1)

print(f"flwr {flwr.__version__}")

DIM = 5
NUM_CLIENTS = 4
ROUNDS = 2


class TinyClient(NumPyClient):
    """Cliente sintetico: 'entrena' sumando cid+1 a todos los pesos."""

    def __init__(self, cid: int):
        self.cid = cid

    def fit(self, parameters, config):
        new_params = [p + (self.cid + 1) for p in parameters]
        return new_params, 10, {"cid": self.cid}


def client_fn(context: Context):
    """Crea clientes sinteticos identificados por el partition-id de Flower."""
    cid = int(context.node_config["partition-id"])
    return TinyClient(cid).to_client()


class InspectingFedAvg(FedAvg):
    """Strategy que registra la actualizacion individual de cada cliente."""

    observed = []

    def aggregate_fit(self, server_round, results, failures):
        assert not failures, f"fallos en la ronda {server_round}: {failures}"
        for client_proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            norm = float(np.sqrt(sum((a ** 2).sum() for a in arrays)))
            cid = fit_res.metrics.get("cid")
            self.observed.append((server_round, cid, norm))
        return super().aggregate_fit(server_round, results, failures)


def server_fn(context: Context):
    """Servidor minimo con pesos iniciales conocidos y Strategy inspectora."""
    initial = ndarrays_to_parameters([np.zeros(DIM, dtype=np.float32)])
    strategy = InspectingFedAvg(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_available_clients=NUM_CLIENTS, initial_parameters=initial)
    return ServerAppComponents(strategy=strategy,
                               config=ServerConfig(num_rounds=ROUNDS))


if __name__ == "__main__":
    run_simulation(
        server_app=ServerApp(server_fn=server_fn),
        client_app=ClientApp(client_fn=client_fn),
        num_supernodes=NUM_CLIENTS,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )
    obs = InspectingFedAvg.observed
    print(f"\nactualizaciones individuales observadas: {len(obs)}")
    for round_num, cid, norm in obs:
        print(f"  ronda={round_num} cliente={cid} norma={norm:.3f}")
    expected = ROUNDS * NUM_CLIENTS
    if len(obs) == expected and all(cid is not None for _, cid, _ in obs):
        print("\nVALIDACION OK: acceso a actualizaciones individuales confirmado")
    else:
        print(f"\nVALIDACION FALLIDA: se esperaban {expected} actualizaciones")
        sys.exit(2)
