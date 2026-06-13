"""Tests unitarios de los agregadores con vectores sinteticos de salida
conocida (fase 7 de la hoja de ruta). Un error en un agregador invalidaria
toda la matriz experimental, por lo que cada propiedad clave se verifica
de forma aislada antes de ejecutar experimentos.
"""

import numpy as np
import pytest

from src.aggregators import build_aggregator
from src.aggregators.base_aggregator import ClientUpdate
from src.aggregators.coordinate_median import CoordinateMedian
from src.aggregators.fedavg import FedAvg
from src.aggregators.krum import Krum
from src.aggregators.trimmed_mean import TrimmedMean


def make_update(client_id: int, values, num_examples: int = 10) -> ClientUpdate:
    """Crea una actualizacion con dos capas (vector y matriz) a partir de
    un escalar o array base, para comprobar la agregacion multi-capa."""
    base = np.asarray(values, dtype=np.float64)
    return ClientUpdate(
        client_id=client_id,
        num_examples=num_examples,
        weights=[base.copy(), np.stack([base, 2 * base])],
    )


class TestFedAvg:
    def test_media_simple_con_pesos_iguales(self):
        updates = [make_update(0, [1.0, 2.0]), make_update(1, [3.0, 4.0])]
        result = FedAvg(num_clients=2).aggregate(updates)
        np.testing.assert_allclose(result[0], [2.0, 3.0])
        np.testing.assert_allclose(result[1], [[2.0, 3.0], [4.0, 6.0]])

    def test_ponderacion_por_num_muestras(self):
        # cliente 0 con 30 muestras, cliente 1 con 10: media ponderada 3:1
        updates = [make_update(0, [0.0], num_examples=30),
                   make_update(1, [4.0], num_examples=10)]
        result = FedAvg(num_clients=2).aggregate(updates)
        np.testing.assert_allclose(result[0], [1.0])


class TestCoordinateMedian:
    def test_mediana_de_vectores_conocidos(self):
        updates = [make_update(0, [1.0, 10.0]),
                   make_update(1, [2.0, 20.0]),
                   make_update(2, [100.0, -5.0])]
        result = CoordinateMedian(num_clients=3).aggregate(updates)
        np.testing.assert_allclose(result[0], [2.0, 10.0])

    def test_outlier_extremo_no_altera_la_mediana(self):
        updates = [make_update(0, [1.0]), make_update(1, [2.0]),
                   make_update(2, [3.0]), make_update(3, [1e9])]
        result = CoordinateMedian(num_clients=4).aggregate(updates)
        np.testing.assert_allclose(result[0], [2.5])


class TestTrimmedMean:
    def test_descarta_exactamente_los_extremos(self):
        # n=5, beta=0.2 -> b=1: descarta el mayor y el menor por coordenada
        values = [[-100.0], [1.0], [2.0], [3.0], [500.0]]
        updates = [make_update(i, v) for i, v in enumerate(values)]
        result = TrimmedMean(num_clients=5, beta=0.2).aggregate(updates)
        np.testing.assert_allclose(result[0], [2.0])

    def test_recorte_por_coordenada_independiente(self):
        # el recorte se aplica coordenada a coordenada, no por cliente entero
        updates = [make_update(0, [0.0, 9.0]),
                   make_update(1, [5.0, 5.0]),
                   make_update(2, [9.0, 0.0])]
        result = TrimmedMean(num_clients=3, beta=1 / 3).aggregate(updates)
        np.testing.assert_allclose(result[0], [5.0, 5.0])

    def test_beta_cero_equivale_a_media(self):
        updates = [make_update(0, [1.0]), make_update(1, [3.0])]
        result = TrimmedMean(num_clients=2, beta=0.0).aggregate(updates)
        np.testing.assert_allclose(result[0], [2.0])

    def test_recorte_excesivo_lanza_error(self):
        updates = [make_update(0, [1.0]), make_update(1, [2.0])]
        with pytest.raises(ValueError):
            TrimmedMean(num_clients=2, beta=0.5).aggregate(updates)


class TestKrum:
    def test_selecciona_el_vector_correcto_ante_outlier_evidente(self):
        # 5 clientes agrupados cerca del origen y un outlier lejano:
        # Krum debe elegir uno de los agrupados, nunca el outlier
        rng = np.random.default_rng(0)
        cluster = [make_update(i, rng.normal(0, 0.01, size=4)) for i in range(5)]
        outlier = make_update(5, np.full(4, 50.0))
        krum = Krum(num_clients=6, f=1)
        result = krum.aggregate(cluster + [outlier])
        assert krum.last_selected != 5
        selected = [u for u in cluster if u.client_id == krum.last_selected][0]
        np.testing.assert_allclose(result[0], selected.weights[0])

    def test_devuelve_los_pesos_exactos_del_seleccionado(self):
        # con todos identicos menos uno, selecciona uno del grupo mayoritario
        group = [make_update(i, [1.0, 1.0]) for i in range(4)]
        attacker = make_update(4, [9.0, -9.0])
        krum = Krum(num_clients=5, f=1)
        result = krum.aggregate(group + [attacker])
        np.testing.assert_allclose(result[0], [1.0, 1.0])

    def test_cota_teorica_invalida_lanza_error(self):
        updates = [make_update(i, [1.0]) for i in range(4)]
        with pytest.raises(ValueError):
            Krum(num_clients=4, f=2).aggregate(updates)  # 4 - 2 - 2 < 1


class TestTodosLosAgregadores:
    def test_actualizaciones_identicas_devuelven_la_media(self):
        """Con todas las actualizaciones identicas, cualquier agregador debe
        devolver exactamente ese valor comun (test de la hoja de ruta)."""
        updates = [make_update(i, [3.0, -1.5, 0.25]) for i in range(10)]
        for name in ("fedavg", "median", "trimmed_mean", "krum"):
            agg = build_aggregator(name, 10, {"krum_f": 3, "trim_beta": 0.3})
            result = agg.aggregate([make_update(u.client_id,
                                                [3.0, -1.5, 0.25])
                                    for u in updates])
            np.testing.assert_allclose(result[0], [3.0, -1.5, 0.25],
                                       err_msg=f"agregador {name}")

    def test_no_mutan_las_actualizaciones_de_entrada(self):
        for name in ("fedavg", "median", "trimmed_mean", "krum"):
            updates = [make_update(i, [float(i)]) for i in range(10)]
            originals = [[layer.copy() for layer in u.weights] for u in updates]
            agg = build_aggregator(name, 10, {"krum_f": 3, "trim_beta": 0.3})
            agg.aggregate(updates)
            for update, original in zip(updates, originals):
                for layer, orig_layer in zip(update.weights, original):
                    np.testing.assert_array_equal(layer, orig_layer)
