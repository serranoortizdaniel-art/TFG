"""Tests del particionado IID y Dirichlet.

Verifican invariantes experimentales baratos pero criticos: usar todas las
muestras una sola vez, reproducibilidad y diferencia real de heterogeneidad.
"""

import numpy as np

from src.datasets.partitioning import (class_distribution, partition_dirichlet,
                                       partition_iid)
from src.utils.seed import numpy_rng


def fake_targets(num_classes=10, per_class=600) -> np.ndarray:
    targets = np.repeat(np.arange(num_classes), per_class)
    numpy_rng(0).shuffle(targets)
    return targets


class TestParticionIID:
    def test_usa_todas_las_muestras_exactamente_una_vez(self):
        targets = fake_targets()
        parts = partition_iid(targets, 10, numpy_rng(1))
        joined = np.sort(np.concatenate(parts))
        np.testing.assert_array_equal(joined, np.arange(len(targets)))

    def test_distribucion_de_clases_equilibrada(self):
        targets = fake_targets(per_class=600)
        parts = partition_iid(targets, 10, numpy_rng(1))
        dist = class_distribution(targets, parts, 10)
        # estratificado: cada cliente recibe 600/10 = 60 muestras por clase
        assert dist.min() >= 59 and dist.max() <= 61

    def test_reproducible_con_misma_semilla(self):
        targets = fake_targets()
        parts_a = partition_iid(targets, 10, numpy_rng(7))
        parts_b = partition_iid(targets, 10, numpy_rng(7))
        for a, b in zip(parts_a, parts_b):
            np.testing.assert_array_equal(a, b)


class TestParticionDirichlet:
    def test_usa_todas_las_muestras_exactamente_una_vez(self):
        targets = fake_targets()
        parts = partition_dirichlet(targets, 10, alpha=0.5, rng=numpy_rng(1))
        joined = np.sort(np.concatenate(parts))
        np.testing.assert_array_equal(joined, np.arange(len(targets)))

    def test_respeta_tamano_minimo(self):
        targets = fake_targets()
        parts = partition_dirichlet(targets, 10, alpha=0.1,
                                    rng=numpy_rng(3), min_size=10)
        assert min(len(p) for p in parts) >= 10

    def test_reproducible_con_misma_semilla(self):
        targets = fake_targets()
        parts_a = partition_dirichlet(targets, 10, 0.5, numpy_rng(7))
        parts_b = partition_dirichlet(targets, 10, 0.5, numpy_rng(7))
        for a, b in zip(parts_a, parts_b):
            np.testing.assert_array_equal(a, b)

    def test_mas_heterogenea_que_iid(self):
        """Con alpha bajo, la desviacion de la distribucion de clases entre
        clientes debe ser claramente mayor que en el reparto IID."""
        targets = fake_targets()
        iid = class_distribution(
            targets, partition_iid(targets, 10, numpy_rng(1)), 10)
        dirichlet = class_distribution(
            targets, partition_dirichlet(targets, 10, 0.1, numpy_rng(1)), 10)
        assert dirichlet.std(axis=0).mean() > 5 * iid.std(axis=0).mean()
