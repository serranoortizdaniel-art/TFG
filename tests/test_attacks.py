"""Tests de los ataques de envenenamiento de datos y de modelo.

Cubren efectos esperados y propiedades de seguridad local: reproducibilidad
por semilla y ausencia de mutacion de entradas compartidas.
"""

import numpy as np

from src.attacks import build_attack
from src.attacks.label_flipping import LabelFlippingAttack
from src.attacks.no_attack import NoAttack
from src.attacks.scaling_attack import ScalingAttack
from src.attacks.sign_flipping import SignFlippingAttack


class TestNoAttack:
    def test_no_modifica_nada(self):
        attack = NoAttack({}, num_classes=10, seed=0)
        labels = np.array([1, 2, 3])
        delta = [np.array([1.0, -2.0])]
        np.testing.assert_array_equal(attack.poison_labels(labels), labels)
        np.testing.assert_array_equal(attack.poison_update(delta)[0], delta[0])


class TestLabelFlipping:
    def test_all_to_next_desplaza_todas_las_clases(self):
        attack = LabelFlippingAttack({"flip_mode": "all_to_next",
                                      "flip_fraction": 1.0},
                                     num_classes=10, seed=0)
        labels = np.array([0, 1, 5, 9])
        np.testing.assert_array_equal(attack.poison_labels(labels),
                                      [1, 2, 6, 0])

    def test_targeted_solo_cambia_la_clase_origen(self):
        attack = LabelFlippingAttack({"flip_mode": "targeted",
                                      "source_class": 0, "target_class": 6,
                                      "flip_fraction": 1.0},
                                     num_classes=10, seed=0)
        labels = np.array([0, 1, 0, 6, 3])
        np.testing.assert_array_equal(attack.poison_labels(labels),
                                      [6, 1, 6, 6, 3])

    def test_flip_fraction_parcial(self):
        attack = LabelFlippingAttack({"flip_mode": "all_to_next",
                                      "flip_fraction": 0.5},
                                     num_classes=10, seed=0)
        labels = np.zeros(100, dtype=int)
        poisoned = attack.poison_labels(labels)
        assert (poisoned == 1).sum() == 50
        assert (poisoned == 0).sum() == 50

    def test_no_muta_el_array_original(self):
        attack = LabelFlippingAttack({"flip_mode": "all_to_next",
                                      "flip_fraction": 1.0},
                                     num_classes=10, seed=0)
        labels = np.array([0, 1, 2])
        attack.poison_labels(labels)
        np.testing.assert_array_equal(labels, [0, 1, 2])

    def test_reproducible_con_misma_semilla(self):
        params = {"flip_mode": "all_to_next", "flip_fraction": 0.3}
        labels = np.arange(10).repeat(20)
        a = LabelFlippingAttack(params, 10, seed=5).poison_labels(labels)
        b = LabelFlippingAttack(params, 10, seed=5).poison_labels(labels)
        np.testing.assert_array_equal(a, b)


class TestSignFlipping:
    def test_invierte_el_signo_con_gamma(self):
        attack = SignFlippingAttack({"gamma": 2.0}, num_classes=10, seed=0)
        delta = [np.array([1.0, -3.0]), np.array([[0.5]])]
        result = attack.poison_update(delta)
        np.testing.assert_allclose(result[0], [-2.0, 6.0])
        np.testing.assert_allclose(result[1], [[-1.0]])

    def test_no_toca_las_etiquetas(self):
        attack = SignFlippingAttack({"gamma": 1.0}, num_classes=10, seed=0)
        labels = np.array([1, 2, 3])
        np.testing.assert_array_equal(attack.poison_labels(labels), labels)


class TestScaling:
    def test_escala_la_actualizacion(self):
        attack = ScalingAttack({"scale_factor": 10.0}, num_classes=10, seed=0)
        delta = [np.array([1.0, -0.5])]
        np.testing.assert_allclose(attack.poison_update(delta)[0], [10.0, -5.0])


class TestRegistro:
    def test_build_attack_construye_cada_tipo(self):
        for name, cls in [("none", NoAttack),
                          ("label_flipping", LabelFlippingAttack),
                          ("sign_flipping", SignFlippingAttack),
                          ("scaling", ScalingAttack)]:
            attack = build_attack(name, {}, num_classes=10, seed=1)
            assert isinstance(attack, cls)
