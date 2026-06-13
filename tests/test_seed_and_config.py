"""Tests del control de semillas y del sistema de configuracion.

Protegen dos contratos transversales del TFG: repetibilidad de ejecuciones y
rechazo temprano de configuraciones ambiguas o imposibles.
"""

import numpy as np
import pytest
import torch

from src.models.cnn import SmallCNN
from src.utils.config import (DEFAULTS, _deep_merge, experiment_name,
                              num_malicious, resolve_defense_params,
                              validate_config)
from src.utils.seed import derive_seed, set_global_seed


class TestSemillas:
    def test_misma_semilla_mismo_modelo(self):
        set_global_seed(42)
        model_a = SmallCNN()
        set_global_seed(42)
        model_b = SmallCNN()
        for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
            assert torch.equal(p_a, p_b)

    def test_semillas_distintas_modelos_distintos(self):
        set_global_seed(42)
        model_a = SmallCNN()
        set_global_seed(43)
        model_b = SmallCNN()
        assert any(not torch.equal(a, b) for a, b in
                   zip(model_a.parameters(), model_b.parameters()))

    def test_derive_seed_estable_y_sensible(self):
        assert derive_seed(42, "client", 0) == derive_seed(42, "client", 0)
        assert derive_seed(42, "client", 0) != derive_seed(42, "client", 1)
        assert derive_seed(42, "client", 0) != derive_seed(43, "client", 0)


class TestConfig:
    def base(self, **overrides) -> dict:
        return _deep_merge(DEFAULTS, overrides)

    def test_defaults_validos(self):
        validate_config(self.base())

    def test_ataque_sin_maliciosos_rechazado(self):
        with pytest.raises(ValueError):
            validate_config(self.base(attack_type="sign_flipping",
                                      malicious_fraction=0.0))

    def test_num_malicious_redondeo(self):
        assert num_malicious(self.base(malicious_fraction=0.2)) == 2
        assert num_malicious(self.base(malicious_fraction=0.0)) == 0
        assert num_malicious(self.base(malicious_fraction=0.3)) == 3

    def test_politica_fija_vs_oracle(self):
        fixed = resolve_defense_params(self.base(
            malicious_fraction=0.1,
            attack_type="sign_flipping"))
        assert fixed["krum_f"] == 3 and fixed["trim_beta"] == 0.3
        oracle = resolve_defense_params(self.base(
            malicious_fraction=0.1, attack_type="sign_flipping",
            aggregator_params={"policy": "oracle"}))
        assert oracle["krum_f"] == 1
        assert oracle["trim_beta"] == pytest.approx(0.1)

    def test_cota_operativa_de_krum_se_valida(self):
        # La cota operativa (n - f - 2 >= 1) si es un error duro. La cota
        # teorica (n > 2f + 2) no se bloquea: la metodologia preve ejecutar
        # el caso de ruptura con 40% de maliciosos (seccion 3.7).
        with pytest.raises(ValueError):
            validate_config(self.base(
                aggregator="krum", num_clients=4,
                aggregator_params={"policy": "fixed", "krum_f": 3,
                                   "trim_beta": 0.3}))
        validate_config(self.base(
            aggregator="krum", num_clients=6,
            aggregator_params={"policy": "fixed", "krum_f": 3,
                               "trim_beta": 0.3}))

    def test_nombre_canonico(self):
        cfg = self.base(dataset="fashion_mnist", partition_type="dirichlet",
                        dirichlet_alpha=0.5, attack_type="sign_flipping",
                        malicious_fraction=0.2, aggregator="krum")
        assert experiment_name(cfg) == \
            "fashion_mnist_dir0.5_sign_flipping_m20_krum"
        clean = self.base(partition_type="iid")
        assert experiment_name(clean) == "fashion_mnist_iid_clean_fedavg"
