"""Registro centralizado de ataques disponibles.

La configuracion YAML solo expone nombres estables. La construccion concreta
queda aqui para que los clientes federados no conozcan detalles de cada ataque.
"""

from __future__ import annotations

from src.attacks.base_attack import BaseAttack
from src.attacks.label_flipping import LabelFlippingAttack
from src.attacks.no_attack import NoAttack
from src.attacks.scaling_attack import ScalingAttack
from src.attacks.sign_flipping import SignFlippingAttack

ATTACKS = {
    "none": NoAttack,
    "label_flipping": LabelFlippingAttack,
    "sign_flipping": SignFlippingAttack,
    "scaling": ScalingAttack,
}


def build_attack(attack_type: str, params: dict, num_classes: int,
                 seed: int) -> BaseAttack:
    """Instancia el ataque configurado.

    `seed` debe derivarse de la semilla global y del cliente concreto
    (`derive_seed(seed, "attack", cid)`) para que cualquier muestreo interno
    del ataque sea reproducible e independiente por cliente.
    """
    if attack_type not in ATTACKS:
        raise ValueError(f"Ataque no soportado: {attack_type}")
    return ATTACKS[attack_type](params=params, num_classes=num_classes, seed=seed)
