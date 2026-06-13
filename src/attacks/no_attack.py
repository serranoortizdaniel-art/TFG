"""Ataque nulo: comportamiento identico al de un cliente benigno."""

from __future__ import annotations

from src.attacks.base_attack import BaseAttack


class NoAttack(BaseAttack):
    name = "none"
    kind = "none"
