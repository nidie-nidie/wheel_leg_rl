"""Dependency-light contracts for applying one fitted model in Isaac/Isaac Lab."""

from .leg_dm_adapter import load_leg_dm_contract
from .wheel_lk_adapter import load_wheel_lk_contract

__all__ = ["load_leg_dm_contract", "load_wheel_lk_contract"]
