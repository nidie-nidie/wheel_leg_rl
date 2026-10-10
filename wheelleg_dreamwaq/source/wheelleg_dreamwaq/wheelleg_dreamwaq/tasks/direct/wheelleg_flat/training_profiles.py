from __future__ import annotations

from dataclasses import replace
from typing import Any

from .commands import STOP_REVERSE_COMMAND_PRACTICE_VERSION, command_contract_payload

TASK_PROFILE_NAMES = ("legacy_v1", "stop_reverse_v1")


def apply_task_profile(env_cfg: Any, name: str) -> None:
    if name not in TASK_PROFILE_NAMES:
        raise ValueError(f"Unknown DreamWaQ task profile: {name}")
    enabled = name == "stop_reverse_v1"
    env_cfg.commands = replace(env_cfg.commands, hold_for_episode=not enabled,
        practice_schedule=STOP_REVERSE_COMMAND_PRACTICE_VERSION if enabled else "disabled")
    env_cfg.reward_weights = replace(env_cfg.reward_weights,
        tracking_vx=2.0 if enabled else 1.0, tracking_vx_enhance=2.0 if enabled else 1.0)
    command_contract_payload(env_cfg.commands)


def task_profile_from_contract(contract: dict) -> str:
    commands = contract["task"]["commands"]
    schedule = commands.get("practice_schedule", "disabled")
    if schedule == "disabled":
        return "legacy_v1"
    if schedule == STOP_REVERSE_COMMAND_PRACTICE_VERSION:
        return "stop_reverse_v1"
    raise ValueError(f"Unknown saved task profile schedule: {schedule}")


def resolve_task_profile(requested: str | None, source_contract: dict | None = None) -> str:
    inferred = "legacy_v1" if source_contract is None else task_profile_from_contract(source_contract)
    if requested is not None and requested not in TASK_PROFILE_NAMES:
        raise ValueError(f"Unknown DreamWaQ task profile: {requested}")
    if source_contract is not None and requested is not None and requested != inferred:
        raise ValueError(f"Requested task profile {requested} differs from saved profile {inferred}")
    return inferred if requested is None else requested


def disable_command_practice(env_cfg: Any) -> None:
    """Disable runtime practice after validating the saved training identity."""
    env_cfg.commands = replace(env_cfg.commands, hold_for_episode=True, practice_schedule="disabled")
