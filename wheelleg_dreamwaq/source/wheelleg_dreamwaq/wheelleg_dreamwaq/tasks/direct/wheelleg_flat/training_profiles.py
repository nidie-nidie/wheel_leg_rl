from __future__ import annotations

from dataclasses import replace
from typing import Any

from .commands import STOP_REVERSE_COMMAND_PRACTICE_VERSION, command_contract_payload
from .leg_reversal import LegReversalConfig, reversal_contract_payload

REVERSAL_PROFILE_BUDGETS = {
    "stop_reverse_reversal12_v1": 12.0,
    "stop_reverse_reversal8_v1": 8.0,
}
TASK_PROFILE_NAMES = ("legacy_v1", "stop_reverse_v1", "stop_reverse_smooth_v1",
                      *REVERSAL_PROFILE_BUDGETS)


def apply_task_profile(env_cfg: Any, name: str) -> None:
    if name not in TASK_PROFILE_NAMES:
        raise ValueError(f"Unknown DreamWaQ task profile: {name}")
    enabled = name != "legacy_v1"
    env_cfg.commands = replace(env_cfg.commands, hold_for_episode=not enabled,
        practice_schedule=STOP_REVERSE_COMMAND_PRACTICE_VERSION if enabled else "disabled")
    env_cfg.reward_weights = replace(env_cfg.reward_weights,
        tracking_vx=2.0 if enabled else 1.0, tracking_vx_enhance=2.0 if enabled else 1.0,
        action_smooth=-0.02 if name == "stop_reverse_smooth_v1" else -0.01)
    env_cfg.leg_reversal = LegReversalConfig(
        enabled=name in REVERSAL_PROFILE_BUDGETS,
        budget_per_second=REVERSAL_PROFILE_BUDGETS.get(name, 12.0))
    command_contract_payload(env_cfg.commands)


def task_profile_from_contract(contract: dict) -> str:
    commands = contract["task"]["commands"]
    schedule = commands.get("practice_schedule", "disabled")
    reversal = contract["task"].get("leg_reversal")
    if reversal is not None:
        if schedule != STOP_REVERSE_COMMAND_PRACTICE_VERSION or contract["task"]["reward_weights"]["action_smooth"] != -0.01:
            raise ValueError("Reversal profile requires the original stop_reverse_v1 rewards/schedule")
        for name, budget in REVERSAL_PROFILE_BUDGETS.items():
            expected = reversal_contract_payload(LegReversalConfig(enabled=True, budget_per_second=budget))
            if reversal == expected:
                return name
        raise ValueError("Unknown saved reversal profile contract")
    if schedule == "disabled":
        return "legacy_v1"
    if schedule == STOP_REVERSE_COMMAND_PRACTICE_VERSION:
        smooth = contract["task"]["reward_weights"]["action_smooth"]
        if smooth == -0.01:
            return "stop_reverse_v1"
        if smooth == -0.02:
            return "stop_reverse_smooth_v1"
        raise ValueError(f"Unknown saved action_smooth weight: {smooth}")
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
