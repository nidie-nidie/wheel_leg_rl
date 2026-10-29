from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
from typing import Any

from wheelleg_dreamwaq.schemas.heading_control import (
    HeadingHoldConfig, RAW_BASELINE_PROFILE, RAW_BASELINE_PROFILES, RUNTIME_GEOMETRY,
    heading_profile_name,
)
from wheelleg_dreamwaq.schemas.normalization import NormalizationV2, RawActionNormalizationV1

from .commands import STOP_REVERSE_COMMAND_PRACTICE_VERSION, command_contract_payload
from .commands import CommandRanges
from .control import ControlLimits
from .leg_acceleration import (
    LegAccelerationConfig,
    acceleration_contract_payload,
)
from .leg_reversal import LegReversalConfig, reversal_contract_payload
from .rewards import (RewardWeights, SplitSmoothRewardWeights,
                      SPLIT_SMOOTH_REWARD_SCHEMA_VERSION, RawSplitStraightRewardWeights,
                      RAW_SPLIT_STRAIGHT_REWARD_VERSION)

REVERSAL_PROFILE_BUDGETS = {
    "stop_reverse_reversal12_v1": 12.0,
    "stop_reverse_reversal8_v1": 8.0,
}
ACCELERATION_PROFILE_FRACTIONS = {
    "stop_reverse_accel80_v1": 0.80,
    "stop_reverse_accel75_v1": 0.75,
    "stop_reverse_accel70_v1": 0.70,
    "stop_reverse_accel85_v1": 0.85,
    "stop_reverse_accel90_v1": 0.90,
}
SPLIT_SMOOTH_PROFILE_WEIGHTS = {
    "stop_reverse_split_smooth_v1": -0.02,
    "stop_reverse_split_smooth04_v1": -0.04,
}
RAW_BASELINE_YAW1_NO_HEADING_PROFILE = "stop_reverse_raw_split_straight_yaw1_headingoff_v1"
RAW_PROFILE_REWARD_WEIGHTS = {
    **{name: RawSplitStraightRewardWeights() for name in RAW_BASELINE_PROFILES},
    RAW_BASELINE_YAW1_NO_HEADING_PROFILE: RawSplitStraightRewardWeights(
        tracking_yaw_rate=1.0, tracking_yaw_rate_enhance=1.0),
}
RAW_TASK_PROFILE_NAMES = tuple(RAW_PROFILE_REWARD_WEIGHTS)
TASK_PROFILE_NAMES = (
    "legacy_v1",
    "stop_reverse_v1",
    "stop_reverse_smooth_v1",
    *REVERSAL_PROFILE_BUDGETS,
    *ACCELERATION_PROFILE_FRACTIONS,
    *SPLIT_SMOOTH_PROFILE_WEIGHTS,
    *RAW_TASK_PROFILE_NAMES,
)


def raw_task_profile_name(reward: dict, heading: dict) -> str:
    heading_profile_name(heading)
    for name, expected in RAW_PROFILE_REWARD_WEIGHTS.items():
        if reward == asdict(expected) and heading["enabled"] == (name == RAW_BASELINE_PROFILE):
            return name
    raise ValueError("保存的原始动作 baseline 奖励与航向组合无效")


def apply_task_profile(env_cfg: Any, name: str) -> None:
    if name not in TASK_PROFILE_NAMES:
        raise ValueError(f"Unknown DreamWaQ task profile: {name}")
    enabled = name != "legacy_v1"
    env_cfg.commands = replace(
        env_cfg.commands,
        hold_for_episode=not enabled,
        practice_schedule=STOP_REVERSE_COMMAND_PRACTICE_VERSION if enabled else "disabled",
    )
    # Rebuild the legacy type so switching back cannot retain split fields.
    base_weights = RewardWeights() if isinstance(env_cfg.reward_weights, RawSplitStraightRewardWeights) else RewardWeights(**{
        field.name: getattr(env_cfg.reward_weights, field.name) for field in fields(RewardWeights)
    })
    env_cfg.reward_weights = replace(
        base_weights,
        tracking_vx=2.0 if enabled else 1.0,
        tracking_vx_enhance=2.0 if enabled else 1.0,
        action_smooth=-0.02 if name == "stop_reverse_smooth_v1" else -0.01,
    )
    if name in SPLIT_SMOOTH_PROFILE_WEIGHTS:
        env_cfg.reward_weights = SplitSmoothRewardWeights(**{
            **asdict(env_cfg.reward_weights), "action_smooth": 0.0,
            "action_smooth_leg": SPLIT_SMOOTH_PROFILE_WEIGHTS[name],
            "action_smooth_wheel": -0.01,
        })
    env_cfg.leg_reversal = LegReversalConfig(
        enabled=name in REVERSAL_PROFILE_BUDGETS,
        budget_per_second=REVERSAL_PROFILE_BUDGETS.get(name, 12.0),
    )
    env_cfg.leg_acceleration = LegAccelerationConfig(
        enabled=name in ACCELERATION_PROFILE_FRACTIONS,
        budget_fraction=ACCELERATION_PROFILE_FRACTIONS.get(name, 0.80),
    )
    raw = name in RAW_TASK_PROFILE_NAMES
    env_cfg.action_clip = None if raw else 1.0
    env_cfg.runtime_geometry_mode = RUNTIME_GEOMETRY["mode"] if raw else "strict"
    env_cfg.heading_hold = HeadingHoldConfig(enabled=name == RAW_BASELINE_PROFILE) if raw else None
    env_cfg.normalization = RawActionNormalizationV1() if raw else NormalizationV2()
    if raw:
        env_cfg.reward_weights = RAW_PROFILE_REWARD_WEIGHTS[name]
    command_contract_payload(env_cfg.commands)


def apply_profile_to_runner(runner_cfg: Any, name: str) -> None:
    raw = name in RAW_TASK_PROFILE_NAMES
    runner_cfg.clip_actions = None if raw else 1.0
    runner_cfg.policy.decoder_action_clip = None if raw else 1.0


def task_profile_from_contract(contract: dict) -> str:
    commands = contract["task"]["commands"]
    schedule = commands.get("practice_schedule", "disabled")
    reward = contract["task"]["reward_weights"]
    split_fields = {"action_smooth_leg", "action_smooth_wheel"}
    saved_schema = contract.get("schemas", {}).get("reward")
    raw_identity = (
        saved_schema == RAW_SPLIT_STRAIGHT_REWARD_VERSION
        or "heading_hold" in contract["task"]
        or contract.get("manifest_version") == "WheelLegBaseTaskContractV2RawHeading"
        or contract.get("schemas", {}).get("action") == "RawActionV1"
        or contract.get("schemas", {}).get("normalization") == RawActionNormalizationV1.schema_version
        or ("action" in contract and contract["action"].get("clip_actions") is None)
    )
    if raw_identity:
        raw_profile = raw_task_profile_name(reward, contract["task"].get("heading_hold"))
        if (saved_schema != RAW_SPLIT_STRAIGHT_REWARD_VERSION
                or schedule != STOP_REVERSE_COMMAND_PRACTICE_VERSION
                or commands.get("hold_for_episode") is not False
                or contract["task"].get("runtime_geometry") != RUNTIME_GEOMETRY
                or contract["task"].get("leg_reversal") is not None
                or contract["task"].get("leg_acceleration") is not None
                or contract["action"].get("clip_actions") is not None
                or contract["schemas"].get("action") != "RawActionV1"
                or contract["schemas"].get("normalization") != RawActionNormalizationV1.schema_version
                or contract.get("manifest_version") != "WheelLegBaseTaskContractV2RawHeading"
                or json.dumps(contract["task"].get("control"), sort_keys=True) != json.dumps(asdict(ControlLimits()), sort_keys=True)
                or contract["task"].get("control_dt") != 0.02
                or json.dumps(commands, sort_keys=True) != json.dumps(command_contract_payload(
                    replace(CommandRanges(), hold_for_episode=False,
                            practice_schedule=STOP_REVERSE_COMMAND_PRACTICE_VERSION)), sort_keys=True)):
            raise ValueError("保存的原始动作/航向 baseline 契约不完整或已更改")
        return raw_profile
    if split_fields.intersection(reward) or saved_schema == SPLIT_SMOOTH_REWARD_SCHEMA_VERSION:
        if (saved_schema != SPLIT_SMOOTH_REWARD_SCHEMA_VERSION
                or schedule != STOP_REVERSE_COMMAND_PRACTICE_VERSION
                or commands.get("hold_for_episode") is not False
                or contract["task"].get("leg_reversal") is not None
                or contract["task"].get("leg_acceleration") is not None):
            raise ValueError("Invalid saved split smooth schema/schedule or combined experiment")
        for name, leg_weight in SPLIT_SMOOTH_PROFILE_WEIGHTS.items():
            expected = asdict(SplitSmoothRewardWeights(
                tracking_vx=2.0, tracking_vx_enhance=2.0, action_smooth_leg=leg_weight))
            if reward == expected:
                return name
        raise ValueError("Unknown or incomplete saved split smooth weights")
    acceleration = contract["task"].get("leg_acceleration")
    if acceleration is not None:
        if schedule != STOP_REVERSE_COMMAND_PRACTICE_VERSION:
            raise ValueError("Acceleration profile requires stop/reverse command practice")
        if contract["task"]["reward_weights"]["action_smooth"] != -0.01:
            raise ValueError("Acceleration profile requires stop_reverse_v1 action_smooth")
        for name, fraction in ACCELERATION_PROFILE_FRACTIONS.items():
            expected = acceleration_contract_payload(
                LegAccelerationConfig(enabled=True, budget_fraction=fraction)
            )
            if acceleration == expected:
                return name
        raise ValueError("Unknown saved acceleration profile contract")
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
