from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from wheelleg_dreamwaq.assets.asset_contract import ASSET_BUNDLE_V2
from wheelleg_dreamwaq.assets.asset_overrides import RUNTIME_COLLISION_POLICY_VERSION
from wheelleg_dreamwaq.kinematics.virtual_leg import (
    VIRTUAL_LEG_KINEMATICS_VERSION,
    build_virtual_leg_contract,
)
from wheelleg_dreamwaq.schemas.action import (
    ACTION_DIM,
    ACTION_SCHEMA_VERSION,
    CANONICAL_JOINT_ORDER,
    PACKET_LEG_ORDER,
    PACKET_TO_POLICY_LEG,
    POLICY_TO_PACKET_LEG,
    WHEEL_JOINT_ORDER,
    WHEEL_JOINT_SIGN_USD,
)
from wheelleg_dreamwaq.schemas.command import COMMAND_DIM, COMMAND_FIELDS, COMMAND_SCHEMA_VERSION
from wheelleg_dreamwaq.schemas.frames import CONTROL_FRAME_VERSION, R_CONTROL_FROM_IMU, R_CONTROL_FROM_USD
from wheelleg_dreamwaq.schemas.normalization import NormalizationV1
from wheelleg_dreamwaq.schemas.observation import (
    ACTOR_OBS_DIM,
    ACTOR_OBS_SCHEMA_VERSION,
    CRITIC_OBS_DIM,
    CRITIC_OBS_SCHEMA_VERSION,
    ActorObsSlices,
    CriticObsSlices,
)
from wheelleg_dreamwaq.schemas.physics import PHYSICS_SCHEMA_VERSION
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import COMMAND_SAMPLING_VERSION, CommandRanges
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.control import ControlLimits
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import REWARD_SCHEMA_VERSION, RewardWeights
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.terminations import TerminationLimits


PHASE1_CONTRACT_VERSION = "Phase1ContractV4"


class ContractMismatchError(RuntimeError):
    """Raised when a checkpoint contract is not valid for the current runtime."""


def _slice_bounds(value: slice) -> list[int]:
    return [int(value.start), int(value.stop)]


def _payload_without_hash(contract: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in contract.items() if key != "contract_hash"}


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


def _canonicalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__qualname__))
        return f"{module}.{name}"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported contract value {type(value).__name__}: {value!r}")


def _first_difference(saved: Any, current: Any, path: str = "contract") -> str | None:
    if type(saved) is not type(current):
        return f"{path}: type {type(saved).__name__} != {type(current).__name__}"
    if isinstance(saved, dict):
        if set(saved) != set(current):
            missing = sorted(set(saved) - set(current))
            extra = sorted(set(current) - set(saved))
            return f"{path}: missing={missing}, extra={extra}"
        for key in sorted(saved):
            difference = _first_difference(saved[key], current[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(saved, list):
        if len(saved) != len(current):
            return f"{path}: length {len(saved)} != {len(current)}"
        for index, (saved_value, current_value) in enumerate(zip(saved, current, strict=True)):
            difference = _first_difference(saved_value, current_value, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if saved != current:
        return f"{path}: {saved!r} != {current!r}"
    return None


def build_phase1_contract(
    *,
    asset_bundle_hash: str,
    sim_dt: float,
    decimation: int,
    solver_position_iterations: int,
    solver_velocity_iterations: int,
    episode_length_s: float,
    q_nominal: tuple[float, float, float, float],
    control: ControlLimits,
    commands: CommandRanges,
    normalization: NormalizationV1,
    reward_weights: RewardWeights,
    termination: TerminationLimits,
    actor_hidden_dims: tuple[int, ...] | list[int],
    critic_hidden_dims: tuple[int, ...] | list[int],
    activation: str,
    actor_obs_normalization: bool,
    critic_obs_normalization: bool,
    clip_actions: float | None,
    is_finite_horizon: bool,
    training: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    if asset_bundle_hash != ASSET_BUNDLE_V2.bundle_hash:
        raise ValueError(
            f"Phase1ContractV4 requires {ASSET_BUNDLE_V2.version} hash "
            f"{ASSET_BUNDLE_V2.bundle_hash}, got {asset_bundle_hash}"
        )
    virtual_leg = build_virtual_leg_contract()
    if virtual_leg["asset_bundle_version"] != ASSET_BUNDLE_V2.version:
        raise ValueError("Virtual-leg audit does not target AssetBundleV2")
    if virtual_leg["asset_bundle_hash"] != asset_bundle_hash:
        raise ValueError("Virtual-leg audit asset hash does not match the selected asset bundle")

    contract: dict[str, Any] = {
        "manifest_version": PHASE1_CONTRACT_VERSION,
        "asset_bundle_version": ASSET_BUNDLE_V2.version,
        "asset_bundle_hash": asset_bundle_hash,
        "asset": {
            "bundle_version": ASSET_BUNDLE_V2.version,
            "bundle_hash": asset_bundle_hash,
            "entry_file": ASSET_BUNDLE_V2.entry_file,
            "default_prim": ASSET_BUNDLE_V2.default_prim,
            "articulation_root": ASSET_BUNDLE_V2.articulation_root,
        },
        "schemas": {
            "action": ACTION_SCHEMA_VERSION,
            "command": COMMAND_SCHEMA_VERSION,
            "actor_observation": ACTOR_OBS_SCHEMA_VERSION,
            "critic_observation": CRITIC_OBS_SCHEMA_VERSION,
            "normalization": normalization.schema_version,
            "control_frame": CONTROL_FRAME_VERSION,
            "command_sampling": COMMAND_SAMPLING_VERSION,
            "physics": PHYSICS_SCHEMA_VERSION,
            "reward": REWARD_SCHEMA_VERSION,
            "virtual_leg_kinematics": VIRTUAL_LEG_KINEMATICS_VERSION,
        },
        "action": {
            "dimension": ACTION_DIM,
            "canonical_joint_order": list(CANONICAL_JOINT_ORDER),
            "packet_leg_order": list(PACKET_LEG_ORDER),
            "policy_to_packet_leg": list(POLICY_TO_PACKET_LEG),
            "packet_to_policy_leg": list(PACKET_TO_POLICY_LEG),
            "wheel_joint_order": list(WHEEL_JOINT_ORDER),
            "wheel_joint_sign_usd": list(WHEEL_JOINT_SIGN_USD),
            "clip_actions": clip_actions,
        },
        "command": {
            "dimension": COMMAND_DIM,
            "fields": list(COMMAND_FIELDS),
        },
        "observation": {
            "actor_dimension": ACTOR_OBS_DIM,
            "critic_dimension": CRITIC_OBS_DIM,
            "actor_slices": {
                "angular_velocity": _slice_bounds(ActorObsSlices.ANGULAR_VELOCITY),
                "projected_gravity": _slice_bounds(ActorObsSlices.PROJECTED_GRAVITY),
                "command": _slice_bounds(ActorObsSlices.COMMAND),
                "leg_position_error": _slice_bounds(ActorObsSlices.LEG_POSITION_ERROR),
                "joint_velocity": _slice_bounds(ActorObsSlices.JOINT_VELOCITY),
                "previous_action": _slice_bounds(ActorObsSlices.PREVIOUS_ACTION),
            },
            "critic_slices": {
                "actor_observation": _slice_bounds(CriticObsSlices.ACTOR_OBS),
                "root_linear_velocity": _slice_bounds(CriticObsSlices.ROOT_LINEAR_VELOCITY),
                "base_height": _slice_bounds(CriticObsSlices.BASE_HEIGHT),
                "joint_acceleration": _slice_bounds(CriticObsSlices.JOINT_ACCELERATION),
                "applied_torque": _slice_bounds(CriticObsSlices.APPLIED_TORQUE),
            },
        },
        "frames": {
            "r_control_from_usd": R_CONTROL_FROM_USD.tolist(),
            "r_control_from_imu": R_CONTROL_FROM_IMU.tolist(),
        },
        "normalization": asdict(normalization),
        "virtual_leg": virtual_leg,
        "task": {
            "sim_dt": sim_dt,
            "decimation": decimation,
            "control_dt": sim_dt * decimation,
            "physics": {
                "sim_dt": sim_dt,
                "decimation": decimation,
                "control_dt": sim_dt * decimation,
                "solver_position_iterations": solver_position_iterations,
                "solver_velocity_iterations": solver_velocity_iterations,
            },
            "episode_length_s": episode_length_s,
            "is_finite_horizon": is_finite_horizon,
            "q_nominal": list(q_nominal),
            "control": asdict(control),
            "commands": asdict(commands),
            "reward_weights": asdict(reward_weights),
            "termination": asdict(termination),
        },
        "network": {
            "actor_hidden_dims": list(actor_hidden_dims),
            "critic_hidden_dims": list(critic_hidden_dims),
            "activation": activation,
            "actor_obs_normalization": actor_obs_normalization,
            "critic_obs_normalization": critic_obs_normalization,
        },
        "training": _canonicalize(training),
        "runtime": _canonicalize(runtime),
    }
    contract["contract_hash"] = _stable_hash(contract)
    return contract


def build_phase1_contract_from_configs(
    *,
    asset_bundle_hash: str,
    env_cfg: Any,
    agent_cfg: Any,
) -> dict[str, Any]:
    scene = env_cfg.scene.to_dict()
    scene.pop("num_envs", None)
    robot = env_cfg.robot_cfg.to_dict()
    robot_spawn = robot.get("spawn")
    if isinstance(robot_spawn, dict):
        robot_spawn.pop("usd_path", None)
        robot_spawn["asset_entry_file"] = ASSET_BUNDLE_V2.entry_file

    algorithm = agent_cfg.algorithm.to_dict()
    algorithm.pop("num_mini_batches", None)
    training = {
        "runner": {
            "num_steps_per_env": int(agent_cfg.num_steps_per_env),
            "obs_groups": agent_cfg.obs_groups,
            "clip_actions": agent_cfg.clip_actions,
        },
        "policy": agent_cfg.policy.to_dict(),
        "algorithm_except_hardware_batching": algorithm,
        "hardware_mutable_fields": ["device", "num_envs", "num_mini_batches"],
    }
    runtime = {
        "collision_policy": RUNTIME_COLLISION_POLICY_VERSION,
        "embedded_ground_absent": True,
        "environment": {
            "action_space": env_cfg.action_space,
            "observation_space": env_cfg.observation_space,
            "state_space": env_cfg.state_space,
            "is_finite_horizon": env_cfg.is_finite_horizon,
        },
        "scene_except_num_envs": scene,
        "simulation": {
            "physics_prim_path": env_cfg.sim.physics_prim_path,
            "dt": env_cfg.sim.dt,
            "render_interval": env_cfg.sim.render_interval,
            "gravity": list(env_cfg.sim.gravity),
            "enable_scene_query_support": env_cfg.sim.enable_scene_query_support,
            "use_fabric": env_cfg.sim.use_fabric,
            "create_stage_in_memory": env_cfg.sim.create_stage_in_memory,
            "physics_material": env_cfg.sim.physics_material.to_dict(),
            "physx": env_cfg.sim.physx.to_dict(),
        },
        "ground": env_cfg.ground.to_dict(),
        "robot_except_asset_absolute_path": robot,
    }
    return build_phase1_contract(
        asset_bundle_hash=asset_bundle_hash,
        sim_dt=float(env_cfg.sim.dt),
        decimation=int(env_cfg.decimation),
        solver_position_iterations=int(
            env_cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count
        ),
        solver_velocity_iterations=int(
            env_cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count
        ),
        episode_length_s=float(env_cfg.episode_length_s),
        q_nominal=tuple(float(value) for value in env_cfg.q_nominal),
        control=env_cfg.control,
        commands=env_cfg.commands,
        normalization=env_cfg.normalization,
        reward_weights=env_cfg.reward_weights,
        termination=env_cfg.termination,
        actor_hidden_dims=agent_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=agent_cfg.policy.critic_hidden_dims,
        activation=agent_cfg.policy.activation,
        actor_obs_normalization=bool(agent_cfg.policy.actor_obs_normalization),
        critic_obs_normalization=bool(agent_cfg.policy.critic_obs_normalization),
        clip_actions=agent_cfg.clip_actions,
        is_finite_horizon=bool(env_cfg.is_finite_horizon),
        training=training,
        runtime=runtime,
    )


def validate_contract(saved: dict[str, Any], current: dict[str, Any]) -> None:
    for label, contract in (("saved", saved), ("current", current)):
        if contract.get("manifest_version") != PHASE1_CONTRACT_VERSION:
            raise ContractMismatchError(
                f"{label} contract version is {contract.get('manifest_version')!r}, "
                f"expected {PHASE1_CONTRACT_VERSION!r}"
            )
        embedded_hash = contract.get("contract_hash")
        calculated_hash = _stable_hash(_payload_without_hash(contract))
        if embedded_hash != calculated_hash:
            raise ContractMismatchError(
                f"{label} contract embedded hash is invalid: {embedded_hash!r} != {calculated_hash!r}"
            )

    if saved["contract_hash"] != current["contract_hash"]:
        difference = _first_difference(_payload_without_hash(saved), _payload_without_hash(current))
        raise ContractMismatchError(
            "Checkpoint contract does not match the current runtime"
            + (f"; first difference: {difference}" if difference else "")
        )
