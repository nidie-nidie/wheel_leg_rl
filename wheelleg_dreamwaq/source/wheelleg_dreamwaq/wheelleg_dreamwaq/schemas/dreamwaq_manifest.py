from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
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
from wheelleg_dreamwaq.schemas.observation import (
    ACTOR_OBS_DIM,
    ACTOR_OBS_SCHEMA_VERSION,
    CRITIC_OBS_DIM,
    CRITIC_OBS_SCHEMA_VERSION,
    HISTORY_LAYOUT_VERSION,
    HISTORY_LENGTH,
    HISTORY_OBS_DIM,
    PROPRIO_RECONSTRUCTION_SOURCE_SLICES,
    PROPRIO_RECONSTRUCTION_TARGET_DIM,
    PROPRIO_RECONSTRUCTION_TARGET_VERSION,
    VELOCITY_TARGET_DIM,
    VELOCITY_TARGET_VERSION,
    ActorObsSlices,
    CriticObsSlices,
)
from wheelleg_dreamwaq.schemas.physics import PHYSICS_SCHEMA_VERSION, unrestricted_velocity_policy
from wheelleg_dreamwaq.schemas.randomization import (
    CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION,
    RANDOMIZATION_SCHEMA_VERSION,
    RandomizationProfileV1,
    closed_chain_reset_contract_payload,
    profile_contract_hash,
    profile_contract_payload,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import COMMAND_SAMPLING_VERSION, CommandRanges
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.control import ControlLimits
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import REWARD_SCHEMA_VERSION, RewardWeights
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.terminations import TerminationLimits


BASE_TASK_CONTRACT_VERSION = "WheelLegBaseTaskContractV1"
DREAMWAQ_ALGORITHM_CONTRACT_VERSION = "DreamWaQAlgorithmContractV1"
DREAMWAQ_EXPORT_CONTRACT_VERSION = "DreamWaQExportContractV1"
ESTIMATOR_MONITOR_STATE_VERSION = "EstimatorMonitorStateV1"
ESTIMATOR_MONITOR_FINAL_WINDOW = 10
GOLDEN_VECTOR_SEED = 20261007
GOLDEN_VECTOR_SHAPE = (32, HISTORY_OBS_DIM)
TORCHSCRIPT_MAX_ABS_ERROR = 1.0e-7


class DreamWaQContractMismatchError(RuntimeError):
    """Raised when any named Phase 2 contract differs from the current runtime."""


def _slice_bounds(value: slice) -> list[int]:
    return [int(value.start), int(value.stop)]


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


def stable_contract_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


def _with_hash(payload: dict[str, Any]) -> dict[str, Any]:
    result = _canonicalize(payload)
    result["contract_hash"] = stable_contract_hash(result)
    return result


def _base_task_payload(
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
    normalization: Any,
    randomization: RandomizationProfileV1,
    reward_weights: RewardWeights,
    termination: TerminationLimits,
    clip_actions: float | None,
    is_finite_horizon: bool,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    if asset_bundle_hash != ASSET_BUNDLE_V2.bundle_hash:
        raise ValueError("WheelLegBaseTaskContractV1 requires the frozen AssetBundleV2")
    virtual_leg = build_virtual_leg_contract()
    if virtual_leg["asset_bundle_hash"] != asset_bundle_hash:
        raise ValueError("Virtual-leg contract and selected asset bundle differ")
    return {
        "manifest_version": BASE_TASK_CONTRACT_VERSION,
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
            "randomization": RANDOMIZATION_SCHEMA_VERSION,
            "closed_chain_reset_cache": CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION,
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
        "command": {"dimension": COMMAND_DIM, "fields": list(COMMAND_FIELDS)},
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
        "randomization": {
            "profile_contract": profile_contract_payload(randomization),
            "profile_hash": profile_contract_hash(randomization),
            "process_start_fields": [
                "wheel_friction",
                "active_leg_reference",
                "active_joint_kp_scale",
                "active_joint_kd_scale",
                "active_joint_effort_limit_scale",
            ],
            "episode_reset_fields": ["root_linear_velocity", "root_angular_velocity", "command"],
            "observation_step_fields": ["actor_observation_noise"],
            "reset_joint_state": {
                "active_leg_position": "per_environment_q_reference",
                "wheel_position": "closed_chain_reset_cache",
                "passive_position": "closed_chain_reset_cache",
                "all_joint_velocity": "zero",
                "root_height": "closed_chain_reset_cache",
                "passive_position_target": "none",
                "passive_velocity_target": "zero",
                "passive_stiffness": 0.0,
                "passive_damping": 0.05,
            },
            "closed_chain_reset_cache": closed_chain_reset_contract_payload(sim_dt=sim_dt),
            "resume": "load_source_dual_tensor_cache_restore_rng_then_full_environment_reset_v3",
        },
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
                "velocity_limit_policy": unrestricted_velocity_policy(),
            },
            "episode_length_s": episode_length_s,
            "is_finite_horizon": is_finite_horizon,
            "q_nominal": list(q_nominal),
            "control": asdict(control),
            "commands": asdict(commands),
            "reward_weights": asdict(reward_weights),
            "termination": asdict(termination),
        },
        "runtime": _canonicalize(runtime),
    }


def build_base_task_contract_from_configs(*, asset_bundle_hash: str, env_cfg: Any, clip_actions: float) -> dict[str, Any]:
    scene = env_cfg.scene.to_dict()
    scene.pop("num_envs", None)
    robot = env_cfg.robot_cfg.to_dict()
    robot_spawn = robot.get("spawn")
    if isinstance(robot_spawn, dict):
        robot_spawn.pop("usd_path", None)
        robot_spawn["asset_entry_file"] = ASSET_BUNDLE_V2.entry_file
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
    payload = _base_task_payload(
        asset_bundle_hash=asset_bundle_hash,
        sim_dt=float(env_cfg.sim.dt),
        decimation=int(env_cfg.decimation),
        solver_position_iterations=int(env_cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count),
        solver_velocity_iterations=int(env_cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count),
        episode_length_s=float(env_cfg.episode_length_s),
        q_nominal=tuple(float(value) for value in env_cfg.q_nominal),
        control=env_cfg.control,
        commands=env_cfg.commands,
        normalization=env_cfg.normalization,
        randomization=env_cfg.randomization,
        reward_weights=env_cfg.reward_weights,
        termination=env_cfg.termination,
        clip_actions=clip_actions,
        is_finite_horizon=bool(env_cfg.is_finite_horizon),
        runtime=runtime,
    )
    return _with_hash(payload)


BASE_TASK_PHASE1_ALLOW_LIST = (
    "asset_bundle_version",
    "asset_bundle_hash",
    "asset",
    "schemas",
    "action",
    "command",
    "observation",
    "frames",
    "normalization",
    "randomization",
    "virtual_leg",
    "task",
    "runtime",
)


def build_base_task_contract_from_phase1_contract(phase1_contract: dict[str, Any]) -> dict[str, Any]:
    missing = set(BASE_TASK_PHASE1_ALLOW_LIST) - set(phase1_contract)
    if missing:
        raise DreamWaQContractMismatchError(f"Phase 1 contract lacks base-task fields: {sorted(missing)}")
    payload = {"manifest_version": BASE_TASK_CONTRACT_VERSION}
    for key in BASE_TASK_PHASE1_ALLOW_LIST:
        payload[key] = phase1_contract[key]
    return _with_hash(payload)


def _require_exact_config(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        raise DreamWaQContractMismatchError(f"{path} differs from DreamWaQAlgorithmContractV1: {actual!r} != {expected!r}")


def build_dreamwaq_algorithm_contract(runner_config: dict[str, Any]) -> dict[str, Any]:
    policy = runner_config.get("policy")
    algorithm = runner_config.get("algorithm")
    if not isinstance(policy, dict) or not isinstance(algorithm, dict):
        raise DreamWaQContractMismatchError("DreamWaQ runner config lacks policy or algorithm dictionaries")
    expected_policy = {
        "class_name": "DreamWaQActorCritic",
        "init_noise_std": 1.0,
        "noise_std_type": "scalar",
        "state_dependent_std": False,
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
        "actor_hidden_dims": [256, 128, 64],
        "critic_hidden_dims": [256, 128, 64],
        "activation": "elu",
        "current_obs_group": "policy",
        "history_obs_group": "policy_history",
        "history_length": 5,
        "current_obs_dim": 25,
        "history_obs_dim": 125,
        "critic_obs_dim": 41,
        "velocity_dim": 3,
        "context_dim": 16,
        "reconstruction_target_dim": 16,
        "cenet_encoder_hidden_dims": [128, 64],
        "cenet_decoder_hidden_dims": [64, 128],
        "actor_context_mode": "deterministic_context_mu",
        "action_mean_clip": 20.0,
        "action_std_min": 1.0e-4,
        "action_std_max": 2.0,
    }
    expected_algorithm = {
        "class_name": "DreamWaQPPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "normalize_advantage_per_mini_batch": False,
        "rnd_cfg": None,
        "symmetry_cfg": None,
        "velocity_coef": 1.0,
        "reconstruction_coef": 1.0,
        "kl_beta": 1.0,
        "strict_finite_checks": True,
        "context_mu_min_feature_std": 1.0e-3,
        "context_mu_min_initial_std_ratio": 0.10,
        "context_monitor_final_window_iterations": 10,
        "velocity_mse_max_zero_baseline_ratio": 0.80,
        "velocity_mse_denominator_floor": 1.0e-8,
    }
    _require_exact_config(runner_config.get("class_name"), "DreamWaQContractOnPolicyRunner", "runner.class_name")
    _require_exact_config(runner_config.get("obs_groups"), {"policy": ["policy"], "critic": ["critic"]}, "runner.obs_groups")
    _require_exact_config(runner_config.get("clip_actions"), 1.0, "runner.clip_actions")
    _require_exact_config(runner_config.get("num_steps_per_env"), 24, "runner.num_steps_per_env")
    for key, expected in expected_policy.items():
        _require_exact_config(policy.get(key), expected, f"policy.{key}")
    for key, expected in expected_algorithm.items():
        _require_exact_config(algorithm.get(key), expected, f"algorithm.{key}")
    mini_batches = algorithm.get("num_mini_batches")
    if not isinstance(mini_batches, int) or isinstance(mini_batches, bool) or mini_batches <= 0:
        raise DreamWaQContractMismatchError("algorithm.num_mini_batches must be a positive hardware field")
    payload = {
        "manifest_version": DREAMWAQ_ALGORITHM_CONTRACT_VERSION,
        "runner_class": "DreamWaQContractOnPolicyRunner",
        "policy_class": "DreamWaQActorCritic",
        "algorithm_class": "DreamWaQPPO",
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "observation_keys": ["policy", "policy_history", "critic"],
        "history": {
            "schema_version": HISTORY_LAYOUT_VERSION,
            "layout": "frame_major",
            "length": HISTORY_LENGTH,
            "current_observation_dim": ACTOR_OBS_DIM,
            "flat_dimension": HISTORY_OBS_DIM,
            "reset_fill": "five_copies_of_first_policy_observation",
        },
        "targets": {
            "velocity": {
                "schema_version": VELOCITY_TARGET_VERSION,
                "dimension": VELOCITY_TARGET_DIM,
                "critic_slice": _slice_bounds(CriticObsSlices.ROOT_LINEAR_VELOCITY),
                "time_index": "current",
            },
            "next_proprio": {
                "schema_version": PROPRIO_RECONSTRUCTION_TARGET_VERSION,
                "dimension": PROPRIO_RECONSTRUCTION_TARGET_DIM,
                "next_critic_slices": [_slice_bounds(value) for value in PROPRIO_RECONSTRUCTION_SOURCE_SLICES],
                "done_mask": "dones.eq(0).view(-1,1)",
            },
        },
        "network": {
            "activation": "elu",
            "encoder": [125, 128, 64, 35],
            "encoder_output_slices": {"velocity": [0, 3], "context_mu": [3, 19], "context_logvar": [19, 35]},
            "actor": [44, 256, 128, 64, 6],
            "critic": [41, 256, 128, 64, 1],
            "decoder": [25, 64, 128, 16],
            "actor_context_mode": "deterministic_context_mu",
            "decoder_condition": "sampled_context_z+detached_velocity+detached_clipped_action",
        },
        "distribution": {
            "init_raw_std": 1.0,
            "state_dependent_std": False,
            "effective_mean_clip": [-20.0, 20.0],
            "effective_std": "clamp(abs(raw_std),1e-4,2.0)",
            "runtime_action_clip": [-1.0, 1.0],
        },
        "ppo": {key: expected_algorithm[key] for key in (
            "value_loss_coef",
            "use_clipped_value_loss",
            "clip_param",
            "entropy_coef",
            "num_learning_epochs",
            "learning_rate",
            "schedule",
            "gamma",
            "lam",
            "desired_kl",
            "max_grad_norm",
            "normalize_advantage_per_mini_batch",
        )},
        "loss": {"velocity_coef": 1.0, "reconstruction_coef": 1.0, "kl_beta": 1.0},
        "tripwires": {
            "context_mu_min_feature_std": 1.0e-3,
            "context_mu_min_initial_std_ratio": 0.10,
            "context_monitor_final_window_iterations": 10,
            "velocity_mse_max_zero_baseline_ratio": 0.80,
            "velocity_mse_denominator_floor": 1.0e-8,
        },
        "storage": {
            "standard_fields_preserved": True,
            "additional_fields": {"next_proprio_target": [16], "reconstruction_mask": [1]},
            "write_semantics": "preallocated_copy_",
        },
        "optimizer": {"type": "Adam", "count": 1, "global_gradient_clip": 1.0},
        "normalization": {"actor_empirical": False, "critic_empirical": False, "velocity_running": False},
        "hardware_mutable_fields": ["device", "num_envs", "num_mini_batches"],
    }
    return _with_hash(payload)


def build_dreamwaq_export_contract() -> dict[str, Any]:
    return _with_hash(
        {
            "manifest_version": DREAMWAQ_EXPORT_CONTRACT_VERSION,
            "schema_version": "DreamWaQPolicyExportV1",
            "artifacts": ["actor.ts", "policy_manifest.json", "golden_vectors.pt"],
            "actor": {
                "format": "torch.jit.script",
                "stateful": False,
                "input": {"name": "history", "dtype": "float32", "rank": 2, "dimension": HISTORY_OBS_DIM, "dynamic_batch": True},
                "output": {"name": "action_mean", "dtype": "float32", "rank": 2, "dimension": ACTION_DIM},
                "effective_mean_clip": [-20.0, 20.0],
                "runtime_action_clip": [-1.0, 1.0],
                "excluded": ["critic", "decoder", "gaussian_sampling", "empirical_normalizer", "history_buffer"],
            },
            "golden_vectors": {
                "generator": "cpu_torch_randn",
                "seed": GOLDEN_VECTOR_SEED,
                "shape": list(GOLDEN_VECTOR_SHAPE),
                "dtype": "float32",
                "max_abs_error": TORCHSCRIPT_MAX_ABS_ERROR,
            },
            "hash_rule": "uppercase_sha256",
        }
    )


def validate_named_contract(saved: dict[str, Any], current: dict[str, Any], expected_version: str) -> None:
    for label, contract in (("saved", saved), ("current", current)):
        if not isinstance(contract, dict) or contract.get("manifest_version") != expected_version:
            raise DreamWaQContractMismatchError(f"{label} contract version is invalid")
        embedded = contract.get("contract_hash")
        calculated = stable_contract_hash({key: value for key, value in contract.items() if key != "contract_hash"})
        if embedded != calculated:
            raise DreamWaQContractMismatchError(f"{label} contract embedded hash is invalid")
    if saved != current:
        raise DreamWaQContractMismatchError(f"{expected_version} does not match the current runtime")


@dataclass
class EstimatorMonitorStateV1:
    schema_version: str = ESTIMATOR_MONITOR_STATE_VERSION
    completed_rollouts: int = 0
    context_mu_feature_std_mean_initial: float | None = None
    recent_context_mu_feature_std_mean: list[float] = field(default_factory=list)

    def record_rollout(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("context_mu_feature_std_mean must be finite and non-negative")
        if self.context_mu_feature_std_mean_initial is None:
            self.context_mu_feature_std_mean_initial = value
        self.completed_rollouts += 1
        self.recent_context_mu_feature_std_mean.append(value)
        del self.recent_context_mu_feature_std_mean[:-ESTIMATOR_MONITOR_FINAL_WINDOW]

    @property
    def final_window_mean(self) -> float | None:
        if not self.recent_context_mu_feature_std_mean:
            return None
        return sum(self.recent_context_mu_feature_std_mean) / len(self.recent_context_mu_feature_std_mean)

    def acceptance_threshold(self, minimum: float, initial_ratio: float) -> float:
        if self.context_mu_feature_std_mean_initial is None:
            raise ValueError("estimator monitor has no initial rollout")
        return max(float(minimum), float(initial_ratio) * self.context_mu_feature_std_mean_initial)

    def passes_context_gate(self, minimum: float, initial_ratio: float, *, require_full_window: bool) -> bool:
        final = self.final_window_mean
        if final is None:
            return False
        if require_full_window and len(self.recent_context_mu_feature_std_mean) != ESTIMATOR_MONITOR_FINAL_WINDOW:
            return False
        return final >= self.acceptance_threshold(minimum, initial_ratio)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "completed_rollouts": self.completed_rollouts,
            "context_mu_feature_std_mean_initial": self.context_mu_feature_std_mean_initial,
            "recent_context_mu_feature_std_mean": list(self.recent_context_mu_feature_std_mean),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EstimatorMonitorStateV1":
        expected = {
            "schema_version",
            "completed_rollouts",
            "context_mu_feature_std_mean_initial",
            "recent_context_mu_feature_std_mean",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("EstimatorMonitorStateV1 keys are missing or unexpected")
        if payload["schema_version"] != ESTIMATOR_MONITOR_STATE_VERSION:
            raise ValueError("EstimatorMonitorStateV1 schema version is invalid")
        completed = payload["completed_rollouts"]
        initial = payload["context_mu_feature_std_mean_initial"]
        recent = payload["recent_context_mu_feature_std_mean"]
        if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0:
            raise ValueError("completed_rollouts must be a non-negative integer")
        if not isinstance(recent, list) or len(recent) > ESTIMATOR_MONITOR_FINAL_WINDOW:
            raise ValueError("recent context monitor window is invalid")
        values = [float(value) for value in recent]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("recent context monitor values must be finite and non-negative")
        if completed == 0:
            if initial is not None or values:
                raise ValueError("empty estimator monitor cannot contain rollout values")
        else:
            if initial is None or not math.isfinite(float(initial)) or float(initial) < 0.0:
                raise ValueError("estimator monitor initial value is invalid")
            if not values or len(values) > completed:
                raise ValueError("estimator monitor recent window is inconsistent with completed rollouts")
        return cls(
            completed_rollouts=completed,
            context_mu_feature_std_mean_initial=None if initial is None else float(initial),
            recent_context_mu_feature_std_mean=values,
        )
