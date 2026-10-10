from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import sys
from typing import Any

import torch

from wheelleg_dreamwaq.kinematics.virtual_leg import OffsetLegGeometry, offset_leg_fk
from wheelleg_dreamwaq.schemas.action import CANONICAL_JOINT_ORDER


RANDOMIZATION_SCHEMA_VERSION = "RandomizationSchemaV1"
RANDOMIZATION_SEED_DERIVATION_VERSION = "RandomizationSeedDerivationV1"
RANDOMIZED_STRESS_PROTOCOL_VERSION = "RandomizedStressProtocolV1"
CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION = "ClosedChainResetCacheSchemaV2"
HISTORICAL_CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION = "ClosedChainResetCacheSchemaV1"
CLOSED_CHAIN_RELAXATION_VERSION = "BoundaryClampedPhysXRelaxationV1"
CLOSED_CHAIN_ROOT_HEIGHT_ALIGNMENT_VERSION = "ClosedChainRootHeightAlignmentV1"
CLOSED_CHAIN_RELAXATION_STEPS = 10
CLOSED_CHAIN_ROOT_LIFT_M = 0.75
CLOSED_CHAIN_PASSIVE_DAMPING = 0.05
CLOSED_CHAIN_MAX_LOOP_ERROR_M = 5.0e-4
CLOSED_CHAIN_MAX_WHEEL_ERROR_M = 5.0e-4
CLOSED_CHAIN_MAX_LENGTH_ERROR_M = 5.0e-4
CLOSED_CHAIN_MAX_PHI0_ERROR_RAD = math.radians(0.25)
CLOSED_CHAIN_MAX_ACTIVE_ERROR_RAD = 1.0e-4
CLOSED_CHAIN_MAX_PASSIVE_WRAPPED_DELTA_RAD = 0.10
CLOSED_CHAIN_MAX_PASSIVE_RAW_DELTA_RAD = math.pi
CLOSED_CHAIN_PHYSICAL_PASSIVE_BRANCH_JOINTS = (
    "jJM",
    "jMK",
    "jKN",
    "jOP",
    "jBE",
    "jEC",
    "jCF",
    "jGH",
)
CLOSED_CHAIN_NOMINAL_BRANCH_SIGNATURE = (-1, 1, -1, -1, 1, 1, 1, 1)
CLOSED_CHAIN_TRACE_STEPS = (0, 1, 5, 10)
RANDOMIZED_STRESS_SEEDS = (31001, 31002, 31003, 31004)
PROCESS_START_RANDOMIZATION_STREAMS = (
    "material_rng",
    "joint_reference_rng",
    "actuator_gain_rng",
    "effort_limit_rng",
)
RUNTIME_RANDOMIZATION_STREAMS = (
    "reset_velocity_rng",
    "observation_noise_rng",
    "command_rng",
)
RANDOMIZATION_STREAMS = PROCESS_START_RANDOMIZATION_STREAMS + RUNTIME_RANDOMIZATION_STREAMS
CPU_STREAMS = frozenset(RANDOMIZATION_STREAMS) - {"observation_noise_rng"}


def canonical_tensor_sha256(named_tensors: dict[str, torch.Tensor]) -> str:
    """Hash tensor identity with the frozen cache byte contract."""

    digest = hashlib.sha256()
    for name, values in sorted(named_tensors.items()):
        tensor = values.detach().to(device="cpu").contiguous()
        array = tensor.numpy()
        if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and sys.byteorder == "big"):
            array = array.byteswap().view(array.dtype.newbyteorder("<"))
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def compute_closed_chain_root_height_offset(
    q_reference: torch.Tensor,
    *,
    q_nominal: torch.Tensor,
    geometries: tuple[OffsetLegGeometry, OffsetLegGeometry],
) -> torch.Tensor:
    """Return the canonical per-environment root-Z correction on CPU float32."""

    if q_reference.ndim != 2 or q_reference.shape[1] != 4:
        raise ValueError(f"Expected q_reference=(num_envs,4), got {tuple(q_reference.shape)}")
    if q_nominal.shape != (4,):
        raise ValueError(f"Expected q_nominal=(4,), got {tuple(q_nominal.shape)}")
    reference = q_reference.detach().to(device="cpu", dtype=torch.float64).contiguous()
    nominal = q_nominal.detach().to(device="cpu", dtype=torch.float64).contiguous()
    nominal_left = offset_leg_fk(nominal[0:2].unsqueeze(0), geometries[0])
    nominal_right = offset_leg_fk(nominal[2:4].unsqueeze(0), geometries[1])
    reference_left = offset_leg_fk(reference[:, 0:2], geometries[0])
    reference_right = offset_leg_fk(reference[:, 2:4], geometries[1])
    if not bool(
        (
            nominal_left.valid.all()
            & nominal_right.valid.all()
            & reference_left.valid.all()
            & reference_right.valid.all()
        ).item()
    ):
        raise ValueError("Root-height alignment requires valid nominal and randomized FK states")
    nominal_down = torch.stack(
        (-nominal_left.wheel_vector_body[0, 2], -nominal_right.wheel_vector_body[0, 2])
    )
    reference_down = torch.stack(
        (-reference_left.wheel_vector_body[:, 2], -reference_right.wheel_vector_body[:, 2]),
        dim=1,
    )
    offsets = (reference_down - nominal_down.unsqueeze(0)).max(dim=1).values
    offsets = offsets.to(dtype=torch.float32).contiguous()
    if not torch.isfinite(offsets).all():
        raise ValueError("Root-height alignment produced a non-finite offset")
    return offsets


def closed_chain_reset_contract_payload(*, sim_dt: float) -> dict:
    if not math.isfinite(sim_dt) or sim_dt <= 0.0:
        raise ValueError("sim_dt must be finite and positive")
    return {
        "schema_version": CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION,
        "algorithm_version": CLOSED_CHAIN_RELAXATION_VERSION,
        "root_height_algorithm_version": CLOSED_CHAIN_ROOT_HEIGHT_ALIGNMENT_VERSION,
        "steps": CLOSED_CHAIN_RELAXATION_STEPS,
        "sim_dt_s": float(sim_dt),
        "root_lift_world_z_m": CLOSED_CHAIN_ROOT_LIFT_M,
        "gravity_during_relaxation": [0.0, 0.0, 0.0],
        "active_joint_order": list(CANONICAL_JOINT_ORDER),
        "physical_passive_branch_joint_order": list(CLOSED_CHAIN_PHYSICAL_PASSIVE_BRANCH_JOINTS),
        "nominal_branch_signature": list(CLOSED_CHAIN_NOMINAL_BRANCH_SIGNATURE),
        "passive_actuator": {
            "stiffness": 0.0,
            "damping": CLOSED_CHAIN_PASSIVE_DAMPING,
            "velocity_target": 0.0,
        },
        "write_order": [
            "read_previous_joint_position",
            "overwrite_active_leg_reference",
            "overwrite_wheel_nominal_position",
            "zero_all_joint_velocity",
            "write_root_pose_velocity",
            "write_full_joint_state",
            "set_active_leg_position_target",
            "set_wheel_zero_velocity_target",
            "scene_write_data_to_sim",
            "sim_step",
            "scene_update",
        ],
        "finalization": [
            "overwrite_active_and_wheel_position",
            "zero_all_joint_velocity",
            "zero_root_velocity",
            "write_full_root_and_joint_state",
            "scene_write_data_to_sim",
            "sim_forward",
            "scene_update",
        ],
        "hard_gate": {
            "max_loop_error_m": CLOSED_CHAIN_MAX_LOOP_ERROR_M,
            "max_wheel_error_m": CLOSED_CHAIN_MAX_WHEEL_ERROR_M,
            "max_length_error_m": CLOSED_CHAIN_MAX_LENGTH_ERROR_M,
            "max_phi0_error_rad": CLOSED_CHAIN_MAX_PHI0_ERROR_RAD,
            "max_active_reference_error_rad": CLOSED_CHAIN_MAX_ACTIVE_ERROR_RAD,
            "max_passive_wrapped_delta_rad": CLOSED_CHAIN_MAX_PASSIVE_WRAPPED_DELTA_RAD,
            "max_passive_raw_delta_rad_exclusive": CLOSED_CHAIN_MAX_PASSIVE_RAW_DELTA_RAD,
        },
        "trace_steps": list(CLOSED_CHAIN_TRACE_STEPS),
        "root_height_alignment": {
            "formula": "max_side(-wheel_z(q_reference) + wheel_z(q_nominal))",
            "input_dtype": "torch.float32",
            "compute_device": "cpu",
            "compute_dtype": "torch.float64",
            "storage_device": "cpu",
            "storage_dtype": "torch.float32",
            "comparison": "torch.equal_after_canonical_recompute",
        },
        "absolute_simulation_time_dependencies": "forbidden",
    }


def validate_closed_chain_reset_cache_artifact(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Closed-chain reset cache must be a dictionary")
    required = {
        "schema_version",
        "algorithm_version",
        "root_height_algorithm_version",
        "contract",
        "identity",
        "q_reset_projected_env",
        "root_height_offset_env",
        "tensor_sha256",
        "actuator_plan",
        "metrics",
        "trace",
    }
    if set(payload) != required:
        raise ValueError(
            "Closed-chain reset cache fields are invalid: "
            f"missing={sorted(required - set(payload))}, extra={sorted(set(payload) - required)}"
        )
    if payload["schema_version"] != CLOSED_CHAIN_RESET_CACHE_SCHEMA_VERSION:
        raise ValueError("Closed-chain reset cache schema version mismatch")
    if payload["algorithm_version"] != CLOSED_CHAIN_RELAXATION_VERSION:
        raise ValueError("Closed-chain reset cache algorithm version mismatch")
    if payload["root_height_algorithm_version"] != CLOSED_CHAIN_ROOT_HEIGHT_ALIGNMENT_VERSION:
        raise ValueError("Closed-chain reset cache root-height algorithm version mismatch")
    contract = payload["contract"]
    if not isinstance(contract, dict) or not isinstance(contract.get("sim_dt_s"), (int, float)):
        raise ValueError("Closed-chain reset cache contract is missing sim_dt_s")
    if contract != closed_chain_reset_contract_payload(sim_dt=float(contract["sim_dt_s"])):
        raise ValueError("Closed-chain reset cache contract payload mismatch")
    q_reset = payload["q_reset_projected_env"]
    if (
        not isinstance(q_reset, torch.Tensor)
        or q_reset.ndim != 2
        or q_reset.dtype != torch.float32
        or not torch.isfinite(q_reset).all()
    ):
        raise ValueError("Closed-chain reset position tensor must be float32 with shape (num_envs,num_joints)")
    if q_reset.device.type != "cpu" or not q_reset.is_contiguous():
        raise ValueError("Closed-chain reset position tensor must be contiguous on CPU")
    root_height_offset = payload["root_height_offset_env"]
    if (
        not isinstance(root_height_offset, torch.Tensor)
        or root_height_offset.device.type != "cpu"
        or root_height_offset.dtype != torch.float32
        or root_height_offset.shape != (q_reset.shape[0],)
        or not root_height_offset.is_contiguous()
        or not torch.isfinite(root_height_offset).all()
    ):
        raise ValueError("Closed-chain reset root-height tensor must be contiguous CPU float32")
    tensor_hash = canonical_tensor_sha256(
        {
            "q_reset_projected_env": q_reset,
            "root_height_offset_env": root_height_offset,
        }
    )
    if payload["tensor_sha256"] != tensor_hash:
        raise ValueError("Closed-chain reset position tensor hash mismatch")
    identity = payload["identity"]
    identity_fields = {
        "asset_bundle_version",
        "asset_bundle_hash",
        "physics_schema_version",
        "randomization_profile_hash",
        "realized_plan_hash",
        "actuator_plan_hash",
        "master_seed",
        "num_envs",
        "joint_names",
        "passive_joint_names",
        "physical_passive_branch_joint_names",
    }
    if not isinstance(identity, dict) or set(identity) != identity_fields:
        raise ValueError("Closed-chain reset cache identity is missing or incomplete")
    if identity["num_envs"] != q_reset.shape[0] or len(identity["joint_names"]) != q_reset.shape[1]:
        raise ValueError("Closed-chain reset cache identity shape mismatch")
    actuator_plan = payload["actuator_plan"]
    actuator_fields = {
        "active_joint_realized_stiffness",
        "active_joint_realized_damping",
        "active_joint_realized_effort_limit",
    }
    if not isinstance(actuator_plan, dict) or set(actuator_plan) != actuator_fields:
        raise ValueError("Closed-chain reset cache actuator plan is missing or incomplete")
    for name, values in actuator_plan.items():
        if (
            not isinstance(values, torch.Tensor)
            or values.device.type != "cpu"
            or values.dtype != torch.float32
            or values.shape != (q_reset.shape[0], len(CANONICAL_JOINT_ORDER))
            or not values.is_contiguous()
            or not torch.isfinite(values).all()
        ):
            raise ValueError(f"Closed-chain reset cache actuator tensor is invalid: {name}")
    if canonical_tensor_sha256(actuator_plan) != identity["actuator_plan_hash"]:
        raise ValueError("Closed-chain reset cache actuator plan hash mismatch")
    if not isinstance(payload["metrics"], dict) or not isinstance(payload["trace"], list):
        raise ValueError("Closed-chain reset cache metrics or trace is invalid")
    return payload


@dataclass(frozen=True)
class RandomizationProfileV1:
    name: str
    enabled: bool
    friction_range: tuple[float, float] = (0.6, 1.4)
    friction_bucket_count: int = 64
    leg_reference_offset_range_rad: tuple[float, float] = (-0.03, 0.03)
    kp_scale_range: tuple[float, float] = (0.95, 1.05)
    kd_scale_range: tuple[float, float] = (0.95, 1.05)
    effort_limit_scale_range: tuple[float, float] = (0.95, 1.05)
    root_linear_xy_range_mps: tuple[float, float] = (-0.10, 0.10)
    root_linear_z_range_mps: tuple[float, float] = (-0.05, 0.05)
    root_angular_range_rad_s: tuple[float, float] = (-0.10, 0.10)
    angular_velocity_noise_rad_s: float = 0.20
    projected_gravity_noise: float = 0.05
    leg_position_noise_rad: float = 0.02
    joint_velocity_noise_rad_s: float = 1.50
    joint_reference_resample_limit: int = 32
    max_leg_length_delta_m: float = 0.010
    max_leg_phi0_delta_rad: float = math.radians(5.0)
    max_lr_phi0_delta_rad: float = math.radians(5.0)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Randomization profile name must be non-empty")
        for field_name in (
            "friction_range",
            "leg_reference_offset_range_rad",
            "kp_scale_range",
            "kd_scale_range",
            "effort_limit_scale_range",
            "root_linear_xy_range_mps",
            "root_linear_z_range_mps",
            "root_angular_range_rad_s",
        ):
            low, high = getattr(self, field_name)
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError(f"Invalid range for {field_name}: {(low, high)}")
        if self.friction_range[0] < 0.0:
            raise ValueError("Friction cannot be negative")
        if self.friction_bucket_count <= 0:
            raise ValueError("friction_bucket_count must be positive")
        if self.joint_reference_resample_limit <= 0:
            raise ValueError("joint_reference_resample_limit must be positive")
        for field_name in (
            "angular_velocity_noise_rad_s",
            "projected_gravity_noise",
            "leg_position_noise_rad",
            "joint_velocity_noise_rad_s",
            "max_leg_length_delta_m",
            "max_leg_phi0_delta_rad",
            "max_lr_phi0_delta_rad",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")


NOMINAL_TRAINING_PROFILE_V1 = RandomizationProfileV1(
    name="NominalTrainingProfileV1",
    enabled=False,
)
FUDAN_STYLE_DOMAIN_RANDOMIZATION_V1 = RandomizationProfileV1(
    name="FudanStyleDomainRandomizationV1",
    enabled=True,
)
NOMINAL_EVALUATION_PROFILE_V1 = RandomizationProfileV1(
    name="NominalEvaluationProfileV1",
    enabled=False,
)


def derive_stream_seed(master_seed: int, stream_name: str) -> int:
    if not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError("master_seed must be a non-negative integer")
    if stream_name not in RANDOMIZATION_STREAMS:
        raise ValueError(f"Unknown randomization stream: {stream_name!r}")
    payload = f"RandomizationSeedV1|{master_seed}|{stream_name}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)
    return seed or 1


def generator_device_for_stream(stream_name: str, env_device: str) -> str:
    if stream_name not in RANDOMIZATION_STREAMS:
        raise ValueError(f"Unknown randomization stream: {stream_name!r}")
    return "cpu" if stream_name in CPU_STREAMS else str(env_device)


def seed_manifest(master_seed: int, env_device: str) -> dict[str, dict[str, int | str]]:
    return {
        stream: {
            "seed": derive_stream_seed(master_seed, stream),
            "generator": "torch.Generator",
            "device": generator_device_for_stream(stream, env_device),
        }
        for stream in RANDOMIZATION_STREAMS
    }


def validate_randomization_rng_state_payload(payload: Any) -> dict[str, Any]:
    expected_fields = {"seed_derivation_version", "streams"}
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("Randomization RNG state fields are missing or incomplete")
    if payload["seed_derivation_version"] != RANDOMIZATION_SEED_DERIVATION_VERSION:
        raise ValueError("Randomization RNG seed-derivation version mismatch")
    streams = payload["streams"]
    if not isinstance(streams, dict) or set(streams) != set(RANDOMIZATION_STREAMS):
        raise ValueError("Randomization RNG streams are missing or incomplete")
    observation_device = streams["observation_noise_rng"].get("device")
    if not isinstance(observation_device, str):
        raise ValueError("Randomization RNG observation device is invalid")
    for stream in RANDOMIZATION_STREAMS:
        stream_state = streams[stream]
        if not isinstance(stream_state, dict) or set(stream_state) != {"device", "state"}:
            raise ValueError(f"Randomization RNG stream {stream!r} is invalid")
        device = stream_state["device"]
        expected_device = generator_device_for_stream(stream, observation_device)
        if device != expected_device:
            raise ValueError(
                f"Randomization RNG device mismatch for {stream}: {device!r} != {expected_device!r}"
            )
        state = stream_state["state"]
        if (
            not isinstance(state, torch.Tensor)
            or state.device.type != "cpu"
            or state.dtype != torch.uint8
            or state.ndim != 1
            or not state.is_contiguous()
            or state.numel() == 0
        ):
            raise ValueError(f"Randomization RNG tensor state is invalid for {stream}")
        try:
            verifier = torch.Generator(device=device)
            verifier.set_state(state)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"Randomization RNG tensor state is unusable for {stream}") from error
    return payload


def profile_contract_payload(profile: RandomizationProfileV1) -> dict:
    return {
        "schema_version": RANDOMIZATION_SCHEMA_VERSION,
        "profile": asdict(profile),
        "seed_derivation_version": RANDOMIZATION_SEED_DERIVATION_VERSION,
        "streams": list(RANDOMIZATION_STREAMS),
        "stream_lifecycle": {
            "process_start": list(PROCESS_START_RANDOMIZATION_STREAMS),
            "runtime": list(RUNTIME_RANDOMIZATION_STREAMS),
        },
        "generator_device_rules": {
            "cpu_streams": sorted(CPU_STREAMS),
            "environment_device_streams": ["observation_noise_rng"],
        },
        "actor_observation": "physical_noise_then_normalize",
        "critic_actor_features": "clean_then_normalize",
        "implicit_torque_semantics": "implicit_pd_estimate",
    }


def profile_contract_hash(profile: RandomizationProfileV1) -> str:
    encoded = json.dumps(
        profile_contract_payload(profile),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()
