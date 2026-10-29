from __future__ import annotations

import copy
from collections.abc import Iterable
import hashlib
import json
import math
from typing import Any

import torch

from wheelleg_dreamwaq.kinematics.virtual_leg import OffsetLegGeometry, offset_leg_fk, wrap_to_pi
from wheelleg_dreamwaq.schemas.frames import transform_control_vector_to_usd
from wheelleg_dreamwaq.schemas.randomization import (
    RANDOMIZATION_SEED_DERIVATION_VERSION,
    RANDOMIZATION_STREAMS,
    RandomizationProfileV1,
    canonical_tensor_sha256,
    derive_stream_seed,
    generator_device_for_stream,
    profile_contract_hash,
    profile_contract_payload,
    seed_manifest,
    validate_randomization_rng_state_payload,
)

from .commands import CommandRanges, sample_commands
from .observations import ActorObservationNoiseV1


def _uniform(
    shape: tuple[int, ...],
    bounds: tuple[float, float],
    *,
    generator: torch.Generator,
    device: str | torch.device,
) -> torch.Tensor:
    low, high = bounds
    unit = torch.rand(shape, generator=generator, device=device)
    return low + (high - low) * unit


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    flattened = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if flattened.numel() == 0:
        raise ValueError("Cannot summarize an empty tensor")
    return {
        "min": float(flattened.min().item()),
        "max": float(flattened.max().item()),
        "mean": float(flattened.mean().item()),
        "std": float(flattened.std(unbiased=False).item()),
    }


def sample_leg_references(
    *,
    count: int,
    q_nominal: torch.Tensor,
    hard_limits: torch.Tensor,
    soft_limit_abs: float,
    geometries: tuple[OffsetLegGeometry, OffsetLegGeometry],
    profile: RandomizationProfileV1,
    generator: torch.Generator,
    device: str | torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target_device = torch.device(device)
    nominal = q_nominal.to(device=target_device, dtype=torch.float32)
    limits = hard_limits.to(device=target_device, dtype=torch.float32)
    if nominal.shape != (4,) or limits.shape != (4, 2):
        raise ValueError("Expected q_nominal=(4,) and hard_limits=(4,2)")
    if not profile.enabled:
        references = nominal.repeat(count, 1)
        return references, {
            "candidate_count": count,
            "accepted_count": count,
            "acceptance_rate": 1.0,
            "max_attempts": 1,
            "rejection_counts": {},
        }

    nominal_left = offset_leg_fk(nominal[0:2].unsqueeze(0), geometries[0])
    nominal_right = offset_leg_fk(nominal[2:4].unsqueeze(0), geometries[1])
    references = torch.empty((count, 4), device=target_device, dtype=torch.float32)
    unresolved = torch.arange(count, device=target_device)
    candidate_count = 0
    rejection_counts = {
        "non_finite": 0,
        "hard_limit": 0,
        "soft_limit": 0,
        "fk_invalid": 0,
        "length_delta": 0,
        "phi0_delta": 0,
        "lr_phi0_delta": 0,
    }
    last_candidates = None
    attempts_used = 0
    for attempt in range(1, profile.joint_reference_resample_limit + 1):
        if unresolved.numel() == 0:
            break
        attempts_used = attempt
        sampled_cpu = _uniform(
            (unresolved.numel(), 4),
            profile.leg_reference_offset_range_rad,
            generator=generator,
            device="cpu",
        )
        candidates = nominal.unsqueeze(0) + sampled_cpu.to(target_device)
        last_candidates = candidates
        candidate_count += candidates.shape[0]

        left = offset_leg_fk(candidates[:, 0:2], geometries[0])
        right = offset_leg_fk(candidates[:, 2:4], geometries[1])
        finite = torch.isfinite(candidates).all(dim=-1)
        hard = ((candidates >= limits[:, 0]) & (candidates <= limits[:, 1])).all(dim=-1)
        soft = (torch.abs(candidates) <= soft_limit_abs).all(dim=-1)
        fk_valid = left.valid & right.valid
        length_delta = (
            (torch.abs(left.length - nominal_left.length[0]) <= profile.max_leg_length_delta_m)
            & (torch.abs(right.length - nominal_right.length[0]) <= profile.max_leg_length_delta_m)
        )
        phi0_delta = (
            (torch.abs(wrap_to_pi(left.phi0 - nominal_left.phi0[0])) <= profile.max_leg_phi0_delta_rad)
            & (torch.abs(wrap_to_pi(right.phi0 - nominal_right.phi0[0])) <= profile.max_leg_phi0_delta_rad)
        )
        lr_phi0_delta = torch.abs(wrap_to_pi(left.phi0 - right.phi0)) <= profile.max_lr_phi0_delta_rad
        checks = {
            "non_finite": finite,
            "hard_limit": hard,
            "soft_limit": soft,
            "fk_invalid": fk_valid,
            "length_delta": length_delta,
            "phi0_delta": phi0_delta,
            "lr_phi0_delta": lr_phi0_delta,
        }
        for name, passed in checks.items():
            rejection_counts[name] += int((~passed).sum().item())
        accepted = finite & hard & soft & fk_valid & length_delta & phi0_delta & lr_phi0_delta
        if accepted.any():
            references[unresolved[accepted]] = candidates[accepted]
        unresolved = unresolved[~accepted]

    if unresolved.numel() > 0:
        failed = unresolved[:8].tolist()
        candidate_preview = [] if last_candidates is None else last_candidates[:8].tolist()
        raise RuntimeError(
            "Closed-chain leg reference sampling exhausted its retry limit: "
            f"env_ids={failed}, last_candidates={candidate_preview}, rejections={rejection_counts}"
        )
    return references, {
        "candidate_count": candidate_count,
        "accepted_count": count,
        "acceptance_rate": count / candidate_count,
        "max_attempts": attempts_used,
        "rejection_counts": rejection_counts,
    }


class WheelLegRandomizationRuntime:
    def __init__(
        self,
        *,
        profile: RandomizationProfileV1,
        master_seed: int,
        device: str | torch.device,
        num_envs: int,
        robot: Any,
        controlled_joint_ids: list[int],
        leg_joint_ids: list[int],
        wheel_body_ids: list[int],
        q_nominal: torch.Tensor,
        geometries: tuple[OffsetLegGeometry, OffsetLegGeometry],
        soft_leg_limit: float,
    ) -> None:
        self.profile = profile
        self.master_seed = int(master_seed)
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.robot = robot
        self.controlled_joint_ids = list(controlled_joint_ids)
        self.leg_joint_ids = list(leg_joint_ids)
        self.wheel_body_ids = list(wheel_body_ids)
        self._generators = self._create_generators()

        hard_limits = robot.data.joint_pos_limits[0, self.leg_joint_ids]
        self.q_reference, reference_audit = sample_leg_references(
            count=self.num_envs,
            q_nominal=q_nominal,
            hard_limits=hard_limits,
            soft_limit_abs=soft_leg_limit,
            geometries=geometries,
            profile=profile,
            generator=self._generators["joint_reference_rng"],
            device=self.device,
        )
        nominal_reference = q_nominal.to(self.device).repeat(self.num_envs, 1)
        self.reference_offset = self.q_reference - nominal_reference
        self._sample_process_parameters()
        if self.profile.enabled:
            self._apply_wheel_materials()
            self._apply_actuator_parameters()
        self._audit = self._build_audit(reference_audit)

    def _create_generators(self) -> dict[str, torch.Generator]:
        generators = {}
        for stream in RANDOMIZATION_STREAMS:
            generator = torch.Generator(device=generator_device_for_stream(stream, str(self.device)))
            generator.manual_seed(derive_stream_seed(self.master_seed, stream))
            generators[stream] = generator
        return generators

    def _sample_process_parameters(self) -> None:
        controlled_ids = self.controlled_joint_ids
        nominal_stiffness = self.robot.data.default_joint_stiffness[0, controlled_ids].to(self.device)
        nominal_damping = self.robot.data.default_joint_damping[0, controlled_ids].to(self.device)
        nominal_effort = self.robot.data.joint_effort_limits[0, controlled_ids].to(self.device)
        if self.profile.enabled:
            material_generator = self._generators["material_rng"]
            self.friction_buckets = _uniform(
                (self.profile.friction_bucket_count,),
                self.profile.friction_range,
                generator=material_generator,
                device="cpu",
            )
            self.friction_bucket_ids = torch.randint(
                0,
                self.profile.friction_bucket_count,
                (self.num_envs,),
                generator=material_generator,
                device="cpu",
            )
            self.wheel_friction = self.friction_buckets[self.friction_bucket_ids]
            gain_generator = self._generators["actuator_gain_rng"]
            self.kp_scale = _uniform(
                (self.num_envs, 6), self.profile.kp_scale_range, generator=gain_generator, device="cpu"
            ).to(self.device)
            self.kd_scale = _uniform(
                (self.num_envs, 6), self.profile.kd_scale_range, generator=gain_generator, device="cpu"
            ).to(self.device)
            self.effort_limit_scale = _uniform(
                (self.num_envs, 6),
                self.profile.effort_limit_scale_range,
                generator=self._generators["effort_limit_rng"],
                device="cpu",
            ).to(self.device)
        else:
            self.friction_buckets = torch.ones(1, device="cpu")
            self.friction_bucket_ids = torch.zeros(self.num_envs, dtype=torch.long, device="cpu")
            self.wheel_friction = torch.ones(self.num_envs, device="cpu")
            self.kp_scale = torch.ones((self.num_envs, 6), device=self.device)
            self.kd_scale = torch.ones((self.num_envs, 6), device=self.device)
            self.effort_limit_scale = torch.ones((self.num_envs, 6), device=self.device)
        self.realized_stiffness = nominal_stiffness.unsqueeze(0) * self.kp_scale
        self.realized_damping = nominal_damping.unsqueeze(0) * self.kd_scale
        self.realized_effort_limit = nominal_effort.unsqueeze(0) * self.effort_limit_scale

    def _shape_spans(self) -> list[tuple[int, int]]:
        counts = []
        for link_path in self.robot.root_physx_view.link_paths[0]:
            link_view = self.robot._physics_sim_view.create_rigid_body_view(link_path)
            counts.append(int(link_view.max_shapes))
        if sum(counts) != int(self.robot.root_physx_view.max_shapes):
            raise RuntimeError(
                f"Unable to resolve articulation collision-shape spans: {counts} != "
                f"{self.robot.root_physx_view.max_shapes}"
            )
        spans = []
        start = 0
        for count in counts:
            spans.append((start, start + count))
            start += count
        return spans

    def _apply_wheel_materials(self) -> None:
        spans = self._shape_spans()
        materials = self.robot.root_physx_view.get_material_properties()
        friction = self.wheel_friction.to(dtype=materials.dtype)
        for body_id in self.wheel_body_ids:
            start, end = spans[body_id]
            materials[:, start:end, 0] = friction[:, None]
            materials[:, start:end, 1] = friction[:, None]
            materials[:, start:end, 2] = 0.0
        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device="cpu")
        self.robot.root_physx_view.set_material_properties(materials, env_ids)
        self.wheel_shape_spans = [spans[body_id] for body_id in self.wheel_body_ids]

    def _apply_actuator_parameters(self) -> None:
        self.robot.write_joint_stiffness_to_sim(
            self.realized_stiffness,
            joint_ids=self.controlled_joint_ids,
        )
        self.robot.write_joint_damping_to_sim(
            self.realized_damping,
            joint_ids=self.controlled_joint_ids,
        )
        self.robot.write_joint_effort_limit_to_sim(
            self.realized_effort_limit,
            joint_ids=self.controlled_joint_ids,
        )
        canonical_index = {joint_id: index for index, joint_id in enumerate(self.controlled_joint_ids)}
        for actuator_name in ("legs", "wheels"):
            actuator = self.robot.actuators[actuator_name]
            if isinstance(actuator.joint_indices, slice):
                joint_ids = list(range(self.robot.num_joints))[actuator.joint_indices]
            elif isinstance(actuator.joint_indices, torch.Tensor):
                joint_ids = actuator.joint_indices.tolist()
            else:
                joint_ids = list(actuator.joint_indices)
            columns = [canonical_index[int(joint_id)] for joint_id in joint_ids]
            actuator.stiffness.copy_(self.realized_stiffness[:, columns])
            actuator.damping.copy_(self.realized_damping[:, columns])
            actuator.effort_limit_sim.copy_(self.realized_effort_limit[:, columns])
            actuator.effort_limit.copy_(self.realized_effort_limit[:, columns])

    def sample_commands(self, count: int, ranges: CommandRanges) -> torch.Tensor:
        return sample_commands(
            count,
            device=self.device,
            ranges=ranges,
            generator=self._generators["command_rng"],
        )

    def sample_root_velocity(self, count: int) -> torch.Tensor:
        if not self.profile.enabled:
            return torch.zeros((count, 6), device=self.device)
        generator = self._generators["reset_velocity_rng"]
        linear_xy = _uniform(
            (count, 2), self.profile.root_linear_xy_range_mps, generator=generator, device="cpu"
        )
        linear_z = _uniform(
            (count, 1), self.profile.root_linear_z_range_mps, generator=generator, device="cpu"
        )
        angular = _uniform(
            (count, 3), self.profile.root_angular_range_rad_s, generator=generator, device="cpu"
        )
        control_velocity = torch.cat((linear_xy, linear_z, angular), dim=-1).to(self.device)
        linear_usd = transform_control_vector_to_usd(control_velocity[:, :3])
        angular_usd = transform_control_vector_to_usd(control_velocity[:, 3:])
        return torch.cat((linear_usd, angular_usd), dim=-1)

    def sample_actor_noise(self) -> ActorObservationNoiseV1 | None:
        if not self.profile.enabled:
            return None
        generator = self._generators["observation_noise_rng"]

        def noise(shape: tuple[int, ...], magnitude: float) -> torch.Tensor:
            return (2.0 * torch.rand(shape, device=self.device, generator=generator) - 1.0) * magnitude

        return ActorObservationNoiseV1(
            angular_velocity=noise((self.num_envs, 3), self.profile.angular_velocity_noise_rad_s),
            projected_gravity=noise((self.num_envs, 3), self.profile.projected_gravity_noise),
            leg_position_error=noise((self.num_envs, 4), self.profile.leg_position_noise_rad),
            joint_velocity=noise((self.num_envs, 6), self.profile.joint_velocity_noise_rad_s),
        )

    def get_rng_state(self) -> dict[str, Any]:
        return {
            "seed_derivation_version": RANDOMIZATION_SEED_DERIVATION_VERSION,
            "streams": {
                stream: {
                    "device": str(generator.device),
                    "state": generator.get_state().clone(),
                }
                for stream, generator in self._generators.items()
            },
        }

    def set_rng_state(
        self,
        payload: dict[str, Any],
        stream_names: Iterable[str] | None = None,
    ) -> None:
        payload = validate_randomization_rng_state_payload(payload)
        streams = payload["streams"]
        selected_streams = tuple(RANDOMIZATION_STREAMS if stream_names is None else stream_names)
        if len(selected_streams) != len(set(selected_streams)):
            raise ValueError("Randomization RNG stream selection contains duplicates")
        unknown_streams = set(selected_streams) - set(RANDOMIZATION_STREAMS)
        if unknown_streams:
            raise ValueError(f"Unknown randomization RNG streams: {sorted(unknown_streams)}")
        for stream in selected_streams:
            generator = self._generators[stream]
            saved = streams[stream]
            if saved.get("device") != str(generator.device):
                raise ValueError(
                    f"Randomization RNG device mismatch for {stream}: "
                    f"{saved.get('device')!r} != {str(generator.device)!r}"
                )
            state = saved.get("state")
            if not isinstance(state, torch.Tensor):
                raise ValueError(f"Randomization RNG tensor state is missing for {stream}")
            generator.set_state(state)

    @property
    def audit(self) -> dict[str, Any]:
        return copy.deepcopy(self._audit)

    @property
    def actuator_plan(self) -> dict[str, torch.Tensor]:
        return {
            "active_joint_realized_stiffness": self.realized_stiffness.detach().to(device="cpu").clone(),
            "active_joint_realized_damping": self.realized_damping.detach().to(device="cpu").clone(),
            "active_joint_realized_effort_limit": self.realized_effort_limit.detach().to(device="cpu").clone(),
        }

    @property
    def actuator_plan_hash(self) -> str:
        return canonical_tensor_sha256(self.actuator_plan)

    def _build_audit(self, reference_audit: dict[str, Any]) -> dict[str, Any]:
        tensors = {
            "friction_buckets": self.friction_buckets,
            "friction_bucket_ids": self.friction_bucket_ids,
            "q_reference": self.q_reference,
            "kp_scale": self.kp_scale,
            "kd_scale": self.kd_scale,
            "effort_limit_scale": self.effort_limit_scale,
        }
        payload = {
            "profile": profile_contract_payload(self.profile),
            "profile_hash": profile_contract_hash(self.profile),
            "master_seed": self.master_seed,
            "seed_derivation_version": RANDOMIZATION_SEED_DERIVATION_VERSION,
            "streams": seed_manifest(self.master_seed, str(self.device)),
            "friction_buckets": [float(value) for value in self.friction_buckets.tolist()],
            "realized": {
                "wheel_friction": _tensor_stats(self.wheel_friction),
                "leg_reference_offset_rad": _tensor_stats(self.reference_offset),
                "kp_scale": _tensor_stats(self.kp_scale),
                "kd_scale": _tensor_stats(self.kd_scale),
                "effort_limit_scale": _tensor_stats(self.effort_limit_scale),
            },
            "closed_chain_sampling": reference_audit,
            "implicit_torque_semantics": "implicit_pd_estimate",
            "realized_plan_hash": canonical_tensor_sha256(tensors),
            "actuator_plan_hash": self.actuator_plan_hash,
        }
        json.dumps(payload, sort_keys=True, allow_nan=False)
        return payload
