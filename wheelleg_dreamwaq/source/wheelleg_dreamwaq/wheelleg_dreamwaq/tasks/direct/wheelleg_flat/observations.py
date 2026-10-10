from __future__ import annotations

from dataclasses import dataclass

import torch

from wheelleg_dreamwaq.schemas.normalization import NormalizationV1
from wheelleg_dreamwaq.schemas.observation import assemble_actor_obs, assemble_critic_obs

from .state import WheelLegState


@dataclass(frozen=True)
class PhysicalActorFieldsV1:
    angular_velocity: torch.Tensor
    projected_gravity: torch.Tensor
    command: torch.Tensor
    leg_position_error: torch.Tensor
    joint_velocity: torch.Tensor
    previous_action: torch.Tensor


@dataclass(frozen=True)
class ActorObservationNoiseV1:
    angular_velocity: torch.Tensor
    projected_gravity: torch.Tensor
    leg_position_error: torch.Tensor
    joint_velocity: torch.Tensor


def build_physical_actor_fields(
    state: WheelLegState,
    *,
    q_reference: torch.Tensor,
) -> PhysicalActorFieldsV1:
    reference = q_reference.to(dtype=state.joint_position.dtype, device=state.joint_position.device)
    if reference.shape[-1] != 4:
        raise ValueError(f"Expected four leg reference positions, got {tuple(reference.shape)}")
    if reference.ndim > 1 and reference.shape[:-1] != state.joint_position.shape[:-1]:
        expected = (*state.joint_position.shape[:-1], 4)
        raise ValueError(
            "Per-environment leg reference batch does not match state: "
            f"{tuple(reference.shape)} != {expected}"
        )
    return PhysicalActorFieldsV1(
        angular_velocity=state.root_angular_velocity,
        projected_gravity=state.projected_gravity,
        command=state.command,
        leg_position_error=state.joint_position[..., :4] - reference,
        joint_velocity=state.joint_velocity,
        previous_action=state.last_applied_action,
    )


def add_actor_observation_noise(
    fields: PhysicalActorFieldsV1,
    noise: ActorObservationNoiseV1,
) -> PhysicalActorFieldsV1:
    pairs = (
        ("angular_velocity", fields.angular_velocity, noise.angular_velocity),
        ("projected_gravity", fields.projected_gravity, noise.projected_gravity),
        ("leg_position_error", fields.leg_position_error, noise.leg_position_error),
        ("joint_velocity", fields.joint_velocity, noise.joint_velocity),
    )
    for name, values, perturbation in pairs:
        if perturbation.shape != values.shape:
            raise ValueError(
                f"Actor observation noise shape mismatch for {name}: "
                f"{tuple(perturbation.shape)} != {tuple(values.shape)}"
            )
    return PhysicalActorFieldsV1(
        angular_velocity=fields.angular_velocity + noise.angular_velocity,
        projected_gravity=fields.projected_gravity + noise.projected_gravity,
        command=fields.command,
        leg_position_error=fields.leg_position_error + noise.leg_position_error,
        joint_velocity=fields.joint_velocity + noise.joint_velocity,
        previous_action=fields.previous_action,
    )


def normalize_actor_fields(
    fields: PhysicalActorFieldsV1,
    normalization: NormalizationV1,
) -> torch.Tensor:
    return assemble_actor_obs(
        angular_velocity=normalization.normalize_angular_velocity(fields.angular_velocity),
        projected_gravity=normalization.normalize_projected_gravity(fields.projected_gravity),
        command=normalization.normalize_command(fields.command),
        leg_position_error=normalization.normalize_leg_position_error(fields.leg_position_error),
        joint_velocity=normalization.normalize_joint_velocity(fields.joint_velocity),
        previous_action=normalization.normalize_previous_action(fields.previous_action),
    )


def build_observations(
    state: WheelLegState,
    *,
    q_reference: torch.Tensor | None = None,
    q_nominal: torch.Tensor | None = None,
    normalization: NormalizationV1,
    actor_noise: ActorObservationNoiseV1 | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (q_reference is None) == (q_nominal is None):
        raise ValueError("Provide exactly one of q_reference or q_nominal")
    reference = q_reference if q_reference is not None else q_nominal
    assert reference is not None
    physical_fields = build_physical_actor_fields(state, q_reference=reference)
    clean_actor_features = normalize_actor_fields(physical_fields, normalization)
    actor_fields = (
        add_actor_observation_noise(physical_fields, actor_noise)
        if actor_noise is not None
        else physical_fields
    )
    actor_obs = normalize_actor_fields(actor_fields, normalization)
    critic_obs = assemble_critic_obs(
        actor_obs=clean_actor_features,
        root_linear_velocity=normalization.normalize_root_linear_velocity(state.root_com_linear_velocity),
        base_height=normalization.normalize_base_height(state.base_height),
        joint_acceleration=normalization.normalize_joint_acceleration(state.joint_acceleration),
        applied_torque=normalization.normalize_applied_torque(state.applied_torque),
    )
    if not torch.isfinite(actor_obs).all() or not torch.isfinite(critic_obs).all():
        raise RuntimeError("ActorObsV1 or CriticObsV1 contains NaN or Inf")
    return actor_obs, critic_obs
