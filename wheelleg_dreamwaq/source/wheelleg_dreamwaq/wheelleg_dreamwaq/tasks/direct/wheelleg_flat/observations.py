from __future__ import annotations

import torch

from wheelleg_dreamwaq.schemas.normalization import NormalizationV1
from wheelleg_dreamwaq.schemas.observation import assemble_actor_obs, assemble_critic_obs

from .state import WheelLegState


def build_observations(
    state: WheelLegState,
    *,
    q_nominal: torch.Tensor,
    normalization: NormalizationV1,
) -> tuple[torch.Tensor, torch.Tensor]:
    nominal = q_nominal.to(dtype=state.joint_position.dtype, device=state.joint_position.device)
    leg_error = state.joint_position[..., :4] - nominal
    actor_obs = assemble_actor_obs(
        angular_velocity=normalization.normalize_angular_velocity(state.root_angular_velocity),
        projected_gravity=normalization.normalize_projected_gravity(state.projected_gravity),
        command=normalization.normalize_command(state.command),
        leg_position_error=normalization.normalize_leg_position_error(leg_error),
        joint_velocity=normalization.normalize_joint_velocity(state.joint_velocity),
        previous_action=normalization.normalize_previous_action(state.last_applied_action),
    )
    critic_obs = assemble_critic_obs(
        actor_obs=actor_obs,
        root_linear_velocity=normalization.normalize_root_linear_velocity(state.root_com_linear_velocity),
        base_height=normalization.normalize_base_height(state.base_height),
        joint_acceleration=normalization.normalize_joint_acceleration(state.joint_acceleration),
        applied_torque=normalization.normalize_applied_torque(state.applied_torque),
    )
    if not torch.isfinite(actor_obs).all() or not torch.isfinite(critic_obs).all():
        raise RuntimeError("ActorObsV1 or CriticObsV1 contains NaN or Inf")
    return actor_obs, critic_obs
