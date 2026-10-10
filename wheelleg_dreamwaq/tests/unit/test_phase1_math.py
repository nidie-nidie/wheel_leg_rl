from __future__ import annotations

import torch

from wheelleg_dreamwaq.schemas.normalization import NormalizationV1
from wheelleg_dreamwaq.schemas.observation import ActorObsSlices, CriticObsSlices
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.control import ControlLimits, compute_action_targets
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.observations import (
    ActorObservationNoiseV1,
    build_observations,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RewardWeights, compute_reward, compute_reward_terms
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.state import WheelLegState
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.terminations import TerminationLimits, compute_dones


def _state(batch: int = 2) -> WheelLegState:
    zeros3 = torch.zeros(batch, 3)
    zeros6 = torch.zeros(batch, 6)
    return WheelLegState(
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.20]]).repeat(batch, 1),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch, 1),
        root_com_pos_w=torch.tensor([[0.0, 0.0, 0.20]]).repeat(batch, 1),
        root_com_linear_velocity=zeros3.clone(),
        root_angular_velocity=zeros3.clone(),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]).repeat(batch, 1),
        base_height=torch.full((batch, 1), 0.20),
        joint_position=torch.tensor([[-0.30, 0.35, -0.30, 0.35, 0.0, 0.0]]).repeat(batch, 1),
        joint_velocity=zeros6.clone(),
        joint_acceleration=zeros6.clone(),
        applied_torque=zeros6.clone(),
        last_applied_action=zeros6.clone(),
        previous_applied_action=zeros6.clone(),
        previous_previous_applied_action=zeros6.clone(),
        command=torch.tensor([[0.0, 0.0, 0.20]]).repeat(batch, 1),
        virtual_leg_length_true=torch.full((batch, 2), 0.20),
        virtual_leg_phi0_true=torch.full((batch, 2), torch.pi / 2.0),
        virtual_leg_length_fk=torch.full((batch, 2), 0.20),
        virtual_leg_phi0_fk=torch.full((batch, 2), torch.pi / 2.0),
        virtual_leg_fk_valid=torch.ones(batch, 2, dtype=torch.bool),
        virtual_leg_wheel_error=torch.zeros(batch, 2),
        loop_closure_position_error=torch.zeros(batch, 4),
    )


def test_action_targets_use_mixed_position_and_velocity_semantics() -> None:
    action = torch.tensor([[2.0, -2.0, 0.5, -0.5, 0.4, 0.4]])
    nominal = torch.tensor([-0.30, 0.35, -0.30, 0.35])
    clipped, leg_targets, wheel_targets = compute_action_targets(action, nominal, ControlLimits())

    assert torch.equal(clipped, torch.tensor([[1.0, -1.0, 0.5, -0.5, 0.4, 0.4]]))
    assert torch.allclose(leg_targets, torch.tensor([[0.05, 0.00, -0.125, 0.175]]))
    assert torch.allclose(wheel_targets, torch.tensor([[10.0, -10.0]]))


def test_phase1_observation_dimensions_are_frozen() -> None:
    actor, critic = build_observations(
        _state(),
        q_nominal=torch.tensor([-0.30, 0.35, -0.30, 0.35]),
        normalization=NormalizationV1(),
    )
    assert actor.shape == (2, 25)
    assert critic.shape == (2, 41)
    assert torch.isfinite(actor).all()
    assert torch.isfinite(critic).all()


def test_observation_supports_per_environment_leg_reference() -> None:
    state = _state()
    references = torch.tensor(
        [
            [-0.30, 0.35, -0.30, 0.35],
            [-0.20, 0.25, -0.20, 0.25],
        ]
    )
    actor, critic = build_observations(
        state,
        q_reference=references,
        normalization=NormalizationV1(),
    )
    expected_error = state.joint_position[:, :4] - references
    assert torch.allclose(actor[:, ActorObsSlices.LEG_POSITION_ERROR], expected_error)
    assert torch.allclose(critic[:, CriticObsSlices.ACTOR_OBS], actor)


def test_actor_noise_is_added_in_physical_units_and_critic_stays_clean() -> None:
    state = _state(1)
    normalization = NormalizationV1()
    reference = torch.tensor([-0.30, 0.35, -0.30, 0.35])
    noise = ActorObservationNoiseV1(
        angular_velocity=torch.full((1, 3), 0.20),
        projected_gravity=torch.full((1, 3), 0.05),
        leg_position_error=torch.full((1, 4), 0.02),
        joint_velocity=torch.full((1, 6), 1.50),
    )
    actor, critic = build_observations(
        state,
        q_reference=reference,
        normalization=normalization,
        actor_noise=noise,
    )
    clean_actor, _ = build_observations(
        state,
        q_reference=reference,
        normalization=normalization,
    )

    assert torch.allclose(
        actor[:, ActorObsSlices.ANGULAR_VELOCITY]
        - clean_actor[:, ActorObsSlices.ANGULAR_VELOCITY],
        torch.full((1, 3), 0.05),
    )
    assert torch.allclose(
        actor[:, ActorObsSlices.PROJECTED_GRAVITY]
        - clean_actor[:, ActorObsSlices.PROJECTED_GRAVITY],
        torch.full((1, 3), 0.05),
    )
    assert torch.allclose(
        actor[:, ActorObsSlices.LEG_POSITION_ERROR]
        - clean_actor[:, ActorObsSlices.LEG_POSITION_ERROR],
        torch.full((1, 4), 0.02),
    )
    assert torch.allclose(
        actor[:, ActorObsSlices.JOINT_VELOCITY]
        - clean_actor[:, ActorObsSlices.JOINT_VELOCITY],
        torch.full((1, 6), 0.075),
    )
    assert torch.allclose(critic[:, CriticObsSlices.ACTOR_OBS], clean_actor)
    assert actor.untyped_storage().data_ptr() != critic.untyped_storage().data_ptr()


def test_tracking_reward_prefers_matching_velocity_and_height() -> None:
    matched = _state(1)
    missed = _state(1)
    matched.command[:] = torch.tensor([[0.5, 0.5, 0.22]])
    matched.root_com_linear_velocity[:, 0] = 0.5
    matched.root_angular_velocity[:, 2] = 0.5
    matched.base_height[:] = 0.22
    missed.command[:] = matched.command

    matched_reward, _ = compute_reward(matched, RewardWeights(), control_dt=0.02, terminated=torch.zeros(1, dtype=torch.bool))
    missed_reward, _ = compute_reward(missed, RewardWeights(), control_dt=0.02, terminated=torch.zeros(1, dtype=torch.bool))
    assert matched_reward.item() > missed_reward.item()


def test_termination_keeps_timeout_separate_from_failure() -> None:
    state = _state(2)
    state.base_height[0] = 0.05
    episode_length = torch.tensor([20, 499])
    terminated, truncated, diagnostics = compute_dones(
        state,
        episode_length,
        max_episode_length=500,
        limits=TerminationLimits(grace_steps=10),
    )
    assert torch.equal(terminated, torch.tensor([True, False]))
    assert torch.equal(truncated, torch.tensor([False, True]))
    assert torch.equal(diagnostics["height_terminated"], torch.tensor([True, False]))


def test_tilt_termination_handles_boundary_sideways_and_inverted() -> None:
    limits = TerminationLimits(grace_steps=0)
    state = _state(4)
    angles = (0.0, limits.max_tilt_rad, 0.5 * torch.pi, torch.pi)
    state.projected_gravity[:] = torch.tensor(
        [[float(torch.sin(torch.tensor(angle))), 0.0, -float(torch.cos(torch.tensor(angle)))] for angle in angles]
    )
    terminated, _, diagnostics = compute_dones(
        state,
        torch.ones(4, dtype=torch.long),
        max_episode_length=500,
        limits=limits,
    )
    assert torch.equal(diagnostics["tilt_bad"], torch.tensor([False, False, True, True]))
    assert torch.equal(terminated, torch.tensor([False, False, True, True]))


def test_orientation_penalty_distinguishes_sideways_and_inverted() -> None:
    state = _state(3)
    state.projected_gravity[:] = torch.tensor(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    terms = compute_reward_terms(state, RewardWeights(), torch.zeros(3, dtype=torch.bool))
    assert torch.equal(terms["orientation"], torch.tensor([0.0, 1.0, 2.0]))


def test_phi0_symmetry_reward_uses_wrapped_difference() -> None:
    state = _state(2)
    state.virtual_leg_phi0_true[:] = torch.tensor(
        [[torch.pi - 0.1, -torch.pi + 0.1], [1.2, 1.2]]
    )
    terms = compute_reward_terms(state, RewardWeights(), torch.zeros(2, dtype=torch.bool))
    assert torch.allclose(terms["phi0_symmetry"], torch.tensor([0.04, 0.0]), atol=1.0e-6)
