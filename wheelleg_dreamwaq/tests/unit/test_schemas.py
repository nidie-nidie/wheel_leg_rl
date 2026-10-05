from __future__ import annotations

import torch

from wheelleg_dreamwaq.schemas.action import (
    ACTION_DIM,
    CANONICAL_JOINT_ORDER,
    CONTROLLED_JOINT_SIGN_USD,
    PACKET_TO_POLICY_LEG,
    POLICY_TO_PACKET_LEG,
    WHEEL_JOINT_SIGN_USD,
    controlled_joint_feedback_usd_to_control,
    packet_to_policy_legs,
    policy_to_packet_legs,
    wheel_targets_control_to_usd,
    wheel_feedback_usd_to_control,
)
from wheelleg_dreamwaq.schemas.command import COMMAND_DIM
from wheelleg_dreamwaq.schemas.observation import (
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    ActorObsSlices,
    CriticObsSlices,
    assemble_actor_obs,
    assemble_critic_obs,
)


def test_frozen_dimensions_and_joint_order() -> None:
    assert ACTION_DIM == 6
    assert COMMAND_DIM == 3
    assert ACTOR_OBS_DIM == 25
    assert CRITIC_OBS_DIM == 41
    assert CANONICAL_JOINT_ORDER == (
        "jIJ",
        "jIO",
        "jAB",
        "jAG",
        "jwheel_left",
        "jwheel_right",
    )
    assert POLICY_TO_PACKET_LEG == (1, 3, 0, 2)
    assert PACKET_TO_POLICY_LEG == (2, 0, 3, 1)
    assert WHEEL_JOINT_SIGN_USD == (1.0, -1.0)
    assert CONTROLLED_JOINT_SIGN_USD == (1.0, 1.0, 1.0, 1.0, 1.0, -1.0)


def test_leg_protocol_reorder_round_trip() -> None:
    policy = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    packet = policy_to_packet_legs(policy)
    assert torch.equal(packet, torch.tensor([[20.0, 40.0, 10.0, 30.0]]))
    assert torch.equal(packet_to_policy_legs(packet), policy)


def test_wheel_control_to_usd_sign() -> None:
    targets = torch.tensor([[5.0, 7.0]])
    assert torch.equal(wheel_targets_control_to_usd(targets), torch.tensor([[5.0, -7.0]]))


def test_wheel_feedback_maps_same_forward_motion_to_same_control_sign() -> None:
    usd_velocity = torch.tensor([[5.0, -7.0]])
    assert torch.equal(wheel_feedback_usd_to_control(usd_velocity), torch.tensor([[5.0, 7.0]]))
    controlled = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, -7.0]])
    assert torch.equal(
        controlled_joint_feedback_usd_to_control(controlled),
        torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 7.0]]),
    )


def test_actor_observation_layout() -> None:
    parts = {
        "angular_velocity": torch.full((2, 3), 1.0),
        "projected_gravity": torch.full((2, 3), 2.0),
        "command": torch.full((2, 3), 3.0),
        "leg_position_error": torch.full((2, 4), 4.0),
        "joint_velocity": torch.full((2, 6), 5.0),
        "previous_action": torch.full((2, 6), 6.0),
    }
    obs = assemble_actor_obs(**parts)
    assert obs.shape == (2, ACTOR_OBS_DIM)
    assert torch.all(obs[:, ActorObsSlices.ANGULAR_VELOCITY] == 1.0)
    assert torch.all(obs[:, ActorObsSlices.PROJECTED_GRAVITY] == 2.0)
    assert torch.all(obs[:, ActorObsSlices.COMMAND] == 3.0)
    assert torch.all(obs[:, ActorObsSlices.LEG_POSITION_ERROR] == 4.0)
    assert torch.all(obs[:, ActorObsSlices.JOINT_VELOCITY] == 5.0)
    assert torch.all(obs[:, ActorObsSlices.PREVIOUS_ACTION] == 6.0)


def test_critic_observation_layout() -> None:
    actor_obs = torch.arange(25, dtype=torch.float32).repeat(2, 1)
    critic_obs = assemble_critic_obs(
        actor_obs=actor_obs,
        root_linear_velocity=torch.full((2, 3), 25.0),
        base_height=torch.full((2, 1), 28.0),
        joint_acceleration=torch.full((2, 6), 29.0),
        applied_torque=torch.full((2, 6), 35.0),
    )
    assert critic_obs.shape == (2, CRITIC_OBS_DIM)
    assert torch.equal(critic_obs[:, CriticObsSlices.ACTOR_OBS], actor_obs)
    assert torch.all(critic_obs[:, CriticObsSlices.ROOT_LINEAR_VELOCITY] == 25.0)
    assert torch.all(critic_obs[:, CriticObsSlices.BASE_HEIGHT] == 28.0)
    assert torch.all(critic_obs[:, CriticObsSlices.JOINT_ACCELERATION] == 29.0)
    assert torch.all(critic_obs[:, CriticObsSlices.APPLIED_TORQUE] == 35.0)
