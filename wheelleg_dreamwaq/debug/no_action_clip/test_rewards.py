from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from debug.no_action_clip.rewards import (
    BASE_REWARD_PROFILE, CONTROL_DT, SPLIT_STRAIGHT_PROFILE,
    SplitStraightRewardWeights, resolve_reward_weights, reward_manifest_fields,
    straight_wheel_cost, validate_reward_manifest, wheel_differential_reward,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RewardWeights, compute_reward
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.state import WheelLegState


def test_exact_command_gate_and_raw_actions():
    commands = torch.tensor([
        [0.8, 0.0, 0.2], [-0.8, -0.0, 0.2], [0.0, 0.0, 0.2],
        [0.0, 0.5, 0.2], [0.8, 0.5, 0.2], [-0.8, -0.5, 0.2],
        [0.8, 1e-12, 0.2], [1e-12, 0.0, 0.2],
    ], dtype=torch.float64)
    actions = torch.tensor([0., 0., 0., 0., 2., -3.], dtype=torch.float64).repeat(8, 1)
    original_commands, original_actions = commands.clone(), actions.clone()
    weights = resolve_reward_weights(SPLIT_STRAIGHT_PROFILE, 0.1)
    reward, cost, enabled = wheel_differential_reward(actions, commands, weights, control_dt=CONTROL_DT)
    torch.testing.assert_close(cost, torch.tensor([25., 25., 0., 0., 0., 0., 0., 25.], dtype=torch.float64))
    assert enabled.tolist() == [True, True, False, False, False, False, False, True]
    torch.testing.assert_close(reward, -0.1 * 0.02 * cost)
    torch.testing.assert_close(actions, original_actions, rtol=0, atol=0)
    torch.testing.assert_close(commands, original_commands, rtol=0, atol=0)
    # 奖励已经计算完成，之后切换停车命令不能改写本步结果。
    commands.zero_()
    assert enabled[0] and cost[0] == 25.0


def test_quadratic_cost_direction_and_leg_independence():
    actions = torch.tensor([[0., 0., 0., 0., .1, 0.], [4., -5., 6., -7., .2, 0.],
                            [0., 0., 0., 0., 0., .2], [0., 0., 0., 0., 2., 2.]])
    commands = torch.tensor([[.8, 0., .2]]).repeat(4, 1)
    cost, _ = straight_wheel_cost(actions, commands)
    torch.testing.assert_close(cost, torch.tensor([.01, .04, .04, 0.]))
    assert cost[1] == cost[0] * 4


@pytest.mark.parametrize("coefficient", [None, 0., -0.1, float("nan"), float("inf"), True])
def test_invalid_coefficient_rejected(coefficient):
    with pytest.raises(ValueError):
        resolve_reward_weights(SPLIT_STRAIGHT_PROFILE, coefficient)


def test_profile_keeps_other_weights_and_legacy_identity():
    old = resolve_reward_weights(BASE_REWARD_PROFILE, None)
    new = resolve_reward_weights(SPLIT_STRAIGHT_PROFILE, .1)
    assert type(old) is RewardWeights
    assert new.action_smooth == 0 and new.action_smooth_leg == -.02 and new.action_smooth_wheel == -.01
    for name, value in asdict(old).items():
        if name != "action_smooth":
            assert getattr(new, name) == value
    for weights in (old, new):
        validate_reward_manifest(reward_manifest_fields(weights))
    with pytest.raises(ValueError):
        resolve_reward_weights(BASE_REWARD_PROFILE, .1)


@pytest.mark.parametrize("field", ["experiment_id", "task_profile", "reward_weights", "reward_contract"])
def test_missing_or_altered_reward_identity_rejected(field):
    valid = reward_manifest_fields(resolve_reward_weights(SPLIT_STRAIGHT_PROFILE, .1))
    missing = deepcopy(valid)
    missing.pop(field)
    with pytest.raises(ValueError):
        validate_reward_manifest(missing)
    altered = deepcopy(valid)
    altered[field] = {} if isinstance(valid[field], dict) else "unknown"
    with pytest.raises(ValueError):
        validate_reward_manifest(altered)


def test_coefficient_and_contract_must_match_saved_weights():
    valid = reward_manifest_fields(resolve_reward_weights(SPLIT_STRAIGHT_PROFILE, .1))
    for name, value in (("wheel_differential", -.2), ("action_smooth_leg", -.04), ("action_smooth", -.01)):
        altered = deepcopy(valid)
        altered["reward_weights"][name] = value
        with pytest.raises(ValueError):
            validate_reward_manifest(altered)
    legacy = reward_manifest_fields(resolve_reward_weights(BASE_REWARD_PROFILE, None))
    legacy["reward_contract"] = valid["reward_contract"]
    with pytest.raises(ValueError):
        validate_reward_manifest(legacy)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["actions", "commands"])
def test_nonfinite_reward_inputs_rejected(field, bad):
    actions, commands = torch.zeros(1, 6), torch.zeros(1, 3)
    (actions if field == "actions" else commands)[0, 0] = bad
    with pytest.raises(FloatingPointError):
        straight_wheel_cost(actions, commands)


def test_wrong_time_step_and_reward_sign_rejected():
    with pytest.raises(ValueError):
        SplitStraightRewardWeights(wheel_differential=0.1)
    with pytest.raises(ValueError):
        wheel_differential_reward(torch.zeros(1, 6), torch.zeros(1, 3),
                                  resolve_reward_weights(SPLIT_STRAIGHT_PROFILE, .1), control_dt=.001)


def test_parent_reward_uses_split_terms_without_composite_or_differential_duplication():
    zeros3, zeros6, zeros2 = torch.zeros(1, 3), torch.zeros(1, 6), torch.zeros(1, 2)
    state = WheelLegState(
        root_link_pos_w=torch.tensor([[0., 0., .2]]),
        root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
        root_com_pos_w=torch.tensor([[0., 0., .2]]), root_com_linear_velocity=zeros3.clone(),
        root_angular_velocity=zeros3.clone(), projected_gravity=torch.tensor([[0., 0., -1.]]),
        base_height=torch.tensor([[.2]]), joint_position=zeros6.clone(), joint_velocity=zeros6.clone(),
        joint_acceleration=zeros6.clone(), applied_torque=zeros6.clone(),
        last_applied_action=torch.tensor([[2., -3., 4., -5., 2., 3.]]),
        previous_applied_action=zeros6.clone(), previous_previous_applied_action=zeros6.clone(),
        command=torch.tensor([[.8, 0., .2]]),
        virtual_leg_length_true=torch.full((1, 2), .2), virtual_leg_phi0_true=zeros2.clone(),
        virtual_leg_length_fk=torch.full((1, 2), .2), virtual_leg_phi0_fk=zeros2.clone(),
        virtual_leg_fk_valid=torch.ones(1, 2, dtype=torch.bool), virtual_leg_wheel_error=zeros2.clone(),
        loop_closure_position_error=torch.zeros(1, 4),
    )
    weights = resolve_reward_weights(SPLIT_STRAIGHT_PROFILE, .1)
    parent_reward, terms = compute_reward(state, weights, control_dt=CONTROL_DT, terminated=torch.tensor([False]))
    assert "action_smooth" not in terms and "wheel_differential" not in terms
    torch.testing.assert_close(terms["action_smooth_leg"], torch.tensor([-.02 * .02 * 54.]))
    torch.testing.assert_close(terms["action_smooth_wheel"], torch.tensor([-.01 * .02 * 13.]))
    weighted, _, _ = wheel_differential_reward(state.last_applied_action, state.command,
                                             weights, control_dt=CONTROL_DT)
    torch.testing.assert_close(weighted, torch.tensor([-.1 * .02]))
    torch.testing.assert_close(parent_reward + weighted, torch.stack([*terms.values(), weighted]).sum(0))
