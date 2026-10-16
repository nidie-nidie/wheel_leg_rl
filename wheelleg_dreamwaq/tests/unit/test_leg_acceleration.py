from copy import deepcopy
import math

import pytest
import torch

from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.leg_acceleration import (
    DEFAULT_REFERENCE_RMS_RAD_S2,
    LEG_ACCELERATION_JOINTS,
    LegAccelerationConfig,
    LegAccelerationReward,
    acceleration_contract_payload,
    validate_acceleration_gain_state,
)


def test_budget_is_per_joint_and_scales_by_fraction():
    cfg = LegAccelerationConfig(enabled=True, budget_fraction=0.80)
    reward = LegAccelerationReward(1, cfg, "cpu")
    assert reward.budget.tolist() == pytest.approx(
        [value * 0.80 for value in DEFAULT_REFERENCE_RMS_RAD_S2]
    )
    assert acceleration_contract_payload(cfg)["joint_names"] == list(LEG_ACCELERATION_JOINTS)


def test_positive_part_cost_uses_actual_first_four_joints_only():
    cfg = LegAccelerationConfig(enabled=True, budget_fraction=0.80, rollout_steps=2)
    reward = LegAccelerationReward(2, cfg, "cpu")
    reward.reset(torch.arange(2))
    values = torch.tensor([
        [0., 0., 0., 0., 1_000., 1_000.],
        [2.0 * reward.budget[0], reward.budget[1], reward.budget[2], reward.budget[3], 1_000., 1_000.],
    ])
    reward.step(values)
    assert torch.equal(reward.excess[0], torch.zeros(4))
    assert reward.excess[1, 0] == pytest.approx(3.0)
    assert torch.equal(reward.cached_reward, torch.zeros(2))


def test_gain_updates_once_at_rollout_boundary_and_checkpoint_round_trips():
    cfg = LegAccelerationConfig(enabled=True, rollout_steps=2)
    reward = LegAccelerationReward(1, cfg, "cpu")
    reward.reset(torch.tensor([0]))
    reward.step(torch.full((1, 6), 100.0))
    assert reward.completed_rollouts == 0
    with pytest.raises(RuntimeError, match="boundary"):
        reward.state_dict()
    reward.step(torch.full((1, 6), 100.0))
    assert reward.completed_rollouts == 1
    assert reward.gain > 0
    state = reward.state_dict()
    restored = LegAccelerationReward(1, cfg, "cpu")
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_partial_reset_clears_only_selected_cached_rows():
    cfg = LegAccelerationConfig(enabled=True)
    reward = LegAccelerationReward(2, cfg, "cpu")
    reward.reset(torch.arange(2))
    reward.step(torch.full((2, 6), 100.0))
    snapshot = reward.cached_reward.clone()
    reward.reset(torch.tensor([0]))
    assert reward.cached_reward[0] == 0
    assert reward.cached_reward[1] == snapshot[1]
    assert reward.excess[0].sum() == 0
    assert reward.excess[1].sum() > 0


def test_invalid_budget_and_state_are_rejected():
    with pytest.raises(ValueError):
        LegAccelerationConfig(reference_rms_rad_s2=(1., 2., 3.)).validate()
    with pytest.raises(ValueError):
        LegAccelerationConfig(budget_fraction=0).validate()
    cfg = LegAccelerationConfig(enabled=True, rollout_steps=1)
    reward = LegAccelerationReward(1, cfg, "cpu")
    reward.reset(torch.tensor([0]))
    reward.step(torch.full((1, 6), 100.0))
    state = reward.state_dict()
    bad = deepcopy(state)
    bad["gain"] = math.nan
    with pytest.raises(ValueError):
        validate_acceleration_gain_state(bad, state["contract"])



def test_batch_shape_mismatch_is_rejected_without_broadcast():
    reward = LegAccelerationReward(2, LegAccelerationConfig(enabled=True), "cpu")
    with pytest.raises(ValueError, match="shape"):
        reward.step(torch.zeros(1, 6))
    assert reward.step_in_rollout == 0
