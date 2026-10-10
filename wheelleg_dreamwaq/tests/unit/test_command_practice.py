from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from test_phase1_math import _state
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.commands import (
    CommandRanges, CommandPracticeState, command_contract_payload,
    command_practice_factor, sample_command_batch,
)
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.rewards import RewardWeights, compute_reward
from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.training_profiles import (
    apply_task_profile, resolve_task_profile, task_profile_from_contract,
)


def _cfg():
    return SimpleNamespace(commands=CommandRanges(), reward_weights=RewardWeights())


def test_real_tick_phase_boundaries_and_final_hold():
    steps = torch.tensor([0, 99, 100, 149, 150, 249, 250, 299, 300, 399, 400, 499, 500, 501])
    assert torch.equal(command_practice_factor(steps),
                       torch.tensor([1, 1, 0, 0, -1, -1, 0, 0, 1, 1, 0, 0, 0, 0]))


@pytest.mark.parametrize("steps", [torch.tensor([-1]), torch.tensor([0.5]), torch.tensor([True])])
def test_invalid_ticks_are_rejected(steps):
    with pytest.raises(ValueError):
        command_practice_factor(steps)


def test_initial_commands_do_not_alias_and_partial_reset_only_changes_selected_rows():
    initial = torch.tensor([[0.5, 0.0, 0.2], [0.0, -0.6, 0.17]])
    practice = CommandPracticeState(2, "cpu")
    practice.reset(torch.tensor([0, 1]), initial)
    initial.zero_()
    practice.steps[:] = torch.tensor([99, 149])
    command = practice.advance()
    assert torch.allclose(command, torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.6, 0.17]]))
    practice.reset(torch.tensor([0]), torch.tensor([[-0.4, 0.0, 0.23]]))
    assert practice.steps.tolist() == [0, 150]
    assert torch.allclose(practice.current_commands(),
                          torch.tensor([[-0.4, 0.0, 0.23], [0.0, 0.6, 0.17]]))


def test_profile_preserves_legacy_behavior_and_reset_rng():
    legacy, new = _cfg(), _cfg()
    apply_task_profile(legacy, "legacy_v1")
    apply_task_profile(new, "stop_reverse_v1")
    assert legacy.commands.hold_for_episode is True
    assert legacy.commands.practice_schedule == "disabled"
    assert legacy.reward_weights == RewardWeights()
    assert new.commands.hold_for_episode is False
    assert new.reward_weights.tracking_vx == new.reward_weights.tracking_vx_enhance == 2.0
    first = torch.Generator().manual_seed(101)
    second = torch.Generator().manual_seed(101)
    a = sample_command_batch(256, device="cpu", ranges=legacy.commands, generator=first)
    b = sample_command_batch(256, device="cpu", ranges=new.commands, generator=second)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    practice = CommandPracticeState(256, "cpu")
    practice.reset(torch.arange(256), b[0])
    practice.steps[:] = 149
    practice.advance()
    assert torch.equal(first.get_state(), second.get_state())


def test_only_vx_reward_contributions_are_doubled():
    legacy, new = _cfg(), _cfg()
    apply_task_profile(new, "stop_reverse_v1")
    state = _state(3)
    state.command[:, 0] = torch.tensor([0.0, 0.5, 1.0])
    state.root_angular_velocity[:] = 0.2
    state.applied_torque[:] = 2.0
    old_total, old_terms = compute_reward(state, legacy.reward_weights,
        control_dt=0.02, terminated=torch.tensor([False, True, False]))
    new_total, new_terms = compute_reward(state, new.reward_weights,
        control_dt=0.02, terminated=torch.tensor([False, True, False]))
    for name in old_terms:
        assert torch.equal(new_terms[name], old_terms[name] * (2.0 if name in
            {"tracking_vx", "tracking_vx_enhance"} else 1.0))
    assert torch.allclose(new_total - old_total, old_terms["tracking_vx"] +
                          old_terms["tracking_vx_enhance"], atol=1e-7)
    assert new_terms["tracking_vx_enhance"][2] < 0.0
    differences = {k for k, v in asdict(new.reward_weights).items()
                   if v != asdict(legacy.reward_weights)[k]}
    assert differences == {"tracking_vx", "tracking_vx_enhance"}


def test_schedule_contract_contains_actual_math_and_unknown_or_conflicting_profiles_fail():
    cfg = _cfg()
    assert command_contract_payload(cfg.commands)["practice_contract"] is None
    apply_task_profile(cfg, "stop_reverse_v1")
    payload = command_contract_payload(cfg.commands)
    practice = payload["practice_contract"]
    assert practice["stage_end_control_steps"] == [100, 150, 250, 300, 400, 500]
    assert practice["factors"] == [1, 0, -1, 0, 1, 0]
    assert practice["final_phase_behavior"] == "hold_last_factor"
    saved = {"task": {"commands": payload, "reward_weights": asdict(cfg.reward_weights)}}
    assert task_profile_from_contract(saved) == "stop_reverse_v1"
    assert resolve_task_profile(None, saved) == "stop_reverse_v1"
    with pytest.raises(ValueError, match="profile"):
        resolve_task_profile("legacy_v1", saved)
    with pytest.raises(ValueError):
        apply_task_profile(cfg, "unknown")
    with pytest.raises(ValueError):
        command_contract_payload(CommandRanges(practice_schedule="unknown"))
    with pytest.raises(ValueError):
        command_contract_payload(CommandRanges(hold_for_episode=False))

def test_isaac_and_numpy_practice_have_identical_boundary_math():
    import importlib.util
    from pathlib import Path
    import numpy as np

    module_path = Path(__file__).resolve().parents[2] / "sim2sim/mujoco/wheelleg_mujoco/command_practice.py"
    spec = importlib.util.spec_from_file_location("numpy_command_practice", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = _cfg()
    apply_task_profile(cfg, "stop_reverse_v1")
    assert command_contract_payload(cfg.commands)["practice_contract"] == module.STAGE_PAYLOAD
    ticks = list(range(502))
    factors = command_practice_factor(torch.tensor(ticks)).numpy()
    for tick, factor in zip(ticks, factors):
        result = module.command_at_tick(np.array([.5, .6, .2]), tick, module.STAGE_PAYLOAD)
        np.testing.assert_array_equal(result, np.array([.5 * factor, .6 * factor, .2]))
