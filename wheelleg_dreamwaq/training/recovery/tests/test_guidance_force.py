from dataclasses import asdict

import numpy as np
import pytest
import torch

from recovery.contracts import GuidanceConfigV1
from recovery.guidance import RecoveryForceGuidance, raw_height_diagnostic, upward_force


def guidance(*, enabled=True):
    config = asdict(GuidanceConfigV1())
    config["force_enabled"] = enabled
    return RecoveryForceGuidance(config, torch.tensor([4.396253988146782, 5.0]))


def step(controller, heights):
    controller.begin_transition()
    for height in heights:
        controller.substep(torch.tensor(height))
    return controller.finish_transition()


def test_force_matches_independent_numpy_reference():
    heights = np.array([-2.0, 0.0, 0.15, 0.2, 0.25, 100.0], dtype=np.float64)
    expected = (1 - np.exp(-20 * np.maximum(0.2 - heights, 0))) * 0.75 * 21.5636258119
    actual = upward_force(torch.from_numpy(heights), torch.tensor(0.75), torch.tensor(21.5636258119, dtype=torch.float64), target_height=0.2, mu=20)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)
    assert torch.equal(actual[3:], torch.zeros(3, dtype=torch.float64))


def test_cost_is_arithmetic_mean_of_submitted_force():
    controller = guidance()
    heights = [[0.0, 0.1], [0.05, 0.15], [0.1, 0.2], [0.2, 0.3]]
    expected = np.mean([(1 - np.exp(-20 * np.maximum(0.2 - np.array(row), 0))) * controller.max_force_n.numpy() for row in heights], axis=0)
    transition = step(controller, heights)
    np.testing.assert_allclose(transition["cost_force_n"], expected, rtol=2e-7)
    assert controller.global_control_steps == 1
    assert controller.force_world[..., :2].count_nonzero() == 0
    assert controller.zero_torque_world.count_nonzero() == 0


def test_last_assisted_step_keeps_cost_and_next_alpha_zero():
    controller = guidance()
    controller.restore_progress(71999)
    transition = step(controller, [[0.1, 0.1]] * 4)
    assert (transition["cost_force_n"] > 0).all()
    assert (transition["alpha_t"] > 0).all()
    assert transition["alpha_next"].count_nonzero() == 0
    assert step(controller, [[0.1, 0.1]] * 4)["cost_force_n"].count_nonzero() == 0


def test_disabled_guidance_is_exactly_zero():
    controller = guidance(enabled=False)
    transition = step(controller, [[0.0, -0.2]] * 4)
    for key in ("cost_force_n", "peak_force_n", "alpha_t", "alpha_next"):
        assert transition[key].count_nonzero() == 0
    controller.restore_progress(24)
    assert controller.alpha.count_nonzero() == 0


def test_local_reset_does_not_change_progress_or_frozen_transition():
    controller = guidance()
    transition = step(controller, [[0.1, 0.1]] * 4)
    other_cost = controller.episode_discounted_cost[1].clone()
    controller.reset(torch.tensor([0]), completed=torch.tensor([True]))
    assert controller.global_control_steps == 1
    assert torch.equal(controller.last_transition["cost_force_n"], transition["cost_force_n"])
    assert torch.equal(controller.episode_discounted_cost[1], other_cost)
    assert int(controller.completed_episode_count) == 1
    controller.reset(torch.tensor([1]), completed=torch.tensor([False]))
    assert int(controller.completed_episode_count) == 1


def test_raw_overflow_is_reported_without_contaminating_actual_force():
    controller = guidance()
    transition = step(controller, [[100.0, 0.1]] * 4)
    diagnostic = controller.statistics()["raw_formula_diagnostic"]["samples"][0]
    assert diagnostic["raw_overflow"] and diagnostic["raw_coefficient"] is None
    assert transition["cost_force_n"][0] == 0
    assert torch.isfinite(transition["cost_force_n"]).all()
    ordinary = raw_height_diagnostic(torch.tensor([0.1, 0.25]), torch.ones(2), target_height=0.2, mu=20)
    np.testing.assert_allclose(ordinary["raw_coefficient"], 1 - np.exp(20 * (np.array([0.1, 0.25], dtype=np.float32).astype(np.float64) - 0.2)))


def test_invalid_substep_order_and_nonfinite_force_fail():
    controller = guidance()
    with pytest.raises(RuntimeError):
        controller.substep(torch.zeros(2))
    controller.begin_transition()
    with pytest.raises(RuntimeError):
        controller.finish_transition()
    with pytest.raises(RuntimeError):
        controller.substep(torch.tensor([float("nan"), 0.1]))
