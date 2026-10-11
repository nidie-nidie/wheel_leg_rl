from dataclasses import replace
from copy import deepcopy
import math

import pytest
import torch

from wheelleg_dreamwaq.tasks.direct.wheelleg_flat.leg_reversal import (
    LegReversalConfig, LegTargetReversalCounter, LegReversalReward, validate_gain_state,
)


def counter(num_envs=1, **kwargs):
    cfg = replace(LegReversalConfig(enabled=True), **kwargs)
    c = LegTargetReversalCounter(num_envs, cfg, "cpu")
    c.reset(torch.arange(num_envs), torch.zeros(num_envs, 4))
    return c


def target(x, n=1):
    return torch.tensor([[x, 0., 0., 0.]] * n)


def test_extremum_hysteresis_counts_directions_not_action_zero_crossings():
    c = counter()
    for x, expected in [(.1, 0), (.2, 0), (.195, 0), (.188, 1),
                        (.188, 1), (.19, 1), (.21, 2)]:
        c.update(target(x))
        assert c.count[0, 0] == expected
    # All targets were positive. The events still correctly count two reversals.
    assert c.direction[0, 0] == 1


def test_subthreshold_noise_is_ignored_but_slow_accumulated_motion_counts():
    c = counter()
    for x in [.004, -.004] * 5:
        c.update(target(x))
    assert c.count.sum() == 0
    for x in [.004, .008, .012, .016, .02, .017, .014, .011, .008]:
        c.update(target(x))
    assert c.count[0, 0] == 1


def test_full_window_and_event_expiration():
    c = counter()
    c.update(target(.1))
    c.update(target(0.))
    assert c.count[0, 0] == 1
    assert not c.eligible.any()
    for _ in range(48):
        c.update(target(0.))
    assert c.eligible.all() and c.rate[0, 0] == 1.
    c.update(target(0.))
    assert c.count[0, 0] == 1
    c.update(target(0.))
    assert c.count.sum() == 0


def test_partial_reset_keeps_other_environment_and_does_not_cross_episode():
    c = counter(2)
    c.update(target(.1, 2))
    c.update(target(0., 2))
    snapshot = {n: getattr(c, n).clone() for n in ("extreme", "direction", "events", "count", "age")}
    cursor = c.cursor
    c.reset(torch.tensor([0]), target(0.))
    assert c.cursor == cursor
    for name in ("extreme", "direction", "count", "age"):
        assert torch.equal(getattr(c, name)[1], snapshot[name][1])
    assert torch.equal(c.events[:, 1], snapshot["events"][:, 1])
    c.update(target(.1, 2))
    assert c.count[0, 0] == 0 and c.count[1, 0] == 2


def test_gain_fixed_within_rollout_and_bounded_reward_excludes_warmup():
    cfg = replace(LegReversalConfig(enabled=True), budget_per_second=8.)
    r = LegReversalReward(1, cfg, "cpu")
    r.reset(torch.tensor([0]), target(0.))
    used = []
    for i in range(96):
        penalty = r.step(target(.1 if i % 2 == 0 else 0.)).clone()
        used.append(r.gain_used)
        if i < 49:
            assert penalty == 0
        if i % 24 != 23:
            with pytest.raises(RuntimeError, match="boundary"):
                r.state_dict()
    assert all(v == 0 for v in used[:72])
    assert all(v == 1 for v in used[72:96])
    assert r.completed_rollouts == 4 and r.gain == 1.
    expected = -.02 * .02 * (50 / 8 - 1)
    assert math.isclose(float(r.cached_reward), expected, abs_tol=1e-8)
    state = r.state_dict()
    restored = LegReversalReward(2, cfg, "cpu")
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert not restored.counter.eligible.any()
    gain = r.gain
    r.reset(torch.tensor([0]), target(0.))
    assert r.gain == gain and r.completed_rollouts == 4


def test_zero_excess_included_and_wrong_state_rejected():
    cfg = replace(LegReversalConfig(enabled=True))
    r = LegReversalReward(2, cfg, "cpu")
    r.reset(torch.arange(2), torch.zeros(2, 4))
    for i in range(72):
        values = target(.1 if i % 2 == 0 else 0., 2)
        values[1] = 0
        r.step(values)
    assert r.latest["mean_excess"][0] > 0
    assert r.latest["valid_windows"] > 0
    state = r.state_dict()
    for key, bad in [("gain", float("nan")), ("gain", 1.1), ("completed_rollouts", 2.5)]:
        mutated = deepcopy(state)
        mutated[key] = bad
        with pytest.raises(ValueError):
            validate_gain_state(mutated, state["contract"])
    with pytest.raises(ValueError):
        validate_gain_state(state, state["contract"], completed_rollouts=99)


@pytest.mark.parametrize("changes", [
    {"amplitude_rad": 0}, {"window_seconds": .015}, {"budget_per_second": float("nan")},
    {"max_penalty_weight": -1}, {"rollout_steps": True},
])
def test_invalid_parameters_rejected(changes):
    with pytest.raises(ValueError):
        replace(LegReversalConfig(), **changes).validate()
