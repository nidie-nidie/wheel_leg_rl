from __future__ import annotations

from pathlib import Path


COLLECTOR = Path(__file__).resolve().parents[1] / "collect_dreamwaq_isaac_trace.py"


def test_isaac_collector_keeps_history_and_terminal_state_explicit() -> None:
    source = COLLECTOR.read_text(encoding="utf-8")
    required = (
        "actor.initialize_policy_input(current_observation)",
        "actor.current_observation(policy_input)",
        "actor.infer(policy_input)",
        "actor.advance_policy_input(",
        '"cenet_estimated_velocity"',
        '"cenet_context_mu"',
        '"cenet_context_logvar"',
        '"next_actor_obs_policy_returned"',
        "env._debug_terminal_state",
    )
    for token in required:
        assert token in source


def test_isaac_collector_forces_nominal_evaluation_randomization() -> None:
    source = COLLECTOR.read_text(encoding="utf-8")
    assert "cfg.randomization = NOMINAL_EVALUATION_PROFILE_V1" in source
