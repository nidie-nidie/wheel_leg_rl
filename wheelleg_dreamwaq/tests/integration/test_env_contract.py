from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import run_project_script


@pytest.mark.integration
def test_environment_observation_reset_and_frame_contract(tmp_path: Path) -> None:
    output = tmp_path / "random-action-smoke.json"
    run_project_script(
        [
            "scripts/smoke_random_actions.py",
            "--num-envs",
            "8",
            "--steps",
            "1000",
            "--output",
            str(output),
            "--headless",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["finite"] is True
    assert report["steps"] == 1000
    assert report["policy_observation_shape"] == [8, 25]
    assert report["critic_observation_shape"] == [8, 41]
    assert report["reset_count"] > 0
    assert report["reset_previous_action_leak_count"] == 0
    assert report["max_actor_joint_velocity_error"] <= 1.0e-5
    assert report["max_critic_joint_acceleration_error"] <= 1.0e-5
    assert report["max_com_velocity_frame_error"] <= 1.0e-5
    assert report["max_loop_closure_position_error"] <= 5.0e-3
    assert report["max_virtual_leg_wheel_error"] <= 5.0e-3
    assert report["max_virtual_leg_length_error"] <= 5.0e-3
    assert report["max_virtual_leg_phi0_error_rad"] <= 0.0523598776
    assert report["virtual_leg_length_range"][0] > 0.05
    assert "phi0_symmetry" in report["reward_terms"]
