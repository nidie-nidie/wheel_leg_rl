from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import run_project_script


@pytest.mark.integration
@pytest.mark.parametrize("randomized", (False, True))
def test_isaac_has_no_operational_velocity_caps(tmp_path: Path, randomized: bool) -> None:
    output = tmp_path / "unrestricted-dynamics.json"
    arguments = ["scripts/check_unrestricted_dynamics.py", "--output", str(output), "--headless"]
    if randomized:
        arguments.append("--randomized")
    run_project_script(arguments)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["physics_schema_version"] == "PhysicsV5"
    assert len(report["robot_bodies"]) == 27
    assert report["joint_velocity_limit_shape"] == [8, 26]
    assert report["policy_shape"] == [8, 25]
    assert report["critic_shape"] == [8, 41]
    assert report["free_body_probes"]["unrestricted"]["linear_speed_mps"] == pytest.approx(200.0, abs=1.e-3)
    assert report["free_body_probes"]["unrestricted"]["angular_speed_rad_s"] == pytest.approx(10.0, abs=1.e-3)
    assert report["finite"] is True
    if randomized:
        assert report["historical_cache_rejected"] is True
