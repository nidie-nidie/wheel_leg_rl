from dataclasses import asdict

import pytest

from recovery.contracts import RecoveryContractV2, stable_hash, validate_resolved_config
from recovery.independent_config import IndependentStandingContractV1, default_independent_config
from recovery.independent_state import StandingGateConfig


def test_new_contract_rejects_old_models_and_has_distinct_identity():
    old = RecoveryContractV2()
    new = IndependentStandingContractV1()
    assert old.sha256 != new.sha256
    assert (new.actor_input_dim, new.critic_input_dim, new.action_dim) == (110, 41, 6)
    assert new.wheel_action_scale == 30 and new.leg_action_scale == 1
    with pytest.raises(ValueError):
        IndependentStandingContractV1.from_dict(asdict(old))
    with pytest.raises(ValueError):
        RecoveryContractV2.from_dict(asdict(new))
    assert IndependentStandingContractV1.from_dict(asdict(new)).sha256 == stable_hash(asdict(new))


def test_fixed_action_or_termination_changes_require_another_version():
    for change in ({"wheel_action_scale": 25.}, {"critic_input_dim": 34}, {"max_recovery_steps": 400},
                   {"success_reward": 10.}, {"action_distribution": "normal"}):
        with pytest.raises(ValueError):
            IndependentStandingContractV1(**change)


def test_unprepared_draft_cannot_launch_and_has_no_old_handoff_requirements():
    config = default_independent_config()
    assert config["schema_version"] == "RecoveryIndependentStandV1"
    assert config["standing"] == asdict(StandingGateConfig())
    assert "walker_export_dir" not in config["environment"]
    assert "guidance" not in config
    with pytest.raises(ValueError, match="几何检查"):
        validate_resolved_config(config, check_paths=False)
