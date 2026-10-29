from copy import deepcopy
import json

import pytest

from recovery.contracts import PROJECT_ROOT, file_sha256, stable_hash, validate_resolved_config
from recovery.independent_config import default_independent_config
from recovery.independent_poses import IndependentPoseBank
from recovery.independent_preflight import independent_preflight


def real_config():
    root = PROJECT_ROOT / "recovery/artifacts/independent-stand-implementation-20261029"
    config = default_independent_config()
    pose_path = root / "pose-generation-check/generated-v3/pose_bank.json"
    bank = IndependentPoseBank.load(pose_path)
    ids = bank.split_ids()
    config["initialization"].update(train_ids=ids["train"], test_ids=ids["test"])
    config["environment"].update(pose_bank_path=str(pose_path), run_dir=str(root / "preflight-test-unused-run"))
    config["geometry_guard"] = json.loads((root / "pose-generation-check/geometry_config.json").read_text(encoding="utf-8"))
    calibration = root / "standing_calibration.json"
    config["assessment"].update(calibration_path=str(calibration), calibration_sha256=file_sha256(calibration))
    return config


def test_real_new_schema_preflight_uses_nominal_splits_without_walker_handoff():
    config = real_config()
    validate_resolved_config(config)
    identity = independent_preflight(config)
    assert identity["schema_version"] == "RecoveryIndependentStandV1"
    assert identity["config_sha256"] == stable_hash(config)
    assert identity["pose_bank_sha256"] == file_sha256(config["environment"]["pose_bank_path"])
    assert "walker_actor_sha256" not in identity


def test_real_four_environment_evidence_cannot_be_used_as_formal_clearance():
    config = real_config()
    config["environment"].update(num_envs=4096, task_type="formal_training")
    config["ppo"]["max_iterations"] = 4000
    validate_resolved_config(config)
    with pytest.raises(ValueError, match="准入证据"):
        independent_preflight(config)


def test_actual_train_test_record_overlap_is_rejected_before_physics():
    config = real_config()
    config["initialization"]["test_ids"]["front"] = deepcopy(config["initialization"]["train_ids"]["front"])
    with pytest.raises(ValueError, match="ID必须分离"):
        validate_resolved_config(config)
