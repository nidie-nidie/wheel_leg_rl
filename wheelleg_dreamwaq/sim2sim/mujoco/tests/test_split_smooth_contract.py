import json
from pathlib import Path
import shutil

import pytest

from wheelleg_mujoco.contract import load_policy_contract, _stable_hash

PROJECT = Path(__file__).resolve().parents[3]
BASELINE = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/exports/run-03"


def prepare(tmp_path, leg_weight=-.02):
    for path in BASELINE.iterdir():
        shutil.copy2(path, tmp_path / path.name)
    path = tmp_path / "policy_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schemas"]["reward"] = "RewardSchemaV3SplitSmooth"
    manifest["training_reward_weights"].update(action_smooth=0., action_smooth_leg=leg_weight,
                                               action_smooth_wheel=-.01)
    return path, manifest


def load(path, manifest):
    manifest["manifest_hash"] = _stable_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return load_policy_contract(path, path.with_name("actor.ts"), PROJECT / "sim2sim/mujoco/model_manifest.json")


@pytest.mark.parametrize("weight", (-.02, -.04))
def test_actual_loader_accepts_new_metadata_and_keeps_adapter_identical(tmp_path, weight):
    old, _, _ = load_policy_contract(BASELINE / "policy_manifest.json", BASELINE / "actor.ts",
                                    PROJECT / "sim2sim/mujoco/model_manifest.json")
    path, manifest = prepare(tmp_path, weight)
    new, _, _ = load(path, manifest)
    assert old.control_dt_s == new.control_dt_s and old.leg_kp == new.leg_kp
    assert old.leg_kd == new.leg_kd and old.wheel_kd == new.wheel_kd
    assert (old.effort_limits == new.effort_limits).all()


@pytest.mark.parametrize("mutation", ("ppo", "missing", "composite", "wheel", "leg", "other",
                                      "schedule", "acceleration", "reversal", "v2_with_split"))
def test_actual_loader_rejects_forged_split_identity(tmp_path, mutation):
    path, manifest = prepare(tmp_path)
    if mutation == "ppo":
        manifest["schema_version"] = "PpoActorExportV1"
    elif mutation == "missing":
        manifest["training_reward_weights"].pop("action_smooth_leg")
    elif mutation in ("composite", "wheel", "leg", "other"):
        key = {"composite": "action_smooth", "wheel": "action_smooth_wheel",
               "leg": "action_smooth_leg", "other": "dof_acc"}[mutation]
        manifest["training_reward_weights"][key] = -.123
    elif mutation == "schedule":
        manifest["command_sampling"]["practice_schedule"] = "disabled"
    elif mutation == "v2_with_split":
        manifest["schemas"]["reward"] = "RewardSchemaV2"
    else:
        manifest["training_leg_" + mutation] = {}
    with pytest.raises(ValueError):
        load(path, manifest)
