from __future__ import annotations

import json
import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np

from wheelleg_mujoco.contract import AdapterContract
from wheelleg_mujoco.control import MixedActionController
from wheelleg_mujoco.model_map import build_model_map

from debug.no_action_clip.contract import BASELINE_EXPORT, PROJECT_ROOT, write_json
from debug.no_action_clip.mujoco_adapter import RawActionRuntime, load_runtime


def test_real_mujoco_raw_controller_and_history():
    baseline = json.loads((BASELINE_EXPORT / "policy_manifest.json").read_text(encoding="utf-8"))
    contract = AdapterContract.from_policy_manifest(baseline)
    model_manifest = json.loads((PROJECT_ROOT / "sim2sim/mujoco/model_manifest.json").read_text(encoding="utf-8"))
    model_path = PROJECT_ROOT / "sim2sim/mujoco/models" / model_manifest["model_xml"]["path"]
    runtime = RawActionRuntime(model_path, contract, model_manifest)
    command = np.array([0.0, 0.0, 0.20])
    first = runtime.reset(command).reshape(5, 25)
    np.testing.assert_array_equal(first, np.repeat(first[:1], 5, axis=0))
    action = np.array([2.5, -2.5, 3.0, -3.0, 2.0, -2.0])
    targets = runtime.controller.prepare(action)
    np.testing.assert_array_equal(targets.clipped_action, action)
    np.testing.assert_array_equal(targets.wheel_velocity_target_mujoco, np.array([50.0, 50.0]))
    assert np.abs(targets.leg_position_target_mujoco).max() <= 1.0
    result = runtime.step(action, command)
    np.testing.assert_array_equal(result.observation.reshape(5, 25)[-1, 19:25], action.astype(np.float32))
    np.testing.assert_array_equal(result.observation.reshape(5, 25)[:-1], first[1:])
    assert np.abs(result.applied_torque[:4]).max() <= 18.0
    assert np.abs(result.applied_torque[4:]).max() <= 9.0
    assert "action_saturation_fraction" not in result.metrics
    reset = runtime.reset(command).reshape(5, 25)
    np.testing.assert_array_equal(reset[:, 19:25], np.zeros((5, 6)))
    # 实验与正式控制器在相同目标下给出相同力矩，未新增动力学处理。
    formal = MixedActionController(build_model_map(runtime.model), contract)
    np.testing.assert_array_equal(runtime.controller.compute_torque(runtime.data, targets),
                                  formal.compute_torque(runtime.data, targets))
    boundary_step = runtime.step(np.ones(6), command)
    assert boundary_step.metrics["action_outside_former_unit_interval_fraction"] == 0.0


def test_real_export_identity_guards(export_dir):
    load_runtime(export_dir)
    target = export_dir.parent / "mujoco-loader-integrity/export"
    shutil.copytree(export_dir, target, dirs_exist_ok=True)
    original = json.loads((target / "experiment_export.json").read_text(encoding="utf-8"))
    checked = []
    for field in ("source_checkpoint_sha256", "training_manifest_sha256", "actor_sha256", "geometry_guard"):
        damaged = dict(original)
        damaged[field] = {} if field == "geometry_guard" else "0" * 64
        write_json(target / "experiment_export.json", damaged)
        try:
            load_runtime(target)
        except ValueError:
            checked.append(field)
        else:
            raise AssertionError(f"MuJoCo 接受了损坏身份：{field}")
    write_json(target / "experiment_export.json", original)
    load_runtime(target)
    write_json(target.parent / "verification.json", {"corrupted_identity_fields_rejected": checked})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path)
    arguments = parser.parse_args()
    test_real_mujoco_raw_controller_and_history()
    if arguments.export is not None:
        test_real_export_identity_guards(arguments.export)
    print("MUJOCO_RAW_ACTION_AND_HISTORY_VERIFIED")
