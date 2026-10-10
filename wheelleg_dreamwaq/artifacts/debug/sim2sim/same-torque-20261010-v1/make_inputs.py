"""Freeze canonical external torque inputs using the unchanged production adapter."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
sys.path.insert(0, str(PROJECT / "sim2sim/mujoco"))
from wheelleg_mujoco.contract import load_policy_contract
from wheelleg_mujoco.model_map import CANONICAL_JOINT_ORDER, CONTROLLED_JOINT_SIGNS
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def main():
    old = ROOT.parent / "ground-on-off-20261010-v1"
    action_path = old / "isaac-shared_first_action_hold-ground-off/actions.json"
    action_record = json.loads(action_path.read_text(encoding="utf-8"))
    actions = np.asarray(action_record["shared_first_action_clipped"], dtype=np.float64)
    assert actions.shape == (8, 6) and np.isfinite(actions).all()
    suite = PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2"
    export = suite / "exports/run-02"
    model = PROJECT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
    manifest = PROJECT / "sim2sim/mujoco/model_manifest.json"
    contract, _, model_manifest = load_policy_contract(export / "policy_manifest.json", export / "actor.ts", manifest)
    runtime = WheelLegMujocoRuntime(model, contract, model_manifest=model_manifest)
    state_path = old / "mujoco-shared_first_action_hold-ground-off/evidence.json"
    states = json.loads(state_path.read_text())
    nominal_names = model_manifest["joint_order"]
    ids = [nominal_names.index(name) for name in CANONICAL_JOINT_ORDER]
    torques, rows = [], []
    for index, action in enumerate(actions):
        command = np.array((0.0, 0.0, 0.20))
        runtime.reset(command)
        prior = next(s for s in states["samples"] if s["environment_index"] == index and s["physics_step"] == 0)
        q = runtime.model_map.joint_position_mujoco(runtime.data)
        qd = runtime.model_map.joint_velocity_mujoco(runtime.data)
        assert np.array_equal(q, np.asarray(prior["joint_position"])[ids])
        assert np.array_equal(qd, np.asarray(prior["joint_velocity"])[ids])
        targets = runtime.controller.prepare(action)
        native = runtime.controller.compute_torque(runtime.data, targets)
        assert not runtime.controller.last_velocity_limit_mask.any()
        canonical = (native * CONTROLLED_JOINT_SIGNS).astype(np.float32).astype(np.float64)
        assert np.all(np.abs(canonical) <= contract.effort_limits)
        torques.append(canonical.tolist())
        rows.append({"environment_index": index, "reset_joint_position_control": (q * CONTROLLED_JOINT_SIGNS).tolist(),
                     "reset_joint_velocity_control": (qd * CONTROLLED_JOINT_SIGNS).tolist(),
                     "leg_target_native": targets.leg_position_target_mujoco.tolist(),
                     "wheel_target_native": targets.wheel_velocity_target_mujoco.tolist(),
                     "initial_reference_torque_native_nm": native.tolist()})
    sources = [action_path, state_path, model, manifest, export / "actor.ts", export / "policy_manifest.json"]
    result = {"schema_version": "SameExternalTorqueInputsV1", "debug_only": True,
              "derivation": "Production MuJoCo PD at t=0 for the shared first action; clipped, canonical, float32-quantized once; held 20ms",
              "not_measured_isaac_drive_torque": True, "canonical_joint_order": list(CANONICAL_JOINT_ORDER),
              "hold_duration_s": 0.020, "effort_limits_nm": contract.effort_limits.tolist(),
              "shared_first_action_clipped": actions.tolist(), "shared_first_torque_control_nm": torques,
              "zero_torque_control_nm": np.zeros((8, 6)).tolist(), "reference_rows": rows,
              "input_sha256": {str(p.resolve()): digest(p) for p in sources}, "generator_sha256": digest(__file__)}
    (ROOT / "inputs.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("INPUTS_COMPLETE", np.asarray(torques).tolist())


if __name__ == "__main__":
    main()
