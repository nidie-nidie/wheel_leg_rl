"""Read-only 1ms sampler: every original mj_step runs exactly once."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(PROJECT / "sim2sim/mujoco"), str(PROJECT / "scripts")]
from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS
from evaluate_mujoco import _failure_reason

MAIN_SCENARIOS = ("nominal_stand", "forward", "reverse")
torch.set_num_threads(1)


def window_stats(acceleration):
    if len(acceleration) < 1000:
        return {"window_count": 0, "rms_mean": [None]*4, "rms_p90": [None]*4}
    count = len(acceleration) // 1000
    rms = np.sqrt(np.mean(acceleration[:count*1000].reshape(count, 1000, 4)**2, axis=1))
    return {"window_count": count, "rms_mean": rms.mean(0).tolist(),
            "rms_p90": np.percentile(rms, 90, axis=0).tolist()}


def capture(runtime, actor, command, observe):
    observation = runtime.reset(command)
    physics, control, histories, actions = [], [], [], []
    original = mujoco.mj_step
    tick = 0
    calls = 0

    def observed_step(model, data, *args, **kwargs):
        nonlocal calls
        assert model is runtime.model and data is runtime.data
        before = runtime.model_map.joint_velocity_control(data)[:4].copy()
        time_before = float(data.time)
        original(model, data, *args, **kwargs)
        calls += 1
        assert abs(float(data.time) - time_before - .001) < 1e-12
        q = runtime.model_map.joint_position_control(data)[:4].copy()
        velocity = runtime.model_map.joint_velocity_control(data)[:4].copy()
        physics.append([tick, float(data.time), *q, *velocity, *((velocity-before)/.001)])

    failure = None
    if observe:
        mujoco.mj_step = observed_step
    try:
        for tick in range(500):
            histories.append(observation.copy())
            with torch.inference_mode():
                action = actor(torch.from_numpy(observation).unsqueeze(0)).numpy().reshape(-1)
            if action.shape != (6,) or not np.isfinite(action).all():
                failure = "invalid_action"
                break
            actions.append(action.copy())
            targets = runtime.controller.prepare(action).leg_position_target_mujoco.copy()
            try:
                result = runtime.step(action, command)
            except FloatingPointError:
                failure = "non_finite_state"
                break
            failure = _failure_reason(runtime, result.metrics)
            if failure:
                break
            q = runtime.model_map.joint_position_control(runtime.data)[:4].copy()
            velocity = runtime.model_map.joint_velocity_control(runtime.data)[:4].copy()
            control.append([float(runtime.data.time), *q, *velocity, *targets,
                            result.metrics["vx_mps"], result.metrics["tilt_rad"],
                            result.metrics["effort_saturation_fraction"]])
            observation = result.observation
    finally:
        mujoco.mj_step = original
    return {"physics": np.asarray(physics).reshape(-1, 14),
            "control": np.asarray(control).reshape(-1, 16),
            "history": np.asarray(histories), "raw_action": np.asarray(actions),
            "failure_reason": failure, "mj_step_calls": calls}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    actor_path = args.export / "actor.ts"
    manifest_path = args.export / "policy_manifest.json"
    contract, manifest, model_manifest = load_policy_contract(
        manifest_path, actor_path, PROJECT / "sim2sim/mujoco/model_manifest.json")
    assert contract.physics_dt_s == .001 and contract.physics_steps_per_action == 20
    runtime = WheelLegMujocoRuntime(PROJECT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml",
                                   contract, model_manifest)
    actor = torch.jit.load(str(actor_path)).eval()
    summaries = []
    for name, values in FORMAL_SCENARIOS:
        if name not in MAIN_SCENARIOS:
            continue
        command = np.asarray(values, dtype=np.float64)
        observed = capture(runtime, actor, command, True)
        plain = capture(runtime, actor, command, False)
        for key in ("control", "history", "raw_action"):
            np.testing.assert_array_equal(observed[key], plain[key])
        assert observed["failure_reason"] == plain["failure_reason"]
        physics = observed["physics"]
        control = observed["control"]
        assert observed["mj_step_calls"] == 20 * len(control)
        if len(control):
            endpoints = physics[np.arange(len(control))*20+19]
            np.testing.assert_array_equal(endpoints[:, 2:6], control[:, 1:5])
            np.testing.assert_array_equal(endpoints[:, 6:10], control[:, 5:9])
        valid = physics[(physics[:, 0] >= 100) & (physics[:, 0] < len(control))]
        stats = window_stats(valid[:, 10:14])
        valid_raw = observed["raw_action"][100:len(control), :4]
        saturation = np.mean(np.abs(valid_raw) >= .999, axis=0).tolist() if len(valid_raw) else [None]*4
        npz = args.output / (name + ".npz")
        np.savez_compressed(npz, **{k: observed[k] for k in ("physics", "control", "history", "raw_action")})
        summaries.append({"scenario": name, "survived": len(control) == 500 and not observed["failure_reason"],
                          "failure_reason": observed["failure_reason"], "valid_control_ticks": len(control),
                          "physics_samples": len(physics), "original_step_calls": observed["mj_step_calls"],
                          "wrapped_vs_plain_exact": True, "metric_valid": stats["window_count"] > 0,
                          "leg_action_saturation_fraction": saturation,
                          "npz_sha256": sha256_file(npz), **stats})
    report = {"schema_version": "SplitSmoothPhysicsSamplerV1", "dt_s": .001,
              "warmup_control_ticks": 100, "window_physics_samples": 1000,
              "definition": "adjacent original mj_step actual qdot difference / .001",
              "physics_columns": ["control_tick", "time", "q0", "q1", "q2", "q3",
                                  "v0", "v1", "v2", "v3", "acc0", "acc1", "acc2", "acc3"],
              "control_columns": ["time", "q0", "q1", "q2", "q3", "v0", "v1", "v2", "v3",
                                  "target0", "target1", "target2", "target3", "vx", "tilt", "effort_sat"],
              "policy_sha256": sha256_file(actor_path), "manifest_sha256": sha256_file(manifest_path),
              "source_completed_iterations": manifest["source_completed_iterations"],
              "sampler_sha256": sha256_file(Path(__file__)), "scenarios": summaries}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
