"""Verify telemetry observes the same production trajectory, including resets."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from joint_telemetry import JointTelemetry
from wheelleg_mujoco.contract import load_policy_contract
from wheelleg_mujoco.observation import build_actor_observation, collect_kinematic_state
from wheelleg_mujoco.runner import WheelLegMujocoRuntime

parser = argparse.ArgumentParser()
parser.add_argument("--export", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
project = Path(__file__).resolve().parents[4]
contract, manifest, model_manifest = load_policy_contract(
    args.export / "policy_manifest.json", args.export / "actor.ts",
    project / "sim2sim/mujoco/model_manifest.json")
model_path = project / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
baseline = WheelLegMujocoRuntime(model_path, contract, model_manifest)
observed = WheelLegMujocoRuntime(model_path, contract, model_manifest)
policy = torch.jit.load(str(args.export / "actor.ts"), map_location="cpu").eval()
torch.set_num_threads(1)
fields = ("qpos", "qvel", "qacc", "ctrl", "qfrc_actuator", "qfrc_passive",
          "qfrc_constraint", "qfrc_bias", "qfrc_applied", "xfrc_applied", "actuator_force")
frames = 240
started = time.perf_counter()
args.output.mkdir(parents=True, exist_ok=True)
with JointTelemetry(observed, args.output, manifest) as logger:
    for frame in range(frames):
        segment = (frame % 80) // 16
        command = np.asarray([(0., .5, 0., -.5, 0.)[segment],
                              .2 if segment == 4 else 0., .20])
        if frame % 80 == 0:
            for runtime in (baseline, observed):
                runtime.reset(command)
                runtime._history[:, 6] = command[0] / contract.normalization["vx_max_abs"]
            logger.reset(command)
            logger.event("key", key="R", command=command.tolist(), paused=False)
        actions = []
        results = []
        for runtime in (baseline, observed):
            state = collect_kinematic_state(runtime.model, runtime.data, runtime.model_map, contract)
            current = build_actor_observation(state, command, runtime.previous_action, contract)
            current[6] = command[0] / contract.normalization["vx_max_abs"]
            runtime._history[-1] = current
            observation = runtime._history.reshape(-1).astype(np.float32, copy=True)
            with torch.inference_mode():
                tensor = torch.from_numpy(observation).unsqueeze(0)
                action = policy(tensor).numpy()[0]
                estimated_vx = float(policy.encoder(tensor)[0, 0]) / contract.normalization["root_linear_velocity_scale"]
            if runtime is observed:
                logger.begin_control(action, command, observation, estimated_vx)
            result = runtime.step(action, command)
            if runtime is observed:
                logger.end_control(result)
            runtime._history[-1, 6] = command[0] / contract.normalization["vx_max_abs"]
            actions.append(action.copy())
            results.append(result)
        np.testing.assert_array_equal(actions[0], actions[1])
        np.testing.assert_array_equal(results[0].applied_torque, results[1].applied_torque)
        for field in fields:
            np.testing.assert_array_equal(getattr(baseline.data, field), getattr(observed.data, field),
                                          err_msg=f"{field}, frame {frame}")
        np.testing.assert_array_equal(baseline._history, observed._history)
    logger.event("jitter_mark", key="M", command=command.tolist())
control_count = 0
with (args.output / "control.csv").open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        assert sum(key.startswith("history_") for key in row) == 125
        assert all(np.isfinite(float(value)) for value in row.values())
        control_count += 1
physics_count = 0
max_force_difference = 0.
with (args.output / "physics.csv").open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        assert all(np.isfinite(float(value)) for value in row.values())
        assert int(row["physics_substep"]) == physics_count % contract.physics_steps_per_action
        assert abs(float(row["time_after_s"]) - float(row["time_before_s"]) - contract.physics_dt_s) < 1e-12
        for name in observed.model_map.joint_names:
            max_force_difference = max(max_force_difference,
                abs(float(row[name + "_actuator_force_nm"]) - float(row[name + "_torque_command_nm"])))
        physics_count += 1
assert control_count == frames
assert physics_count == frames * contract.physics_steps_per_action
assert max_force_difference == 0.
events = [json.loads(line) for line in (args.output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
assert [event["episode_id"] for event in events if event["event"] == "reset"] == [0, 1, 2]
assert any(event["event"] == "jitter_mark" for event in events)
assert events[-1]["event"] == "closed"
assert observed.controller.compute_torque == logger.original_compute
report = {"passed": True, "control_frames": frames, "physics_rows": physics_count, "episodes": 3,
          "state_fields_compared": list(fields), "state_and_action_max_abs_error": 0.,
          "history_max_abs_error": 0., "actual_actuator_vs_command_max_abs_error_nm": max_force_difference,
          "conditions": ["stop", "forward", "stop", "reverse", "turn", "reset"],
          "elapsed_wall_s": time.perf_counter() - started,
          "source_completed_iterations": manifest["source_completed_iterations"]}
(args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report), flush=True)
