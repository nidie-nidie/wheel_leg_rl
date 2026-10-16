from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from debug.sim2sim.collect_dreamwaq_mujoco_trace import NoActionClipProbeController
from wheelleg_mujoco.contract import load_policy_contract, sha256_file
from wheelleg_mujoco.observation import build_actor_observation, collect_kinematic_state
from wheelleg_mujoco.runner import WheelLegMujocoRuntime


PROJECT = Path(__file__).resolve().parents[4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    session = args.session.resolve()
    status = json.loads((session / "status.json").read_text(encoding="utf-8"))
    if status["mode"] != "FORMAL WHEEL CONTACT" or not status["bypass_action_clip"]:
        raise ValueError("Replay requires the formal model and raw-target viewer mode")
    export = Path(status["export"])
    model = PROJECT / "sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml"
    contract, manifest, model_manifest = load_policy_contract(
        export / "policy_manifest.json", export / "actor.ts",
        PROJECT / "sim2sim/mujoco/model_manifest.json",
    )
    if sha256_file(model) != manifest["mujoco_model"]["model_xml_sha256"]:
        raise ValueError("Formal model hash differs from the viewer policy")
    with (session / "trajectory.csv").open(encoding="utf-8", newline="") as stream:
        recorded = list(csv.DictReader(stream))
    times = np.asarray([float(row["time_s"]) for row in recorded])
    np.testing.assert_allclose(
        times, np.arange(1, len(times) + 1) * contract.control_dt_s,
        rtol=0.0, atol=1.0e-8,
    )
    torch.set_num_threads(1)
    policy = torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()
    runtime = WheelLegMujocoRuntime(model, contract, model_manifest)
    runtime.controller = NoActionClipProbeController(runtime.controller)
    # 窗口启动时的指令固定为零速、零偏航和 0.20 m 高度。
    initial_command = np.asarray((0.0, 0.0, 0.20), dtype=np.float64)
    runtime.reset(initial_command)
    runtime._history[:, 6] = 0.0
    checked = (
        "base_height_m", "vx_mps", "yaw_rate_rad_s", "tilt_rad",
        "phi0_left_rad", "phi0_right_rad", "phi0_delta_abs_rad",
        "l0_left_m", "l0_right_m", "max_loop_closure_error_m",
    )
    errors = {name: 0.0 for name in checked}
    errors["estimated_vx_before_action_mps"] = 0.0
    output = session / "reconstructed_control.csv"
    raw_actions = []
    targets_all = []
    summary_rows = []
    writer = None
    with output.open("x", encoding="utf-8", newline="") as stream:
        for index, original in enumerate(recorded):
            command = np.asarray([
                float(original["command_vx_mps"]), float(original["command_yaw_rad_s"]),
                float(original["command_height_m"]),
            ])
            state = collect_kinematic_state(runtime.model, runtime.data, runtime.model_map, contract)
            current = build_actor_observation(state, command, runtime.previous_action, contract)
            current[6] = command[0] / contract.normalization["vx_max_abs"]
            runtime._history[-1] = current
            observation = runtime._history.reshape(-1).astype(np.float32, copy=True)
            with torch.inference_mode():
                tensor = torch.from_numpy(observation).unsqueeze(0)
                action = policy(tensor).numpy()[0]
                estimated_vx = float(policy.encoder(tensor)[0, 0]) / contract.normalization["root_linear_velocity_scale"]
            targets = runtime.controller.prepare(action)
            result = runtime.step(action, command)
            runtime._history[-1, 6] = command[0] / contract.normalization["vx_max_abs"]
            for name in checked:
                error = abs(float(result.metrics[name]) - float(original[name]))
                errors[name] = max(errors[name], error)
                if error > 1.0e-8:
                    raise AssertionError(f"Replay differs at frame {index}, {name}: {error}")
            error = abs(estimated_vx - float(original["estimated_vx_before_action_mps"]))
            errors["estimated_vx_before_action_mps"] = max(errors["estimated_vx_before_action_mps"], error)
            if error > 1.0e-7:
                raise AssertionError(f"Replay CENet estimate differs at frame {index}: {error}")
            row = dict(original)
            row["replay_verified"] = True
            for name, value in zip(runtime.model_map.joint_names, action):
                row[name + "_action_raw"] = float(value)
            for name, value in zip(runtime.model_map.joint_names[:4], targets.leg_position_target_mujoco):
                row[name + "_position_target_rad"] = float(value)
            for name, value in zip(runtime.model_map.joint_names[4:], targets.wheel_velocity_target_mujoco):
                row[name + "_velocity_target_mujoco_rad_s"] = float(value)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            raw_actions.append(action)
            targets_all.append(np.r_[targets.leg_position_target_mujoco, targets.wheel_velocity_target_mujoco])
            summary_rows.append((command, result.metrics))
    raw = np.asarray(raw_actions)
    targets = np.asarray(targets_all)
    groups = {}
    for height in (0.16, 0.20, 0.23, 0.24):
        indices = [i for i, (command, _) in enumerate(summary_rows)
                   if abs(command[0]) < 1.0e-8 and abs(command[1]) < 1.0e-8
                   and abs(command[2] - height) < 1.0e-8]
        if not indices:
            continue
        metrics = [summary_rows[i][1] for i in indices]
        groups[str(height)] = {
            "frames": len(indices),
            "phi_difference_mean_rad": float(np.mean([m["phi0_delta_abs_rad"] for m in metrics])),
            "leg_length_mean_m": [float(np.mean([m[k] for m in metrics])) for k in ("l0_left_m", "l0_right_m")],
            "raw_action_mean": raw[indices].mean(axis=0).tolist(),
            "raw_action_min": raw[indices].min(axis=0).tolist(),
            "raw_action_max": raw[indices].max(axis=0).tolist(),
            "leg_target_mean_rad": targets[indices, :4].mean(axis=0).tolist(),
            "leg_target_at_bounds_fraction": np.mean(
                np.isclose(targets[indices, :4], contract.leg_target_lower, atol=1.0e-10)
                | np.isclose(targets[indices, :4], contract.leg_target_upper, atol=1.0e-10), axis=0,
            ).tolist(),
        }
    summary = {
        "replay_verified": True,
        "frames": len(recorded),
        "checked_metric_max_abs_errors": errors,
        "actor_sha256": manifest["actor_sha256"],
        "joint_order": list(runtime.model_map.joint_names),
        "trajectory_sha256": sha256_file(session / "trajectory.csv"),
        "reconstructed_control_sha256": sha256_file(output),
        "zero_speed_zero_yaw_groups_by_target_height": groups,
    }
    (session / "replay_verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
