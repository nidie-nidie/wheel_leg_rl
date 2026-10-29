import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from debug.no_action_clip.contract import sha256, verify_formal_files, write_json
from debug.no_action_clip.heading_control import HeadingController
from debug.no_action_clip.mujoco_adapter import load_runtime
from debug.no_action_clip.play_mujoco_viewer import policy_observation_for_command
from scripts.evaluate_mujoco import _failure_reason
from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS


def _action(policy, observation):
    with torch.inference_mode():
        return policy(torch.from_numpy(observation).unsqueeze(0)).cpu().numpy().reshape(-1)


def _rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def run_scenario(export, output, policy, name, command_values, enabled, kp, kd, steps):
    command = np.asarray(command_values, dtype=np.float64)
    runtime = load_runtime(export)
    runtime.reset(command)
    heading = HeadingController(kp=kp, kd=kd, enabled=enabled)
    metrics = runtime.last_reset_metrics
    initial_yaw = metrics["yaw_rad"]
    heading.reset(initial_yaw)
    previous_velocity = runtime.model_map.joint_velocity_control(runtime.data)[:4]
    # 关闭组独立运行旧固定命令路径，逐步核对完整动作与动力学结果。
    reference = None
    if not enabled:
        reference = load_runtime(export)
        reference.reset(command)
    rows, valid = [], []
    trajectory_digest = hashlib.sha256()
    for tick in range(1, steps + 1):
        correction = heading.update(metrics["yaw_rad"], metrics["yaw_rate_rad_s"], command[1])
        effective = command.copy()
        effective[1] = correction.command_yaw_rate
        observation = policy_observation_for_command(runtime, effective)
        action = _action(policy, observation)
        targets = runtime.controller.prepare(action)
        if reference is not None:
            old_observation = reference._history.reshape(-1).astype(np.float32, copy=True)
            np.testing.assert_array_equal(observation, old_observation)
            old_action = _action(policy, old_observation)
            np.testing.assert_array_equal(action, old_action)
            old_targets = reference.controller.prepare(old_action)
            np.testing.assert_array_equal(targets.leg_position_target_mujoco,
                                          old_targets.leg_position_target_mujoco)
            np.testing.assert_array_equal(targets.wheel_velocity_target_mujoco,
                                          old_targets.wheel_velocity_target_mujoco)
            old_result = reference.step(old_action, command)
        result = runtime.step(action, effective)
        metrics = result.metrics
        if reference is not None:
            np.testing.assert_array_equal(result.applied_torque, old_result.applied_torque)
            np.testing.assert_array_equal(runtime._history, reference._history)
            np.testing.assert_array_equal(runtime.data.qpos, reference.data.qpos)
            np.testing.assert_array_equal(runtime.data.qvel, reference.data.qvel)
        velocity = runtime.model_map.joint_velocity_control(runtime.data)[:4]
        acceleration = (velocity - previous_velocity) / runtime.contract.control_dt_s
        previous_velocity = velocity.copy()
        failure = _failure_reason(runtime, metrics)
        row = {
            "tick": tick, "time_s": float(runtime.data.time), "survived": int(failure is None),
            "failure_reason": failure or "", "user_vx_mps": float(command[0]),
            "user_yaw_rate_rad_s": float(command[1]), "user_height_m": float(command[2]),
            "policy_yaw_rate_rad_s": float(effective[1]), "heading_mode": correction.mode,
            "heading_target_rad": correction.target_rad,
            "heading_error_before_step_rad": correction.error_rad,
            "heading_correction_rad_s": correction.correction_yaw_rate,
            "heading_saturated": int(correction.saturated),
            **metrics,
            **{f"raw_action_{i}": float(value) for i, value in enumerate(action)},
            **{f"leg_target_{i}_rad": float(value)
               for i, value in enumerate(targets.leg_position_target_mujoco)},
            **{f"wheel_target_{i}_mujoco_rad_s": float(value)
               for i, value in enumerate(targets.wheel_velocity_target_mujoco)},
            **{f"leg_acceleration_{i}_rad_s2": float(value)
               for i, value in enumerate(acceleration)},
        }
        rows.append(row)
        for values in (action, result.applied_torque, runtime.data.qpos, runtime.data.qvel):
            trajectory_digest.update(values.tobytes())
        if failure:
            break
        valid.append(row)
    mode = "on" if enabled else "off"
    csv_path = output / f"{name}-{mode}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "name": name, "heading_enabled": enabled, "command": command.tolist(),
        "expected_steps": steps, "valid_steps": len(valid), "executed_steps": len(rows),
        "completed": len(valid) == steps, "failure_reason": rows[-1]["failure_reason"] or None,
        "initial_yaw_rad": initial_yaw, "csv": str(csv_path), "csv_sha256": sha256(csv_path),
        "trajectory_sha256": trajectory_digest.hexdigest(),
        "disabled_matches_old_path_exactly": True if reference is not None else None,
    }
    if valid:
        yaw = np.unwrap(np.asarray([initial_yaw] + [row["yaw_rad"] for row in valid]))[1:] - initial_yaw
        summary.update({
            "vx_mae_mps": float(np.mean([abs(row["vx_mps"] - command[0]) for row in valid])),
            "yaw_drift_rms_rad": _rms(yaw), "yaw_drift_final_rad": float(yaw[-1]),
            "yaw_rate_rms_rad_s": _rms([row["yaw_rate_rad_s"] for row in valid]),
            "yaw_rate_mae_rad_s": float(np.mean([abs(row["yaw_rate_rad_s"] - command[1]) for row in valid])),
            "heading_correction_rms_rad_s": _rms([row["heading_correction_rad_s"] for row in valid]),
            "leg_acceleration_control_interval_rms_rad_s2": [
                _rms([row[f"leg_acceleration_{i}_rad_s2"] for row in valid]) for i in range(4)],
        })
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="真实 MuJoCo 航向保持开关对照，不执行训练")
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--kp", type=float, default=0.5)
    parser.add_argument("--kd", type=float, default=0.0)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("控制步数必须为正数")
    HeadingController(kp=args.kp, kd=args.kd)
    formal_before = verify_formal_files()
    export = args.export.resolve()
    metadata = json.loads((export / "experiment_export.json").read_text(encoding="utf-8"))
    actor_hash = sha256(export / "actor.ts")
    policy = torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scenarios = []
    for name, command in FORMAL_SCENARIOS:
        pair = [run_scenario(export, output, policy, name, command, enabled, args.kp, args.kd, args.steps)
                for enabled in (False, True)]
        if command[1] != 0.0:
            if pair[0]["trajectory_sha256"] != pair[1]["trajectory_sha256"]:
                raise AssertionError(f"用户转向路径被航向保持改变：{name}")
        scenarios.extend(pair)
    if sha256(export / "actor.ts") != actor_hash:
        raise RuntimeError("评估过程中模型文件发生变化")
    report = {
        "schema_version": "DebugHeadingHoldVerificationV1", "training_started": False,
        "export": str(export), "completed_iterations": metadata["completed_iterations"],
        "actor_sha256": actor_hash, "kp": args.kp, "kd": args.kd,
        "max_yaw_rate_rad_s": 1.0, "formal_files_before": formal_before,
        "formal_files_after": verify_formal_files(), "scenarios": scenarios,
        "source_sha256": {name: sha256(Path(__file__).parent / name) for name in (
            "heading_control.py", "play_mujoco_viewer.py", "verify_heading_mujoco.py")},
    }
    write_json(output / "summary.json", report)
    print("HEADING_HOLD_MUJOCO_VERIFIED", str(output / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
