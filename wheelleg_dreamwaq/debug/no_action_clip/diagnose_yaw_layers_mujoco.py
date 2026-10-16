from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from wheelleg_mujoco.evaluation import FORMAL_SCENARIOS
from wheelleg_mujoco.observation import collect_kinematic_state

from debug.no_action_clip.mujoco_adapter import load_runtime


EXPECTED_TICKS = 500
MODES = ("policy", "wheel_pair_mean", "leg_side_pair_mean", "all_side_pair_mean")


def _intervene(action: np.ndarray, mode: str) -> np.ndarray:
    value = np.asarray(action, dtype=np.float64).copy()
    if mode in {"wheel_pair_mean", "all_side_pair_mean"}:
        mean = float(np.mean(value[4:6]))
        value[4:6] = mean
    if mode in {"leg_side_pair_mean", "all_side_pair_mean"}:
        # Canonical order is left-front, left-rear, right-front, right-rear.
        for left, right in ((0, 2), (1, 3)):
            mean = float(np.mean(value[[left, right]]))
            value[left] = mean
            value[right] = mean
    return value


def _contact_normal_forces(runtime) -> tuple[float, float]:
    floor_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    left_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_GEOM, "left_wheel_proxy")
    right_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_GEOM, "right_wheel_proxy")
    left_force = 0.0
    right_force = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for contact_id in range(runtime.data.ncon):
        contact = runtime.data.contact[contact_id]
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor_id not in pair or (left_id not in pair and right_id not in pair):
            continue
        mujoco.mj_contactForce(runtime.model, runtime.data, contact_id, wrench)
        normal = abs(float(wrench[0]))
        if left_id in pair:
            left_force += normal
        if right_id in pair:
            right_force += normal
    return left_force, right_force


def _policy_action(policy, observation: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(observation.astype(np.float32, copy=False)).unsqueeze(0)
    with torch.inference_mode():
        value = policy(tensor).detach().cpu().numpy().reshape(-1)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise FloatingPointError("Policy returned an invalid six-dimensional action")
    return value.astype(np.float64)


def _mean_array(values: list[np.ndarray]) -> list[float]:
    return np.mean(np.asarray(values, dtype=np.float64), axis=0).tolist()


def _rms_array(values: list[np.ndarray]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.mean(np.square(array), axis=0)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate policy, actuator, and contact contributions to MuJoCo yaw drift.")
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    export_dir = args.export.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(export_dir)
    policy = torch.jit.load(str(export_dir / "actor.ts"), map_location="cpu").eval()
    torch.set_num_threads(1)

    reports = []
    for scenario_name, command_values in FORMAL_SCENARIOS:
        command = np.asarray(command_values, dtype=np.float64)
        for mode in MODES:
            observation = runtime.reset(command)
            rows = []
            raw_actions = []
            applied_actions = []
            targets = []
            velocities = []
            torques = []
            torques_control = []
            yaw_rates = []
            yaw_angles = []
            vx_values = []
            contacts = []
            for step in range(EXPECTED_TICKS):
                policy_action = _policy_action(policy, observation)
                applied_action = _intervene(policy_action, mode)
                prepared = runtime.controller.prepare(applied_action)
                result = runtime.step(applied_action, command)
                state = collect_kinematic_state(runtime.model, runtime.data, runtime.model_map, runtime.contract)
                left_contact, right_contact = _contact_normal_forces(runtime)
                raw_actions.append(policy_action)
                applied_actions.append(applied_action)
                targets.append(np.concatenate((prepared.leg_position_target_mujoco, prepared.wheel_velocity_target_mujoco)))
                velocities.append(state.joint_velocity_control.copy())
                torques.append(result.applied_torque.copy())
                torque_control = result.applied_torque.copy()
                torque_control[4:6] *= runtime.contract.wheel_signs
                torques_control.append(torque_control)
                yaw_rates.append(float(state.angular_velocity_control[2]))
                yaw_angles.append(float(result.metrics["yaw_rad"]))
                vx_values.append(float(state.com_linear_velocity_control[0]))
                contacts.append(np.asarray((left_contact, right_contact), dtype=np.float64))
                rows.append(
                    {
                        "scenario": scenario_name,
                        "mode": mode,
                        "step": step,
                        "command_vx_mps": float(command[0]),
                        "command_yaw_rate_rad_s": float(command[1]),
                        "yaw_rate_rad_s": float(state.angular_velocity_control[2]),
                        "yaw_rad": float(result.metrics["yaw_rad"]),
                        "left_contact_normal_n": left_contact,
                        "right_contact_normal_n": right_contact,
                        **{f"policy_action_{i}": float(policy_action[i]) for i in range(6)},
                        **{f"applied_action_{i}": float(applied_action[i]) for i in range(6)},
                        **{f"joint_velocity_{i}": float(state.joint_velocity_control[i]) for i in range(6)},
                        **{f"torque_{i}": float(result.applied_torque[i]) for i in range(6)},
                    }
                )
                # Keep the policy's own action in the history. This makes the
                # wheel/leg intervention an actuation-only debug ablation.
                if mode != "policy":
                    runtime.previous_action = policy_action.copy()
                    if runtime._history is not None:
                        runtime._history[-1, 19:25] = policy_action.astype(np.float32)
                observation = runtime._history.reshape(-1).astype(np.float32, copy=True)

            raw_array = np.asarray(raw_actions, dtype=np.float64)
            applied_array = np.asarray(applied_actions, dtype=np.float64)
            target_array = np.asarray(targets, dtype=np.float64)
            velocity_array = np.asarray(velocities, dtype=np.float64)
            torque_array = np.asarray(torques, dtype=np.float64)
            torque_control_array = np.asarray(torques_control, dtype=np.float64)
            yaw_array = np.asarray(yaw_rates, dtype=np.float64)
            vx_array = np.asarray(vx_values, dtype=np.float64)
            contact_array = np.asarray(contacts, dtype=np.float64)
            csv_path = output / f"{scenario_name}-{mode}.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            reports.append(
                {
                    "scenario": scenario_name,
                    "command": list(command_values),
                    "mode": mode,
                    "yaw_rate_mean_rad_s": float(np.mean(yaw_array)),
                    "yaw_rate_mae_to_command_rad_s": float(np.mean(np.abs(yaw_array - command[1]))),
                    "yaw_rate_rms_rad_s": float(np.sqrt(np.mean(np.square(yaw_array)))),
                    "yaw_angle_rms_rad": float(np.sqrt(np.mean(np.square(np.unwrap(np.asarray(yaw_angles))))),),
                    "vx_mean_mps": float(np.mean(vx_array)),
                    "vx_mae_to_command_mps": float(np.mean(np.abs(vx_array - command[0]))),
                    "policy_action_mean": _mean_array(raw_actions),
                    "applied_action_mean": _mean_array(applied_actions),
                    "policy_applied_action_delta_rms": _rms_array([raw - applied for raw, applied in zip(raw_actions, applied_actions)]),
                    "target_mean": _mean_array(targets),
                    "joint_velocity_mean": _mean_array(velocities),
                    "joint_velocity_rms": _rms_array(velocities),
                    "torque_mean": _mean_array(torques),
                    "torque_rms": _rms_array(torques),
                    "torque_control_mean": _mean_array(torques_control),
                    "torque_control_rms": _rms_array(torques_control),
                    "wheel_contact_normal_mean_n": _mean_array(contacts),
                    "wheel_action_difference_mean_abs": float(np.mean(np.abs(applied_array[:, 4] - applied_array[:, 5]))),
                    "wheel_velocity_difference_mean_abs_rad_s": float(np.mean(np.abs(velocity_array[:, 4] - velocity_array[:, 5]))),
                    "wheel_target_difference_mean_abs_rad_s": float(np.mean(np.abs(target_array[:, 4] * runtime.contract.wheel_signs[0] - target_array[:, 5] * runtime.contract.wheel_signs[1]))),
                    "wheel_torque_difference_mean_abs_nm": float(np.mean(np.abs(torque_control_array[:, 4] - torque_control_array[:, 5]))),
                    "wheel_contact_difference_mean_abs_n": float(np.mean(np.abs(contact_array[:, 0] - contact_array[:, 1]))),
                    "leg_side_difference_mean_abs": float(np.mean(np.abs(applied_array[:, [0, 1]].mean(axis=1) - applied_array[:, [2, 3]].mean(axis=1)))),
                    "csv": str(csv_path),
                }
            )

    report = {
        "schema_version": "DebugNoActionClipYawLayerDiagnosticV1",
        "export": str(export_dir),
        "expected_ticks_per_scenario": EXPECTED_TICKS,
        "modes": list(MODES),
        "mode_semantics": {
            "policy": "policy output is applied unchanged",
            "wheel_pair_mean": "raw wheel actions 4 and 5 are averaged for actuation; policy history retains original output",
            "leg_side_pair_mean": "raw leg pairs (0,2) and (1,3) are averaged for actuation; policy history retains original output",
            "all_side_pair_mean": "both wheel and corresponding left/right leg actions are averaged for actuation; policy history retains original output",
        },
        "canonical_action_order": ["jIJ", "jIO", "jAB", "jAG", "jwheel_left", "jwheel_right"],
        "reports": reports,
    }
    report_path = output / "yaw-layer-diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
