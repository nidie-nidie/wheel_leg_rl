from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
import torch

from training.recovery.contact_model import load_contact_model
from wheelleg_mujoco.contract import load_policy_contract
from wheelleg_mujoco.metrics import collect_metrics
from wheelleg_mujoco.model_map import CONTROLLED_JOINT_SIGNS
from wheelleg_mujoco.observation import collect_kinematic_state
from wheelleg_mujoco.runner import WheelLegMujocoRuntime
from debug.sim2sim.trace_schema import sha256_file


PROJECT = Path(__file__).resolve().parents[2]
SIM = PROJECT / "sim2sim/mujoco"
COMMAND = np.array([0.0, 0.0, 0.20])
CONTACT_MANIFEST = PROJECT / "recovery/artifacts/mujoco_verification/model/contact_manifest.json"
EXPECTED_CHECKPOINT = "6AC197FBEBBD981ADEB4FD18B55D0E59B3BC81F438B48E5A7252B7E931E50D58"
EXPECTED_ACTOR = "251DBFDE72D78745E4BBA6BE86575F9F2DFBD2242AA256891001DF80A2AA3A04"


def locations():
    logs = PROJECT / "logs/rsl_rl/wheelleg_flat_dreamwaq"
    checkpoint = logs / "2026-10-17_14-57-16_rtx5070_fudan-v1_baseline-yaw1-headingoff-env6144-fresh1000-20261017/model_400.pt"
    export = logs / "2026-10-17_17-56-29_rtx5070_fudan-v1_baseline-yaw1-headingoff-env4096-fresh701-20261017/degeneration_audit/checkpoints_6144/model_400/export"
    return checkpoint, export


def load_actor():
    checkpoint, export = locations()
    contract, metadata, model = load_policy_contract(export / "policy_manifest.json", export / "actor.ts", SIM / "model_manifest.json")
    if Path(metadata["source_checkpoint"]).resolve() != checkpoint.resolve() or metadata["source_completed_iterations"] != 401:
        raise ValueError("导出文件未绑定401实际更新模型")
    identity = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                "actor_sha256": sha256_file(export / "actor.ts")}
    if metadata["source_checkpoint_sha256"] != identity["checkpoint_sha256"]:
        raise ValueError("导出文件与checkpoint哈希不一致")
    if contract.control_dt_s != 0.02 or not np.isinf(contract.action_clip):
        raise ValueError("401控制契约不一致")
    actor = torch.jit.load(str(export / "actor.ts"), map_location="cpu").eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor, contract, model, identity


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def scenarios():
    records = [{"name": "nominal_yaw0", "yaw_rad": 0.0},
               {"name": "nominal_yaw90", "yaw_rad": float(np.pi / 2)}]
    for magnitude in (0.5, 1.0):
        for sign in (-1, 1):
            records.append({"name": f"forward_{magnitude:g}N_{sign:+d}", "force_control_n": [sign * magnitude, 0.0, 0.0]})
    for sign in (-1, 1):
        records.append({"name": f"lateral_0.25N_{sign:+d}", "force_control_n": [0.0, sign * 0.25, 0.0]})
        records.append({"name": f"pitch_0.02Nm_{sign:+d}", "torque_control_nm": [0.0, sign * 0.02, 0.0]})
    return records


class CommandRecorder:
    def __init__(self, controller):
        self.base = controller
        self.effort_limits = controller.effort_limits
        self.torques = []

    def prepare(self, action):
        return self.base.prepare(action)

    def compute_torque(self, data, targets):
        result = self.base.compute_torque(data, targets)
        self.torques.append(result.copy() * CONTROLLED_JOINT_SIGNS)
        return result

    def apply_torque(self, data, torque):
        self.base.apply_torque(data, torque)


def contact_state(model, data):
    floor = model.geom("floor").id
    normal = np.zeros(model.nbody)
    vertical = np.zeros(model.nbody)
    penetration = 0.0
    robot_contacts = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        penetration = max(penetration, -float(contact.dist))
        if floor not in (contact.geom1, contact.geom2):
            robot_contacts += 1
            continue
        robot_is_second = contact.geom1 == floor
        geom = contact.geom2 if robot_is_second else contact.geom1
        body = int(model.geom_bodyid[geom])
        world = np.asarray(contact.frame).reshape(3, 3).T @ force[:3]
        normal[body] += max(0.0, float(force[0]))
        vertical[body] += float(world[2]) * (1.0 if robot_is_second else -1.0)
    return normal, vertical, penetration, robot_contacts


def failure_reason(data, metrics, penetration, robot_contacts, warning_delta):
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        return "non_finite_physics"
    if warning_delta:
        return "mujoco_warning"
    if metrics["max_loop_closure_error_m"] > 0.005:
        return "loop_closure"
    if penetration > 0.005:
        return "penetration"
    if robot_contacts:
        return "robot_contact"
    if metrics["tilt_rad"] > 0.8 or not 0.10 <= metrics["base_height_m"] <= 0.40:
        return "lost_standing"
    if metrics["root_linear_speed_mps"] > 20 or metrics["root_angular_speed_rad_s"] > 35:
        return "root_velocity"
    if metrics["max_hinge_speed_rad_s"] > 1000:
        return "joint_velocity"
    return None


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "min": values.min(axis=0).tolist(), "p005": np.quantile(values, 0.005, axis=0).tolist(),
            "p01": np.quantile(values, 0.01, axis=0).tolist(),
            "p50": np.quantile(values, 0.5, axis=0).tolist(),
            "p99": np.quantile(values, 0.99, axis=0).tolist(),
            "p995": np.quantile(values, 0.995, axis=0).tolist(), "max": values.max(axis=0).tolist()}


def collect(output, duration_s):
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(1)
    actor, contract, formal_manifest, identity = load_actor()
    if identity["checkpoint_sha256"] != EXPECTED_CHECKPOINT or identity["actor_sha256"] != EXPECTED_ACTOR:
        raise ValueError("401 模型身份不符合本次采集约定")
    _, contact_manifest = load_contact_model(CONTACT_MANIFEST)
    formal_xml = SIM / "models" / formal_manifest["model_xml"]["path"]
    runtimes = {
        "formal401": WheelLegMujocoRuntime(formal_xml, contract, formal_manifest),
        "recovery_contact": WheelLegMujocoRuntime(contact_manifest["model_xml"], contract),
    }
    checkpoint, export = locations()
    manifest = {
        "schema": "StandingReference401V1", "started_utc": started_utc,
        "identity": identity, "source_completed_updates": 401,
        "policy_manifest": str(export / "policy_manifest.json"),
        "policy_manifest_sha256": sha256_file(export / "policy_manifest.json"),
        "collector": str(Path(__file__).resolve()), "collector_sha256": sha256_file(__file__),
        "formal_xml": str(formal_xml), "formal_xml_sha256": sha256_file(formal_xml),
        "recovery_contact_manifest": str(CONTACT_MANIFEST),
        "recovery_contact_manifest_sha256": sha256_file(CONTACT_MANIFEST),
        "recovery_xml": contact_manifest["model_xml"], "recovery_xml_sha256": contact_manifest["model_xml_sha256"],
        "device": "cpu", "torch_version": torch.__version__, "mujoco_version": mujoco.__version__,
        "control_dt_s": contract.control_dt_s, "physics_dt_s": contract.physics_dt_s,
        "command": COMMAND.tolist(), "duration_per_episode_s": duration_s,
        "action_mode": "deterministic_mean", "observation_noise": False,
        "domain_randomization": False, "policy_updates": 0, "optimizer_updates": 0,
        "heading_enabled": False, "scenarios": scenarios(),
        "perturbation_interval_s": [5.0, 5.1], "random_seed": 42001,
        "force_application": "base_COM_world_force_and_torque_no_upward_force",
        "state_timing": "post_integration_copy_mj_forward_without_mutating_policy_runtime",
        "solver_torque_timing": "last_original_physics_substep_motor_force",
        "contacts_timing": "recomputed_on_post_integration_copy_at_50Hz_not_substep_peaks",
        "joint_order": list(runtimes["formal401"].model_map.joint_names),
        "control_signs": CONTROLLED_JOINT_SIGNS.tolist(),
        "height_semantics": "base_body_COM_above_floor", "target_contract": {
            "q_nominal": contract.q_nominal.tolist(), "leg_action_scale": contract.leg_action_scale,
            "leg_target_lower": np.asarray(contract.leg_target_lower).tolist(),
            "leg_target_upper": np.asarray(contract.leg_target_upper).tolist(),
            "wheel_action_scale": contract.wheel_action_scale, "leg_kp": contract.leg_kp,
            "leg_kd": contract.leg_kd, "wheel_kd": contract.wheel_kd,
            "effort_limits": contract.effort_limits.tolist(), "history_length": contract.history_length,
            "history_layout": contract.history_layout, "policy_input_dimension": contract.policy_input_dimension},
        "limitations": ["nominal dynamics only; episodes share one policy and deterministic reset",
                        "formal401 model enables only wheel-floor contacts",
                        "recovery_contact uses the existing full floor-contact model with internal contacts disabled",
                        "standing data does not identify hardware motion limits or prove getup success"],
    }
    write_json(output / "manifest.json", manifest)
    episode_records = []
    model_summaries = {}
    for model_name, runtime in runtimes.items():
        runtime.heading.enabled = False
        original = runtime.controller
        recorder = CommandRecorder(original)
        runtime.controller = recorder
        model = runtime.model
        model_map = runtime.model_map
        snapshot = mujoco.MjData(model)
        wheel_ids = model.jnt_bodyid[model_map.joint_ids[4:]]
        other_ids = np.array([i for i in range(1, model.nbody) if i not in wheel_ids])
        body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "world" for i in range(model.nbody)]
        stable_parts = []
        for scenario_index, scenario in enumerate(scenarios()):
            runtime.reset(COMMAND)
            yaw = scenario.get("yaw_rad", 0.0)
            if yaw:
                address = int(model.jnt_qposadr[model_map.free_joint_id])
                runtime.data.qpos[address + 3:address + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
                mujoco.mj_forward(model, runtime.data)
                state = collect_kinematic_state(model, runtime.data, model_map, contract)
                from wheelleg_mujoco.observation import build_actor_observation
                runtime._reset_policy_observation(build_actor_observation(state, COMMAND, runtime.previous_action, contract))
                runtime._prepared_time = None
            rows = []
            failure = None
            warning_before = sum(int(item.number) for item in runtime.data.warning)
            for tick in range(round(duration_s / contract.control_dt_s)):
                time_before = float(runtime.data.time)
                active = 5.0 - 1e-8 <= time_before < 5.1 - 1e-8
                rotation = np.asarray(runtime.data.xmat[model_map.base_body_id]).reshape(3, 3)
                external = np.r_[rotation @ contract.r_control_from_mujoco.T @ np.asarray(scenario.get("force_control_n", [0, 0, 0])),
                                 rotation @ contract.r_control_from_mujoco.T @ np.asarray(scenario.get("torque_control_nm", [0, 0, 0]))] if active else np.zeros(6)
                runtime.data.xfrc_applied[:] = 0.0
                runtime.data.xfrc_applied[model_map.base_body_id] = external
                observation = runtime.prepare_policy_observation(COMMAND)
                with torch.inference_mode():
                    action = actor(torch.from_numpy(observation).view(1, -1)).numpy()[0].copy()
                targets = original.prepare(action)
                recorder.torques.clear()
                step = runtime.step(action, COMMAND)
                actual_torque = runtime.data.qfrc_actuator[model_map.dof_addresses].copy() * CONTROLLED_JOINT_SIGNS
                actuator_force = runtime.data.actuator_force[model_map.actuator_ids].copy() * CONTROLLED_JOINT_SIGNS
                mujoco.mj_copyData(snapshot, model, runtime.data)
                mujoco.mj_forward(model, snapshot)
                state = collect_kinematic_state(model, snapshot, model_map, contract)
                metrics = collect_metrics(model, snapshot, model_map, contract, state)
                normal, vertical, penetration, robot_contacts = contact_state(model, snapshot)
                warning_delta = sum(int(item.number) for item in runtime.data.warning) - warning_before
                failure = failure_reason(snapshot, metrics, penetration, robot_contacts, warning_delta)
                torques = np.asarray(recorder.torques)
                if torques.shape != (contract.physics_steps_per_action, 6):
                    raise ValueError("控制子步记录数量不完整")
                row = {
                    "time_s": float(snapshot.time), "observation": observation,
                    "q_control_rad": state.joint_position_control, "dq_control_rad_s": state.joint_velocity_control,
                    "qpos_mujoco": snapshot.qpos.copy(), "qvel_mujoco": snapshot.qvel.copy(),
                    "projected_gravity": state.projected_gravity_control,
                    "height_m": state.base_height_m, "tilt_rad": metrics["tilt_rad"],
                    "linear_velocity_control_m_s": state.com_linear_velocity_control,
                    "angular_velocity_control_rad_s": state.angular_velocity_control,
                    "base_com_world_m": snapshot.xipos[model_map.base_body_id].copy(),
                    "base_quat_world_wxyz": snapshot.xquat[model_map.base_body_id].copy(),
                    "action_raw": action, "leg_target_rad": targets.leg_position_target_mujoco,
                    "wheel_target_control_rad_s": targets.wheel_velocity_target_mujoco * contract.wheel_signs,
                    "motor_generalized_torque_nm": actual_torque, "actuator_force_nm": actuator_force,
                    "pd_command_mean_nm": torques.mean(axis=0), "pd_command_peak_abs_nm": np.abs(torques).max(axis=0),
                    "effort_saturation_fraction": step.metrics["effort_saturation_fraction"],
                    "body_normal_force_n": normal, "body_vertical_force_n": vertical,
                    "wheel_normal_force_n": normal[wheel_ids], "wheel_vertical_force_n": vertical[wheel_ids],
                    "nonwheel_normal_force_n": normal[other_ids].sum(),
                    "contact_count": snapshot.ncon, "robot_contact_count": robot_contacts,
                    "penetration_m": penetration, "loop_error_m": metrics["max_loop_closure_error_m"],
                    "external_world_force_torque": external, "perturbation_active": int(active and np.any(external)),
                    "warning_count": warning_delta,
                }
                if not all(np.isfinite(value).all() for value in row.values()):
                    raise FloatingPointError("采集字段包含非有限数值")
                rows.append(row)
                if failure:
                    break
            arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            candidate = ((arrays["time_s"] >= 2) & (arrays["tilt_rad"] < 0.10)
                         & (np.linalg.norm(arrays["linear_velocity_control_m_s"][:, :2], axis=1) < 0.15)
                         & (np.linalg.norm(arrays["angular_velocity_control_rad_s"], axis=1) < 2)
                         & (np.max(np.abs(arrays["dq_control_rad_s"][:, :4]), axis=1) < 5)
                         & (np.max(np.abs(arrays["dq_control_rad_s"][:, 4:]), axis=1) < 10)
                         & (arrays["height_m"] > 0.16) & (arrays["height_m"] < 0.24)
                         & (arrays["nonwheel_normal_force_n"] < 1e-6)
                         & (np.min(arrays["wheel_normal_force_n"], axis=1) > 1e-6))
            arrays["reference_candidate"] = candidate
            nominal = scenario["name"].startswith("nominal")
            if nominal and failure is None:
                stable_parts.append({key: value[candidate] for key, value in arrays.items()})
            path = output / f"{model_name}_{scenario_index:02d}.npz"
            np.savez_compressed(path, **arrays)
            record = {"model": model_name, "scenario": scenario, "path": str(path), "sha256": sha256_file(path),
                      "samples": len(rows), "simulated_s": float(arrays["time_s"][-1]), "failure": failure,
                      "reference_candidates": int(candidate.sum()), "nominal_reference_episode": nominal,
                      "tilt_max_rad": float(arrays["tilt_rad"].max()),
                      "loop_error_max_m": float(arrays["loop_error_m"].max()),
                      "penetration_max_m": float(arrays["penetration_m"].max()),
                      "nonwheel_force_max_n": float(arrays["nonwheel_normal_force_n"].max())}
            episode_records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            write_json(output / "episodes.json", episode_records)
        if not stable_parts or not sum(len(part["time_s"]) for part in stable_parts):
            raise ValueError(f"{model_name} 没有可用名义站立参考")
        merged = {key: np.concatenate([part[key] for part in stable_parts]) for key in stable_parts[0]}
        bank = {key: value[::5] for key, value in merged.items()}
        bank_path = output / f"{model_name}_paired_reference.npz"
        np.savez_compressed(bank_path, **bank)
        stats = {key: distribution(merged[key]) for key in (
            "q_control_rad", "dq_control_rad_s", "height_m", "tilt_rad", "linear_velocity_control_m_s",
            "angular_velocity_control_rad_s", "leg_target_rad", "wheel_target_control_rad_s", "action_raw",
            "wheel_normal_force_n", "wheel_vertical_force_n", "motor_generalized_torque_nm",
            "loop_error_m", "penetration_m")}
        stats.update({"angular_speed_rad_s": distribution(np.linalg.norm(merged["angular_velocity_control_rad_s"], axis=1)),
                      "horizontal_speed_m_s": distribution(np.linalg.norm(merged["linear_velocity_control_m_s"][:, :2], axis=1)),
                      "leg_speed_abs_rad_s": distribution(np.abs(merged["dq_control_rad_s"][:, :4])),
                      "wheel_speed_abs_rad_s": distribution(np.abs(merged["dq_control_rad_s"][:, 4:]))})
        model_summaries[model_name] = {"candidate_samples": len(merged["time_s"]), "body_names": body_names,
                                      "total_mass_kg": float(model.body_mass.sum()),
                                      "weight_n": float(model.body_mass.sum() * -model.opt.gravity[2]),
                                      "paired_reference": str(bank_path), "paired_reference_sha256": sha256_file(bank_path),
                                      "paired_reference_samples": len(bank["time_s"]), "statistics": stats}
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT or sha256_file(export / "actor.ts") != EXPECTED_ACTOR:
        raise ValueError("采集期间冻结模型发生变化")
    summary = {"schema": manifest["schema"], "completed": True, "models": model_summaries,
               "episode_count": len(episode_records), "sample_count": sum(item["samples"] for item in episode_records),
               "simulated_s": sum(item["simulated_s"] for item in episode_records),
               "failures": [item for item in episode_records if item["failure"]],
               "wall_time_s": perf_counter() - started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "reference_selection": "nominal episodes only; t>=2s; tilt<0.1; horizontal speed<0.15; angular speed<2; leg abs speed<5; wheel abs speed<10; height 0.16..0.24; both wheel forces>1e-6; nonwheel force<1e-6; episode no failure; not a success calibration",
               "bytes_total_before_summary": sum(path.stat().st_size for path in output.iterdir() if path.is_file()),
               "identity_unchanged": True, "policy_updates": 0, "optimizer_updates": 0}
    write_json(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "models"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=30.0)
    args = parser.parse_args()
    if args.duration_s < 10 or args.duration_s > 60:
        raise ValueError("每段采集时长必须位于10至60秒")
    collect(args.output.resolve(), args.duration_s)
