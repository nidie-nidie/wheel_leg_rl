import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .independent_rewards import compute_independent_reward
from .independent_state import StandingGateConfig, StandingTask
from .standing_reference import StandingReference


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def load_standing_calibration(path, expected_sha256, *, require_formal=False):
    path = Path(path)
    if file_sha256(path) != expected_sha256.upper():
        raise ValueError("独立站立校准文件身份不匹配")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["schema_version"] != "IndependentStandingCalibrationV1":
        raise ValueError("独立站立校准 schema 无效")
    StandingGateConfig(**report["gate_config"]).validate()
    if report["reference"]["samples"] != 560 or not report["computation_verified"]:
        raise ValueError("独立站立校准计算核验缺失")
    if require_formal:
        if (not report["formal_ready"] or not report["positive_401_passed"]
                or not report["negative_6200_passed"] or report["missing_formal_evidence"]
                or len(report["positive_401"]) != 20 or len(report["negative_6200"]) != 12
                or not isinstance(report["physical_negative"], dict)
                or report["physical_negative"]["passed"] is not True
                or len(report["physical_negative"]["episodes"]) != 5
                or not all(item["timely_greater"] for item in report["hold_return_comparison"])):
            raise ValueError("独立站立完整准入校准尚未通过")
    return report


def _tensor(value):
    return torch.as_tensor(value, dtype=torch.float32)


def _longest(values):
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _calibrate_physical_negatives(manifest_path, reference):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"single_wheel", "fast_sliding", "body_supported", "inverted", "low_body"}
    if (manifest["schema_version"] != "IndependentStandingPhysicalNegativesV1"
            or {item["name"] for item in manifest["episodes"]} != required
            or manifest["wheel_body_names"] != ["jwheel_left", "jwheel_right"]
            or manifest["force_source"] != "mj_contactForce_actual_floor_normal_force"
            or manifest["contact_model_sha256"] != reference.identity["contact_model_sha256"]
            or manifest["contact_manifest_sha256"] != reference.identity["contact_manifest_sha256"]
            or manifest["weight_n"] != reference.identity["weight_n"]):
        raise ValueError("真实异常支撑校准身份或种类无效")
    results = []
    for episode in manifest["episodes"]:
        path = manifest_path.parent / episode["file"]
        if file_sha256(path) != episode["sha256"]:
            raise ValueError("真实异常支撑轨迹哈希不同")
        with np.load(path, allow_pickle=False) as archive:
            data = {name: archive[name].copy() for name in archive.files}
        wheel_ids = [manifest["body_names"].index(name) for name in manifest["wheel_body_names"]]
        nonwheel_ids = [index for index, name in enumerate(manifest["body_names"])
                        if name != "world" and index not in wheel_ids]
        if (not np.allclose(data["body_normal_force_n"][:, wheel_ids], data["wheel_normal_force_n"], rtol=0, atol=1e-7)
                or not np.allclose(data["body_normal_force_n"][:, nonwheel_ids].sum(axis=1),
                                   data["nonwheel_normal_force_n"], rtol=0, atol=1e-7)):
            raise ValueError("异常支撑真实接触力与部件名称映射不同")
        weight = reference.identity["weight_n"]
        conditions = {
            "single_wheel": np.any((data["wheel_normal_force_n"].min(axis=1) < .1 * weight)
                                   & (data["wheel_normal_force_n"].max(axis=1) >= .1 * weight)),
            "fast_sliding": np.any(np.linalg.norm(data["linear_velocity_control_m_s"][:, :2], axis=1) > .05),
            "body_supported": np.any(data["nonwheel_normal_force_n"] > .01 * weight),
            "inverted": np.any(data["projected_gravity"][:, 2] > 0.),
            "low_body": np.any(data["height_m"] < .185),
        }
        condition_observed = bool(conditions[episode["name"]])
        if not condition_observed:
            raise ValueError(f"真实异常轨迹没有出现所声明的条件: {episode['name']}")
        task = StandingTask(1, "cpu", reference, reference.identity["weight_n"], mode="eval")
        accepted, failures, success = [], Counter(), False
        if len(data["height_m"]) < 50:
            raise ValueError("异常支撑真实轨迹不足一秒")
        for index in range(len(data["height_m"])):
            result = task.step(q=_tensor(data["q_control_rad"][index:index + 1, :4]),
                dq=_tensor(data["dq_control_rad_s"][index:index + 1]),
                omega=_tensor(data["angular_velocity_control_rad_s"][index:index + 1]),
                gravity=_tensor(data["projected_gravity"][index:index + 1]),
                linear=_tensor(data["linear_velocity_control_m_s"][index:index + 1]),
                height=_tensor(data["height_m"][index:index + 1]),
                wheel_forces=_tensor(data["wheel_normal_force_n"][index:index + 1]),
                nonwheel_force=_tensor(data["nonwheel_normal_force_n"][index:index + 1]),
                world_xy=_tensor(data["base_com_world_m"][index:index + 1, :2]),
                physical_failure=torch.as_tensor(data["physical_failure"][index:index + 1], dtype=torch.bool))
            accepted.append(bool(result.qualified[0]))
            success |= bool(result.success[0])
            failures.update(name for name, value in result.gate_failures.items() if bool(value[0]))
        results.append({"name": episode["name"], "path": str(path), "sha256": episode["sha256"],
                        "samples": len(accepted), "qualified_steps": sum(accepted),
                        "longest_qualified_steps": _longest(accepted), "full_success": success,
                        "source_record_id": episode["source_record_id"],
                        "required_condition_observed": condition_observed,
                        "gate_failure_counts": dict(failures)})
    passed = all(item["required_condition_observed"] and not item["full_success"]
                 and item["longest_qualified_steps"] < 50 for item in results)
    return {"manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
            "contact_manifest_sha256": manifest["contact_manifest_sha256"],
            "contact_model_sha256": manifest["contact_model_sha256"],
            "body_names": manifest["body_names"], "wheel_body_names": manifest["wheel_body_names"],
            "passed": passed, "episodes": results}


def calibrate_reference_data(reference_dir, negative_trace_path, output_path, *, physical_negative_manifest=None):
    torch.set_num_threads(1)
    reference_dir, negative_trace_path = Path(reference_dir), Path(negative_trace_path)
    summary = json.loads((reference_dir / "summary.json").read_text(encoding="utf-8"))
    reference = StandingReference.load(reference_dir / "manifest.json",
        reference_dir / "recovery_contact_paired_reference.npz",
        summary["models"]["recovery_contact"]["paired_reference_sha256"])
    episodes = json.loads((reference_dir / "episodes.json").read_text(encoding="utf-8"))
    if len(episodes) != 20:
        raise ValueError("站立校准必须读取全部 20 段真实401记录")
    positive = []
    reward_distributions = []
    actual_returns = []
    for episode in episodes:
        path = reference_dir / Path(episode["path"]).name
        if file_sha256(path) != episode["sha256"]:
            raise ValueError("401校准轨迹哈希不匹配")
        with np.load(path, allow_pickle=False) as archive:
            data = {name: archive[name].copy() for name in archive.files}
        task = StandingTask(1, "cpu", reference, reference.identity["weight_n"], mode="eval")
        accepted, failure_counts, success_step = [], Counter(), None
        q = _tensor(data["q_control_rad"][:, :4])
        height = _tensor(data["height_m"])
        match = reference.nearest(q, height)
        wheel_values, body_values = [], []
        for index in range(len(q)):
            result = task.step(q=q[index:index + 1], dq=_tensor(data["dq_control_rad_s"][index:index + 1]),
                omega=_tensor(data["angular_velocity_control_rad_s"][index:index + 1]),
                gravity=_tensor(data["projected_gravity"][index:index + 1]),
                linear=_tensor(data["linear_velocity_control_m_s"][index:index + 1]),
                height=height[index:index + 1], wheel_forces=_tensor(data["wheel_normal_force_n"][index:index + 1]),
                nonwheel_force=_tensor(data["nonwheel_normal_force_n"][index:index + 1]),
                world_xy=_tensor(data["base_com_world_m"][index:index + 1, :2]),
                physical_failure=torch.tensor([bool(data["warning_count"][index] or
                    data["loop_error_m"][index] > .005 or data["penetration_m"][index] > .003)]))
            # 终态后继续校准每步物理门限，连续回合状态不会再次产生终奖。
            failures = {name: bool(value[0]) for name, value in result.gate_failures.items()}
            physical_qualified = not any(value for name, value in failures.items() if name != "displacement")
            accepted.append(physical_qualified)
            for name, failed in failures.items():
                failure_counts[name] += int(failed)
            if bool(result.success[0]) and success_step is None:
                success_step = index + 1
            wheel_values.append(task.contact_window.wheel_mean[0].clone())
            body_values.append(task.contact_window.nonwheel_mean[0].clone())
        post = np.asarray(data["time_s"]) >= 2.
        accepted = np.asarray(accepted, dtype=bool)
        # 原401命令按实际腿目标和轮目标换算到新范围，只用于状态奖励尺度诊断。
        action = torch.cat((_tensor(data["leg_target_rad"]), _tensor(data["wheel_target_control_rad_s"]) / 30.), dim=1)
        previous = torch.cat((action[:1], action[:-1]), dim=0)
        previous_previous = torch.cat((previous[:1], previous[:-1]), dim=0)
        valid_targets = ((action.abs() <= 1.).all(dim=1) & (previous.abs() <= 1.).all(dim=1)
                         & (previous_previous.abs() <= 1.).all(dim=1))
        false = torch.zeros(int(valid_targets.sum()), dtype=torch.bool)
        reward, components = compute_independent_reward(q=q[valid_targets],
            dq=_tensor(data["dq_control_rad_s"])[valid_targets], omega=_tensor(data["angular_velocity_control_rad_s"])[valid_targets],
            gravity=_tensor(data["projected_gravity"])[valid_targets], linear=_tensor(data["linear_velocity_control_m_s"])[valid_targets],
            height=height[valid_targets], torque=_tensor(data["motor_generalized_torque_nm"])[valid_targets], actions=action[valid_targets],
            previous_actions=previous[valid_targets], previous_previous_actions=previous_previous[valid_targets],
            wheel_forces_mean=torch.stack(wheel_values)[valid_targets], nonwheel_force_mean=torch.stack(body_values)[valid_targets],
            reference=reference, weight_n=reference.identity["weight_n"], success=false, physical_failure=false, timeout=false)
        if float(reward.max()) > .107001:
            raise ValueError("连续奖励超过设计上界")
        stats = {name: {"min": float(value.min()), "mean": float(value.mean()), "max": float(value.max())}
                 for name, value in components.items()}
        reward_distributions.append({"model": episode["model"], "scenario": episode["scenario"]["name"],
                                     "included_valid_target_samples": int(valid_targets.sum()),
                                     "excluded_target_samples": int((~valid_targets).sum()),
                                     "maximum_original_wheel_target_rad_s": float(np.abs(data["wheel_target_control_rad_s"]).max()),
                                     "components": stats})
        # 相同真实合格片段额外拖延一步的回报比较，终奖20保持原值。
        full_reward = torch.full((len(q),), float("nan"), dtype=torch.float64)
        full_reward[valid_targets] = reward.double()
        contiguous = 0
        hold_start = None
        for index, valid in enumerate(np.asarray(post) & valid_targets.numpy() & accepted):
            contiguous = contiguous + 1 if valid else 0
            if contiguous == 100:
                hold_start = index - 99
                break
        if hold_start is None:
            raise ValueError("参考缺少 100 步连续合格且目标可表示的真实 HOLD 片段")
        hold = full_reward[hold_start:hold_start + 100]
        gamma = .99
        timely = float((hold * gamma ** torch.arange(len(hold))).sum() + gamma ** (len(hold) - 1) * 20.)
        delayed = float((hold * gamma ** torch.arange(len(hold))).sum() + gamma ** len(hold) * (float(hold[-1]) + 20.))
        actual_returns.append({"model": episode["model"], "scenario": episode["scenario"]["name"],
                               "timely": timely, "one_step_delayed": delayed, "timely_greater": timely > delayed})
        positive.append({"model": episode["model"], "scenario": episode["scenario"]["name"],
            "path": str(path), "sha256": episode["sha256"], "samples": len(q),
            "post_2s_qualified_fraction": float(accepted[post].mean()),
            "longest_post_2s_qualified_steps": _longest(accepted[post]),
            "first_full_eval_success_step": success_step,
            "full_eval_success": success_step is not None, "gate_failure_counts": dict(failure_counts),
            "paired_reference_max_leg_error_rad": float((q - match.q).abs().max())})
    negative = defaultdict(lambda: {"steps": 0, "kinematic_rejected_steps": 0, "rejection_counts": Counter()})
    with negative_trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            state = row["state"]
            q, height = _tensor([state["joint_position"][:4]]), _tensor([state["height"]])
            match = reference.nearest(q, height)
            config = StandingGateConfig()
            failures = {
                "tilt": row["tilt_rad"] > config.tilt_rad,
                "height": not config.height_min_m <= state["height"] <= config.height_max_m,
                "leg_pose": float((q - match.q).abs().max()) > config.leg_error_rad,
                "angular_velocity": np.linalg.norm(state["angular_velocity"]) > config.angular_speed_rad_s,
                "leg_velocity": np.max(np.abs(state["joint_velocity"][:4])) > config.leg_speed_rad_s,
                "wheel_velocity": np.max(np.abs(state["joint_velocity"][4:])) > config.wheel_speed_rad_s,
                "horizontal_velocity": np.linalg.norm(state["linear_velocity"][:2]) > config.horizontal_speed_m_s,
                "physical_failure": bool(row["physical_failures"]),
            }
            item = negative[row["record_id"]]
            item["category"] = row["category"]
            item["steps"] += 1
            item["kinematic_rejected_steps"] += int(any(failures.values()))
            item["rejection_counts"].update(name for name, failed in failures.items() if failed)
    if len(negative) != 12:
        raise ValueError("6200校准必须读取全部 12 段真实留出轨迹")
    negative_results = [{"record_id": key, **value, "rejection_counts": dict(value["rejection_counts"]),
                         "full_gate_success_excluded_by_kinematics": value["steps"] == value["kinematic_rejected_steps"]}
                        for key, value in sorted(negative.items())]
    positive_pass = all(item["post_2s_qualified_fraction"] >= .99
                        and item["longest_post_2s_qualified_steps"] >= 500 and item["full_eval_success"] for item in positive)
    negative_pass = all(item["full_gate_success_excluded_by_kinematics"] for item in negative_results)
    physical_negative = None
    missing = []
    if physical_negative_manifest is None:
        missing.append("actual_anomalous_single_wheel_body_support_fast_sliding_gate_rejection")
    else:
        physical_negative = _calibrate_physical_negatives(physical_negative_manifest, reference)
        if not physical_negative["passed"]:
            missing.append("actual_anomalous_support_gate_rejection_failed")
    returns_pass = all(item["timely_greater"] for item in actual_returns)
    report = {"schema_version": "IndependentStandingCalibrationV1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "gate_config": asdict(StandingGateConfig()),
        "reference": reference.identity, "positive_401": positive, "positive_401_passed": positive_pass,
        "negative_6200": negative_results, "negative_6200_trace": str(negative_trace_path),
        "negative_6200_trace_sha256": file_sha256(negative_trace_path), "negative_6200_passed": negative_pass,
        "negative_6200_contact_identity_available": False,
        "reward_distributions_401": reward_distributions, "hold_return_comparison": actual_returns,
        "physical_negative": physical_negative,
        "computation_verified": True, "formal_ready": positive_pass and negative_pass and returns_pass and not missing,
        "missing_formal_evidence": missing,
        "historical_training_archive_read": False, "training_updates_performed": 0}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--negative-trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical-negative-manifest")
    arguments = parser.parse_args()
    report = calibrate_reference_data(arguments.reference_dir, arguments.negative_trace, arguments.output,
                                     physical_negative_manifest=arguments.physical_negative_manifest)
    print(json.dumps({"output": arguments.output, "positive_401_passed": report["positive_401_passed"],
                      "negative_6200_passed": report["negative_6200_passed"], "formal_ready": report["formal_ready"]}))


if __name__ == "__main__":
    main()
