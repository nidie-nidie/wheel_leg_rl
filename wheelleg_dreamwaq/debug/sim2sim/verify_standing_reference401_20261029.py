from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
FIELDS = {
    "time_s", "observation", "q_control_rad", "dq_control_rad_s", "qpos_mujoco", "qvel_mujoco",
    "projected_gravity", "height_m", "tilt_rad", "linear_velocity_control_m_s",
    "angular_velocity_control_rad_s", "base_com_world_m", "base_quat_world_wxyz", "action_raw",
    "leg_target_rad", "wheel_target_control_rad_s", "motor_generalized_torque_nm", "actuator_force_nm",
    "pd_command_mean_nm", "pd_command_peak_abs_nm", "effort_saturation_fraction", "body_normal_force_n",
    "body_vertical_force_n", "wheel_normal_force_n", "wheel_vertical_force_n", "nonwheel_normal_force_n",
    "contact_count", "robot_contact_count", "penetration_m", "loop_error_m", "external_world_force_torque",
    "perturbation_active", "warning_count", "reference_candidate",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def xml_metadata(path, joint_order):
    root = ET.parse(path).getroot()
    qpos, qvel, nq, nv = {}, {}, 0, 0
    for element in root.find("worldbody").iter():
        if element.tag not in ("joint", "freejoint"):
            continue
        kind = "free" if element.tag == "freejoint" else element.get("type", "hinge")
        widths = {"free": (7, 6), "ball": (4, 3), "hinge": (1, 1), "slide": (1, 1)}[kind]
        qpos[element.get("name")], qvel[element.get("name")] = nq, nv
        nq, nv = nq + widths[0], nv + widths[1]
    bodies = list(root.find("worldbody").iter("body"))
    names = ["world"] + [element.get("name") for element in bodies]
    mass = sum(float(element.get("mass", 0)) for body in bodies for element in body.findall("inertial"))
    gravity = np.fromstring(root.find("option").get("gravity"), sep=" ")
    return {"qpos": [qpos[name] for name in joint_order], "qvel": [qvel[name] for name in joint_order],
            "nq": nq, "nv": nv, "body_names": names, "mass": mass, "weight": mass * -gravity[2]}


def longest_run(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return int(np.max(np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1), initial=0))


def stats(values):
    values = np.asarray(values)
    return {"min": values.min(axis=0).tolist(), "p01": np.quantile(values, .01, axis=0).tolist(),
            "median": np.median(values, axis=0).tolist(), "p99": np.quantile(values, .99, axis=0).tolist(),
            "max": values.max(axis=0).tolist()}


def rotated_vectors(quaternions, vectors):
    scalar = quaternions[:, :1]
    axis = quaternions[:, 1:]
    doubled = 2 * np.cross(axis, vectors)
    return vectors + scalar * doubled + np.cross(axis, doubled)


def candidate_mask(a):
    return ((a["time_s"] >= 2) & (a["tilt_rad"] < .10)
            & (np.linalg.norm(a["linear_velocity_control_m_s"][:, :2], axis=1) < .15)
            & (np.linalg.norm(a["angular_velocity_control_rad_s"], axis=1) < 2)
            & (np.max(np.abs(a["dq_control_rad_s"][:, :4]), axis=1) < 5)
            & (np.max(np.abs(a["dq_control_rad_s"][:, 4:]), axis=1) < 10)
            & (a["height_m"] > .16) & (a["height_m"] < .24)
            & (a["nonwheel_normal_force_n"] < 1e-6)
            & (np.min(a["wheel_normal_force_n"], axis=1) > 1e-6))


def acceptance(a, median, weight):
    dq = np.abs(a["dq_control_rad_s"])
    masks = {
        "tilt_lt_0p10": a["tilt_rad"] < .10,
        "height_0p185_to_0p215": (a["height_m"] >= .185) & (a["height_m"] <= .215),
        "leg_pose_each_within_0p10_of_nominal_median": np.max(np.abs(a["q_control_rad"][:, :4] - median), axis=1) < .10,
        "angular_speed_lt_0p8": np.linalg.norm(a["angular_velocity_control_rad_s"], axis=1) < .8,
        "leg_speed_each_lt_1p5": dq[:, :4].max(axis=1) < 1.5,
        "wheel_speed_each_lt_2": dq[:, 4:].max(axis=1) < 2,
        "horizontal_speed_lt_0p05": np.linalg.norm(a["linear_velocity_control_m_s"][:, :2], axis=1) < .05,
        "each_wheel_force_gt_0p05_weight": a["wheel_vertical_force_n"].min(axis=1) > .05 * weight,
        "sum_wheel_force_0p8_to_1p2_weight": (a["wheel_vertical_force_n"].sum(axis=1) > .8 * weight)
            & (a["wheel_vertical_force_n"].sum(axis=1) < 1.2 * weight),
        "nonwheel_force_lt_0p01_weight": a["nonwheel_normal_force_n"] < .01 * weight,
        "no_physical_failure": (a["loop_error_m"] < .005) & (a["penetration_m"] < .005)
            & (a["warning_count"] == 0) & (a["robot_contact_count"] == 0),
    }
    after = a["time_s"] >= 2
    combined = np.logical_and.reduce(list(masks.values()))
    force_mean = np.convolve(a["wheel_vertical_force_n"].sum(axis=1), np.ones(10) / 10, mode="valid")
    smoothed_support = np.r_[np.zeros(9, dtype=bool), (force_mean > .8 * weight) & (force_mean < 1.2 * weight)]
    alternate = np.logical_and.reduce([value for key, value in masks.items()
        if key not in ("wheel_speed_each_lt_2", "sum_wheel_force_0p8_to_1p2_weight")])
    alternate &= (dq[:, 4:].max(axis=1) < 6) & smoothed_support
    return {"post_2s_samples": int(after.sum()),
            "individual_post_2s_fraction": {key: float(value[after].mean()) for key, value in masks.items()},
            "joint_post_2s_fraction": float(combined[after].mean()),
            "longest_joint_hold_post_2s_s": longest_run(combined & after) * .02,
            "alternative_wheel_lt_6_and_trailing_0p2s_total_force": {
                "joint_post_2s_fraction": float(alternate[after].mean()),
                "longest_joint_hold_post_2s_s": longest_run(alternate & after) * .02}}


def verify(artifact):
    manifest = load_json(artifact / "manifest.json")
    summary = load_json(artifact / "summary.json")
    episodes = load_json(artifact / "episodes.json")
    errors, hashes, checks, model_results = [], [], [], {}

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(name)

    def hashed(path, expected):
        actual = sha256(path)
        hashes.append({"path": str(path), "expected": expected, "actual": actual, "matched": actual == expected})
        check(f"hash:{Path(path).name}", actual == expected)

    check("completed_summary", summary["completed"] and summary["episode_count"] == 20)
    check("frozen_cpu_collection", manifest["device"] == "cpu" and manifest["policy_updates"] == 0
          and manifest["optimizer_updates"] == 0 and manifest["source_completed_updates"] == 401)
    policy_path = Path(manifest["policy_manifest"])
    policy = load_json(policy_path)
    hashed(manifest["identity"]["checkpoint"], manifest["identity"]["checkpoint_sha256"])
    hashed(policy_path.parent / policy["actor_file"], manifest["identity"]["actor_sha256"])
    for key in ("policy_manifest", "collector", "formal_xml", "recovery_contact_manifest", "recovery_xml"):
        hashed(manifest[key], manifest[key + "_sha256"])
    check("checkpoint_export_identity", policy["source_checkpoint_sha256"] == manifest["identity"]["checkpoint_sha256"]
          and policy["source_completed_iterations"] == 401)
    check("both_models_present", set(item["model"] for item in episodes) == {"formal401", "recovery_contact"})
    signs = np.asarray(manifest["control_signs"])
    target = manifest["target_contract"]
    total_samples = 0
    for model_name, xml_key in (("formal401", "formal_xml"), ("recovery_contact", "recovery_xml")):
        meta = xml_metadata(manifest[xml_key], manifest["joint_order"])
        records = [record for record in episodes if record["model"] == model_name]
        declared = summary["models"][model_name]
        check(f"{model_name}:ten_scenarios", len(records) == 10 and [r["scenario"] for r in records] == manifest["scenarios"])
        check(f"{model_name}:mass", np.isclose(meta["weight"], declared["weight_n"], atol=1e-10)
              and meta["body_names"] == declared["body_names"])
        arrays = []
        perturbations = []
        for record in records:
            path = Path(record["path"])
            hashed(path, record["sha256"])
            a = load_arrays(path)
            arrays.append(a)
            name = f"{model_name}:{record['scenario']['name']}"
            check(name + ":fields", set(a) == FIELDS)
            check(name + ":finite", all(np.isfinite(value).all() for value in a.values()))
            check(name + ":samples", all(value.shape[0] == 1500 for value in a.values()) and record["samples"] == 1500)
            total_samples += len(a["time_s"])
            check(name + ":time", np.allclose(a["time_s"], .02 * np.arange(1, 1501), atol=2e-10, rtol=0))
            check(name + ":joint_mapping", np.allclose(a["q_control_rad"], a["qpos_mujoco"][:, meta["qpos"]] * signs, atol=1e-12)
                  and np.allclose(a["dq_control_rad_s"], a["qvel_mujoco"][:, meta["qvel"]] * signs, atol=1e-12))
            check(name + ":target_mapping", np.allclose(a["leg_target_rad"], np.clip(np.asarray(target["q_nominal"])
                  + target["leg_action_scale"] * a["action_raw"][:, :4].astype(np.float64), target["leg_target_lower"], target["leg_target_upper"]), atol=1e-7)
                  and np.allclose(a["wheel_target_control_rad_s"], target["wheel_action_scale"] * a["action_raw"][:, 4:].astype(np.float64), atol=1e-6))
            check(name + ":history_and_previous_action", np.array_equal(a["observation"][1:, :100], a["observation"][:-1, 25:])
                  and np.array_equal(a["observation"][1:, -6:], a["action_raw"][:-1]))
            check(name + ":gravity_height", np.allclose(np.linalg.norm(a["projected_gravity"], axis=1), 1, atol=1e-10)
                  and np.allclose(np.arccos(np.clip(-a["projected_gravity"][:, 2], -1, 1)), a["tilt_rad"], atol=1e-10)
                  and np.array_equal(a["height_m"], a["base_com_world_m"][:, 2]))
            wheel_ids = [meta["body_names"].index(name) for name in ("jwheel_left", "jwheel_right")]
            other_ids = [i for i in range(1, len(meta["body_names"])) if i not in wheel_ids]
            check(name + ":contact_mapping", np.array_equal(a["body_normal_force_n"][:, wheel_ids], a["wheel_normal_force_n"])
                  and np.array_equal(a["body_vertical_force_n"][:, wheel_ids], a["wheel_vertical_force_n"])
                  and np.allclose(a["body_normal_force_n"][:, other_ids].sum(axis=1), a["nonwheel_normal_force_n"], atol=1e-12))
            check(name + ":candidate_selection", np.array_equal(candidate_mask(a), a["reference_candidate"])
                  and int(a["reference_candidate"].sum()) == record["reference_candidates"])
            check(name + ":physical_and_torque", record["failure"] is None and (a["warning_count"] == 0).all()
                  and (a["robot_contact_count"] == 0).all() and a["loop_error_m"].max() < .005
                  and a["penetration_m"].max() < .005 and (a["pd_command_peak_abs_nm"] <= np.asarray(target["effort_limits"]) + 1e-10).all())
            scenario = record["scenario"]
            requested = np.r_[scenario.get("force_control_n", [0, 0, 0]), scenario.get("torque_control_nm", [0, 0, 0])]
            active = a["perturbation_active"].astype(bool)
            expected_active = np.zeros(1500, dtype=bool)
            if np.any(requested):
                expected_active[250:255] = True
            external = a["external_world_force_torque"]
            magnitude_ok = np.allclose(np.linalg.norm(external[:, :3], axis=1), expected_active * np.linalg.norm(requested[:3]), atol=1e-12)
            magnitude_ok &= np.allclose(np.linalg.norm(external[:, 3:], axis=1), expected_active * np.linalg.norm(requested[3:]), atol=1e-12)
            check(name + ":perturbation_ticks_magnitude", np.array_equal(active, expected_active) and magnitude_ok)
            if np.any(requested):
                rotation = np.asarray(policy["frames"]["r_control_from_usd"])
                expected_force = rotated_vectors(a["base_quat_world_wxyz"][249:254], np.broadcast_to(rotation.T @ requested[:3], (5, 3)))
                expected_torque = rotated_vectors(a["base_quat_world_wxyz"][249:254], np.broadcast_to(rotation.T @ requested[3:], (5, 3)))
                # 施力读取原运行时最后子步的xmat；归档四元数来自积分完成后的副本。
                # 用一个物理步的转动幅度约束方向差，避免将两种时序误当成同一姿态。
                direction_errors = []
                for offset, expected_vector, requested_vector in ((0, expected_force, requested[:3]), (3, expected_torque, requested[3:])):
                    magnitude = np.linalg.norm(requested_vector)
                    if magnitude:
                        direction_errors.append(float(np.linalg.norm(external[250:255, offset:offset + 3] - expected_vector, axis=1).max() / magnitude))
                direction_bound = 2 * manifest["physics_dt_s"] * np.linalg.norm(a["angular_velocity_control_rad_s"][248:256], axis=1).max() + 1e-6
                check(name + ":perturbation_direction", max(direction_errors) <= direction_bound,
                      {"max_unit_direction_difference": max(direction_errors), "one_substep_rotation_bound": float(direction_bound),
                       "timing": "原始运行时施力方向与积分后副本方向相差一个物理子步"})
                baseline = arrays[0]
                effect = float(np.max(np.abs(a["qvel_mujoco"][250:350] - baseline["qvel_mujoco"][250:350])))
                prefix_error = float(np.max(np.abs(a["qvel_mujoco"][:250] - baseline["qvel_mujoco"][:250])))
                check(name + ":perturbation_effect", prefix_error < 1e-9 and effect > 1e-6)
                perturbations.append({"scenario": scenario["name"], "control_steps": int(active.sum()),
                    "impulse_force_norm_n_s": float(np.linalg.norm(requested[:3]) * .1),
                    "impulse_torque_norm_nm_s": float(np.linalg.norm(requested[3:]) * .1),
                    "pre_perturbation_max_qvel_difference": prefix_error, "response_max_qvel_difference": effect})
        merged = {key: np.concatenate([a[key][a["reference_candidate"]] for a, record in zip(arrays, records)
                  if record["nominal_reference_episode"]]) for key in FIELDS}
        bank_path = Path(declared["paired_reference"])
        hashed(bank_path, declared["paired_reference_sha256"])
        bank = load_arrays(bank_path)
        check(f"{model_name}:nominal_bank_exact", set(bank) == FIELDS
              and all(np.array_equal(bank[key], merged[key][::5]) for key in FIELDS)
              and len(merged["time_s"]) == declared["candidate_samples"]
              and len(bank["time_s"]) == declared["paired_reference_samples"])
        med = np.median(merged["q_control_rad"][:, :4], axis=0)
        nominal_derived = {
            "angular_speed_rad_s": np.linalg.norm(merged["angular_velocity_control_rad_s"], axis=1),
            "horizontal_speed_m_s": np.linalg.norm(merged["linear_velocity_control_m_s"][:, :2], axis=1),
            "leg_speed_abs_rad_s": np.abs(merged["dq_control_rad_s"][:, :4]),
            "wheel_speed_abs_rad_s": np.abs(merged["dq_control_rad_s"][:, 4:]),
        }
        for key, declared_stats in declared["statistics"].items():
            values = merged[key] if key in merged else nominal_derived[key]
            calculated = {"mean": values.mean(axis=0), "std": values.std(axis=0), "min": values.min(axis=0), "max": values.max(axis=0)}
            calculated.update({name: np.quantile(values, probability, axis=0) for name, probability in
                (("p005", .005), ("p01", .01), ("p50", .5), ("p99", .99), ("p995", .995))})
            check(f"{model_name}:summary_statistic:{key}", all(np.allclose(value, declared_stats[name], atol=1e-12)
                  for name, value in calculated.items()))
        results = [{"scenario": record["scenario"]["name"], **acceptance(a, med, meta["weight"])} for a, record in zip(arrays, records)]
        model_results[model_name] = {"episodes": len(records), "samples": sum(len(a["time_s"]) for a in arrays),
            "weight_n": meta["weight"], "nominal_candidates": len(merged["time_s"]), "paired_samples": len(bank["time_s"]),
            "nominal_leg_median_rad": med.tolist(),
            "nominal_statistics": {key: stats(merged[key]) for key in ("q_control_rad", "height_m", "tilt_rad", "dq_control_rad_s",
                "angular_velocity_control_rad_s", "linear_velocity_control_m_s", "wheel_vertical_force_n", "motor_generalized_torque_nm")},
            "nominal_derived_statistics": {key: stats(value) for key, value in nominal_derived.items()},
            "nominal_wheel_force_total_over_weight": stats(merged["wheel_vertical_force_n"].sum(axis=1) / meta["weight"]),
            "max_world_vertical_perturbation_force_abs_n": max(float(np.abs(a["external_world_force_torque"][:, 2]).max()) for a in arrays),
            "perturbation_effects": perturbations, "candidate_acceptance_by_episode": results}
    check("sample_total", total_samples == summary["sample_count"] == 30000)
    result = {"schema": "StandingReference401IndependentVerificationV1", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors, "errors": errors, "check_count": len(checks), "checks": checks, "file_hashes": hashes,
        "input_json_sha256": {name: sha256(artifact / name) for name in ("manifest.json", "episodes.json", "summary.json")},
        "verifier_sha256": sha256(__file__), "models": model_results, "sample_count": total_samples,
        "simulated_s": total_samples * .02, "input_bytes": sum(p.stat().st_size for p in artifact.iterdir()
            if p.is_file() and p.name != "verification.json"),
        "limitations": ["离线核验未运行任何模拟、策略或GPU任务", "接触数据仅为50Hz控制帧，不能证明子步峰值",
            "扰动方向读取原运行时xmat，比积分后副本四元数早一个物理子步；方向核验采用已记录角速度和1ms物理步约束",
            "水平扰动定义在控制坐标系；机身倾斜使旋转后的世界力存在微小垂直分量，名义站立参考没有外力",
            "候选门限覆盖仅描述401数据，不能证明起身策略已达到相同能力", "名义两段包含不同偏航角，但共享确定性策略及动力学，不能视为独立随机试验"]}
    destination = artifact / "verification.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "errors": errors, "checks": len(checks), "sample_count": total_samples,
          "verification": str(destination)}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=PROJECT / "recovery/artifacts/standing-reference401-20261029")
    verify(parser.parse_args().artifact.resolve())
