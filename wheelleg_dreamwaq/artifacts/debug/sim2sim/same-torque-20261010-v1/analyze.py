"""Validate the intervention before comparing the physical response."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from wheelleg_dreamwaq.schemas.frames import quat_rotate_inverse_wxyz, transform_usd_vector_to_control

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
OLD = ROOT.parent / "ground-on-off-20261010-v1"
MODES = ("formal", "direct_shared", "direct_zero")
ENGINES = ("isaac", "mujoco")
JOINTS = ("jIJ", "jIO", "jAB", "jAG", "jwheel_left", "jwheel_right")
SIGNS = np.asarray((1., 1., 1., 1., 1., -1.))
SCENARIOS = ("nominal_stand", "low_stand", "high_stand", "forward", "reverse", "turn_left", "turn_right", "combined")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def sample(evidence, index, ms, engine):
    rows = [s for s in evidence["samples"] if s["phase"] != "constructor_after_cache_load"
            and abs(s["elapsed_s"] - ms / 1000) < 1e-10
            and (engine == "isaac" or s["environment_index"] == index)]
    assert len(rows) == 1, (engine, index, ms, len(rows))
    return rows[0]


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def main():
    inputs = read(ROOT / "inputs.json")
    formal_model_manifest = read(PROJECT / "sim2sim/mujoco/model_manifest.json")
    assert inputs["canonical_joint_order"] == list(JOINTS) and inputs["hold_duration_s"] == 0.020
    assert inputs["not_measured_isaac_drive_torque"]
    assert digest(ROOT / "make_inputs.py") == inputs["generator_sha256"]
    provenance = dict(inputs["input_sha256"])
    for path, expected in provenance.items():
        assert digest(path) == expected, path
    protected = read(ROOT / "protected-before.json")
    for path, expected in protected.items():
        assert digest(path) == expected, path
    records = read(ROOT / "execution.json")
    selected, data = {}, {}
    for engine in ENGINES:
        for mode in MODES:
            matches = [r for r in records if r["engine"] == engine and r["mode"] == mode and r["exit_code"] == 0
                       and r["probe_sha256"] == digest(ROOT / f"probe_{engine}.py")
                       and r["torque_input_sha256"] == digest(ROOT / "inputs.json")
                       and r.get("matching_initial_state_sha256") is None]
            assert matches, (engine, mode)
            record = matches[-1]
            selected[engine, mode] = record
            path = Path(record["output"]) / "evidence.json"
            provenance[str(path)] = digest(path)
            d = read(path)
            data[engine, mode] = d
            assert d["debug_only"] and not d["formal_ranking_eligible"] and d["sources_unchanged"]
            assert d["free_base"] and not d["ground_enabled"]
            assert d["drive_mode"] == mode and d["control_ticks"] == 1
            assert d["test_case"] == "shared_first_action_hold"
            assert d["gravity_m_s2"] == [0., 0., -9.81] and d["reset_physics_time_advanced_s"] == 0.
            assert d["probe_sha256"] == record["probe_sha256"] and d["torque_input_sha256"] == record["torque_input_sha256"]
            assert d["physics_dt_s"] == (0.005 if engine == "isaac" else 0.001)
            assert len(d["samples"]) == (6 if engine == "isaac" else 168)
            for name, value in d["protected_source_sha256"].items():
                assert digest(name) == value and protected[name] == value, name
            if engine == "mujoco":
                assert d["non_contact_compiled_semantics_unchanged"]
                assert d["equality_active"] == [True] * len(formal_model_manifest["equality_constraints"])
                assert max(d["observer_replay_max_errors"]) == 0.
                provenance[str(Path(record["output"]) / "diagnostic-model-identity.json")] = digest(Path(record["output"]) / "diagnostic-model-identity.json")
                for p in (Path(record["output"]) / "model").glob("*.xml*"):
                    provenance[str(p)] = digest(p)
            else:
                actions_path = Path(record["output"]) / "actions.json"
                assert read(actions_path)["shared_first_action_clipped"] == inputs["shared_first_action_clipped"]
                provenance[str(actions_path)] = digest(actions_path)

    ids = {engine: [data[engine, "formal"]["joint_names"].index(name) for name in JOINTS] for engine in ENGINES}
    rotation = np.asarray(formal_model_manifest["r_control_from_mujoco"])

    def velocities(engine, mode, index, ms):
        s = sample(data[engine, mode], index, ms, engine)
        if engine == "isaac":
            q = torch.tensor(np.asarray(s["root_pose_xyzw"])[index, [6, 3, 4, 5]], dtype=torch.float64)
            w = torch.tensor(s["root_velocity_world"][index][3:], dtype=torch.float64)
            angular = transform_usd_vector_to_control(quat_rotate_inverse_wxyz(q, w)).numpy()
            joint = np.asarray(s["joint_velocity"])[index, ids[engine]] * SIGNS
        else:
            angular = rotation @ np.asarray(s["root_qvel"])[3:]
            assert np.max(np.abs(angular - np.asarray(s["angular_control_fresh_jacobian_rad_s"]))) < 1e-12
            joint = np.asarray(s["joint_velocity"])[ids[engine]] * SIGNS
        return angular, joint

    baseline_errors, reset_errors, force_errors = {}, {}, {}
    frame_error, max_speed = 0.0, {}
    for engine in ENGINES:
        old = read(OLD / f"{engine}-shared_first_action_hold-ground-off/evidence.json")
        fields = ("root_pose_xyzw", "root_velocity_world", "joint_position", "joint_velocity") if engine == "isaac" else (
            "root_qpos", "root_qvel", "joint_position", "joint_velocity")
        baseline_errors[engine] = {}
        for field in fields:
            error = max(float(np.max(np.abs(np.asarray(sample(data[engine, "formal"], i, ms, engine)[field]) -
                                             np.asarray(sample(old, i, ms, engine)[field]))))
                        for i in range(8) for ms in (0, 5, 10, 15, 20))
            baseline_errors[engine][field] = error
            assert error == 0.0, (engine, field, error)
        provenance[str(OLD / f"{engine}-shared_first_action_hold-ground-off/evidence.json")] = digest(OLD / f"{engine}-shared_first_action_hold-ground-off/evidence.json")
        for mode in MODES:
            d = data[engine, mode]
            reset_errors[f"{engine}-{mode}"] = {}
            for field in fields:
                error = max(float(np.max(np.abs(np.asarray(sample(d, i, 0, engine)[field]) -
                                                 np.asarray(sample(data[engine, "formal"], i, 0, engine)[field])))) for i in range(8))
                reset_errors[f"{engine}-{mode}"][field] = error
                assert error == 0.0
            max_speed[f"{engine}-{mode}"] = 0.0
            force_errors[f"{engine}-{mode}"] = 0.0
            for s in d["samples"]:
                if s["physics_step"] == 0:
                    continue
                field = "normal_force_world_n" if engine == "isaac" else "normal_force_n"
                assert np.max(np.abs(s[field])) == 0.0
                if engine == "isaac":
                    expected = np.asarray(inputs["zero_torque_control_nm" if mode == "direct_zero" else "shared_first_torque_control_nm"])
                    actual = np.asarray(s["external_force_argument_native_nm"])[:, ids[engine]] * SIGNS
                    if mode == "formal":
                        assert np.max(np.abs(actual)) == 0.0
                    else:
                        error = float(np.max(np.abs(actual - expected)))
                        assert error == 0.0
                        force_errors[f"{engine}-{mode}"] = max(force_errors[f"{engine}-{mode}"], error)
                    speed = np.asarray(s["joint_velocity"])[:, ids[engine]]
                else:
                    angular = rotation @ np.asarray(s["root_qvel"])[3:]
                    frame_error = max(frame_error, float(np.max(np.abs(angular - np.asarray(s["angular_control_fresh_jacobian_rad_s"])))))
                    if mode != "formal":
                        expected = np.asarray(inputs["zero_torque_control_nm" if mode == "direct_zero" else "shared_first_torque_control_nm"])[s["environment_index"]]
                        for field in ("external_ctrl_native_nm", "actuator_force_native_nm", "generalized_actuator_force_native_nm"):
                            error = float(np.max(np.abs(np.asarray(s[field]) * SIGNS - expected)))
                            assert error == 0.0
                            force_errors[f"{engine}-{mode}"] = max(force_errors[f"{engine}-{mode}"], error)
                    speed = np.asarray(s["joint_velocity"])[ids[engine]]
                max_speed[f"{engine}-{mode}"] = max(max_speed[f"{engine}-{mode}"], float(np.max(np.abs(speed))))
            assert max_speed[f"{engine}-{mode}"] < 45.0
            if engine == "isaac":
                before, after = d["drive_before"], d["drive_after"]
                expected = json.loads(json.dumps(before))
                if mode != "formal":
                    for field in ("stiffness", "damping"):
                        for row in expected[field]:
                            for j in ids[engine]: row[j] = 0.0
                        for name in ("legs", "wheels"):
                            expected["actuator_cache"][name][field] = np.zeros_like(expected["actuator_cache"][name][field]).tolist()
                assert after == expected
    assert frame_error < 1e-12

    initial_q_error, height_error, orientation_error = 0.0, 0.0, 0.0
    all_names = data["isaac", "formal"]["joint_names"]
    all_mids = [data["mujoco", "formal"]["joint_names"].index(name) for name in all_names]
    for index in range(8):
        i, m = sample(data["isaac", "formal"], index, 0, "isaac"), sample(data["mujoco", "formal"], index, 0, "mujoco")
        initial_q_error = max(initial_q_error, float(np.max(np.abs(np.asarray(i["joint_position"])[index] - np.asarray(m["joint_position"])[all_mids]))))
        height_error = max(height_error, abs(i["root_pose_xyzw"][index][2] - m["root_qpos"][2]))
        iq = np.asarray(i["root_pose_xyzw"])[index, [6, 3, 4, 5]]
        mq = np.asarray(m["root_qpos"])[3:]
        iq, mq = iq / np.linalg.norm(iq), mq / np.linalg.norm(mq)
        orientation_error = max(orientation_error, float(min(np.linalg.norm(iq - mq), np.linalg.norm(iq + mq))))
        for engine in ENGINES:
            a, j = velocities(engine, "formal", index, 0)
            assert np.max(np.abs(a)) == 0.0 and np.max(np.abs(j)) == 0.0
    # The nominal poses are not bitwise equal across engines. Do not relax the
    # planned tolerance and call that equality: require the extra matched-state runs.
    assert height_error < 1e-6 and orientation_error < 1e-6

    matched_state, matched_rows, matched_aggregate = {}, [], []
    matched_source = Path(selected["isaac", "formal"]["output"]) / "evidence.json"
    for mode in ("direct_shared", "direct_zero"):
        matches = [r for r in records if r["engine"] == "mujoco" and r["mode"] == mode and r["exit_code"] == 0
                   and r["probe_sha256"] == digest(ROOT / "probe_mujoco.py")
                   and r["torque_input_sha256"] == digest(ROOT / "inputs.json")
                   and r.get("matching_initial_state_sha256") == digest(matched_source)]
        assert matches
        record = matches[-1]
        selected["mujoco", mode + "-matched"] = record
        path = Path(record["output"]) / "evidence.json"
        d = read(path)
        data["mujoco", mode + "-matched"] = d
        provenance[str(path)] = digest(path)
        assert d["matching_initial_state_sha256"] == digest(matched_source)
        assert d["debug_only"] and d["sources_unchanged"] and d["free_base"] and not d["ground_enabled"]
        assert d["control_ticks"] == 1 and d["physics_dt_s"] == 0.001 and len(d["samples"]) == 168
        assert d["gravity_m_s2"] == [0., 0., -9.81] and d["reset_physics_time_advanced_s"] == 0.0
        assert d["non_contact_compiled_semantics_unchanged"] and max(d["observer_replay_max_errors"]) == 0.0
        assert d["joint_names"] == data["mujoco", "formal"]["joint_names"]
        assert d["dof_damping"] == data["mujoco", "formal"]["dof_damping"]
        assert d["dof_armature"] == data["mujoco", "formal"]["dof_armature"]
        assert d["equality_active"] == data["mujoco", "formal"]["equality_active"]
        for name, value in d["protected_source_sha256"].items(): assert digest(name) == value == protected[name]
        for p in (Path(record["output"]) / "model").glob("*.xml*"): provenance[str(p)] = digest(p)
        state_errors = {"all_joint_position_rad": 0.0, "root_height_m": 0.0, "normalized_root_orientation": 0.0, "initial_velocity": 0.0}
        for index in range(8):
            i = sample(data["isaac", mode], index, 0, "isaac")
            m = sample(d, index, 0, "mujoco")
            state_errors["all_joint_position_rad"] = max(state_errors["all_joint_position_rad"], float(np.max(np.abs(np.asarray(i["joint_position"])[index] - np.asarray(m["joint_position"])[all_mids]))))
            state_errors["root_height_m"] = max(state_errors["root_height_m"], abs(i["root_pose_xyzw"][index][2] - m["root_qpos"][2]))
            iq = np.asarray(i["root_pose_xyzw"])[index, [6, 3, 4, 5]]
            state_errors["normalized_root_orientation"] = max(state_errors["normalized_root_orientation"], float(np.max(np.abs(iq / np.linalg.norm(iq) - np.asarray(m["root_qpos"])[3:]))))
            state_errors["initial_velocity"] = max(state_errors["initial_velocity"], float(np.max(np.abs(m["root_qvel"]))), float(np.max(np.abs(m["joint_velocity"]))))
        assert state_errors["all_joint_position_rad"] == state_errors["root_height_m"] == state_errors["initial_velocity"] == 0.0
        assert state_errors["normalized_root_orientation"] < 1e-15
        matched_state[mode] = state_errors
        for s in d["samples"]:
            frame_error = max(frame_error, float(np.max(np.abs(rotation @ np.asarray(s["root_qvel"])[3:] - np.asarray(s["angular_control_fresh_jacobian_rad_s"])))))
            assert np.max(np.abs(s["normal_force_n"])) == 0.0
            if not s["physics_step"]: continue
            expected = np.asarray(inputs["zero_torque_control_nm" if mode == "direct_zero" else "shared_first_torque_control_nm"])[s["environment_index"]]
            for field in ("external_ctrl_native_nm", "actuator_force_native_nm", "generalized_actuator_force_native_nm"):
                assert np.array_equal(np.asarray(s[field]) * SIGNS, expected)
            assert np.max(np.abs(np.asarray(s["joint_velocity"])[ids["mujoco"]])) < 45.0
        for ms in (5, 10, 15, 20):
            local = []
            for index in range(8):
                ia, ij = velocities("isaac", mode, index, ms)
                ma, mj = velocities("mujoco", mode + "-matched", index, ms)
                na, nj = velocities("mujoco", mode, index, ms)
                angular_scale = (np.linalg.norm(ia) + np.linalg.norm(ma)) / 2
                joint_scale = (rms(ij) + rms(mj)) / 2
                row = {"mode": mode, "time_ms": ms, "environment_index": index,
                       "angular_error_l2_rad_s": float(np.linalg.norm(ia - ma)), "joint_error_rms_rad_s": rms(ij - mj),
                       "relative_angular_error": float(np.linalg.norm(ia - ma) / angular_scale) if angular_scale > 0.01 else None,
                       "relative_joint_error": rms(ij - mj) / joint_scale if joint_scale > 0.01 else None,
                       "mujoco_initial_alignment_angular_change_l2_rad_s": float(np.linalg.norm(ma - na)),
                       "mujoco_initial_alignment_joint_change_rms_rad_s": rms(mj - nj),
                       "isaac_angular_control_rad_s": ia.tolist(), "mujoco_angular_control_rad_s": ma.tolist(),
                       "isaac_joint_control_rad_s": ij.tolist(), "mujoco_joint_control_rad_s": mj.tolist()}
                local.append(row)
                matched_rows.append(row)
            item = {"mode": mode, "time_ms": ms}
            for metric in ("angular_error_l2_rad_s", "joint_error_rms_rad_s", "relative_angular_error", "relative_joint_error",
                           "mujoco_initial_alignment_angular_change_l2_rad_s", "mujoco_initial_alignment_joint_change_rms_rad_s"):
                values = [r[metric] for r in local if r[metric] is not None]
                item[metric] = float(np.mean(values)) if values else None
            matched_aggregate.append(item)
    assert frame_error < 1e-12

    rows, aggregate = [], []
    for mode in MODES:
        for ms in (5, 10, 15, 20):
            local = []
            for index, name in enumerate(SCENARIOS):
                ia, ij = velocities("isaac", mode, index, ms)
                ma, mj = velocities("mujoco", mode, index, ms)
                aerr, jerr = float(np.linalg.norm(ia - ma)), rms(ij - mj)
                ascale, jscale = float((np.linalg.norm(ia) + np.linalg.norm(ma)) / 2), (rms(ij) + rms(mj)) / 2
                row = {"mode": mode, "time_ms": ms, "environment_index": index, "scenario": name,
                       "angular_error_l2_rad_s": aerr, "joint_error_rms_rad_s": jerr,
                       "leg_error_rms_rad_s": rms((ij - mj)[:4]), "wheel_error_rms_rad_s": rms((ij - mj)[4:]),
                       "angular_response_mean_norm_rad_s": ascale, "joint_response_mean_rms_rad_s": jscale,
                       "relative_angular_error": aerr / ascale if ascale > 0.01 else None,
                       "relative_joint_error": jerr / jscale if jscale > 0.01 else None,
                       "isaac_angular_control_rad_s": ia.tolist(), "mujoco_angular_control_rad_s": ma.tolist(),
                       "isaac_joint_control_rad_s": ij.tolist(), "mujoco_joint_control_rad_s": mj.tolist(),
                       "per_joint_abs_error_rad_s": np.abs(ij - mj).tolist()}
                local.append(row)
                rows.append(row)
            metrics = ("angular_error_l2_rad_s", "joint_error_rms_rad_s", "leg_error_rms_rad_s", "wheel_error_rms_rad_s",
                       "angular_response_mean_norm_rad_s", "joint_response_mean_rms_rad_s", "relative_angular_error", "relative_joint_error")
            item = {"mode": mode, "time_ms": ms}
            for metric in metrics:
                available = [r[metric] for r in local if r[metric] is not None]
                item[metric] = float(np.mean(available)) if available else None
            item["per_joint_mean_abs_error_rad_s"] = np.mean([r["per_joint_abs_error_rad_s"] for r in local], axis=0).tolist()
            aggregate.append(item)
    for p in ROOT.glob("*.py"):
        provenance[str(p)] = digest(p)
    for name in ("execution.json", "inputs.json", "protected-before.json", "diagnostic-plan.md"):
        provenance[str(ROOT / name)] = digest(ROOT / name)
    result = {"schema_version": "SameExternalTorqueAnalysisV1", "passed": True, "debug_only": True,
              "formal_ranking_eligible": False, "protected_files_verified": len(protected),
              "selected_final_executions": list(selected.values()), "total_completed_physics_processes": len(records),
              "baseline_reproduction_max_errors": baseline_errors, "within_engine_initial_state_max_errors": reset_errors,
              "cross_engine_initial_state": {"all_named_joint_position_max_error_rad": initial_q_error,
                  "root_height_max_error_m": height_error, "normalized_orientation_quaternion_l2_error": orientation_error,
                  "initial_planned_tolerance_rad": 1e-6, "joint_tolerance_exceeded": initial_q_error >= 1e-6,
                  "absolute_xy_not_compared": "Different intentional scene origin/reference positions"},
              "matched_initial_state_max_errors": matched_state,
              "matched_initial_aggregate": matched_aggregate, "matched_initial_per_environment": matched_rows,
              "external_torque_delivery_max_error_nm": force_errors,
              "mujoco_frame_vs_fresh_jacobian_max_error_rad_s": frame_error,
              "max_sampled_active_joint_speed_rad_s": max_speed, "canonical_joint_order": list(JOINTS),
              "aggregate": aggregate, "per_environment": rows, "input_sha256": provenance,
              "conclusion": "With the six active PD drives disabled and equal constant external torque applied, a cross-engine angular/joint response gap remains. Its first-5ms aggregate error is smaller than the original-controller reference; the changed excitation/feedback is not a causal decomposition of the original gap. No unique controller, inertia, closure or solver root cause is identified.",
              "limits": ["Constant torque changes excitation and removes active feedback; error changes are not causal percentages.",
                         "Passive damping retains each engine's original integration semantics.",
                         "Physics timesteps remain 5ms and 1ms; integration and constraint differences remain bundled.",
                         "Eight rows are different action conditions, not statistical repeats.",
                         "No ground, no second policy decision, no full sim2sim evaluation or training conclusion."]}
    (ROOT / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    write_report(result)
    plot_response(result)
    print("VERIFICATION_PASSED", json.dumps({k: result[k] for k in ("protected_files_verified", "cross_engine_initial_state", "mujoco_frame_vs_fresh_jacobian_max_error_rad_s")}))
    for row in aggregate: print(json.dumps(row))


def write_report(result):
    ag = {(r["mode"], r["time_ms"]): r for r in result["aggregate"]}
    matched = {(r["mode"], r["time_ms"]): r for r in result["matched_initial_aggregate"]}
    lines = ["# 相同外加关节力矩诊断", "",
             "结论：关闭六个主动 PD、施加相同恒定外加力矩后，两边仍有运动差异。前 5 毫秒平均误差小于原控制对照，到 20 毫秒仍有差异。这确认了本次干预工况下的响应不等价；不能把它直接当作原控制工况的因果分解，也不足以把原来的问题唯一归因于某个电机参数。下一步可针对力矩之后的机械／被动阻尼／约束／积分响应继续定位。", "",
             "这里只比较前 20 毫秒，不接入第二次策略决策。保留重力、自由底座、闭合机构、被动关节参数和正式步长；不修改训练代码、USD、正式 MuJoCo 模型、奖励、动作、命令、随机化、history 或 AdaBoot。", "",
             "## 输入与范围", "",
             "原控制对照使用 run-02 的同一份首动作。直接力矩从 MuJoCo 在相同初态下的正式 PD 计算取得，按原力矩上限截断后统一转成 float32 可精确表示的数值，写入 inputs.json；它不是测得的 Isaac 原驱动力矩。两边施加同一数值并保持 20 毫秒。零力矩对照明确设为 0 Nm，不是零归一化动作。", "",
             "## 结果", "",
             "下表是八个输入条件的平均值，不是八次统计重复。机身角速度差为三维向量差的 L2 范数；六关节速度差为 RMS，单位均为 rad/s。", "",
             "| 时间 ms | 原控制：角速度差 | 相同力矩：角速度差 | 原控制：关节速度差 | 相同力矩：关节速度差 |",
             "|---:|---:|---:|---:|---:|"]
    for ms in (5, 10, 15, 20):
        a, b = ag["formal", ms], ag["direct_shared", ms]
        lines.append(f"| {ms} | {a['angular_error_l2_rad_s']:.6f} | {b['angular_error_l2_rad_s']:.6f} | {a['joint_error_rms_rad_s']:.6f} | {b['joint_error_rms_rad_s']:.6f} |")
    lines += ["", "相同力矩时的响应幅度也发生变化，因此不能仅用绝对误差变化推断根因占比。相对误差定义为每个条件的速度差，除以两边速度幅度的算术平均，再对八个条件平均；不是平均绝对误差除以平均幅度。", "",
              "| 时间 ms | 原控制：相对角速度差 | 相同力矩：相对角速度差 | 原控制：相对关节速度差 | 相同力矩：相对关节速度差 |",
              "|---:|---:|---:|---:|---:|"]
    for ms in (5, 10, 15, 20):
        a, b = ag["formal", ms], ag["direct_shared", ms]
        lines.append(f"| {ms} | {a['relative_angular_error']:.2%} | {b['relative_angular_error']:.2%} | {a['relative_joint_error']:.2%} | {b['relative_joint_error']:.2%} |")
    a, b = ag["formal", 20], ag["direct_shared", 20]
    lines += ["", f"20 毫秒时，相同恒定力矩下的机身角速度幅度均值为 {b['angular_response_mean_norm_rad_s']:.3f} rad/s，原控制为 {a['angular_response_mean_norm_rad_s']:.3f} rad/s。恒定力矩造成更强的响应，所以绝对角速度差增大不能直接解释成更差的电机匹配或策略表现。",
              "", "个别条件的变化不同：初态对齐后，在 5/10/15/20 毫秒，角速度差小于原控制对照的条件数分别为 6/8、5/8、1/8、2/8。表格均值不表示每个输入都改善。"]
    zeros = [ag["direct_zero", ms] for ms in (5, 10, 15, 20)]
    lines += ["", f"零力矩对照在这四个时刻的平均角速度差为 {min(r['angular_error_l2_rad_s'] for r in zeros):.6f}–{max(r['angular_error_l2_rad_s'] for r in zeros):.6f} rad/s，平均关节速度差为 {min(r['joint_error_rms_rad_s'] for r in zeros):.6f}–{max(r['joint_error_rms_rad_s'] for r in zeros):.6f} rad/s。它说明无主动驱动时残留启动运动很小；不能证明受力后的动力学等价。", "",
              "## 初态对齐补测", "",
              "原来的名义初态并非跨引擎逐位一致：被动关节最大角度差 6.285 微弧度，超过本次原定 1 微弧度核验阈值。没有放宽阈值来宣称相等，而是补跑 MuJoCo 的相同力矩、零力矩两组：在 reset 后、首个物理步前，按关节名称写入 Isaac 实测初始位置／速度，root 高度与归一化姿态也对齐。每组仍来自新进程，之后不改状态。绝对 x/y 保留各自场景原点；无地面、均匀重力下它们是平移等价的自由空间条件。", "",
              "补测中，26 个关节初始位置／速度、root 高度和归一化姿态的比较误差全部为零。结果如下：", "",
              "| 时间 ms | 初态对齐后：角速度差 rad/s | 初态对齐后：关节速度差 rad/s | 初态对齐对 MuJoCo 角速度的影响 rad/s |",
              "|---:|---:|---:|---:|"]
    for ms in (5, 10, 15, 20):
        b = matched["direct_shared", ms]
        lines.append(f"| {ms} | {b['angular_error_l2_rad_s']:.6f} | {b['joint_error_rms_rad_s']:.6f} | {b['mujoco_initial_alignment_angular_change_l2_rad_s']:.6f} |")
    lines += ["", "对齐初态后，带力矩时的差异基本不变。这排除了这次发现的几微弧度初值差作为残留运动差异的主要解释；没有证明两个求解器的约束内部状态相同。", "",
              "## 验证", "",
              "- 八个最终引擎／模式组合来自独立新进程，全部 exit=0：六组主对照与两组初态对齐补测。含观察字段完善后的重跑，共执行 14 次物理进程；中间结果保留，不进入最终表格。每组含八个动作条件。",
              "- 两边原控制方式的四项物理状态在 0/5/10/15/20 毫秒与旧无地面记录完全一致。新增观察没有改变对照轨迹。",
              "- 六组主对照中，在各引擎内部，原控制与直接力矩的初始 root pose/velocity、全部 joint position/velocity 完全相同。初态对齐补测另按上一节检查跨引擎匹配。",
              f"- 跨引擎全部同名关节初始位置最大差 {result['cross_engine_initial_state']['all_named_joint_position_max_error_rad']:.3g} rad，root 高度差 {result['cross_engine_initial_state']['root_height_max_error_m']:.3g} m；初始速度均为零。绝对 x/y 是有意不同的场景原点，不作相等判定。",
              "- Isaac 编译驱动和缓存只改变六个主动关节的 stiffness/damping；20 个被动关节及 armature、限位保持不变。Isaac 四个闭合关节、MuJoCo 八条 connect 约束全部保留。",
              "- Isaac 记录实际传给 PhysX 外加力 setter 的参数，MuJoCo 记录每一步 motor force 和对应 generalized actuator force；直接力矩条件均与共同输入完全相同。这里验证的是外加力矩通路，不是测量原 PhysX 隐式驱动力矩。",
              "- 无地面接触力、无 episode reset、无采样到的主动关节 45 rad/s 速度上限越界。",
              f"- MuJoCo 840 个最终快照的机身角速度转换与 fresh-shadow body Jacobian 最大差 {result['mujoco_frame_vs_fresh_jacobian_max_error_rad_s']:.3g} rad/s；40 个无观察器重放的 qpos/qvel 最大差均为零。",
              f"- {result['protected_files_verified']} 个正式源码、配置、USD／模型、架构、缓存和策略文件 SHA256 未变。具体输入、进程、模型副本和结果哈希见 analysis.json。", "",
              "## 解释边界与下一步", "",
              "本次记录到了主动 PD 关闭后、同一恒定外加力矩下的跨引擎响应差异。原控制与恒定力矩的输入波形、反馈都不同，因此不能从误差增减算出电机控制贡献比例，也不能据此排除驱动实现对原控制工况的影响。", "",
              "本测试保留了各自的被动阻尼实现、闭合约束求解和正式物理步长。这些与质量／惯量仍混在一起；当前不能指定其中一个为根因。若继续定位，先复用旧固定底座／直接力矩记录，并与本次自由底座证据比较，选择一个仍未解释的关节响应做针对性对照。不要仅为让某个 checkpoint 站住而调整参数。", "",
              "这不是完整 sim2sim 策略验收，不能据此决定再训练，也没有试验 t-5..t-1 history 或 AdaBoot。", "",
              "图见 response.png；每个条件的向量、六关节分项误差和原始证据路径见 analysis.json。子 agent 复核记录另存 review.md。", ""]
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_response(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    metrics = (("angular_error_l2_rad_s", "Base angular-velocity discrepancy (rad/s)"),
               ("joint_error_rms_rad_s", "Six-joint velocity RMS discrepancy (rad/s)"),
               ("angular_response_mean_norm_rad_s", "Mean base angular-velocity magnitude (rad/s)"),
               ("joint_response_mean_rms_rad_s", "Mean six-joint velocity RMS (rad/s)"))
    for ax, (metric, label) in zip(axes.flat, metrics):
        for mode, color, name in zip(MODES, ("#2864b4", "#d97922", "#45974b"), ("Original controllers", "Equal constant torque", "Zero torque")):
            rows = [r for r in result["aggregate"] if r["mode"] == mode]
            ax.plot([r["time_ms"] for r in rows], [r[metric] for r in rows], marker="o", lw=1.7, color=color, label=name)
        ax.set(xlabel="Time after first input (ms)", ylabel=label, xticks=[5, 10, 15, 20])
        ax.grid(alpha=0.2)
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Free base, gravity on, closure on, no ground — eight input conditions", fontsize=12)
    fig.savefig(ROOT / "response.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
