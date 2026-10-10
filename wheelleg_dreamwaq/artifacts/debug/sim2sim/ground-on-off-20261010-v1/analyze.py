"""Compare the matched 20 ms pair, without policy feedback or production changes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from wheelleg_dreamwaq.schemas.frames import quat_rotate_inverse_wxyz, transform_usd_vector_to_control

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
OLD = ROOT.parent / "initial-ground-contact-20261010-v1"
CASES = ("zero_action", "shared_first_action_hold")
NAMES = ("nominal_stand", "low_stand", "high_stand", "forward", "reverse", "turn_left", "turn_right", "combined")
JOINTS = ("jIJ", "jIO", "jAB", "jAG", "jwheel_left", "jwheel_right")
SIGNS = np.array((1., 1., 1., 1., 1., -1.))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def sample(evidence, case, index, ms, engine):
    rows = [s for s in evidence["samples"] if s["case"] == case and abs(s["elapsed_s"] - ms / 1000) < 1e-10
            and (engine == "isaac" or s["environment_index"] == index)]
    assert len(rows) == 1, (engine, case, index, ms)
    return rows[0]


def main():
    evidence = {(engine, mode, case): read(ROOT / f"{engine}-{case}-ground-{mode}/evidence.json")
                for engine in ("isaac", "mujoco") for mode in ("on", "off") for case in CASES}
    manifest = read(PROJECT / "sim2sim/mujoco/model_manifest.json")
    rotation = np.asarray(manifest["r_control_from_mujoco"])
    inames = evidence["isaac", "on", CASES[0]]["joint_names"]
    mnames = manifest["joint_order"]
    iids, mids = [inames.index(n) for n in JOINTS], [mnames.index(n) for n in JOINTS]
    actions = read(OLD / "isaac-off/actions.json")
    for mode in ("on", "off"):
        for case in CASES:
            assert read(ROOT / f"isaac-{case}-ground-{mode}/actions.json") == actions
    protected = {}
    for (engine, mode, case), data in evidence.items():
        assert data["debug_only"] and not data["formal_ranking_eligible"] and data["sources_unchanged"]
        assert data["ground_enabled"] == (mode == "on") and data["control_ticks"] == 1
        assert data["gravity_m_s2"] == [0., 0., -9.81] and data["reset_physics_time_advanced_s"] == 0.
        assert data["test_case"] == case and data["probe_sha256"] == digest(ROOT / f"probe_{engine}.py")
        assert len(data["samples"]) == (6 if engine == "isaac" else 168)
        for path, expected in data["protected_source_sha256"].items():
            assert digest(Path(path)) == expected, path
            assert protected.get(path, expected) == expected
            protected[path] = expected
        if engine == "mujoco":
            assert max(data["observer_replay_max_errors"]) == 0.
        if mode == "off":
            field = "normal_force_world_n" if engine == "isaac" else "normal_force_n"
            assert all(np.max(np.abs(s[field])) == 0. for s in data["samples"] if s["physics_step"] > 0)

    reset_errors = {}
    for engine in ("isaac", "mujoco"):
        fields = ("root_pose_xyzw", "root_velocity_world", "joint_position", "joint_velocity") if engine == "isaac" else (
            "root_qpos", "root_qvel", "joint_position", "joint_velocity")
        reset_errors[engine] = {field: max(float(np.max(np.abs(np.asarray(sample(evidence[engine, "on", c], c, i, 0, engine)[field]) -
                                                   np.asarray(sample(evidence[engine, "off", c], c, i, 0, engine)[field]))))
                                         for c in CASES for i in range(8)) for field in fields}
        assert max(reset_errors[engine].values()) == 0.

    def velocities(engine, mode, case, index, ms):
        s = sample(evidence[engine, mode, case], case, index, ms, engine)
        if engine == "isaac":
            q = torch.tensor(np.asarray(s["root_pose_xyzw"])[index, [6, 3, 4, 5]], dtype=torch.float64)
            world = torch.tensor(s["root_velocity_world"][index][3:], dtype=torch.float64)
            angular = transform_usd_vector_to_control(quat_rotate_inverse_wxyz(q, world)).numpy()
            joint = np.asarray(s["joint_velocity"])[index, iids] * SIGNS
        else:
            # MuJoCo free-joint rotational qvel is in its child body frame.
            # A separate fresh-shadow Jacobian check verifies this against the production collector.
            angular = rotation @ np.asarray(s["root_qvel"])[3:]
            joint = np.asarray(s["joint_velocity"])[mids] * SIGNS
        return angular, joint

    rows, aggregate = [], []
    for case in CASES:
        for ms in (5, 10, 15, 20):
            local = []
            for index, name in enumerate(NAMES):
                row = {"case": case, "environment_index": index, "scenario": name, "time_ms": ms}
                for mode in ("on", "off"):
                    ia, ij = velocities("isaac", mode, case, index, ms)
                    ma, mj = velocities("mujoco", mode, case, index, ms)
                    row[mode] = {"angular_error_l2_rad_s": float(np.linalg.norm(ia - ma)),
                                 "joint_error_rms_rad_s": float(np.sqrt(np.mean((ij - mj) ** 2))),
                                 "isaac_angular_control_rad_s": ia.tolist(), "mujoco_angular_control_rad_s": ma.tolist(),
                                 "isaac_joint_control_rad_s": ij.tolist(), "mujoco_joint_control_rad_s": mj.tolist()}
                local.append(row)
                rows.append(row)
            item = {"case": case, "time_ms": ms}
            for metric in ("angular_error_l2_rad_s", "joint_error_rms_rad_s"):
                on = float(np.mean([r["on"][metric] for r in local]))
                off = float(np.mean([r["off"][metric] for r in local]))
                item[metric] = {"ground_on_mean": on, "ground_off_mean": off, "off_over_on": off / on,
                                "reduction_percent": (1. - off / on) * 100.}
            aggregate.append(item)
    reproduction = {}
    for engine in ("isaac", "mujoco"):
        old = read(OLD / ("isaac-on" if engine == "isaac" else "mujoco") / "evidence.json")
        fields = ("root_velocity_world", "joint_velocity") if engine == "isaac" else ("root_qvel", "joint_velocity")
        reproduction[engine] = {field: max(float(np.max(np.abs(np.asarray(sample(old, c, i, ms, engine)[field]) -
                                                  np.asarray(sample(evidence[engine, "on", c], c, i, ms, engine)[field]))))
                                         for c in CASES for i in range(8) for ms in (5, 10, 15, 20)) for field in fields}
    frame_check = read(ROOT / "frame-verification.json")
    assert frame_check["passed"] and frame_check["angular_body_qvel_vs_fresh_production_jacobian_max_error_rad_s"] < 1e-12
    for path, expected in frame_check["input_sha256"].items():
        assert digest(Path(path)) == expected
    executions = json.loads((ROOT / "isolated-execution.json").read_text(encoding="utf-8-sig"))
    assert len(executions) == 8 and all(r["exit_code"] == 0 for r in executions)
    inputs = [ROOT / f"{engine}-{case}-ground-{mode}/evidence.json" for engine, mode, case in evidence]
    inputs += [ROOT / f"isaac-{case}-ground-{mode}/actions.json" for case in CASES for mode in ("on", "off")]
    inputs += [ROOT / "isolated-execution.json", ROOT / "frame-verification.json"]
    inputs += [ROOT / "probe_isaac.py", ROOT / "probe_mujoco.py", ROOT / "analyze.py", OLD / "isaac-off/actions.json"]
    inputs += [ROOT / "diagnostic-plan.md"] + list(ROOT.glob("mujoco-*-ground-off/model/*.xml*"))
    inputs += list(ROOT.glob("mujoco-*-ground-off/diagnostic-model-identity.json"))
    result = {"schema_version": "GroundOnOffAnalysisV1", "debug_only": True, "formal_ranking_eligible": False,
              "reset_errors": reset_errors, "protected_sources_verified_unchanged": True,
              "first_episode_cases_in_independent_processes": True, "frame_check": frame_check,
              "conclusion": "Ground contact strongly changes the discrepancy in zero-action startup. Under the shared first action, removing it reduces early discrepancies but does not remove non-contact motion differences; angular error is larger without ground at 20ms. No single-parameter root cause or training decision follows.",
              "old_ground_on_reproduction_max_errors": reproduction, "aggregate": aggregate, "rows": rows,
              "input_sha256": {str(p.resolve()): digest(p) for p in inputs}}
    (ROOT / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    report = ["# 前 20 毫秒地面接触开关诊断", "",
              "结论：地面接触是启动运动差异的重要来源，但不能解释全部差异。零动作时，无地面的两边几乎一致；加入相同首动作后，无地面仍有明显差异，并且角速度误差到 20 ms 反而比有地面大。因此不应只凭这次结果修改地面参数或再次训练。", "",
              "使用相同八环境初态、run-02 首动作、重力、驱动、闭环及正式物理步长。零动作与共享首动作分别保持一个控制周期，不接入后续策略反馈。", "",
              "表中为八个条件的平均误差；角速度是 ControlFrameV1 三维向量差的 L2 范数，关节速度是六个受控关节的 RMS 差。两者单位均为 rad/s。", "",
              "| 输入 | 时间 ms | 角速度差：有地面 | 无地面 | 缩小 % | 关节速度差：有地面 | 无地面 | 缩小 % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        a, j = row["angular_error_l2_rad_s"], row["joint_error_rms_rad_s"]
        report.append(f"| {'零动作' if row['case'] == CASES[0] else '共享首动作'} | {row['time_ms']} | {a['ground_on_mean']:.6f} | {a['ground_off_mean']:.6f} | {a['reduction_percent']:.1f} | {j['ground_on_mean']:.6f} | {j['ground_off_mean']:.6f} | {j['reduction_percent']:.1f} |")
    report += ["", "零动作角速度差在四个采样点缩小约 99.3–99.7%，关节速度差缩小约 98.5–99.8%。共享首动作的角速度差在前 5/10/15 ms 缩小约 44–61%，关节速度差在四个点缩小约 56–74%；但 20 ms 角速度平均差增大约 20%。接触作用与驱动、闭环及积分相互耦合，误差并非可线性相加的来源占比。八环境均值也不能解释为所有条件都改善；每环境结果记录于 analysis.json。", "",
               "后续应优先解释轮地接触启动响应，同时用已有驱动与闭环对照记录分析带动作时残留的非接触差异。本次没有证据指定某个参数为根因，也没有跑完整 sim2sim 策略评估，不能从这 20 ms 测试判断重新训练是否有效。"]
    report += ["", "## 验证与范围", "", "最终八次独立仿真进程均 exit=0。零动作和共享首动作分别从新进程初始化，排除了先运行另一条件造成的求解器状态影响。有地面／无地面的初始 root pose、root velocity、全部关节位置与速度，在每个仿真器内逐项完全相同。Isaac 复用了相同首动作文件；MuJoCo 接收同一文件。无地面的所有 post-step 支撑力为零，重力保持 -9.81 m/s²。MuJoCo 仅移除 floor 及其两条显式接触对；非接触的 compiled dynamics 与正式模型完全相同。四种 MuJoCo 输入／地面模式各 8 条无观察器重放的 qpos/qvel 误差均为零。", "",
              "正式源码、USD 资产、架构文档、缓存、actor 与 MuJoCo 正式模型的 SHA256 逐项核对未变。这是 debug-only 对照，不能作为旧 RootCauseSuite 的正式通过结论或完整策略评估。", "",
              "中间结果先沿用两条件顺序运行的协议，发现与旧 100 ms 探针的 Isaac 首动作结果不完全相同。因此最终只使用按输入隔离新进程的记录；先前中间记录保留用于审计，不进入上表。与旧探针的差见 analysis.json 中 old_ground_on_reproduction_max_errors；该差异发生在不使用网络反馈的物理测试中，不能据此认定神经网络 reset 再次出错。", "",
              "完整每环境向量、逐时刻数值及输入哈希见 [analysis.json](analysis.json)。进一步结论必须区分不同输入和不同时间，不能把误差缩小百分比当作根因占比，也不能仅凭此对照决定重新训练。"]
    (ROOT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "reset_errors": reset_errors, "reproduction": reproduction}, indent=2))


if __name__ == "__main__":
    main()
