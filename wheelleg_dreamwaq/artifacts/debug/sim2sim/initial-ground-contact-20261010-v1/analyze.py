from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main():
    off = json.loads((ROOT / "isaac-off/evidence.json").read_text())
    on = json.loads((ROOT / "isaac-on/evidence.json").read_text())
    mj = json.loads((ROOT / "mujoco/evidence.json").read_text())
    mj_neutral = json.loads((ROOT / "mujoco-observer-neutrality.json").read_text())
    assert len(off["samples"]) == len(on["samples"]) == 43
    assert len(mj["samples"]) == 1616
    neutrality = {}
    for key in ("root_pose_xyzw", "root_velocity_world", "joint_position", "joint_velocity", "wheel_collision_bottom_gap_m"):
        error = max(float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key]))))
                    for a, b in zip(off["samples"], on["samples"]))
        neutrality[key] = error
        assert error == 0.0
    assert mj_neutral["passed"]
    for evidence in (off, on, mj):
        assert evidence["sources_unchanged"]
        for path, expected in evidence["protected_source_sha256"].items():
            assert digest(Path(path)) == expected, path
    cache = torch.load(PROJECT / "artifacts/phase2_dreamwaq/training-suite-20261010-001203-reset-v2/isaac_evaluation/evaluation-reset-cache.pt",
                       map_location="cpu", weights_only=False)
    isaac_joint_names = cache["identity"]["joint_names"]
    model_manifest = json.loads((PROJECT / "sim2sim/mujoco/model_manifest.json").read_text())
    mj_joint_names = model_manifest["joint_order"]
    assert set(isaac_joint_names) == set(mj_joint_names)
    initial_i = next(s for s in on["samples"] if s["case"] == "zero_action" and s["physics_step"] == 0)
    initial_m = next(s for s in mj["samples"] if s["case"] == "zero_action" and s["environment_index"] == 0 and s["physics_step"] == 0)
    reorder = [isaac_joint_names.index(name) for name in mj_joint_names]
    joint_error = float(abs(np.asarray(initial_i["joint_position"])[0, reorder] - np.asarray(initial_m["joint_position"])).max())
    initial = {"isaac_root_link_z_m": initial_i["root_pose_xyzw"][0][2],
               "mujoco_root_link_z_m": initial_m["root_qpos"][2],
               "isaac_wheel_gap_m": initial_i["wheel_collision_bottom_gap_m"][0],
               "mujoco_wheel_gap_m": initial_m["wheel_collision_bottom_gap_m"],
               "isaac_max_abs_root_velocity": float(abs(np.asarray(initial_i["root_velocity_world"])).max()),
               "mujoco_max_abs_root_velocity": float(abs(np.asarray(initial_m["root_qvel"])).max()),
               "all_26_joint_initial_max_difference_rad": joint_error,
               "isaac_all_eight_env_gap_range_m": [np.min(initial_i["wheel_collision_bottom_gap_m"], axis=0).tolist(),
                                                    np.max(initial_i["wheel_collision_bottom_gap_m"], axis=0).tolist()]}
    cases = {}
    for case in ("zero_action", "shared_first_action_hold"):
        irows = [s for s in on["samples"] if s["case"] == case and s["physics_step"] > 0]
        mrows = [s for s in mj["samples"] if s["case"] == case and s["environment_index"] == 0 and s["physics_step"] > 0]
        first_i = [next((s["elapsed_s"] for s in irows if np.linalg.norm(s["normal_force_world_n"][0][side]) > 1e-6), None) for side in (0, 1)]
        first_m = [next((s["elapsed_s"] for s in mrows if s["normal_force_n"][side] > 1e-6), None) for side in (0, 1)]
        impulses = {}
        for end in (0.005, 0.020, 0.100):
            impulse_i = np.sum([np.asarray(s["normal_force_world_n"])[0, :, 2] * on["physics_dt_s"]
                                for s in irows if s["elapsed_s"] <= end + 1e-12], axis=0)
            impulse_m = np.sum([np.asarray(s["normal_force_n"]) * mj["physics_dt_s"]
                                for s in mrows if s["elapsed_s"] <= end + 1e-12], axis=0)
            impulses[str(end)] = {"isaac_vertical_impulse_ns": impulse_i.tolist(), "mujoco_normal_impulse_ns": impulse_m.tolist()}
        cases[case] = {"isaac_first_nonzero_support_sample_s": first_i,
                       "mujoco_first_nonzero_support_sample_s": first_m, "impulses": impulses}
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in (
        ROOT / "isaac-off/evidence.json", ROOT / "isaac-on/evidence.json", ROOT / "mujoco/evidence.json",
        ROOT / "isaac-off/actions.json", ROOT / "mujoco-observer-neutrality.json",
        ROOT / "probe_isaac.py", ROOT / "probe_mujoco.py")}
    result = {"schema_version": "InitialGroundContactAnalysisV1", "debug_only": True,
              "formal_ranking_eligible": False, "initial": initial, "cases": cases,
              "isaac_contact_observer_max_errors": neutrality, "mujoco_observer_exact_replay_cases": 16,
              "protected_sources_verified_unchanged": True, "input_sha256": inputs,
              "conclusion": "Both formal startup paths reset nearly onto the ground at zero velocity. No gravity-driven pre-policy landing wait. Small collider gaps and startup contact response differ; no root-cause attribution is made.",
              "limitations": ["Isaac mesh support excludes convex cooking tolerance and contact/rest offsets.",
                              "5ms Isaac and 1ms MuJoCo sampling do not establish an exact 3ms contact delay.",
                              "Reset-time PhysX contact buffers are not a newly solved episode; only post-step force is used.",
                              "The shared first action is held for 100ms without policy feedback; this is not an eight-scenario policy evaluation."]}
    (ROOT / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    ig = [value * 1000 for value in initial["isaac_wheel_gap_m"]]
    mg = [value * 1000 for value in initial["mujoco_wheel_gap_m"]]
    impulse_i = sum(cases["zero_action"]["impulses"]["0.005"]["isaac_vertical_impulse_ns"])
    impulse_m = sum(cases["zero_action"]["impulses"]["0.005"]["mujoco_normal_impulse_ns"])
    text = f"""# 初态与轮地接触定向诊断（2026-10-10）

结论：未发现“一边从明显高度下落、另一边已等待落稳”的启动差异。两边正式 reset 都把机身写到约 0.20003 m 的 root-link 高度，速度为零，轮子几乎贴地；两边均没有重力下落等待。但是轮子碰撞体的微小间隙和开始支撑的响应并不相同，尚不能把 sim2sim 失败归因于某一项。

## 实际范围

使用当前修复后的正式环境源码、正式八环境评估缓存（seed 20261007），以及新四 seed 训练 suite 的 run-02 actor。Isaac 继承正式 reset/step，经正式 RSL wrapper 完成首次 reset；只在独立配置中增加接触观察器。MuJoCo 使用未修改的 WheelLegMujocoRuntime。两端都运行零动作及同一条首动作保持，分别 100 ms，不使用后续策略反馈。

## 初始读回（nominal_stand，env 0）

| 量 | Isaac | MuJoCo |
| --- | ---: | ---: |
| root-link 初始高度 | {initial['isaac_root_link_z_m']:.9f} m | {initial['mujoco_root_link_z_m']:.9f} m |
| 初始 root 速度最大绝对值 | {initial['isaac_max_abs_root_velocity']:.1f} | {initial['mujoco_max_abs_root_velocity']:.1f} |
| 左轮碰撞体最低点距地面 | {ig[0]:.6f} mm | {mg[0]:.6f} mm |
| 右轮碰撞体最低点距地面 | {ig[1]:.6f} mm | {mg[1]:.6f} mm |
| reset 期间物理时间推进 | {on['reset_physics_time_advanced_s']:.1f} s | {mj['reset_physics_time_advanced_s']:.1f} s |

两侧地面顶面都是 z=0。按名称对齐全部 26 个关节，初始角度最大差 {joint_error:.9g} rad。Isaac 是 authored mesh 的 convexHull；MuJoCo 是半径 0.0625 m 的 sphere proxy。Isaac 的表中间隙由实际 PhysX link pose 和网格顶点求出，不包含 hull cooking 的近似误差及 contact/rest offset；因此不能当作底层求解器的精确碰撞距离。

## 初始化的抬高过程

Isaac cache 生成时会关闭重力，将 root 额外抬高 0.75 m，整理闭链。加载缓存时在同一抬高位置做 forward 验证。此次实测构造后 root 高度为 {off['samples'][0]['root_pose_xyzw'][0][2]:.9f} m；RSL wrapper 调用正式 reset 后，root 高度立刻写回 {initial['isaac_root_link_z_m']:.9f} m，中间没有物理 step。这是写入初态，不是让机器人从 0.75 m 高处自由落下。MuJoCo reset 从 keyframe 写入初态，再 mj_forward，同样不等待落地。

## 支撑力开始出现的样本

零动作：MuJoCo 两轮在 2 ms 样本首次报告非零支撑力；Isaac 两轮在首个 5 ms 样本均报告非零支撑力（约 17.16、12.17 N）。因为两端采样间隔分别是 1 ms 和 5 ms，不能由此声称 Isaac 实际接触晚了 3 ms。

同一首动作：前 20 ms 两边均主要由左轮支撑，右轮没有支撑力。右轮首次非零力样本分别是 MuJoCo 42 ms、Isaac 50 ms。它们是该固定动作下的诊断结果，不能外推到所有 checkpoint 或完整闭环评估。

零动作前 5 ms 的两轮合计竖直支撑冲量：Isaac {impulse_i:.9f} N·s，MuJoCo {impulse_m:.9f} N·s。这证明启动时接触响应存在数值差异，但试验同时保留正式 drive、闭链与积分方式，不能仅凭该差异认定是接触模型、初始微小间隙或闭链造成。

reset 时尚未执行新 episode 的物理求解，PhysX raw contact buffer 可能保留上一求解步的读数。因此本报告仅使用实际物理 step 之后的支撑力，不把 reset 时的 force 值作为初始接触证据。

## 采集与文件验证

Isaac 两个独立 Kit 进程：接触观察器关闭/开启，共 43 个快照、每个快照 8 个环境。root pose、root velocity、26 关节位置/速度、轮子最低点的最大差全部为 0，观察器没有改变本次运动。

MuJoCo 16 个 case、1616 个快照。去掉采集回调后重放相同正式 runtime，全部 16 个 case 的最终 qpos/qvel 与采集运行逐项完全相等；几何刷新只发生在 shadow MjData，未在正式 data 上额外调用 mj_forward。

保护文件在运行前后及分析时逐项 SHA256 核对相同，包括正式源码、USD 资产、架构文档、评估缓存、actor 和 MuJoCo 模型/runtime。未改生产模型或参数，也未训练。此报告是 debug-only 证据，不是旧 RootCauseSuite 的正式 verdict，不改写旧 suite 身份或运行记录。

原始证据：[Isaac off](isaac-off/evidence.json)、[Isaac on](isaac-on/evidence.json)、[MuJoCo](mujoco/evidence.json)、[观察器重放验证](mujoco-observer-neutrality.json)、[机器可读分析与哈希](analysis.json)。

建议下一步用已有记录比较前 5/20 ms 的支撑冲量、关节运动和机身角速度，确定运动差异与接触响应差异的先后关系；本次不修改初态高度、不增加落地等待、不关闭闭链。
"""
    (ROOT / "report.md").write_text(text, encoding="utf-8")
    print(json.dumps({"initial": initial, "observer_errors": neutrality,
                      "zero_action_first_5ms_total_impulse_ns": [impulse_i, impulse_m],
                      "protected_sources_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
