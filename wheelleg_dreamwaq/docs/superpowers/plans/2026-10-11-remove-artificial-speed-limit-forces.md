# 删除人工限速作用实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 按用户明确授权，删除 Isaac 和 MuJoCo 中刚体角速度、刚体线速度及关节转速的人工限速作用。

**Architecture:** Isaac 显式使用有限 float32 最大值作为无操作性限速的引擎哨兵，不能用 `None` 或删除参数来继承 USD/PhysX 的默认上限。MuJoCo 删除 AngularVelocityBiasLimiter 及超速清零电机力矩分支。PhysicsV5、UnrestrictedVelocityPolicyV1 和 MujocoEvaluationContractV3 明确区分新动力学和旧训练工件。

**Tech Stack:** Isaac Lab 2.3.2 / Isaac Sim 5.1、RSL-RL 3.1.2、MuJoCo 3.14、pytest。

本次不启动训练。用户随后明确要求改完调用一个 agent 只读复核；仅使用该 reviewer，禁止它再派生 agent。不改源 USD、MuJoCo XML、质量/惯量、PD、真实输出力矩上限、接触/摩擦、闭环、被动阻尼、动作/命令/奖励/观测/history/CENet/AdaBoot、随机化采样及 reset 时序。速度失败阈值属于终止/诊断，不施加力，保留。

## 1. 留存和规格

- [x] 保存将修改/删除文件的原版本和受保护文件 SHA256，位于 `artifacts/debug/sim2sim/remove-artificial-speed-limits-20261011-v1/`。
- [x] 正式架构更新为 v0.25；§29/ADR-068 明确保留为历史，§30/ADR-069 规定当前 PhysicsV5。
- [x] 共用的速度政策数据为：

```python
{
    "version": "UnrestrictedVelocityPolicyV1",
    "rigid_body_linear_speed_limit": "disabled",
    "rigid_body_angular_speed_limit": "disabled",
    "joint_speed_limit": "disabled",
    "external_speed_limit_force": False,
    "external_speed_limit_torque": False,
    "runtime_velocity_write": False,
    "isaac_velocity_sentinel": float.fromhex("0x1.fffffep+127"),
}
```

这是有限、可准确写入 float32 的引擎哨兵，不是新的电机额定转速，不使用 JSON Infinity 或 NaN。

## 2. 回归用例先行

- [x] MuJoCo 在关节已超过旧 45 rad/s 时仍输出由原 PD 与 effort clipping 决定的力矩，不再强制清零继续加速的分量。
- [x] runtime 在超过旧刚体线/角速度上限时不注入外力；实际 20 个子步轨迹应精确等于相同电机控制下直接 `mj_step` 的参考轨迹。
- [x] 外部实验合法提供的外力/力矩保持不变；reset/history/20 ms 时序回归继续通过。
- [x] 新政策/评估版本不接受旧 PhysicsV4 与旧带制动适配的评估；旧工件不改写。

运行：

```powershell
& .\sim2sim\mujoco\.venv\Scripts\python.exe -m pytest sim2sim/mujoco/tests/test_mujoco_adapters.py -q
```

新增“超速仍按 PD 输出”和“不注入力矩”用例修改前应失败、修改后应通过。

## 3. Isaac 和训练物理契约

- [x] `source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/physics.py`：PhysicsV5、哨兵和政策 payload；dt/decimation/solver iterations 不变。
- [x] `assets/wheelleg.py`：三类速度上限均显式使用哨兵；其余参数逐项不变。
- [x] `schemas/manifest.py` 和 `schemas/dreamwaq_manifest.py`：在任务 physics 字段加入政策，physics schema 进入 hash；既有 checkpoint 和 reset-cache hash 校验保持严格。
- [x] `scripts/export_ppo_actor.py` 与 `scripts/export_dreamwaq_actor.py`：导出当前 PhysicsV5 政策；旧 PhysicsV4 checkpoint 不得重标为 V5。

核心配置：

```python
max_linear_velocity=UNRESTRICTED_SIM_VELOCITY,
max_angular_velocity=UNRESTRICTED_SIM_VELOCITY,
velocity_limit_sim=UNRESTRICTED_SIM_VELOCITY,
```

## 4. MuJoCo

- [x] 删除 `wheelleg_mujoco/angular_limit.py` 和专测旧适配的 `tests/test_angular_limit.py`，原件留存在诊断 source-before。
- [x] `wheelleg_mujoco/runner.py` 删除 limiter 的 import、构造、reset、apply；不读写速度来实现替代限速。
- [x] `wheelleg_mujoco/control.py` 删除速度分支，只保留原 PD 和 effort clipping：

```python
torque[:4] = leg_kp * (target_q - q) - leg_kd * qd
torque[4:6] = wheel_kd * (target_qd - qd)
return np.clip(torque, -effort_limits, effort_limits)
```

- [x] `wheelleg_mujoco/physics.py` 提供自包含政策数据，不依赖 Isaac；cross-engine test 对比其内容和主工程 payload。
- [x] `wheelleg_mujoco/contract.py` 删除不再消费的关节速度字段，严格验证 PhysicsV5 与无制动政策。
- [x] `wheelleg_mujoco/versions.py` 升级 action adapter 为 V2；`model_manifest.json` 仅更新 action adapter 版本和实现 hash，不重建 XML/几何。
- [x] `wheelleg_mujoco/evaluation.py` 使用 MujocoEvaluationContractV3，显式记录无制动政策；排名拒绝旧契约或制动标记。
- [x] `scripts/evaluate_mujoco.py` 保留原 80 rad/s joint-speed 失败阈值，解除其对已删除物理限速字段的依赖。
- [x] 更新 MuJoCo README 和既有单元测试夹具/评估验收，不修改历史诊断结果。
- [x] 复核发现的 5 个 live debug 文件补抓修改前基线，并同步删除旧字段读取和 target-torque 超速清零；trace 中旧 guard 字段明确为 0，不更新旧冻结注册。

## 5. 验收

- [x] 完整纯 Python/算法单元测试：

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit -q
& .\sim2sim\mujoco\.venv\Scripts\python.exe -m pytest sim2sim/mujoco/tests -q
```

- [x] 新增 Isaac headless 诊断脚本与 integration test，读取实际 27 个刚体 USD 属性和 26 个 DOF 限速；确认哨兵到达引擎，原 effort/damping/armature 不变。
- [x] 无接触、无重力的单刚体在初始线速度 200 m/s、角速度 10 rad/s 时短步保持速度，以旧阈值为负面对照，不靠神经网络。
- [x] 8 env 随机动作短测确认观测维度、finite、闭环、reset 正常；不做 PPO 更新。
- [x] 旧 checkpoint/缓存与新 contract 的拒绝测试通过；原四组 TorchScript golden vectors 仍与未修改 actor 一致。
- [x] 使用随机初始化、零优化更新的独立测试包核验实际 DreamWaQ `_manifest_payload`、TorchScript 打包、正式 loader 及当前 debug trace/evaluation；不得伪造已训练性能工件。
- [x] 受保护参数/资产 SHA256 不变；保存测试命令、结果、实际速度政策、修改量和最终架构 hash。

当前工作区不使用 git，使用 source-before 和 SHA256 审计代替提交。用户已明确授权执行，本计划完成后直接在本对话顺序实施，无需重复审批。

## 6. 独立只读复核

- [x] 唯一获用户授权的 reviewer 完成最终复查，确认已修正 debug 旧接口遗漏和孤立测试 import 问题，无未处理 P0/P1。
