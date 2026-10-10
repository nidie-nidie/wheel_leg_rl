# WheelLeg Isaac Sim / MuJoCo 首处分歧诊断报告

## 结论

本轮正式数据采集和对比已完成。三个正式门槛均通过：collector 与正式环境等价、同一 float32 输入的 CPU Actor 输出完全一致、Isaac 原始轨迹与 fresh-reset action replay 等价。所有有效跨引擎场景的 reset 和身份契约也都通过。

首个材料级分歧稳定出现在 **control tick 0 的主动关节位置响应**。零动作实验进一步表明：两边初始关节状态一致，初始目标误差仅 `8.59e-09`，初始 PD 力矩最大误差仅 `1.56e-06 Nm`；但 5 ms 后机身角速度最大误差已达到 `0.13123 rad/s`，20 ms 后关节位置误差达到 `0.00105319 rad`。因此当前问题不在 Actor、归一化、动作顺序或初始 PD 计算，而在动作进入物理系统后的响应。

当前最强候选依次为：

1. 轮地接触几何、首次接触时刻和接触求解不同。
2. 闭链约束表达和约束柔度不同。
3. PhysX 隐式 drive 与 MuJoCo 显式 PD 力矩的积分方式不同，同时物理步长分别为 5 ms 和 1 ms。

现有证据还不能在这三项之间唯一归因，因此现在不应直接调奖励、Actor 或归一化，也不应直接用 PACE 数据做 actuator 拟合来掩盖模型差异。

## 正式门槛

- Collector 等价门槛：通过，report hash `BCD6A9036A76FCD26DF36353DE16B0E0F3D6229F8882BA5CDFF8C3B15FAF6A25`。
- Actor CPU identity：通过，同一输入输出最大误差 `0.0`；两套运行时存储输出最大误差 `3.57628e-07`。
- Isaac fresh-reset replay：通过，report hash `CFCA7CACD433AD1379CD0D13D8725699C353A96D2C0A4A1B0E291F2240C194C9`。

## 场景结果

| Scenario | Isaac | MuJoCo | First material divergence |
|---|---:|---:|---|
| `closed-loop-001` | 1 (none) | 1 (none) | tick 0: `active_joint_position_canonical_post_step_pre_reset` |
| `closed-loop-010` | 10 (none) | 10 (none) | tick 0: `active_joint_position_canonical_post_step_pre_reset` |
| `closed-loop-050` | 50 (none) | 50 (none) | tick 0: `active_joint_position_canonical_post_step_pre_reset` |
| `closed-loop-500-v2` | 500 (timeout) | 270 (tilt) | tick 0: `active_joint_position_canonical_post_step_pre_reset` |
| `zero-action-500` | 26 (tilt) | 24 (tilt) | tick 0: `active_joint_position_canonical_post_step_pre_reset` |
| `policy-replay-500-v2` | 500 (timeout) | 35 (tilt) | tick 0: `active_joint_position_canonical_post_step_pre_reset` |
| `channel-pulse-immediate-0` | 20 (none) | 20 (none) | tick 0: `active_joint_position_canonical_post_step_pre_reset` |
| `channel-pulse-immediate-1` | 20 (none) | 20 (none) | tick 0: `active_joint_position_canonical_post_step_pre_reset` |
| `channel-pulse-immediate-2` | 20 (none) | 20 (none) | tick 0: `active_joint_position_canonical_post_step_pre_reset` |
| `channel-pulse-immediate-3` | 20 (none) | 20 (none) | tick 0: `active_joint_position_canonical_post_step_pre_reset` |
| `channel-pulse-immediate-4` | 20 (none) | 20 (none) | tick 0: `active_joint_position_canonical_post_step_pre_reset` |
| `channel-pulse-immediate-5` | 20 (none) | 20 (none) | tick 0: `active_joint_position_canonical_post_step_pre_reset` |

关键寿命结果：闭环时 Isaac 跑满 500 tick，MuJoCo 在 270 tick 因 tilt 停止；把 Isaac 动作冻结后 replay 到 MuJoCo，MuJoCo 只运行 35 tick 即 tilt。这说明 MuJoCo 的动力学响应无法跟随 Isaac 的同一动作轨迹，不是单纯的闭环 Actor 放大。

零动作时 Isaac 和 MuJoCo 分别在 26、24 tick 因 tilt 停止，证明差异不依赖 Actor 输出。

## 第一个控制周期

闭环 tick 0：

- Actor 输入最大误差：`1.29652e-08`。
- Actor 输出最大误差：`3.57628e-07`。
- 控制目标最大误差：`9.83477e-06`。
- 20 ms 后主动关节位置最大误差：`0.00184403 rad`。
- 20 ms 后主动关节速度最大误差：`0.0967002 rad/s`。

零动作的初始主动关节 torque 基本为零且两边一致，但状态从第一个物理区间开始分叉：

| Time | q max error | qd max error | base linear velocity max error | base angular velocity max error |
|---:|---:|---:|---:|---:|
| 5 ms | 0.000164696 rad | 0.0946406 rad/s | 0.0153881 m/s | 0.13123 rad/s |
| 10 ms | 0.000703933 rad | 0.0977329 rad/s | 0.0309454 m/s | 0.23224 rad/s |
| 15 ms | 0.000891015 rad | 0.026472 rad/s | 0.0215816 m/s | 0.112647 rad/s |
| 20 ms | 0.00105319 rad | 0.0233938 rad/s | 0.0261821 m/s | 0.131497 rad/s |


## 单通道即时激励

原设计在 tick 25 才激励，但零动作姿态在约 tick 24 已倾倒，因此原六组通道实验无效。本轮新增的分析输入从 tick 0 开始激励，六组两边都完整运行 20 tick。每个响应都减去了同一引擎的零动作基线，避免把自然下落误认为通道响应。

| Channel | Joint | Direct response | Isaac peak | MuJoCo peak | Error at tick 19 |
|---:|---|---|---:|---:|---:|
| 0 | `jIJ` | joint_position_rad | 0.0375709 | 0.0376109 | 9.63243e-05 |
| 1 | `jIO` | joint_position_rad | 0.0365092 | 0.0378687 | 0.000391991 |
| 2 | `jAB` | joint_position_rad | 0.0358807 | 0.0376007 | 0.000291013 |
| 3 | `jAG` | joint_position_rad | 0.0360019 | 0.0378633 | 0.000918981 |
| 4 | `jwheel_left` | joint_velocity_rad_s | 2.0736 | 2.1169 | 0.0370828 |
| 5 | `jwheel_right` | joint_velocity_rad_s | 2.08841 | 2.11681 | 0.164293 |

四个腿关节的增量位置响应总体接近，tick 19 的跨引擎误差均低于 `0.001 rad`。两个轮通道的初始增量速度响应也接近，但右轮在 tick 11/19 的非线性差异更大，说明轮地接触、左右闭链/几何或后续耦合响应值得优先检查；它并不能单独证明右轮动作符号错误，因为目标增量和初始响应方向一致。

## 几何与求解差异

- Isaac 使用每轮 14796 点 Mesh 的 `convexHull` 碰撞；局部 AABB 约为 `0.125 x 0.0572 x 0.125 m`。
- MuJoCo 使用半径 `0.0625 m` 的球体代理。reset 时两个球体底部距地面仅约 `1.52 um` 和 `4.47 um`，MuJoCo 在 `2 ms` 首次记录到双轮接触。
- Isaac 的闭链是 4 个 `PhysicsRevoluteJoint`；MuJoCo 用 8 个带 `solref/solimp` 的 `connect` 约束，每个闭链铰链用两个点约束近似。
- Isaac 的主动关节由 PhysX implicit drive 在 5 ms 物理步长中求解；MuJoCo 每 1 ms 显式计算一次 PD 力矩。

这些差异都位于首处分歧之前或同一时刻，是下一轮单变量实验的正确对象。

## 接触数据限制

Isaac 基线没有启用 contact observer，因此当前正式比较不能直接比较接触力。尝试启用可选 observer 时，Isaac Lab 2.3.2 因未展开的 `{ENV_REGEX_NS}/Robot` prim 路径而在初始化阶段失败。该失败没有污染正式数据，部分输出明确排除。下一轮应先修复 observer 路径并重新通过“开/关 observer 不改变轨迹”的等价性门槛，再比较接触时刻和冲量。

## 下一轮顺序

1. 归档当前冻结源哈希和本报告后，修复并门控 Isaac contact observer。
2. 在零动作 20 tick 场景中比较两边首次接触时刻、法向冲量和接触点。
3. 只改 MuJoCo 轮碰撞代理，先从球体改为与 USD convex hull/轮胎截面更一致的几何，保持其余参数不变。
4. 单独测试闭链约束柔度/表达，不与接触几何同时修改。
5. 最后才测试 1 ms/5 ms 步长与显式/隐式 actuator 近似；轮端 `9 Nm` 到实车约 `2 Nm` 的调整属于后续 sim-to-real 标定，不应混入当前跨仿真器归因实验。
