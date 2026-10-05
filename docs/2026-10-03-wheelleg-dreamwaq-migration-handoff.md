# WheelLeg DreamWaQ 迁移讨论 Handoff

日期：2026-10-03  
工作区：`E:\wheel_leg_rl-main`  
状态：仅完成代码审查和设计讨论；尚未创建新工程，也没有修改现有工程代码。

## 1. 项目目标

为当前六自由度轮腿机器人建立一个结构清晰、可逐步验证的 Isaac Lab 强化学习工程。先完成普通 PPO 平地基线，再接入 DreamWaQ。

V1 目标：

- 平地平衡。
- 跟踪 X 方向线速度 `vx`。
- 跟踪 yaw 角速度 `yaw_rate`。
- 跟踪可调机身高度 `base_height`。
- 不训练 Y 方向速度。
- 不训练跳跃。
- 第一阶段不加入复杂地形。

## 2. 三个参考工程

### 当前 WheelLeg Isaac Lab 工程

路径：`E:\wheel_leg_rl-main\wheel_leg\WheelLeg_RL_IsaacLab`

- Isaac Sim 4.5.0。
- Isaac Lab 2.0.1。
- RSL-RL 2.3.3。
- 使用 `DirectRLEnv`。
- 环境文件：`IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\WheelLegRobot\wheellegrobot_env.py`。
- 资产配置：`IsaacLab\source\isaaclab_assets\isaaclab_assets\robots\wheel_leg_robot.py`。
- 实际 USD：`IsaacLab\Robot_USD\Robot_Model.usd`。

应当复用的内容：USD、关节/刚体名称、actuator 参数、机器人生成方式、训练启动方式和 Isaac Lab 接入经验。

### 复旦轮腿工程

路径：`E:\wheel_leg_rl-main\fudan_rl_wheel_leg-main\fudan_rl_wheel_leg-main`

- 使用 Isaac Gym Preview 4，不是 Isaac Lab。
- `plane` 与 `jump` 是两个独立任务。
- 只参考 `plane` 的动作语义、25维本体观测和奖励公式。
- 不复制其 Isaac Gym 环境生命周期和机器人加载代码。

### A1 DreamWaQ 工程

路径：`E:\gogo_2026_09\A1_Base-codex-dreamwaq-pace-baseline-port`

核心参考：

- `gogo-learn\gogo_learn\dreamwaq\model.py`
- `gogo-learn\gogo_learn\dreamwaq\rsl_rl.py`
- `gogo-learn\gogo_learn\dreamwaq\storage.py`

该工程使用 Isaac Lab 2.3.2 和 RSL-RL 3.0.1。当前 WheelLeg 是 RSL-RL 2.3.3，因此不能直接复制接入代码，只能复用 CENet、context latent、速度估计、下一帧重建和 KL loss 的设计。

## 3. 对旧 WheelLeg 工程的结论

旧工程仍有迁移价值，但只适合作为资产和仿真接入参考，不适合作为新任务逻辑的代码基础。

`wheellegrobot_env.py` 超过 1100 行，同时包含配置、场景、控制、IK/FK、VMC、INS、命令、课程、观测、奖励、reset、键盘输入和 CSV 日志，耦合较重。

已发现的问题包括：

- `history_length=5`，实际拼接成六帧。
- `sim_dt=0.01`、`decimation=2`，控制周期应为 `0.02s`，部分逻辑仍按 `0.01s` 更新。
- 奖励依赖 `_get_observations()` 更新的缓存，容易使用一步滞后的数据。
- 平地、站立和跳跃逻辑混在同一任务。
- Actor 直接读取真实机身线速度，不符合 DreamWaQ 的目标。
- 每步写入并刷新 CSV，会影响训练性能。
- 总奖励中 `rew_alive` 被重复相加，虽然当前权重为零。

核心结论：

> 迁移已验证的资产、接口和物理参数，不迁移旧环境文件的整体结构。

## 4. 当前推荐的工程路线

建立一个独立、轻量的 Isaac Lab 工程，而不是从空白实现强化学习框架，也不是以复旦 Isaac Gym 工程为主体。

```text
当前 WheelLeg：USD、actuator、关节信息、Isaac Lab 接入参考
复旦 Plane：平地任务目标、动作和奖励参考
A1 DreamWaQ：CENet、辅助损失和 privileged critic 参考
Isaac Lab/RSL-RL：仿真、并行环境和 PPO 基础设施
```

新工程保留普通 PPO 与 DreamWaQ 两个入口。普通 PPO 是环境正确性的基线，不能一开始只保留 DreamWaQ。

## 5. 已确认的控制接口

动作保持六维，改为纯关节空间控制：

```text
action[0] = L_F_Joint1 位置目标偏移
action[1] = L_H_Joint1 位置目标偏移
action[2] = R_F_Joint1 位置目标偏移
action[3] = R_H_Joint1 位置目标偏移
action[4] = 左轮速度目标
action[5] = 右轮速度目标
```

```python
q_target = q_nominal + leg_action_scale * action[:, 0:4]
wheel_velocity_target = wheel_action_scale * action[:, 4:6]
```

已确认：

- 训练主路径删除五连杆 IK/FK 和 VMC。
- 旧运动学只能作为离线分析工具。
- 不增加硬动作变化率限制。
- 保留 `[-1, 1]` 裁剪、关节限位和 actuator 力矩限制。
- 可以保留软 `action_rate`、`action_smooth` 惩罚。

尚未确定：`q_nominal`、腿动作缩放、轮速缩放。

## 6. 命令接口

推荐固定为：

```text
command = [target_vx, target_yaw_rate, target_base_height]
```

- 不保留 `vy`。
- 不再使用虚拟腿长 `L0` 命令。
- 键盘控制只属于播放脚本，训练 step 内不读取 JSON。
- 命令范围和课程阶段尚未确定。

## 7. Actor、CENet 与 Critic

### Actor 单帧观测提案

```text
机身角速度                         3
投影重力                           3
命令 [vx, yaw_rate, base_height]   3
四个腿关节位置误差                 4
六个关节速度                       6
上一时刻动作                       6
总计                              25
```

Actor 不读取真实机身线速度。历史严格定义为包含当前帧的五帧：

```text
[o_(t-4), o_(t-3), o_(t-2), o_(t-1), o_t]
5 × 25 = 125维
```

reset 时用当前观测填满五帧，避免零历史瞬态。

### DreamWaQ 数据流

```text
125维历史 -> CENet -> 估计速度3维 + context latent 16维
当前观测25 + 估计速度3 + context16 -> Actor MLP 44维输入 -> 六维动作
```

部署时只保留历史观测、CENet 和 Actor。

### Critic 尚未定案

Actor 和 Critic 共用同一个环境奖励，区别在于观测和训练损失。

当前有两个方案：

1. 推荐的平地最小 Critic，41维：Actor 当前观测25 + 真实机身线速度3 + 真实机身高度1 + 六关节加速度6 + 六关节力矩6。
2. 严格参考复旦的141维 privileged observation，其中还包括两帧动作、77维地形高度、质量、质心偏置、默认关节偏置、摩擦和恢复系数。

第一阶段只有平地，也没有完整域随机化，因此当前推荐41维，但用户尚未正式确认。

## 8. 奖励函数

环境每步只产生一个共享奖励：

```text
r_total = 所有环境奖励项之和
```

- Actor 用它计算 advantage。
- Critic 用它形成 return 并拟合 `V(s)`。
- CENet 的速度估计、重建和 KL 是 loss，不是环境 reward。

复旦 Plane 当前启用的奖励项：

```text
tracking_lin_vel          +1.0
tracking_lin_vel_enhance  +1.0
tracking_ang_vel          +1.0
tracking_ang_vel_enhance  +1.0
base_height               +1.0
nominal_state             -1.0
lin_vel_z                 -1.0
ang_vel_xy                -0.2
orientation               -100.0
dof_vel                   -5e-5
dof_acc                   -2.5e-7
torques                   -1e-4
action_rate               -0.01
action_smooth             -0.01
collision                 -1.0
dof_pos_limits            -1.0
```

主要公式：

```python
e_v = (target_vx - actual_vx) ** 2
e_w = (target_yaw_rate - actual_yaw_rate) ** 2
e_h = (target_height - actual_height) ** 2

r_v = exp(-e_v / 0.25)
r_v_enhance = exp(-e_v / 2.5) - 1
r_yaw = exp(-e_w / 0.25)
r_yaw_enhance = exp(-e_w / 2.5) - 1
r_height = exp(-e_h / 0.001)

r_action_rate = -0.01 * ||a_t - a_(t-1)||^2
r_action_smooth = -0.01 * ||a_t - 2*a_(t-1) + a_(t-2)||^2
```

当前适配建议：

- 保留速度、yaw、高度跟踪和常规稳定性/能耗惩罚。
- 删除全部跳跃奖励。
- 复旦 `nominal_state` 依赖虚拟腿角，不能在无五连杆运动学的设计中原样复制。推荐V1先删除，是否改成关节空间镜像对称奖励尚未确定。
- 奖励只能读取具名状态，不能读取 `obs[:, index]`。
- 所有奖励按唯一的 `control_dt` 处理，并逐项记录。

## 9. DreamWaQ 训练目标

```text
PPO clipped surrogate loss
Critic value loss
Entropy bonus
CENet 真实机身线速度估计 MSE
CENet 下一帧本体观测重建 MSE
Context latent KL loss
```

```text
L_total = L_actor
        + value_coef * L_critic
        - entropy_coef * entropy
        + velocity_coef * L_velocity
        + reconstruction_coef * L_reconstruction
        + beta * L_KL
```

第一版建议不加入 AdaBoot。先保证 PPO、CENet 和 next-observation storage 正确，再考虑高级训练机制。

## 10. 防止再次耦合的工程规则

1. V1 只做平地 `vx + yaw + height`；跳跃必须是独立任务。
2. 六维动作、三维命令和观测字段必须有版本化 schema 与维度断言。
3. 固定 `sim_dt=0.01s`、`decimation=2`、`control_dt=0.02s`；奖励、命令、加速度和 episode 时间统一使用 `control_dt`。
4. 环境不能依赖 DreamWaQ；DreamWaQ 只能接收张量，不能访问 Isaac Lab 机器人对象。
5. 奖励不能依赖观测缓存；观测、奖励和 termination 从同一时刻的仿真状态独立构造。
6. 训练 step 中禁止 CSV、JSON 和其他磁盘 I/O。
7. 奖励权重、动作缩放、命令范围和网络参数放在配置中，不散落硬编码。
8. 动作或观测含义改变时必须升级 schema，旧 checkpoint 不允许静默加载。
9. 新功能必须有明确模块、配置和测试，不能继续向 `env.py` 增加大量条件分支。

推荐结构：

```text
wheelleg_dreamwaq/
├── assets/
├── tasks/wheel_leg_flat/
│   ├── env.py
│   ├── env_cfg.py
│   ├── control.py
│   ├── commands.py
│   ├── observations.py
│   ├── rewards.py
│   └── terminations.py
├── algorithms/dreamwaq/
│   ├── model.py
│   ├── ppo.py
│   └── storage.py
├── agents/
├── scripts/
└── tests/
```

## 11. 推荐实施顺序

1. 基线审计：确认 USD、关节方向、限位和 actuator。
2. 最小环境：纯关节动作，随机动作运行1000步，无 NaN、越界和 reset 错误。
3. 普通 PPO：接入三维命令和复旦奖励适配，先学会站立与速度/高度跟踪。
4. DreamWaQ：加入五帧历史、CENet、privileged critic、下一帧 storage、重建与 KL loss。
5. 后续扩展：域随机化、课程、复杂地形、AdaBoot，以及独立跳跃任务。

普通 PPO 基线没有通过之前，不进入 DreamWaQ 集成。

## 12. 下一轮仍需确认

1. 新工程采用“轻量外部 Isaac Lab 工程”，还是“复制当前工程作为运行外壳并新增干净任务”。
2. 是否固定 Isaac Lab 2.0.1 + RSL-RL 2.3.3，还是允许升级到 A1 的版本栈。
3. `q_nominal` 的来源和具体数值。
4. 腿关节、轮速动作缩放。
5. Critic 使用最小41维还是更完整的 privileged state。
6. 是否删除 `nominal_state`，或改成关节空间镜像对称奖励。
7. V1 是否立即加入接触传感器和 collision reward。
8. `vx`、`yaw_rate`、`base_height` 的命令范围。
9. reset、termination 和训练成功指标。
10. 普通 PPO 达到什么标准后才能进入 DreamWaQ 阶段。

## 13. 下一次对话开场

> 请先读取 `E:\wheel_leg_rl-main\docs\2026-10-03-wheelleg-dreamwaq-migration-handoff.md`，把它作为本轮完整上下文。我们正在设计一个新的轮腿 DreamWaQ Isaac Lab 工程，本轮先继续设计，不修改代码。旧 WheelLeg 工程只作为 USD、actuator、关节信息和 Isaac Lab 接入参考；复旦 Plane 提供控制目标与奖励公式；A1 工程提供 DreamWaQ/CENet 参考。已经确定使用六维纯关节空间动作，任务目标是平地 `vx + yaw_rate + base_height`，不训练跳跃。请从 handoff 第12节的未决问题开始，每次只讨论并确认一个问题；首先判断新工程应采用“轻量外部 Isaac Lab 工程”还是“复制当前工程作为运行外壳并新增干净任务”，说明两者的具体工程风险并给出明确推荐。
