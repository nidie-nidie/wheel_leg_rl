# WheelLeg 复旦风格域随机化架构设计

- 日期：2026-10-06
- 修订：v0.8
- 状态：v0.7 增量复核无 P0/P1；实现集成测试发现 v0.7 的“resume 必须保持相同 `num_envs`”与主架构的跨 hardware-profile 续训契约冲突。v0.8 将 resume 明确拆分为同规模精确复用 cache 与跨规模生成目标 cache 两条路径；nominal、未对齐失败和 FK 对齐通过三组同版本 runtime trace 已冻结，增量复核、代码复核与完整训练门禁通过后启动四次正式训练
- 上游文档：`docs/2026-10-03-wheelleg-dreamwaq-architecture.md`
- 关联文档：`docs/2026-10-05-wheelleg-sim2sim-debug-design.md`

## 1. 文档目的

本文只定义普通 PPO 平地基线之后、DreamWaQ 之前的第一版域随机化架构，暂定名称为：

```text
FudanStyleDomainRandomizationV1
```

“复旦风格”表示复用复旦 Plane 工程的采样类别、分布形式和生命周期思路，不表示逐行复制 Isaac Gym 实现，也不表示当前 WheelLeg 与复旦机器人的物理参数相同。

本文需要解决以下问题：

1. 为何当前策略在 Isaac Sim 中能够站立，但迁移到 MuJoCo 后容易失稳。
2. 第一阶段应加入哪些随机化，哪些必须暂缓。
3. 随机量应在进程启动、episode reset 还是每个 observation step 采样。
4. 闭链轮腿机构如何在不破坏约束的前提下随机初始腿构型。
5. Actor、Critic、reward、termination 和 evaluation 如何隔离随机化副作用。
6. 如何保证 nominal PPO、随机化 PPO 和后续 DreamWaQ checkpoint 不会被混用。

本文不是 DreamWaQ 设计文档，不修改 CENet、ActorObsV1/CriticObsV1 的维度和字段顺序、ActionV1、CommandV1、奖励权重或 MuJoCo 模型。

## 2. 背景与当前问题

当前普通 PPO 策略主要在单一 Isaac Sim 名义模型中训练。MuJoCo sim2sim 已经完成动作、观测、控制频率和基本 PD 语义接入，但同一 checkpoint 在 MuJoCo 中的闭环站立明显弱于 Isaac Sim。

现有排查没有证明单一字段接错，也没有证明必须把两个求解器调成逐采样完全重合。更可信的风险是：策略只适应了 Isaac Sim 名义动力学附近很窄的分布，对摩擦、执行器响应、初态构型和传感器误差缺少容忍度。

因此第一版随机化的目标不是掩盖接口错误，也不是寻找一组“能让 MuJoCo 站住”的特调参数，而是让普通 PPO 在一组有来源、可复现、范围保守的训练分布中学习稳定反馈。

以下事项仍然必须先保持一致，随机化不能替代它们：

- 关节名称、顺序、正负号和零位。
- ActionV1 的六维语义与缩放。
- ActorObsV1 的字段顺序、坐标系与 normalization。
- 控制周期、decimation 和动作保持时间。
- 名义质量、质心、惯量、轮径和轮轴位置。
- PD 公式、名义增益、力矩限制和速度限制。
- 闭链自由度与约束拓扑。

## 3. 与主架构的关系

主架构 v0.15 已将完整鲁棒性扩展中的第一段保守随机化前移。根据本轮 sim2sim 结果，名义普通 PPO 与 DreamWaQ 之间加入 `Phase 1R`。

```text
Phase 1N：名义普通 PPO
    |
    |  环境、动作、观测、奖励和 sim2sim 链路通过
    v
Phase 1R：普通 PPO + FudanStyleDomainRandomizationV1
    |
    |  Isaac Sim nominal / randomized evaluation 与 MuJoCo sim2sim 通过
    v
Phase 2：DreamWaQ
```

Phase 1R 仍然使用标准 RSL-RL PPO，不引入 CENet、历史缓存、下一帧重建或 KL loss。这样可以先区分“随机化是否有效”和“DreamWaQ 辅助学习是否有效”，避免两个变量同时变化。

本文是主架构 Phase 1R 的规范性补充；主架构 v0.15 同步采用“普通 PPO -> Phase 1R 随机化 PPO -> DreamWaQ”的阶段顺序、`CheckpointMetadataV6` 和本文的双 cache resume 契约。若两份文档仍有冲突，以版本较新的显式条款为阻塞问题，不能由实现自行选择。

以下已冻结契约保持不变：

| 契约 | 维度/语义 |
|---|---|
| ActionV1 | 4 个腿关节位置目标偏移 + 2 个轮关节速度目标，共 6D |
| CommandV1 | `vx`、`yaw_rate`、`base_height`，共 3D |
| ActorObsV1 | 25D |
| CriticObsV1 | 41D |
| 控制频率 | 50 Hz |
| reward / termination | 使用无噪声仿真真值，不读取随机噪声观测 |

## 4. 复旦 Plane 的实际随机化基线

本节记录本地开源代码的事实来源，避免把历史实验日志或个人记忆当作默认配置。

### 4.1 配置来源

复旦 Plane 的默认域随机化位于：

```text
fudan_rl_wheel_leg-main/fudan_rl_wheel_leg-main/plane/
  wheel_legged_gym/envs/base/legged_robot_config.py:158-181
```

`wheel_legged/wheel_legged_config.py` 没有覆盖 `domain_rand`，因此 Plane 默认继承上述基类配置。

| 项目 | 默认启用 | 范围 |
|---|---:|---:|
| 摩擦系数 | 是 | `[0.6, 1.4]` |
| 恢复系数 | 是 | `[0.6, 1.0]` |
| base 附加质量 | 是 | `[-1.0, 2.0] kg` |
| mass/inertia scalar | 是 | `[0.9, 1.1]` |
| base COM 偏移 | 是 | 各轴 `[-0.02, 0.02] m` |
| 外部 push | 否 | 间隔 `7 s`，最大平面速度 `2 m/s` |
| Kp 比例 | 是 | `[0.95, 1.05]` |
| Kd 比例 | 是 | `[0.95, 1.05]` |
| motor torque 比例 | 是 | `[0.95, 1.05]` |
| default DOF position 偏移 | 是 | `[-0.03, 0.03] rad` |
| action delay | 否 | 配置范围 `[0, 10] ms` |

### 4.2 采样生命周期

复旦实现不是把所有随机量都在每个 episode 重采样：

| 随机量 | 复旦实现生命周期 | 本地证据 |
|---|---|---|
| friction | 创建环境时每个 env 选择一个 bucket，进程内固定 | `legged_robot.py:438-465` |
| restitution | 创建环境时每个 env 连续采样，进程内固定 | `legged_robot.py:466-483` |
| mass / COM / inertia scalar | 创建环境时采样，进程内固定 | `legged_robot.py:561-630` |
| Kp / Kd / torque / default DOF offset | 初始化 buffer 时每 env 采样，进程内固定 | `legged_robot.py:1207-1256` |
| root linear/angular velocity | 每次 episode reset 重采样 | `legged_robot.py:784-804` |
| observation noise | 每次生成 observation 独立采样 | `legged_robot.py:953-992` |

复旦的关键结构是“并行环境持有不同但稳定的动力学实例”，而不是让同一环境的执行器参数在 episode 中持续跳变。

### 4.3 观测噪声语义

按复旦配置和 noise vector 换算，主要物理量级为：

| Actor 字段 | 均匀噪声范围 |
|---|---:|
| base angular velocity | `[-0.2, 0.2] rad/s` |
| projected gravity | 每轴 `[-0.05, 0.05]` |
| leg joint position | `[-0.02, 0.02] rad` |
| joint velocity | `[-1.5, 1.5] rad/s` |
| command | 不加噪声 |
| previous action | 不加噪声 |

复旦 privileged observation 在 actor noise 之前构造，因此 Actor 使用带噪观测，Critic 使用干净状态和随机参数。

### 4.4 不直接复制的实现细节

以下行为只作为参考，不直接迁移：

1. 复旦所谓 inertia randomization 对每个 body 独立采样一个 scalar，并同时缩放该 body 的 mass 与对角惯量，不是纯惯量随机化。
2. 复旦 default DOF offset 可以直接用于树形机构；当前 WheelLeg 是闭链机构，必须先验证约束兼容性。
3. 复旦使用 Isaac Gym tensor/API；本项目使用 Isaac Lab 2.3.2，接入层必须按当前 API 实现。
4. 复旦 privileged observation 已包含随机参数；本文第一版不扩展 41D Critic。
5. 历史训练日志中的更激进范围不等于开源默认配置，不能混入 V1。

## 5. 设计目标与非目标

### 5.1 设计目标

1. 保持当前普通 PPO 算法、网络和任务定义不变，只扩展训练状态分布。
2. 复用复旦已经公开验证过的随机化类别和均匀采样思路。
3. 使用保守范围，先提高 Isaac Sim 内部鲁棒性和 MuJoCo sim2sim 成功率。
4. 所有随机量都可由 seed 复现，并写入 run manifest。
5. nominal evaluation 与 randomized stress evaluation 严格分离。
6. 随机化关闭时，环境行为必须与当前名义版本数值等价。

### 5.2 非目标

V1 明确不做以下事项：

- 不随机 root 初始 roll、pitch 或 yaw 姿态。
- 不施加 episode 内外力或外力矩。
- 不随机质量、质心和惯量。
- 不随机恢复系数。
- 不加入 action delay、通信延迟或控制抖动。
- 不加入 IMU bias、漂移或有色噪声。
- 不修改命令分布、奖励权重、termination 或 episode length。
- 不增加 terrain、台阶、坡面或高度扫描。
- 不扩展 CriticObsV1，不把随机参数直接提供给 Actor。
- 不把当前平地保持高度任务解释为“从倒伏状态自主起立”。

质量随机化暂缓的直接原因是当前 USD 名义总质量仍需与实车重新核对。以未经确认的名义值为中心扩大分布，只会把模型错误扩散到训练中。

## 6. 总体数据流与边界

```text
                         process start
                              |
                 sample persistent env parameters
               friction / q offset / gains / effort
                              |
                              v
episode reset ---> sample root velocity ---> Isaac Lab physics
                                              |
                              clean state -----+-----> reward
                                   |          +-----> termination
                                   |          +-----> clean CriticObsV1
                                   |
                                   +--> physical-unit actor noise
                                              |
                                              v
                                      normalized ActorObsV1
                                              |
                                          PPO Actor
                                              |
                                           ActionV1
```

边界规则：

1. `reward` 和 `termination` 只读取 clean state，不能读取 noisy Actor observation。
2. Actor observation noise 在物理量域加入，之后才执行现有 fixed normalization 和 clip。
3. Critic 的前 25 维由 clean ActorObsV1 语义构造，不能简单复用已经加噪的 Actor tensor。
4. randomization 模块只生成和应用参数，不计算 reward，也不依赖 PPO/RSL-RL 对象。
5. PPO 只接收环境导出的张量，不访问 material、USD prim 或 actuator 对象。
6. MuJoCo 不在训练时动态参与采样；它只作为独立 sim2sim evaluation backend。

## 7. RandomizationSchemaV1

每个 run 必须绑定一个不可变的随机化 profile：

```text
randomization_schema = RandomizationSchemaV1
randomization_profile = NominalTrainingProfileV1
                     or FudanStyleDomainRandomizationV1
                     or NominalEvaluationProfileV1
```

每个随机字段必须声明：

| 字段 | 含义 |
|---|---|
| `enabled` | 是否启用 |
| `distribution` | 当前只允许 `uniform` 或离散 bucket |
| `low/high` | 闭区间边界 |
| `unit` | 物理单位或无量纲 scale |
| `lifecycle` | `process_start`、`episode_reset` 或 `observation_step` |
| `target` | 作用的 body、shape、joint 或 observation slice |
| `coordinate_frame` | 涉及向量时必须声明坐标系 |
| `seed_stream` | 独立 RNG 子流名称 |

禁止用一个没有单位、生命周期和作用对象的字典直接驱动随机化。

### 7.1 生命周期定义

`process_start`：在环境构建期间采样一次，并在首次 rollout 前完成应用；具体字段可按 Isaac Lab API 要求在 physics initialization 前或后写入，但在整个训练进程内保持不变。

`episode_reset`：指定 env reset 时重新采样，只更新这些 env；不影响其他正在运行的 env。

`observation_step`：每次构造 Actor observation 时重新采样；只改变观测张量，不写回仿真状态。

V1 不允许在物理 step 中连续改变 friction、gain、effort limit 或 joint reference。

## 8. FudanStyleDomainRandomizationV1 参数

### 8.1 轮地摩擦

| 属性 | V1 定义 |
|---|---|
| lifecycle | `process_start` |
| distribution | 64 个 bucket，bucket 值从 `U(0.6, 1.4)` 生成 |
| env 采样 | 每个 env 均匀选择一个 bucket |
| target | 左右轮 collision shape |
| 左右关系 | 同一 env 的左右轮使用同一个值 |
| static/dynamic | 两者设为同一个采样值 |
| ground material | 保持 `1.0` |
| combine mode | 保持当前 `multiply` |
| restitution | 保持 `0.0` |

当前地面摩擦为 `1.0` 且 combine mode 为 `multiply`，因此有效摩擦就是轮端采样值。V1 不同时随机地面和轮端，避免同一目标被重复随机化。

当前资产中轮端 collision shape 没有独立 USD physics material，使用 `/physicsScene/defaultMaterial`；该默认材质由 `SimulationCfg.physics_material` 配置为 `multiply/1.0`。独立地面材质同样为 `multiply/1.0`。因此当前 pair 的有效摩擦为 `sampled_wheel_friction * 1.0`。这条继承链属于 asset/runtime contract，未来 USD 或 scene material 变化时必须重新验证。

必须在 physics initialization 完成前或通过 Isaac Lab 支持的运行时 material API 应用；不能只修改 Python 配置但没有写入 PhysX。集成测试必须读回左右轮实际 material 值。

Isaac Lab 2.3.2 的 `randomize_rigid_body_material` 提供 bucket 采样和 PhysX material 写入（`isaaclab/envs/mdp/events.py:155-283`），但它面向 `ManagerBasedEnv`，并默认对每个 shape 独立选择 bucket。本项目是 `DirectRLEnv`，且要求同一 env 左右轮共享一个值，因此实现只能复用其底层 bucket/material-buffer 思路，不能不加约束地直接调用默认 event term。

### 8.2 四个主动腿关节 reference offset

canonical 顺序固定为：

```text
[jIJ, jIO, jAB, jAG]
```

对每个 env、每个主动腿关节独立采样：

```text
delta_q_default ~ U(-0.03, +0.03) rad
q_default_env = q_nominal + delta_q_default
lifecycle = process_start
```

`q_default_env` 同时具有三个语义，必须使用同一份 tensor：

1. episode reset 时写入的四个主动腿关节实际初始 position。
2. reset 后以及零 ActionV1 对应的四个腿关节 position target reference。
3. ActorObsV1 腿关节位置误差的 reference，即 `q_actual - q_default_env`。

轮关节不加 position offset。v0.4 的原始实现把四个主动腿关节写成 `q_default_env`，却把全部被动关节写回 USD nominal。该做法已经在 8 env、seed 42 的真实 Isaac Sim preflight 中失败：初始最大闭链连接点残差为 `12.225 mm`，超过 `5 mm` gate；因此 v0.6 明确禁止每个 episode 重新组合“主动 randomized + 被动 nominal”的不一致状态。

每个 env 的完整 reset position 改为 process-start 派生的 `q_reset_projected_env`：

1. 从 USD/default 全关节 position 开始，四个主动腿关节替换为对应 `q_default_env`，两个轮关节保持 nominal。
2. 按第 8.3 节执行一次悬空、零重力、root pose/velocity 逐步边界钳制和主动关节逐步边界钳制的闭链松弛。主动关节和 root 在单个 PhysX solve 内并非数学意义上的 fixed joint；算法只保证每个 physics step 开始前重新写回边界值。禁止为此创建 fixed joint、kinematic base 或修改 USD。
3. 捕获松弛后的全部关节 position；四个主动腿关节强制精确回写 `q_default_env`，两个轮关节强制回写 nominal，再执行只更新运动学的 `forward` 并通过几何 gate。
4. 将得到的完整 tensor 冻结为 `q_reset_projected_env`，训练期间不再修改。
5. 按 `ClosedChainRootHeightAlignmentV1` 从同一 `q_default_env` 计算每 env 的确定性 root 高度补偿，并与完整 joint cache 一起冻结。

`ClosedChainRootHeightAlignmentV1` 不增加新的随机源。规范计算路径固定为：把最终 float32 `q_default_env` 和 float32 `q_nominal` 复制到 CPU，转换为 float64，使用已审计的 offset FK 计算，再把结果转换为 CPU contiguous float32 后保存/哈希。禁止直接把 CUDA FK 的结果作为 cache canonical tensor。对左右腿分别计算 wheel-axis 在 ControlFrameV1/base frame 中的向下分量：

```text
d_side(q) = -wheel_vector_body_z(q)
delta_root_z_env = max_side(d_side(q_default_env) - d_side(q_nominal))
root_world_z_reset = default_root_world_z + env_origin_z + delta_root_z_env
```

取左右两侧最大值的目的，是保证向下伸得更长的那一侧不会因统一 root 高度而在第 0 步嵌入地面；另一侧允许因左右构型差异保留非负离地间隙。补偿可以为正或负，不做单独随机采样，也不修改 root roll/pitch/yaw。该公式只使用四个主动 reference 和 AssetBundleV2 绑定的 FK 几何；`q_reset_projected_env` 中四个主动关节已被强制回写为 `q_default_env`，因此公式与最终 cache 状态一致。

由于 ActorObsV1 使用 `q_actual - q_default_env` 且不包含真实 base height，相同零误差观测可能对应约毫米级不同的绝对 root height。V1 明确把这视为有意接受的小范围 latent disturbance，而不是精确可观测的高度状态；不修改 25D Actor schema，也不改绝对 base-height reward。训练和 randomized evaluation 必须按 `root_height_offset_env` 分桶报告 base-height error，确认该偏差没有系统性破坏高度跟踪。若后续要求严格绝对高度闭环，必须另行版本化 observation/reward contract，不能在 V1 内静默加入真实高度或绝对关节 reference。

每次 episode reset 必须通过一次 `write_joint_state_to_sim()` 明确写入全部关节状态：

- 全部关节 position 写入对应 env 的 `q_reset_projected_env`。
- 全部主动和被动关节 velocity 写入 `0`。
- root position 写入 nominal XY/orientation 与 `default_root_world_z + env_origin_z + delta_root_z_env`；root velocity 使用第 8.6 节定义的 episode 样本。
- 被动关节保持 `Kp=0`、`Kd=0.05` 并使用零速度 target；四个主动腿 position target 使用 `q_default_env`，两个轮 velocity target 使用零速度。

闭链松弛和 root 高度补偿都不是新的随机变量；它们是 `q_default_env + AssetBundleV2 + PhysicsV4 + realized actuator plan + BoundaryClampedPhysXRelaxationV1 + ClosedChainRootHeightAlignmentV1` 的确定性派生状态。manifest 必须分别记录 seed 决定的 realized-plan hash 和闭链 reset-cache artifact 的文件 SHA-256、联合 tensor SHA-256、schema、两个算法版本及最终误差统计。resume 必须先加载并验证 source run 的原始 cache artifact；目标 `num_envs` 与 source 相同时必须精确复用，不得用当前 GPU 重新生成一个“看起来接近”的 cache 代替。目标 `num_envs` 不同时，source artifact 只用于验证 source checkpoint 完整性，目标环境必须按同一 master seed、相同语义 contract 和目标 env 数生成独立 target cache，并明确记录为新环境状态。reward 中使用绝对关节角、phi0、base height 或真实几何的项目仍读取 clean physical state，不用 observation error 替代。

这一设计与复旦 `default_dof_pos` 同时参与 reset、PD reference 和 joint-position observation 的语义保持一致，但只作用于当前四个主动腿关节。

### 8.3 闭链初态候选验证与一次性边界钳制松弛

当前 WheelLeg 不是树形腿。主动关节独立偏移后，不能沿用名义构型的被动关节状态。每个候选 `q_default_env` 必须先通过离线几何检查：

```text
所有值 finite
四个主动腿关节均位于硬限位内，并位于 RewardWeights.soft_leg_limit 定义的绝对角范围 ±0.95 rad 内
左右腿 FK 都返回有效解
abs(L0_side - L0_nominal_side) <= 0.010 m
abs(wrap(phi0_side - phi0_nominal_side)) <= 5 deg
abs(wrap(phi0_left - phi0_right)) <= 5 deg
```

几何筛选会把名义 `U(-0.03,+0.03)` 变为条件截断分布，因此 manifest 必须记录候选总数、接受率、各拒绝原因以及最终 realized offset 的统计量。`32` 次是首次 preflight 的最大重采上限；preflight 先审计接受率，再为正式训练冻结该上限。达到上限仍失败必须终止环境初始化并报告 seed、env id 和候选值，不能静默回退到 nominal，也不能自动放宽阈值。

几何检查通过不代表 PhysX 的全部被动关节已经处于一致状态。v0.6 在环境构造阶段、run manifest 最终化之前，对全部 env 同步执行一次 `BoundaryClampedPhysXRelaxationV1`：

```text
root pose          = nominal pose + env origin + world Z 0.75 m
root velocity      = 0
gravity            = 0
ground contact     = 无（机器人悬空）
active leg q       = 每个 step 边界重写 q_default_env
active leg target  = q_default_env，使用该 env 已实现的 Kp/Kd/effort limit
wheel q/qd         = nominal / 0，velocity target = 0
all joint qd       = 每个 step 边界写 0
passive actuator   = Kp 0、Kd 0.05、velocity target 0
relaxation         = 10 physics steps = 50 ms simulated time
```

每个 physics step 的写入顺序必须精确冻结为：

1. 读回上一步完整 joint position。
2. 覆盖四个主动腿关节为 `q_default_env`，覆盖两个轮关节为 nominal。
3. 将全部 joint velocity 置零。
4. 写 root pose/root velocity，再写完整 joint state。
5. 写主动腿 position target 与轮端零 velocity target。被动关节的 `Kp=0`、`Kd=0.05` 和零 velocity target 在进入循环前由环境执行器配置建立并完成断言，循环内不重复调用被动 target setter。
6. 调用 `scene.write_data_to_sim()`、`sim.step()`、`scene.update(sim_dt)`。

上述顺序重复 10 次。主动关节和 root 在每次 solve 中允许产生瞬时偏差，下一步边界再被钳回；因此算法名称不得写成 fixed-active/fixed-base projection。第 10 步后读回完整 joint position，将主动腿和轮端再次精确覆盖，将完整 joint velocity 与 root velocity 再次置零，写回完整 root/joint state；随后只执行 `sim.forward()` 与 scene update，不再推进 physics step，然后生成 `q_reset_projected_env` 并运行 hard gate。

重力切换必须放在 `try/finally` 中：进入算法前保存实际 readback；设置零重力后立即 readback 验证；无论成功或异常都恢复原值并再次 readback 验证。10 个 step 会推进底层 PhysX 时间 `0.05 s`，但 `common_step_counter`、`episode_length_buf` 和任务 rollout 计数必须保持零。`FudanStyleDomainRandomizationV1` 禁止依赖绝对 simulation time 的 sensor/event；若以后引入该类模块，必须新增显式 simulation-time reset 方案和 contract 版本。

选择 10 步的依据是 2026-10-06 的真实 Isaac Sim 5.1.0 / Isaac Lab 2.3.2 preflight（8 env、seed 42）。证据文件为 `wheelleg_dreamwaq/artifacts/debug/randomization/boundary-clamped-relaxation-seed42-v2.json`，SHA-256 为 `BFAA2B688EAEDBC3CB367B681D2621D20A313D47791B438B2B7A825C58601F82`：

| 时刻 | 最大闭链残差 | 最大 FK 轮轴误差 | 最大主动 reference 误差 | 最大被动角偏移 | 分支不匹配 |
|---|---:|---:|---:|---:|---:|
| 投影前 | `12.225 mm` | `8.794 mm` | `0 rad` | `0 rad` | `0` |
| 1 physics step | `0.416 mm` | `2.084 mm` | `9.91e-3 rad` | `0.0328 rad` | `0` |
| 5 physics steps | `0.0013 mm` | `0.0099 mm` | `1.63e-5 rad` | `0.0341 rad` | `0` |
| 10 physics steps | `0.0006 mm` | `0.0064 mm` | `1.49e-6 rad` | `0.0341 rad` | `0` |
| 200 physics steps | `0.0008 mm` | `0.0058 mm` | `1.79e-6 rad` | `0.0341 rad` | `0` |

结果证明 nominal 被动状态不可直接使用，也证明 10 步边界钳制松弛已经进入稳定微米量级。正式实现仍必须在第 10 步后逐 env hard fail，而不是因为上述样本通过就跳过验证：

- 全部状态 finite。
- 闭链连接点最大残差 `<= 0.5 mm`。
- FK 轮轴误差 `<= 0.5 mm`。
- FK 与真实 `L0` 误差 `<= 0.5 mm`。
- FK 与真实 `phi0` 误差 `<= 0.25 deg`。
- 四个主动腿关节与 `q_default_env` 的误差 `<= 1e-4 rad`。
- 全部 26 个 joint position 必须满足 USD/PhysX 声明的 finite hard limit；当前 asset 返回近似 float32 无穷界，因此该检查保留但不能作为主要分支判据。
- 全部 20 个被动关节相对 USD nominal 的 raw delta 绝对值 `< pi`，wrapped delta 绝对值 `<= 0.10 rad`。
- 八个物理被动关节 `jJM,jMK,jKN,jOP,jBE,jEC,jCF,jGH` 的符号分支必须逐 env 等于 nominal `[-,+,-,-,+,+,+,+]`；dummy joint 不参与该签名。
- 审计必须包含每个 joint 的 min/max/max-delta，以及 0/1/5/10 步的完整 26D position/velocity trace，不能只保存聚合最大值。

任何 env 不满足条件都必须终止环境初始化并报告 seed、env id、主动候选、全部误差、分支签名和投影后的关节状态；不得增加 episode 隐藏 settling、静默回退 nominal 或放宽训练 gate。

实现期接触 preflight 又暴露了与闭链投影不同的第二个问题。8 env、seed 42 在统一 `0.20003 m` root 高度下，randomized 构型第 0 步最深嵌地为 `7.075 mm`，而 nominal 对照 Gate A/B 最大穿透分别只有 `0.0132 mm`/`0.0136 mm`。这证明失败来自 reference 改变了轮轴向下伸长量，而不是 nominal 接触求解器本身。以下三份证据均由同一脚本版本生成，schema 为 `Phase1RandomizationRuntimeGateV2`，script SHA-256 为 `AC73A1D6A6C22740413D15CD9DD4652D511AAC6A5DD21D88E986DBFA64057578`：

- 未对齐失败 trace：`wheelleg_dreamwaq/artifacts/debug/randomization/phase1r-runtime-gates-v2-unaligned-seed42.json`，SHA-256 `E04AA31FB647E54FFE0AC34C327874C9C60B28D8C9B7B6273FCB16A084649134`；Gate A/B 最大穿透均为 `7.075 mm`，失败项仅为 wheel penetration。
- nominal 对照 trace：`wheelleg_dreamwaq/artifacts/debug/randomization/phase1r-runtime-gates-v2-nominal-seed42.json`，SHA-256 `BAA9A2783FF7825C54B03CCF33A0E97FC368AD634B19C1BB256B6D746E44091B`；Gate A/B 全部通过。

只加入上述 FK 高度补偿后，同一 randomized seed 的 Gate A/B 最大穿透分别降为 `0.157 mm`/`0.403 mm`，主动腿最大速度分别为 `0.701 rad/s`/`0.717 rad/s`，其余闭链、FK、轮速、饱和和 finite gate 全部通过。通过 trace 为 `wheelleg_dreamwaq/artifacts/debug/randomization/phase1r-runtime-gates-v2-aligned-seed42.json`，SHA-256 `8AAAEDDA990453C97E7A0D9159939B946490F044B1D453BDDBF553E47B9DF07A`。因此 root 高度补偿属于 reset 正确性，不是可调训练超参数。

#### 8.3.1 ClosedChainResetCacheSchemaV2

完整 reset cache 必须保存为每个 run 目录下的独立 `closed_chain_reset_cache.pt`，而不是只把 hash 写进 manifest。artifact 至少包含：

- schema/version：`ClosedChainResetCacheSchemaV2`。
- algorithm/version：`BoundaryClampedPhysXRelaxationV1`。
- root-height algorithm/version：`ClosedChainRootHeightAlignmentV1`。
- `q_reset_projected_env` 的完整 tensor、dtype、shape、canonical/USD joint order。
- `root_height_offset_env` 的完整 float32 contiguous tensor，shape 为 `(num_envs,)`，语义为相对 nominal/default root world-Z 的每环境补偿。
- 10 steps、`sim_dt=0.005 s`、root world-Z lift `0.75 m`、零重力、主动/轮端/被动 masks、被动 `Kp/Kd/target`、完整写入顺序、最终 `sim.forward()` 语义。
- AssetBundleV2 hash、PhysicsV4 hash、randomization profile hash、realized-plan hash，以及六个主动关节按 `CANONICAL_JOINT_ORDER` 排列的 realized Kp/Kd/effort-limit tensor/hash；四腿和两轮可另存可读统计，但不能从 cache 身份中省略轮端 Kd/effort limit。
- hard-gate 阈值、逐 joint 统计、最终几何误差和物理分支签名。

文件 SHA-256 与 `q_reset_projected_env + root_height_offset_env` 的联合 canonical-bytes SHA-256 必须同时写入 run manifest 和每个 checkpoint metadata。canonical tensor hash 固定为：tensor `detach -> CPU -> contiguous`；输入依次为 ASCII field name、ASCII shape tuple、ASCII torch dtype、little-endian C-order raw bytes；两个字段按 field name 排序；SHA-256 输出为大写十六进制。fresh run 生成并原子写入 artifact 后，才能最终化 manifest。

resume 流程必须在构造训练环境前从 source manifest 定位 artifact，并验证 source 文件 SHA、联合 tensor SHA、schema/两个 algorithm、asset/physics/profile、seed、source `num_envs`、realized-plan 和 actuator-plan 全部匹配。随后按目标规模分支：

- **同规模 resume：** 目标 `num_envs == source num_envs` 时，把 source 的两个 tensor 一起注入新环境。新环境按本节冻结的 `CPU float64 FK -> CPU contiguous float32` 规范路径重算 `root_height_offset_env`，并与 artifact tensor 执行逐元素 bitwise/`torch.equal` 比较；随后应用 source cache，执行最终 `sim.forward()` 和本节 hard gate，并把完全相同的 artifact 原子复制进新 run 目录。
- **跨规模 resume：** 目标 `num_envs != source num_envs` 时，不得裁剪、重复、补零或直接注入 shape 不符的 source tensor。source artifact 仍必须先完整验证；之后使用 checkpoint 的 master seed、同一 randomization profile/算法和目标 `num_envs` 构造新的 DirectRLEnv，由目标环境执行标准 process-start sampling、10-step relaxation、root-height canonical 重算和全部 hard gate，生成独立 target cache。target manifest/checkpoint 绑定 target artifact/hash，并在 resume metadata 中同时记录 source/target `num_envs`、source/target cache SHA 和 `cache_resume_mode="regenerated_for_target_num_envs"`。这属于主架构定义的“恢复优化状态和 RNG、创建新环境状态”，不声称延续 source 的逐 env 动力学样本。

同规模路径不得重新跑 10-step relaxation 后只做近似比较；跨规模路径不得把新 target cache 冒充 source cache。任一 source artifact 缺失/hash 不符，或任一目标 hard gate 失败，都必须 hard fail。

完成一次性松弛后，正式训练前仍需执行两个分离的 Isaac Sim reset preflight，不能把 root 速度扰动造成的响应误判为闭链初态错误。

**Gate A：闭链初态 gate。** 强制 root linear/angular velocity 全部为零；一次性写入第 8.2 节冻结的 `q_reset_projected_env` 和 `root_height_offset_env`，施加零 ActionV1，并连续运行 `40 physics steps = 10 control steps = 200 ms`。测量窗口必须包含 physics step 之前的第 0 步写入状态，不能只从第一个 solve 之后开始统计；全局 maxima/finite 判定必须由第 0 步状态初始化，不能只把第 0 步保存在 per-env 附件中。要求：

- 闭链连接点最大残差不超过 `5 mm`。
- reset 后左右 `L0/phi0` 与离线 FK 一致。
- 轮端碰撞几何相对地面的额外穿透不超过 `2 mm`。
- 四个主动腿关节绝对速度不超过 `5 rad/s`，两个轮关节绝对速度不超过 `10 rad/s`。
- 任一主动关节达到 `0.99 * realized effort limit` 的状态不得连续超过 `2 physics steps`，整个窗口的 per-joint saturation fraction 不超过 `5%`。
- 全部状态、PD 近似 applied torque、约束残差和 FK 输出保持 finite，不出现 solver/constraint error。
- 被动关节使用 process-start cache，没有被随机；其 `Kp=0`、`Kd=0.05` 和零 velocity target 与松弛/cache contract 一致，不存在被动 position target。

**Gate B：完整随机 reset gate。** Gate A 通过后，保持同一 `q_reset_projected_env + root_height_offset_env` 写入规则，再启用第 8.6 节的随机 root COM linear/angular velocity，使用固定 preflight seed 重复 `200 ms` 零动作测试。仍要求闭链残差、穿透、关节速度、饱和比例和 finite 检查满足 Gate A 的阈值，并包含第 0 步。Gate A 失败归因于 reset cache/root-height/contact 路径；只有 Gate B 失败时，优先调查 root velocity 坐标变换或扰动范围，不能把两者混为同一个结论。

process-start 的 10 个闭链松弛 physics steps 与 Gate A/B 的两个 `200 ms` 窗口是三件不同的事。前者只生成全生命周期复用的 reset cache；后两者只属于独立 preflight/integration test。正式 `DirectRLEnv` episode reset 写入 cache 后立即生成 observation 并进入策略控制，不额外推进物理世界，也不丢弃 control steps。

上述速度、穿透和饱和阈值是 v0.7 preflight 判据，不是训练超参数。首次 nominal、未对齐失败、FK 对齐、relaxation、Gate A 和 Gate B trace 都必须保留；若 nominal 本身违反阈值，应先调查名义模型或测量定义，不能只对 randomized profile 放宽阈值。

若 process-start relaxation hard fail，或 Gate A 在缓存状态下仍失败，V1 必须停止并调查 USD 约束、松弛实现或接触初态；不能恢复为“主动 randomized + 被动 nominal”。若 Gate A 通过而 Gate B 失败，应先修正 root velocity 写入或缩小并版本化该扰动范围。两类失败都不能用更强 damping 掩盖。

### 8.4 主动执行器 Kp/Kd scale

```text
kp_scale ~ U(0.95, 1.05)
kd_scale ~ U(0.95, 1.05)
lifecycle = process_start
```

作用于六个主动关节，每个 env、每个 joint 独立采样：

- 四个腿关节：名义 `Kp=120`、`Kd=4`。
- 两个轮关节：名义 `Kp=0`、`Kd=0.6`。

轮端 `Kp` 名义值为零，因此乘 scale 后仍为零；V1 只随机其 damping。所有被动关节保持现有 `Kp=0`、`Kd=0.05`，不得误包含在 active joint mask 中。

Isaac Lab 2.3.2 的 `Articulation.write_joint_stiffness_to_sim()` 和 `write_joint_damping_to_sim()` 接受 `env_ids` 以及 `(num_envs, num_joints)` tensor，因此按 env 写入 PhysX 在 API 层可行（`articulation.py:652-708`）。但这两个 writer 明确不会同步 actuator model 自身的参数。

当前 `ImplicitActuator.compute()` 又使用其内部 `stiffness/damping` 计算近似 `computed_effort/applied_effort`（`actuator_pd.py:117-142`），而这些值进入当前 reward、Critic 和日志。因此实现必须把同一 realized gain 同步到两处：

```text
PhysX joint stiffness/damping
+ ImplicitActuator stiffness/damping mirror tensors
```

只写 PhysX 会造成机器人真实响应已经变化，但 reward/Critic 看到的近似力矩仍按 nominal gain 计算。集成测试必须同时核对 PhysX buffer、actuator mirror 和已知状态误差下的 torque estimate。

### 8.5 主动执行器 effort limit scale

```text
effort_limit_scale ~ U(0.95, 1.05)
lifecycle = process_start
```

作用于六个主动关节的 effort limit，每个 env、每个 joint 独立采样。名称固定为 `effort_limit_scale`，不称为“真实电机扭矩常数随机化”。

复旦 torque scale 直接乘计算后的 torque；本项目当前使用 Isaac Lab `ImplicitActuator`，随机 effort limit 只能保证饱和上限变化，不保证与复旦逐采样力矩公式完全等价。该差异必须进入 manifest 和评审记录。

`Articulation.write_joint_effort_limit_to_sim()` 同样支持按 env 写入（`articulation.py:805-838`），但不会自动同步 actuator model 的 effort-limit mirror。同一 realized tensor 必须同时写入三处：

```text
PhysX / robot.data.joint_effort_limits
ImplicitActuator.effort_limit_sim
ImplicitActuator.effort_limit
```

其中 `_clip_effort()` 实际读取 `actuator.effort_limit`。少写任何一处都会使仿真饱和、近似 torque reward 和 saturation 日志互相矛盾。当前 implicit actuator 的 `applied_torque` 是根据 position/velocity error、gain 和 limit 得到的 PD 近似值，不是 PhysX 约束求解后的实测关节反力；本文所有 torque reward、Critic 和日志继续沿用这一既有语义，并在 manifest 中明确标注 `implicit_pd_estimate`。

当前 `env.py` 的 `Actuator/effort_saturation_fraction` 缓存 env-0 的一维 effort-limit 快照，不能用于 per-env limit。Phase 1R 必须删除该共享快照，按每个环境实时读取 `robot.data.joint_effort_limits[:, controlled_joint_ids]` 并与对应 `applied_torque` 比较；训练日志、smoke gate 和退出评估统一使用这一 realized per-env 阈值。

当前名义配置和 MuJoCo contract 的轮端 effort limit 都是 `9 Nm`；此前实车信息表明真实轮端可能只有约 `2 Nm`。V1 的 `±5%` scale 只用于当前 Isaac/MuJoCo 名义模型的鲁棒性训练，不能消除 `9 Nm` 与实车约 `2 Nm` 的名义差异。实车训练前必须另行冻结真实 nominal limit 并升级 asset/checkpoint contract。

### 8.6 episode reset root velocity

V1 不改变 root position 和 quaternion，只在每个 episode reset 时增加小范围速度扰动。采样先在 ControlFrameV1 中定义：

```text
linear velocity x/y ~ U(-0.10, +0.10) m/s
linear velocity z   ~ U(-0.05, +0.05) m/s
roll/pitch/yaw rate ~ U(-0.10, +0.10) rad/s
lifecycle = episode_reset
```

`schemas/frames.py` 必须新增具名逆变换 `transform_control_vector_to_usd()`。当前使用 row-vector 约定，其实现为 `values @ R_CONTROL_FROM_USD`，并与 `transform_usd_vector_to_control()` 做 round-trip 测试。

采样值定义为 ControlFrameV1 下的 root COM linear/angular velocity，而不是 root-link 原点速度。线速度和角速度分别经上述逆变换得到 world/USD 向量，按 `[vx,vy,vz,wx,wy,wz]` 拼成 6D tensor，再调用 `Articulation.write_root_velocity_to_sim(root_velocity, env_ids)`；该 API 的目标语义固定为 simulation world frame 下的 root center-of-mass velocity。因为定义本身就是 COM velocity，不引入 link-to-COM 的 `omega x r` 近似。

这不是 root 姿态随机化：初始 roll、pitch、yaw 仍保持 nominal。小角速度扰动用于迫使策略在 episode 开始后主动收敛姿态，同时避免一开始就把闭链轮端以倾斜姿态压入地面。

复旦默认对六个 root velocity 分量使用 `[-0.5, 0.5]`。V1 采用更保守范围，因为当前策略、闭链初态和轮端接触尚未在该量级验证。扩大到复旦完整范围必须创建新 profile，不允许在同名 V1 中静默修改。

所有关节 reset velocity 在 V1 仍为零。关节速度随机化若未来加入，必须单独定义闭链一致性和轮端安全门槛。

### 8.7 Actor observation noise

Actor noise 在 clean physical observation 构造完成后、fixed normalization 之前加入。实现中必须区分三个具名对象：

1. `PhysicalActorFieldsV1`：未归一化的物理字段集合，包括 angular velocity、projected gravity、command、leg position error、joint velocity 和 previous action。
2. `CleanActorFeaturesV1`：由 clean `PhysicalActorFieldsV1` 经过 NormalizationV2 后得到的无噪 25D tensor，只用于 CriticObsV1 的前 25D、回归测试和诊断。
3. `ActorObsV1`：复制 clean physical fields，只对指定字段加入物理量噪声，再独立执行 NormalizationV2 得到的 25D Actor 输入。

三者不得共享会被原地修改的 storage。数据流固定为两条独立路径：

```text
clean PhysicalActorFieldsV1 -> normalize -> CleanActorFeaturesV1 -> Critic

copy(clean PhysicalActorFieldsV1) -> add physical-unit noise
                                   -> normalize -> ActorObsV1 -> Actor
```

噪声范围如下：

| ActorObsV1 slice | 物理量噪声 | lifecycle |
|---|---:|---|
| angular velocity `0:3` | `U(-0.2, 0.2) rad/s` | `observation_step` |
| projected gravity `3:6` | 每轴 `U(-0.05, 0.05)` | `observation_step` |
| command `6:9` | 无 | - |
| leg joint error `9:13` | `U(-0.02, 0.02) rad` | `observation_step` |
| joint velocity `13:19` | `U(-1.5, 1.5) rad/s` | `observation_step` |
| previous action `19:25` | 无 | - |

顺序固定为：

```text
clean physical value
  -> add physical-unit noise
  -> existing NormalizationV2 scale/offset
  -> existing clip
  -> ActorObsV1
```

禁止先 normalization 再套用上述物理量数值。projected gravity 加噪后不额外归一化为单位向量，以保持复旦式逐分量噪声语义；随后仍应用现有 clip。

CriticObsV1 的 `0:25` 必须来自 `CleanActorFeaturesV1`。禁止把物理量噪声直接加到已归一化 25D tensor，也禁止先生成 noisy Actor tensor 后再反向推导 Critic 前缀。

### 8.8 V1 参数总表

| 随机量 | 范围 | 生命周期 | 作用域 |
|---|---:|---|---|
| wheel friction | 64 buckets from `[0.6,1.4]` | process start | 每 env，同 env 左右轮一致 |
| active leg default offset | `[-0.03,0.03] rad` | process start | 每 env、每主动腿关节 |
| active joint Kp scale | `[0.95,1.05]` | process start | 每 env、每主动关节 |
| active joint Kd scale | `[0.95,1.05]` | process start | 每 env、每主动关节 |
| active joint effort limit scale | `[0.95,1.05]` | process start | 每 env、每主动关节 |
| root reset linear velocity | 第 8.6 节 | episode reset | 每 env、每 episode |
| root reset angular velocity | 第 8.6 节 | episode reset | 每 env、每 episode |
| actor observation noise | 第 8.7 节 | observation step | 每 env、每字段 |

## 9. Actor、Critic、reward 与 command 契约

### 9.1 Actor

Actor 仍接收 25D ActorObsV1。随机化参数本身不输入 Actor。部署侧接口、字段顺序、normalization 常量和网络结构不变化。

腿关节位置误差的 reference 从单一 `q_nominal` 变为每 env 的 `q_default_env`。这是 ActorObsV1 字段语义的受控参数化，不改变 slice 和维度；必须写入 checkpoint contract，避免 nominal 与 randomized checkpoint 在不同 reference 下静默混用。

环境内部先构造未归一化的 `PhysicalActorFieldsV1`，再按第 8.7 节的两条独立路径生成 `CleanActorFeaturesV1` 和 `ActorObsV1`。训练 Actor 收到的是在物理量域加噪后独立归一化得到的 ActorObsV1；nominal evaluation 和部署时不加噪，reference 回到 `q_nominal`，因此外部 25D 接口不变。

`CleanActorFeaturesV1` 是训练侧内部 tensor 名称，不作为新的部署 observation schema 导出；checkpoint/training manifest 只记录 `actor_observation=noisy`、`critic_actor_features=clean` 以及对应 profile hash。

### 9.2 Critic

Critic 仍接收 41D CriticObsV1：

```text
CleanActorFeaturesV1 25D
+ true base linear velocity 3D
+ true base height 1D
+ joint acceleration 6D
+ applied joint torque 6D
= 41D
```

这里的前 25D 与 ActorObsV1 具有相同字段、顺序和 normalization，但在 randomized training 中不包含 observation noise，因此不能通过复用 Actor 返回 tensor 构造。

V1 不增加 friction、gain scale、effort scale 或 joint offset privileged fields。这样与复旦 privileged critic 不完全相同，但可以保持现有网络和 checkpoint 结构，先单独验证随机化是否提高闭环鲁棒性。

Critic 看不到 process-start 随机参数是 V1 的有意对照设计，不阻塞首批训练。当前 RSL-RL 3.1.2 训练日志没有稳定导出 explained variance，也没有定义 normalized value loss，因此 v0.2 中基于这两个量的自动阈值取消，不能伪装成可执行 gate。首批四个 Phase 1R run 只记录现有 `Loss/value_function`、episode return、survival、姿态和跟踪指标；是否进入 `CriticObsV2` 由四个 run 完成后的独立架构复核决定。若后续需要自动判定，必须先新增具名的 `CriticDiagnosticsV1`，冻结 returns/value 的采样时刻、explained-variance 公式和归一化分母，再创建新文档版本。

候选 privileged 字段预先登记为 wheel friction 1D、leg default offset 4D、active Kp scale 6D、active Kd scale 6D 和 active effort-limit scale 6D。它们只进入 manifest 的 `critic_v2_candidates`，不进入当前 41D tensor。若升级，必须定义新字段顺序、normalization 和 checkpoint schema；不得修改 CriticObsV1 后继续沿用原名称。

### 9.3 Reward 与 termination

V1 不修改奖励公式和权重。包括速度跟踪、yaw 跟踪、base height、orientation、action rate、torque、phi0 symmetry 等项目，都读取 clean state 和实际 applied action。

termination 不读取 noisy observation，不因一次观测噪声越界而结束 episode。随机化造成真实物理状态越过现有阈值时，仍按当前 termination 结束。

### 9.4 Command

现有 command 采样范围、模式概率、episode 内保持/重采样规则保持不变。randomization RNG 与 command RNG 必须分流；增加或删除一个随机化字段不能改变同一 seed 下的 command 序列。

## 10. 训练与评估 profile

### 10.1 NominalTrainingProfileV1

- 关闭本文全部随机化。
- 用于复现当前普通 PPO 基线和回归测试。
- 关闭随机化后不得仍然消耗会影响 command/reset 的共享 RNG。

### 10.2 FudanStyleDomainRandomizationV1

- 开启第 8 节全部项目。
- 只用于 Phase 1R 从头训练。
- 不允许从 nominal checkpoint resume 后继续记为同一 run。

### 10.3 NominalEvaluationProfileV1

- 所有动力学和 observation noise 回到 nominal。
- policy action 使用 deterministic mean。
- 初态和 command 使用固定评估 seed/protocol。
- 用于四次训练之间的正式横向排名，以及 Isaac Sim 与 MuJoCo 的名义 sim2sim 对比。

### 10.4 Randomized stress evaluation

随机压力评估复用 `FudanStyleDomainRandomizationV1` 的范围，但使用独立、固定的 `RandomizedStressSeedSetV1=(31001,31002,31003,31004)`，并禁用 PPO action sampling。四个 seed 全部执行并等权汇总，不允许因结果不利而删除或替换。它只报告鲁棒性分布，不替代 nominal evaluation，也不能通过挑选某个有利 seed 给模型排名。

`RandomizedStressProtocolV1` 固定为每个 seed 使用 `64 env`、每 env 一个 `10 s / 500 control-step` episode，发生 termination 后该 env 记为失败且不以 reset 后的新 episode 补足。64 个 env 的命令表固定为：16 个站立 `(0,0,0.20)`；8 个前进 `(0.75,0,0.20)`；8 个后退 `(-0.75,0,0.20)`；8 个左转 `(0,0.5,0.20)`；8 个右转 `(0,-0.5,0.20)`；其余 16 个均匀分配给 `(vx,yaw_rate)=(+/-0.75,+/-0.5)` 四种组合，base height 均为 `0.20 m`。汇总时先对每个 seed 的 64 个 episode 等权统计，再对四个 seed 等权统计；不得按 surviving episode 二次加权。

每个 checkpoint 至少分别报告：

```text
Isaac Sim nominal
Isaac Sim randomized stress
MuJoCo nominal sim2sim
```

MuJoCo randomized stress 不属于 V1 必选 gate；只有 MuJoCo 参数随机化也建立独立 schema 后才能加入，不能把 Isaac Sim API 参数名直接套到 MuJoCo。

## 11. RNG 与可复现性

训练 run 的 master seed 默认取最终生效的 `DirectRLEnvCfg.seed`；命令行 run seed 可以覆盖配置值，但覆盖后的值必须同时写入环境配置、checkpoint 和 manifest。master seed 必须派生独立子流：

```text
material_rng
joint_reference_rng
actuator_gain_rng
effort_limit_rng
reset_velocity_rng
observation_noise_rng
command_rng
```

派生算法固定为 `RandomizationSeedDerivationV1`：对 UTF-8 字符串 `RandomizationSeedV1|<master_seed>|<stream_name>` 计算 SHA-256，取前 8 bytes little-endian 并 mask 到 63-bit 正整数，作为该子流 seed。禁止使用 Python `hash()`，也禁止所有模块共享一个全局 `torch.rand` 序列。

generator/device 规则固定为：

| 子流 | generator/device | 规则 |
|---|---|---|
| material / joint_reference / actuator_gain / effort_limit | `torch.Generator(device="cpu")` | process-start 在 CPU 采样，再一次性搬到环境 device |
| reset_velocity / command | `torch.Generator(device="cpu")` | reset 时必须先创建 CPU tensor 并用 CPU generator 采样，再 `.to(env.device)` 写入指定 env；禁止将 CPU generator 传给 CUDA `torch.rand` |
| observation_noise | `torch.Generator(device=env.device)` | 每 observation step 在环境 device 采样，避免 CPU 同步 |

manifest 必须记录每个子流的 seed、generator 类型和 device。该规则保证随机参数样本可复现，但不承诺 RTX 4060 与 RTX 5070 的 PhysX 轨迹逐 bit 一致；跨 GPU 比较使用统计指标和相同随机样本 manifest。

randomization runtime 必须暴露 `get_rng_state()` / `set_rng_state()`，checkpoint 始终完整保存上述七个独立 `torch.Generator.get_state()`。`ContractOnPolicyRunner` 的 checkpoint metadata factory 在每次保存时读取环境 RNG state。本文正式实现使用 `CheckpointMetadataV6`，并验证 stream 名称、generator device/state 与 cache artifact/联合 tensor hash 完整性；七个 state 即使在跨规模恢复时未被选中，也必须通过对应临时 `torch.Generator.set_state()` 的可用性校验。只恢复全局 Python/NumPy/Torch RNG 不视为可复现 resume。

七个 stream 按生命周期固定分组：process-start streams 为 `material_rng`、`joint_reference_rng`、`actuator_gain_rng`、`effort_limit_rng`；runtime streams 为 `reset_velocity_rng`、`observation_noise_rng`、`command_rng`。同规模 resume 恢复 source 的全部七流。跨规模 resume 的 target cache 已经消耗并确定了 target process-start streams，因此只从 source 恢复三个 runtime streams，四个 process-start streams 保留 target 生成 cache 后的状态。新 checkpoint 保存这个混合后的完整七流状态，并在 resume metadata 中逐 stream 记录来源；禁止把 source process-start 游标与 target realized plan/cache 组合后仍宣称七流全部来自 source。

resume 语义固定为 `weights_optimizer_rng_new_environment`，不声称恢复 PhysX 中途状态。顺序必须是：

1. 在构造环境前读取 source manifest，验证并加载 `closed_chain_reset_cache.pt`。使用 checkpoint 的 master seed、目标 hardware profile 和目标 `num_envs` 构造 DirectRLEnv；同规模时注入 source cache，跨规模时生成并验证 target cache。DirectRLEnv 构造中的 `SimulationContext.reset()` 不是任务 `_reset_idx()`；此时只允许完成 cache 应用或 target cache 生成、`sim.forward()` 和 hard gate，不能把它记作 episode reset。
2. `RslRlVecEnvWrapper` 构造时执行第一次完整 bootstrap `env.reset()`；其 command、root velocity 和返回 observation 都是临时值。随后 runner 构造函数调用一次 `get_observations()` 以创建网络，该 observation 同样是临时值。
3. runner 加载 model/optimizer/iteration 后，恢复全局 RNG。环境 generator 同规模时恢复 source 全部七流；跨规模时只恢复 source 的三个 runtime streams并保留 target 的四个 process-start streams。前两步对 runtime streams 的消耗发生在恢复前，不影响恢复后的运行期序列。
4. 恢复完成后强制 `wrapper.reset()` 全部 env。该 reset 采样的 command/root velocity 定义恢复后第一个真实 episode；reset 返回 observation 时会消耗恢复后的第一组 observation-noise 样本，但现有 RSL-RL 3.1.2 接口不会把该返回值传给 `runner.learn()`，因此这组 noisy observation 明确标记为 `post_restore_reset_discarded_policy_obs`。
5. 调用 `runner.learn(..., init_at_random_ep_len=True)`；其先用恢复后的全局 Torch RNG 随机化 episode length，再调用 `get_observations()`。该调用消耗恢复后的第二组 observation-noise 样本，定义为 `first_policy_observation`，是第一份真正进入 Actor 的输入。

不得声称 post-restore `reset()` 返回的 observation 就是第一份策略输入，也不得额外调用 `get_observations()` 改变序列。相同 checkpoint 在相同目标 hardware profile/`num_envs` 下重复 resume 两次必须得到相同的第一个真实 command、root velocity、`post_restore_reset_discarded_policy_obs` 和 `first_policy_observation`；测试必须按上述真实调用序列执行。

训练入口必须把这些数据原子保存为 `ResumeSequenceTraceV1` 的 `resume_sequence_trace.pt`。`schema_version` 的固定值为 `ResumeSequenceTraceV1`；字段集合固定为：`schema_version`、`cache_resume_mode`、`rng_stream_sources`、`post_restore_command`、`post_restore_root_velocity_world_usd`、`post_restore_reset_discarded_policy_obs`、`first_policy_observation`、`tensor_sha256`。其中 `post_restore_root_velocity_world_usd` 表示完成目标 tensor dtype 转换后、实际赋给 `root_state[:, 7:]` 并写入 PhysX 的 float32 数值，不表示转换前的原始 RNG 临时 tensor。四个 tensor 必须分别为 CPU contiguous float32，shape 为 `(num_envs,3)`、`(num_envs,6)`、`(num_envs,25)`、`(num_envs,25)`；`tensor_sha256` 使用第 8.3.1 节相同的 canonical named-tensor hash。trace 不进入语义 contract；`training_summary.json` 中的 `resume_sequence_trace` 子对象必须恰好记录 `schema_version`、`relative_path`、`file_sha256`、`tensor_sha256`。目标 `num_envs` 改变时 tensor shape 和每步 RNG 消耗量随之改变，不要求与 source 逐 env 对齐。resume 可复现的是“同一 checkpoint 以同一目标环境配置启动时的后续随机序列”，不是与未中断 PhysX 轨迹逐 step 完全一致。

run manifest 至少记录：

- master seed 与 seed derivation version。
- 每个 seed substream 的派生 seed、generator 类型和 device。
- randomization schema/profile 名称和内容 hash。
- 每项范围、单位、生命周期和 target joint/body/shape。
- 64 个 friction bucket 的数值或稳定 hash。
- realized 参数的 min/max/mean/std。
- closed-chain 候选重采次数和失败数量。
- `ClosedChainResetCacheSchemaV2` artifact 相对路径、文件 SHA-256、联合 tensor SHA-256、两个 tensor 的 shape/dtype、两个算法版本和 hard-gate 统计。
- Actor noisy、Critic clean 的 observation policy。
- software/dependency manifest 与上游任务 contract hash。

resume provenance 固定使用 `ResumeMetadataV2`，字段集合为：`schema_version`、`mode`、`cache_resume_mode`、`source_checkpoint`、`source_checkpoint_sha256`、`source_run_manifest`、`source_run_manifest_sha256`、`source_completed_iterations`、`source_hardware_profile_name`、`target_hardware_profile_name`、`source_num_envs`、`target_num_envs`、`same_num_envs`、`source_cache_file_sha256`、`source_cache_tensor_sha256`、`target_cache_file_sha256`、`target_cache_tensor_sha256`、`source_realized_plan_hash`、`target_realized_plan_hash`、`rng_stream_sources`。两个 profile name 字段只表示 `portable`、`rtx4060`、`rtx5070` 等 hardware profile；randomization profile 由 active contract 及其 profile hash 表示，resume 时必须完全相同。`cache_resume_mode` 只能为 `reused_source_exactly` 或 `regenerated_for_target_num_envs`；`rng_stream_sources` 必须恰好包含七个 stream，值只能为 `source_checkpoint` 或 `target_process_start`。同规模模式下七流均为 source；跨规模模式下三个 runtime streams 为 source、四个 process-start streams 为 target。

### 11.1 manifest 最终化时序

profile contract 与每个 run 的 realized audit 分开记录：contract 冻结随机化语义和范围，必须完全 seed-independent；realized audit 记录该 run seed 实际采到的 buckets、offsets、gains、limits 和拒绝统计。训练入口顺序固定为：

1. 解析最终 cfg、profile、hardware profile 和 master seed，创建 run directory 并导出 resolved YAML。resume 先读取 source manifest/cache 并完成完整性验证。
2. 构造 `WheelLegFlatEnv`；完成 process-start sampling、几何筛选、material/actuator 写入，并生成 fresh cache 或应用 source cache。
3. fresh run 与跨规模 resume 原子保存当前目标环境生成的 `closed_chain_reset_cache.pt`；同规模 resume 原子复制验证通过的 source artifact。两条路径都禁止把 source/target artifact 身份混写。
4. 从环境读取 immutable audit 和 cache metadata，构建 `Phase1RandomizedContractV3` 与最终 run manifest；使用临时文件加原子 rename 一次性写入最终 manifest。
5. 计算最终 manifest hash，然后创建 RSL-RL wrapper、runner 和 checkpoint metadata factory。
6. 只有最终 manifest 已存在且 hash 已冻结后，才允许开始 rollout 或保存 checkpoint。

环境构造期间 `SimulationContext.reset()` 与 cache 的 `sim.forward()` 不属于 episode reset。第一次完整任务 reset 由 `RslRlVecEnvWrapper` 构造触发；在最终 manifest 写入前不得执行训练 rollout。禁止先写 manifest、环境采样后再原地重写同一个 manifest，否则 checkpoint 中的 manifest hash 会失效。

## 12. CheckpointContract

Phase 1R 创建新契约：

```text
Phase1RandomizedContractV3
```

该契约在 `Phase1RandomizedContractV2` 基础上增加确定性的 root-height alignment 和双 tensor cache 语义；它与历史 nominal `Phase1ContractV4`、历史随机化 `Phase1RandomizedContractV1/V2` 都不兼容。

1. `RandomizationSchemaV1` 和 profile hash。
2. 每个字段的分布、范围、单位、生命周期和 target mask。
3. `q_default_env` 同时用于 reset、零动作 reference 和 observation error reference 的语义。
4. friction bucket 数量、material combine mode 和左右轮共享规则。
5. active/passive joint id 与 canonical order。
6. observation noise 在 normalization 前加入、Actor noisy、Critic clean。
7. root velocity 的 ControlFrameV1 语义及写入变换版本。
8. nominal asset/USD hash 与 actuator nominal 参数。
9. `RandomizationSeedDerivationV1`、固定 stream 名称和 generator/device 规则；不得包含 master seed、派生 seed 或 generator 当前 state。
10. 训练 contract 不包含 `RandomizedStressSeedSetV1`；该 seed set、命令表和聚合规则属于独立的 `RandomizedStressProtocolV1` evaluation contract。
11. implicit torque 语义为 `implicit_pd_estimate`，以及 effort limit 的 PhysX/`effort_limit_sim`/`effort_limit` 三处同步规则。
12. `ClosedChainResetCacheSchemaV2`、`BoundaryClampedPhysXRelaxationV1` 和 `ClosedChainRootHeightAlignmentV1` 的 seed-independent 算法配置、joint masks/order、写入顺序、gravity/root 条件、root-Z 公式、hard-gate 阈值与 artifact 绑定规则。完整 cache tensors 和 seed-dependent hash 不进入语义 contract 本体。

加载规则：

- `Phase1ContractV4 -> CheckpointMetadataV3` 只允许历史诊断/只读评估，不能 resume 到 Phase 1R V3/V6。
- `Phase1RandomizedContractV1 -> CheckpointMetadataV4` 只允许历史诊断/只读评估；其 `passive_position=asset_default`、`passive_targets=none` 语义禁止 resume 到 V3/V6。
- `Phase1RandomizedContractV2 -> CheckpointMetadataV5` 只允许实现期历史诊断/只读评估；其 cache 缺少 `root_height_offset_env`，禁止 resume 到 V3/V6。
- `Phase1RandomizedContractV3` 必须且只允许绑定 `CheckpointMetadataV6`；V3/V6 才允许本文定义的 fresh training 与同版本 resume。
- “V2/V5 只读评估”只表示使用独立历史 reader 做 metadata/model-state 检查或 Actor 导出；若要复现历史环境，必须使用 CacheSchemaV1 和历史 V2 contract 分派。把 V2 Actor 放进 V3 环境属于显式 cross-contract evaluation，必须在输出中标注 source/target，不得称为历史复现。
- randomization profile/schema 或字段语义不匹配时默认 hard fail；hardware profile 名、`device`、`num_envs` 和 `num_mini_batches` 是具名可变运行元数据，不作为模型/schema 兼容性拒绝条件。只允许显式的 evaluation override 跨 randomization contract，并在输出中标注 source/target contract。
- Phase 1R 第一轮正式实验必须从头训练，以避免把 nominal 策略初始化收益误判为随机化训练效果。
- master seed 与派生 seed 只进入 run manifest/checkpoint metadata；七个 generator 当前 state 只进入 checkpoint payload。四个训练 seed 必须共享同一个语义 contract hash。
- randomized checkpoint resume 时必须保存完整七个独立 generator state；同规模恢复全部七流，跨规模按第 11 节的 lifecycle 分组恢复三个 runtime 流并保留四个 target process-start 流，随后强制全环境 reset。缺少任一 state、source cache artifact/hash、跳过 post-restore reset 或改变第 11 节首帧调用顺序时 hard fail。
- 每个 checkpoint metadata 必须绑定 cache artifact 文件 SHA-256 与双字段联合 tensor SHA-256；checkpoint、run manifest 和实际 artifact 三者任一不一致时拒绝加载。
- `CheckpointMetadataV6.closed_chain_reset_cache` 字段表固定为：`relative_path`、`file_sha256`、`tensor_sha256`、`schema_version`、`relaxation_algorithm_version`、`root_height_algorithm_version`、`q_reset_shape`、`q_reset_dtype`、`root_offset_shape`、`root_offset_dtype`。两个 tensor 都必须是 CPU、contiguous、finite，且首维等于 `num_envs`。
- 跨 hardware-profile/`num_envs` resume 时，source checkpoint 先按 source manifest/cache 完整验证；新 run 与其后保存的 checkpoint 只绑定 target cache。source cache 身份保留在 resume metadata 中，不得继续写入 target checkpoint 的 `closed_chain_reset_cache` 主绑定字段。

## 13. 实施前测试与 gate

### 13.1 单元测试

必须覆盖：

1. 每个分布的范围、形状、dtype、device 和 target mask。
2. process-start 参数在 episode reset 后保持不变。
3. episode-reset 参数只更新指定 env id。
4. observation-step noise 每步变化且不写回 clean state。
5. active joint mask 只包含四个腿关节和两个轮关节。
6. default offset 只包含四个主动腿关节。
7. 左右轮 friction 在同一 env 内完全相同。
8. randomization disabled 时输出与当前 nominal path 数值等价。
9. 相同 master seed、环境数量和配置产生相同 realized parameters。
10. command RNG 不随 randomization 字段启停而改变。
11. clean CriticObsV1 与 noisy ActorObsV1 不共享可原地修改的 storage。
12. physical-unit noise 到 normalized tensor 的换算符合 NormalizationV2。
13. `transform_control_vector_to_usd()` 与 `transform_usd_vector_to_control()` 对 batch tensor 双向 round-trip。
14. 相同 master seed 和 stream name 得到相同派生 seed，不同 stream name 得到不同 seed。
15. CPU process-start/reset 子流与 CUDA observation-noise 子流的 generator/device 符合第 11 节。
16. 同一 checkpoint 在相同目标 hardware profile/`num_envs` 下重复 resume 两次时，第 11 节定义的 first real command/root velocity、`post_restore_reset_discarded_policy_obs` 和 `first_policy_observation` 完全一致；`resume_sequence_trace.pt` 的字段、shape、dtype 和 tensor hash 正确。跨 hardware profile 改变 `num_envs` 时 source cache 必须先验证、target cache 必须独立生成并绑定，且不同 run seed 仍共享同一个 contract hash。
17. `PhysicalActorFieldsV1`、`CleanActorFeaturesV1` 和 `ActorObsV1` 的 storage 不别名，且两条 normalization 路径得到预期结果。
18. `ClosedChainResetCacheSchemaV2` 文件/联合 tensor hash、两个 tensor 的 shape/dtype、joint order、两个算法字段和损坏/缺失 artifact hard-fail。
19. suite training fingerprint 排除 seed-dependent realized audit、stream seed 和 cache artifact hash，但保留 profile/schema/算法语义；四个不同 seed 必须得到同一 training fingerprint。
20. `ClosedChainRootHeightAlignmentV1` 对 nominal reference 输出全零，对 randomized reference 输出与规范 CPU float64 FK 公式一致；cache 中补偿被篡改、错序或与规范 float32 重算结果不相等时 hard fail。
21. `ResumeMetadataV2` 两种 cache mode 的字段集合、source/target 主绑定和七流来源映射正确；checkpoint 缺少任一环境 stream 时 hard fail。

### 13.2 Isaac Lab 集成测试

在启动正式训练前，使用少量环境完成以下测试：

- 从 PhysX/Articulation 侧核对 realized friction、stiffness、damping 和 effort limit。
- 断言左右轮 shape 与地面最终解析的 friction combine mode 均为 `multiply`，不能只读回摩擦数值。
- 核对 passive joint gain/limit 未变化。
- 核对 randomized `q_default_env` 的 canonical/USD 重排列没有错位。
- 核对 process-start `BoundaryClampedPhysXRelaxationV1` 的逐步写入顺序、realized actuator 参数、悬空零重力条件和 10 physics steps，并读回最终误差与 reset-cache hash。
- 核对 gravity 在成功和异常路径都恢复且 readback 相同；任务计数器保持零，同时审计底层 physics time 已推进 `0.05 s`。
- 核对全部 joint hard limit、被动 raw/wrapped delta、八关节物理分支签名和 0/1/5/10 步完整 trace。
- 对松弛缓存分别运行第 8.3 节 Gate A 与 Gate B，并单独记录失败归因。
- 核对 episode reset 写入全部关节状态：position 使用 `q_reset_projected_env`，root Z 使用 cache 中的 `root_height_offset_env`，全部 joint velocity 为零，被动关节无 position target，保留冻结的零 velocity target/阻尼语义。
- 核对 Gate A/B 在第一个 physics solve 之前就记录第 0 步碰撞网格最低点，统一 root 高度的故障样本会失败，FK 对齐样本会通过。
- 核对同一 q reference 在 RTX 4060、RTX 5070 和纯 CPU 检查工具中都生成同一 CPU float32 `root_height_offset_env` canonical tensor/hash。
- 核对 PhysX gain/limit、ImplicitActuator mirror 和已知误差下的 torque estimate 使用同一 realized tensor。
- 核对 `Actuator/effort_saturation_fraction` 按每 env live effort limit 计算，不能读取 env-0 快照。
- 关闭 observation noise 时，Actor 和 Critic 前 25D 完全相同。
- 开启 observation noise 时，只有定义的 Actor slice 变化。
- 运行至少 `16 env x 1000 control steps`，无 NaN/Inf、非法接触、约束爆炸或未预期 reset storm。
- 制造部分 env termination，确认 `_get_dones()` 捕获的 transition state 在 `_reset_idx()` 后仅刷新 reset rows；未 reset env 的 Critic joint acceleration 必须继续描述刚返回的 transition，不能因全局 state cache 清空而变成零。
- 检查 reward、termination、command 和 previous action 的原有回归测试仍通过。

process-start relaxation 单独记录 0/1/5/10 步的全部关节状态、闭链残差、FK 误差、主动 reference 误差、被动 delta 和物理分支签名。Gate A 与 Gate B 都固定记录 episode reset 后前 `200 ms`，即 `40 physics steps / 10 control steps`：root pose/velocity、全部关节状态、PD 近似 applied torque、realized effort limit、左右 `L0/phi0`、闭链连接点残差和轮地接触。判定阈值使用第 8.3 节定义；若 nominal 或 randomized profile 失败，应按 gate 类型进入排查，不能简单增加阻尼或静默放宽阈值。训练 episode reset 本身不插入该窗口。

### 13.3 分布审计

训练启动前导出一次只读审计产物：

- profile JSON/manifest。
- 每个 process-start 参数的直方图和统计量。
- reset velocity 的样本统计。
- observation noise 的样本统计及 normalized 后范围。
- closed-chain candidate 总数、接受率、重采次数、拒绝原因及 realized 截断分布。
- 闭链 relaxation/root-height alignment 配置、最终误差统计、cache artifact 文件/联合 tensor SHA-256；两个完整 tensor 保存在 `.pt` artifact，不复制进 JSON manifest。
- `root_height_offset_env` 的 min/max/mean/std，以及按 offset 分桶的 base-height error。

训练 step 内不得做磁盘 I/O；上述审计只在初始化完成后执行一次。

### 13.4 正式训练 gate

只有同时满足以下条件才允许开始 Phase 1R 正式训练：

1. 独立文档复核无阻塞 P0/P1。
2. randomization disabled 与 nominal baseline 回归一致。
3. gain/effort/material 写入得到仿真侧证据。
4. 闭链随机初态通过几何检查和 PhysX smoke gate。
5. clean/noisy observation 分离测试通过。
6. checkpoint contract mismatch 能够 hard fail。
7. 小规模稳定性测试无数值异常。

## 14. 暂缓到后续 profile 的随机化

### 14.1 质量、质心和惯量

必须先重新核对 USD 与实车质量属性。确认名义值后，三者分别定义，不复制复旦“mass 与 diagonal inertia 共用一个 scalar”的实现。

### 14.2 恢复系数

当前 ground restitution 为 `0` 且 combine mode 为 `multiply`。只随机轮端 restitution 不会产生有效变化，因此 V1 暂缓。后续必须同时明确两侧 material 和 combine rule。

### 14.3 Action delay

需要先明确延迟发生在 command、policy output、actuator target 还是总线执行层，并定义整数 control-step buffer。未定义部署通信链路前不加入。

### 14.4 IMU bias 与有色噪声

当前 V1 只有逐步独立白噪声。实机 IMU 的安装变换、零偏、温漂和滤波延迟属于后续传感器模型，不能用白噪声替代。

### 14.5 外力、外力矩与直接 root pose 随机化

V1 先用小范围 reset velocity 产生动态扰动。持续或脉冲外力、直接 roll/pitch 初态以及 fall-recovery 都需要单独安全范围和任务定义。

### 14.6 MuJoCo 参数随机化

MuJoCo stress profile 必须按 MJCF/solver 的实际参数建立独立 schema。目标是测试策略对一组物理不确定性的稳定性，不是选择一组让某个 checkpoint 恰好站住的参数。

## 15. 模块职责与预计代码边界

本文复核通过后，预计只在 `wheelleg_dreamwaq` 内增加或修改以下职责，不改 MuJoCo debug 工具和 USD：

| 模块 | 职责 |
|---|---|
| `schemas/randomization.py` | profile/schema dataclass、版本、manifest 序列化；随机化定义的单一事实来源 |
| `tasks/direct/wheelleg_flat/randomization.py` | RNG 子流、参数采样、闭链候选验证、`BoundaryClampedPhysXRelaxationV1`、cache hard gate、material/actuator/state 应用 |
| `tasks/direct/wheelleg_flat/env_cfg.py` | 选择 profile，不重复声明范围 |
| `tasks/direct/wheelleg_flat/env.py` | 在 process start/reset/observation 生命周期调用 randomization 模块 |
| `schemas/frames.py` | 增加 `transform_control_vector_to_usd()`，保持 ControlFrameV1 双向向量变换的单一事实来源 |
| `tasks/direct/wheelleg_flat/observations.py` | 纯 tensor 函数：reference 支持 `(4,)` 或 `(num_envs,4)`，构造 clean obs 并应用外部传入的 noise；自身不持有 RNG |
| `schemas/manifest.py` | 将 profile contract、RNG、cache schema、root-height alignment 和 implicit torque 语义纳入 `Phase1RandomizedContractV3`；保留 V1/V2/V4 只读识别 |
| `training/checkpoint.py` | 使用 `CheckpointMetadataV6` 保存/恢复环境独立 generator state并绑定双 tensor cache artifact；V3/V4/V5 只读识别 |
| `scripts/train_ppo.py` | fresh 生成/resume 加载 cache；环境 audit 后原子最终化 manifest；冻结真实 resume 首帧顺序 |
| `scripts/run_training_suite.py` | 记录 Phase 1R contract，先完成四次训练，再统一导出并执行四组 MuJoCo nominal sim2sim |
| `tests/unit/test_randomization.py` | schema、采样、生命周期、RNG 和 clean/noisy split 测试 |
| `tests/integration/test_randomization_runtime.py` | 闭链 reset、material/actuator 写入读回和 nominal regression 测试 |
| `tests/unit/test_checkpoint_metadata.py`、`test_manifest.py`、`test_training_suite.py` | 新 contract、RNG state、manifest 时序与四次训练后统一评估的回归测试 |

依赖方向固定为：

```text
schema/config
    |
    v
randomization runtime ---> Isaac Lab objects
    |
    v
environment ---> observation/reward/termination pure tensor functions
    |
    v
RSL-RL PPO
```

`randomization.py` 不得导入 RSL-RL；`observations.py` 不得访问 USD/PhysX prim；部署代码不得依赖 randomization 或 Isaac Lab。

按 v0.7 范围估计，核心实现与训练编排约 `400-650` 行，测试约 `280-440` 行，总计约 `680-1090` 行。新增部分是 process-start 边界钳制松弛、root-height alignment、完整双 tensor cache artifact/恢复绑定、装配分支 hard gate、partial-reset transition 修复与审计 hash；不包含通用解析 IK。若 10-step relaxation 在正式环境规模下不能稳定通过，必须另行设计 IK，不能把它隐藏在本任务估算中。

## 16. 主要风险与处理

| 风险 | 后果 | 处理 |
|---|---|---|
| 闭链主动关节随机、被动关节仍为默认 | 已实测产生 `12.225 mm` 残差 | 禁止该 reset；process-start 生成并验证完整松弛 cache |
| 闭链松弛结果受 asset/solver 漂移 | checkpoint 仍能加载但 reset 几何已变化 | cache artifact 进入 manifest/checkpoint 双 hash；同规模 resume 精确复用，跨规模 resume 先验证 source 再生成并绑定通过 hard gate 的 target artifact |
| 几何残差小但进入错误装配分支 | 训练从非物理构型开始且后续不可解释 | hard limit + raw/wrapped delta + 八关节符号分支 gate |
| 把逐 step 边界钳制误写成 fixed joint | 实现和复核使用不同动力学假设 | 冻结 `BoundaryClampedPhysXRelaxationV1` 的完整写入顺序和 realized actuator 参数 |
| relaxation 推进底层时间 | 绝对时间 sensor/event 提前触发 | V1 禁止绝对时间依赖；计数器/physics time 分别审计，gravity 用 try/finally 恢复 |
| actuator 参数只在 Python tensor 中变化 | 实际物理未随机 | 仿真侧读回/响应测试 |
| 只写 PhysX，未同步 ImplicitActuator mirror | torque reward、Critic 和饱和日志与真实响应不一致 | gain/limit 双写并做 torque-estimate 测试 |
| 使用 env-0 effort-limit 快照 | per-env 饱和日志与退出指标失真 | 删除共享快照，始终读取 live per-env limit |
| material 数值正确但 combine mode 漂移 | 有效摩擦范围静默改变 | 读取/断言 shape 与 ground 的最终 combine mode |
| Actor/Critic 共用同一 noisy tensor | privileged critic 被意外污染 | clean tensor 先构造，Actor copy 加噪 |
| noise 在 normalization 后加入 | 量级错误 | 固定 physical noise -> normalization 顺序 |
| 共用全局 RNG | 改一个字段导致命令序列全变 | 独立 seed substream |
| 独立 RNG 未进入 checkpoint | resume 后命令、reset velocity 和 noise 序列跳变 | checkpoint 保存/恢复七个 generator state |
| resume 首帧调用顺序未冻结 | observation-noise 错一帧，重复实验不可比较 | 明确 discarded reset obs 与 first policy obs，并按真实 wrapper/runner 顺序测试 |
| partial reset 清空全局 transition cache | 未 reset env 的 Critic joint acceleration 被错误置零 | reset 只刷新 reset rows，保留 active rows 的终止前 transition state |
| manifest 在环境采样前冻结 | realized 参数缺失或事后重写使 hash 失效 | 环境 audit 后原子最终化 manifest，再创建 runner |
| nominal/randomized checkpoint 混载 | 实验不可解释 | 新 contract + mismatch hard fail |
| 当前 USD 名义质量不可信 | 质量随机化扩大模型错误 | V1 禁止质量/COM/inertia 随机化 |
| 随机化掩盖 sim2sim 接口错误 | 两边都不稳定且难定位 | nominal regression 与 sim2sim 接口测试继续保留 |
| 把保持高度当成自主起立 | 实车目标被错误外推 | 文档明确 V1 仍从接近站立初态开始 |
| 范围过大导致早期 PPO 无有效样本 | 学习停滞 | V1 使用保守范围；范围升级必须新 profile |

## 17. Phase 1R 退出条件

达到训练 iteration 数不自动表示通过。进入 DreamWaQ 前至少满足：

1. 多 seed 训练无 NaN、reward collapse 或大规模 reset storm。
2. Isaac Sim nominal 指标不显著低于当前 nominal PPO 基线。
3. Isaac Sim randomized stress 的 survival、姿态和命令跟踪达到独立冻结的验收阈值。
4. MuJoCo nominal sim2sim 中能够完成站立、前进、后退和转向测试；不能只挑选单个最有利场景。
5. base height、roll/pitch/yaw、左右 phi0 差、`vx/yaw_rate` 跟踪和 effort saturation 均有统一日志。
6. 所有评估使用固定 protocol 和 deterministic policy，模型选择不读取训练 seed 特供参数。
7. 失败结果能够区分接口错误、随机化范围错误和策略容量/训练问题。

具体数值阈值必须根据当前 nominal PPO 的 Isaac/MuJoCo 统一评估结果单独冻结；它们是 Phase 1R 实验 gate，不在没有基准数据时由本文猜测。

Phase 2 的公平对照固定为：在同一 `FudanStyleDomainRandomizationV1`、同一 Actor observation noise 和同一 41D clean Critic 条件下，将普通 PPO 替换为带五帧历史、CENet、下一帧重建和 KL loss 的 DreamWaQ PPO。DreamWaQ 的 latent reparameterization noise 与本文 sensor-style observation noise 是不同机制；Phase 2 只评估新增历史/CENet/辅助损失的增量效果。

## 18. Architecture Decision Record

| ID | 决策 |
|---|---|
| ADR-R01 | 在名义 PPO 与 DreamWaQ 之间插入普通 PPO 随机化阶段 Phase 1R。 |
| ADR-R02 | V1 复用复旦的随机化类别、均匀采样和生命周期思路，不复制 Isaac Gym 接入层。 |
| ADR-R03 | V1 不直接随机初始 root roll/pitch/yaw。 |
| ADR-R04 | 四个主动腿关节 default offset 同时定义 reset、零动作 target 和 observation reference。 |
| ADR-R05 | 闭链候选必须先通过 FK，再经过 process-start `BoundaryClampedPhysXRelaxationV1` 和 episode reset gate；任一失败均阻塞训练。 |
| ADR-R06 | Actor noise 在物理量域加入，再执行现有 normalization。 |
| ADR-R07 | Critic 保持 41D 并使用 clean observation，不预先加入随机参数。 |
| ADR-R08 | V1 不随机质量、COM、惯量、恢复系数、延迟和外力。 |
| ADR-R09 | Phase 1R 正式实验从头训练，不从 nominal checkpoint warm-start。 |
| ADR-R10 | nominal evaluation 与 randomized stress evaluation 分开报告。 |
| ADR-R11 | 每类随机量使用独立 RNG 子流，command RNG 不受影响。 |
| ADR-R12 | 主架构 v0.15 已同步 Phase 1R 阶段、V6 checkpoint、双 cache resume 与对应 gate；两份文档必须共同通过增量复核后实施。 |
| ADR-R13 | implicit actuator 随机参数必须同时同步 PhysX 与 actuator mirror tensor。 |
| ADR-R14 | root reset 写入值定义为 world frame root COM velocity，并由具名 ControlFrameV1 逆变换生成。 |
| ADR-R15 | effort saturation 指标必须使用 live per-env realized limit。 |
| ADR-R16 | master seed 默认绑定最终生效的 cfg.seed，子流使用 SHA-256 派生并记录 generator/device。 |
| ADR-R17 | 闭链候选筛选后的 realized offset 是截断分布，必须记录接受率和实际统计量。 |
| ADR-R18 | process start 为每个 env 生成或加载完整 `q_reset_projected_env` 与 `root_height_offset_env`；episode reset 写入 position cache、对齐后的 root Z 和全零 joint velocity。 |
| ADR-R19 | 10-step 悬空零重力边界钳制松弛在 fresh process start 与跨 `num_envs` resume 生成 target cache；同规模 resume 精确复用 source artifact；200 ms Gate A/B 只属于独立 preflight；episode reset 不增加 settling。 |
| ADR-R20 | 语义 contract 必须 seed-independent；master/派生 seed 属于 run manifest，七个 generator state 属于 checkpoint payload。resume 恢复后必须全环境 reset，再开始 rollout。 |
| ADR-R21 | run manifest 只能在环境完成 process-start sampling/audit 后原子最终化，runner 在最终 hash 冻结后创建。 |
| ADR-R22 | Actor/Critic 构造固定为 physical-clean、normalized-clean 和 physical-noisy-normalized 两条无别名路径。 |
| ADR-R23 | `realized_plan_hash` 只标识 seed 决定的随机计划；派生的闭链 reset cache 使用独立 artifact/tensor hash，不混淆 RNG 与 solver 浮点结果。 |
| ADR-R24 | `ClosedChainResetCacheSchemaV2` 双 tensor artifact 必须同时绑定当前 run manifest 与 checkpoint；同规模 resume 缺失或不匹配时 hard fail且不重新生成替代，跨规模 resume 验证 source 后生成并绑定具名 target artifact。 |
| ADR-R25 | 闭链 cache hard gate 同时检查几何残差、全关节状态、被动角邻域和八物理关节装配分支。 |
| ADR-R26 | resume 第一份策略观测是 post-restore reset 后第二次 observation-noise draw；第一次 reset 返回值明确丢弃并纳入可复现测试。 |
| ADR-R27 | partial episode reset 只能刷新 reset rows，不能破坏未 reset env 的终止前 transition/acceleration 语义。 |
| ADR-R28 | `ClosedChainRootHeightAlignmentV1` 用左右 wheel-axis 向下伸长增量的最大值确定每 env root Z；它是 reset 几何修正，不是独立 root pose 随机化。 |

## 19. 独立复核重点

复核者应重点回答：

1. 复旦源码范围、生命周期和 observation noise 换算是否准确。
2. `q_default_env` 的三个语义是否与当前 ActionV1/ActorObsV1 一致。
3. `BoundaryClampedPhysXRelaxationV1` 是否忠实描述真实写入/solve 顺序，10-step 证据是否足以支持 cache；装配分支 hard gate、artifact 绑定和失败处理是否完整。
4. `ClosedChainRootHeightAlignmentV1` 的向下伸长公式、左右取最大值、第 0 步接触 gate 和双 tensor cache 是否足以避免统一 root 高度造成的嵌地，同时没有把它误定义为独立 root pose 随机化。
5. Isaac Lab 2.3.2 的按 env writer 与 ImplicitActuator mirror 双写方案是否完整，是否还存在缓存未同步。
6. ground/material combine 语义是否确实产生 `[0.6,1.4]` 有效摩擦。
7. Actor noisy、Critic clean 的张量构造是否会破坏当前 41D contract。
8. root velocity 的坐标系和写入接口是否定义完整。
9. 哪些字段必须进入 checkpoint/manifest 才能避免误加载。
10. disabled profile 是否能严格复现当前 nominal behavior，以及是否存在会阻塞实施的 P0/P1，而不是仅属于后续可调超参数的问题。

## 20. 当前结论与实施前实证阻塞项

`FudanStyleDomainRandomizationV1` 的范围已经足够形成独立实现任务，不需要为了“看起来完整”提前加入 DreamWaQ、质量随机化或外力扰动。

v0.4 假设的“主动 randomized + 被动 nominal”闭链 reset 已被真实运行否定，不能再作为开放问题。v0.6 的 `BoundaryClampedPhysXRelaxationV1` 在 8 env、seed 42 下已把最大闭链残差从 `12.225 mm` 降到 `0.0008 mm` 以内；v0.7 又用失败/nominal/修正三组 trace 证明统一 root 高度是轮端嵌地根因，并验证 `ClosedChainRootHeightAlignmentV1` 能关闭 Gate A/B。正式训练前仍需在生产路径关闭以下三个实证 gate。

第一，完整闭链 reset/cache 路径：在正式环境代码中生成 fresh 双 tensor cache、保存 artifact，或在 resume 时加载 source artifact；通过几何、全关节、root-height 重算和装配分支 hard gate，然后分别完成包含第 0 步的零 root velocity Gate A 和随机 root velocity Gate B。必须证明 episode reset 本身不推进隐藏 physics step，且不同 episode 复用的被动关节状态没有跨 episode 污染。

第二，implicit actuator 同步：PhysX 中的 stiffness/damping/effort limit、`ImplicitActuator.stiffness/damping`、`effort_limit_sim` 和 `effort_limit` 必须使用同一 realized tensor，并证明物理响应、PD 近似 `applied_torque`、torque reward 和 saturation 日志语义一致。若不能一致，执行器随机化必须暂缓，不能带着错误力矩信号训练。

第三，partial-reset transition 与 resume 首帧：正式 smoke 必须证明部分 env reset 不会把未 reset env 的 Critic joint acceleration 清零；重复 resume 必须按第 11 节真实调用序列得到一致的 command、root velocity、discarded observation 和 first policy observation。

这三个实证问题都是正式训练前 gate。v0.7 冻结了 reset、root-height 与 Actor/Critic 数据流基础契约；v0.8 进一步冻结双 cache resume、RNG 生命周期、checkpoint provenance 和 manifest 时序。增量复核无 P0/P1 后，才允许完成实现并启动四次正式训练。
