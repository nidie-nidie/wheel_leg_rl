# WheelLeg Isaac Sim / MuJoCo Sim2Sim 差异诊断设计

**日期：** 2026-10-05  
**修订：** v1.5，记录正式门禁、跨引擎采集结果、即时通道激励和首处分歧结论  
**状态：** 诊断实现、独立代码复核和正式采集已完成；首处分歧已定位到 control tick 0 的物理响应层，Isaac 接触 observer 当前标记为 `unavailable`  
**适用范围：** 第一轮 PPO checkpoint、固定站立命令、Isaac Sim 与 MuJoCo 的同策略对照  
**诊断脚本目录：** `wheelleg_dreamwaq/debug/sim2sim/`  
**诊断产物目录：** `wheelleg_dreamwaq/artifacts/debug/sim2sim/<run-id>/`

## 1. 文档目的

当前同一个 PPO Actor 在两个仿真器中的结果明显不同：

- Isaac Sim 中，第一轮 checkpoint 在固定站立命令下可完成 500 个控制周期。
- MuJoCo 中，同一个 checkpoint、同一个 25 维观测契约和同一个站立命令在正式 trace 的第 270 行因倾斜终止；旧汇总只统计终止前 269 行。

本轮工作的目标不是先把 MuJoCo 参数“调到能站住”，而是建立一套可重复、可审计的跨引擎诊断流程，找到两个系统中**第一个出现有意义差异的信号、控制周期和处理层级**。

只有在差异来源被定位后，才能对执行器、接触、闭链约束、模型参数或适配器做单变量修正。禁止同时修改多个参数后仅凭“看起来更稳”判断问题已经解决。

## 2. 已确认决策

本设计固定以下决策：

1. 第一轮只使用训练 Run 01 的 `model_999.pt`，不同时测试四个 checkpoint。
2. 第一轮只测试站立，不测试前进、后退、转向或组合命令。
3. 命令在整个 episode 内保持为 `[vx=0.0, yaw_rate=0.0, base_height=0.20]`。
4. 使用 1 个环境，运行 500 个控制周期，即 10 秒。
5. 关闭域随机化、命令重采样和人为扰动。
6. 使用现有 MuJoCo 模型 `wheel_leg_urdf4_v1.xml`，本轮不重建或修改 XML。
7. 不修改 USD、奖励、PPO 网络或训练参数。
8. 所有新诊断脚本放在 `wheelleg_dreamwaq/debug/sim2sim/`，不散落到训练脚本、MuJoCo 正式运行包或仓库根目录。
9. 诊断脚本默认只读取现有环境、模型、policy 和 manifest；允许在 `debug/sim2sim/` 内建立独立的 debug-only Isaac 配置和只读 observer，但不得改变生产控制语义或修改共享配置对象。
10. 本轮的完成标准是定位差异，不是要求两个物理引擎长期轨迹完全重合。
11. Isaac 接触采集只有在“无传感器基线”和“启用 ContactReportAPI 的 debug 配置”达到 `instrumented_equivalent` 后才可用于归因；`instrumentation_perturbation` 只保留插桩 trace 供审计，跨引擎非接触比较继续使用无传感器基线；`instrumentation_failed` 不得作为跨引擎可比较运行，API 或配置不可用时标记 `unavailable`。

## 3. 已知事实和问题基线

### 3.1 固定 checkpoint

```text
wheelleg_dreamwaq/logs/rsl_rl/wheelleg_flat_ppo/
  2026-10-05_07-11-36_rtx5070_v4-suite-20261005-071132-run01-seed250509479/
    model_999.pt
```

固定身份：

| 项目 | 值 |
|---|---|
| checkpoint SHA256 | `04E040272EE3D2682DEFA053062D1EBE9AB3D13652DBC8D4760773A23A58197E` |
| Actor TorchScript SHA256 | `5B714AF635AE3C93BD86853BD9141F0B8888BB49C0F275CBCCD531B6AF53B9BC` |
| Phase1 contract hash | `69B5D4AD7CDEB5FF94BE3CC68E2BCE6338AB29F07DE3192EA93247545664600A` |
| source run manifest SHA256 | `CD3DA566179885DD628A09DC66BD57B17F177C00018FA1EF4F93EABD867A87C7` |
| asset bundle version | `AssetBundleV2` |
| asset bundle hash | `E754AE888F5C5379B3B6152CFA5AD6BBAE20E8C7480CF7AD6960782A5267CB46` |

### 3.2 固定 MuJoCo 模型

```text
wheelleg_dreamwaq/sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml
```

固定身份：

| 项目 | 值 |
|---|---|
| model version | `MujocoModelV1` |
| model XML SHA256 | `691C607271ED85EFFA388237E00A7B8F8F41CB66F6BA9C49EA64ABD186D803D1` |
| model manifest SHA256 | `C3DB4FE2797F6AC3628F0A9CB5E0B166FFD0CE802776A335922C474A045F7AE0` |
| dynamics semantics hash | `A846A8E4E198A43DAE8F8D9BD2BA565F7B8DC2C67386FB0028BD2530FE1C42D6` |
| MuJoCo version | `3.14.0` |
| compiled structure | `nbody=28, njnt=27, nq=33, nv=32, nu=6, neq=8, npair=2` |

本轮必须直接使用该模型和对应 manifest。任何 XML、mesh、equality、contact、solver 或 actuator 参数变化都必须生成新的模型身份，不得继续沿用上述 hash。

### 3.3 当前结果

Isaac Sim 固定站立评估：

- 完成 `500/500` 个控制周期。
- 没有 tilt、height、root velocity、joint velocity 或 invalid termination。
- 平均高度绝对误差约 `0.003716 m`。
- 平均 `vx` 绝对误差约 `0.06456 m/s`。
- 平均 yaw rate 绝对误差约 `0.01904 rad/s`。
- action saturation fraction 约 `0.000333`。
- effort saturation fraction 约 `0.001`。

MuJoCo 固定站立评估：

- 正式 terminal-safe trace 共保存 `270/500` 行，第 270 行触发 tilt；旧评估汇总只统计终止前 `269` 行，存活比例写为 `0.538`。
- 汇总中的最大 tilt 约 `0.79854 rad`，主要来自 pitch；该汇总只统计终止前有效行，不包含触发 `tilt > 0.80 rad` 的终止 tick。逐 tick trace 和本诊断后续汇总必须包含终止 tick。
- 高度平均绝对误差约 `0.002063 m`。
- `phi0_left - phi0_right` 的 RMS 约 `0.03231 rad`。
- 最大闭链残差约 `0.0001455 m`。

这些结果只能证明端到端行为不同，不能证明根因是执行器、接触、闭链、坐标系还是策略输入。高度误差较小也不能说明姿态动力学正确。

## 4. 诊断问题

本轮必须回答以下问题：

1. 两个仿真器在 policy 第一次推理前的机器人状态是否一致？
2. 同一个规范化前状态是否构造出相同的 25 维 ActorObsV1？
3. 同一个 25 维输入是否产生相同的 6 维 Actor 输出？
4. 同一个 6 维动作是否转换成相同的四个腿位置目标和两个轮速度目标？
5. 在相同关节状态和目标下，两侧可观测的执行器量分别是什么，第一控制周期后的关节和机身状态响应从什么时候开始不同？
6. 在策略反馈被移除、使用相同动作序列时，两个物理系统从哪个状态量开始分离？
7. 差异最先出现在主动关节、被动闭链关节、轮地接触还是机身状态？
8. MuJoCo 的倾斜是早期小偏差的累积，还是某个接触、约束或饱和事件导致的突变？

## 5. 非目标

第一轮明确不做：

- 不重新训练 PPO。
- 不修改 `phi0` 奖励、base height 奖励或任何奖励权重。
- 不修改 USD 或 MJCF。
- 不调整 Kp、Kd、effort limit、armature、passive damping、摩擦或 solver。
- 不把 MuJoCo 的显式 PD 直接替换成另一套控制器。
- 不测试另外三个 checkpoint。
- 不做前进、后退、转向或组合运动排名。
- 不接入 DreamWaQ、CENet 或部署侧代码。
- 不以可视化主观观感代替 trace 证据。
- 不要求 PhysX 和 MuJoCo 每个物理子步逐点一致。

## 6. 固定契约

### 6.1 控制时钟

| 项目 | Isaac Sim | MuJoCo |
|---|---:|---:|
| policy/control dt | `0.020 s` | `0.020 s` |
| policy frequency | `50 Hz` | `50 Hz` |
| physics dt | `0.005 s` | `0.001 s` |
| physics steps/control tick | `4` | `20` |

两个引擎只在 20 ms 控制边界上天然对齐。不得把 Isaac 的第 1 个 5 ms 子步与 MuJoCo 的第 1 个 1 ms 子步称为等价采样点。

子步分析采用两种形式：

1. 保存两个引擎各自的全部原始子步 trace。
2. 在每个控制周期内计算 `min/max/mean/final/integral` 摘要，再在相同控制周期之间比较。

力矩积分固定为：

```text
torque_impulse[t, j] = sum_k(tau[t, k, j] * physics_dt)
```

不通过插值制造不存在的“共同 1 ms 轨迹”。

### 6.2 ActorObsV1

Actor 输入固定为 25 维：

```text
0:3    control frame 下的机身角速度
3:6    control frame 下的 projected gravity
6:9    CommandV1 [vx, yaw_rate, base_height]
9:13   四个主动腿关节 q - q_nominal
13:19  六个 canonical 主动关节速度
19:25  上一个控制周期的 clipped ActionV1
```

Actor 不输入真实线速度、真实高度或欧拉角。诊断 trace 可以额外记录这些真值，但不得把它们加入 policy 输入。

本诊断把观测分成两层：

- `actor_obs_physical`：完成坐标、顺序和符号转换后，尚未应用 NormalizationV2 scale/clip 的物理量。
- `actor_obs_policy`：应用与训练一致的逐项 scale/clip 后，实际送入 Actor 的 25 维 float32 张量。

当前 Actor 没有 running mean/variance observation normalizer。`actor_obs_policy` 不得再做第二次统计归一化。

`previous_action` 的时序固定为：在 tick `t` 构造观测时使用 tick `t-1` 的裁剪动作；reset 后首帧为六维全零。Actor 推理并裁剪本 tick 动作后，才更新缓存供下一 tick 使用。

### 6.3 ActionV1

Canonical 顺序固定为：

```text
[jIJ, jIO, jAB, jAG, jwheel_left, jwheel_right]
```

转换固定为：

```text
clipped_action = clamp(actor_output, -1, +1)
leg_target = q_nominal + 0.35 * clipped_action[0:4]
wheel_velocity_target = 25.0 * clipped_action[4:6]
```

其中：

```text
q_nominal = [-0.33367134, 0.33367134, -0.33367134, 0.33367134]
```

右轮 USD/MuJoCo native 符号为 `-1` 的模型映射必须显式记录。六个主动关节统一使用：

```text
canonical_from_engine_native = [1, 1, 1, 1, 1, -1]
```

所有名称未特别声明为 native 的跨引擎公共字段都使用 canonical `ActionV1/ControlFrame` 语义。目标、主动关节 `q/qd`、PD 参考值、Isaac host estimate 和 MuJoCo commanded torque 必须同时保存 `*_canonical` 与 `*_engine_native`；右轮的速度目标、速度反馈和广义力矩都应用同一个 `-1` 映射。`all_hinge_*_named[26]` 是按共享名称重排的 engine-native 坐标，不对被动关节猜测额外符号。禁止只保存处理后的结果而无法回查符号转换。

### 6.4 执行器参考值

| 通道 | 控制目标 | Kp | Kd | effort limit | armature |
|---|---|---:|---:|---:|---:|
| 四个腿主动关节 | position target | 120 | 4 | 18 Nm | 0.05 |
| 两个轮主动关节 | velocity target | 0 | 0.6 | 9 Nm | 0.05 |

被动关节使用 `damping=0.05`、`armature=0.005`。

本表冻结的是当前 checkpoint 的仿真身份，用于解释现有 Isaac/MuJoCo 差异；其中轮端 `9 Nm` 不表示已经确认符合实车约 `2 Nm` 的能力。任何实车参数修正属于差异定位后的新契约和单变量实验，不在本轮静默修改。

Isaac Sim 当前使用 PhysX `ImplicitActuator` 位置/速度目标，MuJoCo 当前每个 1 ms 子步显式计算 PD/速度伺服力矩。两侧不存在严格同义的“引擎实际力矩”公开读数：

- Isaac Lab 的 `robot.data.applied_torque` 是宿主侧根据当前 `q/qd/target` 计算并裁剪的近似 PD 镜像，用于奖励和诊断；PhysX 不公开 implicit drive 求解器内部真实力矩。
- MuJoCo 的适配器显式计算力矩，经过 effort 裁剪和自定义 velocity guard 后写入 `data.ctrl`，这是实际提交给 MuJoCo actuator 的命令力矩。
- 因此 Isaac 宿主侧镜像与冻结 PD 公式接近只能作为公式、读取时点和 buffer 刷新的不变性校验，不能证明两个物理引擎的执行器作用相同。

trace 必须按语义拆分以下字段；下列名称描述 canonical 公共量，每项另存对应 `*_engine_native`。不适用于某个引擎的字段使用 `NaN` 并在 metadata 声明：

- `target_command_canonical`：四个腿位置目标和两个轮速度目标；`target_command_engine_native` 保存实际写入对应引擎的六维目标。
- `pd_torque_unclipped_canonical`：使用本子步施力前 canonical `q/qd` 计算的冻结 PD 公式未裁剪结果。
- `pd_torque_effort_clipped_canonical`：仅经过 effort limit 后的冻结参考结果；两项同时保存 engine-native 版本。
- `isaac_host_pd_torque_estimate_canonical`：Isaac Lab 在本子步 `scene.write_data_to_sim()` 内，使用同一个施力前状态计算的 `ImplicitActuator.applied_effort`/`robot.data.applied_torque` 宿主侧近似值；另存 engine-native 版本，MuJoCo 侧为 `unavailable`。
- `mujoco_commanded_torque_canonical`：MuJoCo 使用本子步施力前状态完成 effort 裁剪和 velocity guard 后、写入 `data.ctrl` 的最终命令力矩；另存 engine-native 版本，Isaac 侧为 `unavailable`。
- `physx_internal_drive_torque`：当前公开 API 不可得，固定为 `unavailable`，不得通过其他近似量冒充。
- `effort_limit_event`：冻结 PD 参考值发生 effort 裁剪的事件，两侧都可按同一公式重算。
- `joint_velocity_limit_exceeded`：`abs(qd) >= velocity_limit` 的统一状态谓词，不代表求解器一定触发保护。
- `mujoco_velocity_guard_active`：MuJoCo 适配器的显式门控事件；Isaac 侧为 `unavailable`。

参考力矩公式固定为：

```text
leg_tau_ref   = 120 * (q_target - q) - 4.0 * qd
wheel_tau_ref = 0.6 * (qd_target - qd)
```

随后先按各通道 effort limit 得到 `pd_torque_effort_clipped`。MuJoCo 再执行当前适配器的显式 velocity guard：当 `abs(qd) >= velocity_limit` 且力矩继续推动同方向加速时，将该通道力矩置零。Isaac 的 `velocity_limit_sim` 则交给 PhysX solver 约束并可能主动制动，宿主侧近似力矩不包含该求解器效果，也没有同义的公开触发事件。

D2 的执行器主证据必须是目标一致前提下第一个控制周期后的 `q/qd`、被动闭链量、机身状态和经验证可用的接触响应。跨引擎不得直接对 `isaac_host_pd_torque_estimate` 与 `mujoco_commanded_torque` 设置“相等”验收门限。所有 PD/torque 不变性检查只能使用同一子步的 `state_pre_step`，不能拿推进后的 `state_post_step` 反算后与该子步命令比较。

### 6.5 跨引擎诊断坐标

Isaac 与 MuJoCo 的绝对 world 原点不同，MuJoCo freejoint/body frame 也不等于 base COM。因此绝对 `world position` 只能用于各自引擎内部审计，不能直接相减。

跨引擎位置比较统一使用诊断坐标：

```text
delta_p_engine(t) = p_com_world(t) - p_com_world(t0)
delta_p_diag(t)   = R_diag_from_engine_world * delta_p_engine(t)
p_com_diag(t)     = [delta_p_diag.x, delta_p_diag.y, base_height(t)]
```

其中 `t0` 是第一次 policy 推理前的 COM 状态。本轮 scenario 固定 `initial_yaw_rad=0.0`，因此诊断矩阵明确取对应 manifest 中的 engine-world 到 ControlFrame 旋转：

```text
R_diag_from_isaac_world  = r_control_from_usd
R_diag_from_mujoco_world = r_control_from_mujoco
```

D0 必须验证两矩阵数值一致，并用 engine world 的 `+X/+Y/+Z` 三个基向量做 golden-vector 测试，禁止误用 `r_control_from_imu`。如果后续 scenario 允许非零初始 yaw，必须把初始 root 姿态显式合入 `R_diag_from_engine_world`，不能继续直接复用上述简式。z 分量使用 COM 到地面顶面的高度，避免不同模型参考原点污染比较。

trace 同时保存 engine-native 的绝对 COM 位置和上述 `p_com_diag`。跨引擎位移、速度、接触点和首差异判定全部使用诊断坐标；绝对 engine world 坐标不参与数值等价判断。

平地站立场景的 `base_height` 固定为 base COM 到地面顶面 `z=0` 的垂直距离。`L0/phi0` 使用现有 `VirtualLegKinematicsV1` 的真实髋轴到真实轮轴定义，在各自引擎中先转到同一 ControlFrame；`phi0` 比较使用角度环绕后的最短差值。`loop_closure_error[2]` 分别表示左右腿各自 connect 约束残差的最大欧氏距离。

## 7. 目录设计

新增目录固定为：

```text
wheelleg_dreamwaq/
  debug/
    sim2sim/
      README.md
      trace_schema.py
      stand_scenario.py
      isaac_debug_env.py
      collect_isaac_trace.py
      collect_mujoco_trace.py
      compare_traces.py
      tests/
```

职责如下：

| 文件 | 职责 |
|---|---|
| `README.md` | 记录两个独立 Python 环境下的执行顺序、命令和产物说明 |
| `trace_schema.py` | 定义纯 Python/NumPy 的 `Sim2SimDebugTraceV1` 字段、shape、dtype 和校验器 |
| `stand_scenario.py` | 生成固定站立 scenario、动作序列和身份清单，不访问任何仿真器 |
| `isaac_debug_env.py` | 通过嵌套配置复制构造 debug-only Isaac 环境、ContactSensor 和子步采集子类；不得修改共享正式配置 |
| `collect_isaac_trace.py` | 在 Isaac 环境中采集 reset、控制边界和 PhysX 子步信息 |
| `collect_isaac_production_reference.py` | 使用未覆盖的生产 `WheelLegFlatEnv.step()` 生成 observer 等价门参考轨迹 |
| `collect_mujoco_trace.py` | 使用现有 XML、manifest 和 TorchScript Actor 采集对应 MuJoCo trace |
| `extract_replay_actions.py` | 从 Isaac 闭环 trace 提取 clipped action，并写入来源 hash sidecar |
| `compare_isaac_equivalence.py` | 按自然复现包络比较生产 step 与 debug observer，正式门要求每条路径至少 5 次 |
| `compare_isaac_replay.py` | 比较 Isaac 闭环原轨迹与全新 reset 的同动作开环回放，正式门要求每条路径至少 5 次 |
| `evaluate_actor_cpu.py` | 在单个独立 Python/Torch 环境中读取序列化输入并输出带 hash 的 CPU Actor 结果 |
| `verify_actor_cpu_identity.py` | 将同一序列化 25 维 float32 输入送入同一 CPU TorchScript Actor，执行 `1e-6` 恒等门 |
| `compare_traces.py` | 做字段对齐、误差统计、首差异定位和报告生成，不导入 Isaac 或 MuJoCo |
| `tests/` | schema、时序、映射、hash、比较器和合成首差异测试 |

`trace_schema.py`、`stand_scenario.py` 和 `compare_traces.py` 不得导入 Isaac Lab、Isaac Sim、RSL-RL 或 MuJoCo。两个 collector 分别在现有训练环境和 MuJoCo 独立环境中运行，通过文件交换数据，不要求合并 Python 依赖。

首轮实现不得修改正式训练目录或 `sim2sim/mujoco/wheelleg_mujoco/` 的行为。`configclass.replace()` 底层是浅层 `dataclasses.replace()`，因此不允许只复制顶层对象后原地修改嵌套 `spawn`。启用接触报告必须使用嵌套替换，至少满足以下结构：

```python
debug_robot_cfg = WHEELLEG_CFG.replace(
    spawn=WHEELLEG_CFG.spawn.replace(activate_contact_sensors=True),
)
debug_env_cfg = WheelLegFlatEnvCfg()
debug_env_cfg.robot_cfg = debug_robot_cfg
```

禁止直接修改 `WHEELLEG_CFG`、`WHEELLEG_CFG.spawn`、`WheelLegFlatEnvCfg.robot_cfg` 或正式环境实例持有的共享配置对象。实现后必须测试正式配置中的 `activate_contact_sensors` 仍为 `False`，且 debug 配置与正式配置的 `spawn` 对象身份不同。

启用 Isaac ContactSensor 会添加 PhysX ContactReportAPI，并执行可能写入 rigid-body sleep threshold 的激活逻辑。它不是可预设为轨迹中性的纯 observer。D0 必须记录无传感器和插桩配置中各刚体实际解析后的 ContactReportAPI、sleep threshold 和 contact processing 状态，不能仅根据配置对象推断运行时结果。

左右轮法向接触采集固定使用两个独立 `ContactSensorCfg`，因为 Isaac Lab 的 filtered contact 只支持单个 sensor body 对多个 filtered body：

- 左轮 `prim_path={ENV_REGEX_NS}/Robot/jwheel_left`，右轮 `prim_path={ENV_REGEX_NS}/Robot/jwheel_right`。
- 两者 `filter_prim_paths_expr=["/World/Ground"]`。
- `track_contact_points=True`、`track_friction_forces=False`，并显式填写正确字段名 `max_contact_data_count_per_prim`。
- `force_matrix_w` 和 `net_forces_w` 只解释为法向接触力，不得冒充包含摩擦的总接触力。
- `wheel_contact_active` 统一使用 `norm(normal_force) > 1.0 N`；原始法向力始终保留。
- 第一版若未建立并通过等价门的全机身接触枚举 observer，Isaac 的 `unexpected_contact` 固定为 `-1=unavailable`，不得仅凭两个轮传感器推断“没有其他部位接触”。

必须先对同一 reset 和同一固定动作序列运行无传感器基线与有传感器 debug 配置，比较至少 10 tick 的 q/qd、base pose/velocity 和闭链量。只有满足第 11 节“自然复现包络统一规程”时，接触 observer 才获准用于 D3/D4；否则按该规程降级，比较器不得据此自动归因。

首版 Isaac 子步采集固定使用 `isaac_debug_env.py` 中的 debug-only `DirectRLEnv` 子类，完整复现生产 `step()`。本地 Isaac Lab 2.3.2 没有本设计可直接依赖的公开 physics callback API，因此首版不探索底层 `omni.physx` 回调。该子类只属于诊断层，仍不得修改或 monkey-patch 正式环境类。

## 8. 产物结构

每次诊断生成一个不可覆盖的 run 目录：

```text
wheelleg_dreamwaq/artifacts/debug/sim2sim/<run-id>/
  scenario.json
  action_sequences.npz
  isaac/
    metadata.json
    reset_snapshot.json
    control_trace.npz
    substep_trace.npz
    file_hashes.json
  mujoco/
    metadata.json
    reset_snapshot.json
    control_trace.npz
    substep_trace.npz
    file_hashes.json
  comparison/
    summary.json
    first_divergence.csv
    per_signal_metrics.csv
    control_tick_plots/
    substep_summary_plots/
```

所有 JSON 使用排序 key；浮点时间以整数 `control_tick`、`substep_index` 和明确 `dt` 为权威，`time_s` 只作便捷显示。每个输出文件写入 SHA256，比较器拒绝读取被修改后未更新 hash 的 trace。`file_hashes.json` 还必须包含两个 collector、Isaac debug 环境、训练侧 observation/action adapter、MuJoCo observation/action adapter 和比较器的源文件 SHA256。

NPZ 只允许固定宽度 numeric dtype，不允许 object array 或 pickle。float 字段不可用时写 `NaN`；事件/布尔字段使用 `int8`，其中 `-1=unavailable, 0=false, 1=true`；终止原因使用 `int16` code，并在 metadata 中保存 code-to-name 映射。

## 9. Trace Schema V1

### 9.1 metadata

每个引擎必须记录：

- `schema_version = Sim2SimDebugTraceV1`。
- 引擎名称和版本。
- Python、NumPy、Torch 版本。
- GPU/CPU、仿真 device、Actor 推理 device 和 dtype。
- scenario hash。
- checkpoint、Actor、contract、asset bundle、model 和 manifest hash。
- control dt、physics dt、每控制周期子步数。
- PhysX position/velocity solver iteration、CPU/GPU pipeline、Fabric 开关、contact processing 状态；MuJoCo solver、iteration、tolerance 和 integrator。
- canonical 6 关节顺序、共享 `all_hinge_order[26]`、各引擎 native joint order 和重排映射。`all_hinge_order` 固定取模型 manifest 的 26 个命名 hinge 顺序；两侧按名称重排后保存 native joint coordinate，不对被动关节猜测额外符号变换。
- ControlFrame 变换矩阵和 `R_diag_from_engine_world`。
- 命令、episode 长度、随机种子和所有随机化开关。
- collector、debug 环境、两侧 observation/action adapter 和比较器源文件 SHA256。
- `contact_observation_mode`：`disabled`、`instrumented_equivalent`、`instrumentation_perturbation`、`instrumentation_failed` 或 `unavailable`，以及插桩等价性报告 hash。
- `contact_active_force_threshold_n=1.0`，左右轮 ContactSensor 配置及实际解析的 body/filter 名称。
- `unavailable_fields` 及逐字段原因；Isaac 必须明确声明 `physx_internal_drive_torque` 不可公开读取。
- 开始时间和结束状态。

任一冻结身份不一致时 fail fast，不生成“可比较”结论。

### 9.2 reset snapshot

reset 必须拆成三个命名相位，不能把“写入 reset 值”“约束 forward 后状态”和“生产 API 实际返回给策略的观测”合并：

1. `reset_written_pre_forward`：`_reset_idx()`/`mj_resetDataKeyframe()` 写入完成后、`sim.forward()`/`mj_forward()` 前的 engine state；Isaac 同时保存此时正式环境的 `_state` cache。
2. `reset_forwarded_post_forward`：完成 `scene.write_data_to_sim(); sim.forward()` 或 `mj_forward()` 后，绕过正式环境 `_state` cache 直接读取的 engine state。
3. `reset_returned_to_policy`：生产 `env.reset()`/runtime `reset()` 实际返回的观测；首个 Actor 推理必须原样消费这一份 `actor_obs_policy`，不得由 collector 静默替换为 post-forward 重建观测。

当前 `WheelLegFlatEnv._reset_idx()` 会在 Isaac `sim.forward()` 之前调用 `_capture_state()`，而 `DirectRLEnv.reset()` 在 forward 后调用 `_get_observations()` 时可能复用该 cache。D1 必须把实际返回观测分别与 pre-forward cache 和 post-forward engine state 重建观测比较，明确首次策略输入来自哪一个相位；诊断代码不得顺便改变生产语义。

每个相位按可用性保存：

- root/reference frame 的 engine-native 世界位姿，仅用于引擎内部审计。
- base COM 的 engine-native 世界位置、诊断坐标位置和 control-frame 姿态。
- base COM 线速度、角速度。
- 26 个 hinge 的 engine-native `q/qd`。
- 6 个主动关节的 canonical `q/qd`。
- 四个主动腿的 `q-q_nominal`。
- 左右 `L0`、`phi0` 和闭链残差。
- 左右轮接触状态、诊断坐标接触点、法向、法向力和法向冲量中可用的字段；力与冲量不得互相冒充。
- projected gravity。
- 从该相位状态重建的 25 维 `actor_obs_physical_rebuilt`。
- 应用确定性 scale/clip 后的 `actor_obs_policy_rebuilt`。
- `reset_returned_to_policy` 额外保存生产 API 原样返回的 `actor_obs_policy_returned`，它是第一次 Actor 推理的权威输入。
- previous action 缓存。

三相位都必须带独立名称和 hash。这样可以判断差异来自 reset 写入、第一次约束投影、正式环境缓存，还是 observation adapter；任何差异都不得通过重新计算后覆盖生产返回值。

### 9.3 control trace

每个 20 ms tick 至少记录：

```text
control_tick
control_time_s
command[3]
actor_obs_physical_pre_step[25]
actor_obs_policy_pre_step[25]
previous_action_before_inference[6]
actor_output_raw[6]
action_clipped[6]
target_command_canonical[6]
target_command_engine_native[6]
active_joint_position_canonical_post_step_pre_reset[6]
active_joint_velocity_canonical_post_step_pre_reset[6]
active_joint_position_engine_native_post_step_pre_reset[6]
active_joint_velocity_engine_native_post_step_pre_reset[6]
all_hinge_position_named_post_step_pre_reset[26]
all_hinge_velocity_named_post_step_pre_reset[26]
pd_torque_unclipped_{canonical,engine_native}_{last_substep,mean,peak_abs,impulse}[6]
pd_torque_effort_clipped_{canonical,engine_native}_{last_substep,mean,peak_abs,impulse}[6]
isaac_host_pd_torque_estimate_{canonical,engine_native}_{last_substep,mean,peak_abs,impulse}[6]
mujoco_commanded_torque_{canonical,engine_native}_{last_substep,mean,peak_abs,impulse}[6]
effort_limit_event[6]
joint_velocity_limit_exceeded[6]
mujoco_velocity_guard_active[6]
base_com_position_engine_world_post_step_pre_reset[3]
base_com_position_diag_post_step_pre_reset[3]
base_orientation_control_wxyz_post_step_pre_reset[4]
base_linear_velocity_control_post_step_pre_reset[3]
base_angular_velocity_control_post_step_pre_reset[3]
projected_gravity_post_step_pre_reset[3]
base_height_post_step_pre_reset
virtual_leg_length_post_step_pre_reset[2]
virtual_leg_phi0_post_step_pre_reset[2]
loop_closure_error_post_step_pre_reset[2]
wheel_contact_active[2]
wheel_normal_force_n_mean[2]
wheel_normal_force_n_peak[2]
wheel_normal_impulse_ns[2]
native_terminated_int8
native_truncated_int8
native_termination_reason_code_int16
common_diagnostic_flags_int8[8]
next_actor_obs_policy_returned[25]
next_obs_is_reset_int8
```

这里的 `actor_obs_*_pre_step` 是本 tick Actor 实际消费的观测；所有未写 `pre_step` 的状态结果都固定为完成本 tick 全部物理子步后、自动 `_reset_idx()` 之前冻结的 terminal-safe 状态。`next_actor_obs_policy_returned` 是 `step()` API 最终返回值：非终止 tick 为下一状态观测，终止/超时 tick 为 reset 后新 episode 观测，并由 `next_obs_is_reset_int8=1` 明确标记。禁止把它当作 terminal state 或下一帧重建目标。

上述 `{canonical,engine_native}` 与 `{last_substep,mean,peak_abs,impulse}` 表示 NPZ 中必须展开为确定的扁平 key，不能保存 object/dict。`last_substep` 明确指最后一个子步基于其 `state_pre_step` 产生的值，不得命名为 `final` 并误解为基于 control tick 的 post-step state。`impulse` 统一按每个子步的力矩乘该引擎真实 physics dt 后求和。

`common_diagnostic_flags_int8[8]` 的固定顺序为：

```text
[height, tilt, root_linear_velocity, root_angular_velocity,
 joint_velocity, loop_closure, virtual_leg_length, unexpected_contact]
```

八项 flag 固定使用以下原始状态谓词，不应用 Isaac 原生终止中的 `grace_steps`，也不直接触发任何引擎 reset：

| flag | 统一公式 | Isaac 数据源 | MuJoCo 数据源 |
|---|---|---|---|
| `height` | `base_height < 0.10 or base_height > 0.40` | base COM 到地面顶面的高度 | 同左 |
| `tilt` | `tilt > 0.80 rad` | control-frame 姿态/projected gravity | control-frame 姿态 |
| `root_linear_velocity` | `norm(v_com_control) > 20.0 m/s` | root COM 线速度 | COM 线速度 |
| `root_angular_velocity` | `norm(omega_control) > 35.0 rad/s` | root link 角速度 | base 角速度 |
| `joint_velocity` | `max(abs(qd_all_26_hinges)) > 80.0 rad/s` | 全部 26 个 hinge 按名称读取 | `data.qvel[6:]` 按名称核验 |
| `loop_closure` | `max(left_error, right_error) > 5.0e-3 m` | `_virtual_leg_state` 的两侧 anchor 残差 | equality/connect anchor 残差 |
| `virtual_leg_length` | `min(L0_left, L0_right) <= 0.05 m` | 真实髋轴到真实轮轴 | 同一几何定义 |
| `unexpected_contact` | 存在允许集合 `{ground-left_wheel, ground-right_wheel}` 外的接触 | 仅在全机身接触枚举 observer 通过等价门后可用，否则 `-1` | `data.ncon` 接触对枚举 |

`joint_velocity` 的统一 flag 故意使用全部 26 个 hinge，以免遗漏被动闭链首先失稳；它不等同于 Isaac 原生 termination 当前只检查六个受控关节的实现。原生行为必须只写入 `native_terminated_int8` 和原生 reason code，不得用统一 flag 覆盖。

该数组只用于统一报告，不改变任何引擎的 step、reset 或原生终止行为；不能计算的项目写 `-1=unavailable`。timeout 必须记录为 `native_truncated_int8=1`，不得伪装为 terminated。

control trace 中的 `effort_limit_event`、`joint_velocity_limit_exceeded` 和 `mujoco_velocity_guard_active` 表示该控制周期任一子步是否触发，对对应 substep event 做逻辑 OR；原始逐子步事件仍保存在 substep trace。

若某个引擎不能提供严格同义字段，浮点数组使用 `NaN`、事件数组使用 `-1`，并在 metadata 的 `unavailable_fields` 中说明原因。禁止以语义不同的量填充同一字段。跨引擎接触比较以每个控制周期的法向冲量为主；若 API 只提供力，则按该引擎真实 physics dt 积分，若 API 直接提供冲量则保留其原值和来源。接触字段不可用时，接触归因规则自动降级为“本轮不可评估”，不得因为其他轴数据更完整而自动选择执行器为根因。

### 9.4 substep trace

每个引擎保存自身原始物理子步：

```text
control_tick
substep_index
physics_time_s
active_joint_position_canonical_pre_step[6]
active_joint_velocity_canonical_pre_step[6]
active_joint_position_engine_native_pre_step[6]
active_joint_velocity_engine_native_pre_step[6]
all_hinge_position_named_pre_step[26]
all_hinge_velocity_named_pre_step[26]
base_com_position_engine_world_pre_step[3]
base_com_position_diag_pre_step[3]
base_orientation_control_wxyz_pre_step[4]
base_linear_velocity_control_pre_step[3]
base_angular_velocity_control_pre_step[3]
target_command_canonical[6]
target_command_engine_native[6]
pd_torque_unclipped_canonical[6]
pd_torque_unclipped_engine_native[6]
pd_torque_effort_clipped_canonical[6]
pd_torque_effort_clipped_engine_native[6]
isaac_host_pd_torque_estimate_canonical[6]
isaac_host_pd_torque_estimate_engine_native[6]
mujoco_commanded_torque_canonical[6]
mujoco_commanded_torque_engine_native[6]
effort_limit_event[6]
joint_velocity_limit_exceeded[6]
mujoco_velocity_guard_active[6]
active_joint_position_canonical_post_step[6]
active_joint_velocity_canonical_post_step[6]
active_joint_position_engine_native_post_step[6]
active_joint_velocity_engine_native_post_step[6]
all_hinge_position_named_post_step[26]
all_hinge_velocity_named_post_step[26]
base_com_position_engine_world_post_step[3]
base_com_position_diag_post_step[3]
base_orientation_control_wxyz_post_step[4]
base_linear_velocity_control_post_step[3]
base_angular_velocity_control_post_step[3]
projected_gravity_post_step[3]
virtual_leg_length_post_step[2]
virtual_leg_phi0_post_step[2]
loop_closure_error_post_step[2]
wheel_contact_active[2]
wheel_normal_force_n[2]
wheel_normal_impulse_ns[2]
```

每一行定义一个完整状态转移：`state_pre_step + target/torque -> state_post_step`。Isaac host estimate 与 MuJoCo commanded torque 都与同一行的 pre-step `q/qd` 配对；post-step 状态只用于测量该命令推进一个物理步后的响应。Isaac 每 tick 应有 4 行，MuJoCo 每 tick 应有 20 行。行数、tick 连续性或时间步不符合 metadata 时直接判定采集失败。相邻行必须满足前一行 `state_post_step` 与后一行 `state_pre_step` 落在同引擎自然复现门限内；`physx_internal_drive_torque` 不进入数值数组，因为当前固定为不可得，该事实必须存在于 metadata，不能由 `isaac_host_pd_torque_estimate` 替代。

Isaac 子步采集首版固定使用 debug-only 环境子类，并严格保持以下生产顺序：

```text
_pre_physics_step(action)
is_rendering = sim.has_gui() or sim.has_rtx_sensors()
4 x {
  _sim_step_counter += 1
  _apply_action()
  capture_state_pre_step_without_mutating_env_history()
  scene.write_data_to_sim()
  capture_isaac_host_estimate_paired_with_pre_step_state()
  sim.step(render=False)
  if _sim_step_counter % render_interval == 0 and is_rendering:
      sim.render()
  scene.update(dt=physics_dt)
  capture_state_post_step_without_mutating_env_history()
  append_one_complete_substep_transition()
}
episode_length_buf += 1
common_step_counter += 1
reset_terminated, reset_time_outs = _get_dones()
freeze_state_post_step_pre_reset_and_done_reason()
reward_buf = _get_rewards()
if reset_buf.any():
    _reset_idx(reset_env_ids)
[继续执行与 DirectRLEnv.step 完全相同的 rerender、interval event 和 observation 顺序]
next_obs_returned = _get_observations()
append_control_row_using_frozen_pre_reset_state_and_actual_returned_obs()
```

子步 observer 禁止调用正式环境 `_capture_state()`，因为该函数会更新 `_previous_joint_velocity`，从而改变控制边界关节加速度和奖励。observer 只能直接读取 `robot.data` 并调用无副作用几何函数。`freeze_state_post_step_pre_reset_and_done_reason()` 必须在 `_reset_idx()` 之前 clone 所需张量；若本 tick 终止，control row 仍绑定该终止状态，而不是 reset 后状态。

正式采集前必须用同一 reset 和动作文件完成 10 tick collector-versus-production `env.step()` 等价性测试。等价门比较 q/qd、base pose/velocity、观测和原生 done；门限按第 11 节“自然复现包络统一规程”计算并写入等价性报告。

未来若尝试 Isaac Sim 底层 `omni.physx` callback，该路径不属于 Isaac Lab 公开 API，必须额外通过采样相位测试，证明读取的是当前求解并完成 `scene.update()` 后的数据；通过相位测试后仍然必须执行同一 10 tick collector-versus-production 等价门，不能用相位测试替代等价门。

## 10. 固定站立 Scenario

`stand_scenario.py` 生成唯一的首轮 scenario：

```text
scenario_name        = nominal_stand_debug_v1
num_envs             = 1
control_ticks        = 500
control_dt_s         = 0.020
command              = [0.0, 0.0, 0.20]
hold_command          = true
domain_randomization = false
external_disturbance = false
initial_yaw_rad      = 0.0
random_seed          = 0
previous_action      = zeros(6)
```

scenario 文件必须包含上述 policy、contract、asset 和 model hash。collector 不允许通过命令行静默覆盖这些值；需要改变时生成新的 scenario 版本和新 run-id。

## 11. 诊断实验顺序

为避免与训练文档中的 P0/P1/P2/P3 阶段混淆，本诊断使用 `D0-D5` 编号。

### D0：身份和 schema 门禁

检查：

- checkpoint、Actor、contract、asset bundle、XML 和 manifest hash。
- Actor 输入/输出 shape 与 dtype。
- canonical joint 顺序、符号和 q_nominal。
- 26 个 hinge 的共享名称顺序、两侧 native order 和按名称重排索引；名称集合不一致即失败。
- control dt、physics dt 和子步数。
- solver、pipeline、Fabric、contact processing 和 debug observer 模式。
- `R_diag_from_isaac_world`、`R_diag_from_mujoco_world` 及三个基向量变换结果。
- 命令、previous action 和随机化开关。

任一项失败即停止，不进入动力学比较。

### D1：reset 和静态适配器对齐

步骤：

1. 两边分别写入冻结 reset state，保存 `reset_written_pre_forward`。
2. 执行必要的引擎 forward 或 scene update，不推进完整控制周期，绕过环境 cache 保存 `reset_forwarded_post_forward`。
3. 通过生产 reset API 保存 `reset_returned_to_policy.actor_obs_policy_returned`，并证明第一次 Actor 推理实际消费该 buffer。
4. 分别从 pre-forward 与 post-forward canonical state 离线调用两侧 observation adapter，不覆盖生产返回值。
5. 报告生产返回观测与两份重建观测的逐元素差异，单独标记 Isaac `_state` cache 的影响。
6. 在同一 CPU device 上，用同一份序列化 25 维 float32 输入调用同一个 TorchScript Actor。

D1 负责排除关节顺序、符号、零点、ControlFrame、COM API、normalization、previous action 和 TorchScript 推理差异。

### D2：执行器纯函数和单 tick 对齐

步骤：

1. 使用同一份 canonical `q/qd/action` 计算腿位置目标和轮速度目标。
2. 使用冻结公式分别计算 `pd_torque_unclipped` 和 `pd_torque_effort_clipped`。
3. 分别在两个引擎中执行 1 个控制周期。
4. 对每个 Isaac 子步，在 `scene.write_data_to_sim()` 前冻结 `state_pre_step`，写入后立即读取 `isaac_host_pd_torque_estimate`，并只与该 pre-step 状态计算的 `pd_torque_effort_clipped` 做不变性校验；禁止使用推进后的 q/qd。
5. 验证 MuJoCo effort 裁剪、velocity guard 和 `data.ctrl` 命令链路。
6. 比较目标、六个主动关节、全部 26 个命名 hinge、闭链量、机身状态和经验证可用的接触响应。

这一阶段不宣称两种执行器实现或力矩等价。PhysX 内部 drive torque 不可见，因此 D2 只能判断差异是否已在第一个 20 ms 的状态响应中出现，不能根据两个语义不同的“力矩”字段证明执行器相同或不同。

### D3：开环动力学

依次运行三组仍属于站立诊断的动作序列：

1. `zero_action`：六维动作全零，最多 100 tick。
2. `channel_pulse`：每次只激励一个动作通道，序列为 25 tick 零、25 tick `+0.1`、25 tick 零、25 tick `-0.1`；六个通道分别运行。
3. `isaac_policy_replay`：先记录 Isaac 闭环站立产生的 500 tick clipped action，再在全新 reset 的 Isaac 和 MuJoCo 中开环回放同一动作文件。

`isaac_policy_replay` 是首轮最关键的动力学隔离实验：两边不再根据各自状态重新推理，因此 policy 反馈不会放大早期状态差异。若 MuJoCo 仍快速偏离，问题位于 reset、动作适配、执行器或物理模型，而不是 Actor 对不同观测作出不同反应。

同一动作文件回放到全新 Isaac reset 后，必须先证明它能在同引擎复现门限内复现原始 Isaac 轨迹。复现门限按本节末尾的统一规程计算；若不能复现，说明 reset、随机性或采集时序尚未冻结，该动作文件不能用于跨引擎归因。

### D4：闭环短视野

通过 D0-D3 后，使用同一个 Actor 分别运行：

```text
1 tick -> 10 ticks -> 50 ticks -> 500 ticks
```

每一级都先完成比较报告，再进入下一级。闭环比较同时标记：

- 观测首先分离的 tick。
- Actor 输出首先分离的 tick。
- 状态、各自引擎内部力矩诊断、接触和闭链信号首先分离的 tick。
- 首次饱和、速度保护、接触切换和终止事件。

### D5：单变量验证

D5 只有在 D0-D4 给出明确首差异后才能开始。每次只改变一个候选因素，例如执行器离散计算、轮接触 proxy、摩擦、被动阻尼或 equality 参数，并生成新的 model/scenario identity。

D5 不属于首轮采集脚本的验收范围，也不能在本设计实施时顺便进行。

### 自然复现包络统一规程

collector-versus-production、ContactSensor 插桩和 `isaac_policy_replay` 三类同引擎门禁统一使用以下规程，禁止各脚本自行选择容差：

1. 在完全相同的 engine、device、dtype、`num_envs`、render mode、solver、pipeline、Fabric、observer、reset 和动作文件下独立运行至少 5 次。
2. 对每个浮点 signal `s` 保存全部重复样本，并计算：

```text
observed_envelope_s = max over runs i<j, ticks t and components k:
                      abs(x_i[s,t,k] - x_j[s,t,k])

numeric_floor_s = 32 * eps(dtype_s) * max(1, max_abs_reference_s)

equivalence_limit_s = max(observed_envelope_s, numeric_floor_s)
```

3. `max_abs_reference_s` 取全部基线重复样本中该 signal 的最大绝对值。比较器必须在报告中保存 dtype epsilon、scale、实测包络、numerical floor 和最终 limit，不能只保存一个无法追溯的容差。
4. quaternion 使用第 12.4 节的 geodesic error 先转换为标量 signal；布尔、事件、名称、shape、tick 和 reason code 必须完全一致，不应用浮点 floor。
5. collector-versus-production 任一必需 signal 超过 `equivalence_limit_s` 时为硬失败，不生成“可比较”结论。
6. `isaac_policy_replay` 任一必需 signal 超过该 limit 时，动作文件不能进入跨引擎归因。
7. ContactSensor 插桩比较分三级：
   - 全部必需 signal 不超过 limit：`contact_observation_mode=instrumented_equivalent`，接触数据可参与 D3/D4 归因。
   - 超过 limit，但所有差异仍低于第 12.3 节对应材料差异门限：`contact_observation_mode=instrumentation_perturbation`；插桩 trace 只供审计，不进入跨引擎数值比较。非接触比较继续使用已通过 collector-versus-production 门的无传感器基线，全部 Isaac 接触字段按 `unavailable` 处理。
   - 任一差异达到或超过材料差异门限：`contact_observation_mode=instrumentation_failed`；该插桩运行不能作为跨引擎可比较 trace。
8. 同一门禁的重复样本、源 trace hash、计算脚本 hash 和结果报告必须一并保存。零实测包络不退化为无条件 bitwise gate，而是使用上述 dtype-aware numerical floor；报告仍单独记录 bitwise equality 是否成立。

## 12. 对齐和误差判定

### 12.1 精确身份项

以下项目必须完全一致：

- hash、schema version、关节名称和顺序。
- scenario 和动作序列文件。
- 命令和 previous action 初始化。
- Actor 的输入 shape、输出 shape 和 float32 dtype。
- 同一个序列化 25 维输入送入同一个 TorchScript Actor 时的输出。

Actor 恒等检查固定在 CPU 上执行，同一输入 buffer、同一 TorchScript 文件和同一 float32 dtype 下，输出允许的最大绝对数值误差为 `1e-6`。超出即视为推理路径不一致，不进入物理对比。实际闭环 collector 可以在各自运行 device 上推理，但必须在 metadata 记录，跨设备闭环输出不使用该精确恒等门。

### 12.2 reset 报告门限

初始报告门限固定为：

| 信号 | 门限 |
|---|---:|
| active joint position | `1e-6 rad` |
| active joint velocity | `1e-8 rad/s` |
| base COM diagnostic position | `1e-5 m` |
| base orientation geodesic error | `1e-5 rad` |
| projected gravity | `1e-6` |
| phi0 | `1e-5 rad` |
| L0 | `1e-6 m` |

这些门限用于发现 reset/坐标适配错误。当前完整闭链 XML 与 USD anchor 名义值的已知差异约为 `phi0=1.31e-6 rad`、`L0=0.572e-6 m`，均处于当前门限内。约 `3.08e-4 rad` 的差异来自 CAD 单闭链近似与完整闭链的不同算法，不是 USD 与 MJCF 的引擎几何表示残差，禁止用它作为放宽门限的依据。数值来源为 `docs/2026-10-04-wheelleg-phi0-sim2sim-usd-cleanup-design.md` 的 USD anchor 完整闭链与 XML 完整闭链结果；原始 USD anchor 证据为 `wheelleg_dreamwaq/artifacts/phase1_v4/usd-anchor-audit.json`。

报告必须把 D1 差异分为三层：同一 canonical 输入的离线 adapter 差异、静态模型/anchor 表示差异、引擎 forward/约束投影后差异。若约束 forward 本身使状态超过门限，必须分别展示 forward 前后差异和所属层级，不能简单放宽容差或把所有初态差异都归为 adapter 错误。

### 12.3 动态首差异门限

动态系统中的机器精度差异会被混沌和接触放大，因此报告同时给出：

1. 每个信号的最大绝对误差和 RMS。
2. 第一次超过“材料差异门限”的 control tick。
3. 连续 3 个 control tick 超限后的第一次持久差异。

初始材料差异门限：

| 信号 | 门限 |
|---|---:|
| active joint position | `1e-3 rad` |
| active joint velocity | `1e-2 rad/s` |
| base COM diagnostic position | `1e-3 m` |
| base orientation geodesic error | `0.5 deg` |
| base linear velocity | `0.02 m/s` |
| base angular velocity | `0.02 rad/s` |
| projected gravity | `1e-3` |
| base height | `1e-3 m` |
| phi0 | `1e-3 rad` |
| L0 | `1e-4 m` |
| loop closure error | `1e-4 m` |

这些值是诊断报告阈值，不是“PhysX 与 MuJoCo 等价”的验收标准。由于不存在严格同义的跨引擎 applied torque，本表不设置跨引擎“实际力矩”门限。Isaac host PD estimate 与其同状态参考值、MuJoCo 各显式处理阶段之间的误差作为各自引擎内部不变性报告；跨引擎执行器判断以目标一致后的状态响应为主。接触 active 状态不使用连续数值门限，直接报告首次不一致事件及其前后 5 个控制周期。

### 12.4 姿态误差

姿态不直接逐元素比较 quaternion。统一计算：

```text
q_error = q_isaac^-1 * q_mujoco
orientation_error = 2 * acos(clamp(abs(q_error.w), 0, 1))
```

两个 quaternion 必须先转换成同一 ControlFrame、统一 `[w,x,y,z]` 顺序并归一化。

## 13. 首差异归因规则

比较器按以下优先级生成归因候选，但不能自动宣称最终根因：

1. **身份/输入错误：** D0 hash、shape、dtype 或动作文件不一致。
2. **reset/坐标错误：** D1 在物理推进前已经出现 q、COM、姿态、projected gravity 或观测差异。
3. **观测/推理错误：** 同一 canonical state 产生不同观测，或同一观测产生不同 Actor 输出。
4. **动作适配错误：** 相同 ActionV1 产生不同目标、顺序或符号。
5. **执行器响应差异：** 目标和各自引擎内部力矩不变性校验通过，但第一个控制周期内主动关节 q/qd 或机身状态响应首先分离。不得用 Isaac host estimate 与 MuJoCo commanded torque 的直接差值触发本规则。
6. **闭链差异：** 主动目标和主动关节早期响应接近，但被动关节、phi0/L0 或 closure residual 首先分离。
7. **接触差异：** 经插桩等价门验证后的轮地接触 active、法向响应或切向运动首先分离；任一侧接触字段 unavailable 时，本规则只能标记“不可评估”。
8. **累积动力学差异：** 单步和短视野接近，只在较长开环回放中逐渐分离。
9. **闭环放大：** 开环回放可接受，但闭环时因微小观测差异导致 Actor 输出快速分离。

报告必须列出支持和反对每个候选的直接证据。缺失某类证据不会提高其他候选的可信度。不能因为 MuJoCo 使用显式 PD，或因为接触字段不可用，就在没有状态响应证据的情况下直接把执行器写成已确认根因。

## 14. 当前候选因素及证据状态

| 候选因素 | 当前状态 | 本轮如何验证 |
|---|---|---|
| 关节顺序/符号/零点 | manifest 已定义，但动态链路尚未逐帧证明 | D0-D1 canonical/native 双份 trace |
| Actor 导出 | export verification 为 0，但两引擎构造的输入尚未证明一致 | D1 同输入推理、D4 输入输出 trace |
| previous action 时序 | 文档已定义，运行时尚未跨引擎逐帧核对 | D1 reset、D4 每 tick 前后缓存 |
| ImplicitActuator 与显式 PD | 已知实现语义不同；PhysX 内部 drive torque 不可读，优先级高但尚非结论 | D2 各自内部力矩不变性校验和首周期状态响应 |
| velocity limit 行为 | Isaac solver 制动与 MuJoCo 显式置零不是同义事件 | D2-D4 统一速度阈值状态、MuJoCo guard 和后续状态响应 |
| passive damping/armature | 参数已对齐，求解语义可能不同 | D2-D3 被动关节 trace |
| 闭链 equality/PhysX articulation | solver 机制不同；当前 closure residual 较小 | D3 closure、phi0、L0 与被动关节首差异 |
| 轮地接触 | 当前正式 Isaac 环境无 ContactSensor；只有 debug 插桩通过等价门后才可跨引擎比较 | D3 contact event、法向响应和轮滑移；否则标记不可评估 |
| PhysX 刚体上限与 depenetration | Isaac 固定 `max_depenetration_velocity=1.0`、`max_linear_velocity=100`、`max_angular_velocity=100`，MuJoCo 无严格同义机制 | D3-D4 观察是否只在碰撞修正或高速状态后出现累积差异 |
| COM/ControlFrame/projected gravity | 设计已明确，尚需运行证据 | D1 canonical state golden vector |
| reset 约束投影 | 两边都会执行各自约束更新，可能改变初态 | D1 forward 前后双快照 |

## 15. 测试要求

### 15.1 schema 单元测试

- 必填字段、shape 和 dtype 校验。
- 25 维 observation 和 6 维 action 切片校验。
- canonical 6 关节顺序、26 hinge 名称集合/重排和右轮符号校验。
- control tick/substep 行数和时间连续性校验。
- `NaN/Inf` 检测；仅 metadata 声明 unavailable 的字段允许 `NaN`。
- torque 字段语义互斥校验：Isaac 不得填写 MuJoCo commanded torque，MuJoCo 不得填写 Isaac host estimate，PhysX 内部 drive torque 必须声明 unavailable。
- 原生 termination/truncation 与八项统一诊断 flag 的 shape、dtype 和 codebook 校验。

### 15.2 时序测试

- reset 后 previous action 为零。
- tick `t` 的 observation 使用 tick `t-1` 的 clipped action，且 `actor_obs_policy_pre_step` 必须逐元素等于该 tick 实际送入 Actor 的 buffer。
- 每个 tick Actor 只调用一次。
- Isaac 每 tick 4 个子步，MuJoCo 每 tick 20 个子步。
- 一个 tick 内 target 保持不变，但 MuJoCo 显式伺服力矩按当前 q/qd 每子步重算。
- 每个子步的 PD/host/commanded torque 只与本行 `state_pre_step` 配对；前一行 post-step 必须衔接后一行 pre-step。
- canonical/native 双轨 round-trip 测试，重点覆盖右轮 target、q/qd 和 generalized torque 的 `-1` 映射。
- reset 三相位测试，证明实际首帧 Actor 输入来自生产 reset 返回值，并能检测 pre-forward cache 与 post-forward engine state 的差异。
- terminal tick 测试，证明 control row 使用 `_reset_idx()` 前的冻结状态，同时把 reset 后 API 返回观测仅写入 `next_actor_obs_policy_returned` 并设置 `next_obs_is_reset_int8=1`。
- debug-only step 子类与生产 `env.step()` 的 10 tick 等价性测试；测试必须覆盖 `_sim_step_counter`、render 条件和 post-step 顺序。
- 未来若增加底层 physics callback，该路径除相位测试外仍必须通过同一 10 tick 等价门。
- 有/无 Isaac ContactSensor 配置的同动作轨迹等价性测试；分别验证 `instrumented_equivalent`、`instrumentation_perturbation` 和 `instrumentation_failed` 三种结果。
- 顶层和嵌套配置对象身份测试，确认 debug 配置不会把正式 `WHEELLEG_CFG.spawn.activate_contact_sensors` 改为 `True`。
- 八项统一 flag 的逐项边界测试，尤其验证全部 26 hinge 与 Isaac 原生六受控关节 termination 的语义分离。
- passive hinge 合成首差异测试，证明比较器能在主动关节仍低于门限时定位第一个被动关节分离。

### 15.3 合成比较器测试

用人工生成的两份 trace 验证比较器能准确识别：

- 第 0 tick reset 差异。
- 单个关节符号反转。
- previous action 错一帧。
- 第 N tick 状态响应首先分离，同时验证语义不同的力矩字段不会被直接相减。
- 接触状态单次和持续不一致。
- 接触字段 unavailable 时，比较器输出“不可评估”而不是自动归因到其他候选。
- quaternion 正负号不同但姿态相同。
- 两个引擎子步数不同但控制边界相同。
- timeout 只触发 native truncation，不触发 native termination。

### 15.4 真实数据 smoke test

先完成 Actor CPU 恒等、collector-versus-production 等价性和可选 ContactSensor 插桩等价性，再运行 1 tick 和 10 tick。只有 metadata、shape、时间轴、hash、双轨终止字段和报告均有效后，才允许运行 500 tick 正式采集。

## 16. Fail-Fast 条件

遇到以下任一情况，collector 或比较器立即失败并保留已有产物：

- checkpoint、Actor、contract、asset、XML 或 manifest hash 不匹配。
- joint/body/actuator 名称缺失或重复；启用接触采集时 contact 名称缺失或重复。
- observation/action shape 或 dtype 不匹配。
- command 在 episode 内发生变化。
- 域随机化或命令重采样未关闭。
- reset 后出现未声明的非零 previous action。
- 首个 Actor 输入不是生产 reset API 原样返回的 `actor_obs_policy_returned`，或 collector 静默以 post-forward 重建观测替换它。
- canonical/native 双轨字段缺失、右轮 round-trip 不成立，或公共字段混入 engine-native 符号。
- 子步 torque 与 post-step q/qd 配对，或相邻子步的 post/pre 状态无法连续衔接。
- terminal control row 在 `_reset_idx()` 后读取，或终止状态与 reset 后观测被写入同一状态字段。
- 未在 metadata 声明 unavailable 的字段出现 NaN/Inf。
- control tick 缺失、重复或时间倒退。
- 子步数量与 metadata 不符。
- MuJoCo 出现模型允许集合外的接触对。
- collector 将 PD 参考值、Isaac host estimate、MuJoCo commanded torque 或不可见的 PhysX internal drive torque 写入错误的同义字段。
- collector-versus-production 等价门失败却继续生成“可比较”结论。
- ContactSensor 状态不是 `instrumented_equivalent` 却继续使用 Isaac 接触数据做归因。
- ContactSensor 状态为 `instrumentation_failed` 却继续生成跨引擎“可比较”结论。

物理终止不是采集程序失败。若某个引擎在 500 tick 前因 tilt 等原因终止，应记录自动 reset 前冻结的完整 terminal trace；该 control row 的状态字段全部来自 `state_post_step_pre_reset`，而 `step()` 返回的 reset 后观测只进入 `next_actor_obs_policy_returned`。比较范围以共同有效 tick 为主，同时保留另一引擎的后续独立 trace。

## 17. 首轮完成标准

本轮诊断实现完成必须同时满足：

1. D0 身份门禁通过。
2. 两个引擎都生成符合 `Sim2SimDebugTraceV1` 的 reset、control 和 substep trace。
3. D1 给出 reset 三相位、生产实际返回观测、observation adapter 和 Actor 同输入推理结果，并明确首次 Actor 输入的真实来源。
4. D2 给出第一个控制周期每个子步的 pre/post state、canonical/native 目标、PD 两个参考阶段、Isaac host estimate、MuJoCo commanded torque、各自内部不变性结果和状态响应差异；明确声明 PhysX 内部 drive torque 不可得。
5. D3 至少完成 zero action 和 Isaac policy replay；per-channel pulse 六个通道均有独立结果。
6. D4 完成 1、10、50、500 tick 或明确记录提前终止。
7. `comparison/summary.json` 明确给出第一个数值差异、第一个材料差异和第一个持久差异。
8. 报告把差异定位到至少一个处理层级，并列出下一步单变量实验；若证据不足，必须写“尚未定位”，不能用猜测补全。
9. Isaac 接触路径必须形成可审计状态：`instrumented_equivalent` 可提供接触 trace；`instrumentation_perturbation` 只保留插桩 trace 供审计并回退到无传感器基线做非接触比较；`instrumentation_failed` 拒绝该插桩运行；`unavailable` 明确原因。除第一种外均禁止接触归因。

## 18. 实施边界

预计新增代码只位于：

```text
wheelleg_dreamwaq/debug/sim2sim/
```

预计新增运行产物只位于：

```text
wheelleg_dreamwaq/artifacts/debug/sim2sim/
```

第一轮不应修改：

```text
wheelleg_dreamwaq/source/
wheelleg_dreamwaq/sim2sim/mujoco/models/
wheelleg_dreamwaq/sim2sim/mujoco/wheelleg_mujoco/
wheelleg_dreamwaq/logs/
```

`debug/sim2sim/isaac_debug_env.py` 可以导入正式环境类型，但必须通过嵌套 `.replace()` 或经过身份测试的深复制建立独立配置；允许在副本中启用 ContactReportAPI、添加 ContactSensor，并定义仅供 collector 使用的环境子类。首版不实现底层 physics callback。禁止直接修改共享 `WHEELLEG_CFG`、正式 env cfg、正式类方法，禁止 monkey-patch，禁止让 `source/` 或正式 MuJoCo 包反向依赖 debug 目录。

MuJoCo 接触读取和额外诊断计算优先写在 `collect_mujoco_trace.py` 的只读 adapter 中，不因采集方便修改现有 runner/control。若实施中发现必须更改正式运行代码才能采集某个关键字段，应停止在该边界，先更新本设计并说明：所需 hook、读取时机、为何无法从现有公开 API 获取，以及如何保证 hook 不改变控制结果。

## 19. 与已有文档的关系

本文件不替代以下文档：

- `docs/2026-10-03-wheelleg-dreamwaq-architecture.md`：训练工程总架构和阶段边界。
- `docs/2026-10-04-wheelleg-phi0-sim2sim-usd-cleanup-design.md`：phi0、USD 清理和正式 MuJoCo Sim2Sim 的模型/契约设计。

本文件只定义“为什么两个仿真器行为不同”的诊断层。若诊断结果要求修改模型、控制器或契约，必须回到对应主设计文档升级 schema/version/hash，而不是只在 debug 脚本里做补丁。

## 20. 复核清单

实施前复核以下事项：

- [x] 只使用 Run 01 `model_999.pt`。
- [x] 只运行固定站立命令 `[0.0, 0.0, 0.20]`。
- [x] 使用现有 `wheel_leg_urdf4_v1.xml`，不改 XML。
- [x] 1 个环境、500 tick、50 Hz、无随机化。
- [x] control boundary 与 engine-native substep 分开存储。
- [x] 不把 PD reference、Isaac host estimate、MuJoCo commanded torque 或 PhysX internal drive torque 混为同义量。
- [x] debug Isaac 配置通过嵌套复制获得，顶层、robot 和 spawn 对象均未污染共享正式配置。
- [x] collector 与生产 `env.step()` 已通过等价性门。
- [x] 自然复现包络使用 5 次重复、dtype-aware numerical floor，并保存全部计算证据。
- [x] ContactSensor 状态明确为 `unavailable`：Isaac Lab 2.3.2 初始化时拒绝未展开的 `{ENV_REGEX_NS}/Robot` prim 路径；该运行未进入跨引擎证据。
- [ ] 修复 filtered ContactSensor 的运行时 prim 解析，并在确认字段确为法向力而非合力模长后重新执行插桩等价门。
- [x] Actor 精确恒等测试固定在 CPU，同设备执行。
- [x] 原生 terminated/truncated 与统一诊断 flag 分开记录。
- [x] 八项统一诊断 flag 的公式、阈值和数据源均按表实现。
- [x] `R_diag` 公式、yaw=0 前提和三个基向量测试明确。
- [x] canonical/native joint 值和符号映射均可回查。
- [x] 全部 26 个 hinge 已按共享名称顺序保存并通过集合、重排和被动关节首差异测试。
- [x] previous action 时序可逐 tick 验证。
- [x] 开环回放与闭环测试分开。
- [x] 首处分歧已定位到物理响应层，再进入接触、闭链和 drive 的单变量实验。

## 21. 正式执行结果

### 21.1 产物身份

本轮正式产物根目录：

```text
wheelleg_dreamwaq/artifacts/debug/sim2sim/formal-diagnosis-20261005-v1/
```

聚合报告和机器可读汇总：

```text
analysis/report.md
analysis/diagnosis_summary.json
analysis/scenario_summary.csv
analysis/pulse_response_summary.csv
analysis/artifact_hashes.json
```

### 21.2 正式门禁

以下三项均通过：

1. 5 组 production/reference 与 debug collector 等价门。
2. 两套独立 Python/Torch 环境中的 CPU Actor 同输入恒等门；同一序列化输入的输出最大误差为 `0.0`。
3. 5 组 Isaac 原始闭环动作与 fresh-reset replay 等价门。

所有有效跨引擎比较的 frozen identity、动作序列 identity 和 reset gate 均通过。

### 21.3 首处分歧

所有有效场景的第一个材料级分歧均为：

```text
control tick 0
active_joint_position_canonical_post_step_pre_reset
```

闭环 tick 0 的输入链路仍一致：

- Actor 输入最大误差：`1.29652e-08`。
- Actor 输出最大误差：`3.57628e-07`。
- 六维控制目标最大误差：`9.83477e-06`。

零动作实验中，初始 PD 力矩最大误差仅 `1.55582e-06 Nm`，但 5 ms 后机身角速度最大误差已达到 `0.13123 rad/s`，20 ms 后主动关节位置最大误差达到 `0.00105319 rad`。因此 Actor、归一化、动作顺序、符号和初始 PD 计算均不是当前首要根因；差异发生在物理系统响应阶段。

### 21.4 隔离实验

- 闭环：Isaac 完成 500 tick；MuJoCo 在第 270 行因 tilt 终止。
- 零动作：Isaac 和 MuJoCo 分别在 26、24 tick 因 tilt 终止，证明差异不依赖 Actor 输出。
- Isaac action replay：Isaac 完成 500 tick；相同冻结动作在 MuJoCo 中仅运行 35 tick 即 tilt，证明问题不是仅由闭环 policy 反馈放大。
- 原通道脉冲在 tick 25 才开始，而零动作姿态此前已倾倒，因此原六组结果作废。
- 修正后的即时双极脉冲从 tick 0 开始，六个通道在两个引擎中均完成 20 tick。四个腿关节的基线扣除后位置响应总体接近；两个轮通道的后期非线性差异更大，右轮最明显，但目标增量和初始响应方向一致，不能据此判定动作符号错误。

### 21.5 当前候选排序

1. 轮地接触几何、首次接触时刻和接触求解差异。Isaac 使用 14796 点 Mesh 的 `convexHull`，MuJoCo 使用半径 `0.0625 m` 的球体代理。
2. 闭链表达和约束柔度差异。Isaac 使用 4 个 `PhysicsRevoluteJoint`，MuJoCo 使用 8 个带 `solref/solimp` 的双点 `connect` 约束近似 4 个铰链。
3. 执行器与积分差异。Isaac 使用 5 ms 物理步长的 PhysX implicit drive，MuJoCo 使用 1 ms 物理步长下显式计算的 PD 力矩。

当前证据只能把根因定位到上述物理响应层，尚不能在三个候选之间唯一归因。下一轮必须先修复并门控 Isaac contact observer，再按“接触几何、闭链约束、drive/步长”的顺序做单变量实验。轮端 effort limit 从当前共同使用的 `9 Nm` 调整到实车约 `2 Nm` 属于后续 sim-to-real 标定，不得混入本轮跨引擎归因。
