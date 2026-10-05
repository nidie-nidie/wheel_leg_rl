# WheelLeg Phi0 对称奖励、MuJoCo Sim2Sim 与 AssetBundleV2 设计

日期：2026-10-04  
状态：Implementation Specification v1.8；补充训练/评估/动力学语义指纹与训练监控门禁  
上游文档：`docs/2026-10-03-wheelleg-dreamwaq-architecture.md`

## 1. 文档目的

本文冻结以下增量设计，作为现有 WheelLeg DreamWaQ 架构的补充：

1. 增加基于真实偏置闭链几何的左右虚拟腿 `phi0` 对称奖励。
2. 在当前 `wheelleg_dreamwaq` 工程中建立独立的 MuJoCo sim2sim 子工程。
3. 保留不可变的 `AssetBundleV1`，另建删除 USD 内嵌地面的 `AssetBundleV2`。
4. 将命令改为每个 episode 采样一次并保持，升级到 CommandSamplingV2 与 NormalizationV2。
5. 以四个不同随机 seed 从头训练四次 1000 iterations，并用统一 MuJoCo headless 协议选出最佳 checkpoint。

本文定义实现边界、数据契约、兼容性规则、训练矩阵和验收标准。只有独立文档复核无 P0/P1 后才允许修改 USD 副本和代码。

## 2. 已确认决策

| ID | 决策 |
|---|---|
| D-01 | `phi0` 使用偏置闭链修正后的真实虚拟腿角，不使用旧固件理想五杆近似作为真值。 |
| D-02 | 奖励只约束左右 `phi0` 相等，不约束二者必须等于 `90 deg` 或其他绝对角度。 |
| D-03 | `phi0` 只进入奖励、日志和测试，不进入 ActorObsV1、CriticObsV1 或 ActionV1。 |
| D-04 | 保持 25D Actor、41D Critic、6D Action、50 Hz policy 和当前奖励统一乘 `control_dt` 的语义。 |
| D-05 | MuJoCo sim2sim 位于 `wheelleg_dreamwaq/sim2sim/mujoco/`，运行时不依赖 Isaac Lab 或 RSL-RL。 |
| D-06 | 原 `AssetBundleV1` 不做任何字节级修改；新建完整、自包含的 `AssetBundleV2`。 |
| D-07 | `AssetBundleV2` 不包含 `/wheel_leg_urdf4/GroundPlane`，地面继续由任务配置唯一创建。 |
| D-08 | 新奖励和新资产属于后继训练合约，不能把旧 V1 checkpoint 当作同一训练任务直接 resume。 |
| D-09 | CommandSamplingV2 在 reset 时采样一次，完整 `10 s` episode 内保持不变；模式概率为站立/直行/转向/组合 `20%/30%/20%/30%`。 |
| D-10 | V4 命令范围固定为 `vx in [-1.5,1.5] m/s`、`yaw_rate in [-1,1] rad/s`、`base_height in [0.16,0.24] m`，并使用 NormalizationV2。 |
| D-11 | 第一轮正式 V4 训练固定 `weight_phi0_symmetry=-1.0`，四次训练仅 seed/RNG 不同，不把四个 run 当作权重 A/B。 |
| D-12 | MuJoCo 模型拓扑固定来自本地 `wheel_leg_urdf4_self_mesh_all.xml`，不重新发明一套机构模型。 |
| D-13 | 四次 PPO 均从头训练 1000 iterations；完成后由同一 MuJoCo headless 场景集比较并记录数据。 |
| D-14 | V4 保持 `control_dt=0.02 s` 不变，physics 改为 `sim_dt=0.005 s`、`decimation=4`，articulation solver 固定 `96/4`，用于满足闭环 FK 动态门禁。 |
| D-15 | MuJoCo 3.14.0 对单个 STL 的三角面数上限为 200000；`base_link.STL` 只允许按原三角记录无损分片，不允许减面或重采样。 |
| D-16 | USD 浮点量化使 `jMK/jEC` 的最大主惯量分别越过刚体三角不等式约 `6.82e-13 kg m^2`；只允许把该最大值向零调整一个可表示浮点步，并在 manifest 中逐值记录。 |
| D-17 | `MujocoModelV1` manifest 固定 observation/action adapter 版本和实现文件 hash；加载策略时同时校验模型、adapter、policy manifest 和 Actor hash。 |
| D-18 | `MujocoModelV1` 额外冻结编译后完整动力学语义及 hash，覆盖 timestep/options、全部 joint/dof、body mass/inertia、actuator、equality、contact pair 和 ContractV4 reset；运行时从实际编译模型重新计算后校验。 |
| D-19 | 正式 MuJoCo 报告必须携带统一 `MujocoEvaluationContractV1` 指纹；smoke、不同场景/时长/模型、缺失或重复 checkpoint 的报告不得进入正式排名。 |
| D-20 | 四次训练以种子无关训练指纹证明语义一致：完整 `Phase1ContractV4`、源码/依赖、版本、profile 和 batch 必须相同；只排除 seed/RNG、时间、命令行、resume 元数据，以及包含 seed/run-name/log-dir 的原始 env/agent YAML 文件 hash。原始 hash 仍保留在各 run manifest 中用于追溯。 |
| D-21 | 训练监控结果是正式选优门禁；任一 hard anomaly 都令 suite 进入 `review_required`、清空 `selected` 并禁止宣称 Phase 1 通过。启动期尚未创建 event 文件或尚未出现 episode tag 的可恢复缺失只保留在单次快照，不在最终快照正常后继续误报。 |

## 3. 与主架构的关系

本文不改变主架构中的以下契约：

- `ControlFrameV1`。
- `ActionV1 = [jIJ, jIO, jAB, jAG, jwheel_left, jwheel_right]`，6D。
- `ActorObsV1`，25D。
- `CriticObsV1`，41D。
- `CommandV1`，3D。
- ActorObsV1/CriticObsV1 的维度和字段顺序；V4 归一化版本升级为 `NormalizationV2`。
- `control_dt=0.02 s` 和 50 Hz policy 语义保持不变；V4 physics 使用 `sim_dt=0.005 s`、`decimation=4`。
- 四个腿关节位置目标加两个轮端速度目标的动作语义。
- 环境不依赖 DreamWaQ，普通 PPO 通过后再接入 CENet。

本文对主架构作如下增量升级：

- 原 Phase 1 的 `G-05` 和无镜像奖励结论继续描述旧基线，不被追溯修改。
- 新训练任务增加 `RewardSchemaV2`、`VirtualLegKinematicsV1` 和 `AssetBundleV2`。
- 新训练任务使用 `Phase1ContractV4`，并增加 CommandSamplingV2 与 NormalizationV2；旧 `Phase1ContractV3` 仍只对应 `AssetBundleV1` 和原命令/奖励集合。
- `CheckpointMetadataV3` 的结构若不增加字段可以继续使用，但其中记录的 contract hash、asset hash 和 reward 配置必须指向 V4。
- 本文显式修订主架构旧版“镜像奖励必须位于关节空间”的限制：对称奖励可位于经审计、版本化和测试的闭链几何空间；V4 使用真实 `phi0` 几何差，不使用复旦 `nominal_state`。

## 4. 非目标

本设计明确不做以下事项：

- 不把动作改成 `L0/phi0`、VMC 力或虚拟力矩。
- 不在 Actor 输入中增加 `phi0`、`L0`、真实高度或真实线速度。
- 不用 `phi0` 奖励替代姿态、高度、速度或动作平滑奖励。
- 不增加跳跃、复杂地形或接触规划。
- 不改变 PPO/DreamWaQ 网络结构。
- 不把旧 C 语言 VMC 控制器移植到 MuJoCo 策略执行路径。
- 不要求 Isaac Lab 与 MuJoCo 的状态轨迹逐时刻完全一致。
- 不在本阶段实现 RK3566 或其他实机部署。

## 5. 总体结构

### 5.1 Isaac Lab 训练侧

```text
USD joint/body state
        |
        +--> ActorObsV1 25D --> PPO Actor --> ActionV1 6D --> actuator targets
        |
        +--> CriticObsV1 41D --> PPO Critic
        |
        +--> audited hip/wheel joint-axis positions
        |          |
        |          +--> true phi0_left / true phi0_right
        |                         |
        |                         +--> wrapped mismatch^2 --> reward
        |
        +--> active joint q --> corrected offset FK --> estimated phi0
                                                   --> diagnostic cross-check only
```

### 5.2 MuJoCo sim2sim 侧

```text
MuJoCo state
    |
    +--> observation adapter --> normalized ActorObsV1 25D
                                      |
                                      +--> exported deterministic Actor
                                                     |
                                                     +--> clipped ActionV1 6D
                                                                    |
                                  actuator adapter <----------------+
                                      |
                                      +--> 4 leg PD torques
                                      +--> 2 wheel velocity-servo torques
                                      |
                                      +--> MuJoCo step at 1 kHz
```

### 5.3 资产版本

```text
AssetBundleV1                          AssetBundleV2
immutable legacy bundle               new immutable bundle
contains embedded GroundPlane         contains no embedded GroundPlane
Phase1ContractV3                      Phase1ContractV4
legacy play/resume only               new training/play default
```

## 6. VirtualLegKinematicsV1

### 6.1 物理定义

每侧虚拟腿定义为同轴髋关节轴心 `I` 指向真实轮轴轴心 `W` 的向量。`phi0` 必须沿用已经核对的 VMC/偏置闭链坐标，而不能把 locomotion 前向轴直接替换进公式：

- `s_W`：沿 USD/base `+Y` 的有符号分量。
- `d_W`：沿车体向下方向，即 USD/base `-Z` 的有符号分量。
- 由当前 `R_control_from_usd` 可得等价关系：`s_W = -v_control_x`，`d_W = -v_control_z`。
- 横向分量只用于平面残差审计，不进入 `L0/phi0` 公式。

```text
L0   = sqrt(s_W^2 + d_W^2)
phi0 = atan2(d_W, s_W)
```

因此虚拟腿竖直向下时 `phi0 = pi/2`。该定义与已核对的偏置闭链推导一致，但本奖励没有 `phi0 = pi/2` 的绝对目标。

### 6.2 左右腿统一约定

左右两侧必须先映射到相同的 VMC 侧视坐标后才能比较：

- 两侧 `s_W` 都以 USD/base `+Y` 为正；在当前 ControlFrameV1 中它对应 `-X`，不得误写成 locomotion 前进正方向。
- 两侧 `d_W` 都以车体向下为正。
- 禁止直接比较两个 USD 局部 link frame 中未经变换的角度。
- 禁止通过关节数组位置猜测左右腿；必须按 joint/body name 建立映射。

左右主动关节映射固定为：

```text
left  active q: [jIJ, jIO]
right active q: [jAB, jAG]
```

如果资产审计证明实际物理侧命名与文字标签冲突，以 joint name、轴心位置和低层映射为准，并更新 manifest，不允许只改注释。

### 6.3 两条计算链

#### A. 仿真真实几何链，奖励权威来源

奖励使用仿真当前帧的真实关节轴几何：

1. 读取每侧同轴髋轴的世界坐标 `I_w`。
2. 读取对应轮轴的世界坐标 `W_w`。
3. 计算 `v_w = W_w - I_w`。
4. 用当前 base 姿态将 `v_w` 旋转到 body frame。
5. 在 USD body frame 中取 `s_W=v_body_y`、`d_W=-v_body_z`；若先转换到 ControlFrameV1，则使用 `s_W=-v_control_x`、`d_W=-v_control_z`。
6. 由该 VMC 坐标下的 `s_W`、`d_W` 计算 `L0` 和 `phi0`。

实现前必须由 `wheelleg_dreamwaq/scripts/audit_virtual_leg_geometry.py` 使用 Isaac Sim/USD API 读取 joint local frame、轴向和所连接 body，生成 `wheelleg_dreamwaq/artifacts/phase1_v4/usd-anchor-audit.json`。工件至少包含左右髋轴 `I`、轮轴 `W` 的 prim path、body path、local anchor、world-axis direction、来源 layer 和数值单位。运行时使用该工件冻结的本地轴点随 body transform 转换到世界坐标；只有审计和随机姿态测试证明某个 body origin 与目标轴心重合，才允许 body origin 作为零偏移特例。

该链路不需要额外执行解析闭链 FK 或选择圆交支路；但轴心世界位置仍来自当前物理仿真的闭链约束结果，因此会包含求解器的瞬时闭环残差。奖励始终使用当前真实轴心几何，闭环残差必须单独监测并按本节阈值判定。

#### B. 关节编码器闭链正解，诊断与实机估计来源

根据每侧两个主动关节角，使用修正后的 offset closed-chain FK 计算真实轮轴 `W`，再得到 `L0/phi0`。FK 的几何常量必须从同一份 USD anchor 审计工件推导或校准，USD 是训练资产几何的权威来源；现有 MJCF 推导值只作交叉验证，不得反向覆盖 USD 几何。该实现必须沿用已验证的装配支路和左右输入映射。

此链路在本阶段 Isaac Lab 训练中只用于：

- 与仿真真实几何链逐帧交叉检查。
- 生成单元测试和 golden vector。
- 为未来实机仅使用电机编码器估计 `phi0` 提供同源实现。

它不能驱动 ActionV1，不能把 VMC 引入 PPO 控制主路径。

### 6.4 已知参考点

当前名义主动关节对 `[-0.33367134, +0.33367134] rad` 的已核对参考值为：

| 方法 | 来源 | `L0` | `phi0` |
|---|---|---:|---:|
| 左腿完整偏置闭链 | `wheel_leg_urdf4_self_mesh_all.xml` 树求解 | `0.199507440 m` | `90.539807011 deg` |
| 右腿完整偏置闭链 | 同一 XML，主动关节 `[jAB,jAG]` | `0.199507201 m` | `90.539757564 deg` |
| 左腿 USD anchor 完整闭链 | AssetBundleV2 审计，实际默认关节值 | `0.199506868 m` | `90.539731894 deg` |
| 右腿 USD anchor 完整闭链 | AssetBundleV2 审计，实际默认关节值 | `0.199506678 m` | `90.539725723 deg` |
| 修正后的简化闭式 | `derive_offset_kinematics.py` | `0.199507468 m` | `90.539812684 deg` |
| CAD 单次闭链近似 | `serial_leg_rl/.../offset_leg.py` | `0.199771116 m` | `90.522106136 deg` |
| 旧理想五杆 | 非真值 | 非真值 | `90.000000000 deg` |

这些数值用于回归测试，不构成奖励绝对目标。

`derive_offset_kinematics.py` 中同名 CAD 单闭链实现给出的近似值与 `offset_leg.py` 在第七位有效数字附近存在差异；表中 CAD 行只用于标识历史近似来源，不作为 V4 真值或验收标准。

另一个容易混淆的参考点是导出 MJCF 的原始 `q_front=q_rear=0` 姿态：完整闭链结果约为 `L0=0.120224232 m`、`phi0=90.000010 deg`。它不是当前 PPO 的 `q_nominal`，不能用来替换上表基准。

### 6.5 有效性和失败策略

- 几何真值的 `L0` 必须大于 `L0_min=0.05 m` 且所有分量有限，否则立即报错。
- 关节 FK 出现非法圆交、错误装配支路或 NaN 时必须返回显式 invalid mask；不得沿用上一帧结果。
- 离线 golden test 使用无求解器残差的 USD anchor/FK 参考姿态，固定要求 `abs(L0_fk-L0_true)<0.1 mm`、`abs(wrap(phi0_fk-phi0_true))<0.1 deg`；不允许用运行时 `5 mm` 闭环阈值替代该静态精度要求。
- 运行时每侧闭环 anchor/site 残差 `r_loop` 必须独立满足 `<=5 mm`。`r_loop` 不用于推导 FK 误差，因为约束残差经过连杆几何后不等于轮轴端点误差。
- 动态诊断直接计算 `e_W=norm(W_fk_body-W_true_body)`、`e_L=abs(L0_fk-L0_true)`、`e_phi=abs(wrap(phi0_fk-phi0_true))`，固定硬阈值分别为 `5 mm`、`5 mm`、`3 deg`。正式训练前必须在名义静置、随机动作 1000 step 和全部 reset 初态上通过；训练时继续监测。奖励仍只使用真实轴心几何，不使用 FK 值。
- 离线 golden test 失败、运行时闭环残差超限，或任一直接动态诊断超限时，训练启动/运行立即失败，而不是只写 warning。此时终止 V4 实施并建立“USD/MJCF/闭链几何一致性”专项；不得放宽阈值、沿用 MJCF 常量或把 FK 结果改成奖励真值。
- 角度差统一用 wrap-to-pi，不允许直接相减后平方。

实现期门禁记录：原 `sim_dt=0.01 s`、`decimation=2`、solver `32/4` 在相同白噪声动作测试中出现约 `10 mm` 的 FK-to-axis 轮轴误差；仅增加迭代数仍缺少稳定裕量。改为 `sim_dt=0.005 s`、`decimation=4`、solver `96/4` 后，16 环境、1000 control-step 测试两次一致通过，最大 `r_loop=0.389 mm`、`e_W=2.810 mm`、`e_L=2.319 mm`、`e_phi=0.518 deg`。该修订保持 `control_dt=0.02 s`，不是采用复旦的 `0.01 s` 控制周期。

## 7. RewardSchemaV2

### 7.1 奖励定义

新增正值误差项 `phi0_symmetry`：

```python
delta_phi0 = atan2(
    sin(phi0_left - phi0_right),
    cos(phi0_left - phi0_right),
)
phi0_symmetry = delta_phi0**2
```

总奖励仍遵守当前约定：

```text
weighted_phi0_symmetry = weight_phi0_symmetry * control_dt * phi0_symmetry
weight_phi0_symmetry <= 0
```

该项表达的是“左右角度差越小越好”。例如左右都为 `100 deg` 时，该项误差为零；是否保持合理高度和姿态仍由现有高度、orientation、关节限位和终止条件负责。

### 7.2 状态与观测边界

环境内部状态可增加以下命名字段：

```text
virtual_leg_phi0_true      [num_envs, 2]
virtual_leg_length_true    [num_envs, 2]
virtual_leg_phi0_fk        [num_envs, 2]  # diagnostic
virtual_leg_fk_valid       [num_envs, 2]  # diagnostic
```

它们不改变以下公开张量：

- ActorObsV1 仍为 25D。
- CriticObsV1 仍为 41D。
- DreamWaQ 五帧历史仍为 125D。
- DreamWaQ Actor 输入仍为 44D。
- ActionV1 仍为 6D。

奖励按字段读取状态，不允许使用 observation slice 反推 `phi0`。

### 7.3 日志

每次训练至少记录：

- `reward/phi0_symmetry`：加权后的每步奖励。
- `metric/phi0_delta_abs_rad`：未加权绝对角差。
- `metric/phi0_left_rad`、`metric/phi0_right_rad`。
- `metric/phi0_fk_vs_true_max_abs_rad`。
- `metric/virtual_leg_length_left_m`、`metric/virtual_leg_length_right_m`。

日志指标不能参与 policy 输入或改变 episode 状态。

### 7.4 冻结权重与数值门禁

第一轮正式 `Phase1ContractV4` 固定：

```text
weight_phi0_symmetry = -1.0
```

四次 1000-iteration 训练只比较不同随机 seed，不同时改变奖励权重。该尺度下 `10 deg` 角差每个控制 step 约产生 `-0.00061`，`30 deg` 约产生 `-0.00548`，不会在正常小角差时压过速度、高度和姿态主任务，但会明显惩罚持续大不对称。

正式四次训练前只允许运行有限迭代的数值 smoke test，用于确认奖励有限、梯度和日志量级正常；它不是权重搜索，不产生候选正式 checkpoint。若 smoke test 显示该项导致 NaN、奖励数量级支配或即时策略塌缩，则停止实施并修改文档/contract 后重新复核，不能在四次训练之间临时换权重。

若四个 seed 均未改善 `phi0` 且明显损害主任务，不把权重静默改成 `0.0`。应保留全部工件，将 phi0 奖励降级为诊断量的提案作为新的 ContractV5 决策重新评审。

### 7.5 CommandSamplingV2 与 NormalizationV2

每个环境 reset 时先选择模式，并只采样一次命令：

```text
20% stand:    vx=0,                 yaw=0
30% straight: vx~U(-1.5, 1.5),     yaw=0
20% rotate:   vx=0,                 yaw~U(-1.0, 1.0)
30% combined: vx~U(-1.5, 1.5),     yaw~U(-1.0, 1.0)
all modes:    height~U(0.16, 0.24)
```

采样结果在完整 `10 s` episode 内保持。环境不得保留 5 秒重采样计数器或 step 后重写命令/观测的逻辑。`NormalizationV2` 固定 `vx_max_abs=1.5`、`yaw_rate_max_abs=1.0`；其他字段 scale/clip 与 NormalizationV1 相同。V3/V1 checkpoint 不兼容 V4/V2 normalization。

## 8. AssetBundleV2

### 8.1 不可变版本策略

原资产继续保存在：

```text
E:\wheel_leg_rl-main\wheel_leg_urdf4_usd (1)\wheel_leg_urdf4
```

新资产固定建立在：

```text
E:\wheel_leg_rl-main\wheel_leg_urdf4_usd_v2\wheel_leg_urdf4
```

V2 必须完整复制并拥有自己的五层 USD bundle，不能在运行时 reference V1 中的层。入口文件名继续为 `wheel_leg_urdf4.usd`，版本由 bundle manifest 和根目录区分。

### 8.2 V2 唯一允许的源资产变化

V2 相比 V1 只允许删除内嵌地面子树：

```text
/wheel_leg_urdf4/GroundPlane
/wheel_leg_urdf4/GroundPlane/CollisionPlane
```

删除前必须使用 USD API 枚举该 prim 的完整 prim stack、authoring layer 与 composition arc，并把结果写入 `wheelleg_dreamwaq/artifacts/phase1_v4/groundplane-layer-audit.json`。删除必须在实际 authoring layer 上完成；禁止对二进制 USDC 做字节替换，也禁止只在更强层写一个 inactive/override 意见冒充源层删除。删除后重新保存相关层并为全部五个文件生成新的 size、SHA256 和依赖清单。

机器人 articulation、关节、刚体、质量、惯量、碰撞、材质和 actuator 参考数据不得借本次变更顺带修改。

### 8.3 运行时地面所有权

V2 运行时只有任务创建地面：

```text
prim_path: /World/Ground
translation: (0.0, 0.0, -0.05)
size: (100.0, 100.0, 0.10)
top surface: z = 0.0
static/dynamic friction: 1.0 / 1.0
restitution: 0.0
```

V2 运行时必须删除以下 V1 专用逻辑：

- 查找内嵌 GroundPlane。
- 对内嵌 GroundPlane 写 collision disable override。
- 断言内嵌地面存在且已被禁用。

取而代之的是启动时断言 V2 中禁止路径不存在，且 stage 中唯一训练地面来自任务配置。

### 8.4 Collision policy

V2 使用新的 `WheelOnlyCollisionV2` 标识：

- 禁用全部非轮机器人 collision，只保留左右轮 collision；不允许由源 USD 的 `jMK/jEC` 不对称状态决定运行时行为。
- 地面不再属于机器人资产子树。
- policy 不能包含“禁用资产内地面”的隐含步骤。

### 8.5 合约与 checkpoint

`Phase1ContractV4` 至少新增或更新：

- `asset_bundle_version = AssetBundleV2`。
- V2 五文件 manifest hash。
- `collision_policy_version = WheelOnlyCollisionV2`。
- `reward_schema_version = RewardSchemaV2`。
- `virtual_leg_kinematics_version = VirtualLegKinematicsV1`。
- `command_sampling_version = CommandSamplingV2` 及模式概率、范围、episode hold 语义。
- `normalization_version = NormalizationV2` 及全部字段常量。
- `phi0_symmetry` 权重和完整公式标识。
- `sim_dt=0.005 s`、`decimation=4`、`control_dt=0.02 s`。
- `solver_position_iteration_count=96`、`solver_velocity_iteration_count=4`。

ContractV4 单元测试和 Isaac Lab 集成测试必须逐字段断言上述五个物理值，并验证任一字段变化都会改变 contract hash；只验证 `sim_dt * decimation == control_dt` 不足以阻止退回历史 `0.01 s x 2` 配置。

兼容性规则：

- V3/V1 checkpoint 只能由 V3/V1 环境严格回放或继续训练。
- V4/V2 checkpoint 只能由完全一致的 V4/V2 contract resume。
- 禁止把 V1 checkpoint 的 optimizer、RNG 和 iteration 直接带入 V4 并称为 resume。
- 如果以后允许从 V1 Actor 权重 warm-start V4，必须显式标记为 transfer，重建 optimizer，并记录 parent checkpoint；这不属于第一轮推荐路径。
- 新训练默认 V2，旧 run 的 play 必须根据 checkpoint manifest 明确选择 V1，禁止静默 fallback。

### 8.6 V2 资产审计

在允许训练前，V2 必须通过：

- 五文件路径、大小、SHA256 和依赖闭包审计。
- default prim、up axis、meters-per-unit 和 articulation root 审计。
- 关节名、joint local frame/anchor、关节轴、限位、刚体、质量、惯量和 collision 审计。
- `/wheel_leg_urdf4/GroundPlane` 不存在断言。
- task ground 唯一存在且顶面为 `z=0` 的断言。
- 名义姿态轮子接触高度和 base 初始高度检查。
- V1 与 V2 除 GroundPlane 子树外的机器人语义差异检查。

V2 的实际 root prim 清单必须在删除后由 USD 审计生成，不在实现前凭经验硬编码。

若删除后差异不只位于 GroundPlane 子树、V2 仍组合出 GroundPlane、或五层依赖闭包不自包含，则丢弃本次 V2 副本并从未修改的 V1 重新复制；不得在 V1 上继续编辑，也不得用运行时禁用内嵌地面的旧逻辑把失败的 V2 当作正式结果。

## 9. MuJoCo Sim2Sim V1

### 9.1 目录边界

新目录固定为：

```text
wheelleg_dreamwaq/
  sim2sim/
    mujoco/
      pyproject.toml
      uv.lock
      README.md
      models/
        wheel_leg_rl.xml
        meshes/
      policy_runner.py
      observation_adapter.py
      actuator_adapter.py
      command_source.py
      model_map.py
      manifests/
      tests/
```

职责边界：

- `models/` 只保存 MuJoCo 模型和其自包含 mesh。
- `observation_adapter.py` 构造 ActorObsV1 并应用 checkpoint manifest 指定的 NormalizationV2。
- `actuator_adapter.py` 将 ActionV1 转成六个电机力矩。
- `policy_runner.py` 只负责调度、模型推理、日志和 viewer 循环。
- `command_source.py` 提供键盘和固定命令，不修改策略观测定义。
- `model_map.py` 只按名称解析 MuJoCo joint/body/site/actuator ID。
- `tests/` 保存 schema、golden vector、调度和模型完整性测试。

训练侧负责把 Action/Observation/Normalization/ControlFrame 的冻结值写入带 hash 的 `policy_manifest.json`；sim2sim 只读取并严格校验该 manifest，不直接导入训练包，也不在代码中维护第二套可变契约。跨引擎 golden-vector 测试负责证明 manifest adapter 与训练侧纯 Python schema 一致。

MuJoCo 运行时不得 import `wheelleg_dreamwaq`、Isaac Lab、Isaac Sim 或 RSL-RL。允许依赖 `mujoco`、`torch`、`numpy`，输入只包括自包含 MJCF/mesh、TorchScript Actor 和 JSON manifest。

MuJoCo 使用 `sim2sim/mujoco/pyproject.toml` 与独立 `uv.lock/.venv`，第一版固定 `mujoco==3.14.0`。不得把 MuJoCo 放入根训练项目 dependency group：已实测 Isaac Sim 5.1.0 固定 `websockets==12.0`，而 MuJoCo 3.14.0 要求 `websockets>=13`，单环境无法解析。USD 数据由 Isaac 环境导出为版本化 JSON，MuJoCo 子项目只消费该 JSON。两个环境的 Python/依赖版本和 lock hash 都进入 run manifest，训练环境不得因 sim2sim 改变 PyTorch/CUDA 栈。

### 9.2 MuJoCo 模型来源和身份

`MujocoModelV1` 的唯一机构源固定为：

```text
E:\wheel_leg_rl-main\wheel_leg_debug-main2\mujoco_control_extract\sim\models\wheel_leg_urdf4_self_mesh_all.xml
```

该文件已经包含 `base_free`、`0.001 s` timestep、四组闭链的八条 connect equality、六个主动 motor、轮接触 proxy 和接触排除。不得改用缺少 freejoint 的两个导出 XML，也不得重新手写 body tree。

源 XML 邻接的 `wheel_leg_urdf4_assets/` 中 16 个文件全部为零字节占位文件。项目副本必须从以下目录复制实际使用的非零 STL：

```text
E:\wheel_leg_rl-main\wheel_leg_urdf4_mjcf\wheel_leg_urdf4\meshes
```

`base_link_original.obj` 在完整目录中不存在，因此项目副本将 mesh 引用改为 `base_link.STL`；零字节 `desert.png` 引用直接删除，headless 动力学不依赖该纹理。所有 mesh 复制到 sim2sim 自有 `models/meshes/`，XML 只使用相对路径。`base_link.STL` 只能作为 visual mesh，不得参与碰撞或惯量推断；其 scale/pose 必须从 AssetBundleV2 对应 visual prim 的组合变换导出，禁止凭观察手调。模型构建审计必须比较 USD 与 MuJoCo 名义姿态的 visual world-AABB（中心和尺寸每轴误差 `<=1 mm`），并保存同视角截图供轴向/外观人工确认。

实现期记录：完整 `base_link.STL` 含 `454256` 个三角面，超过 MuJoCo 3.14.0 单 STL `200000` 面限制。构建器按原二进制 STL 三角记录顺序无损分为 `180000 + 180000 + 94256` 三部分；三部分合并后的 local AABB 与源文件逐位相同，`max_abs_local_aabb_error_m=0`。这不是几何简化，三份 mesh 继续只参与 visual。

USD 只读导出器枚举无 `CollisionAPI` 祖先的 15 个 visual mesh，并输出资产根坐标 AABB；MuJoCo 构建器在 authored `q=0` 下对 17 个 mesh geom（base 三分片加其余 14 个 visual）逐顶点计算同一资产根坐标 AABB。实测中心三轴最大绝对误差约 `5.19e-7 m`，尺寸三轴最大绝对误差约 `3.24e-7 m`，通过 `1 mm` 硬门禁。XML world 到资产根的比较必须显式移除源 XML 的 base 参考平移，禁止通过放宽容差掩盖坐标系混用。

#### 9.2.1 MujocoDummyDynamicsV1

已核对的源 XML 总质量约为 `4.276255133 kg`，AssetBundleV1/V2 权威总质量为 `4.396253988 kg`，差值约 `0.119998855 kg`，对应 USD 中必须保留的 12 个 dummy body。源 XML 不能原样用于正式 sim2sim 动力学，导出 MJCF 中 `0.001 kg` 的占位惯量也不是权威来源。

这 12 个 dummy 不是固定配重：它们在 USD articulation 中各自由一个 passive revolute joint 驱动，运行日志也显示非零关节速度。因此 V1 禁止把其名义姿态空间惯量恒定合并到相邻主连杆。

派生 MJCF 必须在源 XML 的 14 个主机构 hinge joint 基础上，按 AssetBundleV2/USD 树形拓扑显式加入 12 个 dummy body 和对应 revolute joint，使模型具有与 Isaac Lab articulation 相同的 26 个树形 hinge joint；四个 articulation 外闭环仍由源 XML 的八条 equality connect 表达。不得根据名称猜测 parent 或变换。

USD 资产中 rigid-body prim 位于资产根节点下的同级位置，其 authored joint position `q=0` 只定义 MJCF 的模型参考构型，不是训练 reset 初态。对每个 dummy，在 USD authored `q=0` 参考构型中先读取相对资产根的 composed transform `T_root_parent^0` 与 `T_root_child^0`，再生成 MuJoCo 嵌套 body 需要的父子局部变换：

```text
T_parent_child^0 = inverse(T_root_parent^0) @ T_root_child^0
```

MuJoCo child `<body pos/quat>` 使用 `T_parent_child^0`。joint anchor、local frame 和 axis 必须由 USD joint 两侧 local frame 组合得到，再转换到 MuJoCo child-body local frame；生成后必须在 world frame 对照 parent/child anchor 重合位置和轴向，禁止把 USD root-local pose 或 axis token 直接复制到嵌套 MJCF。

每个 dummy 从 USD 提取 parent/child prim path、两侧 joint local frame、axis/limit、参考 body transform、质量、COM、完整惯量和惯量坐标系。MJCF 默认 hinge `qpos=0` 对应上述 USD authored 参考构型；headless/交互 reset 则必须把 base freejoint 与全部 26 个 hinge qpos/qvel 设置为 `Phase1ContractV4` 冻结的完整初始状态，包含 `wheelleg.py` 中非零 dummy joint 初态，不能使用 USD authored joint position 替代。设置后执行 `mj_forward`，并在首个 policy tick 前检查 equality、world pose 和闭环残差。

dummy joint 固定使用 passive 参数 `stiffness=0`、`damping=0.05`、`armature=0.005`、friction 为零，不创建 motor actuator；dummy body 不创建 visual/collision geom，只保留惯性。实现生成 `artifacts/phase1_v4/mujoco-dummy-dynamics-audit.json`，分别记录 authored `q=0` 参考构型与 ContractV4 reset 构型，并记录 12 个 body/joint 的 USD prim path、parent、`T_root_body^0`、`T_parent_child^0`、两侧 joint frame、child-local axis、限位、reset qpos 和惯性，以及源/派生模型的 joint 与质量清单。

验收要求：dummy body/joint 数量均为 12；所有名称唯一；总质量与 USD 的绝对误差 `<=1e-6 kg`；逐 dummy 的质量/COM/惯量使用 `rtol=1e-6` 和对应 SI 单位 `atol=1e-8` 比较；参考构型和 ContractV4 reset 构型都要逐 dummy 对照 world body pose、joint anchor、axis direction 和 COM。任一惯量非正定、parent/axis 不唯一、reset qpos 不一致或质量守恒失败均 fail fast。

实现期记录：USD 单精度惯量量化使 `jMK` 与 `jEC` 的最大主惯量仅以约 `6.82e-13 kg m^2` 违反刚体惯量三角不等式，相对量约 `1.31e-8`，MuJoCo 因此拒绝编译。构建器不使用 `balanceinertia`，也不平均三个主惯量；只把违规的最大主惯量用 `nextafter(other_sum, 0)` 向零修正一个浮点边界，并把源值、修正值、绝对修正量和相对违规量写入 `model_manifest.json`。超过 `1e-6` 相对阈值时仍立即失败。

编译后的固定结构断言为：

```text
nbody = 28   # world + base/main bodies + 12 dummy
njnt  = 27   # 1 freejoint + 26 hinge
nq    = 33   # 7 freejoint qpos + 26 hinge qpos
nv    = 32   # 6 freejoint dof + 26 hinge dof
nu    = 6
neq   = 8
npair = 2
```

#### 9.2.2 MuJoCo WheelOnlyCollisionV2

MuJoCo 必须复现训练侧“只有两只轮与任务地面碰撞”的语义，而不是继承源 XML 的 mesh 自碰撞：

- 所有 visual mesh、base proxy、腿部、轮 proxy 和地面 geom 固定 `contype=0`、`conaffinity=0`，完全关闭自动 mask 配对。
- 只为地面与左右两个轮接触 proxy 分别创建显式 `<pair>`；MuJoCo 的 explicit pair 是这两个接触的唯一来源，禁止产生其他接触对。
- 地面顶面高度与 Isaac Lab task ground 一致；两个轮地 pair 固定 `condim=3`、`friction="1.0 1.0 0.0 0.0 0.0"`，即两个切向方向的 sliding friction 均为 `1.0`、torsional/rolling friction 为 `0`。这只对齐当前 V4 的切向摩擦目标，不声称 MuJoCo 与 PhysX 的接触求解语义逐项等价。
- 编译模型审计必须确认 `npair=2`、两个预定义 pair 的 geom 名称正确且 `pair_friction[0:2]=[1.0,1.0]`。运行时 `data.contact` 可以是允许 pair 的任意子集，不要求启动瞬间两轮都接触；但其中出现轮地集合以外的实际 contact 时立即失败。

源 XML 的轮接触 proxy 是半径 `0.0625 m` 的 sphere。V1 明确接受它作为统一的相对选优近似，并把 geom 类型/尺寸写入 manifest；四个 actor 必须使用完全相同的 proxy。该结果不能单独证明真实窄轮接触下的 sim-to-real 稳定性，若后续改成 USD 派生 cylinder/mesh collision，必须升级 `MujocoModel` 版本并重新评估全部候选。

复制后的 `MujocoModelV1` 要求：

- base 使用 freejoint。
- 保留四组闭链 equality constraint。
- 只有六个主动 motor actuator。
- 不包含旧 C 控制器、VMC 状态或强制 wheel override。
- 地面由该 sim2sim 模型明确拥有，顶面语义与 Isaac Lab task ground 一致。
- 使用完整、非零的 mesh；模型不得引用 sim2sim 目录之外的绝对路径。
- 14 个主机构 hinge、site/equality/contact-exclude 和六 motor 拓扑以源 XML 为准；唯一允许的树形拓扑扩展是按 `MujocoDummyDynamicsV1` 从 USD 生成 12 个无碰撞 dummy 惯性分支。
- 质量、惯量、joint anchor/axis 和初始姿态以 AssetBundleV2/USD 审计值为权威；源 XML 数值逐项对照，发现超出明确容差的差异时停止而不是猜测。
- active/passive actuator 参数以 Phase1ContractV4 的任务配置为权威，不使用 USD drive 的零 stiffness/无限 maxForce。
- 碰撞固定使用本节 `MuJoCo WheelOnlyCollisionV2`，不得沿用源 XML 的默认 mesh/base 碰撞和 `friction=1.2`。

`MujocoModelV1` 必须有独立 manifest，记录 XML、mesh、关节/执行器映射、equality、`MujocoDummyDynamicsV1`、`MuJoCo WheelOnlyCollisionV2`、质量、惯量、摩擦、timestep 和上述编译结构维度的 hash。manifest 还必须冻结 root body `base`、freejoint `base_free`、`R_control_from_mujoco`、COM position/velocity API 语义、地面顶面高度、authored `q=0` 参考语义、ContractV4 reset keyframe 以及 observation/action adapter 版本/hash。它与 `AssetBundleV2` 共享语义契约，但不是同一个二进制资产身份。

### 9.3 时间调度

第一版固定：

```text
MuJoCo physics dt = 0.001 s
physics steps per policy action = 20
policy/control dt = 0.020 s
policy frequency = 50 Hz
```

一个 policy tick 的顺序固定为：

1. 从当前 MuJoCo state 构造当前 ActorObsV1。
2. 使用确定性 Actor 得到 ActionV1。
3. 裁剪动作，并按 `previous_previous <- previous`、`previous <- last`、`last <- clipped` 更新动作历史；reset 时三个缓存全部归零。
4. 在接下来的 20 个 physics step 中保持动作目标不变，每个 physics step 重新计算伺服力矩。
5. 进入下一 policy tick。

不能在 20 个 physics step 中重复调用 Actor，也不能只在第一个 physics step 施加一次力矩。

### 9.4 Observation adapter

MuJoCo 原始机身坐标到 ControlFrameV1 的固定旋转为：

```text
R_control_from_mujoco = [[ 0, -1, 0],
                         [ 1,  0, 0],
                         [ 0,  0, 1]]
```

根刚体固定按名称解析为 `base`；freejoint `base_free` 只用于读写根状态，不能把 freejoint 原点误当 COM。MuJoCo freejoint quaternion 顺序固定为 `[w,x,y,z]`，表示原始 MuJoCo body 到 world 的旋转。设 `R_world_from_mujoco` 来自 `data.xmat[base]`，则 `R_world_from_control = R_world_from_mujoco @ R_control_from_mujoco.T`，所有 world vector 先乘 `R_world_from_mujoco.T` 再乘 `R_control_from_mujoco` 才进入策略坐标。

根状态语义固定为：

- COM 世界位置使用 `data.xipos[base]`；base height 为其 `z` 减地面顶面 `z`，禁止使用 `qpos[2]`。
- COM 线速度和角速度使用 `mj_jacBodyCom` 的 `jacp @ qvel`、`jacr @ qvel` 得到 world vector，再按上述矩阵转到 ControlFrameV1。
- `target_vx` 对应 ControlFrameV1 的 COM 线速度 `x`；`yaw_rate` 对应 ControlFrameV1 的角速度 `z`。
- projected gravity 使用 `R_world_from_control.T @ [0,0,-1]`；评估用 roll/pitch/yaw 也从同一个 `R_world_from_control` 提取并统一 wrap/unwrap 规则。

MuJoCo 必须逐字段构造与训练侧相同的 25D ActorObsV1：

```text
0:3    body angular velocity in ControlFrameV1
3:6    projected gravity in ControlFrameV1
6:9    CommandV1 [vx, yaw_rate, base_height]
9:13   four active leg q - q_nominal
13:19  six canonical joint velocities
19:25  previous clipped canonical ActionV1
```

随后应用 policy manifest 中同一个 NormalizationV2 offset、scale 和 clip。禁止由 MuJoCo qpos/qvel 数组顺序直接切片；所有字段必须先按名称重排并应用 canonical sign。

MuJoCo Actor 仍不输入真实线速度、真实高度或欧拉角。

### 9.5 Action/actuator adapter

策略输出先执行：

```text
clipped = clamp(action, -1, +1)
leg_target = q_nominal + 0.35 * clipped[0:4]
wheel_velocity_target = 25.0 * clipped[4:6]
```

伺服器目标：

| 通道 | 控制 | 参数 | 力矩限制 |
|---|---|---|---:|
| 4 个主动腿关节 | position PD | `Kp=120`, `Kd=4` | `18 Nm` |
| 2 个轮关节 | velocity servo | `Kd=0.6` | `9 Nm` |

四个主动腿和两个轮关节使用 `armature=0.05`。闭链被动关节固定 `damping=0.05`、`armature=0.005`，其余 friction 参数为零，与 Phase1ContractV4 一致；不得沿用源 XML default 的 `damping=0.02`、`armature=0.01`。

还必须复现：

- 有限软件关节限位。
- active joint 的 `45 rad/s` 速度安全语义：目标先按策略 scale 裁剪，PD/速度伺服力矩再裁剪到 effort limit；当 `|qd|>=45` 时，禁止施加继续增大 `|qd|` 的力矩并记录 limit event。被动关节 `80 rad/s` 只作异常阈值和评估失败条件。
- canonical joint order。
- 右轮等必要的模型轴符号变换。
- 能在 MuJoCo 中等价表达的 armature、damping 和 friction 参数。

MuJoCo 与 PhysX 的 solver/constraint 语义不同，不能通过调参伪装成逐点轨迹一致。PhysX V4 的 `96/4` articulation solver iteration 与 `0.005 s` physics dt 没有逐项等价的 MJCF 字段，必须连同 equality solver 参数和其他不可等价项写入 `MujocoModelV1` manifest。

### 9.6 策略导出

第一版 sim2sim 只加载训练后导出的确定性 PPO Actor，不加载 Critic、optimizer、RSL-RL runner 或 TensorBoard 状态。

推荐导出物：

```text
actor.ts                  # TorchScript deterministic Actor
policy_manifest.json      # schema/normalization/action/network/contract hashes
```

manifest 至少包含：

- source checkpoint hash。
- `Phase1ContractV4` hash。
- ActorObsV1、ActionV1、NormalizationV2 和 ControlFrameV1 版本/hash。
- canonical joint order、q_nominal、action scale 和 wheel sign。
- Actor 网络结构和数值 dtype。
- CommandSamplingV2 的命令范围和 episode hold 语义。
- `control_dt=0.02 s`、MuJoCo `physics_dt=0.001 s` 和 20:1 调度。
- 初始 base pose、完整初始 joint state 和 `R_control_from_usd` 数值。

ONNX/RK3566 导出属于后续部署阶段，不阻塞第一版 MuJoCo sim2sim。

### 9.7 键盘命令

第一版 viewer 支持：

```text
W/S: target_vx 增/减
A/D: target_yaw_rate 增/减
Q/E: target_base_height 增/减
R: reset
Space: 速度命令归零，高度回到 nominal
Esc: 退出
```

实际按键文字中的“增/减”以 CommandV1 数值为准，所有命令必须裁剪到训练时冻结的 command ranges。键盘只改变 CommandV1，不直接写轮速或关节目标。

### 9.8 日志和可视化

sim2sim 至少输出：

- command 与实际 `vx/yaw_rate/base_height`。
- 6D clipped action、腿目标和轮速目标。
- 六关节位置、速度、施加力矩。
- projected gravity。
- 左右真实几何 `phi0/L0` 和角差。
- policy tick 耗时、物理实时率和 missed deadline 数。

### 9.9 四次训练与 headless 选优协议

训练前生成一个 `training-suite-manifest.json`，记录四个互不相同的随机 seed。四个 run 顺序执行，均从随机初始化的同一网络结构开始，固定 `max_iterations=1000`；除 seed 和由它派生的 RNG 状态外，ContractV4、hardware profile、PPO batch、奖励、资产和网络必须完全相同。

每个 run 结束后从 `run_manifest.json` 生成种子无关训练指纹。指纹保留完整 `Phase1ContractV4`、源码/lock/dependency hash、运行栈、hardware profile、`num_envs`、rollout 和 mini-batch 语义；排除 `seed`、派生 RNG、创建时间、实际命令行和 resume 元数据。`env.yaml` 的 `seed/log_dir` 与 `agent.yaml` 的 `seed/run_name` 会导致原始文件 hash 必然不同，因此这两个原始 hash 不直接进入跨 run 指纹；它们仍保留在 run manifest 中，而环境/算法语义由完整 ContractV4 逐字段覆盖。第二个及后续 run 的指纹不等于第一个时必须立即停止 suite，禁止继续导出或排名。

训练期间在启动后检查一次，并在仍有训练运行时每隔约 30 分钟检查 TensorBoard/event 与进程日志。异常至少包括 NaN/Inf、episode length 突降、termination 激增、value loss 发散、entropy 突然塌缩、动作/力矩长期饱和、`phi0` 误差增大或命令跟踪失效。若四次训练在首个 30 分钟检查点前已全部完成，则以完成后的统一全量检查代替无意义的等待。

监控输出区分 warning 与 hard anomaly。非有限标量、`Termination/invalid>0`、闭链误差超过 `5 mm`、value loss 超过冻结的失控阈值，以及训练结束后 event 文件或必需 tag 缺失属于 hard anomaly；动作/力矩长期饱和、跟踪/`phi0` 趋势恶化、episode/entropy 塌缩和 termination 比例过高属于 warning。启动和中间快照中暂时缺少 event/tag 可以记录，但只要 `post_run` 全量快照恢复正常，就不得把该可恢复的可用性缺失累计成 suite warning。任一 hard anomaly 必须写入最终 ranking 的 `training_monitor_gate`，将 suite 标为 `review_required`，清空 `selected`，并只允许保留 `diagnostic_candidate`。

每个最终 actor 使用完全相同的 MuJoCo deterministic headless 场景集，每个场景持续 `10 s`：

| 场景 | `[vx, yaw_rate, base_height]` |
|---|---|
| nominal stand | `[0.0, 0.0, 0.20]` |
| low stand | `[0.0, 0.0, 0.17]` |
| high stand | `[0.0, 0.0, 0.23]` |
| forward | `[1.0, 0.0, 0.20]` |
| reverse | `[-1.0, 0.0, 0.20]` |
| left turn | `[0.0, 0.6, 0.20]` |
| right turn | `[0.0, -0.6, 0.20]` |
| combined | `[0.8, 0.5, 0.20]` |

每个场景必须从同一个完整名义 keyframe 独立 reset：恢复 root qpos/quaternion、全部 joint qpos/qvel、actuator/internal state、仿真时间和 equality state，并把 `last/previous/previous_previous action` 全部清零；不得把前一场景终态延续到下一场景。命令在 reset 后写入并保持 10 s。所有场景权重相同，场景顺序固定进入 suite manifest。

每个 run 输出逐 tick CSV 和汇总 JSON，至少包含 survival、height MAE/max error、roll/pitch RMS/max、零 yaw 命令场景的 yaw drift、非零 yaw 命令场景的 yaw-rate MAE、左右 `phi0` 差 RMS/max、`vx` MAE、action/effort saturation 和闭链残差。`yaw_drift` 定义为零 yaw 命令场景中的 `unwrap(yaw-yaw_initial)`；非零 yaw 命令场景不计算 yaw drift，只计算 yaw-rate MAE。每项汇总指标先在其适用场景内逐场景计算，再做等权平均；`vx`、height、roll、pitch 和 `phi0` 指标适用于全部场景。

每份正式报告还必须携带 `MujocoEvaluationContractV1` 及其 hash，至少冻结场景顺序/命令、每场景 `500` policy ticks/`10 s`、physics/control period、模型 XML/manifest/完整动力学语义 hash、adapter 版本和失败阈值。排名器必须拒绝 smoke 报告、不同 evaluation contract、场景不完整、checkpoint hash 缺失或重复的输入，不能把不同模型或不同评估条件下的结果放进同一排行榜。

任何出现 NaN、跌倒、闭链失效或场景未完整存活的 run 排在完整存活 run 之后。完整存活 run 的选优分数固定为下式，越低越好，并同时保留所有原始指标供人工复核：

```text
score = vx_mae / 1.5
      + yaw_rate_mae / 1.0
      + height_mae / 0.04
      + roll_rms / rad(10)
      + pitch_rms / rad(10)
      + yaw_drift_rms / rad(15)
      + phi0_delta_rms / rad(10)
```

未完整存活的 run 依次按“完整完成场景数更多、全部场景总 survival fraction 更高、有效存活前缀上的同公式 score 更低”排序，不得把失败后的缺失样本当零误差。若某场景在首个有效 tick 前就失败，其有效前缀 score 固定为 `+inf`。若四个 run 全部失败，可以报告最少失败的诊断候选，但必须明确判定 Phase 1 未通过，不能把该候选称为合格最佳模型。

分数相同到 `1e-6` 时，依次选择 effort saturation 更低、action saturation 更低、最大 tilt 更低的 run。完成训练与 sim2sim 后启动一个覆盖四个 run 的 TensorBoard，不只打开单个 run。

## 10. 跨引擎单一事实来源

以下数据只能有一个权威定义，Isaac Lab、MuJoCo 和未来部署 adapter 必须引用或由 manifest 生成，不能分别手写：

- canonical joint order 和 joint name。
- ActionV1 scale、clip 和 q_nominal。
- ActorObsV1 slice。
- NormalizationV2 offset、scale 和 clip。
- ControlFrameV1 语义。
- command ranges。
- CommandSamplingV2 模式概率与 episode hold 语义。
- policy control period。
- virtual-leg 左右 joint/axis mapping。
- policy export manifest。

USD 和 MJCF 的原始坐标可以不同，但进入策略前必须产生相同语义的 canonical tensor。

## 11. 测试设计

### 11.1 VirtualLegKinematicsV1

- 名义关节角参考值测试。
- 左右镜像姿态得到相同 `phi0` 的测试。
- `phi0` 跨 `-pi/pi` 时 wrapped difference 连续性测试。
- 真实轴心几何链与修正 FK 的随机合法姿态对照。
- USD anchor 审计工件完整性、左右轴向和来源 layer 测试。
- 右腿 `[jAB,jAG]` 名义 golden vector 及左右差值上限测试。
- body origin 与 joint axis 不重合时能检出错误实现的测试。
- 非法闭链、错误支路、NaN 和 `L0` 近零 fail-fast 测试。

离线无求解器残差 golden-test 验收阈值：

```text
abs(L0_fk - L0_true) < 0.1 mm
abs(wrap(phi0_fk - phi0_true)) < 0.1 deg
```

运行时测试另外要求每侧闭环残差 `<=5 mm`，并直接检查第 6.5 节定义的 `e_W<=5 mm`、`e_L<=5 mm`、`e_phi<=3 deg`；不得用闭环残差公式推导轮轴端点或角度误差，也不得把 `0.1 mm/0.1 deg` 直接施加到含物理求解残差的每一帧。

### 11.2 RewardSchemaV2

- 左右角相等时误差为零。
- `+pi/-pi` 邻域角差正确 wrap。
- 交换左右腿不改变误差。
- 奖励权重为负且统一乘 `control_dt`。
- 固定权重为 `-1.0`，并验证 `10 deg/30 deg` 数值尺度。
- 增加字段后 Actor/Critic tensor bitwise schema 不变。
- reset 后第一帧没有跨 episode 的旧 `phi0` 缓存。

### 11.3 AssetBundleV2

- 五层资产 hash/size/依赖闭包测试。
- 禁止 GroundPlane prim 路径测试。
- V1/V2 机器人 prim 语义对照测试。
- 唯一 task ground、顶面高度和碰撞开启测试。
- 名义落地姿态无半轮穿地或悬空异常。
- V3 checkpoint 拒绝 V4 环境、V4 checkpoint 拒绝 V3 环境。
- GroundPlane 删除前 authoring-layer 工件与删除后 prim-stack 为空测试。

### 11.4 MuJoCo sim2sim

- XML、mesh、freejoint、四组 equality 和六 actuator 完整性测试。
- 12 个 dummy body/revolute joint 的 USD parent、composed root pose、父子局部变换、两侧 joint frame、child-local axis、limit、ContractV4 reset qpos 和惯量生成测试。
- 编译模型 `nbody=28/njnt=27/nq=33/nv=32/nu=6/neq=8/npair=2` 断言，以及 authored `q=0`/ContractV4 reset 两种构型的逐 dummy world pose、anchor、axis、COM 对照测试。
- visual world-AABB/pose 审计，以及 `base_link.STL` 不参与 collision/inertia 的测试。
- 编译模型严格只有两个左右轮-地面显式 pair、双切向摩擦 `[1.0,1.0]`，运行时 contact 只能属于允许集合的测试。
- joint/body/site/actuator 名称唯一解析测试。
- qpos/qvel 到 canonical joint order/sign 测试。
- ActionV1 到目标及饱和力矩测试。
- 20:1 physics/policy 调度测试。
- Isaac Lab 与 MuJoCo observation adapter 的 golden-vector 对照测试。
- 非零角速度和非零姿态下的 COM 速度、body/freejoint origin 区分及 `R_control_from_mujoco` golden-vector 测试。
- 导出 Actor 与训练侧 Actor 对相同 25D 输入的输出一致性测试。
- 名义姿态静置、零命令短跑和有限命令短跑无 NaN 测试。
- CommandSamplingV2 episode 内保持、NormalizationV2 与策略 manifest 对齐测试。
- 四个 actor 使用相同 headless 场景顺序、时长和选优公式的可复现测试。
- 每场景独立 keyframe reset、适用场景指标聚合、yaw unwrap 和失败 run 排名测试。
- 编译后完整动力学语义 hash 重算测试，覆盖 joint/dof、body inertia、actuator、equality、contact pair 和 reset keyframe；XML 或 mesh 被修改后必须拒绝。
- 正式评估指纹测试：拒绝 smoke、不同场景/时长/模型、缺失场景及重复 checkpoint。
- 四次训练指纹测试：只改变 seed/run name/log directory 时相同，改变 ContractV4、hardware profile、源码或依赖时不同。
- 监控门禁测试：启动期可恢复的 event/tag 缺失不污染最终结果，post-run 缺失或 hard anomaly 必须阻止正式选择。

两个引擎的 deterministic Actor 输出必须一致；完整动力学轨迹只要求趋势和稳定性可解释，不要求逐帧相同。

## 12. 实施顺序和门禁

依赖关系和审查门禁必须按以下顺序满足：

1. 本文与主架构完成独立只读复核；存在 P0/P1 时先改文档并重新复核。
2. 创建并审计 `AssetBundleV2`，证明只删除了内嵌地面；同时生成 USD anchor/axis 审计工件。
3. 实现 CommandSamplingV2、NormalizationV2、VirtualLegKinematicsV1 与 RewardSchemaV2，并完成单元/Isaac Lab 集成测试。
4. 以指定本地 XML 建立 `MujocoModelV1`、adapters、Actor 导出和跨引擎 golden-vector 测试。
5. 对全部代码和资产副本做独立只读复核；存在 P0/P1 时修复并重新复核。
6. 运行有限迭代 smoke test，确认 ContractV4、奖励尺度和日志正常。
7. 生成四个 seed，依次从头训练四次 1000 iterations；训练中按约定监测异常。
8. 导出四个 Actor，执行统一 MuJoCo headless 评估、保存数据并选出最佳 run。
9. 启动覆盖四个 run 的 TensorBoard；之后再决定是否进入 DreamWaQ 或 RK3566 部署工作。

任何一步失败时都不得通过放宽 contract hash、跳过轴向检查或静默使用旧资产继续推进。

## 13. 验收标准

本设计对应实现只有在以下条件全部满足后才算完成：

- `AssetBundleV1` 的五个源文件哈希未变化。
- `AssetBundleV2` 自包含且不含内嵌 GroundPlane。
- Isaac Lab 场景中只有任务地面承担碰撞。
- `phi0` 真值由髋轴到真实轮轴的几何定义产生。
- 修正 FK 与真值在约定阈值内一致。
- Isaac Lab V4 使用 `sim_dt=0.005 s`、`decimation=4`、solver `96/4`，且 16 环境 1000-step 随机动作门禁通过；`control_dt` 仍为 `0.02 s`。
- 新奖励只约束左右差值，不包含绝对 `90 deg` 目标。
- ActorObsV1、CriticObsV1 和 ActionV1 的维度与顺序不变；归一化显式升级到 NormalizationV2 并拒绝 V3 checkpoint。
- CommandSamplingV2 只在 reset 采样，命令在完整 episode 内保持不变。
- V3/V1 与 V4/V2 checkpoint 不会被静默混用。
- MuJoCo 以 1 kHz physics、50 Hz policy 运行同一个确定性 Actor。
- MuJoCo observation/action golden vector 与训练侧通过。
- MuJoCo 12 个可动 dummy 分支、逐 body 惯量和总质量与 USD 一致，且只有左右轮 proxy 与地面接触。
- viewer 的键盘只修改 CommandV1。
- 四个不同 seed 的 1000-iteration run 均从头训练，且除 seed/RNG 外合约完全一致。
- 四个 actor 完成同一 headless 场景集并生成逐 tick CSV、汇总 JSON 和明确选优结果。
- 全部运行资产和导出模型都有可复现 manifest/hash。

## 14. 风险与约束

| 风险 | 约束 |
|---|---|
| 错把 body origin 当 joint axis | 必须先审计 anchor；通过随机姿态几何测试。 |
| 左右局部坐标符号不一致 | 统一到 VMC 侧视坐标；显式验证 `s_W=-v_control_x`、`d_W=-v_control_z`。 |
| 固定 phi0 权重损害主任务 | 正式训练前做数值 smoke；四个 seed 共用同一权重，失败时回到新 contract 评审，不在 run 间改参。 |
| 修改 USD 导致旧 checkpoint 失效 | V1 永久保留，新建 V2，按 manifest 选择。 |
| MuJoCo 数组顺序与 canonical 顺序混淆 | 全部按名称解析，禁止裸切片。 |
| 源 XML 缺失约 `0.12 kg` 可动 dummy 动力学 | 从 USD 显式生成 12 个 dummy body/revolute joint 惯性分支；禁止名义姿态惯量合并。 |
| MuJoCo 默认 mesh/base 接触污染 sim2sim | visual/非轮 geom 禁用接触，只允许两个显式轮地 pair。 |
| 把 freejoint 原点当作 COM | 高度使用 `xipos`，速度使用 `mj_jacBodyCom`，用非零角速度 golden test 检出。 |
| 把 USD root-local pose 当 MuJoCo parent-local pose | 从 composed root transform 计算 `inverse(T_root_parent) @ T_root_child`，并逐 dummy 做 world-frame golden test。 |
| 把 USD authored `q=0` 当训练初态 | `q=0` 只定义模型参考构型；reset 必须写入 ContractV4 完整 base 与 26 关节状态。 |
| PhysX/MuJoCo 约束差异被误判为策略错误 | 区分张量一致性测试和动力学趋势测试。 |
| V4 physics 子步不足导致闭环 FK 误差越界 | 固定 `sim_dt=0.005 s`、`decimation=4`、solver position iterations `96`、solver velocity iterations `4` 并进入 contract；不得在 hardware profile 中改写。 |
| sim2sim adapter 与训练 schema 漂移 | 训练侧生成唯一 policy manifest，sim2sim 严格校验 hash，并用跨引擎 golden vector 检测。 |
| 旧 checkpoint 被错误续训 | contract hash fail-fast，warm-start 与 resume 分离。 |

## 15. 设计记录

| ADR | 结论 |
|---|---|
| ADR-I01 | `phi0` 定义为修正后真实 `I -> W` 虚拟腿极角。 |
| ADR-I02 | 仿真奖励使用关节轴几何真值，joint FK 只作诊断和未来实机估计。 |
| ADR-I03 | `phi0_symmetry` 使用 wrapped angle difference 的平方，不设置绝对角目标。 |
| ADR-I04 | `phi0` 不进入 Actor/Critic observation，公开张量维度不变。 |
| ADR-I05 | 保留 AssetBundleV1，新建无内嵌地面的完整 AssetBundleV2。 |
| ADR-I06 | 新任务升级为 Phase1ContractV4，不允许从 V3 直接 resume。 |
| ADR-I07 | MuJoCo sim2sim 是项目内独立子工程，运行时不依赖 Isaac Lab/RSL-RL。 |
| ADR-I08 | 第一版 MuJoCo 固定 1 kHz physics 与 50 Hz policy。 |
| ADR-I09 | sim2sim 执行四腿位置 PD、双轮速度伺服，与训练动作契约一致。 |
| ADR-I10 | 第一版加载 TorchScript deterministic Actor；ONNX/RK3566 后置。 |
| ADR-I11 | CommandSamplingV2 每个 episode 只采样一次，模式概率和命令范围固定进入 ContractV4。 |
| ADR-I12 | NormalizationV2 固定 `vx_max_abs=1.5`、`yaw_rate_max_abs=1.0`，V3/V1 checkpoint 不兼容。 |
| ADR-I13 | 第一轮正式 `phi0_symmetry` 权重固定为 `-1.0`，四次训练只改变 seed。 |
| ADR-I14 | `MujocoModelV1` 直接派生自 `wheel_leg_urdf4_self_mesh_all.xml`，用完整 STL 替换零字节资源。 |
| ADR-I15 | 四个 actor 使用统一 headless 场景和固定选优公式比较，原始指标与汇总结果都必须保存。 |
| ADR-I16 | MuJoCo 动力学质量/惯量以 USD 为权威；12 个可动 dummy 必须作为无碰撞 passive body/revolute joint 分支显式生成，禁止名义姿态惯量合并或占位惯量。 |
| ADR-I17 | MuJoCo 只允许左右轮 proxy 与地面显式接触，滑动摩擦为 `1.0`；visual/base/腿部 geom 不参与碰撞。 |
| ADR-I18 | MuJoCo 根高度和速度使用 base COM，固定 `R_control_from_mujoco`；freejoint 原点不得替代 COM。 |
| ADR-I19 | headless 场景独立从同一 keyframe reset；指标按适用场景等权聚合，失败 run 使用 survival-first 排名，全部失败则 Phase 1 不通过。 |
| ADR-I20 | USD authored `q=0` 只定义 MuJoCo 参考构型；dummy 嵌套 pose 由 composed root transform 转为 parent-local，运行 reset 使用 ContractV4 的完整 26 关节初态。 |
| ADR-I21 | V4 使用 `sim_dt=0.005 s`、`decimation=4`、PhysX articulation solver `96/4`；50 Hz policy 和 `control_dt=0.02 s` 不变。 |
| ADR-I22 | `base_link.STL` 为满足 MuJoCo 面数限制进行二进制无损分片；分片合并 AABB 必须与源 STL 完全一致。 |
| ADR-I23 | `jMK/jEC` 只允许记录在 manifest 中的浮点边界惯量修正；任何更大量级违规都 fail fast。 |
| ADR-I24 | USD/MuJoCo authored `q=0` visual AABB 在同一资产根坐标比较，中心与尺寸每轴误差固定 `<=1 mm`。 |
| ADR-I25 | MuJoCo policy 加载同时校验 Actor、policy manifest、模型 manifest/XML 和 observation/action adapter 实现 hash。 |

## 16. 本地依据

- 主架构：`E:\wheel_leg_rl-main\docs\2026-10-03-wheelleg-dreamwaq-architecture.md`
- 真实偏置闭链推导：`E:\wheel_leg_rl-main\offset_closed_chain_kinematics_derivation.html`
- 推导/验证脚本：`E:\wheel_leg_rl-main\wheel_leg_debug-main2\mujoco_control_extract\tools\derive_offset_kinematics.py`
- CAD 单闭链近似来源：`E:\wheel_leg_rl-main\serial_leg_rl\source\serial_leg_rl\serial_leg_rl\kinematics\offset_leg.py`
- ControlFrame 变换：`E:\wheel_leg_rl-main\wheelleg_dreamwaq\source\wheelleg_dreamwaq\wheelleg_dreamwaq\schemas\frames.py`
- 当前 Isaac Lab 奖励：`E:\wheel_leg_rl-main\wheelleg_dreamwaq\source\wheelleg_dreamwaq\wheelleg_dreamwaq\tasks\direct\wheelleg_flat\rewards.py`
- 当前环境配置：`E:\wheel_leg_rl-main\wheelleg_dreamwaq\source\wheelleg_dreamwaq\wheelleg_dreamwaq\tasks\direct\wheelleg_flat\env_cfg.py`
- 当前资产合约：`E:\wheel_leg_rl-main\wheelleg_dreamwaq\source\wheelleg_dreamwaq\wheelleg_dreamwaq\assets\asset_contract.py`
- 当前 V1 USD：`E:\wheel_leg_rl-main\wheel_leg_urdf4_usd (1)\wheel_leg_urdf4\wheel_leg_urdf4.usd`
- MuJoCo 机构源 XML：`E:\wheel_leg_rl-main\wheel_leg_debug-main2\mujoco_control_extract\sim\models\wheel_leg_urdf4_self_mesh_all.xml`
- 可用完整 MuJoCo mesh：`E:\wheel_leg_rl-main\wheel_leg_urdf4_mjcf\wheel_leg_urdf4\meshes`
