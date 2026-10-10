# WheelLeg DreamWaQ Isaac Lab Architecture

日期：2026-10-10  
状态：Architecture v0.25；Phase 2 DreamWaQ/CENet 实现契约已冻结；v0.25 按用户授权删除三类人工限速作用，当前动力学为 PhysicsV5 / UnrestrictedVelocityPolicyV1 / MujocoEvaluationContractV3，见 §30；§29 的 PhysxRigidAngularBiasV1 仅作为历史记录；v0.23 的两个 reset 派生缓存失效修复继续有效；保留 deterministic Isaac evaluation 499-step timeout、网络维度、数学契约、termination 与 MuJoCo 500-tick 语义；旧 PhysicsV4 训练工件继续作为历史对照，不可当作 PhysicsV5 续训来源  
适用范围：新建轮腿机器人平地强化学习工程，不修改或继承旧 WheelLeg 任务实现

## 1. 文档目的

本文定义新 WheelLeg DreamWaQ 工程的系统边界、运行版本、模块职责、张量契约、训练阶段和部署接口。

本文定义工程架构和已经确认的行为，不在本文中创建工程代码。Phase 2 DreamWaQ 的网络维度、张量布局、训练/推理路径、loss、storage、checkpoint、导出和测试契约已经冻结；v0.21 进一步冻结 context-collapse/velocity-estimation 的最低诊断门槛、可复现 Isaac evaluation 协议及其 resume 状态，并让该 evaluator 直接服从当前任务已有的 timeout off-by-one 行为，不改变 loss、梯度、网络、termination 或其他任务语义。仍需实验标定的后续鲁棒性范围与最终部署后端继续由对应 Gate 管理。后续实现必须以本文为上游规范，不得在代码中另行发明不一致的隐式约定。

v0.22 补充 Isaac reset 角速度缓存的设计与验收边界；v0.23 在真实 red probe 确认 COM/height 也存在旧缓存后，冻结两个派生缓存失效的最小修复。目标是让现有观测如实反映本次 reset 状态；不增加 reset 物理步，不改变初态采样、Actor/Critic 输入定义、DreamWaQ history 布局或 MuJoCo 运行参数。设计见 `wheelleg_dreamwaq/docs/2026-10-09-wheelleg-reset-observation-cache-coherence-design.md`；实际代码复核、回归与八场景诊断见 `wheelleg_dreamwaq/docs/2026-10-09-wheelleg-reset-observation-implementation-results.md`。这些结果不替代 RootCauseSuite 正式 G01/G02。

## 2. 项目目标

V1 建立一个可验证、可扩展、可部署的轮腿强化学习基线，按以下顺序推进：

1. 验证 USD、关节顺序、关节方向、限位和 actuator。
2. 建立六维纯关节空间控制的最小 `DirectRLEnv`。
3. 训练普通 asymmetric PPO 平地基线。
4. 普通 PPO 基线通过后，用 `FudanStyleDomainRandomizationV1` 建立 Phase 1R 随机化 PPO，并保留四次独立训练与 MuJoCo sim2sim 结果作为对照证据。
5. Phase 1R 的实现、训练和 sim2sim 诊断链路建立后，可以在不改变环境任务定义的前提下接入 DreamWaQ/CENet。Phase 1R 的 MuJoCo 闭环是否已经稳定不再作为“允许编写 Phase 2 代码”的前置条件；DreamWaQ 必须与同任务定义的 PPO/Phase 1R checkpoint 做受控对比，不能用来掩盖观测、动作、初态或仿真模型映射错误。
6. PPO、Phase 1R 和 DreamWaQ 都稳定后，再扩展质量/质心/惯量、延迟、外力、复杂地形和更完整的 privileged critic。
7. 最终导出只依赖实机可获取信号的策略，在选定的实机计算平台上以 50 Hz 执行。H7、RK3566 或其他平台只影响部署后端和资源预算，不改变训练 schema 与策略语义。

V1 任务目标：

- 平地保持平衡。
- 跟踪机身前向速度 `target_vx`。
- 跟踪偏航角速度 `target_yaw_rate`。
- 跟踪可调机身高度命令 `target_base_height`。
- 不训练横向速度 `vy`。
- 不训练跳跃。
- 不训练复杂地形。

## 3. 非目标

以下内容不属于 V1：

- 复制旧 WheelLeg 的环境文件作为新任务基础。
- 将复旦 Isaac Gym 环境迁移到 Isaac Lab。
- 在主控制路径中使用五连杆 IK/FK 或 VMC。
- 跳跃、腾空控制和跳跃奖励。
- 相机、深度图、激光雷达或地形感知策略。
- AdaBoot。
- 训练 step 内的 CSV、JSON 或其他磁盘 I/O。
- 在普通 PPO 基线通过之前接入 DreamWaQ。
- 在 privileged 字段没有真实含义时，用零填充把 Critic 强行扩展到 141 维。

## 4. 资产契约与参考工程权限边界

### 4.1 正式 WheelLeg USD 资产

V1 唯一允许加载的机器人资产定义为不可变的 `AssetBundleV1`，资产根目录为：

```text
E:\wheel_leg_rl-main\wheel_leg_urdf4_usd (1)\wheel_leg_urdf4
```

入口层固定为：

```text
wheel_leg_urdf4.usd
```

`AssetBundleV1` 由以下五个文件共同构成。任何一个文件的内容、相对路径或大小变化都视为资产身份变化：

| 相对路径 | Bytes | SHA256 |
|---|---:|---|
| `wheel_leg_urdf4.usd` | 18,571 | `F371163FF638D3A151B303A202CE9EACD208E04751538FA1BCCBE4F9E5AA7F44` |
| `configuration/wheel_leg_urdf4_base.usd` | 35,535,228 | `D980BE2205D0078AA7845AEE7C22B5C7F04FD5A05E79D327FC7E3ED28066C28E` |
| `configuration/wheel_leg_urdf4_physics.usd` | 5,371 | `A8F67B6BDD1210D13AE4BD1254BE8BB87662312BC713046E5CB9478A8C0A2CE5` |
| `configuration/wheel_leg_urdf4_robot.usd` | 2,452 | `81415C4FB375DAEC1A8DDFD0FE5CEAB6A1904FB77F29050149DED24C1F3436A1` |
| `configuration/wheel_leg_urdf4_sensor.usd` | 655 | `D31E77E5C105470304DE4F5D8CAA9B3DB2A4D70ECC245510AD96C53F185A8314` |

入口层的固定元数据为：

| 字段 | 固定值 |
|---|---|
| Default prim | `/wheel_leg_urdf4` |
| Articulation root | `/wheel_leg_urdf4/base_link` |
| Up axis | `Z` |
| `metersPerUnit` | `1.0` |

入口层通过相对 reference/payload 组合四个 `configuration` 层，`wheel_leg_urdf4_physics.usd` 还引用基础层。V1 保留该分层结构，不 flatten，也不修改或重新导出任何源 USD。启动时必须逐项校验五个文件的相对路径、大小和 SHA256，并审计资产依赖子树；除运行时匿名/session layer 和经批准的薄 overlay 外，不允许解析到 `AssetBundleV1` 清单之外的资产层。

训练需要的差异通过加载时匿名/session layer 覆盖；如果该方式不能稳定复现，则新增只引用入口层的薄 `training_overlay.usda`。overlay 自身必须具有独立版本和 SHA256，不能替代五个源文件的校验。

已核对出的资产事实：

- 27 个 rigid body。
- 26 个树形/reduced-coordinate revolute joint。
- 4 个启用的闭环 joint：`/wheel_leg_urdf4/jIO/jIO_loop_closure`、`/wheel_leg_urdf4/jKN/jKN_loop_closure`、`/wheel_leg_urdf4/jEC/jAG_loop_closure`、`/wheel_leg_urdf4/jCF/jCF_revolute_joint`。它们保持 `jointEnabled=true`，并通过 `excludeFromArticulation=true` 作为 articulation 外的闭环约束参与物理求解。
- 入口层根部还包含与 default prim 平级的 `/physicsScene`。OpenUSD 静态审计得到入口层 root prim 路径为 `/physicsScene`、`/wheel_leg_urdf4`、`/Render`、`/World`，default prim 为 `/wheel_leg_urdf4`。机器人必须通过 Isaac Lab `UsdFileCfg` 对 default prim 做 reference 加载，禁止把整个入口层直接作为 stage 打开或 sublayer 到任务场景。按 default prim reference 加载时该平级 `PhysicsScene` 不进入机器人命名空间；运行时仍必须断言任务 stage 只有一个由环境配置创建的全局 PhysicsScene，且机器人环境命名空间下不存在 PhysicsScene prim。Phase 0 必须把 root prim 和 reference 后 prim 路径清单保存为审计工件。
- USD 内嵌 `/wheel_leg_urdf4/GroundPlane/CollisionPlane`，且碰撞默认启用。训练加载层必须在首次 physics initialization 前禁用该碰撞体，环境只创建自己的平地；禁止同时存在两块地面。
- 六个受控 joint 的 USD position limit 均为 `[-inf, +inf]`，drive 的 stiffness、damping 为 0，`maxForce` 为无穷。任务配置必须提供显式软件限位、目标限位、速度/力矩限制和 actuator 参数，不能把 USD 默认值当安全边界。
- 该资产的 `base_link` 局部原点不是机身物理中心。Phase 0 运行时实测在初始姿态下 `root COM - root link origin = [0.041710, -1.273885, 0.010447] m`。因此 `root_link` 原点在机身 pitch/roll 时会绕 COM 产生大幅位置变化，禁止把 `root_link z` 当作机身高度。任务高度、奖励、终止和 Critic privileged height 统一使用 `robot.data.root_com_pos_w[:,2] - env_origin_z`；集成测试必须同时记录 root-link 与 root-COM 高度，以防回归。
- 资产总质量为 `4.396253988 kg`；12 个 dummy body 合计 `0.119999997 kg`，约占总质量 `2.73%`。Phase 0 决定 V1 保留这些质量，不修改源 USD，也不通过派生层重标定。
- 源 USD 的镜像 collision 配置不对称已经确认：`jMK` 启用而 `jEC` 禁用。V1 不猜测其机械意图，而是在加载层统一应用 `WheelOnlyCollisionV1`：禁用内嵌地面和全部非轮机器人 collision，只保留左右轮 collision。该覆盖在 physics initialization 前完成，并进入 checkpoint contract；源 USD 保持不变。Phase 1 不启用接触传感器、collision reward 或 collision termination。

闭环腿按真实约束求解，不复制旧工程对 passive joint 的高刚度 PD。passive joint 使用零 stiffness、`0.05` damping 和 `0.005` armature。历史 V3 基线使用 `sim_dt=0.01 s`、`decimation=2` 和 `32/4` articulation solver iteration；V4 几何门禁实测该组合在每个控制 tick 白噪声随机动作下可产生约 `10 mm` 的 FK-to-axis 轮轴误差，不能满足固定 `5 mm` 门槛。V4 因此固定为 `sim_dt=0.005 s`、`decimation=4` 和 `96/4` solver iteration，控制周期仍为 `0.02 s`。16 环境、1000 control-step 门禁实测最大 loop-anchor 残差约 `0.389 mm`、轮轴直接误差约 `2.810 mm`、长度误差约 `2.319 mm`、角度误差约 `0.518 deg`。随机动作测试逐 step 计算四个 excluded loop joint 两侧锚点的世界坐标残差，并同时监控关节速度、加速度、受控关节力矩、有限值和 simulator error。Isaac Lab 2.3.2 的标准 batched `ArticulationData` 不暴露这四个 articulation 外约束的逐约束 wrench，因此 V4 不伪造“内部约束力”指标；如后续启用 PhysX 约束力查询，必须作为新的诊断字段加入而不能改变现有阈值含义。

### 4.1.1 Phase 0 运行时校准冻结值

以下数值构成 `Phase1ContractV4` 首个 PPO checkpoint 的物理契约，修改任一项都必须产生新的 contract hash。历史 `Phase1ContractV3` 继续使用其原有 V1 资产和 collision policy，不得与本表混用：

| 项目 | Phase 0 冻结值 |
|---|---|
| root-link 初始高度 | `0.20003 m` |
| 初始 root-COM 高度 | 约 `0.210477 m`，任务高度始终读取运行时 COM |
| `q_nominal` | `[-0.33367134, +0.33367134, -0.33367134, +0.33367134] rad`，顺序为 ActionV1 前四维 |
| 腿动作 | `q_target = clamp(q_nominal + 0.35 * action, -1.0, +1.0)` |
| 轮动作 | control-frame `25.0 rad/s * action`，再应用 USD `[+1,-1]` 符号 |
| 腿 actuator | effort `18 Nm`，求解器速度限速禁用（§30；历史 V4 为 `45 rad/s`），stiffness `120`，damping `4`，armature `0.05` |
| 轮 actuator | effort `9 Nm`，求解器速度限速禁用（§30；历史 V4 为 `45 rad/s`），stiffness `0`，damping `0.6`，armature `0.05` |
| passive actuator | effort `18 Nm`，求解器速度限速禁用（§30；历史 V4 为 `80 rad/s`），stiffness `0`，damping `0.05`，armature `0.005` |
| V4 physics/control timing | `sim_dt=0.005 s`、`decimation=4`、`control_dt=0.02 s` |
| V4 articulation solver | position iterations `96`、velocity iterations `4` |
| collision policy | `WheelOnlyCollisionV2` |
| 环境平地 | `100 x 100 x 0.10 m` cuboid，顶面 `z=0`，静/动摩擦均为 `1.0`，restitution `0` |
| 闭环残差门槛 | 四个 loop-joint 锚点最大距离 `<= 0.005 m` |
| scene cloning | `replicate_physics=true`、`clone_in_fabric=false`；后者用于规避本栈已复现的 Fabric clone failure，不改变任务张量语义 |

### 4.1.2 AssetBundleV2

`Phase1ContractV4` 不修改 `AssetBundleV1`，而是在 `E:\wheel_leg_rl-main\wheel_leg_urdf4_usd_v2\wheel_leg_urdf4` 建立完整、自包含的五层副本。V2 相对 V1 唯一允许的机器人资产变化是从实际 authoring layer 删除 `/wheel_leg_urdf4/GroundPlane` 子树；机器人 articulation、质量、惯量、关节、碰撞和材质不得顺带变化。V2 使用 `WheelOnlyCollisionV2`，不再包含“禁用资产内嵌地面”的运行时步骤，任务配置仍创建唯一 `/World/Ground`。详细删除、anchor 审计、manifest 和失败回退规则见 `docs/2026-10-04-wheelleg-phi0-sim2sim-usd-cleanup-design.md`。

### 4.2 控制坐标系 ControlFrameV1

策略、命令、奖励、CENet 监督和实机协议统一使用右手 `ControlFrameV1`：

- `+X`：机器人前向。
- `+Y`：横向，符号遵循现有控制器约定。
- `+Z`：向上。

当前 USD `base_link` 的原始轴不等于控制前向轴。根据现有控制器已使用的变换，所有 USD base-frame 三维向量必须先执行：

```text
v_control = R_control_from_usd * v_usd

R_control_from_usd =
[[ 0, -1, 0],
 [ 1,  0, 0],
 [ 0,  0, 1]]
```

该变换适用于 root COM 线速度、root 角速度、projected gravity 以及其他进入观测、奖励或监督目标的三维向量。`target_vx` 永远表示 `ControlFrameV1 +X`，不得解释为 Isaac/USD `base_link +X`。Isaac Lab 中应显式读取 root COM 线速度语义，不能把 root link position、root COM position 或默认 heading 约定混用；依赖 base `+X` 的 heading API 不得未经变换直接使用。

实机 IMU 已按控制坐标正向安装，因此 V1 的预期安装矩阵为：

```text
R_control_from_imu = I
```

这只是已知安装配置，不替代验证。Phase 0/部署一致性测试必须用静止六面、单轴正转和四元数重力方向测试确认轴序与符号。实机四元数统一为 `[w, x, y, z]`、表示 control/body 到 world 的旋转；projected gravity 由世界单位重力向量通过该姿态逆旋转到 `ControlFrameV1` 得到。IMU 正向安装不表示其坐标与原始 USD `base_link` 坐标相同，仿真侧仍必须执行 `R_control_from_usd`。

### 4.3 旧 WheelLeg Isaac Lab 工程

路径：`E:\wheel_leg_rl-main\wheel_leg\WheelLeg_RL_IsaacLab`

只允许参考：

- Isaac Lab 资产配置和加载方式。
- 关节、刚体和 actuator 配置经验。
- 关节限位、速度限制、力矩限制和初始物理参数。
- Isaac Lab 场景注册、训练和播放的接入经验。

禁止迁移：

- 原 `wheellegrobot_env.py` 的整体结构。
- VMC、在线五连杆运动学、INS、键盘、日志、课程和奖励混合实现。
- 依赖观测缓存的奖励逻辑。
- 真实机身线速度作为 Actor 观测。
- `Robot_Model.usd` 作为训练资产；它不再是本项目的资产来源。

### 4.4 复旦 Plane 工程

路径：`E:\wheel_leg_rl-main\fudan_rl_wheel_leg-main\fudan_rl_wheel_leg-main\plane`

只允许参考：

- 六维动作的任务语义。
- 25 维本体观测的组成思想。
- 平地速度、yaw、高度和稳定性奖励公式。
- asymmetric Actor/Critic 的设计经验。

禁止迁移：

- Isaac Gym Preview 4 生命周期和环境基类。
- 机器人加载、仿真 step 和 reset 实现。
- 依赖虚拟腿角的 `nominal_state` 奖励原实现。
- 跳跃任务逻辑。

### 4.5 A1 DreamWaQ 工程

路径：`E:\gogo_2026_09\A1_Base-codex-dreamwaq-pace-baseline-port`

允许迁移的数学和训练思想：

- CENet 编码器和训练期解码器。
- 机身线速度估计。
- 16 维 context latent。
- 重参数化与 KL loss。
- 下一帧本体观测重建。
- privileged critic。
- 部署时使用确定性均值。

禁止直接形成运行时依赖。新工程不得 import A1 仓库代码，也不得复制 A1 机器人任务配置。A1 的 RSL-RL 3.0.1 接入层必须按本工程固定的 RSL-RL 3.1.2 API 重新适配和测试。

A1 Base 的可部署主路径固定为五帧历史、`[128,64]` ELU CENet encoder、3D deterministic velocity、16D `context_mu/context_logvar`、Actor 使用 `estimated_velocity + context_mu`。A1 的 45D 单帧观测、225D 历史、64D Actor 输入和 45D 完整下一帧重建只属于 A1 机器人维度，不复制为 WheelLeg 的假维度。WheelLeg 仅保持相同的数学拓扑和 latent 语义，并使用自己的 25D/125D/44D 契约以及动作条件的 16D 物理本体重建。

相对 A1 Base 的有意差异固定如下，实施时不得把这些差异误判为待补功能：

| 接口面 | A1 Base 参考 | WheelLeg 第一版 |
|---|---|---|
| history layout | Isaac Lab observation-term history 后重排 | 专用 wrapper 直接维护 frame-major `[N,5,25]` |
| Actor context | deterministic velocity + deterministic context | 相同，但只使用 `context_mu`，不输入 `context_logvar/context_z` |
| decoder condition | context + velocity | sampled `context_z` + detached velocity + detached clipped action |
| reconstruction target | 完整下一帧 45D Actor observation | 下一帧 clean 16D 物理本体目标，不含命令和 previous action |
| storage | 保存 full next observation | 只新增 16D target 和 1D valid mask |
| curriculum | 包含 AdaBoot 选项 | 第一版固定关闭 AdaBoot 和 privileged-velocity teacher forcing |

## 5. 固定运行版本

新工程采用轻量外部 Isaac Lab 工程。这里的“外部”表示 WheelLeg 任务和算法代码不复制、不继承 Isaac Lab 或旧 WheelLeg 的任务源码；Isaac Lab 本身作为独立、固定 revision 的外部源码依赖安装，不 vendoring 到 `wheelleg_dreamwaq` Python package 中。

固定软件基线：

| 组件 | 版本 |
|---|---:|
| Isaac Sim | 5.1.0（`isaacsim` package family 5.1.0.0） |
| Isaac Lab | 官方 Git tag `v2.3.2`，commit `37ddf626871758333d6ed89cf64ad702aef127d0` |
| Python | 3.11 |
| PyTorch | 2.7.0 + CUDA 12.8 |
| RSL-RL | 3.1.2 |
| Gymnasium | 由 Isaac Lab 兼容版本锁定 |

精确 wheel、补丁版本和传递依赖由项目 lockfile 固定。禁止在不同机器上分别安装“接近版本”。

### 5.1 安装模式

Windows 训练环境固定采用：

1. 使用 uv 管理项目 Python 3.11 虚拟环境和 `uv.lock`。
2. 从 NVIDIA Python package index 安装 Isaac Sim 5.1.0 对应 package family。
3. 在项目包之外准备 Isaac Lab 官方 Git tag `v2.3.2` 源码 checkout，并严格校验 commit 为 `37ddf626871758333d6ed89cf64ad702aef127d0`，以 editable 方式安装其官方扩展；机器相关的 checkout 绝对路径不得写入 schema 或 checkpoint。PyPI `isaaclab==2.3.2.post1` 及 A1 从该 wheel 提取的源码树只作为 Windows 兼容性参考，不是本项目代码基线，也不得与 Git revision 混称为同一版本。
4. 精确安装 `rsl-rl-lib==3.1.2`，启动时断言实际 import 的版本和模块路径。
5. A1 工程的 dependency override、DLL shim 和本地补丁不作为无条件默认依赖。它们必须列入已知兼容性候选清单，并且只有对应 smoke test 稳定复现问题后才能逐项启用；启用项必须进入 `uv.lock`、dependency manifest 和运行日志。

正式训练、播放、导出和测试命令必须从项目自身 `E:\wheel_leg_rl-main\wheelleg_dreamwaq\.venv` 启动。工作区根目录 `E:\wheel_leg_rl-main\.venv` 当前安装的是历史 `rsl-rl-lib==2.3.3`，只属于旧工具环境；若任何正式入口解析到该解释器、该 `rsl_rl` 模块路径或非 3.1.2 版本，必须在创建 Isaac Sim `AppLauncher` 前失败，禁止依赖当前目录或 `PATH` 偶然选择解释器。

本项目已经稳定复现 Windows 下“Kit 启动后首次导入 `tensordict._C` 发生 access violation”的 DLL 加载顺序冲突。工程因此固定 `tensordict==0.14.2`，并要求所有会创建 Isaac Sim `AppLauncher` 的入口，包括 `train_ppo.py`、`play.py`、`train_dreamwaq.py` 和 `play_dreamwaq.py`，都必须在首次导入 `isaaclab.app` 或创建 Kit 前先导入 PyTorch 和 TensorDict。该处理写入直接依赖、`uv.lock` 与 dependency manifest；它只改变进程初始化顺序，不改变 PPO 张量或模型格式。每个入口都必须由独立子进程 smoke test 覆盖，禁止因同一 Python 进程已经预加载 DLL 而得到假通过。

Windows 已知兼容性候选清单：

| 候选处理 | 已知来源/风险 | 启用条件与验证 |
|---|---|---|
| `starlette==0.45.3` override | A1 的 Isaac Lab 源码 `setup.py` pin 与 Isaac Sim 5.1.0.0 的 FastAPI 约束冲突 | `uv lock` 或 `uv sync` 在本项目依赖集合中复现相同 resolver conflict；启用后重跑完整 lock 与 import smoke test |
| `gymnasium==1.2.0`、`flatdict==4.0.0`、`transformers>=4` override | A1 用于匹配 `2.3.2.post1` wheel METADATA，而源码树仍保留不同精确 pin | 仅当本项目解析或 Isaac Lab import/启动 smoke test 证明需要时启用；不得因 A1 存在就整体照搬 |
| `h5py` preload/site shim | Windows 下 Isaac Sim Kit 与 HDF5/zlib DLL 加载顺序可能冲突 | 分别测试 Isaac Sim 启动前后导入 `h5py`；只有稳定复现 DLL load failure 时启用 |
| `torch`/`tensordict==0.14.2` Kit 前预加载 | 本项目已复现 Kit 后加载 `tensordict._C` 的 Windows access violation | **已启用**；2 iteration PPO 与 checkpoint play smoke test 均必须覆盖该导入顺序 |
| `renderer.debug.aftermath=false` | A1 在特定 RTX 5070/驱动组合上记录过 Vulkan/Aftermath 启动崩溃 | 默认通过启动参数或运行 profile 表达；只有重复启动测试复现崩溃时启用，不修改 Isaac Lab checkout |

Phase 0 必须交付一条从空虚拟环境开始的可重复 bootstrap 命令，并记录 Python、Isaac Sim、Isaac Lab commit、PyTorch、CUDA runtime、RSL-RL 和关键传递依赖的解析结果。`uv.lock` 负责 Python 包版本，单独的 dependency manifest 负责 Isaac Lab checkout revision 和 NVIDIA package index；两者缺一不可。

### 5.2 RSL-RL 3.1.2 接口边界

A1 的 RSL-RL 3.0.1 代码只提供算法语义参考，不直接复制其接入层。Phase 2 实现前必须按 3.1.2 实际 API 完成以下核对：

| 接口面 | 本项目要求 |
|---|---|
| observations / observation sets | 环境 wrapper 返回 TensorDict key `policy`、`policy_history`、`critic`；RSL-RL `obs_groups` 只保留标准 observation set `policy:[policy]`、`critic:[critic]`，自定义 policy 通过 `history_obs_group=policy_history` 显式读取历史 |
| ActorCritic | 构造参数、`act`、`evaluate`、distribution state 和 recurrent mask 按 3.1.2 签名实现 |
| Transition | 逐字段定义 observation、action、reward、done、value、log-probability、distribution statistics 和 DreamWaQ 附加张量 |
| RolloutStorage | 核对 `init_storage`、transition 写入、GAE/returns 和 mini-batch generator 返回结构 |
| runner/algorithm registration | RSL-RL 3.1.2 的 `OnPolicyRunner` 通过模块全局符号和 `eval(class_name)` 解析类；`train_dreamwaq.py` 必须在构造 runner 前显式、幂等地把 `DreamWaQActorCritic` 与 `DreamWaQPPO` 注册到 `rsl_rl.runners.on_policy_runner`，不得依赖任务包导入时的隐式副作用 |
| checkpoint | 显式列出 Actor、Critic、CENet、optimizer、normalization/schema manifest 和 iteration key |
| configuration | 未识别字段必须报错；禁止用无约束 `**kwargs` 静默吞掉旧版本参数 |

自定义类名固定为 `DreamWaQActorCritic`、`DreamWaQPPO` 和 `DreamWaQRolloutStorage`。迁移测试必须构造一批具名 transition，验证写入 storage、mini-batch 取回和 loss 读取的字段逐项一致，而不是只验证张量总维度；注册测试还必须证明未注册时 fail fast、显式注册后 class resolution 指向本项目实现。

当前旧环境的 PyTorch 2.5.1 + CUDA 11.8 不支持 RTX 5070 的 `sm_120`，不能作为新项目运行时。

Isaac Lab 3.0 在 2026-10-03 仍不作为本项目基线。未来升级必须单独建立迁移分支，并通过全部 schema、环境和导出一致性测试。

## 6. GPU 可移植架构

RTX 4060 与 RTX 5070 使用同一套：

- 源代码。
- Python 和 CUDA 运行时版本。
- USD 和 actuator 配置。
- Actor、Critic 和 CENet 网络结构。
- 观测、动作、奖励和归一化定义。
- 仿真步长和控制周期。
- checkpoint 格式。

只允许硬件 profile 修改资源相关参数：

- `num_envs`。
- PPO mini-batch 划分。
- headless/viewer 模式。
- 渲染和视频开关。
- 显存保护参数。

其中 `device`、`num_envs` 和 `num_mini_batches` 必须记录在 run manifest，但不进入跨硬件 profile 的模型兼容性拒绝条件。headless/viewer、渲染和视频开关只控制进程表现，不得进入训练任务张量或物理语义；显存保护参数只允许调整内存分配/启动行为，不能改写 batch 语义之外的算法配置。`sim_dt`、`decimation`、articulation solver、资产、collision、actuator、奖励、观测、动作和网络结构在所有硬件 profile 中绝对不可改变。

推荐 profile：

```text
configs/hardware/portable.yaml   # 两张 GPU 都必须可运行的标准验证配置
configs/hardware/rtx4060.yaml    # 4060 吞吐配置
configs/hardware/rtx5070.yaml    # 5070 吞吐配置
```

工程代码不得根据 GPU 名称改变任务行为。profile 必须由启动参数显式选择。

显存容量通过 `256 -> 512 -> 1024 -> 2048` 环境的基准测试确定。正式数值记录在硬件 profile 中，不写入环境类。训练产生的模型可以在另一张 GPU 上加载；checkpoint 必须保存 model、optimizer、RSL-RL adaptive schedule 的 `alg.learning_rate`、零基 runner iteration、明确的 `completed_iterations` 和可恢复的随机数状态。续训总是创建新 run directory，先校验源 run manifest 和 checkpoint 指定的 ContractV3/V4 或 Phase1RandomizedContractV3，再恢复模型、optimizer、`alg.learning_rate`、Python/NumPy/PyTorch/CUDA RNG；恢复后的 `alg.learning_rate` 必须与 optimizer 全部 parameter group 的单一 learning rate 相等，否则 fail fast，避免下一次 adaptive update 用默认标量覆盖已经恢复的 optimizer learning rate。`--max-iterations` 固定表示目标总轮数，恢复点后的实际更新数为 `target_total - completed_iterations`，禁止把 RSL-RL 保存的零基 iteration 直接作为下一轮而重复更新。不同 GPU、不同 `num_envs` 或不同 mini-batch 划分不保证训练轨迹逐位一致，也不能改变 checkpoint 的网络和 schema 兼容性；续训属于“恢复优化状态和 RNG、创建新环境状态”，不声称恢复中断瞬间的仿真 state。

Phase 1R 的 per-env 闭链 reset cache 按 `num_envs` 绑定。同规模续训必须精确复用 source cache；跨规模续训必须先完整验证 source checkpoint/cache，再按目标 hardware profile 生成并绑定 target cache。跨规模时只从 source 恢复 command、reset velocity 和 observation noise 三个运行期 RNG 流，四个 process-start 随机化流保留 target cache 生成后的状态。source/target cache 与七流来源由 `ResumeMetadataV2` 审计；target checkpoint 只绑定 target cache。

每次运行必须记录 master seed，以及 Python、NumPy、PyTorch、CUDA、Isaac Lab 环境和分布式 rank 的派生 seed。seed、硬件 profile、`num_envs`、rollout length、mini-batch 数和 deterministic flags 属于实验复现元数据，不属于可由硬件 profile 静默改写而不留记录的临时参数。

## 7. 总体系统结构

工程只有一个 WheelLeg 任务环境。普通 PPO 和 DreamWaQ PPO 是两个按阶段二选一的训练后端，不在同一次训练中并行运行。DreamWaQ 仍使用 PPO 做策略优化，只是在 learner 内增加 CENet、辅助损失和扩展 rollout storage。

### 7.1 Phase 1：普通 PPO 基线

```text
WheelLegFlatEnv
    |
    +-- ActorObsV1 25D
    |       |
    |       v
    |   Actor network
    |       |
    |       +---> ActionV1 6D ----------------------+
    |                                                |
    |                                                v
    |                                      WheelLegFlatEnv
    |
    +-- CriticObsV1 41D
    |       |
    |       v
    |   Critic network
    |       |
    |       +---> state value V(s), 1D
    |
    +-- reward / terminated / truncated

Actor observation + action + log probability
Critic observation + value V(s)
Environment reward + done flags
                    |
                    v
        PPO sampling buffer (RolloutStorage)
                    |
                    v
        GAE / returns / mini-batches
                    |
                    v
    Standard PPO update: Actor + Critic
```

只有 Actor network 输出控制机器人的六维动作。Critic network 只输出标量价值 `V(s)`，用于计算 advantage 和 value loss，不输出动作。该阶段没有历史缓存、CENet、重建 loss 或 KL loss。

### 7.2 Phase 2：DreamWaQ PPO

Phase 1R 的实现、四次训练和统一 sim2sim 诊断链路已经提供对照后，训练启动项可以切换为 DreamWaQ PPO。任务环境、奖励、动作、命令、随机化 profile 和物理配置保持不变，普通 PPO learner 不在同一次 run 中同时运行。DreamWaQ run 必须从头训练并使用独立 experiment name，不能把普通 PPO checkpoint 当作完整 DreamWaQ checkpoint 恢复。

```text
WheelLegFlatEnv
    |
    +-- Current ActorObsV1 25D
    |       |
    |       +---> DreamWaQHistoryVecEnvWrapper
    |       |               |
    |       |               +---> Five-frame history 5 x 25 = 125D
    |       |                               |
    |       |                               v
    |       |                CENet encoder 125 -> 128 -> 64 -> 35, ELU
    |       |                               |
    |       |                               +---> estimated velocity 3D
    |       |                               +---> context_mu 16D
    |       |                               +---> context_logvar 16D
    |       |
    |       +---> concatenate:
    |             current obs 25D
    |             + estimated velocity 3D
    |             + context_mu 16D
    |                     = 44D
    |                     |
    |                     v
    |         Actor 44 -> 256 -> 128 -> 64 -> 6, ELU
    |                     |
    |                     +---> ActionV1 6D -------------> WheelLegFlatEnv
    |                               |                              |
    |                               |                              +---> returned next policy 25D (noisy)
    |                               |                              +---> returned next critic 41D (clean)
    |                               |                                         |
    |                               |                                         +---> select critic [0:6] and [9:19]
    |                               |                                                = next physical target 16D
    |                               |
    |                               +---> stop-gradient action 6D ------------------+
    |                                                                                |
    |   context_mu/logvar --reparameterize--> context_z 16D                          |
    |   stop-gradient estimated velocity 3D -----------------------------------------+
    |                                                                                |
    |                     decoder input 25D -> 64 -> 128 -> 16, ELU ------------------+
    |                                                                                +---> predicted next physical target 16D
    |
    +-- CriticObsV1 41D
    |       |
    |       v
    |   Privileged Critic 41 -> 256 -> 128 -> 64 -> 1, ELU
    |       |
    |       +---> state value V(s), 1D
    |
    +-- reward / terminated / truncated

Observations + frame-major history + action + log probability + value
Reward + done + next physical target + reconstruction mask
                            |
                            v
                DreamWaQRolloutStorage
                            |
                            v
PPO surrogate + value + entropy + velocity + reconstruction + KL
                            |
                            v
            Joint update: Actor + Critic + CENet
```

控制动作仍然只由 Actor network 输出。Actor 在 rollout、PPO update、评估和部署四条路径中都使用确定性的 `estimated_velocity + context_mu`，不直接接收 `context_logvar`，也不接收随机 `context_z`。`context_logvar` 只参与训练期重参数化和 KL loss；decoder 接收采样的 `context_z`、停止梯度的预测速度和停止梯度的当前动作，预测动作执行后的下一帧物理本体观测。Decoder 只提供辅助 loss，不预测动作、不替代环境，也不进入部署。Critic 只输出 `V(s)`。DreamWaQ 不是 PPO 旁边的另一个控制器，而是包含 Actor、Critic、CENet 和扩展 storage 的 PPO learner。

### 7.3 Phase 4：实机部署

部署阶段不运行训练器，只执行从 DreamWaQ learner 导出的确定性推理图。计算平台可以是 H7、RK3566 或后续确定的设备，但输入、归一化、历史和 ActionV1 契约不得随平台变化。

```text
IMU + joint feedback + command + previous action
                         |
                         v
                  ActorObsV1 25D
                         |
                         +---> Five-frame history 125D
                         |               |
                         |               v
                         |    CENet encoder, deterministic inference
                         |               |
                         |               +---> estimated velocity 3D
                         |               +---> context_mu 16D
                         |
                         +---> concatenate:
                               current obs 25D
                               + estimated velocity 3D
                               + context_mu 16D
                                      |
                                      v
                                 Actor network
                                      |
                                      +---> ActionV1 6D
                                                   |
                                                   v
                                       limits and safety controller
                                                   |
                                                   v
                             four leg position targets + two wheel velocity targets
```

部署运行时不包含 Critic、PPO optimizer、rollout storage、reward、CENet decoder 或训练 loss。

### 7.4 网络与训练组件输出职责

| 组件 | 主要输出 | 是否直接控制机器人 |
|---|---|---|
| 普通 PPO Actor | ActionV1，6D | 是 |
| DreamWaQ Actor | ActionV1，6D | 是 |
| Privileged Critic | 状态价值 `V(s)`，1D | 否 |
| CENet encoder | 预测线速度 3D、`context_mu` 16D、`context_logvar` 16D | 否；Actor 只接收速度和 `context_mu` |
| CENet decoder | 预测下一帧 `ProprioReconstructionTargetV1`，16D | 否，仅用于训练 loss |
| RolloutStorage | PPO 更新所需的一批 transition | 否，只是训练缓存 |
| PPO update | 更新 Actor、Critic 和可选 CENet 的参数 | 否，不直接产生控制动作 |

三个入口的关系：

```text
train_ppo.py       -> Phase 1，普通 PPO learner
train_dreamwaq.py  -> Phase 2，DreamWaQ PPO learner
exported policy    -> Phase 4，platform-specific deterministic inference backend
```

核心边界：

- 环境只产生状态、观测、奖励和终止信号，不知道 CENet 的存在。
- DreamWaQ 只接收张量和 schema 元数据，不能访问 Isaac Lab `Articulation` 对象。
- 普通 PPO 和 DreamWaQ PPO 共用同一环境和奖励定义，但每次训练只选择一个 learner。
- Actor 和 Critic 始终属于当前选择的 learner，不是环境外的额外并行控制路径。
- Critic、CENet decoder 和训练损失不进入实机部署包。

## 8. 推荐工程目录

工程工作名和 Python package 名统一为 `wheelleg_dreamwaq`。

```text
wheelleg_dreamwaq/
├── .python-version
├── pyproject.toml
├── uv.lock
├── dependency-manifest.toml
├── README.md
├── configs/
│   └── hardware/
│       ├── portable.yaml
│       ├── rtx4060.yaml
│       └── rtx5070.yaml
├── source/
│   └── wheelleg_dreamwaq/
│       └── wheelleg_dreamwaq/
│           ├── __init__.py
│           ├── assets/
│           │   ├── wheelleg.py
│           │   ├── paths.py
│           │   ├── asset_contract.py
│           │   └── asset_overrides.py
│           ├── schemas/
│           │   ├── action.py
│           │   ├── command.py
│           │   ├── observation.py
│           │   ├── frames.py
│           │   ├── normalization.py
│           │   └── manifest.py
│           ├── tasks/
│           │   └── direct/
│           │       └── wheelleg_flat/
│           │           ├── __init__.py
│           │           ├── env.py
│           │           ├── env_cfg.py
│           │           ├── state.py
│           │           ├── control.py
│           │           ├── commands.py
│           │           ├── observations.py
│           │           ├── rewards.py
│           │           ├── terminations.py
│           │           └── agents/
│           │               ├── ppo_cfg.py
│           │               └── dreamwaq_cfg.py
│           ├── algorithms/
│           │   └── dreamwaq/
│           │       ├── __init__.py
│           │       ├── cenet.py
│           │       ├── actor_critic.py
│           │       ├── ppo.py
│           │       ├── storage.py
│           │       └── history_wrapper.py
│           ├── training/
│           │   └── dreamwaq_checkpoint.py
│           └── deployment/
│               ├── dreamwaq_inference.py
│               ├── manifest.py
│               └── observation_adapter.py
├── scripts/
│   ├── bootstrap.ps1
│   ├── audit_asset.py
│   ├── smoke_random_actions.py
│   ├── train_ppo.py
│   ├── train_dreamwaq.py
│   ├── play.py
│   ├── play_dreamwaq.py
│   ├── export_ppo_actor.py
│   └── export_dreamwaq_actor.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conformance/
└── docs/
```

目录规则：

- `env.py` 只负责编排 Isaac Lab 生命周期。
- 观测、奖励、控制和 termination 不写成 `env.py` 内的大段条件分支。
- `schemas` 是训练、播放和实机部署共同依赖的契约层。
- `dependency-manifest.toml` 固定 Isaac Lab checkout revision、Isaac Sim package family 和 index；`bootstrap.ps1` 只负责可重复安装与版本 smoke test，不包含任务逻辑。
- `asset_contract.py` 定义版本化 AssetBundle 的五文件身份、入口层、prim/joint 期望值和审计断言；V3/V1 的 `asset_overrides.py` 负责加载时禁用内嵌地面，V4/V2 不再包含该步骤。两者都不得修改不可变源 bundle。
- `frames.py` 是 ControlFrameV1、关节顺序和符号变换的单一事实来源；`normalization.py` 是字段级固定 offset/scale/clip 的单一事实来源。
- `observation.py` 同时是 ActorObsV1、CriticObsV1 和 ProprioReconstructionTargetV1 slice/版本的单一事实来源。
- `algorithms/dreamwaq` 不依赖 `tasks` 内部对象。
- `deployment` 不依赖 Isaac Sim 或 Isaac Lab。

Phase 2 的实现写集合固定如下，禁止为了 DreamWaQ 复制第二套环境、奖励或动作代码：

| 文件 | 单一职责 |
|---|---|
| `algorithms/dreamwaq/cenet.py` | 纯 PyTorch encoder、reparameterization、decoder 和三项辅助 loss；只接收张量 |
| `algorithms/dreamwaq/actor_critic.py` | RSL-RL 3.1.2 policy 接口、确定性 44D Actor 输入拼接、41D Critic |
| `algorithms/dreamwaq/storage.py` | 在标准 rollout storage 上增加 16D target 与 1D mask |
| `algorithms/dreamwaq/ppo.py` | PPO update、CENet loss、梯度边界和日志，不访问 Isaac Lab robot |
| `algorithms/dreamwaq/history_wrapper.py` | `DreamWaQHistoryVecEnvWrapper`：在 RSL-RL VecEnv/TensorDict 边界维护 `[N,5,25]` history、reset 复制填充和 `policy_history` key |
| `training/dreamwaq_checkpoint.py` | `DreamWaQContractOnPolicyRunner`、Phase 2 checkpoint metadata/save/load/validate/resume；不修改 Phase 1 checkpoint 语义 |
| `tasks/direct/wheelleg_flat/agents/dreamwaq_cfg.py` | 本文冻结的网络、PPO、loss 与 class name 配置 |
| `deployment/dreamwaq_inference.py` | 无状态 `history[125] -> action_mean[6]` 推理模块 |
| `scripts/train_dreamwaq.py` | 启动环境、显式注册 RSL-RL 自定义类、构造 runner、checkpoint/resume |
| `scripts/play_dreamwaq.py` | Isaac Sim 确定性播放与键盘命令，不重复实现网络 |
| `scripts/export_dreamwaq_actor.py` | 从完整 checkpoint 导出推理图、manifest 和 golden vector |
| `sim2sim/mujoco/wheelleg_mujoco/runner.py` | 维护相同五帧 history，并调用导出推理图；不依赖训练包内部类 |

现有 `train_ppo.py`、`play.py` 和 `export_ppo_actor.py` 继续服务普通 PPO/Phase 1R，不加入 DreamWaQ 条件分支。共享 checkpoint、manifest、schema 和 observation helper 可以复用，但必须保持单一实现。

## 9. 仿真和控制时序

固定时序：

```text
sim_dt      = 0.005 s
decimation  = 4
control_dt  = 0.02 s
policy_rate = 50 Hz
```

所有以下计算统一使用 `control_dt=0.02s`：

- episode 时长换算。
- 命令采样模式、概率、范围和是否在 episode 内保持。
- 关节加速度差分。
- action rate 和 action smooth。
- 按时间积分或归一化的奖励。

不参考复旦的 `0.01s` 控制周期，也不得在新任务中同时存在两套控制周期常量。

每个策略 step 的逻辑顺序：

1. 接收并裁剪归一化动作。
2. 计算腿关节位置目标和轮速目标。
3. 在四个物理 step 中持续应用目标。
4. 物理 step 完成后读取 post-physics 具名状态快照。
5. 从该快照独立计算 done 和 reward，并为未结束环境构造下一观测。
6. reset 已结束环境，并只刷新这些环境的状态快照和返回观测。
7. 返回当前观测；DreamWaQ 对发生 reset 的 transition 屏蔽下一帧重建 loss。

done 与 reward 必须使用同一份 reset 前 post-physics 状态。观测必须来自明确的当前状态，禁止通过 reward、termination 或旧观测缓存间接取值。

## 10. 状态快照边界

`state.py` 构造每个控制 step 唯一的 `WheelLegState` 语义快照。快照保留具名 raw state 与转换后的 control state，不把 Isaac Lab 的便捷别名直接泄漏到 reward/observation。快照至少包含：

- world frame 中显式的 root link position/quaternion，以及 root COM position；机身高度固定使用 root COM 相对环境地面的高度。当前 USD 的 root-link 原点存在大幅局部偏置，禁止用 root-link 高度替代 COM 高度。
- root COM linear velocity 固定读取 Isaac Lab 2.3.2 的 `robot.data.root_com_lin_vel_b`，root angular velocity 固定读取 `robot.data.root_link_ang_vel_b`，然后通过 `R_control_from_usd` 转换为 `ControlFrameV1`。禁止把 world-frame 速度直接乘该常数矩阵，也不需要自行用 `v_link + omega x r_com` 重建已有的 COM 速度字段；集成测试必须与 `root_com_lin_vel_w` 经姿态逆旋转后的结果交叉核对。
- `ControlFrameV1` 中的 projected gravity。
- 按 ActionV1 canonical joint order 排列的六个受控关节位置、速度、加速度和 applied torque。
- 当前、上一时刻和上两时刻归一化动作。
- 当前命令。
- episode time 和可选接触状态。

奖励函数使用字段名读取快照，不允许使用 `obs[:, index]`。观测函数只负责从快照按照 schema 排列和固定归一化张量。USD 到 ControlFrameV1 的变换、joint name lookup 和 canonical reorder 在构造快照时完成一次，奖励、Actor、Critic 和 CENet 不得各自重复实现。

## 11. 动作契约 ActionV1

Actor 输出范围为 `[-1, 1]`，维度固定为 6：

| Slice | 名称 | 物理含义 |
|---|---|---|
| `0` | `left_front_leg` | `jIJ` 位置目标偏移 |
| `1` | `left_hind_leg` | `jIO` 位置目标偏移 |
| `2` | `right_front_leg` | `jAB` 位置目标偏移 |
| `3` | `right_hind_leg` | `jAG` 位置目标偏移 |
| `4` | `left_wheel` | `jwheel_left` 前向为正的轮速目标 |
| `5` | `right_wheel` | `jwheel_right` 前向为正的轮速目标 |

ActionV1 的 canonical joint order 固定为：

```text
[jIJ, jIO, jAB, jAG, jwheel_left, jwheel_right]
```

禁止使用 PhysX 内部 joint index 表示该顺序。当前 USD 中这些 joint 的内部索引并不连续，运行时必须按名称解析并断言解析结果唯一。

控制映射：

```text
q_target = q_nominal + leg_action_scale * action[0:4]
wheel_velocity_target_control = wheel_action_scale * action[4:6]
wheel_velocity_target_usd = [1, -1] * wheel_velocity_target_control
wheel_feedback_control = [1, -1] * wheel_feedback_usd
```

两个轮子在 USD 零姿态下的 joint axis 相反，因此 ActionV1 对两轮都定义“机器人前进”为正，而仿真 actuator 目标和反馈映射固定使用 `wheel_joint_sign_usd=[+1,-1]`。该符号同时应用于轮位置、轮速度、轮加速度和 applied torque，使 Actor/Critic 内始终保持 canonical 前进同号语义；禁止只转换目标而把右轮 USD 原始负号直接送入观测。这组符号只适用于当前 USD；硬件 motor sign 必须单独标定并写入部署 manifest。

现有实机协议的腿顺序为 `[jIO, jAG, jIJ, jAB]`，与 ActionV1 不同。部署 adapter 必须显式使用：

```text
policy leg order -> packet leg order: [1, 3, 0, 2]
packet feedback  -> policy leg order: [2, 0, 3, 1]
```

现有实机协议的轮顺序固定为：

```text
packet wheel order = [left wheel (CAN ID 4), right wheel (CAN ID 5)]
policy wheel order = [left wheel, right wheel]
wheel reorder      = [0, 1]
```

轮顺序虽然当前为恒等映射，仍必须作为具名 manifest 字段保存；左右轮 hardware sign 与协议顺序是两个独立契约，不能合并成一个隐式数组。

已知注释冲突：`Sim2Real.h` 把 `jAG`/`jAB` 分别注释为右前/右后，而 MJCF actuator 名称和 `main_mujoco.c` 把它们定义为右后/右前。ActionV1、协议重排、checkpoint 和 manifest 一律以 joint name 与 CAN ID 为权威，不以 `front/hind` 文本标签驱动索引。当前 canonical 顺序不因该注释冲突改变；G-11 首次实机标定必须以低力矩单电机点动确认并记录 `{joint_name, CAN_ID, physical_location, hardware_sign, zero_offset}`，随后才能冻结物理角色文字。

现有 `Sim2Real.c` 中全 `+1` motor sign 和全零 zero offset 只作为接口占位，不作为本架构的实机校准结论。

约束：

- 腿使用位置目标。
- 轮使用速度目标。
- 不使用 VMC、IK 或 FK 生成主控制目标。
- 不使用硬动作变化率限制。
- 保留动作裁剪、关节目标限位和 actuator 限制。
- action rate 和 action smooth 只能作为软奖励项。
- `q_nominal` 和动作 scale 必须写入配置及导出 manifest，训练和实机共用。
- `previous_action` 定义为生成当前观测之前最后一次实际执行的、经 `[-1,1]` 裁剪且尚未乘物理 scale/符号或协议重排的 canonical ActionV1；实现内部命名为 `last_applied_action`，reset 后固定为零向量。
- 腿位置误差统一为 canonical joint position `q - q_nominal`。`q_nominal`、腿 joint sign、hardware zero offset 和安全限位必须在 Phase 0 独立标定，禁止由当前固件占位常量推断。

## 12. 命令契约 CommandV1

命令维度固定为 3：

| Slice | 名称 | 坐标与单位 |
|---|---|---|
| `0` | `target_vx` | `ControlFrameV1 +X` 前向速度，m/s |
| `1` | `target_yaw_rate` | 绕 `ControlFrameV1 +Z` 的 yaw 角速度，rad/s |
| `2` | `target_base_height` | root COM 相对平地的目标高度，m |

V1 不包含 `vy`、绝对 yaw 或虚拟腿长 `L0` 命令。

Actor 不观测真实机身高度。平地条件下，策略通过腿关节位置和投影重力学习高度命令对应的姿态，因此该目标是基于关节几何的间接闭环。复杂地形阶段必须重新评估高度语义。

### 12.1 CommandSamplingV2

`Phase1ContractV4` 在每个环境 reset 时只采样一次命令，并在该环境完整 `10 s` episode 内保持不变。这里的“连续均匀采样”只表示不同 episode 之间可从区间内取得任意实数，不表示一个 episode 内连续改变目标。

先按分类分布选择运动模式，再只对该模式启用的速度分量进行均匀采样：

| 模式 | 概率 | `target_vx` | `target_yaw_rate` |
|---|---:|---:|---:|
| 站立 | `0.20` | `0` | `0` |
| 纯直行 | `0.30` | `U(-1.5, 1.5) m/s` | `0` |
| 原地转向 | `0.20` | `0` | `U(-1.0, 1.0) rad/s` |
| 组合运动 | `0.30` | `U(-1.5, 1.5) m/s` | `U(-1.0, 1.0) rad/s` |

所有模式的 `target_base_height` 都独立采样为 `U(0.16, 0.24) m`，同样保持到 episode 结束。命令只能在 reset 后更新；环境 step 中不得存在按秒数重采样的路径。四种模式的概率、三个范围和 `hold_for_episode=true` 都进入 contract hash 与 run manifest。

### 12.2 未决问题：高度跟踪不等于自主起立

截至 `2026-10-04`，当前 Phase 1 reset 从接近名义站立构型开始：初始 root-COM 高度约为 `0.210477 m`，腿关节位于 `q_nominal` 附近。因而现有 `tracking_base_height` 主要训练策略在已经接近站立的状态下保持或调节高度，不能证明策略具备从低矮、收腿、倒伏或未承载初态主动起立到目标高度的能力。单独扩大 `target_base_height` 范围或提高高度奖励权重也不能替代起立状态分布和安全过程的定义。

当前平地 PPO 基线可以继续用于验证平衡、速度、yaw 和站立附近的高度跟踪，但该 checkpoint 不得标记为“具备自主起立能力”，也不得据此直接进行实车自主起立。后续必须单独确认低姿态 reset 分布、可实现的起立初态、目标高度给定方式或命令斜坡、是否采用 curriculum、失败终止、动作/力矩限制以及实车启动安全状态机。该问题继续由 G-13 跟踪；关闭 G-13 前不改变当前 reset 和高度奖励实现，也不阻塞本轮纯仿真 PPO/sim2sim 对比。

## 13. Actor 观测契约 ActorObsV1

单帧 Actor 观测固定为 25 维：

| Slice | 维度 | 内容 | 实机来源 |
|---|---:|---|---|
| `0:3` | 3 | `ControlFrameV1` 机身角速度 `wx, wy, wz` | BMI088 陀螺仪，经 `R_control_from_imu` 转换 |
| `3:6` | 3 | `ControlFrameV1` projected gravity | IMU 四元数计算 |
| `6:9` | 3 | 归一化 `target_vx, target_yaw_rate, target_base_height` | 上层命令 |
| `9:13` | 4 | `[jIJ,jIO,jAB,jAG]` 相对 `q_nominal` 的位置误差 | 电机位置反馈经 sign/zero/reorder |
| `13:19` | 6 | canonical order 的四腿关节速度和两轮速 | 电机速度反馈经 sign/reorder |
| `19:25` | 6 | 上一时刻 clipped canonical ActionV1 | 控制器内部状态 |

Actor 明确不读取：

- 真实机身线速度。
- 真实机身高度。
- roll、pitch、yaw 欧拉角。
- 绝对 yaw。
- terrain height scan。
- 质量、质心、摩擦等仿真参数。

投影重力由世界单位重力向量旋转到 `ControlFrameV1` 得到。若四元数表示 control/body 到 world 的旋转，则语义等价于：

```text
projected_gravity_control = R_world_from_control^T * [0, 0, -1]
```

它提供 roll/pitch 倾斜信息，但不包含绝对 yaw。V1 使用 yaw rate 命令和陀螺仪 `wz`，不需要绝对航向。

### 13.1 NormalizationV2

`Phase1ContractV4` 使用固定、字段级、可导出的 `NormalizationV2`，不使用运行时经验均值/方差。旧 `Phase1ContractV3` checkpoint 继续绑定 `NormalizationV1`，不得由 V4 代码加载。RSL-RL 3.1.2 原生配置只负责关闭 Actor 和 Critic 的 empirical observation normalizer：

```text
actor_obs_normalization    = false
critic_obs_normalization   = false
```

`velocity_target_normalization` 不是 RSL-RL 3.1.2 原生配置，本项目不得创建该伪开关或另一个速度 running normalizer。CENet 速度监督和输出只使用本文规定的固定 `2.0` scale；启动断言必须确认 Actor、Critic、CENet wrapper 和导出图中不存在第二套经验 normalization。

唯一允许的数据路径为：

```text
raw simulation/sensor state
-> transform to ControlFrameV1
-> canonical joint order/sign/zero conversion
-> q - q_nominal
-> fixed per-field offset and scale
-> fixed clip
-> ActorObsV1
```

任何 Python、RSL-RL wrapper、导出图或实机 adapter 都不得再叠加第二套 normalizer。初始字段定义为：

| ActorObsV1 字段 | offset / 预处理 | 初始固定 scale |
|---|---|---:|
| angular velocity | ControlFrameV1，rad/s | `0.25` |
| projected gravity | 单位向量 | `1.0` |
| `target_vx` | `clip(raw_vx * 0.6666666667, -1, 1)` | 公式已含唯一 scale |
| `target_yaw_rate` | `clip(raw_yaw_rate * 1.0, -1, 1)` | 公式已含唯一 scale |
| `target_base_height` | `clip((raw_h_cmd - 0.20) / 0.04, -1, 1)` | 公式已含 offset/scale |
| leg position error | `q - q_nominal`，rad | `1.0` |
| joint velocity | canonical joint velocity，rad/s | `0.05` |
| previous action | clipped ActionV1 | `1.0` |

`NormalizationV2` 固定 `vx_max_abs=1.5`、`yaw_rate_max_abs=1.0`、`nominal_base_height=0.20`、`height_command_span=0.04`，命令归一化后裁剪到 `[-1,1]`。其余字段 offset、scale 和 clip 与 V1 相同。五帧历史保存的是 Actor 实际消费、已经完成固定 normalization 且按 Phase 1R profile 加噪的 ActorObsV1。CENet 的重建目标不是 noisy policy tensor，也不是完整 ActorObsV1，而是从下一帧 CriticObsV1 `0:25` 的 clean actor features 提取 `ProprioReconstructionTargetV1`，定义见第 16.2 节。

## 14. Critic 观测契约 CriticObsV1

普通 PPO 和第一版 DreamWaQ 共用 41 维 privileged Critic：

| Slice | 维度 | 内容 |
|---|---:|---|
| `0:25` | 25 | 当前 clean ActorObsV1-compatible features；slice、顺序和 normalization 与 ActorObsV1 相同，但位于 Phase 1R observation noise 注入之前 |
| `25:28` | 3 | `ControlFrameV1` 仿真真实 root COM linear velocity |
| `28:29` | 1 | root COM 相对平地的真实高度 |
| `29:35` | 6 | canonical joint order 的六关节加速度 |
| `35:41` | 6 | canonical joint order 的六关节 applied torque |

CriticObsV1 只在训练期间存在，不进入部署运行时。其 `0:25` 是已经按 ActorObsV1 字段契约归一化、但尚未施加 Actor observation noise 的 clean feature 前缀，不从 noisy `policy` TensorDict 复制，也不再次归一化；普通 PPO 无 observation noise 时二者数值相同。真实线速度固定使用 `2.0` scale。真实高度使用与高度命令一致的中心和 span，关节加速度与力矩的固定 scale/clip 在 actuator 审计后冻结。第一阶段不为凑维度加入接触状态；只有接触信息被 reward、termination 或随机化证明有助于 value estimation 时才升级 Critic schema。

后续扩展规则：

- 启用质量随机化后，才加入质量偏差。
- 启用质心随机化后，才加入质心偏置。
- 启用摩擦或恢复系数随机化后，才加入对应参数。
- 训练复杂地形后，才加入 terrain height scan。
- 只有字段与复旦定义完全一致时，维度才可能自然达到 141。
- 每次字段变化必须升级 Critic schema 版本，旧 checkpoint 不允许静默加载。

## 15. 普通 PPO 基线

第一阶段使用 RSL-RL 标准 PPO：

```text
policy observation = ActorObsV1, 25D
critic observation = CriticObsV1, 41D
action             = ActionV1, 6D
```

环境输出 TensorDict 观测组：

```text
policy: [N, 25]
critic: [N, 41]
```

PPO 基线已经允许 asymmetric critic，但不包含：

- 历史缓存。
- CENet。
- 重建 loss。
- KL loss。
- AdaBoot。

这样可以先独立验证资产、控制、观测、奖励和 reset 是否正确。

## 16. DreamWaQ 数据流

### 16.1 历史缓存

DreamWaQ 专用 `DreamWaQHistoryVecEnvWrapper` 维护五帧历史，环境本身仍只产生单帧 ActorObsV1：

```text
history_t = [o_(t-4), o_(t-3), o_(t-2), o_(t-1), o_t]
shape     = [N, 5, 25]
flat      = [o_(t-4)[0:25], o_(t-3)[0:25], o_(t-2)[0:25], o_(t-1)[0:25], o_t[0:25]]
flat shape = [N, 125]
```

wrapper 内部的 canonical 存储必须是连续的 `[N,5,25]` 张量，并以 `history.contiguous().reshape(N,125)` 做 frame-major 展平。禁止使用 Isaac Lab ObsTerm/observation group 的内置 history 功能代替该 wrapper，避免 term-major 与 frame-major 布局混淆。reset 时用 reset 后当前观测复制填满五帧，禁止使用全零历史。

包装顺序固定为：

```text
WheelLegFlatEnv
    -> DreamWaQHistoryVecEnvWrapper(
           subclass/behavior-compatible extension of RslRlVecEnvWrapper,
           clip_actions=1.0)
    -> OnPolicyRunner
```

不能先套普通 Gym history wrapper 再套原生 `RslRlVecEnvWrapper`，因为 2.3.2 的 `get_observations()` 会通过 `unwrapped._get_observations()` 绕过前置 wrapper。专用 wrapper 必须保持 RSL-RL `VecEnv` API，复用原生 action clipping、`terminated|truncated` 合并和 `extras["time_outs"]` 语义，并在 `reset()`、`get_observations()`、`step()` 的 TensorDict 返回值中幂等地追加 `policy_history`。PPO/Phase 1R 继续使用原生 `RslRlVecEnvWrapper`，环境类不得为了 DreamWaQ 增加 history 状态。

专用 wrapper 必须缓存最近一次 reset/step 产生的增强 TensorDict。构造阶段先让 `RslRlVecEnvWrapper.__init__` 完成其内部 `env.reset()`；该次返回 observation 按上游语义被丢弃。随后 wrapper 恰好调用一次 `RslRlVecEnvWrapper.get_observations()`，以这次 observation 复制填满五帧并建立首次缓存。显式 `reset()`，包括 resume 后的全环境 reset，固定执行 `super().reset()` 取得并记录 discarded observation，再恰好调用一次 `super().get_observations()`；第二次 observation 是 `first_policy_observation`，用于复制填满五帧并作为 wrapper 的 reset 返回值。后续 runner 的 `get_observations()` 只返回该缓存，不重新调用 `_get_observations()`、不再次采样 observation noise、也不滚动 history。

缓存及其张量对调用者是只读快照；每次 reset/step 必须构造新的 history/TensorDict 并替换缓存，禁止原地修改此前已经返回、可能仍被 pending transition 引用的 `policy`、`policy_history` 或 `critic` 张量。每次 reset/step 后都必须断言 `policy_history[:, -25:]` 与同一返回 TensorDict 的 `policy` 逐元素相等；resume trace 还必须断言历史五帧全部等于记录的 `first_policy_observation`。该断言失败说明当前帧、历史或 observation-noise RNG 已错一拍。

DreamWaQ wrapper 向算法提供：

```text
policy:         [N, 25]
policy_history: [N, 125]
critic:         [N, 41]
```

### 16.2 CENet

第一版 CENet 固定为无循环、无 dropout、无经验 normalizer 的前馈网络。隐藏层使用 ELU，最后一层不加激活，`nn.Linear` 使用 PyTorch 默认初始化。接口必须显式区分确定性策略路径和随机辅助训练路径，禁止依据 `module.training`、`requires_grad` 或调用者身份隐式切换行为：

```text
encode(history_125)
    -> estimated_velocity_3
    -> context_mu_16
    -> context_logvar_16

infer(history_125)
    -> estimated_velocity_3, context_mu_16

sample_context(context_mu_16, context_logvar_16, epsilon=None)
    -> context_z_16

decode(context_z_16, detached_velocity_3, detached_action_6)
    -> predicted_next_proprio_16
```

上述接口统一接收 rank-2 `float32` 张量，batch 维为第一维：history `[B,125]`、current observation `[B,25]`、velocity `[B,3]`、context `[B,16]`、action `[B,6]`、decoder output `[B,16]`。训练和导出不得依赖固定 batch size。encoder 输出、raw/effective action mean、raw/effective action standard deviation、value、log probability、advantage、`exp(0.5*logvar)`、`exp(logvar)`、`L_PPO/L_velocity/L_next_proprio/L_KL` 与总 loss 必须逐 mini-batch 检查有限值；effective action standard deviation 还必须严格大于零。第一版不对 `context_logvar` 做静默 clamp，因为这会改变已冻结的 A1 对齐目标；一旦指数项溢出或出现非有限值，立即终止该 run 并保留诊断工件。

网络结构和参数量冻结如下：

| 模块 | 拓扑 | 激活 | 参数量 |
|---|---|---|---:|
| CENet encoder | `125 -> 128 -> 64 -> 35` | 隐藏层 ELU，输出线性 | `26,659` |
| Actor mean | `44 -> 256 -> 128 -> 64 -> 6` | 隐藏层 ELU，输出线性 | `53,062` |
| Critic | `41 -> 256 -> 128 -> 64 -> 1` | 隐藏层 ELU，输出线性 | `51,969` |
| CENet decoder | `25 -> 64 -> 128 -> 16` | 隐藏层 ELU，输出线性 | `12,048` |

encoder 的 35D 输出按固定 slice 拆分为 `estimated_velocity[0:3]`、`context_mu[3:19]` 和 `context_logvar[19:35]`。部署路径包含 encoder 与 Actor mean，共 `79,721` 个 Linear 参数；训练期还包含 Critic、decoder 和 RSL-RL 的 6D action standard-deviation 参数。任何层宽、激活、slice 或输出维度变化都必须升级 DreamWaQ algorithm contract，不能由 hardware profile 覆盖。

Phase 1R 继续保持 Actor-noisy/Critic-clean。CENet encoder 输入和 Actor current observation 都来自 Actor 实际消费的 noisy `policy/history`；监督标签只在训练期从 clean `critic` 提取。`ProprioReconstructionTargetV1` 固定为下一帧 CriticObsV1 `0:25` 中 clean actor features 的物理本体切片：

| Target slice | 维度 | 来源 |
|---|---:|---|
| `0:6` | 6 | `next_critic_obs[0:6]`：clean 角速度与 projected gravity |
| `6:16` | 10 | `next_critic_obs[9:19]`：clean 四腿位置误差与六关节速度 |

命令 `next_critic_obs[6:9]` 是外部输入而不是机器人动力学状态；V4 虽在 episode 内保持，但 reset 时仍可改变且 done transition 已被 mask，因此不进入重建目标。clean previous-action `next_critic_obs[19:25]` 等于当前 clipped action `a_t`，已作为 decoder 条件且重建它没有表征学习价值，因此两段都排除。Decoder 的任务是从 noisy history 估计隐变量，并预测“执行 `a_t` 后机器人下一帧的 clean 物理本体观测”，不是预测 observation noise、动作、命令、reward 或环境 reset。普通 PPO 无 observation noise 时，policy 与 clean actor features 数值相同，但 target source 仍按 critic slice 实现，禁止根据 profile 切换代码路径。

Decoder 的存在是为了给没有人工标签的 16D context 提供逐 transition 的密集动力学监督：只有当 context 包含当前运动状态和隐藏动力学信息时，才有助于预测动作执行后的下一状态。它不是 PPO 或环境运行的必要组件，因此 Phase 1 不存在；进入 DreamWaQ 后，它只通过辅助 loss 训练 CENet，部署时删除。`stop_gradient` 只切断反向传播，decoder 前向计算仍能读取 estimated velocity 和当前动作的数值。

第一版 velocity head 是 deterministic regression head，不采样、不参与 KL；`L_KL` 只将 16D context posterior 约束到标准正态先验。Actor 在 rollout、PPO update、play、评估和部署中始终使用 `estimated_velocity + context_mu`。Actor 不读取 `context_logvar` 或 `context_z`，也不允许用 Critic 中的真实线速度替换 estimated velocity。这样 PPO 的 action mean 和 log probability 不受 CENet 重参数化噪声影响，与 A1 Base 的 deterministic actor 主路径一致。

随机 `context_z` 只供训练期 decoder 使用：每次辅助 loss 前向计算显式采样 `epsilon ~ N(0,I)`，并计算 `context_z = context_mu + exp(0.5 * context_logvar) * epsilon`。storage 不保存 `epsilon`；同一 transition 在不同 PPO epoch 中允许为 reconstruction loss 重新采样。给定显式 epsilon 时，`sample_context` 必须可重复，以便单元测试和诊断。第一版不包含 AdaBoot、sampled-velocity head 或 privileged-velocity teacher forcing。

Actor 输入：

```text
normalized current ActorObsV1 25
+ normalized estimated velocity 3
+ context_mu 16
= 44D
```

Actor 输出 ActionV1 的 6D Gaussian action mean。为保留 A1 Base 的分布数值范围且不隐藏异常，分布路径固定为：先对 raw mean 和 raw scalar std 做 finite 检查；raw mean 有限后裁剪到 `[-20,20]` 作为 Gaussian/evaluation/export 共用的 effective mean；`raw_std` 是初始化为 `1.0` 的 6D state-independent 可训练参数，effective std 定义为 `clamp(abs(raw_std),1e-4,2.0)` 并扩展到 batch。禁止使用 `nan_to_num` 把非有限值替换成常数。训练 rollout 从 `Normal(effective_mean,effective_std)` 采样，环境执行前再按 `[-1,1]` 裁剪；`act_inference` 和部署输出同一 effective mean，随后由 runtime 执行环境动作裁剪。

部署时：

- CENet 使用 deterministic velocity output 和 `context_mu`，不采样。
- velocity 输出保持训练时的固定 `2.0` scale，直接进入 Actor，不在部署侧额外归一化。
- 不导出 decoder。
- 不导出 Critic。
- 不导出训练 loss。

### 16.3 训练损失

```text
L_PPO   = L_surrogate + 1.0 * L_value - 0.01 * entropy
L_total = L_PPO
        + 1.0 * L_velocity
        + 1.0 * L_next_proprio
        + 1.0 * L_KL
```

其中：

- `L_velocity = mean((estimated_velocity_t - target_velocity_t)^2)`；监督目标是与 `history_t` 同一当前帧的 `critic_t[25:28]`，其内容是已转换到 ControlFrameV1、并固定乘 `2.0` scale 的真实 root COM linear velocity。禁止错用 `critic_(t+1)`。
- `L_next_proprio` 的目标是下一控制 step 的归一化 `ProprioReconstructionTargetV1` 16D。其定义固定为 `sum(mask * squared_error) / max(sum(mask) * 16, 1)`；全 batch 无有效 transition 时必须返回与 decoder 输出保持计算图连接的零值。
- `L_KL = mean(-0.5 * (1 + logvar - mu^2 - exp(logvar)))`，对 batch 和 16 个 context feature 一起取均值；它只约束 context posterior，不约束 deterministic velocity head。
- CENet loss 不是环境 reward。
- Actor 和 Critic 仍共享同一个环境 reward。
- `velocity_coef=1.0`、`reconstruction_coef=1.0`、`kl_beta=1.0` 固定为第一版 A1 Base 对齐值；变更时必须形成新的 algorithm contract，不能通过临时 CLI 静默覆盖正式 run。

辅助 loss 有界不等价于 estimator 有效。第一版额外冻结两个只用于诊断和 Phase 2 退出验收、不会进入 loss 或梯度的量化 tripwire：

- `context_mu_feature_std_mean` 定义为一个完整 rollout 中对 batch/time 样本轴计算 16 个 `context_mu` feature 的 population standard deviation，再对 feature 取均值。首个完整 rollout 在第一次 optimizer step 前记录 `context_mu_feature_std_mean_initial`；正式 run 的 final 值取最后 10 个完整 iteration 的同名指标算术平均。必须满足 `context_mu_feature_std_mean_final >= max(1.0e-3, 0.10 * context_mu_feature_std_mean_initial)`。该门槛只排除明显 posterior/context collapse，不要求 latent 服从特定尺度。
- 确定性 Isaac Lab evaluation 使用 Actor 在该 evaluation contract 下实际消费的同一 `policy/history`，同时从 clean critic 提取 `critic[25:28]` 标签；记录 `velocity_eval_mse = mean((estimated_velocity-target_velocity)^2)`、`velocity_zero_baseline_mse = mean(target_velocity^2)` 和比值 `velocity_eval_mse_ratio = velocity_eval_mse / max(velocity_zero_baseline_mse, 1.0e-8)`。Phase 2 退出要求 `velocity_eval_mse_ratio <= 0.80`，即 estimator 至少比恒零速度预测器降低 20% MSE。正式 deterministic evaluation 固定使用禁用随机化和 observation noise 的 `NominalEvaluationProfileV1`；训练路径仍保持 Phase 1R 的 noisy policy/history。

上述门槛、final window `10` 和 `1.0e-8` denominator floor 必须进入 `DreamWaQAlgorithmContractV1`，正式 run 禁止 CLI 覆盖。门槛失败不允许把 checkpoint 标记为 Phase 2 通过，但不得通过改变 `kl_beta`、加入 KL warm-up/free-bits 或修改网络来静默修复；这类变化必须形成新的 algorithm contract。另记录 auxiliary-loss gradient norm 与 PPO gradient norm 的比值作为训练诊断，但第一版不为该比值设置通过阈值。

训练侧的 context tripwire 运行状态固定为 `EstimatorMonitorStateV1`：精确包含 `schema_version`、`completed_rollouts`、`context_mu_feature_std_mean_initial` 和 `recent_context_mu_feature_std_mean`。`completed_rollouts` 每完成一次 rollout、在 optimizer update 前加一，并必须与 checkpoint 的 `completed_iterations` 一致；initial 是第一个完整 rollout 的值，之后永不重置；recent list 按时间顺序保存最近最多 10 个完整 rollout 的值。fresh run 在第一次 optimizer step 前创建该状态；resume 必须在收集下一 rollout 前从 checkpoint 精确恢复，禁止把 resumed run 的第一批重新当作 initial。正式 final 值只从该状态的最近 10 项计算；少于 10 项的 smoke 可以记录 partial mean，但不得宣称通过正式 final-window gate。

#### 16.3.1 Deterministic Isaac evaluation contract

Phase 2 的 estimator 和策略对照统一使用 `IsaacEvaluationContractV1` / `IsaacEvaluationV1`，由独立 `scripts/evaluate_isaac.py` 执行，不能由可交互 `play.py`/`play_dreamwaq.py` 的任意 CLI 组合代替。固定协议如下：

| 字段 | 固定值 |
|---|---|
| evaluation seed | `20261007` |
| environment count | `8`，一个环境对应一个固定场景 |
| randomization/noise | `NominalEvaluationProfileV1`，`enabled=false`；禁止随机 episode length |
| control horizon | 每场景 `499` 个 `0.02 s` action step，即 `9.98 s` |
| policy action | deterministic effective mean；runtime 继续裁剪到 `[-1,1]` |
| scenario order | `nominal_stand`、`low_stand`、`high_stand`、`forward`、`reverse`、`left_turn`、`right_turn`、`combined` |
| command vectors | `(0,0,0.20)`、`(0,0,0.17)`、`(0,0,0.23)`、`(1.0,0,0.20)`、`(-1.0,0,0.20)`、`(0,0.6,0.20)`、`(0,-0.6,0.20)`、`(0.8,0.5,0.20)` |

suite 必须先生成或加载一份 `num_envs=8` 的 evaluation closed-chain reset cache，并让 Phase 1R baseline 和四个 DreamWaQ candidate 精确复用同一文件；cache file SHA256、tensor SHA256、schema、relaxation/root-height algorithm version 和 evaluation source fingerprint 都进入 evaluation contract。构造原生 `RslRlVecEnvWrapper` 后保留上游 discarded reset，随后先把八条固定命令写入对应环境，再恰好调用一次 `get_observations()` 取得第一份受评估 observation。DreamWaQ evaluator 使用与训练/部署 golden test 共用的 frame-major history runtime，以该 first policy observation 复制填满五帧；不得把命令写入前的 observation 放入 history 或 metric。

`499` 不是对 `10 s` episode 的重新命名。当前 `DirectRLEnv.step()` 在 done 检查前先把 `episode_length_buf` 加一，而 WheelLeg `compute_dones()` 固定使用 `episode_length >= max_episode_length - 1`；从 reset 后的零计数开始，第 499 次 action step 会产生预期 timeout 并在返回 observation 前 auto-reset。Isaac evaluator 必须接受该第 499 step timeout 为正常完成，不能为了凑足 500 step 修改 `episode_length_s`、counter、termination 或绕过 auto-reset。MuJoCo 的独立正式协议继续保持 `500 tick / 10.0 s`，两者的 horizon 都必须分别写入各自 contract，禁止混称。

每个 tick 在执行动作前，从同一 current TensorDict/history 计算 deterministic action 与 estimated velocity；velocity label 是同一时刻 `critic[:,25:28]`。只对该 tick 开始时仍 active 的环境累计 estimator squared error、zero-baseline squared error，并累计该 action step 返回的 reward。执行 step 后，非 done 环境 append 返回的下一帧 policy；done 环境按 reset observation 复制五帧，但从下一 tick 起不再进入本次场景统计。`terminated` 一律是失败；`truncated` 只有发生在第 `499` action step 的预期 timeout 才算完成，提前 truncated 是协议失败。reset 后 observation 不得作为终止场景的新样本。velocity 两个 MSE 都按全部 active pre-action frame 和三个 velocity feature做 sample-weighted mean，禁止先算场景 ratio 再平均。

每个场景记录 `reward_sum`、`survival_steps`、`survival_fraction=survival_steps/499`、`completed` 和 failure reason；aggregate 记录 `completed_scenarios`、八场景 `mean_survival_fraction`、八场景 `mean_scenario_return`。策略比较 key 固定为：full survival 时 `[0, -round(mean_scenario_return,6)]`；否则 `[1, -completed_scenarios, -round(mean_survival_fraction,6), -round(mean_scenario_return,6)]`。按 Python tuple 的 lexicographic ascending 比较，DreamWaQ candidate 的 key `<=` baseline key 才满足“Isaac performance 不低于 Phase 1R baseline”。

`Phase1RIsaacBaselineV1` 固定为当前最佳诊断候选 run-03，但不把它声明为 MuJoCo 或 Phase 1R 性能通过模型：

```text
checkpoint: E:\wheel_leg_rl-main\wheelleg_dreamwaq\logs\rsl_rl\wheelleg_flat_ppo\2026-10-07_07-32-32_rtx5070_fudan-v1_phase1r-v3-suite-20261007-062147-run03-seed1884612625\model_999.pt
checkpoint SHA256: 31C1A7DB00AD00B68794D721B5B7DDD0E4765B583E3F916710D917D5053B56B4
run manifest SHA256: 026B6D8F151A9DFC3B13E6342EAE5B6851E34C78F2B01EABC2577F1ED4F54984
Phase1RandomizedContractV3 hash: CC57F16B1BB75E169CCF21CD706E10C8D2D23498434ACED982AFFA8FD1F72848
seed/completed iterations: 1884612625 / 1000
```

baseline 与 candidate report 必须绑定 checkpoint、run manifest、base-task contract、evaluation contract/cache/source hash 和逐场景结果。baseline report 由同一 evaluator 在当前 suite 中重新产生，禁止抄用 MuJoCo summary 或历史 play 输出。Phase 1R report 的 estimator 字段必须为 `null`；DreamWaQ report 必须包含两个 MSE、ratio、样本 frame 数和 acceptance boolean。

正式流程区分两类 gate。checkpoint/contract/hash、finite、TorchScript、evaluation protocol/cache 等工件有效性失败属于 hard failure，禁止继续使用该工件。context/velocity tripwire 或 Phase 1R performance comparison 失败属于 acceptance failure：该 checkpoint 不得标记为 Phase 2 通过或 selected，但只要工件仍有效，就必须继续导出并完成八场景 MuJoCo，保留负面证据；不得用性能失败跳过 sim2sim。

第一版使用一个 Adam optimizer 管理 Actor、Critic 和 CENet 参数，对加权总 loss 做一次 backward，但每条梯度路径必须显式满足：

- PPO actor loss 更新 Actor，并通过确定性的 44D Actor 输入回传到 CENet 共享 encoder、velocity 输出和 `context_mu` 输出行；对 `context_logvar` 张量及其 encoder 输出行的直接梯度必须为零。共享隐藏层被 actor loss 更新后仍可能间接改变后续 logvar 数值，这不视为 logvar 进入 Actor 路径。
- velocity loss 更新 CENet 共享 encoder 和 velocity head。
- KL loss 更新 CENet 共享 encoder、`context_mu` 和 `context_logvar` 分支。
- reconstruction loss 通过重参数化 `context_z` 更新 CENet 共享 encoder、`context_mu/context_logvar` 分支和 decoder。
- decoder 输入的 estimated velocity 和当前 action 都必须停止梯度，因此 reconstruction loss 不更新 velocity head，也不能通过动作采样路径更新 Actor。
- critic loss 只更新 Critic。

PPO policy path 必须完全确定：相同参数和 observations/history 必须得到相同 action mean、value 和 log probability，无需保存 CENet epsilon。decoder 的辅助采样使用 PyTorch RNG；checkpoint 必须保存并恢复 CPU/CUDA RNG state。单元测试需要显式注入固定 epsilon 验证重参数化公式，但生产 storage 和导出 manifest 中禁止出现 `context_epsilon` 字段。

Decoder 不得接入 Actor 推理路径。使用同一 optimizer 不等于共享网络参数，所有 loss 系数、global gradient clipping 和各模块 gradient norm 都必须记录；若后续改为多 optimizer或改变上述 stop-gradient 边界，视为算法变更并单独评审。

每个 PPO mini-batch 的 update 应对 history 执行一次 encoder forward，并让同一批 `estimated_velocity/context_mu/context_logvar` 同时服务 Actor、velocity/KL 和 decoder loss；decoder 只额外执行 reparameterization 与自身前向。禁止在同一 update 内为 Actor 和 auxiliary loss 各自重复运行带不同行为的 encoder 路径。

每个 mini-batch 的数值检查顺序固定为：前向后检查全部输出与各 loss；`zero_grad()` 后对 finite `L_total` 执行一次 backward；检查全部非 `None` gradient；多 GPU reduction 后再次检查 gradient；执行 `clip_grad_norm_` 并检查返回的 pre-clip norm 与 clip 后 gradient；然后执行一次 optimizer step，并检查全部模型参数以及 Adam `exp_avg/exp_avg_sq/step` 状态。任一边界失败都必须在写入下一个 transition 或 mini-batch 前终止 run，并保存 iteration、mini-batch 索引、loss、输出范围和各模块 gradient norm；禁止继续训练或用数值替换掩盖错误。

DreamWaQ 继承 Phase 1 PPO 参数：`num_steps_per_env=24`、`num_learning_epochs=5`、`use_clipped_value_loss=true`、`clip_param=0.2`、`value_loss_coef=1.0`、`entropy_coef=0.01`、`learning_rate=1e-3`、adaptive schedule、`gamma=0.99`、`lam=0.95`、`desired_kl=0.01`、`max_grad_norm=1.0` 和 `normalize_advantage_per_mini_batch=false`。Policy 固定 `init_noise_std=1.0`、`noise_std_type=scalar`、`state_dependent_std=false`、Actor/Critic empirical normalization 均关闭，runner `clip_actions=1.0`。`num_mini_batches` 仍是 hardware-mutable，但 rollout batch 必须整除它。第一版不启用 RND、symmetry augmentation、recurrent policy 或额外 optimizer。

DreamWaQ checkpoint 必须包含 Actor、Critic、CENet encoder/decoder、6D action distribution 参数、optimizer state、adaptive learning rate、schema/normalization/algorithm manifest、iteration、`EstimatorMonitorStateV1` 和全部 RNG state。Phase 2 的 canonical reference/sim2sim 导出固定为 TorchScript `actor.ts`，配套 `policy_manifest.json` 和 `golden_vectors.pt`；`actor.ts` 封装无状态 `DreamWaQInferenceActorV1`：输入是 rank-2 `float32`、已经归一化且 frame-major 的 `history [N,125]`，`N` 为动态 batch；内部取最后 25D 作为 current observation，执行 deterministic encoder 与 Actor effective mean，输出 rank-2 `float32`、环境 `[-1,1]` 裁剪之前的 canonical action mean `[N,6]`。五帧 history buffer、reset 复制填充、传感器适配和最终动作裁剪由部署/MuJoCo runtime 负责；导出图不包含 decoder、Critic、Gaussian sampling、normalizer 状态或环境对象。G-10 后续可以增加 ONNX、RKNN 或其他部署后端，但必须以该 TorchScript artifact 和 golden vectors 为数值参考，不能改变输入输出语义。

### 16.4 下一帧和 reset mask

Isaac Lab `DirectRLEnv` 在 episode 结束时会 reset 对应环境，并向 runner 返回 reset 后观测。algorithm 可以在 `process_env_step` 读取该 TensorDict 来提取 16D target，但 storage 不保存完整 next observation；重建 mask 必须满足：

```python
# RslRlVecEnvWrapper returns integer dones: 0 or 1.
reconstruction_mask = dones.eq(0).view(-1, 1)
```

任何发生 reset 的 transition 都不参与下一帧重建 loss，避免把新 episode 初始状态当作上一 episode 的下一状态。

transition 的时间定义固定为：`DreamWaQPPO.act(obs_t)` 先把含 noisy `policy_t/history_t` 和 clean `critic_t` 的 TensorDict 深拷贝为 pending transition 快照，并由以 `o_t` 结尾的 noisy `history_t` 产生 rollout action。不能沿用上游 PPO 对 TensorDict 只保存引用的写法，否则 wrapper 在 `env.step()` 后更新 history 时可能把待写入 storage 的 `history_t` 改成 `history_(t+1)`。RSL-RL storage 中保存的是未裁剪的 Gaussian sample；环境实际执行、`previous_action` 和 decoder condition 统一使用 `a_t=clamp(stored_action_t,-1,1)`，不得新增另一份含义不同的 action。环境 step 后，`DreamWaQPPO.process_env_step(obs_(t+1), rewards, dones, extras)` 从 reset-aware `next_critic_obs` 提取归一化 clean 目标 `p_(t+1)=concat(next_critic_obs[0:6],next_critic_obs[9:19])`。`DreamWaQRolloutStorage` 在 RSL-RL 3.1.2 标准 TensorDict observations、action、reward、combined `dones`、value、log probability、action mean/sigma、return 和 advantage 之外，只新增：

```text
next_proprio_target: [T, N, 16], float32
reconstruction_mask: [T, N, 1], bool
```

Isaac Lab/RSL-RL runner 向 algorithm 提供的 `dones` 已是 `terminated | truncated`，且在 Isaac Lab 2.3.2 `RslRlVecEnvWrapper` 中 dtype 为 `torch.long`；`extras["time_outs"]` 只用于 PPO timeout bootstrap。禁止直接对整型 `dones` 使用 `~dones`，因为 `~0=-1`、`~1=-2`，转为 bool 后都会成为 `True`。mask 必须先通过 `dones.eq(0)` 或等价的 `~dones.to(torch.bool)` 生成，再 reshape 为 `[N,1]`；terminated 和 truncated 都必须被屏蔽。storage 不保存完整 next observation、`context_mu/logvar/z`、`context_epsilon` 或 AdaBoot 字段；这些值在 mini-batch update 时由当前 encoder 重新计算。

history wrapper 的 `step` 必须在底层环境返回 reset-aware `o_(t+1)` 后更新下一次调用所见的 history：非 done 环境 append 完整的 `o_(t+1)`；done 环境用 DirectRLEnv 已 reset 后返回的 `o_reset` 复制填满五帧。当前 transition 的 `history_t` 已由 `DreamWaQPPO.act` 保存，禁止用更新后的 `history_(t+1)` 覆盖它。MuJoCo sim2sim 和部署 runtime 必须复用同一顺序：reset 后以当前观测复制五次，之后每个 50 Hz tick 在动作执行并获得下一观测后才滚动历史。

### 16.5 RSL-RL 类、mini-batch 与 checkpoint 契约

`DreamWaQActorCritic` 的构造签名必须兼容 RSL-RL 3.1.2 runner 的 `(obs: TensorDict, obs_groups: dict[str,list[str]], num_actions: int, **explicit_cfg)` 调用，并保持 `act`、`act_inference`、`evaluate`、`get_actions_log_prob`、distribution properties、`update_normalization`、`reset` 和 `is_recurrent=false` 行为。`update_normalization` 在本 contract 下是显式 no-op，并断言 Actor/Critic empirical normalizer 都已关闭。`load_state_dict(state_dict, strict=True)` 必须严格加载全部模型键并显式 `return True`，使 RSL-RL runner 恢复 optimizer 和 iteration。runner 在调用上游 `resolve_obs_groups` 前、ActorCritic 在构造时都必须断言 `obs_groups` 精确等于 `{"policy":["policy"],"critic":["critic"]}`；缺失、额外 observation set、组内额外 term 或顺序/内容变化都立即失败，`policy_history` 只能通过显式 `history_obs_group` 读取。构造时还必须严格校验三个 TensorDict key、全部维度、网络结构、`num_actions=6` 和 normalization 开关；未知参数直接报错，不沿用上游 `**kwargs` 仅打印后忽略的兼容行为。

自定义 update 接口必须用一个具名 `DreamWaQBatchOutput` 返回：

```text
action_mean, action_sigma, action_log_prob, entropy
value
estimated_velocity, context_mu, context_logvar
predicted_next_proprio
```

该接口对一个 mini-batch 只运行一次 encoder；传入 storage action 后，内部以 `clamp(action,-1,1)` 构造 decoder condition。`DreamWaQPPO` 用该输出计算标准 PPO 与三项 auxiliary loss。不得依赖 `policy._last_*` 这类跨调用可变缓存把 rollout、value 和 auxiliary forward 隐式串联。

`DreamWaQRolloutStorage.Transition` 在标准 transition 上只增加 `next_proprio_target` 和 `reconstruction_mask`。`DreamWaQPPO.__init__` 在完成上游 PPO 初始化后必须把 `self.transition` 替换为该自定义 Transition；`init_storage(training_type,num_envs,num_steps,obs,actions_shape)` 必须构造 `DreamWaQRolloutStorage`，否则 runner 会退回标准 storage 并在第一步丢失附加字段。`add_transitions()` 必须要求两项附加字段均非 `None`，并严格校验 target 为同 device 的 `float32 [N,16]`、mask 为同 device 的 `bool [N,1]`；缺失、shape/dtype/device 不符立即失败，禁止退化为 current observation、全零 target 或全真 mask。两个附加 storage tensor 必须在 storage 构造时预分配为普通张量，并与上游字段一致使用 `buffer[step].copy_(transition_field)` 写入；禁止引用赋值、替换 buffer 或把 rollout 的 inference tensor 直接保存到 update。写入后再修改、清空或复用 pending transition 不得改变 storage 中已保存的 target/mask。`mini_batch_generator` 返回具名 `DreamWaQMiniBatch`，包含 RSL-RL 标准十项 mini-batch 字段以及这两个附加字段；所有字段必须使用同一个 flattened shuffle index。实现可以内部继承标准 storage，但禁止通过不带字段名的尾部位置猜测附加张量。

`DreamWaQPPO` 必须继承 RSL-RL 3.1.2 `PPO`，只覆盖创建自定义 transition/storage、`act` 的 TensorDict snapshot、`process_env_step` 的 target/mask 写入，以及 `update` 中的 DreamWaQ mini-batch 与 auxiliary loss。`process_env_step` 必须先把 next target/mask 写入 pending transition，再复用上游 normalizer no-op、timeout reward bootstrap、storage add、transition clear 与 policy reset 顺序，不能先调用上游方法后再补字段。标准 action rollout 字段、GAE/returns、adaptive-KL learning-rate schedule、advantage normalization、multi-GPU parameter reduction 和 global gradient clipping 继续保持上游语义；`DreamWaQMiniBatch` 必须原样保留 standard storage 的 `old_mu` 与 `old_sigma`，adaptive-KL 的公式、阈值、learning-rate 上下界和所有 parameter-group 更新与 RSL-RL 3.1.2 逐项等价。若因 auxiliary loss 必须复制 update 循环，必须用逐项等价测试锁定这些行为。`rnd` 与 symmetry 配置固定为 `None`。

DreamWaQ checkpoint 保留现有 RSL-RL 顶层字段格式，但不能直接复用只识别 Phase 1 metadata/单一 `manifest["contract"]` 的 `ContractOnPolicyRunner` 验证路径。`training/dreamwaq_checkpoint.py` 必须提供专用 `DreamWaQContractOnPolicyRunner`、`validate_dreamwaq_checkpoint_metadata()` 和 `resume_dreamwaq_runner_from_checkpoint()`；Phase 1 的 `ContractOnPolicyRunner` 和 validator 不因 Phase 2 被放宽。顶层格式为：

```text
model_state_dict
optimizer_state_dict
iter
infos
```

`model_state_dict` 必须以 strict load 覆盖 Actor、Critic、CENet encoder、CENet decoder 和 action std；`optimizer_state_dict` 必须覆盖同一组参数及完全一致的 parameter-group 结构。run manifest 使用三个独立、具名且各自带版本/hash 的字典，禁止继续放入含义不明的单一 `contract` 字段：

| 字典 | 固定版本 | 覆盖范围 |
|---|---|---|
| `base_task_contract` | `WheelLegBaseTaskContractV1` | 从当前 Phase 1R 配置提取的资产、物理、collision、actuator、ControlFrame、Action/Command/ActorObs/CriticObs/Normalization、奖励、reset/termination、命令和域随机化/cache 语义；明确排除 PPO 网络、optimizer、runner/logger 字段 |
| `dreamwaq_algorithm_contract` | `DreamWaQAlgorithmContractV1` | runner class、Actor/Critic/CENet 拓扑、history/reconstruction schema、distribution、PPO、loss、storage、gradient 和 optimizer 语义 |
| `dreamwaq_export_contract` | `DreamWaQExportContractV1` | TorchScript 三件套、输入输出签名、dynamic batch、effective mean/runtime clip、golden-vector seed/误差阈值和 artifact hash 规则 |

`WheelLegBaseTaskContractV1` 可以复用现有 Phase 1 contract builder 的字段生成 helper，但必须由独立 allow-list 生成，不能先生成包含 PPO 网络的 `Phase1RandomizedContractV3` 再通过运行时删除任意键。该 allow-list 必须完整覆盖 `RandomizationProfileV1` 的四个 Actor observation noise 幅度：`angular_velocity_noise_rad_s`、`projected_gravity_noise`、`leg_position_noise_rad`、`joint_velocity_noise_rad_s`，并覆盖 profile/schema、stream lifecycle 与 reset-cache 语义。相同环境配置产生的 base-task hash 必须在 PPO 对照和 DreamWaQ run 间一致；算法差异只进入各自 algorithm contract。

`DreamWaQContractOnPolicyRunner` 固定继承现有 `ContractOnPolicyRunner` 以复用日志聚合和构造参数，但必须完整覆盖 `save()`；它不得调用父类硬编码 Phase 1 metadata 的 `save()`，而是在构造完 `DreamWaQCheckpointMetadataV1` 后直接调用 RSL-RL `OnPolicyRunner.save()` 写入标准顶层字段。`train_dreamwaq.py` 只能构造该 runner，不能构造普通 `ContractOnPolicyRunner`。

`infos` 的 `DreamWaQCheckpointMetadataV1` 必须至少含以下精确字段：`metadata_version`、`saved_at`、`runner_class`、`seed`、`runner_iteration`、`completed_iterations`、`total_timesteps`、`total_time_seconds`、`optimizer_learning_rates`、`algorithm_learning_rate`、`rng_state`、`environment_rng_state`、`estimator_monitor_state`、`run_manifest_sha256`、`base_task_contract_version`、`base_task_contract_hash`、`dreamwaq_algorithm_contract_version`、`dreamwaq_algorithm_contract_hash`、`dreamwaq_export_contract_version`、`dreamwaq_export_contract_hash`、`hardware_profile`、`reset_cache_binding` 和 `resume_provenance`。`runner_class` 必须精确为 `DreamWaQContractOnPolicyRunner`；resume 时 requested seed 必须与 checkpoint seed 相等。`estimator_monitor_state` 必须通过 `EstimatorMonitorStateV1` 严格 key/type/finite/长度校验，且 `completed_rollouts == completed_iterations`；load/resume 必须在下一次 rollout 前恢复该状态。

`reset_cache_binding` 复用 Phase 1R 当前 V2 schema。`resume_provenance` 固定使用 `ResumeMetadataV2`，只允许以下两种互斥 key set：

```text
fresh:
  {schema_version="ResumeMetadataV2", mode="fresh"}

weights_optimizer_rng_new_environment:
  schema_version, mode, cache_resume_mode,
  source_checkpoint, source_checkpoint_sha256,
  source_run_manifest, source_run_manifest_sha256,
  source_completed_iterations,
  source_hardware_profile_name, target_hardware_profile_name,
  source_num_envs, target_num_envs, same_num_envs,
  source_cache_file_sha256, source_cache_tensor_sha256,
  target_cache_file_sha256, target_cache_tensor_sha256,
  source_realized_plan_hash, target_realized_plan_hash,
  rng_stream_sources
```

fresh mode 禁止出现任何 `source_*`、`target_*`、`cache_resume_mode` 或 `rng_stream_sources` 字段；resume mode 必须完整包含上表字段且 `mode="weights_optimizer_rng_new_environment"`。可选上游 `rsl_rl_infos` 必须嵌套保存，不能覆盖这些保留字段。save 前必须验证 optimizer learning rate 与 `alg.learning_rate` 一致；load/resume 必须逐项验证 runner/seed、三个 contract、metadata version、模型键、optimizer 参数组、环境 RNG/cache provenance 和目标总轮数。

resume 只能加载三个 contract、metadata、模型键和 optimizer 参数组都严格匹配的 DreamWaQ checkpoint，并继续沿用“新 run directory + `--max-iterations` 表示总目标轮数”的规则。普通 PPO/Phase 1R checkpoint 可作为对照或独立 Actor 初始化研究的输入，但第一版正式流程不做 warm-start，且绝不能作为完整 DreamWaQ resume checkpoint 静默加载；反向也必须拒绝 DreamWaQ checkpoint 被 Phase 1 runner 当作 PPO checkpoint 加载。

## 17. 奖励架构

环境每个控制 step 只产生一个共享奖励：

```text
r_total = sum(weight_i * reward_i)
```

奖励模块按具名函数拆分，并逐项记录 episode sum。V1 奖励候选集合：

- `tracking_vx`。
- `tracking_vx_enhance`。
- `tracking_yaw_rate`。
- `tracking_yaw_rate_enhance`。
- `tracking_base_height`。
- `lin_vel_z`。
- `ang_vel_xy`。
- `orientation`。
- `dof_vel`。
- `dof_acc`。
- `torques`。
- `action_rate`。
- `action_smooth`。
- `phi0_symmetry`，仅在 `Phase1ContractV4` 启用。
- `collision`，仅在接触传感器决策通过后启用。
- `dof_pos_limits`。

约束：

- 公式可以参考复旦 Plane，控制周期不参考复旦。
- 所有时间敏感项按 `control_dt=0.02s` 重新标定。
- 原复旦 `nominal_state` 不得原样复制。
- 对称奖励可以定义在关节空间或经版本化审计的闭链几何空间，但必须独立命名、测试并进入 contract；`Phase1ContractV4` 采用真实虚拟腿角几何奖励，不复制复旦 `nominal_state`。
- 奖励函数不得依赖 Actor observation 的索引和缓存。
- 奖励权重全部集中在配置中。
- `tracking_vx` 使用 ControlFrameV1 root COM `vx`，`tracking_yaw_rate` 使用 ControlFrameV1 `wz`，高度使用 root COM 相对当前地面的高度。
- 姿态量固定定义为 `upright_cos = -projected_gravity_z`；`orientation = 1 - upright_cos`，因此直立、侧倒、完全倒置分别为 `0、1、2`，禁止只使用 `projected_gravity_xy^2` 而让倒置状态得到零惩罚。

V4 的 `phi0_symmetry` 固定为：

```text
delta_phi0 = atan2(sin(phi0_left - phi0_right), cos(phi0_left - phi0_right))
phi0_symmetry = delta_phi0^2
weight_phi0_symmetry = -1.0
weighted_phi0_symmetry = weight_phi0_symmetry * control_dt * phi0_symmetry
```

该项只约束左右真实 `phi0` 相等，不约束绝对角度。`10 deg` 角差的单步惩罚约为 `-0.00061`，`30 deg` 约为 `-0.00548`，用于提供温和但可见的对称约束，不替代姿态、高度、关节限位和终止条件。角度与长度的权威计算、USD anchor 审计和失败策略见 `docs/2026-10-04-wheelleg-phi0-sim2sim-usd-cleanup-design.md`。

## 18. Reset、termination 和命令边界

`commands.py`、`terminations.py` 和 reset 逻辑必须独立于奖励模块。

架构要求：

- timeout 与 failure termination 分开统计。
- reset 同时重置动作历史、关节加速度历史和 DreamWaQ 五帧历史。
- `terminated` 与 `truncated` 都终止 history continuity，并共同令 reconstruction mask 为 false；timeout bootstrap 是否保留只影响 value return，不得重新开启 CENet 重建目标。
- `Phase1ContractV4` 的命令只在 reset 时按 CommandSamplingV2 采样一次并保持到 episode 结束；环境 step 中不得按物理 step 或秒数再次采样。
- 播放时的键盘命令只存在于 play 脚本，不进入环境 step。
- termination 不得通过奖励值间接触发。
- tilt failure 固定使用 `upright_cos < cos(max_tilt_rad)`；必须能区分直立、阈值边界、90 度侧倒和 180 度倒置，不能只看 projected-gravity XY 范数。

具体命令范围、姿态失败阈值、机身高度阈值和 episode 时长在 PPO 实现前的任务参数评审中冻结。

### 18.1 ResetObservationCacheCoherenceV2

**设计状态：** 用户已授权实施、一个 agent 代码复核及 sim2sim。V2 最小生产修复已实施，独立代码复核最终 P0/P1/P2/P3 均为零；普通、随机化、debug 类各覆盖生成/加载 cache 的六个真实 Isaac 子进程，48 个 case、638 项检查通过。旧 checkpoint 的全新八场景诊断中，Isaac 仍为 8/8、MuJoCo 仍为 0/8；修复不等于旧策略性能恢复。完整身份、原始证据与重训建议见上述专项结果记录；没有启动训练，RootCauseSuite 正式门仍未关闭。

已确认的窄缺陷是：Isaac Lab 2.3.2 的 root velocity writer 更新 `root_com_vel_w` 与 PhysX root velocity，却没有失效派生的 `_root_link_vel_w` 缓存。该缓存通过 `timestamp < _sim_timestamp` 决定是否重算；reset 写入新状态时 simulation timestamp 不前进，因此 `root_link_ang_vel_b` 可能仍返回上一 rollout 的角速度。当前 `_reset_idx()` 随后把该值写入 `self._state`，history wrapper 再使用该观测初始化五帧。这是观测缓存同步缺陷，不能据此断言 MuJoCo 物理模型错误或旧 checkpoint 失败只有这一个原因。

最小生产修改位置为 `WheelLegFlatEnv._reset_idx()`：在本次 root pose、root velocity、joint state 与现有 action/command reset 写入完成后，在现有 `reset_state = self._read_state()` 前，将 `self.robot.data._root_link_vel_w.timestamp` 与 `self.robot.data._root_com_pose_w.timestamp` 都设为 `-1.0`。继续通过第 10 节冻结的 `root_link_ang_vel_b` 与 `root_com_pos_w` API 读取；从已经更新的基础 velocity、root-link pose 与 local COM 重算。沿用现有 `_read_state()` 与 `replace_rows()` 生成和合并状态快照，不另建 Actor/Critic 专用修正值。

独立文档复核发现的 `_root_com_pose_w` 边界已在真实 PhysX nominal 双环境 probe 复现：重复 reset 的 raw COM/height 最大误差约 `0.001051 m`，tilt failure 后高度误差可达 `1.1964 m`；角速度 normalized error 约 `0.03465`。原始报告与 tensor 保存在 `wheelleg_dreamwaq/artifacts/reset-observation-20261009/red-formal-03/`。因此 V1 的角速度-only 候选未实施，先按证据修订为 V2；不扩大到未消费的 combined buffers 或 blanket invalidation。若还有其他字段 split，继续停止并定位。

该操作是 Isaac Lab 2.3.2 固定依赖的局部兼容修复：只修改缓存有效标记，不把角速度强制置零，不替换坐标转换，不增加随机数调用，不修改 `_sim_timestamp`，不调用正 `dt` 的 `scene.update()`，不增加 `sim.step()`。当本次 reset 使用随机初速度时，观测必须反映实际新采样值。若依赖结构变化而该 buffer 不再存在，必须显式失败并重新评审，不得静默跳过。

部分环境 reset 时，buffer 的有效标记可能使下一次读取重算全批次派生速度，但 `self._state` 只合并 `env_ids` 对应行；未结束环境的 post-physics 快照、reward/done、命令、动作历史和控制时间不得因此变化。空 `env_ids` 继续按现有逻辑直接返回。

sim2sim 的 Isaac debug subclass 同样调用正式 `_reset_idx()`，修复必须在共同生产路径生效，并验证其覆写 `reset()` 的 pre-forward/post-forward/returned 三相证据。MuJoCo 保持现有 `mj_resetDataKeyframe -> mj_forward -> collect_kinematic_state -> history fill` 时序；只有同一缺陷在 MuJoCo 端被实际证明时才另行提出运行时代码修改。本轮对 MuJoCo 增加重复 reset 的验收，保留物理参数与 runtime。

最低验收包括：

- 先产生并读取非零终端角速度，再执行 fresh/full repeated reset；noise-disabled Actor `0:3` 与独立 direct PhysX 当前角速度经过相同坐标变换和固定归一化后的结果一致。
- 在 nominal 与随机化配置下，对照 clean Critic `0:3` 角速度、`28:29` 高度以及任务快照的 raw COM 位置/高度；expected 从 direct PhysX root pose 和 COM local offset 独立构造，不得用待测 `root_com_pos_w` 生成。角速度通过但 COM/height 或其他字段 split 时仍属未通过。Actor 仍按原有 observation-noise 流程生成，不能把 noisy Actor 与 clean PhysX 直接要求相等。
- reset 的物理时间、物理步计数、scene-data timestamp 不前进；正常控制 step 仍为 `0.005 s * 4 = 0.02 s`。
- partial reset 和真实 terminated/truncated auto-reset 只更新结束环境；其余环境快照与 history continuity 保持现有语义。
- wrapper 的 discarded/first observation 次数、七个 RNG stream 调用顺序和首次五帧复制规则保持不变；five-frame history 等于本次 first policy observation。probe 必须记录实际 `_get_observations()`/`sample_actor_noise()` 调用计数与七流状态：正式类 wrapper 构造/full reset 各为既有两次底层观测，debug subclass 另有既有 pre-forward capture，需分别记录；连续两次 cached `get_observations()` 的计数与 RNG 状态不变。
- MuJoCo 从非零旧状态重复 reset 后返回当前 keyframe 的观测和五帧 history，reset 不增加物理 step，正常 action 仍执行 20 个 `0.001 s` 物理 step。

旧 suite、checkpoint 和调查产物保持历史身份。修复后的测试和评估必须使用新输出目录/source identity；不得 resume 旧的 paused training suite，不得改写旧调查来标记通过。RootCauseSuiteCoreV1.17 的冻结文档不因本节而被覆盖；正式重新运行 G00/G01/G02 前需建立与新源码匹配的显式诊断修订身份。独立复核和全新 G01/G02 未完成前，P10-P60/C70 正式 verdict 继续阻断；本次不自动调用其他 agent，不启动训练。

## 19. 实机部署边界

部署计算平台尚未冻结。H7 与 RK3566 都只是候选后端；本节先冻结与平台无关的输入、数值和安全接口。策略输入只允许来自以下信号：

- BMI088 角速度，经零偏补偿和 `R_control_from_imu` 转换到 ControlFrameV1。
- IMU 姿态滤波产生的 `[w,x,y,z]` control/body-to-world 四元数。
- 由四元数计算的 projected gravity。
- 四个腿关节位置，经 hardware sign、zero offset 和 protocol-to-policy reorder。
- 四个腿关节速度和两个轮速，经 hardware sign 和 protocol-to-policy reorder。
- 上层三维命令。
- 控制器保存的上一时刻 clipped canonical ActionV1。

角速度预处理责任固定在部署 observation adapter：

```text
omega_control = R_control_from_imu * (omega_raw - gyro_bias)
```

`R_control_from_imu=I` 只表示当前 IMU 正向安装，不表示 `gyro_bias=0`。现有固件读取路径中的 `IMU_Param_Correction` 被注释，不能把原始 BMI088 输出当作已完成零偏补偿。部署后端必须在上电静止窗口估计 bias，并使用运动/方差阈值拒绝无效标定；标定算法版本、单位、窗口、阈值和允许范围写入 manifest，单次上电测得的 bias 写入运行日志。

projected gravity 的唯一策略语义为单位向下重力：

```text
projected_gravity_control = R_world_from_control^T * [0, 0, -1]
```

现有固件局部变量 `gravity_b` 来自 `R^T * [0,0,+9.81]`，符号和量纲都与策略输入不同，禁止直接送入 Actor。若部署后端复用该计算结果，必须显式转换：

```text
projected_gravity_control = -gravity_b / ||gravity_b||
```

静止直立时策略输入的期望值必须接近 `[0,0,-1]`；该断言与四元数方向、六面静止和单轴正转 golden vector 一起冻结。

不要求实机提供：

- 机身真实线速度。
- 机身真实离地高度。
- 绝对 yaw。
- 地形高度。
- Critic privileged state。

实机观测适配器必须与训练 schema 共享：

- 关节顺序。
- 仿真 joint sign、硬件 motor sign、腿 protocol reorder 和轮协议顺序。
- 零位和 `q_nominal`。
- 单位。
- ControlFrameV1 和 IMU 安装变换。
- 对应合约 NormalizationV1/V2 的每字段 offset、scale 和 clip。
- action scale。
- 历史顺序。
- 控制周期。

导出包必须同时携带模型和 manifest。manifest 至少包含：

- Actor、Action、Command、Critic 和 CENet schema 版本。
- HistoryLayoutV1 和 ProprioReconstructionTargetV1 版本。
- 模型输入输出维度。
- checkpoint 所选 `AssetBundleV1/V2` 的五个相对路径、大小、SHA256、入口层和可选 overlay 身份。
- 关节名称、canonical order、腿 protocol reorder、轮协议顺序 `[left(ID4),right(ID5)]` 和全部 sign。
- `q_nominal`。
- hardware zero offset。
- `R_control_from_usd`、`R_control_from_imu` 和四元数约定。
- gyro bias 标定算法/参数、projected gravity 公式与单位向量约定。
- observation normalization 的 offset、scale 与 clip。
- action scales。
- `control_dt`。
- CENet history length、frame-major 展平顺序、velocity scale、encoder/decoder 拓扑、35D 输出 slice、`actor_context_mode=context_mu` 和 decoder 训练契约。
- Actor/Critic hidden dimensions、ELU 激活、PyTorch Linear 默认初始化、loss 系数和 `DreamWaQInferenceActorV1: [N,125] -> [N,6]` 导出签名。
- 模型数值精度、导出格式和后端版本。
- 训练 git revision 或构建标识、master seed、硬件 profile、`num_envs` 和 mini-batch 划分。

Phase 2 reference export 的目录内容固定为：

```text
actor.ts
policy_manifest.json
golden_vectors.pt
```

`golden_vectors.pt` 至少保存按以下固定算法生成的输入与 eager 输出：

```python
generator = torch.Generator(device="cpu").manual_seed(20261007)
history = torch.randn(
    (32, 125), generator=generator, dtype=torch.float32, device="cpu"
).contiguous()
inference_actor = inference_actor.cpu().eval()
with torch.inference_mode():
    expected_action_mean = inference_actor(history).contiguous()
scripted_actor = torch.jit.script(inference_actor)
```

禁止用 `torch.jit.trace` 代替 script。导出脚本必须保存 `scripted_actor` 为 `actor.ts`，重新 `torch.jit.load(actor.ts)`，并在 CPU 上验证 eager 与 TorchScript 的最大绝对误差不超过 `1e-7`。`policy_manifest.json` 记录 `actor.ts` 与 `golden_vectors.pt` 的 SHA256，并包含一个在排除 `manifest_hash` 自身后计算的 canonical manifest hash；manifest 文件本身的 SHA256 由来源 run summary 和 MuJoCo evaluation record 外部记录，禁止设计自引用文件 hash。manifest 还必须记录 dynamic batch、frame-major history、effective mean `[-20,20]`、runtime action clip `[-1,1]` 和来源 checkpoint/三个 contract hash。MuJoCo 正式 evaluator 只接受该三件套，不能直接 import 训练侧 Python policy。

部署 adapter 必须包含独立于策略的安全状态机：传感器有效性检查、启动姿态检查、通信/推理超时、动作限幅、关节软限位、急停和失联回退。策略不能直接绕过这些保护写电机目标。

网络大小已经在 Phase 2 冻结：Actor `[256,128,64]`、CENet encoder `[128,64]`，部署推理图共 `79,721` 个 Linear 参数；Critic 和 decoder 不进入部署模型。选定平台仍必须通过模型文件、工作区 RAM、数值精度和 50 Hz worst-case latency 测试，但测试失败不能静默缩减层宽或改变 schema，只能更换部署后端、数值实现，或通过新 ADR 升级 algorithm/export contract。

## 20. 配置边界

配置分为六类：

1. **Asset 配置**：历史回放使用 `AssetBundleV1`；V4 新训练使用 `AssetBundleV2`。每个 bundle 都固定根路径解析、五文件哈希/大小、入口层和 prim/joint 期望；根路径可以按机器解析，资产清单与身份不能改变。
2. **Schema 配置**：坐标系、动作、命令、观测、history layout、reconstruction target、顺序、固定归一化和 manifest，变更需要 schema 升级。
3. **Task 配置**：控制周期、命令范围、reset、termination、reward 和 actuator。
4. **Algorithm 配置**：网络、PPO、CENet 和 loss 参数。
5. **Training hardware 配置**：并行环境数、batch 划分、headless 和渲染。
6. **Deployment backend 配置**：模型格式、数值精度、线程/加速后端和设备 I/O；不得覆盖 schema 语义。

training hardware profile 和 deployment backend profile 都不允许覆盖 schema 或 task 语义。训练 CLI 可以覆盖实验性超参数，但每次运行必须把最终解析后的完整配置写入日志目录。

Phase 2 第一版 `dreamwaq_cfg.py` 必须定义且只定义以下三类 configclass：

- `DreamWaQActorCriticCfg(RslRlPpoActorCriticCfg)`：声明 policy/CENet、observation key、维度、网络和分布边界。
- `DreamWaQPPOAlgorithmCfg(RslRlPpoAlgorithmCfg)`：声明三项 auxiliary coefficient 和标准 PPO 字段。
- `WheelLegFlatDreamWaQRunnerCfg(RslRlOnPolicyRunnerCfg)`：固定 `class_name="DreamWaQContractOnPolicyRunner"`，组装前两类，并声明 runner/runtime 可变字段。

三类的 `to_dict()` 必须产生 `class_name=DreamWaQContractOnPolicyRunner`、`policy.class_name=DreamWaQActorCritic`、`algorithm.class_name=DreamWaQPPO` 和下表全部固定字段；自定义类构造函数只接收这些显式字段，未知字段必须失败。禁止把 A1 的 `actor_obs_term_dims`、AdaBoot、velocity normalizer 或旧 RSL-RL 字段带入配置。

两个自定义子配置新增字段固定为：

| Configclass | 新增字段 |
|---|---|
| `DreamWaQActorCriticCfg` | `current_obs_group="policy"`、`history_obs_group="policy_history"`、`history_length=5`、`current_obs_dim=25`、`history_obs_dim=125`、`critic_obs_dim=41`、`velocity_dim=3`、`context_dim=16`、`reconstruction_target_dim=16`、`cenet_encoder_hidden_dims=[128,64]`、`cenet_decoder_hidden_dims=[64,128]`、`actor_context_mode="deterministic_context_mu"`、`action_mean_clip=20.0`、`action_std_min=1e-4`、`action_std_max=2.0` |
| `DreamWaQPPOAlgorithmCfg` | `velocity_coef=1.0`、`reconstruction_coef=1.0`、`kl_beta=1.0`、`strict_finite_checks=True`、`context_mu_min_feature_std=1.0e-3`、`context_mu_min_initial_std_ratio=0.10`、`context_monitor_final_window_iterations=10`、`velocity_mse_max_zero_baseline_ratio=0.80`、`velocity_mse_denominator_floor=1.0e-8` |

slice 定义不在 config 中重复：velocity label、reconstruction target 和 history layout 必须从 `schemas/observation.py` 的版本化常量读取。`rnd_cfg` 与 `symmetry_cfg` 使用继承字段并固定为 `None`。

Phase 2 第一版必须解析为以下固定算法配置：

| 配置 | 固定值 |
|---|---|
| `class_name` | `DreamWaQContractOnPolicyRunner` |
| `policy.class_name` | `DreamWaQActorCritic` |
| `algorithm.class_name` | `DreamWaQPPO` |
| TensorDict observation keys | `policy`、`policy_history`、`critic` |
| RSL-RL `obs_groups` | `policy:[policy]`、`critic:[critic]` |
| custom group fields | `current_obs_group=policy`、`history_obs_group=policy_history` |
| history/current/critic/action dims | `125 / 25 / 41 / 6` |
| velocity/context dims | `3 / 16` |
| Actor/Critic hidden dims | `[256,128,64] / [256,128,64]` |
| encoder/decoder hidden dims | `[128,64] / [64,128]` |
| activation | `elu` |
| actor CENet mode | `deterministic_context_mu` |
| sampled velocity / AdaBoot | `false / false` |
| observation/target source | noisy `policy/history` input；clean `critic[25:28]` velocity 与 next `critic[0:6]+critic[9:19]` target |
| action distribution | `init_noise_std=1.0`、scalar state-independent raw std、effective mean clip `[-20,20]`、effective std `clamp(abs(raw_std),1e-4,2.0)`、runtime `clip_actions=1.0` |
| empirical normalization | Actor `false`、Critic `false` |
| auxiliary coefficients | velocity `1.0`、reconstruction `1.0`、KL `1.0` |
| estimator acceptance tripwires | final context feature-std `>= max(1.0e-3, 0.10 * initial)`；deterministic velocity MSE / zero-predictor MSE `<= 0.80`；final window `10` iterations |
| PPO | 与 §16.3 冻结值一致 |

`num_envs`、`device`、`num_mini_batches`、`max_iterations`、save interval、seed、run name 和 logger 保持其既有 hardware/runtime 可变边界；其余上表字段进入 `DreamWaQAlgorithmContractV1` 与 checkpoint manifest。正式 run 禁止通过 CLI 改写固定算法字段。

同一正式多 seed suite 必须生成种子无关训练指纹。Phase 1 名义 suite 使用 `Phase1ContractV4`，Phase 1R suite 使用 `Phase1RandomizedContractV3`；Phase 2 DreamWaQ suite 使用 `WheelLegBaseTaskContractV1 + DreamWaQAlgorithmContractV1 + DreamWaQExportContractV1` 三个 hash。源码/依赖/运行栈、hardware profile、rollout 与 mini-batch 布局进入指纹；Phase 1R/Phase 2 还必须覆盖 randomization profile/schema、stream lifecycle、reset-cache schema 与两个 cache 算法版本。seed、派生 RNG、时间、命令行、realized audit/cache hash 和 resume 元数据不进入跨 seed 相等条件。原始 `env.yaml`/`agent.yaml` hash 仍保留用于审计，但因为其中包含 seed、run name 和 log directory，不直接作为跨 run 相等条件；这些配置的训练语义由对应 contract 覆盖。任一后续 run 指纹不一致时 suite 必须 fail fast。

旧 checkpoint 使用 `Phase1ContractV3`，只允许在 V3/V1 环境严格回放或续训。名义新训练 checkpoint 使用 `Phase1ContractV4`；Phase 1R 随机化训练使用 `Phase1RandomizedContractV3 + CheckpointMetadataV6 + ClosedChainResetCacheSchemaV2`。V4 固定 `AssetBundleV2`、`WheelOnlyCollisionV2`、CommandSamplingV2、NormalizationV2、RewardSchemaV2 和 VirtualLegKinematicsV1；Phase 1R 在这些任务语义上增加具名 randomization profile、独立 RNG streams、双 tensor reset cache 及 source/target resume provenance。contract hash 至少覆盖 Action/Command/Actor/Critic/Normalization/ControlFrame schema、命令模式概率与范围、episode 内保持语义、动作裁剪、finite/infinite-horizon 语义、完整 root 初始 pose/velocity/joint state、完整 rigid/articulation/PhysX/collision/actuator 配置、虚拟腿几何常量与奖励公式/权重、网络和 PPO 参数。机器人配置必须覆盖 `UsdFileCfg` 的 spawner callable 及除机器相关绝对 `usd_path` 之外的全部 spawn 参数；绝对路径由所选 AssetBundle 的相对入口文件和资产 hash 替代。地面配置必须覆盖 prim path、translation、spawner callable、Cuboid 尺寸、collision properties、visual material、physics material以及 friction/restitution combine mode，环境只能通过该配置生成地面，不得另有硬编码副本。只有 `device`、`num_envs` 与 `num_mini_batches` 被明确列为 hardware-mutable，不进入跨 profile 兼容性拒绝条件；它们仍必须记录在每个 run manifest。`max_iterations`、保存间隔、seed 和日志后端属于运行控制元数据，不改变模型兼容性。

## 21. 失败处理和运行时断言

以下情况必须 fail fast：

- 所选 AssetBundle 任一文件缺失，或相对路径、大小、SHA256、依赖层清单、default prim、articulation root、up axis、单位与资产契约不一致。
- 机器人不是通过 `UsdFileCfg` default prim reference 加载，任务 stage 不止一个全局 PhysicsScene，或机器人环境命名空间下出现 PhysicsScene prim。
- V3/V1 环境的内嵌 `GroundPlane/CollisionPlane` 在 physics initialization 后仍启用，或 V4/V2 资产中仍存在该 prim。
- 四个闭环 joint 的数量、名称、enabled 或 `excludeFromArticulation` 状态不一致。
- 找不到预期关节或刚体名称。
- 具名受控 joint 解析结果不唯一，或 canonical order、腿/轮 protocol order、reorder/sign 缺失。
- `R_control_from_usd` 或 `R_control_from_imu` 不是合法正交旋转矩阵。
- 观测、动作或 Critic 维度不一致。
- checkpoint 指定的 NormalizationV1/V2 或 manifest hash 与运行时不一致，或检测到重复 running normalizer。
- actuator、完整初始关节状态、求解器、地面材料、collision policy 或其他 Phase 0 runtime contract 与 checkpoint 不一致。
- history 不是 `[N,5,25]` frame-major，或 reconstruction target/decoder input 不是冻结的 16D/25D 契约。
- encoder、Actor、Critic 或 decoder 拓扑不是 `125-128-64-35`、`44-256-128-64-6`、`41-256-128-64-1`、`25-64-128-16`，35D encoder slice、ELU 激活或 auxiliary coefficient 与 `DreamWaQAlgorithmContractV1` 不一致。
- Actor policy path 读取 `context_logvar/context_z`、storage 含 `context_epsilon`、导出图不是无状态 `[N,125] -> [N,6]`，或部署/MuJoCo reset history 不是五份当前观测。
- history wrapper 重复 `get_observations()` 会推进环境 observation-noise RNG、改变缓存，或 `policy_history[:,-25:] != policy`。
- wrapper 构造/显式 reset 没有遵循“一次 discarded observation + 恰好一次 first policy observation”的固定序列，或 resume 后五帧历史不全等于记录的 `first_policy_observation`。
- RSL-RL 版本、模块来源或配置字段与固定的 3.1.2 接口不一致，或存在未识别配置字段。
- DreamWaQ runner/config/ActorCritic 任一位置解析出的 `obs_groups` 不精确等于 `{"policy":["policy"],"critic":["critic"]}`，包括额外 observation set 或把 `policy_history` 拼入标准 set。
- `train_dreamwaq.py` 在构造 `OnPolicyRunner` 前没有显式注册 `DreamWaQActorCritic`/`DreamWaQPPO`，或注册后的全局符号不是本项目类。
- runner config、algorithm contract 或 checkpoint metadata 的 `runner_class` 不是 `DreamWaQContractOnPolicyRunner`，或 `train_dreamwaq.py` 实际构造了普通 runner。
- 任一创建 Isaac Sim `AppLauncher` 的入口没有先预加载 PyTorch/TensorDict，或独立子进程启动 smoke test 失败。
- observation、reward、action/target、raw/effective mean/std、value、log-probability、advantage、CENet 指数项、任一 loss 或总 loss 出现 NaN/Inf，或 effective std 不严格为正。
- backward 后、distributed reduction 后或 gradient clipping 后任一梯度/总 gradient norm 非有限，或 optimizer step 后任一模型参数/Adam state 非有限。
- `DreamWaQRolloutStorage.add_transitions()` 收到缺失、错误 shape/dtype/device 的 target/mask，或代码尝试用 current observation、零 target、全真 mask 静默替代。
- DreamWaQ storage 的两个附加字段没有预分配后使用 `copy_` 写入，保存了 inference tensor 引用，或 pending transition 后续变化能够修改已写入 rollout 的 target/mask。
- 受控 joint 没有外部有限软件限位、速度/力矩限制或 actuator 参数。
- `sim_dt * decimation != control_dt`，或 V4 的任一冻结值不等于 `sim_dt=0.005 s`、`decimation=4`、`control_dt=0.02 s`、solver position iterations `96`、solver velocity iterations `4`。
- checkpoint schema 与运行环境不一致。
- manifest 与实机 observation adapter 不一致。
- 固定 PyTorch/CUDA 栈无法在当前 GPU 上成功分配张量、执行真实 CUDA kernel 并同步。`torch.cuda.get_arch_list()` 只记录为诊断信息，不得因列表中没有与 capability 完全同名的条目而错误拒绝可执行的 RTX 4060。
- checkpoint 的 model/optimizer state、零基 iteration、`completed_iterations`、run-manifest hash、对应 current active contract、hardware profile、optimizer learning rate、`alg.learning_rate` 或 RNG 恢复信息缺失/不一致，或续训目标总轮数不大于已完成轮数。Phase 1R V6 还必须校验七个环境 RNG stream、当前 target cache 主绑定和 `ResumeMetadataV2` source/target provenance；DreamWaQ 还必须校验三个独立 contract、`DreamWaQCheckpointMetadataV1` 和专用 runner 类型，并与 Phase 1 checkpoint 双向拒绝。
- DreamWaQ checkpoint 缺少或无法严格恢复 `EstimatorMonitorStateV1`，`completed_rollouts != completed_iterations`，initial 被 resume 重置，recent window 超过 10 项、顺序错误或含非有限值。
- Phase 2 export 缺少 `actor.ts`、`policy_manifest.json` 或 `golden_vectors.pt`，任一 hash 不匹配，TorchScript 与 eager 最大绝对误差超过 `1e-7`，或 MuJoCo evaluator 直接依赖训练包内部类。
- deterministic Isaac report 的 seed、八场景顺序/命令、499 action step/9.98 s、nominal profile、base-task/evaluation source、共享 reset cache、baseline checkpoint identity、active-frame MSE 聚合、done/timeout 规则或 performance key 与 `IsaacEvaluationContractV1` 不一致，或实现试图改变现有 timeout/auto-reset 语义来执行 500 个 Isaac action step。
- 正式多 seed suite 的训练指纹不一致，正式 MuJoCo 报告混入 smoke/不同评估契约/重复 checkpoint，或训练监控门禁出现 hard anomaly。

禁止静默截断、补零或重排 checkpoint 输入。

## 22. 测试策略

### 22.1 不启动 Isaac Sim 的单元测试

- `AssetBundleV1` 五文件相对路径、大小、SHA256 和 canonical bundle manifest hash 测试。
- ActionV1、CommandV1、ActorObsV1 和 CriticObsV1 slice 与维度测试。
- NormalizationV1/V2 字段级 golden vector、clip、跨版本拒绝和“无第二套 normalizer”测试。
- CommandSamplingV2 分类概率、速度零分量、范围、reset-only 更新和 episode 内保持测试。
- VirtualLegKinematicsV1 左右真实 `phi0/L0`、wrapped angle difference 与 FK/几何交叉检查测试。
- `R_control_from_usd`、`R_control_from_imu`、四元数与 projected gravity 坐标/符号测试；必须包含 `-gravity_b/||gravity_b||` 正例和未经转换 `gravity_b` 的反例。
- policy joint order、USD name lookup、腿 protocol reorder、轮协议 `[left(ID4),right(ID5)]` 和 action 到物理目标的映射测试。
- 两轮“前进为正”到 USD `[+1,-1]` 目标映射和 USD 反馈到 canonical 同号语义的双向 golden test；Actor joint-velocity、Critic acceleration/torque 都必须使用同一反馈转换。
- upright、tilt 阈值边界、90 度侧倒和 180 度倒置的 orientation/termination 测试。
- history 使用唯一 frame/feature 编号的 `[N,5,25] -> [N,125]` frame-major golden vector，以及 reset 复制填充测试。
- history wrapper 连续两次 `get_observations()` 的 TensorDict 与 observation-noise RNG state 不变，且每个 reset/step 后最后一帧严格等于当前 `policy`。
- wrapper 构造与显式 reset 调用计数测试：一次底层 reset observation 必须被丢弃，随后恰好计算一次 `first_policy_observation`；resume trace 的 discarded/first 两帧顺序固定，首次 history 五帧都等于 first。
- pending-transition 快照测试：`act(obs_t)` 后让 wrapper 执行一次 step 并更新内部 cache，尚未写入 storage 的 `policy_t/policy_history_t/critic_t` 必须逐元素保持不变；storage 中保存的最后一帧必须仍是 `o_t` 而不是 `o_(t+1)`。
- ProprioReconstructionTargetV1 精确从 next clean `critic[0:25]` 提取 `[0:6] + [9:19]`，明确排除命令和 previous action；向 noisy policy 注入非零噪声时 target 必须保持不变。
- CENet encoder 精确拓扑 `125-128-64-35`、35D slice、decoder `25-64-128-16`、Actor `44-256-128-64-6`、Critic `41-256-128-64-1`、ELU 和参数量测试。
- 确定性 Actor 测试：相同参数与 `policy/policy_history` 在 rollout/update/eval/export 路径产生相同 action mean；只改变 `context_logvar` 且保持 velocity/mu 不变时，Actor mean 与 log probability 不变。
- action distribution 测试：raw mean/std 非有限时 fail fast；有限 mean 按 `[-20,20]` 裁剪，effective std 精确等于 `clamp(abs(raw_std),1e-4,2.0)`；rollout/update/inference/TorchScript 共享同一 effective mean。
- reparameterization 测试：显式相同 epsilon 得到相同 `context_z`，不同 epsilon 改变 decoder 路径；生产 storage 不含 epsilon；`context_z` 只用于 reconstruction，不进入 Actor。
- loss 数值测试：velocity/KL 公式、四个 reconstruction 分块、mask 分母和全无效 batch 的 graph-connected zero 都与本文一致。
- autograd 测试：actor loss 更新 Actor、共享 encoder、velocity 与 mu 路径；`context_logvar` 输出张量及 encoder 最后 Linear 的 logvar 输出行直接梯度为零，但共享隐藏层更新后允许后续 logvar 数值间接变化。reconstruction 更新共享 encoder、mu/logvar 和 decoder，但不得经 detached velocity/action 更新 velocity head 或 Actor；critic loss 只更新 Critic。
- RSL-RL 3.1.2 具名 Transition 写入、storage 取回和 mini-batch 字段往返测试；必须断言仅新增 `[T,N,16]` target 与 `[T,N,1]` mask，`None`/错误 shape/dtype/device 都失败，未知配置字段必须失败。测试必须在 `torch.inference_mode()` 中创建 transition 字段，验证写入的是预分配普通 tensor 的 `copy_` 结果，并在修改/清空源 transition 后确认 storage 内容不变。
- `obs_groups` 精确集合测试必须分别覆盖 config 构造前检查和 `DreamWaQActorCritic` 构造检查；缺 key、额外 set、`critic:[critic,policy_history]`、`policy:[policy,policy_history]` 都必须失败。
- estimator tripwire 公式测试必须覆盖 context initial/final threshold、10-iteration final window、velocity zero-baseline ratio 与 denominator floor；tripwire 只读日志/评估张量，不得进入 total loss 或改变 gradient。
- `EstimatorMonitorStateV1` fresh/save/load/resume 测试必须覆盖 initial 永不重置、recent window 截断为最近 10 项、`completed_rollouts == completed_iterations`，并证明中断续训与未中断输入序列得到相同 final-window 状态和结论。
- `IsaacEvaluationContractV1` 单元测试必须冻结 seed、八场景/命令、499 action step/9.98 s、nominal profile、baseline checkpoint SHA/contract、共享 cache 字段、pre-action sample-weighted velocity MSE、第 499 step 预期 timeout/提前 timeout/termination 处理、performance key 和 lexicographic baseline comparison；不得把场景 ratio 先平均，也不得修改任务 timeout 来伪造 500-step Isaac episode。
- done-mask dtype 回归测试必须直接使用 `torch.long([0,1])`，断言 mask 为 `bool([True,False])`；另分别制造 terminated 和 truncated，二者都不得进入 reconstruction loss。
- runner 注册测试：未注册自定义类时失败，显式幂等注册后 `OnPolicyRunner` 解析到本项目类。
- runner 身份测试：config `class_name`、algorithm contract `runner_class`、metadata `runner_class` 和实际 Python 类型必须同时等于 `DreamWaQContractOnPolicyRunner`。
- `DreamWaQActorCritic.load_state_dict(strict=True)` 必须返回 `True`；runner load 后 optimizer、iteration 和 adaptive learning rate 均恢复。Phase 1 与 DreamWaQ checkpoint 必须双向拒绝，三个 contract 任一 hash 改变也必须拒绝。
- checkpoint provenance 测试：metadata 必须保存 seed；fresh provenance 只允许两个固定字段，resume provenance 必须完整包含规定字段；缺失、多余、mode 不匹配或 requested seed 不一致都必须拒绝。
- `DreamWaQInferenceActorV1` 输入 `[N,125]`、输出 `[N,6]`，与训练模型 `act_inference` action mean 的 golden vector 一致；导出图中不得出现 Critic、decoder、Gaussian sampling 或经验 normalizer。
- `DreamWaQInferenceActorV1` 的 rank、`float32` dtype 和动态 batch 测试；错误 rank/dimension 必须失败，`N=1` 与 `N>1` 都必须与训练侧逐元素对齐。导出目录必须精确包含三件套，reloaded `actor.ts` 对 `golden_vectors.pt` 的误差不超过 `1e-7`。
- 数值门禁测试：分别注入非有限 output、loss、gradient、optimizer parameter/state，断言在 backward 前、optimizer step 前或 step 后的对应边界立即失败并保留诊断。
- schema/manifest 不兼容拒绝测试，并覆盖 `clip_actions`、finite-horizon、gravity、root 初速度、完整 solver/rigid/articulation 配置和 PPO `gamma` 等行为漂移。
- current active contract 固定数值测试必须直接断言 `sim_dt=0.005`、`decimation=4`、`control_dt=0.02`、solver position iterations `96`、solver velocity iterations `4`；只检查乘积不算通过。任一字段改变都必须改变 contract hash 并拒绝旧 checkpoint。Phase 1R 还必须覆盖 randomization profile、stream lifecycle、cache schema/算法和 root-height alignment。
- checkpoint 中 model、optimizer、seed、零基 iteration、`completed_iterations`、optimizer learning rate、`alg.learning_rate`、RNG state、hardware profile 和 batch 元数据完整性测试。
- Episode 日志必须按 rollout 内全部 `extras["log"]` 的 key union 聚合，不能因第一个 step 尚无 reset 而丢失后续 `Episode/*` 指标。
- 训练指纹必须允许 seed/run-name/log-dir、realized audit 和 per-seed cache hash 不同，但拒绝 current active contract、源码/依赖、hardware profile 或 batch 语义变化；原始配置文件 hash 继续保留审计价值。
- 训练监控聚合必须让最终正常快照覆盖启动期 event/tag 可用性缺失，同时保留历史数值异常；post-run 必需 tag 缺失属于阻塞异常。

### 22.2 Isaac Lab 集成测试

- `AssetBundleV1/V2` 各自的五层 composition、root prim 路径清单、default prim、articulation root、关节、刚体、限位和 actuator 审计；V1/V2 机器人语义除删除内嵌地面外必须一致。
- 通过 `UsdFileCfg` default prim reference 加载后，机器人命名空间下不存在 PhysicsScene，任务 stage 只有一个环境 PhysicsScene。
- V3/V1 在 physics initialization 前禁用内嵌 GroundPlane；V4/V2 的组合 stage 中该 prim 必须不存在；两者场景都只能存在任务创建的唯一环境平地。
- `robot.data.root_com_lin_vel_b` 与 world-frame COM 速度经姿态逆旋转后的结果一致，再经 `R_control_from_usd` 得到 ControlFrameV1。
- 四个闭环约束存在且生效；静止和随机动作时闭环误差与 solver 状态在阈值内。当前标准 batched API 不提供逐约束 wrench，因此记录 `constraint_wrench_available=false`；只有未来接入经过验证的 PhysX 查询时才增加内部约束力/力矩诊断。
- V4 环境启动时直接断言 `sim_dt=0.005`、`decimation=4`、`control_dt=0.02` 和 articulation solver `96/4`，防止退回同为 50 Hz 的历史 `0.01/2 + 32/4` 配置。
- dummy mass 保留结论、源 `jMK/jEC` collision 不对称，以及 V3 `WheelOnlyCollisionV1` / V4 `WheelOnlyCollisionV2` 只保留左右轮碰撞的回归测试。
- 零动作站立 smoke test。
- 随机动作 1000 step，无 NaN、越界和 reset 异常。
- 多环境 reset 隔离测试。
- ResetObservationCacheCoherenceV2：非零旧角速度后重复 reset、direct PhysX 角速度与 COM/height 对照、clean/noisy 观测区分、生成/加载 cache、partial/auto-reset、五帧 history、无额外物理步与 scene-data timestamp 漂移，以及 sim2sim 两端 reset 回归；见第 18.1 节。
- 奖励逐项有限值与时间尺度测试。
- reconstruction target done mask 测试。
- `terminated`、`truncated`、多环境异步 reset 下的 history、当前 action 和 16D next physical target 时间索引测试。
- DreamWaQ wrapper observation groups 必须精确为 `policy [N,25]`、`policy_history [N,125]`、`critic [N,41]`；环境本体在无 wrapper 时仍只产生 `policy/critic`。
- RTX 5070 profile 的 PPO 短训练启动测试；分别覆盖同 `num_envs` 跨 hardware profile 精确复用 reset cache，以及 `256 -> 32` 跨规模生成 target cache 后继续到明确总轮数。重复的相同目标恢复必须比较七流来源、command、root velocity、discarded reset observation 和 first policy observation。
- DreamWaQ 短训练启动、保存、重新加载、确定性 play 和 resume 测试；checkpoint 必须恢复 Actor/Critic/CENet/optimizer/adaptive learning rate/RNG，并拒绝普通 PPO checkpoint 作为完整 DreamWaQ resume 源。
- `evaluate_isaac.py` 必须在全新子进程用同一份 8-env evaluation reset cache 依次产生 Phase1RIsaacBaselineV1 和 DreamWaQ smoke report，验证命令在 first observation 前生效、八条轨迹的 done/timeout 隔离、history runtime、estimator MSE、performance key、contract/source/cache hash 与 report schema。
- Windows 下分别以全新子进程启动 `train_dreamwaq.py` 和 `play_dreamwaq.py` smoke test，证明 PyTorch/TensorDict 在 `AppLauncher` 前加载；同一进程复用不算通过。

### 22.3 GPU 兼容测试

RTX 4060 与 RTX 5070 都必须通过：

- 真实 CUDA kernel 分配、计算、有限值和同步检查；capability 与 arch list 只作为记录。
- `portable` profile 随机动作 smoke test。
- PPO/DreamWaQ checkpoint 跨 GPU 加载、确定性评估和继续训练 smoke test；不要求后续训练轨迹逐位一致。
- 相同输入下 Actor/CENet 输出容差测试。

### 22.4 Sim-to-real 一致性测试

- 固定 golden observation vectors。
- IMU 静止六面、单轴正转、gyro bias 和四元数重力方向 conformance test。
- Python 训练侧和目标部署 backend 的坐标变换、关节重排及归一化结果对齐。
- 固定五帧历史下 CENet/Actor 输出对齐。
- Python history wrapper、MuJoCo runner 和部署 adapter 的 reset 五份复制、逐 tick append 与 frame-major flatten golden vector 对齐。
- 关节顺序、左右轮 CAN ID、符号、目标重建和动作 scale 对齐。
- policy-to-packet 与 packet-to-policy 往返映射测试。
- 在实际部署数值精度下验证相同输入的模型输出和 50 Hz worst-case latency。

## 23. 日志和实验记录

训练 step 内禁止逐步磁盘写入。允许：

- RSL-RL/TensorBoard 聚合指标。
- 每个 iteration 的低频统计。
- 独立 play/evaluate 脚本中的有限长度 trace。
- 明确开启时的低频视频录制。

必须记录：

- 每个奖励项的 episode sum。
- termination 原因计数。
- 命令与实际速度、高度误差。
- action saturation、关节限位和 actuator saturation 比例。
- 闭环误差和 solver 异常计数；内部约束力/力矩仅在经过验证的查询接口可用时记录，同时始终记录 `constraint_wrench_available`。
- PPO loss、entropy 和 value loss。
- DreamWaQ velocity、KL、reconstruction、PPO 与加权 total loss，以及角速度、projected gravity、腿位置和关节速度四个重建分块的 MSE。
- `context_mu_feature_std_mean_initial`、逐 iteration `context_mu_feature_std_mean`、最后 10 iteration 平均值和对应 collapse threshold；deterministic Isaac evaluation 的 `velocity_eval_mse`、`velocity_zero_baseline_mse` 与比值。
- `EstimatorMonitorStateV1` 的 completed rollout 数、recent-window 内容与 resume provenance；`IsaacEvaluationContractV1`/source/cache/baseline identity、逐场景 reward/survival、aggregate performance key 和 candidate-vs-baseline acceptance。
- Actor、Critic、CENet encoder/context/velocity head 和 decoder 的 gradient norm。
- auxiliary-loss gradient norm、PPO gradient norm 及两者比值；该比值第一版只用于诊断，不作为自动调 coefficient 的反馈。
- raw/effective action mean 与 standard deviation、`context_mu/logvar` 范围、CENet 指数项最大值、pre/post-clip gradient norm、optimizer-step 后参数/Adam state 以及各模块输出/总 loss 的有限值状态。
- Python/Isaac Sim/Isaac Lab commit/PyTorch/CUDA/RSL-RL 版本与 dependency manifest hash。
- master/派生 seed、RNG state、hardware profile、`num_envs`、rollout length 和 mini-batch 划分。
- 当前 AssetBundle、schema、normalization 和 manifest hash。

## 24. 分阶段交付

### Phase 0：环境与资产审计

交付：

- 从空 Python 3.11 环境执行成功的 `bootstrap.ps1`、`uv.lock` 和 dependency manifest。
- 固定版本环境可在 RTX 5070 运行。
- `portable` profile 可启动。
- `AssetBundleV1` 五文件哈希/大小、层依赖清单、入口层 root prim 路径、default prim、articulation root、单位审计与不可变加载证明。
- `UsdFileCfg` default prim reference 加载证明：环境只有一个 PhysicsScene；加载覆盖在 physics initialization 前禁用内嵌 GroundPlane，且不修改源 USD。
- 27 rigid body、26 tree joint、4 loop-closure joint 的名称和启用状态审计。
- 闭环锚点残差 `<=5 mm`、passive joint、dummy mass 保留、源 collision 不对称、`WheelOnlyCollisionV1` 和 solver 稳定性结论。
- `R_control_from_usd`、canonical joint order、腿 protocol reorder、轮协议 `[left(ID4),right(ID5)]` 和 wheel sign 的 golden vector。
- `q_nominal`、joint sign/zero、有限软件限位、actuator、动作 scale 和 Critic privileged scale 的验证结果。
- IMU 正向安装假设 `R_control_from_imu=I`、gyro bias 标定责任和 `-gravity_b/||gravity_b||` 换算的静止/单轴 conformance 测试定义。

退出条件：资产与覆盖断言全部通过；场景只有一块有效平地；闭环误差与 solver 指标在冻结阈值内；canonical frame/joint/action 映射 golden vector 通过；随机动作 1000 step 无 NaN、关节越界、约束爆炸和 reset 错误。

### Phase 1：普通 PPO 平地基线

交付：

- ActorObsV1 25D。
- CriticObsV1 41D。
- ActionV1 6D。
- CommandV1 3D。
- ControlFrameV1 与 NormalizationV2。
- CommandSamplingV2、平地奖励、`phi0_symmetry`、reset 和 termination。
- PPO train/play/checkpoint 流程。
- AssetBundleV2、MuJoCo 12 个可动 dummy 惯性分支、WheelOnlyCollisionV2、COM/ControlFrame adapter 和统一 headless 评估工件。

退出条件：NormalizationV2、CommandSamplingV2、VirtualLegKinematicsV1 与 checkpoint/manifest 一致性测试通过，并达到正式冻结的平衡、速度、yaw、高度和左右 `phi0` 对称指标。

V4 首轮验证固定为四次互相独立、从头开始的 `1000 iteration` PPO 训练。四次运行除随机 seed 和派生 RNG 状态外使用同一个 ContractV4；不得从 V3 checkpoint warm-start，也不得在四次运行之间 resume。完成后使用同一 MuJoCo headless 评估协议比较 base height、base roll/pitch/yaw、左右 `phi0` 差和速度跟踪。迭代数达到 1000 不自动视为 Phase 1 通过。初始计划曾把 G-08 设为进入 Phase 1R 的前置性能门；实际工程随后把 Phase 1R 作为受控诊断阶段执行，并由 G-15 的窄实施许可允许继续 Phase 2。该历史时序偏离不关闭 G-08，也不把任何 PPO 结果改写为性能通过。

### Phase 1R：复旦风格保守域随机化

交付：

- `FudanStyleDomainRandomizationV1`，只包含轮地摩擦、四腿 default offset、六执行器 Kp/Kd/effort scale、episode reset root velocity 和 Actor observation 白噪声。
- `Phase1RandomizedContractV3`、`CheckpointMetadataV6`、`ClosedChainResetCacheSchemaV2` 与 `ClosedChainRootHeightAlignmentV1`。
- Actor noisy / Critic clean 数据路径、per-env actuator/material 写入、partial-reset transition 隔离和七个具名 RNG stream。
- 同规模精确复用 source cache、跨规模生成 target cache 的 `ResumeMetadataV2` 与 `resume_sequence_trace.pt`。
- 四次不同 seed、从头开始、每次 1000 iteration 的普通 PPO，以及四个 checkpoint 的统一 MuJoCo headless sim2sim。

Phase 1R 的“性能通过”仍要求随机化 runtime gate、checkpoint/cache/resume 集成测试和代码复核无 P0/P1，四次训练无 NaN/Inf、invalid termination、约束爆炸或持续饱和异常，并由 MuJoCo 报告确认闭环指标。当前四组 MuJoCo 失败结果必须保留为负面对照，不能改写为 Phase 1R 已通过。

Phase 2 的“允许实施”使用较窄门槛：Phase 1R 环境、随机化、checkpoint 和统一 sim2sim 诊断链已经可运行，四个 checkpoint 已按同一协议评估且失败现象被记录，即可开始 DreamWaQ 代码与受控实验。该许可不等于宣称 PPO sim2sim 已解决；DreamWaQ 必须使用同一个任务、随机化、物理与评估契约对比，若仍失败，必须继续区分 estimator 能力、策略鲁棒性和仿真映射问题。

### Phase 2：DreamWaQ

交付：

- 五帧历史 wrapper。
- CENet `125-128-64-35` encoder、deterministic velocity/context-mu 策略路径和 `25-64-128-16` decoder。
- DreamWaQ ActorCritic、PPO 和 rollout storage。
- frame-major HistoryLayoutV1、速度监督、动作条件的 16D 下一帧物理重建和 KL loss。
- 只新增 next-proprio target/mask 的 RSL-RL 3.1.2 Transition/storage/runner 迁移契约与显式 class registration。
- done transition 重建 mask。
- 无状态 `[N,125] -> [N,6]` deterministic export actor，以及 Isaac/MuJoCo/部署三侧一致的外部 history runtime。
- 完整训练 checkpoint 与只含推理路径的部署 manifest。

退出条件：全部 DreamWaQ 单元/集成/导出测试通过；短训练无 NaN/Inf 且 velocity、KL、reconstruction loss 有界；deterministic Actor 与梯度隔离契约成立；最后 10 iteration 的 `context_mu_feature_std_mean` 满足 `>= max(1.0e-3, 0.10 * initial)`；`IsaacEvaluationContractV1` 下 velocity MSE / zero-predictor MSE 满足 `<= 0.80`，且 candidate performance key 不差于 `Phase1RIsaacBaselineV1`；导出模型、Isaac play 与 MuJoCo runner 的固定 history 输出对齐。acceptance failure 不得标记 Phase 2 通过，但有效 checkpoint 仍必须按相同协议完成 MuJoCo 并保留结果；不能通过跳过失败 run 或修改 MuJoCo 参数制造有利结论。

### Phase 3：鲁棒性扩展

交付顺序：

1. 在 Phase 1R 已有随机化基础上加入 action/通信延迟、控制抖动、episode 内恒定 gyro bias 和具名外力扰动。
2. 重新核对实车名义参数后加入质量、质心、惯量和恢复系数随机化，并按需要扩展摩擦/执行器范围。
3. 按实际启用字段扩展 Critic schema。
4. 命令课程。
5. 复杂地形与 terrain height privileged state。

### Phase 4：实机部署

交付：

- 部署模型和 manifest。
- 选定平台的 observation/protocol adapter。
- 上电 gyro bias 标定、projected gravity 约定和 IMU conformance 实现。
- 50 Hz 推理循环。
- 启动保护、动作限幅和失联回退。
- sim-to-real golden vector 和离线 trace 对齐。

## 25. 已确认决策

| ID | 决策 |
|---|---|
| ADR-001 | 使用独立轻量外部 Isaac Lab 工程，不复制旧 WheelLeg 工程作为外壳。 |
| ADR-002 | 使用 Isaac Sim 5.1.0、Isaac Lab 官方 Git tag `v2.3.2` commit `37ddf626871758333d6ed89cf64ad702aef127d0`、Python 3.11、PyTorch 2.7.0 cu128、RSL-RL 3.1.2；PyPI `2.3.2.post1` 仅作兼容性参考。 |
| ADR-003 | 4060 和 5070 共用代码与模型，只使用不同 hardware profile。 |
| ADR-004 | V1 使用六维纯关节空间动作，四腿关节位置目标加两轮速度目标。 |
| ADR-005 | V4 固定 `sim_dt=0.005s`、`decimation=4`、`control_dt=0.02s`；物理子步调整不改变已确认的 50 Hz 控制周期。 |
| ADR-006 | V1 只训练平地平衡、`vx`、`yaw_rate` 和高度命令，不训练跳跃。 |
| ADR-007 | ActorObsV1 为 25D，不含真实线速度、真实高度和欧拉角。 |
| ADR-008 | CriticObsV1 为 41D，后续按有效 privileged 字段扩展，不预填 141D。 |
| ADR-009 | 普通 PPO 后先建立 Phase 1R 随机化与统一 sim2sim 诊断链；Phase 1R 性能未通过不会被 DreamWaQ 隐藏，但完成负面对照后允许实施 Phase 2。 |
| ADR-010 | DreamWaQ 使用 frame-major 五帧 125D 历史、3D 速度估计、16D `context_mu` 和 44D Actor 输入；Actor 不接收 `context_logvar/context_z`，禁止使用 Isaac Lab 内置 term-major history。 |
| ADR-011 | 第一版 DreamWaQ 使用当前动作作为条件，重建下一帧 16D ProprioReconstructionTargetV1，并包含 KL loss；不重建命令/previous action，不包含 AdaBoot。 |
| ADR-012 | 实机不计算真实线速度、真实机身高度或绝对 yaw 作为策略输入。 |
| ADR-013 | V3 历史合约的唯一正式资产为 `AssetBundleV1`；五个文件均钉路径/大小/SHA256，源文件不可修改或 flatten。V4 新训练使用由 V1 复制并只删除内嵌地面的独立 `AssetBundleV2`。 |
| ADR-014 | 训练和实机统一使用 ControlFrameV1；仿真三维向量固定经过 `R_control_from_usd`，IMU 预期 `R_control_from_imu=I` 并必须实测。 |
| ADR-015 | ActionV1 canonical joint order 固定为 `[jIJ,jIO,jAB,jAG,jwheel_left,jwheel_right]`；腿协议重排、轮协议 `[left(ID4),right(ID5)]` 和仿真/硬件符号由 adapter 显式处理。 |
| ADR-016 | Actor、Critic 与 CENet 使用合约指定的字段级固定 normalization；V3 使用 NormalizationV1，V4 使用 NormalizationV2。仅使用 RSL-RL 原生 Actor/Critic normalization 开关，项目不得实现 velocity running normalizer。 |
| ADR-017 | 部署计算平台暂不冻结；H7、RK3566 等后端不得改变模型 schema、坐标、归一化和动作语义。 |
| ADR-018 | 当前 USD 的真实闭环约束优先，passive joint 不复制旧工程高刚度 PD，必须通过 Phase 0 稳定性门槛。 |
| ADR-019 | 第一版 DreamWaQ 用一个 Adam optimizer 联合更新 Actor、Critic 与 CENet；Actor loss 只经 velocity/mu 路径进入 encoder，reconstruction 对 action 和 estimated velocity 停止梯度并经 sampled z 更新 mu/logvar/decoder，Critic 梯度保持独立。 |
| ADR-020 | Windows 环境采用 uv + Isaac Sim 5.1.0 pip package family + 外部固定 revision 的 Isaac Lab `v2.3.2` editable checkout + RSL-RL 3.1.2；A1 补丁进入候选兼容性表，只有对应 smoke test 复现后才能启用。 |
| ADR-021 | 机器人通过 `UsdFileCfg` default prim reference 加载；资产根部平级 PhysicsScene 不进入机器人命名空间，运行时必须只有一个环境 PhysicsScene。 |
| ADR-022 | 实机角速度必须执行 bias 补偿；策略 projected gravity 是单位向下向量，固件 `gravity_b` 只能经 `-gravity_b/||gravity_b||` 转换后使用。 |
| ADR-023 | 训练记录 master/派生 seed、RNG state、硬件 profile 和 batch 布局；跨 GPU 可加载和续训，但不承诺不同并行布局逐位复现。 |
| ADR-024 | DreamWaQ Actor 在训练、评估和部署中始终使用 deterministic velocity 与 `context_mu`；sampled `context_z` 只用于 decoder 辅助 loss，storage 不保存 `context_epsilon`。 |
| ADR-025 | `jAG/jAB` 的前后文字标注存在一手代码冲突；所有机器接口以 joint name 与 CAN ID 为权威，物理角色必须通过 G-11 低力矩单电机标定冻结。 |
| ADR-026 | V1 保留 USD 的全部质量，包括 12 个 dummy body 的约 `0.12 kg`，不修改或重导出源资产。 |
| ADR-027 | V3 使用 `WheelOnlyCollisionV1`，V4 使用 `WheelOnlyCollisionV2`；二者都只保留左右轮机器人碰撞，V2 不再包含禁用资产内地面的步骤。两版都不启用接触传感器、collision reward 或 collision termination。 |
| ADR-028 | 每个 PPO checkpoint 保存 model、optimizer、optimizer learning rate、`alg.learning_rate`、零基 iteration、`completed_iterations`、累计 timestep/time、seed、Python/NumPy/PyTorch/CUDA RNG state、hardware profile、run-manifest hash 和完整 contract hash；Phase 1R 的 V6 metadata 还保存七个环境 RNG stream 与当前 reset-cache 主绑定。play 和 resume 在构造训练状态前 fail-fast 校验。 |
| ADR-029 | Windows 固定 `tensordict==0.14.2`；所有创建 `AppLauncher` 的 PPO/DreamWaQ 训练和播放入口都在 Kit 前预加载 PyTorch/TensorDict，并由全新子进程 smoke test 覆盖。 |
| ADR-030 | USD 与 canonical 轮反馈使用同一 `[+1,-1]` involution；目标、位置、速度、加速度和 applied torque 都必须显式转换。 |
| ADR-031 | orientation 与 tilt termination 基于 `upright_cos=-projected_gravity_z`，必须惩罚并终止完全倒置状态。 |
| ADR-032 | 4060/5070 运行兼容性由真实 CUDA kernel probe 判断，不要求 capability 字符串与 `torch.cuda.get_arch_list()` 精确匹配。 |
| ADR-033 | PPO 续训创建新 run，`--max-iterations` 表示总目标轮数；恢复模型、optimizer、RSL-RL adaptive `alg.learning_rate` 与 RNG，但不伪称恢复中断瞬间的环境状态。 |
| ADR-034 | `Phase1ContractV3` 必须覆盖机器人 spawner callable 和完整地面生成配置；环境从该配置实例化地面，禁止 contract 与运行时各维护一份硬编码地面语义。 |
| ADR-035 | 新训练升级为 `Phase1ContractV4`，固定 AssetBundleV2、WheelOnlyCollisionV2、CommandSamplingV2、NormalizationV2、RewardSchemaV2 与 VirtualLegKinematicsV1；V3 只用于旧 checkpoint。 |
| ADR-036 | CommandSamplingV2 在 reset 时按 `20%/30%/20%/30%` 选择站立/直行/转向/组合模式，`vx` 范围 `[-1.5,1.5] m/s`、yaw 范围 `[-1,1] rad/s`，命令在完整 episode 内保持不变。 |
| ADR-037 | V4 新增真实虚拟腿角 wrapped-difference 平方奖励，固定权重 `-1.0`；该项不进入 Actor/Critic observation。 |
| ADR-038 | Phase 1R 用四个不同 seed 从头训练四次 1000 iterations，四次只允许 seed/RNG 不同；最终由统一 MuJoCo headless 指标选出最佳 checkpoint。 |
| ADR-039 | MuJoCo sim2sim 的机构源固定为本地 `wheel_leg_urdf4_self_mesh_all.xml`，使用其 freejoint、闭链和六执行器拓扑，并替换零字节资源为本地完整 mesh。 |
| ADR-040 | MuJoCo 动力学以 AssetBundleV2/USD 的质量和惯量为权威；源 XML 缺失的 12 个可动 dummy 必须作为无碰撞 passive body/revolute joint 分支显式生成，禁止名义姿态惯量合并；USD composed root pose 必须转换为 MuJoCo parent-local pose，逐 body 和总质量/惯量审计失败时禁止 sim2sim。 |
| ADR-041 | MuJoCo 使用 WheelOnlyCollisionV2：仅左右轮 proxy 与地面形成显式接触，滑动摩擦 `1.0`；visual、base 和非轮 geom 禁止碰撞。 |
| ADR-042 | MuJoCo 固定 `R_control_from_mujoco=[[0,-1,0],[1,0,0],[0,0,1]]`；根高度与线速度基于 base COM，freejoint 原点不得替代 COM。 |
| ADR-043 | 四个 actor 的每个 headless 场景从同一完整 keyframe 独立 reset；指标按适用场景等权聚合，失败 run 先按完成场景和 survival 排名，若全部失败则 Phase 1R 不通过。 |
| ADR-044 | USD authored `q=0` 只定义 MuJoCo 模型参考构型；headless/viewer reset 必须写入 Phase1ContractV4 冻结的 base freejoint 与全部 26 个 hinge 初态，并断言编译维度 `nbody=28/njnt=27/nq=33/nv=32/nu=6/neq=8/npair=2`。 |
| ADR-045 | 正式四 run suite 使用种子无关训练指纹、`MujocoDynamicsSemanticsV1`、`MujocoEvaluationContractV1` 和训练监控门禁；不同训练/评估语义或任一 hard anomaly 均禁止产生正式 selected 模型。 |
| ADR-046 | Phase 1R 同 `num_envs` resume 精确复用 source reset cache；跨 `num_envs` resume 先验证 source，再生成并绑定 target cache，target checkpoint 不得继续绑定 source cache。 |
| ADR-047 | Phase 1R 同规模 resume 恢复七个环境 RNG stream；跨规模只恢复 source 的三个 runtime streams，四个 process-start streams 保留 target 状态，并用 `ResumeMetadataV2` 记录来源。 |
| ADR-048 | DreamWaQAlgorithmContractV1 冻结 encoder `125-128-64-35`、Actor `44-256-128-64-6`、Critic `41-256-128-64-1`、decoder `25-64-128-16`、ELU、三项 auxiliary coefficient `1.0` 与单 Adam optimizer。 |
| ADR-049 | DreamWaQ storage 只在标准 PPO 字段外增加 16D next-proprio target 和 1D reconstruction mask；Phase 2 reference export 固定用 `torch.jit.script` 生成 TorchScript `actor.ts`、`policy_manifest.json`、`golden_vectors.pt` 三件套和无状态 `history[125] -> action_mean[6]`，history 由 runtime 管理。 |
| ADR-050 | Phase 1R DreamWaQ 的 encoder/Actor 输入使用 noisy policy/history；velocity 和 next-proprio 监督从 clean CriticObsV1 提取，decoder 不学习不可预测的 observation white noise。 |
| ADR-051 | DreamWaQ wrapper 返回的 TensorDict 是不可被后续 step 原地修改的快照；`DreamWaQPPO.act` 仍显式深拷贝 pending observation，防止 RSL-RL 在 env.step 之后才写 storage 时发生 history 错一帧。 |
| ADR-052 | RSL-RL 的 combined `dones` 是 `torch.long`；reconstruction mask 固定使用 `dones.eq(0).view(-1,1)`，禁止直接对整型 `dones` 使用按位取反。 |
| ADR-053 | DreamWaQ wrapper 构造与显式 reset 都保留上游“一次 discarded observation + 恰好一次 first policy observation”时序；首次五帧历史复制 first observation，重复读取只返回缓存。 |
| ADR-054 | Phase 2 config、algorithm contract、metadata 和实际 Python 类型都固定为 `DreamWaQContractOnPolicyRunner`，并使用 `DreamWaQCheckpointMetadataV1`；run manifest 分离 base task、algorithm、export 三个 contract，Phase 1 与 DreamWaQ checkpoint 双向拒绝。 |
| ADR-055 | Actor distribution 在 raw mean/std finite 后使用 A1 对齐的 effective mean `[-20,20]` 与 effective std `clamp(abs(raw_std),1e-4,2.0)`；禁止 `nan_to_num` 静默修复。 |
| ADR-056 | DreamWaQ storage 缺失或错误 target/mask 立即失败，不允许退化为 current observation、零 target 或全真 mask；ActorCritic strict load 必须返回 `True`。 |
| ADR-057 | 每个 DreamWaQ mini-batch 在 forward、backward、distributed reduction、gradient clip 和 optimizer step 后执行有限值门禁。 |
| ADR-058 | DreamWaQ checkpoint 必须保存 seed；`ResumeMetadataV2` fresh mode 只有 schema/mode 两个字段，resume mode 使用完整 source/target provenance，二者严格互斥。 |
| ADR-059 | `golden_vectors.pt` 固定使用 CPU `torch.randn`、seed `20261007`、`float32 [32,125]` 生成，并以 eager/TorchScript `1e-7` 误差阈值验证。 |
| ADR-060 | DreamWaQ storage 的 16D target 与 1D mask 必须预分配并用 `copy_` 写入，禁止引用保存 rollout inference tensor；源 transition 后续复用不得改变已写入数据。 |
| ADR-061 | DreamWaQ 的 RSL-RL `obs_groups` 必须在 runner/config 与 ActorCritic 两端精确等于 `policy:[policy]`、`critic:[critic]`；`policy_history` 只走自定义显式字段。 |
| ADR-062 | Phase 2 estimator 退出门槛固定为 context final feature-std `>= max(1.0e-3,0.10*initial)` 与 deterministic velocity MSE / zero-predictor MSE `<=0.80`；它们只用于验收，不改变 loss 或梯度。 |
| ADR-063 | Phase 2 deterministic Isaac evaluation 固定为 `IsaacEvaluationContractV1`：seed `20261007`、8 env/八场景、NominalEvaluationProfileV1、499 action step/9.98 s、共享 reset cache、pre-action sample-weighted estimator MSE 和固定 Phase1RIsaacBaselineV1 performance key；MuJoCo 独立保持 500 tick/10.0 s。 |
| ADR-064 | context tripwire 的 initial、completed rollouts 和最近 10 项以 `EstimatorMonitorStateV1` 写入 checkpoint，并在 resume 后于下一 rollout 前精确恢复。 |
| ADR-065 | checkpoint/contract/finite/export/evaluation 协议错误是 hard failure；context/velocity/baseline 性能不达标是 acceptance failure。后者禁止标记通过，但有效工件仍必须导出并完成 MuJoCo 留证。 |
| ADR-066 | G-08 保留为未关闭的历史 PPO 性能门，但不再作为已经发生的 Phase 1R/Phase 2 诊断实施时序门；Phase 2 的窄实施许可由 G-15 表达，二者不得混称。 |
| ADR-067 | ResetObservationCacheCoherenceV2 根据真实 red probe，在正式 `_reset_idx()` 读取新状态前只失效 Isaac Lab 2.3.2 `_root_link_vel_w`、`_root_com_pose_w`；保留读取语义、初态采样和快照合并，不推进任何时钟。任何其他 split 仍须停止定位；独立代码复核、运行验收和 RootCauseSuite 正式门分别记录。 |
| ADR-068 | 历史 v0.24：保留 PhysicsV4 的 Isaac 刚体角速度上限 `100 deg/s`，MuJoCo 用 PhysxRigidAngularBiasV1 补充逐刚体世界系偏置力矩；每 `5 ms` 计算并保持五个 `1 ms` 步，不截断 qvel。评估采用 MujocoEvaluationContractV2。当前实现由 ADR-069 / §30 取代，历史工件独立保留。 |
| ADR-069 | v0.25：按用户明确授权解除刚体线/角速度和全部关节转速的人工限速作用；Isaac 显式使用有限 float32 最大值哨兵覆盖默认上限，MuJoCo 删除外加角速度补偿力矩和超速清零电机力矩分支，禁止用速度截断代替。当前 PhysicsV5 与 UnrestrictedVelocityPolicyV1 进入任务/export/reset-cache 身份；MujocoActionAdapterV2 与 MujocoEvaluationContractV3 区分新评估。保留 PD、effort、接触/摩擦、闭环、被动阻尼、armature、termination 和策略数学契约；旧 PhysicsV4 工件禁止重标或直接续训。 |

## 26. 决策门与当前状态

以下项目不阻塞架构评审，但必须在对应阶段前通过单独设计确认：

| Gate | 必须确认的内容 | 最晚时间 | 当前状态 |
|---|---|---|---|
| G-00 | `AssetBundleV1` composition、default-prim 加载/唯一 PhysicsScene、dummy mass、源 collision 不对称、runtime collision policy、闭环误差阈值与 passive damping | Phase 0 退出前 | **已关闭**，见 §4.1/§4.1.1 与 Phase 0 审计工件 |
| G-01 | `q_nominal` 来源、数值、左右对称关系、仿真 joint sign 和有限软件关节限位 | Phase 0 控制实现前 | **已关闭**，见 §4.1.1 与 ActionV1 |
| G-02 | 腿位置/轮速 action scale、actuator 参数、速度/力矩限制、Critic 加速度/力矩 scale 与 clip | Phase 0 随机动作测试前 | **已关闭**，全部进入 Phase1 contract hash |
| G-03 | 普通 PPO Actor 和 Critic hidden dimensions | Phase 1 正式训练前 | **已关闭**：Actor/Critic 均为 `[256,128,64]` ELU |
| G-04 | `vx`、`yaw_rate`、`base_height` 命令范围、采样分布、固定 normalization scale/clip、`nominal_base_height` 和 `height_command_span` | V4 PPO 前 | **已关闭**：见 CommandSamplingV2/NormalizationV2；全部进入 ContractV4 |
| G-05 | V4 reward 权重及虚拟腿几何对称奖励 | V4 PPO 前 | **已关闭**：`phi0_symmetry` 使用 wrapped angle difference 平方、权重 `-1.0`，其余权重沿用基线并进入 ContractV4 |
| G-06 | 接触传感器、collision reward 和碰撞 termination | Phase 1 PPO 前 | **已关闭**：三者均不启用；V3 使用 WheelOnlyCollisionV1，V4 使用 WheelOnlyCollisionV2 |
| G-07 | reset、termination、episode length | Phase 1 PPO 前 | **已关闭**：`10 s` episode、COM 高度/tilt/速度有限阈值与 timeout 已冻结 |
| G-08 | 名义 PPO/Phase 1R 的性能量化门槛 | Phase 1 性能结论前 | **未关闭（历史性能门）**：四次 V4 与四次 Phase 1R 的统一 sim2sim 均未形成完整通过结论；该 gate 不再作为已发生诊断阶段的时序阻塞，Phase 2 实施许可由 G-15 单独表达 |
| G-09 | DreamWaQ Actor、CENet hidden dimensions 和部署模型边界 | Phase 2 实现前 | **已关闭**：Actor `[256,128,64]`、encoder `[128,64]`、decoder `[64,128]`、Critic `[256,128,64]`；部署为 `79,721` 参数的 encoder+Actor mean，资源测试不得静默改结构 |
| G-10 | 最终部署平台、数值精度和 TorchScript 之外的附加部署后端 | Phase 4 前 | 未开始；Phase 2 reference/sim2sim 格式已经冻结为 TorchScript 三件套 |
| G-11 | 实机 motor sign、zero offset、协议方向、gyro bias 标定参数、IMU/projected-gravity conformance 与安全状态机阈值 | 首次带载实机推理前 | 未开始 |
| G-12 | Phase 1 实现经独立只读代码复核且无阻塞 P0/P1 | 首个 2000 iteration run 前 | **已关闭**：第三轮独立只读复核结论为 A，无阻塞 P0/P1；允许启动首个 2000 iteration run |
| G-13 | 区分站立附近高度跟踪与自主起立，冻结低姿态初态分布、起立命令过程/curriculum、失败终止和实车启动安全边界 | 首次策略驱动的实车自主起立前 | **未关闭**：当前 reset 从接近名义站立构型开始，尚未训练或验证自主起立；不阻塞本轮纯仿真 V4 训练 |
| G-14 | ContractV4 文档和实现分别完成独立只读复核且无 P0/P1 | 四次 V4 正式训练前 | **已归档（流程门）**：四次 V4 正式训练及其工件已经于 2026-10-05 完成；本表不再把它写成未来 pending gate。该归档不关闭 G-08，也不声明 V4/Phase 1R 性能通过 |
| G-15 | Phase 1R 实现、四次随机化 PPO 与统一 MuJoCo sim2sim 已形成可比较证据 | Phase 2 实现前 | **已关闭（实施许可）**：四组 checkpoint 已按统一链路评估，MuJoCo 失败保留为负面对照；这不表示 Phase 1R 性能通过，也不允许 DreamWaQ 掩盖映射/物理错误 |

这些 Gate 必须形成明确结论和测试，不允许在实现中以临时常量绕过。

## 27. 架构验收标准

本文通过评审后，实施计划应满足：

1. 不再讨论复制旧 WheelLeg 环境作为主工程路线。
2. 不再以 RSL-RL 2.3.3 为新算法接口目标。
3. 每个实现任务都能归属到本文定义的模块。
4. Actor、Critic、Action、Command、history 和 reconstruction target 的维度、顺序与时间索引没有歧义。
5. PPO 与 DreamWaQ 的环境边界清晰。
6. 4060 与 5070 不产生代码分叉。
7. 平台无关部署输入全部能由实机信号产生，且 H7/RK3566 等后端只实现同一 manifest。
8. 未确认的训练参数均有明确决策 Gate，不被伪装成架构常量。
9. 实现不能修改 `AssetBundleV1` 的任何源 USD，不能绕过 asset、frame、normalization、history、reconstruction 和 manifest 断言。

## 28. 本地参考路径

- 迁移讨论：`E:\wheel_leg_rl-main\docs\2026-10-03-wheelleg-dreamwaq-migration-handoff.md`
- 正式 WheelLeg AssetBundle 根目录：`E:\wheel_leg_rl-main\wheel_leg_urdf4_usd (1)\wheel_leg_urdf4`
- 正式 WheelLeg 入口层：`E:\wheel_leg_rl-main\wheel_leg_urdf4_usd (1)\wheel_leg_urdf4\wheel_leg_urdf4.usd`
- 旧 WheelLeg 环境：`E:\wheel_leg_rl-main\wheel_leg\WheelLeg_RL_IsaacLab\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\WheelLegRobot\wheellegrobot_env.py`
- 旧 WheelLeg 资产：`E:\wheel_leg_rl-main\wheel_leg\WheelLeg_RL_IsaacLab\IsaacLab\source\isaaclab_assets\isaaclab_assets\robots\wheel_leg_robot.py`
- 复旦 Plane：`E:\wheel_leg_rl-main\fudan_rl_wheel_leg-main\fudan_rl_wheel_leg-main\plane`
- A1 DreamWaQ：`E:\gogo_2026_09\A1_Base-codex-dreamwaq-pace-baseline-port\gogo-learn\gogo_learn\dreamwaq`
- 已有控制坐标变换参考：`E:\wheel_leg_rl-main\wheel_leg_debug-main2\mujoco_control_extract\sim\main_mujoco.c`
- 已有 IMU 读取/姿态参考：`E:\wheel_leg_rl-main\wheel_leg_debug-main2\mujoco_control_extract\Application\Task\Src\INS_Task.c`
- 现有实机协议映射参考：`E:\wheel_leg_rl-main\wheel_leg_debug-main2\wheel_leg_debug-main\上下位机通信协议\rm_test-dev\Components\Device\Src\Sim2Real.c`
- 现有实机协议顺序定义：`E:\wheel_leg_rl-main\wheel_leg_debug-main2\wheel_leg_debug-main\上下位机通信协议\rm_test-dev\Components\Device\Inc\Sim2Real.h`
- Isaac Lab 2.3.2 COM 速度字段参考：`E:\gogo_2026_09\A1_Base-codex-dreamwaq-pace-baseline-port\dependencies\isaaclab-src-2.3.2.post1\source\isaaclab\isaaclab\assets\articulation\articulation_data.py`

## 29. v0.24：MuJoCo 刚体角速度限制适配

**历史设计：本节已被 v0.25 的 §30 取代。下述适配器不再存在于当前正式 runtime；公式、结果和工件仅解释旧训练/评估条件。**

本节保留当前 Isaac 训练动力学，并补齐 MuJoCo 缺少的运行时语义。`RigidBodyPropertiesCfg.max_angular_velocity` 的单位为 **deg/s**；当前 `100.0` 实际约为 `1.745329 rad/s`，不是 `100 rad/s`。本轮不推断原作者意图，也不放宽 Isaac 上限。旧 checkpoint、reset cache、Actor/Critic、history、AdaBoot、奖励、动作、命令、随机化、源 USD 和 MuJoCo XML 参数保持原语义。

PhysX 5.6.1 GPU `forwardDynamic2.cu` 对超限 link 计算基于惯量的角速度偏置力矩，而非直接截断输出角速度。公开参考版本为 `107.3-physx-5.6.1` / commit `5ca9f472105a90d70d957c243cb0ef36fe251a9f`。本工程冻结的 MuJoCo 适配器名为 `PhysxRigidAngularBiasV1`，作用于全部 27 个机器人刚体，排除 world；公式为：

```text
limit = radians(100.0)
I_world = R_inertia_world @ diag(body_inertia) @ R_inertia_world.T
tau_world = 0                                      if |omega_world| <= limit
tau_world = -I_world @ omega_world * (1-limit/|omega_world|) / 0.005 otherwise
```

每个 `5 ms` 段开始时，在独立 `MjData` 上用当时的 qpos/qvel 更新运动学，读取各刚体的世界系角速度和惯量主轴朝向。将 `tau_world` 加入 `xfrc_applied` 的力矩分量，保持五个 `1 ms` 步；下一段只替换本适配器自身的旧贡献，保留其他外力矩。该力矩属于刚体限速适配，不属于电机力矩；电机 ctrl、effort limit、PD、关节速度限制和被动阻尼仍按原控制器执行。禁止改写 live qvel、live qpos、模型惯量或闭合约束，禁止在运行中对 live data 调用 `mj_forward` 以计算此项。

适配器的周期计数和保持力矩属于 runtime 状态，随 keyframe reset 清零，不推进任何物理时间。周期来自导出 manifest 已有的 `timing.isaac_sim_dt_s`，必须为 PhysicsV4 冻结的 `0.005 s`；MuJoCo 继续使用 `0.001 s`，每次策略动作仍执行 20 步。PhysicsV4 旧三件套按本节明确的 `100 deg/s` 加载，无需修改旧 actor、golden vectors、model manifest 或 checkpoint；若未来修改 Isaac 上限，必须另行升级相关动力学契约，不得继续沿用该缺省。

新评估契约使用 `MujocoEvaluationContractV2`，在 XML 动力学身份之外显式记录本适配器的版本、上限及单位、更新/保持周期、坐标系、作用对象和力矩规则；source fingerprint 覆盖其实现。旧 V1 评估工件仍是有效的历史证据，但不得与 V2 混合排名或覆盖。八场景、500 tick/10 s、初态、命令、失败阈值与评分规则不变。Isaac 环境未改，因此不需要重复训练或重建其 reset cache 才能检查本次 MuJoCo 改动。

验收必须包含世界系非球形惯量、逐刚体超限力矩、5 ms 保持/reset 和其他外力矩保留测试；正式 runtime 的无地面相同力矩诊断必须复现候选实现；四个既有 1000-iteration checkpoint 必须用新 runtime 重新完成八场景留证，并重新核验四组 TorchScript golden vectors。

该适配器复现已公开的逐 link 偏置力矩项及其采样周期，**不宣称整个 PhysX 与 MuJoCo 求解器已等价**。候选诊断中，保留 Isaac 原上限、在 MuJoCo 按 5 ms 更新该项，闭合开启时 20 ms 平均根角速度差从 `0.431066` 降至 `0.089999 rad/s`；闭合关闭时从 `0.544934` 降至 `0.005342 rad/s`。首个 5 ms 的闭合响应差异仍在；短程改善不能替代策略闭环验收，也不关闭 RootCauseSuite 或 Phase 2 性能门。实施与正式复测记录位于 `wheelleg_dreamwaq/artifacts/debug/sim2sim/angular-limit-alignment-20261010-v1/`，实施计划见 `wheelleg_dreamwaq/docs/superpowers/plans/2026-10-10-mujoco-angular-limit-alignment.md`。

## 30. v0.25：删除人工限速作用

用户于 2026-10-11 明确授权删除三类人工限速，包括 MuJoCo 为对齐旧训练而补加的作用。删除范围是没有真实硬件依据、仅为满足速度上限而额外施加的力/力矩。正常电机 PD、输出 effort 上限、接触/摩擦、重力、闭环反力、被动阻尼及 armature 不在删除范围；穿透修正仍属于接触求解。源 USD、MuJoCo XML/mesh、质量/惯量、动作、命令、奖励、随机化采样、观测、五帧 history、CENet 和 AdaBoot 均不变。

### 30.1 Isaac 显式解除上限

当前 physics schema 为 `PhysicsV5`。单纯删除 `max_linear_velocity`、`max_angular_velocity` 或 `velocity_limit_sim` 会继承 USD/PhysX 默认值，不能视为解除限速。三个字段均显式使用 `UNRESTRICTED_SIM_VELOCITY = float.fromhex("0x1.fffffep+127")`，即有限 float32 最大值 `3.4028234663852886e38`。刚体角速度属性的单位仍为 deg/s，线速度为 m/s，关节速度为 rad/s；这个数是引擎哨兵，不代表真实电机额定速度，不作为 action、command 或 observation 的归一化上限。禁止写入 Infinity/NaN，禁止设置低于该哨兵的替代操作性限速。

`UnrestrictedVelocityPolicyV1` 明确记录 rigid-body linear/angular 与 joint speed limit 均为 `disabled`，`external_speed_limit_force=False`、`external_speed_limit_torque=False` 和 `runtime_velocity_write=False`。政策进入 PPO 与 DreamWaQ base-task 的 `task.physics.velocity_limit_policy` 和 hash，并进入两种 policy export。导出同时记录实际 rigid-body properties 与 actuator 参数，MuJoCo 加载器校验 schema、政策和三组 actuator/两个 rigid 上限的一致性。

`sim_dt=0.005 s`、`decimation=4`、control period `0.02 s`、solver iterations `96/4` 不变。线/角 damping 仍为 0；主动腿 `120/4/18 Nm/0.05 armature`、轮 `0/0.6/9 Nm/0.05 armature`、被动关节 `0/0.05/18 Nm/0.005 armature` 不变；既有随机化仍按原 profile 作用于主动参数。

### 30.2 MuJoCo 删除适配与速度分支

删除正式 `angular_limit.py`、limiter 构造/reset/apply 以及专测该历史适配的测试。运行时不再为限速写 `xfrc_applied` 或 `qfrc_applied`。原有合法外力应保留，不得用全局清零外力作为“删除适配”的实现。

`MixedActionController` 删除 `(abs(qd) >= velocity_limit) & (tau * qd > 0)` 的超速清零分支。控制仅为原腿位置 PD、轮速度 P 和原 effort clipping，真实电机控制仍可以产生制动力矩。删除不再消费的 adapter 速度字段及 `velocity_limit_event_fraction` 指标，不用 qvel/qpos 截断或投影替代。

现有 MuJoCo debug collector/evaluator 和 RootCauseSuite target-torque 参考同步删除旧限速作用及对已删除字段的读取。既有 trace schema 中的限速/guard 事件字段保留并明确写 0，表示当前没有此类作用；原 `80 rad/s` 诊断失败阈值独立保留。绑定旧权重和旧 hash 的冻结诊断入口仍属于历史条件，当前 runtime 拒绝混用，不改写其冻结注册或历史结果。

action adapter 升级为 `MujocoActionAdapterV2`。`model_manifest.json` 仅更新其版本和实现 SHA256；XML 与 `MujocoDynamicsSemanticsV1` hash 不变。MuJoCo 仍为 `0.001 s` / 20 substeps / 500 tick / 10 s；几何、contacts、8 个 equality constraints 和各自 solver 参数不变。

### 30.3 兼容性、评估和训练边界

评估契约升级为 `MujocoEvaluationContractV3`，记录 `PhysicsV5` 与无人工限速政策；旧 V1/V2 报告不得进入当前排名，也不得覆盖。原评估线/角/关节速度失败阈值保留，其中 `80 rad/s` 是诊断失败阈值，不向机器人施力，也不再通过 actuator 物理上限字段读取。

PPO 与 DreamWaQ 的旧 PhysicsV4 checkpoint/run manifest/reset cache 均保留原件。当前模型/optimizer resume 和 Isaac play 沿用严格完整契约校验并拒绝旧物理身份。export 不得把旧 checkpoint 重标为 PhysicsV5；正式 MuJoCo loader 拒绝旧 PhysicsV4 包。旧权重如需在新动力学下测试，须另建明确标记的诊断链，不能伪装为同条件训练/正式验收；本次不新增此类绕过路径。

新随机化环境重新生成 PhysicsV5 reset cache，reset 时序、双 tensor 内容定义和原两个缓存失效修复不变。下一次正式训练必须从新 run 开始，不从旧优化状态直接续训。本次只删除、验证和留证，不启动训练，不宣称站立/速度跟踪或 sim2real 性能改善。

### 30.4 验收

验收需在真实 Isaac runtime 读回 27 个刚体属性和 26 个 DOF 限速，核验原 effort/damping/armature；无接触、无重力单刚体以 `200 m/s` / `10 rad/s` 超过旧上限运行，设置旧 `100 m/s` / `100 deg/s` 的另一个刚体作为负面对照。名义与随机化环境均须短程 finite、reset、闭环和观测维度验收，随机化环境还须证实旧 PhysicsV4 cache 被拒绝。

MuJoCo 验收覆盖超旧关节转速仍按 PD/effort 输出、超旧刚体速度不注入外力、已有外力保持、20 子步实际轨迹与直接物理参考精确相等，以及 history/reset 回归。PPO/DreamWaQ 正确 hash 的旧契约必须被拒绝；新的 export/loader/评估政策必须一致；四组旧 TorchScript golden-vector 数值仍需核验但不重标其物理身份。

实施计划见 `wheelleg_dreamwaq/docs/superpowers/plans/2026-10-11-remove-artificial-speed-limit-forces.md`。原实现、参数 SHA256 和本轮验证工件保存在 `wheelleg_dreamwaq/artifacts/debug/sim2sim/remove-artificial-speed-limits-20261011-v1/`。
