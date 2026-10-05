# WheelLeg DreamWaQ Isaac Lab Architecture

日期：2026-10-04  
状态：Architecture v0.14；保留 Phase1ContractV3 历史回放，Phase1ContractV4 实现与训练前验证进行中  
适用范围：新建轮腿机器人平地强化学习工程，不修改或继承旧 WheelLeg 任务实现

## 1. 文档目的

本文定义新 WheelLeg DreamWaQ 工程的系统边界、运行版本、模块职责、张量契约、训练阶段和部署接口。

本文只描述工程架构和已经确认的行为，不创建工程代码，也不冻结尚需仿真标定的数值参数。后续实现计划必须以本文为上游规范。

## 2. 项目目标

V1 建立一个可验证、可扩展、可部署的轮腿强化学习基线，按以下顺序推进：

1. 验证 USD、关节顺序、关节方向、限位和 actuator。
2. 建立六维纯关节空间控制的最小 `DirectRLEnv`。
3. 训练普通 asymmetric PPO 平地基线。
4. 在不改变环境任务定义的前提下接入 DreamWaQ/CENet。
5. PPO 和 DreamWaQ 都稳定后，再加入域随机化、复杂地形和更完整的 privileged critic。
6. 最终导出只依赖实机可获取信号的策略，在选定的实机计算平台上以 50 Hz 执行。H7、RK3566 或其他平台只影响部署后端和资源预算，不改变训练 schema 与策略语义。

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
| 腿 actuator | effort `18 Nm`，velocity `45 rad/s`，stiffness `120`，damping `4`，armature `0.05` |
| 轮 actuator | effort `9 Nm`，velocity `45 rad/s`，stiffness `0`，damping `0.6`，armature `0.05` |
| passive actuator | effort `18 Nm`，velocity `80 rad/s`，stiffness `0`，damping `0.05`，armature `0.005` |
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

本项目已经稳定复现 Windows 下“Kit 启动后首次导入 `tensordict._C` 发生 access violation”的 DLL 加载顺序冲突。Phase 1 因此固定 `tensordict==0.14.2`，并要求 `train_ppo.py` 与 `play.py` 在创建 `AppLauncher` 前先导入 PyTorch 和 TensorDict。该处理已写入直接依赖、`uv.lock` 与 dependency manifest；它只改变进程初始化顺序，不改变 PPO 张量或模型格式。

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
| observation groups | 明确使用 TensorDict 的 `policy`、`critic` 和 DreamWaQ 扩展字段，不依赖位置元组猜测语义 |
| ActorCritic | 构造参数、`act`、`evaluate`、distribution state 和 recurrent mask 按 3.1.2 签名实现 |
| Transition | 逐字段定义 observation、action、reward、done、value、log-probability、distribution statistics 和 DreamWaQ 附加张量 |
| RolloutStorage | 核对 `init_storage`、transition 写入、GAE/returns 和 mini-batch generator 返回结构 |
| runner/algorithm registration | 使用 3.1.2 的 runner 和类注册入口，不保留 A1 的私有导入副作用 |
| checkpoint | 显式列出 Actor、Critic、CENet、optimizer、normalization/schema manifest 和 iteration key |
| configuration | 未识别字段必须报错；禁止用无约束 `**kwargs` 静默吞掉旧版本参数 |

迁移测试必须构造一批具名 transition，验证写入 storage、mini-batch 取回和 loss 读取的字段逐项一致，而不是只验证张量总维度。

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

显存容量通过 `256 -> 512 -> 1024 -> 2048` 环境的基准测试确定。正式数值记录在硬件 profile 中，不写入环境类。训练产生的模型可以在另一张 GPU 上加载；checkpoint 必须保存 model、optimizer、RSL-RL adaptive schedule 的 `alg.learning_rate`、零基 runner iteration、明确的 `completed_iterations` 和可恢复的随机数状态。续训总是创建新 run directory，先校验源 run manifest 和 checkpoint 指定的 ContractV3/V4，再恢复模型、optimizer、`alg.learning_rate`、Python/NumPy/PyTorch/CUDA RNG；恢复后的 `alg.learning_rate` 必须与 optimizer 全部 parameter group 的单一 learning rate 相等，否则 fail fast，避免下一次 adaptive update 用默认标量覆盖已经恢复的 optimizer learning rate。`--max-iterations` 固定表示目标总轮数，恢复点后的实际更新数为 `target_total - completed_iterations`，禁止把 RSL-RL 保存的零基 iteration 直接作为下一轮而重复更新。不同 GPU、不同 `num_envs` 或不同 mini-batch 划分不保证训练轨迹逐位一致，也不能改变 checkpoint 的网络和 schema 兼容性；续训属于“恢复优化状态和 RNG、创建新环境状态”，不声称恢复中断瞬间的仿真 state。

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

Phase 1 达到进入门槛后，训练启动项切换为 DreamWaQ PPO。任务环境保持不变，普通 PPO learner 不再同时运行。

```text
WheelLegFlatEnv
    |
    +-- Current ActorObsV1 25D
    |       |
    |       +---> Five-frame history 5 x 25 = 125D
    |       |               |
    |       |               v
    |       |         CENet encoder
    |       |               |
    |       |               +---> estimated velocity 3D
    |       |               +---> context latent 16D
    |       |
    |       +---> concatenate:
    |             current obs 25D
    |             + estimated velocity 3D
    |             + context latent 16D
    |                     |
    |                     v
    |               Actor network
    |                     |
    |                     +---> ActionV1 6D -------------> WheelLegFlatEnv
    |                               |                              |
    |                               |                              +---> returned next ActorObsV1
    |                               |                                         |
    |                               |                                         +---> select [0:6] and [9:19]
    |                               |                                                = next physical target 16D
    |                               |
    |                               +---> stop-gradient action 6D --------+
    |                                                                     |
    |   context 16D + stop-gradient estimated velocity 3D ----------------+---> training-only decoder
    |                                                                                |
    |                                                                                +---> predicted next physical target 16D
    |
    +-- CriticObsV1 41D
    |       |
    |       v
    |   Privileged Critic network
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

控制动作仍然只由 Actor network 输出。CENet encoder 输出 Actor 需要的预测线速度和 context；训练期 decoder 接收 context、停止梯度的预测速度和停止梯度的当前动作，预测动作执行后的下一帧物理本体观测。Decoder 只提供辅助 loss，不预测动作、不替代环境，也不进入部署。Critic 只输出 `V(s)`。DreamWaQ 不是 PPO 旁边的另一个控制器，而是包含 Actor、Critic、CENet 和扩展 storage 的 PPO learner。

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
                         |    CENet encoder, deterministic outputs
                         |               |
                         |               +---> estimated velocity 3D
                         |               +---> context 16D
                         |
                         +---> concatenate:
                               current obs 25D
                               + estimated velocity 3D
                               + context 16D
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
| CENet encoder | 预测线速度 3D、context 16D | 否，输出作为 DreamWaQ Actor 输入 |
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
│           │       ├── cenet.py
│           │       ├── actor_critic.py
│           │       ├── ppo.py
│           │       ├── storage.py
│           │       └── history_wrapper.py
│           └── deployment/
│               ├── export.py
│               ├── manifest.py
│               └── observation_adapter.py
├── scripts/
│   ├── bootstrap.ps1
│   ├── audit_asset.py
│   ├── smoke_random_actions.py
│   ├── train_ppo.py
│   ├── train_dreamwaq.py
│   ├── play.py
│   └── export_policy.py
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

`NormalizationV2` 固定 `vx_max_abs=1.5`、`yaw_rate_max_abs=1.0`、`nominal_base_height=0.20`、`height_command_span=0.04`，命令归一化后裁剪到 `[-1,1]`。其余字段 offset、scale 和 clip 与 V1 相同。五帧历史保存的也是已经完成固定 normalization 的 ActorObsV1。CENet 的重建目标不是完整 ActorObsV1，而是从归一化下一帧观测中提取的 `ProprioReconstructionTargetV1`，定义见第 16.2 节。

## 14. Critic 观测契约 CriticObsV1

普通 PPO 和第一版 DreamWaQ 共用 41 维 privileged Critic：

| Slice | 维度 | 内容 |
|---|---:|---|
| `0:25` | 25 | 当前 ActorObsV1 |
| `25:28` | 3 | `ControlFrameV1` 仿真真实 root COM linear velocity |
| `28:29` | 1 | root COM 相对平地的真实高度 |
| `29:35` | 6 | canonical joint order 的六关节加速度 |
| `35:41` | 6 | canonical joint order 的六关节 applied torque |

CriticObsV1 只在训练期间存在，不进入部署运行时。其 `0:25` 直接复用已经归一化的 ActorObsV1，不重新归一化；真实线速度固定使用 `2.0` scale。真实高度使用与高度命令一致的中心和 span，关节加速度与力矩的固定 scale/clip 在 actuator 审计后冻结。第一阶段不为凑维度加入接触状态；只有接触信息被 reward、termination 或随机化证明有助于 value estimation 时才升级 Critic schema。

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

DreamWaQ 专用 wrapper 维护五帧历史，环境本身仍只产生单帧 ActorObsV1：

```text
history_t = [o_(t-4), o_(t-3), o_(t-2), o_(t-1), o_t]
shape     = [N, 5, 25]
flat      = [o_(t-4)[0:25], o_(t-3)[0:25], o_(t-2)[0:25], o_(t-1)[0:25], o_t[0:25]]
flat shape = [N, 125]
```

wrapper 内部的 canonical 存储必须是连续的 `[N,5,25]` 张量，并以 `history.contiguous().reshape(N,125)` 做 frame-major 展平。禁止使用 Isaac Lab ObsTerm/observation group 的内置 history 功能代替该 wrapper，避免 term-major 与 frame-major 布局混淆。reset 时用 reset 后当前观测复制填满五帧，禁止使用全零历史。

DreamWaQ wrapper 向算法提供：

```text
policy:         [N, 25]
policy_history: [N, 125]
critic:         [N, 41]
```

### 16.2 CENet

第一版 CENet 固定语义：

```text
输入：五帧归一化 ActorObsV1，125D
显式输出：归一化 ControlFrameV1 root COM linear velocity，3D
隐式输出：context posterior (mu, logvar)，各 16D
训练采样：context z，16D
decoder 输入：[context z 16D, stop_gradient(estimated velocity) 3D,
              stop_gradient(current clipped canonical ActionV1) 6D]，25D
训练期输出：归一化下一帧 ProprioReconstructionTargetV1，16D
```

`ProprioReconstructionTargetV1` 固定为下一帧 ActorObsV1 的物理本体切片：

| Target slice | 维度 | 来源 |
|---|---:|---|
| `0:6` | 6 | `next_actor_obs[0:6]`：角速度与 projected gravity |
| `6:16` | 10 | `next_actor_obs[9:19]`：四腿位置误差与六关节速度 |

命令 `next_actor_obs[6:9]` 是外部输入而不是机器人动力学状态；V4 虽在 episode 内保持，但 reset 时仍可改变且 done transition 已被 mask，因此不进入重建目标。`next_actor_obs[19:25]` 等于当前动作 `a_t`，已作为 decoder 条件且重建它没有表征学习价值，因此两段都排除。Decoder 的任务是预测“执行 `a_t` 后机器人下一帧的物理本体观测”，不是预测动作、命令、reward 或环境 reset。

Decoder 的存在是为了给没有人工标签的 16D context 提供逐 transition 的密集动力学监督：只有当 context 包含当前运动状态和隐藏动力学信息时，才有助于预测动作执行后的下一状态。它不是 PPO 或环境运行的必要组件，因此 Phase 1 不存在；进入 DreamWaQ 后，它只通过辅助 loss 训练 CENet，部署时删除。`stop_gradient` 只切断反向传播，decoder 前向计算仍能读取 estimated velocity 和当前动作的数值。

第一版 velocity head 是 deterministic regression head，不采样、不参与 KL；`L_KL` 只将 16D context posterior 约束到标准正态先验。训练时 context 使用重参数化采样，评估和部署时使用 `context_mu`。Actor 不允许用 Critic 中的真实线速度替换 estimated velocity，第一版不包含 AdaBoot 或 privileged-velocity teacher forcing。

训练期 context 随机性采用 transition 固定的重参数化噪声：rollout 对每个 transition 采样 `epsilon_t ~ N(0,I)`，并使用 `z_t = mu_t + exp(0.5 * logvar_t) * epsilon_t` 产生 Actor 输入和 decoder 输入。`epsilon_t` 作为 16D DreamWaQ storage 字段保存；PPO 的所有 epoch/mini-batch 对该 transition 重新前向计算当前 `mu/logvar`，但必须复用同一 `epsilon_t`，禁止每次更新重新抽样。这样 PPO 新旧策略比率共享同一外生 latent 噪声，同时保留 actor loss 到 CENet 的重参数化梯度。评估和部署不保存或使用 `epsilon`，固定令 `z=context_mu`。

Actor 输入：

```text
normalized current ActorObsV1 25
+ normalized estimated velocity 3
+ context latent 16
= 44D
```

Actor 输出 ActionV1，6D。

部署时：

- CENet 使用 deterministic velocity output 和 `context_mu`，不采样。
- velocity 输出保持训练时的固定 `2.0` scale，直接进入 Actor，不在部署侧额外归一化。
- 不导出 decoder。
- 不导出 Critic。
- 不导出训练 loss。

### 16.3 训练损失

```text
L_total = L_ppo_actor
        + value_coef * L_critic
        - entropy_coef * entropy
        + velocity_coef * L_velocity
        + reconstruction_coef * L_next_proprio
        + kl_beta * L_KL
```

其中：

- `L_velocity` 的监督目标是 CriticObsV1 `25:28` 中已转换到 ControlFrameV1、并固定乘 `2.0` scale 的真实 root COM linear velocity。
- `L_next_proprio` 的目标是下一控制 step 的归一化 `ProprioReconstructionTargetV1` 16D，只在 reconstruction mask 为 true 时计算。
- `L_KL` 只约束 context posterior 的 `mu/logvar`，不约束 deterministic velocity head。
- CENet loss 不是环境 reward。
- Actor 和 Critic 仍共享同一个环境 reward。

第一版使用一个 Adam optimizer 管理 Actor、Critic 和 CENet 参数，对加权总 loss 做一次 backward，但每条梯度路径必须显式满足：

- PPO actor loss 更新 Actor，并通过 44D Actor 输入回传到 CENet encoder、velocity/context 输出路径。
- velocity loss 更新 CENet 共享 encoder 和 velocity head。
- KL loss 更新 CENet 共享 encoder 和 context posterior head。
- reconstruction loss 通过 `context z` 更新 CENet 共享 encoder/context head，并更新 decoder。
- decoder 输入的 estimated velocity 和当前 action 都必须停止梯度，因此 reconstruction loss 不更新 velocity head，也不能通过动作采样路径更新 Actor。
- critic loss 只更新 Critic。

PPO 更新不得依赖全局 RNG 隐式重现 context。mini-batch 必须显式携带 transition 保存的 `context_epsilon`；在参数和输入不变时，同一 observation/history/epsilon 的 `z`、Actor action mean 和 log probability 必须可重复。改变 `epsilon` 应改变 sampled context，但不得改变 deterministic evaluation 路径。

Decoder 不得接入 Actor 推理路径。使用同一 optimizer 不等于共享网络参数，所有 loss 系数、global gradient clipping 和各模块 gradient norm 都必须记录；若后续改为多 optimizer或改变上述 stop-gradient 边界，视为算法变更并单独评审。

DreamWaQ checkpoint 必须包含 Actor、Critic、CENet encoder/decoder、optimizer state、schema/normalization manifest 和 iteration。部署导出只包含 checkpoint 指定的固定 normalization、五帧历史更新、CENet encoder deterministic path 与 Actor；V4 对应 NormalizationV2。

### 16.4 下一帧和 reset mask

Isaac Lab `DirectRLEnv` 在 episode 结束时会 reset 对应环境，并向 runner 返回 reset 后观测。因此 rollout storage 可以保存返回的 next observation 张量，但重建 mask 必须满足：

```text
reconstruction_mask = not (terminated or truncated)
```

任何发生 reset 的 transition 都不参与下一帧重建 loss，避免把新 episode 初始状态当作上一 episode 的下一状态。

transition 的时间定义固定为：Actor 使用以 `o_t` 结尾的 `history_t` 产生 clipped canonical `a_t`；环境执行后，非 done 环境产生归一化重建目标 `p_(t+1)=concat(o_(t+1)[0:6],o_(t+1)[9:19])`。storage 至少保存 policy observation/history、critic observation、action、reward、`terminated`、`truncated`、value、log probability、action distribution statistics、16D `context_epsilon`、16D next physical target 和 reconstruction mask。wrapper 必须先保存本次 transition 使用的 `history_t`，再更新历史：非 done 环境 append 完整的 `o_(t+1)`；done 环境用 reset 后 `o_reset` 复制填满五帧。禁止先 append 再保存而造成错一帧，也禁止让 done 环境残留上一 episode 的任意历史帧。

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
- CENet history length、frame-major 展平顺序、velocity scale 和 decoder 训练契约。
- 模型数值精度、导出格式和后端版本。
- 训练 git revision 或构建标识、master seed、硬件 profile、`num_envs` 和 mini-batch 划分。

部署 adapter 必须包含独立于策略的安全状态机：传感器有效性检查、启动姿态检查、通信/推理超时、动作限幅、关节软限位、急停和失联回退。策略不能直接绕过这些保护写电机目标。

网络大小必须在选定部署平台上单独通过模型文件、工作区 RAM、数值精度和 50 Hz worst-case latency 预算检查。A1 的隐藏层大小不是本项目默认值；普通 PPO 网络在 Phase 1 前冻结，最终 DreamWaQ Actor 和 CENet hidden dimensions 在 Phase 2 前按候选平台的共同可部署上限冻结。Critic 和 CENet decoder 不计入部署模型。

## 20. 配置边界

配置分为六类：

1. **Asset 配置**：历史回放使用 `AssetBundleV1`；V4 新训练使用 `AssetBundleV2`。每个 bundle 都固定根路径解析、五文件哈希/大小、入口层和 prim/joint 期望；根路径可以按机器解析，资产清单与身份不能改变。
2. **Schema 配置**：坐标系、动作、命令、观测、history layout、reconstruction target、顺序、固定归一化和 manifest，变更需要 schema 升级。
3. **Task 配置**：控制周期、命令范围、reset、termination、reward 和 actuator。
4. **Algorithm 配置**：网络、PPO、CENet 和 loss 参数。
5. **Training hardware 配置**：并行环境数、batch 划分、headless 和渲染。
6. **Deployment backend 配置**：模型格式、数值精度、线程/加速后端和设备 I/O；不得覆盖 schema 语义。

training hardware profile 和 deployment backend profile 都不允许覆盖 schema 或 task 语义。训练 CLI 可以覆盖实验性超参数，但每次运行必须把最终解析后的完整配置写入日志目录。

同一正式多 seed suite 必须生成种子无关训练指纹。完整 `Phase1ContractV4`、源码/依赖/运行栈、hardware profile、rollout 与 mini-batch 布局进入指纹；seed、派生 RNG、时间、命令行、resume 元数据不进入。原始 `env.yaml`/`agent.yaml` hash 仍保留用于审计，但因为其中包含 seed、run name 和 log directory，不直接作为跨 run 相等条件；这些配置的训练语义由完整 ContractV4 覆盖。任一后续 run 指纹不一致时 suite 必须 fail fast。

旧 checkpoint 使用 `Phase1ContractV3`，只允许在 V3/V1 环境严格回放或续训。新训练 checkpoint 使用 `Phase1ContractV4`，固定 `AssetBundleV2`、`WheelOnlyCollisionV2`、CommandSamplingV2、NormalizationV2、RewardSchemaV2 和 VirtualLegKinematicsV1。contract hash 至少覆盖 Action/Command/Actor/Critic/Normalization/ControlFrame schema、命令模式概率与范围、episode 内保持语义、动作裁剪、finite/infinite-horizon 语义、完整 root 初始 pose/velocity/joint state、完整 rigid/articulation/PhysX/collision/actuator 配置、虚拟腿几何常量与奖励公式/权重、网络和 PPO 参数。机器人配置必须覆盖 `UsdFileCfg` 的 spawner callable 及除机器相关绝对 `usd_path` 之外的全部 spawn 参数；绝对路径由所选 AssetBundle 的相对入口文件和资产 hash 替代。地面配置必须覆盖 prim path、translation、spawner callable、Cuboid 尺寸、collision properties、visual material、physics material以及 friction/restitution combine mode，环境只能通过该配置生成地面，不得另有硬编码副本。只有 `device`、`num_envs` 与 `num_mini_batches` 被明确列为 hardware-mutable，不进入跨 profile 兼容性拒绝条件；它们仍必须记录在每个 run manifest。`max_iterations`、保存间隔、seed 和日志后端属于运行控制元数据，不改变模型兼容性。

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
- RSL-RL 版本、模块来源或配置字段与固定的 3.1.2 接口不一致，或存在未识别配置字段。
- observation、reward、action target 出现 NaN 或 Inf。
- 受控 joint 没有外部有限软件限位、速度/力矩限制或 actuator 参数。
- `sim_dt * decimation != control_dt`，或 V4 的任一冻结值不等于 `sim_dt=0.005 s`、`decimation=4`、`control_dt=0.02 s`、solver position iterations `96`、solver velocity iterations `4`。
- checkpoint schema 与运行环境不一致。
- manifest 与实机 observation adapter 不一致。
- 固定 PyTorch/CUDA 栈无法在当前 GPU 上成功分配张量、执行真实 CUDA kernel 并同步。`torch.cuda.get_arch_list()` 只记录为诊断信息，不得因列表中没有与 capability 完全同名的条目而错误拒绝可执行的 RTX 4060。
- checkpoint 的 model/optimizer state、零基 iteration、`completed_iterations`、run-manifest hash、对应 ContractV3/V4、hardware profile、optimizer learning rate、`alg.learning_rate` 或 RNG 恢复信息缺失/不一致，或续训目标总轮数不大于已完成轮数。
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
- ProprioReconstructionTargetV1 精确提取 `[0:6] + [9:19]`，并明确排除命令和 previous action 的 slice 测试。
- CENet shape、25D action-conditioned decoder input、16D output、deterministic inference、context-only KL 和 masked reconstruction 测试。
- CENet transition 固定 `context_epsilon` 测试：同一参数/输入/epsilon 的 sampled context、Actor mean 和 log probability 可重复；不同 epsilon 改变 sampled context；评估路径恒等于 `context_mu`；actor loss 梯度可经固定 epsilon 回传到 `mu/logvar`。
- autograd 测试：reconstruction loss 必须更新 context encoder/decoder，但不得经 detached velocity/action 更新 velocity head 或 Actor。
- RSL-RL 3.1.2 具名 Transition 写入、storage 取回和 mini-batch 字段往返测试；未知配置字段必须失败。
- schema/manifest 不兼容拒绝测试，并覆盖 `clip_actions`、finite-horizon、gravity、root 初速度、完整 solver/rigid/articulation 配置和 PPO `gamma` 等行为漂移。
- ContractV4 固定数值测试必须直接断言 `sim_dt=0.005`、`decimation=4`、`control_dt=0.02`、solver position iterations `96`、solver velocity iterations `4`；只检查乘积不算通过。任一字段改变都必须改变 contract hash 并拒绝旧 checkpoint。
- checkpoint 中 model、optimizer、seed、零基 iteration、`completed_iterations`、optimizer learning rate、`alg.learning_rate`、RNG state、hardware profile 和 batch 元数据完整性测试。
- Episode 日志必须按 rollout 内全部 `extras["log"]` 的 key union 聚合，不能因第一个 step 尚无 reset 而丢失后续 `Episode/*` 指标。
- 训练指纹必须允许 seed/run-name/log-dir 不同，但拒绝 ContractV4、源码/依赖、profile 或 batch 语义变化；原始配置文件 hash 继续保留审计价值。
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
- 奖励逐项有限值与时间尺度测试。
- reconstruction target done mask 测试。
- `terminated`、`truncated`、多环境异步 reset 下的 history、当前 action 和 16D next physical target 时间索引测试。
- RTX 5070 profile 的 PPO 短训练启动测试，以及从该 checkpoint 切换 hardware profile 后继续到明确总轮数的恢复测试。
- DreamWaQ 短训练启动测试。

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
- DreamWaQ velocity、KL、reconstruction 总 loss，以及角速度、projected gravity、腿位置和关节速度四个重建分块的 MSE。
- Actor、Critic、CENet encoder/context/velocity head 和 decoder 的 gradient norm。
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

V4 首轮验证固定为四次互相独立、从头开始的 `1000 iteration` PPO 训练。四次运行除随机 seed 和派生 RNG 状态外使用同一个 ContractV4；不得从 V3 checkpoint warm-start，也不得在四次运行之间 resume。完成后使用同一 MuJoCo headless 评估协议比较 base height、base roll/pitch/yaw、左右 `phi0` 差和速度跟踪。迭代数达到 1000 不自动视为 Phase 1 通过；只有 G-08 根据 TensorBoard 与 sim2sim 数据确认后，才允许开始 Phase 2 DreamWaQ。

### Phase 2：DreamWaQ

交付：

- 五帧历史 wrapper。
- CENet encoder/decoder。
- DreamWaQ ActorCritic、PPO 和 rollout storage。
- frame-major HistoryLayoutV1、速度监督、动作条件的 16D 下一帧物理重建和 KL loss。
- RSL-RL 3.1.2 Transition/storage/runner 迁移契约测试。
- done transition 重建 mask。
- deterministic export actor。
- 完整训练 checkpoint 与只含推理路径的部署 manifest。

退出条件：DreamWaQ 不低于 PPO 基线，CENet loss 有界，reconstruction 的 stop-gradient 测试通过，导出模型与训练模型输出对齐。

### Phase 3：鲁棒性扩展

交付顺序：

1. actuator、延迟、观测白噪声和 episode 内恒定 gyro bias 随机化。
2. 质量、质心、摩擦和恢复系数随机化。
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
| ADR-009 | 普通 PPO 基线通过后才能接入 DreamWaQ。 |
| ADR-010 | DreamWaQ 使用 frame-major 五帧 125D 历史、3D 速度估计、16D context 和 44D Actor 输入；禁止使用 Isaac Lab 内置 term-major history。 |
| ADR-011 | 第一版 DreamWaQ 使用当前动作作为条件，重建下一帧 16D ProprioReconstructionTargetV1，并包含 KL loss；不重建命令/previous action，不包含 AdaBoot。 |
| ADR-012 | 实机不计算真实线速度、真实机身高度或绝对 yaw 作为策略输入。 |
| ADR-013 | V3 历史合约的唯一正式资产为 `AssetBundleV1`；五个文件均钉路径/大小/SHA256，源文件不可修改或 flatten。V4 新训练使用由 V1 复制并只删除内嵌地面的独立 `AssetBundleV2`。 |
| ADR-014 | 训练和实机统一使用 ControlFrameV1；仿真三维向量固定经过 `R_control_from_usd`，IMU 预期 `R_control_from_imu=I` 并必须实测。 |
| ADR-015 | ActionV1 canonical joint order 固定为 `[jIJ,jIO,jAB,jAG,jwheel_left,jwheel_right]`；腿协议重排、轮协议 `[left(ID4),right(ID5)]` 和仿真/硬件符号由 adapter 显式处理。 |
| ADR-016 | Actor、Critic 与 CENet 使用合约指定的字段级固定 normalization；V3 使用 NormalizationV1，V4 使用 NormalizationV2。仅使用 RSL-RL 原生 Actor/Critic normalization 开关，项目不得实现 velocity running normalizer。 |
| ADR-017 | 部署计算平台暂不冻结；H7、RK3566 等后端不得改变模型 schema、坐标、归一化和动作语义。 |
| ADR-018 | 当前 USD 的真实闭环约束优先，passive joint 不复制旧工程高刚度 PD，必须通过 Phase 0 稳定性门槛。 |
| ADR-019 | 第一版 DreamWaQ 用一个 Adam optimizer 联合更新 Actor、Critic 与 CENet；reconstruction 对 action 和 estimated velocity 停止梯度，并固定其余梯度路径和模块级监控。 |
| ADR-020 | Windows 环境采用 uv + Isaac Sim 5.1.0 pip package family + 外部固定 revision 的 Isaac Lab `v2.3.2` editable checkout + RSL-RL 3.1.2；A1 补丁进入候选兼容性表，只有对应 smoke test 复现后才能启用。 |
| ADR-021 | 机器人通过 `UsdFileCfg` default prim reference 加载；资产根部平级 PhysicsScene 不进入机器人命名空间，运行时必须只有一个环境 PhysicsScene。 |
| ADR-022 | 实机角速度必须执行 bias 补偿；策略 projected gravity 是单位向下向量，固件 `gravity_b` 只能经 `-gravity_b/||gravity_b||` 转换后使用。 |
| ADR-023 | 训练记录 master/派生 seed、RNG state、硬件 profile 和 batch 布局；跨 GPU 可加载和续训，但不承诺不同并行布局逐位复现。 |
| ADR-024 | DreamWaQ rollout 为每个 transition 保存 16D `context_epsilon`；PPO 多 epoch 更新复用该 epsilon 并重新计算当前 `mu/logvar`，禁止 mini-batch 内重新抽样；评估和部署使用 `context_mu`。 |
| ADR-025 | `jAG/jAB` 的前后文字标注存在一手代码冲突；所有机器接口以 joint name 与 CAN ID 为权威，物理角色必须通过 G-11 低力矩单电机标定冻结。 |
| ADR-026 | V1 保留 USD 的全部质量，包括 12 个 dummy body 的约 `0.12 kg`，不修改或重导出源资产。 |
| ADR-027 | V3 使用 `WheelOnlyCollisionV1`，V4 使用 `WheelOnlyCollisionV2`；二者都只保留左右轮机器人碰撞，V2 不再包含禁用资产内地面的步骤。两版都不启用接触传感器、collision reward 或 collision termination。 |
| ADR-028 | 每个 PPO checkpoint 保存 model、optimizer、optimizer learning rate、`alg.learning_rate`、零基 iteration、`completed_iterations`、累计 timestep/time、seed、Python/NumPy/PyTorch/CUDA RNG state、hardware profile、run-manifest hash 和完整 Phase1ContractV3/V4 hash；play 和 resume 在构造训练状态前 fail-fast 校验。 |
| ADR-029 | Windows 固定 `tensordict==0.14.2`，训练和播放在 Kit 前预加载 PyTorch/TensorDict，以处理已复现的原生 DLL 加载顺序冲突。 |
| ADR-030 | USD 与 canonical 轮反馈使用同一 `[+1,-1]` involution；目标、位置、速度、加速度和 applied torque 都必须显式转换。 |
| ADR-031 | orientation 与 tilt termination 基于 `upright_cos=-projected_gravity_z`，必须惩罚并终止完全倒置状态。 |
| ADR-032 | 4060/5070 运行兼容性由真实 CUDA kernel probe 判断，不要求 capability 字符串与 `torch.cuda.get_arch_list()` 精确匹配。 |
| ADR-033 | PPO 续训创建新 run，`--max-iterations` 表示总目标轮数；恢复模型、optimizer、RSL-RL adaptive `alg.learning_rate` 与 RNG，但不伪称恢复中断瞬间的环境状态。 |
| ADR-034 | `Phase1ContractV3` 必须覆盖机器人 spawner callable 和完整地面生成配置；环境从该配置实例化地面，禁止 contract 与运行时各维护一份硬编码地面语义。 |
| ADR-035 | 新训练升级为 `Phase1ContractV4`，固定 AssetBundleV2、WheelOnlyCollisionV2、CommandSamplingV2、NormalizationV2、RewardSchemaV2 与 VirtualLegKinematicsV1；V3 只用于旧 checkpoint。 |
| ADR-036 | CommandSamplingV2 在 reset 时按 `20%/30%/20%/30%` 选择站立/直行/转向/组合模式，`vx` 范围 `[-1.5,1.5] m/s`、yaw 范围 `[-1,1] rad/s`，命令在完整 episode 内保持不变。 |
| ADR-037 | V4 新增真实虚拟腿角 wrapped-difference 平方奖励，固定权重 `-1.0`；该项不进入 Actor/Critic observation。 |
| ADR-038 | V4 用四个不同 seed 从头训练四次 1000 iterations，四次只允许 seed/RNG 不同；最终由统一 MuJoCo headless 指标选出最佳 checkpoint。 |
| ADR-039 | MuJoCo sim2sim 的机构源固定为本地 `wheel_leg_urdf4_self_mesh_all.xml`，使用其 freejoint、闭链和六执行器拓扑，并替换零字节资源为本地完整 mesh。 |
| ADR-040 | MuJoCo 动力学以 AssetBundleV2/USD 的质量和惯量为权威；源 XML 缺失的 12 个可动 dummy 必须作为无碰撞 passive body/revolute joint 分支显式生成，禁止名义姿态惯量合并；USD composed root pose 必须转换为 MuJoCo parent-local pose，逐 body 和总质量/惯量审计失败时禁止 sim2sim。 |
| ADR-041 | MuJoCo 使用 WheelOnlyCollisionV2：仅左右轮 proxy 与地面形成显式接触，滑动摩擦 `1.0`；visual、base 和非轮 geom 禁止碰撞。 |
| ADR-042 | MuJoCo 固定 `R_control_from_mujoco=[[0,-1,0],[1,0,0],[0,0,1]]`；根高度与线速度基于 base COM，freejoint 原点不得替代 COM。 |
| ADR-043 | 四个 actor 的每个 headless 场景从同一完整 keyframe 独立 reset；指标按适用场景等权聚合，失败 run 先按完成场景和 survival 排名，若全部失败则 Phase 1 不通过。 |
| ADR-044 | USD authored `q=0` 只定义 MuJoCo 模型参考构型；headless/viewer reset 必须写入 Phase1ContractV4 冻结的 base freejoint 与全部 26 个 hinge 初态，并断言编译维度 `nbody=28/njnt=27/nq=33/nv=32/nu=6/neq=8/npair=2`。 |
| ADR-045 | 正式四 run suite 使用种子无关训练指纹、`MujocoDynamicsSemanticsV1`、`MujocoEvaluationContractV1` 和训练监控门禁；不同训练/评估语义或任一 hard anomaly 均禁止产生正式 selected 模型。 |

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
| G-08 | PPO 进入 DreamWaQ 的量化门槛 | Phase 1 退出前 | **未关闭**：四次 V4 1000-iteration 训练与统一 sim2sim 评估完成后依据数据单独确认 |
| G-09 | DreamWaQ Actor、CENet hidden dimensions 和候选平台共同可部署资源上限 | Phase 2 正式训练前 | 未开始 |
| G-10 | 最终部署平台、模型格式、数值精度和推理后端 | Phase 4 前 | 未开始 |
| G-11 | 实机 motor sign、zero offset、协议方向、gyro bias 标定参数、IMU/projected-gravity conformance 与安全状态机阈值 | 首次带载实机推理前 | 未开始 |
| G-12 | Phase 1 实现经独立只读代码复核且无阻塞 P0/P1 | 首个 2000 iteration run 前 | **已关闭**：第三轮独立只读复核结论为 A，无阻塞 P0/P1；允许启动首个 2000 iteration run |
| G-13 | 区分站立附近高度跟踪与自主起立，冻结低姿态初态分布、起立命令过程/curriculum、失败终止和实车启动安全边界 | 首次策略驱动的实车自主起立前 | **未关闭**：当前 reset 从接近名义站立构型开始，尚未训练或验证自主起立；不阻塞本轮纯仿真 V4 训练 |
| G-14 | ContractV4 文档和实现分别完成独立只读复核且无 P0/P1 | 四次 V4 正式训练前 | **未关闭**：首轮代码复核的四项 P1 已修正并通过端到端 smoke，等待同一独立审查 agent 复核；明确无 P0/P1 前禁止正式训练 |

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
