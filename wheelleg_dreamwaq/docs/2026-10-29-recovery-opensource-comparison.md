# 当前倒地起身策略与开源起身实现的完整对比

核查日期：2026-10-29。比较对象为 **FTSR/getup_gym 的 BipedalWheeled 起身任务、DeepMind Go1Getup、论文作者的 A1 跌倒恢复任务**。网络、观测、动作、奖励和训练入口均按各自起身任务的实际调用核查。

本机依据为已停止训练的 `RecoveryResolvedConfigV3`，最后完成更新 6243，最后完整模型 `model_6200.pt`。本次研究只阅读源码、下载参考文件并分析已有的 12 初态测评；没有修改生产代码、训练配置或再次启动训练、GPU 测评。源码身份来自这次模型与测评记录：`F06D42C9DFBD3D5D8CFC06E11414A83D1531C3BDD22A25087B2088B6247C0FB8`。

## 1. 比较版本与证据

| 对象 | 固定版本与实际入口 | 证据限制 |
| --- | --- | --- |
| 本机 Recovery | `training/recovery/ppo_runner.py`、`env.py`、实际续训配置；RSL-RL 3.1.2 | 12 个留出初态已有实测；本次没有建立各差异的因果消融 |
| FTSR 公开轮腿起身 | `getup_gym` 提交 `238d2962a0fee8e3da4da914cadcea2be00ce402`；注册 `BipedalWheeled → JROwheel + GetupCfg + GetUpPPO` | 公开代码与论文有明显区别；没有执行作者仿真 |
| Go1Getup | MuJoCo Playground `12ad7f3842791ced39f27e5ac96d066bbe361653`；`brax_ppo_config("Go1Getup")` | Brax 只要求 `>=0.14.2`；本文默认行为采用另行固定的 Brax `c8605294264a7b7546268adbb4beabc9eb624000` |
| A1 跌倒恢复 | `quad-fall-recovery-rl` 提交 `7d22ba850559fc8e4367d2f3c00bc9cc9b6a27c8`；`PPO("MlpPolicy")` | SB3 未锁定版本；仅显式配置可直接归属于作者，库默认参考 SB3 v2.7.0；该提交未包含训练权重 |

固定源码、SHA256 与逐项阅读笔记分别保存在 [FTSR 核查笔记](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/recovery-opensource-comparison-20261029/ftsr_findings.md)、[Go1Getup 核查笔记](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/recovery-opensource-comparison-20261029/references/go1/go1_findings.md)、[A1 核查笔记](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/recovery-opensource-comparison-20261029/references/a1/a1_findings.md)。

## 2. 神经网络与学习方式

| 项目 | 本机 Recovery | FTSR 公开起身 | Go1Getup + 指定 Brax | A1 跌倒恢复 |
| --- | --- | --- | --- | --- |
| 基础算法 | PPO，加辅助成本价值网络与混合优势 | PPO，加教师学生潜变量模仿 | PPO | PPO |
| Actor 主网络 | **111→256→128→64→6** | **46→512→256→128→8** | **42→512→256→128→24**，包含 12 均值与 12 尺度 | **30→256→256→16→12** |
| 奖励 Critic | **35→256→128→64→1** | **234→512→256→128→1** | **94→512→256→128→1** | **30→256→256→16→1** |
| 激活 | ELU | ELU | Swish | ReLU |
| 历史 | **五帧直接拼接** | 学生历史编码为 12 维 latent | 当前帧 | 当前帧 |
| 教师编码器 | 没有 | **20→256→128→12** | 没有 | 没有 |
| 学生编码器 | 没有 | **170→256→128→12**，MSE 学习教师 latent | 没有 | 没有 |
| 成本 Critic | 35→256→128→64→1；零辅助时停止成本更新 | 选定算法路径没有独立成本 Critic | 没有 | 没有 |
| 观测归一化 | 固定物理尺度与部分裁剪；没有运行均值方差 | 固定尺度与观测裁剪 | Actor/Critic 运行均值方差 | 关节量按范围映射；没有 VecNormalize |
| 权重初始化 | PyTorch Linear 默认初始化 | PyTorch Linear 默认初始化 | 指定 Brax 默认 LeCun uniform | 参考 SB3 默认正交初始化；Actor 输出增益 0.01 |
| Actor 参数量 | 70,214 | 主 Actor 之外还要计算编码器 | 189,336 | 宽度分配与本机不同 |

本机“前馈”指这个 MLP 的计算方式。五帧数据直接作为输入，不存在一个额外的前馈估计网络把速度或 latent 传入 Actor。当前恢复策略没有 DreamWaQ 的 CENet/VAE。FTSR 的学生编码器具有不同的监督目标，也不能简单称为 DreamWaQ 编码器。

源码：[本机 runner](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/ppo_runner.py:29)、[FTSR 编码器配置](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/envs/bipedal_wheeled/config.py#L360)、[FTSR 输入拼接](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/runners/on_policy_runner.py#L130)、[Go1Getup 专属网络配置](https://github.com/google-deepmind/mujoco_playground/blob/12ad7f3842791ced39f27e5ac96d066bbe361653/mujoco_playground/config/locomotion_params.py#L86)、[Brax 网络与分布](https://github.com/google/brax/blob/c8605294264a7b7546268adbb4beabc9eb624000/brax/training/networks.py#L539)、[A1 网络配置](https://github.com/yerenkl/quad-fall-recovery-rl/blob/7d22ba850559fc8e4367d2f3c00bc9cc9b6a27c8/config.py#L22)。

## 3. Actor 究竟看到了什么

### 本机

每帧 22 维 = 机身角速度 3 + 重力方向 3 + 四个腿关节相对 reference 的位置 4 + 全部关节速度 6 + 上一原始动作 6。五帧形成 110 维，再加当前辅助系数 alpha，合计 **111 维**。相邻帧间隔 20 ms，最旧与最新相隔 80 ms。

角速度乘 0.25、关节速度乘 0.05，其他尺度与裁剪见 [observations.py](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/observations.py:19)。**上一原始动作没有裁剪**，因此该通道的尺度会随策略输出增长。

Critic 使用当前干净 22 维，再加线速度 3、高度 1、力矩估计 6、确认时长比例 1、剩余时间比例 1、alpha 1，共 35 维。Actor 没有直接读取高度、线速度、轮子接触力。确认时长在本次未标定门控下始终无法积累。

### 其他起身任务

| 输入信息 | FTSR 公开起身 | Go1Getup | A1 跌倒恢复 |
| --- | --- | --- | --- |
| 本体观测 | 34：线速度零占位 3、角速度 3、重力 3、指令 3、六腿关节位置误差 6、速度 8、上一动作 8 | 42：角速度 3、重力 3、位置误差 12、速度 12、已存动作 12 | 30：位置 12、速度 12、重力 3、世界角速度 3 |
| Actor 的接触/高度信息 | 教师 latent 直接含轮子三维接触力 6、高度 1、root state 13；学生由历史模仿 | Actor 没有这些显式信息 | Actor 没有这些显式信息 |
| Critic 特权信息 | 本体 34 + 地形高度 187 + 高度 1 + 教师 latent 12 = 234 | 本体 42 加干净 IMU、速度、位置、力矩、高度，合计 94 | 与 Actor 相同 |
| 历史辅助学习 | 教师 latent 与学生 latent 的 MSE | 没有 | 没有 |
| 观测噪声 | 配置开启部分本体噪声与物理随机化 | 默认 gyro ±0.2、gravity ±0.05、q ±0.03、dq ±1.5 | 关节噪声注释关闭；奖励内部另有噪声 |

所以，五帧历史并非全部开源起身方法共有的要求。Go1Getup、A1 使用单帧；最接近本机的 FTSR 轮腿任务使用历史编码与特权监督。是否需要编码器不能单凭这三个结构决定。

源码：[FTSR 实际观测](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/envs/bipedal_wheeled/env.py#L29)、[FTSR 教师输入](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/envs/bipedal_wheeled/env.py#L102)、[Go1Getup 实际观测](https://github.com/google-deepmind/mujoco_playground/blob/12ad7f3842791ced39f27e5ac96d066bbe361653/mujoco_playground/_src/locomotion/go1/getup.py#L231)、[A1 实际观测](https://github.com/yerenkl/quad-fall-recovery-rl/blob/7d22ba850559fc8e4367d2f3c00bc9cc9b6a27c8/quad/envs/quad_env.py#L92)。

## 4. 动作、探索与物理执行

| 项目 | 本机 Recovery | FTSR 公开起身 | Go1Getup | A1 跌倒恢复 |
| --- | --- | --- | --- | --- |
| 机器人 | 六维：四腿执行关节、两轮，含闭链约束 | 八维：六腿关节、两轮，有侧摆关节 | 十二个位置执行关节的四足 | 十二个位置执行关节的四足 |
| 分布 | 无边界 Gaussian；状态无关 log_std | Gaussian；初始 std=1，训练直接学习 std | 状态相关 Gaussian 后接 tanh | 参考 SB3 为 Gaussian，送入环境前裁剪 |
| 动作边界 | **原始动作不裁剪** | ±50，边界较宽 | **tanh 后 ±1** | **±1**，映射本身也饱和到端点 |
| 腿目标 | `q_ref + 0.35a` | 默认站姿 + 侧摆 `0.2a`、俯仰 `0.6a` | **`q_current + 0.5a`** | 映射至完整关节范围的绝对目标 |
| 轮目标 | **`25a rad/s`** | `1a rad/s` | 无轮子 | 无轮子 |
| 动作控制周期 | 20 ms | 20 ms | 20 ms | 100 ms |
| 物理周期 | 5 ms | 5 ms | 4 ms | 10 ms |
| PD | 腿 120/4；轮速度增益 0.6 | 侧摆 80.4/3.5；俯仰约 101/2.5；轮增益 1 | 35/0.5 | PyBullet POSITION_CONTROL；未显式指定 gain |
| 力矩边界 | 腿 18、轮 9 N·m | 腿 120、轮 40 N·m | 髋/外展 23.7、膝 35.55 N·m | hip 20、thigh/calf 55 N·m |
| 其他执行限制 | 有物理关节/闭链/穿透失败；无原始目标角边界 | 显式计算 PD，再按 URDF 限幅 | XML ctrlrange 继承物理关节范围 | 有速度上限；动作间子步插值 |

这些力矩上限属于不同质量、杆长和机构，无法直接比较力量是否足够。六维闭链轮腿的可行翻身动作也不能直接用八维轮腿或四足动作代替。

本机 6200 模型的 std 为 **[4.025, 2.258, 3.851, 2.694, 4.639, 4.671]**，初始均为 0.5。换成目标单位，腿部探索标准差约 **0.79–1.41 rad**，轮速探索标准差约 **116 rad/s**。这是训练 Gaussian 的标准差；确定性测评采用均值动作。

已有确定性测评的 4066 个首回合控制样本中，最大腿目标相对 reference 的偏移为 **9.99 rad**，最大轮速目标绝对值为 **1004.58 rad/s**。这些是命令目标，不能解释为机器人实际达到该速度或角度。日志中的腿/轮 PD 力矩估计触及限幅的比例为 **92.8%/89.3%**。IsaacLab 隐式执行器的 `applied_torque` 是近似 PD 力矩，不能据此宣称求解器实际力矩每个子步均已实测饱和。

动作目标无边界、探索增长、较大的轮速尺度、原始动作反馈通道共同构成值得关注的机制差异。多个巨大动作可能对应相近的限幅控制效果，动作的数值差异仍会进入观测和平滑处罚。当前证据能够确认这一现象，尚不能单独证明它是站稳失败的唯一原因。

来源：[本机动作转换](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/control.py:15)、[本机执行器](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/wheelleg_dreamwaq/assets/wheelleg.py:98)、[隐式力矩估计说明](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/actuators/actuator_pd.py:124)、[本次数值分析](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/recovery-opensource-comparison-20261029/local_evidence.json)、[Brax tanh 分布](https://github.com/google/brax/blob/c8605294264a7b7546268adbb4beabc9eb624000/brax/training/distribution.py#L148)、[FTSR 动作控制](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/envs/bipedal_wheeled/env.py#L252)、[A1 动作执行](https://github.com/yerenkl/quad-fall-recovery-rl/blob/7d22ba850559fc8e4367d2f3c00bc9cc9b6a27c8/quad/envs/quad_env.py#L69)。

## 5. 奖励函数：公式、权重和实际可达性

### 5.1 本机全部奖励

记 `g_z` 为机身重力方向的 z 分量，直立时为 -1；`c=(1-g_z)/2`；`G=1[倾斜角<0.5rad]`；`dt=0.02`。`Δ²a=a_t-2a_(t-1)+a_(t-2)`。下表已经代入实际 dt 和权重。

| 奖励项 | 实际每步公式 |
| --- | --- |
| upright | `+0.04c` |
| progress | `+2(0.99φ_next-φ_prev)`，非终态 `φ_next=c`，成功或失败终态设为 0 |
| height | `+0.02G exp(-((h-0.20)/0.04)²)` |
| stable | `+0.02G exp(-E_pose-E_omega-E_wheel)` |
| torque | `-0.00004 mean((τ/τ_limit)²)` |
| leg_smooth | `-0.0002 mean((Δ²a_leg)²)`，历史完整时启用 |
| wheel_smooth | `-0.0002 mean((Δ²a_wheel)²)`，历史完整时启用 |
| time | `-0.002` |
| success | 成功时 `+10` |
| failure | 失败时 `-10`，含 8 秒未完成 |

其中：`E_pose=mean(((q_leg-q_nominal)/0.20)²)`；`E_omega=sum((omega/0.5)²)`；`E_wheel=mean((wheel_velocity/2)²)`。stable 等价于三种指数奖励相乘，任意一种误差很大都会压低整项。它没有独立的腿关节速度项。整套奖励允许负值，没有最终非负裁剪。

**本次 success 无法触发。** 实际配置的 `support_force_n`、`target_difference_limits`、`torque_difference_limits` 都为 null。环境按未标定路径设置轮子支撑 false、其他部位支撑 true、交接动作不兼容；累计成功确认条件始终不成立。因此，虽然公式有 +10，本次训练没有可获得的成功奖金，也没有通过成功门控进入后续阶段。

这并不意味着其他稠密奖励完全不能学到站立；Go1Getup、A1 同样没有成功奖金。这里需要区分“本机设置的成功分支不可达”和“所有起身方法都必须有成功奖金”。

已有 12 初态测评中各项每步均值：upright **0.03688**，height **0.01750**，stable **3.64×10⁻¹²**。最后训练更新 6243 的对应值为 **0.03677、0.01566、2.50×10⁻¹⁰**。站稳项在这些实际样本中的贡献接近零，朝向和高度仍然提供主要正收益。测评中多数非正面初态最后约 20–25° 倾斜，仍处于 28.65° 的 height 奖励门内，距离 14.32° 的正式姿态要求有差距。

progress 的负值来自折扣势函数差，不能据负均值认定符号错误。静止、非终态且 c 恒定时，progress 为 `-0.02c`，与 upright 合计 `+0.02c`；结束时另有势函数归零与 failure -10。因此 TensorBoard 的总回报同时受姿态、持续时间、平滑代价、终止项和势函数影响，无法直接换算成起身成功率。

来源：[全部公式](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/rewards.py:34)、[固定参数](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/contracts.py:52)、[未标定实际路径](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/env.py:409)、[监督判据](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/batched_supervisor.py:281)。

### 5.2 Go1Getup 的全部九项

记 `e_g=||g-[0,0,-1]||²`，`q0` 为 home，`a` 为 tanh 后动作。加权求和后统一乘 dt，并裁剪为 `[0,10000]`。

| 项目 | 原始公式 | 权重 |
| --- | --- | ---: |
| 朝向 | `exp(-2e_g)` | +1 |
| 高度 | `exp(min(h,0.275))-1` | +1 |
| 站姿 | `1[e_g<0.01] exp(-0.5 sum((q-q0)²))` | +1 |
| 静止 | `1[e_g<0.01 且 h>0.270] exp(-0.5 sum(a²))` | +1 |
| 动作变化 | `sum((Δa)²)+sum((Δ²a)²)` | -0.001 |
| 关节位置边界 | 超出 95% 软范围的线性偏差之和 | -0.1 |
| 力矩 | `sqrt(sum(τ²))+sum(abs(τ))` | -1e-5 |
| 加速度 | `sum(qacc²)` | -2.5e-7 |
| 速度 | `sum(max(abs(dq)-2π,0)²)` | -0.1 |

它把站姿、动作静止、速度、动作变化分开计算；本机把姿态/角速度/轮速合在一个窄指数中。Go1 的直立门约 5.73°，比本机奖励门严格，但其站稳子项没有同时乘上多种速度指数。没有成功奖金或成功提前终止，站住后继续积累收益。[实际奖励与门控](https://github.com/google-deepmind/mujoco_playground/blob/12ad7f3842791ced39f27e5ac96d066bbe361653/mujoco_playground/_src/locomotion/go1/getup.py#L300)

### 5.3 A1 的完整奖励

`r=0.3G+0.3H+K(0.3H+0.6P-0.1(1-C))+0.05(1-D)+0.005(1-V)+0.05B`，不乘 dt。

| 符号 | 源码含义 |
| --- | --- |
| G | 重力向量距离加高斯噪声后，从 [0,2] 线性映射到 [1,0] |
| H | `min(z/0.31,1)`，没有下限 |
| K | `1[G>0.7]`，无噪声约对应倾斜 <34.9° |
| P | 每关节与默认姿态的接近程度平方，再取平均 |
| C | 四足接触比例 |
| D | 归一化动作一阶差分平方均值；实际赋值顺序导致它约为零 |
| V | 归一化关节速度绝对值均值 |
| B | 线速度加噪声后，`exp(-2||v||²)` |

基础朝向和高度在全部阶段可得；接近朝上后增加姿态与接触奖励。没有成功/失败终止奖励。代码中的动作平滑项实际退化为约 +0.05 常量，不能将“配置有该项”当作实际已经平滑。[完整奖励源码](https://github.com/yerenkl/quad-fall-recovery-rl/blob/7d22ba850559fc8e4367d2f3c00bc9cc9b6a27c8/quad/envs/reward.py)、[奖励前更新动作记录](https://github.com/yerenkl/quad-fall-recovery-rl/blob/7d22ba850559fc8e4367d2f3c00bc9cc9b6a27c8/quad/envs/quad_env.py#L84)

### 5.4 FTSR 公开起身的全部启用系数

配置提供四组，实际每步选择第 2、3、4 组。统一乘控制周期 0.02；以下保留原始系数。“关闭”表示该组没有启用。

| 奖励函数 | 第 2 组 | 第 3 组 | 第 4 组 |
| --- | ---: | ---: | ---: |
| termination | -200 | -100 | -500 |
| base_orientation_l2 | -2.6 | -2 | -15 |
| base_orientation_z_l2 | -0.6 | -1 | 关闭 |
| base_height_exp | +4 | +6 | +5 |
| torques_l2 | 关闭 | -1e-6 | -1e-5 |
| dof_acc_l2 | -2.5e-8 | -2.5e-6 | -2.5e-6 |
| dof_vel_l2 | 关闭 | 关闭 | -0.004 |
| action_rate_l2 | -0.06 | -0.02 | -0.1 |
| action_smoothness_l2 | 关闭 | 关闭 | -0.02 |
| dof_pos_bias_l2 | -0.06 | -0.14 | -0.12 |
| two_leg_bias_l2 | -0.02 | -0.02 | 关闭 |
| no_fly_l2 | -2 | 关闭 | -0.2 |
| wheel_contact_force | +4 | 关闭 | 关闭 |
| lin_vel_xy_exp | 关闭 | 关闭 | +7 |
| ang_vel_yaw_exp | 关闭 | 关闭 | +4 |
| feet_distance_l2 | 关闭 | 关闭 | -2 |
| max_velocity_l2 | 关闭 | 关闭 | +1 |
| dof_pos_limits | 关闭 | 关闭 | -5 |
| torque_limits | 关闭 | 关闭 | -0.05 |
| joint_power_l2 | 关闭 | 关闭 | -1e-4 |

原始函数的实际数学含义如下；函数名称与公式不完全一致。

| 函数 | 实际计算 |
| --- | --- |
| base_orientation_l2 | `0.02 sum(g_xy²)+2(g_z+1)²-clip(exp(-sum(g_xy²)/0.25),0,0.1)` |
| base_orientation_z_l2 | 倒置指示量 `1[-g_z<-0.2]` |
| base_height_exp | `exp(-abs(h-h_target)/0.12)` |
| dof_pos_bias_l2 | 左侧平方误差与右侧绝对误差混合，三关节类别求和后再平方；仅双轮接触时启用 |
| two_leg_bias_l2 | 左右两腿对应侧摆、股部、胫部关节角度差平方和 |
| no_fly_l2 | 两轮没有同时满足竖直力 >1N 时为 1 |
| wheel_contact_force | `exp(-0.05 abs(sum(abs(Fz_wheels))-10m))` |
| torques_l2 / dof_vel_l2 | 全部力矩平方和 / 六腿关节速度平方和 |
| dof_acc_l2 | 全部关节 `(dq_prev-dq)/dt` 平方和 |
| action_rate / smoothness | 全部动作一阶 / 二阶差分平方和 |
| lin_vel_xy_exp | `exp(-||v_xy-command_xy||²/0.12)`，要求高度 >0.6m |
| ang_vel_yaw_exp | `exp(-(omega_z-command_yaw)²/0.12)` |
| feet_distance_l2 | 平面轮距离开 [0.32,0.38]m 的偏差，两侧各裁剪到 [0,1] |
| max_velocity_l2 | 速度与指令方向相关的分段 ±1；代码单独处理正负指令，不能按函数名解释成标准 L2 |
| dof_pos_limits | 关节超出软上下界的线性偏差和 |
| torque_limits | 超出配置软力矩上限的线性偏差和 |
| joint_power_l2 | `sum(abs(dq)*abs(τ))`，仅高度 <0.7m 启用 |
| termination | `reset & ~timeout`；默认普通超时不产生此项 |

出处：[全部阶段系数](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/envs/bipedal_wheeled/config.py#L208)、[全部实际函数](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/common/reward_functions.py)、[dt 聚合](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/common/reward_manager.py#L46)。

FTSR 这里的速度跟踪属于同一个倒地恢复策略的最后阶段。与本机最直接的差异包括：**轮子实际承重、腿间对称、独立姿态处罚、分阶段高度目标，以及站立后继续控制的目标**。本机成功判据包含轮子支撑要求，但当前没有对应可达的轮子承重奖励；“判据里存在”和“训练中有可获得的反馈”需要分别看待。

## 6. FTSR 论文、公开实现和本机适配

| 机制 | FTSR 论文 | 选定公开代码 | 本机实际执行 |
| --- | --- | --- | --- |
| 学习阶段 | 扶正、站立、行走，按群体高度比例推进 | 实际组 2→3→4；每步按批量高度与 force_scale 重新选择 | 单套恢复奖励，只有辅助系数按时间撤除 |
| 目标高度 | 阶段目标 | 0.30/0.45/0.60/0.75m，按整批平均高度更新 | 固定 0.20m |
| 竖直辅助 | 指数高度差与时间系数 | 线性高度差、400N 上限 | 指数非负高度差，上限半倍体重约 21.56N |
| 辅助转矩 | 相对旋转与高度系数 | 四元数分量构造环境坐标系 x 转矩 | 关闭 |
| 成本学习 | 辅助成本与受约束 PPO | force 数据没有从环境写入 runner，选定路径保持零 | 实际力成本、独立成本 Critic、GAE 与混合优势 |
| 教师学生 | 教师特权 latent 与学生历史学习 | 有对应网络与 MSE 路径 | 没有 |
| 初始能力 | 约 200 更新基本行走预训练 | 默认 resume=False，没有自动执行论文预训练 | 恢复 Actor 从 fresh 学习；冻结行走候选网络不初始化它 |
| 预算 | 恢复 8000 更新，4000 环境 | 默认最大 1500 更新 | 用户停止于 6243 更新 |

本机辅助公式为：`F_up=0.5mg·alpha·(1-exp(-20 max(0.20-h,0)))`，`alpha=max(1-k/72000,0)`；k 为控制步，3000 次更新以后为零。辅助作用于 base_link 的 COM、方向世界向上，没有横向力或辅助转矩。当前 4000→6243 续训与 6200 测评均为零辅助。

公开代码的 force 缺口仅针对固定提交：runner 用 `extras.get("force")` 取数据，而轮腿子类与父类环境文件没有写入该字段；storage 未得到数据时维持零。不能据此否定作者论文实验，也不能把这个提交当成完整论文算法已连通的证明。[公开施力函数](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/envs/legged_robot.py#L1280)、[成本数据路径](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/storage/rollout_storage.py#L96)

论文的完整方法确实同时使用分阶段奖励、教师学生和辅助训练。我们原定适配范围保留本机奖励与网络，因此当前不能称为完整 FTSR 复现。[论文方法与训练设定](https://arxiv.org/html/2606.14270v1)、[本机辅助公式](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/guidance.py:12)

## 7. PPO 超参数、初态、课程与预算

| 参数 | 本机 | FTSR 公开起身 | Go1Getup + 指定 Brax | A1 显式配置 |
| --- | --- | --- | --- | --- |
| 环境数 | 4096 | 4000，其中 3000 教师 | 8192 | 8 |
| 每环境采样长度 | 24 | 24 | 20 | 1024 |
| 每批交互样本 | 98,304 | 96,000 | 163,840 | 8192 |
| 每批重复 | 5 epochs×4 minibatches | 5×4 | 4×32 | 10×32 |
| 每 minibatch transitions | 24,576 | 24,000，统一随机抽取混合样本 | 5120，即 256 条 20 步序列 | 256 |
| gamma / lambda | 0.99 / 0.95 | 0.99 / 0.95 | 0.97 / 0.95 | 0.97 / 参考默认 0.95 |
| 策略 clip | 0.2 | 0.2 | 指定 Brax 默认 0.3 | 参考默认 0.2 |
| 学习率 | 初始 1e-3，KL 目标 0.01 自适应 | 同类自适应参数 | 固定 3e-4 | 固定 3e-4 |
| entropy coefficient | 0.01 | 0.01 | 0.01，作用于经过 tanh 的分布 | **0** |
| value loss | clipped，系数 1 | clipped，系数 1 | 不裁剪，实际 0.25×MSE | 系数 0.3；参考默认不裁剪 |
| gradient norm | 1 | 1 | 1 | 0.7 |
| advantage normalization | 混合优势统一标准化 | 标准化；教师学生路径详见源码 | 开启 | 参考默认开启 |

相同的 entropy 系数在有界和无边界分布上的效果不同，不能只按系数数值比较。也不能把熵项与原始每步奖励直接比较来证明它“压过奖励”，PPO 还涉及优势标准化与目标函数尺度。

| 训练环境因素 | 本机 | FTSR 公开起身 | Go1Getup | A1 跌倒恢复 |
| --- | --- | --- | --- | --- |
| 初态 | 48 个训练记录，六类固定比例；测试 12 个记录留出 | 默认倒地 root 四元数、root 速度扰动、关节扰动 | 60% 随机掉落、40% home 站姿；初始 settling | 随机四元数与关节角，从 0.5m 沉降 |
| 站姿初态覆盖 | near_upright 类 10% | 初始化与阶段机制不同 | home 40% | 随机覆盖 |
| 课程 | 实际只有一级，概率 [0.18×5,0.10]；没有可晋级的更高一级 | 奖励与目标高度动态切换 | 无多阶段奖励切换 | 无阶段课程 |
| 物理随机化 | **enabled=false**，观测噪声也随之关闭 | 默认部分摩擦/质量/COM/恢复系数随机化开启 | 域随机化训练旗标默认关闭；观测噪声独立开启 | 没有动力学随机化 |
| 回合时限 | **8s** | **20s**；10s 趴地终止开关关闭 | **6s** | **约 102.4s** |
| 成功提前终止 | 有分支，本次未标定而不可达 | 没有持续站稳/交接成功终止 | 没有 | 没有 |
| 其他终止 | 闭链、关节限制、过量穿透等物理失败 | 默认接触立即终止列表为空 | 能量阈值默认 infinity | 主要只有固定时间 |
| 超时语义 | 真终态，失败 -10，无超时 bootstrap | timeout 排除 termination 惩罚 | EpisodeWrapper 截断 | 写入 terminated |

本机的近直立初态也没有保证已经满足全部站稳条件。Go1 的 40% home 覆盖则直接增加了在站立区域学习保持动作的样本。它是否比本机分布更有效属于待验证解释，不能仅靠比例建立因果。

本机 6200 模型已采样 **609,484,800** transitions；停止时全局 6243 更新累计 **613,711,872**。Go1Getup 名义预算 **50,000,000**，指定 Brax 调度静态推导取整至 **52,428,800**；A1 名义 **4,915,200**；FTSR 论文 4000×24×8000 = **768,000,000**，公开默认 1500 更新则为 **144,000,000**。这些涉及不同机构、控制周期、成功定义与奖励，无法换算为相同难度。仅说“训练了五千轮”不足以评价采样是否合理。

来源：[本机实际配置](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/paper-guidance-continuation-20261029-quickstart/training_config.json)、[本机实际课程](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/bootstrap-6144-20261022/curriculum.json)、[FTSR PPO 配置](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/envs/bipedal_wheeled/config.py#L360)、[Go1Getup PPO 分支与继承值](https://github.com/google-deepmind/mujoco_playground/blob/12ad7f3842791ced39f27e5ac96d066bbe361653/mujoco_playground/config/locomotion_params.py#L30)、[Brax 其余默认值](https://github.com/google/brax/blob/c8605294264a7b7546268adbb4beabc9eb624000/brax/training/agents/ppo/train.py#L176)、[A1 显式参数](https://github.com/yerenkl/quad-fall-recovery-rl/blob/7d22ba850559fc8e4367d2f3c00bc9cc9b6a27c8/config.py#L9)。

## 8. 测评结果能够解释到什么程度

当前模型在 12 个留出初态的确定性零辅助测评中：5 个初始倒置案例全部翻到朝上半球；**倾斜角、COM 高度、机身角速度这三个诊断条件也没有任何一个控制步骤同时满足**；10 个案例到 8 秒，两个正面案例发生穿透/闭链物理失败。本次 standalone 测评未达到稳定站立，策略交接能力尚未验证。初态记录、轨迹和物理内容与训练分离，但没有独立随机化动力学参数。正式成功率由于门控未标定仍为 null，独立运动学诊断未达标为 0/12。[测评汇总](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/recovery-evaluation-20261029-stop6200/results/summary.md)

| 差异 | 已经确认的事实 | 对失败原因的解释限度 |
| --- | --- | --- |
| 成功门控 | 本次 +10 与成功退出无法触发 | 任务中的成功分支没有参与学习；独立运动学诊断同样失败，因此问题也不局限于统计门控 |
| stable 奖励 | 训练最后一批与测评中均几乎为零 | 站稳反馈在这些样本中很弱；尚未证明替换公式就能成功 |
| 动作范围与探索 | 巨大目标、增大的 std、PD 估计频繁触及限幅 | 有界控制效果与无界动作数值存在明显脱节；没有量化单独改变边界的收益 |
| FTSR 方法覆盖 | 当前缺少教师学生与阶段奖励、承重/对称目标、预训练初始化 | 当前仅借鉴部分辅助机制，不能期待自动获得整套论文的能力 |
| 初态与课程 | 单级六类比例、10% near_upright、固定名义动力学 | 站稳保持训练覆盖与 Go1、FTSR 不同；无法从代码单独确定最佳比例 |
| 网络宽度 | 本机 Actor 小于 Go1/FTSR，有五帧直接输入 | 容量是结构差异，但现有证据不足以把失败归因于容量或五帧是否编码 |
| PPO 参数 | 与 FTSR 多数基础参数相近，采样已超过 6 亿 | 单独增加更新预算没有已证实的必要性；当前不能给出“再训练多少就会站稳”的承诺 |

## 9. 开源代码自身的限制

静态阅读还发现若干会影响复现解释的细节：FTSR 历史每个控制步有两处追加、reset 未清理对应行，公开 force 成本数据未接通；其混合 minibatch 随机排序后仍用前 num_teacher 行处理教师 latent，行号已无法代表原教师环境身份。Go1 在构造下一观测后才更新已存动作通道，并采用缓存自动复位；A1 平滑项赋值顺序使其退化，测试还有“奖励达到阈值就保持上一动作”的额外规则。各项定位与完整来源见三个核查笔记。

这些实现均提供有价值的起身设计对照；其公开身份、演示或论文成功率不能代替在本机闭链轮腿上的验证。本报告没有把不同项目的回报数值当作统一成功尺度，也没有以源码阅读宣称已经复现作者结果。
