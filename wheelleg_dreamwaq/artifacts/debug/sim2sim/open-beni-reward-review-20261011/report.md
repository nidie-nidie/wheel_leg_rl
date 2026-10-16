# 妙动 OpenBeni 奖励源码核对（2026-10-11）

已找到妙动 Mondo Robotics 官方 [open-beni](https://github.com/Mondo-Robotics/open-beni) 仓库。核对提交 **4545b86a131d05519050df11a424368ef9f06b4f**（2026-09-24T07:54:51Z），包括速度任务、Jump任务、恢复配置和uv.lock锁定的mjlab 1.6.0源码。只阅读和保存参考副本，没有执行外部训练代码、安装依赖或修改当前工程。

这份公开代码提供平地速度跟踪、倒地恢复与跳跃。它的主要相关做法是：腿／轮平滑分项，加入实际运动代价，姿态容忍度随状态变化，正的速度跟踪奖励取决于轮支撑与车身状态。它没有给每次摆腿标注“必要动作”，在检查的公开训练代码中未见ROGER调权实现。

## 当前问题最相关的四个平滑项

腿动作：索引0–3，即L1、R1、L2、R2。轮动作：索引4–5，即LW、RW。第7个动作是头部yaw，未纳入这四个平滑项。

| 项目 | 腿 | 轮 | 我们当前配置 |
|---|---:|---:|---|
| 一阶动作变化惩罚 | −0.05 | −0.5 | 六维合并，−0.01 |
| 二阶动作变化惩罚 | −0.05 | −0.5 | 六维合并，−0.02（stop_reverse_smooth_v1） |

二阶公式仍是 Σ(a_t − 2a_(t−1) + a_(t−2))²，与当前action_smooth同形。变化在分组和动作取值口径；并不是引入了识别抽搐的新算法。[分组配置](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L390)、[公式](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/mdp.py#L198)。

这回答了“轮端是否还能参与平衡”：它保留轮端动作，并对轮端单独设置平滑代价。不能根据轮项权重大10倍就推断实际贡献大10倍，也不能据此保证移植后仍能平衡；代价大小取决于实际动作序列。

该公开版本没有在这四个平滑项内按“救车／停车／运动”豁免某帧动作；恢复与跳跃配置也没有改掉这四项。状态相关的放宽主要发生在姿态要求和其他阶段奖励中。

## Beni-Velocity-Flat 的21个有效配置项

| 奖励项 | 配置权重 | 原始指标／条件 | 证据 |
|---|---:|---|---|
| termination_penalty | -200 | 非时间上限的终止 | [L309](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L309) |
| track_linear_velocity | 2 | 水平速度误差平方 E；exp(−E/0.25) × 支撑因子 | [L310](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L310) |
| track_linear_velocity_sharp | 1 | 相同水平速度误差；exp(−E/0.02) × 支撑因子 | [L315](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L315) |
| track_linear_velocity_l1 | -2 | 水平速度误差绝对值之和，不乘支撑因子 | [L320](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L320) |
| track_yaw_velocity | 2.5 | 偏航角速度误差；exp(−e²/0.25) × 支撑因子 | [L325](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L325) |
| base_height | -100 | 机身高度低于0.11m的不足量 | [L330](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L330) |
| pose | 1 | 偏离默认关节姿态的指数奖励，容忍尺度随命令大小变化；乘支撑因子 | [L336](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L336) |
| leg_symmetry | -2 | 同侧序列中对应左右腿实际关节角度差的平方和 | [L349](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L349) |
| joint_pos_limits | -1 | 实际关节越过软位置限位的量 | [L356](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L356) |
| joint_torques | -2e-05 | 7个受控执行器力的平方和 | [L363](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L363) |
| joint_acc | -2.5e-07 | 7个受控关节实际qacc的平方和 | [L368](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L368) |
| leg_joint_vel | -0.0002 | 4个腿关节实际角速度平方和 | [L373](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L373) |
| base_tilt | -4 | 机身总倾斜角，0表示直立，π表示倒置 | [L379](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L379) |
| vertical_velocity | -5 | 竖直速度平方，原始代价值封顶1 | [L380](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L380) |
| roll_pitch_velocity | -0.1 | roll/pitch方向角速度平方和，原始代价值封顶20 | [L385](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L385) |
| leg_action_rate | -0.05 | 4维腿动作一阶差分平方和 | [L390](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L390) |
| leg_action_acc | -0.05 | 4维腿动作二阶差分平方和；episode_length_buf<3时为0 | [L395](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L395) |
| wheel_action_rate | -0.5 | 2维轮动作一阶差分平方和 | [L400](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L400) |
| wheel_action_acc | -0.5 | 2维轮动作二阶差分平方和；episode_length_buf<3时为0 | [L405](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L405) |
| undesired_contacts | -0.5 | 机身及腿部非法接触数量 | [L410](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L410) |
| body_contact | -3 | 机身／腿／头部是否发生接触 | [L415](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L415) |

每步通常是权重 × 原始指标 × dt；本配置dt=0.002×5=0.01s。表中的权重不是每步实际奖惩值。该配置整体替换了基础velocity任务的rewards字典，而非只追加这21项。

实际joint_acc来自锁定mjlab的MuJoCo qacc，不是我们本次诊断的20ms端点速度差。腿关节速度项只覆盖4腿；qacc和执行器代价覆盖7个受控通道。

## 它怎样给平衡与运动留空间

速度任务的pose使用速度相关容忍尺度：停车0.05、普通运动0.2、较大运动命令0.3rad。尺度越大，相同姿态偏差的奖励损失越小。它们是指数奖励的尺度，不是关节硬限位。

档位依据为命令水平线速度范数加上命令yaw角速度绝对值：低于0.05／0.05到1.5／至少1.5。它把不同量纲命令相加，不能把档位边界简单解释成纯粹的m/s阈值。源码见[pose参数](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_velocity_flat_env_cfg.py#L336)，锁定框架公式见本地[variable_posture](framework/mjlab/tasks/velocity/mdp/rewards.py)。

Jump实际先调用恢复配置，把pose改成有界惩罚：移动命令norm>0.05时尺度0.35，否则0.20；腿姿态误差使用实测参考姿态，另加头部项。near_upright在30°以内为1，从30°到80°逐渐减弱，80°及以上为0。因此倒地时减弱“回到标准腿姿态”的要求，给翻身恢复留出探索空间。这没有关闭动作平滑、总倾斜惩罚或终止惩罚。[恢复pose和near_upright](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/recovery.py#L30)。

Jump任务还会在跳跃阶段改变普通高度、姿态、竖直速度等项的适用程度，加入单独的跳跃姿态和角速度奖惩。[Jump配置](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L13)、[阶段函数](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/jump.py#L691)。

对我们当前问题的推断：可以借鉴按任务／状态调整姿态要求的思路，不必先逐帧判断动作“必要与否”。但这份代码没有证明我们现有8.8Hz振荡的成因，也没有提供可直接移植的ROGER阈值。

## 速度奖励与支撑状态绑定

正的水平速度跟踪和普通yaw跟踪乘standing_support：

- 两轮当前接触力都超过0.1N。
- 机身、腿、头没有近期接触地面。
- 高度因子从0.065到0.10m逐渐增加到1。
- 姿态因子在20°以内为1，到60°逐渐减少到0。

因此倒地滑动或身体拖地，即便速度数值接近命令，也可能得不到正的速度跟踪奖励。注意L1水平速度误差惩罚不乘这个因子；不是所有跟踪奖惩统一开关。[standing_support](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/mdp.py#L222)、[速度公式](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/mdp.py#L107)。

## Jump不是同一份基础奖励配置

恢复配置首先改变termination_penalty为−2500、base_height为−1500的高度误差项、pose为−1.5的有界姿态误差。leg_joint_vel保留原权重，但实际变为4腿角速度平方和，加上未负重轮子超过83.7758rad/s后的超速平方成本。仍是奖励成本，不是额外施加一个制动力矩。[configure_recovery_rewards](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/recovery.py#L85)、[joint_velocity](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/recovery.py#L75)。

然后Jump追加14项，合计35项；部分普通项换成阶段函数：

| Jump新增项 | 配置权重 | 证据 |
|---|---:|---|
| jump_heading | -2.5 | [L35](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L35) |
| large_tilt | -20 | [L37](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L37) |
| jump_roll_angle | -80 | [L39](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L39) |
| jump_pitch_angle | -30 | [L41](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L41) |
| jump_roll_rate | -4 | [L43](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L43) |
| jump_pitch_rate | -2 | [L45](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L45) |
| jump_height_tracking_l1 | -1000 | [L47](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L47) |
| jump_vel_tracking_l2 | -132.583 | [L49](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L49) |
| jump_knee_regulation | 200 | [L50](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L50) |
| jump_direction_progress | 20 | [L51](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L51) |
| jump_reverse_velocity_l1 | -20 | [L53](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L53) |
| single_foot_penalty | 5 | [L54](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L54) |
| jump_takeoff_progress | 200 | [L55](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L55) |
| jump_body_failure | -5000 | [L57](https://github.com/Mondo-Robotics/open-beni/blob/4545b86a131d05519050df11a424368ef9f06b4f/python/beni_rl/beni_jump_env_cfg.py#L57) |

这些Jump权重与阶段条件不应当作我们当前平地跟踪任务的直接建议；列出来是为了避免把基础速度配置误报成README示例训练的完整配置。

## 和我们当前工程的关键边界

| 项目 | OpenBeni公开配置 | 当前wheelleg_dreamwaq |
|---|---|---|
| 平滑分组 | 4腿／2轮分别加权 | 六维合并 |
| 平滑取值 | 原始策略动作，未做[−1,1]裁剪 | 裁剪后的六维动作 |
| 控制周期 | 10ms，100Hz | 20ms，50Hz |
| 腿动作缩放 | 0.25rad | 0.35rad |
| 轮动作缩放 | 20rad/s | 25rad/s |
| 姿态要求 | 默认关节姿态随命令放宽，恢复时进一步按倾斜减弱 | 有身体orientation和虚拟腿phi0对称；没有上述默认关节姿态项 |
| 左右对称 | 左右对应实际关节角差平方 | 虚拟腿phi0差 |
| 正速度奖励条件 | 轮支撑、无身体接触、高度与倾斜因子 | 当前compute_reward_terms没有该支撑因子 |
| 实际关节加速度 | MuJoCo qacc平方和，7受控通道 | 20ms速度差平方和，6受控通道 |
| 网络 | GRU策略／Critic | 当前DreamWaQ/CENet |
| 自动调权 | 固定配置系数和阶段条件，未见ROGER | 当前未实现ROGER |

动作口径已核对：runner的clip_actions=None；各动作项没有设置clip参数，锁定框架默认clip=None；ActionManager保存原始动作，再由动作项做scale/offset。因此它的−0.05不能直接当作我们−0.02的简单倍数。改变当前裁剪语义会涉及动作、history、checkpoint和导出契约，本次没有提出或实施该变更。

机器人、电机、力矩范围、默认姿态、被动膝弹簧也不同。可以借鉴奖励的组织方式，不能将数值直接照搬为可比较的惩罚强度。

## 对后续讨论的建议和限制

优先讨论腿／轮平滑分别加权，以及停车与运动采用不同姿态容忍度。我们已有实际dof_vel和dof_acc，需先明确是否仅改变分组、是否另加姿态要求，不能假设这些实际运动项目前完全不存在。

保持轮端参与平衡，并用实际车身倾斜、跟踪和摔倒结果验证。分组本身并不会自动解决奖励冲突。停车pose更严也可能妨碍必要的重心调整，需验证当前机器人参考姿态是否可用。

OpenBeni目前提供了可阅读的固定奖励设计，没有给出当前机器人抽搐的健康样本或ROGERτ。开源代码和演示效果也不能单独证明其商业产品所有运行条件、奖励和策略均与公开版本相同。

## 证据留存与核验

保存28个仓库文本源码／模型／锁文件，全部SHA1 Git blob身份与固定提交的tree一致；所有Python文件只做AST解析，没有导入或执行。mjlab 1.6.0 wheel依据uv.lock SHA256核验后只读提取6个框架源码文件。21基础项和14Jump追加项从AST静态提取，四个平滑系数再次核对。

93个现有生产文件、模型、导出和正式架构哈希与上一轮诊断身份快照一致，已有tracked改动文件名单未新增。此次只新增参考副本与研究报告，没有修改训练配置或启动训练。

[奖励项机器提取结果](reward-terms.json) · [来源和核验](verification.json) · [固定提交](commit.json) · [锁定框架来源](framework-manifest.json)
