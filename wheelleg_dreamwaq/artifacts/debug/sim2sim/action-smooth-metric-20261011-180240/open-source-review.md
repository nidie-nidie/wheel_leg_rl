# 抑制腿部抽搐：开源源码核对（2026-10-11）

结论：建议先把腿和轮的统计/自适应对象分开，保留已有二阶差分公式作为腿指令平滑候选，再用四腿实际角加速度检验机械效果。当前证据不要求换成复杂算法，也不支持自动确定唯一合格阈值。此为研究建议，未修改正式架构、奖励或控制代码。

## 直接相关的轮腿参考

### Wheel-Legged-Gym

核对固定提交 c354431e5633eb98d9663fa3fbe444201d2de383。

- [_reward_action_smooth](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/envs/base/legged_robot.py#L1677)：仅取动作索引 0、1、3、4，二阶差分平方和；不包含索引 2、5。
- 同文件 _reward_action_rate（1673行）包含所有动作；_reward_dof_acc（1669行）包含所有关节，不能说该工程所有平滑项都只处理腿。
- [_compute_torques](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/envs/base/legged_robot.py#L670)：索引 2、5 是轮速度目标，其余四个是腿位置目标。
- VMC 分支 0、3 是虚拟腿角度目标，1、4 是虚拟腿长度目标，2、5 是轮速度目标；同样继承只含四个腿动作的 action_smooth。因此这里可借鉴的是分组方式，不能宣称其动作物理意义与本工程完全相同。
- 任务平滑系数来自固定 rewards.scales，并非 ROGER；reward_scale 会乘 dt。其动作裁剪、控制模式、尺度与本工程不同，不能直接照抄数值或阈值。

### DIABLO

核对 DDTRobot/Col_DIABLO_Issac_RL 固定提交 590085a027fc9e20b1d7e976714cd195488b96b6；README 说明其基于 Wheel-Legged-Gym，不能把它当作完全独立的两份实验验证。

- [_reward_action_smooth](https://github.com/DDTRobot/Col_DIABLO_Issac_RL/blob/590085a027fc9e20b1d7e976714cd195488b96b6/wheeled_bipedal_gym/envs/base/wheeled_bipedal.py#L1862)：同样只计算索引 0、1、3、4。
- [_compute_torques](https://github.com/DDTRobot/Col_DIABLO_Issac_RL/blob/590085a027fc9e20b1d7e976714cd195488b96b6/wheeled_bipedal_gym/envs/base/wheeled_bipedal.py#L791)：位置目标排除轮索引 2、5，速度目标只保留轮索引 2、5。
- _reward_action_rate（1856行）和 _reward_dof_acc（1852行）仍是全维。
- 存在轮速惩罚函数等额外项；函数存在不代表全部任务启用。此轮不建议复制其轮速惩罚、动作惯性或前馈等控制实现。

## 通用腿式参考

- [ETH legged_gym _reward_dof_acc](https://github.com/leggedrobotics/legged_gym/blob/8fa29acc6fd1910c3d9659eef6310bdd301cde0a/legged_gym/envs/base/legged_robot.py#L841)：实际关节速度差除以控制 dt，平方后求和；同时有 action_rate。固定提交 8fa29acc6fd1910c3d9659eef6310bdd301cde0a。
- [Isaac Lab joint_acc_l2](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab/isaaclab/envs/mdp/rewards.py#L168)：实际关节角加速度平方和，只计算 asset_cfg.joint_ids 指定的关节；支持选定腿关节。action_rate_l2（252行）则计算动作变化。固定提交 b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8。
- “指令平滑”与“实际关节平滑”在代码里是不同对象；角加速度不等于 jerk。加速度项也会惩罚正常运动中的加速和制动，不能单靠数值把正常动作与抽搐完全区分。

## 其他方向：CAPS

[作者项目页](https://ai.bu.edu/caps/) 给出两种 Actor 正则：相邻时刻动作接近、相似观测输出接近；用电机信号频谱 FFT 比较高频变化。它是修改策略优化目标的办法，不是替换某一个 reward term，也不是 ROGER。

本轮只读取作者的方法介绍，没有下载或审查其算法代码。作者实验包含无人机与基准任务，不能据此保证对本工程有效。可借鉴分关节频谱作为离线诊断；频段应由本机器人数据确定，不能任意套用某个截止频率。

## 与当前工程对照

当前 WheelLeg 的 rewards.py:67 action_smooth 覆盖所有六个标准执行动作；rewards.py:62 dof_acc 也覆盖六个关节。已有 action_rate、dof_acc，不能把通用开源中的同类项再重复加一遍。

相同停车，新1000轮：六维均值 2.715708，腿部分 0.032537，轮部分 2.683171（98.80%）。这里支持“腿轮分组能避免总统计量被轮主导”的结论，不证明轮是左腿抽搐根因，也不说明腿部现有惩罚没有作用。沿用相同增益时，各腿仍受惩罚。

## 建议的最小下一步

1. 将四腿动作二阶差分单独作为 ROGER 候选指标，数学公式不变；左/右分别记录，避免饱和右腿的零变化被误当健康标准。
2. 四个腿的实际角加速度单独用于验收；轮信号单独记录，不用全六维总数代替左腿情况。先做离线验证，不直接叠加新惩罚。
3. 阈值仍需要人工判断可接受运动、实际跟踪验收和选定统计窗口。这些参考没有给出适用于本机器人的现成 ROGER 阈值。
4. 只有指标验证后，才制定奖励配置与自适应增益的实现、checkpoint 和对照训练方案。

以上为候选设计，不代表用户已批准改动或训练。所有在线操作只读公开资料；下载的15份源码仅缓存、不执行。源码 SHA256 与 Git blob 哈希已复核。
