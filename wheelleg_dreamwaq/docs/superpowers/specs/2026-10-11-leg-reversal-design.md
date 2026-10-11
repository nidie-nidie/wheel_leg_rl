# 腿目标换向软约束：单种子短训练对照

用户于2026-10-11批准先试验。保留六维 action_smooth=-0.01、stop_reverse_v1 的停车换向练习和速度奖励。新增四腿换向软成本，采用A1启发的ROGER-style有界单项调权，不实现论文多通道critic/GAE或主奖励衰减，不宣称论文保证。

## 计数和成本

信号为 compute_action_targets 输出的四维 leg_position_targets，已裁剪/缩放/限位。每控制步更新一次，不在物理子步或观测/奖励读取更新。每腿记录趋势和局部峰/谷；从极值累计反向至少0.01rad计一次。首次方向不计事件，停顿保留趋势，一周期两次反向。不得用单帧位移门槛漏掉慢移动。

50个20ms控制步构成1秒滑窗，f_j=count_j/1s。reset只清所选环境的极值、方向、年龄、窗口、缓存成本，锚点为reset后的腿目标。未满50步不惩罚、不进入gain统计。跨episode不计反向；全局gain及rollout统计不随部分环境reset清空。

两档预算N=12或8次/(关节·秒)。z_j=max(f_j/N-1,0)，新增单步奖励 r=-dt*0.02*lambda_used*sum_j(z_j)，低于预算新增成本为0。0.02是试验的成本上限，与原smooth的单位不同：旧停车f≈18.7/s时满gain新增成本速率约0.0223(N12)/0.0535(N8)，原smooth约0.0322；仅作为初始试验量级标定，不是安全保证。

每24控制步汇总一次，m_j=mean(z_j)，包含有效窗口的零值；lambda_next=clip(max_j(m_j)^2,0,1)。无有效窗口时为0。初始gain0，同一rollout的24个成本使用旧gain，随后切换。最严重关节驱动一个gain，奖励逐关节求和，不能把静止右腿平均掉。日志记录逐腿rate/excess、有效窗数、gain_used/next和实际新增奖励。PPO学习控制动作，N不由PPO学习。

不加力矩、动作滤波、硬间隔、倾斜免责或termination；观测/网络/history不改。实际qvel与20ms端点加速度一起验收，不能以目标变安静代替机械抖动验收。

## 身份和恢复

具名配置 stop_reverse_reversal12_v1 / stop_reverse_reversal8_v1。disabled配置不向BaseTaskContract增加键，保留旧合同hash；enabled序列化LegTargetReversalV1的全部参数和算法。普通resume仍完整合同相等。gain状态仅enabled合同必须：completed_rollouts、next_gain、最近汇总；只在rollout边界保存。resume仍是weights/optimizer/RNG+new-environment，局部窗口清空并重新等1秒，不声称恢复物理轨迹。

独立 --initialize-from 与 --resume 互斥。源自己的checkpoint/manifest/cache先完整校验；目标只允许增加本约束，原奖励/命令/随机化/物理/动作/网络全部相等。只复制网络及normalizer权重，optimizer/iteration/monitor/gain和过程RNG从新实验开始，记录源路径/hash/iteration和目标合同，不能绕过普通resume。

## 对照和验收

来源为PhysicsV5 stop_reverse_v1 seed41882943 run03 model_999，SHA256=3D621280CF7F11088C972B0DD84CF2E698C29A4C8CDA8BC370BB80A2669DF885；不是昨晚PhysicsV4静止腿模型。三组均从同源weights-only初始化，相同seed41882943/rtx5070/256env/8minibatches/24steps，各200iterations，串行，30分钟监控，训练冻结源码。基础组不加约束。

训练前：迟滞/慢移动/停顿/窗口到期/partial reset/rollout固定gain/保存恢复/旧hash兼容单测，以及真实Isaac时序和checkpoint/resume/play/export/golden。训练后：各组export/golden batch1/3/32、Isaac当前八场景、MuJoCo原八场景/四换向专项、同算法换向和实际qvel/qacc诊断。MuJoCo仅诊断，不改模型和控制。报告逐腿反向次数、实际抖动、vx/yaw误差、存活、姿态和动作/力矩饱和，沿用当前验收门槛。200轮一种子仅探索，不替代4×1000正式验收，不自动扩大训练。
