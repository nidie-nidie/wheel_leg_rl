# WheelLeg 701 checkpoint debug-only 约束微调设计

日期：2026-10-18

## 目标

从 4096 环境、seed `41882943` 的 `model_700.pt` 开始做一个可回退的 debug-only 微调，目标是在速度追踪误差低于 701 轮基线的同时，保持实际腿部角加速度不恶化。正式源码、正式 YAML、原始 checkpoint 和已有评估均保持不变。

701 轮基线证据来自 `debug/sim2sim/provenance/wheelleg_4096_model700_provenance_20261018.json`：

- 速度 MAE：`0.03731354048572227 m/s`
- 排除开始 1 秒后的合并持续腿部角加速度 RMS：`0.17272677856097493 rad/s²`
- 八场景存活率：`1.0`

## 训练边界

- 新实验只写入 `debug/sim2sim/runs/` 下的独立目录。
- 训练输入使用已严格验收的正式复现 model_700.pt；它与旧 debug model_700.pt 已逐位相同，旧 debug SHA 继续作为基线锚点记录。 `debug/no_action_clip/runs/fresh1000-split-straight-lambda01-env4096-20261016/model_700.pt`。
- 4096 环境、seed、任务 profile、观测、动力学、随机化、控制周期和网络结构沿用 701 基线。
- 初始模型只加载 701 的网络权重；Adam 的一阶、二阶历史清零，不带入原训练的优化器动量。
- 首阶段只做 20 次更新 smoke；未通过验收时不继续训练。

## 约束信号

### 实际角加速度窗口约束

在 debug 环境钩子中读取原始 `state.joint_acceleration`，只取四个驱动关节对应的实际角加速度。按每个环境维护最近 50 个控制步（约 1 秒）窗口，计算窗口 RMS。训练信号只惩罚超过 701 基线预算的部分：

`excess = max(window_rms - budget, 0)`

预算先设为基线的 1.05 倍，即约 `0.1813631175 rad/s²`，并在实验 manifest 中固定记录。这个 5% 只用于训练噪声余量，最终验收仍要求持续 RMS 不超过 701 基线；不能靠提高预算来宣称成功。

该惩罚在 rollout reward 中作为单独、具名的 debug 项记录，不改正式 reward contract，不覆盖原有 `dof_acc` 项。窗口在 reset 时清空，避免跨 episode 泄漏。

### 701 策略参考约束

冻结一份 701 actor，仅用于计算当前 actor 与参考 actor 在相同 observation 上的分布差异。采用具名的 reference-policy KL 惩罚，限制单次更新远离已验收策略；参考 actor 不参与反向更新。KL 只用于 debug 训练，不写入正式模型契约。

## 优化器和训练流程

1. 写入独立 manifest，记录源 checkpoint SHA、基线指标、约束预算、窗口长度、KL 系数、Adam 重置标记和全部命令行参数。
2. 启动 debug-only trainer，加载网络权重并创建全新的 Adam 状态。
3. 每次 rollout 记录窗口 RMS、超额惩罚、原始 reward 总和、速度追踪项和 episode reset 事件。
4. 每次 update 记录约束前后速度 MAE 代理、reference KL、梯度范数、策略参数漂移和有限值检查。
5. 20 次更新结束后导出该 smoke checkpoint，在 MuJoCo 采用与基线相同的 headingOFF 八场景、500 控制步、三组切换设置评估。

## 通过条件

- 20 次更新完整结束，checkpoint 和日志有限值，未发生 OOM、NaN 或环境 reset 失败。
- 速度 MAE 小于 `0.03731354048572227 m/s`，或至少相对 701 基线有明确下降并在报告中量化。
- 排除开始 1 秒的持续腿部角加速度 RMS 不超过 `0.17272677856097493 rad/s²`；若只在部分场景满足，判为未通过。
- 存活率为 1.0，换向、转向释放、heading 开关和 reset 事件均达到。
- 报告必须区分训练约束信号与 MuJoCo 行为验收，不用训练 reward 或 KL 单独宣称稳定。

## 主要风险和解释边界

- 实际角加速度来自物理反馈，不能对单个动作直接求导；因此它作为 rollout 惩罚，reference KL 负责限制策略漂移。
- 这次 smoke 只能验证约束方向和局部微调效果，不能证明多种子泛化。
- 即使速度和 RMS 同时改善，也不能把结果归因于某一个原始 reward 分量；需要保留约束项、KL、Adam 重置和同场景对照证据。
- 失败时保留完整日志和 checkpoint，不修改正式工程，也不自动追加训练。


