# 文档复核

日期：2026-10-15。复核对象：`experiment.md`，只读检查正式源码及本地 RSL-RL 3.1.2。没有运行训练，没有修改正式文件，没有调用其他 agent。

## 复核结论

没有发现要求修改正式代码才能完成实验的阻塞矛盾。`experiment.md:43` 已明确使用独立验证进程，并用新进程创建正式实验训练环境，符合 Fudan 独立随机流和环境状态的隔离要求。独立 Generator 创建位置为 `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/randomization.py:203`，状态保存和恢复位于同文件 `:355`、`:367`；环境接口位于 `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py:1001`、`:1004`。

**P2：与旧 run-03 的比较需要限定结论。** `experiment.md:9` 使用相同奖励、种子和硬件规模，并保留当前正式实现；这不能自动证明当前源码、模型、资产及物理行为与 2026-10-11 训练过程完全相同。应在 manifest 记录旧 run-03 的准确路径及 SHA256，把它作为历史参考模型。`experiment.md:17` 已明确原始动作同时进入奖励和 CENet；训练后的改善或恶化应按这个完整实验契约解释，不应直接归因于单个裁剪调用。该项属于结果解释要求，不阻塞实施。

## 实现边界核对

- wrapper 的额外裁剪位于 `dependencies/IsaacLab-v2.3.2/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py:153`。正式 history wrapper 的构造参数允许 `clip_actions=None`，见 `source/wheelleg_dreamwaq/wheelleg_dreamwaq/algorithms/dreamwaq/history_wrapper.py:21`。传入 `None` 可以直接复用五帧更新、完整 reset 和部分 reset 规则；同文件 `:45` 检查最后一帧等于当前观测，`:68` 按 dones 更新 history。
- 环境执行裁剪集中在 `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/control.py:23`，由 `env.py:662` 的 `_pre_physics_step` 调用。实验环境重写此方法，保留前两步动作移动、目标缓存更新及原有条件分支，可以让原始动作进入执行与状态；正式 `_apply_action`、reset、奖励和其他生命周期能够复用。腿目标角边界位于 `control.py:27`，应保留现有配置值。
- 上一动作观测仍有独立裁剪，位于 `source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/normalization.py:56`。`env.py:789` 使用环境配置上的 normalization 实例，允许实验子类只重写 `normalize_previous_action`。该子类还需要在实验 manifest 明确声明原始动作语义，避免把继承的 `NormalizationV2` 名称当作行为完全一致的证明。
- 状态的三步动作缓存分别由 `env.py:702`、`:703`、`:704` 提供，reset 在同文件 `:934`、`:935`、`:936` 清零。正式 `rewards.py:87`、`:90` 使用这些动作缓存计算一阶及二阶差分；原始动作进入缓存后，奖励公式无需修改，但数值尺度会随取消裁剪改变。
- CENet 唯一需要替换的动作条件裁剪位于 `source/wheelleg_dreamwaq/wheelleg_dreamwaq/algorithms/dreamwaq/actor_critic.py:236`。重写 `batch_forward` 并使用原始 actions 即可；`cenet.py:117` 已对估计速度和动作条件执行 detach。需要实测 decoder 收到原始六维值，且动作条件没有重建梯度。
- PPO 在 `source/wheelleg_dreamwaq/wheelleg_dreamwaq/algorithms/dreamwaq/ppo.py:162` 保存原始采样动作，并在 `:187` 写入 storage。`storage.py:82` 复用 RSL-RL 存储，`:96` 原样产生 mini-batch 动作。复用正式 PPO/storage 与文档一致。
- RSL-RL runner 在 `.venv/Lib/site-packages/rsl_rl/runners/on_policy_runner.py:395` 通过 `_construct_algorithm` 创建算法，默认 `:416`、`:422` 使用模块全局名称解析类。实验 runner 重写该方法直接构建实验 ActorCritic 和正式 DreamWaQPPO，可以避免全局注册或 monkey patch。正式 `DreamWaQContractOnPolicyRunner` 在 `source/wheelleg_dreamwaq/wheelleg_dreamwaq/training/dreamwaq_checkpoint.py:303` 检查冻结的 runner 标识，`:338` 起保存正式 metadata；实验应使用独立 checkpoint 实现与身份校验。
- 当前 Actor 的 ±20 限制约束分布均值，随机采样本身可以超过 ±20，见 `actor_critic.py:179`、`:198`。验收应检查 mean 限制与 finite，避免把 ±20 误当作随机样本的运行时硬限制。
- MuJoCo 在 `sim2sim/mujoco/wheelleg_mujoco/control.py:32` 执行动作裁剪，在 `observation.py:114` 再裁剪上一动作，在 `runner.py:95` 保存 controller 返回的动作。实验 adapter 必须同时处理这三处数据含义，不能直接复用旧的仅取消目标裁剪的诊断 wrapper。现有 torque 限制位于 `control.py:48`，应保留。
- RSL-RL 在 `.venv/Lib/site-packages/rsl_rl/runners/on_policy_runner.py:152` 更新 iteration 索引，`:159` 定期保存，`:174` 保存最终模型。文档的 1000 iteration 应对应最后索引 999、completed_iterations=1000；model_0 表示完成第一轮后的 checkpoint。

## 训练前验收要求

执行原始动作转换、上一动作与部分 reset、CENet detach、真实 PPO/storage 更新、checkpoint 身份隔离、TorchScript golden-vector 和真实 Isaac Lab 训练数据通路检查。各项通过后完成代码复核，记录正式文件指纹未变化，再从随机初始化开始本次 1000 iteration。MuJoCo 运动指标用于训练完成后的表现评估，训练奖励提升不能单独证明速度跟踪和腿部振荡得到改善。
