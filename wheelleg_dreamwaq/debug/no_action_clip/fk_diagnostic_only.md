# 运行期间 FK 仅作为诊断的实验配置与实施计划

日期：2026-10-16。承接六维原始动作实验，继续遵守正式代码及资产保持原有字节的约束。用户要求检查 FK 依赖，并允许本次实验暂时不依赖 FK 一致性检查。所有修改限定在本 debug 目录，文档复核与代码复核串行执行，每次最多一个 agent。

## 已核实的依赖

- 正式 `tasks/direct/wheelleg_flat/observations.py:45` 从实测关节角、关节速度、角速度、重力方向、命令和上一动作构造 Actor 观测，Critic 额外使用仿真速度、高度、关节加速度和力矩，均不读取解析 FK 的腿长或 phi0。
- 正式 `tasks/direct/wheelleg_flat/rewards.py:99` 的 phi0 对称奖励使用 `virtual_leg_phi0_true`，由髋轴与轮轴的实际刚体锚点计算，保留这一来源及奖励公式、权重。
- 正式 `env.py:245` 读取实际锚点，`:250` 计算实际腿长及 phi0；解析 FK 从 `:252` 开始，仅用于独立诊断、有限值检查以及当前的误差保护。
- 正式 `schemas/randomization.py:91` 使用 FK 计算出生高度补偿，`env.py:420` 起的 reset 缓存生成也核验初始化几何。本次保持初始化和 reset 算法全部原样。
- MuJoCo 的 `wheelleg_mujoco/metrics.py:59` 直接读取 `data.xanchor` 计算腿长和 phi0，运行期间没有相同的解析 FK 比较终止条件。

2026-10-15 的中断数据只能确定解析 FK 与实际几何存在偏差，尚不能确定 FK 公式是否存在错误。第一处异常环境 env201 的实际闭环锚点最大间隙约 1.674mm，该值不代表全部环境的最大值；轮位置 FK 的全批最大偏差约 5.250mm。理想闭环几何与动态求解结果的偏离需要分别记录。

## 完整行为

实验身份升级为 `DebugNoActionClipV2`，增加独立 `geometry_guard` 契约，覆盖训练 manifest、checkpoint metadata、导出及实验 MuJoCo loader。V1 的失败记录保持原样；V2 从随机初始化重新开始 1000 iteration，种子、256 环境、24 步 rollout、8 mini-batch、5 epoch、奖励和随机化沿用 `experiment.md`。

契约明确区分执行范围：下述运行期间检查适用于 Isaac 训练环境；MuJoCo 继续使用现有的实际腿长及闭环指标采集，不新增对应中断条件。实验 MuJoCo loader 绑定该契约是为确认训练语义，不能解释为 MuJoCo 已经执行相同的几何中断逻辑。

运行期间继续计算解析 FK，记录 wheel、length、phi0 偏差及 FK 不可解比例。上述偏差超过原阈值、或 FK 的 valid 为 false，均不单独触发中断。解析 FK 结果不加入 Actor/Critic 输入，不替换实际几何奖励，不修改动作或力矩。

运行期间继续严格检查实际锚点和几何数值有限、实际腿长大于现有 0.05m，以及实际闭环锚点间隙不超过现有 0.005m。现有 WheelLegState 全字段非有限数值检查、普通终止条件和所有初始化缓存验收继续生效。实际几何检查失败仍然立即终止进程，不将其静默忽略或改为自动 reset。

父类初始化通过 `enforce_guards=False` 调用几何计算时，debug 子类仍然尊重该参数，把初始化阶段的验收交给现有缓存生成过程。运行阶段使用同一批实际刚体数据计算锚点并执行上述真实几何检查。

## 文件与步骤

1. 完成本文件及 `experiment.md` 当前配置说明，使用已有 reviewer 串行复核文档，报告写入 `review_fk_document.md`。
2. 新增 `geometry.py`：纯 Torch 的实际几何检查与 FK 偏差统计函数，依赖正式几何常量和 wrap 函数，不导入 Isaac。
3. 修改 `environment.py`：仅重写 `_virtual_leg_state`，调用父类获取原有真值和 FK 结果，按运行参数执行实际几何检查；日志额外增加 FK 偏差超限比例与不可解比例。动作转换和奖励实现保持当前版本。
4. 修改 `contract.py`：V2 标识和明确的几何检查契约。修改 `train.py`、`experiment.py`、`export.py`、`mujoco_adapter.py`，逐项绑定并校验这个契约，防止混用 V1/V2 的模型和检查行为。
5. 新增 `test_geometry.py`：使用真实 FK 和已记录的失败数值证明该偏差允许继续；分别验证真实锚点非有限、腿长不满足最小值、闭环间隙超限仍然拒绝。验证诊断结果不改变传入的真值与 FK 张量。现有原始动作、PPO/storage 和 TorchScript 测试继续运行。
6. 串行复核代码，报告写入 `review_fk_code.md`。复核完成后使用独立进程执行 Isaac 训练数据验收、checkpoint 身份校验、MuJoCo 加载与 golden-vector 检查；验证进程退出后再创建全新训练进程。
7. 启动 fresh 1000 iteration。检查前 20 轮内的实际几何与 FK 日志，确认能够经过之前第 9 轮的失败点。发生真正几何检查失败时记录并停止；仅 FK 偏差继续记录。训练结果以最终 `model_999.pt` 和 completed_iterations=1000 为准。

依赖方向保持 debug 调用正式模块。正式 source、scripts、configs、架构、USD 和 MuJoCo 参数不作修改，继续核验原有 100 个文件指纹。

## 验收记录

文档复核见 `review_fk_document.md`，代码复核见 `review_fk_code.md`，均无阻塞项。35 项测试通过，其中原始动作与 PPO 测试 10 项、新增几何测试 19 项、正式 FK 测试 6 项。真实 MuJoCo 的控制、history、reset、PD 力矩与保留的物理限幅验收通过。

独立 Isaac 进程输出为 `runs/validation-fk-diagnostic-20261016/`，completed_iterations=1。初始观测、原始动作、部分 reset/history、新增 FK 日志、FK 字段非有限值拒绝及正式 loader 拒绝均通过。相同 8 环境短测中，初始模型与一次更新后的模型指纹分别与 V1 验收相同；这只证明该短测范围内数值行为一致，不能证明长期训练表现。TorchScript 最大绝对误差为 1.12e-7。

实际 V2 checkpoint 的配置及缓存副本损坏、checkpoint infos 和 manifest 的几何契约改变均被拒绝，证据为验证目录 `artifact-integrity/verification.json`。实际 V2 导出的 actor、训练 manifest、源 checkpoint 身份及几何契约改变均被拒绝，证据为验证目录 `mujoco-loader-integrity/verification.json`。

本次训练输出固定为 `runs/fresh1000-fk-diagnostic-20261016/`，启动日志位于同名 `-launch/` 目录。启动后的轮次、FK 偏差统计及进程状态记录在该训练目录的 `startup_verification.json` 和 `progress.json`；最终完成情况由 `training_summary.json` 与最终 checkpoint 确认。原有 100 个正式文件指纹在验收后保持一致。
