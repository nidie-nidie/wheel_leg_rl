# 六维策略动作取消裁剪实验

日期：2026-10-16。用户已授权本目录内的文档、实现、串行 agent 复核及从头训练 1000 iteration。所有新增和修改文件均位于本目录；正式 source、scripts、configs、USD、MuJoCo 模型和正式架构文档保持原有字节。

当前继续实验采用 `DebugNoActionClipV2`；运行期间 FK 仅作诊断、保留实际几何检查的准确范围和实施顺序见 `fk_diagnostic_only.md`。文末的 2026-10-15 执行结果对应 V1。

## 实验配置

实验标识：DebugNoActionClipV2。训练使用项目 `.venv/Scripts/python.exe`、Isaac Lab 2.3.2、Isaac Sim 5.1.0、RSL-RL 3.1.2 和 PyTorch 2.7.0+cu128。硬件配置 rtx5070：256 个环境、每次收集 24 步、8 个 mini-batch、每轮优化 5 次。种子 41882943，随机初始化，1000 iteration，不加载旧模型权重或优化器。

采用 `stop_reverse_v1`，与 2026-10-11 run-03 的任务配置比较：停车、换向练习启用，tracking_vx=2、tracking_vx_enhance=2、action_smooth=-0.01、action_rate=-0.01、phi0_symmetry=-1。其余奖励直接读取该 profile，记录完整解析配置并核对旧 run-03 manifest 的奖励字典。保持现有 Fudan 随机化、地形、命令范围、动作比例、机构、增益、扭矩限制、reset 及训练时序。

## 动作契约

策略动作顺序仍为 `[jIJ,jIO,jAB,jAG,jwheel_left,jwheel_right]`。取消 wrapper 和环境执行中的六维 ±1 裁剪。腿目标为 `clip(q_reference + 0.35*a[:4], -1rad, +1rad)`；轮速目标为 `25*a[4:6]` 并沿用原有轮子符号。目标角和 actuator 的扭矩限制保持现有行为。不新增角速度限制力矩、阻尼、滤波或动作过渡层；0.15–0.35m 腿长仅用于后续 MuJoCo 统计，不增加训练硬限制。

上一动作观测使用实际提交给目标转换的六维原始动作，取消其 ±1 观测裁剪。Actor 的 25 维、Critic 的 41 维及五帧 history 的 125 维保持原尺寸；history 继续沿用当前实现的 t-4 到 t，不进行历史窗口实验。reset 清零动作并按现有行为重复首帧，部分 reset 不污染其他环境。

PPO storage 原本已经保存未裁剪采样动作，继续复用。CENet 重建的动作条件改用该原始动作并保持 detach；动作平滑和 action_rate 也使用该动作历史，公式和权重不变。因此本实验改变的是动作处理及其一致的数据含义，不能将结果解释为只删掉某一个 clamp 调用的效果。

Actor 分布均值的数值保护 ±20、标准差 [1e-4,2]、PPO ratio/value/gradient clipping 和其他物理观测归一化全部沿用。这里“取消动作裁剪”专指六维运行时 ±1 裁剪，不表示移除全部数值和物理边界。

## 文件职责

- `experiment.md`：本配置、执行顺序、验收和结果位置。
- `review_document.md`、`review_code.md`：两个串行 agent 的复核结论。
- `contract.py`：不依赖训练运行时的实验动作契约、路径、JSON 和文件指纹工具，训练与 MuJoCo 共用。
- `experiment.py`：纯 Torch 动作转换、上一动作归一化、CENet 策略子类、独立 runner 与实验 checkpoint 标识。复用正式 DreamWaQPPO 和 storage，不修改或替换正式模块的全局变量。
- `environment.py`：正式环境的实验子类，替换动作目标准备及运行期间几何检查；FK 仅作诊断，保留实际几何检查，其余环境生命周期复用。
- `train.py`：独立训练入口、完整配置与文件指纹、缓存保存、checkpoint 校验、进度记录。输出固定在本目录 `runs/`，不更新正式 latest_run 等指针。
- `export.py`：独立实验导出与 TorchScript golden-vector 校验。导出带 DebugNoActionClipV2 标识和几何检查契约，正式 loader 因缺少正式契约拒绝加载。
- `mujoco_adapter.py`：后续 MuJoCo 使用的实验 controller/runtime；原始动作转换、上一动作及五帧 history 与训练一致，复用正式动力学和 PD 力矩计算。加载时核验导出、源 checkpoint、训练 manifest、训练时正式文件指纹副本，并将当前正式 MuJoCo adapter、model manifest 和模型 XML 与训练时指纹逐项比较；当前物理身份变化必须拒绝加载。
- `test_experiment.py`：真实模块上的数值、梯度、storage、reset/history、checkpoint、TorchScript 与 MuJoCo 数据处理验收。禁止 mock、源码字符串替换和全局 monkey patch。
- `test_mujoco_adapter.py`、`verify_artifacts.py`：真实 MuJoCo 数据通路与导出加载检查，以及真实 checkpoint 副本的文件损坏拒绝验收；只改写本目录 runs 中的检查副本。
- `runs/`：独立日志、配置、训练状态、缓存、checkpoint 和导出结果；受现有 runs 忽略规则保护。

依赖方向：debug 入口和子类调用正式模块；正式模块不得引用 debug。本目录不复制整套工程，不改变 import 后的正式类定义。

## 执行与验收

1. 保存正式训练代码、配置、正式 MuJoCo adapter、架构和资产的 SHA256 清单，记录当前已存在的工作区改动，后续比较以本次开始时的字节为准。
2. 文档完成后仅调用一个 agent，读取本文件和相关正式实现，写出复核结论。解决阻塞问题后再实施。
3. 实现独立子类和入口，验证超出 ±1 的腿、轮动作传递，目标角限幅和轮符号、非有限输入拒绝；对比范围内动作与原实现相同。
4. 验证上一动作与 history 保留原始值，部分 reset 行为；使用真实 CENet decoder 的输入 hook 验证原始动作与 detach；用实际 PPO rollout/storage 更新验证动作没有被夹回 ±1，优化器参数有限。MuJoCo 数据通路检查使用独立 MuJoCo Python 环境，不导入 RSL-RL 训练运行时。
5. checkpoint 使用专有 infos schema，保存模型、优化器、随机状态、环境随机化状态、CENet monitor、completed_iterations 和 manifest hash。拒绝普通 load/resume；本次仅支持 fresh，避免误用 RSL-RL 默认 load 的迭代恢复行为。验证 checkpoint 能重建相同推理输出且正式 metadata validator 拒绝实验文件。
6. 代码完成后再调用一个 agent 复核，检查是否修改正式文件、是否漏掉裁剪、是否改变奖励权重或动力学，并记录结论。每次最多一个 agent，不允许 reviewer 再创建子 agent。
7. 使用单独进程和少量真实 Isaac Lab 环境检查训练数据通路：初始观测、部分 reset/history、超出 ±1 的原始动作及一次 PPO 更新。验证进程完成后退出，再用新进程创建 256 个正式实验训练环境；禁止复用已经推进过的验证实例，也不以全局重新设种子代替 Fudan 各独立 Generator 的恢复。其作用限于训练链路验收；运动表现的数据收集全部使用 MuJoCo。
8. 开始 fresh 1000 iteration，保存 model_0、每 100 轮和最终 model_999，输出 TensorBoard 与进度 JSON。训练遇到非有限数据或配置/源文件指纹变化立即报错终止。禁止自动改奖励或自动追加训练。
9. 训练完成校验最终轮次、checkpoint 身份、参数有限和正式文件未改动，自动导出 actor/encoder；以包含大于 ±1 历史动作的固定输入验证 eager/TorchScript/重新加载结果，记录误差。

后续 MuJoCo 评估使用停车、前进、后退、换向及高度变化的相同命令，报告速度误差、实际腿运动、phi0 差值、目标角和扭矩触限比例。训练奖励提升不等同于这些表现改善。本次用户要求的执行终点是完成文档与代码复核并启动这一次 1000 iteration；最终训练完成情况由训练进度和 summary 文件确认。

## 结果位置

训练目录由入口在 `runs/<Singapore timestamp>/` 创建，包含 `manifest.json`、`formal_files_before.json`、`progress.json`、`training_summary.json`、`model_*.pt`、`closed_chain_reset_cache.pt` 和 TensorBoard 文件。所有模型均使用独立实验标识；与旧模型临时取消动作裁剪的可视化结果分别记录。

## 本次执行结果

文档复核及代码复核均已完成，分别记录在 `review_document.md` 和 `review_code.md`。10 项数值及真实 PPO 测试通过；真实 MuJoCo 的动作、history、reset、扭矩与导出身份检查通过；checkpoint 配置及缓存副本损坏检查通过。独立 Isaac 进程 `runs/validation-reviewed/` 完成一次更新，验证原始动作观测、部分 reset/history、正式 loader 拒绝实验身份，TorchScript 最大绝对误差为 1.12e-7。

完整训练已于 2026-10-15 21:50:30 启动，目录为 `runs/fresh1000-20261015-215030/`。配置为随机初始化、种子 41882943、256 环境、目标 1000 iteration。实际完成索引 0–7 共 8 轮；第 9 轮收集 rollout 时，正式几何检查抛出异常，进程退出。当前状态为 failed，1000 轮没有完成。仅存在第一轮 checkpoint `model_0.pt`，不能把它当成第 8 轮权重或最终模型。

异常为 `Virtual-leg geometry guard failed`：FK 轮位置与仿真真实位置的最大偏差为 0.0052499m，超过 0.005m；第一处异常为 env=201、side=0。该次输出的最大腿长误差为 0.00437285m，最大 phi0 误差为 1.21584°，第一处异常环境的最大闭环锚点误差约 0.00167405m。本次直接触发项为轮位置 FK 误差，不能把它误写成闭环锚点间隙超过 5mm，或据此断言机构不可达。

阈值定义见正式 `source/wheelleg_dreamwaq/wheelleg_dreamwaq/kinematics/virtual_leg.py:15`；检查与抛出异常见正式 `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py:272`、`:286`，由 `env.py:885` 的终止判断前状态读取触发。现有行为会终止整个训练进程。当前日志不足以判断误差增长源于求解误差、几何分支还是具体动作和接触状态。

失败证据保存在同目录 `progress.json` 以及 `runs/fresh1000-20261015-215030-launch/stdout.log`、`stderr.log`，进一步索引为 `runs/fresh1000-20261015-215030/failure_analysis.json`。失败后再次核验 100 个正式文件指纹，变化数为 0；debug 目录外 Git 状态与实验起点相同。正式阈值、动力学和 reset 行为保持原有值。后续需要先采集失败环境前后的原始动作、目标、关节和刚体状态，确认几何偏差来源，才能决定满足本实验边界的处理方法。
