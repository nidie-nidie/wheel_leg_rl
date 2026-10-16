# V2 FK 诊断代码复核

日期：2026-10-16。复核范围为 `geometry.py`、`environment.py`、`contract.py`、`experiment.py`、`train.py`、`export.py`、`mujoco_adapter.py`、`test_geometry.py`、`verify_artifacts.py`、`test_mujoco_adapter.py`，同时只读核对正式依赖与已通过复核的 V2 文档。复核者只写入本报告，没有启动训练，没有调用其他 agent。

没有发现阻塞独立 Isaac Lab 验收的问题。可以使用新进程运行 V2 验收；V2 初始化、部分 reset、PPO 更新、诊断日志、checkpoint、导出及 MuJoCo 身份验收全部通过后，可以另外创建新进程从头训练 1000 iteration。当前纯 Torch 与 MuJoCo 控制测试没有证明 V2 已经越过此前第 9 轮的中断位置，也没有证明最终运动效果改善。

## 实际几何和 FK 的边界

- `environment.py:13` 重写 `_virtual_leg_state`，通过 `super(..., enforce_guards=False)` 获取全部原有结果。调用方要求运行期保护时，`:16` 读取同一批实际髋轴和轮轴锚点，`:20` 执行独立的实际几何检查，`:21` 原样返回父类结果。没有裁剪、填零或替换真实几何及 FK 输出。
- `geometry.py:11` 检查实际机身坐标向量、真实腿长、真实 phi0 和闭环间隙全部有限。`:18` 保留真实腿长 `>0.05m`，`:20` 保留闭环间隙 `<=0.005m`，边界比较与正式定义一致；长度等于下界拒绝，间隙等于上界允许。
- 实验继续继承正式 `tasks/direct/wheelleg_flat/env.py:714` 的 `WheelLegState.assert_finite`，包括全部 FK 张量。关闭的是 FK 偏差阈值及 `fk_valid=False` 的独立运行期中断；FK 输出产生 NaN 或 Inf 时仍然拒绝。正式 `kinematics/virtual_leg.py:187`、`:195` 的圆交点数值保护继续保留。
- 初始化调用传入 `enforce_guards=False` 时，实验子类保持这一语义。正式 `env.py:426` 的缓存计算、`:477` 的初始化几何验收和 `schemas/randomization.py:91` 的参考构型 FK 均未改变。实际闭环、FK 比较、主动参考角、被动构型、物理边界和装配分支的初始化验收继续执行。
- `geometry.py:25` 仅计算诊断统计；`environment.py:43` 仅添加日志。现有 wheel、phi0、腿长、闭环指标继续由父类日志提供。新增 disagreement、invalid 和 length error 指标没有修改输入张量或奖励张量。

## 观测、奖励和物理行为

- 正式 `observations.py:45` 的 Actor 数据与 Critic 扩展字段没有解析 FK 输入；实验未重写观测生成。正式 `rewards.py:99` 的 phi0 对称奖励继续读取 `virtual_leg_phi0_true`。
- `environment.py:35` 完整调用父类奖励计算，返回同一个 reward；修改范围仅包含现有原始动作实验的日志及新增 FK 日志。原始动作转换、目标角限制、扭矩限制、随机化、普通 episode 终止及 reset 算法保持当前实验行为。
- 独立读取 `formal_files_before.json` 的全部 100 个条目，逐个计算当前 SHA256，结果全部一致。正式 source、scripts、configs、架构、USD 与 MuJoCo 资产保持实验开始时的字节。

## V2 身份和负向验收

- `contract.py:9` 使用 `DebugNoActionClipV2`；`:25` 的 geometry_guard 区分 Isaac 运行期实际几何要求、FK 仅作诊断、初始化验收与 MuJoCo 原有实际几何指标行为。该契约没有声称 MuJoCo 执行新增 Isaac 几何中断。
- `train.py:149` 写入训练 manifest；`experiment.py:89`、`:95` 分别核验 manifest 与 checkpoint infos 中的完整字典；`:155` 在保存 checkpoint 时写入。`export.py:56` 写入导出身份；`mujoco_adapter.py:52`、`:66` 分别核验导出和训练 manifest。既有源 checkpoint、manifest、正式 MuJoCo 代码和 XML 指纹绑定继续保留。
- 复核者实际使用当前 V2 checkpoint loader 加载原 V1 的 `runs/validation-reviewed/model_0.pt`，在身份检查处按预期拒绝；使用当前 MuJoCo loader 加载对应 V1 导出，同样按预期拒绝。
- `verify_artifacts.py:42` 修改真实 checkpoint 副本的 infos geometry_guard；`:53` 修改 manifest 副本；`test_mujoco_adapter.py:49` 先确认完整真实导出能够加载，再分别修改单个身份字段并要求拒绝。检查只修改各验证目录中的副本，恢复后再次要求成功加载。V2 的实际负向运行结果需要在本次独立 Isaac 验收生成 V2 artifact 后记录。
- `train.py:118`、`:129` 区分 8 环境验证与 256 环境 fresh 1000 iteration。未加入 resume 或载入旧权重路径，现有正式文件、实验源码和保存配置的指纹保护继续生效。

## 独立执行证据

- 项目 Python 运行 `test_experiment.py` 与 `test_geometry.py`：29 个测试通过；另行运行正式 `tests/unit/test_virtual_leg.py`：6 个测试通过，共 35 项。
- 几何测试使用正式 USD 审计数据与真实 `offset_leg_fk`，根据已记录的关节角计算约 5.250mm 的 wheel FK 偏差，确认真实几何合格时允许继续。该测试仅复核失败记录中的标量判定，没有复现当时完整物理状态。
- 测试覆盖实际向量、长度、phi0、闭环间隙的 NaN、正负 Inf，腿长下界等值、闭环间隙上界等值与超限、有限 FK 输出的 invalid 标志、调用前后输入值不变。
- MuJoCo Python 运行 `test_mujoco_adapter.py`：真实模型中的原始动作、五帧 history、reset、保留的角度和扭矩限制、相同目标下正式 PD 力矩，以及动作恰好为 1 的统计边界全部通过。
- `train.py:78` 已为独立 Isaac 验收设置新增日志检查，`:81` 使用实际 `WheelLegState` 的副本检查 FK 非有限值继续被拒绝。该验收不会修改实际环境状态。完整 V2 进程验收结果应保存在独立的 `runs/validation-fk-diagnostic-<timestamp>/`。
