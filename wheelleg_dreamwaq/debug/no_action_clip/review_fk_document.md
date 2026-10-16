# FK 诊断实验文档复核

日期：2026-10-16。只读复核 `fk_diagnostic_only.md`、`experiment.md` 及正式 Isaac、MuJoCo 源码。没有启动训练，没有修改正式代码，没有调用其他 agent。

## 阻塞项检查

没有发现必须修改正式代码才能实施的阻塞。debug 环境可在重写的 `_virtual_leg_state` 中调用 `super()._virtual_leg_state(..., enforce_guards=False)` 获取原有结果，运行阶段自行检查实际几何。传入该方法的 `enforce_guards=False` 仍需传递初始化阶段语义，保留父类缓存验收。

`experiment.md:29` 明确环境负责动作转换及运行期几何检查，`:31` 明确 V2 导出及几何契约，与同文件 `:5` 及 `fk_diagnostic_only.md:17` 的实验身份一致。`fk_diagnostic_only.md:13` 准确限定 1.674mm 为 env201 的闭环最大间隙，符合失败证据中的 `first_invalid_env_loop_error_max_m`，没有将它当成全部环境最大值。

`fk_diagnostic_only.md:19` 明确严格实际几何中断只适用于 Isaac 训练，MuJoCo 继续采集真实指标，loader 绑定训练语义。正式 MuJoCo `metrics.py:104`、`:105`、`:106` 计算实际腿长和闭环间隙，`:113` 验证指标有限；正式 `runner.py:99` 调用指标收集并返回。文档描述与保持当前 MuJoCo 行为的边界一致。

## 正式数据依赖

- `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/observations.py:45` 构造 Actor 物理字段，使用角速度、重力方向、命令、真实关节角误差、关节速度和上一动作；`:115` 构造 Critic 观测，增加真实速度、高度、关节加速度和力矩。这里没有解析 FK 的腿长、phi0 或 valid 输入。保留这些函数即可保持 Actor、Critic、history 和 CENet 观测定义。
- `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/rewards.py:99` 使用 `virtual_leg_phi0_true` 计算 phi0 对称奖励；其他奖励读取真实关节、速度、姿态、动作和力矩。没有使用解析 FK 量。正式 phi0 奖励函数位于 `kinematics/virtual_leg.py:166`，应原样复用。
- 正式 `env.py:245`、`:246` 读取实际髋轴和轮轴世界锚点，`:249` 变换到机身坐标，`:250` 计算真实腿长和 phi0。`:252`、`:253` 单独计算左右解析 FK。真实奖励来源与解析诊断来源可以保持分离。
- 正式 `env.py:262`、`:264` 使用闭环两端实际世界锚点计算闭环间隙。解析 FK 轮位置误差在 `:260` 计算；两者是不同指标。
- `sim2sim/mujoco/wheelleg_mujoco/metrics.py:60` 使用 `data.xanchor` 的髋轴到轮轴向量，`:64`、`:65` 计算实际腿长和 phi0；`:74` 使用真实闭环 site 位置计算间隙。MuJoCo 没有相同的解析 FK 误差中断路径。

## 初始化与运行期保护

- 正式 `schemas/randomization.py:91` 至 `:103` 使用解析 FK 计算名义和随机参考构型，并要求 FK valid；`:111` 计算出生高度补偿。此路径应保持原有行为。
- 正式 `env.py:426` 在缓存度量中调用 `_virtual_leg_state(..., enforce_guards=False)`；`:477` 的 `_assert_closed_chain_metrics` 仍在 `:481` 至 `:484` 检查实际闭环、轮位置 FK、腿长 FK 和 phi0 FK，此外还检查主动关节参考误差、被动关节变化、物理限位及装配分支。父类方法、调用顺序和这些阈值无需修改。
- 正式运行期合并检查位于 `env.py:268`：实际向量有限、真实腿长、FK valid 及三个 FK 比较指标混合形成 invalid。debug 子类必须替换这一运行期组合检查，单独保留真实向量和锚点有限、真实腿长大于 0.05m；不得仅调用父类默认保护后尝试捕获异常。
- 正式 `kinematics/virtual_leg.py:13`、`:14` 定义真实腿长 0.05m 和闭环间隙 0.005m 边界。长度恰好等于 0.05m 应拒绝，间隙恰好等于 0.005m 应允许，与原有 `<=` 和 `>` 判定一致。
- 正式 `env.py:714` 调用 `WheelLegState.assert_finite`；`state.py:33` 检查所有张量字段。V2 保留这个调用，也保留 FK 字段，因此 FK 产生非有限数值仍会中断。`fk_valid=False` 且 FK 输出全部有限可以继续，这与当前“valid 不单独中断、全字段有限保留”的设计一致。正式 FK 圆交点计算在 `kinematics/virtual_leg.py:187`、`:195` 使用数值保护，`:197` 单独返回 valid；不需要用填零或修改张量来隐藏无效诊断。
- 普通 episode 终止、实际动力学、PD 力矩、动作目标、reset 缓存及随机化不应随 V2 改变。父类 `_get_rewards` 返回的奖励张量保持原值，debug 日志扩展只增加诊断字段。

## V2 身份与验收

`geometry_guard` 至少明确：真实锚点和全字段有限检查、真实腿长下界、闭环间隙上界、三个解析 FK 比较仅作诊断、FK valid 仅作诊断、初始化缓存验收保持现有行为，以及各仿真器的适用范围。训练 manifest、checkpoint infos、导出报告和 MuJoCo loader 应逐项比较这一字典及 V2 实验身份；缺少字段、阈值变化或 V1 身份均立即拒绝。

测试需要覆盖实际几何的边界等值、超限和非有限情况；覆盖有限 FK 输出但 valid=False 的继续行为；覆盖 FK 偏差超阈值但实际几何合格的继续行为；覆盖全字段有限检查保留；覆盖调用前后真值与 FK 张量不变。历史失败标量只能验证对应的判定条件，不能声称已复现当时完整物理状态。

完成代码复核后，使用独立 Isaac 进程验证 reset 缓存初始化、部分 reset/history、PPO 更新和新增日志，并完成身份、TorchScript 与 MuJoCo 验收。正式训练使用新进程与独立 V2 输出目录，保留 V1 失败文件，继续校验原有 100 个正式文件指纹。
