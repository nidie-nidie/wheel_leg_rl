# 2026-10-10：MuJoCo 刚体角速度限制对齐与复测

正式代码已对齐并完成复测。保留 Isaac 原训练动力学，在 MuJoCo 中补上逐刚体限速偏置力矩；四个原 checkpoint 均完成八场景、每场 500 ticks / 10 s，合计 32/32，没有触发原有失败条件。原评估四组均为 0/8。

本轮没有训练，没有新增子 agent。短程诊断与稳定性复测通过不等于 Phase 2 全部性能门通过。

## 实际修改

Isaac `max_angular_velocity=100.0` 的单位仍为 deg/s，约 1.745329 rad/s。四份原 run manifest 的完整 base-task contract 都验证为这个值；本轮不把它改成 100 rad/s。MuJoCo 新增 `PhysxRigidAngularBiasV1`，按世界系惯量和超限角速度计算力矩，每 5 ms 更新并保持五个 1 ms 步。它不截断 qvel，也不修改电机 PD、ctrl 或 effort limit。

对应公开 PhysX 5.6.1 GPU 源码的逐 link 偏置力矩项；参考 commit `5ca9f472105a90d70d957c243cb0ef36fe251a9f`，本地原文 `physx-5.6.1-forwardDynamic2.cu:1261`。该实现和采样周期经过短程复现验证，不宣称两个仿真器整个求解器等价。

新评估采用 `MujocoEvaluationContractV2`，显式记录这项外力语义并覆盖实现 hash。原 suite、旧报告和旧排名均保留。正式架构为 v0.24，SHA256：`FDEF463B22E88F4C219791F0CA019AAAB5F9D2FE7DF47B96EE91B6E240F54225`。

| 文件 | 新增行 | 删除行 |
|---|---:|---:|
| `../docs/2026-10-03-wheelleg-dreamwaq-architecture.md` | 26 | 2 |
| `sim2sim/mujoco/wheelleg_mujoco/runner.py` | 4 | 0 |
| `sim2sim/mujoco/wheelleg_mujoco/contract.py` | 2 | 0 |
| `sim2sim/mujoco/wheelleg_mujoco/evaluation.py` | 5 | 1 |
| `sim2sim/mujoco/tests/test_evaluation_metrics.py` | 23 | 0 |
| `sim2sim/mujoco/wheelleg_mujoco/angular_limit.py` | 75 | 0 |
| `sim2sim/mujoco/tests/test_angular_limit.py` | 83 | 0 |
| `docs/superpowers/plans/2026-10-10-mujoco-angular-limit-alignment.md` | 64 | 0 |

运行代码共四个文件，新增 86 行、删除 1 行；另有两份测试文件与两份文档。准确差异见 `production-changes.diff`；诊断脚本单独位于本目录。

## 相同力矩复测

使用原八组固定电机力矩、完整匹配初态、无地面、原被动阻尼与原 MuJoCo 1 ms 时间步。闭合开/关分别运行；8 个条件是不同输入，不是重复试验。正式 runtime 与候选 5 ms 实现的 qpos/qvel 全时间点最大误差均为 0，初态误差为 0；observer replay 最大误差为 0；实际电机 force/ctrl 与固定输入逐值一致。

| 闭合状态 | 时间 | 修改前平均根角速度差 rad/s | 修改后 rad/s |
|---|---:|---:|---:|
| 开启 | 5 ms | 0.233422 | 0.233422 |
| 开启 | 10 ms | 0.180470 | 0.180372 |
| 开启 | 15 ms | 0.171948 | 0.087395 |
| 开启 | 20 ms | 0.431066 | 0.089999 |
| 关闭 | 5 ms | 0.001458 | 0.001458 |
| 关闭 | 10 ms | 0.004405 | 0.004126 |
| 关闭 | 15 ms | 0.168866 | 0.003845 |
| 关闭 | 20 ms | 0.544934 | 0.005342 |

闭合开启时，20 ms 误差降低约 79.1%；关闭时降低约 99.0%。首个 5 ms 的闭合响应差仍存在。1 ms 更新的候选补偿反而扩大差异，因此没有采用。另一个仅提高 Isaac 上限至 100 rad/s 的诊断候选也复现了高上限结果，但没有修改正式 Isaac。

## 四组旧策略的八场景评估

每组使用原 actor、原三件套、原 suite context 和相同 keyframe。八场景、命令、500 tick 上限、失败阈值与评分没有改动。平均有效存活时间不包含旧报告中的失败帧。

| Run | Seed | 修改前完成场景 | 修改后 | 原平均有效存活 s | 新存活 s | 新评分（低更好） |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 1375451991 | 0/8 | 8/8 | 1.1275 | 10.0 | 4.848664 |
| 02 | 1145850885 | 0/8 | 8/8 | 1.1325 | 10.0 | 2.342594 |
| 03 | 1380466594 | 0/8 | 8/8 | 1.0100 | 10.0 | 1.002711 |
| 04 | 342388981 | 0/8 | 8/8 | 1.0200 | 10.0 | 0.929706 |

现有 MuJoCo 评分下 run-04 最好，run-03 次之。新排名见 `mujoco-ranking-summary.json`，原始逐帧 CSV 与 summary 位于 `mujoco-evaluation/run-01..04/`。ranking 已重新审计所有 CSV、报告、策略与 source hash。

速度跟踪仍不理想。下面为第 2–10 秒的平均实际前向速度，避免启动阶段把稳态表现混在一起；这些均由 CSV 重新计算并核对全程 MAE：

| Run | 前进（目标 +1.00 m/s） | 后退（目标 -1.00 m/s） | 名义站立（目标 0） |
|---|---:|---:|---:|
| 01 | +0.320 | -0.561 | +0.295 |
| 02 | +0.346 | -0.559 | -0.491 |
| 03 | +0.986 | -0.540 | -0.478 |
| 04 | +0.955 | -0.558 | -0.498 |

run-03/04 前进接近目标；四组后退都明显偏慢，站立仍有漂移。run-04 的组合命令 vx=0.8 m/s，稳态平均实际约 0.943 m/s。稳定完成八场景不等于跟踪误差已经合格。

## 验证与保持范围

- 40 项 MuJoCo 单元/回归测试全部通过，包括非球形世界系惯量、逐刚体响应、周期保持/reset、其他外力矩保留、live 状态不被改写，以及错误单位/周期/旧评估契约拒绝。
- 四组 TorchScript golden vectors 最大误差均为 0（阈值 1e-7），checkpoint metadata/base-task hash 和完整 run manifest 相符。
- 原 93 个受保护文件中仅架构文档、MuJoCo contract/runner/evaluation 四个文件按计划修改；其余 89 个保持 hash。另核对 28 个旧工件保持 hash。
- Actor/Critic、CENet history、AdaBoot、loss、奖励、动作映射、命令、随机化、USD、XML/mesh/model manifest、dt、闭合参数、Isaac reset 修复与 cache 均保持原值。没有改变旧 checkpoint 的训练/续训动力学，不需要为本次 MuJoCo 修复重新生成 Isaac reset cache。

## 结论与剩余问题

这项缺失的刚体限速语义是原 MuJoCo 快速失稳的重要原因：只补这一项，四组相同旧策略从全部 0/8 变为全部 8/8。因此，**目前不需要为了修复这次 sim2sim 快速跌倒再重训**。

策略表现仍有缺口。run-04 的 nominal/low/high stand 平均前向速度绝对误差分别约 0.484/0.533/0.430 m/s，站立命令下仍有明显漂移；四组原 Isaac 报告的 `candidate_not_worse` 均为 false，所以“不差于 PPO 基线”的既有性能门仍未通过。后续若要改善站立、倒车和性能基线，仍需要针对策略性能的工作，不能用本轮稳定性结果替代。首步闭合响应差异也尚未消除。

证据链：`aligned-execution.json`、`analysis.json`、`verification.json`。参考版本文件、候选脚本和正式运行器诊断副本均保留在本目录，可由记录中的完整命令重放；再次执行请使用新输出目录，避免覆盖旧证据。
