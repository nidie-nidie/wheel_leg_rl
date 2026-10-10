# 删除人工限速作用：实施与验收记录

日期：2026-10-11。用户授权删除刚体线/角速度和关节转速的人工限速作用，包括此前为 MuJoCo 对齐而添加的外部角速度力矩。实施与独立复核已完成；没有训练，没有修改随机化范围或地形，没有站立/速度跟踪性能结论。

正式规格为 [Architecture v0.25](../../docs/2026-10-03-wheelleg-dreamwaq-architecture.md) §30 / ADR-069，SHA256：`43CB011D1AA4C63CED5F343FBC2FF010DFA46E4F33E9D2F45511DCEEC04EB427`。

## 改动与保留边界

Isaac 显式把全部刚体线/角速度及全部 actuator 的 solver 转速上限设为有限 float32 最大值哨兵。不能直接删除参数或使用 `None`，否则可能继承 USD/PhysX 的默认上限。实际 27 个刚体的 USD 属性和 26 个 DOF 的 PhysX 值均已读回验证。

MuJoCo 删除 `angular_limit.py` 及其 import/构造/reset/apply，删除 controller 的超速清零力矩分支。五个现有 debug 文件同步删除旧接口依赖和重复限速逻辑；既有 trace 的 limiter/guard 事件明确写 0。旧独立 limiter 单测删除，新增测试验证真实控制不再受这些人工限制。

保留电机位置/速度控制、effort clipping、重力、接触/摩擦、闭链、被动阻尼、armature、穿透修正和终止阈值。主动腿参数仍为 `120/4/18 Nm/0.05 armature`，轮为 `0/0.6/9 Nm/0.05 armature`，被动关节为 `0/0.05/18 Nm/0.005 armature`。PD 本身仍可产生正常制动力矩。

源 USD、正式 MuJoCo XML、模型质量/惯量、动作、命令、奖励、观测、history、CENet、AdaBoot、随机化采样、reset 算法和运行时序未改。`80 rad/s` 关节速度失败阈值仍属于诊断/终止，不向机器人施力。

## 契约和历史工件

当前版本为 `PhysicsV5`、`UnrestrictedVelocityPolicyV1`、`MujocoActionAdapterV2`、`MujocoEvaluationContractV3`。速度政策及实际配置进入 PPO/DreamWaQ 任务、导出和 reset-cache 身份；训练/play/resume/export/加载继续严格校验完整契约。

四个旧 PhysicsV4 包与其权重/golden vectors 均保留原件，禁止重标、直接续训或进入当前正式评估。绑定旧包/hash 的冻结诊断注册仍属于历史条件，未修改注册或历史结果；当前 runtime 拒绝混用。下一次正式训练应从新 run 开始。

MuJoCo `model_manifest.json` 仅更新 action adapter 版本与实现 SHA256。正式 XML SHA256 仍为 `691C607271ED85EFFA388237E00A7B8F8F41CB66F6BA9C49EA64ABD186D803D1`，`MujocoDynamicsSemanticsV1` hash 仍为 `A846A8E4E198A43DAE8F8D9BD2BA565F7B8DC2C67386FB0028BD2530FE1C42D6`。

## 验收结果

工件目录：[remove-artificial-speed-limits-20261011-v1](../artifacts/debug/sim2sim/remove-artificial-speed-limits-20261011-v1/)。其中保留修改前源码、SHA256、完整 diff、测试日志/XML、实际引擎 JSON 及可重跑的 CPU 核验脚本。

| 验收 | 结果 | 证据 |
|---|---|---|
| 主工程单元测试 | 167 passed | `unit-tests-final.log` / XML |
| MuJoCo 完整测试（含新增 debug 回归） | 51 passed | `mujoco-tests-final.log` / XML |
| MuJoCo 两文件独立运行 | 20 passed | `mujoco-isolated-tests.log` / XML |
| Isaac 名义/随机化集成测试 | 2 passed | `isaac-integration.log` / XML |
| Isaac 真实引擎短测 | 两种配置均 8 env × 100 steps，观测/reward finite | `isaac-nominal.json`、`isaac-randomized.json` |
| 旧 PhysicsV4 reset cache | 随机化环境明确拒绝 identity mismatch | `isaac-randomized.json` |
| 新 export → 正式 loader | 使用全新随机初始化、0 次优化更新的测试包，接受 PhysicsV5；新 golden 误差 0 | `export-loader-verification.json` |
| 当前 DreamWaQ debug 链 | formal 1 ms 和 Isaac-sync 5 ms trace 各 2 ticks；debug evaluation 正常运行，guard 事件为 0 | `untrained-debug-*`、`export-loader-verification.json` |
| 旧导出包与 exporter 拒绝 | 4 个旧 PhysicsV4 包均被当前 loader 拒绝；DreamWaQ exporter 拒绝重标旧 physics | `export-loader-verification.json`、`export-old-physics-rejection.json` |
| 旧四组 TorchScript golden | 各组最大绝对误差 0，原物理标签保留 | `golden-verification.json` |
| 受保护资产 | 23 个文件 SHA256 全部不变 | `before.json`、`change-audit.json` |

无接触/重力的自由刚体对照：旧上限刚体降到 `100.0 m/s`、`1.74532961845 rad/s`；解除上限的刚体保持 `200.0 m/s`、`10.0000038147 rad/s`。名义与随机化机器人短测的最大闭链误差分别为 `0.00036565779 m`、`0.00031364147 m`，均小于 `0.005 m`；自动 reset 分别为 36、37 次。

MuJoCo 在超过旧关节上限时仍按原 PD/effort 输出，超过旧刚体上限时不添加限速 wrench。运行时 20 子步轨迹与相同控制下直接 `mj_step` 的参考轨迹精确一致；测试同时覆盖已有外部合法 wrench 的保留。新增 root-cause target-torque 回归确认 debug 参考与正式 controller 一致，诊断阈值测试确认只报告失败、不改写速度。

## 独立复核

按用户追加要求，仅调用一个只读 reviewer：`/root/speed_limit_removal_review`，没有派生其他 agent。它独立检查 212 个修改前源码基线、23 个保护文件、新文件、契约和实际引擎报告，并运行 CPU 检验。

首轮发现 5 个 live debug 文件仍读取已删字段，其中 target-torque 参考仍有旧超速清零逻辑；已修正。复核还发现新增测试单独运行依赖其他测试收集时的 import-path 副作用；已显式配置并独立重跑通过。

最终结论：未发现剩余 P0/P1，未发现多删或漏删人工限速作用。复核范围是删除边界、契约和运行正确性；不证明策略运动性能改善。

## 修改量与后续

源码审计覆盖原 117 个正式源文件以及修改 debug 前补抓的 95 个诊断源文件。实际改动 32 个被审计文件：26 个修改、2 个删除、4 个新增；其中包括架构和测试。实施计划和本文是额外的 Markdown 留证。逐文件新增/删除行数和 SHA256 见 `change-audit.json`，完整差异见 `changes.diff`。

本次未开展旧策略在新物理下的正式八场景 sim2sim 性能评估，因为旧包必须保留旧物理身份。未训练的新测试包仅用于接口验收，不能作为站立/速度跟踪证据。接下来先确定新训练配置，再开展新策略的 Isaac 与 MuJoCo 性能评估；是否扩大随机化、加入地形另行确定。
