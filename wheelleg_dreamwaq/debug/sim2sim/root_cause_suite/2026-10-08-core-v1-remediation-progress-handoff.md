# Sim2Sim Root-Cause Suite Core V1.16 修复进度 Handoff

- 记录日期：2026-10-08
- 当前状态：暂停
- 禁止事项：当前代码处于复核整改中，不得据此启动正式 Core 全流程或发布根因结论

## 一、已经完成

1. 冻结设计仍为 `RootCauseSuiteCoreV1.16`，设计 SHA256：
   `363720317500D6F98A4BBA81CFC7F2E9B5EF5A774974418542A46F602C072FE6`。
2. 正式架构已核对为 Architecture v0.21，SHA256：
   `5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE`。
3. 文档独立复核已通过。
4. 首轮代码实现完成后，修复前单元测试结果为：
   - 项目 Python：`62 passed, 1 skipped`；skip 原因为 Windows 当前环境不能创建测试符号链接。
   - MuJoCo Python：`13 passed`。
5. 已调用独立 agent 做首轮只读代码复核，并保存报告：
   `2026-10-08-sim2sim-root-cause-suite-code-review-01.md`。
6. 首轮代码复核结论为不通过：`P0=4`、`P1=5`、`P2=2`。

## 二、首轮代码复核发现的关键阻塞

1. G00-G03 的覆盖不足，但旧 finalizer 固定写入空 `failed_gates`，可能越过证据门输出强结论。
2. P30/P40/P50 ratio pair 没有在运行时核验配置、reset、激励和 semantic allow-list。
3. P50 的 t=0 接触样本、coupon property、有效接触与 normal gate 没有形成 fail-closed 链。
4. manifest、selected attempt、stage hash、verify、report 与 finalizer 之间存在联合篡改绕过路径。
5. run-id/write scope、worker 第三方 import 前守卫、resume 输入指纹、P20 mismatch 分类、P60 replay seal、C70 plant gate 和 evidence 粒度均需修复。

## 三、本轮已经开始的整改

以下改动均只位于 `debug/sim2sim/root_cause_suite/`：

1. `contracts.py`
   - 新增配置、身份漂移、stage 执行和证据完整性异常类型。
   - 新增 run-id 与 canonical path containment 校验。
2. `integrity.py`（新增）
   - 正式输入、策略、suite source、解释器、distribution metadata、项目可控 cache/Kit tree 的身份快照框架。
3. `worker_bootstrap.py`（新增）
   - 计划保证 worker 在导入第三方模块前安装 `PythonWriteGuard`。
   - 记录 bootstrap argv、环境和 write-audit 证据。
4. `run_suite.ps1`
   - 在首次创建目录前校验 run-id 和 run-root containment。
   - 新增 `WHEELLEG_ROOT_CAUSE_RUN_ROOT`。
5. `__main__.py`
   - 新增父进程 write guard 和 parent bootstrap 记录。
6. `orchestrator.py`
   - 已重写为 guarded worker bootstrap、run identity、definition hash、selected attempt、dependency state hash、strict verify/report/rebuild 模型。
   - 当前尚未重新运行测试，不能视为已验证实现。
7. `cli.py`
   - 开始按冻结契约区分 exit code `2/3/4/5`。
8. `causal_contract.py`（新增）
   - 新增 scenario identity、configuration semantics、repeatability key、JSON-pointer semantic diff 和 ratio-pair fail-closed validator。
9. `mujoco_worker.py`
   - 已开始接入 causal identity。
   - 已开始把 P20 的“测量有效”和“发现物性 mismatch”拆开。
   - 已开始补 common-sphere property/contact identity。
   - 该文件刚完成局部修改，尚未做 AST、单元或真实 worker 验证。

## 四、当前明确未完成

1. `mujoco_worker.py` 当前局部修改后的验证与必要修正。
2. Isaac worker 的 causal identity、P50 raw t=0 样本/clearance/property/contact gate。
3. `core_stage_worker.py` 使用统一 guarded bootstrap、完整 G00-G03 gate、ratio binding、P60 原子 replay seal 和 C70 plant gate。
4. `analysis.py` 的 result-manifest binding、P50 normal/tangential gate 和 G01 threshold binding。
5. `finalize.py` 读取真实 gate 状态、只接受 orchestrator 已验证 attempt、P20/P50/C70 新分类和完整 evidence 展开。
6. report 原子重建、final-state 联合验证和新 schema。
7. 首轮复核指出的 mutation tests 和真实 Isaac/MuJoCo 集成测试。
8. 修复后的第二次独立代码复核。
9. 正式一键 Core 诊断运行、结果独立复核和最终正式文件 after-hash。

## 五、当前是否已经有结论

**没有可认证的正式根因结论。**

此前手工 probe 曾观察到：

- common-sphere 法向接触后出现材料级差异；
- nominal friction 滑移有差异，而 friction=0 对照差异显著减小；
- P60 同动作整机短时开环仍分歧；
- C70 的初步计算把 DreamWaQ checkpoint 判为 plant error 的放大器。

这些只能称为“待复核的诊断线索”，不能称为正式结论。原因是首轮代码复核已经证明旧证据链可能越过不完整 gate、缺少 ratio causal binding，并可能在 P50 接触证据无效时误分类。

因此当前准确表述是：

> 本轮已经找到“接触/摩擦层值得优先验证、checkpoint 可能放大 plant error”的线索，但尚未形成通过 Core V1.16 证据门和独立结果复核的根因结论。

## 六、恢复工作时的固定顺序

1. 完成上述未完成整改，不运行正式 Core。
2. 重跑 AST、项目 Python、MuJoCo Python 和针对性 mutation tests。
3. 调用独立 agent 做第二次只读代码复核；P0/P1 必须清零。
4. 只有复核通过后，运行一键 Core 诊断。
5. 对完整 trace、threshold、verdict 和 report 调用独立结果复核。
6. 最后重新核验正式文件 hash，并输出正式结论；若证据仍不足，明确写“本轮没有找到可认证根因”。
