# WheelLeg DreamWaQ Reset Observation Remediation Handoff

日期：2026-10-09  
工作区：`E:\wheel_leg_rl-main`  
工程目录：`E:\wheel_leg_rl-main\wheelleg_dreamwaq`  
当前阶段：DreamWaQ Phase 2 首个 1000-iteration checkpoint 已完成训练与 Isaac/MuJoCo 评估；后续训练已暂停。Sim2Sim RootCauseSuiteCoreV1.17 已完成设计、实现、独立文档复核、独立代码复核、测试、实际诊断和独立结果复核，并在冻结的 G01/G02 硬门处正确停止。已确认正式 Isaac reset observation 路径存在 stale root angular velocity cache 缺陷，尚未证明该缺陷是 MuJoCo 首次失稳的唯一根因。

## 1. 本文用途

本文用于把 2026-10-07 至 2026-10-09 这次长对话的结尾，直接转换为下一次对话的开头。下一位执行者应先完整读取本文，再读取本文列出的正式架构、RootCauseSuite 冻结设计、运行调查和独立复核报告。

本文不替代正式架构文档，也不授权绕过任何诊断门禁。必须区分以下状态：

- **已实现并验证**：代码、测试和运行证据都存在。
- **已确认缺陷**：证据足以支持精确分类，可以进入正式修复阶段。
- **被门禁阻断**：按冻结设计必须停止，继续运行不会产生有效 verdict。
- **尚未证明**：不能因为候选合理就写成根因。

当前最重要的结论是：

> 已经确认生产 `WheelLegFlatEnv` 的重复 reset 可以返回旧的 root angular velocity 缓存，DreamWaQ history wrapper 会把这份错误 Actor observation 复制为五帧初始 history。冻结分类为 `SIM2SIM_ADAPTER_BUG`，窄因是 `isaac_reset_returned_observation_stale_root_angular_velocity_cache`。该结论允许开始修复正式 reset observation 路径，但不允许直接修改 MuJoCo 参数、关节限位、奖励、动作、命令或随机化，也不允许把它升级为 MuJoCo 首次失稳的唯一根因。

## 2. 权威文档与冻结身份

### 2.1 正式架构

路径：

```text
E:\wheel_leg_rl-main\docs\2026-10-03-wheelleg-dreamwaq-architecture.md
```

当前身份：

```text
Architecture v0.21
SHA256 5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE
```

Phase 2 的网络维度、history 布局、CENet 数学契约、Actor/Critic 输入、storage、checkpoint、导出和评估语义仍以该文档为上游规范。本次调查没有修改这些契约。

### 2.2 RootCauseSuite 冻结设计

路径：

```text
E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\root_cause_suite\2026-10-07-sim2sim-root-cause-suite-design.md
```

当前身份：

```text
RootCauseSuiteCoreV1.17
SHA256 8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD
```

独立设计复核：

```text
E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\root_cause_suite\2026-10-08-sim2sim-root-cause-suite-v1.17-review.md
SHA256 58E10632A61023C1E42A0CE197165BABE5EA77345192E7003424846C9AA5AB63
结论：批准
计数：P0=0, P1=0, P2=0
```

### 2.3 RootCauseSuite 最终代码复核

路径：

```text
E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\root_cause_suite\2026-10-08-sim2sim-root-cause-suite-code-review-18.md
```

身份与结论：

```text
SHA256 3F18F0BDDBD5FB628EDA6CF4488313295AC10E098EF53B5A2AC9F04A5C8147BE
Decision: APPROVED
Counts: P0=0, P1=0, P2=0
```

### 2.4 本次根因调查和结果复核

调查报告：

```text
E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\root_cause_suite\2026-10-08-g01-reset-observation-runtime-investigation.md
SHA256 8C4C562AC6B2C64AF2384A1A9CF79906558A94B4FFCE1E56989B7FC76F51E4C8
```

独立结果复核：

```text
E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\root_cause_suite\2026-10-08-g01-reset-observation-result-review-01.md
SHA256 B55C82A689C72D9BE85711FA5F7F5A0C02347AF45203D0D71DFC2919543490A0
结论：APPROVED
计数：P0=0, P1=0, P2=2, P3=2
```

这里的 `APPROVED` 只批准 stale-cache 调查结论、冻结分类和停止决定，不批准在当前 reset 语义下继续生成 P10-P60/C70 verdict。

## 3. Phase 2 训练与评估的当前状态

### 3.1 训练 suite

正式 suite：

```text
E:\wheel_leg_rl-main\wheelleg_dreamwaq\artifacts\phase2_dreamwaq\training-suite-20261007-185711
```

计划种子：

```text
917410003
424960785
386665039
1265093471
```

实际状态：

- run-01：seed `917410003`，1000 iteration 完成。
- run-02：seed `424960785`，启动后在日志 iteration 65 被用户要求停止，状态为 `interrupted_by_user`。
- run-03、run-04：未开始。
- suite 状态：`paused_after_run_01`。

**不得在修复正式 reset observation 后继续 resume 这个旧 suite。** 该 suite 的 source fingerprint 和训练语义绑定修复前源码。如果后续决定重新训练，应创建新的 suite ID，并从头执行四个独立 seed。

### 3.2 DreamWaQ run-01 身份

checkpoint：

```text
E:\wheel_leg_rl-main\wheelleg_dreamwaq\logs\rsl_rl\wheelleg_flat_dreamwaq\2026-10-07_18-57-15_rtx5070_fudan-v1_dreamwaq-v1-suite-20261007-185711-run01-seed917410003\model_999.pt
SHA256 F3FE2AAEE7975AD64FB7C5F0686D7632AFAD81A16419415135AF56680595BA4C
```

导出 Actor：

```text
E:\wheel_leg_rl-main\wheelleg_dreamwaq\artifacts\phase2_dreamwaq\training-suite-20261007-185711\run-01-evaluation\export\actor.ts
SHA256 4C2C64C2D9F9F88532FCEFF4AE45AA049F970646761D2D0ACE0A29678621D6C7
```

run-01 在冻结 Isaac 八场景评估中：

- 8/8 场景完成，平均 survival 为 1.0。
- estimator velocity acceptance 通过。
- context acceptance 通过。
- 但平均回报约 `22.0053`，低于共享 Phase 1R run-03 baseline 的约 `27.6767`，因此 `candidate_not_worse=false`。

run-01 在正式 MuJoCo 八场景中：

- 0/8 场景完成。
- 平均 survival fraction `0.09525`。
- action saturation fraction 约 `0.70150`。
- 多数场景在约 24-57 tick 因 tilt 或 base height 失败。

因此 run-01 是已完成训练并可用于诊断的 checkpoint，不是通过模型，也不能证明 DreamWaQ 提高了 sim2sim 鲁棒性。

### 3.3 训练后的 debug-only 对照

Broad Sim2Sim diagnosis：

```text
E:\wheel_leg_rl-main\wheelleg_dreamwaq\artifacts\debug\sim2sim\dreamwaq-run01-diagnosis-20261007-v1\analysis\report.md
```

该报告曾观察到：

- 统一 MuJoCo 为 `5 ms` 物理步长没有解决 DreamWaQ 失败，并显著恶化 Phase 1R run-03。
- 相同动作、零动作和开环 replay 下，差异仍在第一物理区间出现。
- policy feedback 会放大差异，但不是第一物理差异的唯一来源。

Unbounded diagnosis：

```text
E:\wheel_leg_rl-main\wheelleg_dreamwaq\artifacts\debug\sim2sim\dreamwaq-run01-unbounded-20261007-v1\report.md
```

结果：解除 runtime action clip、腿目标 clip、effort clip、velocity guard 和 MJCF limits 后，平均 survival 从 `0.09525` 降至 `0.02925`。正式限位是在抑制失控反馈，不是当前已证明的首发根因。

这些报告仍是有效的 debug-only 历史证据，但 RootCauseSuite 后来发现 reset/history 硬门失败。因此它们不能在修复并重过 G01/G02 前升级为正式物理 verdict。

## 4. 本次完成的 RootCauseSuite 工作

### 4.1 设计与实现

在以下目录内完成了 Core V1 设计和代码，写入范围保持在 debug suite 及其 runs：

```text
E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\root_cause_suite
```

套件包括：

- G00 冻结身份、写入边界和解释器门。
- G01 engine 内重复性和 reset 三相 identity。
- G02 observation/history/Actor/adapter 等价门。
- G03 instrumentation neutrality。
- P10-P60 逐层物理隔离。
- C70 checkpoint sensitivity/amplifier 分析。
- PowerShell 唯一入口、跨解释器 worker、证据封存、resume/seal、finalize 和 fail-closed verdict。

设计和实现经过多轮独立复核，最终代码复核 18 为 `APPROVED`，P0/P1/P2 均为 0。

### 4.2 最终测试与身份验收

2026-10-09 重新执行的最终验收：

```text
项目解释器，排除 test_mujoco_worker.py：281 passed, 4 skipped in 401.47s
MuJoCo 冻结解释器：12 passed in 16.13s
py_compile：52 files, exit_code=0
scenario catalog：22 scenarios，identity hash 匹配
formal snapshot：17 entries，all_expected=True
```

四个 skip 都是当前 Windows 用户缺少目录 symlink 创建权限，不是测试失败。

验收产物：

```text
debug\sim2sim\root_cause_suite\runs\completion-audit-20261009-non-mujoco-rerun
debug\sim2sim\root_cause_suite\runs\completion-audit-20261009-mujoco-rerun
debug\sim2sim\root_cause_suite\runs\completion-audit-20261009-pycompile-rerun
debug\sim2sim\root_cause_suite\runs\completion-audit-20261009-catalog-rerun-03
debug\sim2sim\root_cause_suite\runs\completion-audit-20261009-formal-snapshot-rerun-03
```

冻结 catalog identity：

```text
8374479CD53585DFFB5F2840C20F6E387A211007170E4BE76A69C32F1A97EE67
```

### 4.3 正确停止的位置

RootCauseSuite 没有继续生成 P10-P60 和 C70 正式结论。原因不是代码没跑完，而是冻结设计要求：只要 reset returned observation/history 不可信，后续物理和 checkpoint 因果判断全部无效。

当前门状态：

```text
G00: PASS
G01/G02: BLOCKED by reset returned observation identity split
P10-P60: BLOCKED, no valid physical verdict
C70: BLOCKED, no valid checkpoint verdict
```

下一次不得通过删除 `reset_returned_policy_hash`、改用 post-forward physical hash、只修 debug worker 或放宽 repeatability key 来绕过该门。

## 5. 已确认缺陷

### 5.1 现象

基线 run：

```text
debug\sim2sim\root_cause_suite\runs\verification-20261008-review18-isaac-p10-a-smoke
```

三次 P10_A repetition 的以下内容一致：

- pre-forward initial-condition hash；
- post-forward state hash；
- excitation/profile identity；
- 除故意不同的 `profile_repetition` 元数据外，完整物理 trace 逐位一致。

唯一分裂的是 reset API 返回给 Actor 的 policy state：

```text
rep0 reset_returned_policy_hash
D01A74F826E72AC358ABEC0CD7A34F2A3E367AEC4E825E5ACC2D85588A36DD50

rep1/rep2 reset_returned_policy_hash
3E46EDBC5F3F7F9DC924C62547F0B3DF99D15976FB6E5B30598401A61CD4B442
```

### 5.2 精确差异

只有 ActorObsV1 的 `0:3` angular velocity slice 不同：

```text
rep0   [0.0, 0.0, 0.0]
rep1/2 [-1.1614577e-05, 3.3813692e-05, -1.0379885e-05]
```

rep1/2 的三个 float32 值逐位等于上一 rollout 末端 base angular velocity 乘冻结 normalization scale `0.25`。同一 reset sample 的 direct post-forward 物理 angular velocity 精确为零。previous action 六维数组在所有 capture 中逐位全零。

因此这不是 PhysX reset 物理状态变化、RNG 变化、command 变化、previous action 污染或 solver nondeterminism，而是 reset observation/cache coherence 缺陷。

### 5.3 正式生产路径复现

正式 probe：

```text
debug\sim2sim\root_cause_suite\runs\diagnostic-20261008-formal-reset-observation-01
```

它直接实例化正式 `WheelLegFlatEnv`。五次 reset 的 Actor angular slice 中，第一次为零，后四次约有 `0.265-0.267` 的 y 轴归一化角速度。Actor indices `3:25` 保持逐位一致。

更精确的证据边界是：生产类和正式 reset API 路径在 nominal 单环境诊断配置下复现；它不是完整训练配置实例的逐项复现。

### 5.4 源码数据流

正式任务：

```text
source\wheelleg_dreamwaq\wheelleg_dreamwaq\tasks\direct\wheelleg_flat\env.py
_read_state()        lines 653-691
_capture_state()     lines 693-697
_current_state()     lines 699-700
_get_observations()  lines 761-769
_reset_idx()         lines 852-918
当前 SHA256 5399B29B92B54851EB0A5FA880EA9386C82F202A5C337F19ABE4EDF300C2969C
```

上游 Isaac Lab：

```text
dependencies\IsaacLab-v2.3.2\source\isaaclab\isaaclab\envs\direct_rl_env.py
reset order lines 292-331
normal step scene.update(dt=physics_dt) lines 383-384

dependencies\IsaacLab-v2.3.2\source\isaaclab\isaaclab\assets\articulation\articulation.py
write_root_com_velocity_to_sim() lines 497-525

dependencies\IsaacLab-v2.3.2\source\isaaclab\isaaclab\assets\articulation\articulation_data.py
update(dt) lines 97-102
root_link_vel_w cache refresh lines 487-504
root_link_ang_vel_b derivation lines 813-819
```

当前机制：

1. `_reset_idx()` 写 nominal root velocity 和 joint state。
2. `write_root_com_velocity_to_sim()` 更新 COM velocity buffer，但不会使所有 derived root-link velocity cache 失效。
3. 任务立即读取 `robot.data.root_link_ang_vel_b` 并缓存到 `self._state`。
4. `DirectRLEnv.reset()` 执行 write/forward 后直接获取 observation，没有正常 step 中的正 `dt` scene-data update。
5. `_get_observations()` 返回已缓存的 `self._state`。
6. DreamWaQ history wrapper 把返回的 current observation 复制为五帧 history，所以 stale angular velocity 被复制五次。

history wrapper：

```text
source\wheelleg_dreamwaq\wheelleg_dreamwaq\algorithms\dreamwaq\history_wrapper.py
initial reset history fill line 60
done history refill line 68
```

### 5.5 三个 counterfactual

全部是 debug-only 内存实验，没有修改正式源码：

1. 只调用 `_capture_state()`：无效。
2. `scene.update(dt=0.0)` 后重新读取：无效，因为 timestamp 没有推进。
3. `scene.update(dt=physics_dt)` 后重新读取：三次 returned-policy hash 和 repeatability key 完全一致，完整物理 trace SHA256 仍与 baseline 相同。

第三项只是证明“推进 scene-data timestamp 后重新读取”在所测路径中足以消除 split。它不是可以直接复制进正式环境的最终修复方案，因为 `scene.update(physics_dt)` 可能引入额外物理时间语义。

### 5.6 冻结分类

```text
SIM2SIM_ADAPTER_BUG
```

窄因：

```text
isaac_reset_returned_observation_stale_root_angular_velocity_cache
```

## 6. 已确认与尚未确认的边界

### 6.1 已确认

- 正式 `WheelLegFlatEnv` reset observation 路径受 stale root angular velocity cache 影响。
- 重复 reset 时，返回给 Actor 的 observation 可能不是 post-forward 真实 reset 状态。
- DreamWaQ 五帧 history 会复制该错误 observation。
- 当前 G01 repeatability key split 被该缺陷完整解释。
- 可以开始针对正式 reset observation 路径设计和实施最小修复。

### 6.2 尚未确认

- 该缺陷不是已经证明的 MuJoCo 首次失稳唯一根因。
- 尚无修复后 G01/G02 通过证据。
- 尚无修复后 P10-P60 物理归因。
- 尚无修复后 P60-D exact replay 或 C70 checkpoint amplifier verdict。
- 尚未量化训练期间跨 episode history 污染对 run-01 checkpoint 的影响。
- 尚未排除闭链、drive/contact、质量/惯量和整机耦合等其他候选。

### 6.3 独立结果复核保留意见

必须在下一阶段保留以下限制：

- 三个内存 counterfactual 的产物没有绑定 exact wrapper source/hash，属于 provenance P2，不推翻结论，但以后应补强。
- per-record pre/post phase 原始 canonical payload 未保存，已有 hash 和 trace 可核验，但不能从产物单独重建全部 phase payload。
- 调查报告中“jointly necessary and sufficient”措辞过强；只能说在三个预声明 counterfactual 中，只有正 `dt` timestamp 推进后重新读取修复了 split。
- formal probe 是生产类/API 在 nominal 单环境诊断配置下复现，不等于完整训练配置实例复现。

## 7. 下一次对话的主任务

下一次主线不是继续旧训练，也不是调 MuJoCo 参数，而是：

> 为正式 Isaac reset observation/cache coherence 缺陷设计并实现最小生产修复，通过独立设计复核、独立代码复核和全新 G01/G02 验收；只有硬门重新通过后，才恢复 P10-P60/C70 和现有 checkpoint 的有效 sim2sim 判断。

### 7.1 第一阶段：正式修复设计

先完整阅读本 handoff、Architecture v0.21、RootCauseSuiteCoreV1.17、调查报告和结果复核。随后只读检查正式 `env.py`、DreamWaQ history wrapper 和 Isaac Lab 2.3.2 reset/articulation cache 实现。

设计必须满足：

- reset 返回前的 Actor/Critic observation 使用当前 reset 物理状态。
- 第一次 reset 与后续重复 reset 语义一致。
- 上一个 episode 的 root velocity 不得泄漏到新 episode。
- 不额外推进物理时间，不改变 reset 初态。
- 不改变 command、previous action、randomization、termination、奖励、动作或控制周期。
- 不通过硬编码 Actor `0:3=0` 隐藏其他 cache coherence 问题。
- 不依赖修改 Isaac Lab vendor 源码，除非有明确证据证明任务层无法正确修复并得到用户批准。
- 明确 checkpoint compatibility：修复后的环境可以加载旧 checkpoint 做诊断，但旧 checkpoint 不能被视为在新语义下训练得到。

不得直接把诊断用的 `scene.update(physics_dt)` 原样写入正式代码。应比较可维护方案，例如正确刷新/失效 articulation derived cache、调整任务状态采集时机，或使用不推进 physics 的官方同步路径，并用运行证据选择。

设计文档完成后，调用独立 agent 做只读文档复核。P0/P1 必须清零后才能修改正式代码。

### 7.2 第二阶段：实现与回归测试

预计最小正式修改集中在：

```text
source\wheelleg_dreamwaq\wheelleg_dreamwaq\tasks\direct\wheelleg_flat\env.py
```

可能需要增加测试或 debug 证据代码，但不得无证据扩张到 USD、奖励、动作、命令、随机化或 MuJoCo 正式参数。

至少覆盖：

- 在制造非零 terminal root angular velocity 后重复 reset。
- direct PhysX post-forward state 与 reset returned Actor `0:3` 一致。
- Actor indices `3:25`、command 和 previous action 没有非预期变化。
- reset 后五帧 history 都等于正确的 first policy observation。
- repeated reset identity 可重复。
- 修复不额外推进 simulation time 或产生隐藏物理 step。
- fresh、done-reset、play/evaluation 和 resume 相关路径不回归。

实现后调用独立 agent 进行代码复核。P0/P1 必须清零。

### 7.3 第三阶段：更新冻结身份并重跑硬门

正式 `env.py` 修改后，RootCauseSuiteCoreV1.17 绑定的 `formal_env` SHA256 必然变化。不得静默覆盖旧哈希或修改既有 run 产物。

应当：

1. 保留 V1.17 和既有 evidence 不变。
2. 创建明确的 post-fix suite/design revision 或 remediation identity 更新。
3. 独立复核新的设计和正式源码身份。
4. 使用全新 run ID，从当前 source identity 重新执行 G00、G01 和 G02；禁止 resume 旧失败 run。
5. 调用独立 agent 复核新运行结果。

硬门验收至少要求：

- 三次 P10_A 的 pre-forward、post-forward、returned-policy、excitation 和 repeatability key 全部一致。
- reset returned observation 与 direct physical state 一致。
- 五帧 history identity 正确。
- Actor/adapter 等价门通过。
- formal file guard 除已批准的源码身份更新外无漂移。

### 7.4 第四阶段：恢复完整诊断

只有 G01/G02 全新通过后：

1. 继续 G03。
2. 执行 P10-P60 的完整物理隔离。
3. 执行 P60-D exact replay。
4. 执行 C70 双 anchor checkpoint sensitivity。
5. 使用现有 DreamWaQ run-01 checkpoint 重新评估 Isaac/MuJoCo，判断 runtime reset 修复本身能解释多少差异。

如果旧 checkpoint 在修复后的环境仍然失败，不能据此否定修复。旧 checkpoint 可能已经在受污染的 reset history 下学习。只有在环境硬门通过后，才能决定是否创建全新的四 seed 训练 suite。

## 8. 下一阶段禁止事项

- 不继续旧 `training-suite-20261007-185711` 的 run-02/run-03/run-04。
- 不在 G01/G02 通过前启动新的 1000-iteration 正式训练。
- 不修改 USD 或 AssetBundle。
- 不修改奖励、动作、命令、随机化、termination 或 control clock。
- 不修改 MuJoCo XML、接触、闭链、质量、惯量、PD、effort limit 或物理步长。
- 不再次通过“全部解除限位”寻找正式结论；已有 debug 证据显示它会更快失败。
- 不修改 checkpoint 或把旧 checkpoint 包装成新训练结果。
- 不把 `SIM2SIM_ADAPTER_BUG` 误写成“MuJoCo adapter 已被证明错误”；当前窄因位于 Isaac reset observation cache。
- 不把本次缺陷写成 MuJoCo 首次失稳的唯一根因。
- 不修改既有 RootCauseSuite run 产物；新证据必须写入新 run ID。

## 9. 下一次对话的完成条件

下一次对话至少应完成以下闭环之一：

### 目标 A：修复与硬门闭环

- 正式修复设计完成并通过独立文档复核。
- 正式代码最小修改完成并通过独立代码复核。
- 单元/集成测试通过。
- 新 source identity 被显式冻结。
- 全新 G01/G02 运行通过并经过独立结果复核。

### 目标 B：发现新的阻塞矛盾

如果无法在不改变 reset 物理语义的前提下修复，应给出准确文件、行号、实验和失败证据，并明确声明本轮未完成修复。不得用未经验证的 workaround 让门禁表面通过。

## 10. 可直接用于下一次对话的 Prompt

```text
请先完整读取以下 handoff，并将其作为本次对话的完整上下文：

E:\wheel_leg_rl-main\wheelleg_dreamwaq\docs\2026-10-09-wheelleg-dreamwaq-reset-observation-remediation-handoff.md

随后完整读取以下权威文档和复核件：

1. E:\wheel_leg_rl-main\docs\2026-10-03-wheelleg-dreamwaq-architecture.md
2. E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\root_cause_suite\2026-10-07-sim2sim-root-cause-suite-design.md
3. E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\root_cause_suite\2026-10-08-g01-reset-observation-runtime-investigation.md
4. E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\root_cause_suite\2026-10-08-g01-reset-observation-result-review-01.md

先核对以下冻结身份：

- Architecture v0.21：5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE
- RootCauseSuiteCoreV1.17：8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD
- G01 调查报告：8C4C562AC6B2C64AF2384A1A9CF79906558A94B4FFCE1E56989B7FC76F51E4C8
- 独立结果复核：B55C82A689C72D9BE85711FA5F7F5A0C02347AF45203D0D71DFC2919543490A0

当前已经确认的缺陷分类是：

SIM2SIM_ADAPTER_BUG

窄因是：

isaac_reset_returned_observation_stale_root_angular_velocity_cache

正式 WheelLegFlatEnv 在重复 reset 后可能把上一个 rollout 的 root angular velocity 缓存返回给 Actor；DreamWaQ history wrapper 随后把该错误 observation 复制成五帧 history。该问题已经在生产类/API 路径的 nominal 单环境诊断配置下复现，并通过独立结果复核，但尚未证明它是 MuJoCo 首次失稳的唯一根因。

本次对话的主线是修复正式 Isaac reset observation/cache coherence 缺陷。请先只读检查：

- source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py
- source/wheelleg_dreamwaq/wheelleg_dreamwaq/algorithms/dreamwaq/history_wrapper.py
- dependencies/IsaacLab-v2.3.2 中 DirectRLEnv reset 和 ArticulationData cache/timestamp 路径

然后按以下顺序执行：

1. 写一份最小生产修复设计。修复必须让 reset returned observation 使用当前 reset 物理状态，同时不得额外推进物理时间，不得改变奖励、动作、命令、随机化、termination、控制周期或 MuJoCo 语义。不要把诊断用 scene.update(physics_dt) 直接当成最终修复。
2. 调用一个独立 agent 对修复设计做只读复核；P0/P1 清零后再改正式代码。
3. 实施最小正式代码修改并增加重复 reset、direct PhysX 对齐、五帧 history、无额外物理步和 fresh/done-reset 路径回归测试。
4. 调用一个独立 agent 做代码复核；P0/P1 清零后再运行正式门禁。
5. 保留 RootCauseSuiteCoreV1.17 和既有 run 不变；为修复后的 formal_env SHA256 创建显式的新 source identity/design revision，不能静默改旧哈希。
6. 使用全新 run ID 重跑 G00/G01/G02，禁止 resume 旧失败 run；随后调用独立 agent 复核测试和运行结果。
7. 只有 G01/G02 全新通过后，才继续 G03、P10-P60、P60-D 和 C70，并用现有 DreamWaQ run-01 checkpoint 重测 Isaac/MuJoCo。

当前 DreamWaQ 正式训练 suite 只完成 run-01；run-02 在 iteration 65 被用户停止，run-03/run-04 未开始。正式 env 修复后不得继续 resume 旧 suite；如果最终需要重训，必须创建新的 suite ID，从头执行四个独立 seed。

暂时不要训练，不要修改 USD、奖励、动作、命令、随机化、termination、MuJoCo 参数、关节限位或 checkpoint。不要把当前缺陷表述为 MuJoCo 首次失稳的唯一根因。如果无法形成安全修复或新的有效结论，请明确记录阻塞证据，不要为了让门禁通过而放宽 identity 契约。
```
