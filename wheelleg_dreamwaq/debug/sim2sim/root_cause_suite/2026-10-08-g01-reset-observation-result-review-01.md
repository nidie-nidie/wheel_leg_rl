# G01 Reset Observation Result Review 01

**复核日期：** 2026-10-09  
**复核类型：** 独立、只读运行结果复核  
**冻结设计：** `RootCauseSuiteCoreV1.17`  
**冻结设计 SHA256：** `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD`  
**源码、既有运行产物与正式工程修改：** 无

## Findings

### P0

无。

### P1

无。

### P2-1：三个 counterfactual 的运行产物没有绑定实际内存 wrapper 源码

涉及产物：

- `runs/diagnostic-20261008-p10-post-forward-state-refresh-01`
- `runs/diagnostic-20261008-p10-scene-update-refresh-01`
- `runs/diagnostic-20261008-p10-scene-update-physics-dt-01`

三个 stdout 分别在第 25-27 行记录了三次：

- `POST_FORWARD_STATE_REFRESH_APPLIED=1`
- `POST_FORWARD_SCENE_UPDATE_REFRESH_APPLIED=1`
- `POST_FORWARD_SCENE_UPDATE_PHYSICS_DT_APPLIED=1`

结果文件足以独立确认“带这些标签的两个负对照没有改变结果，带 physics-dt 标签的正对照使 returned-policy identity 收敛”。但是，各 run 的 `identity.json` 没有保存内存 wrapper 的源码、源码 SHA256、精确 monkeypatch 目标或启动命令，因此无法只凭封存产物证明每个 wrapper 除报告所述操作外没有执行其他逻辑。

最小补强：今后同类非正式 counterfactual 应把 exact wrapper 源码置于 `NON_EVIDENTIARY`，记录 SHA256、patch target、调用顺序及参数，或把等价 canonical source payload/hash 写入 run identity。

影响：这是 counterfactual provenance 缺口，不推翻本次 stale-cache 结论。原始 Actor 分量、位级终端角速度关系、直接 PhysX trace、正式类路径复现和源码数据流形成了不依赖该 provenance 的交叉证据。

### P2-2：per-record pre/post phase 原始 payload 未随 smoke 产物保存

`result.json` 保存了三个 repetition 的 `pre_forward_initial_condition_hash`、`post_forward_state_hash`、`reset_returned_policy_hash`、`excitation_hash` 和完整 record identity，但没有保存可逐项重算前两个 hash 的 per-record canonical phase payload。

本次能够独立完成：

- 比较三个记录中的 pre/post hash；
- 重算 `profile_identity_hash`、`excitation_hash`、`repeatability_key` 和 `record_identity_hash`；
- 从 `trace.npz` 逐数组验证 post-reset 后完整物理轨迹一致；
- 从原始组件捕获重建并重算两个 `reset_returned_policy_hash`。

本次不能仅由封存产物重新生成 `BB2A...48B39` 和 `A063...A5FC` 的原始 phase payload 内容。最小补强是为每个 repetition 保存 canonical pre/post payload 或精确 evidence ref，并由 verifier 重新计算 hash。

影响：不改变“存储的 pre/post identity 相同”和“物理 trace 相同”的结论，但限制了对 pre/post hash 内部字段完整性的独立再审计能力。

### P3-1：调查报告第 138 行的“jointly necessary and sufficient”表述过强

现有实验严格证明的是：

- 单独重新 `_capture_state()` 不充分；
- `scene.update(0.0)` 后重新读取不充分；
- 在所测路径中，`scene.update(physics_dt)` 后重新读取足以消除本次 identity split。

它没有穷举手动失效特定缓存、直接重写 derived buffer 或其他合法刷新方法，因此不能证明该组合是全局唯一的必要条件。建议将表述限定为“在三个预声明 counterfactual 中，只有推进 scene-data timestamp 后重新读取修复了该 split”。

### P3-2：formal probe 证明生产类/API 路径复现，不等于完整训练配置实例复现

`isaac_worker.py:3662` 使用 `make_debug_env_cfg()` 构造单环境、固定命令的诊断配置，随后在 `isaac_worker.py:3669-3670` 的 `mode == "formal"` 分支直接实例化 `WheelLegFlatEnv(cfg)`。因此调查报告“使用 `WheelLegFlatEnv` 而非 debug subclass”的事实正确；更精确的边界是“生产类和 reset API 路径在 nominal、单环境诊断配置下复现”。

此外，`instrumentation.json` 记录 `mode=formal` 与配置前后 hash，但没有直接保存 `type(env)`、正式 `env.py` SHA256 或完整 cfg hash。当前源码分支和产物 mode 可以共同确定类选择，但运行产物本身的类/source provenance 仍可加强。

## 复核结论

**APPROVED**

计数：`P0=0, P1=0, P2=2, P3=2`。

这里的 `APPROVED` 只表示本调查的核心运行结论、冻结分类和阻断决定成立。它不表示批准继续使用当前 reset 语义生成 P10-P60/C70 正式 verdict。

**运行门状态仍为 BLOCKED：在正式 reset observation 路径修复、独立代码复核并用全新证据重新通过 G01/G02 前，P10-P60 物理 verdict 与 C70 checkpoint verdict 均不得成立。**

## 独立重算方法

只读解析指定 JSON、stdout 和 NPZ；使用 canonical JSON 规则 `sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False` 与 SHA256 重算 identity；使用 NumPy 对 dtype、shape、float32 位模式和 trace 数组逐项比较。未启动 Isaac、MuJoCo 或 pytest，未修改任何既有运行目录。

## 1. Baseline 三重复

对象：`runs/verification-20261008-review18-isaac-p10-a-smoke`

### 1.1 文件身份

- `result.json`: `20933E6A51E95C2C799AA1737293A3A8294F82B8DAAAFCF50B39E8928E7AB905`
- `trace.npz`: `FD42D4ADC980CB103D1E525BD82A48D63849BCA8EEAF08111BCF95A05D77B82A`
- `smoke.stdout.txt`: `9916312C1A33AE81EE49B1888A022428EEDEF38F887C91221E86E1964722F222`

### 1.2 per-record identity 重算

三个 record 的以下字段完全相同：

- `configuration_hash`: `1AF7DDE7CA3AD91E14C688CCB55DB59FECF64F34493B9407F4569BB92C49E0D2`
- `pre_forward_initial_condition_hash`: `BB2AE1BFEC8B2FB07F59DE2E8F2F0ECEC9AFC1926ECB2B8929F237BC3E648B39`
- `post_forward_state_hash`: `A063A5BB86BD754598617CBF5FCFDA4D6F305969CC920F7F58AAFDC77A56A5FC`
- per-record `excitation_hash`: `1C2DC3BE4342DDEFA0432D0FDE79EE9FA635298487FE2F9FF9FB12E0D581A062`

唯一分裂的构成字段是 `reset_returned_policy_hash`：

| repetition | returned-policy hash | 重算 repeatability key |
|---:|---|---|
| 0 | `D01A74F826E72AC358ABEC0CD7A34F2A3E367AEC4E825E5ACC2D85588A36DD50` | `B90B4E411CE4A6F572CB95037C13215FEB9DB9333F740A66D14681FFE85EADCA` |
| 1 | `3E46EDBC5F3F7F9DC924C62547F0B3DF99D15976FB6E5B30598401A61CD4B442` | `14C4F0E161517335037A80CC4535201367F6E61A01169686D6A395AA4E1B2FF7` |
| 2 | `3E46EDBC5F3F7F9DC924C62547F0B3DF99D15976FB6E5B30598401A61CD4B442` | `14C4F0E161517335037A80CC4535201367F6E61A01169686D6A395AA4E1B2FF7` |

对三个 record 分别重算后：

- `profile_identity_hash` 一致；
- `excitation_hash` 一致；
- 由五个冻结构成字段重算的 `repeatability_key` 与记录一致；
- 去除 `record_identity_hash` 后重算的 record SHA 与记录一致。

所以“只有 returned-policy phase 导致 key 分裂”的 JSON identity 结论成立。

### 1.3 物理 trace 重算

`trace.npz` 共 23 个数组。沿 repetition 轴逐位比较：

- 唯一不同数组是身份元数据 `profile_repetition`，三列按设计分别记录 0、1、2；
- 其余 22 个数组全部逐位相同。

逐位相同的数组覆盖：

- `time_s`；
- 26 hinge 的 `all_hinge_position`、`all_hinge_velocity`；
- base 线速度、角速度和 COM 位置；
- closure residual；
- 受控关节位置/速度；
- commanded input、canonical/host torque；
- 动量、动能、limit event、termination/truncation。

因此调查报告第 30 行“三次完整物理 trace 一致”的表述成立，只需明确排除故意不同的 repetition 标签元数据。

reset sample 的直接 PhysX angular velocity 三次均为精确 `[0.0, 0.0, 0.0]`。rollout 末端三次均为：

`[-4.645830631488934e-05, 0.000135254769702442, -4.151953908149153e-05]`。

## 2. 原始 Actor 组件捕获

对象：`runs/diagnostic-20261008-p10-returned-policy-components-03`

- `diagnostic.stdout.txt`: `6AA1F4E8B32D19E2494A829D045622A84AB8217301C057BC9B34E80CFA486387`
- 组件捕获位于 stdout 第 25-28 行。
- 该 run 的 `result.json` 与 `trace.npz` 分别与 baseline 逐字节同哈希。

四次 helper 调用中：

- call 0/1 的 Actor bytes SHA 为 `925AF015C06790F5B78CD6D8E953C694A63671024639A22F35329C79EC7E011A`；
- call 2/3 的 Actor bytes SHA 为 `F3249C7FD740C8B80EB39E6FA4118F5AC43544F7AD7E119DDE57347146ECEFB9`；
- 四次 previous-action bytes SHA 都为 `9D908ECFB6B256DEF8B49A7C504E6C889C4B0E41FE6CE3E01863DD7B61A20AA0`，六个元素逐位全零；
- call 0 与 call 1 的 Actor 完全一致；
- call 2 与 call 3 的 Actor 完全一致；
- 两组之间只有 float32 indices `0,1,2` 不同，indices `3..24` 逐位一致。

差异值为：

`[-1.1614576578722335e-05, 3.38136924256105e-05, -1.0379884770372882e-05]`。

将 baseline 末端 angular velocity 按 float32 乘冻结 scale `0.25` 后，三个 float32 bit pattern 与上述值完全一致：

- scaled bits: `[3074612282, 940430125, 3073254724]`
- captured bits: `[3074612282, 940430125, 3073254724]`

根据 `isaac_worker.py:916-937` 的 returned-policy payload schema，使用原始 `<f4` Actor、全零 `<f4` previous action 和五份 Actor history 独立重建 payload，重算得到：

- call 0/1: `D01A74F826E72AC358ABEC0CD7A34F2A3E367AEC4E825E5ACC2D85588A36DD50`
- call 2/3: `3E46EDBC5F3F7F9DC924C62547F0B3DF99D15976FB6E5B30598401A61CD4B442`

它们与 baseline 三个 repeatability record 的 returned-policy hash 精确一致。调查报告关于 raw Actor 仅 `0:3` 变化、previous action 不变、后续 reset 读取上一 rollout 末端角速度乘 `0.25` 的结论成立。

## 3. 两个负对照与一个正对照

### 3.1 `_capture_state()` 负对照

对象：`runs/diagnostic-20261008-p10-post-forward-state-refresh-01`

- stdout marker 在第 25-27 行出现三次；
- `result.json`: `20933E6A51E95C2C799AA1737293A3A8294F82B8DAAAFCF50B39E8928E7AB905`；
- `trace.npz`: `FD42D4ADC980CB103D1E525BD82A48D63849BCA8EEAF08111BCF95A05D77B82A`。

两者都与 baseline 逐字节相同，因此该负对照没有修复 identity split。

### 3.2 `scene.update(0.0)` 负对照

对象：`runs/diagnostic-20261008-p10-scene-update-refresh-01`

- stdout marker 在第 25-27 行出现三次；
- `result.json`: `20933E6A51E95C2C799AA1737293A3A8294F82B8DAAAFCF50B39E8928E7AB905`；
- `trace.npz`: `FD42D4ADC980CB103D1E525BD82A48D63849BCA8EEAF08111BCF95A05D77B82A`。

两者都与 baseline 逐字节相同，因此该负对照没有修复 identity split。

### 3.3 `scene.update(physics_dt)` 正对照

对象：`runs/diagnostic-20261008-p10-scene-update-physics-dt-01`

- stdout marker 在第 25-27 行出现三次；
- `result.json`: `0A684EE176FF240C2E03DCCCD31AC9EF77DF6912E1B52AD25D9008489D28BF8F`；
- `trace.npz`: `FD42D4ADC980CB103D1E525BD82A48D63849BCA8EEAF08111BCF95A05D77B82A`。

三次 record 独立重算后均为：

- returned-policy hash: `B17F8A84A22C031DA1B9CF8C58A4C75C343B0D36F228BD4C01339DA570CEF601`
- repeatability key: `12CAAA613549800365F5C6FAB476CB844F7FDE3A07979A6B3900704FD14CAA56`

三次 pre-forward、post-forward、excitation 仍与 baseline 相同。正对照的 `trace.npz` 与 baseline 文件 SHA256 完全相同，且逐数组比较也无差异。因此该 intervention 改变了 reset returned observation identity，没有改变后续物理 trace。

## 4. 源码数据流核对

调查报告的数据流解释与当前源码一致：

1. 正式 `WheelLegFlatEnv._reset_idx()` 在 `env.py:882-884` 写 root pose/velocity 和 joint state，在 `env.py:911-918` 立即调用 `_read_state()` 并缓存为 `self._state`。
2. `_read_state()` 在 `env.py:653-690` 读取状态，其中 `env.py:669` 从 `robot.data.root_link_ang_vel_b` 取得 Actor angular velocity 来源。
3. `_get_observations()` 在 `env.py:761-769` 调用 `_current_state()`；`env.py:699-700` 在 `self._state` 已存在时直接返回缓存，不重新读取 articulation data。
4. Isaac Lab `write_root_com_velocity_to_sim()` 在 `articulation.py:513-525` 更新 `root_com_vel_w` 并写 PhysX，但没有使 `_root_link_vel_w` timestamp 失效。
5. `root_link_vel_w` 只在其 timestamp 小于 `_sim_timestamp` 时重新计算，见 `articulation_data.py:487-504`。
6. `ArticulationData.update(dt)` 在 `articulation_data.py:97-102` 通过加 `dt` 推进 `_sim_timestamp`。因此 `dt=0.0` 不产生新的缓存世代。
7. `DirectRLEnv.reset()` 在 `direct_rl_env.py:313-331` 执行 `_reset_idx()`、`scene.write_data_to_sim()`、`sim.forward()` 后直接返回 `_get_observations()`，中间没有正常 step 路径 `direct_rl_env.py:383-384` 的 `scene.update(dt=physics_dt)`。
8. debug direct capture 在 `isaac_debug_env.py:164-184` 直接读取 PhysX view，而不是复用任务 `_state`；其 reset 在 `isaac_debug_env.py:279-310` 分离 pre-forward、post-forward 和 returned-policy 三相，因此能够同时观察“直接物理速度为零”和“API returned Actor angular velocity 非零”。

这条路径与两个负对照、一个正对照的结果相互一致。

## 5. 正式 `WheelLegFlatEnv` 路径确认

对象：`runs/diagnostic-20261008-formal-reset-observation-01`

- `instrumentation.json`: `F7B58E36D2A02BE8306DCAE6ECB252C010215F2DBAD0731B0296C6506F1E2984`
- `trace.npz`: `89E7A7040004586051A40C2DE04CAB80F1ED81BF4DCB0265E4AD3EB3F4D79DF7`
- `diagnostic.stdout.txt`: `95ED6A7F142ED96972880F6A37DD3E9EA01A110AB1D61998F41916B21D2F79FC`

`instrumentation.json` 记录：

- `mode = "formal"`；
- `formal_config_hash_before == formal_config_hash_after == AD92A93B61E3436DB443081E2A521926B73944E13B017FEF4F656C320100A02`；
- 声明的 trace SHA 与实际文件 SHA 一致。

实际类选择由 `isaac_worker.py:3669-3670` 决定：formal 分支直接执行 `env = WheelLegFlatEnv(cfg)`；debug subclass 只用于其他 mode。

五次 reset 的 tick-0 Actor angular slice 为：

| repetition | Actor `0:3` |
|---:|---|
| 0 | `[0.0, 0.0, 0.0]` |
| 1 | `[0.00020298079471103847, 0.26674848794937134, 0.00031355838291347027]` |
| 2 | `[0.00041443054215051234, 0.2653515934944153, 0.00035527837462723255]` |
| 3 | `[0.00031034532003104687, 0.2669265866279602, 0.00034599131322465837]` |
| 4 | `[0.0003949041129089892, 0.2665711045265198, 0.0006611178396269679]` |

每一行都与该 capture 时刻由同一 formal class 路径读取的 cached `base_angular_velocity_control * 0.25` 逐位一致。`NOMINAL_EVALUATION_PROFILE_V1` 的 randomization disabled，正式 randomization 实现在 `tasks/direct/wheelleg_flat/randomization.py:322-324` 对 root velocity 返回六维全零。因此该 artifact 确实证明生产 `WheelLegFlatEnv` reset API 路径可在 nominal reset 后返回非零 angular-velocity observation，且不是 debug subclass 的 observer 注入。

限制：formal instrumentation 中 `base_angular_velocity_control` 也通过 `robot.data` cache 读取，不是 direct PhysX view，所以它证明的是 formal cache/returned-observation 一致性；formal 配置下 post-forward 直接物理零值没有在此 trace 中另设独立 direct-view channel。直接物理零值来自 baseline debug direct capture及其相同 nominal reset/物理 trace。

## 6. 冻结设计分类

冻结设计的相关规则：

- `§6.4`，第 400-413 行：pre-forward、post-forward、returned-policy 是不同且不可替代的 phase identity；returned-policy 是首次交给 Actor 的权威输入。
- `§6.4`，第 426 行：形成同一 repeatability envelope 必须同时匹配 configuration、三个 phase hash 与 excitation hash。
- `§7.2`，第 594-613 行：无法形成匹配 key 的重复包络时，依赖 probe 不得作材料级判定。
- `§7.3`，第 623-650 行：history/observation 硬门失败分类为 `SIM2SIM_ADAPTER_BUG`，并使 checkpoint 与物理 verdict 全部无效。
- `§8.1`，第 1136-1140 行：G01 不可用先使依赖 probe `INCONCLUSIVE`；G02 失败固定分类为 `SIM2SIM_ADAPTER_BUG`；只有前置门均允许时才进入物理和 checkpoint 分类。

本次证据同时满足：

1. 同一物理 reset/trace 的 `reset_returned_policy_hash` 不可重复，G01 对该 exact key 不可用；
2. 分裂发生在权威 Actor observation 的 angular-velocity slice，属于 observation/history hard-gate failure；
3. 不能用 post-forward hash 替换 returned-policy hash 来绕过门禁。

因此冻结分类应为：

`SIM2SIM_ADAPTER_BUG`

更窄的已支持子因可写为：

`isaac_reset_returned_observation_stale_root_angular_velocity_cache`

P10-P60 的物理归因和 C70 的 checkpoint amplifier 归因在该门关闭前全部无效。允许继续做不进入 verdict 的诊断实验，不允许形成正式 root-cause verdict。

## 7. 是否能升级为 MuJoCo 首次失稳的唯一根因

不能。

本批证据没有完成：

- 正式 reset 路径修复后的全新 G01/G02；
- 修复后同一 checkpoint 的有效 Isaac/MuJoCo 首帧及 20 tick 对齐；
- 修复后 P10-P60 的完整物理证据链；
- P60-D exact replay 与 C70 双 anchor amplifier 分析；
- 训练时 reset contamination 对已训练 checkpoint 的定量影响；
- 排除闭链、drive/contact、静态/动态惯量和整机耦合等其他候选。

正对照只证明 cache refresh 消除了本次 Isaac repeatability identity split，并且没有改变该 P10_A 物理 trace。它没有运行修复后的 MuJoCo sim2sim 闭环，也没有证明首次 MuJoCo 失稳随之消失。

因此调查报告第 172-174 行“不认证 sole cause、不生成有效 P10-P60/C70 verdict、不量化 checkpoint 训练影响”的限制正确。任何“这就是 MuJoCo 首次失稳唯一根因”的升级都属于无证据过推断。

## 8. 源码 SHA256

| 文件 | SHA256 |
|---|---|
| `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py` | `5399B29B92B54851EB0A5FA880EA9386C82F202A5C337F19ABE4EDF300C2969C` |
| `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/envs/direct_rl_env.py` | `3A7573303D83EC8C816D11CA6E8C8997C92F6765AE5D2E0D2A369E65D2CEC3C0` |
| `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/assets/articulation/articulation.py` | `FB179AACDD848906367D01105020E1D36412BD0F1DA7187BE09F0781FF589309` |
| `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/assets/articulation/articulation_data.py` | `D69495C66C409FB20823E4E467F9D90E63163B2050B77D2AC8E433221D1D208A` |
| `debug/sim2sim/isaac_debug_env.py` | `3CDB03D723BEB8D2024E7A4F512ADF8C51560C3F1C8209AA9B9D01E7725F56C2` |
| `debug/sim2sim/root_cause_suite/isaac_worker.py` | `BCF6A3530FAB09E6FFC3A6F1633B225BC3C8D2AC6BB5A39DA7294F8706D644B8` |

调查报告 SHA256：

`8C4C562AC6B2C64AF2384A1A9CF79906558A94B4FFCE1E56989B7FC76F51E4C8`

## 最终状态

- 结果复核：`APPROVED`
- `SIM2SIM_ADAPTER_BUG` 分类：批准
- P10-P60/C70 正式 verdict：`BLOCKED`
- “MuJoCo 首次失稳唯一根因”：`NOT PROVEN`
- 正式训练代码、USD、MuJoCo 参数、checkpoint、既有运行产物：未修改
