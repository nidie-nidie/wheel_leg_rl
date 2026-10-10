# Sim2Sim RootCauseSuite Core V1.16 独立代码复核 02

日期：2026-10-08  
工作区：`E:\wheel_leg_rl-main`  
套件根目录：`E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\root_cause_suite`  
冻结设计：`RootCauseSuiteCoreV1.16`  
冻结设计 SHA256：`363720317500D6F98A4BBA81CFC7F2E9B5EF5A774974418542A46F602C072FE6`  
审查方式：严格只读；除本报告外未修改代码、测试、设计、运行证据或正式工程文件。

## 独立结论

- P0：**4**
- P1：**4**
- P2：**2**
- 结论：**NOT APPROVED**
- 是否允许进入完整 Core V1 真实诊断运行：**不允许**

本地已有的 `79 passed, 1 skipped`、MuJoCo `10 passed` 以及真实 G03 成功证据均已独立复核，结果属实；但测试通过不等于冻结设计已经实现。当前 G01、G02 仍可在缺失关键实质检查的情况下通过，P30/P40/P50 的因果身份还不是冻结设计要求的完整编译后语义身份，P50 还存在可直接触发的法向门 fail-open。因此不满足“P0=0 且 P1=0”这一开跑条件。

## Findings

### P0-1：G01 统计的是声明覆盖，不是已执行并冻结的重复性门

**位置**

- `core_stage_worker.py:370-483`
- `analysis.py:60-91`
- 冻结设计：`2026-10-07-sim2sim-root-cause-suite-design.md:594-598,426-470`

**可复现逻辑**

1. `run_g01()` 实际只对两个引擎分别执行一次 `p60_c` nominal zero-action baseline；`core_stage_worker.py:371-389` 生成的是 profile 定义集合，不是各 profile 在各引擎中的实际重复执行记录。
2. `core_stage_worker.py:465-467` 的 `declared_coverage` 仅检查每个声明 profile 的计数是否大于等于 3，不能证明每个 verdict-bearing exact key 已经被真实运行、重复三次并冻结。
3. 后续阶段在 `analysis.py:60-91` 从本阶段自己的 trace 重新计算 envelope，并使用 `max(floor, 5 * envelope)`；它没有绑定 G01 预先冻结的 `threshold_snapshot.json`，也没有在 envelope 超过数值地板时把该 key 标记为 nondeterministic/unavailable。

**影响**

G01 可以在未执行完整 nominal/dynamic repeatability matrix 的情况下返回通过。P30/P40/P50/P60 还可以用阶段内重新放宽的阈值掩盖噪声，形成后验阈值和伪通过。此时根因分类不再满足冻结设计的“先冻结重复性地板，再做跨引擎比较”。

**建议**

真实执行并缓存每一个精确 repeatability key，保存输入身份、输出身份、重复样本、envelope 与可用性；在任何跨引擎阶段之前原子冻结 threshold snapshot。所有消费者必须按精确 key 读取快照；缺 key、身份不匹配或 envelope 高于允许地板时必须 fail-closed，而不是阶段内扩大阈值。补充“声明存在但未执行”“动态 key 缺失”“高噪声被扩大阈值吸收”的变异测试。

### P0-2：G02 不是冻结设计要求的双引擎 reset/history/adapter 实质门

**位置**

- `core_stage_worker.py:486-678`
- 冻结设计：`2026-10-07-sim2sim-root-cause-suite-design.md:623-629`

**可复现逻辑**

1. `core_stage_worker.py:497` 只调用 `runner.isaac(..., "one-tick")`，没有对应的 MuJoCo one-tick worker 结果。
2. `core_stage_worker.py:507-587` 使用同一份 Isaac observation 构造两个本地 `DebugPolicyAdapter`，比较的是两个本地适配器对同一输入的行为，不是 Isaac 与 MuJoCo 两条真实观测、history、CENet、actor、clipping 和 native target 链。
3. `core_stage_worker.py:589-604` 的 polarity 检查调用纯函数工具，没有向两个物理引擎施加六通道正负微脉冲，也没有比较实际首响应方向、幅值和时钟。
4. 当前输出没有冻结设计要求的三个 reset phase、25D/125D 跨引擎比较、CENet 输出、raw/clipped action、joint target、previous action 和 policy/control clock 的完整成对证据。

**影响**

reset 初态、history 填充、观测切片、previous-action 时序、动作缩放/裁剪、执行器极性或 policy clock 的真实 sim2sim 错误都可能在 G02 通过后继续存在，后续 P 阶段可能把接口错误误归因给物理或 checkpoint。

**建议**

实现冻结设计中的双引擎 one-tick trace：在相同 reset 身份下导出三个 reset phase、25D observation、125D history、CENet velocity/context、raw/clipped action、native joint target、previous action 和全部时钟；再对六通道分别执行正负微脉冲并读取两个引擎的真实首响应。任何字段缺失、不可比较或身份不一致均应阻止 G02 通过。

### P0-3：P30/P40/P50 的因果身份是粗粒度摘要，且会掩盖非法协变量

**位置**

- `causal_contract.py:53-172,248-358`
- `scenario_catalog.py:181-189`
- `mujoco_worker.py:226-236,1195-1205`
- `isaac_worker.py:925-935,1784-1812`
- `test_causal_contract.py:18-129`
- 冻结设计：`2026-10-07-sim2sim-root-cause-suite-design.md:361-424,1338-1409`

**可复现逻辑**

1. `causal_contract.py:53-121` 构造的 identity 主要由 source/invariant hash 和 mechanics/closure/actuation/contact/clock 等粗粒度对象组成，不是 composed config、编译后模型和运行时解析结果的完整递归规范化语义。
2. `scenario_catalog.py:181-189` 的允许差异只覆盖少量 coarse path，没有冻结设计要求的逐 hinge、逐 constraint、逐 contact pair/solver coefficient 精确路径。
3. `mujoco_worker.py:226-236` 在 P30 identity 中屏蔽所有 joint damping。独立变异证明：只修改无关的 `base_free.damping`，`_robot_invariant_identity(..., scenario="p30_open_direct")` 仍完全相同。
4. 同一实现对 P40 屏蔽所有 constraint active 状态，而不是仅屏蔽指定的闭链变换路径。
5. `mujoco_worker.py:1195-1205` 的 P50 invariant 没有完整绑定 contact pair 的 `condim/solref/solimp` 和几何接触语义；`isaac_worker.py:925-935,1784-1812` 也是手选摘要，未递归绑定完整 `SimulationCfg`、`PhysxCfg`、解析后的 physics scene、drive/material/contact 语义。
6. `causal_contract.py:124-172` 的 configuration identity 未包含冻结设计要求的 worker source hash。
7. 当前 validator 没有把 transform manifest 的逐路径差异与实际 compiled semantic diff 做一一对应，也没有证明两个引擎应用的是同一个抽象变换因子。
8. `test_causal_contract.py:18-129` 只覆盖粗粒度 synthetic gravity/reset/excitation 变异，没有设计要求的 MuJoCo option、PhysX、contact solver、逐约束和 transform-manifest 变异。

**影响**

P30/P40/P50 的 baseline/ablation pair 可以在无关甚至结论相关的隐藏物理量发生变化时仍通过身份门，导致 explanation ratio 不是单一因果变量的可信估计。这是根因结论的核心伪通过风险。

**建议**

为每个引擎输出完整、递归、规范化的 composed/compiled/runtime semantics；从冻结变换规范生成精确 allow-list artifact，并将其 hash、worker source hash、解析后物理语义全部纳入 configuration identity。validator 必须计算实际 semantic diff，逐路径与 transform manifest 对齐，并验证双引擎抽象因子投影一致。补齐冻结设计列出的强制变异测试。

### P0-4：P50 切向分类的法向前置门存在 fail-open

**位置**

- `core_stage_worker.py:1202-1245,1282-1295`
- `finalize.py:306-328`
- 冻结设计：`2026-10-07-sim2sim-root-cause-suite-design.md:532,1148-1149`

**可复现逻辑**

1. `core_stage_worker.py:1202-1245` 已经计算 contact/property/clearance/t=0/freefall 等 hard gate，这是正确方向。
2. 但 `core_stage_worker.py:1282-1295` 使用：`normal_gate = not impact["normal_primary_supported"] or not normal_impulse["material"]`。
3. 当 P50-A hard gate 失败时，`normal_primary_supported` 被置为 `False`；于是表达式第一项为 `True`，即使法向 impulse 明显 material mismatch，`normal_gate` 仍然通过。
4. 独立构造值 `normal_primary_supported=False`、`normal_impulse.material=True`，结果 `normal_gate=True`，复现了 fail-open。
5. `finalize.py:306-328` 会继续把 `tangential_primary_supported` 提升为 primary/contributor，未阻止无效法向证据进入根因分类。

**影响**

接触事件、t=0、属性或 free-fall hard gate 失败时，套件仍可能给出“纯切向接触根因”，直接违反冻结设计中“P50-A 正常或法向 impulse 在容差内才允许纯切向分类”的约束。

**建议**

把“法向证据有效”和“法向不构成 primary”拆成两个变量。切向前置条件应为：`(normal_evidence_valid and raw_normal_not_primary) OR valid_normal_impulse_within_tolerance`。接触事件或 hard gate 失败时必须返回 failed/inconclusive，禁止切向 primary。增加覆盖所有布尔组合的 truth-table 测试和当前反例回归测试。

### P1-1：resume 遇到已选 attempt 的哈希损坏时直接中止，没有按设计重跑

**位置**

- `orchestrator.py:696-713`
- `test_orchestrator.py:62-72`
- 冻结设计：`2026-10-07-sim2sim-root-cause-suite-design.md:1282-1289`

**可复现逻辑**

`orchestrator.py:696-713` 对 previous selected attempt 直接调用 `verify_attempt(previous)`，没有捕获 evidence-integrity failure 并创建下一 attempt。`test_orchestrator.py:62-72` 反而明确断言 tampered complete attempt 会抛错，固化了与冻结设计“running/failed/missing/hash mismatch 从新 attempt 重跑”相反的行为。

**影响**

中断恢复或证据损坏后不能自动保留旧 attempt 并重新执行该 stage；长流程可能无法 resume。更重要的是，“身份漂移应中止”和“证据损坏应重跑”两个状态没有按设计区分。

**建议**

仅在 run identity/definition identity 漂移时中止；selected attempt 的缺失、哈希损坏或不完整应保留原目录、分配新 attempt 并从头重跑。添加 missing file、artifact tamper、running marker、failed result 四类 resume 测试。

### P1-2：run write-boundary 只盘点缓存类路径，G03 disabled child 也没有保持基础写保护

**位置**

- `integrity.py:172-211`
- `guard_semantics_worker.py:355-365,398-432`
- `contracts.py:138-159`
- 冻结设计：`2026-10-07-sim2sim-root-cause-suite-design.md:137-147`

**可复现逻辑**

1. `controlled_project_snapshot()` 只记录 source roots 中已有 `__pycache__`/`.pytest_cache`、根 pytest cache 和 Kit logs/data/cache，未对套件外全部项目可控文件建立 before/after inventory。
2. 因此 unguarded/native child 若在 `source/`、`sim2sim/` 等目录创建普通文件，不会被该快照发现。
3. G03 的 enabled/disabled child 由 `guard_semantics_worker.py:398-432` 直接启动，而不是通过 `worker_bootstrap`。disabled child 在 `guard_semantics_worker.py:355-365` 完全不安装 `PythonWriteGuard`。
4. `contracts.py:138-159` 的 canonical serialization 在 scalar leaf 上会尝试导入 NumPy；该第三方导入发生在 disabled child 内，没有 Python write guard。禁用 FileHandler wrapper 的对照不应同时移除所有基础写保护。

**影响**

当前 G03 证据证明本次样本没有观察到差异，但不能完整证明所有 suite-external 写入都被阻止或被 before/after inventory 捕获。仍存在正式工程文件或缓存污染未被证据链发现的风险。

**建议**

所有 child 都经 stdlib-only bootstrap 启动；disabled 对照只关闭被评估的 FileHandler wrapper，保留 audit/write guard。before/after inventory 应覆盖套件外全部项目可控文件，或使用完整、显式、可审计的允许列表，并对普通新文件、修改文件和 native child 写入添加变异测试。

### P1-3：记录了实际 worker argv/env，但 verify 没有把实际命令与冻结定义做语义绑定

**位置**

- `orchestrator.py:306-403`
- `test_orchestrator.py:100-119`

**可复现逻辑**

`verify_attempt()` 检查 bootstrap module、return code、环境标志、guard、output root 等字段，但没有读取并重建 `commands/worker.json` 中实际 `argv`、`argv_hash`、cwd、interpreter、source hashes，再与 stage definition 中的冻结 args/source identity 逐项比较；也没有把 bootstrap 记录的 `argv/process_argv` 与预期命令做 exact equality。现有测试只断言这些字段被记录，没有篡改/错命令拒绝测试。

**影响**

证据包可以说明“有一份命令记录”，但独立 verifier 不能证明被选 attempt 确实由冻结定义中的命令执行。运行身份绑定仍不完整，影响 resume 与审计可信度。

**建议**

在 verify 中从冻结 stage definition 重建预期命令，逐项核对 interpreter、module/script、args、cwd、关键环境和 source hash；同时核对 bootstrap 与 command artifact 的双记录一致性。增加 argv 参数替换、顺序变化、解释器变化、source hash 变化和环境删除的 mutation tests。

### P1-4：P60 replay source seal 在 attempt 提交前写入 run manifest，失败恢复不具事务原子性

**位置**

- `core_stage_worker.py:1317-1324,1396-1454`
- `orchestrator.py:245-264`
- 冻结设计：`2026-10-07-sim2sim-root-cause-suite-design.md:1003-1040`

**可复现逻辑**

P60 在 source 生成后立即调用 `seal_replay_source()` 修改 run manifest，随后才执行 fresh replay 和 MuJoCo replay。若后续步骤失败，P60 attempt 仍是 `.incomplete`，但 run-level seal 已持久化。resume 会创建新 attempt，而不是从已封存 source 继续；若新 source 有任何 bit difference，会因“已有另一 identity”而中止。即使内容相同，run manifest 也曾在没有 completed/selected P60 attempt 的状态下引用一个中途副作用。

**影响**

“source 只生成一次并在 replay 前封存”的目标部分实现了，但失败/恢复路径不是事务性的，可能造成孤儿 seal、无法 resume 或 run manifest 与 selected attempt 暂时不一致。

**建议**

把 replay source 生成设计为独立可提交 substage/attempt：先完成 source artifact 与 hash，再原子地同时选中 source attempt 并写入 run seal，之后 B/C 只读该已选 source。或者在 P60 resume 时强制复用已封存 source artifact，而不是重新生成。当前 B/C reset identity 检查可保留。

### P2-1：PythonWriteGuard 对部分 `dir_fd` audit 参数的索引判断不完整

**位置**

- `python_write_guard.py:229-232`
- `test_python_write_guard.py`

**可复现逻辑**

`os.remove`/`os.rmdir` 的 CPython audit 参数通常是 `(path, dir_fd)`，但实现只在 `len(args) > 2` 时读取 `dir_fd`；`os.chmod` 的 `(path, mode, dir_fd)` 也只在 `len(args) > 3` 时读取。因此受支持平台上的相对 `dir_fd` 路径可能不会按真实目标解析。现有单元测试没有逐事件覆盖这些参数布局。

**影响**

本次 Windows capability manifest 显示相关 `dir_fd` 能力不可用，因此没有证明当前证据已被利用；但跨平台或未来运行时的 fail-closed 写保护不完整。

**建议**

按每个 CPython audit event 的正式参数签名解析，不复用模糊长度判断；为 remove/rmdir/chmod/chown/utime/rename/replace/mkdir/open 分别添加可用与不可用 capability 测试。

### P2-2：report 可重建，但最终报告和 evidence refs 尚未达到冻结设计的审计细度

**位置**

- `report.py:60-107`
- `finalize.py:187-214,556-563`
- `test_report.py:10-37`
- 冻结设计：`2026-10-07-sim2sim-root-cause-suite-design.md:1202-1261`

**可复现逻辑**

1. `report.py` 能从已验证 stage 重新生成 bundle，这是正确的；但生成内容主要是 JSON dump 和简短表格，没有完整呈现冻结设计要求的正证据链、逐候选排除理由、阈值敏感性和下一步允许修改的正式契约。
2. `finalize.py:556-563` 的 `evidence_refs` 只有 stage/probe/artifact SHA，缺少设计要求的具体文件、signal、time/window、threshold、observed value 等定位字段。
3. `test_report.py:10-37` 只验证确定性、一个 artifact hash 单元格和一句文本，没有覆盖删除报告后重建、报告内容篡改、引用文件缺失及逐引用可解析性。

**影响**

机器完整性链可以重建报告，但人工审计和独立复算仍需要回到多个原始 JSON 搜索字段；报告尚不能完全承担冻结设计定义的最终审计入口。

**建议**

让每条 evidence ref 包含并校验 stage、attempt、相对文件、JSON/CSV signal path、time/window、threshold、value 和 artifact hash；报告按设计顺序输出正证据、排除项、阈值敏感性和 next-contract。添加删除/重建、篡改、缺失引用和引用字段错配测试。

## 第一次代码复核逐项处置

| Review-01 项目 | 状态 | 独立复核依据 |
|---|---|---|
| P0：G00-G03 实质性门 | **PARTIAL** | G00 已绑定核心 stage 定义；G03 真实证据和 bitwise/exact 比较成立。但 G01 仍是声明覆盖，G02 仍不是双引擎实质门，见 P0-1、P0-2。 |
| P0：P30/P40/P50 因果对身份 | **PARTIAL** | `causal_contract.py` 已有运行时 validator 并被阶段调用，但输入身份是粗粒度摘要，mask 过宽，缺完整 compiled semantics 和强制 mutation coverage，见 P0-3。 |
| P0：P50 t=0/contact/property gate | **PARTIAL** | t=0、contact、property、clearance、free-fall gate 已实现；切向前置逻辑仍可在 hard gate 失败时 fail-open，见 P0-4。 |
| P0：manifest/verify/report/finalizer 防篡改链 | **CLOSED** | `orchestrator.py:819-961` 验证冻结 core stage 集、selected attempt、artifact 与 replay seal；`orchestrator.py:974-980` 在 report 重建前验证；`finalize.py:620-658` 生成并绑定 final state。已有 tamper tests 覆盖 stage/artifact/final bundle。实际 argv 的语义绑定作为单独 P1 保留。 |
| P1：run-id/write boundary 与导入顺序 | **PARTIAL** | run-id 和主 worker bootstrap 顺序已改进；但 suite-external inventory 不完整，G03 disabled child 失去基础 guard，见 P1-2。 |
| P1：resume 全身份绑定 | **PARTIAL** | run identity、definition identity、selected attempt 已绑定；但 hash-invalid selected attempt 直接中止，实际 argv 也未与冻结 definition 精确比对，见 P1-1、P1-3。 |
| P1：P20 测量/测量失败/golden 分离 | **CLOSED** | `core_stage_worker.py:1012-1043` 分开输出 measurement、audit 和 golden；`finalize.py:296,309-315` 不再把 golden mismatch 当作 plant 根因。 |
| P1：P60 replay 原子封存与 B/C reset 身份 | **PARTIAL** | `core_stage_worker.py:1396-1454` 已检查 B/C reset identity，replay source 也有 run-level seal；但 seal 在 P60 attempt 提交前写入，失败恢复存在孤儿 seal，见 P1-4。 |
| P1：C70 plant-material 前置与证据明细 | **CLOSED** | `core_stage_worker.py:1488-1522` 要求 P60 material plant failure；`analysis.py:1021-1139` 生成双 anchor、group、sign、reduction；`finalize.py:495-542` 输出逐组合证据。 |
| P2：report 重建能力 | **CLOSED** | `orchestrator.py:974-980` 先验证已选 stages，再调用 finalizer覆写重建报告；最终 state/hash 可再次验证。报告审计细度不足作为新 P2-2 单列。 |
| P2：CLI exit 3/5 | **CLOSED** | `cli.py:44-49` 映射 `IdentityDriftError -> 3`、`EvidenceIntegrityError -> 5`；独立 monkeypatch 调用实测返回 3 和 5。 |

## 真实 G03 证据独立核验

证据文件：`runs/integration-g03-exact-20261008-213922/worker/result.json`  
SHA256：`CB650AF43D612C9C4EBBA6CFE8B077BE779F8996E25CABF7CFCFF60563D4B84F`

独立检查结果：

- `passed=true`，`invalid_evidence_pipeline=false`，contact evidence 可用。
- 六个 bootstrap gate 均通过，且每个记录的 `evidence_sha256` 与对应 bootstrap 文件的实算 SHA256 相同。
- Isaac formal/debug/contact/system_observer：所有比较字段 `bitwise_equal=true`，连续字段最大绝对误差为 0，离散字段全部相等；formal config 和 RNG 状态未变化。
- MuJoCo formal/collector：identity 相同，全部 `exact_fields=true`，最大误差为 0。
- guard semantics：通过，FileHandler 语义相同。

因此，本次真实 G03 证据本身可信；本报告的阻塞项不是否定 G03 样本，而是指出 G01/G02、因果身份和 P50 分类逻辑仍未满足冻结设计。

## 实际执行的验证

### 身份与证据哈希

- 冻结设计 SHA256：`363720317500D6F98A4BBA81CFC7F2E9B5EF5A774974418542A46F602C072FE6`，与声明一致。
- Review-01 SHA256：`E1F733629A23806B49C27BB084D70B75F74C558664E3FE7DC01D05666094F150`。
- remediation handoff SHA256：`C916A48BD24F1DEB24A9CF5DB06FEA97A1C2B5B7B0EEEAA5C9C15E987E27924C`。
- 最新真实 G03 result SHA256：`CB650AF43D612C9C4EBBA6CFE8B077BE779F8996E25CABF7CFCFF60563D4B84F`。

### 项目侧测试

```powershell
$env:PYTHONNOUSERSITE=1
$env:PYTHONDONTWRITEBYTECODE=1
$env:PYTHONPYCACHEPREFIX='<suite-unique-temp>\pycache'
E:\wheel_leg_rl-main\wheelleg_dreamwaq\.venv\Scripts\python.exe -B -m pytest `
  debug\sim2sim\root_cause_suite\tests `
  --ignore=debug\sim2sim\root_cause_suite\tests\test_mujoco_worker.py `
  -q -o cache_dir='<suite-unique-temp>\pytest-cache' `
  --basetemp='<suite-unique-temp>\pytest-tmp'
```

结果：`79 passed, 1 skipped in 11.09s`。skip 为 `test_workspace_guard.py:32` 的本机 symlink creation unavailable。

### MuJoCo 侧测试

```powershell
$env:PYTHONNOUSERSITE=1
$env:PYTHONDONTWRITEBYTECODE=1
$env:PYTHONPYCACHEPREFIX='<os-temp>'
$env:PYTHONPATH='E:\wheel_leg_rl-main\wheelleg_dreamwaq;E:\wheel_leg_rl-main\wheelleg_dreamwaq\sim2sim\mujoco'
E:\wheel_leg_rl-main\wheelleg_dreamwaq\sim2sim\mujoco\.venv\Scripts\python.exe -B -m pytest `
  debug\sim2sim\root_cause_suite\tests\test_mujoco_worker.py `
  -q -o cache_dir='<os-temp>' --basetemp='<os-temp>'
```

结果：`10 passed in 1.34s`。

### 定向变异与 CLI 验证

- P30 隐藏协变量变异：把编译模型中非目标 `base_free.damping` 从 0 改为 99，P30 invariant identity 仍相等：`True`。
- P50 fail-open 反例：`normal_primary_supported=False` 且 `normal_impulse.material=True` 时，当前 `normal_gate=True`。
- CLI 独立调用：`IdentityDriftError -> 3`，`EvidenceIntegrityError -> 5`。

### 污染检查

- 未启动完整 Core V1。
- 未启动训练。
- 未修改正式训练、USD、MuJoCo 参数、测试、设计或现有运行证据。
- 测试使用的唯一 suite 内临时目录已在确认绝对路径后删除；终检时不存在 `_review02-temp-*` 残留。

## 最终裁决

**NOT APPROVED。**

当前不得进入完整 Core V1 真实诊断运行。最低解锁条件是：修复并验证 P0-1 至 P0-4、P1-1 至 P1-4，使 P0=0、P1=0；随后重新执行项目侧与 MuJoCo 侧测试、G00-G03 真实 bootstrap，并由独立复核确认因果身份变异、P50 truth table、resume 损坏恢复和 P60 失败恢复均 fail-closed。P2 可以在不影响结论可信度的前提下单独收尾，但不得用 P2 标签降级当前 P0/P1。
