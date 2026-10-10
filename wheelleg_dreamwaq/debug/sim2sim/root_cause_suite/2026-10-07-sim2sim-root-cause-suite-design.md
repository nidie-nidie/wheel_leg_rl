# WheelLeg Isaac Sim / MuJoCo 全链路根因诊断套件设计

**日期：** 2026-10-08（V1.17 修订）  
**设计版本：** RootCauseSuiteCoreV1.17  
**状态：** Core V1 范围保持冻结；V1.17 根据真实 MuJoCo compiled pair smoke 修正 P30 exact-diff 投影中的阻塞矛盾，等待定向独立复核后继续实施，尚未运行完整新诊断  
**代码与受控产物边界：** `wheelleg_dreamwaq/debug/sim2sim/root_cause_suite/`  
**正式工程边界：** `source/`、正式 USD、正式 MJCF、训练脚本、checkpoint、奖励、动作、命令、随机化和 termination 全部只读

本文件同时保留后续完整诊断矩阵，但范围标签具有强制含义：

- `[CORE]`：本轮 Core V1 必须实现、测试和执行。
- `[EXTENDED]`：仅保留为 Extended V2 设计，不得在本轮顺手实现。
- 未列入 `[CORE]` 的新 probe、传感器、参数点或模型生成器不得在实现阶段临时加入。
- 实现中允许新增的未逐文件列名代码，只能是完成既定契约所必需的 CLI、schema、日志、断言、测试夹具和兼容处理；不得扩展诊断能力。
- 若实现时发现阻塞矛盾，必须先给出准确文件与行号证据并修订本设计，不能静默扩大范围。

## 1. 设计目标

本套件要把当前“DreamWaQ 在 Isaac 中可训练、在 MuJoCo 中迅速失败”的现象拆成可验证的因果链，并用一次可恢复的执行流程回答下面的问题：

1. 诊断代码本身是否可信，是否改变了被观察系统。
2. 两侧 reset、五帧 history、观测、Actor、动作、符号、目标和时钟是否一致。
3. 不使用 checkpoint 输出时，两个物理系统是否已经分歧。
4. 分歧主要来自质量/惯量、执行器/积分、闭链约束，还是轮地接触/摩擦。
5. 在可观测的底层物理门全部通过后，剩余失败是否才可归因于 checkpoint 鲁棒性。
6. 如果证据不足以唯一归因，具体还缺哪一项观测或实验，而不是用猜测补全结论。

最终输出不是一句“看起来像接触问题”，而是一个带证据引用、排除链、阈值、置信度和剩余不确定性的机器可读 verdict。

Core V1 不承诺一定得到唯一根因。只要证据链完整，`INCONCLUSIVE`、`MULTIPLE_PHYSICS_MISMATCHES` 或“本轮没有找到可认证根因”都是合格结果；禁止为了给出确定答案而越过未执行的 Extended V2 门。

本设计不承诺物理求解器内部不可见量一定能够被唯一反演。它承诺的是：

- 能直接证明的结论才进入主 verdict。
- 只能证明“是贡献因素”时，不写成“唯一根因”。
- 多个独立门同时失败时，输出 `MULTIPLE_PHYSICS_MISMATCHES`。
- 数据无效、观测缺失或候选仍耦合时，输出 `INCONCLUSIVE`。
- 诊断链自身失效时，输出 `INVALID_EVIDENCE_PIPELINE`，不继续伪造物理结论。

## 2. 当前权威身份与历史证据

### 2.1 主架构身份

当前工作区中的权威架构是：

```text
Architecture v0.21
SHA256 5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE
```

早期独立 AI 复核针对的是 `Architecture v0.18` 和哈希
`5BC72668F2A233D66D75C069950EC6571C0416580EEF223D8F9DA8A9E5C81591`。
该复核可以作为历史设计意见，但不能覆盖当前 `v0.21` 的文件身份与已实现状态。

### 2.2 本设计冻结读取的正式文件

| 正式输入 | 当前 SHA256 |
|---|---|
| `sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml` | `691C607271ED85EFFA388237E00A7B8F8F41CB66F6BA9C49EA64ABD186D803D1` |
| `sim2sim/mujoco/model_manifest.json` | `C3DB4FE2797F6AC3628F0A9CB5E0B166FFD0CE802776A335922C474A045F7AE0` |
| `sim2sim/mujoco/wheelleg_mujoco/runner.py` | `B0C973A5335856E3991ECCF6014BA9B5CF65E638A8BD4701FC07EED1A446EDBB` |
| `sim2sim/mujoco/wheelleg_mujoco/control.py` | `83080BC57A0399B11610CE5E06B4F607ADA1C0FB995344E23E531658EBBEFD62` |
| `sim2sim/mujoco/wheelleg_mujoco/observation.py` | `DD7F972D77FCCA25378BF403F61A67755C8AB4BA28738C97ABF2BBDD733970BD` |
| `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py` | `5399B29B92B54851EB0A5FA880EA9386C82F202A5C337F19ABE4EDF300C2969C` |
| `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env_cfg.py` | `AB7BAA693A73EFBB0F933A605EF02F94C0BDBCF95E468D3CF400E590F59101F5` |
| `artifacts/phase1_v4/asset-bundle-v2-manifest.json` | `CC4BF6006F103E9520E522BD8636BCDDAF69B2F121BFA73E65D984A9CFAA903B` |
| `artifacts/phase1_v4/mujoco-usd-data.json` | `411B3913D03F4523132177E42B416AFA736A026E7ADB567A81AD28ECB75D53C5` |
| `artifacts/phase1_v4/usd-anchor-audit.json` | `6231C16A830401B46516880821D3E8C4225BF8D2963367619D0F7FBC9D69BB5F` |
| `artifacts/phase2_dreamwaq/training-suite-20261007-185711/run-01-evaluation/isaac/evaluation-reset-cache.pt` | `008EE5D50E1C622D4AC0408E3C5E1C0B8CFCA618ABD3163E221A39D203B3DECE` |
| `artifacts/phase2_dreamwaq/training-suite-20261007-185711/run-01-evaluation/isaac/dreamwaq-run01/summary.json` | `3F8FA420604B65A514CB767CF770DA6F00E48610C8832B5487AA3A02AFCDB9BA` |
| `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/app/app_launcher.py` | `7AB2C742CFA64E2C4E2EA4601D9441E5E8C9BB8F556A121B0D10C337F9B6093E` |
| `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/sim/simulation_cfg.py` | `D8035DF355C576AD7C876789BBA54BE4EBB361F5C16786A4EC8578143EEBC118` |
| `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/envs/direct_rl_env.py` | `3A7573303D83EC8C816D11CA6E8C8997C92F6765AE5D2E0D2A369E65D2CEC3C0` |
| `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/sim/simulation_context.py` | `D95F2EAA138B4FFF845AC72439CAA2ECA5DEAB4F0D5609E06F163D20EEF4DF93` |
| `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/utils/logger.py` | `1C800C9876973CED42850AF89FBA68676A920EDA27CDE94E254D454DB32AD63E` |
| `.venv/Lib/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py` | `B0C233FD227C57ED38EAFCA2EF728CED31BFD53F444A98682714856CD08716C9` |
| `.venv/Lib/site-packages/isaacsim/kit/kernel/config/kit-core.json` | `A57318200F25E9AB84710C6840398B42B1A4D61AB65CB8561D254D6B3ADB2224` |

套件必须在执行前后重新计算这些哈希。任一正式文件发生变化，当前 run 立即失效。

### 2.3 固定策略输入

以下路径均相对 `wheelleg_dreamwaq/` 项目根目录，路径和内容哈希共同构成策略身份：

| 策略 | Actor 相对路径 | Actor SHA256 | policy manifest 相对路径 | policy manifest 文件 SHA256 | 用途 |
|---|---|---|---|---|---|
| DreamWaQ run-01 | `artifacts/phase2_dreamwaq/training-suite-20261007-185711/run-01-evaluation/export/actor.ts` | `4C2C64C2D9F9F88532FCEFF4AE45AA049F970646761D2D0ACE0A29678621D6C7` | `artifacts/phase2_dreamwaq/training-suite-20261007-185711/run-01-evaluation/export/policy_manifest.json` | `322232B78674217579E491FF9B88E1E688DB899A2C5064830F0D3149B0E992C3` | 当前失败对象 |
| Phase 1R run-03 | `artifacts/phase1_randomized_v3/training-suite-20261007-062147/exports/run-03/actor.ts` | `BFA655DF083EC8634DE18E4966E85A64219752F4E2875D3B2548BC7AA0E70AC6` | `artifacts/phase1_randomized_v3/training-suite-20261007-062147/exports/run-03/policy_manifest.json` | `53866E1078CFA2E2E337578ECC7DEE1A0E804B2D0D2892C90BA8FD7D6D9ADE40` | 已知较强但未完全通过的基线 |

checkpoint 和 Actor 均为只读。套件不得改写、重新导出或就地替换这些文件。
`artifacts/phase1_v4/` 只提供 §2.2 冻结的 AssetBundle/source-audit 物理证据，不是 Phase 1R run-03 的策略来源；实现不得由该目录推断或搜索策略文件。

### 2.4 已经成立、无需重复争论的事实

现有证据已经确认：

- reset 当前帧与五帧复制 history 的最大误差约 `4.6e-7`。
- 初始相同动作映射后的目标最大误差约 `2.98e-7`。
- 初始 PD 参考最大误差约 `5.53e-5 Nm`。
- CPU 上同一个 float32 输入经过同一个 TorchScript Actor 的输出恒等门已通过。
- 动作顺序、左右轮符号、主要反馈极性没有发现反转。
- DreamWaQ 在 MuJoCo 正式八场景为 `0/8`；同步到 `5 ms x 4` 仍为 `0/8`。
- Phase 1R run-03 在正式 MuJoCo 为 `5/8`，同步到 `5 ms x 4` 反而降为 `0/8`。
- 零动作和冻结 Isaac 动作回放仍会分歧。
- 首个材料级状态差异出现在第一个物理区间内，早于第二次策略推理。
- `+/-4 Nm` pitch 力矩响应在两个引擎中的方向一致。
- 将 MuJoCo 轮碰撞从球体临时替换为 mesh 只带来有限改善，不能单独解释全部分歧。

因此新套件不再从“Actor 是否算错”开始猜测，而是先把这些结论重新纳入自动门禁，再重点分解物理层。

## 3. 严格范围

### 3.1 允许的操作

- 读取正式源码、USD、MJCF、manifest、Actor、checkpoint 和已有 debug 产物。
- 在 `debug/sim2sim/root_cause_suite/` 下新增诊断代码、测试、设计文档、运行目录和模型副本。
- 由正式 USD/MJCF 生成带明确 debug 身份的新副本，并只在副本上关闭地面、重力、闭链、drive、碰撞或替换碰撞几何。
- 启动多个独立 Isaac 或 MuJoCo 子进程；同一时间最多运行一个 Isaac Kit 进程。
- 对 debug 副本施加外力、外力矩、直接关节 effort、目标脉冲和规定的初态扰动。
- 读取现有 debug collector 的纯函数和 schema；必要时把可复用逻辑复制到套件内部，避免修改旧诊断代码。

### 3.2 禁止的操作

- 不训练，不继续第二至第四次 DreamWaQ 正式训练。
- 不修改 `source/` 中的训练环境、DreamWaQ、PPO、随机化或 runner。
- 不修改正式 USD、正式 MJCF、正式 model manifest 或正式 MuJoCo runtime。
- 不修改奖励、动作缩放、命令、termination、随机化、历史长度、CENet 或 Actor 权重。
- 不为了让某个 checkpoint 站住而搜索 Kp、Kd、摩擦、solver、质量或惯量。
- 不把 debug 模型的结果作为正式八场景排名结果。
- 不把 Isaac host-side implicit-drive torque estimate 与 MuJoCo `data.ctrl` 当作同义实测力矩。
- 不把单次可视化观感作为 verdict 证据。

### 3.3 写入边界

所有项目可控的新文件必须位于：

```text
wheelleg_dreamwaq/debug/sim2sim/root_cause_suite/
```

Isaac Sim/Kit 的 logs、data、cache、user config 和 crash dump 在本机均可通过 `--portable-root`、Kit settings 和 pre-instantiation launch config 重定向或关闭，因此不属于可豁免写入。父进程必须先创建当前 run 的绝对目录 `<run>/runtime_cache/kit/`，并在启动 worker 时把该路径作为 argv 传入；worker 必须先用标准库解析并校验该路径，再允许导入 `isaaclab.app`。pre-launch 还必须计算并核验 §2.2 的 `kit-core.json` SHA256，解析 JSON 并断言 `/rtx-transient/resourcemanager/localTextureCachePath` 键存在且冻结默认值精确为 `${omni_global_cache}/texturecache`；文件/key/value 任一漂移时 G00 失败。`AppLauncher` 的 `kit_args` 必须包含独立 argv token `--portable-root <absolute-kit-root>`，以及 `--/crashreporter/enabled=false`、`--/crashreporter/dumpDir=<absolute-kit-root>/data`、`--/log/file=<absolute-kit-root>/logs/kit.log`、`--/app/userConfigPath=<absolute-kit-root>/data/user.config.json` 和 `--/rtx-transient/resourcemanager/localTextureCachePath=<absolute-kit-root>/cache/texturecache`。最后一项不能依赖 `${omni_global_cache}` 的默认解析，因为本机 portable root 不保证该 token 离开用户目录。

仅传 `--/crashreporter/enabled=false` 不能作为“插件未加载”的证据，因为本机 `SimulationApp.__init__` 在解析 Kit settings 前会依据 `launch_config["enable_crashreporter"]` 决定是否把 `carb.crashreporter-*` 加入 plugin wildcard。每个 worker 必须使用 suite-local `RootCauseSuiteAppLauncher(AppLauncher)`：只覆盖 `_create_app()`，先复制 `self._sim_app_config`，强制写入并断言 `enable_crashreporter is False`，再调用 `super()._create_app()`；不得 monkey-patch 上游类或修改上游文件。只有 §2.2 冻结的 `AppLauncher` 与 `SimulationApp` 源码哈希均匹配，且 AST/source contract 仍证明 crash reporter wildcard 只在该布尔值为真时加入，才允许使用这个 private-method seam。`kit_args` 中的 crash setting 和 dump path 继续保留为 defense-in-depth。启动后 worker 必须读取 resolved settings/tokens，证明 logs/data/general cache/config/dump 全部位于当前 run、crash reporter setting 为 false，并通过 `carb.get_framework().get_plugins()` 断言已加载 plugin 的 `impl.name` 中不存在 `carb.crashreporter-*`。RTX texture cache 采用更强的精确等式：`canonical(resolved_setting) == canonical(requested_setting) == canonical(<run>/runtime_cache/kit/cache/texturecache)`；仅“仍在 run 内”但落到 sibling 目录也不合格。缺少任一 pre-load 或 post-start 证明、texture cache setting 缺失、三方不精确相等、路径不绝对、解析后逃出当前 run，或 crash reporter 已加载时立即失败，不创建场景。GPU driver 自身不可控 cache 仅可作为明确外部白名单记录，不能用来豁免任何 Kit resolved setting。

Python 与 pytest 的项目可控缓存也属于本套件写入：唯一入口 `run_suite.ps1` 必须在启动首个 Python 进程前设置 `PYTHONNOUSERSITE=1`、`PYTHONDONTWRITEBYTECODE=1` 和指向当前 run `runtime_cache/pycache/` 的 `PYTHONPYCACHEPREFIX`，父进程与全部 worker 的解释器 argv 都必须在模块名之前显式包含 `-B`；只在 Python 代码中后设环境变量不合格。pytest 必须显式使用 `-o cache_dir=<run>/runtime_cache/pytest-cache --basetemp=<run>/runtime_cache/pytest-tmp`。运行前后记录 suite root 之外的项目文件清单；允许既有 `__pycache__` 保持不变，但出现新的或内容变化的项目可控 cache/file 时，G00 使 run 失效。项目 `.venv/Lib/site-packages/isaacsim/kit/{logs,data,cache}` 的既有完整目录树、文件集合、大小和 SHA256 必须进入 before/after 不变检查；任何变化都是正式工程写入。操作系统、GPU driver 和用户级不可控 cache 只能在明确列出的外部路径白名单中记录，不能覆盖项目目录。

Isaac Lab Python logger 也是项目可控写入。每个 Python 进程必须在任何第三方 import、`logging.FileHandler` 构造或场景创建之前，以标准库入口安装 suite-local `PythonWriteGuard`。守卫必须冻结并记录以下 CPython audit 事件：`open`（同时覆盖 builtins `open`、`os.open` 与 `os.fdopen`，依据 mode 及 `O_WRONLY|O_RDWR|O_CREAT|O_TRUNC|O_APPEND|O_EXCL|getattr(os, "O_TMPFILE", 0)|getattr(os, "O_TEMPORARY", 0)` mutation/write flag mask 判定；Windows `O_TEMPORARY` 必须即使没有其他 write flag 也按 delete-on-close 文件系统变更处理）、`os.mkdir`、`os.remove`、`os.rmdir`、`os.rename`（包含 `os.replace`）、`os.link`、`os.symlink`、`os.truncate`、`os.chmod`、`os.chown` 和 `os.utime`。安装时必须生成 capability manifest，逐项记录 flag 常量值、audit API 是否在当前平台可调用，以及每个 API 是否出现在 `os.supports_dir_fd`；不存在的 `O_TMPFILE`、`O_TEMPORARY`、`os.chown`、`dir_fd` 支持或其他平台可选能力记为 `unavailable_on_platform`，不得以整数 `0`、空集合或静默 skip 伪装成已测试或已覆盖。`open` 事件的 path 若为整数 fd，守卫必须在事件发生时解析其实际对象：Windows 使用标准库 `msvcrt.get_osfhandle` 加 `ctypes` 调用 `GetFinalPathNameByHandleW`；支持 `/proc/self/fd` 的平台读取该链接；标准输入/输出/错误 fd 和安装时显式登记的匿名 pipe 可以按 capability manifest 白名单，其余无法解析或无法证明位于 run 内的可写 fd 必须 fail-closed。所有实际可用事件涉及的 source/destination path 都必须解析整数 fd 或受支持的 `dir_fd` 后做 canonical allow-list；只允许当前 suite run 和操作系统空设备。当前 API 不支持 `dir_fd` 时不得构造伪正例；API 支持但组合无法可靠解析时必须 fail-closed。调用 `sys.addaudithook` 后必须立刻发送带随机 nonce 的 suite-local `sys.audit("wheelleg.root_cause_suite.guard_probe", nonce)`，只有新 hook 在自身 ledger 中精确记录该 nonce 才能标记 `installed=true` 并继续第三方 import；若已有 hook 阻止安装、probe 未到达或到达次数不是 1，进程立即失败。该事件集合定义“Python 层可审计文件系统变更”，不宣称覆盖 Kit/C++ native 或 GPU driver 写入；后两者继续由 portable-root、resolved-setting、before/after inventory 和明确外部白名单负责。同时对标准库 `logging.FileHandler.__init__` 安装进程局部、可恢复的登记包装器。包装器只能在调用原始构造器前记录原始参数并校验 canonical filename，不得改变 path/mode/encoding/delay，不得吞掉异常，合法路径必须恰好调用原始构造器一次；退出时在 `finally` 中恢复原方法。任何 suite 外 FileHandler，包括 `delay=True`，都必须在原始构造器和文件打开之前拒绝。audit hook 不可移除，因此持续覆盖包装器恢复后的 Python 层事件。G03 必须用 suite 内文件证明启用/禁用登记包装器时 open/write/close 的字节结果和异常语义一致，并分别验证上述 mutation 事件在 suite 内允许、suite 外拒绝。

每个 Isaac worker 安装守卫后、导入 Isaac Lab 前先保存 root 与 named logger 的 `pre_handlers`；任何既有 suite 外 `FileHandler` 立即失败。构造 `SimulationContext`/环境之前 clone debug env config，设置 `cfg.sim.save_logs_to_file=True` 和 absolute `cfg.sim.log_dir=<run>/runtime_cache/isaaclab/logs/`，并证明没有修改正式 `WHEELLEG_CFG`。pre-construction 必须核验 §2.2 冻结的 `direct_rl_env.py`、`simulation_context.py` 与 `utils/logger.py` hash/source/import contract：`DirectRLEnv` 必须从冻结的 `isaaclab.sim` origin 绑定同一 `SimulationContext` 对象，先保存 `self.cfg=cfg`，在 `SimulationContext.instance() is None` 时精确调用 `SimulationContext(self.cfg.sim)`，已有 context 时抛错；worker 在 clone config 后及环境构造前也必须显式断言 `SimulationContext.instance() is None`。`SimulationContext` 必须把 `logging_level/save_logs_to_file/log_dir` 原样传给 `configure_logging`，且非空 `log_dir` 不回退到 `%TEMP%/isaaclab/logs`。构造后遍历 root logger 与 `logging.Logger.manager.loggerDict` 中全部 `logging.FileHandler`，要求至少存在一个 Isaac Lab handler，且每个 handler 的 canonical `baseFilename` 父目录都精确等于 canonical requested/expected log directory。stage manifest 保存 requested config、expected directory、`pre_handlers`、`created_lifetime_handlers`、`post_handlers`、完整 Python audit event ledger 和 exact-parent equality；`None`、False、相对路径、run 内错误 sibling、suite 外路径、无 handler、预建 context、错误 import origin、任一 handler 逃逸或任一受审计 Python 文件系统变更逃逸均在场景执行前 fail-closed。不得通过把 `save_logs_to_file=False` 来绕过该证据门。

## 4. 顶层方法选择

本设计采用“Core V1 分层矩阵 + 因果消融”的方法，而不是只跑一个参数 sweep。完整矩阵保留为 Extended V2，但不进入本轮代码范围。

### 4.1 未采用的方法

**只做大范围参数搜索：** 可能找到一个让 MuJoCo 站住的参数组合，但无法区分是在修复错误、补偿另一个错误，还是过拟合单个 checkpoint。

**只比较闭环轨迹：** 策略会放大微小状态差异，无法判断差异由策略产生还是由 plant 先产生。

**只做单一候选实验：** 接触、闭链和 drive 在完整机器人上强耦合，单独把某个参数改好并不能排除其他候选。

### 4.2 采用的方法

1. 先验证证据链和 adapter。
2. 再移除 checkpoint，证明 plant 是否独立分歧。
3. 通过“无地面、无重力、直接 effort、关闭闭链、共同碰撞几何”等消融逐层去掉耦合项。
4. 使用正负对称激励和 baseline subtraction，消除 reset 偏置与自然下落。
5. 使用差分中的差分比较某个组件在两个引擎中的增量效应。
6. 最后才回到 checkpoint，判断它是根因、放大器，还是只是受害者。

### 4.3 Core V1 与 Extended V2 的冻结边界

| 阶段 | Core V1 本轮执行 | Extended V2 保留但本轮不实现 |
|---|---|---|
| G00-G03 | 身份、重复性、adapter、插桩中立性全部执行 | 无 |
| P10 | 零重力静止、自由落体、闭链 on/off 关键对照 | 更长时域与额外初态扰动 |
| P20 | 编译后 per-body mass/COM/inertia 静态审计；动量读回解析 golden | 六轴动态 wrench、有效惯量辨识 |
| P30 | 开链/闭链 direct-effort 与 formal-target 全部核心对照 | 额外幅值和 cadence 矩阵 |
| P40 | 零输入松弛、闭链 on/off 对称 effort | compliant constraint 多档柔度实验 |
| P50 | 共同球体法向落体、共同球体切向滑移 | common mesh、完整 rolling/spin-up 矩阵 |
| P60 | 整机零动作、共享第一动作、精确开环 replay | 整机刚性快照、common-geometry 三组合矩阵 |
| C70 | 只判断 checkpoint 是否放大已存在 plant error | checkpoint-only 主因认证与完整八场景重评 |

Core V1 最多推进到 P60 的核心对照，并在 C70 给出 amplifier 分析。它不能用“未执行的 Extended 探针通过”作为排除证据，也不能输出 checkpoint 是唯一根因。

## 5. 软件架构

### 5.1 单一入口

唯一受支持的用户入口为 suite-local PowerShell bootstrap：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\debug\sim2sim\root_cause_suite\run_suite.ps1 run
```

辅助命令仅用于恢复和审计：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\debug\sim2sim\root_cause_suite\run_suite.ps1 run --resume <run-id>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\debug\sim2sim\root_cause_suite\run_suite.ps1 verify <run-id>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\debug\sim2sim\root_cause_suite\run_suite.ps1 report <run-id>
```

`run_suite.ps1` 在启动任何 Python 解释器之前设置 `PYTHONNOUSERSITE=1` 和 `PYTHONDONTWRITEBYTECODE=1`，然后只允许调用项目冻结解释器 `wheelleg_dreamwaq/.venv/Scripts/python.exe -B -m debug.sim2sim.root_cause_suite ...`。`-B` 必须是解释器参数，不能在 Python 代码导入后补设。`__main__.py` 仍作为内部模块入口，但启动时必须断言 `sys.dont_write_bytecode is True` 且上述环境变量正确；不带 bootstrap/`-B` 的直接 `python -m` 明确不受支持并拒绝运行。这样 `debug/__init__.py` 与 `debug/sim2sim/__init__.py` 首次导入前已经禁止 bytecode 写入。

默认 `run` 只执行全部 `[CORE]` 独立分支。只有身份门、schema 门或插桩等价门失败时才停止依赖它的后续阶段；某个物理候选已经得到支持时，其他 Core 独立物理分支仍继续执行，以检测多重问题。CLI 不提供启用 Extended probe 的隐藏开关。

### 5.2 进程隔离

父进程只负责纯 Python 编排、hash、schema、分析和报告，不同时 import Isaac 与 MuJoCo。

- Isaac worker：`wheelleg_dreamwaq/.venv/Scripts/python.exe -B`；父进程显式传入 `--kit-root <run>/runtime_cache/kit`，worker 在任何 Isaac import 前用标准库校验路径并安装 `PythonWriteGuard`，然后由 `RootCauseSuiteAppLauncher` 构造 §3.3 的 `kit_args` 并在 `SimulationApp` 实例化前强制 `enable_crashreporter=False`
- MuJoCo worker：直接调用冻结解释器 `sim2sim/mujoco/.venv/Scripts/python.exe -B`，禁止通过 `uv run`、pip 或任何会同步/解析依赖的 launcher 启动
- 分析器：项目 `.venv` 的 `python.exe -B`，只依赖 Python、NumPy、Torch 的 CPU 功能

所有父、子进程命令行必须显式包含 `-B`，并继承 `PYTHONNOUSERSITE=1`、`PYTHONDONTWRITEBYTECODE=1` 和指向当前 run 的 `PYTHONPYCACHEPREFIX`；进程启动即断言 `sys.dont_write_bytecode`。G00 在启动前后核验 `run_suite.ps1`、两个 `python.exe`、`pyvenv.cfg`、对应 `pyproject.toml`/`uv.lock`，`.venv/Lib/site-packages/*.dist-info/` 下 `METADATA`、`RECORD`、`direct_url.json` 的文件集合与 SHA256，以及项目 `.venv/Lib/site-packages/isaacsim/kit/{logs,data,cache}` 的完整文件树；任一变化均为正式运行时身份漂移。run manifest 必须记录父进程与每种 worker 的完整 argv、环境契约、`sys.flags.dont_write_bytecode`、bootstrap SHA256 和 §2.2 七个 Isaac runtime source/config payload SHA256。每个 Isaac stage manifest 还必须记录 requested/resolved portable root、Kit log/data/general-cache/config/dump 路径、`kit-core.json` hash/key/default-value proof、requested/resolved/expected RTX `localTextureCachePath`、三方 canonical equality、pre-instantiation `enable_crashreporter`、resolved crash-reporter setting、loaded plugin name 列表，以及 Python write audit、Isaac Lab logger requested config/expected directory、`pre_handlers`、`created_lifetime_handlers`、`post_handlers` 和 exact-parent equality。编排器和测试不得调用 `uv run`、`uv sync`、pip、conda 或其他环境管理命令。这样运行期间不会解析 lock、同步 `.venv`、访问全局 uv cache 或写入项目内默认 Kit portable 目录或 `%TEMP%/isaaclab/logs`。

进程隔离单位是“不可变模型/配置族”，不是每个轴、每个正负号都重新启动一次 Kit：

- 不同模型 variant、drive 模式、闭链模式、接触探针类型和 G03 插桩对照使用独立 worker 进程。
- 同一不可变配置族中的不同轴、正负激励和重复样本，可以在一个 worker 进程内串行执行。
- 每个 case 仍必须执行 fresh reset，核验配置 hash、model hash、reset 三相状态、仿真时钟、随机状态和上一 case 残留为零。
- fresh-reset 核验失败时，该进程内本 case 及其后续 case 全部失效；编排器重启新进程后从该 case 重跑。
- 同一时间最多一个 Isaac Kit 进程，Isaac 和 MuJoCo 永不在同一解释器中 import。

这样做是为了同时排除状态污染，并避免数百次无意义的 Kit 冷启动：

- 两套依赖在同一解释器中的版本冲突。
- 上一个 case 遗留的仿真状态、GPU state 或全局配置污染。
- 并发 Isaac Kit 对 GPU 和本地数据库的竞争。

### 5.3 计划新增文件

| 文件 | 职责 | 允许依赖 |
|---|---|---|
| `run_suite.ps1` | 在 Python 启动前设置只读环境并强制项目解释器 `-B` | PowerShell、项目 `.venv` |
| `__init__.py` | 包身份，不产生副作用 | 标准库 |
| `__main__.py` | CLI 入口 | `cli.py` |
| `cli.py` | 参数解析、exit code、用户命令 | `orchestrator.py`、`report.py` |
| `contracts.py` | run/stage/trace/verdict schema、枚举和版本 | 标准库、NumPy |
| `workspace_guard.py` | 路径白名单、正式 hash 前后快照、原子写入 | 标准库 |
| `python_write_guard.py` | 第三方 import 前安装 Python 文件内容写入/路径 mutation audit hook、FileHandler 生命周期登记与 pre/post 快照 | 仅标准库 |
| `scenario_catalog.py` | Core 实验、配置族、重复性族、主信号/时间窗/聚合量、依赖图和目的 | `contracts.py` |
| `orchestrator.py` | 顺序调度、resume、子进程、stage 状态 | 上述纯模块 |
| `variant_builders.py` | 只生成 Core 所需的 no-ground、gravity-off、closure-off、drive-off 和 sphere coupon 副本 | 标准库；Isaac worker 可用 USD API |
| `isaac_bootstrap.py` | suite-local `AppLauncher` 子类、pre-instantiation launch-config hardening 与 Kit path assertions | Isaac Lab `AppLauncher`、标准库 |
| `isaac_worker.py` | Isaac 场景创建、施力、采集和自校验 | Isaac Lab、正式环境只读 import |
| `mujoco_worker.py` | MuJoCo 场景创建、施力、采集和自校验 | MuJoCo、正式 adapter 只读 import |
| `trace_contract.py` | 引擎无关 trace 字段、shape、单位和校验 | NumPy |
| `compiled_properties.py` | per-body mass/COM/inertia 归一化、frame 转换、静态审计和解析 golden | NumPy、`trace_contract.py` |
| `body_mapping_v1.json` | 冻结 27 个 canonical body 与 Isaac/MuJoCo/source-audit 名称映射及顺序 | 数据文件，由 `compiled_properties.py` 校验 |
| `analysis.py` | 对齐时间、基线扣除、奇响应、动量/能量账本 | NumPy |
| `verdict.py` | 依据固定决策表生成主因、次因与置信度 | `contracts.py`、分析 JSON |
| `report.py` | 生成 `report.md`、CSV 和摘要 | 纯 Python |
| `tests/` | 单元、合成、MuJoCo、Isaac smoke 和端到端测试 | 对应测试环境 |

依赖方向固定为：

```text
formal code/models (read-only)
          ↓
engine workers + variant builders
          ↓ files only
trace_contract → analysis → verdict → report
          ↑
orchestrator / CLI
```

正式 `source/` 和正式 `sim2sim/mujoco/wheelleg_mujoco/` 不得反向 import 本套件。

### 5.4 运行目录

```text
debug/sim2sim/root_cause_suite/
  runs/
    <run-id>/
      run_manifest.json
      run_state.json
      factor_path_allowlist.json
      input_hashes_before.json
      input_hashes_after.json
      runtime_cache/
        pycache/
        pytest-cache/
        pytest-tmp/
        python-write-audit.jsonl
        kit/
          logs/
          data/
          cache/
        isaaclab/
          logs/
      commands/
      models/
      stages/
        G00_integrity/
        G01_repeatability/
        G02_adapter/
        G03_instrumentation/
        P10_rest/
        P20_static_properties/
        P30_actuator/
        P40_closure/
        P50_contact/
        P60_full_robot/
        C70_checkpoint/
      analysis/
        gate_results.json
        evidence.csv
        verdict.json
        report.md
        file_hashes.json
```

每个 stage 先写入同目录下临时 sibling，全部文件完成并通过 hash 校验后再原子改名。`--resume` 只复用 `complete` 且输入 hash 未变化的 stage。

Core V1 不创建 `rigid_snapshot`、`common_mesh`、constraint-compliance sweep 或六轴 wrench 产物目录。若这些目录在本轮 run 中出现，G00 范围门必须使 run 失效。

## 6. 通用数据契约

### 6.1 所有实验都必须记录

- engine、版本、设备、seed、场景版本、worker 源文件 hash。
- 正式输入和 debug 模型副本的来源 hash、变换列表和输出 hash。
- physics dt、control dt、substep 数和每次控制写入时点。
- pre-forward、post-forward、首个 policy observation 三相 reset 快照。
- 编译后的 per-body mass、COM pose、COM-frame inertia、body/link frame 定义和名称映射；原始值与统一到 canonical frame 后的值必须同时保存。
- base COM pose/velocity、全系统 COM、总线动量、总角动量和动能。
- 26 个 hinge 的名称顺序、位置和速度。
- 6 个受控关节的 canonical/native 状态、目标和动作。
- 闭链 site/anchor 残差；可获得时记录 constraint impulse/work，不能获得时明确为 unavailable。
- 轮接触状态、接触点、法向量、法向冲量；摩擦力不可直接获得时，通过总水平动量变化和轮滑移间接测量，并标记测量语义。
- 每个 contact 字段都必须带 `availability` 和 `measurement_semantics`。现有 observer 的 `unexpected_contact=-1` 只能在 worker 边界转换为 `value=null, availability=false`，不得作为“没有非轮接触”的数值证据。
- Core V1 不新增全 body contact observer。非轮接触默认 `unavailable`；凡归因时间窗可能越过非轮接触或正式 termination，而又没有通过 G03 的观察量，相关接触结论最高只能为 contributor 或 `INCONCLUSIVE`。
- 施加的外力、外力矩、直接 effort 或 target profile 的精确数组与 hash。
- 终止前冻结状态和 reset 后返回观测必须分开存储。

### 6.2 时间对齐

跨引擎只在共同物理时刻比较：`0, 5, 10, 15, 20, 40, 100, 200, 400 ms`。MuJoCo 保留全部 `1 ms` 子步；Isaac 保留全部 `5 ms` 子步。不得把 1 ms 样本伪装成 PhysX 中不存在的等价状态。

控制边界仍固定为 `20 ms`。本套件不会把 MuJoCo 改成 `5 ms x 4` 后的结果当正式结果；该配置只作为已经存在的 timing sensitivity 证据。

### 6.3 正负对称激励

外力、外力矩和关节激励使用 `+u`、`0`、`-u` 三组：

```text
odd_response  = (response(+u) - response(-u)) / 2
even_response = (response(+u) + response(-u)) / 2 - response(0)
```

- `odd_response` 用于比较线性主响应并抵消自然下落和 reset 偏置。
- `even_response` 用于发现接触非线性、饱和、单侧约束和碰撞修正。

只看 `+u` 的单边结果不能形成高置信度因果结论。

### 6.4 重复性包络

重复性按不可替换的“配置签名 + 初态签名 + 激励签名”预声明，不能只用 zero-action 的静止包络覆盖动态场景。每个 scenario 必须保存完整、稳定排序的 `configuration_semantics` 和 `excitation_semantics`，不能只保存不透明的 variant 文件 hash。

`configuration_semantics` 是从 composed USD/PhysX view 或 compiled `MjModel` 读回的规范化 JSON，至少包含：

```text
schema_version, engine
topology: body/joint/dof/actuator/constraint/contact-pair counts and ordered names
environment: ground enabled/geometry/pose/filter, gravity, base_mode
timing: physics_dt, control_dt, substeps, write/sample cadence
engine_options: complete engine-specific result-determining options and resolved values
static_model: per-body property hash, joint axis/range/armature/frictionloss hash,
              geom shape/pose/filter hash
closure: mode, ordered constraint names/types/endpoints/enabled/solver fields
actuation: drive_mode, external_controller_enabled, target_neutralization,
           all 26 robot hinge names and resolved stiffness/damping/armature,
           six actuator names/transmission/gain/bias/gear/limits
contact: mode, ordered pair/material identities, friction coefficients,
         combine mode/condim/solref/solimp where available
observer_mode
```

其中 `engine_options` 不是手写的少数字段，而是完整、递归、稳定排序的结果决定配置：

- MuJoCo 必须直接从 compiled `model.opt` 序列化本机 MuJoCo 3.14 暴露的全部公共字段：`ccd_iterations`、`ccd_tolerance`、`cone`、`density`、`disableactuator`、`disableflags`、`enableflags`、`gravity`、`impratio`、`integrator`、`iterations`、`jacobian`、`ls_iterations`、`ls_tolerance`、`magnetic`、`noslip_iterations`、`noslip_tolerance`、`o_friction`、`o_margin`、`o_solimp`、`o_solref`、`sdf_initpoints`、`sdf_iterations`、`sleep_tolerance`、`solver`、`timestep`、`tolerance`、`viscosity` 和 `wind`。枚举和 bit field 同时保存稳定整数值与符号名。正式 `model_semantics.py` 没有覆盖这些字段，因此该完整快照由 suite worker 独立实现，不修改正式 adapter。
- Isaac/PhysX 必须序列化非渲染 `SimulationCfg` 的 `physics_prim_path`、`device`、`dt`、`render_interval`、`gravity`、`enable_scene_query_support`、`use_fabric`、`create_stage_in_memory` 和完整递归 `physics_material`；必须递归序列化 `PhysxCfg` 的全部字段，而不是选取子集，包括 solver type、articulation/contact solve order、position/velocity iteration count 上下界、CCD、stabilization、external-force-per-iteration、enhanced determinism、bounce/friction thresholds/correlation distance，以及全部 GPU contact/patch/pair/aggregate/stack/heap/temp-buffer/partition/soft-body/particle capacity。还必须从 composed `/physicsScene` 的 PhysxSceneAPI 读回对应 resolved attributes，并逐 articulation 读回实际 position/velocity solver iteration counts。字段集合以 §2.2 冻结 `simulation_cfg.py` 的递归 dataclass/configclass field introspection 为准；源码新增或缺失字段时不是静默忽略，而是 G00 runtime-contract failure。

上述 `engine_options` 路径在 drive、input、closure 和 friction ratio pair 中全部不可变化；这些值只用于证明实验除声明因素外同构，不要求 Isaac 与 MuJoCo 使用同名枚举或相同数值。

为了让 JSON-pointer diff 可执行，`robot_hinges`、`constraints`、`actuators` 和 `contact pairs/materials` 都必须编码为以冻结 canonical name 为 key 的 object，而不是会因删除/排序而整体移位的匿名 array；名称列表另存为排序审计字段。catalog 在加载时把上表中的“26 个 hinge”或“冻结 constraint 列表”展开为具体、无通配符的 JSON-pointer 集合并写入 `factor_path_allowlist.json`，其 SHA256 进入 run manifest。validator 不接受运行时正则、前缀匹配或 `*` 通配符。

`excitation_semantics` 至少包含 `input_kind`、单位、canonical channel/order/sign、实际输入数组、canonical torque-equivalent profile hash、正负号集合、start/stop 和 sample schedule。`source_model_sha256`、`model_artifact_sha256`、`transform_manifest_sha256`、`configuration_semantics_hash` 和 worker source hash 是独立 identity 字段；它们必须校验，但不作为因素差分中的伪语义字段。

每个 scenario 的派生身份为：

```text
configuration_semantics_hash = sha256(canonical_json(configuration_semantics))
configuration_hash = sha256(engine + source_model_sha256 + model_artifact_sha256 +
                            transform_manifest_sha256 + configuration_semantics_hash +
                            worker_source_hash)
pre_forward_initial_condition_hash = sha256(reset_written_pre_forward canonical root
                                            pose/velocity + all joint position/velocity +
                                            command + seed + RNG state + reset artifact hash)
post_forward_state_hash = sha256(reset_forwarded_post_forward canonical root pose/velocity +
                                 all joint position/velocity + closure residuals)
reset_returned_policy_hash = sha256(actor_obs_policy_returned + previous action +
                                    initialized policy history), or unavailable
excitation_hash   = sha256(canonical_json(excitation_semantics))
repeatability_key = sha256(configuration_hash + pre_forward_initial_condition_hash +
                           post_forward_state_hash + reset_returned_policy_hash +
                           excitation_hash)
```

三个 reset phase hash 具有不同语义，禁止合并或互相替代。`pre_forward_initial_condition_hash` 表示 worker 实际写入且尚未经过 constraint/forward 的共同输入，是消融 pair 的初态身份；`post_forward_state_hash` 是约束投影后的结果身份；`reset_returned_policy_hash` 是生产 API 首次返回给 Actor 的权威输入身份。hash 均使用同一 canonical 顺序、单位、dtype/endianness 和可用性标记；跨引擎数值等价仍由 G02 的容差门判断，不能要求两个不同引擎的字节 hash 相等。

ratio validator 必须分别在 Isaac before/after 和 MuJoCo before/after 上递归比较 `configuration_semantics` 与 `excitation_semantics`，得到 JSON-pointer path 差异集合；两个引擎都必须满足同一 factor allow-list。因素到唯一允许 path 的投影冻结为：

| factor | domain | 允许变化的 compiled semantic paths |
|---|---|---|
| `drive_mode` | configuration | `/actuation/drive_mode`、`/actuation/external_controller_enabled`、`/actuation/target_neutralization`；Isaac 另要求 26 个 robot hinge 的 resolved `stiffness/damping` 全部形成 exact diff；MuJoCo 另要求正式值非零的 20 个 passive robot hinge `dof_damping` 形成 exact diff。MuJoCo 的 6 个 controlled hinge `jIJ,jIO,jAB,jAG,jwheel_left,jwheel_right` 在正式 XML 与 drive-off variant 中 compiled `dof_damping` 都为 0，因此它们必须被完整记录并证明为不变量，不能伪造为已变化 path；六通道显式 PD/velocity servo 的启用差异由 `external_controller_enabled` 表示，其冻结 Kp/Kd、motor gain/bias/gear/limits 仍作为不变量记录。armature、motor gain/bias/gear/limits 不允许变化 |
| `input_kind` | excitation | `/input_kind`、`/input_units`、`/actual_input_values`；channel/order/sign、start/stop、sample schedule 和 `canonical_torque_equivalent_profile_hash` 必须相同 |
| `closure_mode` | configuration | `/closure/mode` 与冻结 constraint 列表逐项 `/enabled`；constraint 名称、endpoint、solver fields 和 topology counts 必须相同。closure-off variant 必须保留 constraint topology，只切换 enabled |
| `friction` | configuration | 仅 common-sphere coupon/floor 对的 friction coefficient values：Isaac `static_friction/dynamic_friction`，MuJoCo compiled pair `friction[0:5]`；material/pair identity、combine mode、condim、solref、solimp 与所有 geometry 字段必须相同 |

任何不在表中的 semantic path 变化、允许 path 未发生预期变化、两个引擎的 factor 投影不一致，或 transform manifest 声明的字段与 compiled diff 不一一对应时，ratio 证据 fail-closed。`model_artifact_sha256` 可以因合法 variant 不同而不同，但 source hash 必须相同；文件 hash 差异既不能被当作额外 factor，也不能用来掩盖 compiled semantics 中的隐藏变化。mutation test 必须覆盖“合法 factor 之外再改一个 joint armature/body mass/contact solver 字段”，并额外覆盖 MuJoCo `integrator`、`noslip_iterations`、`ccd_tolerance`、`enableflags`/`disableflags`，以及 PhysX `solver_type`、`enable_ccd`、`enable_stabilization`、`bounce_threshold_velocity`、`friction_offset_threshold`；任一隐藏变化都必须被拒绝。

只有 `configuration_hash`、`pre_forward_initial_condition_hash`、`post_forward_state_hash`、`reset_returned_policy_hash` 和 `excitation_hash` 都完全相同的同引擎重复样本才能形成包络。pre-forward hash 必须来自实际写入的 reset pose/velocity、26 关节 q/qd、command、seed/RNG state 和 reset-cache/artifact SHA，而不是只哈希请求参数；post-forward 与 returned-policy hash 必须来自各自命名相位的实际读回。不同高度、初始竖直/切向速度、command、seed、closure、drive、friction、first-action/replay、轴、符号 profile 或 contact coupon 不得共享包络。P50-A 的三个高度 x 两个初始竖直速度因此是 6 个不同 repeatability keys。下表是 repeatability family namespace；其中每一个进入 verdict 的不同 key 都必须在每个引擎内独立重复至少 3 次，不能用同 family 的“代表关节”替代其他 hash。

| repeatability family | 配置与代表激励 |
|---|---|
| `reset_nominal_zero` | nominal formal robot，ground on，formal drive，zero action，20 tick |
| `p10_rest_closure_on` | no ground，zero-g，drive off，closure on，zero input |
| `p10_rest_closure_off` | no ground，zero-g，drive off，closure off，zero input |
| `p10_freefall_closure_on` | no ground，gravity on，drive off，closure on，zero input |
| `p30_open_direct` | fixed base，no ground，closure off，direct effort；每个 verdict-bearing joint 与完整 `+tau/0/-tau` profile 分别 hash/重复 |
| `p30_open_target` | fixed base，no ground，closure off，formal target；每个 verdict-bearing joint 与完整 `+target/0/-target` profile 分别 hash/重复 |
| `p30_closed_direct` | fixed base，no ground，closure on，direct effort；每个 verdict-bearing joint/profile 分别 hash/重复 |
| `p30_closed_target` | fixed base，no ground，closure on，formal target；每个 verdict-bearing joint/profile 分别 hash/重复 |
| `p40_closure_on_pulse` | free base，no ground，zero-g，closure on，drive off，对称腿 effort profile |
| `p40_closure_off_pulse` | 与上一行仅 closure off 不同，使用同一 canonical effort profile |
| `p50_sphere_impact` | common sphere，friction 0，中间高度，零初始竖直速度 |
| `p50_sphere_slide_zero_friction` | common sphere，friction 0，正向代表切向速度 |
| `p50_sphere_slide_nominal_friction` | common sphere，冻结 nominal friction，正向代表切向速度 |
| `p60_zero_action_drive_off` | formal full robot，ground/closure on，drive off，zero action |
| `p60_first_action_formal_drive` | formal full robot，ground/closure on，formal drive，共享第一动作 |
| `p60_open_loop_replay` | formal full robot，ground/closure on，formal drive，冻结 replay 序列及其 hash |

同一不可变配置中的重复样本允许在一个 worker 进程中按 fresh reset 串行执行。诊断 observer 与正式路径等价门保持 5 次。每个场景只能引用 `scenario_catalog.py` 中同时绑定的 `repeatability_key`、`configuration_hash`、三个 reset phase hash 和 `excitation_hash`；任何错误族绑定均为 schema failure，不得在看见跨引擎结果后改用更宽的其他族。每个信号的有效容差为：

```text
effective_tolerance = max(legacy_material_floor,
                          5 * within_engine_repeat_envelope)
```

`legacy_material_floor` 沿用现有 `compare_traces.py` 的量级：

| 信号 | floor |
|---|---:|
| observation/action/target | `1e-6` |
| 主动/被动关节位置 | `1e-3 rad` |
| 主动/被动关节速度 | `1e-2 rad/s` |
| base COM 位移/高度 | `1e-3 m` |
| base 线速度 | `2e-2 m/s` |
| base 角速度 | `2e-2 rad/s` |
| base orientation geodesic | `8.726646e-3 rad`（`0.5 deg`） |
| projected gravity | `1e-3` |
| virtual leg `phi0` | `1e-3 rad` |
| virtual leg length | `1e-4 m` |
| loop closure residual | `1e-4 m` |

若引擎内重复误差本身超过 legacy floor，报告必须先标记 nondeterminism；跨引擎误差只有超过匹配配置族扩大后的容差才算材料级分歧。缺少匹配族包络时该信号为 `repeatability_unavailable`，不能回退到 zero-action 包络。

### 6.5 响应差异与消融解释率

对每个物理 probe，分析器计算：

- 共同时间点上的最大绝对误差和 RMSE。
- 相对响应误差：误差除以两侧响应幅值的较大者，并设置单位相关的最小尺度，避免除以接近零的数。
- 首个数值、材料和持续 3 个共同采样点的分歧。
- 正负激励的奇响应误差与非线性偶响应。
- 动量、角动量、能量和外部冲量的守恒残差。

每个 probe 必须在 `scenario_catalog.py` 中事前冻结以下字段：

```text
primary_signals
comparison_window_ms
scalar_metric = rmse | max_abs | event_delta
channel_reduction = named_channel | l2 | max
normalization_scale
guard_metrics
repeatability_family
configuration_hash
configuration_semantics_hash
pre_forward_initial_condition_hash
post_forward_state_hash
reset_returned_policy_hash
excitation_hash
comparison_profile_hash = null | <matched comparison profile>
ratio_pair_id = null | <frozen pair id>
baseline_scenario_id = null | <scenario id>
ablation_scenario_id = null | <scenario id>
expected_improvement_direction = null | lower
allowed_ablation_factors = [] | [drive_mode,input_kind] | [closure_mode] | [friction]
guard_thresholds
replay_source_spec = null | <P60-D frozen generator identity and selection rule>
replay_source_identity_hash = null | <sealed source/action identity before replay>
```

`allowed_ablation_factors` 是按字典序写入 JSON 的精确集合，不是“至少允许这些差异”。validator 必须按 §6.4 在两个引擎内分别比较规范化 configuration/excitation semantics；实际 factor 集合、实际 JSON-pointer path 集合与 allow-list 投影不完全相等时 fail-closed。derived file/configuration hash 只做身份核验，不参加因素推断。所有合法 ratio pair 在每个引擎内的 `pre_forward_initial_condition_hash` 必须相同；不得把 post-forward 差异混入初态身份。

post-forward/returned-policy 的 pair guard 按类别冻结：P30 target/direct 与 P50-C nominal/zero-friction 在激励开始前必须具有相同 `post_forward_state_hash`，且 `reset_returned_policy_hash` 在可用时也必须相同；否则 pair 无效。P40 closure on/off 只要求 pre-forward hash 相同，允许 `post_forward_state_hash` 与 returned-policy hash 不同，并把该差异作为闭链投影的待测结果而不是污染。P40 的同一 closure-mode 重复样本仍必须同时匹配全部三个 phase hash。

`max_abs` 和 `RMSE` 都必须输出，但只有 `scalar_metric` 指定的量能进入主 verdict；`guard_metrics` 只负责阻断异常，不得被事后替换成更有利的主指标。Core V1 的主指标冻结为：

| probe | primary signal | window | scalar metric | channel reduction | 是否使用 explanation ratio |
|---|---|---:|---|---|---|
| P30 open/closed direct/target | 六个 canonical joint `odd_response.velocity` | `0..100 ms` | normalized RMSE | L2 | 仅使用下表两组冻结配对 |
| P40 closure on/off | 预声明主动与被动关节 `odd_response.velocity` | `0..100 ms` | normalized RMSE | L2 | closure on 对 closure off |
| P50-A normal impact | coupon rigid body `COM velocity z` 与首次接触事件时刻 | 接触前 `20 ms` 至接触后 `100 ms` | RMSE；事件另作 guard | named channels | 否，使用直接隔离失败规则 |
| P50-C tangential slide | coupon rigid body `COM velocity x` | `0..400 ms` | normalized RMSE | named channel | nominal friction 对 friction=0 |
| P60-B closure/contact 回装 | `base_angular_velocity_y` | `0..100 ms` | normalized RMSE | named channel | 否，只作整机佐证 |
| P60-C shared first action | 六个 canonical joint velocity | `0..100 ms` | normalized RMSE | L2 | 否，P60-B 激励不同，不得伪造配对 ratio |
| P60-D open-loop replay | `base_angular_velocity_y` | `0..100 ms` | normalized RMSE | named channel | 否，只判断同动作 plant 分歧与 feedback 放大 |

允许进入主 verdict 的 explanation-ratio 配对只有以下四组；catalog 中出现其他非空 `ratio_pair_id` 时 G00 必须失败：

| ratio pair id | baseline scenario | ablation scenario | allowed factors | expected direction | guard thresholds |
|---|---|---|---|---|---|
| `P30_OPEN_TARGET_TO_DIRECT` | `P30_B_OPEN_FORMAL_TARGET` | `P30_A_OPEN_DIRECT_EFFORT` | `[drive_mode,input_kind]` | `lower` | bypass 全断言通过；两侧 direct odd-response 幅值均 `>10x` 对应 effective tolerance；无 effort/velocity saturation；相同 joint/sign/timing/window 与 `comparison_profile_hash` |
| `P30_CLOSED_TARGET_TO_DIRECT` | `P30_C_CLOSED_FORMAL_TARGET` | `P30_C_CLOSED_DIRECT_EFFORT` | `[drive_mode,input_kind]` | `lower` | 与上一行相同，且 closure mode/hash 相同、只有 drive mode/input kind 不同 |
| `P40_CLOSURE_ON_TO_OFF` | `P40_B_CLOSURE_ON` | `P40_B_CLOSURE_OFF` | `[closure_mode]` | `lower` | P20-S 无静态物性 hard failure；无 contact；相同 effort profile/window；closure-off 机械响应可观测 |
| `P50_NOMINAL_TO_ZERO_FRICTION` | `P50_C_NOMINAL_FRICTION` | `P50_C_ZERO_FRICTION` | `[friction]` | `lower` | 两侧均发生有效接触；P50-A 法向门通过或法向冲量误差不超过有效容差；相同初速度/profile/window |

P30 的 matched comparison profile 事前定义为同一 joint、sign、start/stop 和初始 canonical torque：腿通道 `tau0 = formal_Kp * delta_q_target`，轮通道 `tau0 = formal_Kd * delta_v_target`；direct 分支施加该 `tau0` 脉冲，target 分支施加对应 `delta_q_target/delta_v_target`。两者 `excitation_hash` 因 `input_kind` 和 actual input values 不同而必须不同，configuration semantics 只允许 `drive_mode` 投影中的 paths 不同，但 pre-forward、post-forward、returned-policy guard、canonical torque-equivalent profile hash 和 `comparison_profile_hash` 必须相同；幅值必须低于 effort/velocity limit 且不得由结果反推。其他 ratio pair 的 excitation semantics/hash 和 pre-forward hash 必须相同，并遵守上一段的类别专属 post-forward/returned-policy guard；configuration semantics 只允许表中 `allowed_ablation_factors` 对应 paths 不同。所有 pair 的 source model hash 必须相同，artifact/transform/configuration hash 分别作为 before/after identity 记录。

候选消融的解释率定义为：

```text
explanation_ratio = 1 - mismatch_after_ablation / mismatch_before_ablation
```

它只在上述冻结 pair 中、before/after 使用同一预声明主信号、标量指标、通道聚合、归一化尺度、时间窗和 `comparison_profile_hash`，且两个引擎各自的 configuration/excitation semantic path 差异精确等于 allow-list 的 `allowed_ablation_factors`、pre-forward 初态相同、类别专属 post-forward/returned-policy guard 通过、未引入新 hard-gate 失败时有效。`baseline_scenario_id`、`ablation_scenario_id`、两者的 configuration/semantics/三个 reset phase/excitation/comparison-profile hash、逐引擎 semantic diff paths、allowed factors 和 expected direction 必须写入 evidence；`evidence.csv` 还必须逐信号输出独立 ratio，主 verdict 只引用预声明主指标。
如果 `mismatch_before_ablation` 没有超过 effective tolerance，解释率不定义，候选直接记为 `not_supported`；禁止用接近零的分母制造高解释率。

- `>= 0.70`：可作为该候选的强支持证据。
- `>=0.30` 且 `<0.70`：只能标记为 contributor。
- `< 0.30`：不能用该消融支持主因结论。

`0.70/0.30` 是 `VerdictPolicyV1` 的审计阈值，不是假装存在的物理定律。最终报告必须同时给出原始误差和阈值敏感性，避免只展示一个标签。

## 7. 分阶段实验与 Core/Extended 排除链

## 7.1 G00：身份、范围和环境完整性门

### 为什么做

任何后续动态差异只有在“比较的是预期文件、预期策略和预期运行时”时才有意义。该门防止把文件漂移、旧 Actor、错误 manifest 或误改正式模型当成物理差异。

### 执行内容

1. 按 §2.3 的两个精确相对路径加载策略，计算主架构、正式 USD bundle、正式 MJCF、model manifest、正式 adapter、环境、Actor 和 policy manifest 的 hash；禁止扫描或从 `phase1_v4` 猜测 Phase 1R 策略位置。
2. 记录 Python、Torch、Isaac Sim、Isaac Lab、RSL-RL、MuJoCo、CUDA 和 GPU 身份。
3. 解析两个 policy manifest，验证 ActionV1、ActorObsV1、history、normalization、ControlFrameV1、时钟和模型身份。
4. 建立写入路径守卫，解析所有目标路径的绝对路径，拒绝任何逃出 suite root 的写入。
5. 验证 scenario catalog 只包含本文件标记为 `[CORE]` 的 probe；发现 Extended-only scenario 或未声明参数点立即失败。
6. 记录 official `run_suite.ps1`、父进程和各 worker 的完整 argv、环境、`sys.flags.dont_write_bytecode` 与 bootstrap/source hashes；验证父、子解释器都从启动时使用 `-B`，并验证 Python/pytest cache 目标位于当前 run。对 suite root 外项目文件以及项目内既有 Isaac Kit `logs/data/cache` 做完整 before snapshot。
7. 每个 Isaac worker 在导入 Isaac Lab 前安装 `PythonWriteGuard`，保存 pre-handler 快照，并核验 `kit-core.json` hash 与精确 setting key/default value、absolute portable root 和显式 absolute RTX texture-cache path；在构造环境前 clone config、核验 absolute Isaac Lab log dir，并断言没有预建 `SimulationContext`。`RootCauseSuiteAppLauncher` 在 `SimulationApp` 实例化前记录并断言 `enable_crashreporter=False`，启动后核验 resolved log/data/general-cache/config/dump paths 全在当前 run，且 texture cache requested/resolved/expected canonical path 三方精确相等、resolved crash setting 为 false、loaded plugin list 不含 `carb.crashreporter-*`；`DirectRLEnv`/`SimulationContext`/logger 的 frozen source/import/call contract 必须通过；环境构造后核验 pre、created-lifetime、post 三组 `FileHandler` 和完整 Python audit ledger，全部内容写入和 mkdir/remove/rmdir/rename/link/symlink/truncate/chmod/chown/utime 路径必须位于当前 run，Isaac Lab handler 父目录必须精确等于 `<run>/runtime_cache/isaaclab/logs`。缺任一证明、源码/config/logger hash 或 AST/key/source contract 不匹配、预建 context、错误 import origin、texture cache 仍解析到 `${omni_global_cache}`/用户目录/错误 sibling、Isaac Lab logger 回退 `%TEMP%`/错误 sibling/逃逸、受审计路径逃逸或项目内 Kit tree 变化立即失败。
8. 运行结束后再次计算全部正式输入 hash 与 inventory，并逐项比较。

### 要找出什么

- 是否正在运行预期的 DreamWaQ run-01 和 Phase 1R run-03。
- 是否有正式源码、模型或 manifest 在诊断期间被改动。
- 是否存在版本漂移导致旧诊断无法重现。

### 通过后排除什么

- 排除“拿错 checkpoint/Actor”。
- 排除“正式 MJCF 或 adapter 已被偷偷改过”。
- 排除“报告混用了不同版本产物”。

### 失败处理

输出 `INVALID_EVIDENCE_PIPELINE`，停止所有依赖阶段。不得输出任何物理根因。

## 7.2 G01：同引擎重复性与数值噪声门

### 为什么做

如果同一引擎、同一输入、同一 seed 的轨迹本身不稳定，跨引擎的细小差异可能只是 GPU/solver 非确定性，固定阈值会产生误报。

### 执行内容

1. Isaac 和 MuJoCo 各自重复 3 次 `nominal zero-action 20 tick`，建立 `reset_baseline` 包络。
2. 按 §6.4 对每个 verdict-bearing `repeatability_key` 运行至少 3 次，并建立该精确 key 的逐信号最大 pairwise envelope；不得只运行 namespace 中的代表 case。
3. Isaac 正式 step 与 debug observer 等价路径各重复 5 次，串行执行。
4. 对 reset、共同采样时刻、事件和终止时刻计算引擎内包络，并保留每对重复样本的原始差值。
5. 在任何跨引擎分析开始前冻结 `threshold_snapshot.json`；后续不得增加重复样本、切换代表 case 或改变族映射来放宽阈值。

### 要找出什么

- 哪些信号是逐位稳定的。
- 哪些信号需要使用自然复现包络而不是理想零误差。
- 是否存在无法支撑精细归因的非确定性。

### 通过后排除什么

- 排除“首个材料分歧只是一次偶发数值抖动”。
- 为后续每一类动态门提供匹配工况、有证据的容差。

### 失败处理

若关键状态的引擎内 envelope 已超过 legacy material floor，相关信号标记 `nondeterministic_unusable`。某配置族缺少有效包络时，依赖该族的 probe 不得执行材料级判定。如果 base、全系统动量和关节响应均不可用，则整体输出 `INCONCLUSIVE`；不允许通过放大阈值或借用其他配置族包络掩盖问题。

## 7.3 G02：reset、history、Actor 和 adapter 语义门

### 为什么做

该门验证“进入物理引擎之前”的整个数字链。只有它通过，后续状态分歧才可以被解释为 plant 差异。

### 执行内容

1. 比较 reset 写入前、constraint/forward 后和 API 返回给 policy 的三相状态。
2. 验证首帧 25D ActorObsV1 的每个切片、单位、归一化和 ControlFrame 方向。
3. 验证 DreamWaQ 125D history 是五个相同的当前帧，frame-major 排列，末 25D 等于当前帧。
4. 把同一个序列化 float32 输入送入两份独立加载的同一 TorchScript Actor，要求 CPU 输出最大误差 `<=1e-6`。
5. 比较 CENet velocity/context、Actor raw output、clipped action、六维 canonical/native target 和 previous-action 时序。
6. 对六个动作通道分别施加极小正负脉冲，验证 target 方向、native 符号和第一响应方向。
7. 验证 20 ms policy clock；记录但不把 `5 ms x 4` timing copy 当作正式语义。

### 要找出什么

- 初态是否真正相同，而不是只看 XML/USD 文本。
- 两侧首个 history 是否都由五帧当前状态填充。
- 观测、CENet、Actor、裁剪、动作顺序、轮端符号和 previous-action 是否有错。
- policy clock 或 target refresh 是否存在一拍偏移。

### 通过后排除什么

- 排除 history 初始化和 frame layout 错误。
- 排除 CENet/Actor 对同一输入计算不一致。
- 排除动作顺序、缩放、裁剪和主要反馈极性错误。
- 排除初始 PD reference 由 adapter 算错。

### 失败后的分类

- pre-forward 写入状态不一致：`SIM2SIM_ADAPTER_BUG`，子因 `reset_serialization`。
- pre-forward 一致、post-forward 才分离且闭链残差变化：暂不归 adapter，转入 `P40` 检查闭链投影。
- history/observation/action/target 任一硬门失败：`SIM2SIM_ADAPTER_BUG`。
- 该门失败时，checkpoint 和物理 verdict 全部无效。

## 7.4 G03：诊断插桩中立性门

### 为什么做

现有 debug collector 会覆盖 step 以读取子步，ContactSensor 也可能改变 PhysX 配置。如果 observer 改变轨迹，采到的数据不能代表正式训练环境。

### 执行内容

1. 比较正式 `WheelLegFlatEnv.step()` 与 debug 子步 collector 的 5 组 fresh-reset 10-tick 轨迹。
2. 比较 ContactSensor 关闭/开启后的同动作轨迹和解析后的 PhysX 属性。
3. 比较新增的动量/能量 observer 开启/关闭后的轨迹。
4. 验证采集只读字段不会额外推进仿真、抽取随机数或改变 render/step 计数。
5. MuJoCo worker 比较正式 runner 只读路径与 collector 路径的相同动作结果。

### 要找出什么

- debug step override、contact observer 或状态读取是否改变被测系统。
- collector 是否错读了 post-reset 状态、错过 terminal 状态或多走一步。

### 通过后排除什么

- 排除“观察工具制造了首处分歧”。
- 排除 terminal/reset 采样时点错误。

### 失败处理

失败的 observer 标记 `instrumentation_perturbation` 或 `instrumentation_failed`，相关字段不得进入归因。若 debug step 本身改变共同状态轨迹，整个 run 为 `INVALID_EVIDENCE_PIPELINE`；若只有可选 contact observer 不可用，其他阶段可以继续，但 contact 分类最多为 `INCONCLUSIVE`，不能继续使用污染字段。

## 7.5 P10 `[CORE]`：无接触静止、自由落体和初始约束预应力

### P10-A：无地面、零重力、零动作、drive 关闭、闭链开启

**为什么做：** 理想情况下系统从零速度开始，没有外力、接触和 actuator 输入，不应自行产生总动量或持续动能。任何运动只能来自初态不满足约束、约束投影、数值稳定项或模型内部预应力。

**要观察：** 全系统线/角动量、动能、base 运动、26 关节运动、闭链残差和 constraint impulse/work。

**能查出：** reset 后闭链是否在第一个物理步释放能量；两侧约束表达是否导致不同的自然松弛。

**通过后排除：** 排除“没有接触和控制时，闭链自己把机器人弹开”这一类主因。

**不能单独排除：** 不能排除有地面后的接触-闭链耦合，也不能排除 drive 差异。

### P10-B：无地面、重力开启、零动作、drive 关闭

**为什么做：** 自由落体的全系统 COM 加速度应接近重力，与总质量无关。它检查重力方向、坐标、base freejoint、积分和内部约束是否错误地向外系统注入动量。

**要观察：** COM z 加速度、水平 COM 漂移、总角动量、内部相对运动和 closure residual。

**能查出：** 重力轴/单位错误、base 自由度设置错误、约束导致的非物理外部反作用或严重积分异常。

**通过后排除：** 排除 gross gravity/frame/free-base 错误。

**不能单独排除：** 自由落体通过并不证明质量和转动惯量正确，因为平动重力加速度与质量无关。

### P10-C：重复 P10-A，但关闭闭链

**为什么做：** 用同一引擎的 `closure_on - closure_off` 增量识别闭链本身造成的运动，而不是直接比较两个拓扑不同的绝对轨迹。

**判定：** 若 closure-on 有材料级自发运动，closure-off 消失，且两引擎的 closure effect 显著不同，则给 `CLOSED_CHAIN_CONSTRAINT_MISMATCH` 增加强支持；若两种模式都运动，则优先检查初态、惯量或 worker 施力语义。

## 7.6 P20：质量、质心和转动惯量

### P20-S `[CORE]`：编译后静态物性审计

**为什么做：** XML/USD 文本相似并不证明两个引擎编译后的 per-body mass、COM pose 和 COM-frame inertia 相同。静态审计几乎没有动态仿真成本，并能在解释响应差异前先排除模型编译或 frame 转换错误。

**执行：**

1. source audit 的唯一输入冻结为 `artifacts/phase1_v4/mujoco-usd-data.json`（SHA256 `411B3913D03F4523132177E42B416AFA736A026E7ADB567A81AD28ECB75D53C5`），并同时验证其 embedded `asset_bundle_hash=E754AE888F5C5379B3B6152CFA5AD6BBAE20E8C7480CF7AD6960782A5267CB46` 与 `asset-bundle-v2-manifest.json` 的 active AssetBundleV2 身份一致。哈希或 embedded identity 不一致时 P20-S 不是物理失败，而是 G00 identity failure。
2. Isaac 从编译后的 articulation/root view 读取 per-body mass、COM pose 和 inertia；MuJoCo 从编译后的 `mjModel` 读取对应量。
3. 使用下表冻结的 27-body mapping 和明确的 body/COM frame 变换，把两侧值统一到 canonical body frame；实现时原样写入 `body_mapping_v1.json` 并校验其 SHA256，禁止按运行时名字相似度猜测映射。原始值和转换后值都写入 trace。
4. 两侧分别与 source audit table 比较，再做跨引擎逐 body 比较；禁止只比较总质量而忽略分布。
5. 计算总质量、系统 COM，以及名义姿态下关于系统 COM 的复合空间惯量。总质量同时核对已审计锚点 `4.396253988146782 kg`。
6. 静态物性容差预先固定为：per-body mass `max(1e-7 kg, 1e-6 relative)`，COM position `1e-6 m`，inertia element `max(1e-9 kg m^2, 1e-5 relative)`，COM orientation geodesic `1e-6 rad`，总质量锚点 `1e-6 kg`。

冻结 body mapping 如下；除 root 的 `base_link -> base` 外，其余名称恒等：

| canonical body | Isaac compiled name | MuJoCo compiled name | source-audit key |
|---|---|---|---|
| `base` | `base_link` | `base` | `base_link` |
| `jIO` | `jIO` | `jIO` | `jIO` |
| `jOP` | `jOP` | `jOP` | `jOP` |
| `jwheel_left` | `jwheel_left` | `jwheel_left` | `jwheel_left` |
| `jIO_dummy_child_link1` | same | same | same |
| `jIO_dummy_child_link2` | same | same | same |
| `jAG` | same | same | same |
| `jGH` | same | same | same |
| `jwheel_right` | same | same | same |
| `jAG_dummy_child_link1` | same | same | same |
| `jAG_dummy_child_link2` | same | same | same |
| `jIJ` | same | same | same |
| `jJM` | same | same | same |
| `jMK` | same | same | same |
| `jKN` | same | same | same |
| `jKN_dummy_child_link1` | same | same | same |
| `jKN_dummy_child_link2` | same | same | same |
| `jMK_dummy_child1` | same | same | same |
| `jMK_dummy_child2` | same | same | same |
| `jAB` | same | same | same |
| `jBE` | same | same | same |
| `jEC` | same | same | same |
| `jCF` | same | same | same |
| `jCF_dummy_child_link1` | same | same | same |
| `jCF_dummy_child_link2` | same | same | same |
| `jEC_dummy_child_link1` | same | same | same |
| `jEC_dummy_child_link2` | same | same | same |

表中的 `same` 在数据文件中必须展开为完整字符串；它只用于提高本文可读性。body 数量、顺序、集合或任一名字不一致均为 fail-closed，不能跳过未知 body 后继续比较。

**解析读回 golden：** 两个 worker 各自创建一个 suite 内生成的单刚体，无重力、无接触、无约束、无 actuator。固定参数为 `mass=2.3 kg`、body-frame COM offset `[0.11,-0.07,0.05] m`、互不相等且满足刚体惯量三角不等式的 principal inertia `[0.031,0.047,0.073] kg m^2`、principal-axis orientation 为绕归一化轴 `[1,2,3]` 旋转 `0.61 rad`、body origin `[0.3,-0.2,0.5] m`、非轴对齐 COM linear velocity `[0.7,-0.4,0.9] m/s`、world angular velocity `[1.1,-0.8,0.6] rad/s`，固定参考原点为 world `[0,0,0]`。读取 t=0 状态并验证：

```text
P = m * v_COM
H_COM = R_world_from_principal * diag(I_principal) * R_world_from_principal^T * omega_world
H_world_origin = H_COM + r_world_origin_to_COM x P
```

golden 必须同时验证关于 COM 和关于固定 world origin 的角动量。负向 mutation tests 至少包含：把 link-origin velocity 错当 `v_COM`、把旋转方向写成 `R^T I R`、遗漏 `r x P`；三种错误都必须超过 golden tolerance 并使测试失败。这项测试验证 worker 的参考点、frame 和角动量合成公式，不把仿真器积分误差混入读回验证。

**失败处理：**

- source audit 与某引擎编译值不一致：支持 `MASS_OR_INERTIA_MISMATCH`，子因 `compiled_static_property`。
- 两引擎静态值各自符合 source audit，但 worker golden 失败：输出 `INVALID_EVIDENCE_PIPELINE`；动量和角动量字段不得进入任何 verdict。
- 静态审计通过只能排除“编译后的静态物性表不一致”，不能排除动态积分、约束组合后的有效惯量或 wrench 坐标问题。

### P20-D `[EXTENDED]`：六轴动态 wrench 与有效惯量辨识

Extended V2 才执行以下矩阵：无地面、零重力、drive 关闭，分别 closure-on/off；对 base 施加 x/y/z 三轴 `+F/0/-F` 和 roll/pitch/yaw 三轴 `+T/0/-T` 短脉冲，比较 `Delta P` 与外部冲量、`Delta H` 与外部角冲量，并估计六轴有效惯量响应。

该阶段用于识别静态表相同但动态组合响应不同、惯量 frame 解释错误或 base wrench 坐标错误。Core V1 不实现、不运行，也不得把它标成已排除。

## 7.7 P30 `[CORE]`：执行器、drive 和积分语义

该阶段使用两条互相独立的通道，目的是把“机械体响应”与“目标控制器如何施力”分开。

### P30-A：固定 base、无地面、关闭闭链、直接 effort 脉冲

**为什么做：** 直接向六个受控关节施加同一 canonical effort 波形，并把两侧所有 actuator/drive stiffness 与 damping 置零，从而真正绕过 PhysX implicit drive 和 MuJoCo 显式 PD。该 probe 检查关节轴、armature、局部惯量和积分；formal damping 随 P30-B 一并恢复，不在本 probe 中混入。

**drive-off 是声明式模型 variant，必须在环境创建前完成：**

- Isaac debug config clone 中，腿、轮和 passive actuator group 的 stiffness/damping 全部为零；不得修改正式 `WHEELLEG_CFG`。
- 两侧 drive-off 的 robot hinge 集合冻结为 26 个名称，`base_free` 明确排除：controlled 为 `jIJ,jIO,jAB,jAG,jwheel_left,jwheel_right`；passive 为 `jOP,jIO_dummy_child_link1,jIO_dummy_child_link2,jGH,jAG_dummy_child_link1,jAG_dummy_child_link2,jJM,jMK,jKN,jKN_dummy_child_link1,jKN_dummy_child_link2,jMK_dummy_child1,jMK_dummy_child2,jBE,jEC,jCF,jCF_dummy_child_link1,jCF_dummy_child_link2,jEC_dummy_child_link1,jEC_dummy_child_link2`。worker 必须证明这个集合与 compiled robot hinge set 完全相等，不得只使用正式 `ModelMap` 的六个 controlled joints。
- 创建 articulation 后，同时读取并断言 actuator object 缓存 gains、`ArticulationData` resolved gains 和 PhysX view 编译 gains 全部为零。
- position/velocity targets 写为当前状态或零速中性值，并证明改变这些 target 不改变零增益下的响应。
- `set_joint_effort_target()` 后记录最终写入 PhysX 的 effort target、clipped effort 和 `applied_torque` host estimate；零增益、无饱和时 host estimate 必须与声明 effort 在 `1e-6 Nm` 内一致。
- MuJoCo 不调用正式 `MixedActionController.compute_torque()`，因此绕过外部显式 PD/velocity servo；debug variant 必须按上面的冻结名称解析并清零全部 26 个 robot hinge 的 `dof_damping`，不允许保留任何 passive joint 的正式 `0.05` damping。正式 `<motor>` 必须保持 fixed gain `1`、bias `0`、gear `1`，因为把 motor gain 置零会使 `data.ctrl` 不产生任何 actuator force。
- suite worker 必须把 canonical effort 通过冻结符号 `[+1,+1,+1,+1,+1,-1]` 转成 native control 后再写 `data.ctrl`。读取 `actuator_force` 与受控 DOF 的 `qfrc_actuator` 后再乘同一符号还原 canonical；在无饱和单 actuator golden 中，canonical input、canonical actuator force 和 canonical generalized force 的最大误差均须 `<=1e-12 Nm`。
- 编译后输出全部 26 个名称对应的 joint id、DOF address、source damping 和 compiled damping；断言名称集合无缺失/额外项、DOF address 唯一、26 项 `dof_damping=0`。再逐 actuator 断言 `gaintype=fixed`、`gainprm[0]=1`、`biasprm=0`、gear `1`，并验证右轮 `+tau_canonical -> -ctrl_native -> +tau_canonical` round-trip。任何 passive damping 非零、motor gain 为零或把 `base_free` 纳入清零集合的 variant 必须被拒绝。
- exact ratio diff 必须区分“被完整审计的 26 个 hinge”与“实际数值发生变化的 path”。正式 XML 的六个 controlled hinge `jIJ,jIO,jAB,jAG,jwheel_left,jwheel_right` 在 `sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml:58,72,92,106,124,170` 已是 `damping=0`，drive-off 后仍为 0；其余 20 个 passive hinge 的正式 `damping=0.05` 被清零。因此 MuJoCo `drive_mode` allow-list 只展开这 20 条 `/actuation/robot_hinges/<name>/dof_damping` 差异路径，同时要求六个 controlled hinge 的 compiled 值在两侧精确相等。把 unchanged path 填入 exact-diff 集合或用声明值覆盖 compiled readback 都属于无效证据。
- 任一断言失败时 P30-A/P30-C 为 `invalid_drive_bypass`，不得进入 actuator verdict。

**执行：** 六个通道逐个运行 `+tau/0/-tau`；记录主动关节和相邻被动链的角度、速度、总能量、施加 effort、最终 engine effort buffer 和饱和事件。

**要找出：** 相同 effort 下的局部关节响应是否一致；左右轮和四条腿的符号、增益或局部惯量是否异常。

**通过后排除：** 排除 gross joint-axis、canonical sign、armature 和局部机械响应错误。

**失败含义：** 说明差异早于 target controller，不能把问题只归因于 implicit drive。结合 P20/P40 再分为惯量或闭链问题。

### P30-B：固定 base、无地面、关闭闭链、正式 target 脉冲

**为什么做：** 在与 P30-A 相同的机械配置下恢复正式控制语义：Isaac implicit position/velocity drive 与 MuJoCo 每子步显式 PD/速度伺服。若直接 effort 匹配而 target 模式不匹配，差异就位于 actuator/drive/update cadence，而不是刚体惯量或接触。

**执行：**

- 四个腿通道使用小幅位置目标脉冲。
- 两个轮通道使用小幅速度目标脉冲。
- 六通道都运行正、零、负三组。
- 保存每个物理子步的 state-before, target, PD reference, Isaac host estimate 和 MuJoCo commanded torque。

**判定：**

- P30-A 通过、P30-B 失败：强支持 `ACTUATOR_INTEGRATION_MISMATCH`。
- P30-A 和 P30-B 都失败：actuator 不是唯一解释，优先查看 P20/P40。
- P30-A 和 P30-B 都通过：排除无接触、开链条件下的 actuator 主因；仍需 P30-C 检查闭链耦合。

### P30-C：固定 base、无地面、闭链开启，重复 direct effort 与 target

**为什么做：** drive 与闭链 solver 在完整机构中可能耦合。该实验比较 `closure-on - closure-off` 的 drive 增量。

direct-effort 分支必须复用 P30-A 的同一 drive-off 契约和全部运行时断言；target 分支必须使用未经调参的正式 stiffness/damping、target refresh 和 effort limit。两个分支不得共享已被原地修改的 actuator 对象，必须来自不同不可变配置族/worker 进程。

**判定：**

- 只有 target+closure-on 失败，而 direct-effort+closure-on 通过：drive/constraint 求解耦合，主分类仍为 `ACTUATOR_INTEGRATION_MISMATCH`，附带 closure contributor。
- direct-effort+closure-on 也失败且 P40 的 closure effect 失败：主分类转为 `CLOSED_CHAIN_CONSTRAINT_MISMATCH`。

### 该阶段明确不能做的事

Isaac 内部真实 implicit-drive torque 不可见，因此不能把 host estimate 和 MuJoCo command 做逐值相等门。主证据必须是相同输入后的状态、动量和能量响应。

## 7.8 P40：闭链表达、柔度和约束能量

### 为什么做

Isaac 使用 4 个 `PhysicsRevoluteJoint`，MuJoCo 使用 8 个 compliant `connect` 约束近似 4 个铰链。二者可能在 nominal pose 看起来闭合，但在第一物理步释放不同的约束冲量，或者在 actuator/contact 载荷下表现出不同柔度。

### P40-A `[CORE]`：无地面、零重力、零输入的约束松弛

该实验复用 P10-A/C，重点分析：

- reset forward 前后 site/anchor 残差变化。
- 第一物理步的动能增量。
- 闭链残差的衰减/振荡。
- 总动量守恒和内部能量注入。

它要找的是“闭链松弛缓存或约束投影是否在仿真启动时释放能量”。如果 pre-forward 状态一致但 post-forward/step 后分离，并且 closure-off 后消失，就排除 observation/Actor，支持闭链根因。

### P40-B `[CORE]`：对称腿 effort 脉冲，闭链 on/off

对左右对称的前/后腿关节施加相同和反相的正负 effort，记录：

- 主动和八个物理被动关节的奇响应。
- 左右 virtual-leg length/phi0。
- 每个 closure site 的残差与等效约束功。
- base reaction 和总角动量。

使用差分中的差分：

```text
closure_effect_engine = response_closure_on - response_closure_off
cross_engine_closure_error = closure_effect_isaac - closure_effect_mujoco
```

这样做是为了排除开链模型本身的局部关节响应，把证据集中在“加上闭链后发生了什么”。

### P40-C `[EXTENDED]`：约束柔度敏感性，不是参数搜索

允许使用预先声明的三个 MuJoCo debug-only 约束语义：formal、近似更硬、近似更软；Isaac 侧只使用正式 joint 和 closure-off 两种。三个 MuJoCo 点只用于验证 formal mismatch 对 constraint compliance 是否单调敏感，不用于挑一个“最好看”的参数。

Core V1 不生成这三个柔度 variant，也不得用未执行的 P40-C 解释率升级主因。Core 的闭链判定只依赖 P10-A/C、P30-C 和 P40-A/B。

### 要找出什么

- 是否存在初始闭链预应力。
- 两侧约束对相同关节负载的增量响应是否不同。
- MuJoCo formal `connect` 柔度是否足以解释大部分完整系统误差。

### 通过后排除什么

如果零输入、direct effort 和 closure effect 都在容差内，可以排除闭链作为独立主因；接触下仍可能存在闭链-接触耦合，留给 P60。

### 失败后的分类

满足以下全部条件时强支持 `CLOSED_CHAIN_CONSTRAINT_MISMATCH`：

1. adapter/reset pre-forward 门通过。
2. closure-on 材料失败，closure-off 显著改善。
3. 预声明 P40-B closure on/off 主指标的 `explanation_ratio >= 0.70`，或 P10/P30/P40 中至少两个独立 closure probe 同向失败。
4. P20-S 没有先证明 compiled static property mismatch；P20-D 未执行时必须保留动态惯量残余不确定性。

## 7.9 P50：接触几何、法向求解与摩擦探针

完整机器人上的轮地接触同时包含几何、质量、闭链和 actuator。该阶段先使用最小接触探针，再回到完整机器人。

### P50-A `[CORE]`：共同球体的无摩擦竖直落体

**配置：** 两个引擎都使用同半径、同质量、同惯量的单刚体球；平面位置和法向一致；摩擦设为 0；从预先固定的三个高度和两个初始竖直速度释放。

**为什么做：** 同时去掉轮 mesh、闭链、关节和 actuator，只比较法向碰撞检测、穿透修正、恢复和时间积分。

**记录：** 首次接触时刻、接触点/法向、最大穿透、法向冲量、反弹速度、settling time、机械能损失。

**能查出：** 两个 solver 的法向接触语义在最简单几何上是否已经差异过大。

**通过后排除：** 排除“任何接触都会在两个引擎中完全不同”这种 gross solver 问题。

**不能排除：** 不能排除 convex mesh、滚动或摩擦差异。

### P50-B `[EXTENDED]`：共同轮形 mesh 的无摩擦竖直落体

**配置：** 两侧都使用由同一轮 mesh/convex hull 来源生成的碰撞几何；质量和惯量仍与 P50-A 相同。

**为什么做：** 比较共同 sphere 与共同 mesh 的增量，区分“solver 本身”与“复杂碰撞几何表示”。

**判定：**

- sphere 通过、mesh 失败：支持碰撞几何/convex 处理差异。
- sphere 和 mesh 都失败：支持法向 contact solver/integration 差异。
- 两者都通过：法向最小探针不是主因，继续摩擦和完整机器人。

Core V1 不生成 common-mesh coupon，因此不能区分“sphere solver 通过但复杂凸几何失败”。该残余候选必须写入 `remaining_uncertainty`。

### P50-C `[CORE]`：共同球体切向滑移

**配置：** 共同 sphere 运行摩擦 0 与冻结 nominal friction 两组，施加 `+vx/0/-vx` 初速度；不施加 actuator。共同 mesh 分支留给 Extended V2。

**为什么做：** Isaac GPU ContactSensor 当前不能直接提供与 MuJoCo 同义的摩擦力，因此使用全系统水平动量变化、滑移距离和速度衰减作为共同可观测量。

**记录：** 水平冲量、速度衰减、滑移距离、法向冲量、接触持续时间和滚动/滑动状态。

**能查出：** friction combine、摩擦锥/锥近似、切向 solver 和材料语义是否导致材料级差异。

**通过后排除：** 排除简单刚体切向摩擦作为主要来源。

### P50-D `[EXTENDED]`：被动滚动与受控轮 spin-up

**配置：** 单轮刚体分别给定初始角速度和小的轴向直接 torque；运行正负方向，比较滚动比、角速度衰减和 base reaction。

**为什么做：** 当前完整系统的最大非线性差异集中在轮通道。该 probe 不包含闭链，可以检查轮形、接触 patch、滚动摩擦和轮轴 inertia 的组合。

### P50 的分类规则

- 最小 contact probe 在共同几何下失败：支持 `CONTACT_OR_FRICTION_MISMATCH`，子因按 normal/tangential/rolling 细分。
- Core 的共同 sphere 通过但完整 formal robot 失败：不能直接支持 geometry mismatch，只能将 complex geometry/contact coupling 保留为候选。
- probe 全通过但完整机器人 ground test 失败：不能排除接触，说明更可能是 contact 与闭链/drive 的耦合，转入 P60。

## 7.10 P60：完整机器人分层回装

该阶段按照“每次只加回一个耦合”的顺序，把前面拆开的组件重新组合。

### P60-A `[EXTENDED]`：地面开启、整机刚性快照、无 actuator

**为什么做：** 将整机名义姿态烘焙为一个 compound rigid body，保留变换后的碰撞几何和按平行轴定理合成的总质量/惯量，但移除关节、闭链运动和控制器。不得用“超大 stiffness”假装锁定，因为那会再次引入不同的 joint solver。

**要观察：** reset penetration、首次接触、base pitch/roll、COM 运动、总接触冲量和静止后的微动。

**能查出：** 完整机器人几何与接触在不含机构运动时是否已经产生不同的 base 响应。

**通过后排除：** 排除静态整机碰撞几何/质量分布作为唯一主因。

**失败含义：** 结合 P50 判断是接触 solver 还是 formal geometry；若 P20 同时失败，则可能是质量/惯量与接触共同问题。刚性快照只作为接触支持证据，生成器的空间惯量必须先通过解析校验，否则该 probe 无效。

Core V1 不实现刚性快照生成器，不能引用 P60-A 作为排除证据。

### P60-B `[CORE]`：地面开启、闭链开启、drive 关闭、零动作

**为什么做：** 在接触上加入可动闭链，但仍移除 policy 和 actuator target。它直接重现当前已经看到的“零动作也分歧”，并用前面 probe 的结果解释分歧。

drive-off 必须复用 P30-A 的零 gain、target 中性化和运行时断言契约；不得仅发送零动作而保留正式 position/velocity drive。

**要观察：** 第一接触时刻、闭链残差、约束能量、轮滑移、总水平/角动量和 base pitch-rate。

**Core 排除逻辑：**

- P60-B 失败且 P10/P40 闭链 probe 同向失败：增加闭链主因支持，但因 P60-A 未执行，仍保留整机 geometry/contact contributor。
- P60-B 失败、P40 通过且 P50-A/C 失败：增加 contact 主因支持，但不能区分复杂整机 geometry。
- P40 与 P50-A/C 都通过而 P60-B 失败：候选仍耦合，输出 contributor 或 `INCONCLUSIVE`，不能强行二选一。

### P60-C `[CORE]`：地面开启、正式 drive、共享第一动作

**为什么做：** 加回 actuator，但仍只执行一个冻结动作，确保第二次 policy 推理没有参与。它与 P30、P40、P50 的结果组合，用于解释第一个 5/20 ms 分歧。

**要观察：** 第一动作 target、每子步 joint/base/closure/contact response、饱和和能量账本。

**排除逻辑：**

- P60-B 已失败：P60-C 只能说明 drive 是否进一步放大，不能把全部问题归 Actor。
- P60-B 通过、P60-C 失败且 P30-B 失败：执行器/积分主导。
- P60-B/P30 均通过而 P60-C 失败：检查仅在接触状态出现的 drive-contact 耦合，输出 multiple/contributor。

### P60-D `[CORE]`：精确 Isaac 动作开环回放

**为什么做：** 使用唯一、预声明来源的 fresh Isaac 闭环生成 clipped float32 动作序列，在两个引擎中开环重放，移除后续 policy feedback。

replay source 在 catalog 中冻结为：

```text
source_scenario_id = P60_D_REPLAY_SOURCE_DREAMWAQ_RUN01
actor = artifacts/phase2_dreamwaq/training-suite-20261007-185711/
        run-01-evaluation/export/actor.ts
actor_sha256 = 4C2C64C2D9F9F88532FCEFF4AE45AA049F970646761D2D0ACE0A29678621D6C7
policy_manifest = same export directory/policy_manifest.json
policy_manifest_sha256 = 322232B78674217579E491FF9B88E1E688DB899A2C5064830F0D3149B0E992C3
profile = NominalEvaluationProfileV1, randomization/noise disabled
engine = Isaac
configuration = formal full robot, ground/closure/formal drive enabled
environment_count = 8
scenario_name = nominal_stand
environment_index = 0
command = [0.0, 0.0, 0.20]
seed = 20261007
repetition_index = 0
horizon = 499 policy action steps at 20 ms
evaluation_contract_hash = 8A3D9CC722B7F947B6C9D0DA8E4B68B90877D40E990FAE8D00495D5E5A7CAD80
reset_cache = artifacts/phase2_dreamwaq/training-suite-20261007-185711/
              run-01-evaluation/isaac/evaluation-reset-cache.pt
reset_cache_file_sha256 = 008EE5D50E1C622D4AC0408E3C5E1C0B8CFCA618ABD3163E221A39D203B3DECE
reset_cache_tensor_sha256 = 446817DB7509AAC24F66D7111391FF9223D50FA8B33469C31CFF9B79CE3DF21D
reset_cache_identity_hash = D87209736F3B070EB3DB3B93879457DB7A26A5098E132F6CD00181826E05A8A4
reset_cache_schema = IsaacEvaluationResetCacheIdentityV1 / ClosedChainResetCacheSchemaV2
reset_cache_algorithms = ClosedChainRootHeightAlignmentV1 /
                         BoundaryClampedPhysXRelaxationV1
selection_rule = generate exactly once; no fallback actor/seed/repetition/scenario
```

source worker 必须在环境构造前验证 §2.2 的 reset-cache 文件与 evaluation summary 哈希，并验证 cache tensor/identity/schema/algorithm/`num_envs=8`；环境构造后必须证明 `FORMAL_SCENARIOS[0] == ("nominal_stand", (0.0,0.0,0.20))`，且 source trace 只读取 cache/env row `0`。随后先保存 actor/manifest/configuration/三个 reset phase/RNG identity，再生成唯一闭环 trace。生成完成后、任何 MuJoCo replay 或 C70 分析启动前，编排器把 `source_trace_sha256`、`clipped_action_sequence_sha256`、实际 action count，以及由上述全部字段、reset-cache identity 和 environment row 共同计算的 `replay_source_identity_hash` 原子写入 run manifest。hash 是生成结果的绑定，不是事后从多个候选中选择；source 提前 termination、hash 不符、cache/env row 不符或长度不是 499 时，P60-D/C70 失效，禁止换用 1-env 新 cache、另一 evaluation cache/env row、Phase 1R、另一 seed 或另一 repetition。

同一动作文件必须先在全新 Isaac reset 上开环回放，并在对应同引擎 repeatability envelope 内复现 source Isaac 轨迹；等价比较覆盖 499 个 active pre-action frame、每步 pre-reset/terminal trace 和预期第 499 step timeout，auto-reset 后返回的下一 episode observation 不进入 replay 轨迹误差。该 fresh-reset replay 等价门失败时，不得进入 MuJoCo replay 或 C70。跨引擎 P60-D 与 C70 只接受同一个 `replay_source_identity_hash`。

**能查出：** 同一动作序列下 plant 是否仍偏离；策略是否只是放大已有误差。

**通过后排除：** 若开环状态在容差内而闭环动作迅速分离，才可以把注意力转向 checkpoint sensitivity。

### P60-E `[EXTENDED]`：预声明几何消融，不做搜索

使用以下固定组合：

1. formal：Isaac convex-hull wheel / MuJoCo sphere proxy。
2. common sphere：两侧使用同尺寸 sphere proxy。
3. common mesh：两侧使用同来源轮 mesh/convex 语义。

对每组运行 P60-A/B/C。若 common geometry 相对 formal 的 `explanation_ratio >= 0.70`，支持 geometry 主因；只改善 `0.30..0.70` 时标记为 contributor。不得增加更多半径或材质点来挑选最优结果。

Core V1 不生成这些整机 geometry variant；P50 的共同球体结果只能识别 gross normal/tangential solver 差异，不能替代 P60-E 的整机 geometry 认证。

## 7.11 C70 `[CORE amplifier-only]`：checkpoint 鲁棒性与策略放大

### 为什么最后才做

checkpoint 只能在输入链和 plant 门有清晰结论后评价。一个策略在另一个错误 plant 上失败，不等于策略本身没有鲁棒性；同样，一个策略把很小的合法 plant 差异放大到倾倒，也确实说明其鲁棒余量不足。

### Core V1 执行内容

1. 只读加载 §2.3 精确冻结路径的 DreamWaQ run-01 和 Phase 1R randomized_v3 run-03 Actor/manifest；Phase 1R 必须来自 `artifacts/phase1_randomized_v3/training-suite-20261007-062147/exports/run-03/`，不得从 `phase1_v4` 或目录扫描结果替代；不重新导出 checkpoint，不重新训练，也不修改正式八场景结果。
2. ActorObsV1 feature groups 固定为下表，名称、slice 和维度必须与两个 policy manifest 完全一致；不允许按结果拆分、合并或丢弃通道：

| feature group | ActorObsV1 slice | dim |
|---|---:|---:|
| `angular_velocity` | `0:3` | 3 |
| `projected_gravity` | `3:6` | 3 |
| `command` | `6:9` | 3 |
| `leg_position_error` | `9:13` | 4 |
| `joint_velocity` | `13:19` | 6 |
| `previous_action` | `19:25` | 6 |

3. 只从上述冻结 `replay_source_identity_hash` 对应的 P60-D Isaac 与 MuJoCo 五帧 ActorObsV1 序列构造共同物理扰动。对每个 policy tick `t in {0,20,40,60,80,100} ms`、group `g` 和 frame `f=0..4`（oldest 到 newest），定义 `e[f,g,t] = IsaacActorObsV1 - MuJoCoActorObsV1`；它已经处于 manifest 冻结的 NormalizationV2 空间。分析必须分别使用 `anchor in {isaac,mujoco}` 两个 baseline：DreamWaQ baseline 是该 anchor 的完整 `5x25` history，Phase 1R baseline 是同一 anchor 序列的最新 25D frame。对每个 anchor，DreamWaQ 主分析把 `s*e[0:5,g,t]` 按 frame-major 加到 baseline history，Phase 1R 把 `s*e[4,g,t]` 加到 baseline current frame，其中 `s in {+1,-1}`。未属于 `g` 的通道保持该 anchor baseline 不变。幅值只能取实测误差，不能搜索使某个 Actor 更差的方向或尺度。
4. 每个 anchor 独立记录 `replay_source_identity_hash`、generator Actor/manifest SHA、source scenario/seed/repetition、baseline engine、P60-D trace SHA256、五帧 observation-sequence SHA256、125D/25D baseline、CENet 输出、raw/clipped action 和饱和。Phase 1R 的 CENet 字段显式 unavailable。比较的是各策略相对同一 anchor 下自身未扰动 baseline 的 action 变化，不把 125D 与 25D 向量直接当作同维输入，也不允许只选择结果更支持某种结论的 anchor。
5. 主分析以 RMS 消除 group 维数和五帧拼接长度的机械影响：DreamWaQ 的 feature denominator 是五帧该 group 全部 normalized elements 的 RMS，Phase 1R 的 denominator 是最新帧该 group normalized elements 的 RMS；两者的 action response 都是六个 raw ActionV1 通道 delta 的 RMS。禁止使用 125D/25D L2 总范数或把 DreamWaQ 的五帧 L2 与 PPO 单帧 L2 直接相比。
6. 每个 anchor 下 DreamWaQ 还必须运行 latest-frame-only sensitivity guard：只对 frame 4 加 `s*e[4,g,t]`，frames 0..3 完全保持该 anchor baseline；Phase 1R 输入不变。guard 中 DreamWaQ 与 Phase 1R 的 feature denominator 都固定为 `rms(e[4,g,t])`，其余 group、tick、action RMS、阈值和 eligibility 不变。若同一 anchor 的 full-history 主分析与 latest-frame-only guard 在 `amplifier` 和 `not_distinguished` 之间翻转，该 anchor 结果为 `inconclusive`，原因 `history_reduction_sensitive`；guard 不得替换主分析或生成新的 checkpoint 排名。
7. 计算 plant 初始误差到 action 差异和饱和的增益：

```text
state_error(t) -> policy_input_error(t) -> action_error(t) -> state_error(t+1)
```

8. 比较 DreamWaQ 与 Phase 1R 对同一物理 feature 扰动的敏感度，不把 run-03 的 `5/8` 写成正式通过。

### Core V1 amplifier 阈值

定义 `rms(x)=sqrt(mean(x^2))`。对每个 eligible physical feature group `g`、扰动符号 `s in {+,-}` 和冻结 tick `t in {0,20,40,60,80,100} ms`，使用 raw action 定义：

```text
feature_rms_DreamWaQ(anchor,g,t) = rms(vec(e[0:5,g,t]))
feature_rms_Phase1R(anchor,g,t)  = rms(e[4,g,t])
action_rms_policy(anchor,g,s,t)  = rms(delta_action_raw_policy(anchor,g,s,t))
G_policy(anchor,g,s) = median_valid_t(action_rms_policy(anchor,g,s,t) /
                                      max(feature_rms_policy(anchor,g,t), 1e-6))
R(anchor,g) = geometric_mean_s(
    G_DreamWaQ(anchor,g,s) / max(G_Phase1R(anchor,g,s), 1e-6))
D(anchor,g) = median_s(
    G_DreamWaQ(anchor,g,s) - G_Phase1R(anchor,g,s))
```

tick 只有在对应 feature RMS `>1e-6`、两策略 baseline/perturbed raw action 均有限、没有 input clipping/schema/normalization failure 时才有效。feature group 只有在该 anchor 的正负两个方向都存在、每个方向至少 3 个有效 tick 时才 eligible。每个 anchor 的 `gain_ratio` 和 `gain_difference` 分别是其 eligible groups 的 median；不得把所有 125D/25D 分量拼接后计算单个 Jacobian norm。两个 anchors 的 full-history 与 latest-frame-only 中间结果、有效 tick mask、trace/observation hashes 和 exclusion reason 都必须写入 evidence。

- 每个 anchor 的 `amplifier`：至少 2 个 eligible feature groups；每组正负两个方向均满足 DreamWaQ gain 更高；该 anchor 的 `gain_ratio >= 1.50` 且 `gain_difference >= 0.25`。阈值必须同时满足，不能二选一。
- 每个 anchor 的 `not_distinguished`：全部 eligible groups 的 `R(anchor,g)` 均在 `[0.80,1.25]` 内，该 anchor 的 `abs(gain_difference) < 0.10`，且 clipped-action saturation fraction 差的绝对值 `<0.05`。
- 每个 anchor 的 `inconclusive`：eligible group 少于 2、符号/feature group 排序不一致、history guard 翻转，或结果落在上述 amplifier 与 tie zone 之间。
- `checkpoint_role=not_evaluated`：底层输入链无效、P60-D 序列无效或 Actor artifact identity 失败。

最终 `checkpoint_role` 只有在 Isaac-anchor 与 MuJoCo-anchor 得到相同的 `amplifier` 或相同的 `not_distinguished` 时才采用该共同结论；任一 anchor 为 inconclusive、两者分类不同或 anchor identity/hash 不完整时，最终强制 `checkpoint_role=inconclusive`，原因 `baseline_anchor_sensitive` 或对应 identity failure。不得对两个 anchor 的 gains 取平均来掩盖分歧。

饱和只在每个 anchor 内作独立佐证：`saturation_fraction` 按 `abs(raw-clipped)>1e-6` 的 channel-tick 比例计算；DreamWaQ 比 Phase 1R 高 `>=0.10`，或首次饱和至少早 1 个 20 ms control tick，记为该 anchor 的 `saturation_amplifier=true`。它可以提高报告置信度，但不能在 raw-gain 条件不满足时单独生成 `amplifier`，也不能跨 anchor 合并。报告还必须输出阈值附近 `[1.25,1.50)` 的敏感性，不得把边界值四舍五入后升级。

### Core V1 允许的结论

- 若底层 plant 门已经失败，且两个 baseline anchors 都满足上述 raw-gain `1.50/0.25` 双阈值与 history guard，可标记 `checkpoint_role=amplifier`；饱和差异只作佐证，不能替代 raw-gain 门。
- 若两个 baseline anchors 都落入上述 tie zone，则标记 `checkpoint_role=not_distinguished`；中间区、anchor 分歧或方向不一致标记 `inconclusive`，不能声称 DreamWaQ 特有问题。
- 即使 Core 的底层门全部通过，也只能输出 `INCONCLUSIVE` 并建议 Extended V2；不得输出 `CHECKPOINT_ROBUSTNESS_FAILURE` 作为 primary。
- Core V1 不重跑完整八场景，不生成 common-mesh/common-sphere 整机策略排名。

### `[EXTENDED]` checkpoint-only 认证门

只有 Extended V2 补齐 P20-D、P40-C、P50-B/D、P60-A/E，且正式模型 P60 前 100 ms 开环 plant 没有材料级分歧，再完成冻结八场景对照后，才允许评估 `CHECKPOINT_ROBUSTNESS_FAILURE` 是否为 primary。该规则本轮只保留为未来门禁，不实施对应代码。

## 8. 自动判定逻辑

## 8.1 先判定证据是否有效

判定器按固定顺序运行，禁止因为结果“符合预期”而跳过前置门：

1. G00 失败：`INVALID_EVIDENCE_PIPELINE`。
2. G01 的对应配置族关键状态不可复现或缺少有效包络：依赖 probe 为 `INCONCLUSIVE`，原因 `nondeterministic_evidence` 或 `repeatability_unavailable`。
3. G02 失败：`SIM2SIM_ADAPTER_BUG`。
4. G03 失败且无中立替代 observer：`INVALID_EVIDENCE_PIPELINE` 或 `INCONCLUSIVE`。
5. P20-S 的 worker 动量读回 golden 失败：`INVALID_EVIDENCE_PIPELINE`，禁止使用动量/角动量证据。
6. 只有 1-5 均允许时才进入物理和 checkpoint 分类。

## 8.2 物理类别的必要证据

| 类别 | Core V1 的 class-specific primary rule | 主要排除/降级条件 |
|---|---|---|
| `MASS_OR_INERTIA_MISMATCH` | P20-S 的冻结 source audit 身份有效，27-body mapping 完整，worker golden 通过，且 source audit 与至少一个引擎 compiled per-body mass/COM/inertia 超过静态容差。该直接身份失败不要求 explanation ratio | P20-S 通过只能排除 compiled static property mismatch；P20-D 未执行，动态有效惯量仍为 residual uncertainty |
| `ACTUATOR_INTEGRATION_MISMATCH` | P30 bypass 全断言通过；对应 open 或 closed direct effort 响应在容差内；同机械配置 formal target 材料失败；冻结 `P30_*_TARGET_TO_DIRECT` ratio `>=0.70`。第二个 P30 配对或 P60-C 同向结果用于提高置信度 | direct effort 失败、motor/PhysX bypass 无效、饱和或配对 identity 不同，均不得只归 actuator |
| `CLOSED_CHAIN_CONSTRAINT_MISMATCH` | `P40_CLOSURE_ON_TO_OFF` ratio `>=0.70`，或 P10/P30/P40 中至少两个独立 closure-on/off probe 同向失败；全部配对必须无 contact 且 P20-S 无先行静态 hard failure | 仅完整机器人有地面场景失败时只能 contributor/INCONCLUSIVE；P20-D 未执行必须列动态惯量残余不确定性 |
| `CONTACT_OR_FRICTION_MISMATCH:normal` | P50-A common-sphere identity/property audit 通过，pre-contact freefall 通过，两引擎均发生有效接触，且至少两个预声明 height/velocity 初态的 post-contact 主指标持续材料失败。该直接隔离失败不要求 explanation ratio | 只有一个初态失败、接触字段 unavailable、pre-contact 已失败或 full robot 才失败时，最高 contributor/INCONCLUSIVE |
| `CONTACT_OR_FRICTION_MISMATCH:tangential` | P50-A 法向门通过或法向冲量误差在容差内；P50-C nominal friction 材料失败；冻结 `P50_NOMINAL_TO_ZERO_FRICTION` ratio `>=0.70` | friction=0 仍同样失败、法向门失败或 contact event 不一致时，不得认证纯切向摩擦 primary |
| `CHECKPOINT_ROBUSTNESS_FAILURE` | Core V1 不可输出 primary，只记录 `checkpoint_role=amplifier|not_distinguished|inconclusive|not_evaluated` | 需要 Extended V2 全部门和正式八场景重评 |

## 8.3 主因、贡献因素和多重问题

每个候选有四种状态：

- `supported_primary`：满足 §8.2 对该类别冻结的 primary rule，且不存在使该证据无效的更早 hard failure。不得用其他类别的 ratio 代替。
- `supported_contributor`：有独立材料证据，但只满足该类别部分规则、其合法消融 ratio 为 `[0.30,0.70)`，或只在耦合场景成立。
- `not_supported`：对应隔离 probe 未超过有效容差，或该类别合法消融 ratio `<0.30`。
- `inconclusive`：probe/observer/repeatability unavailable、证据互相矛盾、落在未定义中间区，或 Core 缺少认证该候选所需的 Extended probe。

最终规则：

1. 恰好一个 `supported_primary`：输出该类别。
2. 两个或以上独立 `supported_primary`：输出 `MULTIPLE_PHYSICS_MISMATCHES`，并排序列出类别。
3. 没有 primary，但有 contributor：输出 `INCONCLUSIVE`，列出候选和缺失的判别实验。
4. Core 全部已执行物理候选 not-supported，但仍有未执行 Extended 候选：输出 `INCONCLUSIVE`，原因 `extended_scope_required`；不得升级 checkpoint-only 根因。
5. Core 全部门都通过且未观察到原问题：仍输出 `INCONCLUSIVE`，子状态 `CORE_NO_MATERIAL_DIVERGENCE`，说明本轮核心矩阵未定位根因，不得声称问题已永久解决。

## 8.4 置信度

- `high`：满足该类别的 class-specific primary rule，并有第二条独立证据。ratio 类别要求至少一个合法 ratio `>=0.70`；直接身份类别要求两个独立读回/初态条件同向失败。动态 probe 还必须包含正负激励，且重复性和插桩门通过。
- `medium`：单条直接身份失败，或一个隔离 probe 加一个完整机器人 probe 支持，或合法 ratio 在 `0.30..0.70`。
- `low`：只存在相关性、只能依赖不可直接观测量或候选仍耦合。低置信度不得作为唯一 primary；应落入 `INCONCLUSIVE`。

## 8.5 已知证据如何进入而不污染新 run

已有报告和 quick experiment 只写入 `historical_evidence`，不会替代当前 run 的硬门。新 run 的 verdict 必须只由其自身完整、hash-verified 的 stage 计算。历史数据只能用于：

- 解释为什么设计包含某个 probe。
- 检查新结果是否复现既有趋势。
- 新结果冲突时触发显式 discrepancy，而不是自动选择旧结果。

## 9. 报告与产物

## 9.1 `verdict.json`

至少包含：

```json
{
  "schema_version": "RootCauseVerdictV1",
  "scope": "core_v1",
  "run_id": "...",
  "run_valid": true,
  "conclusion_status": "identified|multiple|inconclusive|invalid",
  "primary_classification": "...",
  "confidence": "high|medium|low",
  "supported_primary": [],
  "supported_contributors": [],
  "excluded_candidates": [],
  "inconclusive_candidates": [],
  "first_material_divergence": {},
  "failed_gates": [],
  "executed_core_probes": [],
  "unexecuted_extended_probes": [],
  "checkpoint_role": "amplifier|not_distinguished|inconclusive|not_evaluated",
  "evidence_refs": [],
  "remaining_uncertainty": []
}
```

每个 evidence ref 必须指向一个 stage、具体文件、信号、时间点、阈值、实测值和 hash。

## 9.2 `report.md`

报告按以下顺序写：

1. 一句话 verdict 和置信度。
2. run 是否有效，正式文件前后 hash 是否一致。
3. 首个材料分歧发生在哪一层、哪一时刻。
4. 主因的正向证据。
5. 每个被排除候选及其排除实验。
6. contributor 与策略放大关系。
7. 阈值敏感性和重复性包络。
8. 仍不可观测或不可唯一归因的部分。
9. 下一步允许修改哪个正式契约；在报告阶段不实际修改。

报告必须同时包含“这个实验发现了什么”和“它排除了什么”。只列误差表而没有推理链视为不合格。

## 9.3 `evidence.csv`

每行是一条可审计论据：

```text
stage,probe,engine_pair,signal,time_window_ms,scalar_metric,
channel_reduction,repeatability_family,baseline_value,ablation_value,
scenario_id,configuration_hash,configuration_semantics_hash,
pre_forward_initial_condition_hash,post_forward_state_hash,
reset_returned_policy_hash,excitation_hash,comparison_profile_hash,
baseline_scenario_id,baseline_configuration_hash,baseline_configuration_semantics_hash,
baseline_pre_forward_initial_condition_hash,baseline_post_forward_state_hash,
baseline_reset_returned_policy_hash,baseline_excitation_hash,
ablation_scenario_id,ablation_configuration_hash,ablation_configuration_semantics_hash,
ablation_pre_forward_initial_condition_hash,ablation_post_forward_state_hash,
ablation_reset_returned_policy_hash,ablation_excitation_hash,
semantic_diff_paths_sha256,allowed_ablation_factors,
baseline_engine,baseline_trace_sha256,baseline_observation_sha256,
replay_source_identity_hash,replay_generator_actor_sha256,
replay_generator_manifest_sha256,replay_source_scenario_id,replay_source_seed,
replay_source_repetition_index,replay_environment_count,replay_environment_index,
replay_environment_scenario_name,replay_evaluation_contract_hash,
replay_reset_cache_file_sha256,replay_reset_cache_tensor_sha256,
replay_reset_cache_identity_hash,replay_reset_cache_schema,
replay_reset_cache_root_height_algorithm,replay_reset_cache_relaxation_algorithm,
replay_action_sequence_sha256,
expected_improvement_direction,guard_status,
effective_tolerance,explanation_ratio,availability,status,supports,
excludes,artifact_sha256
```

不适用字段写空字符串，不得复用含义。C70 每个 anchor、feature group、sign 和 reduction mode 各有独立 evidence 行；ratio 行必须同时填充 before/after identity 与 semantic diff fields。

## 9.4 原始 trace

原始 trace 使用 NPZ 加 JSON sidecar。所有字段必须有单位、坐标系、采样相位、参考点、frame、availability 和 unavailable 原因。分析器禁止读取未通过 sidecar hash 校验的 NPZ，也禁止把 `-1`、`NaN` 或空数组自行猜测成 unavailable；availability 必须由 worker 显式声明并通过 schema 校验。

## 10. 错误处理、恢复和可重复执行

### 10.1 fail-closed

以下任一情况立即使 stage 无效：

- 输入或输出 hash 不匹配。
- 模型副本的变换清单与实际 XML/USD 不一致。
- shape、dtype、单位、joint/body 名称或时钟不一致。
- 出现未声明 NaN/Inf。
- terminal 状态与 reset 后观测混写。
- worker 退出但没有完整 `stage_status.json`。
- 正式文件前后 hash 不同。
- 写入路径逃出 suite root。

### 10.2 resume

`--resume` 的规则：

1. run manifest、源 hash、套件代码 hash 和场景定义 hash 必须与原 run 一致。
2. 只复用状态为 `complete` 且文件 hash 全部通过的 stage。
3. `running`、`failed`、缺文件或 hash 不一致的 stage 从头执行。
4. 已完成 stage 永不原地覆盖；强制重跑写入新 attempt 目录，并更新显式选择记录。

### 10.3 exit code

| code | 含义 |
|---:|---|
| `0` | 完整 verdict 已生成；verdict 可以是某种 failure 分类 |
| `2` | CLI/配置错误 |
| `3` | 正式输入身份漂移 |
| `4` | worker 或 stage 不完整 |
| `5` | schema/hash/分析内部矛盾 |

“发现机器人有物理 mismatch”不是程序失败，因此仍使用 exit code 0；“证据不完整却生成结论”才是程序失败。

## 11. Debug 模型副本的生成规则

### 11.1 USD

- 不直接编辑正式 AssetBundleV2。
- 每个 run 在 `models/isaac/` 下创建完整副本或只读引用加 debug override layer；所有可变 opinion 必须落在 suite 内的新 layer。
- override layer 必须记录 source layer hash、被覆盖的 prim/property、旧值、新值和理由。
- 关闭地面、重力、drive、碰撞或闭链必须使用彼此独立的 variant，不允许一个隐含修改多个因素。
- closure-off 必须保留四个冻结 loop-joint prim 及其 endpoint/solver 属性，只把 `physics:jointEnabled` 设为 false：`/wheel_leg_urdf4/jIO/jIO_loop_closure`、`/wheel_leg_urdf4/jKN/jKN_loop_closure`、`/wheel_leg_urdf4/jEC/jAG_loop_closure`、`/wheel_leg_urdf4/jCF/jCF_revolute_joint`。删除 prim、改变 endpoint 或 solver 属性均不是 Core 的合法 `closure_mode` 消融。
- 生成后重新打开 composed stage，读取最终解析值并与变换清单比较，不能只相信写入调用成功。
- drive-off variant 除 USD/PhysX resolved 属性外，还必须来自独立的 debug actuator config clone；正式 Python config 对象的深层 hash 在创建前后必须一致。

### 11.2 MJCF

- 从正式 XML 解析为结构化树后写入 `models/mujoco/`，禁止正则替换 XML 文本。
- mesh 路径使用明确的只读 source hash 或复制到 run 模型目录；任何复制都写 hash。
- 每个 variant 必须带唯一 model name、variant schema 和源 XML hash。
- 编译后记录 `nbody/njnt/nq/nv/nu/neq/npair`、质量、惯量和 actuator/constraint 表，验证变换只影响声明字段。
- closure-off 必须保留八个冻结 `<connect>` equality 的名称、site endpoints、solref/solimp 和 `neq=8`，只把 compiled `eq_active0` 设为 false；不得删除 equality。
- drive-off variant 必须按 P30 冻结的 26 个 robot hinge 名称结构化清零全部 joint damping，但保留 `<motor>` 的 fixed gain `1`、bias `0` 和 gear `1`；“drive off”在 MuJoCo 侧指 worker 不调用外部 `MixedActionController.compute_torque()`，直接写入经过 canonical/native sign 转换的 effort。必须记录编译后的完整 name-to-DOF map、`gaintype/gainprm/biasprm/gear/dof_damping`，并拒绝任一 passive damping 非零、motor gain 为零、遗漏右轮符号或仍调用正式 PD 的实现。

### 11.3 `[EXTENDED]` 刚性快照

- 在名义姿态把每个 link 的碰撞几何变换到 base frame。
- 使用 link mass、COM 和 inertia 计算 compound 总质量、总 COM 和关于总 COM 的空间惯量。
- Isaac 和 MuJoCo 两侧从同一中间 JSON 生成模型，而不是各自独立重算。
- 解析验证总质量与 `4.396253988146782 kg` 的已审计值一致到浮点容差；总 COM/惯量两侧输入逐项一致。
- 若某个碰撞几何无法等价表示，P60-A 标记 unavailable，不允许以近似结果形成高置信度结论。

本节不属于 Core V1 实现范围。Core 代码中不得出现刚性快照生成入口或未使用的占位实现。

### 11.4 变体不是修复

所有 variant 都是 diagnosis-only。即使某个 variant 让策略跑满，也只能说明某个因素具有解释力；正式模型是否修改必须另立设计、版本、hash、回归和重新评估流程。

## 12. 测试与验收

## 12.1 纯 Python 单元测试

必须覆盖：

- 路径守卫拒绝 `..`、符号链接/junction 逃逸和正式目录写入。
- official bootstrap 必须在第一个 Python import 之前设置 cache 环境并用 `python.exe -B` 启动父进程；在临时复制或预先确认不存在 bytecode 的包层级执行 bootstrap smoke 后，`debug/__pycache__`、`debug/sim2sim/__pycache__`、suite 外 `.pyc`、默认 `.pytest_cache` 和默认 pytest temp 均不得新建。内部模块入口在 `sys.dont_write_bytecode=False`、缺少 bootstrap marker 或任一 child argv 缺 `-B` 时必须拒绝运行；故意在 suite root 外生成项目可控 cache 时 G00 必须失败，既有 cache 未变化时不得误报。
- `RootCauseSuiteAppLauncher` 单元测试必须用 fake/stub `SimulationApp` 在构造边界捕获 launch config，证明 `enable_crashreporter=False` 在构造之前已经成立；source-hash/AST contract 必须证明冻结 `SimulationApp` 仍只在该配置为真时加入 `carb.crashreporter-*`。pre-launch config test 必须验证冻结 `kit-core.json` hash、精确 setting key 和默认值。Isaac worker 缺少 `--portable-root`、缺少 `--/rtx-transient/resourcemanager/localTextureCachePath`、任一路径非绝对/逃出当前 run、texture cache 仍是 `${omni_global_cache}/texturecache`、resolved texture cache 改到 run 内其他 sibling 目录、pre-instantiation flag 非 false、resolved log/data/general-cache/config/dump 任一路径逃逸、resolved crash setting 非 false，或 loaded plugin list 出现 crash reporter 时必须在场景创建前失败；项目 `.venv/.../isaacsim/kit/{logs,data,cache}` 任一文件变化必须使 G00 失败。
- `PythonWriteGuard` 必须在第三方 import 前安装并覆盖 pre、created-lifetime、post 三组 handler，以及 §3.3 冻结的全部 Python audit 事件、write/mutation flag mask、整数 fd 策略、安装 nonce 自证和 capability manifest。测试必须证明合法 FileHandler 恰好调用原始构造器一次且 path/mode/encoding/delay 和写入字节不变；suite 外普通写入在 `open` 前失败；在 `SimulationContext` 前注入 suite 外 `logging.FileHandler(..., delay=True)` 时必须在原始构造器/文件打开前失败且目标文件不存在；构造后新增逃逸 handler 或普通写入也必须失败。Windows 上若存在 `os.O_TEMPORARY`，必须预先创建 suite 内与 suite 外两个现有文件，然后分别以仅 `os.O_RDONLY | os.O_TEMPORARY`、不含 `O_CREAT/O_TRUNC/O_APPEND/O_WRONLY/O_RDWR` 的 flags 调用 `os.open`：suite 内 case 必须由原始调用成功并在 close 后体现 delete-on-close，suite 外 case 必须由守卫在原始调用前拒绝，且文件仍存在、内容与 metadata 未变；不得用不存在目标触发的底层 `FileNotFoundError` 充当通过。不存在 `O_TMPFILE`、`O_TEMPORARY`、`os.chown` 或其他可选 API 时，测试必须断言 capability manifest 精确记录 `unavailable_on_platform`，不得静默 skip 或伪造通过。`os.fdopen` 必须覆盖 suite 内文件 fd、suite 外文件 fd、标准流/登记 pipe 和无法解析 fd；只有前者与明确白名单可通过。测试必须预装一个在 `sys.addaudithook` 事件抛出 `RuntimeError` 的 blocker，证明 nonce probe 检测到 hook 未安装并在第三方 import 前失败；正常安装必须精确记录一次 nonce。每一种实际可用的 mkdir/remove/rmdir/rename-or-replace/link/symlink/truncate/chmod/chown/utime 事件至少有一个 suite 内正例和一个 suite 外拒绝例；只有相应 API 出现在 `os.supports_dir_fd` 时才执行相对 path + `dir_fd` 可解析正例和不可解析 fail-closed 例，否则必须断言 capability manifest 对该 API 记录 `dir_fd=unavailable_on_platform`。manifest 缺任一组、包装器未在 `finally` 恢复、audit 中出现 suite 外允许变更均使 G00 失败。
- Isaac Lab logger source-contract test 必须验证冻结 `direct_rl_env.py` 的 import origin、`self.cfg=cfg`、`SimulationContext.instance() is None` guard、精确 `SimulationContext(self.cfg.sim)` 调用和 existing-context error；把调用改成 `SimulationContext()`、预建 context 或绑定到错误 source origin 的 mutation 必须在环境构造前失败。还必须验证冻结 `simulation_context.py` 把 `cfg.logging_level/save_logs_to_file/log_dir` 原样传给 `configure_logging`，冻结 `utils/logger.py` 只在 `log_dir is None` 时回退 `%TEMP%/isaaclab/logs`。debug config 必须 clone 后设置 `save_logs_to_file=True` 和 exact absolute `<run>/runtime_cache/isaaclab/logs`，不得修改正式 config；`None`、`False`、相对路径、run 内错误 sibling、suite 外路径、没有 `FileHandler` 或任一 handler 父目录不精确相等时必须失败。
- 正式 hash before/after 快照与漂移检测。
- stage DAG、依赖阻断、resume 和原子完成语义。
- scenario catalog 拒绝 Extended-only probe，并要求每个 Core probe 声明配置族、完整 configuration/excitation semantics、精确 configuration/semantics/三个 reset phase/excitation hash、重复性族、主信号、时间窗、scalar metric、channel reduction 和 guard metrics；ratio probe 还必须声明 comparison profile、pair identity、allowed factors 和 expected direction。
- trace shape、dtype、单位、坐标、时间和 unavailable 字段校验。
- `unexpected_contact=-1` 输入必须规范化为 `value=null, availability=false`，且 verdict 不能把 unavailable 当作 false。
- reset 三相、terminal/pre-reset 与 returned reset observation 分离。
- canonical/native 顺序和右轮符号 round-trip。
- history 五帧复制、frame-major 和 previous-action 时序。
- 正负激励的 odd/even response。
- per-body mass/COM/inertia frame 转换、27-body mapping、复合空间惯量、动量、角动量、能量和 impulse 账本的合成解析例；错误 COM/link velocity、反向 inertia rotation、遗漏平行移项的 mutation 必须失败。
- material threshold、按配置族 repeat envelope、orientation geodesic 和 explanation ratio 边界。
- repeatability binding 必须同时匹配 configuration、三个 reset phase 和 excitation hash；把 P30 closure-on direct 绑定到 open/target 包络、把 P60 first-action 绑定到 replay 包络等错误映射必须失败。单独改变 pre-forward reset height、root velocity、任一 joint q/qd、command、seed/RNG state 或 reset artifact hash 时必须产生 cache miss；只改变 post-forward 或 returned-policy phase 也必须产生同场景 repeatability cache miss。P50-A 的六个初态不得共享 envelope。
- mismatch 聚合必须使用 catalog 预声明 metric；故意交换 RMSE/max-abs 后 verdict 必须拒绝。
- ratio pair 必须来自冻结 allow-list，并携带 baseline/ablation scenario identity、configuration/semantics/三个 reset phase/excitation/comparison-profile hash、逐引擎 semantic diff paths 和精确 `allowed_ablation_factors` 集合；任何 pair 的 pre-forward hash 不同必须失败；P30/P50 post-forward 或 returned-policy guard 不同必须失败；P40 closure on/off 的 pre-forward 相同但 post-forward 不同必须被接受并作为结果，不能误判为 pair 污染。遗漏 P30 的 `input_kind`、额外改变未允许 path、合法 factor 外附加 armature/body-mass/contact-solver 变化、改变任一 MuJoCo `MjOption` 字段、改变任一递归 PhysX/SimulationCfg 物理字段、P30 初始 canonical torque 不匹配、transform manifest 与 compiled diff 不对应，或对 P60-B/C/D 伪造 explanation ratio 必须失败。
- verdict 决策表的每个类别、multiple、inconclusive 和 invalid pipeline。
- P60-D replay source 必须精确绑定 DreamWaQ run-01 Actor/manifest、预声明 scenario/profile/command/seed/repetition/horizon、`environment_count=8`、`nominal_stand/environment_index=0`，以及冻结 evaluation reset-cache 的 file/tensor/identity/schema/algorithm hashes；并在任何 replay/C70 前原子冻结 source trace、action sequence 和 `replay_source_identity_hash`。替换为 1-env cache、另一 cache/env row、Phase 1R、另一 seed/repetition、非 499 长度、事后从多个候选选取、fresh-reset Isaac replay 不等价，或 C70 引用不同 replay identity 时必须失败。
- C70 六个冻结 ActorObsV1 feature groups、Isaac/MuJoCo 双 baseline anchors、full five-frame 与 latest-frame-only reduction、六通道 action RMS、正负号与固定 tick 聚合、`1.50/0.25` amplifier 双阈值、`[0.80,1.25]` tie zone、`0.10/1 tick` saturation 佐证、history-reduction-sensitive、baseline-anchor-sensitive 和中间区 `inconclusive` 边界；交换/篡改 anchor trace、observation hash 或 replay source identity 必须失败。
- 人工构造的相互矛盾证据必须拒绝生成高置信度主因。
- JSON/NPZ sidecar hash 被篡改时 fail-closed。

## 12.2 MuJoCo 集成测试

- 正式模型只读加载，所有 debug variant 均可独立编译。
- 每个 variant 的编译结构和变换 allow-list 符合预期。
- compiled `MjOption` 全字段快照必须与 §6.4 字段集合完全相等；改变 `integrator`、`noslip_iterations`、`ccd_tolerance`、`enableflags` 或 `disableflags` 的 mutation 必须改变 semantics hash，并在 ratio pair 中被拒绝。
- 1-tick/20-ms collector smoke。
- compiled per-body property dump 与 source audit；单刚体 `P=m v`、`H=I omega` runtime golden。
- no-ground、zero-g、direct-effort、closure-off 和共同 sphere contact coupon smoke。
- drive-off 编译后的 motor fixed gain `1`、bias `0`、gear `1`；冻结 26 个 robot hinge 名称必须与 compiled hinge set 完全相等且 DOF address 唯一，26 项 damping 全为零，`base_free` 不在集合中。已知 canonical effort 经 `[+1,+1,+1,+1,+1,-1]` 映射到 native `data.ctrl`，再由 `actuator_force` 和 `qfrc_actuator` round-trip 回 canonical，误差 `<=1e-12 Nm`。任一 passive damping 保留 `0.05`、motor gain 置零或清零 `base_free` 的 mutation 必须失败。
- closure-off 仍须 `neq=8`、八个 connect 名称/endpoints/solver fields 不变且仅 `eq_active0=false`；删除 equality 的 mutation 必须失败。
- 正负 effort/force/torque 确实施加在声明轴和时段。
- 正式 adapter 路径与 debug collector 的共同字段一致。
- 测试前后正式 XML、manifest 和 runtime hash 不变。

## 12.3 Isaac Lab 集成测试

- 每个不可变配置族使用全新 Kit 进程；同族 case 复用进程时，每个 case 必须通过 fresh-reset 身份核验。
- 每个 Kit 进程使用当前 run 内独立 absolute portable root，并显式把 RTX `localTextureCachePath` 指向 `<absolute-kit-root>/cache/texturecache`；`SimulationApp` 构造前的 launch config 必须为 `enable_crashreporter=False`，启动后的 resolved logs/data/general-cache/user-config/crash-dump 路径全部位于其中，texture cache requested/resolved/expected 三方 canonical path 必须精确相等，resolved crash setting 为 false，且 `carb.get_framework().get_plugins()` 不含 crash reporter plugin。真实 1-tick worker 必须保存这些证据；省略 portable root、省略/篡改 texture-cache setting、让它解析到用户 `omni_global_cache`、改到 run 内 sibling、绕过 suite launcher 或伪造 pre-load flag 时必须被 G00 拒绝，并验证 `kit-core.json` 与项目 `.venv/.../isaacsim/kit/{logs,data,cache}` 前后不变。
- 同一真实 1-tick worker 必须在任何 Isaac import 前安装 `PythonWriteGuard`，保存 pre-handler 快照；环境构造前把 cloned `cfg.sim.log_dir` 设为 `<run>/runtime_cache/isaaclab/logs`、保持 `save_logs_to_file=True`，验证 `direct_rl_env.py` import/call contract 并断言 `SimulationContext.instance() is None`；构造后保存 `pre_handlers`、`created_lifetime_handlers`、`post_handlers` 与完整 Python audit ledger。所有受审计内容写入和路径 mutation 必须位于当前 run，Isaac Lab handler canonical parent 必须与 requested/expected 目录精确相等。默认 `None` 导致 `%TEMP%`、相对路径、错误 sibling、逃逸路径、预建 context、错误 import origin、额外 file handler 或 suite 外 mkdir/rename/remove 的 mutation 必须被 G00 拒绝，正式 `WHEELLEG_CFG` 前后必须不变。
- 递归序列化的 `SimulationCfg`/`PhysxCfg` 字段集合必须与冻结源码 introspection 完全一致，并与 `/physicsScene` 和 articulation resolved 值相互校验；改变 `solver_type`、`enable_ccd`、`enable_stabilization`、`bounce_threshold_velocity` 或 `friction_offset_threshold` 的 mutation 必须改变 semantics hash，并在 ratio pair 中被拒绝。
- 正式 step 与 debug collector 的 5-run observer equivalence。
- debug config/override 不改变正式 `WHEELLEG_CFG` 对象和值。
- reset 三相和五帧 history golden vector。
- ContactSensor 开/关等价门；不通过时 contact force 字段不得参与 verdict。
- compiled per-body property dump 与 source audit；单刚体 `P=m v`、`H=R I R^T omega` runtime golden。
- no-ground、zero-g、drive-off、direct-effort、closure-off 和共同 sphere contact coupon smoke。
- drive-off 同时验证 debug config gains、actuator object 缓存 gains、ArticulationData resolved gains 和 PhysX view gains 全零；已知 effort 下无饱和的 host estimate 与输入误差 `<=1e-6 Nm`。
- closure-off 保留四个冻结 loop-joint prim/endpoints/solver fields，只把 resolved joint-enabled 设为 false；删除 prim 的 mutation 必须失败。
- 施加 base wrench 和 joint effort 的坐标/符号 golden vector。
- 运行结束后正式 env、env_cfg 和 AssetBundle hash 不变。

## 12.4 跨引擎端到端测试

1. 只通过 official `run_suite.ps1` 使用最小场景集合运行主命令，必须生成完整 run manifest、stage 状态和 report；manifest 中父/子 argv 都包含 `-B`，且 suite root 外不得出现新 bytecode/cache。
2. 中断一个 stage 后 `--resume`，已完成 stage hash 不变，未完成 stage 重新执行。
3. 故意篡改一个 trace，`verify` 必须失败，`report` 不得生成新 verdict。
4. 使用 synthetic traces 覆盖所有 verdict 分支，验证 decision table 没有顺序依赖。
5. 使用真实 1-tick 数据通过 G00-G03 和 P20-S，再允许 Core 物理矩阵。
6. 运行后扫描正式文件 hash 和修改时间；hash 必须全部相同。
7. 同一配置族中连续运行两个 case，第二个 case 的 reset snapshot、时钟和 state 必须与独立进程运行等价；故意注入残留时必须触发进程重启。
8. Core run 中故意加入一个 Extended-only scenario，G00 必须拒绝执行。

## 12.5 代码复核门

实现完成后、运行真实 Core 诊断矩阵之前，调用一个独立 agent 做只读代码复核。复核重点：

- 是否有任何写入逃出 debug suite。
- variant 是否一次只改一个声明因素。
- collector 时序、reset、terminal 和坐标是否正确。
- direct effort 是否真正绕过 drive。
- compiled property frame 和动量读回 golden 是否足以检测 reference-point 错误。
- 每个 verdict 使用的 metric 和 repeat envelope 是否与 catalog 预声明一致。
- unavailable 非轮接触是否被错误解释为“无接触”。
- verdict 是否可能把 contributor 错写为唯一根因。
- tests 是否能检测错误实现，而不只是覆盖 happy path。

必须清除所有 P0/P1 后才能运行完整诊断；代码变化后重新跑相关测试和复核。

## 13. 实施、复核与执行顺序

本轮顺序固定，不允许跳过复核门：

1. 更新并冻结本设计；生成 SHA256 和逐条首轮复核处置表。
2. 调用独立 agent 只读复核文档。P0/P1 必须清零；P2 必须修复或在文档中给出可审计的不采纳理由。
3. 编写 Core V1 逐文件实施计划并做本地自检，不扩展 §4.3 范围。
4. 先写纯 Python 失败测试，再实现 `contracts.py`、`workspace_guard.py`、`trace_contract.py`、`compiled_properties.py`、`analysis.py`、`verdict.py` 和 `report.py`。
5. 实现 `scenario_catalog.py`、`orchestrator.py`、fake worker 和 resume/范围端到端测试。
6. 实现 Core MJCF variants、MuJoCo worker 和集成测试。
7. 实现 Core USD/config variants、Isaac worker 和集成测试；不实现 rigid snapshot/common mesh。
8. 完整运行静态/单元/合成/worker smoke 测试。
9. 调用独立 agent 做只读代码复核。P0/P1 清零后才能运行真实诊断；修复后重跑受影响测试。
10. M1：运行 G00-G03、P10、P20-S，检查证据链、静态物性和读回 golden。
11. M2：运行 P30、P40、P50-A/C；只有依赖门有效的 probe 才进入判定。
12. M3：运行 P60-B/C/D 和 C70 amplifier-only；生成自动 verdict 与全部原始证据引用。
13. 对完整测试输出、trace、threshold snapshot、verdict 和 report 调用独立 agent 做结果复核。
14. 修正报告层错误时重建 report；若需修改采集或分析代码，则返回步骤 8，并重新进行代码与结果复核。
15. 最后核验正式文件 before/after hash，输出根因结论；若证据不足，明确输出“本轮没有找到可认证根因”和缺失的 Extended probe。

这套顺序先通过 synthetic evidence 锁住分析器，再连接昂贵仿真器；三次独立复核分别检查规格、实现和证据解释，不能互相替代。

## 14. 必须保持不变的现有语义

以下正式语义在整个套件中只读：

- ActionV1 六维顺序、`[-1,1]` clip、腿 `0.35 rad` 和轮 `25 rad/s` 缩放。
- 右轮 native sign `-1` 映射。
- ActorObsV1 25D、DreamWaQ 5x25 frame-major history 和 reset 五帧复制。
- DreamWaQ CENet/Actor 数学契约和 TorchScript Actor。
- ControlFrameV1、NormalizationV2、CommandV1。
- Isaac `0.005 s x 4`、正式 MuJoCo `0.001 s x 20` 和共同 20 ms policy clock。
- 四腿 position drive、两轮 velocity drive、正式 Kp/Kd、effort/velocity limits。
- Phase 1R 随机化配置和 RNG 语义；诊断场景中关闭随机化不等于修改配置。
- USD AssetBundleV2、WheelOnlyCollisionV2 和正式地面。
- 正式 MuJoCo XML 的 sphere proxy、8 个 connect、solver/contact 参数。
- 奖励、phi0、base height、termination 和八场景 evaluator。
- DreamWaQ run-01 与 Phase 1R run-03 的 Actor/checkpoint 文件。

debug variant 对其中某项的临时关闭或替换必须显式标记 `diagnosis_only`，且只能存在于 suite run 目录。

## 15. 设计的已知限制

1. PhysX implicit drive 内部实际 torque 不可直接读取，因此 actuator 归因依赖“direct effort 通过、target drive 失败”的响应证据，而不是 torque 逐点恒等。
2. 两个求解器不会长期轨迹逐点一致；本设计关心共同短时刻、响应方向、材料阈值和消融解释率。
3. contact solver 与闭链 solver 在完整系统中可能不可完全分离；此时正确输出是 multiple 或 inconclusive。
4. Core V1 不执行六轴动态惯量辨识、common mesh、rolling、rigid snapshot 或整机 geometry matrix，因此不能把这些候选写成已排除。
5. 该套件可以定位 simulator/adapter/checkpoint 层级，但不能直接证明哪一个仿真器更接近实车。后者需要实机辨识数据。
6. 该套件不替代后续 sim-to-real 参数标定，也不会自动决定正式模型应如何修改。
7. Core V1 没有通过 G03 的全 body contact observer，非轮接触默认为 unavailable；这会限制倾倒后接触归因的置信度。
8. Core V1 只能识别 checkpoint 对已观测 plant error 的放大作用，不能认证 checkpoint-only 根因。

## 16. 本设计完成标准

实现后的套件只有同时满足以下条件才算完成：

1. 单一命令可以从 fresh run 执行到最终 report，也可以可靠 resume。
2. 所有项目可控的新文件都在 suite root 内。
3. 正式文件 before/after hash 完全一致。
4. G00-G03 能阻断无效证据。
5. 所有 `[CORE]` probe 都在报告中明确“发现”“排除”和“本 probe 不能排除”。
6. verdict 能区分 adapter、compiled static property、actuator/integration、closure、sphere normal/tangential contact、checkpoint amplifier、multiple 和 inconclusive，并明确列出未执行的 Extended 候选。
7. 任一高置信度物理结论必须满足 §8.2 的类别专属规则并有第二条独立证据；ratio 类别至少一个合法消融证据，直接身份类别至少两个独立读回或初态条件，不得强迫所有类别使用 explanation ratio。
8. DreamWaQ 与 Phase 1R 对照只用于鲁棒性解释，不被写成新的正式排名。
9. 独立文档复核和独立代码复核均无 P0/P1。
10. 独立结果复核确认 verdict 没有越过可用证据；若无根因，报告明确写出本轮未定位。
11. 最终报告可从 hash-verified 原始数据完整重建。

## 17. 独立复核意见处置

### 17.1 首轮复核

| 复核项 | Core V1 处置 |
|---|---|
| P1 direct-effort 未真正关闭 drive | 已采纳并强化：环境创建前 clone config；Isaac 三层 gain 置零与 target 中性化；MuJoCo 绕过外部 PD、joint damping 置零但 motor fixed gain 保持 1；失败即 `invalid_drive_bypass` |
| P2 静态 mass/COM/inertia 与动量读回缺 golden | 静态编译物性审计和单刚体读回 golden 纳入 P20-S Core；六轴动态辨识留在 Extended |
| P2 zero-action 重复性不能覆盖动态工况 | 改为 16 个精确 configuration/excitation family；同族 fresh reset 复用进程，禁止跨 hash 借用包络 |
| P2 explanation ratio 聚合未冻结 | scenario catalog 必须声明主信号、窗口、metric、通道聚合、scale 和 guard；逐信号 ratio 同时输出 |
| P2 缺 base orientation geodesic 阈值 | 增加 `8.726646e-3 rad (0.5 deg)` 动态材料阈值 |
| P3 每个 case 冷启动 Kit 成本过高 | 改为每个不可变配置族一个进程，同族多 case fresh reset；跨配置仍隔离 |
| P3 非轮接触占位语义不明确 | Core 将其规范化为 unavailable，不新增全 body observer，并限制相关 verdict 置信度 |

### 17.2 第二轮复核

第二轮独立 agent 结论为 `BLOCKED`，`P0=0, P1=4, P2=2`。本次 V1.2 逐项处置如下：

| 复核项 | Core V1.2 处置 |
|---|---|
| P1 MuJoCo direct-effort 把 motor gain 清零会导致零施力 | 已修正：保留 fixed gain 1/bias 0/gear 1，只绕过外部 PD 并清零 joint damping；canonical effort 经右轮符号映射写 ctrl，actuator force 与 qfrc round-trip golden |
| P1 P20-S body mapping/source identity/golden 退化 | 已冻结 source audit 文件与 SHA、embedded bundle identity、完整 27-body mapping；golden 使用非零 COM offset、非恒等惯量旋转、互异主惯量和非轴对齐速度，并增加三类 mutation test |
| P1 repeatability family 对 P30/P60 仍不精确 | 已把 configuration/excitation hash 纳入 schema，分别冻结 open/closed direct/target、P60 zero/first/replay 等 16 个 family；错误绑定 fail-closed |
| P1 universal explanation-ratio rule 与直接证据冲突 | 已冻结合法 ratio pair allow-list 和 baseline/ablation identity；P20-S 与 P50-A 使用类别专属直接失败规则，P60-B/C/D 禁止伪造 ratio；§8 统一改为 class-specific primary rule |
| P2 C70 amplifier 阈值未冻结 | 已冻结 feature-group/sign 聚合、`1.50/0.25` 双阈值、`[0.80,1.25]` tie zone、saturation `0.10/1 tick` 佐证和中间区 `inconclusive` |
| P2 Python/pytest cache 可逃出 suite | 已要求 `PYTHONDONTWRITEBYTECODE`、`PYTHONPYCACHEPREFIX`、pytest cache/basetemp 全部落在 run 内，并增加 before/after inventory 与负向测试 |

上述修订没有增加 Extended probe，只把既有 Core 的语义、身份、阈值和判定条件改为可执行、可审计契约。任何新增 probe 仍必须先证明是 P0/P1 阻塞，否则进入 Extended V2。

### 17.3 第三轮定向复核

第三轮独立 agent 对 Core V1.2 的结论为 `BLOCKED`，`P0=1, P1=2, P2=1`。本次 V1.3 逐项处置如下：

| 复核项 | Core V1.3 处置 |
|---|---|
| P0 裸 `uv run` 可能同步 `.venv`、lock 或使用 suite 外 cache | MuJoCo worker 固定直接调用 `sim2sim/mujoco/.venv/Scripts/python.exe`；编排器与测试禁止调用 uv/pip/conda；G00 对两个解释器、`pyvenv.cfg`、project/lock 和 dist-info 身份做前后核验，所有 Python cache 指向 run 内 |
| P1 P30 target/direct 实际同时改变 `drive_mode` 和 `input_kind`，与单字段 allow-list 矛盾 | `allowed_ablation_factors` 改为精确集合；P30 唯一允许 `[drive_mode,input_kind]`，P40/P50 分别只允许 `[closure_mode]`/`[friction]`；validator 要求实际语义差异集合完全相等 |
| P1 repeatability key 未绑定 reset/初态 | V1.3 首次加入 reset identity；经第六轮复核，V1.6 已进一步拆成 pre-forward 初态、post-forward 状态和 returned-policy 三个 phase hash，避免把闭链投影结果误当成消融初态污染 |
| P2 C70 未冻结 feature groups 和 DreamWaQ 五帧归约 | 冻结六个 ActorObsV1 slice、六个固定 policy tick、DreamWaQ 五帧 group RMS、PPO 最新帧 RMS、六通道 raw-action RMS，并增加 latest-frame-only sensitivity guard；history reduction 改变 amplifier/tie 结论时强制 `inconclusive` |

这些修订仍未新增任何 Extended probe，也未改变正式 ActorObsV1、history、策略、模型或物理参数。

### 17.4 第四轮定向复核

第四轮独立 agent 对 Core V1.3 的结论为 `BLOCKED`，`P0=1, P1=2, P2=1`。本次 V1.4 逐项处置如下：

| 复核项 | Core V1.4 处置 |
|---|---|
| P0 Isaac/Kit 默认 portable 会写项目 `.venv/Lib/site-packages/isaacsim/kit/{logs,data,cache}` | 当前 run 新增 `runtime_cache/kit`；Isaac worker 在任何 Isaac import 前必须传绝对 `--portable-root` 和显式 log/config/dump/crash settings，启动后读回 resolved paths；项目内既有 Kit 三目录完整 tree 纳入 before/after 不变检查；缺参数或路径逃逸在场景创建前失败 |
| P1 MuJoCo drive-off 未冻结 precise damping set | 冻结 6 controlled + 20 passive 共 26 个 robot hinge 名称；compiled set 必须完全相等并记录唯一 DOF address，26 项 damping 全为零，`base_free` 排除；保留任一 passive `0.05` 的 mutation 必须失败 |
| P1 ratio factor 缺可执行 normalized semantics | 新增完整 `configuration_semantics`/`excitation_semantics`、独立 identity hashes 和逐引擎 JSON-pointer diff；冻结 drive/input/closure/friction 的唯一允许 compiled paths，transform manifest 与 compiled diff 必须一一对应，隐藏 armature/body-mass/contact-solver 变化 fail-closed |
| P2 C70 baseline 来源未冻结 | 对 hash-verified Isaac-anchor 和 MuJoCo-anchor 分别运行 full-history 与 latest-frame-only 分析；记录 anchor trace/observation hash；只有两个 anchors 得到同一 amplifier 或 tie 结论才采用，否则 `inconclusive:baseline_anchor_sensitive` |

这些修订没有增加 probe，只关闭现有 Core 的写入边界、配置身份和 checkpoint 解释歧义。closure-off 也被收紧为保留 topology、只切换 enabled，避免合法因素携带隐藏拓扑变化。

### 17.5 第五轮定向复核

第五轮独立 agent 对 Core V1.4 的结论为 `BLOCKED`，`P0=2, P1=1, P2=0`。本次 V1.5 逐项处置如下：

| 复核项 | Core V1.5 处置 |
|---|---|
| P0 仅用 Kit setting 关闭 crash reporter 太晚，`SimulationApp` 可能已经把 `carb.crashreporter-*` 加入 plugin wildcard | 新增 suite-local `RootCauseSuiteAppLauncher`，只覆盖 `_create_app()`，在 `SimulationApp` 构造前复制 launch config 并强制 `enable_crashreporter=False`；冻结 `AppLauncher`/`SimulationApp` 源码哈希与 AST seam；真实 worker 同时保存 pre-load flag、resolved setting 和 loaded plugin list，任一层失败即不创建场景 |
| P0 只在 Python 代码中设置 `PYTHONDONTWRITEBYTECODE` 无法阻止父包首次 import 生成 `.pyc` | 唯一入口改为 `run_suite.ps1`；它在首个解释器启动前设置环境，并用 `python.exe -B -m ...` 启动父进程；所有子进程同样显式 `-B`，内部入口断言 `sys.dont_write_bytecode` 和 bootstrap marker；增加从无 cache 状态启动的负向/端到端测试 |
| P1 normalized semantics 没有覆盖完整的 MuJoCo/PhysX 结果决定选项 | MuJoCo 快照冻结完整公共 `MjOption` 字段集合；Isaac 快照递归覆盖完整非渲染 `SimulationCfg`、全部 `PhysxCfg`、physics material、resolved `/physicsScene` 与 articulation iteration counts；所有 engine-option paths 对 ratio pair 均不可变，并增加 integrator/noslip/CCD/flags 与 PhysX solver/CCD/stabilization/contact-threshold mutation tests |

这些修订没有新增物理 probe，也没有改变任何正式物理参数；它们只保证既定 Core 实验不会写出 suite、不会在证据采集前加载 crash reporter，并能检测隐藏 solver 配置漂移。

### 17.6 第六轮定向复核

第六轮独立 agent 对 Core V1.5 的结论为 `BLOCKED`，`P0=0, P1=1, P2=1`。本次 V1.6 逐项处置如下：

| 复核项 | Core V1.6 处置 |
|---|---|
| P1 单一 `initial_condition_hash` 同时包含 pre/post-forward，会在 P40 真正出现闭链投影差异时把目标现象误判为 pair 污染 | 拆成 `pre_forward_initial_condition_hash`、`post_forward_state_hash` 和 `reset_returned_policy_hash`；ratio pair 只统一要求 pre-forward 相同；P30/P50 另有 post/returned guard，P40 明确允许 post/returned 不同并把差异作为结果；同一场景重复性仍绑定全部三个 phase hash；schema、evidence 和 mutation tests 同步更新 |
| P2 P60-D/C70 replay 未预声明生成 Actor 和选择规则 | 固定 DreamWaQ run-01 Actor/manifest、Isaac nominal formal scenario、command `[0,0,0.20]`、seed `20261007`、repetition `0`、499-step horizon 和“只生成一次、无 fallback”规则；在任何 replay/C70 前原子冻结 source/action/identity hashes；同一动作先通过 fresh-reset Isaac replay 等价门，C70 必须引用同一 replay identity |

这两项修订只修正证据身份与选择偏差，不新增物理 probe，不改变正式 reset、termination、Actor、动作或仿真参数。

### 17.7 第七轮定向复核

第七轮独立 agent 对 Core V1.6 的结论为 `CONDITIONAL APPROVAL`，`P0=0, P1=0, P2=1`；批准进入实施计划，但要求在开始实现前关闭该 P2。本次 V1.7 处置如下：

| 复核项 | Core V1.7 处置 |
|---|---|
| P2 P60-D replay 虽已固定 Actor/seed/场景，仍可在生成前选择 1-env cache 或 8-env cache 的不同环境行 | 冻结现有 run-01 正式 evaluation reset-cache 的绝对相对路径、file/tensor/identity SHA、schema、两个 cache algorithm、`environment_count=8`、`scenario_name=nominal_stand`、`environment_index=0` 和 evaluation contract hash；cache 与 summary 加入 §2.2 前后哈希；`replay_source_identity_hash` 必须包含这些字段；mutation test 覆盖 1-env cache、另一 cache 和另一 env row |

该修订消除了 replay 生成前最后一个可选择自由度，不改变正式 evaluation cache，也不生成新的 cache。

### 17.8 第八轮定向复核

第八轮独立 agent 对 Core V1.7 的结论为 `CONDITIONAL APPROVAL`，`P0=0, P1=0, P2=1`。第七轮 replay P2 已充分清零；本次 V1.8 处置新发现如下：

| 复核项 | Core V1.8 处置 |
|---|---|
| P2 Kit `localTextureCachePath` 默认使用 `${omni_global_cache}/texturecache`，portable root 在本机不保证把它重定向出用户目录 | `kit_args` 显式增加 `--/rtx-transient/resourcemanager/localTextureCachePath=<absolute-kit-root>/cache/texturecache`；requested/resolved 路径进入 stage manifest 与启动后绝对路径断言；缺失、逃逸或仍解析到用户/global cache 的 mutation 和真实 1-tick worker 均 fail-closed；GPU driver cache 仅作为明确外部白名单 |

该修订只关闭 Kit 写入边界，不改变场景、模型、物理参数或 Core probe。

### 17.9 第九轮定向复核

第九轮独立 agent 对 Core V1.8 的结论为 `CONDITIONAL APPROVAL`，`P0=0, P1=0, P2=1`。第八轮 texture-cache 功能处置已充分；本次 V1.9 关闭其审计身份缺口：

| 复核项 | Core V1.9 处置 |
|---|---|
| P2 `kit-core.json` 本体未冻结，且 texture cache 只验证“位于 run 内”而非 requested/resolved/expected 精确相等 | §2.2 冻结 `kit-core.json` SHA256；G00/run manifest 对该 payload 做前后哈希并在 pre-launch 解析精确 key/default value；post-start 强制三方 canonical path 精确相等；mutation 增加 run 内错误 sibling 目录，真实 1-tick 同样验证；不再只依赖 wheel `RECORD` |

该修订只把已冻结的 setting key 变成可审计身份和精确等式，不改变任何运行语义。

### 17.10 第十轮定向复核

第十轮独立 agent 对 Core V1.9 的结论为 `CONDITIONAL APPROVAL`，`P0=0, P1=0, P2=1`。第九轮 `kit-core.json` 审计缺口已充分关闭；本次 V1.10 处置新发现如下：

| 复核项 | Core V1.10 处置 |
|---|---|
| P2 Isaac Lab `SimulationCfg.log_dir=None` 会让 Python `FileHandler` 默认写 `%TEMP%/isaaclab/logs` | 新增 run-local `runtime_cache/isaaclab/logs`；环境构造前 clone debug config 并固定 `save_logs_to_file=True`、absolute `sim.log_dir`；冻结 `simulation_context.py`/`utils/logger.py` hash 与 pre-write source contract；stage manifest 保存 requested/expected/resolved handler 文件；构造后要求全部 `FileHandler` 父目录精确相等；覆盖 `None`、False、相对路径、错误 sibling、逃逸和额外 handler mutation，并纳入真实 1-tick |

该修订只关闭 Isaac Lab Python 日志写入边界，不修改正式 env config 或仿真语义。

### 17.11 第十一轮定向复核

第十一轮独立 agent 对 Core V1.10 的结论为 `CONDITIONAL APPROVAL`，`P0=0, P1=0, P2=2`。既有 logger config/path 处置没有引入 P0/P1，但首次写入前的实际调用链和 handler 生命周期证据尚未闭合；本次 V1.11 逐项处置如下：

| 复核项 | Core V1.11 处置 |
|---|---|
| P2 `DirectRLEnv` 位于 env 与 `SimulationContext` 之间但未冻结，无法证明 hardened `cfg.sim` 是首次 logger 构造使用的对象 | §2.2 新增 `direct_rl_env.py` 精确 SHA；runtime manifest 从六项增为七项；冻结 import origin、`self.cfg=cfg`、singleton guard、精确 `SimulationContext(self.cfg.sim)` 调用和 existing-context error；worker 在 clone 后/构造前显式断言无既有 context；增加默认构造、预建 context 和错误 origin mutation |
| P2 构造后 handler 扫描会漏掉被 `configure_logging()` 移除的既有 handler，也不能证明生命周期内没有逃逸创建 | 新增第三方 import 前安装的标准库 `PythonWriteGuard`；V1.11 当时将其概括为覆盖全部 Python 写入，V1.12 已把实际事件集合与边界精确化；FileHandler 登记包装器在原始构造/打开前拒绝逃逸并保存 created-lifetime；manifest 同时保存 pre/created/post；固定 `delay=True` 逃逸 mutation 和合法路径中立性测试 |

这些修订只加强既有 G00 写入与身份门，不增加物理 probe，不修改正式 env、Isaac Lab、Kit 或 Python 标准库文件；登记包装器只在 worker 进程内临时生效并在 `finally` 恢复。

### 17.12 第十二轮定向复核

第十二轮独立 agent 对 Core V1.11 的结论为 `CONDITIONAL APPROVAL`，`P0=0, P1=0, P2=2`。第十一轮两个 P2 已充分清零；本次 V1.12 关闭新发现的身份路径和 audit 覆盖表述缺口：

| 复核项 | Core V1.12 处置 |
|---|---|
| P2 Phase 1R run-03 只冻结内容哈希，未冻结绝对来源路径，可能与 `phase1_v4` run-03 混淆 | §2.3 同时冻结两个策略的精确相对 Actor/manifest 路径与文件哈希；明确 `phase1_v4` 仅为 AssetBundle/source-audit 证据；G00 和 C70 禁止目录扫描、猜测或替代来源 |
| P2 “全部 Python 写入”表述大于只审计 write-mode `open` 的实际范围 | §3.3 明确定义 audit 事件集合与 write-flag mask，加入 mkdir/remove/rmdir/rename/link/symlink/truncate/chmod/chown/utime；所有 source/destination path 均 canonical allow-list，无法解析时 fail-closed；明确不覆盖 Kit/C++ native/GPU driver，并由既有 portable-root/resolved-setting/inventory 门负责；单元与真实 1-tick mutation 同步扩展 |

这些修订不增加 Core 物理实验，不修改任何正式输入，只把策略身份和 Python 层文件系统审计边界改为可直接实现和可验证的契约。

### 17.13 第十三轮 Windows audit 复核

第十三轮独立 agent 对 Core V1.12 的结论为 `BLOCKED`，`P0=0, P1=1, P2=0`。Phase 1R 路径身份已充分关闭；Python audit 边界仍遗漏 Windows delete-on-close flag 与平台能力证据。本次 V1.13 处置如下：

| 复核项 | Core V1.13 处置 |
|---|---|
| P1 write-flag mask 遗漏 Windows `O_TEMPORARY`，且不存在的 `O_TMPFILE`/`os.chown` 没有能力门控 | `open` mutation mask 加入 `getattr(os, "O_TEMPORARY", 0)` 并明确其无其他 write flag 时也视为变更；安装时输出 flag/API capability manifest；不可用项明确记 `unavailable_on_platform`；Windows `O_TEMPORARY` 增加 suite 内/外正反例，不可用 API 增加 manifest 断言，禁止静默 skip |

该修订只关闭 Windows Python audit 的可实现性和证据完整性，不改变任何实验、正式文件或物理语义。

### 17.14 第十四轮 Python audit 复核补充

另一独立 agent 对 V1.11 的深层标准库审计指出，除 V1.13 已处理的 `O_TEMPORARY` 与平台能力外，还必须关闭整数 fd 和 hook 安装自证缺口。本次 V1.14 处置如下：

| 复核项 | Core V1.14 处置 |
|---|---|
| P2 `os.fdopen()` 的 `open` audit path 可为整数 fd，无法直接 canonicalize | 冻结跨平台 fd 解析：Windows `msvcrt` + `GetFinalPathNameByHandleW`，支持 `/proc/self/fd` 的平台读取链接；标准流和安装时登记的匿名 pipe 白名单化，其他无法解析/无法证明 run-local 的可写 fd fail-closed；增加 suite 内/外/标准流/pipe/未知 fd mutation tests |
| P2 已有 hook 可在 `sys.addaudithook` 事件抛 `RuntimeError`，导致新 hook 静默未安装 | 安装后立即发送带随机 nonce 的 suite-local audit probe；ledger 必须精确收到一次才允许第三方 import；增加 blocker hook 负例和正常单次到达正例 |

这些补充仍只强化 G00/G03 的证据边界，不增加物理 probe，不修改正式环境或模型。

### 17.15 第十五轮 Windows capability 复核

第十五轮独立 agent 对 Core V1.14 的结论为 `BLOCKED`，`P0=0, P1=1, P2=0`。事件矩阵、整数 fd 与 nonce 自证已充分；剩余问题是 Windows `O_TEMPORARY` 负例可能因目标不存在而假通过，以及本机 `os.supports_dir_fd` 为空时测试无条件要求不可执行的正例。本次 V1.15 处置如下：

| 复核项 | Core V1.15 处置 |
|---|---|
| P1 `O_TEMPORARY` 与 `dir_fd` 测试未按真实平台能力设计 | `O_TEMPORARY` 正反例均预建现有文件，只使用 `O_RDONLY|O_TEMPORARY`，suite 外必须证明守卫先于原调用拒绝且文件/内容/metadata 不变；capability manifest 逐 API 记录 `os.supports_dir_fd`，仅支持时执行 `dir_fd` 正反例，不支持时必须断言 `unavailable_on_platform`，禁止静默 skip |

该修订只消除测试假通过与不可执行分支，不改变审计事件、物理实验或正式工程语义。

### 17.16 实施期 P20-S golden 可编译性勘误

V1.15 实施到 MuJoCo 单刚体读回 golden 时，MuJoCo 3.14 对冻结的 `diaginertia="0.031 0.047 0.083"` 报错：`inertia must satisfy A + B >= C`。该失败有精确数值依据：`0.031 + 0.047 = 0.078 < 0.083`，因此原参数不是任何物理刚体的合法主惯量，P20-S worker 无法按冻结契约创建该刚体。这是 §0 允许的实施期阻塞矛盾，不是数值容差或求解器调参问题。

| 复核项 | Core V1.16 处置 |
|---|---|
| P1 P20-S 单刚体 golden 的第三主惯量 `0.083` 违反刚体惯量三角不等式，MuJoCo 3.14 无法编译 | 只把第三主惯量改为 `0.073 kg m^2`；三个主惯量仍互不相等，非恒等旋转、非零 COM offset、非轴对齐线/角速度、固定 world origin 以及三类 mutation test 全部保持不变；禁止使用 `balanceinertia` 静默改写冻结输入 |

该勘误不增加、删除或重排任何 Core/Extended probe，不修改正式模型、训练语义、策略、阈值或 verdict 规则。Isaac 与 MuJoCo worker 必须使用同一个修正后的合法 golden，并继续把实际编译后惯量读回值写入证据。

### 17.17 实施期 P30 MuJoCo exact-diff 投影勘误

V1.16 实施到真实 MuJoCo P30 target/direct compiled pair smoke 时，原冻结 allow-list 要求 26 个 robot hinge 的 `dof_damping` path 全部形成 exact diff，但正式 XML 的六个 controlled hinge `jIJ,jIO,jAB,jAG,jwheel_left,jwheel_right` 已为 `damping=0`，drive-off variant 清零后仍为 0；只有其余 20 个 passive hinge 从正式 `0.05` 变为 0。把六条 unchanged path 强行加入 actual diff 会违反 §6.4 的“实际差异集合精确等于 allow-list”与 compiled-readback 原则。这是 §0 允许的实施期阻塞矛盾，不是放宽门禁。

| 复核项 | Core V1.17 处置 |
|---|---|
| P1 MuJoCo P30 drive-mode allow-list 把六个本来就是零阻尼的 controlled hinge 错列为必须变化 path | 保留全部 26 hinge 的 compiled 审计与 drive-off 零阻尼断言；actual exact diff 只展开正式值非零的 20 个 passive hinge `/actuation/robot_hinges/<name>/dof_damping`；六个 controlled hinge 在 before/after 中必须精确相等；外部显式 PD/velocity servo 的启用差异只由 `/actuation/external_controller_enabled` 表示，冻结 Kp/Kd 和 motor gain/bias/gear/limits 继续作为不变量 |

第一次 V1.17 定向独立复核确认该技术修订成立、正式 XML/compiled 数值与外部显式控制器证据充分，且没有扩大物理实验范围；唯一 P1 是本节加入前 §18 仍保留 V1.16 旧门禁。本节与下方当前门禁同步关闭该文档版本冲突，随后必须再次独立复核。该勘误不修改正式模型、控制器、策略、阈值、probe 或 verdict 规则。

## 18. 当前门禁

本文件是设计规范，不是实施计划。下一步顺序为：

1. 使用独立 agent 重新复核 Core V1.17 的 P30 MuJoCo exact-diff 投影勘误与本节门禁同步，复核报告必须记录所读文件 SHA256。
2. 清除该定向复核的 P0/P1；P2 修复或给出明确不采纳理由。
3. 同步设计哈希、`contracts.py`、factor-path allow-list artifact、worker compiled semantics 与 mutation tests 后继续既有实施，不重开已通过的其他设计项。
4. 完成代码、独立代码复核、测试、诊断执行与独立结果复核。

在 V1.17 定向复核通过前暂停后续 P30/P40/P50 因果身份实现和新 Isaac/MuJoCo 诊断；已完成的 suite-local 基础代码、测试与失败 smoke 证据保留，不修改任何正式工程文件。
