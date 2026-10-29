# 无动作裁剪：腿轮分项平滑与直行轮端差速奖励

日期：2026-10-16。实验 profile：`no_clip_split_straight_v1`；实验标识：`DebugNoActionClipSplitStraightV1`。

## 已确定的配置

基准为 `runs/fresh1000-fk-diagnostic-20261016/model_999.pt` 对应的 `DebugNoActionClipV2` 配置。该模型从零训练 1000 iteration，seed=41882943，256 环境；使用 `stop_reverse_v1`、六维合并 `action_smooth=-0.01`，没有轮端差速奖励。

本次实现两个奖励变化。保留原无裁剪 profile 和旧模型的加载能力，新 profile 不修改正式训练源文件。

### 腿轮分项平滑

`a_t` 为本控制步实际传给控制器的六维原始策略动作；顺序为 `jIJ, jIO, jAB, jAG, jwheel_left, jwheel_right`。

```text
d_t = a_t - 2*a_(t-1) + a_(t-2)
r_leg   = 0.02 * (-0.02) * sum(d_t[:4]^2)
r_wheel = 0.02 * (-0.01) * sum(d_t[4:6]^2)
```

复用工程已有的 `SplitSmoothRewardWeights` 和奖励计算。旧合并项 `action_smooth=0`，不再额外计罚；六维一阶差分 `action_rate=-0.01` 保持原值。分项按维度求和，不除以维数，不更改动作尺度。

### 直行轮端差速

```text
enabled = (command_vx != 0) and (command_yaw_rate == 0)
cost = (a_t[4] - a_t[5])^2 if enabled else 0
r_wheel_differential = -lambda * 0.02 * cost
```

条件读取本控制步用于奖励的原始命令，在停车/换向 schedule 推进之前计算。使用精确零值，不添加接近零的范围。前进、后退直行启用；停车、原地转向、行进转向关闭，包括任意非零的微小偏航命令。

两个轮动作采用相同前进方向定义，均乘以 25 后得到该方向的目标轮速。因此上述平方等于 `((target_left_control-target_right_control)/25)^2`。右轮写入仿真器时有负号，不直接相减两个仿真器关节坐标下的目标。

这是固定系数的软奖励，不修改动作，不取左右平均，不增加滤波、额外力矩或 ROGER。必要的差速修正仍然可以由策略输出，但启用期间也会产生代价。该实验的有效性需要训练后的 MuJoCo 对照验证，当前诊断没有证明某一个奖励项是偏航或腿抖动的唯一原因。

用户已确定本次训练使用 `lambda=0.1`，实际负权重为 `wheel_differential=-0.1`。新 profile 的入口要求显式指定有限、严格大于零的 `--wheel-differential-weight`，没有默认值。实际配置和 manifest 保存对应的负权重 `wheel_differential=-lambda`；缺少参数或传入零、负数、NaN、Inf 均立即报错。旧 profile 拒绝附带该参数，避免参数被忽略。

## 保持的训练和部署设置

- 六维运行时动作不裁剪到 ±1；腿目标角仍限于 ±1 rad，腿动作尺度 0.35，轮动作尺度 25，Actor 均值限幅仍为 ±20。
- 前后速度跟踪主项和增强项各 2.0；偏航速度跟踪主项和增强项各 1.0；其余旧奖励权重、sigma 不变。
- 停车/换向练习、命令范围和采样方式、随机化、终止条件、reset 和实际几何检查保持原有语义；旧 ROGER-style 加速度/换向试验关闭。
- 网络、CENet、storage、PPO、五帧 t-4..t history、观测和归一化不变。
- USD、MuJoCo XML、接触、PD 和力矩限制不变。MuJoCo 继续使用无动作裁剪控制器；训练奖励不添加到部署控制器。

## 文件和依赖方向

全部实现文件位于 `debug/no_action_clip/`，正式模块不引用 debug。

| 文件 | 职责 |
|---|---|
| `rewards.py` | 独立奖励权重子类、参数校验、精确命令条件和差速成本、可序列化奖励契约及实验身份校验；只依赖 Torch 和正式纯奖励模块 |
| `environment.py` | 计算推进命令前的差速成本；复用父环境分项奖励后只增加一次差速奖励，维护单步和 episode 日志 |
| `train.py` | 增加显式 profile/系数参数；原入口默认不变；保存真实权重、奖励契约和独立实验标识 |
| `experiment.py` | runner/checkpoint 绑定新奖励契约；拒绝身份、系数或保存权重不一致；继续只支持从零训练 |
| `export.py` | 导出保存真实实验标识及奖励契约，网络和 golden-vector 公式不变 |
| `mujoco_adapter.py` | 核验新导出和训练奖励身份，复用相同无裁剪控制器和物理模型 |
| `play_mujoco_viewer.py` | 从已核验的导出读取实验标识，用于窗口和会话记录，键盘和控制逻辑保持原有值 |
| `test_rewards.py` | 数值验证命令条件、原始动作、腿轮隔离、不重复计罚、参数校验及契约一致性 |
| `verify_rewards_mujoco.py` | 在真实 MuJoCo 模型和已有策略上验证奖励公式及动作/物理结果不受计算奖励影响；检查实际旧 checkpoint、导出加载兼容性 |

`ACTION_CONTRACT` 继续使用 `DebugNoActionClipV2` 的动作语义标识；训练实验标识和新增奖励契约分别识别奖励变化。新旧 checkpoint 不使用相同训练身份。新契约绑定 manifest、checkpoint infos 和导出；常规 resume 仍拒绝跨配置，当前入口继续拒绝所有 resume。

## 实施和验收顺序

- [x] 写入本文件，检查公式、条件和保存路径与用户决定一致。
- [x] 实现纯奖励模块和单元测试，覆盖前进/后退、停车、两种转向、任意微小非零偏航命令、超出 ±1 的动作、左右交换、平方量级和无重复扣分。
- [x] 接入 debug 环境，保证差速项只累计一次，且在父环境推进命令之前计算。
- [x] 接入训练参数、manifest/checkpoint/export/MuJoCo loader；原 profile 和实际旧工件继续可以加载。
- [x] 运行有针对性的纯数值测试和真实 MuJoCo 检查，核对正式文件 SHA256。
- [x] 记录测试命令、数量、结果和未运行的验收。

测试使用项目 Python 运行纯 Torch 测试、MuJoCo Python 运行真实模型检查；不启动 Isaac，不执行 PPO 更新，不产生新训练 checkpoint。本次不运行原 `test_experiment.py` 中的 PPO 更新测试。新配置的真实 Isaac 训练链路、保存/重载及新权重 TorchScript 导出验收，随后续明确授权的训练执行；不能把旧权重的兼容性检查当成新训练已经通过。

命令从 `E:\wheel_leg_rl-main\wheelleg_dreamwaq` 执行：

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m pytest debug/no_action_clip/test_rewards.py tests/unit/test_split_action_smooth.py debug/no_action_clip/test_experiment.py -k 'not real_ppo_storage_and_update' -q
$env:PYTHONPATH = (Get-Location).Path + ';' + (Join-Path (Get-Location) 'source/wheelleg_dreamwaq') + ';' + (Join-Path (Get-Location) 'sim2sim/mujoco')
.\sim2sim\mujoco\.venv\Scripts\python.exe -m debug.no_action_clip.verify_rewards_mujoco
```

本次训练入口选择 `--reward-profile no_clip_split_straight_v1 --wheel-differential-weight 0.1`。用户已确定从零训练 1000 轮，seed=41882943，4096 环境，单 GPU `cuda:0`。8 环境、1 轮只用于已经完成的真实训练链路验收；本次 4096 环境训练独立初始化。每轮采集 24 步，共 98304 个样本，1000 轮共 98304000 个样本，8 个 minibatch 的设置保持原值。每次运行采用独立目录，保留现有模型和记录。

验收目录为 `runs/validation-split-straight-lambda01-20261016`，本次训练目录为 `runs/fresh1000-split-straight-lambda01-env4096-20261016`。启动前核对目录没有已有训练工件；训练期间只写入 runs 内的进度和日志，不修改已经绑定 SHA256 的源文件和本文档。沿用每半小时检查训练进度的安排。

## 验证结果

2026-10-16 完成实现和本次范围内的验证，没有启动训练。

- 上述纯 Torch 回归命令：`53 passed, 1 deselected`。排除项为旧的真实 PPO 更新测试；本次没有进行优化器更新。覆盖新增命令条件和原始动作、参数/保存契约拒绝、正式分项算术以及旧无裁剪数据通路和 TorchScript。
- `verify_rewards_mujoco`：真实 MuJoCo 中完成前进、后退、停车、原地转向、行进转向和 `1e-12` 非零偏航指令六组，每组 50 个控制步。差速项仅前进/后退直行启用；奖励计算前后动作不变，与不计算奖励的独立参考实例逐步比较 qpos、qvel、力矩和观测，全部逐元素相同。这项检查验证实现没有改写控制器，不能作为新奖励已经改善运动表现的证据。
- MuJoCo 奖励公式验收使用的 `lambda=0.1` 仅为测试输入，不作为训练配置或默认值。真实报告保存为 `runs/reward-verification-20261016/verification.json`，其中 `training_coefficient=null`。
- 原 `test_mujoco_adapter` 真实模型动作、history、reset、轮符号和力矩检查通过。
- 实际旧 `model_999.pt` 加载检查通过，标识为 `DebugNoActionClipV2`，完成轮次 1000。`verify_artifacts` 在副本上检查 env.yaml、agent.yaml、reset cache 以及几何契约损坏均拒绝；完好副本仍可加载。原模型未修改。
- 旧导出加载和保存的 golden vectors 检查通过，MuJoCo 进程中重新加载输出最大绝对误差为 0。这是旧工件兼容性验收，不代表新奖励训练后的网络已经通过导出验收。
- `compileall -q debug/no_action_clip` 通过；训练入口 `--help` 显示新 profile 和显式系数参数。真实命令行使用新 profile 且省略系数时，在创建 AppLauncher 之前报错退出，没有启动仿真或训练。
- 原冻结清单中 100 个正式文件 SHA256 全部相同。修改范围限于本 debug 目录。

上述验证完成时，新配置尚未执行真实 Isaac 训练链路验收、checkpoint 保存/重载或新权重导出，也尚未评估训练后的速度跟踪、偏航和腿抖动。随后用户确定训练系数为 0.1 并授权训练；真实训练与导出结果以对应 runs 目录中的 manifest、progress、training_summary 和导出报告为准。

## 文档和代码的独立复核

用户随后授权调用一个 agent 复核文档和代码。[复核记录](review_split_straight.md)：未发现阻塞问题或需要修改的确定性错误；独立复跑 53 项纯 Torch 测试、实际旧 checkpoint 加载、真实 MuJoCo 控制器/history/reset、旧导出和 golden 校验，100 个正式文件 SHA256 一致。没有修改代码或启动训练。

新 profile 的真实环境 `_get_rewards()`、命令切换及 episode 累计仍需在训练前的 Isaac 链路验收中实际执行，当前静态时序核对不替代这项验收。

## 本次训练启动验收

`runs/validation-split-straight-lambda01-20261016` 已完成 8 环境、1 轮真实 Isaac 训练。六维原始动作进入 Actor/Critic、部分 reset 和五帧 history 检查通过；保存的新身份 checkpoint 重载通过，正式 loader 按预期拒绝实验模型。新导出通过 TorchScript golden-vector 检查，最大绝对误差 `1.2665987014770508e-7`；真实 MuJoCo 加载和身份损坏拒绝检查通过。

实际 TensorBoard 日志的差速成本均值为 `0.811638355255127`，差速奖励均值为 `-0.0016232768539339304`，满足 `-0.1 * 0.02 * cost`。腿、轮分项奖励均已执行，合并项没有重复出现；全部已记录标量有限，100 个正式文件 SHA256 不变。证据见该目录的 `training_acceptance.json`、`training_summary.json` 和 `mujoco-loader-integrity/verification.json`。这项验收确认训练和导出链路可以执行，运动效果由 1000 轮训练后的 MuJoCo 评估确定。

## 2026-10-16 停止训练与可视化

用户在完成 729 轮后要求停止本次 4096 环境训练，训练进程已经退出，每半小时检查已经暂停；本次没有完成 1000 轮。最近持久化的 checkpoint 为 `runs/fresh1000-split-straight-lambda01-env4096-20261016/model_700.pt`，对应已完成 701 轮。第 702–729 轮没有保存为新的 checkpoint，不能将当前可视化称为 729 轮模型。

该 checkpoint 的 SHA256 为 `A351C42E999F873FDA0921940BA9F2C57FB2143485971DF50A8757B067507FB9`，导出 Actor 的 SHA256 为 `FDCA84A3B9B4F993DF481F09DBAD14CC7AFBCDB861D530642EA522EE50837464`。checkpoint 重载与导出检查通过，TorchScript golden-vector 最大绝对误差 `7.152557373046875e-7`。已使用该导出打开 MuJoCo 键盘可视化，会话位于同一 runs 目录下的 `viewer-701-20261016-164336/`。当前尚未完成该模型的完整八场景评估。

停止证据为该运行的 `progress.json`、`monitor_snapshot.json`，以及同名 `-launch/` 目录中的 `user_stop.json`；导出身份见 `export/experiment_export.json`。训练权重、runs 日志和运行时会话继续遵循仓库已有忽略规则。本次 GitHub 提交保存代码、配置契约和诊断文档，模型文件保存在本地。用户已确定下一次训练采用 6144 个环境；当前代码默认值仍为本次运行使用的 4096，不在此次提交中启动下一次训练。

## 2026-10-16 MuJoCo 航向保持开关对照

用户要求接入航向控制器。本次新增 debug 可视化航向保持，设计、控制公式和验证见 [heading_control.md](heading_control.md)。航向误差通过 Kp=0.5、Kd=0 产生网络的偏航速度指令，限制为 ±1 rad/s；六维动作仍由原模型产生。停车和直行保持参考朝向，非零用户转向指令直接传递，结束转向后捕获新朝向。H 切换，R 重置时重新捕获；未训练、未修改正式代码和 XML。

使用上述完成 701 次更新的同一导出，真实 MuJoCo 的开启/关闭两组各八场景、500 控制步全部完成。10 秒末停车、前进、后退的航向偏移绝对值分别由 22.10°/21.72°/33.06° 降至 4.35°/4.65°/6.34°，速度跟踪误差基本相同，仍有残余偏航。控制间隔上的四驱动关节角加速度 RMS 变化均小于 0.1%，不作为腿部抖动已经消除的证据。

`runs/heading-hold-verification-20261016/summary.json` 和 CSV 记录原始命令、纠偏命令、实际运动、六维动作、目标角/轮速和角加速度。关闭组与独立旧固定命令路径逐步相同；转向场景开关两组轨迹相同。针对性测试 `23 passed, 1 deselected`，100 个正式文件及 Actor 指纹不变。任意键盘切换与长时间保持效果需继续观察。

## 2026-10-16 正式 baseline 接入

正式配置 `stop_reverse_raw_split_straight_yaw2_headingp9_v1` 接入原始动作、腿轮平滑 -0.02/-0.01、直行差速 -0.1，并设置偏航两项权重各 2，加入 P=9/D=0 航向命令处理。主架构更新到 v0.30；本实验已有模型、checkpoint 和导出文件保持原身份。

正式 MuJoCo 运行器与解析器升级后，旧 debug 加载器通过 `baseline_runtime_compatibility.json` 固定指纹记录逐文件核对允许的变更，仍核对原训练清单、模型、源 checkpoint 和全部导出 SHA256。当前覆盖完成 701 次更新的模型；未实测的旧 debug 导出继续拒绝该运行器变更。修改前后用八场景各 32 步比较动作、目标、观测/history、力矩与 qpos/qvel，具体结果与实现验收记录保存到 `docs/2026-10-16-dreamwaq-baseline-implementation-results.md`。

正式新模型可视化使用 `play_mujoco_viewer.py --formal-export <正式三件套目录>`；航向参数从 manifest 恢复，键盘 H 使用正式共享控制器开关。原有 `--export` 分支继续使用 debug 控制方式。两个分支分别执行自己的单个航向控制器。

本次正式接入的最终检查为 260 项单元测试、77 项 MuJoCo 测试通过，真实 8 环境 fresh 1 次更新及恢复到总计 2 次更新、部分 reset/history、优化器步数 40→80 和导出检查通过；正式同 batch golden-vector 最大绝对误差为 0。单个复核 agent 指出的两处身份校验问题已修复并复查。旧 701 轮工件经当前加载器执行八场景各 500 步，全部存活；修改前后两份旧模型各八场景 32 步的数组精确相等。

接口模型只完成 2 次更新，其八场景仍因倾斜提前结束，不能据此评价新 baseline 的训练效果。正式 6144 环境/1000 轮训练尚未启动。完整证据与待完成的行为验收见 [实现验收记录](../../docs/2026-10-16-dreamwaq-baseline-implementation-results.md)。
