# WheelLeg DreamWaQ：过去五帧与 AdaBoot 核对、设计建议和逐文件修改量

日期：2026-10-10（Asia/Singapore）。状态：修改前的核对与范围草案；不是已实现或已验收的声明。

用户最新要求：CENet 使用 `t-5..t-1`；Actor 保留当前帧 `t`；尝试启用 AdaBoot 的动态概率。此要求取代此前误说的 `t-4..t`。

当前正式架构仍为 `Architecture v0.23`，SHA256 为 `43AC9306FD065DA259C60033E702C264230BB6B9D767B3974EFCE940587B94FA`。本次仅新增本核对文件，没有更新正式架构、修改源码或启动训练。

## 1. 已核实的事实

- 主架构 `../../docs/2026-10-03-wheelleg-dreamwaq-architecture.md:884` 明确规定 `[o(t-4),...,o(t)]`；第 222、960、962、1086 行明确排除第一版 AdaBoot 和 privileged-velocity 替换。
- `algorithms/dreamwaq/history_wrapper.py:45` 强制 history 最后 25 维等于当前 policy；第 68 行追加 step 返回的新观测，因此当前训练确实包含 `o(t)`。
- `deployment/dreamwaq_inference.py:30` 把全部 125 维送进 encoder，第 33 行把最后 25 维当作 Actor 当前帧。当前导出不能直接装载“不包含当前帧”的 history 而保持相同含义。
- `algorithms/dreamwaq/actor_critic.py:176` 始终拼接 `current_obs + estimated_velocity + context_mu`。对本工程 `source/scripts/configs` 搜索未发现 AdaBoot 开关、概率函数或抽签逻辑。
- storage 当前额外字段只有 next-proprio target 与 reconstruction mask；启用 AdaBoot 还需保存逐 transition 的速度来源 mask。
- PPO 中的 `time_outs` reward bootstrap 与 AdaBoot 是两个不同机制。

以上包内路径统一相对于 `source/wheelleg_dreamwaq/wheelleg_dreamwaq/`；下面以 `scripts/`、`tests/`、`sim2sim/`、`debug/` 开头的路径相对于项目根 `E:\wheel_leg_rl-main\wheelleg_dreamwaq`。

## 2. A1 参考实现及实际启用情况

A1 根目录：`E:\gogo_2026_09\A1_Base-codex-dreamwaq-pace-baseline-port`。以下 A1 包路径相对于 `gogo-learn/gogo_learn/dreamwaq/`。

| 参考文件和行号 | 已核实行为 |
|---|---|
| `model.py:54-67` | 总奖励的总体标准差除以 `max(abs(mean),1e-6)` 得到 CV；计算 `p=1-tanh(CV)` |
| `rsl_rl.py:32-57` | 默认开启；最近 256 个完成 episode 的奖励缓冲；概率上下界 0/1 |
| `rsl_rl.py:347-367` | 每个环境累计原始 reward，done 时记录总和并清零；少于两个完成样本时 p=0 |
| `rsl_rl.py:85-98` | 每次 rollout act 重新计算当前 p；用 `rand([N,1]) < p` 逐环境选择估计速度 |
| `model.py:563-570` | `where(mask, estimated_velocity, privileged_velocity)`，选择会改变 Actor 的速度输入 |
| `storage.py:17,38,54-60,79,98` | 保存 mask，mini-batch 按相同索引读取 |
| `rsl_rl.py:166-177` | PPO update 复用 rollout mask，不重新抽签 |
| `model.py:536-539` | 推理直接走估计器，不向 Actor 注入真值速度 |

论文 v2 的式 (8) 也是 `p_boot=1-tanh(CV(R))`，见 [DreamWaQ 原论文 II-C](https://arxiv.org/html/2301.10602v2#S2.SS3)。论文以 learning iteration 描述调度；这里逐 act 调度、256 样本滑窗和数值保护是本地 A1 的具体实现选择，不能把它们说成论文唯一规定。

启用证据：A1 `gogo-learn/isaaclab_ext/a1_locomotion/agents/rsl_rl_ppo_cfg.py:31` 默认 `adaboot=True`；本地两个 logs 根共找到 76 份 agent.yaml，其中 75 份包含 `adaboot: true`，1 份未包含该字段，没有发现包含 `adaboot: false` 的保存配置。这是保存配置的核对，不是对全部训练过程的重新验收。

一份实际保存配置：`gogo-learn/logs/rsl_rl/gogo_a1_local_pace_midrough_ab_baseline/2026-09-19_01-04-56-953993_reward_ab_baseline_seed20060604_4096e_3000it_20260919_010451/params/agent.yaml:83-86`，明确开启、buffer=256、min=0、max=1。配置源码第 512-536 行另有专门的 `NoAdaBoot` 实验类，明确关闭；不能把整个 A1 仓库说成每个任务都开启。

这里 p 的定义是“使用估计速度的概率”。例如 p=0.7 表示约 70% 的选择使用估计速度，约 30% 使用训练期真值速度。当前 WheelLeg 没有 p 变量，始终使用估计速度，按输入行为相当于 p=1。此前所谓“0%”指真值使用率，不是该定义下的 p。

## 3. 建议的明确实现边界

### 3.1 history 与导出接口

- 新训练的 TensorDict 保持 `policy[25]`、`policy_history[125]`、`critic[41]`。`policy_history_t=[o(t-5),...,o(t-1)]`；`policy_t=o(t)`。
- wrapper 从 t 推进到 t+1 时，为未结束环境追加缓存的 `o(t)`，不能追加返回的 `o(t+1)`；结束环境以本次 reset 后新观测复制填满五帧，不能带入上一 episode 的帧。
- 构造/full reset 的底层观测次数、噪声调用、重复 get_observations 的幂等性沿用现有实现。早期不足五帧时沿用第一帧复制补齐，不增加物理预热步。
- 三种导出选择：单参数 150 维拼接；两个参数分别传 history/current；导出图内部保存 history。建议第一种，保持现有单参数、无状态 TorchScript 调用。
- V2 的 150 维是部署接口打包 `[past_history125,current25]`，不是把 CENet 扩成 150 维。encoder 只读取前 125 维，Actor 使用最后 25 维当前观测。
- MuJoCo V2 外部缓存可保存六帧：前五帧交给 CENet，最后一帧交给 Actor。V1 五帧 125 维与 PPO 单帧 25 维保持各自旧语义。

### 3.2 AdaBoot

建议采用本地 A1 已有规则，适配 WheelLeg 的 RSL-RL 3.1.2 接口，不形成 A1 运行时 import 依赖：

- 默认开启，buffer=256，概率上下界 0/1，CV 使用总体标准差与 `max(abs(mean),1e-6)`。
- 少于两个完成 episode 样本时 p=0；之后每次 rollout act 根据已有完成奖励更新 p。p 可以升降，不是按训练轮数单调递增，也不是直接由速度 MSE 决定。
- p 以本 runner 管理的环境所完成的 episode 为统计范围。保存原始环境 reward 总和，在 PPO timeout reward 补值前记录；不改环境 reward。
- 每个 transition/environment 抽一个 bool mask。Actor 的速度输入为 `where(mask,estimated_velocity_t,critic_t[25:28])`；context 仍为 `context_mu`。
- rollout、log probability、old_mu/old_sigma 和 update 必须对应同一次选择。storage 保存 mask，update 使用保存的 mask，禁止重新抽签。
- 速度监督仍针对 estimated velocity；decoder 仍使用 detached estimated velocity 和 detached clipped action；辅助 loss 保持现有定义。CENet 和 Actor 的网络层及参数数量不变。
- play、Isaac 正式评估、导出、MuJoCo 和部署统一只用估计速度，不注入真值。AdaBoot 开关关闭时同样只用估计速度，不进行抽签。
- 抽签使用训练侧 Torch RNG，纳入现有 checkpoint Torch RNG 捕获；不使用或增加环境七个随机化/noise stream 的采样。
- logger 记录 p、CV、完成样本数及实际 estimator mask 比例；冷启动的 CV 标记为不可计算，不输出伪造统计量。

### 3.3 checkpoint、版本与复现

- 建议新增 `DreamWaQAlgorithmContractV2`、`DreamWaQExportContractV2`、`DreamWaQPolicyExportV2` 和 `DreamWaQCheckpointMetadataV2`；显式记录 history 时间偏移、Actor 当前帧和 AdaBoot 规则。
- 保留 V1 推理/导出读取路径，以及既有 MuJoCo V1 三件套读取路径。按保存版本选接口，不能将 V1 输入含义静默改成 V2。
- V1 checkpoint 到 V2 训练不作为 resume 接受。已有训练结果保留，不改旧 manifest/hash 或历史报告。
- V2 checkpoint 保存奖励滑窗、概率状态及已有 RNG/optimizer 状态。滑窗和 p 在 resume 后恢复；现有 resume 会创建新环境，因此未完成 episode 的 reward 累计必须清零，不能接到旧环境累计值上。history 同样按新环境首帧重建。
- 既有 `init_at_random_ep_len=True` 保持不变。按 A1 的采样期累计规则，首次从采样开始至 done 的奖励段也进入缓冲；不能为了 AdaBoot 私自改变 episode_length 初始化。
- 正式架构建议升级为 v0.24，明确 V1 基线与新 V2 实验的边界，修订 history、velocity 注入、storage、checkpoint、导出与验收条款。当前文档仍未修改或宣告新版本冻结。

## 4. 功能文件修改量

估算口径：新增行加改写的原有行；不是精确 git diff 的新增/删除数。范围已按具体函数和依赖核对，实际行数在实现后才能统计。表中不包括重新训练的日志和工件。

| 文件 | 预计改动行数 | 职责及依赖方向 |
|---|---:|---|
| `schemas/observation.py` | 20-35 | 新时间窗口标识与 past-history 推进 helper；依赖 torch，供 wrapper/deployment 使用 |
| `algorithms/dreamwaq/history_wrapper.py` | 15-25 | 缓存当前帧、推进过去五帧、替换旧 last==current 断言；依赖 schema/RSL wrapper |
| `algorithms/dreamwaq/adaboot.py`（新增） | 100-150 | CV、奖励滑窗、cold-start、状态导出/校验；供 PPO/checkpoint 使用，不依赖环境 |
| `algorithms/dreamwaq/actor_critic.py` | 25-45 | 训练期可选速度来源 mask，update 复用；推理继续使用估计器 |
| `algorithms/dreamwaq/storage.py` | 20-35 | 新增 bool mask 的 transition、copy_ 和 mini-batch 返回 |
| `algorithms/dreamwaq/ppo.py` | 35-60 | 记录 raw episode reward、计算 p/抽签、写 mask、update/日志接入 |
| `tasks/direct/wheelleg_flat/agents/dreamwaq_cfg.py` | 5-10 | 增加明确的 AdaBoot 字段；网络维度保持原数值 |
| `schemas/dreamwaq_manifest.py` | 50-85 | V2 窗口/AdaBoot/storage/150D 导出契约；保留 V1 构造与严格校验 |
| `training/dreamwaq_checkpoint.py` | 45-75 | V2 状态保存/恢复/校验及 V1 只读兼容；保留新环境 resume 边界 |
| `deployment/observation_adapter.py` | 25-45 | V2 past-history/current 缓存；V1 adapter 保留 |
| `deployment/dreamwaq_inference.py` | 15-25 | 新 V2 150D 无状态 graph，encoder/Actor 分别取输入；V1 graph 保留 |
| `deployment/manifest.py` | 15-30 | 按版本生成 golden inputs、保存/重新加载、动态 batch 校验 |
| `deployment/__init__.py` | 2-4 | 导出新 V2 名称 |
| `scripts/train_dreamwaq.py` | 3-8 | 新状态审计/训练 summary 接入；主体继续使用统一 cfg/builder/wrapper |
| `scripts/play_dreamwaq.py` | 10-20 | 按 checkpoint 版本选 contract/history，使用当前帧与过去五帧 |
| `scripts/evaluate_isaac.py` | 10-20 | 同上，并生成当前 source fingerprint 的评估报告 |
| `scripts/export_dreamwaq_actor.py` | 15-25 | 按版本选择 graph、adapter 和输入 metadata |
| `sim2sim/mujoco/wheelleg_mujoco/contract.py` | 20-35 | 识别严格 V2 150D/time-window；保留 PPO/V1 |
| `sim2sim/mujoco/wheelleg_mujoco/runner.py` | 15-25 | V2 六帧打包；V1 五帧与 PPO 单帧仍按旧契约运行 |
| `sim2sim/mujoco/wheelleg_mujoco/versions.py` | 1-3 | 增加 V2 adapter identity |
| `sim2sim/mujoco/wheelleg_mujoco/evaluation.py` | 2-5 | 将 V2 识别为 DreamWaQ；当前只识别 V1 的分支需同步 |
| `debug/sim2sim/dreamwaq_debug_contract.py` | 15-30 | 接受 V2 150D，encoder 只读取过去 125D，缓存/metadata 对齐 |
| `debug/sim2sim/trace_schema.py` | 2-5 | 在保持旧 trace 可读的同时允许 V2 输入维度 |
| `debug/sim2sim/collect_dreamwaq_isaac_trace.py` | 3-8 | reset/history 断言区分 125D encoder history 与 150D 打包输入 |
| `debug/sim2sim/collect_dreamwaq_mujoco_trace.py` | 3-8 | 同上；实际 runtime 输入与 adapter 继续逐项比较 |

功能合计：25 个文件，其中 1 个新增；约 **471-816 行**。

## 5. 测试文件修改量

| 文件（项目相对路径） | 预计改动行数 | 必须验证的行为 |
|---|---:|---|
| `tests/unit/test_dreamwaq_core.py` | 25-45 | 帧编号时间窗口、current/past 分开、网络参数数量不变 |
| `tests/unit/test_dreamwaq_export.py` | 40-70 | V2 adapter、150D eager/script/load 对齐与动态 batch；V1 保留 |
| `tests/unit/test_dreamwaq_adaboot.py`（新增） | 90-150 | CV/p/cold-start/raw-reward/done/缓冲边界/finite/state 校验 |
| `tests/unit/test_dreamwaq_storage.py` | 15-25 | mask 快照、copy_、batch_idx 对齐及不被后续 step 改写 |
| `tests/unit/test_dreamwaq_ppo.py` | 40-65 | p=0/1、mixed mask、rollout/update log-prob 对齐、不重抽签及关闭兼容 |
| `tests/unit/test_dreamwaq_checkpoint.py` | 35-60 | V2 调度状态/RNG 恢复、未完成奖励清零、旧版本只读/跨契约 resume 拒绝 |
| `tests/unit/test_dreamwaq_manifest.py` | 20-35 | 窗口/概率规则进入 hash；错误版本组合必须拒绝 |
| `tests/integration/test_dreamwaq_startup.py` | 20-40 | Isaac 短 rollout、train/inference 对齐、V2 checkpoint/resume/export |
| `tests/integration/probes/reset_observation_coherence.py` | 20-35 | 真 Isaac 构造/full/partial/auto-reset 的过去帧与当前帧，无旧 episode 泄漏 |
| `tests/integration/test_reset_observation_coherence.py` | 5-10 | 校验上述新增运行结果，保留观测/RNG 调用计数验收 |
| `sim2sim/mujoco/tests/test_mujoco_adapters.py` | 30-50 | V2 六帧时序、重复 reset、同 action 下物理步和状态与旧 adapter 一致 |
| `sim2sim/mujoco/tests/test_evaluation_metrics.py` | 10-20 | V2 evaluator 分支正确读取任务契约 |
| `debug/sim2sim/tests/test_dreamwaq_v2_inputs.py`（新增） | 60-100 | V2 实际 TorchScript fixture、debug encoder 分片与 MuJoCo collector 短 trace |
| `debug/sim2sim/tests/test_trace_schema.py` | 8-15 | 25/125/150 维按版本验证，非法宽度/组合拒绝 |

测试合计：14 个文件，其中 2 个新增；约 **418-720 行**。

正式架构及 README 后续修改：根架构文档 50-90 行、项目 README 5-10 行、MuJoCo README 5-10 行，合计 60-110 行。本核对草案自身不计入后续实施量。

全范围预计：42 个功能/测试/正式文档文件，约 **949-1646 行**。其中包含旧版本只读兼容、训练/部署两端、通用 debug 和实质测试；不代表网络需要重写。这是逐文件估算，不是已完成的修改数。

## 6. 实施依赖顺序与验收

1. 更新正式架构：明确 V2 history、p 的含义/规则、storage mask、checkpoint 与导出输入；V1 的已有训练/评估结果保留。
2. history schema/helper → Isaac wrapper → 外部 V2 adapter。先以唯一帧编号验证 t-5..t-1、current=t、reset 补齐、异步 done、重复 get 幂等。
3. AdaBoot helper → ActorCritic 速度选择 → storage mask → PPO 采样/更新/监控。固定参数下必须验证 rollout 与 update 用同一个 mask 得到相同 mean/log-prob；更新时不抽签。
4. cfg/algorithm/export contracts → checkpoint/runner → train/play/evaluate/export 入口。验证 V2 同契约 resume 恢复完成奖励历史和 RNG；新环境未完成累计清零；V1→V2 resume 拒绝。
5. TorchScript V2 → MuJoCo contract/runtime/evaluator → 通用 debug collectors。golden input 为 `[32,150]`，再验证动态 batch 1/3、重新加载与训练侧输出；最大绝对误差沿用 `1e-7`。
6. 真实 Isaac integration 包括构造/full/partial/terminated/truncated reset、history/current 时间关系、噪声/RNG 调用计数、短 rollout 与 checkpoint/resume。测试所需短更新属于集成验收，不等同于启动四次正式训练；本次未运行。
7. 仅在代码与接口验收后进行新的训练实验和八场景 sim2sim。旧 checkpoint 可用于 V1 回放/接口诊断，不能宣称已经按 V2 history/AdaBoot 训练。

## 7. 0 行修改的边界

- `algorithms/dreamwaq/cenet.py`：网络层、估计输出维度和 decoder 条件维度保持原样。
- 上游 RSL-RL 3.1.2 和 Isaac Lab 2.3.2 源码：0 行，接入留在工程自有类。
- `scripts/run_dreamwaq_training_suite.py`、`scripts/evaluate_mujoco.py`：既有动态 contract/单参数策略调用可以沿用，预计 0 行；新 source identity 会自动进入新运行工件。
- 普通 PPO、Phase 1R 环境/随机化、USD、正式 MuJoCo XML/控制器/观测物理量/初态与参数：0 行。reset 缓存失效修复继续保留；不增加 reset 物理步。
- velocity 标签仍为 `critic_t[25:28]`，下一帧 clean 16D reconstruction、done mask、KL、Gaussian/GAE/adaptive-KL/global clip 继续沿用。
- reset cache identity 不依赖本次 history 接口，不能仅因 history 改动重新生成或修改物理 cache。Isaac evaluation source fingerprint 将变化，正式可比报告需要用新 evaluator 重新生成 PPO baseline 评估；不需要据此重训 PPO。
- 冻结的 RootCauseSuite、历史 run-01 分析器和 artifact 内诊断脚本保持旧版本，0 行。通用 V2 trace 接入不等于改写或通过冻结 suite 的 gate；若以后需正式 V2 suite，应另立版本。
- 不添加新奖励、动作、命令、随机化 profile 或 MuJoCo 参数试验。

## 8. 本次核对范围和限制

核对包含现有主架构、训练/RSL 调用顺序、history/ActorCritic/storage/PPO/checkpoint、play/evaluation/export、MuJoCo runtime 与通用 debug 输入，以及相关测试入口；A1 核对的是本地参考实现，不是声称其实现等同于官方公开代码的全部版本。

本次没有启用 AdaBoot、没有改变 history，没有启动训练，也没有对新设计声称测试通过。新代码的验收必须在实现后按上述顺序执行。过去帧与 AdaBoot 的尝试不能作为已经解释 Isaac/MuJoCo 电机及闭链物理响应差异的证据。
