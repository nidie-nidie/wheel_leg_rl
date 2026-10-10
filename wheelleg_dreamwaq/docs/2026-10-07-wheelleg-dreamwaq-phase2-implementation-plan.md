# WheelLeg DreamWaQ Phase 2 Implementation Plan

日期：2026-10-07  
状态：Architecture v0.21 文档复核已通过，Phase 2 代码已实现；真实 Isaac 集成、最终独立代码复核和正式训练待完成

## 1. 目标和硬边界

本计划实现 Architecture v0.21 的第一版 DreamWaQ/CENet，并在代码复核通过后执行四次不同 seed、每次 1000 iteration 的正式训练，随后完成冻结的 deterministic Isaac 对照和现有八场景 MuJoCo sim2sim 评估。

以下语义不得修改：AssetBundleV2/USD、WheelOnlyCollisionV2、PhysicsV4 `0.005/4/0.02 s + 96/4`、ActionV1、CommandSamplingV2、ActorObsV1、CriticObsV1、NormalizationV2、RewardSchemaV2、termination、Phase 1R `FudanStyleDomainRandomizationV1`、closed-chain reset cache、MuJoCo XML/控制器/场景/失败阈值/排名规则。

## 2. 顺序和审查门

1. 修改 Architecture、handoff、README 和本计划。
2. 独立 agent 对文档做只读复核；有 P0/P1 时修正文档并重新复核。
3. 实现 schema/contract/config、history、CENet/inference、ActorCritic、storage、PPO、runner/checkpoint、train/play/export、deterministic Isaac evaluator、MuJoCo 接入。
4. 运行纯 Python 单元测试和静态 contract 测试。
5. 独立 agent 对全部代码和集成测试做只读复核；有 P0/P1 时修复并重新复核。
6. 复核通过后，在全新 Windows 子进程运行 2 iteration fresh、同规模/跨规模 resume、play/export、`evaluate_isaac.py` 和 TorchScript golden-vector 集成测试。
7. integration 若触发代码修改，重复纯测试和独立代码复核；无代码修改时直接进入正式训练门。
8. 执行四次 fresh 1000 iteration 正式训练，四个 seed 必须不同并写入 suite manifest。
9. 每个 checkpoint 独立导出 `actor.ts`、`policy_manifest.json`、`golden_vectors.pt`。
10. 用同一份 8-env evaluation reset cache 重新评估冻结的 Phase1RIsaacBaselineV1，并评估四个 DreamWaQ checkpoint；生成逐场景/aggregate、velocity MSE ratio 和 candidate-vs-baseline 结论。
11. 对所有 hard-valid 的四个导出策略运行相同八场景 MuJoCo sim2sim，生成逐场景 CSV、summary 和 ranking；context/velocity/baseline acceptance 失败的结果也必须保留和评估。

## 3. 实现文件

新增：

- `source/wheelleg_dreamwaq/wheelleg_dreamwaq/algorithms/dreamwaq/{cenet,actor_critic,storage,ppo,history_wrapper}.py`
- `source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/{dreamwaq_manifest,isaac_evaluation}.py`
- `source/wheelleg_dreamwaq/wheelleg_dreamwaq/training/dreamwaq_checkpoint.py`
- `source/wheelleg_dreamwaq/wheelleg_dreamwaq/deployment/{dreamwaq_inference,manifest,observation_adapter}.py`
- `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/agents/dreamwaq_cfg.py`
- `scripts/{train_dreamwaq,play_dreamwaq,export_dreamwaq_actor,evaluate_isaac,run_dreamwaq_training_suite}.py`

修改：

- `schemas/observation.py`：HistoryLayoutV1、velocity label 和 ProprioReconstructionTargetV1 单一事实来源。
- `schemas/manifest.py`：抽取独立 allow-list 的 base-task payload helper，Phase 1 hash 保持不变。
- `training/checkpoint.py`：仅公开可复用的 RNG/cache/LR helper，不放宽 Phase 1 validator。
- `agents/__init__.py`、`schemas/__init__.py`：导出新配置与契约。
- `sim2sim/mujoco/wheelleg_mujoco/{contract,runner,evaluation,versions}.py` 与 `scripts/evaluate_mujoco.py`：向后兼容 PPO V1，同时支持 DreamWaQ history artifact。
- `scripts/export_ppo_actor.py`：只增加独立 base-task identity，使 PPO/DreamWaQ 使用相同任务身份比较；不改变 PPO actor 数值。

`schemas/isaac_evaluation.py` 只保存 `IsaacEvaluationContractV1`、八场景、Phase1RIsaacBaselineV1 身份、MSE 聚合和 performance-key 纯函数；不得 import Isaac Lab。`scripts/evaluate_isaac.py` 负责 AppLauncher、环境、共享 reset cache、checkpoint 模型装载和 report I/O，不修改环境 command/randomization 定义。

依赖方向固定为 `schemas -> algorithms -> training -> scripts`；`algorithms/dreamwaq` 不依赖 task/env，`deployment` 不依赖 Isaac/RSL-RL，MuJoCo 不导入训练包内部类。

## 4. 核心实现验收

- history wrapper 缓存不可变 TensorDict snapshot；构造/reset 均是一次 discarded observation 加恰好一次 first policy observation。
- CENet、Actor、Critic、decoder 拓扑和参数量精确匹配 Architecture v0.21。
- `obs_groups` 在 config/runner 和 ActorCritic 两端精确拒绝任何额外 set/term。
- storage 只新增 target/mask，预分配后使用 `copy_`；inference tensor 不跨入 update 引用。
- DreamWaQ mini-batch 原样保留标准十项字段、`old_mu`、`old_sigma` 和同一 shuffle index。
- PPO 每个 mini-batch 只运行一次 encoder、一次 backward、一次 optimizer step，并执行完整 finite gate。
- checkpoint 使用三个独立 contract 和 `DreamWaQCheckpointMetadataV1`，与 Phase 1 checkpoint 双向拒绝；`EstimatorMonitorStateV1` 的 initial/completed rollouts/recent-10 必须保存和恢复。
- base-task contract 完整覆盖四个 Actor observation noise 幅度和七个 RNG stream。
- context final feature-std 和 velocity zero-baseline MSE ratio 按 v0.21 记录并验收，不进入 loss。
- deterministic Isaac 评估固定 seed `20261007`、8 env/八场景、NominalEvaluationProfileV1、`499` 个 action step/`9.98 s`、共享 cache 和 run-03 baseline 身份；第 499 step 的既有 timeout 是正常完成，禁止修改 termination 或 auto-reset 来凑足 500 个 Isaac step；active pre-action 样本按元素加权，policy performance 使用冻结 lexicographic key。

## 5. 测试顺序

```powershell
uv run pytest tests/unit -q
uv run pytest sim2sim/mujoco/tests -q
uv run pytest tests/integration -q
```

integration 必须包含独立子进程启动、异步 terminated/truncated reset、1-2 iteration fresh/save/load/resume、同规模与跨规模 cache provenance、`EstimatorMonitorStateV1` 中断等价、deterministic play、export、dynamic-batch TorchScript reload，以及 Phase1R baseline/DreamWaQ 共用 cache 的 `evaluate_isaac.py` smoke；评估测试必须断言第 499 action step timeout 被接受为完成、提前 timeout 和任意 termination 被判失败。

`golden_vectors.pt` 固定 CPU `torch.randn`、seed `20261007`、`float32 [32,125]`；eager 与 reloaded TorchScript 最大绝对误差不超过 `1e-7`，目录精确包含三件套。

## 6. 正式训练与 MuJoCo

正式 suite 使用 `rtx5070`、`fudan-v1`、四个 fresh 且互不相同的 seed、每次 1000 iteration。禁止 resume、warm-start 或从普通 PPO checkpoint 初始化完整 DreamWaQ learner。suite 必须在每个 run 前核对 source/dependency/base-task/algorithm/export fingerprint 一致。

正式训练完成后，每个 run 先通过 checkpoint/manifest/TensorBoard/finite 等 hard-validity 检查，再导出并进入 deterministic Isaac evaluation。context/velocity/baseline tripwire 是 acceptance gate，不是丢弃证据的 hard gate：失败 run 标记不通过，但仍进入 MuJoCo。MuJoCo 使用现有模型、50 Hz 控制、八场景、500 tick/场景和既有失败阈值；不为 DreamWaQ 修改任何参数。最终工件至少包括 suite manifest、共享 Isaac evaluation cache/contract、Phase1R baseline report、四个 DreamWaQ Isaac report、四个 training summary、四套 TorchScript 三件套、32 个 MuJoCo 场景 CSV、四个 MuJoCo evaluation summary 和 ranking summary。
