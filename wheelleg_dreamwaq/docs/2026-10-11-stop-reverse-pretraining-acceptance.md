# 停车/换向与速度奖励：训练前验收

2026-10-11；Architecture v0.26，stop_reverse_v1，PhysicsV5。

工作区原状态已提交并推送到 GitHub main：c5fbc91cff1091bdacfc9158d8b0834e919288e3；之前两条提交仍完整保留。新实现位于 feat/stop-reverse-v1-20261011。

文档与代码均由各一个 agent 顺序复核，全部 P0/P1 已关闭。代码复核的第二次换向遗漏已修复；旧条件误报 true，新条件判 false，有对应回归测试。

验证结果：

- 主工程单测：181 passed；MuJoCo：56 passed。
- 真实 PhysX：230 checks，全部通过；旧命令 reward、下一帧新命令、history、独立真实步计数、partial reset、timeout、额外读取均留证。
- 新 profile 便携 fresh：2 updates；32/16 环境同/跨规模 resume：总计 3 updates，只执行新增的 1 update；自动识别保存 profile。
- 显式改为 legacy_v1 的 resume 被拒绝。
- fixed-command play：160 steps，schedule disabled，finite。
- 导出三件套与实际 MuJoCo loader 可用，TorchScript golden 误差不超过 1e-7，batch 1/3/32 输出维度正确。
- 当前 PhysicsV5 的 IsaacEvaluationV2 八场景报告可生成，真实物理单位 MAE 与旧 estimator 定义分别保留；历史 baseline 为 null/not_comparable。
- 独立动态 MuJoCo 四场景 smoke 可生成；性能不冒充通过。

原始工件位于 `artifacts/phase2_dreamwaq/stop-reverse-acceptance-20261011-v1/`。正式训练尚未在此记录中开始；上述小测试权重不进入正式四次结果。

正式执行命令：

```powershell
.venv/Scripts/python.exe scripts/run_dreamwaq_training_suite.py --mode formal --profile rtx5070 --task-profile stop_reverse_v1 --candidate-only --runs 4 --iterations 1000 --monitor-interval-seconds 1800
```

训练中冻结代码和架构；只有种子不同，均从零开始。维持 FudanStyleDomainRandomizationV1、平地、PD/effort/阻尼/armature、动作/命令范围、观测/history/CENet/PPO/AdaBoot 和 termination。全部训练后完成每组 export/golden、当前 Isaac、原八场景 MuJoCo、独立停车/换向 MuJoCo 与四组排名。性能未达标时仍保存全部评估，不中途改参数。
