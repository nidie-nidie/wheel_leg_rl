# 停车/换向练习与速度奖励：四种子训练和 sim2sim 结果

2026-10-11；Architecture v0.26 §31 / ADR070；stop_reverse_v1；PhysicsV5。

四个不同种子均从零完成 1000 iteration，所有导出、Isaac 和 MuJoCo 评估均执行完毕。全部固定场景和动态场景在测试时长内存活，但四个候选都未通过完整性能目标。run-03 最适合继续诊断停车和前后速度跟踪；它不是已经全部验收通过的策略。

## 1. 实际实施和复核

按用户要求先提交原工作区到 GitHub main，再写文档、调用一个文档 reviewer、实施代码、调用一个代码 reviewer，最后验证和训练。没有并行委托实施。

- 原工作区提交：c5fbc91cff1091bdacfc9158d8b0834e919288e3，旧提交历史完整保留。
- 文档提交：e27326a；代码和训练前验收提交：aaa5ac583f449d1595a1c1dd7247bb7db4b7c685。
- 分支：feat/stop-reverse-v1-20261011。
- [设计](superpowers/specs/2026-10-11-stop-reverse-speed-tracking-design.md)、[实施计划](superpowers/plans/2026-10-11-stop-reverse-speed-tracking.md)。
- [文档复核](2026-10-11-stop-reverse-document-review.md)、[代码复核](2026-10-11-stop-reverse-code-review.md)。全部 P0/P1 已关闭；第二次换向的验收遗漏已修复并回归。
- [训练前验收](2026-10-11-stop-reverse-pretraining-acceptance.md)：主单测 181 passed、MuJoCo 单测 56 passed；真实 PhysX 230 checks 全通过；fresh/resume、跨环境数 resume、固定命令 play、实际导出与加载均留证。

实际行为变化只有新训练 profile 的停车/换向练习，以及 tracking_vx、tracking_vx_enhance 权重各由 1 改为 2。yaw 跟踪权重未增加。tracking_vx_enhance 仍按原定义惩罚较大的速度误差，函数和符号未改变。

练习使用真实完成的控制步：原命令 2 秒 → 停车 1 秒 → 反向 2 秒 → 停车 1 秒 → 原方向 2 秒 → 停车至 episode 结束。高度保持初始采样值。stand 类命令全程停车；其他命令按原采样类别练习。命令切换不 reset history。

保留原 Fudan 随机化、平地、USD/XML、真实 PD/effort/阻尼/armature/接触/闭合链、动作和命令范围、观测/CENet/PPO/AdaBoot 及 termination。使用已经删除人工速度制动力矩的 PhysicsV5。legacy_v1 默认 profile 仍保留。

## 2. 正式运行

配置为 rtx5070、256 环境、8 mini-batches，四组只有种子不同，均未 resume 旧策略。每组监控初始、30 分钟和结束状态；训练期间没有改源代码或参数。

| 组 | 种子 | 完成 iteration | 训练耗时 | 最后训练日志 vx MAE，m/s |
| --- | ---: | ---: | ---: | ---: |
| run-01 | 1066832402 | 1000 | 45.45 分钟 | 0.1471 |
| run-02 | 272508862 | 1000 | 45.36 分钟 | 0.1727 |
| run-03 | 41882943 | 1000 | 45.39 分钟 | 0.1628 |
| run-04 | 638953086 | 1000 | 45.34 分钟 | 0.1638 |

最后训练日志混合了随机化、不同高度和练习阶段，不等同于固定命令验收。

监控未发现非有限数值或硬异常；四组均记录 phi0_symmetry_degraded 警告。该警告没有隐藏或当作通过项，应在后续分析中保留。

实际启动命令：

```powershell
.venv/Scripts/python.exe scripts/run_dreamwaq_training_suite.py --mode formal --profile rtx5070 --task-profile stop_reverse_v1 --candidate-only --runs 4 --iterations 1000 --monitor-interval-seconds 1800 --output E:\wheel_leg_rl-main\wheelleg_dreamwaq\artifacts\phase2_dreamwaq\training-suite-20261011-stop-reverse-v1
```

## 3. 固定场景与导出验收

- Isaac：4 × 8 = 32 个完整场景，每场 499 控制步，9.98 秒。
- 原正式 MuJoCo：4 × 8 = 32 个完整场景，每场 500 步，10 秒。
- 独立停车/换向 MuJoCo：4 × 4 = 16 个完整场景，每场 500 步，10 秒。
- 每组 export 的 actor.ts、policy_manifest.json、golden_vectors.pt 三件套完整；batch 1/3/32 检查全部通过，TorchScript golden 最大绝对误差四组均为 0。
- 四组 context 和速度估计器门槛均通过。Isaac 归一化 estimator MSE/zero-baseline ratio 分别为 0.009730、0.014682、0.009288、0.011415，低于既定 0.8 门槛。它不是下面物理单位速度 MAE。
- 全部报告、CSV、export 和 checkpoint 的记录哈希已独立核验。训练源代码指纹未改变。

完整存活只证明这些测试时长内没有触发既定失败条件；不能代替停车、速度跟踪、方向保持等性能验收。

MuJoCo 固定场景 vx 平均绝对误差，单位 m/s。前进目标 +1，后退目标 -1；停车目标 0。名义/低/高高度分别为 0.20/0.17/0.23 m。原固定场景统计包含启动过程，未替换其窗口。

| 组 | 名义停车 | 低高度停车 | 高高度停车 | 前进 | 后退 |
| --- | ---: | ---: | ---: | ---: | ---: |
| run-01 | 0.0789 | 0.1515 | 0.0445 | 0.0892 | 0.1224 |
| run-02 | 0.1111 | 0.0955 | 0.0961 | 0.1285 | 0.2725 |
| run-03 | **0.0273** | 0.1398 | 0.0628 | **0.0440** | **0.1034** |
| run-04 | 0.0959 | 0.1358 | 0.0257 | 0.0487 | 0.2050 |

预先定义的停车门槛是三个高度 vx MAE 均不超过 0.1，前进/后退均不超过 0.2。run-01、03 在低高度停车失败；run-02 在名义停车和后退失败；run-04 在低高度停车和后退失败。

Isaac 八场景也不是全过：四组后退误差分别为 **0.2457、0.2963、0.2702、0.2614 m/s**，全部高于 0.2。run-02 名义停车 0.1026 也超过门槛。因此，源策略在训练仿真器中的跟踪仍有不足，不能把所有问题归为 MuJoCo 迁移差异。

run-03 固定左右转向的 yaw MAE 分别为 0.1335、0.1764 rad/s。其余场景及方向漂移、动作/力矩饱和、姿态和闭合链误差见原始报告。

![四组固定命令速度轨迹](../artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/figures/fixed-velocity-comparison.png)

## 4. 停车与换向专项结果

动态诊断采用初始 vx ±0.5 m/s 或 yaw ±0.6 rad/s，高度 0.20 m，依照六阶段练习。每个稳态窗口排除阶段开始后的 0.5 秒；失败或不完整片段不能通过。停车标准：稳态 vx MAE ≤ 0.1 m/s，yaw MAE ≤ 0.15 rad/s。换向标准：两次换向阶段（stage_index 2/4，即第 3/第 5 段）都必须在 1 秒内完成连续 0.2 秒误差 ≤ 0.15 的确认。

| 组 | 前进→停车→后退 | 后退→停车→前进 | 左转→停车→右转 | 右转→停车→左转 |
| --- | --- | --- | --- | --- |
| run-01 | 未通过，停车误差 | 未通过，停车误差 | 未通过 | 未通过 |
| run-02 | 未通过，停车误差 | 未通过，停车误差 | 未通过 | 未通过 |
| run-03 | **通过** | **通过** | 未通过，换向稳定响应 | 未通过，换向稳定响应 |
| run-04 | 未通过，停车误差 | 未通过，停车误差 | 未通过 | 未通过 |

run-03 的两条直线场景中，六个停车阶段的稳态 vx MAE 为 0.0337–0.0735 m/s，yaw MAE 为 0.0960–0.1318 rad/s。四次前后换向确认时间为 **0.68、0.58、0.54、0.70 秒**。

这表示它进入了约定的停车误差范围，仍会有运动。六段停车过程中，世界 XY 平面 COM 总路程为 **0.2228–0.2914 m**，包含开始减速和之后的漂移；不是瞬时刹停距离，也不是终点净位移。

左右换向的转动方向能切换，但角速度有持续振荡，难以连续 0.2 秒留在误差带内。左转起始场景的两次确认均未到达；右转起始场景一次为 1.72 秒，另一次未到达。因此不是“完全不会转”，而是稳定响应没达标。

![run-03 停车与换向轨迹](../artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/figures/run-03-command-practice.png)

## 5. 候选选择与下一步判断

原八场景综合排序使用原 cost，越低越好，排序仍是 run-02、run-01、run-04、run-03；cost 分别为 5.1392、5.5148、5.5738、6.1786。该综合排序未改公式，不把 run-03 重标为正式综合第一名。

针对用户现在关心的停车和前后速度跟踪，**run-03 是本轮最有用的诊断候选**：名义停车和前后速度误差最好，并且两个直线练习都通过。四组完整性能验收均 false，不能将任一组当作全部达标的最终策略。

要继续达到全部目标，建议继续做有针对性的训练调整。优先分析 run-03 的低高度停车、左右角速度振荡，以及 Isaac 后退偏差，再决定相应命令练习或奖励的调整。当前结果不支持仅靠再加训练轮数就一定修好，也不支持立即扩大随机化或加入地形。先把平地停车、前后跟踪和转向稳定性做好，再提高训练难度。

旧 PhysicsV4 策略与当前 PhysicsV5 不能用旧 baseline 合约直接比较；历史 baseline 和旧 Phase 2 acceptance 均为 null/not_comparable。本轮也没有独立的“只加练习”或“只加权重”对照，不能量化每个改动的单独贡献。

## 6. 工件与复现范围

结果目录：[training-suite-20261011-stop-reverse-v1](../artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/)。

- [独立核验总表](../artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/results-summary.json)。
- [套件 manifest](../artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/training-suite-manifest.json)、[原八场景排序](../artifacts/phase2_dreamwaq/training-suite-20261011-stop-reverse-v1/mujoco-ranking-summary.json)。
- exports/run-01..04：可部署 actor、manifest 和 golden 三件套。
- training-records/run-01..04：原 run_manifest 与 training_summary 的逐字节副本。
- monitor/run-01..04：初始、半小时和结束监控。
- isaac_evaluation/run-01..04：当前物理配置的八场景报告及共享 reset cache。
- mujoco_evaluation/run-01..04：原八场景报告和 32 个 CSV。
- mujoco_command_practice/run-01..04：动态报告和 16 个 CSV。
- figures：直接由对应 CSV 绘制的完整轨迹图，没有平滑处理。

原始训练 checkpoint、TensorBoard 和运行日志保留在本地 logs/rsl_rl/wheelleg_flat_dreamwaq；不随源码提交。每个 checkpoint 的本地绝对路径和 SHA256 均由工件记录，四组可推理导出包随结果提交。跨机器读取时需解析已有报告中的本机绝对路径。

冻结标识：

- Architecture SHA256：236CF37B870909323D8D87351A27E699EBCE736C138460DEFCEA79834B75452A。
- 训练源代码指纹：274842FD7E9C4EE34907BBC1AE92BAE338F8770D8E022D7B8E20BD9D460A2F1D。
- 训练配置指纹：D3FA8A2064785C314BDD0F05BD6BD942ADB424937FA98572CD4CA93C787C1318。
- 正式 MuJoCo XML SHA256：691C607271ED85EFFA388237E00A7B8F8F41CB66F6BA9C49EA64ABD186D803D1。
