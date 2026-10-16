# DreamWaQ 腿部抽搐 debug 索引

更新：2026-10-13。用途：接续调查时先查看已有证据，避免把已确认现象反复当成新发现。正式架构和生产语义不在此修改。

**目前已确认：右腿目标长期固定、轮反馈参与左腿周期修正、六维平滑罚分主要来自轮子；现有加速度试验的罚分和增益主要受右腿归一化超额驱动。尚未确认抽搐的唯一根因或已经有效的生产修复。**

## 记录入口

以下路径相对 wheelleg_dreamwaq/artifacts/debug/sim2sim：

| 记录 | 对象与已确认内容 | 未解决内容 |
|---|---|---|
| joint-telemetry-20261011-141922/diagnostic-notes.md | run-03 1401 轮键盘记录；左腿目标振荡，右腿两动作裁成 -1；电机实际力矩符合命令 | 最初起振原因 |
| 同目录 observation-feedback/report.md | history/动作时序和归一化核对；估速有偏差；离线输入依赖指向轮速/轮动作历史 | 离线依赖不是闭环因果结论 |
| 同目录 observation-feedback/smoothness-and-pitch-notes.md | 平滑罚分确实加入；右腿指令恒定导致其动作平滑罚分为 0；总罚分轮子主导 | 加权能否修复 |
| leg-jitter-indicators-20261011-182657/report.md | 左腿约 8.8 Hz 周期运动；动作变化和实际角加速度分别衡量指令与机械响应 | 健康阈值及根因 |
| night-model-leg-comparison-20261011-v1/report.md | 旧 PhysicsV4 模型四腿安静但溜车；换当前物理很快失败 | 不能从安静推断控制合格；不与当前模型作单变量归因 |
| reversal-noise-diagnosis-20261011-v1/report.md | 只数目标换向对探索采样敏感、忽略幅度；旧 baseline 已有右腿饱和 | 实际训练因果占比 |
| leg-motion-metric-analysis-20261012-v1/report.md | 推荐实际四腿角加速度指标；九条旧轨迹右腿目标固定；右腿小值不是健康阈值 | 验证后的预算值 |
| leg-acceleration-budget-search-20261012-v1/results.md | baseline/80/85/90 匹配 200 轮试验；85% 只是诊断候选；增益基本封顶，均未完整通过控制验收 | 有效最终配置 |
| 上述目录 85/feedback-root-cause-20261012-v1/report.md | 42 个闭环分支；轮换向依赖 pitch 角速度，左腿依赖轮动作历史；纠正估速不能解决主要抖动 | 训练形成与 sim2sim 诱发/增强的区分 |
| 上述目录 saturation-reward-audit-20261012-v1/report.md | 新模型八场景右腿饱和；站立 smooth 99.7% 来自轮子；训练加速度罚分约 91% 来自右腿；gain 长期为 1 | 饱和来源及单项因果 |
| right-leg-saturation-lineage-20261013-v1/report.md | 本次新增：18 个现有 checkpoint 溯源；原始第 101 与 201 轮模型之间，当前 MuJoCo 停车行为变成持续右腿饱和；早于减抖试验；B0 精确追溯旧饱和参考 | 精确首次训练 iteration 和具体学习诱因 |

| split-action-smooth-20261013-v1/failure-analysis-20261013-v1/report.md | 分项平滑 fresh1000 后续诊断：固定侧由右变左；前进/后退实际加速度恶化，停车平均有所改善；最终新增力矩饱和主要在轮子；九个快照、十八场 MuJoCo 追踪及原数据核验 | 腿权重加倍的单项训练因果、轮腿直接因果及有效生产修复 |

## 当前调查应从哪里接着走

第 101–201 轮对应原始 1000 轮训练 run-03 / seed41882943，而不是 85% 的新增 200 轮。相同初态和指令的 MuJoCo 检查中，第 101 轮右腿目标还变化，第 201 轮在运行 0.08 s 后两个右腿目标均持续固定。不要把第 101 轮当健康参考，它的速度和腿运动仍很差。

当前 85% 初始化链：
原始 run-03 / 1000 轮 → 原奖励 baseline 另训 200 轮 → 85% 另训 200 轮；后两段使用 weights-only、新优化器和新环境。模型路径和 SHA256 见新溯源报告及 checkpoint-inputs.json。

继续调查时，优先查早期学习目标/动作表示如何接受这类模式，以及 MuJoCo 反馈如何触发或维持它。已确认时序/归一化和电机执行公式的检查，在没有新证据时无需全部重做。

旧模型、原奖励 baseline、新加速度试验应标明各自身份；不要混用百分比、窗口或训练/部署统计。已有报告对右腿参考不健康、gain 封顶和指标局限已明确提示。

## 本次结果入口

[新增溯源报告](../artifacts/debug/sim2sim/right-leg-saturation-lineage-20261013-v1/report.md)  
[checkpoint 对照图](../artifacts/debug/sim2sim/right-leg-saturation-lineage-20261013-v1/checkpoint-lineage.png)  
[独立核验](../artifacts/debug/sim2sim/right-leg-saturation-lineage-20261013-v1/verification.json)

本次未训练、未启动 Isaac、未改生产代码/奖励/模型/观测/控制参数，没有调用子 agent。


## 2026-10-13：分项平滑训练实验

用户现已授权修改训练奖励：腿 action_smooth=-0.02，轮=-0.01，使用独立 profile从零训练1000轮。旧参数和模型保留原语义；先文档复核、再代码复核与验证，运动验收只在MuJoCo。上文“未训练/未改生产”属于早先调查记录，不代表此实验后续状态。

[实施计划](superpowers/plans/2026-10-13-split-action-smooth.md)；[debug计划](../artifacts/debug/sim2sim/split-action-smooth-20261013-v1/plan.md)。当前状态：文档/代码复核和实现已完成，fresh1000 已结束；候选未通过完整 MuJoCo 验收，未自动追加腿 −0.04。原训练结果见 [results.md](../artifacts/debug/sim2sim/split-action-smooth-20261013-v1/results.md)。


## 分项平滑实验失败原因：本次后续诊断

[完整诊断报告](../artifacts/debug/sim2sim/split-action-smooth-20261013-v1/failure-analysis-20261013-v1/report.md)；[独立数值/哈希核验](../artifacts/debug/sim2sim/split-action-smooth-20261013-v1/failure-analysis-20261013-v1/verification.json)；[目标与实际运动图](../artifacts/debug/sim2sim/split-action-smooth-20261013-v1/failure-analysis-20261013-v1/target-and-motion.png)。

这次实际变量是腿二阶动作差分权重 −0.01→−0.02，等权拆分本身与旧公式等价。没有发现重复扣罚、history/reset 错误或新旧部署参数不一致。新候选仍固定一侧目标、另一侧周期修正，只是由右固定换成左固定；完成第 201 轮的已保存快照已经出现左固定。固定侧在完整八秒稳态中目标恒定、smooth 为零。

1 ms 正式一秒窗口统计：停车四腿平均加速度 RMS 28.399→25.339（改善10.78%）；前进4.934→20.310；后退16.282→24.102。三场景平均恶化40.58%，不是全场景恶化。新增总力矩饱和主要来自轮电机，最终停车/前进四腿力矩均未达到限幅；前后专项0/4的新增失败主要为停车偏航，不能说所有速度阶段失败。

此次只分析已有训练日志/原轨迹并在 MuJoCo 回放九个已有 checkpoint、十八场；没有启动训练或 Isaac、没有更改生产代码/奖励/物理参数。一个已有 agent 串行复核，24个现有分项奖励测试通过，十二个冻结源文件哈希不变。已确认奖励代理量允许长期固定一侧，尚不能把 −0.02 或轮反馈认定为唯一训练根因。不要直接重复“继续加大 smooth 就会修复”的尝试。
