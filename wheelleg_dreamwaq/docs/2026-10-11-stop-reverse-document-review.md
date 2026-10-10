# 停车/换向与速度奖励：独立文档复核

日期：2026-10-11。复核者：/root/stop_reverse_document_review。范围：只读文档与本地源码；未实施、训练或调用子 agent。
结论：最终无剩余 P0/P1 阻塞，可进入实施；代码和性能尚未验收。

| 文档 | 复核 SHA256 |
|---|---|
| Architecture v0.26 | E81D66D8B1FE446A8B38DF433C857474B558F3DB87A49D1F39097A39650602F3 |
| 停车/换向设计 | B62F938661E4CAD31492856A50A7CD95EF8181ACB49037E61E7EC4CD487761B0 |
| 实施计划 | CC14CB945310565E05006260FB513434A161923F28D7E32D9346A305CEDDC1E8 |

已关闭的发现：
1. P1：原 schemas/manifest.py:270 与 dreamwaq_manifest.py:235 仅 asdict(commands)。现要求完整阶段 payload/hash，关闭时 null，MuJoCo 独立核验。
2. P1：export_dreamwaq_actor.py:77–89 原未保存权重。现要求 training_reward_weights 完整明写并保持 base-task hash。
3. P1：DirectRLEnv:395–410 / env.py:916–923 会在返回前 partial reset；observations.py:115–120 / normalization.py:59–60 的 critic 速度已归一化且裁剪。现规定 act 前物理单位 state 的 vx/wz、active_before、排除 postreset；estimator 数学不变。
4. P2：run_training_suite.py:27–48、242 漏 reward/estimator 标签。补已存在的可选 Reward/*/Loss/*，不增加 PPO 必需标签。
5. P2：metrics.py:94–112 无世界位置；现明确停车段 COM 世界 XY 累计路程及净位移，含段首边界；响应按 10 个连续有效样本的窗口结束；未完成段不能判成功。
6. P2：MuJoCo runner.py:96–98 第500步仍生成下一帧；明确499/500/501系数0、末阶段持续保持并进入hash。

确认：真实练习计数不依赖 RSL 随机 episode_length_buf；reward 对旧命令、下一观测对新命令；partial reset 只清对应行并同步缓存；命令变化不清 history；vx 两项1→2其余公式/权重不变；保存 profile 完整验证后才关闭固定运行的课程；PhysicsV5候选V2不冒充旧baseline通过；动态MuJoCo日志区分执行与下一观测命令；原八场景及动力学不变；四次串行新训练只换种子，性能失败不省略其余sim2sim。

实施之后仍必须完成单元测试、一次独立代码复核、真实Isaac probe、fresh/resume/play/export/golden、动态诊断与正式四次训练。
