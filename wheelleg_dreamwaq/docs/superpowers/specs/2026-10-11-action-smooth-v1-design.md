# 动作平滑加倍与一种子从零训练

用户已批准 action_smooth=-0.01 → -0.02，先使用 run-03 相同种子 41882943 从零训练 1000 轮；不自动启动四组。

新增 stop_reverse_smooth_v1，相对 stop_reverse_v1 唯一有效环境配置差异为 action_smooth=-0.02。legacy_v1、stop_reverse_v1 保留原行为，action_rate=-0.01、两个 vx 权重=2。命令练习、数学、物理、随机化、USD/XML、动作、history、CENet、PPO、reset、termination 保持原实现。

已保存的完整 task contract 通过相同练习 schedule 和 reward_weights.action_smooth=-0.01/-0.02 重建 profile；未知值拒绝。普通 resume 禁止跨奖励配置，本实验不使用 --resume，网络、优化器与 RNG 从种子初始化。

修改 training_profiles.py、train_dreamwaq.py CLI choices、run_dreamwaq_training_suite.py 的枚举/候选模式/专项条件。正式四组套件门槛保持 4×1000。扩展 profile 唯一差异、实际奖励贡献、旧/新配置重建与冲突拒绝测试，参数化现有真实 Isaac fresh/resume/play/export/golden 集成测试。

Architecture v0.27 §32 留证，§31 保留原配置。CPU 与真实 Isaac 新 profile 验收通过后启动 rtx5070（256 env、8 mini-batches）、1000 iteration、seed41882943，新独立 run，无 --resume。训练冻结代码/架构，1800秒监控，保留旧工件。

一种子八场景 MuJoCo 采用独立诊断入口，仅改 suite 模式/数量和脚本根路径，保存 diff/hash；原模型验证、初态、八场景、tick、metrics/failure/report 相同，不冒充正式四组。完成导出/golden、Isaac 当前八场景、MuJoCo 八场景、停车/换向专项及固定停车逐腿/轮子运动对照。

主对照为旧 stop_reverse_v1 seed41882943 的1000轮模型；1401轮模型作额外参考。同种子不保证跨源码版本训练逐位一致。按实际抖动、车速和稳定性判断，不能把平均奖励比例当作梯度或成功保证，性能失败仍保留全部报告。

自行复核：授权、唯一配置差异、旧模型兼容、严格续训、种子/轮数、评估语义与验收均明确。本次不调用未获授权的 agent。
