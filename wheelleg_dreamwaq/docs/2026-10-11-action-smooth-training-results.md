# Action Smooth 加倍：重新训练与 sim2sim 结果（2026-10-11）

结论：action_smooth 从 −0.01 加到 −0.02 有部分抑制抖动的效果，但没有解决左腿抽搐。新策略前进跟踪变差，停车换向时新增偏航超标。本次建议保留新旧模型作对照，暂不把相同配置扩大到四个种子。只有一个种子的实验结果，不能据此断言跨种子的平均效果或把抽搐根因归结为某一个奖励项。

## 实际执行与范围

- 独立配置 stop_reverse_smooth_v1；唯一有效训练配置差异为 action_smooth=−0.02。action_rate=−0.01，两个 vx 跟踪权重均为 2，其余训练条件相同。
- seed=41882943，256 环境、8 minibatches，从零训练 1000 iterations；不是旧 checkpoint 的 resume。
- training_summary：starting_completed_iterations=0、iterations_executed=1000、completed_iterations=1000、resumed_from=null。
- 训练用时 48.67 分钟；按 1800 秒间隔检查，hard_anomalies 为空，仍有 phi0_symmetry_degraded 警告。
- 架构 v0.27 §32/ADR071；实现提交 5fb657d。正式四种子评估入口保留原有约束，本次使用单种子诊断入口。
- USD、物理模型、PD/接触、动作、命令练习、随机化、地形、观测/history、算法参数未改。110 个受保护文件的 SHA256 在训练后复核一致。
- 新旧 Isaac 评估使用同一实际 reset cache 和评估条件；MuJoCo 使用同一模型、场景、时序、失败条件和评分。训练奖励身份及单种子上下文被单独核验，不能要求其完整合同哈希等于旧四种子合同。

主对照是同种子的旧 1000 轮模型。旧 1401 轮模型为用户此前观察到抽搐的补充参考，训练长度不同，不能替代主对照。

新 checkpoint SHA256：8BC8B90DDEAAA3DA4B0A7CCCB434C73C6FBA4DE9A2043B131F385876060B975F。
模型路径：E:\wheel_leg_rl-main\wheelleg_dreamwaq\logs\rsl_rl\wheelleg_flat_dreamwaq\2026-10-11_16-13-27_rtx5070_fudan-v1_dreamwaq-action-smooth2-run03-fresh1000-20261011-161144-seed41882943\model_999.pt。

## 左腿是否更平稳

相同 MuJoCo 初态、停车命令 [0,0,0.2]，各运行 10 秒；排除前 2 秒，统计 2–10 秒的 400 个控制帧。日志不增加物理步数或改变控制输出。下表是实际关节角速度 RMS，单位 rad/s。

| 关节 | 旧1000 | 新1000 | 相对旧1000下降 | 旧1401参考 |
|---|---:|---:|---:|---:|
| 左前关节 jIJ | 0.9137 | 0.7167 | 21.6% | 0.7204 |
| 左后关节 jIO | 0.5834 | 0.5195 | 10.9% | 0.6657 |

与旧 1401 轮相比，新模型左前关节 RMS 只下降 0.5%，左后关节下降 22.0%。因此不能告诉用户此前看到的左前抽搐已经明显消失。

左前目标的步间变化 RMS 从 0.04794 降到 0.03613 rad，左后却从 0.01896 增到 0.02292 rad。实际关节速度减小不等于每个关节的目标更平滑。

轮子目标步间变化 RMS：旧1000为 23.26/19.90，新1000为 23.12/19.73 rad/s，频繁切换基本保留。新模型右腿 jAB/jAG 在整个统计窗口的动作都处在下限，左/右 phi0 均值约 86.3°/110.6°，左右不对称仍在。右腿目标恒定不能当作整体策略动作平滑的证据。

停车速度绝对误差从 0.02498 降到 0.01626 m/s，但新模型仍有缓慢后漂；pitch RMS 从 1.41° 增到 1.90°。

数据与同尺度曲线：[stationary-comparison.json](../artifacts/phase2_dreamwaq/action-smooth2-run03-fresh1000-20261011-161144/stationary-comparison.json)，[stationary-comparison.png](../artifacts/phase2_dreamwaq/action-smooth2-run03-fresh1000-20261011-161144/stationary-comparison.png)。

## 八场景与停车换向验收

Isaac 八场景与 MuJoCo 八场景均全部跑完；分别为每场景 499 个有效评估步、500 个控制帧。完整存活不代表速度与停车换向性能全部通过。下表为实际 vx MAE，单位 m/s，越小越好。

| 仿真器 | 场景 | 旧1000 | 新1000 |
|---|---|---:|---:|
| isaac | 停车 | 0.01539 | 0.07564 |
| isaac | 低姿停车 | 0.02939 | 0.04990 |
| isaac | 高姿停车 | 0.02855 | 0.02597 |
| isaac | 前进 +1 m/s | 0.08890 | 0.19972 |
| isaac | 后退 −1 m/s | 0.27023 | 0.21324 |
| isaac | 组合指令 | 0.05621 | 0.14373 |
| mujoco | 停车 | 0.02727 | 0.01866 |
| mujoco | 低姿停车 | 0.13983 | 0.01088 |
| mujoco | 高姿停车 | 0.06278 | 0.01706 |
| mujoco | 前进 +1 m/s | 0.04396 | 0.12454 |
| mujoco | 后退 −1 m/s | 0.10337 | 0.08112 |
| mujoco | 组合指令 | 0.07320 | 0.07101 |

新模型 MuJoCo 的停车/前后移动 vx 门槛通过；该门槛只检查三个停车场景 ≤0.10 m/s 和前后移动 ≤0.20 m/s，不检查全部转向或偏航表现。Isaac 后退 vx MAE=0.21324 m/s，仍未通过 0.20 m/s 门槛；前进虽然未超过现有门槛，但由 0.08890 增至 0.19972 m/s，已接近上限。

MuJoCo 前进 vx MAE 从 0.04396 增至 0.12454 m/s，左转/右转/组合指令的 yaw-rate MAE 也从 0.13346/0.17641/0.19558 增至 0.17985/0.20074/0.29164 rad/s。前进场景的 yaw 漂移 RMS=1.540 rad、最大绝对漂移=2.614 rad，不能仅根据 vx 判断直行稳定。

四个停车换向专项都运行至 500 帧，但新模型性能均未通过。阶段编号从 0 开始。

| 专项 | 旧1000性能 | 新1000性能 | 新模型未通过项 |
|---|---|---|---|
| forward_stop_reverse | 通过 | 未通过 | 阶段1: stop_yaw_mae；阶段3: stop_yaw_mae |
| reverse_stop_forward | 通过 | 未通过 | 阶段3: stop_yaw_mae |
| left_stop_right | 未通过 | 未通过 | 阶段2: reverse_response_time；阶段4: reverse_response_time；阶段5: stop_yaw_mae |
| right_stop_left | 未通过 | 未通过 | 阶段2: reverse_response_time；阶段4: reverse_response_time |

前后移动方向仍能正确切换；新增失败主要是停车后的 yaw-rate 误差，不是完全不会前进/后退。转向换向专项仍未能及时进入并保持允许误差范围。

详细数据：[movement-comparison.json](../artifacts/phase2_dreamwaq/action-smooth2-run03-fresh1000-20261011-161144/movement-comparison.json)，[Isaac summary](../artifacts/phase2_dreamwaq/action-smooth2-run03-fresh1000-20261011-161144/isaac_evaluation/summary.json)，[MuJoCo summary](../artifacts/phase2_dreamwaq/action-smooth2-run03-fresh1000-20261011-161144/mujoco_evaluation/summary.json)，[停车换向 summary](../artifacts/phase2_dreamwaq/action-smooth2-run03-fresh1000-20261011-161144/mujoco_command_practice/summary.json)。

## 实现、导出与训练曲线核验

训练前单元测试 186 项通过，新配置的实际 Isaac fresh/resume/play/export 集成用例通过。普通 resume 继续拒绝跨奖励配置恢复。导出 TorchScript batch 1/3/32 输出形状正确，golden-vector 最大误差为 0；新导出 actor/golden/checkpoint 文件哈希和评估报告/轨迹哈希再次核对。

训练最后 100 轮（900–999），去除权重和 dt 后的动作二阶差分平方和均值由 8.36314 降到 5.32369，下降 36.3%。这说明动作变化惩罚确实影响了训练行为，但该值涵盖随机化和探索，不能代替部署运动验收。同期 vx MAE 从 0.14918 增到 0.15841，phi0 左右差和动作饱和比例也增加。

记录：[experiment-manifest.json](../artifacts/phase2_dreamwaq/action-smooth2-run03-fresh1000-20261011-161144/experiment-manifest.json)，[final-verification.json](../artifacts/phase2_dreamwaq/action-smooth2-run03-fresh1000-20261011-161144/final-verification.json)，[训练曲线](../artifacts/phase2_dreamwaq/action-smooth2-run03-fresh1000-20261011-161144/training-curve-comparison.png)。
TensorBoard 对照：[6009 scalars](http://127.0.0.1:6009/#scalars)。

结果整理时修正了编排脚本的一个对照断言：它错误地要求包含训练奖励身份的完整评估合同相等。现改为分别核验各自完整身份，并逐项比较实际评估条件；原 stderr 保留。此修正未改变训练结果、评估条件或物理参数，详见 experiment-manifest 的 comparison_check_correction。

## 后续判断

本次只完成一个种子的重新训练与诊断，未进行四种子的统计验证。保留独立的新配置和 checkpoint；当前证据不足以建议按此配置扩大训练或把它替换为最终模型。下一步应优先解释右腿持续下限、轮子快速切换与左腿修正之间的关系，再决定是否分别约束腿/轮动作或调整奖励；这些属于后续实验，本次没有继续改参数或追加训练。
