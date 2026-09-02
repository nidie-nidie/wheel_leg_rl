# 实验设计

主实验顺序：

1. Phase 0：机器人模型和运动学验证
2. Phase 1：不使用 RL 的 position-control bring-up
3. Experiment A：Pure Joint PPO 站立 baseline
4. Experiment C：不加入串联腿运动学先验的 CTS
5. Experiment D：Kinematic-Prior CTS
6. Experiment B：IK + Position Residual PPO baseline
7. Experiment E：如有需要，再做 VMC baseline

第一版任务：

```text
原地稳定站立
```

第一版 RL baseline 的 reward 要少而清晰，先不要塞很多复杂项。

## 当前有没有做消融

现在还没有真正开始四组消融实验。当前完成的是主 baseline 的工程框架和一次粗糙地形站立训练。

注意：2026-08-23 的旧训练使用的是轮端 `effort_limit=1.45 Nm` 的配置。2026-08-25 已按领控 9025 官方数据把轮端额定扭矩改为 `2.42 Nm`、额定转速改为 `51.31 rad/s`，所以旧 checkpoint 只能作为历史调试结果，不能作为当前参数下的正式 baseline。

后面每个消融实验都要单独建 run，并至少固定这些条件：

- 同一个 URDF / USD 资产版本
- 同一个环境数、训练 iteration 数和随机种子策略
- 同一套 reward 主体
- 只改变被消融的那一个模块，例如是否加入 kinematic prior、是否加入 IK residual

如果仍使用当前 PPO 配置，每组实验就是各训练 6000 个 iteration。

## 运动学更新后要不要重训

只改 `offset_leg.py` 的 4.3 运动学解算时，当前 Pure Joint PPO 站立环境不一定要重训，因为现在的 actor observation、action 和 reward 没有调用这个独立运动学模块。

但这次同时更新了轮端 9025 的 actuator 参数，这会改变训练环境里的轮端力矩和速度限制。因此下一次正式 baseline 建议重新跑一轮，不要直接拿旧 `model_5999.pt` 当最终结果。

后续只要实验里用到 IK residual、VMC baseline、Kinematic-Prior CTS 或任何基于 `L0/theta/JH` 的特征，就必须使用 4.3 版本重新训练，旧运动学下的 checkpoint 不再可比。

## 当前工程参考了什么

开源/已有部分：

- Isaac Lab 的 DirectRLEnv 任务组织方式。
- Isaac Lab 的 URDF 导入、场景配置、terrain generator、actuator 配置。
- RSL-RL 的 PPO runner、actor-critic 配置、checkpoint 和 TensorBoard logging。
- mevius2-master 作为“地形和训练工程应该长什么样”的参考对象，不是直接把它的任务代码搬过来。

我们自己搭的部分：

- `serial_leg_rl` 这个独立新工程结构。
- 串联腿 URDF 的仿真副本生成和 dummy link 处理。
- 轮腿机器人的 joint 分组、actuator 配置、初始姿态。
- 原地站立 DirectRLEnv：action、observation、reward、termination、reset。
- 粗糙地形站立任务配置。
- 独立运动学模块、几何检查脚本和单元测试。
- 中文文档和实验记录结构。

## 需要学习

- 什么是 baseline，为什么先做最朴素的 Pure Joint PPO。
- 什么是消融实验：每次只改一个变量，否则看不出收益来自哪里。
- PPO 的 `iteration`、`rollout`、`minibatch`、`epoch`、`checkpoint` 分别是什么意思。
- 怎么读 TensorBoard：reward 上升不等于动作可用，还要看 termination、速度、action rate 和视频。
- 怎么写实验记录：配置、现象、证据、下一步，而不是只写“好像变好了”。
