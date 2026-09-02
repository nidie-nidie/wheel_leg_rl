# Sim2Real 记录

最终部署目标：

```text
机载观测历史 -> Student encoder/actor -> 主动关节位置目标
```

运行时不应该依赖：

- Teacher
- IK
- VMC
- Jacobian torque mapping
- 仿真真值
- 运动学先验输入

MuJoCo Sim2Sim 等 Isaac Lab 里有可用 Student policy 之后再做。现在只在架构里预留。

## 需要学习

- sim2sim 和 sim2real 的区别：前者换仿真器，后者上真实硬件。
- 为什么要先有可播放的 policy，再做 MuJoCo sim2sim。
- 传感器延迟、电机延迟、通信周期、控制频率对策略的影响。
- domain randomization 的边界：随机化能提升鲁棒性，但不能替代正确的模型和限幅。
- 上实车前必须做小幅动作、低增益、急停和限位保护。
