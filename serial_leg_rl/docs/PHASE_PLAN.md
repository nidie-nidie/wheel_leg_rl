# 分阶段计划

## Phase 0：模型和运动学

目标：不启动 Isaac Sim，也能验证 URDF 几何和运动学公式。

已做：

- 新建干净工程骨架
- 保持原始 URDF 不动
- 建立中文文档
- 按 HTML 4.3 实现偏置闭链运动学模块
- 实现 URDF 几何抽取和一致性检查
- 实现 FK / IK / Jacobian 离线测试
- 生成仿真用 URDF 副本和 USD

下一步：

- 继续核对 Isaac / PhysX 中闭链约束的表达方式
- 用可视化和仿真 forward 继续检查左右腿方向、轮轴位置和初始姿态

## Phase 1：Position-Control Bring-Up

目标：不训练 RL，只验证仿真机器人能稳定跟踪主动关节位置目标。

需要验证：

- 四个腿部主动关节方向
- 轮端速度方向
- PD / MIT 控制参数
- 关节限幅
- 力矩限幅
- 初始站立姿态
- 仿真不会爆炸或严重穿模

## Phase 2：Pure Joint PPO Standing

目标：只用实机可用观测，训练原地站立。

不使用：

- IK action path
- VMC
- Teacher
- privileged kinematic prior

## Phase 3：Asymmetric Actor-Critic

目标：Actor 仍只看 Student observation，Critic 可以看 privileged information。

重点检查：

- privileged information 没有进 Actor
- 导出部署模型时只导出 Actor / Student

## Phase 4：CTS

先做不含运动学先验的 CTS，再做 Kinematic-Prior CTS。

关键对比：

```text
CTS without kinematic prior
vs
Kinematic-Prior CTS
```

两者应尽量保持相同 reward、action、terrain、randomization 和网络规模。

## 学习路线

- Phase 0 学 URDF/USD、闭链运动学、Jacobian、几何测试。
- Phase 1 学 actuator、PD/MIT 控制、关节方向、限幅和延迟。
- Phase 2 学 PPO、reward、observation、checkpoint、TensorBoard。
- Phase 3 学 asymmetric actor-critic、privileged information 和部署边界。
- Phase 4 学 CTS、teacher/student、latent、消融实验设计。
- Sim2Real 阶段学 domain randomization、sim2sim、实机标定、安全保护。
