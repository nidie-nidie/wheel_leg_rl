# 动作空间

主线动作空间：

```text
[jIJ_des, jIO_des, jAB_des, jAG_des, left_wheel_vel_des, right_wheel_vel_des]
```

腿部动作是电机侧 PD / MIT 控制模式下的位置目标。
轮端动作是速度目标。

第一版站立 baseline 的动作链路不使用 IK / VMC：

```text
policy action -> 缩放后的主动关节位置目标 -> 电机控制
```

动作缩放、默认关节角、限幅都必须写成配置参数，不要硬编码在环境逻辑里。

## 需要学习

- policy 输出通常是归一化 action，不是直接裸关节角。
- action scale、default joint position、joint limit 三者的关系。
- position-control RL 和 torque-control RL 的差别。
- 为什么 action rate penalty 能减少高频抖动。
- 轮端速度目标和腿部位置目标混在一个 action 向量里时，量纲和缩放必须分开记录。
