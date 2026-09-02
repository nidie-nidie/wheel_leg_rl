# 原地站立任务规格

第一版 RL 任务只做：

```text
原地稳定站立
```

暂时不做速度跟踪、跳跃、复杂地形和 Sim2Sim。

## 目标

让机器人在平地上保持：

- body roll / pitch 接近 0
- base 高度在合理范围
- 左右腿姿态不过度劈叉
- 轮端只做必要的平衡辅助
- 腿部关节目标平滑，不高频抖动

## Action

主线 action 定义：

```text
[jIJ_des, jIO_des, jAB_des, jAG_des, left_wheel_vel_des, right_wheel_vel_des]
```

腿部四个输出是主动关节位置目标。轮端两个输出是速度目标。

第一版不走：

```text
policy -> IK -> q_des
policy -> VMC -> torque
```

## Student Observation

第一版 Student actor 只使用实机可获得的量：

- base angular velocity
- projected gravity
- 四个腿部主动关节位置
- 四个腿部主动关节速度
- 两个轮端关节速度
- command
- previous action
- observation history

不使用：

- true base linear velocity
- contact force
- terrain truth
- `L0/theta/dL0/dtheta`
- IK 结果
- 仿真 body pose 真值

## Reward 第一版建议

先保持 reward 少而清楚：

- upright reward：鼓励 `projected_gravity` 接近直立
- height reward：鼓励 base 高度接近目标
- angular velocity penalty：抑制 roll/pitch/yaw 高速摆动
- action rate penalty：抑制 action 高频变化
- joint velocity penalty：抑制腿部高频抖动
- wheel velocity penalty：原地站立时抑制轮子无意义高速转动
- termination penalty：摔倒或异常终止惩罚

不要第一版就加入几十个 reward。每个 reward 都要能解释物理意义。

## Termination 第一版建议

- roll / pitch 超过安全阈值
- base 高度过低或过高
- NaN / Inf
- 关节位置超过仿真安全范围
- 机器人明显飞出工作区

## 后续扩展

原地站立稳定之后再逐步加入：

1. 小速度前后移动
2. yaw 转向
3. 命令跟踪
4. domain randomization
5. CTS / Kinematic-Prior CTS

## 需要学习

- episode、reset、termination 的含义。
- reward shaping 怎么影响策略学到的动作风格。
- 为什么站立任务也需要轮端约束，否则策略可能靠高速转轮刷稳定性。
- base height reward 要结合地形高度，不然粗糙地形上会误判高度。
- 粗糙地形会增加接触和重置开销，训练时间通常比平地长。
