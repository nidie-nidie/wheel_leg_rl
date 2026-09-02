# 执行器

已确认硬件：

- 腿部电机：达妙 DM8009P / DM-J8009P-2EC，MIT 控制模式
- 轮端电机：领控 9025

## 当前仿真参数

当前代码位置：

```text
serial_leg_rl/source/serial_leg_rl/serial_leg_rl/assets/wheel_leg_robot.py
```

腿部电机：

```text
effort_limit = 20.0 Nm
velocity_limit = 45.0 rad/s
velocity_limit_sim = 80.0 rad/s
stiffness = 80.0
damping = 3.0
delay = 0 到 1 个仿真步
```

轮端电机：

```text
effort_limit = 2.42 Nm
velocity_limit = 51.31 rad/s
velocity_limit_sim = 51.31 rad/s
stiffness = 0.0
damping = 0.35
delay = 0 到 1 个仿真步
```

passive joints 只是为了让导入后的树结构闭链相关关节不要完全无阻尼乱飞，不代表真实主动电机。

passive joints / 闭链近似关节：

```text
effort_limit = 20.0 Nm
velocity_limit = 40.0 rad/s
velocity_limit_sim = 80.0 rad/s
stiffness = 60.0
damping = 2.5
delay = 0
```

这里的 `stiffness/damping` 是第一版 Isaac standing baseline 的工程近似：让 URDF 导入后的树结构关节保持 nominal pose，避免 `jOP/jGH` 等支撑链自由下垂。它不等价于真实硬件上有这些电机，也不等价于真正的 PhysX 闭链约束。

## 哪些是确认值，哪些是暂估值

DM8009P：

- 20 Nm 是官方公开参数里的额定扭矩。
- 40 Nm 是官方公开参数里的峰值扭矩。
- 额定转速公开参数为 100 rpm，空载最高转速和电压有关，24 V 约 160 rpm，48 V 约 320 rpm。
- 参考资料：达妙官网 DM-J8009P-2EC 页面 `https://www.dmbot.cn/index.php?c=show&id=105`；达妙英文手册镜像 `https://damiao.enactic.ai/en/products/hardware/dm-j8009p-2ec-v1.0/`。
- 当前训练使用 20 Nm 作为仿真力矩限制，是按额定扭矩保守处理。
- 当前 `velocity_limit=45 rad/s` 是仿真暂估值，不是直接照抄官方额定转速。

领控 9025：

- 官方数据按你确认的领控 9025 参数记录。
- 额定扭矩：2.42 Nm。
- 峰值扭矩：4.6 Nm。
- 额定转速：490 rpm，换算为约 51.31 rad/s。
- 转速常数：20 rpm/V。
- 当前训练使用额定扭矩 2.42 Nm 和额定转速 51.31 rad/s 作为仿真限制。
- 峰值扭矩 4.6 Nm 先只作为硬件能力记录，不默认给策略长期使用。

待确认项目：

- 实际可用的 `Kp/Kd`
- action latency
- 每个电机的编码器方向和零偏
- 轮端电机的命令接口细节

第一版仿真参数可以先用保守估计值，但训练前每个假设都要记录在这里。

## 需要学习

- MIT 控制模式里 `p_des`、`v_des`、`kp`、`kd`、`tau_ff` 的含义。
- Isaac Lab 里 actuator 的 `effort_limit`、`velocity_limit`、`stiffness`、`damping` 分别影响什么。
- 额定扭矩、峰值扭矩、持续电流、峰值电流之间的区别。
- 为什么 sim2real 前要做电机零位、方向、延迟和带宽标定。
