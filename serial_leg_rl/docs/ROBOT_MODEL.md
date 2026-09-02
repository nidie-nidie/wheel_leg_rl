# 机器人模型

真实模型来源：

```text
../wheel_leg_urdf4/urdf/wheel_leg_urdf4.urdf
```

## 已确认关节映射

腿部主动关节：

| URDF joint | Meaning | Isaac Lab legacy name |
| --- | --- | --- |
| `jIJ` | 左腿前侧电机 | `L_F_Joint1` |
| `jIO` | 左腿后侧电机 | `L_H_Joint1` |
| `jAB` | 右腿前侧电机 | `R_F_Joint1` |
| `jAG` | 右腿后侧电机 | `R_H_Joint1` |

轮端关节：

| URDF joint | Meaning |
| --- | --- |
| `jwheel_left` | 左轮 |
| `jwheel_right` | 右轮 |

## URDF 注意事项

- 这个 URDF 是机械闭链机器人导出的树结构文件。
- `dummy` links 暂时当作闭链约束点的位置来源。它们的位置有用，但质量已确认不可信。
- 原始 URDF 中所有关节都是 `continuous`，没有可靠的关节限位、力矩限幅和速度限幅。
- 标准 URDF 不能直接表达闭链约束；后续需要在仿真专用资产里补充约束。

## 仿真用资产副本

原始 URDF 保持不动。训练启动时，代码会生成一份只供 Isaac Lab / Isaac Sim 导入使用的副本：

```text
/tmp/serial_leg_rl_isaac_assets/wheel_leg_urdf4_sim.urdf
```

Isaac Sim 转换出的 USD 在：

```text
/tmp/serial_leg_rl_isaac_assets/wheel_leg_urdf4.usd
/tmp/serial_leg_rl_isaac_assets/configuration/wheel_leg_urdf4_base.usd
/tmp/serial_leg_rl_isaac_assets/configuration/wheel_leg_urdf4_physics.usd
/tmp/serial_leg_rl_isaac_assets/configuration/wheel_leg_urdf4_sensor.usd
```

本地地面资产在：

```text
/tmp/serial_leg_rl_isaac_assets/flat_terrain.usda
```

生成逻辑在：

```text
serial_leg_rl/source/serial_leg_rl/serial_leg_rl/assets/sim_urdf.py
serial_leg_rl/source/serial_leg_rl/serial_leg_rl/assets/local_terrain.py
```

## 初始站立姿态

训练环境里的零 action 会回到 nominal joint pose。当前 nominal pose 不是全 0，而是用 4.3 理想五连杆运动学选出的接近 `0.20 m` 机身高度的姿态：

```text
jIJ = -0.30 rad
jIO =  0.35 rad
jAB = -0.30 rad
jAG =  0.35 rad
```

4.3 运动学里，这组角度对应的轮心大约在髋关节下方 `0.197 m`。它只是第一版 standing baseline 的初始姿态，后续如果补上真正闭链约束或实测零位，需要重新校准。

## dummy link 为什么保留小质量

仿真副本里对 dummy link 做了三件事：

- 保留 link 和 joint 的位置关系，因为它们是闭合链路约束点的位置来源。
- 删除 dummy link 的 visual / collision mesh，避免空 mesh 或错误 mesh 影响导入。
- 把 dummy link 的质量改成 `0.001 kg`，惯量改成很小的对角占位值。

不直接设成 0 的原因是：物理引擎和 URDF/USD 导入器通常不喜欢动态刚体出现 0 质量或奇异惯量。0 质量容易导致不可逆质量矩阵、导入失败或仿真数值爆炸。小质量占位值的意思是“这个点在动力学里尽量轻，但仍然是合法刚体”。

这不是在说 dummy link 的真实质量就是 1 g；它只是仿真导入的工程处理。

## 闭链是否成立怎么检查

当前 Isaac 训练环境还没有真正加入 PhysX 闭链约束。也就是说，URDF 导入后本质上仍然是树结构；passive joints 的 PD 只是让这些关节保持 nominal pose，属于第一版可训练 baseline 的工程近似。

真正要检查闭链是否成立，不能只看机器人有没有站住，而要量对应 dummy 点之间的距离。`view_standing_env.py` 里已经加入闭链点距离诊断：

```text
closure_max=...
closure_pairs=...
```

如果闭链真的闭合，对应距离应该接近 0，通常至少要到毫米级。如果距离是几厘米或更大，就说明 Isaac 里只是靠 PD 撑住了形状，没有形成真正的几何闭链。

当前先检查这些闭链点：

```text
jIO_dummy_child_link1 <-> jMK_dummy_child1
jIO_dummy_child_link2 <-> jMK_dummy_child2
jAG_dummy_child_link1 <-> jEC_dummy_child_link1
jAG_dummy_child_link2 <-> jEC_dummy_child_link2
```

后续如果要从“闭链近似”升级到“真正闭链”，需要在 USD/PhysX 资产里加约束，而不是只靠 URDF。

## 需要学习

- URDF 是树结构，为什么标准 URDF 不能直接表达闭链。
- Isaac Sim 导入 URDF 后为什么会生成 USD。
- link 质量、惯量、collision、visual 对物理仿真的影响。
- 闭链机构在 MuJoCo 里用 equality/connect，在 Isaac/PhysX 里需要用另一套约束或近似方式。
