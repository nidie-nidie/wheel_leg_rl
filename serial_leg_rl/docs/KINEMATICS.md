# 运动学

运动学模块实现的是 `../offset_closed_chain_kinematics_derivation.html`
第 4.3 节的 CAD 简化偏置闭链模型。

它不是动力学模型，只负责几何关系：

- 主动关节角到虚拟腿长、虚拟腿角的映射
- 由关节速度计算虚拟腿长速度、虚拟腿角速度
- Jacobian
- Newton 逆运动学
- 基本奇异性诊断

## 坐标约定

每一侧腿都在自己的局部腿平面里计算，原点是该侧两根主动轴的共同髋轴 `I`。

- `s`：沿 base `+Y` 为正
- `z`：沿 base `+Z` 向上为正
- `d = -z`：车体向下为正

输入顺序固定为：

```text
q = [q_front, q_rear]
```

左腿：

```text
q = [jIJ, jIO]
```

右腿：

```text
q = [jAB, jAG]
```

## 主要正运动学模型

```text
J = R(q_front) v_IJ
L = R(q_rear)  v_IL
P = k_a L
用理想五连杆内层 B=L, D=J, C=M 的半角闭式求 phi2
M = L + l_LM [cos(phi2), sin(phi2)]
W = P + k_b (M - L)
s_W = W_s
d_W = -W_z
L0 = sqrt(s_W^2 + d_W^2)
phi0 = atan2(d_W, s_W)
theta_leg = pi/2 - phi0
```

这里故意不使用第 4.2 节的通用两圆交点实现。虽然 4.2 的圆交和 4.3 的半角闭式在数学约束上等价，但当前代码按 4.3 写成“理想五连杆先求 M，再用平行四边形关系求 W”，这样和后续控制文档、固件迁移时的公式名字一致。

当前代码位置：

```text
serial_leg_rl/source/serial_leg_rl/serial_leg_rl/kinematics/offset_leg.py
```

`phi2` 分支固定为 `sigma_phi=-1`，对应 HTML 4.3 里按有向线 `L -> J` 选择的实际装配支路。这个分支和旧 4.2 里按 `J -> L` 写的 `sigma_M=+1` 是同一个物理点，只是有向线换了，所以符号标签不同。

最终部署的 Student policy 不应该依赖这个模块。这个模块主要用于：

- 离线验证运动学
- 作为 Teacher 的训练先验
- 作为 IK residual / VMC baseline 等对比实验的依赖

## check_kinematics_geometry.py 在检查什么

脚本位置：

```text
serial_leg_rl/scripts/check_kinematics_geometry.py
```

它不是在跑仿真，也不是在证明强化学习能站住。它做的是静态几何核对：

- 从真实 URDF `wheel_leg_urdf4/urdf/wheel_leg_urdf4.urdf` 提取左右腿的固定几何点。
- 打印 `v_ij`、`v_il`、`v_ip(raw)`、`l_jm`、`l_lm`、`l_pw`、`k_a`、`k_b`。
- 把提取出的几何和当前代码里的左腿参考常量做绝对误差比较。

这些 print 的意义：

- `v_ij`：髋轴 I 到前主动杆末端 J 的参考向量。
- `v_il`：髋轴 I 到后主动杆闭链点 L 的参考向量。
- `v_ip(raw)`：URDF 里髋轴 I 到输出起点 P 的原始参考向量；当前 4.3 运行时会使用 CAD 共线约束 `P=k_a L`。
- `l_jm`、`l_lm`、`l_pw`：内层五杆和输出杆的长度。
- `k_a`：`IP / IL`，把 L 放大到 P。
- `k_b`：`PW / LM`，把 `L -> M` 放大到 `P -> W`。

它确定“代码里的几何常量没有抄错 URDF”，不能单独证明“整套控制正确”。运动学正确性还要看：

- 单测里的零位结果是否合理。
- 解析 Jacobian 是否和中心差分一致。
- IK/FK 回代是否能收敛到同一组关节角。
- 之后如果接 MuJoCo，需要比较解析 W 和仿真 forward 后的轮轴位置。

## 需要学习

- 平面五连杆正运动学：主动角、被动杆角、装配支路。
- `atan2`、半角替换、判别式和分支选择。
- Jacobian 的物理意义：关节角速度到腿长/腿角速度的瞬时映射。
- 虚功关系 `tau = J^T F`，它是力/力矩映射，不是动力学模型。
- 运动学、微分运动学、动力学三者的边界。
