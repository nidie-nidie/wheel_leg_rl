# 观测空间

第一版 Student / 机载可用观测：

- base 角速度
- 重力在机体系下的投影 `projected_gravity`
- 腿部主动关节位置
- 腿部主动关节速度
- 轮端关节速度
- command
- previous action
- observation history

下面这些量不能进入 Student actor，除非以后明确实机能实时获得：

- 仿真真值 base linear velocity
- 接触力
- 地形真值
- 只有仿真里才有的 body pose
- privileged dynamics parameters
- 运动学先验特征，例如 `L0`、`theta`、`dL0`、`dtheta`

Teacher / critic 后续可以使用 privileged information，但每个实验都必须单独记录清楚，
避免 Student actor 偷看到实机没有的量。

## 需要学习

- actor observation、critic observation、privileged information 的区别。
- 为什么 Student actor 不能使用仿真真值。
- observation history 能在没有显式速度/延迟模型时补一部分动态信息。
- observation normalization / clipping 为什么能防止训练数值炸掉。
- 每加一个观测量都要问：实机有没有、频率够不够、噪声多大、延迟多少。
