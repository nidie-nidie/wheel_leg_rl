# Final Identification Contract

- 状态：**NOT FROZEN**
- 最后更新：2026-09-28
- 适用会话：`FINAL_IDENTIFICATION_SESSION`
- 当前固件锁：`PACE_FINAL_CONTRACT_READY = 0`

该合同只冻结 RL 与执行器之间的最低层接口，不要求先确定网络结构、观测量、奖励函数、训练算法或 domain randomization。

## 1. 必填接口

所有 `TBD` 清零前不得解锁正式采集。

| 字段 | 冻结值 | 说明 |
|---|---|---|
| contract version | `TBD` | 单调递增整数 |
| action update rate | `TBD Hz` | RL/低层目标更新频率 |
| command hold | `TBD` | 例如 zero-order hold |
| DM action semantics | position target | 是否为绝对位置或 nominal pose + offset |
| DM L_front nominal pose | `TBD rad` | canonical 坐标 |
| DM L_rear nominal pose | `TBD rad` | canonical 坐标 |
| DM R_rear nominal pose | `TBD rad` | canonical 坐标 |
| DM R_front nominal pose | `TBD rad` | canonical 坐标 |
| DM maximum expected amplitude | `TBD rad` | 训练和部署共同包络 |
| DM required bandwidth | `TBD Hz` | 正式 chirp 必须覆盖 |
| DM Kp/Kd | `20.0 / 0.6` 或 `TBD` | 若改变必须重新采集和拟合 |
| DM dq_des policy | `0` 或 `TBD` | |
| DM tau_ff policy | `0` 或 `TBD` | |
| LK primary action mode | `TBD torque/velocity` | 必须二选一 |
| LK action scale/limit | `TBD` | canonical Nm 或 rad/s |
| control-to-CAN scheduling | `TBD` | update 与 500 Hz bus command 的关系 |
| emergency safety envelope | `TBD` | 速度、转矩/电流、温度、超时、位置中止 |
| fixture/load condition | `TBD` | 架空、固定基座或规定负载 |
| final stage layout version | `TBD` | 独立 FIT/VALIDATION 段 |
| experiment config hash | `TBD uint32` | 绑定全部字段和固件 build |

## 2. 正式数据覆盖

- DM `FIT` 与 `VALIDATION` 必须是不同轨迹，不能随机拆分同一条 chirp。
- 两段均使用最终部署的 Kp、Kd、dq_des 和 tau_ff 语义。
- 轨迹覆盖 nominal pose、预计最大 action 幅值和主要带宽，但正常运行不得触发安全裁剪。
- LK 主命令模式必须有独立 `FIT` 与 `VALIDATION` 段。
- 非主 LK 模式可以保留为 `DIAGNOSTIC`，不得混入主模型优化。
- 同一 raw 文件必须保留 stage config、Tx-confirmed 命令、反馈、时间年龄、status、event 和 footer。
- 轮地接触/打滑属于单独的接触模型问题；架空 LK 数据只能拟合执行器本体。

## 3. 固件解锁步骤

1. 完成本合同并由控制、硬件和仿真负责人共同签字。
2. 更新正式 stage layout，使 DM 和最终 LK 主模式各有独立 FIT/VALIDATION。
3. 生成全部合同字段的 canonical 配置文件并计算 32 bit `experiment_config_hash`。
4. 更新 `Config/pace_experiment_config.h` 中全部 `PACE_FINAL_*` 字段。
5. 更新 firmware build ID，重新 build，并记录 `.axf/.hex` SHA-256。
6. 在代码评审中确认正式配置不会回退到 commissioning 的 stage 语义。
7. 最后才将 `PACE_FINAL_CONTRACT_READY` 改为 `1`。
8. 用 host `CONFIGURE(FINAL)` 验证 session header 和每条 stage config 都携带相同非零 hash。

## 4. 正式拟合要求

- DM 正式模型使用参考 `pace_sim2real.CMAESOptimizer` 或经评审的等价模拟回放优化。
- 参数至少包含 armature、viscous friction、static/dynamic friction、encoder bias 和 delay。
- LK 使用与最终命令模式匹配的独立模型。
- manifest 固定标记 `model_maturity = final`。
- Isaac 与 MuJoCo 必须读取同一 manifest，不允许手工重排或复制参数。
- 每通道分别报告 fit 与 validation 的 RMSE、P95、最大误差、延迟误差、方向错误和饱和率。
- validation 失败时不得通过扩大 domain randomization 掩盖执行器模型错误。

## 5. 签字

| 角色 | 姓名 | 日期 | 结论 |
|---|---|---|---|
| 实机/硬件 | | | |
| 低层控制 | | | |
| RL | | | |
| Isaac | | | |
| MuJoCo | | | |
