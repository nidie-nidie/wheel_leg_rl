# 相同力矩下的机械响应隔离诊断

**目标：** 在已确认的相同恒定力矩、逐关节对齐初态、无地面工况中，区分被动阻尼、闭合约束及步长的影响。用户已授权继续检查这些因素；本次自行执行，不新增子 agent。

**方法：** 从 same-torque-20261010-v1 复制诊断器到本目录。正式训练、USD、MuJoCo XML、架构及历史结果保持只读。每次从新进程启动，只运行 20 ms 物理过程，不调用后续策略，不进行训练或策略排名。

**依赖：** Isaac Lab/PhysX、MuJoCo、NumPy；输入固定为已有的 8×6 float32 恒定力矩和 Isaac reset 实测状态。八组是不同输入条件，并非八次统计重复。

## 对照

| 组 | 改动 | 保留 |
|---|---|---|
| baseline，两端 | 无新增物理改动；验证新采集器复现旧同力矩轨迹 | 六主动 PD 关闭、被动 damping=0.05、闭合机构、重力、自由底座、各自正式步长 |
| passive_off，两端 | 仅 20 个被动关节 damping=0 | 其余 baseline 条件 |
| closure_off，两端 | 仅临时关闭 Isaac 四个闭合关节、MuJoCo 八个 connect | 其余 baseline 条件 |
| dt_5ms，仅 MuJoCo | dt 从 1 ms 改为 5 ms，4 步保持总时长 20 ms | 其余 baseline 条件，包括约束 solref/solimp |

闭合关闭后，正式闭合几何验收不再适用。因此 Isaac 统一使用 DirectRLEnv 原有物理循环的诊断拷贝，不计算奖励、终止或闭合几何观测。baseline 必须逐值复现旧轨迹，才接受这个采集方式。MuJoCo 仍通过 runtime.step，步长组只使用内存中的诊断 contract 副本；正式 manifest 不修改。

## 文件及执行

- [x] `build_probes.py`：保护哈希，生成两个诊断拷贝及精确 diff。
- [x] `probe_isaac.py`：临时 stage 闭合开关、PhysX/actuator 被动阻尼修改及实际外加力记录。
- [x] `probe_mujoco.py`：仅诊断模型内存修改，逐项检查编译语义差异、初态对齐、无观察器重放。
- [x] `run_diagnosis.py`：顺序运行七组，新进程日志和输入/脚本哈希完整保存。
- [x] `analyze.py`：旧小力矩证据核验及新数据比较，保存分条件数据、报告和验证。

运行 `E:\wheel_leg_rl-main\.venv\Scripts\python.exe -B build_probes.py`，再运行同 Python 的 `run_diagnosis.py` 和 `analyze.py`。Isaac 使用工程主 venv；MuJoCo 使用 sim2sim/mujoco/.venv。

## 验收与解释边界

- [x] 原 93 个保护文件 SHA256 不变，旧诊断输入及原始结果哈希记录。
- [x] baseline 两端 0/5/10/15/20 ms root/joint 状态逐值复现上一轮同力矩记录。
- [x] 每组 26 个关节 q/qd、root 高度及归一化姿态完全匹配同一初态，初速度为零，reset 不推进时间。
- [x] 直接外加力与输入完全一致，主动 PD 关闭，无接触，主动速度不越界。
- [x] 各干预只有表中指定的物理参数不同；MuJoCo 无观察器重放完全一致。
- [x] 以角速度向量 L2、六主动关节速度 RMS、全部关节分项及响应幅度比较。相对误差逐条件计算后平均。
- [x] 保留旧 P30/P40 是开发探针、未通过整套 RootCauseSuite gate 的说明；P30 open 的 profile hash 不同，不能宣称正式验收通过。

若关闭闭合显著减小差异，表述为本工况差异依赖闭合约束；不能直接推导应取消真实机构的闭合或某项正式参数错误。若改步长影响结果，只能定位积分与约束求解的耦合，不能把 dt=5 ms 宣称修复。关闭阻尼也会改变运动幅度，不能将误差下降当作原问题的根因贡献比例。本测试不能决定 checkpoint sim2sim 合格或是否重新训练。

## 数据驱动的补充：刚体角速度上限

七组结果显示，closure_off 在 5/10 ms 很接近，15/20 ms 再次拉开；没有关节位置或速度限位触发。进一步查到 `assets/wheelleg.py:51` 设置 `max_angular_velocity=100.0`，本地 Isaac Lab `schemas_cfg.py:92–93` 明确单位为 deg/s。

补充仅 Isaac 的四组：闭合 on/off × 刚体上限 100/10000 deg/s；后者约 174.53 rad/s，远高于本次运动。其余条件不变，独立新进程。用 `build_cap_probe.py` 生成额外拷贝 `probe_isaac_cap.py`，`run_cap_probe.py` 运行，`analyze_cap.py` 验证。原七组脚本不再修改。增加各 link 角速度和全部运行时刚体 USD 上限属性记录；100 deg/s 两组必须逐值复现已有 baseline/closure_off。

补测目的是验证是否存在限幅造成的非线性响应差异；即使确认，也不在本次修改正式参数或训练。较高上限不作为正式推荐值。
