# MuJoCo 刚体角速度限制对齐实施计划

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. 本轮由主 agent 顺序执行；用户未授权新增子 agent。

**Goal:** 保留 Isaac 的训练动力学，在 MuJoCo 中补上已有 PhysX 刚体角速度限制，再检查相同力矩响应和四组旧策略的八场景表现。

**Architecture:** 新增 MuJoCo 外力适配器，按 Isaac 的 5 ms 周期计算各机器人刚体的角速度限制力矩，并在五个 1 ms 物理步中保持该世界系力矩。运行器负责 reset 和每步调用；评估契约记录运行时适配器，区分 XML 动力学和补充的外力语义。策略、模型 XML 和训练端保持原语义。

**Tech Stack:** Windows PowerShell、项目 Python 3.11、MuJoCo、NumPy、pytest、TorchScript。

## 已确认的证据与范围

- 原架构 v0.23 SHA256：`43AC9306FD065DA259C60033E702C264230BB6B9D767B3974EFCE940587B94FA`。
- Isaac 配置 `assets/wheelleg.py` 的 `max_angular_velocity=100.0` 单位为 deg/s，约 1.745329 rad/s。本轮保留这个实际训练条件，不把旧 checkpoint 解释成在 100 rad/s 下训练。
- PhysX 5.6.1 GPU 源码 `forwardDynamic2.cu` 使用惯量和超限角速度产生偏置力矩；并非输出速度硬截断。参考 commit：`5ca9f472105a90d70d957c243cb0ef36fe251a9f`，本地副本位于本轮诊断目录。
- 诊断副本中，5 ms 更新的补偿将闭合开启时 20 ms 平均根角速度差从 0.431066 降至 0.089999 rad/s；闭合关闭时从 0.544934 降至 0.005342 rad/s。1 ms 更新不等价且表现更差，禁止采用。
- 该实现对应 PhysX 已公开的逐 link 偏置力矩项和更新周期，不宣称移植了整个 PhysX/TGS 求解器。

## 文件与依赖

| 文件 | 职责与依赖方向 |
|---|---|
| `sim2sim/mujoco/wheelleg_mujoco/angular_limit.py`（新增） | 只依赖 MuJoCo/NumPy；计算世界系惯量和角速度限制力矩；管理 5 ms 更新、保持与 reset；提供可序列化契约 |
| `sim2sim/mujoco/wheelleg_mujoco/contract.py` | 从 policy manifest 已有 timing 字段读取 Isaac 物理周期，传给运行器；原有三件套及 hash 校验保留 |
| `sim2sim/mujoco/wheelleg_mujoco/runner.py` | 依赖 contract 和新适配器；在 reset 后清空适配器状态，在每次 `mj_step` 前调用；动作控制器不承担限速逻辑 |
| `sim2sim/mujoco/wheelleg_mujoco/evaluation.py` | MuJoCo 评估契约升级 V2，显式记录角速度适配语义；source fingerprint 自动覆盖新增模块；拒绝混用旧评估契约 |
| `sim2sim/mujoco/tests/test_angular_limit.py`（新增） | 验证单位、世界系惯量、逐刚体超限响应、更新/保持/reset、外部力矩保留和实时状态不被 observer 改写 |
| `sim2sim/mujoco/tests/test_evaluation_metrics.py` | 补充 V2 契约 fixture 与角速度适配身份校验 |
| `../docs/2026-10-03-wheelleg-dreamwaq-architecture.md` | 升级 v0.24，仅补充这项 sim2sim 外力适配语义、兼容性和证据边界 |
| `artifacts/debug/sim2sim/angular-limit-alignment-20261010-v1/` | 原文件备份、候选诊断、正式复测、四组新评估、hash/diff 和结论；旧 suite 不覆盖 |

## Task 1：写入架构

- [x] 将架构升级 v0.24，保留现有冻结契约，新增 `PhysxRigidAngularBiasV1` 和 `MujocoEvaluationContractV2` 的运行时边界。
- [x] 明确公式：当 `|omega| > radians(100)`，`tau_world = -I_world @ omega * (1-limit/|omega|) / 0.005`；否则为零。`I_world = R_inertia_world @ diag(body_inertia) @ R_inertia_world.T`。
- [x] 每次计算发生在 5 ms 段开始，保持五个 1 ms 步；每段重新读取当时的 qpos/qvel，在独立 MjData 上更新运动学。禁止覆盖 live qvel 或调用 live `mj_forward`。

## Task 2：测试与实现

- [x] 新测试先运行，确认缺少模块时失败。使用一个非球形惯量的自由刚体检验世界系力矩，不仅重复实现表达式；验证 z 轴旋转后的 x/y 惯量互换。
- [x] 实现独立类：初始化校验更新周期是物理周期整数倍；`reset()` 清空周期计数和保持力矩；`apply(data)` 只在段开始重新计算，替换自身上次贡献并保留其他外部力矩。
- [x] 运行 `sim2sim/mujoco/.venv/Scripts/python.exe -B -m pytest sim2sim/mujoco/tests/test_angular_limit.py -q`，应全部通过。
- [x] 在 contract 追加默认 0.005 s 的 Isaac 周期字段，在 `from_policy_manifest` 中读取 `timing['isaac_sim_dt_s']`。
- [x] 运行器构造适配器；keyframe reset 之后调用适配器 reset；控制循环在 apply_torque 后、`mj_step` 前调用适配器 apply。新增逻辑不写 ctrl、history、previous_action 或模型参数。
- [x] V2 evaluation contract 增加 `rigid_body_angular_limit`，包括版本、deg/s 与 rad/s、更新周期、保持方式、世界系及作用对象。ranking 重新验证该字段；旧 V1 评估不混入本轮。

## Task 3：回归和相同力矩诊断

- [x] 运行完整 MuJoCo tests；保持 USD、XML、网格、动作/观测适配器与模型 manifest hash 不变。
- [x] 从原短程 probe 新建本轮副本：继承正式 runtime 新适配器，避免候选补偿重复施加；保留相同八组力矩、匹配初态、无地面、闭合开/关和 5/10/15/20 ms 采样。
- [x] 新进程复跑两个 MuJoCo case；与候选 5 ms 补偿逐状态比较，预期完全相同；重新计算与 Isaac 原配置的误差、验证力矩提交和 reset 不推进时间。

## Task 4：四组旧策略八场景评估

- [x] 对原 suite `training-suite-20261010-001203-reset-v2` 的四个 actor 各运行 `scripts/evaluate_mujoco.py`，使用原 manifest、原 suite context、run index 1..4 和 completed iterations 1000，输出到本轮独立目录。
- [x] 不加 `--smoke`，八场景各最多 500 ticks；已有失败条件、动作与命令保持原值。
- [x] 运行 `scripts/rank_mujoco_runs.py` 生成新排名并验证全部 CSV/report/source hash；不改旧 suite manifest 或标记训练成功。
- [x] 四个 actor 再验 golden vectors，max error <=1e-7；核对 actor、checkpoint、三件套和 reset-cache 的 hash 未变。

## Task 5：形成结论

- [x] 记录代码差异及准确行数、测试结果、相同力矩响应、四组新旧场景完成率/存活时长与失败原因。
- [x] 明确是否只缩小动力学差异、是否八场景通过，以及剩余首步闭合响应是否仍存在。不得从短程改善直接宣称 sim2sim 闭环成功或必须重训。
- [x] 本轮不训练。history、CENet、AdaBoot、loss、reward、randomization、USD、MuJoCo XML/dt/闭合参数、reset 初态及时间语义保持原样；旧 checkpoint 的训练/续训条件不变。
