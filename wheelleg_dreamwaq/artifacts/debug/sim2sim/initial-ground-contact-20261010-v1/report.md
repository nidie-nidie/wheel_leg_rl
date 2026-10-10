# 初态与轮地接触定向诊断（2026-10-10）

结论：未发现“一边从明显高度下落、另一边已等待落稳”的启动差异。两边正式 reset 都把机身写到约 0.20003 m 的 root-link 高度，速度为零，轮子几乎贴地；两边均没有重力下落等待。但是轮子碰撞体的微小间隙和开始支撑的响应并不相同，尚不能把 sim2sim 失败归因于某一项。

## 实际范围

使用当前修复后的正式环境源码、正式八环境评估缓存（seed 20261007），以及新四 seed 训练 suite 的 run-02 actor。Isaac 继承正式 reset/step，经正式 RSL wrapper 完成首次 reset；只在独立配置中增加接触观察器。MuJoCo 使用未修改的 WheelLegMujocoRuntime。两端都运行零动作及同一条首动作保持，分别 100 ms，不使用后续策略反馈。

## 初始读回（nominal_stand，env 0）

| 量 | Isaac | MuJoCo |
| --- | ---: | ---: |
| root-link 初始高度 | 0.200029984 m | 0.200030000 m |
| 初始 root 速度最大绝对值 | 0.0 | 0.0 |
| 左轮碰撞体最低点距地面 | 0.054687 mm | 0.001521 mm |
| 右轮碰撞体最低点距地面 | 0.152851 mm | 0.004470 mm |
| reset 期间物理时间推进 | 0.0 s | 0.0 s |

两侧地面顶面都是 z=0。按名称对齐全部 26 个关节，初始角度最大差 4.49581116e-06 rad。Isaac 是 authored mesh 的 convexHull；MuJoCo 是半径 0.0625 m 的 sphere proxy。Isaac 的表中间隙由实际 PhysX link pose 和网格顶点求出，不包含 hull cooking 的近似误差及 contact/rest offset；因此不能当作底层求解器的精确碰撞距离。

## 初始化的抬高过程

Isaac cache 生成时会关闭重力，将 root 额外抬高 0.75 m，整理闭链。加载缓存时在同一抬高位置做 forward 验证。此次实测构造后 root 高度为 0.950029969 m；RSL wrapper 调用正式 reset 后，root 高度立刻写回 0.200029984 m，中间没有物理 step。这是写入初态，不是让机器人从 0.75 m 高处自由落下。MuJoCo reset 从 keyframe 写入初态，再 mj_forward，同样不等待落地。

## 支撑力开始出现的样本

零动作：MuJoCo 两轮在 2 ms 样本首次报告非零支撑力；Isaac 两轮在首个 5 ms 样本均报告非零支撑力（约 17.16、12.17 N）。因为两端采样间隔分别是 1 ms 和 5 ms，不能由此声称 Isaac 实际接触晚了 3 ms。

同一首动作：前 20 ms 两边均主要由左轮支撑，右轮没有支撑力。右轮首次非零力样本分别是 MuJoCo 42 ms、Isaac 50 ms。它们是该固定动作下的诊断结果，不能外推到所有 checkpoint 或完整闭环评估。

零动作前 5 ms 的两轮合计竖直支撑冲量：Isaac 0.146662250 N·s，MuJoCo 0.096894207 N·s。这证明启动时接触响应存在数值差异，但试验同时保留正式 drive、闭链与积分方式，不能仅凭该差异认定是接触模型、初始微小间隙或闭链造成。

reset 时尚未执行新 episode 的物理求解，PhysX raw contact buffer 可能保留上一求解步的读数。因此本报告仅使用实际物理 step 之后的支撑力，不把 reset 时的 force 值作为初始接触证据。

## 采集与文件验证

Isaac 两个独立 Kit 进程：接触观察器关闭/开启，共 43 个快照、每个快照 8 个环境。root pose、root velocity、26 关节位置/速度、轮子最低点的最大差全部为 0，观察器没有改变本次运动。

MuJoCo 16 个 case、1616 个快照。去掉采集回调后重放相同正式 runtime，全部 16 个 case 的最终 qpos/qvel 与采集运行逐项完全相等；几何刷新只发生在 shadow MjData，未在正式 data 上额外调用 mj_forward。

保护文件在运行前后及分析时逐项 SHA256 核对相同，包括正式源码、USD 资产、架构文档、评估缓存、actor 和 MuJoCo 模型/runtime。未改生产模型或参数，也未训练。此报告是 debug-only 证据，不是旧 RootCauseSuite 的正式 verdict，不改写旧 suite 身份或运行记录。

原始证据：[Isaac off](isaac-off/evidence.json)、[Isaac on](isaac-on/evidence.json)、[MuJoCo](mujoco/evidence.json)、[观察器重放验证](mujoco-observer-neutrality.json)、[机器可读分析与哈希](analysis.json)。

建议下一步用已有记录比较前 5/20 ms 的支撑冲量、关节运动和机身角速度，确定运动差异与接触响应差异的先后关系；本次不修改初态高度、不增加落地等待、不关闭闭链。
