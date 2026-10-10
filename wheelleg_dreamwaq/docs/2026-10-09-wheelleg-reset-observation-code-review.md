# Reset Observation V2 独立代码复核记录

日期：2026-10-09。用户明确授权代码完成后调用一个 agent 复核；主 agent 复用原文档 reviewer `reset_document_review`，没有调用第二个 agent。该 agent 只读复核，没有修改文件、启动仿真或训练、再调用其他 agent。本记录由主 agent 根据 reviewer 返回内容整理。

## 第一轮与修订

第一轮计数为 `P0=0, P1=0, P2=2, P3=0`，批准两个派生缓存失效的生产修改，提出两项测试补强：

1. MuJoCo 原测试只写了非零速度和 previous action，没有实际更新旧 history。测试增加 `runtime.step(np.full(6, 0.2), command)`，证明 history 与初始值不同后再 reset。
2. debug subclass 的 reset 验收需要保存 pre-forward、post-forward、returned 三相原始 payload，以及两级上游 reset wrapper 的来源身份。probe 增加深拷贝三相数据、前后角速度/COM 对照与 returned/first 对齐；pytest 增加 evidence SHA、八个 case 集合及 debug 三相字段检查。

同时补充 generated/loaded reset cache 两条构造路径、七流 RNG restore 后重复 reset 的 bitwise 对照、empty reset 的状态/缓存 timestamp/时钟/RNG no-op，以及正式类 first-observation resume trace 对照。

## 最终独立复核

同一个 agent 复查修改后的源码，最终 `P0=0, P1=0, P2=0, P3=0`。被复核的当前文件身份：

| 文件 | SHA256 |
|---|---|
| `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py` | `1EF7C18B830BE13DA6DD6BB0865A61A63822926F6403C69730B08C5E013D6A22` |
| `tests/integration/probes/reset_observation_coherence.py` | `DE7EC4008CAEF4F94B08BFE3923C2DEC0EA2DFC05C91E0934775553D61B4327A` |
| `tests/integration/test_reset_observation_coherence.py` | `834DC336F9C03AE06A9CB97BD6803DBCCD336F8ED20DE949CAF38EAF04CC4B20` |
| `sim2sim/mujoco/tests/test_mujoco_adapters.py` | `758C868C3F3AE73E98E3B4E3B863D9B66C1E331741878374F8BB95A33F7EB630` |

reviewer 确认生产改动仍只有 `_read_state()` 前两项缓存 timestamp 失效；没有改变物理步、读取 API、采样/噪声/命令时序、partial 行合并、网络或 MuJoCo runtime。复核还包括 AST 解析和实际噪声 payload 类型的全新 Python 进程导入。

## 对既有 green 产物的只读独立复核

reviewer 另行只读检查主 agent 已运行的六份 green 产物，确认 48 个 case 齐全，全部报告 passed，evidence SHA 与生产源码身份匹配。原始 tensor 重算的角速度最大误差 0、COM 最大误差 `4.768e-7 m`、高度最大误差 `1.490e-8 m`，均低于冻结阈值。日志分别记录 Isaac integration `3 passed in 41.71s`、MuJoCo adapter `7 passed in 0.71s`、unit `164 passed in 3.57s`。

以上是独立复核既有证据，不是 reviewer 重新执行 pytest。主 agent 随后也重新校验六份 hash、源码来源和 tensor，并记录于 `artifacts/reset-observation-20261009/post-fix-evaluation/verified-comparison.json`。

复核批准进入用户授权的八场景修复后诊断；不等于 RootCauseSuite 正式 G00/G01/G02 已通过，也不预先批准重训。后续八场景结果与建议见 [实施与结果记录](2026-10-09-wheelleg-reset-observation-implementation-results.md)。
