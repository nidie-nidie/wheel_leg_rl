# Reset Observation 文档独立复核记录

日期：2026-10-09  
范围：Architecture v0.22 与 ResetObservationCacheCoherenceV1 设计/实施准备  
复核方式：用户明确授权的一个 agent `reset_document_review`，只读复核两轮；没有第二个 agent，也没有训练或 Isaac/MuJoCo 仿真。  
记录方式：主 agent 将 reviewer 返回的发现、修订与结论整理落盘；这份记录不冒充代码复核或运行报告。

## 1. 第一轮

审查身份：

| 文档 | SHA256 |
|---|---|
| 主架构初稿 | `996A738A4943CC804C92BE2CDEE742FD5E5471389E0CBB7DA62E05D369C304EE` |
| 专项设计初稿 | `A59A930CD24A0004387E4C4B43CCCC52CC520E241A7936D5524189A7E0BDD75A` |

发现计数：`P0=0, P1=1, P2=1, P3=0`。

### P1：角速度-only 补丁不能证明完整 reset coherence

初稿架构第 1237 行要求 clean Critic 角速度和任务状态与当前物理状态一致；设计第 83 行只失效 `_root_link_vel_w`，第 156–158 行只检查角速度前三维。

reviewer 独立核对的源码证据：

- `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/assets/articulation/articulation.py:427` 更新 root-link pose；第 438–443 行的缓存失效列表没有 `_root_com_pose_w`。
- `articulation_data.py:513–522` 在同 timestamp 下可以保留旧 COM pose；第 881 行的 `root_com_pos_w` 使用该缓存。
- 正式 `env.py:667,671` 读取 COM 位置/高度，`observations.py:118` 将高度写入 Critic。

reviewer 用 AST 抽取原始 getter/writer、假 PhysX 后端做纯缓存模型：同 timestamp 下把 root-link 高度从 `0.17` 写为 `0.25`，仅失效角速度缓存后 COM 高度仍为 `0.17`。这是静态模型证据，没有运行真实 PhysX，不能写成完整训练配置的 COM 缺陷已复现。

修订：把一行失效明确标为角速度候选；新增 direct PhysX root pose + COM local offset 算出的 raw COM 位置/高度与 Critic `28:29` 对照；覆盖 full、partial 和 auto-reset。若其他字段存在 split，Task 2 停止，先记录运行证据并修订窄设计，不扩大或绕过补丁。

### P2：cached 读取缺少调用计数与七流 RNG 断言

初稿架构第 1240 行要求 discarded/first 调用次数与 RNG neutrality；设计第 151–164 行只比较张量，不能证明没有重采样。

修订：probe 记录实际 `_get_observations()` 和 `sample_actor_noise()` 计数及七个 generator state。正式类 wrapper 构造/full reset 的底层观测次数为二；debug subclass 的既有 pre-forward capture 使该数为三，分别保留。连续 cached 双读必须证明计数、七流状态和张量均不变；只读 instrumentation 的源码和 SHA256 入报告，不另调观测/噪声函数构造 expected。

## 2. 第二轮

同一 reviewer 复查上述修订，最终：

```text
P0=0, P1=0, P2=0, P3=0
第一轮两项在设计层已闭环。
```

审查身份：

| 文档 | SHA256 |
|---|---|
| 主架构 v0.22 | `34968F8FC73D8B4F9D5B63870EC8CB809D49BC0FA55BC5DFFBEEEB764F2B155A` |
| 修订设计，复核时正文 | `70EAD3DA8903C3CC378ECCE81A0FFB40DC830CA61740D1D10E7BD003703DC9C9` |

reviewer 确认新增 COM 对照的定义与上游源码一致：直接使用 PhysX `get_coms()[:,0,:3]` 和 root pose 重建 world COM，不复用待测 `_root_com_pose_w`；Critic 高度切片 `28:29` 与 schema 一致。formal/debug 次数与 cached 双读 RNG 验收符合当前 wrapper 和 diagnostic 路径。

批准的是先真实 probe 诊断、仅当前置对照没有其他 split 时实施角速度候选的文档/准备方案。没有批准“一行代码已完成整个 coherence”，没有运行验收通过结论。

实施提醒：分别采集和落盘角速度、COM/height 判定后再汇总失败。不能因预期的角速度断言先失败，导致 COM 项没检查却把前置条件当作满足。该提醒已写入设计的 Task 1。

第二轮复核后，主 agent 只更新设计状态、复核记录链接，并写入 reviewer 的上述实施提醒；候选代码和物理契约未改变。最终设计文件 SHA256：

```text
143252DEA8A711D5600CD9F620189193C4209F2B160FBF16476C552394E71DCD
```

## 3. 自查与当前边界

主 agent 已解析设计中的三个 Python 片段，确认语法有效；核对主架构版本/第 18.1 节/ADR-067、COM 对照和 RNG 断言一致，并重新计算六个生产/vendor/debug/MuJoCo 源码 SHA256，仍等于修改前值。生产 `env.py` 保持：

```text
5399B29B92B54851EB0A5FA880EA9386C82F202A5C337F19ABE4EDF300C2969C
```

没有运行 pytest、训练、Isaac 或 MuJoCo 仿真；没有修改生产代码、USD、奖励、动作、命令、随机化、termination、MuJoCo runtime/参数或冻结诊断产物。真实 COM 边界、角速度候选效果和新 G01/G02 仍待运行证明；P10-P60/C70 门没有因本次文档复核关闭。
