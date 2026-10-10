# WheelLeg Reset Observation Remediation Design

日期：2026-10-09  
版本：ResetObservationRemediationV1（待独立设计复核）  
范围：正式 Isaac reset observation/cache coherence；不训练。

## 冻结输入与结论边界

已完整读取 handoff、Architecture v0.21、RootCauseSuiteCoreV1.17、G01 调查与结果复核。2026-10-09 本地 SHA256 核验结果：

| 输入 | SHA256 |
|---|---|
| Architecture v0.21 | 5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE |
| RootCauseSuiteCoreV1.17 | 8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD |
| G01 investigation | 8C4C562AC6B2C64AF2384A1A9CF79906558A94B4FFCE1E56989B7FC76F51E4C8 |
| Independent result review | B55C82A689C72D9BE85711FA5F7F5A0C02347AF45203D0D71DFC2919543490A0 |
| formal env before repair | 5399B29B92B54851EB0A5FA880EA9386C82F202A5C337F19ABE4EDF300C2969C |

缺陷分类保持 `SIM2SIM_ADAPTER_BUG`，窄因保持 `isaac_reset_returned_observation_stale_root_angular_velocity_cache`。该缺陷解释既有 G01 identity split；不是已证明的 MuJoCo 首次失稳唯一根因。旧 run 和 V1.17 文档不可覆盖或修改。

## 只读源码证据

路径均相对项目 `E:/wheel_leg_rl-main/wheelleg_dreamwaq`：

- 正式 `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py:653-700,852-918`：reset 写 root pose、COM velocity、joint state 后立即 `_read_state()`，并只替换 reset rows；reward/done 已使用 reset 前快照。
- `algorithms/dreamwaq/history_wrapper.py:19-25,55-70`：构造/显式 reset 的 discarded observation 与 first observation 次序固定；done history 使用底层返回的 reset observation。wrapper 不修补状态。
- `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/envs/direct_rl_env.py:292-331,389-415`：全环境 reset 在任务 reset 后 write/forward 再返回观测；step 内 done reset 没有额外 forward 或 scene update。
- `articulation/articulation.py:207-218,414-444,497-525,580-650`：Articulation.reset 只清 actuator/wrench；pose writer 不失效 `_root_com_pose_w`，COM velocity writer 不失效 `_root_link_vel_w`；joint writer 已失效 body pose/velocity/aggregate buffers。
- `articulation/articulation_data.py:97-102,469-577,583-737,813-830`：derived buffer 仅在 timestamp 小于 `_sim_timestamp` 时读取；root link angular velocity 从 COM angular velocity派生，两者角速度物理语义相同。COM pose 从当前 root pose/local COM 派生。
- `sim/simulation_context.py:533-540`：forward 更新 articulation kinematics 和 fabric(0,0)，不积分、不调用 simulation step。正式 `_read_state()` 的 body-link pose读取本身已触发同一 kinematic update（articulation_data.py:590-600）。

源码还表明 COM pose 存在潜在同类失效遗漏，设计必须覆盖 Critic 高度；该源码风险不被写成新的运行确证缺陷。

## 方案比较与选择

1. **选择：reset writes 后失效遗漏的派生缓存，保持状态采集时点。** 任务层显式失效 `_root_link_vel_w`、`_root_com_pose_w` 及两个依赖它们的 combined root state buffers。原 `_read_state()` 重新计算派生量，原 `replace_rows()` 继续隔离非 reset 环境。没有新时间戳世代、RNG、物理写入、step 或 forward。
2. 延迟状态采集到 `_get_observations()`：需要 pending reset bookkeeping；debug reset 还会在 forward 前读取 observation，done reset 无 forward。新增时序和快照合并风险超过本次需求，暂不采用。
3. 从 COM 角速度直接替代 root-link 角速度，或推进 `scene.update(physics_dt)`：前者绕开 stale root-link cache并遗漏 COM pose，后者改变 data clock/加速度历史。均不采用。直接 PhysX readback只用于独立验收，不另建生产状态路径。

## 最小生产修改

唯一生产写集合：`source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py`。

在 `_reset_idx()` 三个 state writer 完成后、任何 reset `_read_state()` 之前加入：

```python
# Isaac Lab 2.3.2 state writers leave these derived root caches current.
# Invalidate them without advancing the data clock or the simulation.
self.robot.data._root_link_vel_w.timestamp = -1.0
self.robot.data._root_com_pose_w.timestamp = -1.0
self.robot.data._root_link_state_w.timestamp = -1.0
self.robot.data._root_com_state_w.timestamp = -1.0
```

四个 buffer 使用上游同样的失效标记；本栈 `_sim_timestamp` 从 0 开始且正常只增加正 dt，因此 `-1.0` 严格小于当前 timestamp。整个 batched buffer 的 cache generation 被失效，但重新派生计算只读；任务仍只合并 reset rows，非 done transition state、关节加速度历史和 reward/done 输入保持原值。上游 root link pose/COM velocity/joint pos/joint vel writers已更新基础缓存，body buffers已失效；root_state_w 的 pose/COM velocity 两段由writers更新，不需要额外失效。COM local pose读取已由joint writer失效。

不修改 wrapper、vendor、USD、env_cfg、奖励、动作、命令、随机化、termination、控制周期、MuJoCo、关节限位、checkpoint。不存在 Actor angular slice 置零；随机化非零 reset velocity必须被真实保留。fresh/full/partial/done/reset/resume共同经过该 `_reset_idx()`。

本设计依赖固定 Isaac Lab 2.3.2 private buffer contract。新增测试绑定 vendor SHA256 与实际 property dependency/source；vendor升级或buffer集合漂移必须失败，不允许 `getattr(..., None)` 静默降级。运行证据必须证明reset前采样与post-forward直接物理状态一致；若不一致，停止并修订设计，不加入隐式step。

## 回归验收

单元测试放 suite/tests，使用固定 vendor ArticulationData 实际 property 源码和 task `_reset_idx` 方法，最少stub外部PhysX/Kit接口；不得用重写缓存算法证明自身实现。先在修复前运行产生 stale observation，再实施并变绿。

1. 制造非零 terminal world angular velocity并先物化derived缓存；full/repeated/partial reset使用nominal和非零reset COM velocity、非恒等quaternion及非零COM offset。
2. 同时核验 Actor angular、Critic COM velocity/height、combined root-link/root-COM state，证明非零reset速度未被置零，pre/post其他合法字段无漂移。
3. partial reset核验未reset行的transition字段和velocity history保持逐位一致；空reset不触及缓存。
4. 使用正式DreamWaQ wrapper构造、显式reset、done-step；五帧严格等于first/current policy，非done append，重复get不新增noise采样。resume capture的discarded/first序列保持原有调用数。
5. data timestamp、simulation time、physics step counter在显式reset前后相同；done-step严格只有decimation=4个物理步。修复代码不得调用scene.update、sim.step、额外forward或改动history/noise/RNG。
6. 全新Kit集成probe直接实例化formal WheelLegFlatEnv，独立PhysX root transforms/velocities/COM offset与25D/41D返回观测对齐，保存原始canonical phase payload、actual class、formal/vendor/wrapper source SHA、完整argv及cache/source identity。多次reset用零动作短rollout制造污染，明确记录非零terminal值；覆盖fresh、5次重复reset、2-env异步done reset、wrapper和resume capture，不执行训练。

纯测试通过不能代替真实PhysX门；nominal诊断配置的通过不能被描述为完整训练配置验收。集成另覆盖随机化profile的非零velocity/noise和loaded reset-cache配置，旧suite不resume。

## 独立复核与身份顺序

1. 独立agent只读复核本设计；记录exact SHA和P0/P1计数，清零后才改正式env。
2. 最小修复+回归测试；独立agent只读代码复核P0/P1清零后才运行真实门。
3. 新增显式 `RootCauseSuiteCoreV1.17-ResetObservationRemediationV1` identity/design revision。旧V1.17文档、hash和既有runs保持不变。新revision严格继承全部阈值/catalog/phase/reset/replay/adapter语义，只允许 `formal_env` SHA白名单变化，并绑定本修复设计、设计/代码复核、vendor/handoff身份。新旧revision必须显式选择；不采用临时monkeypatch或 `validate_expected=False`。
4. 身份实现需让父进程、core_stage_worker与子worker一致验证revision；原 `integrity.py` 和 `core_stage_worker.py` 两处hardcoded FORMAL_FILES都必须受同一显式契约管理，禁止仅改其中一处或静默覆盖旧hash。
5. 全新run ID执行G00/G01/G02，禁止resume旧失败run。门禁结果保存后暂停，独立结果复核通过再继续G03及Core矩阵。默认launcher当前会直接运行全部stage，因此新revision需支持明确stop-after-G02及同一新run的后续审核放行；不得标记未执行stage通过。
6. 三次P10_A的pre-forward/post-forward/returned-policy/excitation/repeatability key全部一致，reset API与direct view对齐，五帧history及Actor/adapter通过，formal guard仅批准的env SHA变化。缺任一项则后续门保持阻断。

## checkpoint与后续边界

现有run-01 checkpoint可在修复环境只读加载作诊断，身份不变；它仍是旧环境语义训练所得。run-02 iteration 65用户停止、run-03/04未开始；禁止resume旧training-suite-20261007-185711。本轮暂不训练。G01/G02全新通过并结果复核后才允许G03、P10-P60、P60-D、C70与现有run-01 Isaac/MuJoCo重评；若将来重训必须新suite从头四seed。

## 当前执行状态

2026-10-09：冻结输入已核验；本设计待独立复核。尚未修改正式env，尚无修复后G01/G02或物理/策略verdict。
