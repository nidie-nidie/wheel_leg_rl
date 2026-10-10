# Reset Observation V2 实施与 sim2sim 结果

日期：2026-10-09。按用户授权完成代码、一个 agent 独立代码复核、回归测试和八场景修复后诊断；没有启动训练。正式架构为 Architecture v0.23。

建议在修复后的环境下重新训练新的 DreamWaQ 候选，但开训前先完成两端观测/动作/物理语义的剩余正式核对。现有 checkpoint 是在有 reset 缓存缺陷的代码上训练的，修复后 MuJoCo 仍 0/8；它不能作为修复后训练结果的验收依据。当前证据不足以认定 reset 是 sim2sim 失败的唯一原因，也不能保证仅重训就解决问题。训练必须使用新 suite/source identity，不继续旧 paused suite。

## 实际修改与范围

唯一生产修改是 `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py` 的 `_reset_idx()`，本次 state 写入完成后、`_read_state()` 前失效两个派生缓存：

```python
self.robot.data._root_link_vel_w.timestamp = -1.0
self.robot.data._root_com_pose_w.timestamp = -1.0
reset_state = self._read_state()
```

实际新增三行解释注释与上述两个失效语句。让原 API 从本次 pose/velocity 重算角速度和 COM/高度，再沿用原有快照合并、噪声、normalization 和 history 流程；没有强制角速度为零，也没有推进 reset 的时间。

生产 `env.py` 修改前 SHA256 为 `5399B29B92B54851EB0A5FA880EA9386C82F202A5C337F19ABE4EDF300C2969C`，修改后为 `1EF7C18B830BE13DA6DD6BB0865A61A63822926F6403C69730B08C5E013D6A22`。原文保存在 `artifacts/reset-observation-20261009/source-before/env.py.txt`。

新增真实 Isaac probe 与 pytest 入口，扩展 MuJoCo adapter reset 测试；没有其他生产源码变更。USD、奖励、动作、命令、randomization profile/七流 RNG、termination、控制周期、网络/optimizer、checkpoint schema、MuJoCo model/runtime/参数均保留。冻结文件逐项校验仅 architecture 与 formal env 改变，其他项一致；记录见 `protected-source-check.json`。

## 缺陷复现与回归

未修改生产代码时，真实 formal nominal 双环境 probe 保存了 98 项检查，其中 35 项失败。角速度 normalized error 约 `0.03465`，重复 reset 的 raw COM/height error 约 `0.001051 m`，tilt failure reset 的 raw height error 达 `1.1964 m`。时钟、非 reset 行、观测调用次数、RNG/history 时序没有相应失败。按 V1 停止条件，先记录证据并升级为 V2，未实施角速度-only 补丁。

canonical red 证据是 `artifacts/reset-observation-20261009/red-formal-03/{summary.json,evidence.pt}`。此前 `red-formal-01/02` 是测试准备/接口问题的调查产物，保留历史，不能当作 canonical red。

修复后运行：

| 验收 | 结果 | 原始日志/产物 |
|---|---|---|
| 全部纯单元测试 | 164 passed，3.57 s | `artifacts/reset-observation-unit-01.log` |
| Isaac reset integration | 3 passed，41.71 s；内部六个全新 Kit 子进程 | `artifacts/reset-observation-green-02.log`、`artifacts/reset-observation-20261009/pytest-green-02/` |
| MuJoCo adapter | 7 passed，0.71 s | `artifacts/reset-observation-mujoco-tests-02.log` |
| 原 checkpoint/export golden replay | 32-vector stored replay 误差 0；batch 1/3/32 同 batch eager/scripted 误差均为 0 | `post-fix-evaluation/golden-vector-replay.json` |

六个 Isaac probe 分别覆盖 formal nominal、Fudan 随机化、实际 sim2sim debug subclass；每种都运行新生成 cache 和加载已有 cache。每个八个 case：constructor、四次 repeated reset、partial、timeout auto-reset、真实 tilt failure auto-reset。共 48 个 case、638 项检查通过。还验证缓存双读的 bitwise/RNG neutrality、七流 restore 后 reset 可重现、empty reset no-op、first resume trace 和 debug 三相原始数据。

独立 direct PhysX root pose/velocity/COM local offset 对照：raw angular error 最大 0，raw COM 最大 `4.76837158203125e-7 m`，raw height 最大 `1.4901161193847656e-8 m`。reset 时钟/步计数/data timestamp 不前进；正常 step 仍为四个 0.005 s 物理步；partial 未 reset 行及其 history 保持既有语义。MuJoCo 测试先正常 step 实际污染 history，再连续四次 reset 对照 keyframe、观测、history 与 time。

golden replay 的临时检查曾错误地将 batch 1/3 结果与 stored batch-32 的前缀按 `1e-7` 比较，受 float32 的 batch 计算舍入影响出现约 `1e-6` 差异。核对现有 exporter 后按同 batch stored replay、同 batch eager/scripted 比较执行，误差均为 0；没有放宽官方阈值或修改导出代码。原始差异与检查修订也保留在 replay JSON。

## 独立代码复核

只复用用户授权的同一个 `reset_document_review` agent。第一轮 `P0=0, P1=0, P2=2, P3=0`，两项 P2 是 MuJoCo 旧 history 未实际污染、debug 三相和来源证据不足。补强后最终 `P0=0, P1=0, P2=0, P3=0`；reviewer 同时只读核对已有 green 产物和 tensor，没有启动仿真或训练。详见 [代码复核记录](2026-10-09-wheelleg-reset-observation-code-review.md)。

## 八场景修复后诊断

全新输出目录：`artifacts/reset-observation-20261009/post-fix-evaluation/`。Isaac 先用固定 Phase 1R run-03 PPO 生成八环境 nominal reset cache，再让 DreamWaQ run-01 使用同一份 cache 和新 baseline report。沿用 499 control steps/9.98 s。MuJoCo 用原 DreamWaQ 导出三件套和原 checkpoint 训练 context，沿用八场景各 500 ticks/10 s 的协议，失败即停止该场景。

| 模型/引擎 | 完整场景 | 主要结果 | 与原报告比较 |
|---|---:|---|---|
| PPO run-03 / Isaac | 8/8 | 平均回报 `27.676682819068446` | aggregate 精确相同 |
| DreamWaQ run-01 / Isaac | 8/8 | 平均回报 `22.00532790293437`；低于 PPO | aggregate 和 estimator 精确相同 |
| DreamWaQ run-01 / MuJoCo | 0/8 | 平均存活 `0.9525 s`，动作饱和占比 `70.1504%` | aggregate 精确相同，八份 CSV hash 全部相同 |

DreamWaQ Isaac velocity MSE 为 `0.0016300842766609582`，相对零估计器 ratio 为 `0.0036810867976171803`，既有 velocity 门通过；baseline comparison 为 false。

| MuJoCo 场景 | 存活秒数 | 实际失败原因 |
|---|---:|---|
| nominal_stand | 1.10 | tilt |
| low_stand | 1.14 | tilt |
| high_stand | 1.12 | base_height |
| forward | 0.48 | tilt |
| reverse | 1.02 | tilt |
| left_turn | 1.14 | tilt |
| right_turn | 1.12 | base_height |
| combined | 0.50 | base_height |

这些结果与缓存回归不矛盾：标准评估从全新进程的初始状态开始，结束后不再把新 reset 观测送入下一 episode 的动作；它没有覆盖训练中连续 episode 的非零终端状态 reset。MuJoCo 的 actor、runtime 与模型也没有变更，旧权重不会因修复 Isaac 缓存而自动变好。具体是否由旧训练污染、两端映射或物理差异主导，不能由本次结果区分。

## 结果身份与正式门边界

DreamWaQ checkpoint SHA256 `F3FE2AAEE7975AD64FB7C5F0686D7632AFAD81A16419415135AF56680595BA4C`；actor.ts SHA256 `4C2C64C2D9F9F88532FCEFF4AE45AA049F970646761D2D0ACE0A29678621D6C7`。没有新 checkpoint 或新导出。

`provenance-before.json` 保存运行时 architecture、env、evaluator、checkpoint、导出三件套和训练 context 的身份；`architecture-at-evaluation.md` 保存评估时架构原文，其 SHA256 为 `4E0987DE794964371995D6957BCDBFE052B5C1B51A79C2185C7F3F478E9F159A`。评估后的文档状态更新不改变这个运行时身份。

`verified-comparison.json` 保存六份 reset summary/evidence/source hash 核对、tensor 重算、三份 evaluation report hash 校验、Isaac 共享 cache/contract 校验，以及 MuJoCo CSV 与旧结果逐项 hash 比较。三个新评估报告分别在 `isaac-baseline/summary.json`、`isaac-dreamwaq/summary.json`、`mujoco/summary.json`，均有原始日志。

当前八场景运行是用户授权的诊断评估，保留 evaluator 自身的 schema/context 校验，没有修改或跳过校验；它不是 RootCauseSuiteCoreV1.17 的正式 verdict。旧 RootCauseSuite 硬编码了旧 architecture/env 身份；没有改写该门、resume 旧 run 或冒充 G00/G01/G02 通过。下一步先建立与新源码绑定的显式诊断修订，按新 run 完成 G00/G01/G02，并按 gate 进入两端语义检查，再安排修复后新的训练候选。重训时不得续旧 suite，结果仍需重新验收。

## 实际执行命令

以下命令从 `E:\wheel_leg_rl-main\wheelleg_dreamwaq` 执行。验收输出已存在，复跑必须换新目录，不能覆盖本次证据。

```powershell
& .\.venv\Scripts\python.exe -B -m pytest tests/unit -q -p no:cacheprovider --basetemp=artifacts/reset-observation-20261009/pytest-unit-01
& .\.venv\Scripts\python.exe -B -m pytest tests/integration/test_reset_observation_coherence.py -q -p no:cacheprovider --basetemp=artifacts/reset-observation-20261009/pytest-green-02
& .\sim2sim\mujoco\.venv\Scripts\python.exe -B -m pytest sim2sim/mujoco/tests/test_mujoco_adapters.py -q -p no:cacheprovider
& .\.venv\Scripts\python.exe -B scripts/evaluate_isaac.py --checkpoint 'logs/rsl_rl/wheelleg_flat_ppo/2026-10-07_07-32-32_rtx5070_fudan-v1_phase1r-v3-suite-20261007-062147-run03-seed1884612625/model_999.pt' --reset-cache 'artifacts/reset-observation-20261009/post-fix-evaluation/nominal-reset-cache.pt' --output 'artifacts/reset-observation-20261009/post-fix-evaluation/isaac-baseline' --headless
& .\.venv\Scripts\python.exe -B scripts/evaluate_isaac.py --checkpoint 'logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-07_18-57-15_rtx5070_fudan-v1_dreamwaq-v1-suite-20261007-185711-run01-seed917410003/model_999.pt' --reset-cache 'artifacts/reset-observation-20261009/post-fix-evaluation/nominal-reset-cache.pt' --baseline-report 'artifacts/reset-observation-20261009/post-fix-evaluation/isaac-baseline/summary.json' --output 'artifacts/reset-observation-20261009/post-fix-evaluation/isaac-dreamwaq' --headless
& .\sim2sim\mujoco\.venv\Scripts\python.exe -B scripts/evaluate_mujoco.py --policy 'artifacts/phase2_dreamwaq/training-suite-20261007-185711/run-01-evaluation/export/actor.ts' --manifest 'artifacts/phase2_dreamwaq/training-suite-20261007-185711/run-01-evaluation/export/policy_manifest.json' --suite-context 'artifacts/phase2_dreamwaq/training-suite-20261007-185711/run-01-evaluation/evaluation-suite-context.json' --run-index 1 --completed-iterations 1000 --output 'artifacts/reset-observation-20261009/post-fix-evaluation/mujoco'
```
