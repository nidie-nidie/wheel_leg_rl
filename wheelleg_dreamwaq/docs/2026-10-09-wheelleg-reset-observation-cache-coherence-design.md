# Reset Observation Cache Coherence Design and Implementation Plan

> **执行方式：** 主 agent 按 `executing-plans` 逐项执行；用户另行明确授权同一个 `reset_document_review` agent 复核代码。没有调用第二个 agent，不启动训练，不修改 MuJoCo 参数。

**日期：** 2026-10-09  
**设计版本：** ResetObservationCacheCoherenceV2（V1 文档复核历史保留）  
**状态：** V2 两个缓存失效语句已实施；独立代码复核最终 `P0=0, P1=0, P2=0, P3=0`。六个真实 Isaac 子进程的 48 个 case、638 项检查通过，单元测试 164 passed，MuJoCo adapter 7 passed；八场景诊断已完成，旧模型 MuJoCo 仍 0/8，没有训练。详见第 11 节与结果记录；正式 RootCauseSuite 门仍未关闭。  
**Goal：** reset 后，现有角速度、COM 位置和高度观测与本次物理状态一致，五帧 history 使用正确的 first policy observation。  
**Architecture：** 在正式 `WheelLegFlatEnv._reset_idx()` 读取新状态前，明确失效 Isaac Lab 2.3.2 的 `_root_link_vel_w` 与 `_root_com_pose_w` 两个派生缓存。继续使用现有 API、坐标转换、观测组装和 partial-reset 快照合并；其他字段存在 split 时停止并定位，不扩大补丁。  
**Tech Stack：** Isaac Lab 2.3.2 / Isaac Sim 5.1.0 / PyTorch 2.7.0+cu128 / TensorDict 0.14.2 / RSL-RL 3.1.2 / MuJoCo 3.14.0。

## 1. 用户确认的范围

用户已确认：修复的是 reset 后生成神经网络输入的过程，保留原来的输入字段、维度、数学契约、初态和控制时序。随后明确授权按文档修改代码、调用一个 agent 复核、跑 sim2sim，并给出是否需要重训的结论；没有授权本次启动训练。

正式架构是 `../../docs/2026-10-03-wheelleg-dreamwaq-architecture.md`，当前 Architecture v0.23。V1 文档阶段为 v0.22；真实 COM red 证据后升级为 v0.23。修改前 v0.21 SHA256：

```text
5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE
```

v0.23 第 18.1 节和 ADR-067 是当前修复的上游规范。10 月 9 日 handoff、RootCauseSuiteCoreV1.17、调查和结果复核保留原有历史身份，不追改其冻结 SHA256。第 2、3 节保留 V1 诊断过程；V2 的真实证据与当前结果见第 10、11 节。

## 2. 原因和证据边界

修复前生产 `env.py` SHA256：

```text
5399B29B92B54851EB0A5FA880EA9386C82F202A5C337F19ABE4EDF300C2969C
```

以下行号均指修复前源码，相对路径以 `wheelleg_dreamwaq/` 为根：

| 文件与行号 | 已确认行为 |
|---|---|
| `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py:882` | 写本次 root pose、root velocity 和 joint state。 |
| 同文件 `:669` | 角速度读取 `robot.data.root_link_ang_vel_b`。 |
| 同文件 `:911` | `_reset_idx()` 立即 `_read_state()`，之后缓存/合并为 `self._state`。 |
| 同文件 `:699` | `_current_state()` 优先返回已有快照。 |
| `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/assets/articulation/articulation.py:515` | velocity writer 更新 COM velocity 数据，并在 `:525` 写 PhysX；没有失效 `_root_link_vel_w`。 |
| `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/assets/articulation/articulation_data.py:493` | 只有派生缓存 timestamp 小于 `_sim_timestamp` 时，才从当前 COM velocity 重算 root-link velocity。 |
| 同文件 `:813`、`:868` | body-frame angular velocity 经过 root-link velocity 派生缓存。 |
| `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/envs/direct_rl_env.py:315` | full reset 调用 `_reset_idx()`，随后 write/forward，再返回观测；没有正常物理 step 的 scene-data timestamp 更新。 |

旧读数进入 Actor 的链路为：

```text
上一 rollout 的 root-link velocity 派生缓存
  -> reset 写入新 PhysX/COM velocity，但缓存仍被认为有效
  -> root_link_ang_vel_b 返回旧角速度
  -> _read_state() 把旧值写入本次状态快照
  -> 原有 noise / normalization / Actor observation
  -> first policy observation 的五份 history
```

调查：`../debug/sim2sim/root_cause_suite/2026-10-08-g01-reset-observation-runtime-investigation.md`。结果复核：`../debug/sim2sim/root_cause_suite/2026-10-08-g01-reset-observation-result-review-01.md`。

已验证生产类/API 的 nominal 单环境复现和 stale-cache 数据流。尚未验证本设计的一行失效操作在完整运行配置下修复成功；也未证明旧 checkpoint 的 MuJoCo 失败只有这个原因。设计必须通过运行证据，不能把理论推导写成测试结果。

第一轮独立文档复核发现另一个需要先诊断的缓存边界：`articulation_data.py:513` 的 `_root_com_pose_w` 在同 timestamp 下不重算，而 `articulation.py:427,438-443` 的 root-link pose writer 没有明确失效它。生产 `env.py:667,671` 读取该 COM 位置/高度，`observations.py:118` 将高度送入 Critic。独立 AST getter/writer 静态模型能复现旧 COM pose，但未运行真实 PhysX；本设计先补 direct COM/height 对照，不直接把新缓存加入生产补丁。

## 3. 方案选择

| 方案 | 影响 | 决定 |
|---|---|---|
| 在 reset 读状态前失效 `_root_link_vel_w` | 精确覆盖已定位的角速度派生缓存；保留读取 API 和初态；固定依赖的一个私有 buffer 写入。 | 作为角速度候选；先诊断 COM/height，任何其他 split 均停止并修订。 |
| 正 `dt` 的 `scene.update()` 后重新读状态 | 不等于 `sim.step()`，但会推进场景数据和传感器计时，并触发关节加速度差分更新；现有诊断曾消除 split。 | 本次不采用；影响范围大于已定位的缓存。 |
| reset 中增加真实物理 step | 改变 first observation 的物理初态；并行环境中还会推进未结束环境；需要重新定义启动控制和 sim2sim 时序。 | 不属于此次缓存修复。 |

`root_com_ang_vel_b` 虽可绕开该派生缓存，但正式架构固定 `root_link_ang_vel_b`；本次保留冻结读取语义。

## 4. 精确生产修复

该角速度候选的唯一生产文件：

```text
source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py
```

V1 的一行候选未实施：真实 red probe 同时确认 COM/height split，按前置停止条件修订为 V2。当前在现有 `_reset_idx()` 的 `reset_state = self._read_state()` 前实施以下两个精确失效操作：

```python
            # Isaac Lab 2.3.2 writers update the base pose/COM velocity but leave
            # these derived root caches valid at the unchanged reset timestamp.
            # Recompute them from this reset without advancing any clock/physics.
            self.robot.data._root_link_vel_w.timestamp = -1.0
            self.robot.data._root_com_pose_w.timestamp = -1.0
            reset_state = self._read_state()
```

机制：writer 已更新本次 COM velocity，pose writer 已更新 root-link quaternion；`-1.0` 使下一次 `root_link_vel_w` 读取走重算分支。重算不修改 PhysX，不修改 simulation timestamp；角速度仍沿用原有 inverse rotation、ControlFrameV1、noise 和 normalization。

这段代码只说明角速度缓存机制，不承诺所有 `WheelLegState` 或整个 Critic 正确。COM 位置/高度独立对照失败时，保留 red 证据并停止一行补丁实施，先定位实际运行路径、写出新的窄设计，再继续；不能删除失败断言、换用缓存数据做 expected 或强制将高度归零。

继续沿用现有 `reset_state.applied_torque[env_ids] = 0.0`、full-reset 快照替换和 partial-reset `replace_rows()`。空 `env_ids` 仍提前返回。buffer timestamp 是全批次的有效标记，因此重算可覆盖全批次数据，但新快照只合并被 reset 的行；必须验证其他行没有变化。

不增加 blanket cache invalidation，不改 Isaac Lab vendor source。固定版本 buffer 不存在时由属性访问明确失败，不添加静默 fallback。该兼容点的依赖源码身份如下：

| 依赖文件 | SHA256 |
|---|---|
| `articulation_data.py` | `D69495C66C409FB20823E4E467F9D90E63163B2050B77D2AC8E433221D1D208A` |
| `articulation.py` | `FB179AACDD848906367D01105020E1D36412BD0F1DA7187BE09F0781FF589309` |
| `direct_rl_env.py` | `3A7573303D83EC8C816D11CA6E8C8997C92F6765AE5D2E0D2A369E65D2CEC3C0` |

## 5. sim2sim 与 checkpoint 边界

`WheelLegSim2SimDebugEnv` 继承正式环境，在覆写 `reset()` 时同样调用 `_reset_idx()`，所以修复放在共同路径，而不是只修 debug worker。正式类与 debug subclass 都要实际验收三相 reset、返回观测和 history。

MuJoCo `wheelleg_mujoco/runner.py:66` 已执行 `mj_resetDataKeyframe -> mj_forward -> collect_kinematic_state -> history fill`；角速度从当前 `data.qvel` 计算。本次保留其 runtime，仅增加重复 reset 的回归测试；没有证据时不增加 MuJoCo warmup 或修改参数。

本设计不改 checkpoint schema、网络 state dict 或三个 contract builder。实际修改 `env.py` 后 source fingerprint 会改变；正式架构 SHA256 本次已改变。不得沿用旧 paused training suite 做续训或把旧报告改名为新结果。现有 checkpoint 已在合法校验路径下做修复后诊断评估，保留原始来源与当前评估源码身份；遇到校验拒绝不得绕过。该诊断不替代 RootCauseSuite 正式 verdict；训练未执行。

## 6. 文件职责与依赖方向

| 文件 | 操作 | 职责 |
|---|---|---|
| `../../docs/2026-10-03-wheelleg-dreamwaq-architecture.md` | 已修改 | v0.23 上游约束、ADR-067、验收边界。 |
| 本文件 | 本次新增 | 窄设计、精确代码插入点、按序实施计划。 |
| 正式 `tasks/direct/wheelleg_flat/env.py` | 已修改 | 失效已定位的派生缓存；保留现有状态生成路径。 |
| `tests/integration/probes/reset_observation_coherence.py` | 已新增 | 全新 Isaac 子进程中的实际 reset/物理状态/history 对照及来源记录；不创建 optimizer。 |
| `tests/integration/test_reset_observation_coherence.py` | 已新增 | 使用现有 `run_project_script()` 启动 probe，校验报告，不复用训练 fixture。 |
| `sim2sim/mujoco/tests/test_mujoco_adapters.py` | 已扩展 | 非零旧状态后的重复 reset 与 history/time 回归；runtime 不变。 |

依赖方向：architecture -> design/plan -> 正式 env -> 原有状态/观测 -> 原有 history -> Actor。测试依赖生产 API 和独立 PhysX 状态作为对照；生产 env 不依赖测试/debug。MuJoCo 不引入训练侧 Python policy。

## 7. 实施步骤

以下清单已按 V2 完成；V1 的角速度-only 前置条件已按第 10 节升级，red/green 与范围证据见第 11 节。代码块为当前测试/实现摘录，实际源文件为运行时依据。

### Task 1：先建立真实失败用例

- [x] 创建独立 probe 与 pytest 入口；按现有 Windows 启动契约，在 `AppLauncher` 前 import PyTorch/TensorDict。
- [x] 在正式环境和 debug subclass 上分别运行 nominal、两个环境的 probe。测试配置使用固定 command，准备阶段产生非零 angular velocity 并正常执行一个控制 step；禁止直接伪造 `_root_link_vel_w.data` 来制造失败。
- [x] 保存实际类名、配置/profile、probe 源码 SHA256、生产/依赖源码 SHA256、原始 direct PhysX quaternion/velocity/COM local offset、返回 policy/critic、raw COM 位置/高度、五帧 history，以及操作前后时钟、观测/噪声调用计数和七流状态。使用全新输出目录。
- [x] 未修改生产代码时，至少重复-reset 用例必须因返回角速度与 direct PhysX 不一致而失败；如果不能复现，停止代码修改并调查测试设置。
- [x] 单独记录角速度与 COM/height 的失败。若 COM/height 或其他字段已有 split，先停止 Task 2，写运行证据并修订候选；不能把角速度-only 补丁当完整修复。

probe driver 必须先独立采集并落盘角速度与 COM/height 各项结果，再汇总抛出失败；下列顺序断言函数用于通过阶段的校验，不能让预期的角速度失败提前中断 COM 诊断，以致误判 Task 2 前置条件。此要求是第二轮 reviewer 的实施提醒，不改变候选机制。

核心对照函数放入新 probe，使用下列完整实现；不从待测 `self._state` 生成 expected：

```python
import torch
from wheelleg_dreamwaq.schemas.frames import (
    quat_rotate_inverse_wxyz,
    quat_rotate_wxyz,
    transform_usd_vector_to_control,
)


def expected_clean_angular_observation(env):
    view = env.robot.root_physx_view
    root_transform = view.get_root_transforms().clone()
    root_velocity = view.get_root_velocities().clone()
    quaternion_wxyz = root_transform[:, 3:7][:, [3, 0, 1, 2]]
    angular_body = quat_rotate_inverse_wxyz(quaternion_wxyz, root_velocity[:, 3:6])
    angular_control = transform_usd_vector_to_control(angular_body)
    return env.cfg.normalization.normalize_angular_velocity(angular_control)


def expected_root_com_position_and_height(env):
    view = env.robot.root_physx_view
    root_transform = view.get_root_transforms().clone()
    quaternion_wxyz = root_transform[:, 3:7][:, [3, 0, 1, 2]]
    local_com = view.get_coms().to(env.device).clone()[:, 0, :3]
    root_com_position = root_transform[:, :3] + quat_rotate_wxyz(quaternion_wxyz, local_com)
    height = (root_com_position[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)
    return root_com_position, height


def reset_clock_signature(env):
    return (
        float(env.sim.current_time),
        int(env._sim_step_counter),
        int(env.common_step_counter),
        float(env.robot.data._sim_timestamp),
    )


def assert_reset_policy_and_history(env, wrapper):
    before = reset_clock_signature(env)
    observations, _ = wrapper.reset()
    assert reset_clock_signature(env) == before
    expected = expected_clean_angular_observation(env)
    torch.testing.assert_close(observations["critic"][:, :3], expected, rtol=0.0, atol=1.0e-6)
    expected_com, expected_height = expected_root_com_position_and_height(env)
    torch.testing.assert_close(env._state.root_com_pos_w, expected_com, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(env._state.base_height, expected_height, rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(
        observations["critic"][:, 28:29],
        env.cfg.normalization.normalize_base_height(expected_height),
        rtol=0.0,
        atol=1.0e-5,
    )
    if not env.cfg.randomization.enabled:
        torch.testing.assert_close(observations["policy"][:, :3], expected, rtol=0.0, atol=1.0e-6)
    frames = observations["policy_history"].view(env.num_envs, 5, 25)
    assert torch.equal(frames, observations["policy"].unsqueeze(1).repeat(1, 5, 1))
    assert torch.equal(observations["policy"][:, 19:25], torch.zeros_like(observations["policy"][:, 19:25]))
    cached = wrapper.get_observations()
    assert torch.equal(cached["policy"], observations["policy"])
    assert torch.equal(cached["policy_history"], observations["policy_history"])
```

probe 完整 case 集合和必须记录的断言：

| Case | 操作 | 判定 |
|---|---|---|
| fresh/full repeated | fresh reset；写非零 world angular velocity、正常 step；连续四轮 reset 对照。 | terminal angular velocity 非零；独立角速度、COM/height、history、时钟检查全部通过。 |
| partial | 先正常 step 保存 post-physics 快照；只对 `[0]` 调用真实 `_reset_idx()`。 | row 0 状态角速度与 COM/height 匹配 direct PhysX；row 1 每个快照字段 bitwise 不变；时钟不变。 |
| timeout auto-reset | 仅令 row 0 的 `episode_length_buf = max_episode_length - 2`，执行正常 step；不改 termination 配置。 | row 0 truncated 且 reset；row 1 未结束；row 0 独立角速度、COM/height 和 history 检查通过；整批只执行规定四个物理 step。 |
| failure auto-reset | 测试准备阶段将 row 0 姿态设到真实 tilt failure；通过正常 step 的 `compute_dones()` 触发。 | row 0 terminated；不得 monkeypatch done；其余约束同 timeout case。 |
| randomized | 使用原有 Fudan profile 与固定已记录 RNG 状态，执行相同 reset 用例。 | clean Critic 的角速度/高度与 raw COM 状态匹配独立物理对照；Actor physical-noise-normalize 残差匹配原有噪声流；同 seed/RNG 的重复序列一致。 |
| debug subclass | 对真实 `WheelLegSim2SimDebugEnv` 重复以上 nominal case。 | pre/post/returned 原始 payload 和 identity 可重算；不只比较存储 hash。 |

partial/auto-reset 的检查必须从真实 production API 的结果取值。randomized case 必须在 probe 中记录实际噪声调用与 RNG 前后状态，不能清掉噪声后冒充 Fudan 配置。

上述核心函数中的 cached 张量比较不能替代 RNG/调用数断言。probe 用只记录再委托原方法的 instrumentation 分别统计 `_get_observations()` 和 `sample_actor_noise()`；不得改变返回值或另行消耗随机数，instrumentation 原文与 SHA256 必须入报告。正式类 wrapper 构造/full reset 各为两次底层 observation/noise 调用（discarded、first）；debug subclass 现有 pre-forward capture 使该数为三，单独记录，不把三改成二。连续调用两次 wrapper cached `get_observations()` 前后，以上计数必须不变，七个 stream 的 generator state 必须逐字节相等，三组返回张量必须 bitwise 相等。RNG 快照使用 `env.get_randomization_rng_state()` 并立即 deep-copy；不调用 `_get_observations()` 或 `sample_actor_noise()` 构造 expected。

### Task 2：实施单一生产修复

- [x] 复核 Task 1 的失败原因与第 2 节数据流一致；自查不能标作独立复核。
- [x] COM/height 已由真实 red probe 定位并纳入 V2；两项都必须 green，其他字段 split 则停止。
- [x] 只在第 4 节指定位置插入缓存失效语句和解释注释。
- [x] 再运行同一 probe、相同测试设置、全新输出目录，要求失败用例转为通过；保留修复前后源码 hash。
- [x] 若不通过，停止扩大补丁，回到数据流调查；不追加物理 step、正 `dt` 的 scene update 或 Actor 数值替换。

### Task 3：MuJoCo 重复 reset 验收

- [x] 在现有 `test_mujoco_adapters.py` 添加以下完整用例。该文件已有 `replace`、`np`、`mujoco`、`MODEL_PATH`、`_contract()` 和 `WheelLegMujocoRuntime`。

```python
def test_repeated_reset_discards_old_motion_and_repeats_current_history() -> None:
    contract = replace(_contract(), policy_input_dimension=125, history_length=5, history_layout="frame_major")
    runtime = WheelLegMujocoRuntime(MODEL_PATH, contract)
    command = np.array((0.0, 0.0, 0.20))
    expected = runtime.reset(command).copy()
    expected_qpos = runtime.data.qpos.copy()
    expected_qvel = runtime.data.qvel.copy()
    expected_time = float(runtime.data.time)
    for _ in range(4):
        runtime.data.qvel[3:6] = (0.4, -0.6, 0.8)
        mujoco.mj_forward(runtime.model, runtime.data)
        moving = collect_kinematic_state(runtime.model, runtime.data, runtime.model_map, contract)
        assert np.linalg.norm(moving.angular_velocity_control) > 0.1
        advanced = runtime.step(np.full(6, 0.2), command)
        assert not np.array_equal(advanced.observation, expected)
        runtime.previous_action.fill(0.7)
        actual = runtime.reset(command)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(runtime.data.qpos, expected_qpos)
        np.testing.assert_array_equal(runtime.data.qvel, expected_qvel)
        assert float(runtime.data.time) == expected_time
        frames = actual.reshape(5, 25)
        np.testing.assert_array_equal(frames, np.repeat(frames[:1], 5, axis=0))
```

MuJoCo 的 `mj_resetDataKeyframe` 会恢复 keyframe time；这里断言 reset 返回时间等于 fresh keyframe time，不要求等于上一 rollout 的末尾时间。Isaac reset 时钟则保留 reset 前值；两端本来就不要求时钟绝对数值相同。

### Task 4：回归、自查和结果记录

- [x] 运行既有纯单元测试，重点检查 `test_state.py` 的 partial-row 合并以及 DreamWaQ history/math/checkpoint/export 的现有契约；单元测试不代替实际 Isaac probe。
- [x] 运行新增 Isaac probe 入口与 MuJoCo adapter 测试。禁止运行 `test_dreamwaq_startup.py`，它会触发短训练 fixture，超出本次范围。
- [x] 比较修改文件清单与第 6 节：生产修改只有 `env.py`，测试只服务本缺陷；USD、奖励、动作、命令、随机化、termination、MuJoCo runtime、网络与 optimizer 参数不变。
- [x] 记录通过/失败数量、原始输出与 hash，明确区分自查、独立复核、诊断门和策略性能。未做独立复核不得宣称已完成独立复核。

后续执行命令；新增测试文件落地之前不运行：

```powershell
Set-Location 'E:\wheel_leg_rl-main\wheelleg_dreamwaq'
& .\.venv\Scripts\python.exe -B -m pytest tests/integration/test_reset_observation_coherence.py -q -p no:cacheprovider
& .\.venv\Scripts\python.exe -B -m pytest tests/unit -q -p no:cacheprovider
uv run --frozen --project .\sim2sim\mujoco pytest .\sim2sim\mujoco\tests\test_mujoco_adapters.py -q -p no:cacheprovider
```

## 8. 此后才可安排的诊断与训练

本次设计不修改 RootCauseSuiteCoreV1.17，也不放宽它的硬门。正式修复通过代码复核后，需要建立绑定新 source identity 的显式诊断修订，以全新 run ID 重跑 G00/G01/G02；不能 resume 旧失败 run。通过后才可恢复 P10-P60/C70 和现有 checkpoint 的正式 sim2sim 判断。

用户随后授权的八场景修复后诊断已执行，保留旧 checkpoint 的训练来源，并另记当前源码与新输出身份；没有执行 RootCauseSuite 正式 P10-P60/C70。checkpoint/resume 短训练与新的四 seed suite 仍未授权。性能仍失败时保留失败记录，不倒推缓存修复无效，也不把缓存修复写成已解决全部 sim2sim 问题。

## 9. 设计自查

- 精确修改点位于所有 reset 路径共有的 `_reset_idx()`，不依赖只在 full reset 执行的 `reset()` hook。
- 继续使用冻结角速度 API；新采样角速度不会被强制归零。
- 缓存失效与状态快照重建在同一次 reset 中先后执行；没有只更新 Actor 的旁路。
- 不修改 scene-data timestamp，不增加物理 step，不改变 randomization RNG 调用次数。
- partial reset 的全批次派生重算与 per-row 状态合并已区分，需要用真实测试证明隔离。
- noisy Actor 和 clean PhysX 的对照责任已区分；MuJoCo keyframe time 与 Isaac 全局 time 的断言已区分。
- 一行补丁只是角速度候选；新增 raw COM/height 独立对照与失败停止条件，不用角速度通过推断全部状态正确。
- 正式/debug 底层观测次数分别为二/三；连续 cached 双读需同时证明张量、调用计数和七流 RNG 不变。
- 文档准备完成与生产修复、独立复核、G01/G02、策略性能验收分开记录。

第一轮独立文档复核发现 `P0=0, P1=1, P2=1, P3=0`。修订补上 P1 的 COM/height 边界及停止条件、P2 的调用计数和七流 RNG 断言后，同一 reviewer 第二轮复核通过，最终 `P0=0, P1=0, P2=0, P3=0`。当时批准范围是先真实 probe 诊断、前置条件满足才实施角速度候选的文档/准备方案；不是生产修复或运行验收。两轮历史记录见 `2026-10-09-wheelleg-reset-observation-document-review.md`。随后用户另行授权的 V2 代码复核与真实运行结果见第 11 节；仍只使用同一个 agent。

## 10. V2：真实 red 证据后的最小修订

2026-10-09 用户授权实施、一个 agent 代码复核、sim2sim 和是否重训的评估。`artifacts/reset-observation-20261009/red-formal-03/summary.json` 与 `evidence.pt` 是真实生产类双环境 nominal 证据，全部检查分别落盘：角速度和 COM/height 都 split，时钟、partial 非 reset 快照、调用次数、RNG/history 时序均没有相应失败。repeat raw COM error 约 `0.001051 m`，failure reset raw height error `1.1964 m`。原 V1 Task 2 按规则暂停，未实施角速度-only 补丁。

V2 只增加 `_root_com_pose_w.timestamp = -1.0`，与角速度派生缓存同时在 `_read_state()` 前失效。其 getter 从 writer 已更新的 root-link pose 加 local COM offset 重算；没有 PhysX 写入、时间、RNG 或观测字段变化。未消费的 combined root buffers不加入补丁，不改 vendor/MuJoCo。原 V1 的“其他字段 split 则停止”约束继续适用于 V2 的运行回归。

第 4 节 V2 代码块取代 V1 一行候选；原 Task 2 的 COM 未 split 前置条件由“COM 已定位并纳入 V2，两项均必须 green”取代。其余测试、复核、旧身份和不训练约束保留。V1 原文已在 `artifacts/reset-observation-20261009/source-before/design-v1.md` 保存。V2 须交本次用户授权的单一代码 reviewer 一并复核，不冒用 V1 文档审核结论。

## 11. V2 实施与验收完成状态

第 7 节任务已按 V2 修订完成。真实 red 仅使用正式 nominal 类先定位两种缓存；随后 green 分别覆盖正式 nominal、Fudan 随机化、debug subclass，每组均有新生成与加载已有 reset cache 两个全新子进程。生产只增加三行注释与两个 timestamp 失效语句；没有其他生产文件变更。

| 项目 | 实际结果 |
|---|---|
| 生产修复 | `env.py` SHA256 `1EF7C18B830BE13DA6DD6BB0865A61A63822926F6403C69730B08C5E013D6A22` |
| 独立代码复核 | 同一 agent，两项测试 P2 已补齐，最终 P0/P1/P2/P3 全为 0 |
| 单元测试 | 164 passed |
| Isaac 集成测试 | 3 passed，内部六个全新 Kit 子进程；48 个 case、638 项检查通过 |
| direct PhysX 原始 tensor 对照 | 角速度最大误差 0；COM 最大误差 `4.768e-7 m`；高度最大误差 `1.490e-8 m` |
| MuJoCo adapter | 7 passed，包含真实推进并污染 history 后的四次 reset |
| 原导出 TorchScript replay | 32-vector stored golden 误差 0；batch 1/3/32 与同 batch checkpoint eager 推理误差均为 0 |
| 八场景修复后诊断 | PPO Isaac 8/8，DreamWaQ Isaac 8/8，DreamWaQ MuJoCo 0/8；三个 aggregate 与原评估精确相同 |
| 正式 RootCauseSuite | 未建立新修订、未运行 G00/G01/G02，未关闭正式 P10-P60/C70 门 |
| 训练 | 未启动；旧 paused suite 不得以新源码继续 |

代码复核见 [独立代码复核记录](2026-10-09-wheelleg-reset-observation-code-review.md)。完整命令、来源、报告和重训建议见 [实施与结果记录](2026-10-09-wheelleg-reset-observation-implementation-results.md)。修复后诊断目录为 `artifacts/reset-observation-20261009/post-fix-evaluation/`；其中 `architecture-at-evaluation.md` 保留运行时架构原文，`provenance-before.json` 保留运行时身份，后续文档状态更新不追改原始身份。
