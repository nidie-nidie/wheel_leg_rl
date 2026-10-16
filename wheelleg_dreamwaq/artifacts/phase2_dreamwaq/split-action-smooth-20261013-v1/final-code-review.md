# 腿轮 action_smooth 分项：最终代码复核

复核日期：2026-10-13。复核方式：只读检查实际工作区文件、架构 §34、实施计划、前一轮文档复核、生产 diff、新增单元/集成/MuJoCo 测试及 debug sampler/worker。未修改生产代码，未启动训练，未调用其他 agent。

## 结论

**P0=0，P1=0，P2=1。当前实现可以开始训练。**

此前已记录的验证结果为：

- `wheelleg_dreamwaq/tests/unit`：238 passed。
- `wheelleg_dreamwaq/sim2sim/mujoco/tests`：68 passed。
- `sampler-preflight`：`nominal_stand`、`forward`、`reverse` 三个主场景均完成 500 个控制步和 10,000 次原始 `mj_step`；包装采样与普通采样的控制、history、raw action、末端 q/v 均 exact；每场 8 个 1 ms 窗口有效。

## 核对通过项

### 1. 奖励公式没有重复计罚

`wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/rewards.py:43-59,102-108` 使用独立的 `SplitSmoothRewardWeights`。分项配置把继承的 `action_smooth` 固定为 0，并只返回 `action_smooth_leg` 和 `action_smooth_wheel`。两项分别对 `d[:4]` 与 `d[4:]` 求平方和，未做维数平均，也没有额外动作缩放。

`.../rewards.py:112-125` 只遍历实际返回的 terms，因此不会同时收取旧总项和两个分项。等权 `-0.01/-0.01` 等价性、腿轮隔离和腿权重加倍均有 `tests/unit/test_split_action_smooth.py:28-57` 覆盖。

### 2. 旧 profile 和 checkpoint 身份仍然严格

`.../training_profiles.py:49-64` 在切换回旧 profile 时重新构造原 `RewardWeights` 类型，避免残留分项字段。`.../training_profiles.py:76-126` 先识别 V3 split schema，再验证完整权重、停车/换向 schedule 和与 reversal/acceleration 试验的互斥条件；缺失或矛盾值拒绝。相关往返和拒绝测试见 `tests/unit/test_split_action_smooth.py:60-118`。

`schemas/dreamwaq_manifest.py:143-156,219-237` 只按实际权重选择 V2/V3 schema，完整 reward payload 继续进入 base task contract hash。网络、CENet、history、storage、动作张量和 checkpoint 张量布局均未被改变。

### 3. MuJoCo V3 边界正确

`sim2sim/mujoco/wheelleg_mujoco/contract.py:88-109,145-151` 只对 `DreamWaQPolicyExportV1` 接受 `RewardSchemaV3SplitSmooth`，并严格检查完整权重、支持的腿权重 `-0.02/-0.04`、轮权重、vx 两项和 StopReverse schedule；PPO 仍只接受 V2，伪装 V3 会拒绝。原有 actuator、physics、history 和模型 hash 检查继续执行。

真实 loader 的接受/拒绝覆盖在 `sim2sim/mujoco/tests/test_split_smooth_contract.py:30-60`，fresh/resume/export/golden/MuJoCo smoke 覆盖在 `tests/integration/test_split_action_smooth.py:14-46`。

### 4. debug sampler 没有改变物理

`artifacts/debug/sim2sim/split-action-smooth-20261013-v1/sample_physics.py:38-48,51-78` 包装原始 `mujoco.mj_step`，每次只调用一次并在 `finally` 恢复；只记录 q、qdot 和相邻原始步差分得到的 1 ms qddot。控制、目标处理、reset、物理 timestep 和原始 controller 未改动。已有 preflight 工件记录三场景的 10,000 次调用和 wrapped/plain exact 结果。

`run_experiment.py:157-178,224-237` 的长跑从零启动 1000 iteration，不使用 `--resume` 或 `--initialize-from`；只有首轮主速度/存活通过而减抖门槛未通过时，才最多追加同种子腿 `-0.04` 的另一档，符合 Architecture §34.4 的有界授权。

## P2：采样脚本的调用次数约束目前依赖 preflight，而不是脚本自身断言

位置：`artifacts/debug/sim2sim/split-action-smooth-20261013-v1/sample_physics.py:121-126`。

脚本记录 `original_step_calls`，但自身没有显式断言它等于 `20 * valid_control_ticks`，也没有把这个条件写进 `metric_valid`。当前 preflight 已经实际证明三场景为 `500 * 20 = 10,000`，因此这不是训练阻塞；后续正式验收时建议把该条件写成断言，避免采样器未来被误改后仍只凭窗口数量通过。

## 训练判定

没有发现 P0/P1 阻塞。可以开始 `stop_reverse_split_smooth_v1` 的独立 fresh 1000 iteration 训练。训练过程中保持代码冻结，并按现有 worker 记录 checkpoint、export、golden vector、MuJoCo 八场景、四组停车/换向和 1 ms/20 ms 实际关节指标。若首轮满足存活和主要速度门槛但减抖不足，才允许同种子从零追加一次 `stop_reverse_split_smooth04_v1`；不要在训练中途改权重，也不要把通过数值门槛直接表述为肉眼已无抖动。
