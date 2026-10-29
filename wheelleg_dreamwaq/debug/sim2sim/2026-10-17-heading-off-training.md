# 关闭训练航向保持的 1000 轮对照

用户已确认：只关闭训练时的航向保持，从零训练同种子 1000 轮，检查腿部运动是否改善。本文记录已批准方案和执行步骤，不改变现有航向开启配置。

## 条件

对照为 `2026-10-17_00-02-28_rtx5070_fudan-v1_baseline-headingp9-env6144-fresh1000-20261017/model_999.pt`。新增 profile 为 `stop_reverse_raw_split_straight_yaw2_headingoff_v1`，与该对照的有效任务配置唯一差异为 `heading_hold.enabled: true → false`。保留控制器配置中的 P=9、D=0、最大偏航速度 1 rad/s，便于训练后使用同一控制器做开关评估。关闭时源命令直接进入网络和奖励，不叠加纠偏。

6144 环境，seed=41882943，cuda:0，每轮 24 步，8 个 mini-batch，5 个 epoch，目标 1000 轮。禁止 resume 和 weights-only 初始化。前后和偏航跟踪主项/增强项各 2；腿/轮平滑 −0.02/−0.01，合并项 0，action_rate −0.01，差速 −0.1，其他奖励与旧配置完全一致，ROGER 关闭。

动作无 ±1 执行裁剪，腿目标角限幅、history t−4…t、Actor/Critic/CENet、PPO、随机化、动力学、USD/XML、命令练习和失败条件保持现状。

差速门控公式继续使用有效命令。当航向关闭时，有效命令等于源命令，因此直行差速项可能比航向开启时更经常启用。这是关闭航向反馈之后的既有门控响应，本实验同时记录它，不能把结果解释为已经分离了纠偏指令与门控占用率的各自因果影响。

## 文件和职责

- `schemas/heading_control.py`：增加关闭 profile 标识及严格的完整航向契约识别；不改变控制器更新公式。
- `tasks/direct/wheelleg_flat/training_profiles.py`：让两个 profile 共用原始动作、奖励与几何配置，只选择不同的 enabled 值；恢复时区分两种身份。
- `schemas/dreamwaq_manifest.py`、`deployment/manifest.py`、`sim2sim/mujoco/wheelleg_mujoco/contract.py`：接受具名关闭配置，继续拒绝未知参数、字段缺失和非法开关类型；开关进入原有契约 hash。
- `sim2sim/mujoco/wheelleg_mujoco/evaluation.py`：默认评估身份读取保存的真实开关，继续拒绝与模型身份不同的默认评估模式；独立双模式诊断沿用现有入口。
- `scripts/train_dreamwaq.py`：增加一个 profile 入口。
- `tests/unit/test_heading_control.py`、`tests/unit/test_split_action_smooth.py`、MuJoCo 航向测试：验证关闭时命令直通、配置往返、原权重、严格身份和真实 runtime。
- 正式主架构 §35.4：记录独立关闭配置和保持不变的范围。
- 独立启动目录：保存 launch、检查脚本、冻结指纹、stdout/stderr 和监控记录；运行目录保存模型、训练摘要、导出和评估。

## 执行与验收

- [x] 实施具名配置，确认关闭配置只改变 enabled 值及派生 hash，算法契约完全相同。
- [x] 相关单元测试与真实 MuJoCo 检查通过，拒绝开关类型/参数损坏以及跨配置 resume。
- [x] 独立 8 环境、1 次更新训练接口验收通过：配置与开启对照仅 enabled 不同、有限损失、fresh checkpoint 重载、正式导出及 MuJoCo 加载；该模型不作为 1000 轮结果。
- [ ] 从零启动 6144 环境 1000 轮，保存启动命令、有效配置与源码指纹；验证第一份 model_0 完成 1 次更新、优化器 40 步，无已加载优化器。
- [ ] 每半小时检查进程身份、真实迭代、损失有限性、GPU、checkpoint 和指纹；正常推进时保持安静，异常不自动重启或改变条件。
- [ ] 结束后严格核对 1000 次更新、CENet 1000 rollouts 和 model_999；正式三件套同 batch golden-vector 最大误差 ≤1e−7。
- [ ] MuJoCo 八场景各 500 控制步，航向开启/关闭及三组切换协议，比较本次与上述 6144 环境 1000 轮对照的速度、停车、偏航、存活、四驱动关节实际角加速度及力矩饱和。Isaac 仅用于训练和训练接口验收。

本次为单种子训练对照，不能把结果推广为多种子结论。训练过程中冻结源码，不追加训练或自动调整奖励。

## 训练前验证证据

纯单元测试 77 项、MuJoCo 航向与加载测试 15 项通过。独立验证运行位于 `logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-17_12-35-06_rtx5070_fudan-v1_validation-headingoff-env8-update1-20261017`，仅完成 1 次更新，优化器步数 40；正式导出同 batch golden 最大误差 0.0。MuJoCo 实际加载该导出，确认初始开关关闭、有效命令等于源命令并完成物理 step。原航向开启的 1000 轮正式导出继续能够加载，未改写其原件。检查记录位于 `debug/sim2sim/runs/validation-headingoff-env8-update1-20261017-launch/monitor_snapshot.json`。

补充默认评估模式检查后，MuJoCo 航向、加载和评估测试共 39 项通过，相关纯单元测试重验 52 项通过。目标训练唯一启动目录为 `debug/sim2sim/runs/baseline-headingoff-env6144-fresh1000-seed41882943-20261017-launch`，实际运行路径由其中 `launch.json` 记录，监控使用 `debug.sim2sim.inspect_heading_off_training --launch <启动目录>`。训练期间本文与源码一起冻结。
