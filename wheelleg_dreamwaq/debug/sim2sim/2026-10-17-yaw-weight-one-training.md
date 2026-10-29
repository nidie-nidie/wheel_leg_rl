# 偏航两项权重恢复为 1 的训练对照

日期：2026-10-17。用户确认本次只调整偏航跟踪两项权重，再从零训练 1000 轮。

## 条件与比较对象

新增具名配置 `stop_reverse_raw_split_straight_yaw1_headingoff_v1`。直接对照为已完成的 `logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-17_12-38-59_rtx5070_fudan-v1_baseline-headingoff-env6144-fresh1000-20261017`，该模型训练时航向保持关闭、偏航两项权重各 2。

本次仅将 `tracking_yaw_rate`、`tracking_yaw_rate_enhance` 各从 2 改为 1；完整 base task contract 除这两个字段与派生 hash 外必须相等。完整 algorithm/export contract 必须相等。既有两个 yaw2 配置保持原值，旧模型、导出及日志保留原身份。

从零训练，6144 环境，seed41882943，cuda:0，24 步/轮，8 个 mini-batch，5 个 epoch，1000 轮；不使用 resume 或 initialize-from。航向保持关闭，保留 P=9、D=0、max_yaw_rate=1 的部署开关参数。

原始六维动作、腿目标角边界、腿/轮平滑 −0.02/−0.01、合并平滑 0、action_rate −0.01、差速 −0.1、前后速度两项各 2、其他奖励与 sigma、命令、history、网络/CENet/PPO、随机化、PhysicsV5、USD/XML、MuJoCo 初态与控制均保持。ROGER 关闭。

## 实施顺序与职责

1. `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/training_profiles.py`：保存三个独立具名配置，按完整权重与完整航向配置共同识别身份；新增配置只改变上述两项权重，切换回旧配置恢复旧值，跨配置普通 resume 拒绝。
2. `source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/dreamwaq_manifest.py`：原始动作契约允许已经严格识别的新具名配置，保持旧版本、hash 与载荷。
3. `scripts/train_dreamwaq.py`：新增一个 CLI profile 选项。
4. `sim2sim/mujoco/wheelleg_mujoco/contract.py`：部署元数据复用相同具名权重/航向识别，拒绝未知权重或组合；不改变电机控制、物理或评估实现。
5. 新增 `tests/unit/test_yaw_weight_profile.py`，并运行既有拆分平滑、航向、契约、checkpoint 与 MuJoCo 测试：检查两项之外配置精确相等、旧配置不变、权重缩放、契约往返、非法组合和跨配置恢复拒绝。
6. 主架构 §35.5：记录新具名对照与边界。本文和独立启动目录记录实验条件；新增监控只允许两个预期权重变化，读取真实迭代、第一份 checkpoint、有限标量和冻结指纹。

本次在当前工作区串行实施，不调用其他 agent。修改范围限制于上述身份、入口、测试与记录文件；不修改 rewards.py 的公式或默认值。

## 验证、启动与监控

- [x] 单元测试覆盖 profile 往返、旧值保留、两个实际偏航奖励分量减半且其他分量相同、未知组合拒绝。
- [x] 既有两份正式 1000 轮 checkpoint 严格加载，以及原导出真实 MuJoCo 加载保持可用。
- [x] 独立 8 环境 1 次更新训练接口验收，实际 manifest 与直接对照只相差两个权重和派生 hash；算法/导出契约精确相等；fresh model_0 完成 1 次更新、优化器 40 步。
- [x] 正式导出三件套，同 batch golden 最大绝对误差 ≤1e−7；MuJoCo 实际加载、命令直通和原始动作物理 step 通过。接口模型不计作目标训练完成。
- [ ] 唯一启动 `baseline-yaw1-headingoff-env6144-fresh1000-20261017`，使用独立 launch 目录及日志；6144 环境、1000 轮、seed41882943，冻结源码和设计文档。
- [ ] 每半小时检查实际命令、Loss/total 的 step+1、有限标量、CUDA/OOM/NaN、checkpoint、GPU 和冻结指纹；正常推进保持安静，异常不改变条件、不重复启动。

启动前结果：80 项项目单元测试、27 项 MuJoCo 检查通过。第一次 8 环境验收完成更新后，在契约校验处发现局部导入遗漏，保留该失败记录；修复导入并新增三个 raw profile 的契约版本回归检查后，重新执行完整接口验收。

通过的独立验收目录为 `logs/rsl_rl/wheelleg_flat_dreamwaq/2026-10-17_14-54-16_rtx5070_fudan-v1_validation-yaw1-headingoff-env8-update1-20261017-v2`，启动记录在 `debug/sim2sim/runs/validation-yaw1-headingoff-env8-update1-20261017-v2-launch`。其 `monitor_snapshot.json` 确认 fresh、1 次更新、优化器 40 步、115 个冻结文件、有限标量和预期两个字段的差异。正式导出误差为 0；`mujoco_verification.json` 记录新接口模型及旧 yaw2 开启/关闭模型各 5 次真实控制 step，三者 golden 最大误差均为 0，非法权重被拒绝。短验收仅验证训练和部署接口，不评价行为表现。

## 完成验收

以 training_summary 的 completed_iterations=1000、iterations_executed=1000、starting_completed_iterations=0、CENet completed_rollouts=1000 及 model_999.pt 为准。严格加载最终 checkpoint，核对有限值、学习率和 reset 缓存绑定；正式三件套保持同 batch golden 最大误差 ≤1e−7。

行为数据只在 MuJoCo 收集。运行现有 `debug.sim2sim.evaluate_baseline_mujoco --formal-export <本次export> --output <本次evaluation> --heading-mode both --transitions`，保留八场景各 500 控制步、三组切换协议、XML、初态、0.02 秒控制周期和失败条件。

分别在相同部署开关下，与上述 yaw2/headingoff 1000 轮模型比较速度 MAE、停车蠕动、航向漂移、实际偏航速度、存活、四驱动关节角加速度 RMS 和力矩饱和。另报排除最初 1 秒的持续角加速度；检查换向、转向释放、H 开关和 reset 是否实际达到事件。提前失败则明确相应行为未验收，不以 reward 判断减抖。

本次只有一个种子，结果不能推广为多种子结论。两个偏航分量一起减半，结论仅适用于此联合权重变化。差速门控公式保留，有效命令与状态分布变化可能改变其利用率，需记录。完成后保存本次结果并暂停定时检查，不自动追加训练或继续搜索其他权重。
