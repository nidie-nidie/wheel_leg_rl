# Wheelleg 4096 / model_700 溯源记录

记录时间：10/18/2026 16:04:56

这份记录固定后续 debug 约束微调使用的行为基线。它只引用已有工件；没有修改正式源码、训练配置或 checkpoint。

## 参考 checkpoint

- 路径：`E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\no_action_clip\runs\fresh1000-split-straight-lambda01-env4096-20261016\model_700.pt`
- SHA256：`A351C42E999F873FDA0921940BA9F2C57FB2143485971DF50A8757B067507FB9`
- 来源实验：`DebugNoActionClipSplitStraightV1`，profile `no_clip_split_straight_v1`
- 4096 环境，seed `41882943`，fresh 初始化，起始完成 `0` 次更新
- `model_700.pt` 的 checkpoint index 为 700，因此它代表完成 `701` 次更新；原请求上限是 `1000`，同一历史运行后来保存到 `729` 次后停止，不能把后来停止点当成 model_700 的训练边界。
- manifest SHA256：`82AB66E75ABA245860EA260B10D68748D8678E313A03951A35ADA2829217BB38`

## 奖励函数和权重

旧 debug manifest 中记录的完整权重如下：

```json
{"action_rate":-0.01,"action_smooth":0.0,"action_smooth_leg":-0.02,"action_smooth_wheel":-0.01,"ang_vel_xy":-0.2,"dof_acc":-2.5E-07,"dof_pos_limits":-1.0,"dof_vel":-5E-05,"height_sigma":0.001,"lin_vel_z":-1.0,"orientation":-100.0,"phi0_symmetry":-1.0,"soft_leg_limit":0.95,"termination":-5.0,"torques":-0.0001,"tracking_base_height":1.0,"tracking_sigma":0.25,"tracking_vx":2.0,"tracking_vx_enhance":2.0,"tracking_yaw_rate":1.0,"tracking_yaw_rate_enhance":1.0,"wheel_differential":-0.1}
```

其中真实关节角加速度项是 `dof_acc = sum(joint_acceleration^2)`，权重 `-2.5E-07`；腿/轮动作二阶差分平滑分别是 `-0.02` / `-0.01`，合并平滑项为 `0`。速度追踪两项各为 `2`，增强速度项各为 `2`；偏航两项为 `1` / `1`。

奖励项按 `weight * control_dt * term` 加权，control_dt 为 `0.02` 秒。真实加速度、动作平滑和轮差速公式详见 JSON；源码位置：RewardWeights 18-37; SplitSmoothRewardWeights 43-54; RawSplitStraightRewardWeights 60-71; term formulas 85-132; weighted sum 135-148。

实现来源核对：旧 checkpoint 的 debug 奖励实现 SHA256 为 `150D64C582DA20C5BCD5C3C39A76CC440C2E08D2A03AC01CB021E845EF183AE4`，旧 debug 环境实现 SHA256 为 `D8040CCB081EA569052F7AA3A13BCDC24F5AD6B55AB239727262B1D069C096EA`；当前正式 `rewards.py` SHA256 为 `6893CB691B06F77D3D8E8AE1869B9039487E175CFF0B1764A6C5E7C49FC96194`。旧 debug 环境是在父环境奖励之后额外接入 `wheel_differential`，当前正式入口把同名项放进 `RawSplitStraightRewardWeights/compute_reward_terms`。manifest 的权重相同，但实现文件不同，因此实现等价性必须单独验证，不能仅凭权重相同声称完全复现。记录时 Git 对正式 rewards.py 显示 ` M source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/rewards.py`；本次只写入 debug 溯源文件，没有改这个正式文件。

## 正式入口的 4096 复现

- 路径：`E:\wheel_leg_rl-main\wheelleg_dreamwaq\logs\rsl_rl\wheelleg_flat_dreamwaq\2026-10-17_17-56-29_rtx5070_fudan-v1_baseline-yaw1-headingoff-env4096-fresh701-20261017\model_700.pt`
- SHA256：`21DEB4222A2B260008DE6B9829447C92E64B21BE268CF0AB66AF6E163062E984`
- 启动目录：`E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\runs\baseline-yaw1-headingoff-env4096-fresh701-seed41882943-20261017-launch`
- 实际命令：`scripts/train_dreamwaq.py --profile rtx5070 --num-envs 4096 --max-iterations 701 --seed 41882943 --run-name baseline-yaw1-headingoff-env4096-fresh701-20261017 --task-profile stop_reverse_raw_split_straight_yaw1_headingoff_v1 --headless --device cuda:0`
- profile：`stop_reverse_raw_split_straight_yaw1_headingoff_v1`；4096 环境；seed `41882943`；device `cuda:0`；每轮 `24` 步；`8` mini-batch；5 epoch；fresh 701 次更新；未 resume。
- training_summary：completed_iterations=`701`，iterations_executed=`701`，starting_completed_iterations=`0`，resumed_from=null，最终 `model_700.pt`。
- run_manifest SHA256：`631782A8F4C34D438155EB304405D742D8C35C465AE8B0972C814B02AB6ECBFB`；training.md SHA256：`00C1E16D02334482B3E157CD8D4C42FFE91B66EEE26AE35E46802603097A69E4`

正式复现的奖励权重与旧 debug 参考相同：前后速度追踪各 2，增强速度各 2，偏航追踪各 1，增强偏航各 1；`dof_acc=-2.5e-7`；腿/轮平滑 `-0.02/-0.01`；合并平滑 0；action rate `-0.01`；轮差速 `-0.1`。heading_hold 训练开关为 false，但部署参数仍保留 P=9、D=0、max_yaw_rate=1；原始六维动作没有运行时 ±1 clip，腿目标角限幅为 [-1,1] rad。

## PPO / DreamWaQ / 物理条件

- PPO：gamma `0.99`，lambda `0.95`，clip `0.2`，value loss coef `1`，entropy coef `0.01`，learning rate `0.001`，adaptive schedule，global gradient clip `1`。
- DreamWaQ auxiliary：velocity coef `1`，reconstruction coef `1`，KL beta `1`。
- 物理：sim_dt `0.005` s，control_dt `0.02` s，decimation `4`，episode `10` s，solver position/velocity iterations `96/4`。
- 命令范围：vx `-1.5, 1.5` m/s，yaw rate `-1, 1` rad/s，base height `0.16, 0.24` m；StopReverse practice stages `100, 150, 250, 300, 400, 500`。
- 域随机化：profile hash `28C4E8DEB19C35F447BE71C5C803986CE9105A0A0D95DD641E8C9C40885DFA18`，摩擦范围 `0.6, 1.4`，kp/kd/effort scale `[0.95,1.05]`，观察噪声与 command/reset 使用独立 generator stream；master seed `41882943`。

## 4096 复现核对

已有 `post_training_analysis.json` 记录显示：旧 debug 与正式入口在 `model_0`、`model_100`、…、`model_700` 的模型张量、优化器张量、学习率、CPU/CUDA RNG 和环境 RNG 均逐位相同；8个 checkpoint 的 `model_tensors_exact_equal` 均为 true，最大参数误差为 0。文件级 SHA 不同是 checkpoint 文件封装/元数据差异，不能据此判断策略参数不同。

旧701 headingOFF MuJoCo 参考：8场景完整存活，速度 MAE `0.0373135404857223` m/s，合并持续腿部角加速度 RMS `0.172726778560975` rad/s²，持续均值绝对加速度 `0.0345402139268735` rad/s²。这个持续 RMS 是后续“腿部不变差”的行为上限参考，不从训练 reward 估计。

机器可读行为来源：`E:\wheel_leg_rl-main\wheelleg_dreamwaq\logs\rsl_rl\wheelleg_flat_dreamwaq\2026-10-17_17-56-29_rtx5070_fudan-v1_baseline-yaw1-headingoff-env4096-fresh701-20261017\post_training_analysis.json`，SHA256 `82A94CB9358A319BCCACA5E4979B48B0D9B64B552AB3A0B2B883303B0BEA9B23`。

## 后续使用边界

这个 `model_700.pt` 是“701 次更新后的行为稳定性参考”，不是理论最优模型。后续 debug 约束微调应从它复制出独立 run，先测出持续关节角加速度 RMS 的参考值，再用时间窗口约束保持不高于参考值加容差，同时把速度追踪 MAE 作为必须改善的目标。单凭训练 reward 不能替代 MuJoCo 行为指标；旧 checkpoint 也没有完整保存原训练每个 PPO batch 的真实 rollout，因此本记录用于固定来源和契约，不声称已由它单独证明某个因果根因。

机器可读完整记录：`E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\provenance\wheelleg_4096_model700_provenance_20261018.json`



