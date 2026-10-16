# 腿轮分项平滑及直行差速奖励复核

日期：2026-10-16。复核者：`split_straight_reward_review`。仅调用一个 agent，未创建其他 agent；复核期间未启动训练或 Isaac。

## 结论

没有发现阻塞问题，也没有发现文档与实现之间需要修改的确定性错误。本次代码保持原样。

复核范围为 `debug.md`、`rewards.py`、`environment.py`、`train.py`、`experiment.py`、`export.py`、`mujoco_adapter.py`、`play_mujoco_viewer.py`、`test_rewards.py` 和 `verify_rewards_mujoco.py`，并核查正式奖励、环境、动作契约、Isaac Lab 本地 step 顺序及实际旧工件。

## 已核对的实现

- 腿、轮二阶动作平滑权重分别为 -0.02、-0.01；正式奖励识别分项权重子类后删除旧合并项，没有重复扣分。
- 差速成本为两个方向一致的原始轮动作之差的平方，只有原始 `command_vx != 0` 且 `command_yaw_rate == 0` 时启用。没有改写动作、目标或力矩。
- 差速项在父环境推进命令之前计算，之后加入总奖励和 episode 累计值各一次；父环境通过权重实例字段创建累计项，包含 `wheel_differential`。Isaac Lab 的 episode reset 位于奖励累计之后。
- 新旧实验使用不同训练身份，系数、保存权重及奖励契约需要一致；旧默认 profile 与实际旧训练 manifest 的权重相同。
- 新 profile 缺少显式差速系数时，在创建 AppLauncher 之前拒绝启动；没有训练系数默认值。

## agent 独立执行的检查

- 纯 Torch 回归：`53 passed, 1 deselected`，排除 PPO 更新测试，没有执行优化器更新。
- 实际旧 `model_999.pt` 加载通过，实验标识 `DebugNoActionClipV2`，完成轮次 1000。
- 真实 MuJoCo 控制器、轮符号、history、reset 和力矩边界检查通过。
- 旧导出 loader 和保存的 golden vectors 检查通过，最大绝对误差为 0。
- 原清单的 100 个正式文件 SHA256 一致。

## 尚未执行的验收

新 profile 尚未运行真实 Isaac 训练链路，没有新训练 checkpoint、导出或运动表现结果。本次对命令切换和 episode 累计的确认来自静态调用顺序核对；单元测试没有实际执行新 profile 的 `RawActionEnv._get_rewards()`。

后续训练前的链路验收需要实际检查新 profile 的奖励日志、episode 累计、停车/换向交界和部分 reset，以及新 checkpoint 保存/重载和导出。现有测试通过不能证明新奖励已改善偏航、腿抖动或速度跟踪。
