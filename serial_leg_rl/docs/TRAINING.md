# 训练入口

当前先做平地站立训练，4096 个环境，PPO 配置里已经设为 6000 个 iteration。

完整训练只需要这一条命令：

```bash
cd /home/shun/桌面/wheel_leg_rl
source wheel_leg/WheelLeg_RL_IsaacLab/IsaacLab/.venv/bin/activate
TERM=xterm ./wheel_leg/WheelLeg_RL_IsaacLab/IsaacLab/isaaclab.sh -p serial_leg_rl/scripts/train_standing_rsl_rl.py --task SerialLeg-Standing-Direct-v0 --num_envs 4096 --headless
```

TensorBoard 查看训练曲线：

```bash
cd /home/shun/桌面/wheel_leg_rl
source wheel_leg/WheelLeg_RL_IsaacLab/IsaacLab/.venv/bin/activate
python -m tensorboard.main --logdir logs/rsl_rl/serial_leg_standing_direct --port 6006
```

浏览器打开：

```text
http://localhost:6006
```

可视化检查当前仿真模型，不加载 PPO：

```bash
cd /home/shun/桌面/wheel_leg_rl
serial_leg_rl/scripts/view_standing_gui.sh
```

窗口里看形状，同时终端里看 `closure_max` 和 `closure_pairs`。脚本里已经关掉 NGX/DLSS/DL denoiser，避免 viewport 初始化阶段段错误。

脚本内容在 `serial_leg_rl/scripts/view_standing_gui.sh`。

## 训练轮数

当前 PPO 配置在 `serial_leg_rl/source/serial_leg_rl/serial_leg_rl/tasks/standing/agents/rsl_rl_ppo_cfg.py`：

```text
max_iterations = 6000
num_steps_per_env = 24
save_interval = 100
```

4096 个环境时，每个 iteration 采样：

```text
4096 * 24 = 98304
```

6000 个 iteration 总采样量约为：

```text
6000 * 4096 * 24 = 589824000
```

这是一条网络训练 6000 个 PPO iteration。以后做四组消融实验时，如果配置不改，就是每一组各训练 6000 个 iteration。

## Checkpoint

checkpoint 默认保存在：

```text
logs/rsl_rl/serial_leg_standing_direct/<运行时间>/
```

例如当前已经完成的一次训练在：

```text
logs/rsl_rl/serial_leg_standing_direct/2026-08-23_04-25-39/
```

里面的 `model_100.pt`、`model_200.pt` 等文件是按 `save_interval=100` 落盘保存的快照，最后还有 `model_5999.pt`。

神经网络参数更新和 checkpoint 更新不是一回事：

- 神经网络参数更新：每个 PPO iteration 内部都会用新采样数据做梯度下降，参数在显存/内存里持续变化。
- checkpoint 更新：隔 100 个 iteration 把当时的网络参数、优化器状态等写成 `.pt` 文件，方便恢复训练或播放策略。

所以训练不是每 100 轮才学一次；它一直在学，只是每 100 轮保存一次。

## 当前速度

2026-08-23 这次 6000 iteration 粗糙地形训练大约跑了 13 小时 43 分钟。主要耗时来自 4096 个环境的物理仿真采样，粗糙地形会让接触、碰撞和重置更重；学习网络本身的耗时反而很小。

## 需要记录

每次训练至少记录：

- 任务名、环境数、iteration 数、是否使用粗糙地形
- reward 曲线、episode length、termination 原因
- `Perf/total_fps`、`Perf/collection time`、`Perf/learning_time`
- checkpoint 路径和你认为最好的一版模型
- 训练中是否出现 `nan`、`inf`、机器人飞出、长期轮速过大
