# 训练命令

完整训练：

```bash
cd /home/shun/桌面/wheel_leg_rl
source wheel_leg/WheelLeg_RL_IsaacLab/IsaacLab/.venv/bin/activate
TERM=xterm ./wheel_leg/WheelLeg_RL_IsaacLab/IsaacLab/isaaclab.sh -p serial_leg_rl/scripts/train_standing_rsl_rl.py --task SerialLeg-Standing-Direct-v0 --num_envs 4096 --headless
```

TensorBoard：

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

这条命令用平地、1 个环境、零 action 看默认物理状态。先观察腿部闭链点有没有分开、被动杆有没有乱摆、机器人是否一落地就倒。窗口里看形状，同时终端里看 `closure_max` 和 `closure_pairs`。

如果想看完整命令，脚本内容在：

```text
serial_leg_rl/scripts/view_standing_gui.sh
```

如果 GUI 窗口在 viewport / DLSS 初始化阶段段错误，先跑 headless 诊断：

```bash
cd /home/shun/桌面/wheel_leg_rl
source wheel_leg/WheelLeg_RL_IsaacLab/IsaacLab/.venv/bin/activate
TERM=xterm ./wheel_leg/WheelLeg_RL_IsaacLab/IsaacLab/isaaclab.sh -p serial_leg_rl/scripts/view_standing_env.py --task SerialLeg-Standing-Direct-v0 --num_envs 1 --num_steps 300 --print_interval 20 --headless --device cuda:0
```

如果 headless 能创建环境，但 GUI 仍崩，问题优先看 Isaac Sim 渲染/viewport 配置，不是 PPO，也不是闭链本体。
