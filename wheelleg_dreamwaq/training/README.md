# 训练代码阅读入口

当前源码集中在本目录。`wheelleg_dreamwaq/` 提供行走任务，`recovery/` 提供倒地起身和两种策略交接，`scripts/` 提供行走训练及工具入口。

## 先选择要理解的过程

| 过程 | 从这里进入 |
| --- | --- |
| 行走 DreamWaQ 训练 | [scripts/train_dreamwaq.py](scripts/train_dreamwaq.py) |
| 行走普通 PPO 训练 | [scripts/train_ppo.py](scripts/train_ppo.py) |
| 倒地起身 PPO 训练 | [recovery/train.py](recovery/train.py) |
| 起身后交给行走策略 | [recovery/evaluate.py](recovery/evaluate.py)，接着查看 [supervisor.py](recovery/supervisor.py) |

## 行走代码的阅读顺序

1. [train_dreamwaq.py](scripts/train_dreamwaq.py)：参数、环境创建、训练器、保存与恢复。首先找到环境和训练器的创建位置。
2. [env_cfg.py](wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env_cfg.py) 和 [training_profiles.py](wheelleg_dreamwaq/tasks/direct/wheelleg_flat/training_profiles.py)：本次实验使用哪些任务参数和配置。
3. [env.py](wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py)：从 `_pre_physics_step`、`_apply_action`、`_get_observations`、`_get_rewards`、`_get_dones`、`_reset_idx` 理解一个控制步和一次重置。
4. [observations.py](wheelleg_dreamwaq/tasks/direct/wheelleg_flat/observations.py)、[control.py](wheelleg_dreamwaq/tasks/direct/wheelleg_flat/control.py)、[rewards.py](wheelleg_dreamwaq/tasks/direct/wheelleg_flat/rewards.py)：输入、执行目标及学习信号。
5. [history_wrapper.py](wheelleg_dreamwaq/algorithms/dreamwaq/history_wrapper.py) 和 [actor_critic.py](wheelleg_dreamwaq/algorithms/dreamwaq/actor_critic.py)：观测历史、速度与上下文估计、策略和价值网络；其余算法文件在同目录。

遇到字段含义不清楚时，查看 [schemas](wheelleg_dreamwaq/schemas) 中对应定义。机器人资产入口位于 [assets/wheelleg.py](wheelleg_dreamwaq/assets/wheelleg.py)。

## 起身代码的阅读顺序

1. [train.py](recovery/train.py)：读取解析配置、检查运行身份、建立训练器。
2. [contracts.py](recovery/contracts.py) 和 [env_cfg.py](recovery/env_cfg.py)：观测和动作约定、时间步长及环境配置。
3. [env.py](recovery/env.py)：初态恢复、动作执行、观测历史、奖励、终止和监督器调用。
4. [observations.py](recovery/observations.py)、[control.py](recovery/control.py)、[rewards.py](recovery/rewards.py)：输入如何组成，动作如何转换，奖励如何计算。
5. [ppo_runner.py](recovery/ppo_runner.py)：起身策略采样和 PPO 更新。
6. [batched_supervisor.py](recovery/batched_supervisor.py) 与 [supervisor.py](recovery/supervisor.py)：物理就绪、共同确认、截止和策略交接；[evaluate.py](recovery/evaluate.py) 组织实际组合评估。

起身验证状态和已有模型证据见 [起身说明](recovery/README.md)。详细设计见 [起身与行走双模式设计](../docs/2026-10-22-wheelleg-fall-recovery-design.md)。

## 运行位置

命令在 `wheelleg_dreamwaq/` 工程根目录执行：

```powershell
& '.venv/Scripts/python.exe' training/scripts/train_dreamwaq.py --help
& '.venv/Scripts/python.exe' -m recovery.train --config recovery/artifacts/unclipped-actions-20261024/training_config.json --help
& '.venv/Scripts/python.exe' -m pytest tests/unit -q
```

首次配置环境使用 [bootstrap.ps1](scripts/bootstrap.ps1)。从新路径安装当前包：

```powershell
uv pip install --python .venv/Scripts/python.exe --no-deps --editable .
uv pip install --python sim2sim/mujoco/.venv/Scripts/python.exe --no-deps --editable .
```

也可在仅需要临时指定源码的进程中设置 `$env:PYTHONPATH="$PWD/training;$PWD/sim2sim/mujoco;$env:PYTHONPATH"`。Python 导入名称仍为 `wheelleg_dreamwaq`、`recovery`。模型、日志和旧实验报告继续使用原数据目录；源码摘要读取本目录，旧 checkpoint 的严格身份检查继续生效。

起身入口在添加 Isaac 启动参数时会先检查必需的 `--config`，因此上面的帮助命令也指定了一份已有配置；`--help` 会在运行前退出。
