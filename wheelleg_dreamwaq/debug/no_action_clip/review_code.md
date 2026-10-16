# 实验代码复核

日期：2026-10-15。复核范围为本目录全部 Python 实现、`experiment.md` 和 `review_document.md`，并只读检查其调用的正式工程、RSL-RL 3.1.2 与 MuJoCo 实现。复核者没有修改正式文件，没有启动训练，没有调用其他 agent。

没有发现阻塞本次从头训练 1000 iteration 的代码问题。最终独立 Isaac Lab 验收已经成功完成，见 `runs/validation-reviewed/training_summary.json`；正式训练使用单独的新进程。本记录中的数值测试不能证明训练后的运动表现。

## 动作及训练数据

- `train.py:117`、`train.py:132` 同时将 runner 配置和 history wrapper 的 `clip_actions` 设置为 `None`。`environment.py:11` 替换正式环境的动作准备过程，`experiment.py:43` 直接使用六维原始动作生成目标并缓存原始动作，未再次执行 ±1 动作裁剪。
- `experiment.py:36` 保留上一动作原始值，`train.py:111` 把此 normalization 实例交给正式观测构造。Actor、Critic 和五帧 history 均能保留超过 ±1 的动作；`train.py:55` 的真实环境检查覆盖初始动作清零、部分 reset 和其他环境 history 的连续更新。
- `experiment.py:66` 将原始动作交给真实 CENet decoder。正式 `algorithms/dreamwaq/cenet.py:117` 对动作条件和估计速度执行 detach。`test_experiment.py:82` 的真实 decoder 输入 hook 与梯度检查确认动作条件保持原值，重建损失没有向该动作张量传播梯度。
- `experiment.py:128` 复用正式 DreamWaQPPO 和 storage。正式 `algorithms/dreamwaq/ppo.py:162` 保存原始采样动作；`test_experiment.py:100` 检查 storage 与采样动作逐元素相同，并完成真实 PPO 更新。分布均值 ±20、标准差范围及 PPO 的数值保护继续保留。
- 动作历史同时用于原有 action_rate 和 action_smooth，取消裁剪后这些代价的数值也会变化。`experiment.md:17` 已准确说明该实验契约，训练效果需要按完整数据含义解释。

## 物理行为和隔离

- `experiment.py:48` 保留腿目标角上下限和动作比例；轮子沿用正式控制坐标到 USD 的符号转换。`environment.py` 继承正式 `_apply_action`、actuator、奖励、随机化、reset 和环境生命周期，未增加滤波、动作过渡或限制角速度的额外力矩。
- `train.py:126` 将完整奖励配置与旧 run-03 manifest 逐项比较，使用 stop_reverse_v1，种子 41882943，256 个环境，24 步 rollout，8 个 mini-batch、5 个学习 epoch。正式实验分支设置 1000 iteration；验证分支单独设置 8 个环境和一次更新。没有加载旧模型的路径，`experiment.py:167` 明确拒绝 load/resume。
- 独立读取 `formal_files_before.json` 的全部 100 个文件并计算 SHA256，全部与实验开始时一致。当前 Git 状态与 `workspace_before.txt` 的比较没有差异；已有正式工程修改属于实验开始前状态。本次目录在 Git 中作为独立未跟踪目录出现，未改写正式 latest_run。
- `experiment.py:145` 保存模型、优化器、随机状态、环境独立随机状态、CENet monitor 和迭代身份；`experiment.py:85` 核验 checkpoint 与训练 manifest、种子及 completed_iterations 的绑定。model_0 表示已完成一次更新；最终应为 model_999、completed_iterations=1000。本实验没有承诺支持 resume。
- `experiment.py:77` 核验保存的环境配置、算法配置和 reset cache；`experiment.py:132` 在每轮日志与 checkpoint 保存前检查这些文件、正式文件清单、全部受保护正式文件、manifest 和实验源码指纹。`train.py:203` 对执行异常写出 failed 状态并以非零退出码终止。

## 导出及 MuJoCo

- `export.py:24` 使用实验 checkpoint 和原有推理 actor 导出，固定输入的上一动作覆盖 −4 到 +4，并检查 batch 1、3、32 的 eager、TorchScript 和重新加载输出。
- `mujoco_adapter.py:15` 直接转换原始动作，保留目标角限制并继承正式 PD 和扭矩限制。`mujoco_adapter.py:33` 把原始上一动作写入当前观测后再更新五帧 history，reset 时仍为零。表示原始动作超出原区间的统计使用 `abs(action)>1`，与训练端一致。
- `mujoco_adapter.py:47` 核验 actor、encoder、golden vector、源 checkpoint 和训练 manifest 的 SHA256、实验身份、动作契约、参考物理契约及轮次；训练时正式文件清单副本与 manifest 绑定，当前全部正式 MuJoCo adapter、model manifest 和实际模型 XML 必须符合训练时指纹。导出无法静默切换到后续修改的物理模型。
- 正式 loader 拒绝实验 checkpoint 的验证已记录在 `runs/validation-reviewed/training_summary.json`。后续播放和评估必须使用本目录的实验 adapter。

## 验收证据

- 复核者独立运行项目 Python 的 `pytest debug/no_action_clip/test_experiment.py -q`：10 个测试通过。
- 复核者独立运行 MuJoCo Python 的 `test_mujoco_adapter.py`：真实模型中的原始动作、history、reset 和扭矩限制检查通过；相同目标下的力矩与正式控制器逐元素相同。
- 复核者独立加载 `runs/validation-final/model_0.pt`，已保存配置和缓存的身份校验通过；独立执行实验 `load_runtime` 并 reset，输出形状为 `(125,)`。
- 复核者已读取最终独立进程的 `runs/validation-reviewed/training_summary.json`：真实 Isaac Lab 的初始观测、原始动作进入 Actor/Critic、部分 reset、一次 PPO 更新及正式 loader 拒绝检查均通过。部分 reset 后 7 个环境连续运行；模型参数指纹发生变化；completed_iterations=1；100 个正式文件指纹保持一致。TorchScript 最大绝对误差分别为 batch 1 的 5.96e-8、batch 3 的 1.12e-7、batch 32 的 0。
- `runs/artifact-integrity/verification.json` 记录真实 checkpoint 副本的 env.yaml、agent.yaml 和 reset cache 逐项损坏均被拒绝，恢复后可加载。`runs/mujoco-loader-integrity/verification.json` 记录源 checkpoint、训练 manifest 和 actor 的错误身份字段均被拒绝。相关检查只修改 debug/runs 下的副本。

旧 run-03 是历史参考，当前正式源码和物理资产不因奖励字典、种子相同就自动等同于其历史训练状态。取消动作裁剪后是否改善速度跟踪、phi0 对称性和实际腿部振荡，仍须用这次训练所得模型进行 MuJoCo 评估。
