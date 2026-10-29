# MuJoCo 航向保持 debug 接入

日期：2026-10-16

目标：给现有模型增加可开关的航向保持输入，在同一个 701 次更新的模型上观察停车、前进和后退的偏航变化。

## 控制边界

航向控制器根据机身朝向计算神经网络的偏航速度指令。网络六维输出、轮速目标转换和电机力矩计算继续沿用现有实现。正式训练配置、模型工件、USD/XML、奖励和随机化保持现有内容。

- 默认参数：Kp=0.5、Kd=0，输出偏航速度限制为 ±1 rad/s，与当前模型的训练命令范围一致；命令行允许设置非负有限 Kp/Kd。
- 停车/直行：保持记录的朝向，使用最短角差 `atan2(sin(target-yaw), cos(target-yaw))`；输出 `clip(Kp*error-Kd*body_yaw_rate, -1, 1)`。D 项使用当前机身 z 轴角速度作为阻尼信号，不声称它严格等于 Euler 航向角导数。
- 用户转向：偏航输入非零时直接传递用户指令，并更新参考朝向；结束转向的第一个控制步重新捕获当前朝向，随后保持。
- 初始化、R 重置和 H 开关：捕获当前朝向并清除上一转向状态。关闭时传递用户指令。
- 暂停期间不更新控制器；X 只把用户速度和转向指令归零，不强行恢复 reset 时的朝向。
- 输入和控制结果必须有限，错误立即报告。

航向角与角速度复用当前 MuJoCo 指标的控制坐标定义。每次策略推理前，只更新当前 history 最后一帧的三个命令分量，使网络本步看见实际使用的纠偏指令；不移动 history，不更改其他观测。后续物理步进照常追加下一帧。开关两种状态使用相同的命令写入时序。

本 debug 实现限于运行时，不计算训练奖励。正式训练中的 P=9、D=0 航向保持、差速门控及保存/部署契约已记录到 [baseline 修改计划](../../docs/2026-10-16-dreamwaq-baseline-modification-design.md)，该正式实现尚未实施。

## 文件与实施步骤

- [x] 新增 `heading_control.py`：航向保持状态、参数和偏航速度指令计算；只依赖 Python 标准库。
- [x] 修改 `play_mujoco_viewer.py`：命令行参数、H 开关、当前命令写入、HUD 和 CSV/status 记录。
- [x] 新增 `test_heading_control.py`：角度跨越 ±π、纠偏方向、转向释放、开关/reset、阻尼、限幅与非法值拒绝。
- [x] 新增 `verify_heading_mujoco.py`：使用真实模型与现有八场景、500 控制步，比较航向保持开启/关闭，保存原始输入、纠偏输入、实际运动及六维网络输出。
- [x] 运行针对性测试、真实 MuJoCo 对照与 100 个正式文件指纹核对。
- [x] 打开同一模型的新可视化会话，验证 READY 与会话状态。

## 验证与记录

命令从项目根目录执行，PYTHONPATH 包含项目根目录、`source/wheelleg_dreamwaq` 和 `sim2sim/mujoco`。

```powershell
.\sim2sim\mujoco\.venv\Scripts\python.exe -m pytest debug/no_action_clip/test_heading_control.py -q
.\sim2sim\mujoco\.venv\Scripts\python.exe -m debug.no_action_clip.verify_heading_mujoco --export debug/no_action_clip/runs/fresh1000-split-straight-lambda01-env4096-20261016/export --output debug/no_action_clip/runs/heading-hold-verification-20261016
```

比较分别记录各场景的存活控制步、速度误差、航向漂移、实际偏航速度、纠偏指令与实际四关节角加速度 RMS；实际结果写入运行目录。纠偏输入来自航向控制器，六维动作始终来自同一 TorchScript 模型。关闭状态与旧固定命令路径的动作、history、目标、扭矩和真实轨迹逐元素比较。

本次没有启动训练；控制器是否改善当前模型的运动表现以真实 MuJoCo 结果为依据。

## 已执行的结果

使用 `fresh1000-split-straight-lambda01-env4096-20261016/export` 中完成 701 次更新的模型，Actor SHA256 为 `FDCA84A3B9B4F993DF481F09DBAD14CC7AFBCDB861D530642EA522EE50837464`。Kp=0.5、Kd=0；未更新网络参数。

- 针对性测试：`test_heading_control.py` 和 `test_mujoco_adapter.py -k 'not identity_guards'` 共 `23 passed, 1 deselected`。排除项需要外部 export fixture，本次不运行工件损坏测试。覆盖当前命令写入后，真实 MuJoCo 的 qpos/qvel/时间和其他历史分量不变，以及随后 history 只追加一次。
- 对照报告：`runs/heading-hold-verification-20261016/summary.json`，包含 16 个场景结果和对应 CSV。开启、关闭两组各八个场景，全部完成 500 个控制步（10 秒），无失败。
- 关闭组八场景每一步的观测、六维动作、腿目标角、轮目标转速、施加力矩、history、qpos/qvel 均与独立旧固定命令路径逐元素相等。三种有转向指令的场景，开启/关闭组真实轨迹指纹完全相同。
- 正式文件指纹前后均为 100/100 相同；Actor 文件指纹前后相同。正式架构仍为 v0.29，SHA256 仍为 `EAC93CCEA82C3B9429FCBD4327BABC5DEB8F9D180557AB965CC12F5C9334D0A7`。

| 场景 | 10 秒末航向偏移：关→开 | 平均绝对速度误差：关→开 |
|---|---|---|
| 停车，目标高度 0.20 m | −22.10° → −4.35° | 0.03770 → 0.03727 m/s |
| 前进 +1 m/s | −21.72° → −4.65° | 0.02425 → 0.02430 m/s |
| 后退 −1 m/s | −33.06° → −6.34° | 0.06135 → 0.06163 m/s |

三种高度停车及前进、后退的航向漂移 RMS 均降低。四个驱动关节在控制间隔上的角加速度 RMS 变化均小于 0.1%；这里按相邻控制帧实际关节速度差除以 0.02 s 计算，不能据此排除物理子步内的振动，也不能声称已经解决腿部抖动。

结论：当前模型在这些固定场景下能响应航向纠偏指令，偏航减少，速度误差基本相同，仍有残余航向误差。尚未用这组固定命令验收任意键盘切换、外部推扰或长时间运行。用户可在可视化中按 H 继续比较。

## 参考与使用

比例航向控制参考 [legged_gym 的 legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py) 和 [Isaac Lab v2.3.2 velocity_command.py](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py)。这两处参考通过航向误差产生偏航速度命令；本实验默认 Kd=0。

可视化需显式传入本次 export 和 `--heading-hold`。H 切换航向保持；W/S 调整前后速度，A/D 调整转向速度，X 把这两项归零；R 重置；Q/E 调整目标机身高度（0.16–0.24 m）。当前网络的前四维动作控制四个腿部驱动关节目标角，能够通过腿部构型变化伸缩腿。输入为目标机身高度，尚无左右腿长度的独立目标输入。

可视化固定启用 MuJoCo 原生世界坐标轴；X/Y/Z 轴分别显示为红色、绿色、蓝色。status 的 `coordinate_frame` 记录原生世界坐标枚举值 7。

本次会话为 `runs/fresh1000-split-straight-lambda01-env4096-20261016/viewer-heading-701-20261016-190723/`。stdout 已输出 `MUJOCO_VIEWER_READY`，stderr 为空，实际进程命令绑定本次 export；status 确认航向保持开启、Kp=0.5、Kd=0，仿真正在推进。11.44 秒时记录到用户前后速度指令 −0.8 m/s、用户偏航指令 0、策略纠偏指令约 +0.0324 rad/s，失败字段为空。
