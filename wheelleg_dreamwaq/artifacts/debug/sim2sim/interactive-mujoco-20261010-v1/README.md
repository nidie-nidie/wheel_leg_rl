# run-04 MuJoCo 键盘可视化

用户要求打开当前 MuJoCo 策略并用键盘控制命令，随后要求启用机身碰撞。此入口独立放在调试目录，复用正式 `WheelLegMujocoRuntime`、当前角速度补偿和 run-04 导出策略。原模型和策略身份通过原 contract loader 校验；`--base-contact` 生成私有模型，只增加已有 `base_proxy` 与 floor 的显式接触，不改训练/正式评估路径。

2026-10-10 修正了首次可视化入口的错误：实测 passive viewer 的 `sync(state_only=True)` 会修改 `data.xfrc_applied`，清除 runtime 的角速度限制外力矩。旧 live-data viewer 在零速命令下 1.20 秒 tilt 超过 0.8 rad，而无窗口评估稳定。显示改为独立 model/data 副本后，500 次动作的 qpos、qvel、xfrc_applied 与无窗口参考完全一致。见 `viewer-sync-probe.json`。机身碰撞缺失和这个显示入口问题是两个独立因素。

窗口现在只读物理状态副本，鼠标对显示副本的物理扰动不传回 runtime。显示碰撞盒只改变独立显示模型的 geom group，不改物理形状。橙色半透明盒子是模型原有的近似 base box，不代表精细 STL 机身的完整碰撞外形。

原 passive viewer 的用户回调不能拦截其内置按键，导致 W/D 同时触发显示快捷键和我们的命令。现改用 `keyboard_viewer.py` 的专用 GLFW 渲染窗口，只安装我们的按键处理器。窗口标题为 `WheelLeg MuJoCo | Run-04 keyboard control`，W/S、A/D 不再改变渲染状态；左键拖动旋转镜头，滚轮缩放。见 `keyboard-fix-report.md` 和 `keyboard-viewer-verification.json`。

键盘事件进入队列，在下一次动作边界消费。用户再次要求提高速度上限后，调试窗口默认 vx [-10.0, 10.0] m/s，可用 `--max-speed` 指定；yaw [-1, 1] rad/s、高度 [0.16, 0.24] m。更新当前观测中的命令，保留旧历史帧，随后按原 20 个 1 ms 物理步执行策略动作。R 使用原 runtime reset，并重建 history。

训练速度范围仍为 [-1.5, 1.5] m/s。正式 observation builder 会把超过范围的命令截到 ±1；本调试入口仅对速度命令索引 6 取消这道截断，保留 `vx_max_abs=1.5` 换算比例，因此 10.0 m/s 对应网络输入约 6.667。reset 五帧及每次新 history 帧同步采用此值。高于 1.5 的目标属于训练范围外测试，不保证实际速度达到目标。`--max-speed 1.5` 恢复原控制范围和网络输入语义；正式代码、manifest、训练配置不变。

点击 MuJoCo 画面使窗口获得焦点。每按一次调整一档，命令保持到下一次调整：

| 键 | 操作 |
| --- | --- |
| W / S | 前后速度命令加 / 减 0.1 m/s |
| A / D | 左转 / 右转命令加 / 减 0.1 rad/s |
| Q / E | 高度加 / 减 0.01 m |
| X | 前后速度和转向命令归零 |
| R | 恢复初始姿态并继续，保留当前命令 |
| P | 暂停 / 继续 |
| Esc | 关闭窗口 |

初始命令为 (0, 0, 0.20)，策略自动运行；镜头跟随机器人。X 只把命令归零，当前 run-04 的零速后退问题仍会显示。机身高度越界、倾角超过 0.8 rad 或闭环残差超限时，会明确显示 FALL 并暂停；R 重置后继续。`--session` 指定的目录保存逐帧 `trajectory.csv` 和实时 `status.json`，包含命令、机身高度/倾角、碰撞盒最低 z、机身触地数量和失败原因。

W/S 调整的是速度目标，顶部 `Command vx` 显示目标值，`Actual vx` 显示实际运动。W 把目标增加一档，例如 0 变成 +0.1；S 把目标减少一档，例如 0 变成 -0.1。从正速度按一次 S 只是减速，不一定立刻变成倒退。当前策略仍可能在小正速度或零速度命令下后退，不能用实际运动方向判断按键是否生效。

运行：

```powershell
& 'E:\wheel_leg_rl-main\wheelleg_dreamwaq\sim2sim\mujoco\.venv\Scripts\python.exe' -B 'E:\wheel_leg_rl-main\wheelleg_dreamwaq\artifacts\debug\sim2sim\interactive-mujoco-20261010-v1\play_keyboard.py' --base-contact
```

验证使用 `--base-contact --smoke`：检查键盘命令、调试范围裁剪、正负高速度目标在 reset/history/实际策略调用中不会被原 ±1.5 限制截断、当前 command 与 history 衔接、原 reset 和 20 次动作执行。`verify_base_contact.py` 在人为设置 2 mm 穿入的相同姿态下确认旧模型无机身接触、新模型有 4 个接触并产生正支撑力；随后确认新增接触对不改变未触地的 10 秒名义站立轨迹。通过后启动实际 viewer，并核对 `MUJOCO_VIEWER_READY`、进程窗口和实时 status。

`verify_keyboard_viewer.py` 打开实际 OpenGL 窗口，直接调用注册给 GLFW 的处理器，验证 W/S、A/D、Q/E、X/R/P 和 Escape 的事件转发及释放忽略，同时确认渲染选项、qpos、qvel、外力矩不变；此检查不模拟操作系统键盘输入。smoke 另验证暂停/继续与 Escape 退出请求。

当前 `--base-contact` 是用户要求的可视化实验配置；正式架构仍为 v0.24 / WheelOnlyCollisionV2，正式 XML、USD、checkpoint 和奖励不变。不能把此调试模型的结果直接混进正式八场景排名。实际长期/键盘后倒情况由新的逐帧记录判断，不把旧窗口中的后倒直接归因给策略。
