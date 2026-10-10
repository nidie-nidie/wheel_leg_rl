# 键盘命令与 MuJoCo 内置显示快捷键冲突

日期：2026-10-10。用户报告 W 切换显示、D 切换碰撞等功能，无法按预期只控制机器人。

根因是 `mujoco.viewer.launch_passive(key_callback=...)` 的回调与原生 Simulate 键盘处理并存，用户回调不能消费并阻止原生快捷键。旧会话的逐帧记录也有速度、转向目标变化，说明按键同时进入了两个处理路径，而非速度命令没有更新。

修正只在调试显示入口：新增 `keyboard_viewer.py`，直接使用 GLFW 和 MuJoCo 渲染 API，安装单一键盘处理器；`play_keyboard.py` 改用这个窗口，保留独立显示 model/data、run-04 策略、原 runtime 和已有 base-contact 调试模型。没有启用 Simulate 的显示快捷键。GLFW Escape 键码为 256，退出判断同步使用 `KEY_ESCAPE`。

验证分两层：`verify_keyboard_viewer.py` 创建实际 OpenGL 窗口，调用同一注册处理器，确认控制按键只转发一次，释放不触发命令，渲染选项与物理状态/外力矩不变；`play_keyboard.py --base-contact --smoke` 验证实际命令加减、范围裁剪、history 衔接、reset、暂停/继续、退出请求和 20 次动作执行。此回归检查不模拟操作系统键盘输入，实际桌面窗口另通过进程、就绪日志和实时状态确认启动。

使用时查看顶部 `Command vx` 和 `Actual vx`：W/S 分别将速度目标增加/减少 0.1 m/s，A/D 分别将转向目标增加/减少 0.1 rad/s。当前策略小速度和零速跟踪仍有问题，命令增加不保证实际速度立即转正；按键冲突修正不能视为策略跟踪问题已解决。

历史 `viewer-fix-report.md` 和 `viewer-fix-verification.json` 保留了此前外力矩缓存、base 接触修正的验证状态。此次代码身份和启动证据单独保存在 `keyboard-fix-verification.json`，不替换历史文件哈希。正式架构、训练代码、奖励、USD、原 MuJoCo XML/参数、checkpoint 和导出策略均未修改。
