# PACE 采集软件

这是 STM32H723 固件、PC 工具和真实 C 调试平台的共同入口。当前交付采用方案 2：电机装在整机上、机身固定、轮子悬空；沿用现有 CAN ID 和用户确认正确的电机方向。

四个腿关节按左腿、右腿、双腿同相、双腿异相分组；轮子分别做转矩和速度激励。26 段中包含两个拟合幅值和独立验证段。目标以 50 Hz 零阶保持，CAN 调度 500 Hz，采样 250 Hz。一轮约 211 秒，板子复位后先等待 10 秒。

Python 已放在本目录 `.venv`，与 DreamWaQ 环境独立。新的电脑可运行 `tools/setup_pc.ps1`，需要 Python 3.11。完整操作说明见 [software-release.md](docs/software-release.md)。

在当前工作区根目录打开 PowerShell：

```powershell
Set-Location E:\wheel_leg_rl-main\wheelleg_dreamwaq\PACE
& .\.venv\Scripts\python.exe tools/verify_software.py --include-target
```

检查通过后，使用 `build/firmware_gcc/commissioning/` 内的新 HEX 烧录。不要使用 `MDK-ARM` 目录里之前遗留的 HEX。烧录器使用 HEX 自带地址，起始 Flash 为 `0x08000000`。

试采前先创建记录计划；`--firmware-binary` 指向实际烧录的新文件：

```powershell
& .\.venv\Scripts\python.exe tools/prepare_session.py --output data/commissioning_001.plan.json --firmware-binary build/firmware_gcc/commissioning/wheel_leg_motor_PACE.hex
& .\.venv\Scripts\python.exe wheel_leg_motor_PACE_PC/tools/capture_session.py --port COM7 --output data/commissioning_001.raw --metadata-plan data/commissioning_001.plan.json
```

把 COM7 换成串口号。先看到 `Saving raw capture to`，再复位板子。默认是单向发送模式：PC 只接收，不发送 START；提前停止使用实物急停或电源开关，关闭 PC 录制不会让电机停止。完成后会得到 raw、固定快照 sidecar 和质量报告；异常数据保留用于诊断。

```powershell
& .\.venv\Scripts\python.exe wheel_leg_motor_PACE_PC/tools/decode_session.py data/commissioning_001.raw --summary data/commissioning_001.summary.json --npz data/commissioning_001.npz
& .\.venv\Scripts\python.exe wheel_leg_motor_PACE_PC/tools/fit_session.py data/commissioning_001.raw data/commissioning_001.actuator.json
```

试采模型为 `provisional`。零位、整机运动范围与最终增益合同尚未实测确认，正式 final 采集保持锁定。现有方向无需重新标定；50 Hz 目标语义已对齐，但低幅值试采使用 Kp20/Kd0.6，DreamWaQ 当前腿关节为 Kp120/Kd4，不能直接认为动力学等价。这里得到的是装机负载响应，不能把拟合等效惯量直接加到闭链模型的电机 armature 上。

测试平台直接编译生产状态机、Board、Motor 和 Protocol C 源码，只替换 HAL 边界。它验证程序逻辑和数据链路；本轮没有连接实板，实板 UART/CAN 时序、机械范围及停止动作需由首次低幅值试采确认。
