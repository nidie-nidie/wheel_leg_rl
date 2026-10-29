# wheel_leg_motor_PACE

独立 STM32H723 六电机采集固件。完整构建、烧录和试采入口见 [PACE README](../README.md) 及 [软件交付说明](../docs/software-release.md)。

生产模块为 Board（CAN/UART/时间）、Motor（编码/反馈/注册表）、PACE（状态机/安全/采集/传输）、Protocol（帧/CRC/命令）和 Config（配置/映射）。真实 C 状态机由 STM32 与 `../host_debug` 共用，RTOS 任务只封装同一 poll 入口。没有引入 VMC/LQR/IMU 或 DreamWaQ 训练依赖。

当前方案 2 配置为 26 段低幅值 commissioning：50 Hz 目标保持、500 Hz CAN 调度、250 Hz 采样、10 Hz 状态。启动方式沿用单向自启动，等待 10 秒且每次复位只采一轮；DM Kp20/Kd0.6，LK 转矩和速度分开验证。线上 V2、stage128B、sample126B；完整字段见 [V2 实验合同](docs/v2-experiment-contract.md)。

电机 ID、通道和现有方向继续以 [pace_motor_manifest.h](Config/pace_motor_manifest.h) 为准；用户已确认现有方向。`calibration_verified=false` 表示零位与机械工作范围合同还未实测确认，不要求重新测方向。PACE 右腿顺序与 DreamWaQ 不同，PC 按关节名字映射。

Keil 工程位于 `MDK-ARM/wheel_leg_motor_PACE.uvprojx`；RX DMA 的 AXI 内存由 `Config/pace_memory.sct` 控制。另有可复现 GNU 构建 `../firmware_gcc/build.py`，默认输出新 commissioning HEX。历史 MDK build 产物不能当成本轮的新固件。

软件验证入口：

```powershell
# 在 PACE 目录执行
& .\.venv\Scripts\python.exe tools/verify_software.py --include-target
```

实板试采说明见 [commissioning_checklist.md](docs/commissioning_checklist.md)。默认不开放 final，不把未实测零位、范围或最终增益的试采结果标为正式模型。
