# wheel_leg_motor_PACE

这是面向偏置五连杆轮腿实机的独立 STM32H723 数据采集工程。它只负责四台 DM8009 和两台 LK9025 的命令、反馈、安全监控与原始数据输出，不包含 VMC、LQR、底盘控制、IMU 或强化学习代码。

## 当前状态

- 固件、Keil 工程文件、二进制协议和 PC 工具已经实现。
- Python 协议、带宽、调度、解码、拟合和双仿真适配测试已通过。
- 当前电脑未发现 Keil MDK、ARMClang 或 GCC ARM 工具链，因此尚未完成真实 ARM 编译。
- 尚未下载到 STM32H723，也未完成六电机上电、CAN/UART 时序和急停验收。
- `FINAL_IDENTIFICATION_SESSION` 由编译期合同锁禁止启动；当前只允许 `COMMISSIONING_SESSION`。

## 固定硬件接口

| 项目 | 配置 |
|---|---|
| MCU | STM32H723VGTx |
| CPU/AHB | 400 MHz / 100 MHz，VOS1；24 MHz HSE |
| FDCAN | FDCAN3，PD12/PD13，Classic CAN 1 Mbit/s |
| CAN 可靠性 | 自动重传开启，8 个 Tx Event FIFO 元素，8 个 Tx FIFO 元素 |
| PC 串口 | UART7，PE7/PE8，921600 baud，8N1 |
| UART DMA | RX: DMA1 Stream3 circular；TX: DMA1 Stream4 normal |
| 采样时钟 | TIM6，500 Hz |
| 单调时间 | TIM5，1 MHz，32 bit |

## 六通道顺序

所有固件、raw 文件、PC、Isaac 和 MuJoCo 代码只使用以下顺序：

```text
0 L_front
1 L_rear
2 R_rear
3 R_front
4 L_wheel
5 R_wheel
```

CAN ID、关节名、方向和零位的唯一来源是 [pace_motor_manifest.h](Config/pace_motor_manifest.h)。首次上电前，方向、零位和 `calibration_verified` 仍需实机确认。

## 工程结构

```text
MDK-ARM/   Keil 项目、启动文件和链接配置
Core/      STM32 初始化、IRQ 和最小 FreeRTOS 任务
Board/     FDCAN、UART DMA 和时间端口
Motor/     DM8009/LK9025 编码、解码和六通道注册表
PACE/      实验状态机、激励、安全、采集和传输
Protocol/  raw frame、host command 和 CRC-32
Config/    motor manifest、实验参数和最终合同锁
docs/      格式、实施和硬件验收文档
tools/     Keil 项目生成与静态检查
```

## Keil 构建

1. 在安装了 Keil MDK 和 STM32H7 Device Pack 的电脑打开 `MDK-ARM/wheel_leg_motor_PACE.uvprojx`。
2. 确认目标为 `wheel_leg_motor_PACE`，器件为 STM32H723VGTx。
3. 全量 rebuild，先处理所有编译/链接警告，再下载。
4. 不要直接用 CubeMX 覆盖工程；当前 `.ioc` 仍保留部分历史元数据，生成前必须人工核对时钟、DMA、FDCAN message RAM 和 FreeRTOS 文件清单。
5. 下载后先断开机械负载，按 [commissioning_checklist.md](docs/commissioning_checklist.md) 执行。

本机可执行的项目静态检查：

```powershell
python tools/test_keil_project.py
python Protocol/tests/test_motor_manifest.py
python Protocol/tests/test_motor_protocol.py
python Protocol/tests/test_pace_frame.py
python Protocol/tests/test_bandwidth_budget.py
python PACE/tests/test_can_schedule.py
python PACE/tests/test_capture_transport.py
```

## 默认工程试采

DM 主辨识基线固定为：

```text
q_des  = symmetric chirp target
dq_des = 0
Kp     = 20.0
Kd     = 0.6
tau_ff = 0
```

默认阶段依次为静态、DM 拟合、DM 留出验证、LK 转矩、LK 速度和停机。六通道快照始终为 500 Hz，但 CAN 命令频率按阶段分配，以避免六电机全 500 Hz 时约 78% 的保守总线负载。

## 数据协议

- sample: 固定 126 B，500 Hz；
- status: 64..128 B，10 Hz；
- 持续最坏流量: `126*500 + 128*10 = 64280 B/s`；
- UART7 8N1 理论容量: `921600/10 = 92160 B/s`；
- 最坏持续占用: `69.75%`；
- 每个拟合命令必须来自 FDCAN Tx Event 确认后的量化 payload。

完整字段见 [session_format_v1.md](docs/session_format_v1.md)。

## 安全边界

当前未启用最终 RL 工作区间的硬位置限位，但仍保留独立的速度、转矩/电流、温度、反馈超时、CAN、UART overflow 和总时长中止。任何中止都会使 session 失效并重复发送 DM disable/LK stop 安全命令。

正式采集前必须完成 [final_identification_contract.md](docs/final_identification_contract.md)；只修改 `PACE_FINAL_CONTRACT_READY` 而没有冻结合同和配置哈希，不构成有效发布。
