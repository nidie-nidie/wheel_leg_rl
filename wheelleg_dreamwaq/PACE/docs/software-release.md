# 软件交付与试采步骤

## 当前配置

| 项目 | 值 |
|---|---|
| 硬件 | STM32H723VGTx，原工程时钟及引脚配置 |
| 工况 | 电机装在整机上，机身固定，轮子悬空 |
| 方向 | 保留现有代码，来源 user_confirmed_existing_code |
| 关节顺序 | PACE: jIJ,jIO,jAG,jAB,jwheel_left,jwheel_right |
| DreamWaQ 对应 | 按名字重排为 jIJ,jIO,jAB,jAG,jwheel_left,jwheel_right |
| 串口 | UART7 PE7/PE8，921600，8N1，无流控 |
| CAN | FDCAN3 PD12/PD13，Classic CAN 1 Mbit/s |
| 目标更新 / CAN 调度 / 采样 | 50 / 500 / 250 Hz |
| 协议 | V2；PC 可读取 V1 用于诊断 |
| 默认运行 | 单向发送，复位等待 10 秒，再自启动一次 |
| 总采集时长 | 211 秒，共 26 段 |
| 模型成熟度 | commissioning / provisional |

## 代码与构建

新电脑需要 Python 3.11 和 MSVC C/C++ Build Tools（本机已配置）。在 PACE 目录运行 `tools/setup_pc.ps1` 建独立 Python 环境，再运行 `.venv/Scripts/python.exe tools/bootstrap_arm_toolchain.py` 下载固定版本的官方便携 Arm GNU 工具链。工具链仅放在 `PACE/build`；现有环境可直接验证。

`host_debug` 编译真实生产 C 源码；HAL fake 注入 TIM5/TIM6、独立 CAN Tx Event、Rx、UART 错误及任务轮询。原 Keil 工程保留，新 GNU 构建使用同版本官方 STM32 startup 和 FreeRTOS Cortex-M7 GCC port，来源和 hash 位于 `firmware_gcc/vendor/sources.json`。

```powershell
& .\.venv\Scripts\python.exe tools/verify_software.py --include-target
```

报告为 `build/release/software_verification.json`，详细输出为同目录 `check_*.txt`；同时运行 MSVC AddressSanitizer，覆盖 HAL 按 DLC 复制 64 字节后才拒绝长帧的内存安全回归。native 报告另有参与编译源文件、可执行程序与测试 raw 的 SHA256。新 ARM 输出位于 `build/firmware_gcc/commissioning/`，包括 ELF、HEX、MAP 与构建记录。构建验证了 Flash/RAM 容量和 DMA 所在地址，但没有证明板子已经启动。

UART RX DMA 使用 `.pace_dma`，由 Keil scatter 与 GNU linker 放在 AXI SRAM；其余内存沿用原工程容量。DMA1/2 对 DTCM 的访问限制见 [ST 官方说明](https://community.st.com/stm32-mcus-60/dma-is-not-working-on-stm32h7-devices-97)。当前程序未开启 D-cache；以后若开启，需要同时处理 DMA 缓存一致性。

## 烧录与开采

1. 将机身固定好、轮子悬空，确认这次低幅值运动不会碰撞。沿用原方向；零位与机械范围仍需观察实际动作。实物停止开关可触及。
2. 烧录新生成的 commissioning HEX。板子复位后 10 秒会自行使能和试采；烧录后的自动复位也会触发等待，因此应先准备好工况。
3. 按 PACE README 创建 `commissioning_001.plan.json`。其中记录代码、实际 HEX、DreamWaQ 控制基线和固定工况。不要覆盖已有计划；下一轮换新名字。
4. 开启 PC capture，看到保存路径后复位板子。已经错过 header 的文件会被判为不完整，此时开一个新文件并重新复位。
5. 约 3 分 31 秒采集完成后，固件先发送 DM disable / LK stop，等待匹配的 CAN Tx Event，再发送末尾事件和 footer。这里的确认表示报文已发到总线，物理停稳仍需观察。
6. 查看自动质量报告，再执行 decode；有任何故障或缺帧时保留 raw 和 sidecar 排查。PC 超时或 Ctrl+C 只会影响录制，单向固件需要实物急停。

录制从不清空输入 buffer。首数据等待至少 15 秒；中断、超时、缺头、缺尾、CRC、sequence、尾后数据、饱和/故障、源快照不一致都能被记录或拒绝拟合。录制文件已有时自动选下一个名字，以屏幕显示的实际路径为准。raw 是原始来源，CSV/NPZ 可重建，sidecar 不要手工修改。

## 质量、拟合与边界

停止成功的文件尾不代表所有样本合格。公共库入口和 CLI 共用门禁，分段求导和延迟，不跨独立试验拼接求导；FIT 与 VALIDATION 分开，温度采用最近过去的 status 并保留 age。模拟数据只用于检查软件，不用于报告实际电机参数。

TIM5 时间戳记录 HAL 回调服务时刻。CAN 入队、Tx Event 服务和反馈服务都是可观测信息，但不是电机收到/执行指令的绝对时刻。当前 250 Hz 记录不支持宣称已测得微秒级或低于一个采样周期的物理延迟。

零位、姿态中心、机械限位、最终幅值、增益与 LK 模式合同确认后，才准备并重新编译 final 配置。默认 final 请求会被拒绝，而且拒绝后不能误启动上一次的 recipe。仅修改一个 hash 无法解锁。

低增益试采与 DreamWaQ 现有基线的差异写入计划。输出先供数据诊断和初始拟合，不会自动写入正在训练的 DreamWaQ 工程；装机闭链有效惯量不能直接覆盖仿真 armature。Isaac/MuJoCo 独立适配器要求明确运行周期和参数语义，避免延迟单位或负载惯量误用。

## 本轮验收结果

2026-10-12 最终统一验收返回0：78项Python检查通过，真实生产C的主动/被动模式及两种AddressSanitizer运行均通过；真实C完整输出经碎片FakeSerial录制、sidecar、解码与质量门禁通过。GNU目标编译/链接和ELF向量、内存位置检查通过。当前HEX SHA256：`8a2a839058509004ae95345ff60fc18107dcf945d55e14182d42fba7f221ca1e`。

构建仍有预期告警：MSVC final合同关闭造成常量条件提示；GNU单向模式留下未使用的host接收函数。验收记录这些告警，未把编译成功称为零告警或实板验证通过。实际board_validation为未执行。
