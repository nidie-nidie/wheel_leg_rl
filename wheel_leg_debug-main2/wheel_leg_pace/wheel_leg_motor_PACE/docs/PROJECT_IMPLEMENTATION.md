# wheel_leg_motor_PACE 项目实施文档

- 文档状态：Milestone A 软件实现完成，等待 Keil 编译、下载和实机 commissioning 验收
- 文档日期：2026-09-28
- 目标平台：STM32H723VGTx
- 目标工程：`wheel_leg_motor_PACE`
- 目标采集对象：4 个腿部 DM 电机 + 2 个轮毂 LK 电机

## 1. 项目目标

建立一个独立的 Keil 工程，只负责偏置五连杆串腿的六通道电机数据采集。该工程不承担底盘控制、VMC、LQR、姿态控制或仿真功能。

每个实机会话输出一份带版本信息的原始数据文件。后续在 PC 端完成：

1. 原始数据解码和清洗；
2. 腿部 DM 执行器参数拟合；
3. 轮毂 LK 执行器参数拟合；
4. Isaac/Isaac Lab 回放；
5. MuJoCo 回放；
6. 交叉验证和误差报告。

实机端不运行 PACE 优化，也不需要包含 Isaac 或 MuJoCo 代码。

硬件采集分为两轮：

1. `COMMISSIONING_SESSION`：低幅值、短时间的工程试采，用于验证 Keil 工程、六通道数据链路、安全机制和初步可辨识性；
2. `FINAL_IDENTIFICATION_SESSION`：在 RL 最低层执行器接口冻结后执行的正式采集，用于产生最终 Isaac/MuJoCo 共用执行器模型。

当前实施目标是先达到 `COMMISSIONING_SESSION` 可执行、可解码、可初步拟合。工程试采数据默认标记为诊断或临时拟合数据，不直接声明为最终模型数据。

## 2. 已确定的边界

### 2.1 实机端负责的内容

- 初始化 STM32、FDCAN3、UART7 和定时基准；
- 管理六个电机的 CAN 命令和反馈；
- 按可重复的实验配置生成激励；
- 以统一时间基准记录命令和反馈；
- 维护采样环形缓冲区；
- 通过二进制协议向 PC 发送数据；
- 处理电机掉线、超速、超温、超限、通信超时和急停。

### 2.2 实机端明确不负责的内容

- 偏置五连杆正逆运动学；
- VMC、LQR、跳跃状态机和底盘控制；
- IMU 姿态融合；
- PACE CMA-ES 或其他参数优化；
- CSV 转换、绘图和统计分析；
- Isaac/Isaac Lab actuator 实现；
- MuJoCo actuator 实现和闭链回放。

### 2.3 数据定义

“六通道原始数据”不是只有六个电机的反馈位置。500 Hz sample 必须同时保留：

- 六个电机最近一次 Tx-confirmed 量化命令和命令年龄；
- 六个电机的原始反馈和反馈年龄；
- 时间戳和序号；
- 当前实验阶段；
- CAN 通信状态；
- 有效性和饱和标记。

温度、电机状态码和累计通信统计通过 10 Hz status frame 保留，不在每个 500 Hz sample 中重复。

## 3. 总体数据链路

```text
PC 发送实验配置
        |
        v
UART7 命令解析
        |
        v
PACE 实验状态机 -----> 安全检查
        |
        v
500 Hz 六通道快照和分阶段 CAN 命令调度
        |
        v
FDCAN3 电机反馈回调
        |
        v
六电机状态表 + 时间戳
        |
        v
采样快照 -> 环形缓冲区 -> UART7 DMA 二进制传输
        |
        v
PC raw_session.bin
        |
        v
解码/归一化 -> DM/LK 参数拟合 -> Isaac/MuJoCo 回放
```

PC 只发送实验配置和开始/停止命令，不在实验过程中逐采样发送实时命令。这样可以避免 PC 调度抖动影响激励时间，同时保证实机可以记录实际发送的命令。

## 4. 硬件和实时约束

| 项目 | 冻结方案 |
|---|---|
| MCU | STM32H723VGTx |
| 电机总线 | FDCAN3 |
| PC 数据链路 | UART7，初版 921600 baud |
| 采样目标 | 500 Hz，目标间隔 2 ms |
| 时间戳 | MCU 单调时钟，微秒单位 |
| 传输格式 | 二进制帧 + 序号 + CRC |
| 调度方式 | 硬件定时基准触发，FreeRTOS 任务执行非阻塞处理 |
| 实时缓存 | RAM 环形缓冲区 |
| 文本 CSV | 只在 PC 端生成，不在实机实时生成 |

### 4.1 UART7 带宽预算

UART7 的现有配置见 `wheel_leg_debug-main/rm_test-dev/Core/Src/usart.c`，配置为 921600 baud、8N1、无硬件流控。每个数据字节在线路上占 10 bit，因此理论上限为：

```text
921600 / 10 = 92160 byte/s
92160 / 500 = 184.32 byte/sample
```

因此，直接把第 7 节中的命令和反馈全部按 `float32` 发送在 500 Hz 下必然无法工作，不再将其描述为“带宽较紧张”。协议 v1 固定采用量化整数和阶段配置帧：

- 500 Hz sample frame 线上总长度固定为 126 B，包含帧头和 CRC；
- sample stream 占用 `126 * 500 = 63000 byte/s`，即 UART7 理论容量的 68.4%；
- 10 Hz status frame 的基础长度为 64 B，允许扩展到最多 128 B；
- 按最大 status frame 计算，持续流量为 `126 * 500 + 128 * 10 = 64280 byte/s`，占 UART7 理论容量的 69.75%；
- 持续传输总占用不得超过 UART7 理论容量的 70%；
- session header、stage config、event 和 footer 为低频帧，不得以连续突发方式阻塞 sample stream；
- 不以提高串口波特率作为第一版成立条件，后续只有在确认 USB 转串口芯片、驱动和线材后才能单独启用更高波特率。

实时采集不能沿用低优先级 ASCII telemetry 任务，也不能在 DMA 忙时静默丢样本。环形缓冲区不能接收完整帧时，必须锁存 overflow 标志、中止当前 session，并发送安全停止命令。

### 4.2 CAN 总线预算和分阶段调度

电机总线使用 FDCAN3、Classic CAN、1 Mbit/s。新工程必须保留现有 FDCAN3 的 `AutoRetransmission = ENABLE`，不能照搬 FDCAN1 的关闭配置。

若六个电机都在 500 Hz 下执行“一条命令对应一条反馈”，则共有 6000 frame/s。按每个 8 B 标准帧约 130 bit 的工程估算，总线占用约为 78%，还未计入极端位填充、错误帧和重发，不能作为默认调度方式。

`sample_rate_hz` 与每台电机的 `command_rate_hz` 必须分开定义。六通道状态快照始终为 500 Hz，但只有当前辨识对象使用 500 Hz 命令，其余电机使用保持命令或低频状态轮询：

| 阶段 | DM 8009 命令频率 | LK 9025 命令/轮询频率 |
|---|---:|---:|
| `STAGE_STATIC` | 每台不高于 100 Hz | 每台不高于 100 Hz |
| `STAGE_DM_FIT` / `STAGE_DM_VALIDATION` | 四台各 500 Hz | 每台不高于 100 Hz |
| `STAGE_LK_TORQUE` / `STAGE_LK_VELOCITY` | 每台不高于 100 Hz | 两台各 500 Hz |
| `STAGE_STOP` | 安全停止命令后降至不高于 100 Hz | 安全停止命令后降至不高于 100 Hz |

配置器必须按实际 DLC、发送频率以及“一命令一反馈”的保守假设计算每个阶段的总线占用。设计目标不高于 65%，计算结果超过 70% 的实验配置必须拒绝进入 `ARMED`。Phase 3 还必须记录每台电机的发送帧数、反馈帧数、入队失败、Tx Event 丢失、CAN 错误计数、发送周期和反馈年龄。

### 4.3 时间戳和“实际发送值”的定义

每条电机命令必须区分以下时刻和数据：

1. `requested_command`：激励器生成的浮点目标；
2. `encoded_command`：完成限幅和协议量化后写入 CAN payload 的整数值；
3. `enqueue_timestamp`：成功进入 FDCAN Tx FIFO 的 MCU 时间；
4. `tx_timestamp`：Tx Event 报告的帧开始发送时间；
5. `rx_timestamp`：FDCAN Rx header 报告的反馈帧开始接收时间。

用于拟合的“实际发送值”是与成功 Tx Event 对应的 `encoded_command`，不是未经量化的 `requested_command`，也不是仅调用发送函数时保存的值。FDCAN 的 16 bit Tx/Rx 时间戳必须映射到 MCU 单调微秒时间并处理回绕。

## 5. 六通道统一顺序

新工程固定使用 MuJoCo 当前执行器顺序，不在不同模块中重新排序。

| 索引 | 统一名称 | MuJoCo 关节/执行器 | 实机数组 | 当前 CAN ID |
|---:|---|---|---|---:|
| 0 | `L_front` | `jIJ` / `Left_front_joint_act` | `DM[0]` | 1 |
| 1 | `L_rear` | `jIO` / `Left_rear_joint_act` | `DM[1]` | 2 |
| 2 | `R_rear` | `jAG` / `Right_rear_joint_act` | `DM[2]` | 3 |
| 3 | `R_front` | `jAB` / `Right_front_joint_act` | `DM[3]` | 6 |
| 4 | `L_wheel` | `jwheel_left` / `Left_Wheel_act` | `LK[0]` | 4 |
| 5 | `R_wheel` | `jwheel_right` / `Right_Wheel_act` | `LK[1]` | 5 |

右腿在当前 MuJoCo 数组中是 `rear` 在 `front` 前面。这不是问题，关键是所有端都读取同一个 manifest。首次上电验证时仍需确认 CAN ID、方向符号和零位。

## 6. Motor manifest

工程必须有一份唯一的六电机映射定义。初版可以在 Keil 中使用 `Config/pace_motor_manifest.h`，后续 PC 端应从同一份定义生成或校验 Python/JSON manifest。

每个电机至少包含：

```text
canonical_index
motor_name
motor_type                 // DM 或 LK
can_id
hardware_slot
mujoco_joint
isaac_joint
position_sign
velocity_sign
torque_sign
position_zero_rad
gear_ratio
actuator_model             // leg_dm 或 wheel_lk
```

任何坐标变换都必须显式写入 manifest，不能在采集、转换和仿真代码中分别隐式修正。

## 7. 原始数据协议

### 7.1 Session header

所有 UART 帧使用 little-endian 显式序列化，不允许直接 `memcpy` C 结构体，以免编译器 padding 改变线上布局。所有 frame type 共用以下 10 B 帧头：

| 字段 | 类型 | 字节数 | 说明 |
|---|---|---:|---|
| `magic` | `uint8[2]` | 2 | 固定字节 `0x50 0x41`，即 ASCII `PA` |
| `protocol_version` | `uint8` | 1 | v1 固定为 1 |
| `frame_type` | `uint8` | 1 | header=`0x01`、sample=`0x02`、stage config=`0x03`、status=`0x04`、event=`0x05`、footer=`0x06` |
| `frame_length` | `uint16` | 2 | 包含公共前缀和 CRC 的线上总字节数 |
| `sequence` | `uint32` | 4 | 从 0 开始，每发送一条 UART frame 加 1 |

每个数据文件开头必须写入 session header；除公共帧头外，其 payload 至少包含：

```text
robot_variant
motor_order_version
firmware_build_id
session_id
sample_rate_hz
start_timestamp_us
experiment_config_hash
wire_endianness
sample_frame_size_bytes
status_rate_hz
uart_baud
can_nominal_bitrate
scale_manifest_hash
```

### 7.2 Sample frame

协议 v1 的每个 500 Hz sample frame 固定为 **126 B**。线上布局如下：

| Offset | 字段 | 类型 | 字节数 | 说明 |
|---:|---|---|---:|---|
| 0 | `magic` | `uint8[2]` | 2 | `0x50 0x41` |
| 2 | `protocol_version` | `uint8` | 1 | 固定为 1 |
| 3 | `frame_type` | `uint8` | 1 | sample 固定为 `0x02` |
| 4 | `frame_length` | `uint16` | 2 | 固定为 126 |
| 6 | `sequence` | `uint32` | 4 | 公共帧头序号，所有 frame type 共用 |
| 10 | `sample_time_us` | `uint32` | 4 | 相对 session 起点的微秒时间 |
| 14 | `stage_id` | `uint8` | 1 | 当前实验阶段 |
| 15 | `config_seq` | `uint8` | 1 | 对应最近一条 stage config |
| 16 | `active_mask` | `uint8` | 1 | 当前执行高频激励的电机 |
| 17 | `online_mask` | `uint8` | 1 | 六个电机在线状态 |
| 18 | `tx_valid_mask` | `uint8` | 1 | 已有成功 Tx Event 的电机 |
| 19 | `rx_valid_mask` | `uint8` | 1 | 已有有效反馈的电机 |
| 20 | `saturation_mask` | `uint8` | 1 | 本阶段发生命令限幅的电机 |
| 21 | `reserved` | `uint8` | 1 | v1 必须为 0 |
| 22 | `safety_flags` | `uint16` | 2 | 越界、超温、超时等锁存状态 |
| 24 | `can_error_flags` | `uint16` | 2 | bus-off、error-passive、FIFO/event 丢失等状态 |
| 26 | `dm[4]` | 见下表 | 64 | 每台 16 B，canonical index 0..3 |
| 90 | `lk[2]` | 见下表 | 32 | 每台 16 B，canonical index 4..5 |
| 122 | `crc32` | `uint32` | 4 | CRC-32/ISO-HDLC，覆盖 offset 0..121 |

DM block 固定为 16 B，并直接保存 MIT 协议量化后的整数码：

| block offset | 字段 | 类型 | 字节数 |
|---:|---|---|---:|
| 0 | `cmd_q_raw` | `uint16` | 2 |
| 2 | `cmd_dq_raw` | `uint16` | 2 |
| 4 | `cmd_tau_raw` | `uint16` | 2 |
| 6 | `fb_q_raw` | `uint16` | 2 |
| 8 | `fb_dq_raw` | `uint16` | 2 |
| 10 | `fb_tau_raw` | `uint16` | 2 |
| 12 | `tx_age_us` | `uint16` | 2 |
| 14 | `rx_age_us` | `uint16` | 2 |

其中 `cmd_q_raw` 和 `fb_q_raw` 使用 16 bit；`cmd_dq_raw`、`cmd_tau_raw`、`fb_dq_raw`、`fb_tau_raw` 的有效数据均为低 12 bit。命令字段表示最近一次成功 Tx Event 对应的 CAN payload，不表示尚未发送的请求值。

LK block 固定为 16 B：

| block offset | 字段 | 类型 | 字节数 | 说明 |
|---:|---|---|---:|---|
| 0 | `cmd_primary_raw` | `int32` | 4 | 转矩模式保存符号扩展后的 `iqControl`，速度模式保存 `speedControl` |
| 4 | `fb_encoder_raw` | `uint16` | 2 | 单圈原始编码器值 |
| 6 | `fb_turn_count` | `int16` | 2 | MCU 解绕圈数的低 16 bit，PC 按二进制补码回绕继续展开 |
| 8 | `fb_speed_raw` | `int16` | 2 | LK 原始速度值 |
| 10 | `fb_iq_raw` | `int16` | 2 | LK 原始电流控制值 |
| 12 | `tx_age_us` | `uint16` | 2 | 距最近成功 Tx Event 的时间 |
| 14 | `rx_age_us` | `uint16` | 2 | 距最近有效反馈的时间 |

所有 `age_us` 在 65535 us 处饱和；没有有效历史数据时清除对应 valid bit。`sample_time_us` 在单个 session 内使用 32 bit 相对时间，session 最长限制为 60 min，PC 端仍必须支持无符号回绕展开。

### 7.3 Stage config、status 和 event frame

`command_mode`、`Kp/Kd`、命令频率和激励配置不在每个 sample 中重复。每个阶段开始前必须发送一条 stage config，并在任何配置变化时递增 `config_seq` 后重新发送。stage config 至少包含：

```text
config_seq
stage_id
command_mode[6]
command_rate_hz[6]
dm_kp_raw[4]
dm_kd_raw[4]
excitation_type
amplitude
frequency_range
duration
safety_limit_set_id
experiment_config_hash
```

温度和低频诊断使用 64 B 基础 status frame，以 10 Hz 发送。基础字段占 60 B，随后允许加入 0..64 B 的可选 TLV 扩展，最后追加 4 B CRC，因此线上总长度为 64..128 B。所有基础计数均为“自上一条 status frame 以来”的 `uint16` 增量，累计值由 PC 端恢复：

| 字段 | 类型 | 字节数 |
|---|---|---:|
| 公共帧头 | - | 10 |
| `status_time_us` | `uint32` | 4 |
| `motor_state_nibbles` | `uint8[3]` | 3 |
| `dm_temp_mos` | `int8[4]` | 4 |
| `dm_temp_rotor` | `int8[4]` | 4 |
| `lk_temp` | `int8[2]` | 2 |
| `tx_count_delta` | `uint16[6]` | 12 |
| `rx_count_delta` | `uint16[6]` | 12 |
| `enqueue_fail_delta` | `uint16` | 2 |
| `tx_event_lost_delta` | `uint16` | 2 |
| `can_error_delta` | `uint16` | 2 |
| `tx_fifo_high_water` | `uint8` | 1 |
| `can_state` | `uint8` | 1 |
| `extension_length` | `uint8` | 1 |
| `extension_tlv` | `uint8[0..64]` | 0..64 |
| `crc32` | `uint32` | 4 |

无扩展时合计 64 B，扩展区全满时合计 128 B。每个 TLV 使用 `type:uint8 + length:uint8 + value[length]`，未知 type 必须依据 length 跳过。扩展区可用于每通道入队失败、每通道 Tx Event 丢失、32 bit 累计计数或协议错误快照。`frame_length` 必须等于 `64 + extension_length`，CRC 位于帧末尾并覆盖其之前的全部字节。

基础计数增量发生 16 bit 饱和时必须发送 event frame 并标记该 session 的计数不再完整。安全中止、阶段切换和总线状态突变也必须立即发送 event frame，不能等待下一条 status frame。

### 7.4 传输要求

- 使用二进制固定帧或带长度字段的帧；
- 每帧有递增序号；
- 每帧有 CRC；
- PC 能检测丢帧、重复帧和乱序；
- sample frame 编码后的长度必须严格等于 126 B，status frame 必须在 64..128 B 范围内且与 `extension_length` 一致；
- 500 Hz sample 加 10 Hz status 的持续 UART7 占用不得超过理论容量的 70%；
- 不允许使用 C 结构体直接映射线上帧，所有字段必须按规定 offset 显式写入；
- 实验结束时发送 session footer，包含总帧数、丢帧数和 overflow 标记；
- 原始二进制文件保留，CSV/NPZ/PT 都由 PC 转换生成。

## 8. 实验状态机

```text
BOOT
  -> SAFE_IDLE
  -> CONFIGURED
  -> ARMED
  -> RUNNING
  -> STOPPING
  -> COMPLETE

任何状态 --故障--> FAULT
FAULT --复位/重新上电--> SAFE_IDLE
```

### 8.1 两类硬件会话

| 会话类型 | 目的 | 数据地位 |
|---|---|---|
| `COMMISSIONING_SESSION` | 验证方向、零位、时序、带宽、安全停机、激励可执行性和 PC 解码/初拟合链路 | `DIAGNOSTIC` 或 `PROVISIONAL_FIT` |
| `FINAL_IDENTIFICATION_SESSION` | 覆盖已冻结的 RL 低层动作范围和带宽，产生正式拟合与留出验证数据 | `FIT` 和 `VALIDATION` |

`FINAL_IDENTIFICATION_SESSION` 开始前必须冻结以下最低层接口：

- RL/低层控制 action 更新频率和保持方式；
- DM 默认姿态、预计位置目标幅值和主要频率范围；
- DM 部署时是否保持 `tau_ff = 0`；
- LK 9025 最终采用转矩命令还是速度命令；
- 实机紧急中止边界。

网络结构、观测量、奖励函数、训练算法和 domain randomization 不属于正式 PACE 采集的前置条件。

### 8.2 会话内部阶段

每个会话生成一份带版本信息的原始文件，文件内部由明确标记的数据段组成：

| 阶段 | 内容 |
|---|---|
| `STAGE_STATIC` | 零位、静态保持、通信、噪声和温度检查 |
| `STAGE_DM_FIT` | 四个 DM 同步、对称的位置目标 chirp；正式会话中标记为 `FIT` |
| `STAGE_DM_VALIDATION` | 不同频率、幅值或相位的位置轨迹；不参与同轮参数优化 |
| `STAGE_LK_TORQUE` | 两个 LK 9025 的正反向转矩/电流激励 |
| `STAGE_LK_VELOCITY` | 两个 LK 9025 的正反向速度目标激励 |
| `STAGE_STOP` | 平滑停止、统计计数和 session footer |

工程试采可以缩短或省略部分轮毂阶段。正式会话必须在同一原始文件中保留独立的 `FIT` 和 `VALIDATION` 数据段，不能随机拆分同一条轨迹后同时用于拟合和评价。

### 8.3 DM 主拟合基线

四个 DM 8009 使用相同的主控制接口，并通过方向、中心位置和幅值形成机械上对称的同步运动：

```text
command_mode = DM_MIT
q_des        = chirp_target
dq_des       = 0
Kp           = 20.0
Kd           = 0.6
tau_ff       = 0
```

`STAGE_DM_FIT` 和 `STAGE_DM_VALIDATION` 均使用上述 `Kp/Kd`。阶段之间改变的是轨迹，不是主控制增益。采集工程不运行 VMC 或 LQR，也不叠加来自底盘控制器的转矩前馈。

### 8.4 LK 采集定义

`LK` 是当前控制代码对两台 LK 9025 轮毂电机的类型前缀，不是 PACE 算法模块。LK 数据用于辨识轮端命令延迟、正反向响应、死区、摩擦、空载惯量和转矩/速度闭环特性。

LK 不能直接套用 DM 的位置 chirp。工程试采固件必须支持并区分转矩和速度两种独立数据段，实际试采按机械安全和准备情况执行其中一种或两种；正式会话根据已冻结的 RL 低层接口选择主模型，并可保留另一模式作为诊断。仅在悬空条件下采集的 LK 数据不等同于轮地接触和打滑模型。

### 8.5 激励和安全原则

- sample frame 必须记录最近一次成功 Tx Event 对应的量化命令码和 `tx_age_us`，不能把理想轨迹或仅成功入队的值当作实际发送值；
- 如需保留 `requested_command`，只能作为诊断字段，拟合默认使用 Tx-confirmed `encoded_command`；
- 每个阶段必须通过 stage config 记录 command mode、`Kp/Kd` 原始码、各电机命令频率、持续时间、幅值和频率配置；
- 每条反馈必须保留由 FDCAN Rx timestamp 映射得到的 `rx_timestamp` 或逐采样 `rx_age_us`；
- 正常激励范围不得触发安全边界；
- 当前阶段不冻结最终 RL 关节工作限位；
- MCU 必须保留独立的紧急位置、速度、转矩、温度和反馈超时中止边界；
- 安全边界触发后立即平滑撤销命令或失能、标记 session 无效，不通过持续裁剪命令生成可用于拟合的数据。

## 9. Firmware 工程结构

```text
wheel_leg_motor_PACE/
├─ MDK-ARM/
├─ Core/
│  ├─ Inc/
│  └─ Src/
├─ Board/
│  ├─ Inc/
│  └─ Src/
├─ Motor/
│  ├─ Inc/
│  └─ Src/
├─ PACE/
│  ├─ Inc/
│  └─ Src/
├─ Protocol/
│  ├─ Inc/
│  └─ Src/
├─ Config/
├─ Drivers/
├─ Middlewares/FreeRTOS/
├─ docs/
└─ README.md
```

### 9.1 模块职责

| 模块 | 责任 | 不应包含 |
|---|---|---|
| `Core` | STM32 启动、HAL、时钟、中断入口 | 机器人控制逻辑 |
| `Board` | CAN、UART、定时器、GPIO、DMA 端口 | PACE 参数拟合 |
| `Motor` | DM/LK 协议、状态解码、六电机注册表 | VMC、LQR |
| `PACE` | 实验状态机、激励、安全、采样调度 | 仿真接口 |
| `Protocol` | 配置命令、数据帧、CRC、状态响应 | 电机动力学模型 |
| `Config` | 顺序、CAN ID、限幅、默认实验参数 | 分散的魔法数字 |

不从原工程复制 `ChassisL_Task.c`、`ChassisR_Task.c`、VMC、INS、遥控器、音乐、裁判系统和跳跃控制模块。

## 10. PC 端职责和输出

PC 端后续以现有 `a1_motor_id_flow_20260726_clean` 的通用 PACE 优化器为基础，新增 wheel-leg 数据适配层。A1 的采集器、12 关节 CSV 转换器和 A1 专用配置不能直接复用。

PC 端处理顺序固定为：

```text
raw_session.bin
    -> frame decoder
    -> manifest/order/sign normalization
    -> leg_dm_dataset + wheel_lk_dataset
    -> separate actuator fitting
    -> common fitted model manifest
    -> Isaac replay
    -> MuJoCo replay
    -> train/validation report
```

MuJoCo 当前的六执行器顺序和几何模型可以复用，但当前直接把控制器扭矩写入 `d->ctrl` 的逻辑不能作为最终电机模型。Isaac 侧需要把当前六通道 manifest 和拟合参数接入目标 articulation。

## 11. 复用和重写范围

| 来源 | 直接复用 | 需要适配或重写 |
|---|---|---|
| `wheel_leg_debug-main` | STM32、CAN、DM/LK 解码、FDCAN 回调、串口/DMA 基础 | 独立采集任务、二进制协议、环形缓冲、安全状态机 |
| `mujoco_control_extract` | 六执行器顺序、五连杆几何、闭链模型、符号映射 | 真实电机延迟/摩擦/偏置模型、真实数据回放 |
| `a1_motor_id_flow` | PACE 优化器、delay/bias 思路、交叉验证方法 | 六通道数据格式、DM/LK 两类 actuator、A1 专用脚本 |

按最终跨仿真闭环计算，现有代码资产约有一半可以复用；最大的新增工作是采集协议和两个仿真端的 actuator 适配，而不是底层 CAN 驱动。

## 12. 实施阶段和验收标准

当前实施状态：

| 阶段 | 软件状态 | 仍需外部验证 |
|---|---|---|
| Phase 0 | 已完成 | 首次上电后回填方向、零位和校准状态 |
| Phase 1 | 工程和 Keil 项目文件已生成 | 本机无 Keil/ARM 编译器，需在装有 MDK 的电脑编译下载 |
| Phase 2 | 协议、注册表和安全停机路径已实现 | 六台实机 CAN ID、方向、使能和急停 |
| Phase 3 | 采集协议、状态机、Tx Event、500 Hz 快照和 DMA 传输已实现 | 示波/CAN/UART 实测时序与持续传输验收 |
| Phase 4 | 解码、归一化、CSV/NPZ/PT 导出已实现 | 用首份实机 raw 文件核对物理单位和符号 |
| Phase 5 | 临时 DM/LK 拟合与共用 manifest 已实现 | 运行低幅值 `COMMISSIONING_SESSION` 并生成 provisional 报告 |
| Phase 6 | 最终合同模板和固件锁已实现 | 冻结 RL 最低层执行器接口并填写非零合同哈希 |
| Phase 7 | Isaac/MuJoCo 适配和合成测试已实现 | 正式 PACE/Isaac 优化、最终实机数据和双仿真回放 |
| Phase 8 | 检查表已建立 | 完成全部硬件安全回归记录 |

### Phase 0: 文档和接口冻结

交付：

- `pace_motor_manifest`；
- 原始数据字段表；
- 命令/状态协议；
- `COMMISSIONING_SESSION` 与 `FINAL_IDENTIFICATION_SESSION` 定义；
- DM 主拟合命令 `q_des=chirp`、`dq_des=0`、`Kp=20.0`、`Kd=0.6`、`tau_ff=0`；
- 实验阶段和数据用途定义；
- 安全限值表。

验收：同一个六通道索引在 Firmware、PC、Isaac 和 MuJoCo 中含义一致，并且每个原始数据段能够区分会话类型、阶段和数据用途。

### Phase 1: 独立 Keil 工程骨架

交付：

- `wheel_leg_motor_PACE.uvprojx`；
- STM32H723 启动和 HAL；
- FDCAN3、UART7、定时器和 DMA；
- 最小 FreeRTOS 运行环境；
- 无底盘控制代码。

验收：工程能编译、下载、启动，并通过串口输出版本信息。

### Phase 2: 六电机通信和安全

交付：

- DM/LK 命令发送；
- 六通道反馈解码；
- CAN ID 到 canonical index 的映射；
- 掉线和急停；
- 六电机安全零命令。

验收：六个电机都能被识别，状态表中的顺序、方向和在线状态正确。

### Phase 3: 采集协议和实验状态机

交付：

- 500 Hz 采样调度；
- 126 B 固定 sample frame、64..128 B 的 10 Hz status frame、stage config、event、footer、序号和 CRC；
- 环形缓冲区和 UART DMA 传输；
- UART7 持续占用不高于理论容量 70% 的静态预算检查；
- FDCAN3 自动重传、Tx Event FIFO、MessageMarker、发送入队结果和 Rx timestamp；
- 500 Hz sample rate 与分阶段 per-motor command rate 调度；
- 每阶段 CAN 负载估算、入队失败、Tx Event 丢失、发送周期抖动和反馈年龄统计；
- 工程试采/正式采集会话类型；
- 静态、DM 拟合、DM 验证、LK 转矩、LK 速度和停机阶段；
- 固定 `Kp=20.0`、`Kd=0.6`、`tau_ff=0` 的 DM 同步对称 chirp；
- overflow 和丢帧报告。

验收：

- 编码器对每个 sample 必须输出恰好 126 B；即使 status 每次都使用 128 B 上限，500 Hz sample 加 10 Hz status 的持续 UART7 占用也不得超过 70%；
- PC 能还原完整 session，帧数、时间戳、`config_seq` 和序号连续，无静默丢样本；
- 正常实验配置的保守 CAN 负载不高于 65%，超过 70% 的配置不能进入 `ARMED`；
- 活跃 500 Hz 通道不得出现大于等于 4 ms 的命令发送间隔，enqueue 到 Tx Event 的最大延迟必须小于 2 ms；
- 不允许出现未上报的 Tx FIFO 入队失败、Tx Event FIFO 丢失、UART overflow 或反馈失效；
- 报告每个通道发送周期、反馈年龄和 enqueue-to-Tx 延迟的 P50、P95、P99 和最大值。

### Phase 4: PC 解码和数据归一化

交付：

- raw binary decoder；
- manifest 校验；
- DM/LK 数据拆分；
- CSV/NPZ/PT 输出；
- 命令与反馈可视化。

验收：六个通道都能绘制命令、位置、速度和扭矩/电流曲线，单位和符号经过人工核验。

### Phase 5: 工程试采和临时拟合

交付：

- 低幅值、短时间的 `COMMISSIONING_SESSION`；
- 六通道方向、零位、时序、反馈年龄和数据完整性报告；
- DM `20.0/0.6/0` 基线下的临时 PACE fit；
- LK 转矩模式和速度模式的诊断曲线；
- chirp 安全幅值和可用频率范围建议；
- 临时模型 manifest，显式标记 `model_maturity = provisional`。

验收：工程试采无未解释丢帧或安全故障，PC 能完成一次端到端初步拟合；该结果只用于工程验证和后续接口决策，不作为最终 sim2real 模型发布。

### Phase 6: 最终辨识接口冻结

交付：

- action 更新频率和保持方式；
- DM 默认姿态、目标幅值和频率覆盖范围；
- DM `tau_ff` 部署约定；
- LK 最终主命令模式；
- 正式会话紧急中止边界；
- 与固件 build ID 绑定的 final experiment config hash。

验收：正式采集配置不存在未定义字段，并且其动作范围和带宽覆盖计划中的 RL 低层输出。

### Phase 7: 正式参数拟合和双仿真回放

交付：

- 一次包含独立 `FIT` 和 `VALIDATION` 数据段的 `FINAL_IDENTIFICATION_SESSION`；
- 腿部 DM PACE fit；
- 轮毂 LK actuator fit；
- Isaac replay adapter；
- MuJoCo replay adapter；
- 同一参数 manifest 的跨仿真读取；
- 最终模型 manifest，标记 `model_maturity = final`。

验收：拟合段和验证段分别报告每个通道的 RMSE、P95、延迟和方向误差。

### Phase 8: 硬件安全回归和实验定版

交付：

- 急停测试记录；
- 电机掉线测试记录；
- 超限停止测试记录；
- 采集文件完整性检查；
- 固件版本、manifest 版本和实验配置绑定。

验收：任何异常都能使六个电机进入安全状态，并且不会生成未标记的无效数据。

## 13. 当前不纳入第一版的内容

- 外部扭矩传感器接口；
- SD 卡本地存储；
- USB 高速传输；
- 在线参数拟合；
- 机身 IMU 融合；
- 自动闭链姿态求解；
- 自动调节激励幅值；
- RL 网络结构、观测量、奖励函数和训练算法；
- domain randomization 参数定版。

第一版先保证六通道数据完整、可复现、可解码。若 UART7 带宽或机械负载条件不足，再单独增加 USB/SD 或外部传感器方案。

## 14. 关键风险

1. DM 回传扭矩是驱动器估算值，不等同于独立扭矩传感器测量值；第一版拟合结果定义为“有效执行器模型”。
2. 如果实验过程中基座自由运动，仅六电机数据可能无法区分电机动力学和整机负载；第一版应优先使用固定或受控负载条件。
3. 轮毂 LK 与腿部 DM 的控制接口不同，必须分模型拟合。
4. UART7 921600 baud 在 500 Hz 下最多提供 184.32 B/sample；未压缩的逐通道 float 帧必然无法传输，协议 v1 必须保持 126 B sample frame、最大 128 B 的 10 Hz 状态帧和不高于 70% 的持续占用。
5. 任何坐标、符号和零位修正都必须进入 manifest，不能靠绘图脚本临时修正。
6. 工程试采的动作范围可能不覆盖最终 RL 输出；除非配置完全一致并通过留出验证，否则不能将 provisional 数据直接升级为最终模型。
7. 如果把六台电机都按 500 Hz 命令并假设一命令一反馈，1 Mbit/s CAN 负载约为 78%；必须使用分阶段命令频率，且依据 Tx Event 和 Rx timestamp 验证真实时序。
8. FDCAN3 虽已启用自动重传，但错误重发会增加延迟；重传不能替代 Tx Event、错误计数和反馈年龄监控。

## 15. 实施前冻结项

以下默认值作为实施基线：

- 工程目录：`wheel_leg_motor_PACE/`；
- 采样目标：500 Hz；
- sample frame：little-endian、固定 126 B、CRC-32/ISO-HDLC；
- status frame：10 Hz、64 B 基础布局、线上总长度上限 128 B；
- UART7 持续占用：不超过理论容量 70%；
- 六通道顺序：MuJoCo 当前顺序；
- 激励配置：PC 下发配置，MCU 本地执行；
- 实时传输：UART7 二进制帧；
- 电机总线：FDCAN3 Classic CAN 1 Mbit/s，保留自动重传；
- 调度语义：六通道快照保持 500 Hz，命令频率按阶段分别配置，保守 CAN 负载目标不高于 65%，超过 70% 拒绝运行；
- 发送语义：以 Tx Event 确认后的量化命令码作为实际发送值，同时记录 `tx_age_us` 和 `rx_age_us`；
- 实机只输出原始数据，不执行 PACE 优化；
- DM 和 LK 分成两类 actuator model；
- DM 主拟合接口：`q_des=chirp`、`dq_des=0`、`Kp=20.0`、`Kd=0.6`、`tau_ff=0`；
- 采集分为工程试采和最终辨识两轮；
- 当前实现里程碑是完成工程试采闭环，最终辨识由 RL 最低层接口冻结结果解锁；
- 当前不冻结最终 RL 关节工作限位，但实机必须有独立的越界中止边界；
- 第一版不引入 VMC、IMU、底盘控制和外部扭矩传感器。

当前可以进入 Keil 编译和低幅值工程试采。正式大批量采集仍必须等待 Phase 6 的最低层接口冻结完成；固件中的 `PACE_FINAL_CONTRACT_READY` 当前固定为 `0`，因此正式会话保持锁定。
