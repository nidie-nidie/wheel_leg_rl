# PACE Session Format v1

本文档描述 `wheel_leg_motor_PACE` 与 PC 之间的线协议。所有多字节整数和 `float32` 均为 little-endian；所有输出帧末尾使用 CRC-32/ISO-HDLC。

## 1. 通用输出帧头

每条 MCU -> PC 帧都以固定 10 B 帧头开始：

| Offset | 字段 | 类型 | 长度 |
|---:|---|---|---:|
| 0 | magic | `uint8[2] = {0x50,0x41}` (`PA`) | 2 |
| 2 | protocol_version | `uint8 = 1` | 1 |
| 3 | frame_type | `uint8` | 1 |
| 4 | frame_length | `uint16` | 2 |
| 6 | sequence | `uint32` | 4 |

`sequence` 在 session 的所有 frame type 之间共用并逐帧加一。

| Type | 名称 | 长度 |
|---:|---|---:|
| `0x01` | session header | 58 B |
| `0x02` | sample | 126 B |
| `0x03` | stage config | 76 B |
| `0x04` | status | 64..128 B |
| `0x05` | event | 30 B |
| `0x06` | footer | 34 B |

## 2. Session Header

| Offset | 字段 | 类型 |
|---:|---|---|
| 10 | session_type (`1=commissioning`, `2=final`) | `uint8` |
| 11 | motor_order_version | `uint8` |
| 12 | wire_endianness (`1=little`) | `uint8` |
| 13 | reserved | `uint8` |
| 14 | session_id | `uint32` |
| 18 | sample_rate_hz | `uint16` |
| 20 | status_rate_hz | `uint16` |
| 22 | start_timestamp_us | `uint32` |
| 26 | experiment_config_hash | `uint32` |
| 30 | sample_frame_size_bytes | `uint16` |
| 32 | reserved | `uint16` |
| 34 | uart_baud | `uint32` |
| 38 | can_nominal_bitrate | `uint32` |
| 42 | scale_manifest_hash | `uint32` |
| 46 | firmware_build_id | `uint32` |
| 50 | robot_variant | `uint32` |
| 54 | CRC32 | `uint32` |

## 3. Sample Frame

固定 126 B，当前无线 UART7 配置以 250 Hz 发送（电机/CAN 控制仍为 500 Hz）。

| Offset | 字段 | 类型 |
|---:|---|---|
| 10 | sample_time_us，session 相对时间 | `uint32` |
| 14 | stage_id | `uint8` |
| 15 | config_seq | `uint8` |
| 16 | active_mask | `uint8` |
| 17 | online_mask | `uint8` |
| 18 | tx_valid_mask | `uint8` |
| 19 | rx_valid_mask | `uint8` |
| 20 | saturation_mask | `uint8` |
| 21 | reserved | `uint8` |
| 22 | safety_flags | `uint16` |
| 24 | can_error_flags | `uint16` |
| 26 | DM0 block | 16 B |
| 42 | DM1 block | 16 B |
| 58 | DM2 block | 16 B |
| 74 | DM3 block | 16 B |
| 90 | LK0 block | 16 B |
| 106 | LK1 block | 16 B |
| 122 | CRC32 | `uint32` |

每个 DM block：

| Block offset | 字段 | 类型 | 说明 |
|---:|---|---|---|
| 0 | cmd_q_raw | `uint16` | 最近 Tx-confirmed 命令，16 bit |
| 2 | cmd_dq_raw | `uint16` | 有效数据为低 12 bit |
| 4 | cmd_tau_raw | `uint16` | 有效数据为低 12 bit |
| 6 | fb_q_raw | `uint16` | 16 bit |
| 8 | fb_dq_raw | `uint16` | 有效数据为低 12 bit |
| 10 | fb_tau_raw | `uint16` | 有效数据为低 12 bit |
| 12 | tx_age_us | `uint16` | 距最近 Tx Event |
| 14 | rx_age_us | `uint16` | 距最近有效反馈 |

每个 LK block：

| Block offset | 字段 | 类型 |
|---:|---|---|
| 0 | cmd_primary_raw | `int32` |
| 4 | fb_encoder_raw | `uint16` |
| 6 | fb_turn_count | `int16` |
| 8 | fb_speed_raw | `int16` |
| 10 | fb_iq_raw | `int16` |
| 12 | tx_age_us | `uint16` |
| 14 | rx_age_us | `uint16` |

`age_us` 在 65535 us 饱和；没有时间历史时清除对应 valid bit。拟合只能使用 valid、未饱和且有 stage config 上下文的样本。

## 4. Stage Config

固定 76 B，必须出现在任何引用新 `config_seq` 的 sample 之前。

| Offset | 字段 | 类型 |
|---:|---|---|
| 10 | config_seq | `uint8` |
| 11 | stage_id | `uint8` |
| 12 | command_mode[6] | `uint8[6]` |
| 18 | command_rate_hz[6] | `uint16[6]` |
| 30 | dm_kp_raw[4] | `uint16[4]` |
| 38 | dm_kd_raw[4] | `uint16[4]` |
| 46 | excitation_type | `uint8` |
| 47 | reserved | `uint8` |
| 48 | amplitude | `float32` |
| 52 | frequency_start_hz | `float32` |
| 56 | frequency_end_hz | `float32` |
| 60 | duration_ms | `uint32` |
| 64 | safety_limit_set_id | `uint16` |
| 66 | reserved | `uint16` |
| 68 | experiment_config_hash | `uint32` |
| 72 | CRC32 | `uint32` |

`command_mode`：`1=DM_MIT`、`5=LK_TORQUE`、`6=LK_VELOCITY`；其他数值只用于使能/失能控制，不应出现在拟合阶段配置中。

## 5. Status Frame

基础 64 B，以 10 Hz 发送；TLV 扩展使总长度最多为 128 B。

| Offset | 字段 | 类型 |
|---:|---|---|
| 10 | status_time_us | `uint32` |
| 14 | motor_state_nibbles | `uint8[3]` |
| 17 | dm_temp_mos | `int8[4]` |
| 21 | dm_temp_rotor | `int8[4]` |
| 25 | lk_temp | `int8[2]` |
| 27 | tx_count_delta | `uint16[6]` |
| 39 | rx_count_delta | `uint16[6]` |
| 51 | enqueue_fail_delta | `uint16` |
| 53 | tx_event_lost_delta | `uint16` |
| 55 | can_error_delta | `uint16` |
| 57 | tx_fifo_high_water | `uint8` |
| 58 | can_state | `uint8` |
| 59 | extension_length | `uint8` |
| 60 | extension_tlv | `uint8[0..64]` |
| `frame_length-4` | CRC32 | `uint32` |

所有 `delta` 都是相对上一条 status 的增量。发生 `uint16` 饱和时必须发 event，并将 footer 的 `statistics_complete` 清零。

## 6. Event 与 Footer

Event 固定 30 B：

| Offset | 字段 | 类型 |
|---:|---|---|
| 10 | event_time_us | `uint32` |
| 14 | event_code | `uint16` |
| 16 | severity | `uint8` |
| 17 | motor_index | `uint8` |
| 18 | argument0 | `uint32` |
| 22 | argument1 | `uint32` |
| 26 | CRC32 | `uint32` |

Footer 固定 34 B：

| Offset | 字段 | 类型 |
|---:|---|---|
| 10 | stop_time_us | `uint32` |
| 14 | total_frames | `uint32` |
| 18 | dropped_frames | `uint32` |
| 22 | overflow | `uint8` |
| 23 | statistics_complete | `uint8` |
| 24 | stop_reason | `uint16` |
| 26 | experiment_config_hash | `uint32` |
| 30 | CRC32 | `uint32` |

## 7. PC -> MCU Host Command

命令 magic 为 `0x50 0x43` (`PC`)，版本为 1。帧结构为：

| Offset | 字段 | 类型 |
|---:|---|---|
| 0 | magic | `uint8[2]` |
| 2 | version | `uint8` |
| 3 | command_id | `uint8` |
| 4 | frame_length | `uint16` |
| 6 | sequence | `uint32` |
| 10 | payload | `uint8[]` |
| `frame_length-4` | CRC32 | `uint32` |

命令 ID：`1=CONFIGURE`、`2=START`、`3=STOP`、`4=STATUS`、`5=ABORT`。`CONFIGURE` 可携带 1 B session type；命令 sequence 必须严格连续。

## 8. 带宽约束

```text
sample: 126 B * 250 Hz = 31500 B/s
status worst case: 128 B * 10 Hz = 1280 B/s
total sustained: 64280 B/s
UART7 capacity at 921600 8N1: 92160 B/s
occupancy: 69.75%
```

session header、stage config、event 和 footer 只能是低频短帧。任何实现不得扩大 sample frame 或提高 status 频率而不重新计算持续带宽。
