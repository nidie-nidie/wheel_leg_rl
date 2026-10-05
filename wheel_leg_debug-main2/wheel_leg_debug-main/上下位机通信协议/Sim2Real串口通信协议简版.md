# Sim2Real 串口通信协议简版

本文档只定义上位机和下位机通信结构。所有多字节数据使用 **大端序 big-endian / network byte order**。所有 `float` 使用 **IEEE-754 float32，4 字节**，也按大端序发送。结构体按 **1 字节对齐 packed**，不允许编译器自动 padding。

## 1. 帧格式

每一帧统一格式：

| 字段 | 类型 | 字节 |
|---|---|---:|
| `sof` | `uint16` | 2 |
| `ver` | `uint8` | 1 |
| `msg_type` | `uint8` | 1 |
| `payload_len` | `uint16` | 2 |
| `seq` | `uint32` | 4 |
| `t_us` | `uint64` | 8 |
| `payload` | bytes | N |
| `crc32` | `uint32` | 4 |

固定值：

```c
sof = 0xAA55
ver = 1
msg_type = 1  // 上位机 -> 下位机 command
msg_type = 2  // 下位机 -> 上位机 state
```

帧头大小：

```text
2 + 1 + 1 + 2 + 4 + 8 = 18 bytes
```

帧总长度：

```text
18 + payload_len + 4
```

`crc32` 对 `sof` 到 `payload` 的全部字节计算，不包含 `crc32` 字段本身。

## 2. Command：上位机发给下位机

用途：上位机已经把 policy action 换算成物理控制目标，下位机只需要执行。

### 2.1 Payload 结构

| 字段 | 类型 | 数量 | 单位 | 字节 |
|---|---|---:|---|---:|
| `mode` | `uint8` | 1 | 无 | 1 |
| `reserved` | `uint8` | 1 | 无 | 1 |
| `ttl_ms` | `uint16` | 1 | ms | 2 |
| `leg_q_des_rad` | `float32` | 4 | rad | 16 |
| `leg_dq_des_rad_s` | `float32` | 4 | rad/s | 16 |
| `wheel_dq_des_rad_s` | `float32` | 2 | rad/s | 8 |
| `leg_kp` | `float32` | 4 | N*m/rad | 16 |
| `leg_kd` | `float32` | 4 | N*m*s/rad | 16 |
| `wheel_kp` | `float32` | 2 | N*m/(rad/s) | 8 |
| `wheel_kd` | `float32` | 2 | N*m*s/rad | 8 |
| `tau_limit_nm` | `float32` | 6 | N*m | 24 |
| `dq_limit_rad_s` | `float32` | 6 | rad/s | 24 |

Payload 总大小：

```text
1 + 1 + 2 + 16 + 16 + 8 + 16 + 16 + 8 + 8 + 24 + 24 = 140 bytes
```

Command 整帧大小：

```text
18 + 140 + 4 = 162 bytes
```

### 2.2 数组顺序

```text
leg_q_des_rad:
[jIO, jAG, jIJ, jAB]

leg_dq_des_rad_s:
[jIO, jAG, jIJ, jAB]

wheel_dq_des_rad_s:
[jwheel_left, jwheel_right]

tau_limit_nm:
[jIO, jAG, jIJ, jAB, jwheel_left, jwheel_right]

dq_limit_rad_s:
[jIO, jAG, jIJ, jAB, jwheel_left, jwheel_right]
```

### 2.3 mode 定义

| mode | 数值 | 含义 |
|---|---:|---|
| `IDLE` | 0 | 空闲，不输出 |
| `READY` | 1 | 准备，使能但不跑策略 |
| `RUN` | 2 | 正常执行控制 |
| `DAMPING` | 3 | 阻尼保护 |
| `ESTOP` | 4 | 急停 |

下位机超过 `ttl_ms` 没收到新 command，必须自动进入 `DAMPING` 或 `ESTOP`。

## 3. State：下位机发给上位机

用途：下位机回传传感器状态，上位机用它拼 policy observation。

### 3.1 Payload 结构

| 字段 | 类型 | 数量 | 单位 | 字节 |
|---|---|---:|---|---:|
| `status` | `uint8` | 1 | 无 | 1 |
| `reserved` | `uint8` | 1 | 无 | 1 |
| `fault_code` | `uint16` | 1 | 无 | 2 |
| `cmd_seq_echo` | `uint32` | 1 | 无 | 4 |
| `imu_gyro_rad_s` | `float32` | 3 | rad/s | 12 |
| `imu_quat_wxyz` | `float32` | 4 | 无 | 16 |
| `active_leg_q_rad` | `float32` | 4 | rad | 16 |
| `active_leg_dq_rad_s` | `float32` | 4 | rad/s | 16 |
| `wheel_dq_rad_s` | `float32` | 2 | rad/s | 8 |
| `joint_tau_nm` | `float32` | 6 | N*m | 24 |
| `motor_current_a` | `float32` | 6 | A | 24 |
| `bus_voltage_v` | `float32` | 1 | V | 4 |
| `temperature_c` | `float32` | 6 | degC | 24 |

Payload 总大小：

```text
1 + 1 + 2 + 4 + 12 + 16 + 16 + 16 + 8 + 24 + 24 + 4 + 24 = 152 bytes
```

State 整帧大小：

```text
18 + 152 + 4 = 174 bytes
```

### 3.2 数组顺序

```text
active_leg_q_rad:
[jIO, jAG, jIJ, jAB]

active_leg_dq_rad_s:
[jIO, jAG, jIJ, jAB]

wheel_dq_rad_s:
[jwheel_left, jwheel_right]

joint_tau_nm:
[jIO, jAG, jIJ, jAB, jwheel_left, jwheel_right]

motor_current_a:
[jIO, jAG, jIJ, jAB, jwheel_left, jwheel_right]

temperature_c:
[jIO, jAG, jIJ, jAB, jwheel_left, jwheel_right]
```

训练和部署时的 policy observation 应只使用主动腿关节、轮子和 IMU。`projected_gravity` 不由下位机发送；上位机用 `imu_quat_wxyz` 自己计算。

如果 `imu_quat_wxyz` 表示 body 到 world 的旋转 `q_wb`，则：

```text
projected_gravity = R_wb^T * [0, 0, -1]
```

其中 `[0, 0, -1]` 是世界系单位重力方向，不是 `9.81`。

## 4. C 结构体定义

```c
#pragma pack(push, 1)

typedef struct {
    uint16_t sof;          // 帧头固定值，必须为 0xAA55
    uint8_t  ver;          // 协议版本，当前固定为 1
    uint8_t  msg_type;     // 消息类型：1=上位机到下位机 cmd，2=下位机到上位机 state
    uint16_t payload_len;  // payload 字节数：CmdPayload=140，StatePayload=152
    uint32_t seq;          // 本端递增帧号，用于丢包、乱序、回显检查
    uint64_t t_us;         // 本端发送时间戳，单位 us
} FrameHeader;             // 18 bytes

typedef struct {
    uint8_t  mode;                 // 控制模式：0=IDLE，1=READY，2=RUN，3=DAMPING，4=ESTOP
    uint8_t  reserved;             // 保留字节，当前填 0
    uint16_t ttl_ms;               // 命令有效时间，单位 ms；超时后下位机进入 DAMPING 或 ESTOP

    float leg_q_des_rad[4];        // 主动腿目标角度，单位 rad，顺序 [jIO, jAG, jIJ, jAB]
    float leg_dq_des_rad_s[4];     // 主动腿目标角速度，单位 rad/s，顺序 [jIO, jAG, jIJ, jAB]；当前可全 0
    float wheel_dq_des_rad_s[2];   // 轮子目标角速度，单位 rad/s，顺序 [jwheel_left, jwheel_right]

    float leg_kp[4];               // 主动腿位置环 Kp，单位 N*m/rad，顺序 [jIO, jAG, jIJ, jAB]
    float leg_kd[4];               // 主动腿速度阻尼 Kd，单位 N*m*s/rad，顺序 [jIO, jAG, jIJ, jAB]
    float wheel_kp[2];             // 轮子速度环 Kp，单位 N*m/(rad/s)，顺序 [jwheel_left, jwheel_right]
    float wheel_kd[2];             // 轮子阻尼项 Kd，单位 N*m*s/rad，顺序 [jwheel_left, jwheel_right]

    float tau_limit_nm[6];         // 力矩限幅，单位 N*m，顺序 [jIO, jAG, jIJ, jAB, jwheel_left, jwheel_right]
    float dq_limit_rad_s[6];       // 速度限幅，单位 rad/s，顺序 [jIO, jAG, jIJ, jAB, jwheel_left, jwheel_right]
} CmdPayload;              // 140 bytes

typedef struct {
    uint8_t  status;               // 下位机状态：0=IDLE，1=READY，2=RUN，3=DAMPING，4=ESTOP，5=FAULT
    uint8_t  reserved;             // 保留字节，当前填 0
    uint16_t fault_code;           // 下位机错误码；0 表示无错误
    uint32_t cmd_seq_echo;         // 最近一次被下位机接收/执行的 command seq

    float imu_gyro_rad_s[3];       // IMU 角速度，单位 rad/s，机身坐标系顺序 [wx, wy, wz]
    float imu_quat_wxyz[4];        // IMU 姿态四元数，顺序 [w, x, y, z]；建议定义为 body 到 world 的旋转 q_wb

    float active_leg_q_rad[4];     // 主动腿实际角度，单位 rad，顺序 [jIO, jAG, jIJ, jAB]
    float active_leg_dq_rad_s[4];  // 主动腿实际角速度，单位 rad/s，顺序 [jIO, jAG, jIJ, jAB]
    float wheel_dq_rad_s[2];       // 轮子实际角速度，单位 rad/s，顺序 [jwheel_left, jwheel_right]

    float joint_tau_nm[6];         // 电机估算/反馈力矩，单位 N*m，顺序 [jIO, jAG, jIJ, jAB, jwheel_left, jwheel_right]
    float motor_current_a[6];      // 电机电流，单位 A，顺序 [jIO, jAG, jIJ, jAB, jwheel_left, jwheel_right]
    float bus_voltage_v;           // 母线电压，单位 V
    float temperature_c[6];        // 电机或驱动温度，单位 degC，顺序 [jIO, jAG, jIJ, jAB, jwheel_left, jwheel_right]
} StatePayload;            // 152 bytes

#pragma pack(pop)
```

## 5. 控制频率

| 项 | 建议 |
|---|---:|
| 上位机发送 command | 50 Hz |
| 下位机返回 state | 50 Hz 或 100 Hz |
| 下位机电机闭环 | 500 Hz 到 1 kHz |
| `ttl_ms` | 50 ms |

## 6. 串口参数

| 参数 | 值 |
|---|---|
| baudrate | 921600 |
| data bits | 8 |
| parity | None |
| stop bits | 1 |
| flow control | None |
