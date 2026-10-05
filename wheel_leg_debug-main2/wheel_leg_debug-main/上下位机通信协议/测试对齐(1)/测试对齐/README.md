# 测试对齐工具

本目录用于轮腿机器人真机和 Isaac Sim/USD 仿真之间的信息流对齐调试。

目录内已包含默认使用的
`wheel_leg_urdf4_cod_real_closed_chain_floating_base_artic.usd`。脚本会按
`align_common.py` 所在目录定位该文件，因此整个 `测试对齐` 目录可以一起搬运。
若要使用其他模型，运行时传入 `--usd-path /absolute/path/to/model.usd`。

协议沿用：

- 大端序 big-endian / network byte order
- `CmdFrame = 162 bytes`
- `StateFrame = 174 bytes`
- 主动腿顺序：`[jIO, jAG, jIJ, jAB]`
- 轮子顺序：`[jwheel_left, jwheel_right]`
- 不接收、不使用被动关节角度

## 1. 仿真数据校验解析工具

只接收下位机 `state`，把主动腿角度和轮子速度同步到 USD，可用于检查关节映射。

```bash
cd /home/bubble/港中深暑研

./IsaacLab-2.3.2/_isaac_sim/python.sh 测试对齐/01_sim_state_joint_viewer.py \
  --show \
  --transport serial \
  --serial-port /dev/ttyUSB0 \
  --baudrate 921600
```

UDP 接收：

```bash
./IsaacLab-2.3.2/_isaac_sim/python.sh 测试对齐/01_sim_state_joint_viewer.py \
  --show \
  --transport udp \
  --udp-bind-port 15002
```

## 2. 电机参数调控双向交互 UI

UI 可调：

- `mode`
- `ttl_ms`
- 主动腿 `q_des / dq_des`
- 轮子 `dq_des`
- 腿部 `Kp / Kd`
- 轮子速度环 `Kp / Kd`

同时接收下位机 `state` 并驱动 USD 做真机-仿真对照。

```bash
./IsaacLab-2.3.2/_isaac_sim/python.sh 测试对齐/02_motor_control_ui.py \
  --show \
  --transport udp \
  --udp-bind-port 15002 \
  --udp-remote-host 127.0.0.1 \
  --udp-remote-port 15001
```

串口双向通信：

```bash
./IsaacLab-2.3.2/_isaac_sim/python.sh 测试对齐/02_motor_control_ui.py \
  --show \
  --transport serial \
  --serial-port /dev/ttyUSB0 \
  --baudrate 921600
```

## 3. IMU 姿态对齐 UI

接收下位机 `state` 中的 `imu_quat_wxyz` 和 `imu_gyro_rad_s`，显示 roll/pitch/yaw，并用四元数驱动 USD 内 `base_link` 的姿态。

```bash
./IsaacLab-2.3.2/_isaac_sim/python.sh 测试对齐/03_imu_pose_align_ui.py \
  --show \
  --transport serial \
  --serial-port /dev/ttyUSB0 \
  --baudrate 921600
```

## 4. 独立 Qt 串口收发调试 UI

不依赖 IsaacLab/Isaac Sim，只用于快速验证 USB 串口是否能正常收发协议帧。

功能：

- 打开 `/dev/ttyUSB*`、`/dev/ttyACM*` 等串口
- 接收并解析下位机 `state` 帧，显示状态、IMU、关节、电流、电压、温度
- 单发或周期发送上位机 `command` 帧
- 手动填写 command 字段，包括腿/轮目标、PD、力矩/速度限幅
- 支持 raw hex 直接写串口，方便测下位机解析器

```bash
cd /home/bubble/港中深暑研

python3 测试对齐/04_qt_serial_comm_ui.py \
  --serial-port /dev/ttyUSB0 \
  --baudrate 921600
```

## 注意

脚本 3 默认认为 `imu_quat_wxyz` 是 `[w, x, y, z]` 且表示 body 到 world 的姿态。如果实机 IMU 给的是 world 到 body，或顺序是 `[x, y, z, w]`，需要先在下位机或上位机转换后再发送。
