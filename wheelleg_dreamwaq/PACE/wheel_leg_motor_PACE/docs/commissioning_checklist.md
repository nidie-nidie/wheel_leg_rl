# Commissioning Checklist

本检查表用于第一次低幅值 `COMMISSIONING_SESSION`。任何安全、方向、时序或数据完整性项目失败时，停止试验并保留原始文件，不得继续扩大幅值。

## 1. 环境与人员

- [ ] 车辆可靠架起，四个腿部关节和两个轮子全行程不碰撞支架、线束或地面。
- [ ] 实体急停可触达，并有一人只负责观察机械状态和急停。
- [ ] 电源具有限流，首次上电使用可接受的最低电流限制。
- [ ] CAN 收发器、终端电阻、电机电源和逻辑地已经检查。
- [ ] UART7 使用可稳定支持 921600 baud 的 USB 转串口设备和短线。
- [ ] PC 端已安装 Python、NumPy、SciPy 和 pyserial。
- [ ] 原始数据目录为空或使用新的唯一文件名，不覆盖任何旧 raw 文件。

## 2. 构建记录

| 项目 | 记录值 |
|---|---|
| 日期/操作者 | |
| Git/文件快照标识 | |
| Keil 版本 | |
| STM32H7 Device Pack | |
| 固件 build ID | `0x20260928`（如已修改请记录实际值） |
| motor manifest hash | `0x4D4F5431`（如已修改请记录实际值） |
| experiment config hash | `0x50414331`（如已修改请记录实际值） |
| `.axf/.hex` SHA-256 | |
| USB-UART 型号 | |
| CAN 分析仪型号 | |

- [ ] `wheel_leg_motor_PACE.uvprojx` 全量 rebuild 无 error。
- [ ] 编译 warning 已逐条审阅，不存在隐式声明、截断、未对齐或栈溢出风险。
- [ ] map 文件确认 RAM 足以容纳 128×128 B capture ring、任务栈和 HAL buffer。
- [ ] 工程清单不含 Chassis、VMC、LQR、INS、Music、USB 或跳跃控制模块。
- [ ] `PACE_FINAL_CONTRACT_READY == 0`，正式会话仍锁定。

## 3. 无负载通信检查

- [ ] 先断开电机动力或保持驱动器失能，确认 MCU 启动无重复复位。
- [ ] FDCAN3 标称速率实测为 1 Mbit/s。
- [ ] FDCAN3 自动重传开启。
- [ ] Tx Event FIFO 有效，成功入队的命令都产生匹配的 MessageMarker 事件。
- [ ] 未确认命令不会写入 `last_confirmed_command`。
- [ ] UART7 实测为 921600 baud、8N1、无硬件流控。
- [ ] PC 能发送 CONFIGURE/START/STOP/STATUS/ABORT，错误 CRC 或错误 sequence 会被拒绝。

## 4. 六通道映射

逐台只使能一台电机并核对：

| Index | 名称 | 期望反馈 CAN ID | 已识别 | 正方向正确 | 零位合理 | 线束标签一致 |
|---:|---|---:|---|---|---|---|
| 0 | L_front | `0x011` | [ ] | [ ] | [ ] | [ ] |
| 1 | L_rear | `0x012` | [ ] | [ ] | [ ] | [ ] |
| 2 | R_rear | `0x013` | [ ] | [ ] | [ ] | [ ] |
| 3 | R_front | `0x016` | [ ] | [ ] | [ ] | [ ] |
| 4 | L_wheel | `0x144` | [ ] | [ ] | [ ] | [ ] |
| 5 | R_wheel | `0x145` | [ ] | [ ] | [ ] | [ ] |

- [ ] 右腿顺序确认是 `R_rear` 后 `R_front`。
- [ ] 将实测 sign、zero、gear ratio 写回 `Config/pace_motor_manifest.h`。
- [ ] 重新计算 manifest hash、重新构建固件，并把 `calibration_verified` 改为 `true`。
- [ ] PC manifest parser 与新固件 header hash 一致。

## 5. 安全停机

- [ ] DM 零增益探测命令不产生可见冲击。
- [ ] LK 零转矩命令不产生持续转动。
- [ ] STOP 在 ARMED 和 RUNNING 状态均能撤销命令。
- [ ] ABORT 触发六通道安全命令并进入 FAULT。
- [ ] 断开任意一台电机反馈后，30 ms feedback timeout 能中止 session。
- [ ] 模拟 CAN passive/bus-off 后能够中止。
- [ ] 模拟 UART/capture overflow 后能够中止并在 footer 标记无效。
- [ ] 超速、超转矩/电流和超温阈值路径至少各做一次受控软件注入测试。
- [ ] 急停后不存在未标记为无效的拟合样本。

注意：当前默认不启用最终 RL 工作区间的硬位置限位。首次 commissioning 必须依靠架空、低幅值、人员急停以及速度/转矩/温度/超时边界，不得据此扩大动作范围。

## 6. 低幅值 Session

PC 命令示例：

```powershell
python wheel_leg_motor_PACE_PC/tools/capture_session.py `
  --port COM7 `
  --output data/commissioning_001.raw
```

- [ ] session header 声明 `COMMISSIONING_SESSION`。
- [ ] `STAGE_STATIC` 无非预期运动、异常噪声或温升。
- [ ] `STAGE_DM_FIT` 四台 DM 做同步机械对称的小幅 chirp。
- [ ] `STAGE_DM_VALIDATION` 与拟合轨迹的幅值、频率或相位不同。
- [ ] DM 实际 stage config 为 `Kp=20.0`、`Kd=0.6`、`dq_des=0`、`tau_ff=0`。
- [ ] `STAGE_LK_TORQUE` 两轮正反向响应方向正确。
- [ ] `STAGE_LK_VELOCITY` 两轮正反向响应方向正确。
- [ ] `STAGE_STOP` 平滑停止并产生 footer。
- [ ] 任一阶段出现冲击、机构接近奇异位形、线束拉扯、异常声响或温升时立即 ABORT。

## 7. UART 与 CAN 验收

- [ ] sample 固定 126 B，平均 500 Hz。
- [ ] status 为 64..128 B，平均 10 Hz。
- [ ] 持续编码流量不超过 64512 B/s；128 B status 的理论值应为 64280 B/s。
- [ ] capture ring 无 overflow，UART DMA 无 error。
- [ ] 静态阶段保守 CAN 负载约 15.6%。
- [ ] DM 阶段保守 CAN 负载约 57.2%。
- [ ] LK 阶段保守 CAN 负载约 36.4%。
- [ ] 任意配置超过 70% 时无法进入 ARMED。
- [ ] 活跃 500 Hz 通道最大命令周期小于 4 ms。
- [ ] enqueue 到 Tx Event 最大延迟小于 2 ms。
- [ ] Tx Event lost、unmatched event、queue full、HAL error 均为 0。

记录每个通道：

| 通道 | Tx count | Rx count | period P95/max us | Tx latency P95/max us | Rx age P95/max us |
|---|---:|---:|---|---|---|
| L_front | | | | | |
| L_rear | | | | | |
| R_rear | | | | | |
| R_front | | | | | |
| L_wheel | | | | | |
| R_wheel | | | | | |

## 8. PC 数据验收

```powershell
python wheel_leg_motor_PACE_PC/tools/decode_session.py data/commissioning_001.raw `
  --csv derived/commissioning_001.csv `
  --npz derived/commissioning_001.npz `
  --summary derived/commissioning_001.summary.json
```

- [ ] decoder 无 CRC、length、sequence、timestamp、config 或 footer 错误。
- [ ] footer: `overflow=0`、`dropped_frames=0`、`statistics_complete=1`。
- [ ] 六通道 raw integer 保留，单位转换后的方向与实机一致。
- [ ] sample 的命令全部来自 Tx-confirmed payload；无效/age 饱和样本被排除。
- [ ] raw 文件只读归档，派生文件写入其他目录。

临时拟合：

```powershell
python wheel_leg_motor_PACE_PC/tools/fit_session.py `
  data/commissioning_001.raw `
  models/commissioning_001.actuator.json
```

- [ ] 输出 `model_maturity = provisional`。
- [ ] 四个 DM 都有 fit 与独立 DM validation 指标。
- [ ] 两个 LK 都有独立 torque/velocity 子模型。
- [ ] LK commissioning 报告明确标记“无独立 holdout”，不冒充最终验证。
- [ ] Isaac 与 MuJoCo adapter 读取同一 manifest 且顺序完全一致。

## 9. Commissioning 结论

- [ ] **PASS**：全部安全、时序、完整性和临时拟合项目通过，可以进入接口冻结讨论。
- [ ] **FAIL**：记录失败项、保留 raw 和 build artifact，修复后重新从低幅值开始。

结论/异常记录：

```text

```
