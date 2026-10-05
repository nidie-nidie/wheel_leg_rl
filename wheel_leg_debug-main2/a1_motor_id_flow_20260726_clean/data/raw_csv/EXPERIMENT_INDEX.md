# A1 chirp 实机采集数据索引

## 文件命名约定

```text
A1chirp_<time>_f<start>-f<end>Hz_d<chirp>s_r<ramp>s_Kp<jointKp>_Kd<jointKd>_ampHip<hipAmp>_ampTh<thighAmp>_ampCalf<calfAmp>_cH<hipCenterAbs>_cFT<frontThighCenter>_cRT<rearThighCenter>_cCalf<calfCenter>_v<maxVel>_a<maxAccel>.csv
```

关键字段：

- `Kp` / `Kd`: 本次采集中 12 个关节统一使用的关节 PD 参数。
- `ampHip`: 4 个 hip 关节的 chirp 幅度绝对值，单位 rad。
- `ampTh`: 4 个 thigh 关节的 chirp 幅度绝对值，单位 rad。
- `ampCalf`: 4 个 calf 关节的 chirp 幅度绝对值，单位 rad。
- `cH`: hip center 的绝对值；程序内部按左右腿自动赋正负号。
- `cFT`: front thigh center。
- `cRT`: rear thigh center。
- `cCalf`: calf center。
- `v`: `--max-vel` 轨迹预检上限，单位 rad/s。
- `a`: `--max-accel` 轨迹预检上限，单位 rad/s²。

说明：文件名中没有把 12 个关节逐个展开，因为同一类关节幅度相同。每个关节实际幅度映射见下一节。

## 关节幅度映射

Unitree raw joint order：

```text
FR_hip, FR_thigh, FR_calf,
FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf,
RL_hip, RL_thigh, RL_calf
```

每条实验使用的 12 关节幅度：

| 文件时间 | FR_hip | FR_thigh | FR_calf | FL_hip | FL_thigh | FL_calf | RR_hip | RR_thigh | RR_calf | RL_hip | RL_thigh | RL_calf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260726_061600` | 0.03 | 0.05 | 0.05 | 0.03 | 0.05 | 0.05 | 0.03 | 0.05 | 0.05 | 0.03 | 0.05 | 0.05 |
| `20260726_062823` | 0.20 | 0.30 | 0.30 | 0.20 | 0.30 | 0.30 | 0.20 | 0.30 | 0.30 | 0.20 | 0.30 | 0.30 |
| `20260726_063144` | 0.20 | 0.30 | 0.30 | 0.20 | 0.30 | 0.30 | 0.20 | 0.30 | 0.30 | 0.20 | 0.30 | 0.30 |
| `20260726_064152` | 0.20 | 0.35 | 0.35 | 0.20 | 0.35 | 0.35 | 0.20 | 0.35 | 0.35 | 0.20 | 0.35 | 0.35 |
| `20260726_071554` | 0.20 | 0.38 | 0.38 | 0.20 | 0.38 | 0.38 | 0.20 | 0.38 | 0.38 | 0.20 | 0.38 | 0.38 |
| `20260726_071950` | 0.22 | 0.38 | 0.38 | 0.22 | 0.38 | 0.38 | 0.22 | 0.38 | 0.38 | 0.22 | 0.38 | 0.38 |

单位均为 rad。CSV 中实际发送的位置指令仍以每个时刻的 `q_des_*` 列为准。

## 数据文件

| 本地文件名 | 远端原始路径 | 参数摘要 |
| --- | --- | --- |
| `A1chirp_20260726_061600_f0p10-f1p50Hz_d8_r2_Kp20_Kd2_ampHip0p03_ampTh0p05_ampCalf0p05_cH0p05_cFT0p60_cRT0p90_cCalf-1p55_v20_a250.csv` | `/home/unitree/a1_chirp_logs/a1_joint_chirp_20260726_061600_small.csv` | `Kp=20`, `Kd=2`, `hip/thigh/calf amp=0.03/0.05/0.05`, `0.1→1.5Hz`, chirp `8s`, ramp `2s` |
| `A1chirp_20260726_062823_f0p10-f1p4Hz_d20_r2_Kp25_Kd2_ampHip0p20_ampTh0p30_ampCalf0p30_cH0p10_cFT0p80_cRT1p00_cCalf-1p50_v20_a250.csv` | `/home/unitree/a1_chirp_logs/a1_joint_chirp_20260726_062823_wide_4hz.csv` | `Kp=25`, `Kd=2`, `hip/thigh/calf amp=0.20/0.30/0.30`, `0.1→4Hz`, chirp `20s`, ramp `2s` |
| `A1chirp_20260726_063144_f0p10-f1p8Hz_d20_r2_Kp25_Kd2_ampHip0p20_ampTh0p30_ampCalf0p30_cH0p10_cFT0p80_cRT1p00_cCalf-1p50_v20_a800.csv` | `/home/unitree/a1_chirp_logs/a1_joint_chirp_20260726_063144_wide_8hz_accel800.csv` | `Kp=25`, `Kd=2`, `hip/thigh/calf amp=0.20/0.30/0.30`, `0.1→8Hz`, `max-accel=800` |
| `A1chirp_20260726_064152_f0p10-f1p10Hz_d20_r2_Kp25_Kd2_ampHip0p20_ampTh0p35_ampCalf0p35_cH0p10_cFT0p80_cRT1p00_cCalf-1p50_v25_a1600.csv` | `/home/unitree/a1_chirp_logs/a1_joint_chirp_20260726_064152_wide_10hz_tc035_vel25_accel1600.csv` | `Kp=25`, `Kd=2`, `hip/thigh/calf amp=0.20/0.35/0.35`, `0.1→10Hz`, `max-vel=25`, `max-accel=1600` |
| `A1chirp_20260726_071554_f0p10-f1p10Hz_d20_r2_Kp25_Kd2_ampHip0p20_ampTh0p38_ampCalf0p38_cH0p10_cFT0p80_cRT1p00_cCalf-1p50_v25_a1600.csv` | `/home/unitree/a1_chirp_logs/a1_joint_chirp_20260726_071554_wide_10hz_tc038_vel25_accel1600.csv` | `Kp=25`, `Kd=2`, `hip/thigh/calf amp=0.20/0.38/0.38`, `0.1→10Hz`, `max-vel=25`, `max-accel=1600` |
| `A1chirp_20260726_071950_f0p10-f1p10Hz_d20_r2_Kp25_Kd2_ampHip0p22_ampTh0p38_ampCalf0p38_cH0p10_cFT0p80_cRT1p00_cCalf-1p50_v25_a1600.csv` | `/home/unitree/a1_chirp_logs/a1_joint_chirp_20260726_071950_wide_10hz_h022_tc038_vel25_accel1600.csv` | `Kp=25`, `Kd=2`, `hip/thigh/calf amp=0.22/0.38/0.38`, `0.1→10Hz`, `max-vel=25`, `max-accel=1600` |

## 采集完整性摘要

| 时间 | chirp 频率 | chirp 时长 | ramp | 样本行数，不含表头 | 通信状态 |
| --- | --- | ---: | ---: | ---: | --- |
| `20260726_061600` | 0.1→1.5Hz | 8s | 2s | 5000 | `send_status=610`, `receive_status=0` |
| `20260726_062823` | 0.1→4Hz | 20s | 2s | 11000 | `send_status=610`, `receive_status=0` |
| `20260726_063144` | 0.1→8Hz | 20s | 2s | 11000 | `send_status=610`, `receive_status=0` |
| `20260726_064152` | 0.1→10Hz | 20s | 2s | 11000 | `send_status=610`, `receive_status=0` |
| `20260726_071554` | 0.1→10Hz | 20s | 2s | 11000 | `send_status=610`, `receive_status=0` |
| `20260726_071950` | 0.1→10Hz | 20s | 2s | 11000 | `send_status=610`, `receive_status=0` |

远端 A1 上的原始 CSV 暂未重命名，以保留采集现场原始记录。
