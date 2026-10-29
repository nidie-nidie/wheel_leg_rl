# PACE PC 工具

完整试采步骤见 [PACE README](../README.md) 与 [软件交付说明](../docs/software-release.md)。本目录负责原始串口录制、V1/V2 解码、质量检查、分段初始拟合和独立仿真适配。

使用 PACE 独立 Python 3.11 环境，依赖固定于 `../requirements.lock.txt`，不修改 DreamWaQ 环境。以下命令在 PACE 目录运行。

```powershell
& .\.venv\Scripts\python.exe tools/prepare_session.py --output data/commissioning_001.plan.json --firmware-binary build/firmware_gcc/commissioning/wheel_leg_motor_PACE.hex
& .\.venv\Scripts\python.exe wheel_leg_motor_PACE_PC/tools/capture_session.py --port COM7 --output data/commissioning_001.raw --metadata-plan data/commissioning_001.plan.json
```

默认 passive：不清空串口 buffer、不发送 START，等待板子自启动。看到保存路径再复位板子。主动模式需固件 `PACE_ONE_WAY_AUTOSTART=0` 且 RX 连线可用，再显式加 `--active`；默认 HEX 为单向自启动模式，PC 不能使它停止。

保存三类结果：原始 `.raw`、`.session_manifest.json` 固定快照和 `.quality.json` 报告。录制中断/超时也保留原始来源，并标记不完整；同名文件不覆盖。metadata plan 中 `firmware_binary={path,sha256}` 会在采集时重新核验文件；缺少新 build 时不编造二进制身份。

```powershell
& .\.venv\Scripts\python.exe wheel_leg_motor_PACE_PC/tools/decode_session.py data/commissioning_001.raw --summary data/commissioning_001.summary.json --csv data/commissioning_001.csv --npz data/commissioning_001.npz
& .\.venv\Scripts\python.exe wheel_leg_motor_PACE_PC/tools/fit_session.py data/commissioning_001.raw data/commissioning_001.actuator.json
```

增量录制与离线解码共用 scanner。CRC/长度/version/sequence/阶段上下文/末尾多余数据等问题保留用于诊断；V1 可以读，但缺失 V2 合同不能作为已验证拟合输入。库入口与 CLI 共用 `require_fit_quality`，不允许绕过快照、fault、有效性及独立验证段要求。

分段求导和延迟，每个分段保留 role、group、segment、中心、50 Hz 保持方式。温度取过去最近一条状态消息并附 age。时戳是 TIM5 回调服务时刻，不能将其称为电机执行时刻。

DM 与 LK 使用不同初始模型；commissioning 输出仅为 provisional。装机闭链拟合的等效惯量不能直接当作 motor-only armature 写入已有资产。DreamWaQ 当前训练代码尚未加载该模型；本轮只保存实际控制基线与按关节名字的映射，后续实际模型接入需独立验证。

MuJoCo replay 必须显式给 `runtime_period_s`，Isaac 参数/目标适配也检查延迟语义，避免采样周期被误用为 physics 周期。没有实测 final 合同、零位和工作范围时，不输出正式模型。

统一验证：

```powershell
& .\.venv\Scripts\python.exe tools/verify_software.py --include-target
```

测试包括真实 C passive/active 完整 raw 经碎片 FakeSerial 录制，以及坏 CRC、假 magic、内嵌帧、缺头缺尾、末尾多余数据、启动等待、source tamper、fault 和延迟有效性反例。合成动力学数据仅验证拟合器能否恢复已知参数，不代表实际电机参数。
