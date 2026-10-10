# wheel_leg_motor_PACE_PC

PC 端工具负责保存原始串口流、严格解码、数据归一化、DM/LK 分模型辨识，以及向 Isaac 和 MuJoCo 提供同一份六通道 actuator manifest。

## 依赖

基础链路：

```text
Python 3.8+
numpy
scipy
pyserial
jsonschema
```

安装基础依赖：

```powershell
python -m pip install -r requirements.txt
```

可选依赖：

- `torch`：导出 PACE `.pt` 数据和运行 Isaac/PACE；
- `cmaes`、Isaac Sim、Isaac Lab、`pace_sim2real`：正式 CMA-ES 回放优化；

当前电脑没有 `torch` 和 `cmaes`，但原始数据、CSV/NPZ、确定性初始拟合和适配层测试可直接运行。

## 一次工程试采

```powershell
python tools/capture_session.py `
  --port COM7 `
  --output data/commissioning_001.raw
```

脚本先发送 `CONFIGURE(COMMISSIONING)`，再发送 `START`，并将 UART 字节原样写入 `.raw`。不要把终端日志混入该文件。

按需提前停止：

```powershell
python tools/capture_session.py --port COM7 --output data/short.raw --stop-after 15
```

`--session-type final` 已预留，但当前固件合同锁会拒绝正式会话。

## 解码与归一化

```powershell
python tools/decode_session.py data/commissioning_001.raw `
  --csv derived/commissioning_001.csv `
  --npz derived/commissioning_001.npz `
  --summary derived/commissioning_001.summary.json
```

解码器会检查：

- magic、版本、长度和 CRC；
- 全流 sequence 的丢失、重复和乱序；
- 32 bit 时间回绕与 2 ms 采样间隔；
- `config_seq` 与 stage config 关联；
- session header、stage config 与 footer 的配置哈希一致；
- Tx/Rx age、饱和和有效位；
- status 中的 CAN 入队失败、Tx Event 丢失、CAN 错误与总线故障状态；
- footer overflow、drop 和统计完整性；
- 固件 motor manifest 版本及哈希。

raw 二进制文件是唯一源数据，CSV/NPZ/PT 都是可重建产物。

## 六通道拟合

```powershell
python tools/fit_session.py data/commissioning_001.raw `
  models/commissioning_001.actuator.json `
  --normalized-npz derived/commissioning_001.npz
```

输出遵循 `models/motor_model_schema.json`：

- DM8009：PACE 参数族 `armature + viscous friction + static/dynamic friction + encoder bias + global delay`；
- LK9025：转矩模式的惯量/摩擦模型和速度模式的一阶闭环模型；
- commissioning 输出固定标记为 `model_maturity = provisional`；
- final session 才能输出 `model_maturity = final`，且轮毂两种模式必须各有独立 holdout 数据。

本地 DM 拟合器是确定性的有效转矩平衡 initializer，主要用于打通工程链路和给正式优化提供初值。正式 Isaac 模型使用 `fit/pace_optimizer_bridge.py` 导出的 PACE 数据合同，并复用参考包中的 `pace_sim2real.CMAESOptimizer`。

导出正式 PACE 输入：

```python
from fit.pace_optimizer_bridge import export_pace_dataset
export_pace_dataset(dataset, "derived/dm_fit.pt")
```

## Isaac 与 MuJoCo

Isaac 合同：

```powershell
python -m isaac.leg_dm_adapter models/commissioning_001.actuator.json
python -m isaac.wheel_lk_adapter models/commissioning_001.actuator.json
```

MuJoCo 参数和有状态延迟执行器：

```powershell
python mujoco/actuator_replay.py models/commissioning_001.actuator.json
```

`mujoco/actuator_replay.py` 中的 `ActuatorReplay` 先计算带编码器偏置的 DM MIT-PD，再延迟并限幅实际转矩；LK 的转矩和速度模式分别处理。`joint_parameter_patch()` 同时返回 MuJoCo 的 armature、damping 和 frictionloss，避免继续使用理想转矩直通模型。

## 测试

```powershell
python tests/test_commands.py
python tests/test_decoder.py
python tests/test_fitting.py
```

合成拟合测试使用已知六通道参数，检查参数与延迟恢复，并确认两个仿真适配器不重排通道。
