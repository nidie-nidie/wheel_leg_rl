# 键盘调试速度上限 Implementation Plan

**Goal:** 用户要求提高控制速度上限；键盘窗口默认速度目标从 ±1.5 提高到 ±2.0 m/s。

**Architecture:** 仅修改独立调试入口 `play_keyboard.py` 和本目录说明。新增 `--max-speed` 正有限数参数，默认 2.0。沿用导出策略的 `vx_max_abs=1.5` 换算比例，当前观测索引 6 使用 `command[0] / vx_max_abs`，允许此次实验超过 ±1；reset 重复帧和 step 后新帧使用同一值。正式 observation builder、runtime、manifest、训练命令范围和物理参数不改。

**Tech Stack:** Python、NumPy、TorchScript、MuJoCo、既有 GLFW 显示入口。

- [x] `play_keyboard.py`：增加 `--max-speed` 参数和有限正数校验；替换键盘 vx 裁剪范围；添加本地 reset 包装同步全部初始 history 帧；动作前当前帧和动作后最新 history 帧都写入原比例换算后的速度目标；窗口和 status 显示可控范围。
- [x] 扩展已有 smoke：分别按 W/S 达到正负上限并 reset/tick，确认所有 history 速度命令均为 ±`max_speed/1.5`，不被第二道裁剪吞掉；保留低速、命令往返、裁剪、reset、暂停和退出检查。以默认 2.0 和显式 1.5 分别运行，确认负数参数被拒绝。
- [x] 更新 `README.md`，注明 2.0 超过训练范围、`--max-speed 1.5` 可恢复原范围。运行原 101 文件身份检查。记录检查输出和源码哈希，启动专用窗口核对实际命令范围与就绪状态。

本次修改只提高策略收到的速度目标，不声称策略实际达到 2.0 m/s。0 到 ±1.5 范围内的换算、动作、history 时序和物理求解保持原语义。

后续用户明确要求将键盘目标上限提高到 ±10 m/s。沿用上述实现，仅将默认 `--max-speed` 改为 10.0 并更新 README；正负目标在网络中的归一化值约为 ±6.667。重新运行既有 smoke 与正式文件身份检查，启动新窗口；验证记录单独保存在 `speed-limit-10-verification.json`，此前 2.0 的记录保留。
