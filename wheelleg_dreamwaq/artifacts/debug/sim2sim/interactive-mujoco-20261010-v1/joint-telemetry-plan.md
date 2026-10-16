# MuJoCo 六关节抖动记录与诊断计划（2026-10-11）

用户已授权：补齐刚才缺失的六关节输出和控制指令，重新打开同一 1401 轮策略，复现左腿抖动。不训练，不修复未知原因，不改变动作、模型、PD、随机化、history、命令时序和速度范围。

## 范围及职责

- 修改 artifacts/debug/sim2sim/interactive-mujoco-20261010-v1/play_keyboard.py：增加 --joint-telemetry 开关、调用记录器、记录键盘事件和复位、M 症状标记；原来的 trajectory.csv 继续输出。
- 新增同目录 joint_telemetry.py：观察现有 controller.compute_torque，不改变返回值；在下一次调用之前读上一个物理步的结果，最后一步在 runtime.step 返回后补齐。每次原始 compute_torque 仍调用一次，原始 mj_step 不插入额外调用。
- 新增同目录 verify_joint_telemetry.py：比较有/无记录器的相同策略轨迹，包含停车、前进、后退、转向及复位，验证完整状态、history 和 torque 完全一致；验证每个控制帧恰好有 20 行物理数据。
- 新增同目录 analyze_joint_telemetry.py：从日志计算左右腿动作跳变、速度波动、目标误差和力矩饱和，绘图；指标用于定位，不能单独宣称根因。
- 会话生成 telemetry_manifest.json、control.csv、physics.csv、events.jsonl，保留旧会话记录。

## 记录契约

canonical 顺序：jIJ 左前、jIO 左后、jAB 右前、jAG 右后、jwheel_left 左轮、jwheel_right 右轮。
所有六关节状态/目标/力矩字段使用 MuJoCo 坐标；右轮与策略控制坐标的符号为 -1，manifest 明确说明。
control.csv：episode/control 索引、wall 时间、模拟前后时间、三项用户命令、125 维实际策略输入、六维原始/裁剪动作、目标、实际前后位置/速度、最后物理步的施加力矩。
physics.csv：每 1 ms 的前后位置/速度、该步目标、PD/轮速度控制的未裁剪与裁剪力矩、实际 actuator force、qfrc_actuator/passive/constraint/bias/applied、qacc。
MuJoCo force/acceleration 数据是该次 mj_step 在积分前计算的值，时间对应 time_before_s；积分后的 qpos/qvel 对应 time_after_s。constraint 包含接触和闭链，不能称为单独接触力。
events.jsonl：初始化/复位、所有按键（包括暂停时）、自动倒下暂停、M 抖动标记、关闭；记录 wall 时间和 episode/control 编号。R 保持当前命令，这是既有行为。
仅加日志开关，默认旧路径不挂记录器；用户诊断窗口启用记录器。

## 实施和验收

- [x] 实现纯观察记录器并接入窗口。
- [x] 固定同一策略、命令、复位过程比较启用/禁用的 qpos、qvel、qacc、ctrl、force、history，最大误差必须为 0。
- [x] 验证六关节映射、125 维输入、20 个物理子步、复位分段、日志无缺字段/非有限值。
- [x] 开启诊断窗口，确认实际策略身份、窗口存活、三份日志增长，用户复现抖动时可按 M 标记。
- [x] 分析已收集数据，只给有证据的定位结果，不能把自动停车基线说成用户复现。
