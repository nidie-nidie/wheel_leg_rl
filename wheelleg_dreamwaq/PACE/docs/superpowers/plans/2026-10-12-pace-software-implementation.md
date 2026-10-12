# PACE 可信采集与 DreamWaQ 对齐 Implementation Plan

> **For agentic workers:** Execute the owned tasks in parallel, review each boundary, and run the native C / PC integration suite before reporting completion. Steps use checkbox syntax for tracking.

**Goal:** 完成用户选择的方案2的采集软件：真实C验证、固件正确性修复、分段独立实验、PC严格录制/解码/拟合门禁和可追溯来源，让用户按交付步骤低幅值烧录试采。

**Architecture:** 原生产C源码由STM32与host_debug共同编译，HAL/RTOS只在测试边界替换。固件在线输出V2，PC保留V1诊断读取；稳定sample格式与低频扩展stage绑定实际控制、分段与中心。采集来源和标定快照保存在raw sidecar；错误/不完整会话不能导出正式模型。

**Tech Stack:** STM32H723 HAL + FreeRTOS C、MSVC/GCC native C、Python unittest + numpy/scipy/pyserial/jsonschema。

## 授权与范围

用户已批准设计方案2及host_debug位置，要求软件工作推进到能低幅值试采。用户确认现有电机方向准确：保留现有manifest方向与硬件ID，记录来源user_confirmed_existing_code；轴方向、激励正负号、仿真轴符号分别处理，零位与工作范围不冒充已实测。

当前交付覆盖采集、解码、质量检查和现有provisional initializer；不把缺少实测最终合同的数据标为final，不在本轮改动正在训练的DreamWaQ源文件，不宣称已经获得真实执行器参数。完整机器人优化与实测模型训练接入依赖用户采得的数据。

## 冻结边界

### V2线上协议

- common header version=2；PC显式接受V1/V2；sample126、status64..128、session header58、event30、footer34保持长度。
- V2 stage长度128；原字段offset10..71保持。
- offset72 role u8：0 DIAGNOSTIC，1 FIT，2 VALIDATION，3 STATIC，4 STOP。
- offset73 group/active mask u8。
- offset74 segment_id u16，会话内唯一；offset76 action_update_rate_hz u16。
- offset78 command_hold_method u8：1 zero-order hold；offset79 center_source u8：1 initial_feedback，2 verified_contract。
- offset80 phase_rad float32；offset84..99 dm_center_rad[4] float32；offset100..123 excitation_multiplier[6] float32。
- offset124..127 CRC32。所有数值显式little-endian，不用结构体内存序列化。
- session header reserved13=timebase：1 TIM5_ISR_SERVICE；reserved32..33=control_rate_hz500。
- config hash绑定显式序列化recipe内容/所有stage/固定control和采样周期及控制语义；runtime测量center通过stage+raw SHA256绑定，不混入开始前未知的recipe hash。
- Tx/Rx timestamp为TIM5在HAL回调消费事件时采集的服务时间，明确不代表电机执行时刻；移除错误的FDCAN/TIM5高位拼接。精细硬件event trace作为后续功能，当前不以虚假精度宣传延迟。

### 实验

stage类别仍为1..6；新增stage_count和最大32段。commissioning完整26段：static1；腿FIT4组×2幅(.075/.15)=8，腿VALIDATION4组×1幅(.12)=4；轮torque两轮×2幅(.05/.10)=4 FIT+两轮各1(.075) VALIDATION；轮velocity两轮×2幅(2/4)=4 FIT+两轮各1(3) VALIDATION；STOP1。每段保持50Hz新目标，CAN当前主动500Hz/保持100Hz调度，sample250Hz。组定义与所有multipliers线上可追溯；保留原腿基线方向[+1,-1,-1,+1]，第二双侧组改变明确激励配方、不改轴映射。

低幅值Kp20/Kd.6不变。正式配置函数必须真正使用final字段并校验完整性，ready=0时拒绝；不允许default commissioning仅换hash冒充正式stage。正式参数只能由已验收实机合同生成。

## Task 1：真实C平台（owner pace_search）

Files: Create `PACE/host_debug/build.py`、`run_checks.py`、`README.md`、`fakes/{main,cmsis_os,fdcan,tim,usart}.h`、`fakes/host_hal.c/.h`、`tests/test_pace_host.c`；编译原Board/Motor/PACE/Protocol源文件，不复制实现。

- [x] 构建器通过vswhere/VC工具链发现MSVC，使用vcvarsall环境，所有产物在host_debug/build；GCC在其他系统可用。
- [x] HAL fake区分CAN入队、TxEvent确认、RxFIFO反馈，按真实callback驱动registry。fake TIM5/6与UART只替代边界。
- [x] 实际应用共享接口：
```c
void pace_app_poll(void);
void pace_app_transport_poll(void);
```
两个RTOS task只是循环调用上述函数加osDelay，测试直接调用同一单步。
- [x] 用原C验证初始状态、26段normal complete、50Hz保持、阶段/recipe hash、short CAN拒绝、无Tx-confirm不发布确认命令、STOP/FAULT/队列/时间回绕/未来时间age/tick积压。
- [x] 提供fixture CLI生成V2 raw，通过真实C encoder与transport发送，PC直接消费这个文件。
- [x] 完成`python host_debug/run_checks.py`，输出compiler/version/native断言与fixture路径；失败返回非零。

## Task 2：固件experiment/protocol（owner workspace_scan）

Files: Modify `wheel_leg_motor_PACE/PACE/Inc/pace_experiment.h`、`PACE/Src/pace_experiment.c`、`PACE/Src/pace_excitation.c`、`Config/pace_experiment_config.h`、`Protocol/Inc/pace_frame.h`、`Protocol/Src/pace_frame.c`；相关原Python静态检查更新为实际V2契约。

- [x] stage结构支持role、mask、segment、action_rate、hold、phase、center_source、6 multipliers；配置含实际stage_count。
- [x] 保持目标时钟与CAN发送时钟分别计算：
```c
const uint32_t update_us = 1000000U / stage->action_update_rate_hz;
const uint32_t held_elapsed_us = (elapsed_us / update_us) * update_us;
```
激励基于held_elapsed_us，发送mask仍按各电机CAN周期；禁止处理早于stage_start的历史时间使无符号下溢跳段。
- [x] 实现26段recipe和独立holdout；进入begin采真实中心并填入每stage帧。
- [x] 内容hash用显式LE字段CRC32，不hash带padding的结构体；stage配置的hash不随runtime中心漂移。
- [x] `pace_experiment_default_config`仅创建commissioning；另`pace_experiment_final_config`返回bool且完整消费正式合同，未ready返回false。
- [x] V2编码严格按offset表与128 slot预算；返回长度与CRC由native C验证。
- [x] 改完向PC owner传达实际C字段名和枚举；不得静默变更冻结offset。

## Task 3：应用/Board正确性（owner root）

Files: Modify `wheel_leg_motor_PACE/PACE/Src/pace_app.c`、`PACE/Inc/pace_app.h`、`Board/Src/pace_can_port.c`、`Board/Inc/pace_can_port.h`、`Board/Src/pace_time.c`、`Board/Src/pace_uart_port.c`、`Motor/Src/pace_motor_registry.c`。

- [x] snapshot临界区内先复制registry并取当前TIM5时间，离开后仅刷新副本age；latest数据不配旧tick时间。poll消费积压时只用最新tick触发一次当前时刻控制，不能追赶突发发送。
- [x] age模块对跨回绕正向差值正常计算，对确实未来timestamp拒绝有效年龄，不能变成巨大正数；native regression给定100000/100300证明不假offline。
- [x] CAN只接受standard/classic/data/8B已知ID，并校验LK响应opcode；新增RX lost/full/invalid统计和通知，lost触发session不完整。
- [x] 采用单一TIM5 ISR service timestamp；不再把FDCAN epoch假设为TIM5低16位。
- [x] safe commands拆分pending/inflight，入队仅转inflight，确认matching disable/stop且新tx_count才清除；所有safe确认后才发完整footer。超时改为明确stop-unconfirmed故障footer而非成功。
- [x] terminal frames各自检查剩余槽位；ordinary帧预留两个terminal槽，不因合法重试污染overflow统计。
- [x] ARMED要求完整enable次数、最新online反馈、使能CAN待确认完成；probe时序明确。
- [x] 初始化全部会话/自启动状态；autostart10s与enablepair重复为poll非阻塞阶段，原顺序/1ms间隔不变。
- [x] RX错误在task上下文恢复DMA或显式fault；不在HAL callback中隐式调用无限重试。
- [x] 共享poll API与RTOS wrappers原功能一致，host command模式通过compile-time override测试。

## Task 4：PC scanner/capture/metadata/quality（owner failure_review）

Files: Create `wheel_leg_motor_PACE_PC/pace_raw/scanner.py`、`quality.py`、`session_metadata.py`；Modify `schema.py`、`decoder.py`、`normalize.py`、`tools/capture_session.py`、`tools/decode_session.py`、`tools/fit_session.py`及tests。

- [x] shared scanner按version/type/length提前拒绝不可能长度，CRC失败逐字节resync；incremental和offline一致。
- [x] 零rate/非法mode/重复header/未关联stage/footer后frame或bytes标issue而不崩溃；closed会话任何尾随数据均不进入fit。
- [x] capture默认passive等待>=15s首数据，不清输入buffer；active显式选择；写原始bytes与确定性sidecar、超时/中断状态及raw SHA256；不会覆盖已有文件。
- [x] V2字段解码、segment/role/centers/hold进入normalize；温度与status做有age的因果关联，不伪造同步温度。
- [x] metadata保存校准固定快照与现有方向user confirmation、fixture、期望/线上recipe、raw/source hash；V1诊断保留但来源不确定不能晋升final。
- [x] `require_fit_quality(dataset)`作为所有fit/export入口门禁，不仅CLI；footer完整、无issue/hash一致、合法有效段及独立holdout。

## Task 5：分段拟合与模型来源（owner failure_review）

Files: Modify `fit/common.py`、`fit_leg_dm.py`、`fit_wheel_lk.py`、`model_manifest.py`、`pace_optimizer_bridge.py`、`models/motor_model_schema.json`、必要isaac/mujoco独立adapter。

- [x] 按family/role取segment列表，各段独立gradient/延迟，仅合法regression行联合优化；不能跨段求导或跨段填延迟。
- [x] baseline来源取真实stage Kp/Kd，检查mode/dq_des/tau_ff与stage一致；保留旧20/.6 initializer兼容。
- [x] 新commissioning两种LK均有holdout，保持已有dual模式依赖显式，不把无独立验证torque升级final。以velocity响应为主要报告。
- [x] delay输出秒与语义，target-delay与effort-delay不混淆；独立adapter必须验证调用dt，无法一致时拒绝运行而非静默改变延迟。
- [x] wheel模型合并校验source_session/config/calibration/sample_period；禁止仅换channels保留另一来源身份。
- [x] 正式PACE export按连续合法segment输出，mask/age/saturation门禁一致，不导出坏样本。
- [x] 利用已知合成参数+独立holdout验证拟合，实际C fixture只验证采集/解码质量，不声称假反馈能验证物理辨识。

## Task 6：交付集成（owner root）

Files: Create `PACE/README.md`、`docs/software-release.md`、`tools/setup_pc.ps1`、`tools/verify_software.py`、`tools/prepare_session.py`；更新firmware/PC README与commissioning checklist实际250Hz/V2。

- [x] 独立PACE Python环境安装依赖，不修改DreamWaQ环境。
- [x] Windows native全部测试 + PC unit + C raw→FakeSerial→capture→decoder→quality门禁集成，统一命令返回明确结果；追加packet切碎/CRC/缺头缺尾/footer后数据/10s启动/中断反例。
- [x] 查找现成Keil/ARM compiler，完成可用目标build；如无法生成HEX，交付实际可用Keil工程和准确未验证项，不能称已有可烧HEX。
- [x] 交付低幅值autostart/passive试采命令、COM选择、固定工况、停止方式、首次试采质量报告及后续final合同生成/拒绝路径。
- [x] 最后`git diff --check`、fresh tests、检查修改只在PACE，保护已有DreamWaQ修改。记录软件通过与仍需实板确认的不同结果。

## Review checkpoints

先native compile确认真实C源码能执行；再看端到端normal会话；每个已确认问题提供反例失败→修复通过证据；最后完整release验证。严格保留实机验证边界：逻辑/HAL fake验证不能证明真实IRQ延迟、电机机械停稳或驱动增益等价。

## 执行验收记录（2026-10-12）

最终统一命令 `.venv/Scripts/python.exe tools/verify_software.py --include-target` 返回0。Python检查44+16+11+7=78项通过；真实生产C passive/active 与各自AddressSanitizer全通过。正常fake HAL会话26段、52,750samples、54,915frames、6,785,770bytes；仅用于逻辑/协议，不代表实际动力学。

新GNU HEX已生成并独立核对当前源码输入与输出SHA256；startup/FreeRTOS强中断向量、Flash/RAM和AXI DMA地址验证通过。HEX SHA256为 `8a2a839058509004ae95345ff60fc18107dcf945d55e14182d42fba7f221ca1e`。检查报告在 `PACE/build/release/software_verification.json`，源码与产物绑定在 target build_manifest 与 native_report。

软件交付范围为可烧录的低幅值commissioning、PC录制/质量检查/分段初始拟合和真实C debug平台。final默认锁定，实测零位/机械范围/最终控制合同落实后再配置；本轮没有烧录、没有操作电机，没有实板时序/停止动作/栈水位结果，也没有把初始等效参数写入DreamWaQ。现有方向沿用用户确认正确的代码，不要求重做方向标定。
