# PACE 与当前 DreamWaQ 的采集对齐设计草案

日期：2026-10-12。
状态：代码对照已完成；本文是新增采集范围的设计草案，不代表固件已经实现、实机增益已经验收或模型已经接入训练。

用户已确认：电机安装在整机上，机身固定，轮子悬空。此前确认的 C 主机调试位置为 `PACE/host_debug/`，真实 C 实现与串口模拟方法继续沿用。

## 1. 要解决的问题

当前 PACE 适合作为低幅值 commissioning 链路，不能直接称为当前 DreamWaQ 的最终执行器辨识数据。两端不仅有正确性问题，还没有共同的控制、时间、通道和实验来源契约。采集时长增加不能解决这些差异。

本轮目标是把应改内容及验证边界具体化：先修数据可信度和建立真实 C / PC 回归基础，再丰富试验工况，最后将已验证的模型显式接入 DreamWaQ。训练算法、奖励和执行器接口接入是不同工作范围，不能通过修改一份模型 JSON 宣称训练已经使用了 PACE。

## 2. 当前源码事实

以下路径相对 `wheelleg_dreamwaq/`，数值来自当前工作树，而非旧架构文档。

| 项目 | DreamWaQ 当前实现 | PACE 当前实现 | 需要对齐的内容 |
|---|---|---|---|
| 腿部接口 | `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/control.py:27`，q_reference + 0.35 × clipped_action；绝对目标裁剪到 [-1, 1] rad | `PACE/wheel_leg_motor_PACE/PACE/Src/pace_experiment.c:304`，围绕启动时反馈中心作 0.15 rad chirp | 坐标标定、参考姿态、幅值包络及裁剪语义 |
| 腿部增益 | `source/wheelleg_dreamwaq/wheelleg_dreamwaq/assets/wheelleg.py:98`，名义 Kp=120、Kd=4，effort=18 Nm | `PACE/wheel_leg_motor_PACE/Config/pace_experiment_config.h:22`，Kp=20、Kd=0.6；PC initializer 强制该 baseline | 已验收的实机增益作为正式合同来源；不能直接把仿真增益写入自动运行配置 |
| 轮部接口 | `control.py:28`，速度目标最大 ±25 rad/s；仿真用 Kd=0.6、effort=9 Nm 的速度误差反馈 | LK A2 速度模式试采幅值 4 rad/s，另有 A1 转矩诊断 | 以速度模式为主要拟合对象；实机 A2 内部速度环不能等同仿真的 Kd=0.6 |
| 策略时间 | `schemas/physics.py:8`，physics 5 ms，decimation=4，动作更新 20 ms，即 50 Hz | firmware tick/调度 2 ms，即 500 Hz；快照 4 ms，即 250 Hz；目标随 chirp 每个调度时刻变化 | 动作更新周期、目标保持、CAN 重发与快照周期分别记录 |
| 通道顺序 | `schemas/action.py:8`，jIJ、jIO、jAB、jAG、左轮、右轮 | `Config/pace_motor_manifest.h:40`，jIJ、jIO、jAG、jAB、左轮、右轮 | 按 joint name 显式换序，不能按数组下标直接拷贝 |
| 坐标符号 | `schemas/action.py:22`，USD 右轮轴符号为 -1 | firmware manifest 所有 sign 暂为 +1、零位为 0、calibration_verified=false | 硬件轴→控制轴→仿真轴分开转换，保留校准来源 |
| 模型接入 | Isaac 使用 ImplicitActuator；MuJoCo `sim2sim/mujoco/wheelleg_mujoco/control.py:46` 使用名义 PD 与速度阻尼 | PC 内有独立适配器与 ActuatorReplay | 当前训练和正式 sim2sim 路径没有调用这些 PACE 适配器 |

DreamWaQ 训练入口还会启用 Kp/Kd/effort 约 ±5% 等随机化（`scripts/train_dreamwaq.py:239`、`schemas/randomization.py:292`）。这不是实测辨识结果。当前实现没有执行器延迟随机化字段。

## 3. 选择的方案

| 方案 | 收益 | 代价与限制 |
|---|---|---|
| 单纯延长旧 chirp、提高导出频率 | 改动少，可以多得到样本 | 无法解决增益、顺序、目标保持或轮 holdout 缺口；更高频率还需要验证吞吐 |
| **分阶段对齐控制合同，复用已有 raw，增加独立实验与少量诊断记录（推荐）** | 每项数据都有明确用途，可在真实 C 和 PC 链上验证，保留历史数据解释能力 | 固件与 PC 必须共同改造；模型接入需要后续独立验收 |
| 将 PACE、整机 IMU/接触遥测和策略部署一次合并 | 可做整机实机回放 | 当前执行器与录制正确性尚未解决，范围会同时跨越观测、传感器、接触模型和部署 |

采用第二项。保留固定机身、轮子悬空作为明确 load_case。它覆盖装机传动、腿部闭链和悬空轮响应；地面摩擦、接触、打滑与浮动机身动力学采用其他数据工况验证。

## 4. 数据：先利用已有字段，再增加有用途的信息

### 4.1 已有 raw 信息

`Protocol/Inc/pace_frame.h:44` 起，250 Hz sample 已有：

- DM：Tx-confirmed 原始位置/速度/前馈目标，反馈位置/速度/转矩，Tx/Rx age。
- LK：原始主命令、编码器/圈数、反馈速度/电流，Tx/Rx age。
- 每帧：stage/config、有效位、在线位、饱和与安全/CAN 标记。
- StageConfig：每电机模式与 CAN 命令频率、DM Kp/Kd、激励幅值/频率/持续时间。
- 10 Hz Status：DM MOS/rotor 温度、LK 温度以及通信统计。
- 会话 Header/Event/Footer：版本、配置 hash、故障和完整性标记。

PC 当前 initializer 并未充分使用这些字段。DM 回归主要用位置、速度、由速度差分得到的加速度和 PD 推算力矩；LK 主要用目标与反馈速度/加速度。首先将温度与状态按时间关联进派生数据，统一质量门禁，并在质量报告中显示反馈新鲜度、掉帧、饱和、温度和停机结果。温度关联保留原 status 时间和 age，不能假装是同步 250 Hz 温度采样。

当前 DM 回包没有独立的实测电流/母线电压；LK A1/A2 的 data[0] 被代码称为 status，但实际编码更像命令应答码，不能不经协议确认便解释为故障位。需要新传感器或新 CAN 查询的量单独设计，并计入总线预算。反馈转矩和电流换算转矩是电机报告/估计量，不等于外置转矩传感器实测。

### 4.2 第一批新增信息：会话元数据

每个 `.raw` 配套不可变的 `session_manifest.json`，保存完整内容与校验 hash，而不只保存容易被手工改动的常数 hash：

- raw 的 SHA-256、session ID、试验 ID、repeat ID、FIT/VALIDATION/DIAGNOSTIC 角色。
- 捕获的 firmware build/config、源代码/二进制身份及对应的原始线上配置。
- 控制合同：action/control/sample 周期、目标保持方式、每电机模式、实际量化 Kp/Kd、dq_des/tau_ff、命令与安全限幅、参考姿态、目标包络。
- 校准快照：电机/关节名、CAN ID、sign、zero、gear ratio、单位与 calibration_verified。
- fixture/load_case=`mounted_body_fixed_wheels_suspended`，固定方式、安装姿态、额外负载与装机结构版本/资产 hash。
- 轨迹配方：组别、幅值、频率/相位或 seed、持续时间、独立验证轨迹 ID。
- 所对照的 DreamWaQ 低层控制合同与资产内容 hash。绑定实际参数内容，不能只绑定 ActionV1 这个名字。

旧 v1 raw 缺少的来源标记为未验证，允许解码和故障诊断，不通过猜测填值升级为 final 模型。sidecar 中的计划值与线上实测/声明值分别保存；不匹配则拒绝正式拟合，不能通过 sidecar 覆盖原始线上值。

### 4.3 必须来自固件的时序信息

CAN Tx-confirm 只证明控制器完成发送，既不证明电机已收到，也不证明开始执行。响应估计中的执行器延迟必须说明观测边界，不能增加无法实测的 motor_execution_time 字段。PC sidecar 不能补造真实动作更新时间。部署匹配实验应在固件区分 50 Hz 新目标与既定 CAN 重发，在 20 ms 内保持同一个目标；同时保留单独的原生高频诊断轨迹，以区分电机动态和目标更新方式的影响。

为了判断重复快照和查毫秒级延迟，设计选通道、短时的 Tx/Rx trace，记录真实事件 timestamp、事件序号、通道、命令模式和原始 CAN payload。事件批量打包，不能每个 CAN 事件单发一个 UART 帧。完整格式在实际帧预算和 decoder 双端兼容测试后冻结；若新增线上帧或改变长度，使用显式协议版本/能力声明，不能把扩展 v1 silently 当成原 v1。

250 Hz latest-state 快照不能还原 4 ms 内的全部 500 Hz 发送/接收事件。trace 的用途是诊断和验证时序，不能把缺少 trace 的旧快照数据宣传成具有同等事件分辨率。

## 5. 实验：针对装机闭链与轮速度丰富覆盖

1. **静态与标定检查**：记录实际中心、各轴符号/零位、温度、反馈新鲜度和保持误差。仿真 nominal 必须经过实测坐标标定才成为硬件目标。
2. **腿部 FIT**：在机构可行范围内进行单侧/成对分组及不同相位的激励，记录所有四腿反馈。改变多个可区分的输入方向；当前四腿只共享一个 chirp 乘固定正负号，不能充分区分闭链交叉影响。
3. **腿部 VALIDATION**：保留现有独立验证思想，采用不同轨迹、幅值或相位，并覆盖保持和换向。验证轨迹不参与参数优化，不能随机拆同一条 chirp 作为独立 holdout。
4. **轮速度 FIT**：正反向、多幅值、近零低速和受控换向/停止，首先验证当前速度闭环 gain/time constant 和延迟；分别记录左右轮。近零数据用于评估死区/摩擦是否需要新模型，当前一阶拟合不会自动输出这些新参数。
5. **轮速度 VALIDATION**：新增独立轨迹。当前固件只有一个 LK velocity 段，PC 明确报告没有独立轮验证数据。
6. **轮转矩 DIAGNOSTIC**：保留已有低幅值 A1 段用于检查电流/转矩换算和响应。若之后正式采用转矩模型，同样需要独立 FIT/VALIDATION。
7. **重复性**：同一已验收配置进行多个完整重复会话，保留 repeat ID 和温度，检验参数与误差的稳定性。延长一条 chirp 不等于获得独立重复。
8. **部署匹配验证**：50 Hz 更新目标并保持 20 ms，覆盖静止、正转、停车、反转；随后可用固定来源的策略目标记录作为额外验证输入。策略回放记录其 clipping/限幅和轨迹来源。

幅值按已验收的实机包络分级；120/4、0.35 rad、25 rad/s 目前是仿真对照值，不是本次默认试采值。先用原 commissioning 配置完成链路与低幅值验证，再决定正式低层合同应采用哪些已验证增益、速度和限幅。

此次工况不要求凑齐策略的 IMU/机身观测；固定机身常量不能冒充传感器实测。后续整机部署观测另行接入真实 IMU/姿态估计。LK 可选低频母线电压与真实故障状态在本地厂家手册 State1 查询中有来源，但当前代码没有实现，必须先按应答 opcode 分发，避免把 State1 的电压/故障字错误解析为速度/电流。

正式腿模型必须用含现有闭链、重力与连杆惯量的装机仿真回放验证。当前 scalar initializer 的 effective inertia 可能包含负载和机构影响，不能未经区分直接累加到已有机器人资产的 motor armature；否则可能重复计入惯量。

## 6. 两端和第三处接入的职责

### 固件

- 先修时间域、历史 tick/新反馈快照、CAN 数据长度、使能/反馈新鲜度、停机确认与终止帧预留等已发现问题。
- 将共享应用单步逻辑与硬件端口分开，真实 C host harness 编译同一份状态机和编解码代码。
- commissioning 与 final 有独立实际配置；不能仅换 hash 就调用旧 commissioning 布局。
- stage 定义支持明确的实验角色、重复段/分组、目标保持及独立轮 holdout；不再把六种固定 stage 等同整个实验能力。
- 最终线上扩展只加入确有消费方且经预算验证的字段。

### PC

- 捕获模式和 10 s 自动启动保持一致；数据原样落盘并绑定 metadata/标定快照。
- decoder 和 capture 共用帧扫描规则；会话关闭后不再接受有效数据，错误会话不能由任何拟合入口绕过门禁。
- Kp/Kd 来自真实 stage/控制合同，替换现有固定 20/0.6 的假设；历史 baseline 的支持保留且明确标记。
- 不依赖固定 stage ID 推测新增段用途，FIT/VALIDATION/DIAGNOSTIC 明确可检验。保留 trial_id/segment_id/config_seq/工况边界，逐段计算导数与延迟历史，再联合优化兼容的段；不能将不连续或不同配置的行直接拼接为等间隔长序列。
- 温度、age、异常、覆盖率与 holdout 进入派生质量报告。
- 模型输出物理时间 `delay_s`、`delay_semantics` 与辨识时间分辨率，保留源 sample_period。当前腿 initializer 延迟的是目标位置，而独立 replay 延迟整个 PD effort；先统一语义，再验证两端回放，不能只把单位换成秒。旧 delay_steps 先换算成秒再按接收执行器周期处理。不能把 4 ms 的一步直接当成 5 ms 或 20 ms 的一步。
- 主模型以当前轮速度语义验证；LK 转矩诊断结果不自动当成具有独立验证的 final 模型。当前 fit_wheel_lk、model schema 和独立 replay 要求双模式，还借 torque 模型将 velocity 响应换算为 effort；第一阶段保留此显式依赖并标记每种模式的成熟度，后续若采用 velocity-only 模型，必须共同修改拟合器/schema/适配器并定义响应如何转为仿真作用力矩。

### DreamWaQ 后续接入

- 复用 scripts/export_dreamwaq_actor.py:68 的导出控制/资产/actuator/timing 内容生成对照合同。按 joint name 映射六通道，分别验证硬件→控制→Isaac/MuJoCo 的符号与零位。Isaac 物理为 5 ms，当前 MuJoCo 物理为 1 ms，两者动作均为 20 ms；delay 在各自物理时间轴上验证。
- 明确选择基于辨识的执行器实现，而不是仅导入 JSON 后仍走原名义 PD。
- 先完成固定机身装机回放、50 Hz 目标保持和物理延迟测试，再做训练/部署验收。
- 保留模型来源、资产 hash、控制合同和随机化范围关系。扩大随机化不能代替修正错误模型。

## 7. 采样与吞吐

当前正常流约 `250 × 126 + 10 × 64 = 32,140 B/s`，UART 921600 8N1 线速数据容量为 92,160 B/s，约占线速 34.9%，未计事件与阶段帧。

这不等于还有 65% 的真实发送余量。`PACE/Src/pace_transport.c:23` 使用 blocking UART；`pace_app_transport_task` 每帧再 `osDelay(1)`。RTOS 一 tick 延时的实际停顿取决于相位、抢占和任务调度，不能机械地把它当测得的固定 1 ms，也不能只根据线速将全量采样翻倍。

当前 capture slot 最大 128 B，sample 已 126 B；向每电机添加 Tx/Rx 序号会同时涉及协议、slot 和吞吐。新 trace 用短时选通道及批量帧，并通过实际 transport 压测确定可持续预算。

CAN 目前估算每对命令/反馈 260 bits：腿主实验约57.2%，轮主实验约36.4%，六电机同时500 Hz约78%，会超过当前配置检查的70%阈值。这是现有代码预算模型，不是总线实测。暂保留按实验组分配 CAN 频率，而不把全电机频率统一抬高。

## 8. 验证与交付顺序

### A. 已认可的 host_debug 基础与正确性修复

位置：`PACE/host_debug/`。编译原 C 状态机、激励、协议和采集模块；替换时间/CAN/UART/RTOS 边界，不在测试中复制状态机。FakeCAN 分开入队、Tx-confirm 和反馈事件；FakeSerial 调用真实 capture，再交给真实 decoder 和拟合质量门禁。

验收至少覆盖：正常阶段转换、START/STOP/FAULT、tick积压期间新反馈、时间回绕、入队成功但尚未确认、终止帧队列容量、10 s 启动、分片/粘包/CRC坏帧、缺头缺尾、footer 后额外数据。离线测试不能证明 CAN/定时器时间域在实板已经一致或电机机械停稳。

当前机器 PATH 未发现原生编译器，但通过 vswhere 找到 VS2022 C++ 工具链安装目录 `E:/pow-mining-lab/.toolchain/vs2022`；可进一步验证并复用，构建产物仍放 `PACE/host_debug/build/`。

### B. 控制/标定/工况 metadata 和实验角色

先补可绑定的数据来源、真实目标保持语义、轮速度独立 holdout，以及分组/多幅值配方。所有正式拟合入口对合同、校准、会话完整性和独立验证使用同一个门禁。

### C. 精细时序与仿真接入

需要更细延迟分辨率时才实现批量 trace 并做吞吐验证。将已验证模型接入固定机身 Isaac/MuJoCo 回放后，再决定重训练范围。

本草案没有把新增增益或幅值写入自动运行的固件，也没有进行串口通信、下载或实机试采。当前程序正确性问题详见兄弟目录 `wheel_leg_motor_PACE/docs/2026-10-12-pace-code-review.md`。
