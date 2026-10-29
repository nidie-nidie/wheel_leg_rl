# PACE 代码审查与分阶段整理建议

日期：2026-10-12。范围：当前 PACE 固件应用层、Board、Motor、Protocol，以及 PC 录制、解码、归一化、拟合入口。未完整审计供应商 HAL/RTOS 或辨识算法数学；没有修改程序源码、连接电机或下载固件。

## 判断

可以继续维护，推荐分阶段修复和重构。已有电机编码/解码、CRC/帧序列化、激励、安全阈值检查、采集队列等可复用单元。复杂度主要集中在总控和跨模块的状态、时间、会话完整性约定。

`PACE/Src/pace_app.c` 共 1270 行、44 条文件级 static 数据声明，同时承担硬件初始化、UART 接收和命令组帧、tick 排队、使能、实验启动、CAN 编码、状态快照、安全、六种输出帧、统计、故障及停机善后、两个 RTOS 任务入口。实验 state 之外还有 header/footer/terminal/arm/safe pending/transport 等散落状态。

模块有划分，但底层 CAN ISR 直接修改电机 registry，上层 task 再关全部中断复制状态；实验模块又依赖 registry、协议 frame、DM 编码器。改启停流程必须跨好几层推理。主要问题是状态归属和时间语义，单纯减少 if 或拆文件不能解决。

以下位置均相对固件根目录；标为 PC 的位置相对兄弟目录 `wheel_leg_motor_PACE_PC`。P1 为优先修正的正确性问题，P2 为条件缺陷/结构风险。代码可证明的触发条件不等于已证明它造成了旧实机异常。

## 发现

### [P1] 历史 tick 与最新反馈混用，年龄下溢

Location: `PACE/Src/pace_app.c:1242`、`:960`、`:330`；`Motor/Src/pace_motor_registry.c:137`；`PACE/Src/pace_safety.c:46`。
Problem: 队列只保存 ISR 时间，任务稍后使用该旧时间刷新当前 registry 的 RX/TX 年龄。关中断只能保证复制一致，不能让当前快照成为历史快照。
Impact: tick=100000 us、最新 RX=100300 us 时，无符号年龄为 4294966996 us，新鲜反馈被标离线/超时；样本 age 也可能饱和，补跑旧 tick 会密集发送历史目标。
Recommendation: 明确调度、实际采样、总线事件三个时间；快照绑定实际读取时刻，过期 tick 使用明确超时/丢周期策略；若要求边界采样，保存对应快照/历史事件。不要仅把负年龄裁成零。
Risk: high，控制时序和安全语义。
Safe to auto-edit: no，先确定时间契约。
Verification: 本轮独立验证无符号算术；后续真实 C harness 覆盖延迟调度、反馈晚于 tick、正常回绕和积压。实机触发频率未测。

### [P1] FDCAN 与 TIM5 没有建立计数器起点映射

Location: `Board/Src/pace_time.c:29`；`Board/Src/pace_can_port.c:95`、`:264`、`:313`。
Problem: FDCAN 使用 INTERNAL 16 bit timestamp，却直接拼接 TIM5 的高 16 位。两个计数器分别启动，未采样/校准起点偏移。项目 HAL 头 `stm32h7xx_hal_fdcan.h:1269` 说明当前单位为 CAN bit time；频率相同不能证明起点相同。
Impact: RX/TX age、enqueue-to-TX 延迟可带偏移，低位回绕附近可映射到错误区间；这是独立于历史 tick 混用的另一项问题。
Recommendation: 时间端口独占单位、起点、回绕、最大事件年龄和映射；保留原始时间与校准证据。
Risk: high，时间与延迟辨识语义。
Safe to auto-edit: no。
Verification: 不同起点、16/32 bit 回绕、延迟取出测试；真实 offset 需要板上测量。

### [P1] 固件默认运行方式与 PC 默认录制方式冲突

Location: `Config/pace_experiment_config.h:17`；`PACE/Src/pace_app.c:1195`；PC `tools/capture_session.py:91`、`:104`、`:130`、`:159`。
Problem: 固件默认单向 UART、上电 10 秒后自动启动；PC 默认主动 CONFIGURE/START，未收到数据只等 5 秒。15 秒启动等待只用于 passive。打开后还无条件清掉串口输入缓冲。
Impact: 从上电按默认命令录制，可在第一字节输出前超时；晚连接/清缓冲可丢唯一 header。默认固件不会受理 PC STOP/ABORT。
Recommendation: 选择并统一自动/主机模式、等待策略和文档；录制器区分等 header、接收、等 footer、不完整退出，保留原始字节和原因。
Risk: medium/high，启停行为变化。
Safe to auto-edit: needs confirmation，先确定希望保留的运行方式。
Verification: 模拟串口覆盖 10 秒延迟、晚打开、已有 header、正常结束、超时和 Ctrl+C；未做实机启动试验。

### [P1] Footer 后仍接受可拟合样本

Location: PC `pace_raw/decoder.py:285`；`tools/fit_session.py:53`。
Problem: footer 被保存但不关闭会话；后续 sequence/time 连续的样本继续关联旧 header/config。
Impact: 会话结束后的数据可混入拟合，footer total_frames 也未约束实际全部数据。
Recommendation: 统一单会话生命周期验证；关闭后拒绝额外帧，或明确分离多会话。下游只消费已验证会话。
Risk: medium，异常文件接受规则变化。
Safe to auto-edit: needs confirmation。
Verification: 本轮调用真实 decode_bytes 和质量门禁，header+config+sample+footer+sample 输出两样本、两者 fit_eligible=True、issues=[]，质量门禁接受。

### [P1] 质量门禁只属于 CLI，拟合库/优化导出绕过

Location: PC `tools/fit_session.py:53`、`:77`；`fit/fit_leg_dm.py:183`、`:255`；`fit/fit_wheel_lk.py:190`、`:350`；`fit/pace_optimizer_bridge.py:62`。
Problem: 完整会话检查只在总流水线脚本。独立腿/轮入口没有同等检查。PACE 导出按 stage/role 选行，未检查样本/各电机 fit_eligible、age、饱和等质量标记。
Impact: 同一 raw 从不同入口得到不同接受结论；存在 CRC 等问题的会话仍可经其他入口产出模型。审查中将四腿全部标无效后，优化导出仍向 mock 保存器传入全部 3500 行。
Recommendation: 建立共享质量验证入口，区分诊断数据与允许拟合的数据；生成模型和优化导出都执行同一规则。CLI 只承担参数和输出。
Risk: medium，以前可运行的坏数据会被拒绝。
Safe to auto-edit: needs confirmation。
Verification: 审查中合成 crc_error 数据由总门禁拒绝，直接腿/轮拟合仍生成模型且完整模型格式校验接受；后续增加入口一致性回归。

### [P2] 会话完成不包含安全停止发送的确认

Location: `PACE/Src/pace_app.c:323`、`:239`、`:1251`；`Board/Src/pace_can_port.c:159`、`:284`；`PACE/Src/pace_experiment.c:341`。
Problem: safe pending 在 CAN 入队成功就清位，真实 Tx Event 确认稍后发生；随后立即可排 terminal/footer，未要求 safe pending/pending CAN 清空。
Impact: COMPLETE/footer 不能证明六个停止命令已确认发送；甚至某条停止命令尚未入队也可输出 footer。这不证明实机一定停不住。
Recommendation: 明确实验结束、停止发送/确认、数据排空几个阶段及超时结果；总控不能用一个 COMPLETE 混合它们。
Risk: high，停止和故障处理。
Safe to auto-edit: no，分阶段设计。
Verification: 部分入队失败、Tx Event 延迟、BUS_OFF、pending 满与正常停止；总线确认不等于电机执行完成。

### [P2] 终止事件和 footer 要两个槽，但只检查一次余量

Location: `PACE/Src/pace_app.c:248`、`:252`、`:262`；`PACE/Src/pace_capture.c:43`。
Problem: 有效容量 127、深度 126 时 event 占最后一槽；footer 再 push 就置 overflow/rejected，尽管它仍 pending 可重试。消费者若恰好在两个 push 间释放空间则不触发，实现无此保证。
Impact: 可等待背压被当作丢失，footer 统计被污染。
Recommendation: 每次提交检查余量/分次推进；区分未提交可重试与真正丢数据。
Risk: medium，终止统计。
Safe to auto-edit: needs confirmation。
Verification: 真实 C 队列加终止流程，覆盖深度 125/126/127、消费者穿插和重试。

### [P2] CAN 接收忽略真实长度，异常短帧可污染反馈

Location: `Board/Src/pace_can_port.c:80`、`:294`、`:303`、`:319`。
Problem: mask=0 接收全部标准 ID；未检查 DataLength/RxFrameType，未初始化 data[8] 始终按 DLC=8 解码。RX FIFO full/lost 未接入通知和完整性统计。
Impact: 已知 feedback ID 的短帧可把未填写字节变成有效反馈，接收丢消息也可能静默。正常完整电机帧不必然触发。
Recommendation: Board 层校验帧元信息、传真实有效数据，记录 malformed/RX-loss；registry 只应用已验证事件。
Risk: medium，硬件异常路径。
Safe to auto-edit: needs confirmation。
Verification: 已知 ID 的 0..7 B/remote/8 B、未知 ID、RX FIFO 溢出。

### [P2] 两套字节扫描规则不能可靠恢复；非法 header 还可除零

Location: PC `tools/capture_session.py:51`、`:72`、`:75`；`pace_raw/decoder.py:210`、`:220`、`:139`。
Problem: FooterDetector 与 decoder 分别维护魔数/长度/CRC 恢复。错误声明长度导致等待/终止或按错误长度跳过。rate=0 虽报告 issue，仍参与除法。
Impact: 合法 footer 可被错误前缀遮住；非法 header 可让诊断工具崩溃。
Recommendation: 共用增量帧扫描器；扫描、单帧解析、会话验证分清职责。未经语义校验的 header 不可驱动计算。
Risk: medium，损坏文件恢复规则。
Safe to auto-edit: needs confirmation。
Verification: 本轮真实解码：伪 PA 宣称 128 B 后加合法 footer，得到 0 帧/truncated_frame；合法 CRC 的零 rate header 后第二 sample 抛 ZeroDivisionError。

### [P2] 测试与文档维护另一套过期契约

Location: `PACE/tests/test_capture_transport.py:6`、`:58`；`PACE/tests/test_can_schedule.py:31`；`Protocol/tests/test_motor_protocol.py:9`；`README.md:93`。
Problem: 吞吐模拟仍按 500 Hz 连续字节消费，而实际导出 250 Hz，blocking transmit 后每帧 osDelay(1)。很多测试检查源码字符串或独立 Python 重写算法，未执行 C 状态机/ISR。文档的 DMA、带宽和编译状态与代码/旧产物不一致。
Impact: 测试通过不能证明实际采集正确，结构清理还容易被字符串断言阻碍。
Recommendation: 保留有用旧测试，补真实 C host harness 和串口 mock；共享并验证采样/协议权威定义。
Risk: low/medium，测试和构建组织。
Safe to auto-edit: needs confirmation，纳入第一阶段设计。
Verification: 见执行记录。

## 后续风险

- Host 模式先以当前时间执行 STOP，后处理旧 tick，`pace_experiment.c:337` 无符号阶段时间可下溢而跳过 1 秒 STOP。默认单向模式不解析主机命令。
- ARMED 只看 rx_valid，未共同要求反馈最新性/online/使能重复完成；`pace_app.c:860`、`pace_experiment.c:229`。
- FINAL 分支仍取 commissioning 默认配置，只改 hash/limit ID；nominal/action rate/hold/amplitude/bandwidth/LK mode 等未真正驱动正式命令。当前锁定期间不影响 commissioning，不能只改 READY 开锁。
- 模型合并未约束来源/采样周期：`fit_wheel_lk.py:218` 接受旧 model，`model_manifest.py:196` 只换通道，旧 source_session/sample_period_s 会保留。审查中不同来源/周期的合成数据已显示此行为。
- 250 Hz 数据 delay_steps 的单位为 4 ms，仿真适配器每调用一次推进整数延迟；应核对调用周期与 sample_period_s。一旦调用为 500 Hz 就可能把延迟减半；当前调用入口未完整审计，不声称已经发生。
- 配置和 motor manifest hash 是人工固定常量，PC 还读取当前兄弟固件头文件。历史数据应绑定自己的固件/配置/标定快照，不能假定当前源文件就是历史权威。

## 适合的模块边界

| 单元 | 职责 | 状态归属 |
|---|---|---|
| 实验逻辑 | 配置、阶段、激励，输出语义目标/阶段事件 | 仅实验上下文 |
| 电机 I/O/快照 | 编码、总线事件、确认命令、反馈、时间映射 | pending/registry，明确 ISR 与任务规则 |
| 会话协调 | 等待/使能/运行/停止/故障/排空 | 单个会话上下文和状态转移入口 |
| 记录器 | 各类 raw frame、顺序、计数、终止输出 | sequence/统计/输出进度 |
| UART 传输 | 队列消费、背压、发送结果 | 单个传输任务拥有状态 |
| PC 流/会话验证 | 增量扫描、会话边界、共同质量结论 | 每个流/会话上下文 |

不要只是把全局变量搬到公共头文件，或让每个函数拿到整个巨型 context。每项状态有一个修改入口，其他模块通过快照、结果或事件使用它。实验核心不负责 wire frame 或 STM32 接线。

## 不宜在清理中顺手改变

- 六电机顺序、CAN ID、payload 量化、raw offsets/CRC 是兼容边界。
- 保留 Tx-confirmed 量化命令；入队成功不能替代确认。
- 正式合同锁、饱和拒绝、安全阈值、中止检查是有意设计。
- ring 的单生产者/单消费者和短临界区意图；同步规则明确后再调整，不能盲删 volatile。
- 供应商 HAL/RTOS、生成文件、时钟/DMA/链接区域不纳入第一轮结构整理。

## 推荐推进顺序

1. 固定时间/会话/停机契约，选择希望保留的自动或主机模式；建立独立 PACE PC 环境、真实 C harness 和串口 mock。
2. 先逐项修时间与快照、录制边界、共同质量门禁；每项都有原失败用例，不把算法、硬件配置和大量布局一起改。
3. 验证后按状态归属提取会话、记录和命令边界；app 留硬件/RTOS 接线与清晰事件顺序，先保持公开接口与 wire 协议。
4. 板上验证控制/采样周期、时间映射、正常/异常终止、RX 溢出及停止确认，再关联旧录制根因。

局部整理便宜，但不能解决时间和完整性问题；全部重写会重新承担硬件/协议/标定风险。推荐保留已有单元、渐进修复核心边界。

## 本次验证与限制

- Protocol/tests：15 项通过；PACE/tests：11 项通过。主要是 Python 模拟/字符串检查，不能代替真实 C 固件验证。
- PC 完整 unittest 发现：10 项实际测试通过，2 个模块加载失败，环境缺 pyserial/jsonschema。未安装依赖到训练环境；这两项是环境限制，不是已证明的采集 bug。
- 独立内存复现：footer 后样本通过真实总质量门禁、零 rate 除零、假 PA 遮住 footer、旧 tick 年龄下溢；无硬件访问。
- 重解现有 raw：003/004 当前各报一次时间断续，仍有截断/缺 footer，旧 summary 的 153/227 已过期；005/009 缺 header；007/008 缺 footer；最初 commissioning 和 001 footer 无效；002 有 CRC/sequence/缺 footer；006/link_smoke 为空。局部 sample fit_eligible 不等于整场会话可拟合。
- 旧 Keil 日志曾显示 ARMClang 6.22 构建成功；本轮未重新 ARM 编译、测板或确认旧 hex 与全部现源码一致。README 和旧日志不能替代新验证。
