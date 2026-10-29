# 辅助力起身训练延长到 8000 次

用户于 2026-10-29 授权从第 4000 次模型继续训练，随后明确要求停止训练并测评。本次续训于 2026-10-29 14:25:17（Asia/Singapore）人工停止，最后完成第 6243 次更新，最后完整检查点为 model_6200.pt。原定 8000 次目标未完成。环境规模 4096、seed=42001、每 50 次保存；原始源码、算法、奖励、物理配置和辅助撤除日程没有修改。原 4000 次运行目录与验收证据保留。

## 入口和身份

原 V3 CLI 严格恢复要求完整配置身份相同，预算和运行目录包含在配置 SHA256 中。使用 artifact 内显式续训入口 continue_training.py，调用现有训练、checkpoint 和环境 API；不改写生产源码、父 checkpoint 或其 index，也不放宽现有严格 checkpoint 加载函数。

入口读取父 run_manifest 的完整 identity，并以该 identity 严格加载 model_4000.pt。新配置只允许 environment.run_dir 与 assessment.updates 两项变化，新增配置身份和入口脚本字节身份单独记录。实际引擎读回的完整接触模型和执行实现身份必须与父运行一致。启动前通过独立复核，检查父验收、模型和授权文件 SHA256、资源及重复启动证据。

父模型 SHA256 为 D0E4958BE5A33581E29EC08C53AD56D61E45D351E24810BFD518417A4980202A；生产源码 SHA256 为 F06D42C9DFBD3D5D8CFC06E11414A83D1531C3BDD22A25087B2088B6247C0FB8。新增配置 SHA256 为 F2C8795EFD2A33D8A2E237EDDBF9191EB60A3A5BE221E5CE6DFD576A6467EE40。

## 恢复和训练行为

依次恢复策略、PPO 优化器、学习率、成本网络及其优化器、辅助控制步；然后恢复环境命名随机状态、课程和帧号，刷新观测，最后恢复全局 RNG。用户明确授权跳过历史日志检查，新增阶段建立独立统计归档，覆盖 4001–8000 次更新，statistics_scope=extension_only。历史累计统计不导入；父统计描述仅作为来源引用保存在 manifest。物理环境按现有恢复规则开始新回合，不宣称恢复 PhysX 连续物理状态。父模型及其完整身份校验继续执行。

guidance.end_updates=3000 保持，初始 k=96000、alpha=0，续训全部 4000 次实际辅助力、成本、成本优势及罚项均应为零。零辅助时成本优化器跳过更新，因此最终成本网络和优化器必须与父状态完全相同且有限；策略参数必须实际变化。学习率沿用父值并继续原 adaptive KL 调整。

新增运行位于 recovery/artifacts/paper-guidance-continuation-20261029-quickstart/training4096to8000。更新记录应严格连续 4001–8000，k=24×completed_updates，最终 k=192000、alpha=0；保存模型 4050,4100,…,8000，共 80 组。

## 完成核验与跟进

要求进程退出码为零，内置训练验证 passed=true、completed_updates=8000，新增 measured_update_count=4000、environment_steps=393216000，累计环境控制步为 786432000。完整新增更新序列、模型索引及哈希、身份、最终日志屏障、worker 正常关闭均单独核验。失败统计仅报告新增阶段，历史运行独立保留；不进行父 metrics 归档的全量读取检查。

沿用自动化 2000 跟进本次续训，正常推进时保持安静；完成、明确故障、停止推进或需要处理时通知。资源不足时等待，不缩减规模，不干预其他项目进程；故障后不自动重启。完成或明确故障后暂停自动化。

成功门控尚未标定，起身、站稳、机械限制和完整交接能力仍未验收。原训练记录的物理失败及几何极值继续保留，延长训练不改变能力判定标准。

## 实际运行状态

2026-10-29 10:30:11（Asia/Singapore）按最新用户授权启动 quickstart 续训，supervisor PID=28960、Python 入口 PID=26864、实际训练 Python PID=31760。CPU 前置检查通过，新增统计恢复逻辑独立复核未处理 P1/P2=0。正式入口 SHA256 为 D341260044A93D0D2B8183AD749CBF9DB171F3254B9909C6555F621E2125375C，launcher SHA256 为 33ED46120D869308CFA45787FA5E45DB2177D5166CFF2B6B3088032F0A3C812B。

14:25:17 按用户要求停止已核实身份的实际训练 Python，入口、监督和所属工作进程随后退出。退出码为 -1，状态记为 stopped_by_user_request；未生成正常训练结束的日志屏障，不宣称完成 8000 次验收。已核实 5 个无关 Python 进程仍然存在。

实际新增 2243 次更新。最后完整模型 model_6200.pt 及其 index 保留，模型 SHA256 为 D344C8E9739DDE63A3BA0B37BA5D58B2A65142D88640662BB47CCA3D3DF1DE0E。停止身份、进程与退出证据保存在 quickstart/user_requested_stop.json。

14:31:22–14:32:14 使用该模型完成同源 Isaac 仿真诊断，12 个保留 test 初态均与训练记录、产生轨迹和物理内容分离。每个类别两条初态，每条最多 8 秒，使用确定性均值动作，逐步确认零辅助。评估运行退出码为 0，实际运动回放耗时 41.59 秒。4066 条首回合逐步记录与 12 条结果的步数、连续性、终止原因、最终高度、回报及轨迹 SHA256 交叉核对通过。

5 个初始倾角至少 90° 的样本均曾进入倾角小于 90° 的范围；12 条样本均未满足连续 1 秒的运动学稳定条件（倾角小于 0.25 rad、COM 高度 0.17–0.25 m、角速度范数小于 0.5 rad/s）。10 条达到 8 秒时限，2 条前向倒地样本因地面穿透或闭链约束故障提前结束。两条接近直立的初态最终倾角也约为 22°。这些固定初态结果表明存在部分翻转能力，稳定起身尚未通过本次诊断。

评估产物位于 recovery/artifacts/recovery-evaluation-20261029-stop6200/results：summary.json、summary.md、first_episode_trace.jsonl、result_verification.json。成功门控仍未标定，正式成功率记为 null，未验证交接能力。

对自动化 2000 的暂停请求返回“Automation does not exist in the app”，当前不存在可暂停的该自动化。禁止重新启动训练。旧续训故障证据保留在 recovery/artifacts/paper-guidance-continuation-20261029；当前调度、进程身份和增量日志位于 quickstart 目录。
