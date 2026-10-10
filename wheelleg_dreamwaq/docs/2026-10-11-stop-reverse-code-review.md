# 停车/换向与速度奖励：独立代码复核

日期：2026-10-11。复核者：/root/stop_reverse_code_review。范围：git diff e27326a、git status 中新增源文件/测试及相关调用链；未修改实现、启动训练或调用子 agent。

结论：发现一项 P1，主 agent 已修复，本复核者重新验证；最终无剩余 P0/P1，可进入剩余真实集成验收。此结论不代表正式训练或性能目标完成。fresh/resume/play/export/golden 等运行门槛通过后才可启动四次正式训练。

## 已关闭的 P1

动态诊断遗漏第二次换向的响应门槛。原 sim2sim/mujoco/wheelleg_mujoco/command_practice.py:103 仅在 index == 2 时检查反向响应；第 4 阶段（tick 300–399，从负向切回原方向）完全忽略命令，只要阶段完整，也会被接受。因此 held_error_band_confirmation_time_s=null 仍可导致整体 performance_accepted=true，错误宣称换向达标。

最小修复：条件改为 index in {2, 4}，两个换向段都检查 1 秒内完成连续 10 个有效样本的误差带确认。主 agent 已应用；新增回归在 sim2sim/mujoco/tests/test_command_practice.py:94。

确定性复现：完整 500 帧有效轨迹，其余阶段完美跟踪，tick 300–399 的 vx 恒为 0（目标 +0.5）。仅在内存恢复旧条件得到 pre_fix_accepted=true；当前实现得到 fixed_accepted=false，第 4 阶段失败项为 reverse_response_time。没有改源码做对照，没有训练策略。

另有非阻塞 P2：README 新增 Markdown backtick/fence 原被反斜杠转义。主 agent 已去除，复核最终 diff 确认关闭。

## 关键约束与准确证据

下列 commands.py、env.py、training_profiles.py 均位于 source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/；suite 指 scripts/run_dreamwaq_training_suite.py；动态统计指 sim2sim/mujoco/wheelleg_mujoco/command_practice.py。

- reward→next command：env.py:783 计算旧命令 reward，:806/:817 构建旧 Command/Tracking 日志，:847 才推进命令。
- 真控制步与 partial reset：env.py:775 按 common_step_counter 去重，commands.py:124 只增加独立 steps；env.py:905 保留原 reset 采样，:907 只重置对应行。episode_length_buf 只用于原 termination，不作为练习 clock。
- 缓存与 history：env.py:780 显式同步 cached state.command，:938 保留 partial-reset clone 路径；真实 probe 已验证下一 actor/critic/history 帧、额外读取和真实 done 的历史填充。命令切换不清 history。
- 阶段与 RNG：commands.py:68 完整阶段 payload 进入两个 manifest；:92 的 tick 数学覆盖全部边界与 499/500/501。纯推进不访问 RNG，reset 初始命令独立保存。legacy_v1 默认仍 disabled/hold=true/原权重。
- train/resume：scripts/train_dreamwaq.py:233 在构建 base-task 前识别并应用保存 profile；:272 完整 checkpoint/hash 验证保持。training_profiles.py:36 拒绝显式不同 profile；不匹配的保存权重/阶段仍由完整 contract hash 拒绝。
- fixed play/evaluation：scripts/play_dreamwaq.py:132 完整身份验证后才在 :145 关闭 schedule，:103 同步 cached command。scripts/evaluate_isaac.py:173 验证训练身份，:264 建独立默认 scoring/disabled schedule 的名义环境，:302 同步固定首帧命令。
- export/loader：scripts/export_dreamwaq_actor.py:130 保留既有完整 saved-artifact 校验，:83 保留真实训练 command payload，:84 明写全部训练 reward weights。MuJoCo contract.py:121 拒绝未知/矛盾阶段 payload。网络尺寸及旧 PhysicsV4 不重标规则不变。
- 范围守恒：完整 diff 未改变 PPO/CENet/history、reward 公式、PhysicsV5、USD/XML/mesh、control/action、randomization、reset 初态/时钟或 termination 实现；两个 manifest 的关闭状态新字段改变 hash 属于已批准行为。
- Isaac V2：scripts/evaluate_isaac.py:322 在 act 前读取 cached state 的真实物理单位 vx/wz，按 active_before 累积；:338 estimator 仍按 critic[:,25:28] 原定义。done 后 reset observation 不计入物理 MAE。
- scoring/baseline：schemas/isaac_evaluation.py:335/:337 要求原 RewardWeights() 与 disabled command 配置；V1 matcher 未放宽。evaluate_isaac.py:386 的 candidate-only 不进入 baseline 比较，:419 标记 not_comparable。suite:349/:350 的 baseline/phase2 acceptance 为 null。
- 动态统计：command_practice.py:71 要求连续有效前缀，:79 排除前 0.5 秒，:86 的 COM 世界 XY 路径含段首边界，:62 返回 10 样本窗口结束时点，:96 未完成阶段必失败。diagnostic runner scripts/evaluate_command_practice_mujoco.py:80 排除失败帧并结束本场景。
- 四次与失败路径：suite:67 生成不同种子，:159 显式传 profile且不传 resume，:115 限定 formal 四次/1000/rtx5070。Isaac、正式 MuJoCo、动态 MuJoCo 分别有独立错误处理（:293/:316/:336）；性能失败不会省略后续 sim2sim 或其他候选。
- fingerprint：scripts/run_training_suite.py:73 枚举 source/scripts/configs，:79 包含 MuJoCo package。实际调用确认新增 training_profiles.py、command_practice.py、evaluate_command_practice_mujoco.py 均在映射中。

## 复核者实际验证

- 主工程针对性测试：.venv/Scripts/python.exe -m pytest tests/unit/test_command_practice.py tests/unit/test_current_isaac_evaluation.py tests/unit/test_training_suite.py -q，22 passed，exit 0。
- MuJoCo 针对性测试（sim2sim/mujoco 下）：.venv/Scripts/python.exe -m pytest tests/test_command_practice.py -q，5 passed，exit 0。
- 内存旧/新条件对照：pre_fix=true/current=false，第 4 段 reverse_response_time；新增三个源工具 fingerprint 覆盖全部为 true。
- 读取主 agent 完整测试的 JUnit：.git/verification/practice-final-unit-v1.xml 为 181 tests，0 failures/errors/skipped；practice-final-mujoco-v1.xml 为 56 tests，0 failures/errors/skipped。复核者核对文件，未重跑完整套件。
- 读取主 agent 的 .git/verification/practice-isaac-probe-v1/test_real_physx_command_reward0/practice-probe.json：passed=true，230 checks，0 failed，PhysicsV5，control_dt=0.02。该仿真由主 agent 执行，复核者核对 JSON。
- git diff --check：exit 0，无输出。

核验时 suite source fingerprint：274842FD7E9C4EE34907BBC1AE92BAE338F8770D8E022D7B8E20BD9D460A2F1D。正式训练前应重新生成并冻结最终来源记录。

## 文档依据与剩余运行门槛

已完整读取 Architecture v0.26 §31/ADR-070、设计、实施计划、独立文档复核。SHA256：

- Architecture：E81D66D8B1FE446A8B38DF433C857474B558F3DB87A49D1F39097A39650602F3。
- 设计：B62F938661E4CAD31492856A50A7CD95EF8181ACB49037E61E7EC4CD487761B0。
- 计划：CC14CB945310565E05006260FB513434A161923F28D7E32D9346A305CEDDC1E8。
- 独立文档复核：0CB021629D52A8684CF02E84BC77798BB5404E408CB4E909E2BC04A677812AE5。

尚待运行：便携 fresh 两次 update、同/跨规模 resume、固定命令 play、真实 export/loader/golden、candidate-only 八场景与动态入口 smoke。通过后启动四个新种子各 1000 iteration，并保留全部四次固定/动态 sim2sim 及真实性能失败项。
