# 停车、换向练习与前后速度奖励调整（StopReverseTaskProfileV1）

日期：2026-10-11。状态：待独立文档复核；复核无 P0/P1 后按用户授权实施。
正式架构：Architecture v0.26 §31 / ADR-070。提交前快照：c5fbc91cff1091bdacfc9158d8b0834e919288e3（已推送 main，历史完整保留）。

## 1. 用户已批准的目标与边界

本轮只新增连续的停车/换向命令练习，将前后速度的两个奖励系数由 1.0 改为 2.0，并打通这些配置对应的 checkpoint、play、export、验收和四种子训练入口。用户已批准“文档 → 一个 agent 文档复核 → 实施 → 一个 agent 代码复核 → 验证 → 四个独立种子各 1000 iteration → sim2sim”的顺序。agent 串行使用，不为编码另开并发 agent。

不扩大随机化，不增加地形，不修改 USD/XML/mesh、PhysicsV5、PD/effort/damping/armature、动作、命令范围和归一化、观测维度、history t−4..t、CENet/PPO/AdaBoot、reset 初态与物理时钟、termination 和现有八场景 MuJoCo 评估。训练全部从新权重开始；旧 PhysicsV4 工件不重标、不续训。

## 2. 两种明确的任务配置

保留默认 legacy_v1：CommandSamplingV2 在 reset 采样一次并保持整集，RewardSchemaV2 的两个 vx 系数仍为 1.0。普通 PPO 与未选择新任务配置的入口保持原行为。

新 stop_reverse_v1 由一个纯 Python 配置函数同时设置：
- commands.practice_schedule = "StopReverseCommandPracticeV1"；
- commands.hold_for_episode = false；
- reward_weights.tracking_vx = 2.0；
- reward_weights.tracking_vx_enhance = 2.0。

CommandRanges 新增 practice_schedule，默认为 "disabled"。新增 command_contract_payload：以 asdict(commands) 为基础，启用时追加 practice_contract，完整保存版本、stage_end_control_steps=[100,150,250,300,400,500]、factors=[1,0,−1,0,1,0]、control_dt_s=0.02、真实步时钟与 reward→next-observation 更新顺序；disabled 时 practice_contract=null。两个 manifest builder 使用此 payload，将实际阶段定义纳入 PPO/base-task/export manifest 与 contract hash。CommandSamplingV2 仍专指 reset 的原采样算法；阶段函数是新具名契约 StopReverseCommandPracticeV1，不能悄悄覆盖默认 hold 行为。新增关闭状态元数据也会改变新构建的 contract hash，严格验证不得绕过；目前没有正式 PhysicsV5 训练 checkpoint 需要迁移。

新配置保持初始四种模式 20% 站立、30% 直行、20% 原地旋转、30% 组合；vx 范围 ±1.5 m/s，yaw ±1 rad/s，高度 U(0.16,0.24) m。直接复用原 reset 命令和 command_rng，不新增 RNG、不在阶段边界重采样。初始方向随机，因此两种方向都覆盖。

## 3. 一集内的练习

设 reset 得到初始命令 c0=(vx0,wz0,h0)。下面只乘前两项，高度始终 h0；站立模式仍始终为零速度。

| 已完成的真实控制步 | 名义时间 | 下一策略帧中的命令 |
|---|---|---|
| 0–99 | 0–2 s | (vx0,wz0,h0) |
| 100–149 | 2–3 s | (0,0,h0) |
| 150–249 | 3–5 s | (−vx0,−wz0,h0) |
| 250–299 | 5–6 s | (0,0,h0) |
| 300–399 | 6–8 s | (vx0,wz0,h0) |
| 400 起至结束 | 8 s 起 | (0,0,h0) |

直行组练习前进/后退 → 停车 → 相反方向；旋转组练习原地左/右转 → 停转 → 反向转；组合组两个分量同步变化。没有真实速度清零、状态写回、临时力矩、重新 reset 或历史清空。停车是命令为零，由策略和原 PD 控制自行减速。

名义 episode 仍为 10 s。Isaac 原 499-step timeout 保留，所以最后停车段实际到 9.98 s。RSL-RL 的 init_at_random_ep_len 保留；它可能缩短首次 episode，但绝不能作为练习阶段时钟。每个环境独立维护真实控制步计数，从 reset 的 0 开始；partial reset 只清本行。训练中步数不足的 episode 按真实经历的前缀记录。

阶段切换是阶跃命令，不新增斜坡或自动速度 curriculum。本轮不承诺权重翻倍一定解决速度跟踪，且同时改变练习与训练动力学，不能用结果声称单独某项改动具有因果效果。

## 4. 更新时序与缓存

现有 DirectRLEnv.step 先计算 done/reward，再 partial reset，再生成 observation（IsaacLab direct_rl_env.py:395–422）。新增推进函数放在 _get_rewards 的所有旧命令奖励与日志计算完成之后：
1. 当前动作使用策略看到的旧命令执行；本动作 reward/Tracking 日志仍对旧命令。
2. 仅当 common_step_counter 超过上次处理 tick，逐行增加真实练习步数并计算下一命令。
3. 将下一命令写入 _commands，并同步当前 WheelLegState.command，包含 partial reset 后曾克隆的缓存。
4. 随后的 reset 将终止行重置到新 c0 和阶段 0；观察返回新命令。
5. history wrapper 仍每个 step 追加一帧，只有真实 done 才填充 reset 五帧。

额外 get_observations 或重复读取奖励不能推进阶段、消费 command_rng、追加 history 或推进物理时间。初始命令保存在独立张量，不能与 _commands alias。应验证边界 99/100/149/150/249/250/299/300/399/400，以及缓存 command、actor/critic command 和 history 最新帧一致。

## 5. 奖励契约

只变训练配置，不改 compute_reward_terms 的公式、tracking_sigma=0.25、control_dt=0.02 和其余系数。tracking_vx_enhance 的原公式是 exp(−vx_error_squared/2.5)−1，属于宽范围误差惩罚；系数翻倍也会翻倍其负值，不得写成额外的正奖励。yaw 权重仍为 1.0。

测试要证明两项 vx 加权贡献恰好翻倍，其余加权项完全相同；在误差 0、0.5、1.0 m/s 上检查数值和总差。PPO 默认权重与静态命令保持原值。

## 6. checkpoint / play / export

train_dreamwaq 增加 --task-profile legacy_v1|stop_reverse_v1，fresh 默认 legacy_v1，正式新 suite 显式传 stop_reverse_v1。resume 未指定时从已验证的源 run manifest 命令配置识别 profile；显式指定不同 profile 必须拒绝。配置必须在构建 base-task 与验证 checkpoint 之前应用。

play 和 Isaac policy loader 同样从 run manifest 的 base-task 命令配置重建具名训练 profile，再执行完整 contract/hash 验证；未知 schedule 或不匹配的保存配置拒绝。固定命令 play、名义八场景 evaluation 在训练身份验证完成后关闭 schedule，固定命令不能被覆盖。非固定 play 可以选择保持训练练习；必须在输出中说明实际模式。

export 使用保存的 base-task 和完整校验，command_sampling 保存完整阶段 payload 与 hold 字段；新增 training_reward_weights 明确保存全部训练奖励系数，不能仅靠 base-task hash 暗示两个系数。MuJoCo 不运行随机训练课程；固定八场景继续由评估器指定命令。25/125/41/6 维和 TorchScript 输入输出不变。

## 7. 当前 PhysicsV5 的独立 Isaac 验收

已确认现有 suite 强制使用旧 PhysicsV4 的 PPO run-03 基线（run_dreamwaq_training_suite.py:36–40、258 起）；evaluate_isaac.py:186–203 对照当前配置严格验证旧基线，且 evaluate_isaac.py:370–373 强制 baseline-report。新 PhysicsV5 / 新命令奖励 profile 与其不兼容，不能靠关闭校验完成比较。

保留旧 IsaacEvaluationContractV1 与 baseline 比较路径。新增显式 --candidate-only 模式和 IsaacEvaluationContractV2，专用于当前 PhysicsV5 的单候选验收：
- 同一八场景、seed、名义 randomization、reset-cache、499 步、确定性 Actor 和 estimator MSE 定义不变；
- 实际评估 reward 使用原 RewardWeights() 的统一固定评分系数（vx 两项为 1），schedule 关闭；V2 显式记录实际评分权重与固定命令运行配置；
- 训练 profile 和其 2 倍系数仍由训练 base-task hash 完整绑定；
- 实际 vx/yaw MAE 固定在 active_pre_action_frame：act 前从 direct_env._current_state() 的 root_com_linear_velocity[:,0] 与 root_angular_velocity[:,2] 读取 m/s、rad/s，只计 active_before；缓存必须对应当前 base_observations，此时尚未执行 step。不得拿归一化/裁剪后的 critic[:,25] 当真实 m/s；done 的 post-reset 帧不得计入。本 tick 末仍存活与终止的 pre-action 样本均有效，不采 postreset；synthetic partial-reset 测试证明这一点。estimator MSE 继续用原归一化 critic[:,25:28]，不改数学。记录存活、estimator 与已有结果，不提供旧 baseline comparison；
- report baseline_comparison=null，baseline 状态为 not_comparable，旧 Phase 2 baseline 接受项为 null，绝不伪装为通过。

suite 新 profile 必须显式 candidate-only；在训练前验证参数，免得四次训练后才遇到基线阻塞。四个候选共享一个新 PhysicsV5 名义 evaluation reset cache。V1 的旧契约匹配函数不放宽。

## 8. sim2sim 与效果报告

全部训练完成后，每个 checkpoint 独立 export，核验 saved-state 与 TorchScript golden-vector，再运行原 MujocoEvaluationContractV3 的八场景 500 tick / 10 s；四次只允许 seed 不同，其余 fingerprint 相同。

另加独立的动态命令诊断，不修改原八场景排名/失败判据：
- 直行两组：(±0.5,0,0.20)；
- 原地转向两组：(0,±0.6,0.20)；
- 使用同一六段 timeline，记录每帧当前执行命令、下一观测命令、实际速度、姿态和失败原因。
- MuJoCo runtime.step(action,next_command) 的参数控制下一策略观测；当前 action 已由旧命令生成，所以逐步日志的 target 必须使用当前执行命令。下一命令更新要在本次 step 形成的新 observation/history 中生效，禁止提前修改上一历史帧。
- 共享阶段数值由无新依赖的命令练习契约实现/镜像并做逐边界一致性测试；严禁为诊断改变 physics 参数。

报告停车稳态 vx MAE、yaw MAE、停车距离和净位移、反向后的首次正确符号时间。停车距离定义为停车段内 base COM 世界 XY 的累计水平路程，包含阶段起点边界位置；另报该段起终点净位移，不能用 body vx 积分替代。响应时间从当前执行命令阶段的起点算，以连续 10 个有效控制样本（0.2 s）窗口的结束时点判定达到误差带，不能把窗口起点当确认时间；未完成阶段可保留有效前缀统计但性能判定必须失败。记录持续 0.2 s 达到 |vx error|≤0.15 m/s 或 |yaw error|≤0.15 rad/s 的时间；未达到记 null。稳态窗口排除每段最初 0.5 s，短段不足时不冒填结果。任何跌倒后的帧不计入有效跟踪统计，终止前缀不能当作整段成功。

本轮预先声明的效果目标：原八场景全部存活；站立三个场景 vx MAE≤0.10 m/s；forward/reverse vx MAE≤0.20 m/s；动态停车段稳态 vx MAE≤0.10 m/s、yaw MAE≤0.15 rad/s，反向运动/旋转在 1 s 内进入并保持上述误差带 0.2 s。这些是报告中的性能目标，不是给机器人施加的速度限制。未达标仍保留全部四次训练与评估，明确给出失败项，不在中途改权重/参数。不能仅凭总 return 提高宣布修复。

## 9. 实施文件与依赖方向

| 文件（wheelleg_dreamwaq 相对路径） | 职责 |
|---|---|
| source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/commands.py | 原采样保持不动；具名阶段字段、纯步数→系数函数、独立练习状态 |
| source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/training_profiles.py（新） | 具名配置应用/识别；只依赖 commands、rewards 和 dataclasses |
| source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py | reset 行初始化与 reward 后单次推进、缓存同步 |
| source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/manifest.py 与 dreamwaq_manifest.py | 消费完整 command_contract_payload，绑定真实六段定义 |
| scripts/export_dreamwaq_actor.py | 明写 training_reward_weights，保留完整命令 profile |
| scripts/train_dreamwaq.py | CLI、profile 应用/严格 resume、manifest 记录 |
| scripts/play_dreamwaq.py | 保存 profile 重建与 fixed-command 优先 |
| scripts/evaluate_isaac.py | profile 重建、显式 candidate-only、实际速度记录 |
| source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/isaac_evaluation.py | 独立 V2 评估契约，V1 保持 |
| scripts/run_training_suite.py | 半小时快照补充实际存在的 Reward/* 与 Loss/* 可选标签，不新增 PPO 必需标签 |
| scripts/run_dreamwaq_training_suite.py | 传 profile、分开当前验收与历史基线、训练后执行所有评估 |
| sim2sim/mujoco/wheelleg_mujoco/command_practice.py（新）与 contract.py | NumPy 消费/核验导出阶段 payload，无 Isaac 依赖；正式 loader 拒绝未知/矛盾 profile |
| scripts/evaluate_command_practice_mujoco.py（新） | 独立六段动态命令 MuJoCo 诊断与报告 |
| tests/unit/test_command_practice.py（新） | 时间边界、partial reset、RNG、profile/reward/hash 与未知配置拒绝 |
| tests/integration/probes/command_practice.py（新） | 真实 Isaac 首帧/边界/history/cache/额外读取验证 |
| tests/integration/test_command_practice.py（新） | probe、训练+resume+fixed play+export golden 的串行入口 |
| sim2sim/mujoco/tests/test_command_practice.py（新） | MuJoCo timeline 和下一帧/history 对齐、统计窗口 |
| docs + README | 当前 profile、执行命令、验收与结果记录 |

命令数学/配置 → 环境 → train/play/evaluate；manifest 消费配置，不反向操纵环境；算法/history 不依赖练习器。MuJoCo 动态诊断消费导出 manifest 和 runtime，不反向修改 Isaac 训练代码或 XML。

## 10. 执行门槛与训练

先独立文档复核，解决全部 P0/P1。再实施与单元测试，独立代码复核，解决全部 P0/P1 后完成真实 Isaac probe、小规模两次 update、同/跨规模 resume、固定命令 play、TorchScript golden 和 MuJoCo 动态入口 smoke。测试 smoke 的权重不进入正式四次。

正式命令：
```powershell
.venv\Scripts\python.exe scripts/run_dreamwaq_training_suite.py --mode formal --profile rtx5070 --task-profile stop_reverse_v1 --candidate-only --runs 4 --iterations 1000 --monitor-interval-seconds 1800
```

沿用 rtx5070、FudanStyleDomainRandomizationV1、平地和原超参数。suite 生成四个不同种子并在启动前写 manifest，训练串行、每次从零开始，每半小时记录 iteration、NaN/finite、episode/termination、实际速度误差、动作/effort 饱和、reward 分项和 estimator 指标。训练开始后冻结代码/架构与 suite fingerprint；若确需修复，停止并用新 suite 重新留证，不能混成同条件四次。

hard gate 只用于数值错误、工件/身份错误和已有严重异常；性能差、估计器接受失败与旧基线不可比较均不得阻止用户要求的全部 sim2sim。每次新评估失败要记录原因，继续处理其他已完成的 checkpoint，最终状态区分“执行完成”和“性能达标”。新 profile 本次训练授权已获得，无需在文档/代码复核完成后再次要求用户确认。


终点外行为：StopReverseCommandPracticeV1 的 stage_end_control_steps 最后 500 是诊断 horizon，final_phase_behavior='hold_last_factor'；400 及以后均返回系数 0，包括 499、500、501，不能在第 500 次 step 生成下一策略帧时出现索引越界。负数/非整数步数拒绝。此行为也进入 practice_contract/hash。
