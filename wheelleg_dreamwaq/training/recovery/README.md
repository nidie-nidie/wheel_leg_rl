# 轮腿倒地起身

起身训练代码目录：`E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/`。模型和实验记录继续保存在工程根目录的 `recovery/artifacts/`。

该目录用于倒地姿态库、完整接触模型、起身观测与奖励、独立 PPO 训练、checkpoint 与导出、双策略监督器、MuJoCo 组合运行及相关测试。

实现要求、公式及验证状态以[倒地起身设计](../../docs/2026-10-22-wheelleg-fall-recovery-design.md)为准。当前起身动作使用V2契约，变更与执行安排见[起身动作执行计划](../../docs/superpowers/plans/2026-10-24-recovery-unclipped-actions.md)。此前名义/随机条件四环境PPO、checkpoint恢复继续更新、MuJoCo五向下落和GPU接口报告保留原源码身份；完整起身与连续交接成功率尚未验收。

起身腿目标为 `q_reference+0.35*a[:4]` rad，轮目标为 `25*a[4:6]` rad/s；训练采样动作及导出均值均保留有限原值，起身腿目标取消±1 rad裁剪。0.35和25继续用于无量纲动作到目标角偏移、轮角速度的单位换算，25不表示轮速上限。冻结701行走策略继续使用原腿目标±1 rad限幅及原导出契约，兼容比较和MuJoCo按实际策略分别解码。

观测归一化和字段裁剪、全部奖励与惩罚、非有限状态/关节范围/闭链/地面穿透及配置要求的内部间隙检查继续保留；腿18 Nm、轮9 Nm的力矩上限及既有随机化能力范围保持。上一原始动作历史不裁剪。

## 代码入口

| 功能 | 文件 |
| --- | --- |
| 完整配置、动作、观测及奖励 | `contracts.py`、`config.py`、`control.py`、`observations.py`、`rewards.py` |
| 参数随机化、姿态库与课程 | `randomization.py`、`pose_bank.py`、`collect_pose_bank.py` |
| 独立 Isaac PPO | `env.py`、`env_cfg.py`、`ppo_runner.py`、`train.py` |
| 完整接触及GPU安全证明、CPU原生几何复核 | `contact_model.py`、`geometry_gpu.py`、`geometry_constants.py`、`geometry_binary.py`、`geometry_guard.py` |
| 标量/批量两策略状态机与完整日志归档 | `supervisor.py`、`batched_supervisor.py`、`metrics_protocol.py`、`metrics_stream.py`、`metrics_worker.py`、`metrics.py`、`metrics_archive.py` |
| 状态缓存、姿态张量池与阶段计时 | `env.py`、`pose_pool.py`、`performance.py` |
| 保存、严格恢复与均值网络导出 | `checkpoint.py`、`export.py` |
| 真实交接标定及 MuJoCo 组合评估 | `mujoco_runtime.py`、`calibrate.py`、`evaluate.py` |
| 正式训练证据检查 | `preflight.py` |

## 运行

在 `E:/wheel_leg_rl-main/wheelleg_dreamwaq` 执行。主 `.venv` 用于 Isaac，`sim2sim/mujoco/.venv` 用于 MuJoCo；后者需要源码路径：

```powershell
$env:PYTHONPATH = "$PWD/training;$PWD/sim2sim/mujoco;$env:PYTHONPATH"
& '.venv/Scripts/python.exe' -m recovery.config --create recovery/artifacts/my-run/resolved_config.json
& '.venv/Scripts/python.exe' -m recovery.config --check recovery/artifacts/my-run/resolved_config.json
& '.venv/Scripts/python.exe' -m recovery.train --config recovery/artifacts/my-run/resolved_config.json --headless --device cuda:0
```

生成的配置使用 `RecoveryResolvedConfigV2` 和 `RecoveryContractV2`，保留尚未提供的数据/标定路径为 `null`；填写真实文件后运行，不能把空配置当正式训练。V2恢复在原运行目录使用相同配置追加 `--resume <checkpoint路径>`，更新次数是总完成目标。恢复保存的优化器及随机流后开启新回合，旧回合标记为观察中断。旧V1配置和checkpoint不接受隐式续训，本轮从独立目录fresh开始。

用户授权的4096个环境、2000次更新、每50次保存的fresh V2训练已执行完成，实际时间为2026-10-24 05:41:03至08:39:06（Asia/Singapore），用时10682.7064582秒（2小时58分3秒），共196608000个环境步，退出码为0，更新平均耗时5.32秒。训练报告记录 `passed=true`、参数确已改变，共保存40组模型和索引；最终模型为[model_2000.pt](../../recovery/artifacts/unclipped-actions-20261024/training4096x2000/model_2000.pt)。起身与连续交接能力尚未验收。

本次训练的[完整配置](../../recovery/artifacts/unclipped-actions-20261024/training_config.json)、[调度记录](../../recovery/artifacts/unclipped-actions-20261024/scheduled_training.json)、[实际启动记录](../../recovery/artifacts/unclipped-actions-20261024/training_launch.json)和[退出记录](../../recovery/artifacts/unclipped-actions-20261024/training_exit.json)已保存，运行目录为[training4096x2000](../../recovery/artifacts/unclipped-actions-20261024/training4096x2000/)。训练日志95800个包全部确认消费，待处理数量为0且已完成最终排空。[最终只读验收](../../recovery/artifacts/unclipped-actions-20261024/final_acceptance.json)的66项检查全部通过，记录 `training_execution_passed=true`，已确认更新、模型、身份与完整归档的一致性。原记录注明 heartbeat 任务 `2000` 已暂停并禁止再次启动 fresh 训练；本次目录整理未重新检查其他训练进程。

本轮623665次尝试中619569次记录任务失败，4096次在训练结束时截断；物理就绪和接受交接计数均为0。支撑及兼容校准阈值为空，全部196608000步均被 `calibration_unavailable` 阻断，这些计数不能换算为真实起身成功率0%。最大地面穿透记录27608.2003 m异常，CPU原生复查结果不能用GPU保守上界解释，根因尚未确定。内部间隙100%凸包重叠来自 `diagnostic` 模式及已知空腔凸包问题，不能视为实际碰撞率。具体依据见[独立物理统计复核](../../recovery/artifacts/unclipped-actions-20261024/final_physics_metrics_review.md)。

本轮[代码校验](../../recovery/artifacts/unclipped-actions-20261024/unit/verification.json)通过126项测试及32项子测试；[导出组件](../../recovery/artifacts/unclipped-actions-20261024/checkpoint-export/verification.json)使用35条真实姿态记录，批次1/3/32及重载误差为0。[独立对抗性复核](../../recovery/artifacts/unclipped-actions-20261024/adversarial_review.md)未发现阻断训练的P1/P2。上述证据及配置预检对应源码 `A7A4A7BDCCF0F6EA98AD654A8DC853C29D25E3DA4202340F87228BF008334B10`；本轮实际配置、物理回读与训练报告保存在运行目录，能力验收继续单独登记。2026-10-28 目录整理改变源码路径及摘要，旧身份不会自动改写为新身份，严格恢复仍须匹配。

`interface_validation` 只做接口核验；`bootstrap_training` 用于首次学习；`formal_training` 需要完整物理与真实终态证据。初训尚未校准时不触发交接成功奖励。内部凸包的 `diagnostic` 模式只用于初训，正式训练要求 `strict`；未抽查内部间隙的状态保持 `null`，不会伪报通过。

用户指定的首轮训练目录为 `recovery/artifacts/bootstrap-6144-20261022/`，环境数 6144、首轮总目标 200 次更新、每 25 次保存。2026-10-22 23:25（+08:00）后台启动，真实场景包含6144环境，首个PPO更新耗时43.00秒。2026-10-23 00:10:10，完成63次更新后，几何请求编码的 `json.dumps` 抛出CPU `MemoryError`，训练退出；最新持久化模型为 `run/model_50.pt`，第51至63次更新未保存。本轮检查没有重启训练。运行配置、更新日志、checkpoint、逐步统计与启动日志均保存在原目录。

GPU批量几何、状态机、状态缓存、姿态张量池、有界通信及完整日志已接入，独立对抗性复核完成。最终源码116项测试、32项子测试及4/256环境真实接口通过；用户决定保留另一项目训练，暂缓6144环境性能与耐久验收。完整证据见[GPU实现与验证](../../docs/2026-10-23-wheelleg-recovery-gpu-validation.md)。几何实现 `RecoveryPlaneSupportFP64SafeLowerBoundV2` 用FP64支持点下界证明安全，所有可能越界状态均按最多256环境一块，同步交给同帧CPU原生MuJoCo复核。bootstrap内部间隙仍轮转诊断32个环境，formal strict同步检查全部环境。

日志采用37个int64字段、两个固定槽位、每槽两个packet。6144环境的双槽逻辑pinned载荷6.9375MiB，allocator圆整预算8MiB，共享内存另占6.9375MiB；worker保留完整原始记录和分母，槽满背压。CPU逐环境姿态RNG及有序课程保留。优化后的源码与执行身份见各次 `run_manifest.json` 和checkpoint索引，原运行身份不作为新代码身份。完整方案及组件测量见[性能优化方案](../../docs/2026-10-23-wheelleg-recovery-performance-plan.md)；6144环境吞吐、200次更新耐久和起身能力尚未完成验收。原故障时内存历史缺失，不能据此认定为已证实的内存泄漏。

`--import-compatible` 仅用于动作和固定物理契约一致的显式性能兼容导入，证据须绑定父identity、新identity及当前source SHA，并核验geometry、FSM、logging、interface4、interface256产物。该入口不将旧V1动作语义转换为V2；本轮fresh训练不使用此入口。普通V2 `--resume` 继续执行完整严格身份检查。故障路径在Kit关闭前将traceback写入真实stderr和持久化 `training_failure.json`，并以非零状态退出。

MuJoCo模型、采集、校准和评估入口分别运行 `python -m recovery.contact_model --help`、`recovery.collect_pose_bank --help`、`recovery.calibrate --help`、`recovery.evaluate --help`。配置值由各入口dataclass定义，实际采集保留参数计划、完整关节、根状态、速度、真实五帧及前序动作。交接校准使用 `RecoveryCompatibilityCalibrationV2`，以 `recovery_contract_sha256` 绑定起身契约。导出提供 `export_actor`、`golden_inputs_from_pose_bank`、`load_export`，从源V2训练配置、实际V2 checkpoint和至少32条真实姿态记录生成 `RecoveryExportV2` 包；均值直接输出，配置和导出不再包含 `deployment_action_limit`。

以下示例要求该V2运行已经生成第50次更新checkpoint；实际路径与本次运行配置一致。

```powershell
& '.venv/Scripts/python.exe' -m recovery.export --checkpoint recovery/artifacts/my-run/run/model_50.pt --config recovery/artifacts/my-run/resolved_config.json --output-dir recovery/artifacts/my-run/export_50
```

活动回合使用紧凑数组，完成的逐步记录立即归档到分段 gzip JSONL；checkpoint 保存归档索引哈希及删失副本。恢复会核验索引与分段，旧 checkpoint 的统计按保存时的序号截止，避免混入后续训练记录。

## 验证

```powershell
& '.venv/Scripts/python.exe' -m pytest training/recovery/tests --ignore=training/recovery/tests/test_mujoco_recovery.py --ignore=training/recovery/tests/test_geometry_guard.py -q
& 'sim2sim/mujoco/.venv/Scripts/python.exe' -m pytest training/recovery/tests/test_mujoco_recovery.py training/recovery/tests/test_geometry_guard.py -q
```

验证记录位于 `recovery/artifacts/isaac_verification_*`、`mujoco_verification_v2`、`geometry_bootstrap_review` 等目录。完整五类起身、每类至少 100 个独立初态、真实终态校准、交接后连续行走及每类 90% 目标仍须后续实测；接口验证 checkpoint 不能作为能力验收结果。
