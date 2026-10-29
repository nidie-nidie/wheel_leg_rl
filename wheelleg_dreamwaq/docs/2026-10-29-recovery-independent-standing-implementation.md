# 独立站立实现与验收记录

2026-10-29，按“先独立复核文档、上传现有代码保留历史、再实施”的要求完成。本轮实现新任务及小规模接口检查，正式4096环境训练和700回合最终行为评估尚未执行。

## Git与独立复核

实施前完整代码快照：[`10ba30fc7117789eeabbd0725c97d113f51583b1`](https://github.com/nidie-nidie/wheel_leg_rl/commit/10ba30fc7117789eeabbd0725c97d113f51583b1)，分支为 [`codex/recovery-standing-20261029`](https://github.com/nidie-nidie/wheel_leg_rl/tree/codex/recovery-standing-20261029)。该提交接在原`bd43c9a`之后，原有22个提交均保留；没有重写历史。

独立文档审阅先发现并修正两项P2：reset写入状态后尚无真实接触力，不能要求即时双轮承重；新独立站立任务应使用自己的准入和60/20/20初态规则。实施阶段另做跨模块审阅，修正：

- reset/restore后Isaac COM位姿与root link速度缓存未失效。
- 初态的root link线速度被传入接受COM线速度的写入接口。
- 局部reset探针先推进两步，随后新建诊断缓冲，导致回合长度错开。探针后已真实reset全部环境。
- 训练500次后才检查12个周期测评ID；已移到启动前并共用选择函数。
- 资源检查使用GPU0，而配置可选择别的GPU；本实验已固定`cuda:0`。

终复核未处理P1/P2均为0。[独立复核记录](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/independent-stand-implementation-20261029/independent_review_final.json)为源码审阅结论，完整正式准入仍有下文列出的待补证据。

## 实现内容

新schema为`RecoveryIndependentStandV1`，由现有配置和训练入口显式分发。旧V2/V3路径继续保留。

| 部分 | 实际实现 |
| --- | --- |
| Actor | 五帧×22维＝110维本体观测，256/128/64 ELU前馈网络 |
| Critic | 41维，补充速度、高度、力矩、接触窗口、任务阶段和保持位移 |
| 动作 | 六维tanh高斯；四腿绝对目标±1rad，两轮目标±30rad/s |
| PPO | 存储潜在量u和动作a，稳定Jacobian、变换后熵、解析潜在KL、有限标准差和学习率范围 |
| 参考 | 冻结的560组成对q/h；同一最近索引同时提供腿姿态和高度 |
| 奖励 | 扶正、高度、腿姿态、机身稳定、腿/轮减速、双轮支撑及控制代价分别计算 |
| 持续站立 | 10步真实接触窗口；50步连续合格后进入HOLD；训练保持100步，独立评估保持500步 |
| 初态 | 92个真实物理状态；训练采样为60%倒地、20%站立、20%近站立扰动；训练/测试轨迹与初态分离 |
| 物理 | 复用真实闭链与几何检查；Isaac 5ms×4，MuJoCo 1ms×20；力矩限制18/9Nm |
| 日志 | 每更新和每回合标量、TensorBoard、小checkpoint index；固定最多12环境的失败轨迹，512MiB配额 |
| 恢复 | 策略、Adam、学习率、显式根/全部关节、历史、接触窗口、任务、RNG及诊断半回合 |
| 评估 | 固定12留出姿态周期评估与600+100最终套件；各类成功数、Wilson区间、失败原因和基础姿态相关性 |

该任务使用普通PPO，辅助力为零，不构建成本网络；401只提供参考数据，不导入其Actor或优化器。PhysX求解器的接触冲量缓存没有保存，恢复报告明确采用显式物理状态恢复语义。

主要文件位于[独立训练入口](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/independent_train.py)、[配置](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/independent_config.py)、[环境](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/independent_env.py)、[PPO](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/independent_ppo.py)、[奖励](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/independent_rewards.py)、[状态机](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/independent_state.py)、[准入](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/independent_preflight.py)和[CPU测评器](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/training/recovery/independent_evaluate.py)。

## 已执行的验证

受验source SHA256：`14388C314C12C225FDAE57C99A44E416C8E468C2EA963324F2C0244AFDDBE00C`。

| 检查 | 真实结果 |
| --- | --- |
| 实施前基线 | 原225项通过 |
| 最终完整单元验收 | 主环境246项（186个testcase＋60个subtest），MuJoCo环境25项；总271项，失败/错误/跳过均0 |
| 401门控校准 | 20段，第2秒后覆盖100%，每段连续28秒，完整评估保持通过 |
| 旧6200反例 | 12段共4066帧均违反至少一项必要运动学条件；旧接触身份限制明确记录 |
| 新真实支撑反例 | 单轮、机身支撑、快速滑动、倒置、低位共5×60帧，完整成功误接受0 |
| 动作边界 | 6类×6关节×正负边界＝72例通过；所记录硬物理失败0 |
| 实际Isaac PPO | 4环境，先2次更新→保存→真实推进→恢复→再2次更新；共384个训练环境步骤 |
| 恢复检查 | 当前观测精确恢复，恢复后的两次更新继续改变策略参数 |
| 局部reset | 真实推进后只reset env0，另3环境物理状态、历史、窗口及任务计数精确不变 |
| 导出一致性 | 同一批实际110维观测，CPU TorchScript最大动作误差8.94e−8 |
| 最终日志 | update1..4连续有限，提交=消费=4，pending0；同步fsync结束，cutoff/尾部哈希匹配 |
| TensorBoard | `Train/mean_step_reward`确实记录update1..4；没有完整回合，episode_count为0 |
| CPU导出接口 | 本次4更新Actor完成前倾/站立两个留出回合，成功0/2、均起身超时；用于检查完整调用，不作学习结论 |
| 进程退出 | 实际接口检查exit_code0，`training_verification.passed=true` |

实际4环境入口计时16.77秒，PPO rollout/optimization累计10.82秒。当前日志中的硬物理失败计数为空。一次先前接口检查因诊断回合长度错位以exit_code1退出，证据保留于`isaac-interface-v2`和`isaac-supervisor-v2`；修复后使用全新v3目录，没有覆盖失败记录。

证据入口：[综合实施验收](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/independent-stand-implementation-20261029/implementation_acceptance.json)、[单元验收](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/independent-stand-implementation-20261029/final-unit-v2/verification.json)、[实际4环境报告](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/independent-stand-implementation-20261029/isaac-interface-v3/training_verification.json)、[站立校准](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/independent-stand-implementation-20261029/standing_calibration.json)、[72例动作边界](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/independent-stand-implementation-20261029/action_physical_check.json)。

关键文件SHA256：

| 文件 | SHA256 |
| --- | --- |
| 站立校准 | `29DD81F4321696E7A655EED6F092AD9331EC8ECD330BC140B9B361C360F43589` |
| 独立终复核 | `1B2221E0FDEFB76338129F55CC2AA39818FC101E2A6D98C8F2ABEDB8377E41D9` |
| 完整单元验收 | `564223ECDF3DC44E8BE31E6577F59513E744FC086C93267DBD35D88C4169CB5A` |
| 实际4环境报告 | `3649ACFEFC2796E0F417A673345EFB9EBCBE5437950172D3DE2BC42895CD94D8` |
| 接口model_4.pt | `AF909E23918CA695C2F9E20BAEFB1E9EA43281362CC7186B2A671E8309FD997B` |

模型与日志位于[isaac-interface-v3](/E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/independent-stand-implementation-20261029/isaac-interface-v3)，TensorBoard目录为其下`tensorboard`。模型只有4次接口更新，不是候选站立策略。Git提交包含源码、依赖锁和说明；本机采集数据、模型与运行输出依现有gitignore保存在artifact目录，测试需要这些实际数据。

## 正式训练前仍需完成

- 四环境在线覆盖完整HOLD成功/中断和完整终态日志。本次每环境96个训练控制步骤尚无完整回合；状态边界检查与实际401轨迹校准不能替代这一项。
- 汇总401、6200与六类倒地初态的完整分项奖励分布，检查近零项和量级。
- 取得真实4096环境资源测量并组装专用正式准入证据。当前4环境报告不能改标为4096环境资源报告。
- 单独启动固定4096环境×4000次更新实验，然后执行600+100最终评估；报告训练完成与行为达标各自结论。

成功门控校准通过只说明本次参考和反例的判据计算。当前没有新策略自主起身、站稳或行走交接达标的证据。

## 复核命令

以下命令只核验本次小型证据与模型，不读取旧20GB历史归档，也不启动训练：

```powershell
$env:PYTHONPATH='E:/wheel_leg_rl-main/wheelleg_dreamwaq/training;E:/wheel_leg_rl-main/wheelleg_dreamwaq/sim2sim/mujoco;E:/wheel_leg_rl-main/wheelleg_dreamwaq'
& 'E:/wheel_leg_rl-main/wheelleg_dreamwaq/.venv/Scripts/python.exe' 'E:/wheel_leg_rl-main/wheelleg_dreamwaq/debug/sim2sim/verify_independent_standing_implementation.py' --artifact 'E:/wheel_leg_rl-main/wheelleg_dreamwaq/recovery/artifacts/independent-stand-implementation-20261029'
```

新任务配置通过`debug/sim2sim/prepare_independent_standing_config.py`生成，必须绑定实际参考库、校准、姿态库、几何配置与全新运行目录。`--formal`只生成正式配置草案；正式入口会检查独立准入，缺失证据时拒绝启动。
