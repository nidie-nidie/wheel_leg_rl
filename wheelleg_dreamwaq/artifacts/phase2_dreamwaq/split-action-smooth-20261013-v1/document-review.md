# 腿轮 action_smooth 分项：独立文档复核

复核日期：2026-10-13。范围：只读核对 Architecture v0.29 状态及 §34、完整实施计划、完整 debug plan、debug 索引新增段，以及当前奖励/profile/合同/checkpoint/export/MuJoCo 接口。没有修改生产代码，没有启动 Isaac 或训练，没有调用其他 agent。唯一写入为本报告。

结论：设计主体可实施，但应先修正文档遗漏的一个 P1：MuJoCo 的策略合同加载器现在只接受 RewardSchemaV2，新导出将带 RewardSchemaV3SplitSmooth，会在评估前直接被拒绝。修正文档并列入最小元数据加载改动/验证后，可以按原范围实施。P0=0；P1=1；P2=1。

## P1：新 reward schema 不能通过现有 MuJoCo loader

证据：

- E:/wheel_leg_rl-main/docs/2026-10-03-wheelleg-dreamwaq-architecture.md:1928 定义新配置使用 RewardSchemaV3SplitSmooth。
- E:/wheel_leg_rl-main/wheelleg_dreamwaq/docs/superpowers/plans/2026-10-13-split-action-smooth.md:30–36 的生产文件清单不含 MuJoCo 合同加载器；第70–72行却要求用现有评估入口加载新导出。
- E:/wheel_leg_rl-main/wheelleg_dreamwaq/scripts/export_dreamwaq_actor.py:77 原样复制 base["schemas"]；第84行原样复制训练权重。因此实现动态 schema 后，导出得到 V3 而非 V2。
- E:/wheel_leg_rl-main/wheelleg_dreamwaq/sim2sim/mujoco/wheelleg_mujoco/contract.py:98–117 对 PPO 和 DreamWaQ 共用 expected_schemas，reward 在第107行固定为 RewardSchemaV2；第115行任一字段不匹配即拒绝。
- E:/wheel_leg_rl-main/wheelleg_dreamwaq/scripts/evaluate_leg_acceleration_mujoco.py:92、scripts/evaluate_command_practice_mujoco.py:34 均先调用这个 loader，尚未创建 runtime 就会失败。

只读复现：在 MuJoCo venv 中读取原 run-03 policy_manifest.json，仅在内存把 reward schema 改为 RewardSchemaV3SplitSmooth 并重新计算 manifest_hash，调用实际 load_policy_contract。实际输出：ValueError: Unsupported policy manifest schema。没有写入修改后的 manifest，也没有执行仿真积分。

建议最小修订：

1. 把 sim2sim/mujoco/wheelleg_mujoco/contract.py 列入实现清单，只扩展训练奖励元数据识别，不修改 ActionV1、history、controller、runner、模型或时序。
2. PPO 继续只接受 RewardSchemaV2；DreamWaQ 可接受 V2 或严格验证后的 V3SplitSmooth。V3 的 action_smooth=0、完整分项字段、受支持的腿/轮权重、tracking_vx 两项和保存 schedule 应一致；未知/不完整/矛盾配置拒绝。
3. 训练前的小流程应将新导出交给真实 load_policy_contract，并检查旧导出继续被接受、PPO 伪装 V3 被拒绝。TorchScript golden 验证本身不能覆盖这个失败。
4. 更新架构/debug文档对新增元数据加载边界的说明；无需更改 MuJoCo 物理参数或动作控制。model_manifest 的 observation/action adapter hashes 只绑定 observation.py/control.py（contract.py:177–192），本项不要求修改这两份适配器或重标模型。

## P2：1 ms 指标的定义和统计窗口仍不够具体

证据：

- 实施计划第72行要求 true1ms q/qdot/qddot，第82行要求 main1ms RMS aggregate 降低50%，但没有定义 qddot 的取值、warmup/window 对齐及 aggregate 顺序。
- scripts/evaluate_leg_acceleration_mujoco.py:119、125–127 明确定义既有20ms qddot为控制步前后 qdot 差除 control_dt；第60–68行是50tick窗口内先求每关节RMS、再平均窗口；第175行再平均四关节。
- wheelleg_mujoco/runner.py:90–94 每控制步执行既有20个1ms积分。只读包装 mj_step 可行，无需修改 runner。
- 实施计划第70行已经冻结20ms warmup100/window50，而1ms部分尚未明确沿用相同时间范围。

建议在 debug 计划中补充确定性口径即可，不扩大生产范围：1ms qddot=(相邻既有物理步的qdot差)/0.001；初始qdot由原reset后读取；只观察已有mj_step，不增加step/mj_forward；舍弃前100控制tick对应的2秒；后8秒按1秒窗口统计（1000个1ms样本），先每关节/每窗口RMS，再对窗口及四关节、三个主场景等权平均，与20ms同一物理时间窗口。per-joint RMS/p90及任何恶化应单独列出。可同时保存 data.qacc 作为另一个诊断字段，但不能把两个口径混为同一指标。

这不是启动实现的架构阻塞；但应在分析脚本落地前定清楚，以免训练后再改变成功标准。

## 其余设计核对

### 奖励数学和旧配置兼容

- rewards.py:16–36 的原 RewardWeights 保留全部字段/默认值；采用独立 frozen 子类可在不改变旧 asdict payload 的情况下增加两项。
- rewards.py:67–74 的当前二阶动作差恰为文档三帧公式；第64–66行是一阶项。分成前四维和后两维求和能在等权时恢复旧值，仅涉及浮点加法顺序差异。
- compute_reward 在 rewards.py:91–94 遍历实际返回 terms 并乘相应权重。因此新配置移除旧 composite key 后，只返回两个分项即可保证无重复计罚；仅把旧系数设0但仍返回旧日志不满足文档“无旧总项”的完整要求。
- env.py:157–161 根据 vars(weights)建立 episode sums；第819–821行按实际 weighted_terms 累加/生成 Reward日志；第912–914行输出全部 episode sums。计划将禁用 composite 从 split episode sums 排除是必要且足够的日志改动。
- 旧 profile 的 dataclass/公式/schema/payload应完全不变。dreamwaq_manifest.py:152只需对实际weights选择schema；第236行asdict已自动保存新增权重；第102–105行hash绑定实际payload。
- training_profiles.py:42–47 当前replace会保留子类类型和新增字段；因此文档要求切回旧 profile 时重建原 RewardWeights 合理。要保留其余原字段值，而不是顺带覆盖用户已有reward配置。

### profile/checkpoint/play/export

- training_profiles.py:59–93 当前保存profile识别主要由schedule及action_smooth决定；新profile必须先识别完整分项schema/字段/权重并验证互斥，不能落入legacy路径。第96–102行保留显式profile与保存profile冲突拒绝即可。
- dreamwaq_checkpoint.py:416–428 用完整current contract严格比较；第451–460行再绑定合同hash。因此无需更改checkpoint张量布局或一般resume验证。
- play_dreamwaq.py:118–125 从保存合同识别并apply profile再构建当前合同，沿用入口可行。
- export_dreamwaq_actor.py:77、84 原样复制schema及训练权重；不需要改导出数学。上面的P1是下游loader遗漏，不是导出需要伪装V2。
- weights_initialization.py:49–59 的白名单不含新profile，确实拒绝为新实验跨奖励初始化；无需增加例外。 fresh1000 与小规模fresh2/resume3验证是两个独立run，不能把验证run接着训至1000充作fresh1000。

### frozen边界和训练身份

- env.py:701–703 现有state三帧动作字段已有；第933–935行reset清零。新奖励不需要新增history或改观测。
- wheelleg_mujoco/control.py:28–40 仍负责原动作clip/腿位置目标/轮速度目标；第42–48行原PD/effort不变。文档要求原策略输出交给这个controller，符合现有接口。
- configs/hardware/rtx5070.yaml 实际num_envs=256、num_mini_batches=8，与文档一致。
- baseline导出manifest的source_completed_iterations=1000，source_checkpoint指向run03 model_999.pt。已读取实际checkpoint文件确认SHA256=3D621280CF7F11088C972B0DD84CF2E698C29A4C8CDA8BC370BB80A2669DF885，与三份文档一致。
- dreamwaq_checkpoint.py:491–496定义completed_iterations=iter+1，与正式1000轮结束于index999一致。
- 旧配置、模型及其hash保留；v0.29是新增具名试验，未重讨论冻结网络数学契约。

### 验收和有限追加试验

- wheelleg_mujoco/evaluation.py:25–34 有八个既有静态场景；evaluate_command_practice_mujoco.py:17–21 有四个专项，非smoke路径在第51行跑500tick。计划引用现有场景，不改变正式四模型排名门。
- command_practice.py:66–120已实现停车误差与换向时间等原门；要求全四专项通过才称“完整控制验收”与实现相符。
- 20ms和1ms两种实际加速度都纳入，且需要目标/饱和/个别关节恶化报告、视觉确认；没有把“动作平滑”偷换成“实际抖动消失”。
- 饱和>=90%、下降50%与速度绝对/相对门明确被称为工程探索门槛，不是硬件安全结论。
- 最多追加腿-0.04、轮-0.01，同种子fresh1000：是有界单变量比较。触发条件是存活和主要速度过关但减抖不足；四专项未全过必须如实保留失败项，不得因此称完整通过。若主要控制恶化，不自动扩大权重。三份文档对此没有矛盾。
- 现有旧模型不是健康阈值，只是同种子对照；一次种子试验仍不能证明统计泛化或所有扰动中的平衡动作保留，文档已保持这个限制。

## 实施前处理清单

- [ ] 修订计划并加入MuJoCo reward schema纯元数据支持和真实loader回归。
- [ ] 明确1ms差分与warmup/window/aggregate，固定后再写数据脚本。
- [ ] 实施、独立代码复核、单元与小流程验证通过后，才启动独立fresh1000。
