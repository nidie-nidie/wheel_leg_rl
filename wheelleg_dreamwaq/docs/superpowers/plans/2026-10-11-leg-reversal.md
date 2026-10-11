# Leg Target Reversal Implementation Plan

> **For agentic workers:** 当前会话使用executing-plans，用户禁止私自调用子agent。

**Goal:** 保留基线奖励，完成四腿换向软约束和12/8次每秒短对照。
**Architecture:** 纯张量模块计算逐腿迟滞事件和滑窗率，环境每控制步缓存成本、每24步完成调权；PPO/CENet/storage不修改。条件化合同/checkpoint扩展和独立weights-only初始化保留普通resume严格性。
**Tech Stack:** Python/PyTorch/RSL-RL3.1.2、Isaac Lab、MuJoCo、pytest。

- [x] 保存protected hash和已有dirty改动；架构v0.28 §33/ADR072，spec自行检查。
- [x] 新增tasks/direct/wheelleg_flat/leg_reversal.py：LegReversalConfig、LegTargetReversalCounter、LegReversalReward；实现spec的0.01rad/50步/逐腿z/24步gain公式及state校验。
- [x] env_cfg.py默认disabled；env.py初始化、_pre_physics_step更新并缓存、_get_rewards加入新成本/日志、_reset_idx清所选行；旧compute_reward和动作路径不变。
- [x] training_profiles.py新增两个显式配置和严格重建；schemas/dreamwaq_manifest.py仅enabled加入task.leg_reversal，disabled与旧hash相等。
- [x] training/dreamwaq_checkpoint.py条件必需gain状态和可选weights初始化身份，严格校验/恢复；不改PPO或runner.learn。
- [x] 新增training/weights_initialization.py：验证源自己的合同，只允许目标添加新约束，strict加载网络，拒绝跨物理/原奖励初始化。
- [x] scripts/train_dreamwaq.py互斥--initialize-from，复用同规模源reset cache，manifest/checkpoint/summary记录初始化身份；所有组optimizer/iteration/monitor/过程RNG新起点。
- [x] tests/unit/test_leg_reversal.py：迟滞、慢移动、停顿、窗口到期、partial reset、零有效窗、gain上限、固定rollout、state往返；tests/unit/test_leg_reversal_checkpoint.py：旧hash/profile重建、非法初始化/metadata、resume。
- [x] tests/integration/probes/leg_reversal.py：真实PhysX每控制步一次、重复读取/partial reset不走时间；tests/integration/test_leg_reversal.py：真实短fresh/resume、weights-only、play/export/golden。
- [x] .venv/Scripts/python.exe -m pytest tests/unit -q；新Isaac集成；git diff/protected hash确认控制/物理/命令/随机化/history不变。
- [ ] 独立实验脚本与artifact目录，三组同源weights-only各200轮，保存来源fingerprint/命令/30分钟监控；正式4×1000套件不改。
- [ ] 每组export/golden、Isaac八场景、MuJoCo八场景和四换向专项，复用原规则；同目标计数和实际关节运动验收。
- [ ] 写结果报告和最终hash核验，结论限定探索，不自动扩大或推送。
