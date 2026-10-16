# Action Smooth V1 Implementation Plan

> **For agentic workers:** 使用 executing-plans 在当前会话执行；用户禁止未授权子 agent。

**Goal:** 仅将 action_smooth 加倍，seed41882943 从零训练1000轮，并评估真实运动。
**Architecture:** 新旧具名配置并存，合同通过 schedule+reward 重建，普通 resume 继续禁止换奖励。独立一种子评估不改变正式四组门槛。
**Tech Stack:** Python/PyTorch/RSL-RL、Isaac Lab、MuJoCo、pytest。

- [x] 架构 v0.27 §32/ADR071，保存 design 与本计划，自行核对范围。
- [x] training_profiles.py：TASK_PROFILE_NAMES 增加 stop_reverse_smooth_v1；enabled=name!="legacy_v1"；replace(... action_smooth=-0.02 if name=="stop_reverse_smooth_v1" else -0.01)；重建时 -0.01→旧，-0.02→新，其余拒绝。
- [x] train_dreamwaq.py：choices 增加新 profile。
- [x] run_dreamwaq_training_suite.py：choices/校验增加新 profile，两个 stop_reverse 配置都要求 candidate-only 与动态专项；FORMAL_RUN_COUNT/ITERATIONS 不动。
- [x] test_command_practice.py：新旧配置除 action_smooth 完全相同；非零动作二阶差分的实际 smooth 项恰好加倍，其它奖励项不变；旧/新 round-trip，跨 profile 与未知 smooth 拒绝，重新应用旧配置恢复原权重。
- [x] integration/test_command_practice.py：现有完整训练/加载/导出测试参数化两个 profile，并检查各自 smooth 权重；新 profile 与旧 profile 跨配置恢复拒绝。
- [x] 单测：.venv/Scripts/python.exe -m pytest tests/unit -q --basetemp artifacts/phase2_dreamwaq/action-smooth-acceptance-20261011/unit，所有通过。
- [x] 集成：.venv/Scripts/python.exe -m pytest tests/integration/test_command_practice.py -k stop_reverse_smooth_v1 -q --basetemp artifacts/phase2_dreamwaq/action-smooth-acceptance-20261011/integration；真实 fresh/resume/play/export/golden/当前评估通过，不以测试权重作性能验收。
- [x] 核对 diff、保留旧训练合同与物理哈希；保存验收，提交本次范围文件，不含已有 viewer 改动。
- [x] 在 artifacts 新实验目录写编排脚本：调用现有 _stream_process（1800秒监控）和 _monitor_report；记录新训练、旧模型、异常状态与 source fingerprint，准备独立八场景入口并确认场景执行段完全相同。
- [x] 从零训练：.venv/Scripts/python.exe scripts/train_dreamwaq.py --profile rtx5070 --task-profile stop_reverse_smooth_v1 --seed 41882943 --max-iterations 1000 --run-name dreamwaq-action-smooth2-seed41882943 --headless；实际 run_name 加时间戳，无 --resume。
- [x] 训练后核对 starting_completed_iterations=0、iterations_executed=1000、completed_iterations=1000、fresh provenance、唯一奖励差异、源码冻结及无硬异常。
- [x] 导出/golden（batch1/3/32）与实际 loader；运行 Isaac 当前八场景、独立 MuJoCo 八场景、停车换向专项，逐阶段保存错误与结果。
- [x] 同初态同命令停车10秒对照旧1000/新1000/旧1401：腿目标步间差、实际关节速度/加速度、轮子目标跳变、phi0/pitch与vx误差。
- [x] 结果报告说明改善/退化和是否值得扩到四种子，不自动追加训练或改参数。

执行状态：fresh1000、导出/golden、Isaac 与 MuJoCo 八场景、四个停车换向专项及三模型同条件停车对照均完成。训练无硬异常，phi0_symmetry_degraded 警告仍在。
验收区别：实现与导出功能通过；性能存在退化，新配置不是最终验收合格模型。结果见 ../../2026-10-11-action-smooth-training-results.md。
结论：暂不扩到四种子，不追加训练。保留对照 TensorBoard 6009；不改动既有 viewer/telemetry 工作。
