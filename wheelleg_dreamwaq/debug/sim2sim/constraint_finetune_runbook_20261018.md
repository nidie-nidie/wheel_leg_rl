# Debug-only constrained fine-tune runbook

唯一 smoke 目录：
wheelleg_dreamwaq/debug/sim2sim/runs/constraint-finetune-4096-fresh20-20261018

输入是已严格验收的正式复现 model_700.pt；它与旧 debug model_700.pt 逐位相同，旧 debug SHA 写在 debug_constraint_manifest.json。训练只加载网络权重，reset_adam=true，不调用 resume。

约束常数：
- 50 个控制步窗口，约 1 秒
- 701 基线持续 RMS 0.17272677856097493 rad/s²
- 训练预算 0.181363117489024 rad/s²
- qddot penalty coefficient 1.0
- reference policy KL coefficient 0.05

启动前：
    $env:PYTHONPATH='E:\wheel_leg_rl-main\wheelleg_dreamwaq;E:\wheel_leg_rl-main\wheelleg_dreamwaq\source\wheelleg_dreamwaq;E:\wheel_leg_rl-main\wheelleg_dreamwaq\sim2sim\mujoco'
    E:\wheel_leg_rl-main\wheelleg_dreamwaq\.venv\Scripts\python.exe wheelleg_dreamwaq/debug/sim2sim/run_constraint_finetune.py --preflight

唯一启动：
    $env:OPENBLAS_NUM_THREADS='1'
    E:\wheel_leg_rl-main\wheelleg_dreamwaq\.venv\Scripts\python.exe wheelleg_dreamwaq/debug/sim2sim/run_constraint_finetune.py --headless

输出：
- debug_constraint_manifest.json
- constraint_metrics.jsonl
- update_metrics.jsonl
- training_summary.json
- model_19.pt
- final_model_path.txt
- failure.json（仅失败时）

禁止修改正式 source/config/scripts，禁止 --resume、重复启动、追加训练或停止其他项目。训练失败时保留 failure.json 和已有工件，不自动重启。

