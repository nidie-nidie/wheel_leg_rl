# WheelLeg debug constrained fine-tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 从 4096 环境 seed 41882943 的 701 更新模型开始，使用实际腿部角加速度窗口超额惩罚和 701 策略参考 KL 约束，执行 20 次 debug-only smoke 微调，并以 MuJoCo 指标判断速度是否改善且腿部不恶化。

**Architecture:** 新增三个 debug 模块：纯数学约束、WheelLeg 环境子类、DreamWaQ PPO 子类；启动器在运行时替换 RSL-RL 算法注册，不修改 source、正式脚本或 YAML。启动器只加载 701 网络权重并创建新 Adam，写入独立 run 目录和侧车 manifest；最终通过现有 MuJoCo 评估程序验证。

**Tech Stack:** Python 3.11、PyTorch 2.7、Isaac Lab、RSL-RL 3.1.2、pytest、现有 MuJoCo evaluator。

> 用户要求 debug-only，因此本计划保存于 wheelleg_dreamwaq/debug/sim2sim/，不写默认的 docs/superpowers/plans/。

---

### Task 1: 实现并测试约束数学

**Files:**
- Create: wheelleg_dreamwaq/debug/sim2sim/constraint_finetune_math.py
- Create: wheelleg_dreamwaq/debug/sim2sim/tests/test_constraint_finetune_math.py

- [ ] **Step 1: Write the failing test**

测试 window_rms、excess_over_budget、diagonal_gaussian_kl 和 NaN 拒绝：
    def test_window_rms_and_excess():
        values = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
        assert torch.allclose(window_rms(values), torch.tensor(2.5))
        assert torch.equal(excess_over_budget(torch.tensor([0.2, 0.1]), 0.15),
                           torch.tensor([0.05, 0.0]))

    def test_gaussian_kl_zero_for_identical_distribution():
        mean = torch.zeros(2, 6)
        std = torch.ones(2, 6)
        assert torch.allclose(diagonal_gaussian_kl(mean, std, mean, std), torch.zeros(2))

    def test_nonfinite_is_rejected():
        with pytest.raises(FloatingPointError):
            validate_constraint_tensor(torch.tensor([float("nan")]), "metric")

- [ ] **Step 2: Run test to verify it fails**

Run:
    $env:PYTHONPATH='E:\wheel_leg_rl-main\wheelleg_dreamwaq\source\wheelleg_dreamwaq'
    E:\wheel_leg_rl-main\wheelleg_dreamwaq\.venv\Scripts\python.exe -m pytest wheelleg_dreamwaq/debug/sim2sim/tests/test_constraint_finetune_math.py -q
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement torch-only functions:
    def window_rms(values):
        return values.square().mean().sqrt()

    def excess_over_budget(rms, budget):
        return (rms - float(budget)).clamp_min(0.0)

    def diagonal_gaussian_kl(mean, std, reference_mean, reference_std):
        ratio = std.square() + (mean - reference_mean).square()
        return 0.5 * (ratio / reference_std.square() - 1.0
                      + 2.0 * (reference_std.log() - std.log())).sum(-1)

    def validate_constraint_tensor(value, name):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN or Inf")

Clamp both standard deviations to 1e-6 and preserve device/dtype.

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command. Expected: all tests PASS.

- [ ] **Step 5: Commit**

    git add wheelleg_dreamwaq/debug/sim2sim/constraint_finetune_math.py wheelleg_dreamwaq/debug/sim2sim/tests/test_constraint_finetune_math.py
    git commit -m "feat: add debug constraint math"

### Task 2: 实现实际角加速度窗口环境

**Files:**
- Create: wheelleg_dreamwaq/debug/sim2sim/constraint_finetune_env.py
- Create: wheelleg_dreamwaq/debug/sim2sim/tests/test_constraint_finetune_env_state.py

- [ ] **Step 1: Write the failing test**

Test a torch-only QddotWindowTracker: selected reset clears only selected rows, and six-joint input uses only the first four controlled leg joints.

- [ ] **Step 2: Run test to verify it fails**

    E:\wheel_leg_rl-main\wheelleg_dreamwaq\.venv\Scripts\python.exe -m pytest wheelleg_dreamwaq/debug/sim2sim/tests/test_constraint_finetune_env_state.py -q
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Write minimal implementation**

QddotWindowTracker maintains a [num_envs, 50, 4] ring buffer, write pointer and count. update accepts [N,6] raw joint acceleration, uses [:,:4], returns per-environment RMS and max(rms - 0.181363117489024, 0). reset(env_ids) clears only selected rows.

ConstrainedWheelLegEnv subclasses WheelLegFlatEnv. It fixes window_steps=50, budget=0.181363117489024 rad/s², penalty_coef=1.0; _get_rewards calls super first, updates from current_state().joint_acceleration, adds -penalty_coef*excess to the rollout reward, and stores detached debug metrics. _reset_idx calls super then clears selected rows. It never changes cfg.reward_weights or formal reward code.

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command. Expected: all tracker tests PASS.

- [ ] **Step 5: Commit**

    git add wheelleg_dreamwaq/debug/sim2sim/constraint_finetune_env.py wheelleg_dreamwaq/debug/sim2sim/tests/test_constraint_finetune_env_state.py
    git commit -m "feat: add debug qddot window constraint"

### Task 3: 实现 reference-KL PPO 子类

**Files:**
- Create: wheelleg_dreamwaq/debug/sim2sim/constraint_finetune_ppo.py
- Create: wheelleg_dreamwaq/debug/sim2sim/tests/test_constraint_finetune_ppo.py

- [ ] **Step 1: Write the failing test**

Use a fake policy and batch to verify reference KL is zero for identical distributions and that set_reference_from_policy creates a frozen deep copy.

- [ ] **Step 2: Run test to verify it fails**

    $env:PYTHONPATH='E:\wheel_leg_rl-main\wheelleg_dreamwaq\source\wheelleg_dreamwaq'
    E:\wheel_leg_rl-main\wheelleg_dreamwaq\.venv\Scripts\python.exe -m pytest wheelleg_dreamwaq/debug/sim2sim/tests/test_constraint_finetune_ppo.py -q
Expected: FAIL because the module does not exist.

- [ ] **Step 3: Write minimal implementation**

ConstrainedDreamWaQPPO subclasses DreamWaQPPO and fixes reference_kl_coef=0.05. set_reference_from_policy deep-copies the current DreamWaQActorCritic to the same device, eval mode, and requires_grad=False. reference_distribution uses _encoding, _effective_mean and _effective_std under no_grad, never sample_context, so it does not consume training RNG. reference_kl computes KL(current || reference) and checks finite values.

Copy the frozen DreamWaQPPO.update implementation into the debug subclass. Keep PPO, value, velocity, reconstruction, context KL, gradient clipping, Adam step, and finite checks identical. Add reference_kl_coef * reference_kl.mean() only to total_loss. Return reference_policy_kl and parameter drift in the update metrics. Do not edit formal ppo.py.

- [ ] **Step 4: Run test to verify it passes**

Run the Step 2 command. Expected: reference KL, freezing, and finite-value tests PASS.

- [ ] **Step 5: Commit**

    git add wheelleg_dreamwaq/debug/sim2sim/constraint_finetune_ppo.py wheelleg_dreamwaq/debug/sim2sim/tests/test_constraint_finetune_ppo.py
    git commit -m "feat: add debug reference policy constraint"

### Task 4: 实现独立 debug 启动器

**Files:**
- Create: wheelleg_dreamwaq/debug/sim2sim/run_constraint_finetune.py
- Create: wheelleg_dreamwaq/debug/sim2sim/constraint_finetune_runbook_20261018.md

- [ ] **Step 1: Write the failing preflight test**

Preflight must reject a missing source checkpoint, a nonmatching SHA, an existing nonempty output directory, or any --resume flag; it must not initialize Isaac.

- [ ] **Step 2: Run preflight to verify the boundary**

With a missing checkpoint, expect a clear failure. With the approved formal 701 checkpoint, expect JSON preflight_ok=true and no CUDA/Isaac process.

- [ ] **Step 3: Write minimal launcher**

Parse formal source checkpoint, with the old debug checkpoint SHA recorded as an exact-equality anchor, 4096 envs, 20 iterations, seed 41882943, rtx5070, cuda:0, and the existing yaw1 headingOFF task profile. Set the same seeds and TF32 flags as formal training. Build the same env and runner configs/contracts. Load only source model_state_dict, never call resume; the runner creates a new Adam. Temporarily wrap registration so formal classes are registered first and rsl_rl.runners.on_policy_runner.DreamWaQPPO points to the debug subclass. Construct ConstrainedWheelLegEnv and the history wrapper, then call set_reference_from_policy after loading weights. Write debug_constraint_manifest.json with source SHA, baseline values, budget, window, KL coefficient, reset_adam=true, command, contract hashes, debug module hashes, and Git HEAD. Write constraint_metrics.jsonl and update_metrics.jsonl. Refuse to overwrite an existing run; on exception write failure.json and never restart.

Use run directory wheelleg_dreamwaq/debug/sim2sim/runs/constraint-finetune-4096-fresh20-20261018. Keep formal checkpoint metadata valid and put debug-only data in sidecars.

- [ ] **Step 4: Write runbook**

Document the one launch command, PYTHONPATH, output files, constants, acceptance thresholds, forbidden actions, and failure handling.

- [ ] **Step 5: Run preflight and CPU import smoke**

    $env:PYTHONPATH='E:\wheel_leg_rl-main\wheelleg_dreamwaq;E:\wheel_leg_rl-main\wheelleg_dreamwaq\source\wheelleg_dreamwaq;E:\wheel_leg_rl-main\wheelleg_dreamwaq\sim2sim\mujoco'
    E:\wheel_leg_rl-main\wheelleg_dreamwaq\.venv\Scripts\python.exe wheelleg_dreamwaq/debug/sim2sim/run_constraint_finetune.py --preflight
Expected: preflight_ok=true; --dry-run verifies registration and manifest without calling learn.

- [ ] **Step 6: Commit**

    git add wheelleg_dreamwaq/debug/sim2sim/run_constraint_finetune.py wheelleg_dreamwaq/debug/sim2sim/constraint_finetune_runbook_20261018.md
    git commit -m "feat: add debug constrained fine-tune launcher"

### Task 5: 执行 20 次更新 smoke

**Files:**
- Create: wheelleg_dreamwaq/debug/sim2sim/runs/constraint-finetune-4096-fresh20-20261018/ (runtime artifacts only)

- [ ] **Step 1: Start the only smoke run**

Set project, source, and MuJoCo PYTHONPATH; set OPENBLAS_NUM_THREADS=1 only for this process. Run the launcher with the approved formal source checkpoint, with the old debug checkpoint SHA recorded as an exact-equality anchor, 4096 envs, max_iterations 20, seed 41882943, rtx5070, cuda:0, the existing task profile, and --headless. Expected: exactly one run, 20 completed updates, model_19.pt, optimizer step 800, finite tensors.

- [ ] **Step 2: Read-only monitoring**

Read the complete process command, stdout/stderr, constraint_metrics.jsonl, update_metrics.jsonl, latest checkpoint, and GPU/commit memory. Do not start another CUDA diagnostic, stop other projects, or restart on failure. On OOM, NaN, contract mismatch, or partial artifacts, write failure report and do not continue.

- [ ] **Step 3: Strict completion check**

Confirm training_summary.json, source SHA, debug manifest, model_19.pt, optimizer step 800, reset-cache and RNG metadata. Use load_dreamwaq_checkpoint_artifact and finite scans.

### Task 6: MuJoCo 验收和报告

**Files:**
- Create: wheelleg_dreamwaq/debug/sim2sim/runs/constraint-finetune-4096-fresh20-20261018/evaluation/
- Create: wheelleg_dreamwaq/debug/sim2sim/runs/constraint-finetune-4096-fresh20-20261018/conclusion.md

- [ ] **Step 1: Export smoke actor**

Use the existing export script into the run export directory and verify actor.ts, policy_manifest.json and golden_vectors.pt with same-batch max error <=1e-7.

- [ ] **Step 2: Run the existing MuJoCo evaluator**

Keep the eight scenarios, 500 control steps, three transition groups, headingOFF/ON modes, and failure conditions; output only inside this run.

- [ ] **Step 3: Recompute comparison metrics**

From CSV recompute headingOFF speed MAE; sustained four-joint angular-acceleration RMS after the first 50 steps and combined 450-step RMS; parking creep, heading drift, actual yaw rate, survival, torque saturation, and transition/reset event arrival. Compare to 701 baseline speed MAE 0.03731354048572227 and sustained RMS 0.17272677856097493. Training reward and KL cannot substitute behavior metrics.

- [ ] **Step 4: Write conclusion**

State whether speed MAE decreased, whether sustained leg RMS stayed within baseline, actual constraint/KL/Adam-reset statistics, failures, and the single-seed/20-update limitation.

- [ ] **Step 5: Commit debug code and runbook, never large runtime checkpoints**

    git add wheelleg_dreamwaq/debug/sim2sim/constraint_finetune_*.py wheelleg_dreamwaq/debug/sim2sim/tests wheelleg_dreamwaq/debug/sim2sim/constraint_finetune_runbook_20261018.md
    git commit -m "test: verify constrained fine-tune smoke"

### Verification checklist

- [ ] Pure math, tracker, and reference-KL pytest tests pass.
- [ ] Formal source/, configs/, and scripts/ are unchanged.
- [ ] 701 network weights are loaded with reset_adam=true and no resume.
- [ ] Exactly one smoke run ends with model_19.pt and optimizer step 800.
- [ ] MuJoCo behavior metrics show speed improvement without leg RMS degradation, or the report states failure clearly.


