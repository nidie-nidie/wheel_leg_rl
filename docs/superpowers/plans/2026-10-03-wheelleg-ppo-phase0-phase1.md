# WheelLeg PPO Phase 0/1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the independent `wheelleg_dreamwaq` Isaac Lab project through Phase 0 asset validation and the Phase 1 asymmetric PPO flat-ground baseline, then run 2000 training iterations and expose the run through TensorBoard.

**Architecture:** The project is an external Python package that depends on an exact Isaac Lab checkout but does not import the old WheelLeg task. One `DirectRLEnv` produces the frozen 25D policy observation, 41D critic observation, 6D action target, rewards, and done signals. The training script constructs RSL-RL 3.1.2 directly, avoiding `isaaclab_tasks` discovery and keeping DreamWaQ code out of Phase 1.

**Tech Stack:** Windows, Python 3.11, uv, Isaac Sim 5.1.0.0, Isaac Lab Git `v2.3.2` commit `37ddf626871758333d6ed89cf64ad702aef127d0`, PyTorch 2.7.0+cu128, RSL-RL 3.1.2, pytest, TensorBoard.

---

### Task 1: Bootstrap the external project

**Files:**
- Create: `wheelleg_dreamwaq/.python-version`
- Create: `wheelleg_dreamwaq/pyproject.toml`
- Create: `wheelleg_dreamwaq/dependency-manifest.toml`
- Create: `wheelleg_dreamwaq/README.md`
- Create: `wheelleg_dreamwaq/scripts/bootstrap.ps1`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/__init__.py`

- [x] Create the directory skeleton and exact dependency declarations.
- [x] Create a clean external Isaac Lab checkout at `wheelleg_dreamwaq/dependencies/IsaacLab-v2.3.2` and verify its commit.
- [x] Run `uv lock` and `uv sync`; introduce only compatibility overrides that the resolver or smoke tests prove necessary.
- [x] Verify Python, torch/CUDA, Isaac Sim, Isaac Lab source path, and `rsl-rl-lib==3.1.2` imports.

Expected verification:

```powershell
uv run python scripts/check_runtime.py
```

The command must print the exact versions, RTX 5070 CUDA capability, and the pinned Isaac Lab commit, then exit 0.

### Task 2: Implement frozen schemas and pure unit tests

**Files:**
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/action.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/command.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/observation.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/frames.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/normalization.py`
- Create: `wheelleg_dreamwaq/tests/unit/test_schemas.py`
- Create: `wheelleg_dreamwaq/tests/unit/test_frames.py`
- Create: `wheelleg_dreamwaq/tests/unit/test_normalization.py`

- [x] Write failing tests for dimensions, slices, joint order, protocol reorder, wheel signs, rotation orthogonality, projected gravity, normalization, and clipping.
- [x] Implement immutable schema constants and pure tensor functions.
- [x] Run the unit suite and require all tests to pass without launching Isaac Sim.

Expected verification:

```powershell
uv run pytest tests/unit -q
```

### Task 3: Implement AssetBundleV1 audit and non-destructive overrides

**Files:**
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/assets/paths.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/assets/asset_contract.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/assets/asset_overrides.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/assets/wheelleg.py`
- Create: `wheelleg_dreamwaq/scripts/audit_asset.py`
- Create: `wheelleg_dreamwaq/tests/unit/test_asset_contract.py`
- Create: `wheelleg_dreamwaq/tests/integration/test_asset_stage.py`

- [x] Verify all five asset paths, sizes, SHA256 values, default prim, root prim paths, articulation root, joint names, rigid bodies, loop closures, and units.
- [x] Load the robot with `UsdFileCfg` using the default prim reference.
- [x] Disable the embedded GroundPlane collision through a runtime stage override before physics initialization; do not modify any source USD.
- [x] Assert exactly one global PhysicsScene and no PhysicsScene under an environment robot namespace.
- [x] Save the Phase 0 audit report as JSON under `wheelleg_dreamwaq/artifacts/phase0/`.

### Task 4: Implement the Phase 1 DirectRLEnv

**Files:**
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env_cfg.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/state.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/control.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/commands.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/observations.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/rewards.py`
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/terminations.py`

- [x] Use `sim_dt=0.01`, `decimation=2`, and a 50 Hz control loop.
- [x] Resolve controlled joints by name in canonical order `[jIJ,jIO,jAB,jAG,jwheel_left,jwheel_right]`.
- [x] Apply four leg position targets and two signed wheel velocity targets with finite software limits.
- [x] Produce normalized `policy:[N,25]` and privileged `critic:[N,41]` observations.
- [x] Resample `[target_vx,target_yaw_rate,target_base_height]` commands without curricula or domain randomization.
- [x] Implement flat-ground tracking and regularization rewards, per-term logging, timeout, low-height, and excessive-tilt termination.
- [x] Reset canonical previous action, commands, root state, and joint state without history or CENet state.

### Task 5: Validate Phase 0 dynamics

**Files:**
- Create: `wheelleg_dreamwaq/scripts/smoke_random_actions.py`
- Create: `wheelleg_dreamwaq/scripts/smoke_zero_actions.py`
- Create: `wheelleg_dreamwaq/tests/integration/test_env_contract.py`

- [x] Launch one environment and record resolved joint/body ordering and default state.
- [x] Run a zero-action standing smoke test.
- [x] Run 1000 random action steps and reject NaN/Inf, software-limit violations, constraint explosions, or reset leakage.
- [x] Adjust only Phase 0 calibration values such as nominal pose, actuator gains, finite limits, and action scales when supported by measured results.
- [x] Freeze the successful values in the Phase 0 audit report and environment config.

### Task 6: Add RSL-RL 3.1.2 PPO integration

**Files:**
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/agents/ppo_cfg.py`
- Create: `wheelleg_dreamwaq/scripts/train_ppo.py`
- Create: `wheelleg_dreamwaq/scripts/play.py`
- Create: `wheelleg_dreamwaq/scripts/tensorboard.ps1`
- Create: `wheelleg_dreamwaq/configs/hardware/portable.yaml`
- Create: `wheelleg_dreamwaq/configs/hardware/rtx4060.yaml`
- Create: `wheelleg_dreamwaq/configs/hardware/rtx5070.yaml`

- [x] Configure observation groups as `policy:[policy]` and `critic:[critic]`.
- [x] Disable RSL-RL empirical actor/critic normalization because NormalizationV1 is explicit.
- [x] Use `[256,128,64]` ELU Actor/Critic networks, 24 rollout steps, PPO clip 0.2, five epochs, adaptive learning rate, and checkpoint interval 100.
- [x] Make hardware profiles change only environment count and mini-batch count.
- [x] Save resolved configs, manifest hashes, seed, versions, and checkpoints in each run directory.

### Task 7: Run verification and short PPO training

**Files:**
- Create: `wheelleg_dreamwaq/tests/integration/test_ppo_startup.py`

- [x] Run all pure unit tests.
- [x] Run asset and environment integration checks.
- [x] Run a 2-iteration PPO startup test with a small environment count and the full RTX 5070 profile.
- [x] Confirm TensorBoard event files and a loadable checkpoint are produced.

Expected verification:

```powershell
uv run pytest tests/unit -q
uv run python scripts/audit_asset.py --headless
uv run python scripts/smoke_random_actions.py --num-envs 16 --steps 1000 --headless
uv run python scripts/train_ppo.py --profile portable --num-envs 32 --max-iterations 2 --headless
```

### Task 8: Train 2000 iterations and launch TensorBoard

- [x] Obtain an independent read-only code review with no blocking P0/P1 findings before launching the formal run.
- [ ] Start the RTX 5070 profile training for exactly 2000 learning iterations.
- [ ] Monitor the process for finite rewards/losses, checkpoint production, and simulator errors.
- [ ] After training exits successfully, launch TensorBoard against `logs/rsl_rl/wheelleg_flat_ppo` on an available localhost port.
- [ ] Open the TensorBoard URL in Codex and report the final run directory, checkpoint, training duration, and key metrics.

Expected command:

```powershell
uv run python scripts/train_ppo.py --profile rtx5070 --max-iterations 2000 --headless
```
