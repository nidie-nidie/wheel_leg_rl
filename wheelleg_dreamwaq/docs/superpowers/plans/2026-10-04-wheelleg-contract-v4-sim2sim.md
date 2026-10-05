# WheelLeg Contract V4 and MuJoCo Sim2Sim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase1ContractV4 with episode-held commands, NormalizationV2, true-geometry `phi0` symmetry reward, AssetBundleV2, and a self-contained MuJoCo 3.14.0 sim2sim/evaluation path, then run four independent 1000-iteration PPO trainings and select the best checkpoint headlessly.

**Architecture:** Keep Isaac Lab task code independent of MuJoCo and keep MuJoCo runtime independent of Isaac Lab. Build all cross-engine behavior from shared pure-Python schemas and frozen JSON/manifests. Preserve AssetBundleV1 and V3 replay while making new training default to AssetBundleV2 and ContractV4.

**Tech Stack:** Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.2, RSL-RL 3.1.2, PyTorch 2.7.0 cu128, OpenUSD/PhysX, MuJoCo 3.14.0, pytest, TensorBoard.

**Execution note:** The workspace has no Git metadata, so verification checkpoints replace commit steps. The user already selected inline execution in this task.

---

### Task 1: Freeze Contract V4 schemas and unit behavior

**Files:**
- Modify: `pyproject.toml`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/command.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/normalization.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/manifest.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/commands.py`
- Modify: `tests/unit/test_normalization.py`
- Modify: `tests/unit/test_manifest.py`
- Create: `tests/unit/test_commands.py`

- [x] **Step 1: Add failing tests for CommandSamplingV2**

Test deterministic mode masks and values using a seeded `torch.Generator`: stand forces `vx=yaw=0`; straight forces `yaw=0`; rotate forces `vx=0`; combined permits both; all heights stay in `[0.16,0.24]`; sampled `vx/yaw` stay in `[-1.5,1.5]` and `[-1,1]`.

- [x] **Step 2: Add failing tests for NormalizationV2 and ContractV4**

Assert `[-1.5,+1.5]` maps to `[-1,+1]`, `[-1,+1] yaw` maps to `[-1,+1]`, nominal height maps to zero, and V3/V4 contracts reject each other.

- [x] **Step 3: Run the focused tests and confirm failure**

Run:

```powershell
uv run pytest tests/unit/test_commands.py tests/unit/test_normalization.py tests/unit/test_manifest.py -q
```

Expected: failures for missing V2 types/version fields.

- [x] **Step 4: Implement schema versions and sampler**

Implement:

```python
COMMAND_SAMPLING_VERSION = "CommandSamplingV2"
NORMALIZATION_SCHEMA_VERSION = "NormalizationV2"
PHASE1_CONTRACT_VERSION = "Phase1ContractV4"
```

Use one categorical mode draw with probabilities `(0.20, 0.30, 0.20, 0.30)` and independent continuous uniform draws inside the selected mode. Remove time-based resampling from the schema; commands are reset-only.

- [x] **Step 5: Run focused and all pure unit tests**

```powershell
uv run pytest tests/unit/test_commands.py tests/unit/test_normalization.py tests/unit/test_manifest.py -q
uv run pytest tests/unit -q
```

Expected: all pass.

### Task 2: Build and select AssetBundleV2 without touching V1

**Files:**
- Create: `sim2sim/mujoco/pyproject.toml`
- Create: `sim2sim/mujoco/uv.lock`
- Create: `scripts/create_asset_bundle_v2.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/assets/paths.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/assets/asset_contract.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/assets/asset_overrides.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/assets/wheelleg.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/assets/__init__.py`
- Modify: `tests/unit/test_asset_contract.py`
- Modify: `tests/integration/test_asset_stage.py`
- Create: `artifacts/phase1_v4/groundplane-layer-audit.json` (generated)
- Create: `artifacts/phase1_v4/asset-bundle-v2-manifest.json` (generated)
- Create: `../wheel_leg_urdf4_usd_v2/wheel_leg_urdf4/**` (generated copy)

- [x] **Step 1: Add failing V2 contract tests**

Assert V1 hashes remain unchanged, V2 has five frozen files, V2 has no embedded-ground path, and `asset_root("AssetBundleV2")` resolves the copy.

- [x] **Step 2: Implement the V2 builder**

The script must:

1. Verify V1 before copying.
2. Copy all five files into a new self-contained root.
3. Open the composed V2 stage through OpenUSD.
4. Record `/wheel_leg_urdf4/GroundPlane` prim stack and authoring layer.
5. Remove the prim from its actual authoring layer via USD API.
6. Reopen and assert both GroundPlane paths are absent.
7. Hash all five V2 files and write the audit/manifest.
8. Re-verify V1 hashes after completion.

- [x] **Step 3: Generate AssetBundleV2**

```powershell
uv run python scripts/create_asset_bundle_v2.py --headless
```

Expected: V2 directory and both JSON artifacts exist; V1 hashes are unchanged.

- [x] **Step 4: Implement version-aware loading**

Default new environment construction to V2. Remove the V1-only embedded-ground disable call from V4 runtime. Keep wheel-only collision override and assert the embedded GroundPlane prim is absent.

- [x] **Step 5: Verify asset tests**

```powershell
uv run pytest tests/unit/test_asset_contract.py -q
uv run pytest tests/integration/test_asset_stage.py -q
```

Expected: all pass and exactly two robot collision shapes remain enabled.

### Task 3: Implement true virtual-leg geometry and `phi0` reward

**Files:**
- Create: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/kinematics/__init__.py`
- Create: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/kinematics/virtual_leg.py`
- Create: `scripts/audit_virtual_leg_geometry.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/state.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/rewards.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env_cfg.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py`
- Modify: `scripts/smoke_random_actions.py`
- Modify: `tests/unit/test_phase1_math.py`
- Create: `tests/unit/test_virtual_leg.py`
- Modify: `tests/integration/test_env_contract.py`
- Create: `artifacts/phase1_v4/usd-anchor-audit.json` (generated)

- [x] **Step 1: Add failing pure-math tests**

Test `L0=sqrt(s^2+d^2)`, `phi0=atan2(d,s)`, wrap-to-pi continuity, left/right equality, and a known nominal reference near `L0=0.199507 m`, `phi0=90.54 deg`.

- [x] **Step 2: Implement pure virtual-leg math**

Provide tensor functions for body-frame `I -> W` vectors, `L0`, `phi0`, wrapped difference, symmetry error, and direct FK diagnostic errors. Keep these functions free of Isaac imports.

- [x] **Step 3: Generate and validate the USD anchor audit**

The Isaac script must resolve named hip/wheel joint frames through USD APIs, record body-local anchors and axes, and fail if either side is missing or ambiguous.

```powershell
uv run python scripts/audit_virtual_leg_geometry.py --headless
```

- [x] **Step 4: Integrate state capture and reward**

Resolve hip/wheel body IDs once. Transform frozen local anchors through current body poses, rotate `W-I` into the base frame, compute true left/right `L0/phi0`, add them to `WheelLegState`, and add:

```python
phi0_symmetry = wrap_to_pi(phi0_left - phi0_right).square()
weight_phi0_symmetry = -1.0
```

The existing reward loop supplies the common `control_dt` multiplier.

- [x] **Step 5: Add logging and runtime guards**

Log true left/right angles, absolute wrapped difference, L0, direct FK diagnostics, and loop residual. Fail on non-finite geometry, `L0 <= 0.05`, loop residual over 5 mm, or direct diagnostic errors over 5 mm/5 mm/3 degrees.

- [x] **Step 6: Verify unit and Isaac smoke tests**

```powershell
uv run pytest tests/unit/test_virtual_leg.py tests/unit/test_phase1_math.py -q
uv run python scripts/smoke_random_actions.py --num-envs 16 --steps 1000 --output artifacts/phase1_v4/random-action-v4.json --headless
uv run pytest tests/integration/test_env_contract.py -q
```

Expected: finite values, no reset leaks, loop/FK thresholds pass, reward contains `phi0_symmetry`.

Implementation result: the frozen V4 physics configuration is `sim_dt=0.005`, `decimation=4`, articulation solver `96/4`; this preserves `control_dt=0.02`. The final 16-env/1000-step gate passed with maximum loop/FK errors below the fixed limits.

### Task 4: Complete environment ContractV4 integration

**Files:**
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env_cfg.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/manifest.py`
- Modify: `scripts/train_ppo.py`
- Modify: `scripts/play.py`
- Modify: `scripts/audit_asset.py`
- Modify: `README.md`
- Modify: `tests/integration/test_ppo_startup.py`

- [x] **Step 1: Remove in-episode command resampling**

Delete `_command_resample_steps`, `_resample_due_commands`, and the `step` override that changes commands. Reset remains the sole sampler. Fixed/keyboard play continues to overwrite only `CommandV1`.

- [x] **Step 2: Freeze V4 contract fields**

Include AssetBundleV2 hash/version, WheelOnlyCollisionV2, CommandSamplingV2 probabilities/ranges/episode hold, NormalizationV2, RewardSchemaV2, VirtualLegKinematicsV1, and the fixed `phi0` weight in the contract hash.

- [x] **Step 3: Preserve V3 rejection semantics**

Do not silently load V3 checkpoints in V4. Error messages must identify contract version/hash mismatch before policy construction.

- [x] **Step 4: Run full training-side verification**

```powershell
uv run pytest tests/unit -q
uv run pytest tests/integration -q
uv run python scripts/train_ppo.py --profile portable --num-envs 64 --max-iterations 2 --seed 401 --run-name v4-smoke --headless
```

Expected: tests pass; smoke run writes event file, manifest, and checkpoint with ContractV4.

### Task 5: Build the self-contained MuJoCo model and adapters

**Files:**
- Create: `sim2sim/mujoco/README.md`
- Create: `sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml` (generated)
- Create: `sim2sim/mujoco/models/meshes/**` (generated copy)
- Create: `sim2sim/mujoco/model_manifest.json` (generated)
- Create: `sim2sim/mujoco/wheelleg_mujoco/__init__.py`
- Create: `sim2sim/mujoco/wheelleg_mujoco/model_map.py`
- Create: `sim2sim/mujoco/wheelleg_mujoco/observation.py`
- Create: `sim2sim/mujoco/wheelleg_mujoco/control.py`
- Create: `sim2sim/mujoco/wheelleg_mujoco/metrics.py`
- Create: `sim2sim/mujoco/wheelleg_mujoco/runner.py`
- Create: `scripts/build_mujoco_model.py`
- Create: `tests/sim2sim/test_mujoco_model.py`
- Create: `tests/sim2sim/test_mujoco_adapters.py`
- Create: `artifacts/phase1_v4/mujoco-dummy-dynamics-audit.json` (generated)

- [x] **Step 1: Add failing model-identity tests**

Load the generated XML with MuJoCo and assert:

```text
nbody=28, njnt=27, nq=33, nv=32, nu=6, neq=8, npair=2
```

Assert exactly two named floor-wheel pairs and both pair sliding coefficients equal `[1.0,1.0]`.

- [x] **Step 2: Create the isolated MuJoCo runtime project**

Pin Python 3.11, `mujoco==3.14.0`, `torch==2.7.0`, `numpy`, and `pytest` in `sim2sim/mujoco/pyproject.toml`. This subproject has its own `.venv` and `uv.lock`; it must not depend on the root Isaac project package because Isaac Sim 5.1.0 pins `websockets==12.0` while MuJoCo 3.14.0 requires `websockets>=13`.

- [x] **Step 3: Implement the model builder**

Use the local source XML:

```text
E:\wheel_leg_rl-main\wheel_leg_debug-main2\mujoco_control_extract\sim\models\wheel_leg_urdf4_self_mesh_all.xml
```

Copy nonzero STL files from `wheel_leg_urdf4_mjcf/wheel_leg_urdf4/meshes`, replace `base_link_original.obj` with visual-only `base_link.STL`, remove the desert texture, disable all automatic collision masks, and create only two explicit floor-wheel pairs.

Consume `artifacts/phase1_v4/mujoco-usd-data.json`, generated in the Isaac environment by a read-only OpenUSD export script, and generate the 12 passive dummy body/joint branches. Treat USD authored `q=0` as model reference only, calculate each nested body transform as `inverse(T_root_parent) @ T_root_child`, transform joint frames/axes into child-local coordinates, and write ContractV4 reset values as a named keyframe.

- [x] **Step 4: Compile and audit the model**

```powershell
uv sync --project sim2sim/mujoco
uv run --project sim2sim/mujoco python scripts/build_mujoco_model.py
uv run --project sim2sim/mujoco pytest tests -q
```

Expected: exact dimensions, mass `4.396253988 kg` within `1e-6`, six motors, no non-wheel contacts, and generated audit/manifest.

- [x] **Step 5: Implement shared observation and action adapters**

Use `R_control_from_mujoco=[[0,-1,0],[1,0,0],[0,0,1]]`, `data.xipos[base]` for COM height, `mj_jacBodyCom` for COM velocity/angular velocity, fixed NormalizationV2, canonical name/sign mapping, four position PD channels, and two wheel velocity channels.

- [x] **Step 6: Add adapter golden tests**

Test nonzero pose/angular velocity, freejoint-origin versus COM separation, canonical wheel signs, action clipping/scales, torque saturation, and 20 physics steps per policy tick.

```powershell
uv run --project sim2sim/mujoco pytest tests -q
```

Expected: all pass.

Implementation result: MuJoCo compiles with `(nbody,njnt,nq,nv,nu,neq,npair)=(28,27,33,32,6,8,2)` and total mass `4.396253988146782 kg`. The 454256-face base STL is losslessly partitioned into three visual meshes, two USD quantization-scale inertia corrections are explicitly audited, and USD/MuJoCo visual AABB center/size errors are below `1e-6 m`.

### Task 6: Export PPO actors and implement deterministic headless evaluation

**Files:**
- Create: `scripts/export_ppo_actor.py`
- Create: `scripts/evaluate_mujoco.py`
- Create: `scripts/run_training_suite.py`
- Create: `tests/sim2sim/test_evaluation_metrics.py`
- Modify: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/training/checkpoint.py`
- Modify: `scripts/train_ppo.py`

- [x] **Step 1: Implement TorchScript actor export**

Load and contract-validate one PPO checkpoint, export only the deterministic actor, and write `policy_manifest.json` containing checkpoint/contract/schema hashes, network sizes, NormalizationV2, joint order, q nominal, action scales, frames, and time periods.

- [x] **Step 2: Add evaluation metric tests**

Test yaw unwrap, per-scenario equal weighting, applicable scenario subsets, empty-prefix `+inf`, survival-first failed-run ranking, and tie-break order.

- [x] **Step 3: Implement eight deterministic 10-second scenarios**

Each scenario starts from the same ContractV4 keyframe with zero action history. Record per-tick CSV and summary JSON for height, roll/pitch/yaw, yaw-rate, `vx`, `phi0`, saturation, closure residual, and survival.

- [x] **Step 4: Implement four-run orchestration**

Generate four distinct random seeds, write `training-suite-manifest.json`, run four fresh 1000-iteration trainings sequentially, record subprocess logs/return codes, export each final actor, evaluate all four, and write a ranked suite summary. Never resume or warm-start between runs.

- [x] **Step 5: Verify export/evaluation with smoke checkpoints**

```powershell
uv run --project sim2sim/mujoco pytest tests/test_evaluation_metrics.py -q
$runDir = Get-Content -LiteralPath artifacts/phase1/latest_run.txt
$checkpoint = Get-ChildItem -LiteralPath $runDir -Filter 'model_*.pt' |
    Sort-Object { [int]($_.BaseName -replace 'model_','') } |
    Select-Object -Last 1
uv run python scripts/export_ppo_actor.py --checkpoint $checkpoint.FullName --output artifacts/phase1_v4/export-smoke
uv run --project sim2sim/mujoco python scripts/evaluate_mujoco.py --policy ../../artifacts/phase1_v4/export-smoke/actor.ts --manifest ../../artifacts/phase1_v4/export-smoke/policy_manifest.json --smoke
```

Expected: actor outputs match the RSL-RL inference policy for fixed 25D vectors and headless smoke emits finite CSV/JSON.

Implementation result: deterministic actor export matches RSL-RL inference exactly on 32 seeded vectors. A one-run/one-iteration orchestration smoke completed training, checkpoint export, all eight MuJoCo scenarios, survival-first ranking, and TensorBoard scalar artifact generation without resuming.

Review hardening result: the suite now compares a seed-independent training fingerprint before exporting later runs; formal MuJoCo reports carry a frozen evaluation-contract hash and reject smoke/mixed/duplicate inputs; the model manifest/runtime validate a recomputed compiled-dynamics semantic hash; and TensorBoard anomalies feed a blocking final monitor gate. A two-run independent-seed orchestration smoke completed the full training/export/eight-scenario/ranking path with matching fingerprints.

### Task 7: Independent code review gate

**Files:**
- Review all files changed/created by Tasks 1-6.

- [x] **Step 1: Run the complete pre-review verification**

```powershell
uv run pytest tests/unit -q
uv run pytest tests/integration -q
uv run --project sim2sim/mujoco pytest tests -q
uv run python scripts/smoke_random_actions.py --num-envs 16 --steps 1000 --output artifacts/phase1_v4/random-action-v4.json --headless
```

- [ ] **Step 2: Invoke an independent read-only code review agent**

The reviewer must inspect code, generated V2 asset/manifests, the MuJoCo model, tests, and the design docs. Any P0/P1 blocks training. Fix and re-review until the result explicitly states no blocking P0/P1.

### Task 8: Four formal trainings, monitoring, headless selection, TensorBoard

**Files:**
- Generate: `artifacts/phase1_v4/training-suite-manifest.json`
- Generate: four `logs/rsl_rl/wheelleg_flat_ppo/**` run directories
- Generate: `artifacts/phase1_v4/evaluation/**`
- Generate: `artifacts/phase1_v4/training-suite-summary.json`

- [ ] **Step 1: Launch the formal suite**

```powershell
uv run python scripts/run_training_suite.py --profile rtx5070 --iterations 1000 --runs 4
```

- [ ] **Step 2: Monitor every active training at approximately 30-minute intervals**

Inspect process state, logs, event scalars, episode length, termination counts, NaN/Inf, entropy, value loss, tracking errors, action/effort saturation, and `phi0` metrics. Save each inspection in the suite artifact directory. If a run completes before 30 minutes, perform the complete post-run scalar scan instead of waiting.

- [ ] **Step 3: Evaluate and rank all four actors**

Run the fixed MuJoCo scenario suite, preserve every CSV/JSON, and rank full-survival models by the frozen score. If all runs fail, mark Phase 1 failed and report the least-bad diagnostic candidate without calling it a qualified best model.

- [ ] **Step 4: Open one TensorBoard over all four runs**

```powershell
uv run tensorboard --logdir logs/rsl_rl/wheelleg_flat_ppo --host 127.0.0.1 --port 6007
```

Expected: TensorBoard is reachable at `http://127.0.0.1:6007`, suite summary identifies the selected checkpoint or explicitly records Phase 1 failure, and no required process remains unattended.
