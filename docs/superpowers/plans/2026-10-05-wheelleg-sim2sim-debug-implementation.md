# WheelLeg Sim2Sim Debug Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, auditable Isaac Sim / MuJoCo trace pipeline that finds the first meaningful divergence for the frozen Run 01 standing policy without changing production training or simulation behavior.

**Architecture:** Add a pure NumPy contract/scenario/comparison layer under `wheelleg_dreamwaq/debug/sim2sim`, plus one collector per simulator. The Isaac collector uses a debug-only `WheelLegFlatEnv` subclass that preserves the production `DirectRLEnv.step()` order while observing pre/post substep state; the MuJoCo collector reads the existing frozen runtime modules directly and does not modify them. Both collectors exchange versioned JSON/NPZ artifacts only.

**Tech Stack:** Python 3.11, NumPy, PyTorch/TorchScript, Isaac Lab 2.3.2 / Isaac Sim 5.1.0, MuJoCo 3.14.0, pytest.

**Repository note:** `E:\wheel_leg_rl-main` and `wheelleg_dreamwaq` currently have no `.git` metadata. Commit steps are replaced by SHA256 checkpoints and test logs; no history operation is possible in this workspace.

---

### Task 1: Pure trace schema and file integrity

**Files:**
- Create: `wheelleg_dreamwaq/debug/__init__.py`
- Create: `wheelleg_dreamwaq/debug/sim2sim/__init__.py`
- Create: `wheelleg_dreamwaq/debug/sim2sim/trace_schema.py`
- Create: `wheelleg_dreamwaq/debug/sim2sim/tests/test_trace_schema.py`

- [ ] **Step 1: Write failing schema tests**

Test fixed dimensions, numeric-only NPZ arrays, canonical/native right-wheel round-trip, missing-field rejection, file hashes, and substep continuity:

```python
def test_controlled_native_round_trip():
    native = np.array([1, 2, 3, 4, 5, -6], dtype=np.float64)
    canonical = engine_native_to_canonical(native)
    np.testing.assert_array_equal(canonical, [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(canonical_to_engine_native(canonical), native)

def test_substep_continuity_rejects_gap():
    trace = valid_substep_trace(rows=2)
    trace["active_joint_position_canonical_pre_step"][1, 0] += 0.1
    with pytest.raises(ValueError, match="substep continuity"):
        validate_substep_trace(trace, physics_steps_per_action=2)
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
E:\wheel_leg_rl-main\wheelleg_dreamwaq\.venv\Scripts\python.exe -m pytest wheelleg_dreamwaq\debug\sim2sim\tests\test_trace_schema.py -q
```

Expected: import failure because `trace_schema.py` does not exist.

- [ ] **Step 3: Implement the schema**

Implement constants and pure functions with no simulator imports:

```python
SCHEMA_VERSION = "Sim2SimDebugTraceV1"
CONTROLLED_SIGNS = np.array((1, 1, 1, 1, 1, -1), dtype=np.float64)

def engine_native_to_canonical(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[-1] != 6:
        raise ValueError("expected six controlled values")
    return values * CONTROLLED_SIGNS.astype(values.dtype, copy=False)

def canonical_to_engine_native(values: np.ndarray) -> np.ndarray:
    return engine_native_to_canonical(values)
```

Add `sha256_file`, stable sorted JSON writing, non-overwriting run-directory creation, NPZ writing/loading without pickle, metadata hash verification, exact required field/shape/dtype validation, and separate reset/control/substep validators.

- [ ] **Step 4: Run schema tests**

Expected: all schema tests pass.

- [ ] **Step 5: Record SHA checkpoint**

Run `Get-FileHash` for the four created files and save output in the implementation log.

### Task 2: Frozen standing scenario and action sequences

**Files:**
- Create: `wheelleg_dreamwaq/debug/sim2sim/stand_scenario.py`
- Create: `wheelleg_dreamwaq/debug/sim2sim/tests/test_stand_scenario.py`

- [ ] **Step 1: Write failing deterministic-scenario tests**

Cover exact command `[0, 0, 0.20]`, 500 ticks, zero previous action, immutable frozen hashes, deterministic scenario hash, zero action, six channel pulses, and action replay validation.

- [ ] **Step 2: Implement `StandScenarioV1`**

Use a frozen dataclass and sorted JSON payload:

```python
@dataclass(frozen=True)
class StandScenarioV1:
    name: str = "nominal_stand_debug_v1"
    control_ticks: int = 500
    control_dt_s: float = 0.020
    command: tuple[float, float, float] = (0.0, 0.0, 0.20)
    random_seed: int = 0
```

The generator must verify the fixed actor, checkpoint, policy-manifest, model-manifest, XML, asset and contract hashes before writing `scenario.json` and `action_sequences.npz`.

- [ ] **Step 3: Run scenario tests**

Expected: deterministic hashes and exact action shapes pass.

### Task 3: Pure comparison and first-divergence reporting

**Files:**
- Create: `wheelleg_dreamwaq/debug/sim2sim/compare_traces.py`
- Create: `wheelleg_dreamwaq/debug/sim2sim/tests/test_compare_traces.py`

- [ ] **Step 1: Write failing synthetic comparison tests**

Cover reset tick 0, right-wheel sign error, previous-action one-frame shift, quaternion sign equivalence, passive-hinge first divergence, unavailable contact, terminal/reset separation, first numeric/material/persistent divergence, and torque-semantic non-comparison.

- [ ] **Step 2: Implement quaternion and threshold helpers**

```python
def quaternion_geodesic_error(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> np.ndarray:
    lhs = lhs_wxyz / np.linalg.norm(lhs_wxyz, axis=-1, keepdims=True)
    rhs = rhs_wxyz / np.linalg.norm(rhs_wxyz, axis=-1, keepdims=True)
    dot = np.abs(np.sum(lhs * rhs, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
```

Implement natural-envelope calculation, signal alignment at control boundaries, material thresholds from design v1.3, evidence-ranked candidate classification, CSV/JSON reports, and compact PNG plots only if matplotlib is available; plot absence must not invalidate numeric comparison.

- [ ] **Step 3: Run comparator tests**

Expected: all synthetic divergence locations and classifications pass.

### Task 4: MuJoCo collector

**Files:**
- Create: `wheelleg_dreamwaq/debug/sim2sim/collect_mujoco_trace.py`
- Create: `wheelleg_dreamwaq/debug/sim2sim/tests/test_mujoco_collector.py`

- [ ] **Step 1: Write failing headless collector tests**

Verify frozen manifest loading, 26 named hinges, reset three phases, 20 substeps per control tick, canonical/native pairs, pre/torque/post ordering, actual TorchScript input/output shapes, no unexpected contact-pair omission, and deterministic 1-tick output.

- [ ] **Step 2: Implement direct MuJoCo observation helpers in the debug file**

Reuse public functions from `wheelleg_mujoco` without modifying that package. Resolve all 26 hinge IDs from `model_manifest["joint_order"]`; capture COM Jacobian velocities, base orientation in ControlFrame, projected gravity, virtual legs, loop residuals, contacts, pre/post state, and native/canonical targets/torques.

- [ ] **Step 3: Implement reset and control collection**

For each physics substep:

```python
pre = capture_state(model, data)
torque_native = controller.compute_torque(data, targets)
controller.apply_torque(data, torque_native)
mujoco.mj_step(model, data)
post = capture_state(model, data)
append_transition(pre, targets, torque_native, post)
```

At each control boundary, store the observation actually passed to Actor before inference, the clipped action, pre-reset terminal state, returned next observation, native termination flags, and common diagnostic flags.

- [ ] **Step 4: Run 1-tick and 10-tick tests in the MuJoCo uv environment**

```powershell
uv run --project wheelleg_dreamwaq\sim2sim\mujoco python -m pytest wheelleg_dreamwaq\debug\sim2sim\tests\test_mujoco_collector.py -q
```

Expected: pass with 20 and 200 substep rows respectively.

### Task 5: Isaac debug-only environment and collector

**Files:**
- Create: `wheelleg_dreamwaq/debug/sim2sim/isaac_debug_env.py`
- Create: `wheelleg_dreamwaq/debug/sim2sim/collect_isaac_trace.py`
- Create: `wheelleg_dreamwaq/debug/sim2sim/tests/test_isaac_contract.py`

- [ ] **Step 1: Write pure/static contract tests**

Test nested config replacement does not mutate `WHEELLEG_CFG`, expected body/joint names are unique, and the debug step source contains the exact `DirectRLEnv.step()` ordering plus explicit pre-step and post-step capture points.

- [ ] **Step 2: Implement debug configuration and optional wheel sensors**

Construct a fresh `WheelLegFlatEnvCfg`, set one environment, fixed seed/device, disable randomization/resampling, and create a nested robot spawn replacement. Add left/right `ContactSensor` instances only in instrumented mode and never mutate formal environment config objects.

- [ ] **Step 3: Implement side-effect-free state capture**

Read `robot.data` directly, resolve the shared 26-hinge order once, call `_virtual_leg_state()` only for geometry, and never call `_capture_state()` from a physics substep observer. Capture canonical/native controlled state and direct all-hinge native state.

- [ ] **Step 4: Implement exact debug `step()` copy**

Copy Isaac Lab 2.3.2 `DirectRLEnv.step()` order. Capture pre-state after `_apply_action`, host estimate immediately after `scene.write_data_to_sim`, post-state after `scene.update`, and clone the final terminal state after `_get_dones()` but before `_reset_idx()`. Return the production API observation unchanged and mark whether it is reset output.

- [ ] **Step 5: Implement reset three-phase capture**

Record `_reset_idx()` output/cache before forward, direct engine state after forward, and the exact reset API observation consumed by Actor. Do not clear or refresh `_state` merely to make the two match.

- [ ] **Step 6: Run Isaac smoke gates**

Launch headless with one environment and run 1 tick, 10 ticks, then collector-versus-production repeated gates. Expected: four substep rows per tick, exact done/returned observation semantics, and no formal-config mutation.

### Task 6: Documentation and full test suite

**Files:**
- Create: `wheelleg_dreamwaq/debug/sim2sim/README.md`
- Update only if implementation evidence requires clarification: `docs/2026-10-05-wheelleg-sim2sim-debug-design.md`

- [ ] **Step 1: Document exact two-environment commands**

Include immutable artifact paths, scenario generation, MuJoCo collection, Isaac collection, comparison, output layout, and interpretation of unavailable contact/torque fields.

- [ ] **Step 2: Run pure and MuJoCo regression suites**

```powershell
E:\wheel_leg_rl-main\wheelleg_dreamwaq\.venv\Scripts\python.exe -m pytest wheelleg_dreamwaq\debug\sim2sim\tests\test_trace_schema.py wheelleg_dreamwaq\debug\sim2sim\tests\test_stand_scenario.py wheelleg_dreamwaq\debug\sim2sim\tests\test_compare_traces.py -q
uv run --project wheelleg_dreamwaq\sim2sim\mujoco python -m pytest wheelleg_dreamwaq\sim2sim\mujoco\tests wheelleg_dreamwaq\debug\sim2sim\tests\test_mujoco_collector.py -q
```

- [ ] **Step 3: Run existing project unit tests**

```powershell
E:\wheel_leg_rl-main\wheelleg_dreamwaq\.venv\Scripts\python.exe -m pytest wheelleg_dreamwaq\tests\unit -q
```

### Task 7: Independent code review gate

**Files:** all files created by Tasks 1-6.

- [ ] **Step 1: Record file hashes and test evidence**

Produce a list of changed paths, SHA256 values, and exact pass/fail counts.

- [ ] **Step 2: Dispatch an independent review agent**

Ask it to inspect only correctness risks: production behavior mutation, pre/post phase errors, reset/terminal contamination, canonical/native signs, hashes/schema, and missing tests. Do not begin formal collection until all P0/P1 findings are closed and re-reviewed.

### Task 8: Data collection and cross-engine comparison

**Files:** output only under `wheelleg_dreamwaq/artifacts/debug/sim2sim/<run-id>/`.

- [ ] **Step 1: Generate the frozen scenario and action sequences**
- [ ] **Step 2: Run MuJoCo reset, 1-tick, 10-tick, zero-action, six pulses, replay and closed-loop collection**
- [ ] **Step 3: Run the corresponding Isaac collections and five-repeat equivalence gates**
- [ ] **Step 4: Run comparison and generate `summary.json`, CSV metrics and plots**
- [ ] **Step 5: Inspect the first numerical, material and persistent divergences**
- [ ] **Step 6: Report evidence for/against reset, adapter, actuator response, closed-chain, contact, accumulation and closed-loop amplification; do not modify model parameters in this task**

---

## Plan Self-Review

- Spec coverage: v1.3 reset phases, terminal state, substep phase, 26 hinges, canonical/native signs, contact gating, natural envelope, D0-D4, independent review and final data comparison all map to explicit tasks.
- Placeholder scan: no TBD/TODO or unspecified implementation step remains.
- Type consistency: both collectors emit `Sim2SimDebugTraceV1`; six controlled fields use canonical/native pairs; 26-hinge fields remain named engine-native; control state is post-step pre-reset; Actor observation is pre-step actual input.
