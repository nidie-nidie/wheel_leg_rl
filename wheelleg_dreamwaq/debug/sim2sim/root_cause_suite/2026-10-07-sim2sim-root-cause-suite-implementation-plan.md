# WheelLeg Sim2Sim Root Cause Suite Core V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This workspace is not a Git repository, so the verification artifact for each task is a test log plus SHA256 inventory rather than a commit.

**Goal:** Implement, test, execute, and independently review the frozen RootCauseSuiteCoreV1.16 diagnostic pipeline entirely under `debug/sim2sim/root_cause_suite/`, without changing formal training, assets, policies, randomization, rewards, commands, termination, or MuJoCo semantics.

**Architecture:** A pure-Python parent process owns identity checks, stage DAG execution, resume, analysis, verdict, and reporting. Isaac and MuJoCo run in separate frozen interpreters and communicate only through hash-verified JSON/NPZ artifacts. Debug variants are structured copies under the run directory, and every physical conclusion is gated by identity, repeatability, adapter, instrumentation, and compiled-semantics checks.

**Tech Stack:** PowerShell 7/Windows PowerShell bootstrap, Python 3.11 standard library, NumPy, TorchScript CPU inference, MuJoCo 3.14, Isaac Sim 5.1/Isaac Lab 2.3.2, pytest.

**Frozen design:** `RootCauseSuiteCoreV1.16`, SHA256 `363720317500D6F98A4BBA81CFC7F2E9B5EF5A774974418542A46F602C072FE6`.

---

## File Map

Create only beneath `debug/sim2sim/root_cause_suite/`:

- `run_suite.ps1`: only supported launcher; creates a run-local cache root and invokes project Python with `-B`.
- `__init__.py`, `__main__.py`, `cli.py`: package identity and CLI routing.
- `contracts.py`: enums, dataclasses, canonical JSON/hash helpers, stage/result schemas, frozen input identities.
- `workspace_guard.py`: path confinement, atomic JSON/NPZ writes, formal-file and project-inventory snapshots.
- `python_write_guard.py`: pre-import audit hook, integer-fd handling, capability manifest, nonce self-test, FileHandler registry.
- `trace_contract.py`: trace sidecars, required fields, units/frames/availability, NPZ verification.
- `compiled_properties.py`: body mapping, inertia/frame transforms, composite mass properties, momentum/energy golden calculations.
- `body_mapping_v1.json`: exact 27-body mapping from design §7.6.
- `scenario_catalog.py`: only Core V1 scenarios, ratio pairs, repeatability families, metrics, dependencies, and replay identity.
- `analysis.py`: common-time alignment, repeat envelopes, odd/even response, mismatch metrics, explanation ratios, C70 sensitivity.
- `verdict.py`: frozen decision table; never emits checkpoint-only primary in Core V1.
- `report.py`: `evidence.csv`, `verdict.json`, `report.md`, and artifact hash index.
- `variant_builders.py`: structured MuJoCo copies and Isaac override/config specifications for no-ground, gravity-off, closure-off, drive-off, and sphere coupons.
- `mujoco_worker.py`: compiled semantics, P10/P20/P30/P40/P50/P60 collection, formal replay, and runtime goldens.
- `isaac_bootstrap.py`: hardened AppLauncher boundary and Kit path/source contracts.
- `isaac_worker.py`: equivalent Isaac collectors and runtime goldens using only cloned configs and suite-local layers.
- `orchestrator.py`: run creation, stage DAG, worker subprocesses, atomic completion, resume, verify, and report flow.
- `fake_worker.py`: deterministic subprocess used by orchestration tests.
- `tests/`: pure unit, synthetic, MuJoCo integration, Isaac integration, and minimal end-to-end tests.
- `runs/`: generated runtime artifacts only.

Existing `debug/sim2sim/*.py` modules may be imported read-only, especially `dreamwaq_debug_contract.py`, `trace_schema.py`, `isaac_debug_env.py`, and existing collectors. They are not modified.

## Task 1: Bootstrap, CLI, and Frozen Contracts

**Files:**
- Create: `run_suite.ps1`
- Create: `__init__.py`
- Create: `__main__.py`
- Create: `cli.py`
- Create: `contracts.py`
- Test: `tests/test_bootstrap.py`
- Test: `tests/test_contracts.py`

- [ ] Write tests that reject a missing bootstrap marker, missing `-B`, wrong design SHA, unknown stage/status values, non-canonical JSON, and any Core catalog entry named for an Extended probe.
- [ ] Run:

```powershell
.\.venv\Scripts\python.exe -B -m pytest debug\sim2sim\root_cause_suite\tests\test_bootstrap.py debug\sim2sim\root_cause_suite\tests\test_contracts.py -q -o cache_dir=debug\sim2sim\root_cause_suite\runs\plan-test\runtime_cache\pytest-cache --basetemp=debug\sim2sim\root_cause_suite\runs\plan-test\runtime_cache\pytest-tmp
```

Expected: failures for missing modules.

- [ ] Implement frozen constants and canonical helpers with these public interfaces:

```python
DESIGN_VERSION = "RootCauseSuiteCoreV1.16"
DESIGN_SHA256 = "363720317500D6F98A4BBA81CFC7F2E9B5EF5A774974418542A46F602C072FE6"
SCHEMA_VERSION = "RootCauseSuiteRunV1"

def canonical_json_bytes(payload: object) -> bytes: ...
def stable_hash(payload: object) -> str: ...
def sha256_file(path: Path) -> str: ...
def require_bootstrap() -> None: ...
```

- [ ] Implement `run_suite.ps1` so it creates `<suite>/runs/<run-id>/runtime_cache/{pycache,pytest-cache,pytest-tmp}`, sets `WHEELLEG_ROOT_CAUSE_BOOTSTRAP=1`, `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPYCACHEPREFIX`, and runs only `.venv\Scripts\python.exe -B -m debug.sim2sim.root_cause_suite`.
- [ ] Implement CLI commands `run`, `verify <run-id>`, and `report <run-id>` with exit codes 0/2/3/4/5 from design §10.3.
- [ ] Re-run the tests and a bootstrap `--help` smoke; expected PASS and no suite-external `.pyc` or `.pytest_cache` creation.

## Task 2: Workspace and Python Write Guards

**Files:**
- Create: `workspace_guard.py`
- Create: `python_write_guard.py`
- Test: `tests/test_workspace_guard.py`
- Test: `tests/test_python_write_guard.py`
- Test helper: `tests/write_guard_subprocess.py`

- [ ] Write failing tests for `..`, symlink/junction escape, atomic sibling writes, before/after hash drift, write-mode `open`, mkdir/remove/rmdir/rename/link/symlink/truncate/chmod/utime, FileHandler `delay=True`, integer fd, nonce blocker, `O_TEMPORARY`, and per-API `os.supports_dir_fd` capability evidence.
- [ ] Implement path confinement:

```python
def canonical_path(path: os.PathLike[str] | str, *, dir_fd: int | None = None) -> Path: ...
def require_within(path: Path, roots: tuple[Path, ...]) -> Path: ...
def atomic_write_json(path: Path, payload: object) -> Path: ...
def snapshot_tree(root: Path) -> dict[str, dict[str, object]]: ...
def compare_snapshots(before: dict, after: dict) -> list[dict[str, object]]: ...
```

- [ ] Implement `PythonWriteGuard.install()` using only standard library imports before any third-party import. Its ledger records event, raw args, resolved paths, decision, and timestamp; its capability manifest distinguishes absent constants/APIs from value `0`.
- [ ] Implement Windows integer-fd resolution using `msvcrt.get_osfhandle` plus `ctypes.windll.kernel32.GetFinalPathNameByHandleW`; permit only run-local resolved files, fd 0/1/2, and explicitly registered anonymous pipes. Unknown writable fds fail closed.
- [ ] After `sys.addaudithook`, emit `wheelleg.root_cause_suite.guard_probe` with a random nonce and require exactly one matching ledger entry before imports continue.
- [ ] Wrap `logging.FileHandler.__init__` only within a context manager, reject escaped filenames before the original constructor, record created handlers, call the original exactly once for legal paths, and restore it in `finally`.
- [ ] Execute the write-guard tests in dedicated subprocesses because audit hooks cannot be removed. On Windows, pre-create both `O_TEMPORARY` files, use only `O_RDONLY|O_TEMPORARY`, and prove the external file remains unchanged after rejection.

## Task 3: Trace and Compiled-Property Contracts

**Files:**
- Create: `trace_contract.py`
- Create: `compiled_properties.py`
- Create: `body_mapping_v1.json`
- Test: `tests/test_trace_contract.py`
- Test: `tests/test_compiled_properties.py`

- [ ] Write failing tests for missing unit/frame/phase/availability metadata, unknown sentinel values, inconsistent row counts, wrong common-time samples, corrupted NPZ sidecars, incomplete 27-body mapping, and incorrect inertia transforms.
- [ ] Store the exact 27 mappings from design §7.6; expand every `same` to the literal body name and require exact ordered-set equality.
- [ ] Implement trace interfaces:

```python
def write_trace(directory: Path, arrays: Mapping[str, np.ndarray], fields: Mapping[str, FieldSpec]) -> TraceIdentity: ...
def load_verified_trace(directory: Path) -> VerifiedTrace: ...
def normalize_availability(value: object, *, sentinel: object | None = None) -> Availability: ...
def validate_common_times(times_s: np.ndarray) -> None: ...
```

- [ ] Implement quaternion/rotation helpers, COM-frame inertia conversion, parallel-axis composition, system COM, linear momentum, angular momentum about COM/world origin, and kinetic energy.
- [ ] Add the fixed single-body golden from design §7.6 using the legal principal inertia `[0.031,0.047,0.073] kg m^2`, and mutation tests for link-origin velocity, `R.T @ I @ R`, and omitted `r x P`. Reject any implementation that enables `balanceinertia` to rewrite the frozen input.
- [ ] Run both test files and require all mutation cases to fail for the intended reason.

## Task 4: Catalog, Analysis, Verdict, and Reports

**Files:**
- Create: `scenario_catalog.py`
- Create: `analysis.py`
- Create: `verdict.py`
- Create: `report.py`
- Test: `tests/test_scenario_catalog.py`
- Test: `tests/test_analysis.py`
- Test: `tests/test_verdict.py`
- Test: `tests/test_report.py`

- [ ] Define only G00-G03, P10-A/B/C, P20-S, P30-A/B/C, P40-A/B, P50-A/C, P60-B/C/D, and C70. Catalog validation rejects P20-D, P40-C, P50-B/D, P60-A/E, wildcard semantic paths, or any unlisted ratio pair.
- [ ] Freeze the four ratio pairs exactly: `P30_OPEN_TARGET_TO_DIRECT`, `P30_CLOSED_TARGET_TO_DIRECT`, `P40_CLOSURE_ON_TO_OFF`, `P50_NOMINAL_TO_ZERO_FRICTION`.
- [ ] Implement `effective_tolerance=max(material_floor, 5*repeat_envelope)`, common-time interpolation only at `0,5,10,15,20,40,100,200,400 ms`, first numerical/material/persistent divergence, odd/even response, normalized RMSE, event delta, and explanation ratio.
- [ ] Validate pair identity using exact JSON-pointer diffs, pre-forward equality, class-specific post-forward/returned-policy guards, and the allowed factor set.
- [ ] Implement C70 using both Isaac and MuJoCo anchors, six feature groups, `+/-` perturbations, full-history and latest-frame-only modes, median gains, ratio/difference thresholds, and saturation support. Phase 1R CENet fields are unavailable rather than NaN evidence.
- [ ] Implement the decision table from design §8.2. `CHECKPOINT_ROBUSTNESS_FAILURE` is impossible as Core primary; zero primary with contributors or remaining Extended probes is `INCONCLUSIVE`.
- [ ] Generate deterministic `evidence.csv`, `verdict.json`, `report.md`, and `file_hashes.json`; every evidence reference includes the artifact SHA.
- [ ] Run synthetic tests covering every primary, multiple, contributor, invalid, repeatability-unavailable, and no-material-divergence branch.

## Task 5: Stage DAG, Resume, and Fake End-to-End Flow

**Files:**
- Create: `orchestrator.py`
- Create: `fake_worker.py`
- Test: `tests/test_orchestrator.py`
- Test: `tests/test_fake_end_to_end.py`

- [ ] Write failing tests for dependency blocking, independent-branch continuation, atomic stage completion, interrupted temp directories, resume hash mismatch, Extended scenario rejection, worker command lacking `-B`, and trace tampering.
- [ ] Implement immutable run directories and stage attempts. A stage becomes reusable only after `stage_state.json` says `complete` and every listed file hash verifies.
- [ ] Build worker commands as argument arrays, never shell strings, using the exact project or MuJoCo interpreter and `-B` before `-m`.
- [ ] Capture stdout/stderr, argv, environment contract, return code, start/end time, and worker source hash under `commands/`.
- [ ] Run the fake pipeline through `run_suite.ps1`, interrupt one stage, resume it, tamper one trace, and verify that `verify` fails closed.

## Task 6: Structured Variant Builders

**Files:**
- Create: `variant_builders.py`
- Test: `tests/test_variant_builders.py`

- [ ] Write failing tests proving the formal XML/hash never changes and each variant changes only its declared semantic paths.
- [ ] Parse MJCF with `xml.etree.ElementTree`, absolutize mesh paths, and create run-local variants for: no ground, gravity off, closure inactive while retaining all eight equality constraints, drive off with exactly 26 damping values zero, fixed base, and the common-sphere coupons.
- [ ] Emit a transform manifest containing source SHA, output SHA, operation list, old/new values, and allowed factor. Compile every output and compare normalized semantics to the manifest.
- [ ] Represent Isaac variants as immutable data specifications consumed by `isaac_worker.py`; do not import Isaac in the parent process and do not modify the formal USD.
- [ ] Test mutations that also alter armature, mass, contact solver, MuJoCo option, equality endpoint, motor gain, or a second factor; each must be rejected.

## Task 7: MuJoCo Worker and Integration Tests

**Files:**
- Create: `mujoco_worker.py`
- Test: `tests/test_mujoco_worker.py`

- [ ] Implement subcommands `identity`, `properties`, `golden`, `robot-probe`, `sphere-impact`, `sphere-slide`, and `replay` using the frozen MuJoCo interpreter.
- [ ] Serialize every public `model.opt` field listed in design §6.4, with enum integer and symbolic names; serialize topology, 26 hinges, six actuators, eight connect constraints, contact pairs, and per-body properties.
- [ ] Implement direct effort by bypassing `MixedActionController.compute_torque`, applying canonical/native sign `[1,1,1,1,1,-1]`, and proving actuator/generalized-force round trips within `1e-12 Nm`.
- [ ] Collect reset phases, substeps, common state, closure residual, contact impulse, momentum/energy, and exact excitation arrays for all MuJoCo Core probes. Use at least three repeats per verdict-bearing repeatability key.
- [ ] Reuse the frozen policy adapter and existing observation/control helpers read-only for P60-C/D; keep formal timing at 1 ms x 20.
- [ ] Run compiler, one-tick, golden, no-ground/zero-g, drive-off, closure-off, sphere, first-action, and short replay integration tests; verify formal XML/manifest/runtime hashes unchanged.

## Task 8: Isaac Bootstrap, Worker, and Integration Tests

**Files:**
- Create: `isaac_bootstrap.py`
- Create: `isaac_worker.py`
- Test: `tests/test_isaac_bootstrap.py`
- Test: `tests/test_isaac_worker.py`

- [ ] Implement source/hash contracts for AppLauncher, SimulationApp, DirectRLEnv, SimulationContext, SimulationCfg, logger, articulation, articulation data, and actuator PD before Kit starts.
- [ ] Implement `RootCauseSuiteAppLauncher._create_app()` to copy launch config, force/assert `enable_crashreporter=False`, then call `super()` exactly once. Add explicit portable-root/log/data/config/dump/texture-cache Kit args.
- [ ] Install `PythonWriteGuard` before importing Isaac modules. Clone debug config, set run-local `sim.log_dir`, retain `save_logs_to_file=True`, and assert no pre-existing `SimulationContext`.
- [ ] After startup, verify resolved paths, exact texture cache equality, crash setting, loaded plugins, FileHandler pre/created/post evidence, and project Kit tree invariance.
- [ ] Implement Isaac equivalents of MuJoCo worker subcommands, using suite-local USD/config copies/layers and the existing `WheelLegSim2SimDebugEnv` read-only where behavior matches. Formal `WHEELLEG_CFG` deep hash must remain unchanged.
- [ ] Implement drive-off gain checks at config, actuator object, ArticulationData, and PhysX view levels; closure-off preserves all four loop-joint prims and only disables them.
- [ ] Run fake launcher/source-contract tests first, then a real headless one-tick worker, property golden, and one smoke per immutable configuration family. All Isaac subprocesses are serialized.

## Task 9: Full Core Orchestration and Code Review Gate

**Files:**
- Modify: `orchestrator.py`
- Modify: `scenario_catalog.py`
- Test: `tests/test_real_minimal_end_to_end.py`

- [ ] Wire G00-G03, P10, P20-S, P30, P40, P50, P60, and C70 according to the catalog DAG. A supported physical candidate does not suppress independent branches.
- [ ] Implement P60-D source generation exactly once from DreamWaQ run-01, 8-env reset cache row 0, `nominal_stand`, seed `20261007`, 499 actions, and fresh-reset Isaac replay equivalence before MuJoCo replay.
- [ ] Run the full static/unit/synthetic suite and minimal real G00-G03/P20-S end-to-end test.
- [ ] Invoke an independent code-review agent. Fix every P0/P1, rerun affected tests, and repeat review until P0/P1 are zero.

## Task 10: Execute M1, M2, M3 and Review Results

**Generated files only:** `runs/<run-id>/...`

- [ ] M1: run G00-G03, P10, and P20-S; stop only on evidence-pipeline hard gates. Inspect hashes, repeatability, reset/history/Actor, instrumentation neutrality, and static-property/golden results.
- [ ] M2: run P30, P40, P50-A/C and generate per-probe analysis. Preserve raw traces even when a candidate is unsupported.
- [ ] M3: run P60-B/C/D and C70. Seal replay identity before any replay or checkpoint analysis.
- [ ] Run `verify <run-id>` and `report <run-id>`, then independently review test logs, stage manifests, traces, threshold snapshot, evidence, verdict, report, and before/after formal hashes.
- [ ] If review identifies a report-only issue, rebuild report. If it identifies collection/analysis code defects, return to Task 9, rerun code review, and create a new stage attempt or run rather than overwriting evidence.
- [ ] Final output must state one of: certified primary, multiple physics mismatches, inconclusive with contributors, `CORE_NO_MATERIAL_DIVERGENCE`, invalid pipeline, or “本轮没有找到可认证根因”. It must never claim checkpoint-only primary from Core V1.

## Verification Commands

Pure and synthetic suite (PowerShell session; cache remains inside the suite):

```powershell
$cache = (Resolve-Path .\debug\sim2sim\root_cause_suite).Path + '\runs\test-runtime\runtime_cache'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPYCACHEPREFIX = "$cache\pycache"
.\.venv\Scripts\python.exe -B -m pytest .\debug\sim2sim\root_cause_suite\tests -q -o "cache_dir=$cache\pytest-cache" --basetemp "$cache\pytest-tmp"
```

MuJoCo integration:

```powershell
$cache = (Resolve-Path .\debug\sim2sim\root_cause_suite).Path + '\runs\test-mujoco\runtime_cache'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPYCACHEPREFIX = "$cache\pycache"
.\sim2sim\mujoco\.venv\Scripts\python.exe -B -m pytest .\debug\sim2sim\root_cause_suite\tests\test_mujoco_worker.py -q -o "cache_dir=$cache\pytest-cache" --basetemp "$cache\pytest-tmp"
```

Isaac integration:

```powershell
$cache = (Resolve-Path .\debug\sim2sim\root_cause_suite).Path + '\runs\test-isaac\runtime_cache'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPYCACHEPREFIX = "$cache\pycache"
.\.venv\Scripts\python.exe -B -m pytest .\debug\sim2sim\root_cause_suite\tests\test_isaac_bootstrap.py .\debug\sim2sim\root_cause_suite\tests\test_isaac_worker.py -q -o "cache_dir=$cache\pytest-cache" --basetemp "$cache\pytest-tmp"
```

Full Core run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\debug\sim2sim\root_cause_suite\run_suite.ps1 run
```

Final verification/report:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\debug\sim2sim\root_cause_suite\run_suite.ps1 verify <run-id>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\debug\sim2sim\root_cause_suite\run_suite.ps1 report <run-id>
```

## Self-Review

- Spec coverage: every Core stage, hard gate, ratio pair, replay identity, C70 constraint, artifact, review gate, and explicit Extended exclusion maps to a task above.
- Placeholder scan: no `TBD`, `TODO`, “implement later”, hidden Extended hook, or unspecified worker command remains.
- Type/interface consistency: all workers emit files consumed through `trace_contract.py`; analysis consumes only verified traces; verdict consumes only analysis JSON; report consumes verdict/evidence; formal modules never import the suite.
- Scope: no formal source, USD, MJCF, policy, checkpoint, reward, action, command, randomization, termination, or evaluator file is modified.
