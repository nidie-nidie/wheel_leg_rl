# WheelLeg Sim2Sim Debug Pipeline

This directory is an isolated diagnostic surface. It does not modify the formal
training environment, the USD/MJCF models, or the frozen policy artifacts.

## Fixed Inputs

- Actor: `artifacts/phase1_v4/formal-4x1000-20261005/exports/run-01/actor.ts`
- Policy contract: the adjacent `policy_manifest.json`
- MuJoCo model: `sim2sim/mujoco/models/wheel_leg_urdf4_v1.xml`
- Scenario: one environment, 50 Hz control, `[vx, yaw_rate, height] = [0, 0, 0.20]`

Every collector verifies the frozen hashes before stepping physics.

## DreamWaQ Run-01 Broad Diagnosis

The 2026-10-07 run-01 diagnosis is entirely debug-only. It supports 25D PPO
actors and 125D DreamWaQ actors, preserves the five-identical-frame reset
history, and exposes CENet velocity and context values without editing the
serialized actor.

The collected evidence and final report are under:

```text
artifacts/debug/sim2sim/dreamwaq-run01-diagnosis-20261007-v1
```

The timing evaluator accepts these explicit profiles:

- `formal_1ms`: MuJoCo `0.001 s x 20`, PD refreshed every 1 ms.
- `isaac_sync_5ms`: copied MuJoCo model at `0.005 s x 4`, PD refreshed every 5 ms.
- `hold_5ms`: MuJoCo `0.001 s x 20`, one PD command held for five substeps.

Example timing run from the project root:

```powershell
uv run --project .\sim2sim\mujoco python -m debug.sim2sim.evaluate_debug_mujoco `
  --actor <actor.ts> --policy-manifest <policy_manifest.json> `
  --model-manifest sim2sim\mujoco\model_manifest.json `
  --timing-profile isaac_sync_5ms `
  --output artifacts\debug\sim2sim\timing-example
```

The broad diagnosis uses four matched trace groups: zero action, the identical
first action at substep resolution, exact replay of the first 20 Isaac actions
in MuJoCo, and 20-tick closed loop. Generate the final summary with:

```powershell
.\.venv\Scripts\python.exe -m debug.sim2sim.analyze_dreamwaq_diagnosis `
  artifacts\debug\sim2sim\dreamwaq-run01-diagnosis-20261007-v1 `
  --output artifacts\debug\sim2sim\dreamwaq-run01-diagnosis-20261007-v1\analysis
```

The 2026-10-07 result is that timestep synchronization is not a solution:
DreamWaQ remains `0/8`, while Phase 1R run-03 falls from `5/8` to `0/8` under
`0.005 s x 4`. The first material mismatch occurs in physical state within the
first 5 ms under an identical action, before a second policy inference. Treat
Isaac implicit-drive torque as a host estimate and MuJoCo `data.ctrl` as an
explicit command; they are diagnostic references, not identical measurements.

## Environments

Use the project venv for pure tests and Isaac Sim:

```powershell
E:\wheel_leg_rl-main\wheelleg_dreamwaq\.venv\Scripts\python.exe
```

Use the MuJoCo uv project for MuJoCo collection and tests:

```powershell
uv run --project E:\wheel_leg_rl-main\wheelleg_dreamwaq\sim2sim\mujoco python
```

Do not collect `test_mujoco_collector.py` with the Isaac venv; MuJoCo is
intentionally absent from that environment.

## Minimal Run

From `E:\wheel_leg_rl-main\wheelleg_dreamwaq`:

```powershell
$run = 'artifacts\debug\sim2sim\example-1tick'

.\.venv\Scripts\python.exe -m debug.sim2sim.stand_scenario `
  --output $run

.\.venv\Scripts\python.exe -m debug.sim2sim.collect_isaac_trace `
  --output "$run\isaac" --control-ticks 1 --mode closed_loop `
  --headless --device cuda:0

uv run --project .\sim2sim\mujoco python -m debug.sim2sim.collect_mujoco_trace `
  --output "$run\mujoco" --control-ticks 1 --mode closed_loop

.\.venv\Scripts\python.exe -m debug.sim2sim.compare_traces $run
```

The same commands support 10- and 500-tick closed-loop runs. `zero_action` and
`channel_pulse` use arrays from `action_sequences.npz`; pass both
`--action-sequence` and `--action-key`.

For a 500-tick Isaac diagnostic, the collector extends only its debug episode
guard to 10.02 s. The formal task checks timeout at `max_episode_length - 1`,
so this preserves 500 control rows without changing the training environment.

Extract the actual Isaac clipped actions before `isaac_policy_replay`:

```powershell
.\.venv\Scripts\python.exe -m debug.sim2sim.extract_replay_actions `
  "$run\isaac\control_trace.npz" "$run\replay_actions.npz"
```

The replay collector validates the adjacent JSON sidecar, including the NPZ
hash, key, row count, and source Isaac control-trace hash.

## Actor CPU Identity Gate

After one verified Isaac and MuJoCo run exist, evaluate the same serialized
float32 input with independently loaded copies of the same Actor on CPU:

```powershell
.\.venv\Scripts\python.exe -m debug.sim2sim.verify_actor_cpu_identity `
  --isaac-run <isaac-directory> --mujoco-run <mujoco-directory> `
  --output artifacts\debug\sim2sim\gate\actor-cpu
```

The exact same-input output limit is `1e-6`. Cross-engine input and output
differences are reported separately and do not alter that exact gate.

## Isaac Observer Equivalence Gate

The debug collector overrides `step()` to expose substeps. Before formal
cross-engine collection, run at least five independent 10-tick trajectories for
both the unmodified production step and the debug collector, then compare them:

```powershell
.\.venv\Scripts\python.exe -m debug.sim2sim.collect_isaac_production_reference `
  --output artifacts\debug\sim2sim\gate\production-01 `
  --control-ticks 10 --headless --device cuda:0

.\.venv\Scripts\python.exe -m debug.sim2sim.collect_isaac_trace `
  --output artifacts\debug\sim2sim\gate\debug-01 `
  --control-ticks 10 --mode closed_loop --headless --device cuda:0

.\.venv\Scripts\python.exe -m debug.sim2sim.compare_isaac_equivalence `
  --production <five-production-directories> `
  --debug <five-debug-directories> `
  --output artifacts\debug\sim2sim\gate\comparison
```

Run the ten Isaac processes serially. Concurrent Kit processes can contend for
the same key-value database and GPU, which invalidates the frozen gate setup.

## Isaac Replay Gate

Extract one closed-loop Isaac action file, then replay that same NPZ from a
fresh reset at least five times. Compare five original closed-loop runs and
five replay runs:

```powershell
.\.venv\Scripts\python.exe -m debug.sim2sim.compare_isaac_replay `
  --original <five-closed-loop-directories> `
  --replay <five-replay-directories> `
  --output artifacts\debug\sim2sim\gate\replay-comparison
```

The gate rejects a replay whose sidecar source hash is not one of the supplied
original Isaac traces.

## Matched Pitch-Torque Response

Use the zero-action mode to compare the early whole-body response to the same
body-local ControlFrameV1 pitch torque. The pulse interval is end-exclusive:

```powershell
$root = 'artifacts\debug\sim2sim\pitch-torque-response'

.\.venv\Scripts\python.exe -m debug.sim2sim.collect_isaac_trace `
  --output "$root\isaac-plus4" --control-ticks 20 --mode zero_action `
  --base-pitch-torque-nm 4 --pitch-torque-start-tick 5 `
  --pitch-torque-end-tick 10 --contact-sensors `
  --headless --device cuda:0

uv run --project .\sim2sim\mujoco python -m debug.sim2sim.collect_mujoco_trace `
  --output "$root\mujoco-plus4" --control-ticks 20 --mode zero_action `
  --base-pitch-torque-nm 4 --pitch-torque-start-tick 5 `
  --pitch-torque-end-tick 10
```

Repeat each engine with `0 Nm` and `-4 Nm`, using the directory names expected
by `analyze_pitch_torque_response.py`. Analyze all six runs with:

```powershell
.\.venv\Scripts\python.exe -m debug.sim2sim.analyze_pitch_torque_response `
  $root --output "$root\analysis"
```

The analyzer verifies hashes, action identity, command identity, signed pulse
symmetry, and timing before computing the centered odd response. GPU PhysX
does not support the attempted global-ground contact filter, so the Isaac
observer reads each wheel's unfiltered `net_forces_w`. Isaac Lab defines this
field as the net normal-force vector, excluding friction. Its vector norm is
recorded as normal-force magnitude; Isaac friction remains unavailable and is
stored as `NaN`.

## Output Contract

Each engine directory contains:

- `metadata.json`: versions, clocks, orders, hashes, solver facts, and unavailable fields.
- `reset_snapshot.json`: written pre-forward, forwarded post-forward, and returned observation phases.
- `control_trace.npz`: one row per 20 ms control tick.
- `substep_trace.npz`: 4 PhysX or 20 MuJoCo rows per control tick.
- `file_hashes.json`: hashes of collectors, schema, reports, and emitted traces.

The comparison directory contains `summary.json` and `per_signal_metrics.csv`.
The summary separates metadata identity, reset alignment, dynamic divergence,
and evidence-ranked candidates.

## MuJoCo Isolation Experiments

These experiments are debug-only. They reuse the frozen ActionV1 scaling and
PD controller but do not modify the formal runtime or model manifest.

- `fixed_base`: zero gravity, no ground contact, and no `base_free`. Use it to
  isolate joint signs, actuator response, and closure residuals.
- `suspended`: zero gravity and no ground contact, with `base_free` retained.
  Use it to inspect internal reaction and left/right symmetry.
- `grounded_pitch`: the formal grounded model with a control-frame pitch
  perturbation and a constant canonical wheel common action. It identifies the
  initial wheel correction direction and pitch coupling. A constant open-loop
  action is not a stabilizing controller and is not used to select a final gain.

From the project root, collect one run with the MuJoCo environment:

```powershell
$python = '.\sim2sim\mujoco\.venv\Scripts\python.exe'
$env:PYTHONPATH = '.;sim2sim/mujoco'

& $python -m debug.sim2sim.collect_mujoco_isolation_trace `
  --scenario fixed_base --channel 4 --amplitude 0.1 `
  --control-ticks 50 --pulse-start-tick 5 --pulse-end-tick 20 `
  --output artifacts\debug\sim2sim\isolation\fixed-wheel-left

& $python -m debug.sim2sim.collect_mujoco_isolation_trace `
  --scenario suspended --channel 4 --amplitude 0.1 `
  --control-ticks 50 --pulse-start-tick 5 --pulse-end-tick 20 `
  --output artifacts\debug\sim2sim\isolation\suspended-wheel-left

& $python -m debug.sim2sim.collect_mujoco_isolation_trace `
  --scenario grounded_pitch --initial-pitch-deg 5 --wheel-action 0.2 `
  --control-ticks 50 `
  --output artifacts\debug\sim2sim\isolation\pitch-p5-wheel-p02
```

`--wheel-action` is a normalized canonical ActionV1 value. With the frozen
contract, `0.2` maps to a `5 rad/s` wheel-speed target; it does not mean
`0.2 rad/s`. Scenario-specific CLI arguments are rejected when supplied to an
incompatible scenario instead of being silently ignored.

Summarize several runs together:

```powershell
& $python -m debug.sim2sim.analyze_mujoco_isolation `
  artifacts\debug\sim2sim\isolation\pitch-* `
  --output artifacts\debug\sim2sim\isolation\analysis
```

Each run contains `metadata.json`, `isolation_trace.npz`, and
`file_hashes.json`. The directory is first written under a temporary sibling
and renamed only after all three files are complete. The isolation trace records
the six canonical actions, native and canonical targets, native and canonical
joint state, tick-mean/substep-peak/impulse torque, effort and velocity-limit
fractions, base motion, wheel contact force, virtual-leg state, and all eight
equality-site closure residuals.

The analyzer verifies the emitted hashes, scenario/model identities, row count,
tick sequence, and control clock before reading a run. Its pitch response groups
report pitch and pitch rate relative to the zero-action baseline at common time
horizons, together with zero crossings and overshoot. They deliberately do not
label one constant action as the best controller.

## Semantics

- All cross-engine controlled-joint fields use canonical order
  `[jIJ, jIO, jAB, jAG, jwheel_left, jwheel_right]`.
- Engine-native pairs are stored separately; the right wheel uses sign `-1`.
- Isaac `applied_torque` is a host-side implicit-drive estimate. MuJoCo
  `data.ctrl` is an explicit commanded torque. They are never compared as the
  same physical measurement.
- Baseline Isaac contact fields are unavailable. Instrumented contact traces
  remain `instrumented_unverified` until a same-engine equivalence gate passes.
- A terminal control row always uses the state frozen before reset; the actual
  reset observation is stored only as `next_actor_obs_policy_returned` with
  `next_obs_is_reset_int8 = 1`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest `
  debug\sim2sim\tests\test_trace_schema.py `
  debug\sim2sim\tests\test_stand_scenario.py `
  debug\sim2sim\tests\test_compare_traces.py `
  debug\sim2sim\tests\test_isaac_contract.py `
  debug\sim2sim\tests\test_isaac_equivalence.py `
  debug\sim2sim\tests\test_replay_actions.py `
  debug\sim2sim\tests\test_actor_cpu_identity.py `
  debug\sim2sim\tests\test_isaac_replay.py `
  debug\sim2sim\tests\test_pitch_torque_pulse.py `
  debug\sim2sim\tests\test_analyze_pitch_torque_response.py `
  debug\sim2sim\tests\test_dreamwaq_debug_contract.py `
  debug\sim2sim\tests\test_collect_dreamwaq_isaac_contract.py `
  debug\sim2sim\tests\test_analyze_dreamwaq_diagnosis.py -q

uv run --project .\sim2sim\mujoco python -m pytest `
  debug\sim2sim\tests\test_mujoco_collector.py `
  debug\sim2sim\tests\test_isolation_scenarios.py `
  debug\sim2sim\tests\test_isolation_models.py `
  debug\sim2sim\tests\test_mujoco_isolation_collector.py `
  debug\sim2sim\tests\test_analyze_mujoco_isolation.py `
  debug\sim2sim\tests\test_evaluate_debug_mujoco.py `
  debug\sim2sim\tests\test_collect_dreamwaq_mujoco_trace.py -q
```
