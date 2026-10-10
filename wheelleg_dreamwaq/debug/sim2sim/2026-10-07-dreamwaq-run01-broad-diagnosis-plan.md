# DreamWaQ Run-01 Broad Sim2Sim Diagnosis Plan

> **Scope:** Every code change stays under `debug/sim2sim/`. Generated traces,
> copied MJCF files, and reports stay under `artifacts/debug/sim2sim/`. Formal
> training, USD, MuJoCo runtime/model, reward, action, command, randomization,
> checkpoint, and evaluation files are read-only inputs.

**Goal:** Determine whether DreamWaQ run-01 fails in MuJoCo primarily because
of the `0.001 s x 20` MuJoCo integration/control cadence, or because the policy
amplifies a broader Isaac/MuJoCo dynamics gap after the first matched action.

**Decision rule:** First evaluate an explicitly non-formal MuJoCo copy with
`0.005 s x 4` and unchanged 20 ms policy control. Treat the timing change as a
solution only if all eight scenarios complete 500 ticks. Otherwise execute all
five diagnostic groups below and report evidence without tuning formal files.

## Files

- Create `debug/sim2sim/dreamwaq_debug_contract.py`:
  validated policy/history/CENet inspection and debug-only timing profiles.
- Create `debug/sim2sim/collect_dreamwaq_mujoco_trace.py`:
  configurable MuJoCo timing/cadence collector with 125D history and substeps.
- Create `debug/sim2sim/collect_dreamwaq_isaac_trace.py`:
  nominal Isaac collector with matching history, CENet, action, and substeps.
- Create `debug/sim2sim/evaluate_debug_mujoco.py`:
  eight-scenario non-ranking evaluator for DreamWaQ and Phase 1R actors.
- Create `debug/sim2sim/analyze_dreamwaq_diagnosis.py`:
  compare zero-action, one-step, replay, closed-loop, timing, and baseline data.
- Modify `debug/sim2sim/trace_schema.py`:
  allow validated 25D or 125D policy-input traces without changing V1 defaults.
- Modify `debug/sim2sim/README.md`:
  document the new debug-only commands and interpretation limits.
- Add focused tests under `debug/sim2sim/tests/` for history/CENet inspection,
  timing-copy isolation, torque refresh cadence, and dynamic trace validation.

## Execution

### Task 1: Freeze formal input identities

- [x] Record hashes for the formal MJCF, model manifest, MuJoCo runtime/control/
      observation modules, and Isaac environment/config.
- [x] Recompute the same hashes after all diagnostics and require equality.

### Task 2: Implement and test debug contracts

- [x] Load PPO or DreamWaQ TorchScript actors from explicit paths.
- [x] Initialize DreamWaQ history as five copies of current ActorObsV1.
- [x] Expose encoder velocity, context mean, and context log-variance without
      changing the frozen actor graph.
- [x] Define `formal_1ms`, `isaac_sync_5ms`, and `hold_5ms` timing profiles.
- [x] Generate copied MJCF files under the debug artifact root with absolute
      mesh paths and an explicit debug marker.

### Task 3: Run the timing gate first

- [x] Evaluate DreamWaQ run-01 on all eight scenarios at `0.005 s x 4` with PD
      torque recomputed each 5 ms substep.
- [x] Stop the gate only if all eight scenarios complete; otherwise continue
      automatically with Tasks 4-8.

### Task 4: Zero-action free response

- [x] Collect matched nominal reset, zero action, and at least 20 control ticks
      from Isaac and MuJoCo.
- [x] Compare base motion, joints, virtual legs, contacts, closure residuals,
      and torque despite policy output being observational only.

### Task 5: Identical one-action response

- [x] Use the DreamWaQ reset action as one shared open-loop action.
- [x] Record every Isaac 5 ms and MuJoCo 1 ms/5 ms substep, targets, unclipped
      and clipped PD torque, host/commanded torque, joint state, and base state.

### Task 6: Isaac-action open-loop replay

- [x] Collect the first 20 closed-loop Isaac actions from a fresh reset.
- [x] Replay the exact clipped float32 action rows in MuJoCo at formal timing.
- [x] Separate plant divergence under identical actions from policy feedback.

### Task 7: Closed-loop first 20 ticks

- [x] Collect both engines from a fresh nominal reset.
- [x] Compare current 25D observation, full 125D history, CENet velocity,
      context mean/log-variance, raw/clipped action, targets, torque, and state.
- [x] Locate the first numerical, material, and persistent divergence.

### Task 8: Phase 1R run-03 control comparison

- [x] Evaluate run-03 at formal timing and the same `0.005 s x 4` debug timing.
- [x] Compare its timing sensitivity and survival pattern with DreamWaQ run-01.

### Task 9: Verification and report

- [x] Run pure debug unit tests and MuJoCo integration tests.
- [x] Verify artifact hashes and trace dimensions.
- [x] Produce one machine-readable summary and one concise evidence report.
- [x] Verify all formal input hashes remain unchanged.

## Result

- DreamWaQ formal `0.001 s x 20`: `0/8`, mean survival `0.095250`.
- DreamWaQ synchronized `0.005 s x 4`: `0/8`, mean survival `0.103500`.
- Phase 1R run-03 formal `0.001 s x 20`: `5/8`, mean survival `0.655000`.
- Phase 1R run-03 synchronized `0.005 s x 4`: `0/8`, mean survival `0.326750`.
- Reset history, first actor output, target, and initial PD reference align.
- A material base-angular-velocity difference appears within the first 5 ms
  under the same first action, before the second policy inference.
- Zero-action and exact Isaac-action replay also diverge, locating the first
  mismatch in cross-engine plant/actuator/contact/constraint dynamics. CENet
  and closed-loop actions respond to and then amplify that mismatch.

## Frozen Semantics

The diagnosis must not change ActionV1 scaling/clipping, joint order/signs,
leg/wheel PD gains, effort/velocity limits, reset pose, commands, rewards,
randomization profiles, USD/MJCF dynamics parameters, termination thresholds,
history layout, CENet dimensions, or actor weights. Timing variants are labeled
debug-only and are never accepted as formal MuJoCo evaluation or ranking data.
