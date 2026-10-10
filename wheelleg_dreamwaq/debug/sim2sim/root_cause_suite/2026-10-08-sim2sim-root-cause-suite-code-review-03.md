# Sim2Sim RootCauseSuite Core V1 Code Review 03

Date: 2026-10-08  
Scope: `wheelleg_dreamwaq/debug/sim2sim/root_cause_suite`  
Review mode: independent, read-only review after the first implementation remediation pass  
Result: **NOT APPROVED**

## Severity Summary

- P0: 4
- P1: 3
- P2: 1

The real Core diagnostic run remains blocked until all P0/P1 findings are closed by code, regression tests, and a fresh independent review.

## Findings

### P0-1: P60 replay result variable is undefined

`core_stage_worker._run_p60()` binds the MuJoCo replay result as `_mujoco_replay_result` but later passes `mujoco_replay_result` to `compare_p60_replay_trace_directories()`. A real P60 execution therefore raises `NameError` before producing evidence.

Required closure: add a stubbed `_run_p60()` execution test, fix the binding, and later run a real P60 smoke after code-review approval.

### P0-2: G02 mixes evidence validity with diagnostic failure

The adapter gate treats first-response magnitude differences as hard adapter failures and maps almost every failed gate to `SIM2SIM_ADAPTER_BUG/adapter_chain`. `_run_g02()` then writes `passed=false` and raises, so the orchestrator marks the stage failed and the finalizer cannot reach its G02 verdict branch.

Required closure: distinguish worker/evidence validity from the diagnostic gate result; classify reset serialization and digital-chain failures separately from post-forward/closure or plant-response differences; permit a terminal, hash-bound G02 diagnostic verdict without pretending the worker crashed.

### P0-3: P50 four-state verdict semantics are collapsed to booleans

The current normal and tangential booleans erase the difference between primary support, contributor, not-supported, and inconclusive. In particular, one valid material normal condition and a tangential explanation ratio in `[0.30, 0.70)` must not be written as exclusion evidence.

Required closure: derive explicit four-state candidate statuses from evidence availability, gate validity, valid/material condition counts, and the frozen explanation-ratio thresholds.

### P0-4: Closure verdict omits frozen combination and P20 preconditions

The current closure status only inspects the P40 result. It does not implement the frozen rule: P40 ratio `>=0.70`, or at least two independent same-direction closure probes among P10/P30/P40, with P20-S free of static hard failure and no contact contamination.

Required closure: add one closure evidence aggregator with primary, contributor, not-supported, and inconclusive branches plus boundary tests.

### P1-1: G01 resume/reseal cannot replace an invalid selected attempt

When a selected G01 attempt is invalid and reruns as a later attempt, the new replay seal is compared against the old manifest seal and rejected. A reselection must atomically replace the G01 selection and seal and invalidate all downstream selections bound to the prior G01 state.

### P1-2: Evidence references are not bound to the exact selected attempt or value

Evidence artifacts are only constrained to the run root. The JSON signal path is not resolved and the stored observed value is not compared with the artifact. A row can therefore reference another attempt while claiming the selected attempt identity.

Required closure: require the artifact to live under the exact selected stage/attempt worker directory, resolve the supported JSON path, and compare the extracted value with the declared observed value. Add cross-attempt, missing-path, and wrong-value mutation tests.

### P1-3: Valid but unusable repeatability aborts instead of producing INCONCLUSIVE

G01 can complete with exact coverage while one or more repeatability keys are unusable. Downstream envelope lookup currently raises an integrity error, which converts numerical nondeterminism into pipeline corruption.

Required closure: preserve schema/hash corruption as fail-closed, but represent unavailable or nondeterministic evidence as `INCONCLUSIVE` and prevent material claims that depend on it.

### P2-1: G02 polarity trusts self-reported signs

The pulse gate reads `driven_channel_target_sign` and `driven_channel_velocity_sign` from worker output instead of deriving both signs from the raw odd-response vectors and cross-checking the reported fields.

Required closure: derive signs from raw vectors, require reported and derived values to match, and add a sign-mutation test.

## Confirmed Closed Areas

- causal identity and factor-path allow-list enforcement
- write guard, `dir_fd`, argv/environment capture, and G03 bootstrap isolation
- normal P60 replay-seal ownership and identity concept
- actual G01 threshold snapshot binding
- latest standalone G02 and G03 smoke evidence generation
- frozen formal-file snapshot remained unchanged at review time
