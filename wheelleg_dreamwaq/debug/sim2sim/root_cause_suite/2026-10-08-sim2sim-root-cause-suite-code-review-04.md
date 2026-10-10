# Sim2Sim RootCauseSuite Core V1 Code Review 04

Date: 2026-10-08  
Scope: strictly limited to the files named in the review request  
Review mode: independent, read-only static review; no tests were run  
Accepted test record from the main thread: `138 passed, 1 skipped`; MuJoCo `12 passed`  
Result: **NOT APPROVED**

## Severity Summary

- P0: 1
- P1: 3
- P2: 1

Approval requires P0=0 and P1=0. The real Core diagnostic run remains blocked.

## Review-03 Finding Status

### P0-1: P60 replay result variable is undefined - CLOSED

`core_stage_worker.py:1604-1631` now binds `mujoco_replay_result` and passes that exact object to `compare_p60_replay_trace_directories()`. `tests/test_core_stage_worker.py:39-181` directly exercises `_run_p60()` and checks object identity.

### P0-2: G02 mixes evidence validity with diagnostic failure - CLOSED

`adapter_gate.py:374-407` and `adapter_gate.py:623-669` separate digital-chain failures from post-forward projection and plant-response differences. `core_stage_worker.py:730-750` records a successful evidence-producing stage with a separate diagnostic result. `orchestrator.py:128-147`, `orchestrator.py:935-964`, and `orchestrator.py:1085-1116` implement terminal blocking and partial finalization. `finalize.py:102-242` emits the hash-bound terminal adapter report.

### P0-3: P50 four-state verdict semantics are collapsed to booleans - CLOSED

`finalize.py:588-645` implements explicit primary, contributor, not-supported, and inconclusive states. `finalize.py:1024-1038` uses those states independently for normal and tangential candidates, and `finalize.py:1334-1407` no longer writes contributor evidence as exclusion evidence. Boundary coverage is present at `tests/test_finalize.py:210-269`.

### P0-4: Closure verdict omits frozen combination and P20 preconditions - CLOSED

The decision logic is implemented at `finalize.py:439-585`: P10/P30/P40 votes are aggregated, P40 strong-ratio support is recognized, P20 audit/golden/static-mismatch preconditions are enforced, and contact-exclusion gates are required. Boundary tests are at `tests/test_finalize.py:272-373`. A separate evidence-attribution defect remains open below as P1-N2.

### P1-1: G01 resume/reseal cannot replace an invalid selected attempt - CLOSED

`orchestrator.py:88-125` atomically replaces the selected G01 attempt and replay seal while invalidating transitive dependents. The execution path is at `orchestrator.py:1009-1053`; the reselection regression test is at `tests/test_orchestrator.py:167-219`.

### P1-2: Evidence references are not bound to the exact selected attempt or value - CLOSED

`finalize.py:758-870` now enforces selected-attempt ownership, artifact hash equality, JSON-path existence, and canonical observed-value equality. Cross-attempt, missing-path, and wrong-value tests are at `tests/test_finalize.py:56-207`.

### P1-3: Valid but unusable repeatability aborts instead of producing INCONCLUSIVE - OPEN

The numerical-unavailability path now reaches an INCONCLUSIVE partial report, but the required fail-closed distinction for malformed G01 evidence is incomplete. `finalize.py:245-290` accepts `repeatability_usable=false` without calling the existing snapshot verifier at `finalize.py:873-898`. The test at `tests/test_finalize.py:451-504` proves the partial finalizer succeeds even though its G01 fixture has no threshold-snapshot file, hash, or validated schema.

Minimum fix: call `_load_g01_threshold_snapshot(g01_worker, g01)` before accepting the terminal condition; derive unusable keys only from that validated snapshot. Keep schema/path/hash inconsistencies as `EvidenceIntegrityError`, and reserve INCONCLUSIVE for a valid snapshot containing unusable numerical keys.

### P2-1: G02 polarity trusts self-reported signs - OPEN

`adapter_gate.py:282-293` correctly derives signs from raw odd-response vectors and compares them with reported signs, but `adapter_gate.py:366-370` stores mismatches only as metrics. They do not enter any gate. `tests/test_adapter_gate.py:191-200` explicitly expects a reported-sign mismatch to leave the result passing.

Minimum fix: make both reported/derived consistency checks fail closed, either as digital-chain gates or as `EvidenceIntegrityError`; add target-sign and velocity-sign mutation coverage. Raw response remains authoritative, but contradictory redundant evidence must not be accepted as internally consistent.

## New Findings

### P0-N1: Full Core verification recomputes the P60 input identity without its replay seal

Execution includes `replay_source_seal_identity_hash` for both P60 and C70 at `orchestrator.py:737-748`. Verification adds that field only for C70 at `orchestrator.py:1251-1260`. Therefore the recomputed P60 input hash differs from the selected P60 record, and a completed full Core run fails verification before finalization.

Minimum fix: apply the replay-seal identity branch to both `P60_full_robot` and `C70_checkpoint` in `verify_stage_evidence()`, matching `_stage_input_identity()`. Add a focused regression test that constructs or executes a P60 selection and verifies the recomputed input identity.

### P1-N1: A resumed G02 terminal diagnosis can retain stale downstream selections

When G02 becomes terminal, `orchestrator.py:935-950` marks physics stages blocked but does not remove previously selected P10-C70 attempts. Later, `orchestrator.py:1195-1208` requires a terminal run to contain exactly the four gate-stage selections. A resume that replaces an invalid prior G02 attempt with a terminal attempt can therefore fail partial verification because stale physics selections remain in the manifest. Existing terminal tests at `tests/test_orchestrator.py:65-149` cover only fresh runs, not this resume transition.

Minimum fix: when selecting a terminal G02 attempt, atomically retain only the frozen gate-stage prefix or invalidate all transitive G02 dependents before writing the manifest. Add a resume test starting with downstream selections and replacing G02 with a terminal attempt.

### P1-N2: The closure aggregate verdict is attributed to a P40-only evidence row

`finalize.py:467-585` can produce primary support from P10+P30 even when P40 does not support closure; this case is explicitly tested at `tests/test_finalize.py:337-339`. However, `finalize.py:1294-1332` writes the aggregate `closure_status` and `supports=CLOSED_CHAIN_CONSTRAINT_MISMATCH` on one row whose exact source value is only `$.closure_on.normalized_rmse` from the P40 artifact. P10/P30 votes, P20 preconditions, and all three contact-exclusion proofs are present only in derived diagnostics and are not exact value-bound evidence references for that aggregate claim.

Minimum fix: emit selected-attempt-owned evidence references for every aggregate input actually used: P10 on/off material inputs, P30 open/closed material inputs, P40 on/off material and ratio, P20 audit/golden/mismatch fields, and each contact-exclusion result. Do not label the P40 scalar row with an aggregate status that may have been established entirely by P10+P30.

## Final Decision

**NOT APPROVED**

Open blocking findings: P0-N1, P1-3, P1-N1, and P1-N2. P2-1 also remains open. No architecture re-review or real simulation was performed.
