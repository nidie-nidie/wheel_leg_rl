# Sim2Sim RootCauseSuite Core V1 Code Review 05

Date: 2026-10-08  
Review mode: independent, defect-first, read-only static review  
Result: **APPROVED**

## Conclusion

- P0: 0
- P1: 0
- P2: 3
- Review-04 `P0-N1`, `P1-3`, `P1-N1`, `P1-N2`, and `P2-1` are closed.
- No new P0/P1 regression was found in terminal partial finalization, full verification, resume/reseal, or exact evidence binding.
- Approval threshold `P0=0 && P1=0` is satisfied. The real Core diagnostic run is no longer blocked by code review.

This review did not rerun pytest because the request permits writing only this report and pytest would create cache/temp artifacts. The supplied records were accepted: project interpreter `142 passed, 1 skipped`; MuJoCo interpreter `12 passed`. Read-only inspection found 143 project node IDs and 12 MuJoCo node IDs in the latest named pytest caches, consistent with those records.

## Frozen Baseline

- Architecture: `E:\wheel_leg_rl-main\docs\2026-10-03-wheelleg-dreamwaq-architecture.md`
  - Declared version: `Architecture v0.21` at line 4.
  - Observed SHA256: `5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE`.
- Design: `E:\wheel_leg_rl-main\wheelleg_dreamwaq\debug\sim2sim\root_cause_suite\2026-10-07-sim2sim-root-cause-suite-design.md`
  - Declared version: `RootCauseSuiteCoreV1.17` at line 4.
  - Observed SHA256: `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD`.
- Factor-path allowlist: `factor_path_allowlist.json`
  - Observed SHA256: `4CF2A34D4E3725A78DC0028AD92FF6DCD04E820874CFFCD8757F267F8DADDD85`.
- The main-thread `FORMAL_FILES` 13/13 statement was used as an accepted review premise; this review made no formal-file changes.

## Review-04 Finding Status

### P0-N1: P60 verification omitted the replay seal - CLOSED

Execution and verification now build the same input identity:

- `orchestrator.py:721-768` adds `replay_source_seal_identity_hash` for both `P60_full_robot` and `C70_checkpoint` during execution.
- `orchestrator.py:1286-1304` applies the identical two-stage branch during verification.
- `tests/test_orchestrator.py:151-168` constructs a selected G01 replay source plus P60 and verifies the recorded P60 input identity successfully.

The seal also contains the selected G01 attempt and stage-state hash (`orchestrator.py:398-439`), so a G01 reseal changes the P60/C70 input identity rather than merely changing a side artifact.

### P1-3: Valid-but-unusable G01 evidence was accepted without its independent snapshot - CLOSED

- `finalize.py:245-288` now calls `_load_g01_threshold_snapshot()` before producing the terminal `INCONCLUSIVE` result and derives unusable keys only from the loaded snapshot.
- `finalize.py:1096-1148` checks the independent filename, containment, file hash, schema/identity, key record shape, sample counts, usable flags, reasons, aggregate counts, aggregate usability, and embedded identity.
- `tests/test_finalize.py:475-542` covers a valid unusable snapshot.
- `tests/test_finalize.py:544-590` proves a missing snapshot is rejected rather than converted into `INCONCLUSIVE`.

Malformed or inconsistent snapshot evidence therefore fails before report generation. A lower-severity semantic-hardening gap remains under P2-N1.

### P1-N1: Resumed terminal G02 retained stale downstream selections - CLOSED

- `orchestrator.py:88-145` computes all transitive dependents and replaces the G02 selection while removing every selected descendant.
- The replacement is used both when a verified prior terminal attempt is reused (`orchestrator.py:1029-1048`) and when a newly executed attempt becomes terminal (`orchestrator.py:1082-1093`).
- The mutated selection set is committed by one atomic manifest write; no intermediate manifest containing the new terminal G02 plus stale descendants is written.
- Against the frozen Core DAG at `orchestrator.py:1171-1206`, the affected closure is P10, P20, P30, P40, P50, P60, and C70; G00, G01, and independent G03 are retained.
- `tests/test_orchestrator.py:171-223` covers the resume transition from a completed physics selection to G02 attempt-0002 terminal and checks that only G00-G03 remain selected.

Fresh and resume paths therefore converge on the exact terminal selection prefix required by `orchestrator.py:1231-1248`.

### P1-N2: Closure aggregate was attributed to one P40 scalar - CLOSED

- `_closure_status()` consumes the P10, P30, P40, P20, and contact-exclusion inputs at `finalize.py:463-581`.
- `_closure_evidence_rows()` emits exact selected-attempt-owned rows for all actual aggregate inputs at `finalize.py:750-974`:
  - P10 on/off `max_abs` and both repeat envelopes;
  - P30 open/closed material booleans for direct and target modes;
  - P40 on/off material booleans and explanation ratio;
  - P20 audit, golden, and static-mismatch fields;
  - P10/P30/P40 contact-exclusion results.
- `_build_evidence_refs()` enforces selected-attempt ownership, artifact hash, JSON path existence, and exact canonical value equality at `finalize.py:996-1093`.
- The pre-existing P40 normalized-RMSE row now has a local `material|within_tolerance` status and no closure support/exclusion claim at `finalize.py:1544-1567`; aggregate support comes from the dedicated rows added at `finalize.py:1568-1581`.
- `tests/test_finalize.py:884-919` checks the required aggregate paths and explicitly rejects attaching the aggregate claim to `$.closure_on.normalized_rmse`.

The P10+P30 primary case is therefore no longer represented as if it were established by P40 alone.

### P2-1: Reported polarity could contradict raw vectors without failing - CLOSED

- `adapter_gate.py:257-310` derives target and velocity signs from `odd_target_response` and `odd_velocity_response`; these raw vectors remain the inputs to the physical direction gates.
- `adapter_gate.py:351-355` raises `EvidenceIntegrityError` when either reported sign contradicts that raw derivation.
- `tests/test_adapter_gate.py:191-223` covers reported velocity mutation, reported target mutation, and a raw negative velocity that cannot be hidden by a positive reported sign.

The repair fails closed on contradictory redundant evidence without making the reported scalar authoritative over the raw response vector.

## New Findings

### P0

None.

### P1

None.

### P2-N1: Threshold snapshot validation does not enforce all frozen metadata semantics

Files:

- `repeatability.py:490-503`
- `finalize.py:1096-1148`
- `tests/test_repeatability.py:149-156`

Reproducible logic: `validate_threshold_snapshot()` verifies schema, self-consistent identity hash, and basic key/index map shape, but it does not assert the frozen values of `frozen_before_cross_engine_physics`, `minimum_repetitions`, the two rule strings, `material_floors`, or execution/index completeness. `_load_g01_threshold_snapshot()` adds useful per-key and aggregate checks, but accepts any positive `sample_count`; a caller can alter those semantic fields, recompute the unkeyed identity hash and matching result metadata, and pass validation.

This is P2 rather than P1 because the current production G01 path constructs the snapshot through `build_threshold_snapshot()` (`core_stage_worker.py:523-526`), which enforces exact repetitions and emits the frozen constants, and the selected worker source is identity-bound. No current Core execution path supplies an external snapshot builder.

Minimum fix: make `validate_threshold_snapshot()` assert the exact frozen metadata, require each key's sample count/repetition set to match `minimum_repetitions`, and validate bidirectional consistency among `executions`, `keys`, and `index`. Add mutation tests that recompute `identity_hash` after each semantic mutation; the existing test only changes metadata without recomputing the hash and therefore exercises hash integrity, not semantic validation.

### P2-N2: Repeatability terminal evidence refs do not directly cite the snapshot that supplies the reasons

Files:

- `finalize.py:279-323`
- `finalize.py:325-359`
- `tests/test_finalize.py:532-541`

Reproducible logic: the terminal analysis publishes `snapshot["identity_hash"]` and snapshot-derived `unusable_reasons`, but its G01 evidence row cites `result.json` at `$.repeatability_usable`. The external `threshold_snapshot.json` is validated and indirectly bound by the selected stage state, yet no formal evidence ref points directly to its identity or unusable-key records.

Impact is auditability, not verdict correctness: invalid snapshots are already rejected, and selected-attempt hashes bind the artifact set. The report reader nevertheless has one extra indirection when verifying why repeatability was unusable.

Minimum fix: add a G01 terminal evidence row for `threshold_snapshot.json` at `$.identity_hash`, with snapshot-derived unusable-key details in the row guard data; optionally add one exact row per unusable key.

### P2-N3: Reported sign fields are value-checked but not strictly type-checked

Files:

- `adapter_gate.py:283-304`
- `tests/test_adapter_gate.py:191-223`

Reproducible logic: Python equality treats `True == 1` and `1.0 == 1` as true. A JSON boolean or floating-point reported sign can therefore pass the reported/derived consistency comparison when the derived sign is `+1`, even though the intended schema is an integer sign.

This is P2 because both current workers emit `int(np.sign(...))` (`isaac_worker.py:2992-2995`, `mujoco_worker.py:1816-1819`), and the raw vectors still drive the physical gates.

Minimum fix: require `type(reported_sign) is int` and membership in `{-1, 0, 1}` before comparing with the derived sign. Add boolean and float mutation tests for target and velocity signs.

## Residual Test Blind Spots

- `tests/test_orchestrator.py:151-168` directly covers P60 seal recomputation but not a C70-specific case. The implementation branch is shared and exact, so this is not a blocker.
- `tests/test_orchestrator.py:171-223` materializes only a direct P10 stale selection. The helper is transitive and the frozen DAG review confirms P30-C70 are included, but an explicit stale P60/C70 resume fixture would improve regression strength.
- The fake G01 terminal test at `tests/test_orchestrator.py:113-148` stops at stage verification and its fake worker does not emit a valid threshold snapshot. Finalizer behavior is covered separately, but there is no single fake end-to-end test that runs terminal G01 through orchestrator verification, partial finalization, and `verify_final_state()`.
- The supplied full test results were not independently rerun in this review because doing so would violate the one-output-file write restriction.

## Final Decision

**APPROVED**

P0=0 and P1=0. Review-04's blocking findings are closed, and no new blocking defect was found. The P2 items should be tracked as evidence-schema and auditability hardening; they do not block the real Core V1 diagnostic run under the current frozen worker/source identities.
