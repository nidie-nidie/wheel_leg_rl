# RootCauseSuite G01 Runtime Hotfix Independent Code Review 18

**Review date:** 2026-10-09  
**Frozen design:** `RootCauseSuiteCoreV1.17`  
**Decision:** **APPROVED**  
**Counts:** P0 = 0, P1 = 0, P2 = 0

## Conclusion

Both review-17 P1 findings are closed in the final on-disk implementation.

The real `_collect_robot_probe()` path captures returned observation, command, previous action and transformed pre/post root position without dtype or endianness coercion. The captured arrays are later passed to strict identity helpers that enforce exact `<f4`, exact frozen shapes, finite values and explicit `{dtype, shape, values}` encoding.

The clock helper now rejects non-finite absolute time, origin, dt, relative time and expected time before accepting the sample, while retaining the finite drift check and deterministic schedule written to the trace.

Previously closed phase-separation, returned-policy, serial repetition and schedule behavior have not regressed. No new P0, P1 or P2 defect was found.

- Real Isaac `p10_a` smoke: **APPROVED**.
- Full fresh Core V1 run: **CONDITIONALLY APPROVED after the smoke passes** identity, trace, repeatability and formal-file verification. It must be a fresh run under the current source identity, not a resume of an older failed run.

## Findings

### P0

None.

### P1

None.

### P2

None.

## Review-17 Finding Closure

### Review-17 P1-1: Producer-side dtype/endianness coercion - CLOSED

**Capture boundary**

- `_capture_authoritative_identity_array()` converts Torch/array input to NumPy without a dtype argument and returns a copy (`isaac_worker.py:788-796`).
- `_to_numpy()` preserves Torch/NumPy dtype and copies without numeric conversion (`isaac_worker.py:595-605`).
- Returned observation is captured through this helper at `isaac_worker.py:2592-2595`; its full environment layout is checked at `2596-2599`.
- Command and previous action use the same capture path at `isaac_worker.py:2600-2605`.
- Transformed pre/post root positions use the same path at `isaac_worker.py:2610-2621`.
- `transform_usd_vector_to_control()` casts the rotation matrix to the input tensor's dtype and therefore preserves the authoritative vector dtype (`source/.../schemas/frames.py:28-32`).
- The captured arrays are independent copies and are stored per repetition at `isaac_worker.py:2725-2735`, so subsequent environment mutation cannot change identity input.

**Strict identity boundary**

- `_canonical_float32_identity_value()` requires exact shape, exact little-endian float32 `<f4`, finite values and emits `{dtype, shape, values}` (`isaac_worker.py:799-823`).
- `_repeatability_profile_state()` applies it to root position `(3,)`, orientation `(4,)`, linear velocity `(3,)`, angular velocity `(3,)`, q `(26,)`, qd `(26,)`, closure residual `(2,)` and command `(3,)` (`isaac_worker.py:826-913`).
- `_repeatability_returned_policy_state()` applies it to returned observation `(25,)`, previous action `(6,)` and initialized history `(5,25)` (`isaac_worker.py:916-937`).
- Aggregate identities consume the captured values at `isaac_worker.py:2773-2824`.
- Every repetition/profile record consumes the corresponding captured values at `isaac_worker.py:2899-2953`; later repetitions cannot bypass validation even though the aggregate causal identity is based on repetition zero.

**Independent probe**

The final source preserved the adversarial raw dtypes and then rejected all four at the strict boundary:

```text
captured_dtypes <f8 >f4 <f8 <f8
strict_case 1 rejected
strict_case 2 rejected
strict_case 3 rejected
strict_case 4 rejected
```

The four cases are returned observation, previous action, transformed root position and command respectively.

### Review-17 P1-2: Non-finite clock fail-open - CLOSED

- `_normalized_serial_physics_time_s()` converts its three scalar clock inputs once (`isaac_worker.py:1046-1048`).
- Absolute time, origin and dt must all be finite (`isaac_worker.py:1049-1052`).
- Positive dt remains mandatory (`isaac_worker.py:1053-1054`).
- Relative and expected time are calculated only after input validation and must also be finite (`isaac_worker.py:1055-1065`).
- The finite drift check remains active at `isaac_worker.py:1066-1070`.
- The real robot probe supplies Isaac absolute time, per-reset origin and physics dt to the helper at `isaac_worker.py:2682-2689`, then writes only its verified deterministic schedule to the trace at `2690-2693`.

Independent no-write probing rejected all nine NaN/+Inf/-Inf input combinations and a separate finite drift case:

```text
absolute_time_s: NaN/+Inf/-Inf rejected
origin_s: NaN/+Inf/-Inf rejected
physics_dt_s: NaN/+Inf/-Inf rejected
finite_drift rejected
```

The executed regression tests cover the same nine non-finite input cases and finite drift (`tests/test_isaac_worker.py:259-286`).

## Previously Closed Items Rechecked

### Aggregate and per-profile post-forward separation

- `_reset_phase_identity_payload()` requires reset identity for pre-forward and rejects it for post-forward (`isaac_worker.py:940-967`).
- Aggregate pre-forward includes `first_common_reset`, while aggregate post-forward supplies no reset identity (`isaac_worker.py:2792-2814`).
- Per-profile pre-forward includes `common_reset`, while post-forward again supplies no reset identity (`isaac_worker.py:2923-2947`).
- `phase_state()` returns only the frozen physical state fields (`isaac_worker.py:2773-2790`), so seed/RNG/reset-artifact identity cannot leak through the nested state object.
- An independent probe confirmed that passing reset identity to post-forward is rejected.

### Frozen payload layout

- All requested reset state dimensions are exact and all encoded leaves include dtype, shape and values.
- Returned policy identity remains observation `(25,)`, previous action `(6,)` and history `(5,25)`.
- NaN/Inf physical values are rejected by the strict identity helper (`isaac_worker.py:815-818`).
- Test coverage includes every requested reset-state shape, non-canonical dtype/endianness, encoded payload structure and returned-policy layout (`tests/test_isaac_worker.py:440-682`).

### Serial schedule and repetition mapping

- The per-reset clock origin is captured before stepping (`isaac_worker.py:2606`).
- The verified expected schedule is passed to trace row construction (`isaac_worker.py:2682-2693`).
- Serial arrays retain exact shape/dtype equality and semantics-major/repetition-minor merge ordering (`isaac_worker.py:997-1030`).
- The three-origin test executes the pure clock helper, merges three schedules and checks exact samples, initial zero, dt spacing and the `0.02 s` boundary (`tests/test_isaac_worker.py:222-256`).
- Profile-to-run mapping remains `serial_profile_index = profile_index // repetitions` plus `repetition_runs[repetition]` (`isaac_worker.py:2899-2903`).

### Write and formal boundaries

- The reviewed changes introduce no new output destination. Existing probe output and Isaac logs remain under the guarded run/output tree.
- Formal configuration before/after identity remains checked before result finalization (`isaac_worker.py:2990-2994`).
- No modification to formal training code, USD, MuJoCo parameters, checkpoints, design or prior evidence was found in this review scope.

## Test and Residual-Risk Assessment

Provided verification evidence:

- Targeted tests: **60 passed**.
- `py_compile`: **passed**.
- Non-MuJoCo suite: **281 passed, 4 skipped**; skips are Windows symlink privilege only.
- MuJoCo interpreter suite: **12 passed**.
- Adversarial producer probes preserve `<f8`, `>f4`, `<f8`, `<f8` and then fail strict identity validation.
- Nine non-finite clock cases are rejected.

Independent review actions:

- Read the final implementation and tests and followed every reviewed value through the real `_collect_robot_probe()` call path.
- Executed only no-write `python -B` adversarial probes. No pytest or Isaac process was started because this review permits writing only the new report.

Residual integration boundary:

- Unit tests exercise the capture/strict-helper composition but do not instantiate the complete Isaac producer. This is not a current code defect because the production wiring was verified directly in the final source. The approved `p10_a` smoke is the required runtime acceptance gate before the full fresh Core run.

## Reviewed SHA256

| File | SHA256 |
|---|---|
| `2026-10-07-sim2sim-root-cause-suite-design.md` | `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD` |
| `2026-10-08-sim2sim-root-cause-suite-code-review-17.md` | `E45EBE3C2D2007695F1BD268F488CF201BBFEC90A95A83C56F9162201F9EA3CC` |
| `isaac_worker.py` | `BCF6A3530FAB09E6FFC3A6F1633B225BC3C8D2AC6BB5A39DA7294F8706D644B8` |
| `tests/test_isaac_worker.py` | `FF355E190E384CEF55A0536777D6A6B3E5570EC366612C403BBC6D4EDA3A6D23` |
| `repeatability.py` | `07308D8B1C47C54088B7CF28EDC801DDE8C31A3DC316DE1813BB9166ECB8B19F` |
| `scenario_catalog.py` | `D60E67496650D6F36A09F21FAE106D33337BC9EB2AAE9F002ED0384A40160D10` |
| `isaac_probe_env.py` | `F4A192BEB6A2871F168D482A55DAE47011D7F3BDD92234AF80234FDC01C1F00E` |
| `../isaac_debug_env.py` | `3CDB03D723BEB8D2024E7A4F512ADF8C51560C3F1C8209AA9B9D01E7725F56C2` |
| formal `schemas/frames.py` | `3EEAA49092C766A26D8D54B1CD5DF40C737E2A661EB405109269F349B7BF1D63` |

## Final Decision

**APPROVED** - P0 = 0, P1 = 0, P2 = 0.

- Real Isaac `p10_a` smoke: **APPROVED NOW**.
- Full fresh Core V1 run: **APPROVED ONLY AFTER** the smoke completes successfully and its result/trace hashes, repeatability records, source identity and formal-file guards verify cleanly.
