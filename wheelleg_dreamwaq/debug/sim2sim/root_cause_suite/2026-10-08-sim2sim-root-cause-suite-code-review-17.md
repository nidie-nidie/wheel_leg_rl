# RootCauseSuite G01 Runtime Hotfix Independent Code Review 17

**Review date:** 2026-10-09  
**Frozen design:** `RootCauseSuiteCoreV1.17`  
**Decision:** **BLOCKED**  
**Counts:** P0 = 0, P1 = 2, P2 = 0

## Conclusion

Review-16 P1-1 is closed: neither the aggregate nor per-profile post-forward identity contains seed, RNG or reset-artifact identity, and the shared payload builder rejects any attempt to attach those fields to a post-forward phase.

Review-16 P2-1 is also closed as originally stated: clock normalization is now a pure function, the real robot-probe path calls it, and the test executes three nonzero absolute origins before merging the resulting schedules.

Review-16 P1-2 is not fully closed. The strict identity helpers correctly reject malformed raw inputs, but the real producer path coerces several authoritative values to float32 before those helpers see them. Original dtype and endianness drift can therefore be hidden. A second P1 was found in the new clock helper: NaN absolute/origin time passes the drift comparison and is replaced by a clean expected timestamp.

A real Isaac `p10_a` smoke and a full fresh Core V1 run are **not approved** until both P1 findings are fixed and independently re-reviewed.

## Findings

### P0

None.

### P1-1: Runtime capture coerces authoritative dtype before strict identity validation

**Files and lines**

- `isaac_worker.py:788-812` correctly requires exact shape, finite values and dtype string `<f4`, and encodes `{dtype, shape, values}`.
- `isaac_worker.py:815-902` correctly applies that helper to root position, orientation, linear/angular velocity, 26 q, 26 qd, two closure residuals and command.
- `isaac_worker.py:905-926` correctly applies it to returned actor observation `(25,)`, previous action `(6,)` and initialized history `(5,25)`.
- The real producer path first forces returned observation to float32 at `isaac_worker.py:2565-2570`.
- It first forces command and previous action to float32 at `isaac_worker.py:2575-2580`.
- It first forces both pre- and post-forward root positions to float32 at `isaac_worker.py:2585-2601`.
- These already-normalized arrays are the values later passed through `phase_state()` and returned-policy construction at `isaac_worker.py:2754-2805` and `isaac_worker.py:2914-2932`.
- Tests at `tests/test_isaac_worker.py:511-605` call the strict helpers directly. They do not exercise or guard the producer-side coercions above.

**Reproduction**

A no-write probe against the final on-disk source reproduced the bypass:

```text
producer_cast_masks_returned_obs_dtype <f8 <f4
producer_cast_masks_previous_action_endianness >f4 <f4
producer_cast_masks_root_dtype <f8 <f4
producer_cast_masks_command_dtype <f8 <f4
```

The first token on each line is the raw producer dtype; the second is the identity payload dtype after the same conversion used by the real call path. The helper rejects raw float64 and `>f4`, but it cannot reject information that was discarded before invocation.

**Impact**

- A source regression from float32 to float64 for returned observation, command, previous action or transformed root position is silently canonicalized rather than invalidating the stage.
- A non-canonical endian source is likewise hidden.
- The payload proves the dtype of the converted copy, not the dtype of the authoritative value read from Isaac.
- This conflicts with the frozen requirement that dtype/endianness mismatch invalidates the stage (`design.md:413`, `1276`).

**Minimum fix**

Remove `dtype=np.float32` from the identity capture paths for returned observation, command, previous action and pre/post root position. Preserve the raw NumPy dtype and let `_canonical_float32_identity_value()` validate it before any conversion. If a coordinate transform requires a canonical output dtype, validate the transform input and output explicitly rather than silently casting. Add a producer-boundary regression that supplies float64 and `>f4` values before capture and proves the same path used by `run_robot_probe()` rejects them.

**Review-16 status:** P1-2 remains **OPEN**. Shape checks, finite checks and payload encoding are closed; original producer dtype/endianness validation is not.

### P1-2: Clock normalization fails open for NaN and can synthesize clean evidence

**Files and lines**

- `_normalized_serial_physics_time_s()` is defined at `isaac_worker.py:1022-1044`.
- It checks only `physics_dt_s <= 0.0` at `isaac_worker.py:1033-1034` and does not require finite absolute time, origin, dt, relative time or expected time.
- The comparison at `isaac_worker.py:1039` is `abs(relative - expected) > 1e-10`. For NaN, that condition is false.
- The real probe passes Isaac's absolute clock and the reset origin into this helper at `isaac_worker.py:2663-2670`, then writes the returned value into the trace at `isaac_worker.py:2671-2674`.
- The behavioral test at `tests/test_isaac_worker.py:221-255` covers three valid finite origins but no NaN/Inf or non-finite dt case.

**Reproduction**

The final on-disk helper produced:

```text
nan_absolute accepted 0.005
nan_origin accepted 0.005
nan_dt accepted nan
```

Thus a NaN Isaac absolute clock or reset origin is replaced with the expected finite schedule value. The emitted trace can look valid even though the authoritative clock was invalid.

**Impact**

- Undeclared NaN in authoritative timing does not immediately invalidate the stage.
- A clock failure can be concealed by the generated expected schedule, defeating the purpose of the actual-vs-expected guard.
- This violates the frozen fail-closed rules for clock mismatch and undeclared NaN/Inf (`design.md:1276-1277`).

**Minimum fix**

Convert all scalar inputs once, then require `math.isfinite()` for `absolute_time_s`, `origin_s` and `physics_dt_s`; also require finite relative and expected results before comparison. Reject non-finite values with `EvidenceIntegrityError` or the suite's equivalent integrity exception. Add executed tests for NaN and positive/negative Inf in absolute time, origin and dt, plus the existing finite drift rejection.

### P2

None.

## Review-16 Closure Status

| Review-16 finding | Status | Evidence |
|---|---|---|
| P1-1 post-forward contamination | **CLOSED** | `_reset_phase_identity_payload()` adds reset identity only for pre-forward and rejects it for post-forward (`isaac_worker.py:929-956`). Aggregate post-forward construction omits reset identity (`2775-2795`), as do per-profile records (`2914-2928`). |
| P1-2 exact shape/dtype/endianness encoding | **NOT CLOSED** | Helpers and payload schema are correct (`788-926`), but producer-side float32 coercions at `2565-2601` hide original dtype/endianness before validation. |
| P2-1 source-only clock regression | **CLOSED** | Pure helper at `1022-1044`, actual call at `2663-2670`, executed three-origin merge test at `tests/test_isaac_worker.py:221-255`. The newly found NaN fail-open is a separate P1. |

## Confirmed Correct

### Post-forward phase separation

- `_reset_phase_identity_payload()` requires reset identity for pre-forward and rejects non-`None` reset identity for post-forward (`isaac_worker.py:929-956`).
- The aggregate pre-forward hash includes `first_common_reset`; aggregate post-forward does not (`isaac_worker.py:2773-2795`).
- Per-profile pre-forward records include `common_reset`; post-forward records do not (`isaac_worker.py:2904-2928`).
- `phase_state()` returns only the frozen physical fields, so seed/RNG/artifact identity cannot leak through the nested state object (`isaac_worker.py:2754-2771`).

### Shape, value and encoded-layout checks

- Every robot reset field has the requested exact row shape (`isaac_worker.py:852-901`).
- Raw state fields that reach the helper without producer coercion, including orientation, linear/angular velocity, q, qd and closure residual, are strictly checked as `<f4` and finite.
- Returned policy payload explicitly encodes observation `(25,)`, action `(6,)` and history `(5,25)` (`isaac_worker.py:905-926`).
- The encoding includes all three identity components: dtype, shape and values (`isaac_worker.py:808-812`).

### Normal finite clock dataflow

- The helper returns the integer-index-derived schedule rather than a large-number subtraction result (`isaac_worker.py:1035-1044`).
- The actual call passes the per-reset origin and current Isaac absolute time (`isaac_worker.py:2581`, `2663-2670`).
- Trace rows use the returned expected schedule (`isaac_worker.py:2671-2674`).
- The three-origin test proves identical merged schedules, exact initial zero, physics-dt increments and the `0.02 s` boundary (`tests/test_isaac_worker.py:221-255`).

### Existing serial/reset boundaries

- Profile order, repetition mapping, RNG restore, P60 row-0 cache selection and tensor copy/stack ownership remain unchanged from review-16.
- No new write target or formal-project mutation was found in the reviewed dataflow. The changes remain inside the debug suite.

## Verification Assessment

Provided validation evidence:

- Targeted suite: **49 passed**.
- `py_compile`: **passed**.
- Non-MuJoCo project suite under frozen cache/basetemp boundaries: **270 passed, 4 skipped**; skips are Windows symlink privilege only.
- MuJoCo interpreter suite: **12 passed**.
- Reported direct adversarial checks reject raw float64, `>f4` and 25-joint inputs and verify `{dtype,shape,values}` payloads.

Independent review actions:

- Read the frozen design, review-16, final implementation and final tests.
- Followed aggregate and per-profile payloads through their real call sites into `build_repeatability_record()`.
- Executed only no-write `python -B` adversarial probes. No pytest or Isaac run was started because the requested write boundary permits only this report.

The supplied green tests do not cover producer-side coercion or non-finite clock inputs, so they do not close the two P1 findings.

## Reviewed SHA256

| File | SHA256 |
|---|---|
| `2026-10-07-sim2sim-root-cause-suite-design.md` | `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD` |
| `2026-10-08-sim2sim-root-cause-suite-code-review-16.md` | `7EC7A762A130B6A7683A7EFA5B0C0348A59E6CFF805BEF289DEFC5C037D4C0BE` |
| `isaac_worker.py` | `9D0AF3D7D585FD0FCEFB142ED9CAADE448C4B82F755678907F3199EB5C93E4D3` |
| `tests/test_isaac_worker.py` | `5373C021C34B362864207BB284224CFE30CAA84DC54FD9A664F7F2BCFA1CD5F0` |
| `repeatability.py` | `07308D8B1C47C54088B7CF28EDC801DDE8C31A3DC316DE1813BB9166ECB8B19F` |
| `scenario_catalog.py` | `D60E67496650D6F36A09F21FAE106D33337BC9EB2AAE9F002ED0384A40160D10` |
| `isaac_probe_env.py` | `F4A192BEB6A2871F168D482A55DAE47011D7F3BDD92234AF80234FDC01C1F00E` |
| `../isaac_debug_env.py` | `3CDB03D723BEB8D2024E7A4F512ADF8C51560C3F1C8209AA9B9D01E7725F56C2` |
| formal `randomization.py` | `92A1A41A0E225856EBFCD2998287D781AEB65685558873D42969FBB2E2FF1CE6` |
| Isaac Lab `direct_rl_env.py` | `3A7573303D83EC8C816D11CA6E8C8997C92F6765AE5D2E0D2A369E65D2CEC3C0` |

## Final Decision

**BLOCKED** - P0 = 0, P1 = 2, P2 = 0.

- Real Isaac `p10_a` smoke: **NOT APPROVED**.
- Full fresh Core V1 run: **NOT APPROVED**.

After both P1 fixes, rerun the focused adversarial tests first, then one fresh Isaac `p10_a` smoke. A full fresh Core V1 run should be allowed only after the smoke evidence passes identity and repeatability verification.
