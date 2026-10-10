# RootCauseSuite G01 Runtime Hotfix Independent Code Review 16

**Review date:** 2026-10-09  
**Frozen design:** `RootCauseSuiteCoreV1.17`  
**Decision:** **BLOCKED**  
**Counts:** P0 = 0, P1 = 2, P2 = 1

## Conclusion

The serial-repetition architecture is coherent: one environment is used per unique excitation semantic, the same custom randomization RNG state is restored before each fresh reset, trace columns are rebuilt in semantics-major/repetition-minor order, P60 selects frozen reset-cache row 0, and the final clock correction converts Isaac's non-resetting global `_sim_step_counter` into a verified per-reset schedule.

Two frozen evidence-identity requirements are still violated:

1. `post_forward_state_hash` contains reset/RNG/artifact fields assigned exclusively to `pre_forward_initial_condition_hash`.
2. Reset phase state serialization accepts non-canonical dtypes and incorrect vector lengths, then discards dtype/endianness before hashing.

Both affect verdict-bearing repeatability evidence. A real Core V1 fresh run is not approved until both P1 findings are closed and independently re-reviewed.

## P0 Findings

None.

## P1 Findings

### P1-1: Post-forward identity is contaminated by pre-forward reset identity

**Evidence**

- The design assigns written root pose/velocity, 26-joint q/qd, command, seed, RNG state and reset artifact to pre-forward identity (`2026-10-07-sim2sim-root-cause-suite-design.md:400-402`).
- It assigns only forwarded root pose/velocity, 26-joint q/qd and closure residuals to post-forward identity (`design.md:403-404`) and prohibits merging the phase identities (`design.md:413`).
- `common_reset_identity()` creates the pre-only seed/RNG/reset-artifact fields at `isaac_worker.py:2640-2653`.
- The top-level post-forward hash expands those fields at `isaac_worker.py:2698-2706`.
- Each repeatability record also expands them into `post_forward_payload` at `isaac_worker.py:2830-2836`.

**Reproduction**

A read-only pure-Python probe constructed the same post-forward physical state with and without `common_reset_identity`. `stable_hash` changed (`post_common_reset_changes_hash=True`). The emitted hash therefore does not represent the frozen post-forward state contract.

**Impact**

- The phase hash cannot be audited as the constraint-projected physical state alone.
- Seed/RNG/artifact identity is represented in both pre-forward and post-forward phases.
- A post-forward identity change can be reported even when the projected state is byte-identical.

**Minimum fix**

Remove `**first_common_reset` from the top-level post-forward payload and `**common_reset` from each record's post-forward payload. Keep them in pre-forward payloads. Add a regression proving that reset identity changes affect only pre-forward hash, while physical/closure changes affect post-forward hash.

### P1-2: Phase state does not enforce or encode exact shape and dtype/endianness

**Evidence**

- The design requires canonical order, units, dtype/endianness and availability (`design.md:413`).
- Any shape or dtype mismatch must invalidate the stage (`design.md:1272-1277`).
- `profile_row()` accepts every floating, integer or boolean dtype (`isaac_worker.py:801-815`) and returns `row.tolist()` (`isaac_worker.py:820`), discarding dtype and endianness.
- Exact shapes are not checked for root position, quaternion, linear/angular velocity, 26-joint q/qd or two closure residuals (`isaac_worker.py:834-873`). Only command length is checked, without an exact dtype requirement (`isaac_worker.py:860-865`).
- Tests cover field membership and non-finite rejection, but not dtype, endianness or authoritative vector lengths (`tests/test_isaac_worker.py:365-436`).

**Reproduction**

The final on-disk source produced:

```text
dtype_hash_equal True
all_hinge_position_named accepted_length 25
base_orientation_control_wxyz accepted_length 5
loop_closure_error accepted_length 3
```

The first result used otherwise identical zero-valued float32 and float64 states. The other results show malformed authoritative dimensions are accepted rather than rejected.

**Impact**

- Numerically equal float32/float64 payloads can have the same phase hash although dtype is part of frozen identity.
- Integer or boolean physical state can be accepted.
- A 25-joint vector, five-element quaternion or three-element closure residual can enter verdict-bearing records.

**Minimum fix**

Enforce exact canonical floating dtype/endianness and row shape before hashing: root position `(3,)`, orientation `(4,)`, linear velocity `(3,)`, angular velocity `(3,)`, q `(26,)`, qd `(26,)`, closure residual `(2,)`, command `(3,)`. Either reject other dtypes or serialize a validated canonical `{dtype, shape, values}` object. Add mutations for float64, integer, boolean, non-native/big-endian data, extra dimensions and every wrong vector length.

## P2 Findings

### P2-1: The per-reset clock regression is source-text only

**Evidence**

- The implementation captures `physics_time_origin_s` after each reset (`isaac_worker.py:2474`).
- It verifies actual relative time against `(tick * decimation + substep + 1) * physics_dt` within `1e-10`, then writes the expected schedule to the trace (`isaac_worker.py:2556-2575`).
- This compensates for the global counter in `isaac_debug_env.py:323-345` and yields the shared vector required by `repeatability.py:239-250`.
- The regression only searches worker source strings (`tests/test_isaac_worker.py:205-215`); it does not execute the normalization/merge behavior.

**Impact**

The implementation is correct by inspection, but a later refactor could leave the searched strings while disconnecting their dataflow.

**Minimum fix**

Extract a pure schedule helper or test equivalent data with three distinct nonzero absolute origins. Assert one exact merged schedule, initial `t=0`, `physics_dt` increments and the frozen `0.02 s` excitation boundary.

## Confirmed Correct or Unchanged

### Serial profile and trace order

- The catalog emits contiguous repetition groups inside each excitation semantic (`scenario_catalog.py:137-212`).
- `_serial_robot_profiles()` validates exact repetitions `0..N-1` and invariant semantics (`isaac_worker.py:877-901`).
- `_merge_serial_repetition_arrays()` restores semantics-major/repetition-minor order (`isaac_worker.py:904-937`).
- Records use `profile_index // repetitions` and `repetition_runs[repetition]` (`isaac_worker.py:2790-2793`).
- Tests cover environment count, noncontiguous repetition rejection and merged order (`tests/test_isaac_worker.py:133-169`).

### RNG and reset semantics

- Custom randomization state is captured once, restored before every repetition, then followed by a fresh `env.reset(seed=0)` (`isaac_worker.py:2351`, `2450-2454`).
- Generator states are cloned and restored per stream (`randomization.py:355-391`).
- Isaac Lab `seed()` resets global libraries but does not replace the suite's custom streams (`direct_rl_env.py:420-438`).

### P60 reset cache

- Only P60 may receive the frozen cache (`isaac_worker.py:2309-2319`).
- A one-environment serial P60 probe selects source row 0 (`isaac_probe_env.py:221-231`); multi-environment replication remains explicitly guarded (`isaac_probe_env.py:232-247`).
- Metadata records row 0 for every expanded profile (`isaac_worker.py:2871-2875`).

### Returned policy and tensor ownership

- Returned identity contains actor observation, previous action and five initialized history frames (`isaac_worker.py:2674-2683`).
- Reset snapshots are detached/cloned (`isaac_debug_env.py:279-310`); commands/actions are copied and merged traces are newly allocated. No actionable alias was found.

### Final clock correction

- Capturing the global counter origin before the first new step makes the first relative sample one physics step after reset.
- Actual relative time is guarded at `1e-10`; the trace uses integer-index-derived expected time, giving byte-stable values across repetitions and satisfying the downstream `1e-12` shared-vector check.
- The reset row remains exactly `t=0` (`isaac_worker.py:2496-2508`). No off-by-one defect was found.

### Write boundary and formal semantics

- No new write target was introduced. Isaac logs stay under `<run>/runtime_cache/isaaclab/logs` (`isaac_worker.py:2331-2333`); result/trace writes remain under guarded worker output.
- Formal configuration is hashed before and after the probe (`isaac_worker.py:2322`, `2878-2879`). No write to formal training source, USD, MuJoCo formal parameters or checkpoints was found.
- `root-cause-core-v1-20261009-061548` froze an older worker identity. The final source requires a new fresh run after P1 closure; the old run must remain historical evidence.

## Verification Assessment

Provided for the final files:

- `tests/test_isaac_worker.py`: **28 passed**.
- `py_compile`: **passed**.

Independent review:

- Read the frozen design and final implementation/test paths around profile construction, reset, time, phase identity and repeatability records.
- Executed a no-write pure-Python adversarial probe confirming both P1 reproductions.
- Did not execute a real Isaac/PhysX smoke run. It should run only after P1 closure.

## Reviewed SHA256

| File | SHA256 |
|---|---|
| `2026-10-07-sim2sim-root-cause-suite-design.md` | `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD` |
| `isaac_worker.py` | `7CA1F14DFEAF64E101AABC28384858D36CBF0C4574EB2211BA4913B9191320CD` |
| `tests/test_isaac_worker.py` | `ABC5A072A085B682B0A7EFBF26CD5F5512BB881BB1F7FFA4D619C1B5A0CBB3EF` |
| `repeatability.py` | `07308D8B1C47C54088B7CF28EDC801DDE8C31A3DC316DE1813BB9166ECB8B19F` |
| `scenario_catalog.py` | `D60E67496650D6F36A09F21FAE106D33337BC9EB2AAE9F002ED0384A40160D10` |
| `isaac_probe_env.py` | `F4A192BEB6A2871F168D482A55DAE47011D7F3BDD92234AF80234FDC01C1F00E` |
| `../isaac_debug_env.py` | `3CDB03D723BEB8D2024E7A4F512ADF8C51560C3F1C8209AA9B9D01E7725F56C2` |
| formal `randomization.py` | `92A1A41A0E225856EBFCD2998287D781AEB65685558873D42969FBB2E2FF1CE6` |
| Isaac Lab `direct_rl_env.py` | `3A7573303D83EC8C816D11CA6E8C8997C92F6765AE5D2E0D2A369E65D2CEC3C0` |

## Final Decision

**BLOCKED** - P0 = 0, P1 = 2, P2 = 1. The final clock hotfix and serial execution model are accepted, but a real Core V1 run is not permitted until P1-1 and P1-2 are fixed and re-reviewed.
