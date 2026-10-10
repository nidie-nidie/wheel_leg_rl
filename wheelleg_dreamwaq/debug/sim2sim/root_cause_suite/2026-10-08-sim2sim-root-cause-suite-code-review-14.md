# Sim2Sim RootCauseSuite Core V1 Code Review 14

Date: 2026-10-08  
Review mode: independent, defect-first, read-only second-round review  
Frozen design: `RootCauseSuiteCoreV1.17`  
Previous review: `2026-10-08-sim2sim-root-cause-suite-code-review-13.md`  
Result: **BLOCKED**

## Conclusion

- P0: 0
- P1: 0
- P2: 1
- Review-13 P1-1: **CLOSED**
- Review-13 P1-2: **CLOSED**
- Review-13 P2-1: **PARTIALLY CLOSED; one consumer-side malformed-tag bypass remains**

The exact composed-USD infinity declaration and reset/contact fail-closed repairs are materially correct. The producer accepts only 12 frozen `(prim path, attribute)` declarations accounting for the 17 robot and 6 sphere inspection occurrences. Reset serialization now limits unavailable state to six contact fields, requires exact contact dtype/shape/sentinels, and rejects non-finite authoritative state before per-profile world position is removed.

However, a mapping that simultaneously removes `kind` and changes `schema_version` to a non-reserved value is treated as ordinary data even when it retains `semantics`, `source`, and `attribute`. A self-consistent result with all identity hashes recomputed is accepted by `validate_result_identity()`. Review-13 P2-1 is therefore not strictly closed.

Because this review explicitly requires strict closure of P1-1, P1-2, and P2-1, the decision is **BLOCKED** despite P0/P1 being zero. A disposable non-evidentiary Isaac smoke may still be useful, but this revision should not be used for an authoritative full Core V1 run until the P2 gap is closed.

## Findings

### P0

None.

### P1

None.

### P2

#### P2-1: combined malformed-tag mutations bypass `validate_result_identity()`

Files and lines:

- `contracts.py:25-29,126-133,231-311`
- `causal_contract.py:262-282,390-451`
- `tests/test_contracts.py:115-177`
- `tests/test_causal_contract.py:279-331`
- design `2026-10-07-sim2sim-root-cause-suite-design.md:1264-1279`

Reproducible logic:

1. `validate_declared_infinity_payload()` calls a mapping tag-like only if it contains `kind+semantics`, contains `kind+attribute`, or its `schema_version` begins with `RootCauseDeclaredInfinity` (`contracts.py:254-263`).
2. Remove `kind` from a valid tag and set `schema_version="WrongVersion"`. The mapping retains sentinel-specific `semantics`, `source`, and `attribute`, but none of those predicates matches.
3. The validator visits the remaining strings as ordinary data. Exact-key/value checks at `contracts.py:264-277` are never reached.
4. Put that object at the otherwise legal `physxScene:maxBiasCoefficient` value path and recompute `configuration_semantics_hash`, `configuration_hash`, and `repeatability_key`. `validate_result_identity()` accepts it.
5. Tests mutate one field at a time, add an extra key, move a valid tag, or partially replace a shaped sentinel. They do not combine a missing discriminator with a non-reserved version or manually rehash a malformed result.

Read-only behavior probes:

```text
valid                          ACCEPT
missing_kind                   REJECT malformed infinity tag
missing_kind_wrong_version     ACCEPT
only_semantic_shape            ACCEPT
validate_result_identity(...)  ACCEPT self-consistent malformed tag
```

Impact:

- The trusted current producer does not generate this object, so this remains P2 rather than P1.
- Future producer drift or an internally rehashed malformed artifact can pass the consumer with a non-schema value in a sentinel-reserved field.
- Hash consistency is proven, but sentinel schema validity is not proven for every malformed representation.
- Recognized shaped tags cannot be partially forged because `contracts.py:304-311` checks the Cartesian index set. The gap remains when all leaves evade tag recognition.

Minimum repair:

1. At a path where `declaration_for_path(path)` returns a spec, require any mapping value to equal the exact declared tag before heuristic classification.
2. Fail closed on combinations of sentinel-specific keys even when both `kind` and the reserved schema prefix are damaged.
3. Add a consumer test removing `kind`, changing `schema_version`, recomputing all identity hashes, and requiring rejection.
4. Add the equivalent all-leaves malformed mutation for shaped `physics:centerOfMass`.

## Review-13 Closure Status

### P1-1: exact legal infinity path contract - CLOSED

The implementation no longer performs whole-object `Inf` replacement.

- `contracts.py:80-116` freezes 12 exact `(prim path, attribute)` declarations.
- Each declaration binds sign, field-specific semantics, attribute name, source, and exact scalar or `(3,)` shape (`contracts.py:58-76,126-133`).
- `_usd_value()` supplies the composed prim path and attribute at the USD read boundary (`isaac_probe_env.py:52-83`).
- `encode_declared_usd_value()` rejects every NaN, undeclared infinity, wrong sign, wrong shape, mixed finite/infinite shaped value, and non-numeric leaf (`contracts.py:173-228`).
- `ProbeEnvironmentIdentity` validates the complete variant/closure/drive/compiled payload after confirming the exact four closure prim paths (`isaac_probe_env.py:385-546`). Raw non-finite mass, inertia, gain, material, actuator, or undeclared solver values therefore fail closed even when they did not pass through `_usd_value()`.
- `result_identity_fields()` validates Isaac configuration semantics and forbids every infinity tag in transform semantics before hashing (`causal_contract.py:262-307`).
- `validate_result_identity()` repeats the same check before accepting stored hashes (`causal_contract.py:390-425`), subject only to the P2 malformed-tag recognition gap above.
- Negative tests cover raw non-finite mass, inertia, stiffness, damping, material, and solver fields; wrong sign; undeclared fields; illegal paths; every tag field; extra keys; and partial shaped sentinels (`tests/test_contracts.py:41-177`, `tests/test_causal_contract.py:279-331`).

The only allowed solver infinity is the exact composed `/physicsScene` `physxScene:maxBiasCoefficient` declaration. All other raw or tagged solver infinities remain undeclared and are rejected.

### Disposable inspection reconciliation

The inspection is correctly isolated under:

`runs/inspection-20261008-infinity-paths/NON_EVIDENTIARY`

Robot `p10_a-r3` contains 17 occurrences:

- 8 top-level closure records: four exact closure prim paths times `physics:lowerLimit`/`physics:upperLimit`.
- 8 configuration-semantic closure records: the same four exact prim paths times the same two attributes.
- 1 physics-scene occurrence: `/physicsScene`, `physxScene:maxBiasCoefficient`.

The four exact closure prim paths are:

1. `/World/envs/env_0/Robot/jIO/jIO_loop_closure`
2. `/World/envs/env_0/Robot/jKN/jKN_loop_closure`
3. `/World/envs/env_0/Robot/jEC/jAG_loop_closure`
4. `/World/envs/env_0/Robot/jCF/jCF_revolute_joint`

Sphere `sphere-impact` contains 6 occurrences:

- 1 physics-scene `physxScene:maxBiasCoefficient` positive infinity.
- 3 `physics:centerOfMass` negative-infinity vector elements at the exact coupon prim.
- 1 `physxRigidBody:maxContactImpulse` positive infinity at the exact coupon prim.
- 1 `physxRigidBody:maxLinearVelocity` positive infinity at the exact coupon prim.

These 23 occurrences map exactly to the current 12 unique source declarations. No inspected mass, inertia, gain, material, or additional solver field requires an infinity declaration.

The disposable files were produced by the earlier three-key generic tag (`schema_version`, `kind`, `semantics="unbounded"`). They are path-discovery artifacts only and would not pass the current five-key consumer schema. This is correct for a directory explicitly marked `NON_EVIDENTIARY`; the artifacts must not be selected as formal Core evidence.

### P1-2: reset/contact non-finite masking - CLOSED

- Only six frozen contact fields are eligible for unavailable markers (`isaac_worker.py:626-636`).
- Contact integer fields must retain exact `int8`; contact force/impulse fields must retain exact `float32` (`isaac_worker.py:661-696`).
- Shapes are derived from exact `wheel_contact_active[...,2]` and checked for every contact field (`isaac_worker.py:667-685`).
- Disabled wheel-contact state requires exact all-`-1` active values and exact all-NaN normal force/impulse/vector values. Mixed sentinels fail (`isaac_worker.py:698-713`).
- Enabled wheel-contact state requires exact `0/1` activity and finite normal fields (`isaac_worker.py:714-723`).
- Friction-force unavailability and unexpected-contact unavailability each have their own exact sentinel and reason (`isaac_worker.py:724-775`).
- Every non-contact reset field is authoritative. Floating NaN or either infinity raises rather than becoming unavailable (`isaac_worker.py:734-750`).
- `_repeatability_profile_state()` validates the selected complete state before removing `base_com_position_engine_world` (`isaac_worker.py:788-803`). A non-finite world position is rejected before omission.
- Tests exercise disabled/enabled contact states, mixed sentinels, wrong dtype, wrong shape, NaN/+Inf/-Inf in representative authoritative joint/root fields, and non-finite world position before removal (`tests/test_isaac_worker.py:201-319`).

The frozen runtime producer always includes `base_com_position_engine_world` (`../isaac_debug_env.py:238-256`). The serializer does not independently require a complete authoritative key set; that is residual schema hardening, but it does not reopen review-13 P1-2's non-finite masking defect for the current producer.

## Robot Transform Return

The earlier misplaced-return defect remains closed.

- `_isaac_robot_transform_semantics()` has a reachable independent return at `isaac_worker.py:216-246`.
- `_isaac_sphere_transform_semantics()` has its own return at `isaac_worker.py:377-389`.
- The test calls target and direct variants and asserts the returned actuation/closure payload (`tests/test_isaac_worker.py:89-113`).

## Frozen Scope and Boundary

Verified unchanged:

- Architecture v0.21 SHA256: `5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE`
- Design `RootCauseSuiteCoreV1.17` SHA256: `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD`
- Factor-path allowlist SHA256: `4CF2A34D4E3725A78DC0028AD92FF6DCD04E820874CFFCD8757F267F8DADDD85`
- `../isaac_debug_env.py` SHA256 remains the review-13 value: `3CDB03D723BEB8D2024E7A4F512ADF8C51560C3F1C8209AA9B9D01E7725F56C2`

All reviewed implementation and test changes are inside `debug/sim2sim/root_cause_suite`. The architecture, design, factor allowlist, and external debug environment were not changed by this hotfix.

## Test Assessment

Accepted supplied records; this review did not rerun pytest:

- Project `.venv`, excluding `test_mujoco_worker.py`: `242 passed, 4 skipped`.
- The four skips are reported as Windows symlink-creation privilege only.
- `sim2sim/mujoco/.venv`, `test_mujoco_worker.py`: `12 passed`.
- Targeted hotfix tests: `74 passed`.
- `py_compile`: passed.

The regression record supports closure of both P1 findings. It does not close P2-1 because current tests cover one-field mutations, not the reproduced missing-discriminator plus wrong-version representation.

## Reviewed SHA256

### Frozen and prior-review files

- `docs/2026-10-03-wheelleg-dreamwaq-architecture.md`: `5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE`
- `2026-10-07-sim2sim-root-cause-suite-design.md`: `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD`
- `2026-10-08-sim2sim-root-cause-suite-code-review-13.md`: `C2B1D0A6E718D01022CF7FE70248329C31DAB3DBF2AEB650FE17A41E93161C4D`
- `factor_path_allowlist.json`: `4CF2A34D4E3725A78DC0028AD92FF6DCD04E820874CFFCD8757F267F8DADDD85`
- `../isaac_debug_env.py`: `3CDB03D723BEB8D2024E7A4F512ADF8C51560C3F1C8209AA9B9D01E7725F56C2`

### Current implementation and tests

- `contracts.py`: `A410B107F03A7F85C2CFC06C6BB76933ED9DCEB6AD3C2F0E0E1CA43179781130`
- `isaac_probe_env.py`: `F4A192BEB6A2871F168D482A55DAE47011D7F3BDD92234AF80234FDC01C1F00E`
- `causal_contract.py`: `5ACACEB60C6C653D87A272D6C7C8BE1A26DB26EE9EEB55BB0E7F379CCB690176`
- `isaac_worker.py`: `6C720DC307C85CAD9A344EE0E29465B79B60422CA06C64DAF07B3EEC9BEC05C8`
- `tests/test_contracts.py`: `92693616883238155E7E5E700C0B66283DD5312D67764C1FA1FFC66BE8B75224`
- `tests/test_causal_contract.py`: `DB73BC7747B624130BF8A4634C2F6EB7493FCAED9DFE4A4D54732038333686CD`
- `tests/test_isaac_worker.py`: `0018113F5616F64FDA54F68140572E1564025D0DE8823E6917D084FC0987AD4B`

### Disposable inspection artifacts

- `p10_a-r3/identity.json`: `C205CC046B883177B45A99AD7AB68A8716B18FD1C3DDED59018C0A94C7427FB7`
- `p10_a-r3/result.json`: `2D3DF9B71758B0A321CB66E8BBCDB9318BC1221C5DA1A6079E8FF8CCB2C1AE5F`
- `sphere-impact/identity.json`: `17FA04D0B996C8245D69C070071B2079CFB430B20028B38C0502C222C01BF919`
- `sphere-impact/result.json`: `808A6680760518A7A6057A75E55DDB117CE7FD80E6B27A1E77FEE8E049A5A6D2`

## Decision

**Final decision: BLOCKED.**

Review-13 P1-1 and P1-2 are closed, the exact 17/6 inspection paths are represented by frozen source declarations, reset physical NaN/Inf no longer becomes unavailable, world position is validated before omission, and the robot transform return remains correct. Review-13 P2-1 is not strictly closed because a combined malformed tag can evade classification and pass a fully rehashed `validate_result_identity()` payload.
