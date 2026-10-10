# Sim2Sim RootCauseSuite Core V1 Code Review 13

Date: 2026-10-08  
Review mode: independent, defect-first, read-only G01 hotfix review  
Frozen design: `RootCauseSuiteCoreV1.17`  
Result: **BLOCKED**

## Findings

### P0

None.

### P1

#### P1-1: every encountered infinity is treated as declared without a frozen path contract

Files and lines:

- `contracts.py:23,46-61`
- `isaac_probe_env.py:46-64,373-483`
- `isaac_worker.py:118-212`
- `causal_contract.py:59-146,210-266,310-361`
- `tests/test_contracts.py:37-60`
- design `2026-10-07-sim2sim-root-cause-suite-design.md:1270-1279`

Reproducible logic:

1. `encode_declared_infinities()` accepts only a value. It receives no JSON pointer, source field, schema, or allow-list proving that an infinity was declared in advance.
2. Therefore `encode_declared_infinities({"bodies":{"base":{"mass":float("inf")}}})` produces a valid `RootCauseDeclaredInfinityV1` object even though infinite mass is not a declared unbounded limit.
3. `validate_variant_identity()` recursively applies the encoder to complete `variant`, `closure_joints`, `drive`, and `compiled` payloads. These include body properties, articulation and solver/USD attributes, joint stiffness/damping/armature, position/velocity/effort limits, material properties, and actuator configuration.
4. Downstream causal validation checks section/mode consistency and re-hashes the payload, but never validates where an infinity tag is legal.
5. The new test proves deterministic encoding for a container named `limits`; it does not prove that the path was declared. No negative test injects infinity into mass, inertia, gains, material, or solver fields.

Impact:

- The immediate `stable_hash(transform_payload)` crash is avoided, but the frozen rule changes from "reject undeclared NaN/Inf" to "treat every Inf as declared and unbounded."
- A real runtime/configuration anomaly can become hash-valid causal evidence instead of invalidating the stage.
- A function name and tag text are not a declaration. The declaration must be bound to a frozen field/path and source semantics.

Minimum repair:

1. Freeze exact legal infinity paths, producer/origin, and sign semantics. Derive them using a disposable Isaac inspection, then commit the declaration before accepting formal evidence.
2. Make the encoder path-aware or use field-specific constructors. Inf outside the frozen paths and every NaN must raise.
3. Remove the whole-object calls at `isaac_probe_env.py:477-483`, or validate every visited path against the frozen declaration set.
4. Add positive tests for each legal unbounded field and negative tests for mass/inertia, stiffness/damping, material, and PhysX solver fields.

#### P1-2: G01 reset identity converts any non-finite physical state into `debug_observer_disabled`

Files and lines:

- `isaac_worker.py:625-670,2260-2267,2393-2407`
- `isaac_debug_env.py:131-162,231-256`
- `repeatability.py:139-173`
- `trace_contract.py:62-81`
- design `2026-10-07-sim2sim-root-cause-suite-design.md:400-426,1264-1279`

Reproducible logic:

1. `_phase_hash_payload()` and `_repeatability_profile_state()` test only whether a floating array contains any non-finite value.
2. Every such array is replaced by `{"availability":"unavailable","reason":"debug_observer_disabled","shape":[...]}`.
3. They do not check the field name, whether contact observers were disabled, whether the array equals the expected sentinel, or whether the value came from an authoritative physical state.
4. The reset snapshot mixes intentional contact sentinels with joint position/velocity, root pose/velocity, projected gravity, virtual-leg state, and closure error.
5. NaN or Inf in `active_joint_position_canonical`, `all_hinge_velocity_named`, `base_orientation_control_wxyz`, or another physical field is therefore mislabeled as an unavailable observer and becomes a valid phase hash and repeatability key.

Impact:

- Different corrupt states with the same shape collapse to the same identity marker.
- This can hide the reset corruption that G01 is intended to detect and violates the requirement that phase hashes represent actual written/read-back physical state.
- The trace writer correctly rejects non-finite available arrays at `trace_contract.py:118-125`, but that does not protect a pre-forward field reduced before repeatability hashing. `trace_contract.py:79-80` separately states that NaN/Inf cannot be inferred as unavailable.
- This code predates the hotfix, but it is in the requested G01/P10 path and blocks the requested no-masking acceptance criterion.

Minimum repair:

1. Define an exact sentinel policy keyed by field and probe configuration. Only frozen contact-observer fields may be unavailable when contact sensors are disabled.
2. Require exact type, shape, and sentinel contents for those fields, including all-NaN versus exact integer `-1`; reject mixed finite/non-finite arrays.
3. Require all authoritative pose, velocity, joint, orientation, gravity, kinematic, and closure fields to be finite before either phase hash or repeatability record is built.
4. Add NaN/+Inf/-Inf mutation tests for joint and root fields, plus positive tests for every permitted disabled-observer sentinel.

### P2

#### P2-1: the infinity tag is deterministic but has no consumer-side schema validator

Files and lines:

- `contracts.py:23,46-61`
- `causal_contract.py:269-297,310-361`
- `tests/test_contracts.py:37-60`

The producer emits a stable version, fixed keys, explicit sign, and canonical JSON, so identical output is deterministic and hash-verifiable. However, no recursive validator enforces the exact key set, allowed `kind`, exact `semantics="unbounded"`, legal placement, or absence of malformed tag-like objects. `validate_result_identity()` verifies byte hashes, not tag semantics.

This is P2 because intact outer artifact hashes and a trusted producer still protect current evidence. It remains a schema gap: future producer drift or an internally recomputed malformed tag can be self-consistent.

Minimum repair: add one recursive validator that rejects raw NaN/Inf, malformed/extra-key tags, and tags outside the P1-1 legal-path declaration. Invoke it before configuration/transform hashing and in `validate_result_identity()`. Add wrong-version, wrong-kind, wrong-semantics, extra-key, and illegal-path tests.

## Confirmed Repairs

### Immediate G01 crash path

The failed formal attempt matches the stated trigger. `p10_a.stderr.txt` reached `isaac_worker.py:2297`, then `stable_hash(transform_payload)` failed because canonical JSON rejected a non-finite value. Trace artifacts existed, but no `result.json` was produced and the stage stayed incomplete.

The hotfix encodes Python-float infinity before the transform payload is hashed. For values traversing `_usd_value()` or the final `ProbeEnvironmentIdentity` encoder, the original exception is removed. Raw canonical JSON remains strict at `contracts.py:177-188`, and NaN passed to the new encoder still raises at `contracts.py:52-54`.

### Robot transform semantics return

`_isaac_robot_transform_semantics()` now has a reachable return at `isaac_worker.py:215-245`. It returns the actuation mode/controller/neutralization fields, robot-hinge stiffness/damping, closure mode, and constraint enabled state. `_isaac_sphere_transform_semantics()` has a separate return at `isaac_worker.py:376-388`.

`tests/test_isaac_worker.py:82-106` calls target and direct robot variants, so a regression to implicit `None` will fail. This closes the reported misplaced-return defect.

### Causal contract behavior

The restored transform payload fits the existing exact-diff flow: `result_identity_fields()` hashes it, `validate_result_identity()` recomputes the hash, and ratio validation compares transform and compiled semantic paths. Identical infinity tags are stable invariants and create no diff.

The unresolved issue is declaration, not hash determinism. Until P1-1 is closed, downstream hash equality proves byte equality but not that infinity was permitted at that causal field.

## G01/P10 Blocker Sweep

- The original transform serialization blocker is addressed for Python floats reached by the new encoding boundaries.
- No second definite static infinity failure was established in frozen `SimulationCfg`/`PhysxCfg` data. Those payloads remain strict; a real smoke is still needed after the P1 fixes to prove no additional legal unbounded field reaches a raw `stable_hash()` path.
- Available robot trace arrays remain fail-closed for NaN/Inf through `trace_contract.py:93-129`.
- The failed trace is useful diagnostic evidence but cannot become completed G01/P10 evidence because the causal result and stage classification were not written.
- Source hashes changed after `root-cause-core-v1-20261009-045026`. The frozen source-identity contract should reject authoritative resume; after approval, use a new fresh run identity.

## Test Assessment

Accepted supplied record: frozen-environment targeted tests `17 passed`. This review did not rerun pytest.

The added tests cover basic tag determinism, Python-float NaN rejection, raw Inf rejection by canonical JSON, and reachable robot transform output for target/direct variants.

Missing coverage:

- legal path versus unexpected infinity in a physical/configuration field;
- the actual `ProbeEnvironmentIdentity` infinity that triggered the run;
- malformed tag rejection by a downstream consumer;
- NumPy scalar/array non-finite inputs at the encoder boundary;
- reset-state NaN/Inf mutation and exact observer-sentinel handling;
- end-to-end causal identity/ratio validation containing a legal infinity tag.

## Reviewed SHA256

- `2026-10-07-sim2sim-root-cause-suite-design.md`: `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD`
- `contracts.py`: `F723EC53C2942A62A4FCF9BB363167FA044F6E00939E8B10E8DD6BE365FE619E`
- `isaac_probe_env.py`: `F0F8F40B407C0C5F11C293433AFE4751BB1869CB7D30F30C7D3561C0F0D45A2D`
- `isaac_worker.py`: `CA1717F4E227E0A4531F49A388D3E40D06EB185D06EAD5F3F1D6402A6374EEA3`
- `tests/test_contracts.py`: `80A4CB1E2EEFDC9656C5C329D6F26B54E06E9106AE9BEC3B884CAB87B2AAD651`
- `tests/test_isaac_worker.py`: `D3B62D8611D5E88A2427CD9F0B7C677CEB307D522FC06AFA48F012F1549F0554`
- `../isaac_debug_env.py`: `3CDB03D723BEB8D2024E7A4F512ADF8C51560C3F1C8209AA9B9D01E7725F56C2`
- `causal_contract.py`: `E220624AADA15D41FDBD0139AEDE712665879916A4C12200681B218631D8AED1`
- `repeatability.py`: `07308D8B1C47C54088B7CF28EDC801DDE8C31A3DC316DE1813BB9166ECB8B19F`
- `trace_contract.py`: `C9BC01DB48DDBE40359E560A2455917797207508CBE5B19FDB6B85EAB71E8710`
- failed `p10_a.stderr.txt`: `EAF2A0F97CAF79B45A46E43C710F806C3D0000BC6BE88031C46E0F9A4CF0E510`
- failed `p10_a/trace/trace.json`: `A31E1F78704A21DB170B2302CFAAD10CB028FCF2B274BC179CB9FE6C77F1B002`
- failed `p10_a/trace/trace.npz`: `0B117D33EBD64655DDD5CE9544CC71E975ADB2376E9AB020CC7232DC9F1A34ED`

## Decision

- P0: 0
- P1: 2
- P2: 1

**Final decision: BLOCKED.**

The robot-return defect and immediate transform serialization crash are fixed, but the revision does not preserve the frozen distinction between declared and undeclared infinity, and G01 reset hashing can misclassify real numerical corruption as a disabled observer. Do not start an authoritative Isaac `p10_a` smoke or full fresh Core V1 run with this revision.

A disposable, explicitly non-evidentiary inspection may be used only to enumerate exact legal infinity paths for the P1-1 allow-list. Its output must not be selected as Core evidence.
