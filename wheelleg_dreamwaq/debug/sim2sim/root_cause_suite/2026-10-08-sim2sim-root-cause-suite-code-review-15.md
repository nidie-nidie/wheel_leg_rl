# Sim2Sim RootCauseSuite Core V1 Code Review 15

Date: 2026-10-08  
Review mode: independent, defect-first, read-only final incremental review  
Previous review: `2026-10-08-sim2sim-root-cause-suite-code-review-14.md`  
Result: **APPROVED**

## Conclusion

- P0: 0
- P1: 0
- P2: 0
- Review-14 P2-1: **CLOSED**
- Final decision: **APPROVED**

The combined malformed-tag bypass reported in review-14 is closed for both scalar and shaped declarations. A mapping at a legally declared infinity path is now validated as an exact tag before any heuristic tag-like classification. The former `missing kind + WrongVersion` payload therefore cannot be treated as ordinary data, even after all causal identity hashes are recomputed.

The illegal-path classifier also fails closed when at least three reserved sentinel keys are present. The existing exact tag, raw NaN/Inf, illegal-path, sign, shape, source, semantics, and shaped-completeness protections remain in force. No P1 regression was found in producer encoding, consumer validation, reset handling, or robot transform semantics.

## Findings

### P0

None.

### P1

None.

### P2

None.

## Review-14 P2-1 Closure

### Declared-path mappings are exact by construction

At `contracts.py:254-280`, `validate_declared_infinity_payload()` now calls `declaration_for_path(path)` before evaluating the generic tag-like heuristic.

When a declaration exists:

1. Any mapping must have exactly `schema_version`, `kind`, `semantics`, `source`, and `attribute` (`contracts.py:257-262`).
2. The complete mapping must equal `declared_infinity_tag(spec)` (`contracts.py:263-266`).
3. Shaped tags still record their exact integer suffix and participate in the complete Cartesian-index check (`contracts.py:267-280,310-317`).
4. The function returns immediately after validating the exact tag, so malformed mappings cannot be recursively reinterpreted as ordinary data.

This directly closes the review-14 reproducer. Removing `kind` causes the exact-key check to fail regardless of the replacement `schema_version` value.

### Illegal-path tag-like mappings fail closed

For paths without a declaration, the classifier at `contracts.py:281-297` retains the prior `kind+semantics`, `kind+attribute`, and reserved schema-prefix checks and adds:

```text
len(keys & DECLARED_INFINITY_TAG_KEYS) >= 3
```

The review-14 malformed object retains four reserved keys (`schema_version`, `semantics`, `source`, `attribute`) and is therefore rejected even at an undeclared path. A mapping with only two ordinary reserved names is not automatically classified as a sentinel.

### Consumer regressions exercise the real acceptance boundary

`tests/test_causal_contract.py:279-308` adds `_rebind_minimal_identity()`, which recomputes:

- `configuration_semantics_hash`
- `configuration_hash`
- `repeatability_key`

The scalar regression at `tests/test_causal_contract.py:366-390`:

1. Builds a valid `physxScene:maxBiasCoefficient` identity.
2. Removes `kind` and changes `schema_version` to `WrongVersion`.
3. Recomputes all dependent hashes.
4. Calls `validate_result_identity()` and requires `EvidenceIntegrityError`.

The shaped regression at `tests/test_causal_contract.py:393-420` applies the same mutation to all three `physics:centerOfMass` leaves, recomputes the identity hashes, and requires consumer rejection. This covers the exact all-leaves evasion left open by review-14 rather than relying on the producer to reject the malformed value before a result exists.

## Read-Only Behavior Verification

The following narrow probes were run with the project interpreter using `-B`; no source or evidence file was written:

```text
valid_declared          ACCEPT
combined_declared       REJECT malformed infinity tag
combined_illegal        REJECT malformed infinity tag
ordinary_two_reserved   ACCEPT
shaped_all_combined     REJECT malformed infinity tag at leaf 0
```

A read-only scan of 200 existing suite JSON files found no ordinary non-tag mapping with three or more reserved sentinel keys. This is supporting evidence only, not a replacement for schema tests.

## P1 Regression Check

No P1 regression was found.

- The 12 exact `(prim path, attribute)` declarations, sign, shape, source, and field-specific semantics are unchanged (`contracts.py:25-133`).
- Producer encoding remains path-aware and rejects NaN, undeclared Inf, wrong sign, wrong shape, and mixed shaped values (`contracts.py:173-228`).
- `ProbeEnvironmentIdentity` producer validation is unchanged; `isaac_probe_env.py` retains SHA256 `F4A192...F00E`.
- `result_identity_fields()` and `validate_result_identity()` consumer call sites are unchanged; `causal_contract.py` retains SHA256 `5ACACE...0176`.
- Both production declaration resolvers remain exact and safe when queried at root or parent mappings: `isaac_probe_env.py:508-540` and `causal_contract.py:219-259` guard path equality, length, type, and exact attribute positions before returning a declaration.
- Reset/contact finite-state enforcement and pre-removal world-position validation are unchanged; `isaac_worker.py` retains SHA256 `6C720D...05C8`.
- Robot and sphere transform return paths are unchanged.
- Frozen architecture, design, and factor-path allowlist hashes are unchanged.

The new exact-mapping branch does not weaken raw non-finite rejection or shaped completeness. It also does not change valid producer output: an exact five-key declared tag remains accepted.

## Test Assessment

Accepted supplied verification record:

- Targeted current-source tests: `76 passed`.

This review did not rerun pytest. The two additional tests are correctly placed at the downstream consumer boundary and reproduce the required self-consistent hash rebinding. The previous broader regression record applies to the unchanged surrounding files; the only current implementation change is the reviewed validator branch in `contracts.py`.

## Reviewed SHA256

### Current changed files

- `contracts.py`: `FFE462AAD2DE8CEEE42652B067A5609BFAAF9CC8A81370B41316ECE108B3D0B4`
- `tests/test_contracts.py`: `34CC3564B935E5898C114F950C81E3BBE492D262304105782AD3E937CCB5AFEF`
- `tests/test_causal_contract.py`: `2EBD74A2C7BDBBDC665DD37A714384FEF4A77ADDE8971BAEA5321A953021F084`

### Unchanged reviewed boundaries

- `2026-10-08-sim2sim-root-cause-suite-code-review-14.md`: `0078947A1AC30CCFD733499AD95737AEB1D707078B0072FC53B28ECB01FAB842`
- `isaac_probe_env.py`: `F4A192BEB6A2871F168D482A55DAE47011D7F3BDD92234AF80234FDC01C1F00E`
- `causal_contract.py`: `5ACACEB60C6C653D87A272D6C7C8BE1A26DB26EE9EEB55BB0E7F379CCB690176`
- `isaac_worker.py`: `6C720DC307C85CAD9A344EE0E29465B79B60422CA06C64DAF07B3EEC9BEC05C8`
- Architecture v0.21: `5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE`
- Design `RootCauseSuiteCoreV1.17`: `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD`
- `factor_path_allowlist.json`: `4CF2A34D4E3725A78DC0028AD92FF6DCD04E820874CFFCD8757F267F8DADDD85`

## Decision

**Final decision: APPROVED.**

Review-14's only P2 finding is closed. No P0, P1, or P2 defect remains in the requested incremental scope, and no code-review blocker remains for starting a fresh authoritative Core V1 run under the frozen entry and identity rules.
