# Sim2Sim RootCauseSuite Core V1 Code Review 12

Date: 2026-10-08  
Review mode: independent, defect-first, read-only incremental review  
Result: **APPROVED**

## Conclusion

- P0: 0
- P1: 0
- P2: 0
- Review-11 P1-N1 is closed. Manifest loading no longer changes acceptance state; resume and Core verify/report accept identity only after command-specific stage-set, definition, and scope checks pass. `IdentityDriftError` also unconditionally returns `run_identity_accepted=False`.
- Review-11 P2-1 is closed. Launcher argument/configuration failures map to code `2`; preflight layout/evidence failures map to code `5`. Formal launcher tests assert exact codes under Windows PowerShell 5.1 and PowerShell 7.
- Catalog data now crosses the canonical JSON boundary before entering the run identity. Tuple/list write-before/write-after inequality is removed without changing the frozen catalog hash.
- Previously closed ownership, TOCTOU/reparse, manifest-less resume, valid resume/reselection, write-guard, and formal-file boundaries remain closed.

**Final decision: APPROVED. Real Core V1 execution through the frozen launcher is authorized.**

## Frozen Baseline

All requested hashes were recomputed and match:

- Architecture v0.21: `5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE`.
- RootCauseSuiteCoreV1.17 design: `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD`.
- Factor-path allowlist: `4CF2A34D4E3725A78DC0028AD92FF6DCD04E820874CFFCD8757F267F8DADDD85`.
- `run_suite.ps1`: `BA75535293D68C6F0F4042FB4D86FA0B9B36ADA86355C678AA04A102E513C7FF`.
- `__main__.py`: `9733343F7853A8C01BF27B55909DFA7D88ADE5F235CCA732D96A69B9757A70BC`.
- `cli.py`: `009123EA3D001E6B6A744BBEE5F4435569D40F517F4A93A6E18281607B2CBA08`.
- `orchestrator.py`: `9078E0940DAD31360C910E8B467E6EAB21C0B59C8B467952DE344CD95137ECA7`.
- `integrity.py`: `DD9EF92ADCFF95251637D734B0451233D66FB7EB793B31CB09896E3D8D4507EB`.
- `tests/test_run_suite_launcher.py`: `E63B609727AAF00E38A766E742C4EEDC99A2341A47BADD004B7DB4B1EDF55247`.
- `tests/test_orchestrator.py`: `CF3370FA7F446E03B71F9F93B74F112B4BC9E11953C74433E5B212E9E8E81776`.
- `tests/test_scenario_catalog.py`: `CA779A6155290454A5A1B4BF81A55E7EBDD059BE439877496675AE958A38A136`.

The supplied 17-entry `formal_snapshot`, with `all_expected=True`, was accepted. A read-only direct evaluation of `catalog_snapshot()` returned `8374479CD53585DFFB5F2840C20F6E387A211007170E4BE76A69C32F1A97EE67`. The `launcher-*` run count is `0`. No pytest command was run for this review.

## Review-11 P1-N1

### CLOSED: complete command identity precedes acceptance

Files and lines:

- `orchestrator.py:989-998`
- `orchestrator.py:1172-1268`
- `orchestrator.py:1523-1548`
- `cli.py:36-75`
- `__main__.py:39-64`
- `tests/test_run_suite_launcher.py:105-155`
- `tests/test_run_suite_launcher.py:388-459`

Evidence:

1. `_load_manifest()` at `orchestrator.py:989-998` only reads and verifies the manifest and its full or light run identity. It does not set `run_identity_accepted`.
2. `run_definitions()` resets acceptance to false at line 1172. Resume validates the existing lexical root, manifest, and layout at lines 1175-1187, then checks exact stage ids, current definition hash, and requested scope at lines 1216-1222. It sets acceptance true only at line 1223, before any launch-history or stage write.
3. Fresh mode obtains and rereads its exclusive initialization claim, constructs and writes the formal manifest at line 1265, reloads it through `_load_manifest()` at line 1267, and sets acceptance true only at line 1268. Claim-only, malformed, non-round-trippable, or identity-invalid state cannot be accepted.
4. `verify_stage_evidence(require_core=True)` requires `scope == "core_v1"`, exact Core stage ids, exact definition payload, and matching definition hash at lines 1536-1547. Acceptance occurs only at line 1548.
5. `cli.py:61-63` returns code `3` with `run_identity_accepted=False` for every `IdentityDriftError`, independent of orchestrator state. Evidence-integrity and stage-execution handlers preserve state at lines 64-69, allowing valid post-acceptance failure evidence.
6. `__main__.py:43-63` writes `parent_guard.json` only when the returned acceptance flag is true.

The drift test at `tests/test_run_suite_launcher.py:388-416` creates internally self-consistent manifests whose generic identity passes but whose stage set, definitions, or scope disagree with the current Core command. It invokes the formal script under each available PowerShell engine, requires code `3`, requires a byte-for-byte unchanged run tree, and requires no `parent_guard.json`.

The positive test at `tests/test_run_suite_launcher.py:419-459` enters a new process, passes the real `run_definitions(..., resume=True)` identity gate, then raises `StageExecutionError`. It requires code `4` and valid parent/write-guard evidence. The repair therefore does not suppress legitimate post-acceptance failure evidence.

## Review-11 P2-1

### CLOSED: launcher exit codes follow the frozen taxonomy

Files and lines:

- `run_suite.ps1:12-25`
- `run_suite.ps1:31-67`
- `run_suite.ps1:125-153`
- `run_suite.ps1:162-166`
- `tests/test_run_suite_launcher.py:233-251`
- `tests/test_run_suite_launcher.py:254-385`

Argument and configuration checks call `Exit-WithCode(..., 2)`: invalid or absent command, missing frozen interpreter, missing resume/verify/report run id, invalid run id, and a run path escaping the runs root all map explicitly to code `2`.

The complete fresh/non-fresh filesystem preflight is enclosed by `try/catch` at `run_suite.ps1:128-153`. Missing roots/manifests/caches and link/reparse/layout violations therefore map explicitly to evidence-integrity code `5`, rather than escaping with a host-specific generic code. Once Python is launched, its frozen code is propagated unchanged at lines 162-166.

The formal-script tests assert exact code `2` for configuration cases under both PowerShell engines at `tests/test_run_suite_launcher.py:233-251`. Missing runs, manifest-less debris, claim-only/corrupt manifests, and all three real junction positions assert exact code `5` and no-write invariants at lines 254-385. No unsupported syntax was found; the launcher remains compatible with Windows PowerShell 5.1.

## Catalog Manifest Round-Trip

Files and lines:

- `integrity.py:164-173`
- `integrity.py:216-258`
- `tests/test_scenario_catalog.py:34-37`

`catalog_snapshot()` serializes the dataclass-derived catalog through `canonical_json_bytes()` and immediately parses that JSON before storing `scenarios`. Tuple-valued Python fields therefore become the same list-valued structure returned after a persisted manifest is loaded. The identity hash is computed over that normalized payload.

This is the correct boundary because the run manifest is a JSON artifact. The change does not omit fields, reorder semantically ordered arrays, or coerce values beyond the serialization already performed by the manifest. Canonical JSON encodes tuples and lists identically, so the frozen identity remains `8374479CD53585DFFB5F2840C20F6E387A211007170E4BE76A69C32F1A97EE67`; direct read-only evaluation confirmed it.

The full identity stores the normalized snapshot at `integrity.py:223-239`, and verification compares the complete current snapshot at lines 254-258. `tests/test_scenario_catalog.py:34-37` enforces equality across another canonical manifest round trip. This closes the fresh-Core failure where an identity could be valid before writing but unequal immediately after loading only because of tuple/list representation.

## Core Scope And Terminal Paths

Files and lines:

- `orchestrator.py:1430-1439`
- `orchestrator.py:1523-1548`
- `tests/test_orchestrator.py:334-383`
- `tests/test_orchestrator.py:387-427`

The formal `run()` path still invokes `run_definitions()` with `full_identity=True` and `scope="core_v1"`. Core verify/report cannot accept a light/test run: scope, stage order, full definition payload, and definition hash must match the current Core contract before acceptance.

The terminal G02 and valid-but-unusable G01 tests now create genuine `core_v1` runs with full identities and replace `core_definitions()` only with the exact test definition set before calling `require_core=True`. They exercise the Core scope/identity gate instead of relying on a relaxed verifier. No production gate needs to be weakened.

## Ownership And Resume Regression Review

No regression was found in previously closed controls:

- Lexical paths are preserved, and every run root, runtime cache, cache child, and manifest is checked as a plain non-link/non-reparse entry with exact canonical containment at `orchestrator.py:93-282`.
- Fresh ownership remains an atomic `O_CREAT | O_EXCL` claim at `orchestrator.py:423-490`. Fresh mode validates the exact empty cache skeleton before claim, validates claimed layout after claim, and rereads the persisted claim before building the run identity at `orchestrator.py:1188-1248`.
- Resume requires the lexical run root and formal manifest before cache, claim, identity, launch-history, or stage writes at `orchestrator.py:1175-1187`. Missing run, manifest-less debris, claim-only state, corrupt manifest, and identity drift remain fail-closed.
- Completed-attempt reuse requires full attempt verification at `orchestrator.py:1321-1367`; a tampered selected attempt is discarded and rerun. G01 resealing and terminal G02 replacement preserve transitive downstream invalidation at lines 1387-1405.
- The formal launcher creates cache directories only for a fresh `run`; resume, verify, and report validate all existing bootstrap entries before Python at `run_suite.ps1:125-153`.
- The manifest identity binds scope, stage ids, definitions, definition hash, run identity, initialization claim, launch history, selections, and replay seal at `orchestrator.py:509-530`. Generic validation checks its self-hash, run id, claim, design, allowlist, definition hash, and order at lines 533-576.

The revision does not alter formal training code, USD, production MuJoCo parameters, checkpoints, or the frozen formal-file set.

## Test Record Assessment

Accepted supplied records:

- Launcher and catalog: `51 passed`.
- Terminal Core targeted: `3 passed`.
- Project interpreter, excluding the MuJoCo-only file: `196 passed, 4 skipped`.
- MuJoCo interpreter: `12 passed`.
- All four skips are the stated local symbolic-link privilege limitation. Corresponding real Windows junction tests at run root, runtime cache, and cache child executed and passed.
- Post-test `launcher-*` temporary run count: `0`.

Coverage directly addresses both review-11 findings and the catalog blocker: two formal PowerShell engines, exact exit codes, internally self-consistent command drift, byte-for-byte no-write checks, positive post-acceptance failure evidence, full-identity terminal Core verification, and catalog JSON round-trip equality.

Residual risks and test blind spots:

- A complete real Core V1 run has not yet executed. Approval means the launcher and identity/evidence gates are fit to start it; it does not predict whether every Isaac/MuJoCo worker or physical probe will complete.
- Symbolic-link creation remains unavailable under the current account. Equivalent Windows junction paths pass, and the shared reparse-bit rejection logic was reviewed; this is not blocking.
- The accepted-failure regression uses the real resume identity gate in an isolated subprocess with test scope instead of forcing a production Core worker failure. It directly covers the acceptance/result/parent-guard state machine; full-run fault injection would be additional coverage, not an approval requirement.
- A hostile external process can still attempt path replacement between user-space checks. Launcher post-create checks, Python revalidation, exclusive claim, post-claim validation, and claim reread preserve the designed fail-closed behavior for suite contenders. No new write path was introduced.

## Findings Summary

### P0

None.

### P1

None.

### P2

None.

## Final Decision

**APPROVED**

`P0=0`, `P1=0`, `P2=0`. Review-11 P1-N1 and P2-1 are closed, catalog identity survives manifest round-trip without changing the frozen hash, and no regression was found in ownership, resume, reparse, selection, write-guard, or formal read-only controls. The suite may proceed to one real Core V1 execution through `run_suite.ps1 run`.
