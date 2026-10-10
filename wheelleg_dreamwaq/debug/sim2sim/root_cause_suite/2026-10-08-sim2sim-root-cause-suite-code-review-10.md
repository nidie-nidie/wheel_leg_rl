# Sim2Sim RootCauseSuite Core V1 Code Review 10

Date: 2026-10-08  
Review mode: independent, defect-first, read-only incremental review  
Result: **NOT APPROVED**

## Conclusion

- P0: 0
- P1: 1
- P2: 0
- The `orchestrator.py` repair closes review-09 P1-N1 when `run_definitions(..., resume=True)` is called directly: a missing lexical run root or manifest is rejected before run-local cache, claim, identity, manifest, or stage writes, and a present manifest receives full layout and identity validation.
- The frozen official entry does not preserve that property end to end. `run_suite.ps1` still unconditionally creates `<run>/runtime_cache/{pycache,pytest-cache,pytest-tmp}` before Python and before the resume manifest/reparse checks. `run_suite.ps1 run --resume <missing-id>` therefore creates the supposedly missing run, and manifest-less debris is modified before rejection.
- The same launcher write means review-08 link/reparse protection is complete inside the Python orchestrator but not at the official-entry boundary: `New-Item -Force` can reach an existing run-root or cache junction before Python rejects it.
- The supplied test records were accepted without rerunning pytest. The new tests bypass the frozen launcher.
- The threshold `P0=0 && P1=0` is not met.

**Final decision: NOT APPROVED. Real Core V1 execution is not authorized.**

## Frozen Baseline

Recomputed hashes match the request:

- Architecture v0.21: `5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE`.
- RootCauseSuiteCoreV1.17 design: `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD`.
- Factor-path allowlist: `4CF2A34D4E3725A78DC0028AD92FF6DCD04E820874CFFCD8757F267F8DADDD85`.
- `orchestrator.py`: `73E088E682EB8E01AA93B8E83C08ECD957007DEBD30A3C10D210F60C491F46B8`.
- `tests/test_orchestrator.py`: `D62E5D6201B5AC47E9AD9A7E4AF0DD7E2FED83DBB7F050171520669F8A0B9EB2`.
- `run_suite.ps1`: `DDDDB778BE4C3897FE00232BB60004772B76171F53A81B9C3EDE824CC3F569FA`.
- The supplied 17-entry `formal_snapshot`, `all_expected=True`, was accepted.

No tests were executed for this report.

## Review-09 P1-N1 Status

### Orchestrator-level repair: CLOSED

Files and lines:

- `orchestrator.py:1168-1182`
- `orchestrator.py:1208-1217`
- `orchestrator.py:533-573`
- `orchestrator.py:985-994`
- `tests/test_orchestrator.py:566-594`

The direct orchestrator ordering is correct:

1. `_run_root()` preserves the lexical entry at `orchestrator.py:780-785`.
2. Missing run and missing manifest are rejected at lines 1171-1181 before cache normalization, claim, identity, manifest, or stage writes.
3. Lines 1175-1182 validate the run root, runtime cache, cache children, and manifest as plain, non-reparse, canonically exact entries.
4. `_load_manifest()` at lines 1210-1217 and 985-994 performs full manifest/run identity validation before launch history or stage state is written.
5. `_read_manifest_payload()` at lines 533-573 validates manifest hash, run id, claim, design, definitions, allowlist, and source/run identity.

Direct calls therefore fail closed for missing run, missing manifest, claim-only manifest, and invalid full-manifest identity. Malformed JSON also stops before resume state is written, although it propagates `JSONDecodeError` rather than normalizing it to `EvidenceIntegrityError`.

The tests at `tests/test_orchestrator.py:566-594` correctly prove the direct-call behavior: no cache/manifest is added to manifest-less debris, and a missing run directory is not created.

### Official-entry repair: OPEN

Review-09 P1-N1 remains open at the executable-system boundary. See P1-1.

## P0

None.

## P1

### P1-1: The frozen launcher writes the resume target before run/manifest/reparse validation

Files and lines:

- `run_suite.ps1:23-35`
- `run_suite.ps1:40-58`
- `__main__.py:23-40`
- `__main__.py:43-60`
- `orchestrator.py:1168-1217`
- `tests/test_orchestrator.py:566-594`
- `tests/test_run_suite_launcher.py:18-82`
- `2026-10-07-sim2sim-root-cause-suite-design.md:1283-1290`
- `2026-10-07-sim2sim-root-cause-suite-design.md:1480-1485`

Reproducible logic:

1. `run_suite.ps1` identifies `run --resume <id>` at lines 23-29 and computes the lexical run path at lines 40-45.
2. Regardless of whether the command is fresh run, resume, verify, or report, lines 46-49 execute `New-Item -ItemType Directory -Force` for all three cache children.
3. For a nonexistent resume id, those calls materialize the run root and cache skeleton before Python starts at line 61. The new `orchestrator.py:1171-1174` missing-run guard cannot observe the original missing state through the official entry; it sees an existing root and rejects the now-modified directory only for missing manifest at lines 1178-1181.
4. For existing manifest-less debris, the launcher adds `runtime_cache` before rejection. This contradicts the direct test's no-cache/no-debris-modification assertion at `tests/test_orchestrator.py:580-582`.
5. After a caught `EvidenceIntegrityError`, `cli.py` returns an error code and `__main__.py:43-60` writes `parent_guard.json` under the invalid run root. The official path can therefore add another run artifact without accepting a valid prior run identity.
6. If the lexical run root, `runtime_cache`, or a missing cache child is a Windows junction/reparse point, PowerShell writes occur before `orchestrator.py:1175-1182`. They can traverse or populate the redirected target before Python rejects it.

Impact:

- A mistyped/nonexistent resume id leaves a new run directory and cache skeleton even though resume requires a pre-existing identity-bound run.
- A manifest-less or otherwise invalid run can be changed before rejection, so official-entry behavior does not match the new no-write resume contract or its tests.
- The pre-Python write weakens the end-to-end review-08 reparse guarantee and can redirect project-controlled directory creation outside the intended lexical run path.
- No new claim, run identity, selected attempt, or stage evidence is accepted, so this remains P1 rather than P0; the provenance/write boundary is nevertheless blocking.

Minimum repair:

1. Make launcher cache-skeleton creation mode-specific. Only a fresh `run` may create it. `run --resume`, `verify`, and `report` must not call `New-Item` before proving that the lexical run and full manifest already exist and are acceptable.
2. Preserve the orchestrator's current full layout/manifest/identity validation as the authoritative resume gate. Setting `PYTHONPYCACHEPREFIX` does not require eager directory creation because the parent uses `-B` and `PYTHONDONTWRITEBYTECODE=1`.
3. Ensure an invalid resume does not emit `parent_guard.json` into an unaccepted run. Complete the standard-library identity preflight before that evidence path becomes writable, or emit failure evidence only after a valid run identity is accepted.
4. Add frozen-entry tests for missing run, missing manifest/debris, claim-only manifest, corrupt manifest, and run-root/runtime-cache/cache-child junctions. Each negative case must prove no new local or redirected file/directory was created. Current launcher tests cover only argument normalization.

## P2

None.

## Fresh Ownership and Reparse Regression Check

Inside `orchestrator.py`, the review-08 repairs remain structurally intact:

- Only the fresh branch at `orchestrator.py:1183-1196` prepares the cache skeleton.
- Only the fresh branch at `orchestrator.py:1218-1247` creates an initialization claim and new run identity.
- `_create_initialization_claim()` still uses `O_CREAT | O_EXCL` through `orchestrator.py:423-490`.
- Claimed-layout revalidation and persisted-claim comparison still precede `_new_run_identity()` at `orchestrator.py:1219-1242`.
- Lexical-path and link/reparse validation remains at `orchestrator.py:93-282`.
- Two-contender, validation/claim mutation, symbolic-link, and real Windows junction tests remain at `tests/test_orchestrator.py:90-307`.

No new regression was found in Python fresh ownership. The blocker is that the official launcher writes before these protections run, so review-08 reparse closure cannot be certified end to end.

## Valid Resume Regression Check

No regression was found after a valid manifest passes the gate:

- Hash-verified complete attempts are reused; a tampered selected attempt is rerun into a new attempt directory at `orchestrator.py:1310-1401`, covered by `tests/test_orchestrator.py:495-508`.
- G01 reselection replaces its seal and invalidates only transitive downstream selections through `orchestrator.py:303-322`, covered by `tests/test_orchestrator.py:511-563`.
- Terminal G02 reselection invalidates transitive downstream selections through `orchestrator.py:325-342`, covered by `tests/test_orchestrator.py:440-492`.
- Stage-set, definition-hash, and scope drift remain rejected at `orchestrator.py:1210-1217`.

These properties execute only after the frozen launcher has touched the target path and therefore do not cure P1-1.

## Test Record Assessment

Accepted supplied records:

- Targeted resume/ownership: `7 passed`.
- Complete project suite with the MuJoCo-only test excluded: `159 passed, 4 skipped`.
- MuJoCo interpreter: `12 passed`.
- The four skips are accepted as the stated local symbolic-link privilege limitation; the three real Windows junction positions reportedly executed and passed.

The records support the direct-orchestrator repair and unchanged valid-resume behavior. They do not cover P1-1: `tests/test_run_suite_launcher.py:18-82` extracts and executes only the argument-normalization block, while the new resume tests instantiate `RootCauseOrchestrator` directly and bypass `run_suite.ps1` and `__main__.py`.

## Findings Summary

### P0

None.

### P1

- P1-1: the frozen launcher creates or modifies the resume target before run, manifest, and reparse validation.

### P2

None.

## Final Decision

**NOT APPROVED**

`P0=0`, `P1=1`, `P2=0`. Review-09 P1-N1 is closed within `orchestrator.py`, but not through the frozen official launcher. Because the official resume path still writes cache/run state before validating the prior run and can reach reparse targets before Python rejects them, the approval threshold is not met and real Core V1 must remain blocked.
