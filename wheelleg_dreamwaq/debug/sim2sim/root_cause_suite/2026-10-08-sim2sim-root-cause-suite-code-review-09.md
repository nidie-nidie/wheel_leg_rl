# Sim2Sim RootCauseSuite Core V1 Code Review 09

Date: 2026-10-08  
Review mode: independent, defect-first, read-only incremental review  
Result: **NOT APPROVED**

## Conclusion

- P0: 0
- P1: 1
- P2: 0
- Review-08 P1-1, atomic fresh ownership and the validation/claim TOCTOU, is closed.
- Review-08 P1-2, lexical path preservation and Windows link/reparse rejection, is closed.
- One blocking resume-provenance defect remains: `resume=True` creates a new initialization claim and run identity when no valid prior manifest exists, so a missing or polluted pre-manifest directory can be adopted as a resumed run.
- The supplied test records were accepted without rerunning pytest.
- The approval threshold `P0=0 && P1=0` is not met.

**Final decision: NOT APPROVED. Real Core V1 execution is not yet authorized.**

## Frozen Baseline

- Architecture v0.21 SHA256: `5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE`.
- RootCauseSuiteCoreV1.17 design SHA256: `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD`.
- Factor-path allowlist SHA256: `4CF2A34D4E3725A78DC0028AD92FF6DCD04E820874CFFCD8757F267F8DADDD85`.
- `orchestrator.py` SHA256: `585B12A1B60D5BF5B0846B7FBFFB35983F0C566239D0BA26CDD8DBA90F67029C`.
- `tests/test_orchestrator.py` SHA256: `8FD3D03EC35928F0B61F6574E0B5206554F5F104E8F30072F5E50EFCCA94CCD3`.
- `run_suite.ps1` SHA256: `DDDDB778BE4C3897FE00232BB60004772B76171F53A81B9C3EDE824CC3F569FA`.
- The supplied `formal_snapshot` result of 17 entries, including policies, with `all_expected=True` was accepted as a review premise.

No tests were executed during this report-only review.

## Review-08 Finding Status

### P1-1: Atomic fresh ownership and TOCTOU - CLOSED

Files and lines:

- `orchestrator.py:423-452`
- `orchestrator.py:455-498`
- `orchestrator.py:509-580`
- `orchestrator.py:1168-1259`
- `tests/test_orchestrator.py:70-162`

The repair now has the required ownership sequence:

1. The fresh bootstrap skeleton is validated before claim creation at `orchestrator.py:1181-1187`.
2. `_create_initialization_claim()` writes directly to `run_manifest.json` using `_open_new_binary()` with `O_CREAT | O_EXCL | O_WRONLY` at lines 423-427 and 487-490. Only one contender can acquire the path.
3. The claimed layout is revalidated before identity construction at lines 1214-1218. A foreign entry inserted in the validation-to-claim gap therefore fails before `_new_run_identity()`.
4. The persisted claim is reread and compared with the in-memory winner claim at lines 1219-1225.
5. Only after those checks does line 1239 build the run identity.
6. The claim is embedded in the manifest identity payload at lines 509-530 and independently validated whenever the manifest is read at lines 533-552.
7. `_atomic_json()` uses a PID plus random nonce temporary filename and exclusive creation at lines 439-446, then atomically replaces the claim file. The previous shared `run_manifest.json.tmp` collision is removed.

The two-contender test at `tests/test_orchestrator.py:89-128` synchronizes both callers immediately before the real O_EXCL claim and proves exactly one owner succeeds. The gap-mutation test at lines 131-162 inserts pollution during claim creation and proves post-claim validation rejects it before identity or stage creation.

The remaining unavoidable operating-system interval between individual path checks is handled fail-closed for legitimate suite contenders: the exclusive manifest claim establishes one owner, the winner revalidates the complete fresh layout, and later manifest reads revalidate the run/cache/manifest paths. No new P0/P1 remains from review-08 P1-1.

### P1-2: Lexical path and Windows link/reparse handling - CLOSED

Files and lines:

- `orchestrator.py:93-105`
- `orchestrator.py:108-159`
- `orchestrator.py:162-251`
- `orchestrator.py:254-282`
- `orchestrator.py:770-785`
- `orchestrator.py:533-552`
- `tests/test_orchestrator.py:165-306`

`_run_root()` now returns the lexical `<runs>/<run-id>` entry at `orchestrator.py:780-785`; it no longer resolves away the entry before validation. `_plain_directory()` and `_plain_file()` use `lstat`, reject symbolic links and Windows reparse attributes, require the expected file type, resolve strictly, and require exact canonical equality at lines 101-147.

The checks cover:

- the lexical run-root entry at lines 150-159;
- `runtime_cache` at lines 170-183;
- every runtime-cache entry and each required cache child at lines 184-204;
- `run_manifest.json` at lines 205-219;
- runtime-cache creation races at lines 254-282;
- all resume/verify manifest reads through `_validate_existing_run_layout()` at lines 533-552.

Fresh mode validates before and after cache normalization at `orchestrator.py:1177-1187`, then validates the exact claimed layout again at line 1218. Resume and verification load the manifest only after the same plain-path, reparse, canonical-containment, and manifest-file checks.

The new tests cover run-root, runtime-cache, and cache-child symbolic links at `tests/test_orchestrator.py:165-219`, plus real Windows junctions at all three positions at lines 222-306. The supplied `5 passed, 3 skipped` result is coherent: the three account-restricted symbolic-link creations skip on WinError 1314, while the three Windows junction cases execute and pass.

The implementation deliberately avoids depending on `Path.is_junction()`. It uses `lstat().st_file_attributes` plus `FILE_ATTRIBUTE_REPARSE_POINT`, with a numeric fallback, so the added reparse check does not require a newer pathlib API. Both frozen project and MuJoCo environments identify as Python 3.11; no new Python-version regression was found.

## New Finding

### P1-N1: Resume without a valid prior manifest initializes a new run and can adopt unowned stage debris

Files and lines:

- `run_suite.ps1:23-35,40-58`
- `orchestrator.py:1168-1187`
- `orchestrator.py:1201-1245`
- `orchestrator.py:1246-1259`
- `tests/test_orchestrator.py:565-576`
- `2026-10-07-sim2sim-root-cause-suite-design.md:1283-1290`

Frozen design §10.2 requires resume to begin from an existing run manifest whose run, source, suite-code, scenario, and stage identities match. The current branch does not require that precondition.

Reproducible logic:

1. `run_suite.ps1 run --resume <id>` creates or ensures the three runtime-cache directories before Python starts.
2. `run_definitions(..., resume=True)` permits a newly created or pre-existing run root at `orchestrator.py:1168-1185`.
3. If `run_manifest.json` is absent, `existing_manifest` is false at lines 1201-1203.
4. The shared `else` branch at lines 1213-1245 creates a new initialization claim, builds a new run identity, and constructs a new manifest even though the caller explicitly requested resume.
5. Launch history then records `resume: true` at lines 1246-1255, although no prior identity-bound run existed.

This is not theoretical. `tests/test_orchestrator.py:565-576` creates `stages/A/attempt-0001.incomplete` without any manifest, calls `resume=True`, and expects the orchestrator to create a new manifest and execute `attempt-0002`. Such a directory cannot be proven to belong to the requested run because the prior manifest and run identity are absent. The test therefore codifies adoption of pre-identity stage debris rather than the frozen resume contract.

Impact: a mistyped or nonexistent resume id silently becomes a new run whose provenance says `resume=true`; a directory containing arbitrary non-manifest stage state can be incorporated under a newly generated identity. The unselected incomplete attempt is not reused as evidence, but its presence predates and is not bound by the new run identity. This is a P1 provenance and evidence-boundary defect.

Minimum fix: split the manifest branch by mode. When `resume=True`, require an existing plain, canonical, hash-valid full run manifest before creating any claim, identity, or stage state; absence or a claim-only/incomplete manifest must fail closed. Only fresh mode may create the O_EXCL initialization claim and new identity. Replace the manifest-less incomplete-attempt test with a rejection test, while retaining the existing valid-manifest resume/reselection tests.

If recovery from a claim-only initialization crash is desired, define a separate explicit recovery contract. It must not be treated as ordinary `--resume` without a complete prior run identity.

## Regression Assessment

### Manifest identity

The initialization claim is included in `_manifest_identity_payload()` and independently self-hashed. A completed manifest cannot drop or mutate the claim without failing manifest or claim validation. No regression was found here.

### Write guard and cache identity

The ownership/link changes occur before worker startup and do not alter worker command construction, bootstrap environment, write-guard installation, or run-local cache destinations. Canonical cache paths are now checked more strictly. No regression was found.

### Valid resume and verify

Resume from a valid full manifest still loads and verifies run identity before reusing attempts. Verify/report use the lexical run root and reject missing, linked, reparse-point, non-canonical, or identity-invalid manifests. These valid-manifest paths remain safe. P1-N1 concerns only the manifest-absent resume branch.

### Exception recovery

A crash after exclusive claim creation but before full-manifest replacement leaves a claim-only file. Subsequent manifest loading rejects it rather than treating it as evidence, which is fail-closed. Reusing the same id is not automatic; this is acceptable unless an explicit initialization-recovery contract is added.

### Test interpreter boundary

`tests/test_mujoco_worker.py` imports `mujoco` directly and the frozen design assigns that test to `sim2sim/mujoco/.venv/Scripts/python.exe`. Its initial collection failure under the project interpreter is therefore a command/interpreter-boundary mistake, not a product-code defect. The corrected project run with explicit `--ignore` and the separate MuJoCo interpreter result are the relevant records.

## Test Record Assessment

Accepted supplied results:

- Targeted ownership/link boundary: `5 passed, 3 skipped`.
- Complete orchestrator: `27 passed, 3 skipped`.
- Project interpreter with explicit MuJoCo-test exclusion: `158 passed, 4 skipped`.
- MuJoCo interpreter: `12 passed`.

The three targeted skips are explained by unavailable privileged symbolic-link creation; the corresponding real Windows junction cases ran successfully. The fourth project skip is consistent with the supplied environment explanation. The tests adequately close review-08's two findings, but the manifest-less resume test currently asserts behavior that conflicts with frozen §10.2.

## Findings Summary

### P0

None.

### P1

- P1-N1: manifest-less resume initializes a new identity and can adopt unowned stage debris.

### P2

None.

## Final Decision

**NOT APPROVED**

`P0=0`, `P1=1`, `P2=0`. Review-08's ownership and link/reparse blockers are closed, but the current resume-without-manifest path violates the frozen provenance contract. Real Core V1 must remain blocked until resume requires a valid prior full manifest and the conflicting test is corrected.
