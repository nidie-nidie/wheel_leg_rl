# Sim2Sim RootCauseSuite Core V1 Code Review 07

Date: 2026-10-08  
Review mode: independent, defect-first, final incremental review of review-06 P1-1  
Result: **APPROVED**

## Conclusion

- P0: 0
- P1: 0
- P2: 0
- Review-06 P1-1 is closed.
- Fresh `run`, `run --resume <id>`, `verify <id>`, and `report <id>` now use the untyped `$forwardArgs` array consistently.
- No regression was found in argument forwarding, run-id selection, bootstrap environment, runtime-cache placement, write-guard bootstrap, or source/run identity behavior.
- The supplied validation records are consistent with the reviewed implementation: cross-engine launcher tests `7 passed`; complete project tests `149 passed, 1 skipped`; MuJoCo tests `12 passed`.

The approval threshold `P0=0 && P1=0` is satisfied. **Final decision: APPROVED.**

## Reviewed Delta

- `run_suite.ps1` SHA256: `DDDDB778BE4C3897FE00232BB60004772B76171F53A81B9C3EDE824CC3F569FA`.
- `tests/test_run_suite_launcher.py` SHA256: `10C741E4ADFEE41540BE9739D01C4BFFBEC7C43F664BC291DCE2387EE85CC835`.
- Review-06 P1-1 required a separate untyped forwarding array plus executable Windows PowerShell 5.1 and PowerShell 7 coverage for zero, one, and two remaining arguments.

No tests were executed during this review. The supplied results were accepted as review evidence, as requested.

## Review-06 P1-1 Status

### CLOSED: null remaining arguments no longer flow through the typed parameter

Files and lines:

- `run_suite.ps1:7-15`
- `run_suite.ps1:23-35`
- `run_suite.ps1:58-62`
- `tests/test_run_suite_launcher.py:12-27`
- `tests/test_run_suite_launcher.py:30-82`

`run_suite.ps1:12-15` creates an untyped empty array and copies `$RemainingArgs` only when the typed parameter is non-null:

```powershell
$forwardArgs = @()
if ($null -ne $RemainingArgs) {
    $forwardArgs = @($RemainingArgs)
}
```

The implementation no longer assigns an array expression back into the `[string[]]$RemainingArgs` parameter. After the copy block, every relevant operation uses `$forwardArgs`:

- Resume lookup: `run_suite.ps1:25`.
- Resume count and indexing: `run_suite.ps1:27-28`.
- Verify/report count and indexing: `run_suite.ps1:33-34`.
- Final Python argument forwarding: `run_suite.ps1:58`.

There is no downstream use of `$RemainingArgs` after line 14. The Windows PowerShell 5.1 typed-array coercion identified by review-06 is therefore removed from the execution path.

## Command Semantics

### Fresh run

With no remaining arguments, `$forwardArgs` stays a real empty array. `IndexOf` returns `-1`, line 30 generates the fresh run id, and line 58 constructs the Python argv without a null or empty trailing argument.

### Resume

For `run --resume <id>`, `$forwardArgs` contains the two values in their original order. Lines 25-28 select the supplied id, and line 58 forwards the unchanged `--resume <id>` pair to Python.

### Verify and report

For `verify <id>` and `report <id>`, lines 33-34 require at least one remaining value and select the first value as the run id. Line 58 forwards the same command and id. Missing ids still fail before cache/bootstrap execution, preserving the prior fail-closed behavior.

## Test Adequacy

`tests/test_run_suite_launcher.py:12-15` extracts the normalization block directly from the formal launcher rather than maintaining an independent duplicate. The parametrization at lines 30-45 covers zero, one, and two remaining arguments, and lines 47-82 execute that exact block through both `powershell.exe` and `pwsh.exe`, checking array count, resume index, and value ordering.

The static test at lines 18-27 additionally confirms that:

- the former typed-variable reassignment is absent;
- `IndexOf` uses `$forwardArgs`;
- count access uses `$forwardArgs`;
- final argument concatenation uses `$forwardArgs`.

This directly covers the engine-specific failure from review-06 and is sufficient to close P1-1. The test does not launch the complete Core pipeline, but full launcher side effects are outside the narrow normalization regression and are covered by the supplied complete regression records.

## Bootstrap, Cache, Write Guard, And Identity

The change is confined to argument normalization and consumption before the existing run setup:

- Run-root containment remains at `run_suite.ps1:40-45`.
- Runtime-cache creation remains at `run_suite.ps1:46-49`.
- Bootstrap, run-id/run-root, and bytecode/cache environment variables remain at `run_suite.ps1:51-56`.
- The frozen project interpreter and `-B -m debug.sim2sim.root_cause_suite` invocation remain at `run_suite.ps1:18-20` and `run_suite.ps1:58-62`.

Fresh execution now reaches those controls instead of failing before them. Resume, verify, and report derive the same run id as before, so their run root and cache identity remain unchanged. The launcher and launcher-test hashes have changed as expected; a new run will freeze the current suite source identity, while an older identity must continue to fail closed on source drift.

## Findings

### P0

None.

### P1

None.

### P2

None.

## Residual Risk

- The cross-engine test executes the extracted normalization block rather than launching the full Core command. This avoids creating run artifacts and still exercises the exact code responsible for P1-1.
- The supplied full project and MuJoCo regressions were not independently rerun in this report-only review.

Neither item changes the approval decision for this narrowly scoped fix.

## Final Decision

**APPROVED**

`P0=0`, `P1=0`, `P2=0`. Review-06 P1-1 is closed, no blocking regression was found, and the approval threshold is satisfied.
