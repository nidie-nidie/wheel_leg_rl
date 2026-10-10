# Sim2Sim RootCauseSuite Core V1 Code Review 06

Date: 2026-10-08  
Review mode: independent, defect-first, two-file incremental review  
Reviewed files: `run_suite.ps1`, `tests/test_run_suite_launcher.py`  
Result: **NOT APPROVED**

## Conclusion

- P0: 0
- P1: 1
- P2: 0
- The new normalization works under PowerShell 7 but does not eliminate the original null-array crash under Windows PowerShell 5.1.
- `run --resume <id>`, `verify <id>`, and `report <id>` retain their expected non-null argument ordering. The blocking path is fresh `run` with no remaining arguments.
- The new test is textual only and passes despite the runtime defect.

Approval requires `P0=0 && P1=0`; therefore the incremental change is **NOT APPROVED**.

## P0

None.

## P1

### P1-1: Assigning the array expression back to the typed parameter remains null in Windows PowerShell 5.1

Files and lines:

- `run_suite.ps1:7-12`
- `run_suite.ps1:20-31`
- `run_suite.ps1:55`
- `tests/test_run_suite_launcher.py:7-13`

`$RemainingArgs` is declared as `[string[]]` at `run_suite.ps1:7-8`. Line 12 assigns `@($RemainingArgs)` back into that same type-constrained variable.

For a fresh `run`, the incoming value is null. Under Windows PowerShell 5.1, assignment back into the `[string[]]` parameter coerces the result to null again. Consequently, `[Array]::IndexOf($RemainingArgs, '--resume')` at line 22 still throws `Value cannot be null. Parameter name: array`. The Core process is never launched.

Under PowerShell 7, the same expression produces a non-null array and line 22 returns `-1`. The fix is therefore engine-dependent and does not close the reported Windows PowerShell failure.

The non-null forms remain ordered correctly:

- `run --resume <id>` keeps `--resume`, then `<id>`.
- `verify <id>` keeps `<id>` as the first remaining argument.
- `report <id>` keeps `<id>` as the first remaining argument.

Minimum fix: normalize into a separate untyped variable instead of assigning back into the typed parameter, then use that variable for `IndexOf`, `.Count`, indexing, and final argument construction. For example:

```powershell
$forwardArgs = @()
if ($null -ne $RemainingArgs) {
    $forwardArgs = @($RemainingArgs)
}
```

Replace the downstream `$RemainingArgs` uses at lines 22, 24-25, 30-31, and 55 with `$forwardArgs`.

The added test at `tests/test_run_suite_launcher.py:7-13` is insufficient. It only confirms that the literal normalization text occurs before `IndexOf` and `.Count`; it does not execute either PowerShell engine and cannot detect typed-array coercion. The regression coverage must execute the zero-, one-, and two-remaining-argument cases under Windows PowerShell 5.1, with PowerShell 7 retained as a compatibility case.

## P2

None.

## Final Decision

**NOT APPROVED**

`P0=0`, `P1=1`, `P2=0`. Fresh `run` still reproduces the original null-array startup failure under Windows PowerShell 5.1, so this launcher change does not satisfy the approval gate.

No tests were run for this narrowed review, and no file other than this report was modified.
