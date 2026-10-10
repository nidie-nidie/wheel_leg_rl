# Sim2Sim RootCauseSuite Core V1 Code Review 08

Date: 2026-10-08  
Review mode: independent, defect-first, fresh-run bootstrap skeleton incremental review  
Result: **NOT APPROVED**

## Conclusion

- P0: 0
- P1: 2
- P2: 0
- The reported single-process failure is fixed: a fresh run can accept the exact launcher-created skeleton containing only `runtime_cache/{pycache,pytest-cache,pytest-tmp}`, and the three reviewed pollution forms fail closed.
- Sequential reuse of an already materialized run remains rejected, and the fresh-only branch does not alter normal resume loading.
- Two blocking gaps remain: fresh ownership is not claimed atomically, and link rejection is incomplete for a resolved run-root alias and Windows junction/reparse-point directories.
- The supplied records were accepted without rerunning tests: targeted `4 passed`; complete project `153 passed, 1 skipped`; MuJoCo `12 passed`.

Approval requires `P0=0 && P1=0`; therefore the result is **NOT APPROVED**.

## Reviewed Baseline

- `orchestrator.py` SHA256: `54DDC62619BE86F60770E0C83B3D22CD09C971D0AAC9671BF4F6D9C5CF22158F`.
- `tests/test_orchestrator.py` SHA256: `8F60FA4CC3537D0842AFDFFEDD9A25BF6243E02782C2695B7A79781809AAAB9D`.
- Frozen design SHA256: `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD`.

No tests were executed during this report-only review.

## Correctly Implemented Behavior

The following parts of the repair are consistent with design §3.3 and §16:

- `orchestrator.py:46` freezes the three launcher-created cache directory names.
- `orchestrator.py:89-103` accepts only one top-level entry named `runtime_cache`, requires exactly the three expected children, and rejects ordinary files, extra directories, non-directory children, non-empty cache directories, and ordinary symbolic links that remain visible at that point.
- `orchestrator.py:911-918` applies the skeleton exception only to fresh mode and then ensures the three cache directories exist for both fresh and resume flows.
- `tests/test_orchestrator.py:65-77` covers the canonical positive skeleton.
- `tests/test_orchestrator.py:80-101` covers a foreign top-level file, a file inside a cache directory, and a foreign cache subdirectory.
- A completed or partially materialized sequential run contains top-level state beyond the canonical skeleton and is rejected before a new fresh manifest is accepted.
- The failed startup directory described in the request, containing only the three empty canonical cache directories and no manifest/stage state, now matches the intended recoverable bootstrap state.

Design consistency is otherwise sound: §3.3 at design lines 133-149 requires all project-controlled caches and writes to remain under the suite run, and §16 at lines 1480-1492 requires one command to complete a fresh run while retaining reliable resume and independent code approval.

## P0

None.

## P1

### P1-1: Fresh-run ownership is still a check-then-act operation, so duplicate-run protection is not atomic

Files and lines:

- `run_suite.ps1:30`
- `orchestrator.py:89-103`
- `orchestrator.py:911-918`
- `orchestrator.py:932-974`
- `orchestrator.py:244-248`
- `tests/test_orchestrator.py:65-101`

The launcher generates fresh ids with second-level precision at `run_suite.ps1:30`. Two launches in the same second can therefore select the same run root and both create the same canonical empty cache skeleton.

Each process can then execute this sequence:

1. Observe `run_root.exists()` and pass `_validate_fresh_bootstrap_skeleton()` at `orchestrator.py:911-913`.
2. Re-create the already existing directories with `exist_ok=True` at lines 914-918.
3. Observe no `run_manifest.json` at lines 932-943.
4. Independently build the run identity at line 957.
5. Write the manifest at line 974.

There is no exclusive claim between steps 1 and 5. The manifest helper also uses the shared fixed sibling name `run_manifest.json.tmp` at `orchestrator.py:244-248`, so concurrent writers can overwrite/remove the same temporary path. If both processes pass the manifest-existence check, they can race on the manifest and later on `attempt-0001.incomplete`, producing an exception, lost launch history, or mixed ownership rather than a deterministic duplicate-run rejection.

The same window allows a foreign file, directory, or replacement link to be inserted after the skeleton validation and before manifest creation. The run identity does not bind the run-root skeleton itself, so this post-validation mutation is not converted into an identity failure.

This violates the requested repeated-run protection and leaves a real TOCTOU boundary in the fresh path. It is P1 because it can corrupt or invalidate run evidence under a plausible same-second duplicate launch, although it does not unconditionally affect a single launch.

Minimum fix: after validating the canonical skeleton, atomically claim fresh ownership using a run-local create-new primitive, such as `os.open(..., O_CREAT | O_EXCL | O_WRONLY)` on a dedicated initialization record. Only the winning process may build identity or write the manifest. The claim must remain identity-bound or transition atomically into the manifest; a fixed shared `.tmp` path must not be used by competing owners. Revalidate the skeleton after acquiring the claim. Add a deterministic two-contender test proving exactly one fresh owner succeeds and a mutation-between-validation-and-claim test proving fail-closed behavior.

### P1-2: The link checks do not fail closed for a resolved run-root alias or Windows junctions

Files and lines:

- `contracts.py:184-204`
- `orchestrator.py:89-103`
- `orchestrator.py:515-528`
- `orchestrator.py:944-958`
- `orchestrator.py:282-296`
- `tests/test_orchestrator.py:65-101`
- `2026-10-07-sim2sim-root-cause-suite-design.md:133-149,1345,1480-1492`

`RootCauseOrchestrator._run_root()` passes the lexical `<runs>/<run-id>` path through `require_path_within()` at `orchestrator.py:524-528`. `require_path_within()` resolves the candidate before returning it at `contracts.py:194-204`. Consequently, if the run-id entry is a symbolic link or junction to another in-bounds directory, the validator receives the resolved target rather than the link entry. The `run_root.is_symlink()` check at `orchestrator.py:90` therefore cannot detect the original link.

An in-bounds alias can then pass the canonical skeleton check, while the new manifest records the requested id at `orchestrator.py:944-958`. Later manifest loading requires that id to equal `Path(run_root).name` at `orchestrator.py:295-296`; for an alias whose target has another name, the run becomes internally inconsistent only after state has already been written.

The nested checks at `orchestrator.py:96` and `orchestrator.py:102` use only `Path.is_symlink()`. On Windows, directory junctions and other reparse-point directories are not comprehensively represented by that predicate. A `runtime_cache` or cache-child junction can therefore appear as a normal directory, expose the three expected empty names, and redirect later cache writes outside the intended run. This conflicts with design §3.3's write boundary and the explicit symlink/junction fail-closed requirement at design line 1345.

No new tests cover run-root links, runtime-cache links, child links, Windows junctions, or reparse-point substitution. The three pollution cases only use ordinary files and directories.

Minimum fix: preserve and inspect the lexical run-id entry before resolution, and reject every link/reparse-point component for the run root, `runtime_cache`, and all three cache children. On Windows/Python 3.11 this requires an `lstat`/file-attribute or reparse-tag check rather than relying only on `is_symlink()`; on platforms exposing `is_junction()`, reject that as well. Then resolve each accepted path and prove exact containment. Repeat the no-link/canonical checks after acquiring the atomic fresh claim. Add tests for an in-bounds run-root symlink alias, runtime-cache and child symlinks, and Windows junction cases when the platform supports creating them.

## P2

None.

## Requested Check Results

### Design §3.3 and §16

The canonical bootstrap exception is consistent with the required run-local cache layout and restores the single-command fresh path. The unresolved TOCTOU and reparse-point cases still violate the design's fail-closed write boundary and prevent §16 completion.

### Repeated run protection

- Sequential duplicate fresh run: protected by the exact top-level skeleton check.
- Concurrent duplicate fresh run with the same id: not protected atomically; P1-1 applies.

### Resume

The new validator is gated by `not resume` at `orchestrator.py:912`, so normal resume manifest loading, identity verification, and completed-attempt reuse are not directly changed. The unified cache-directory `mkdir(..., exist_ok=True)` preserves the prior cache targets. No resume regression was found in this delta.

### Symlink and pollution handling

- Ordinary foreign files/directories and non-empty cache directories: fail closed.
- Ordinary nested symbolic links visible to the validator: fail closed.
- Run-root link aliases resolved before validation: not fail closed.
- Windows junction/reparse-point directories: not comprehensively rejected.

### Run identity and source snapshot timing

In the uncontended canonical path, identity is built after cache skeleton validation and before manifest/stage execution, and the run directory does not contaminate the suite source snapshot. That ordering is correct. It does not, however, bind or lock the validated skeleton; concurrent mutation during identity construction remains possible under P1-1.

### Windows path behavior

Canonical containment rejects a run-root target that resolves outside `runs`, but it does not preserve the fact that an in-bounds lexical run-id entry was itself a link. Nested junctions are not passed through the same containment guard. P1-2 therefore remains blocking on the frozen Windows execution platform.

## Test Coverage Assessment

The four new targeted cases correctly cover the reported ordinary bootstrap state and three ordinary pollution forms. They do not cover either blocking boundary identified above. The supplied full regression counts do not substitute for missing concurrency and Windows reparse-point cases.

## Final Decision

**NOT APPROVED**

`P0=0`, `P1=2`, `P2=0`. The original single-process `FileExistsError` is repaired, but fresh ownership and Windows link handling are not yet fail-closed, so the approval threshold is not met.
