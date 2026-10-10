# Sim2Sim RootCauseSuite Core V1 Code Review 11

Date: 2026-10-08  
Review mode: independent, defect-first, read-only incremental review  
Result: **NOT APPROVED**

## Conclusion

- P0: 0
- P1: 1
- P2: 1
- Review-10 P1-1 is closed for missing, manifest-less, claim-only, corrupt, and statically linked/reparse resume targets: non-fresh launcher commands perform only plain-entry validation before Python and no longer create cache directories.
- The new `CliExecution` gate correctly suppresses `parent_guard.json` when `_load_manifest()` itself rejects missing, corrupt, claim-only, source-drifted, formal-drifted, or otherwise identity-invalid manifests.
- One blocking identity-state defect remains. `_load_manifest()` sets `run_identity_accepted=True` before resume/core command-specific stage-set, definition, and scope identity checks. Those later checks can raise `IdentityDriftError`, but `cli.py` preserves the already-true flag and `__main__.py` writes `parent_guard.json` into the rejected run.
- One non-blocking exit-code defect remains. Launcher preflight failures use an uncaught PowerShell `throw`, producing the shell's generic failure code rather than the frozen code `5` used for evidence-integrity failures. The new tests assert only nonzero for these cases.
- The supplied test records were accepted without rerunning pytest. No `launcher-*` run directories remain.
- The approval threshold `P0=0 && P1=0` is not met.

**Final decision: NOT APPROVED. Real Core V1 execution is not authorized.**

## Frozen Baseline

All requested hashes were recomputed and match:

- Architecture v0.21: `5C7AF59F56750B50BA3CF14D731DF619489C0BBDBCCDA620B885B610087BB3AE`.
- RootCauseSuiteCoreV1.17 design: `8889B6102BC7F255638B8208A7A9C35A6B3F3E4644E87C9C4A3F3053A69DC4FD`.
- Factor-path allowlist: `4CF2A34D4E3725A78DC0028AD92FF6DCD04E820874CFFCD8757F267F8DADDD85`.
- `run_suite.ps1`: `BFE2E7FDF4CB90EFF013F5421B70130663D04F06422142A1B2C2AB28F9F71DC0`.
- `__main__.py`: `9733343F7853A8C01BF27B55909DFA7D88ADE5F235CCA732D96A69B9757A70BC`.
- `cli.py`: `FD4CC4413A6BD2366EF6250CE58B44166317BC8A3E0B83C91CFC4BD3C37EEDBC`.
- `orchestrator.py`: `421E168FD4262606C1AA61629A5737E58F9C4F9A57644FCD7CA23136D408F554`.
- `tests/test_run_suite_launcher.py`: `012261C9E71230DE2451FE59AE5A2FCD7A3B6D0C3B5F18027A9B9C4746455B94`.
- `tests/test_orchestrator.py`: `D62E5D6201B5AC47E9AD9A7E4AF0DD7E2FED83DBB7F050171520669F8A0B9EB2`.
- The supplied 17-entry `formal_snapshot`, `all_expected=True`, was accepted.

No tests were executed for this report.

## Review-10 P1-1 Status

### Official entry write-before-validation: CLOSED

Files and lines:

- `run_suite.ps1:23-38`
- `run_suite.ps1:43-48`
- `run_suite.ps1:50-130`
- `run_suite.ps1:132-145`
- `tests/test_run_suite_launcher.py:158-289`

`$isFreshRun` is true only for `run` without `--resume` at lines 23-34. `New-Item` exists only in that fresh branch at lines 109-122. Resume, verify, and report instead require, in order, an existing plain non-reparse run root, manifest, runtime cache, and all three cache children at lines 123-130. These checks execute before environment setup and Python launch at lines 132-145.

The preflight uses `File.GetAttributes()` and rejects `FileAttributes.ReparsePoint` for each required entry. Because no non-fresh `New-Item` occurs, a pre-existing run-root, runtime-cache, or cache-child junction is rejected before redirected content can be created. Python then repeats lexical/canonical/reparse validation before manifest loading.

The official launcher tests invoke `run_suite.ps1` itself under Windows PowerShell 5.1 and PowerShell 7. They cover missing run for resume/verify/report, manifest-less debris for all three commands, claim-only/corrupt resume manifests, and all three real junction positions. Their tree snapshots and `parent_guard.json` assertions address the review-10 reproduction rather than testing an extracted implementation fragment.

## P0

None.

## P1

### P1-N1: Post-load identity drift retains an accepted flag and writes parent evidence

Files and lines:

- `orchestrator.py:989-999`
- `orchestrator.py:1216-1223`
- `orchestrator.py:1522-1544`
- `cli.py:36-46`
- `cli.py:61-63`
- `__main__.py:39-64`
- `tests/test_run_suite_launcher.py:212-244`

Reproducible logic:

1. `_load_manifest()` validates the stored manifest and run identity, then unconditionally sets `self.run_identity_accepted = True` at `orchestrator.py:998`.
2. Resume performs additional identity checks only after that return: stage-id equality, current worker-definition hash equality, and scope equality at lines 1218-1223. Each mismatch raises `IdentityDriftError` while the flag remains true.
3. Verify/report similarly call `_load_manifest()` at line 1529 before comparing the stored Core definitions with the current frozen definitions at lines 1535-1544. A rehashed, internally self-consistent manifest with drifted definitions reaches this later identity failure after the flag is true.
4. The stored full run identity does not bind the manifest's stage list, definition payload, or scope. `integrity.py:212-275` binds formal inputs, suite sources, catalog, interpreters, controlled-project snapshot, environment, and initial argv, so these later manifest/current-command comparisons are not redundant.
5. `cli.py:61-63` catches `IdentityDriftError` and constructs its result through the common helper at lines 40-46, which copies the already-true orchestrator flag.
6. `__main__.py:44-63` consequently writes `parent_guard.json` even though the command rejected the run for identity drift.

Concrete cases include:

- a valid manifest resumed with a different stage set, worker definition set, or scope;
- a manifest whose stage/definition/scope fields and self-hashes were changed consistently while its otherwise-current full run identity remained unchanged;
- verify/report rejecting current Core-definition drift after `_load_manifest()` accepted only the stored run identity.

This violates the requested rule that missing, claim-only, corrupt, or identity-drift runs must not receive parent evidence. It is P1 because the suite mutates a run after rejecting its active identity contract, so `parent_guard.json` no longer proves that the command accepted the complete identity required for that operation.

Minimum repair:

1. Separate manifest/run-identity verification from the acceptance transition. `_load_manifest()` should return a verified manifest without setting the flag, or accept the expected scope/stage/definition identity and set the flag only after all command-specific identity checks pass.
2. For resume, mark accepted only after lines 1218-1223 succeed. For verify/report, mark accepted only after exact Core scope, stage set, and definition payload/hash checks succeed.
3. As defense in depth, the `IdentityDriftError` handler in `cli.py` should return `run_identity_accepted=False` rather than preserving mutable orchestrator state.
4. Add official-entry tests under both PowerShell engines that trigger scope/stage/definition identity drift and assert exit code `3`, byte-for-byte unchanged run tree, and no `parent_guard.json`.
5. Add a positive failure-path test showing that a fully accepted valid run which later reaches `StageExecutionError` returns code `4` and does write parent guard evidence.

## P2

### P2-1: Launcher preflight failures return an undocumented shell exit code

Files and lines:

- `run_suite.ps1:50-130`
- `tests/test_run_suite_launcher.py:158-209`
- `tests/test_run_suite_launcher.py:247-289`
- `2026-10-07-sim2sim-root-cause-suite-design.md:1292-1302`

The new preflight correctly fails before Python, but its validation helpers use uncaught PowerShell `throw`. Under `-File`, that path terminates with the shell's generic failure code rather than the frozen CLI taxonomy. Before this change, the same missing run, missing manifest, or reparse-layout failure reached `EvidenceIntegrityError` and returned code `5` through `cli.py`.

The design defines only codes `0`, `2`, `3`, `4`, and `5`; code `5` covers schema/hash/internal evidence contradictions. The new tests assert only `returncode != 0` for missing run, manifest-less debris, and junction rejection at lines 175, 203, and 282. Only the claim-only/corrupt cases that reach Python assert exact code `5` at line 240.

This remains fail-closed and cannot authorize invalid evidence, so it is P2 rather than P1. The launcher should map preflight layout/evidence failures to `5` and argument/configuration failures to `2`, with exact-code assertions under both PowerShell engines.

## Python Acceptance Semantics

The new mechanism is otherwise correctly structured:

- `CliExecution` carries return code and acceptance state separately at `cli.py:18-21`.
- Missing, claim-only, corrupt, manifest-hash-invalid, design/allowlist drift, source drift, formal drift, interpreter drift, and environment drift all fail inside `_read_manifest_payload()` or `verify_run_identity()` before `orchestrator.py:998`; their flag remains false and `__main__.py` does not write parent evidence.
- Fresh mode writes its completed manifest at `orchestrator.py:1265`, then rereads it through `_load_manifest()` at lines 1266-1267 before the flag becomes true. Claim-only or incomplete fresh state therefore cannot become accepted.
- Once a fresh or resumed run has passed the complete intended identity gate, later stage failure leaves the flag true. `cli.py:67-69` returns code `4`, and `__main__.py:44-63` writes the parent guard. That is the required positive failure behavior, although no focused test currently proves it.

P1-N1 is specifically the gap between generic `_load_manifest()` acceptance and the later command-specific identity comparisons.

## Orchestrator Regression Check

No regression was found in the previously closed ownership/resume machinery:

- Fresh ownership still uses `_open_new_binary()` with `O_CREAT | O_EXCL` at `orchestrator.py:423-490`.
- Fresh mode validates the bootstrap skeleton before claim, validates the claimed layout after claim, rereads the persisted claim, and only then constructs the identity at `orchestrator.py:1189-1253`.
- Lexical path preservation and plain/reparse/canonical validation remain at `orchestrator.py:93-282` and `784-789`.
- Resume resets acceptance state at line 1173, rejects missing run/manifest before writes at lines 1176-1188, and loads the full prior identity before launch-history or stage writes.
- Completed-attempt reuse and tampered-attempt rerun remain at `orchestrator.py:1324-1407`.
- G01 reselection and replay-seal replacement still invalidate only transitive downstream selections; terminal G02 reselection still invalidates all transitive downstream selections. The unchanged tests remain at `tests/test_orchestrator.py:440-563`.

The launcher fresh branch performs compatible pre/post plain-directory checks around creation, and the Python orchestrator still enforces the exact canonical bootstrap skeleton before claiming ownership. The first fresh launch is not blocked by the new non-fresh preflight.

Residual risk: the launcher tests do not execute a complete fresh Core launch, and they do not deterministically inject a path replacement between the PowerShell precheck and `New-Item`. The existing post-create checks plus the Python claimed-layout revalidation retain the review-09 fail-closed model for ordinary suite contenders, but this hostile mutation interval remains a test blind spot.

## Test Record Assessment

Accepted supplied records:

- Official launcher: `29 passed`.
- Orchestrator: `28 passed, 3 skipped`.
- Project interpreter with the MuJoCo-only file excluded: `181 passed, 4 skipped`.
- MuJoCo interpreter: `12 passed`.
- All four skips are accepted as the stated symbolic-link privilege limitation; corresponding real Windows junction cases executed successfully.
- Post-test `launcher-*` temporary run count: `0`.

Coverage strengths:

- Both Windows PowerShell 5.1 and PowerShell 7 execute the formal script.
- Missing run and manifest-less debris cover resume, verify, and report.
- Claim-only/corrupt manifests prove no parent guard after `_load_manifest()` rejection.
- Real junctions cover run root, runtime cache, and cache child and assert no redirected tree change.

Coverage gaps:

- No test exercises an `IdentityDriftError` raised after `_load_manifest()` has set the acceptance flag.
- No test proves parent evidence is retained for a valid accepted run whose stage later fails.
- Preflight cases do not assert the frozen exact exit code.

The first gap exposes P1-N1; the latter two support the stated minimum repair and P2 finding.

## Findings Summary

### P0

None.

### P1

- P1-N1: post-load stage/definition/scope identity drift retains `run_identity_accepted=True` and writes `parent_guard.json` into a rejected run.

### P2

- P2-1: pre-Python launcher integrity failures return an undocumented generic shell code, and tests assert only nonzero.

## Final Decision

**NOT APPROVED**

`P0=0`, `P1=1`, `P2=1`. Review-10's write-before-validation defect is closed, the fresh ownership and resume/reselection machinery remains intact, and the supplied launcher tests materially improve the boundary coverage. However, parent evidence can still be written after command-specific identity drift because acceptance is marked too early. The approval threshold is not met, so real Core V1 must remain blocked.
