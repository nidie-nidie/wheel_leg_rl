# Stop/reverse practice and vx reward implementation plan

> Use executing-plans inline. The user permits one sequential documentation reviewer and one sequential code reviewer; do not delegate implementation.

**Goal:** Add the stop_reverse_v1 DreamWaQ training profile, verify it, run four fresh 1000-iteration seeds, and evaluate all four in MuJoCo.
**Architecture:** Pure tick-based command practice feeds the existing task. Training profile settings are hashed; policy, storage, history mathematics and physics remain unchanged. Current-physics candidate evaluation stays separate from the historical V1 baseline.
**Tech stack:** Python 3.11, PyTorch 2.7, Isaac Lab 2.3.2, RSL-RL 3.1.2, MuJoCo 3.14.

Design: ../specs/2026-10-11-stop-reverse-speed-tracking-design.md.

## Task 1 — Document gate

- [x] Review spec and Architecture v0.26 §31 against commands.py, env.py, train/play/evaluation.
- [x] Invoke one agent to review documents only; save findings and resolve all P0/P1 before implementation.
- [x] Commit reviewed documents locally.

## Task 2 — Pure practice/profile and tests

Files: source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/{commands.py,training_profiles.py}; tests/unit/test_command_practice.py.

- [x] Add boundary tests before implementing:
```python
steps = torch.tensor([0,99,100,149,150,249,250,299,300,399,400,499])
expected = torch.tensor([1,1,0,0,-1,-1,0,0,1,1,0,0])
assert torch.equal(command_practice_factor(steps), expected)
```
Run: .venv/Scripts/python.exe -m pytest tests/unit/test_command_practice.py -q. First fail must identify the missing implementation.
- [x] Add immutable practice_schedule="disabled" to CommandRanges; implement factor and per-row initial_command/steps state. Keep reset sampling unchanged. Reject unknown schedule, invalid hold combination and invalid tick values.
- [x] Implement named profile application using dataclasses.replace:
```python
env_cfg.commands = replace(env_cfg.commands, hold_for_episode=False,
    practice_schedule="StopReverseCommandPracticeV1")
env_cfg.reward_weights = replace(env_cfg.reward_weights,
    tracking_vx=2.0, tracking_vx_enhance=2.0)
```
legacy_v1 applies disabled/hold true and 1/1. Recognition uses the saved schedule then strict current base-task validation; it cannot accept arbitrary restored weights.
- [x] Test no alias, partial reset, fixed height, profile defaults, reward doubling, remaining reward equality, hash changes, same seed command RNG continuity, wrong schedule and conflicting profile rejection.

Add command_contract_payload to both schemas/{manifest.py,dreamwaq_manifest.py}; include enabled stage boundaries/factors/control_dt/update timing and disabled=null. Export adds training_reward_weights. Add MuJoCo command_practice.py plus strict loader validation; its NumPy timeline consumes that payload without importing Isaac. Update run_training_suite.py to include existing optional Reward/* and Loss/* tags without increasing required PPO tags.

## Task 3 — Environment timing

File: source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py.

- [x] Initialize practice state and last_processed_common_tick; reset only selected rows after original command sampling.
- [x] At the end of _get_rewards, after original log construction:
```python
if practice_enabled and self.common_step_counter > last_processed_common_tick:
    advance_all_rows_once()
    self._commands.copy_(next_commands)
    self._state.command.copy_(self._commands)
    last_processed_common_tick = self.common_step_counter
```
Do not use episode_length_buf as practice clock. Do not call simulator writes, reset or RNG in advance.
- [x] Verify original reward command, next observation command, cloned partial-reset cache, extra reads and per-done history behavior with a real Isaac probe.

## Task 4 — Entry points, strict identities and evaluation gate

Files: scripts/{train_dreamwaq.py,play_dreamwaq.py,evaluate_isaac.py,run_dreamwaq_training_suite.py}; source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/isaac_evaluation.py.

- [x] Add --task-profile to training/suite and apply before base-task construction. Resume without an explicit flag recognizes the saved profile; conflicting explicit flag fails before learning.
- [x] Rebuild saved profile in play/evaluation before full checkpoint validation. Fixed play disables only runtime practice afterward, with true hold; copy commands and cached state together.
- [x] Add --candidate-only and a separate V2 evaluation builder. Explicitly bind actual fixed scoring RewardWeights(), disabled practice, no historical baseline. Preserve V1 builder and comparison validator.
- [x] Add active-frame vx/yaw MAE to V2 reports. The first observation and all rewards use fixed commands, with 499-step timeout.
- [x] Suite passes new profile, skips legacy baseline only in explicit candidate-only mode, and records baseline acceptance=null/not_comparable. Estimator/performance failure is reported without preventing subsequent MuJoCo evaluations.
- [x] Test JSON round trips, new-vs-old contract mismatch, unknown profile rejection, fresh/resume identity and suite gate semantics.

## Task 5 — Independent dynamic MuJoCo diagnostic

Files: scripts/evaluate_command_practice_mujoco.py; sim2sim/mujoco/tests/test_command_practice.py.

- [x] Generate four 500-tick traces at initial commands ±0.5 vx and ±0.6 yaw; use six fixed phases in the spec.
- [x] Pass next_command to runtime.step while logging current executed command. Never edit the preceding history row.
- [x] Apply existing evaluation failure checks and report valid prefix/failure distinctly. Measure per-stage steady errors, stop displacement and sustained band response time, with null for unreached/incomplete segments.
- [x] Test all boundaries against Isaac factor and test history continuity plus synthetic settled/not-settled/failed statistics. Do not change formal eight-scenario metrics, runtime controller or model files.

## Task 6 — Code review and runtime verification

- [x] Run main unit and MuJoCo unit suites; inspect failures.
- [x] Invoke one agent to review code against spec; fix all P0/P1 and relevant omissions.
- [x] Run real Isaac command probe, portable fresh 2-update train, same/cross-size resume to 3 total updates, fixed-command play, actual exporter/loader and golden vectors. Require produced JSON/checkpoint markers, not just process exit code.
- [x] Re-run affected tests after review fixes, update review report, architecture implementation status and README; commit locally.
- [x] Freeze hashes before formal suite.

## Task 7 — Four training seeds and sim2sim

- [ ] Start the exact formal command in the spec, four fresh seeds, 1000 total updates each.
- [ ] Monitor every 1800 s and record actual tracking, termination, saturation and estimator diagnostics; stop only for hard numerical/artifact issues. Do not alter sources mid-suite.
- [ ] After all training, export/golden, candidate-only Isaac eight scenarios, formal MuJoCo eight scenarios, independent dynamic MuJoCo four scenarios, and formal ranking for every checkpoint.
- [ ] Save report comparing the four current-physics candidates. Separate completed evidence, speed targets, estimator gate and unavailable old-baseline comparison; give explicit remaining failures and next training recommendation.
