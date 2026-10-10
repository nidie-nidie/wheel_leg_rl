# WheelLeg Phase 1R Domain Randomization Implementation Plan

> **For Codex:** Execute this plan task by task. Do not start the four formal training runs until all unit/integration gates pass and an independent code review reports no blocking P0/P1.

**Goal:** Implement `FudanStyleDomainRandomizationV1` for the existing WheelLeg PPO environment, preserve a seed-independent checkpoint contract, verify closed-chain reset and implicit-actuator semantics, then run four fresh 1000-iteration trainings and evaluate every checkpoint in MuJoCo.

**Architecture:** A seed-independent schema defines ranges and lifecycle. A DirectRLEnv-owned runtime derives independent RNG streams, samples immutable process-start parameters, applies material/actuator changes, owns per-env joint references, reset velocity and Actor noise, and exports an immutable audit plus checkpointable RNG state. Reward/termination remain on clean state; Critic uses clean normalized Actor features while Actor receives independently normalized noisy features.

**Stack:** Python 3.11, PyTorch 2.7, Isaac Sim 5.1, Isaac Lab 2.3.2, RSL-RL 3.1.2, MuJoCo, pytest, TensorBoard.

---

## Task 1: Freeze the pure schema, RNG and frame utilities

**Files:**
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/randomization.py`
- Modify: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/frames.py`
- Create: `wheelleg_dreamwaq/tests/unit/test_randomization.py`
- Modify: `wheelleg_dreamwaq/tests/unit/test_frames.py`

1. Add failing tests for profile serialization, seed derivation, stream independence, range validation and frame round-trip.
2. Implement immutable nominal/randomized profiles, fixed stream names, SHA-256 seed derivation and seed-independent contract payloads.
3. Add `transform_control_vector_to_usd()` using the existing row-vector convention.
4. Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_randomization.py tests/unit/test_frames.py -q
```

## Task 2: Split clean and noisy observation construction

**Files:**
- Modify: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/observations.py`
- Modify: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/commands.py`
- Modify: `wheelleg_dreamwaq/tests/unit/test_normalization.py`
- Modify: `wheelleg_dreamwaq/tests/unit/test_commands.py`
- Modify: `wheelleg_dreamwaq/tests/unit/test_phase1_math.py`

1. Add tests for `(4,)` and `(num_envs,4)` references, physical-unit noise, no storage aliasing, clean Critic prefix and CPU-generator command sampling.
2. Introduce pure `PhysicalActorFieldsV1` construction and separate clean/noisy normalization paths.
3. Make command sampling occur on the generator device and copy to the requested output device.
4. Preserve exact nominal output when noise is absent.
5. Run the affected unit tests.

## Task 3: Implement the DirectRLEnv randomization runtime

**Files:**
- Create: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/randomization.py`
- Modify: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env_cfg.py`
- Modify: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py`
- Create: `wheelleg_dreamwaq/scripts/preflight_randomization.py`
- Create: `wheelleg_dreamwaq/tests/integration/test_randomization_runtime.py`

1. Resolve active/passive joints and wheel collision-shape spans once.
2. Sample and geometrically validate per-env `q_default_env`; retain rejection statistics.
3. Apply shared left/right wheel material, per-env gains and effort limits; synchronize PhysX plus all implicit-actuator mirror tensors.
4. Store per-env references and use them for reset, zero-action targets and joint-error observations.
5. Sample reset root COM velocity in ControlFrameV1, convert to USD/world, and sample Actor noise without touching clean state.
6. Replace the env-0 effort-limit cache with live per-env limits.
7. Expose immutable audit plus `get_rng_state()` / `set_rng_state()`.
8. Implement Gate A and Gate B traces in the standalone preflight script; do not add settling to training reset.
9. Run nominal-disabled regression and randomized integration gates.

## Task 4: Extend contract, manifest and checkpoint semantics

**Files:**
- Modify: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/schemas/manifest.py`
- Modify: `wheelleg_dreamwaq/source/wheelleg_dreamwaq/wheelleg_dreamwaq/training/checkpoint.py`
- Modify: `wheelleg_dreamwaq/scripts/train_ppo.py`
- Modify: `wheelleg_dreamwaq/tests/unit/test_manifest.py`
- Modify: `wheelleg_dreamwaq/tests/unit/test_checkpoint_metadata.py`

1. Upgrade to `Phase1RandomizedContractV1`; include profile semantics but exclude run/evaluation seeds and generator state.
2. Upgrade checkpoint metadata and store the seven environment generator states.
3. Finalize the run manifest atomically only after environment initialization/audit, before wrapper/runner creation.
4. On resume, discard wrapper bootstrap reset, restore all RNG states, force a full reset, then enter rollout.
5. Add tests proving four different training seeds share a contract hash and repeated resume produces the same post-restore reset sequence.

## Task 5: Update suite orchestration and monitoring

**Files:**
- Modify: `wheelleg_dreamwaq/scripts/run_training_suite.py`
- Modify: `wheelleg_dreamwaq/tests/unit/test_training_suite.py`
- Modify as needed: `wheelleg_dreamwaq/scripts/export_ppo_actor.py`

1. Write Phase 1R artifacts under a new versioned directory.
2. Train all four fresh seeds first; only after all pass monitoring, export and evaluate all four in MuJoCo.
3. Keep the 1800-second interval monitor and hard-stop on non-finite values, invalid termination, loop closure above 5 mm or explosive value loss.
4. Preserve seed-independent training fingerprint comparison while retaining seed-specific audit provenance.
5. Run suite unit tests.

## Task 6: Run the full pre-training verification gate

1. Run all unit tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit -q
```

2. Run the randomized preflight with a small environment count and retain Gate A/B JSON traces.
3. Run integration tests, including nominal regression and `16 env x 1000 control steps` randomized stability.
4. Run a one-iteration PPO startup smoke and MuJoCo smoke export/evaluation.
5. Do not proceed if any gate is relaxed, skipped or only visually inspected.

## Task 7: Independent code review

1. Launch one independent review agent after implementation and verification.
2. Ask it to inspect only correctness, contract consistency, RNG/resume, closed-chain reset, actuator/material synchronization, Actor/Critic separation and suite ordering.
3. Fix all P0/P1 findings and rerun affected tests; re-review only the changed risk area if needed.

## Task 8: Four fresh 1000-iteration trainings

Run from `wheelleg_dreamwaq`:

```powershell
.\.venv\Scripts\python.exe scripts\run_training_suite.py --mode formal --profile rtx5070 --iterations 1000 --runs 4 --monitor-interval-seconds 1800
```

Requirements:
- Four unique random seeds, no resume.
- No source edits while the suite source fingerprint is active.
- Inspect each 30-minute snapshot and final snapshot for non-finite values, loss explosion, reset storm, saturation, tracking degradation and loop-closure errors.
- If a hard anomaly occurs, stop before starting the next run and diagnose it.

## Task 9: Evaluate all four checkpoints in MuJoCo and report

1. Export every final Actor only after all four trainings complete.
2. Run the frozen stand/forward/backward/yaw/combined MuJoCo protocol for all four.
3. Record base height, roll/pitch/yaw, left/right phi0 difference, vx/yaw tracking, survival time, effort saturation and exact failure reason.
4. Produce the existing ranking artifact, but report every run rather than only the best candidate.
5. State explicitly which runs passed all scenarios; for every failure, identify the first failed scenario, tick/time, threshold and likely category: interface, randomized-policy robustness, contact/dynamics mismatch or training failure.

