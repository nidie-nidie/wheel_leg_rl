# Ground on/off diagnosis implementation plan

Goal: test whether disabling ground contact substantially reduces the existing Isaac/MuJoCo motion divergence during the first 20 ms.

Architecture: reuse the completed initial-ground-contact probe and its ground-on evidence. Run diagnostic ground-off variants from the same eight-environment reset cache, with gravity, closure, drives, actions and timesteps unchanged. This is debug-only evidence, not a RootCauseSuite verdict or policy evaluation. The user approved this design and execution in the conversation; no additional agents are authorized.

## Files and execution

- `probe_isaac.py`: copy the existing probe; inherit production reset/step. Add `--no-ground` and `--control-ticks` (default 1). In `_setup_scene`, after the normal setup but before physics initialization, disable CollisionAPI on ground prims in the temporary stage. Keep geometry present for the existing gap observer. Keep contact sensors on and verify all post-step wheel contact forces are zero.
- `probe_mujoco.py`: copy the existing probe. Use the existing `build_robot_variant(..., operations=("no_ground",))` to write an isolated XML and manifest. Verify compiled options, reset, joints, bodies, actuators and equalities exactly equal the validated formal model; verify all removed explicit pairs involve the floor, and collision masks remain zero. Supply an explicitly generated diagnostic dynamics hash to the unchanged runtime. The absent ground has a virtual z=0 reference for gap observation only.
- `analyze.py`: validate input hashes and compare raw physical velocities at common 5/10/15/20 ms. Transform world/body angular velocities to ControlFrameV1 using the existing frame contract; align the six controlled joints by name and signs. Report angular vector L2 error and six-joint velocity RMS error, per environment and aggregate. Verify within-engine ground-on/off reset states and actions match.

Ground-on inputs: `../initial-ground-contact-20261010-v1/isaac-on/evidence.json`, `mujoco/evidence.json`, and `isaac-off/actions.json`. The historical `isaac-off` name means sensors off, **not** ground off.

Run Isaac with `.venv/Scripts/python.exe -B probe_isaac.py --headless --device cuda:0 --contact-sensors --no-ground --control-ticks 1`, supplying output, cache, actor and actions-from paths. Run MuJoCo with `sim2sim/mujoco/.venv/Scripts/python.exe -B probe_mujoco.py --no-ground --control-ticks 1`, supplying output and actions-from. Run analysis with the main Python environment.

Acceptance: both physics processes exit zero; correct 20 ms time/sample counts; no episode boundary; no post-step ground forces; gravity remains -9.81; within-engine initial states and shared actions unchanged; all protected production source/model/asset/cache/actor hashes unchanged; actual diagnostic inputs and results hashed. Use the existing observer-neutrality evidence plus exact no-observer MuJoCo replay for this variant. Self-review locally without another agent.

## Execution refinement and audit

Initial paired runs completed, but the Isaac first-action trace differed when the preceding zero-action case lasted 20 ms versus 100 ms. To eliminate that ordering confound, add `--case` to the same diagnostic probes and launch zero-action and shared-first-action conditions in separate new processes, for both ground modes and both engines: eight final executions, with arguments and exit codes recorded in `isolated-execution.json`. Each Isaac process inherits the production initialization/cache load and reset with no intervening episode physics. MuJoCo replays each of its eight conditions without the observer and compares final qpos/qvel exactly. Intermediate results remain auditable but are excluded from the final result table. The recorded outcome does not establish an observation/reset bug.

Verify the MuJoCo body-frame angular qvel conversion against the unchanged production kinematic collector on independently forwarded data for all 672 final snapshots. Final analysis consumes only isolated-process evidence and records all input hashes.

Interpretation: a decrease supports prioritizing contact-startup effects, including their coupling with drive/closure. A persistent gap demonstrates a non-contact component. The experiment alone cannot identify a single solver parameter, prove a closure/drive root cause, establish policy success, or decide that further training will solve the transfer failure.
