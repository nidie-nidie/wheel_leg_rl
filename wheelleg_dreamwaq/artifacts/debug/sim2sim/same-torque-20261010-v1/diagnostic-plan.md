# Same external joint torque diagnostic plan

Goal: distinguish the action-to-torque controller contribution from the remaining mechanical/solver response in the first 20 ms without ground contact.

The user approved the free-base, gravity-on, closure-on experiment and requested one subagent review after execution. Implementation and simulation remain in this session; only the completed evidence and conclusion will be delegated for review. No training or production configuration changes are authorized by this experiment.

## Experimental contract

- Reuse the eight nominal reset states, run-02 first-action file, reset cache, production physics steps (Isaac 5 ms; MuJoCo 1 ms), gravity and free base from the completed ground-off diagnostic.
- Baseline: run the shared first action with production controllers for one 20 ms control cycle in each engine, in separate fresh processes. Require exact reproduction of the previous ground-off physical trace before interpreting added observations.
- Direct conditions: zero external torque, and a fixed six-joint external torque derived from MuJoCo's production PD calculation at the shared first-action reset state. Store float32-representable canonical values once; both engines read the same file. Hold the torque for 20 ms; do not evaluate another policy action.
- In Isaac only, after the final reset, write stiffness and damping zero for the six active joints and their actuator caches. Keep the 20 passive joints, armature, limits, closure joints, reset, scene, gravity, integration and solver options unchanged. `_apply_action` writes the fixed canonical torque after the existing sign transform. The inherited step and reset still run.
- MuJoCo already has six direct motor actuators with zero active joint damping. Replace only the runtime controller's `compute_torque` callback with the fixed native torque. Keep model, passive damping, limits, solver and runtime stepping unchanged. Validate unit gears, fixed gain, zero bias and force/control range before running.
- Log the force argument submitted to the Isaac PhysX actuation-force setter and MuJoCo actuator/generalized force from each solve. These establish external torque delivery with active PD disabled; they are not claims about directly measured original PhysX implicit drive forces.
- Verify within-engine initial states match baseline exactly. Compare cross-engine initial states after frozen joint/frame mapping; record tolerances and residuals rather than assuming bitwise agreement across engines.
- Observe joint state and base angular velocity at 5/10/15/20 ms. MuJoCo refreshes geometry only on shadow data, never on the production solver data. Verify frame conversion against a fresh-shadow body Jacobian and exact observer-free replay. Check zero ground forces and no episode resets or active velocity-limit crossings.

## Files and order

1. `protected-before.json`: captured hashes of 93 production source/config/model/asset/architecture/cache/policy files; validate against preceding ground-off evidence before any edits.
2. `make_inputs.py` -> `inputs.json`: validate old action and reset files; compute initial reference torque, apply effort limits, convert to canonical coordinates and quantize once to float32. Record input hashes, targets and reset values. Zero-torque condition is explicit zero, not zero normalized action.
3. `probe_isaac.py`: isolated copy of the existing probe, extended with `--drive-mode`, `--torques-from`, drive identity, direct external torque and setter-boundary logging. Use one condition per new process.
4. `probe_mujoco.py`: isolated copy, same modes/input file, motor delivery checks and the existing exact observer-free replay.
5. `run_diagnosis.py` -> `execution.json`: sequentially execute baseline and two direct conditions in both engines, with exact arguments, return codes, duration and separate logs.
6. `analyze.py` -> `analysis.json`, `report.md`, `response.png`: verify hashes, mode/drive identity, timing, initial states, force delivery, frame conversion, baseline reproduction and replay before comparing motions; record per-environment vectors as well as aggregates and per-joint errors.
7. One read-only subagent reviews code, evidence, interpretation and limitations. Save its review. Address any correctness issue and repeat affected checks before reporting.

`build_probe_copies.py` records and generates the copies with guarded replacements and unified diffs. The existing probes are immutable inputs, not edited in place.

## Decision and limits

Shrinking error supports an important controller/drive contribution, not a unique root cause or a percentage of causal attribution. Persistent error supports a remaining mechanical/constraint/integration component; retained passive damping can itself differ in integration semantics. Constant torque is an intervention, not the original unknown Isaac motor-force waveform. Therefore compare response magnitudes and relative errors too, so weaker excitation alone cannot masquerade as improved agreement.

This does not prove sim2sim policy success, justify production parameter fitting, identify one solver parameter or decide that retraining will fix transfer. First localize the observed difference; do not change history or AdaBoot.

## Execution checklist

- [x] Input file and protected hashes validated.
- [x] Six main independent processes and two matched-state supplementary processes exit zero with complete 20 ms evidence.
- [x] Baseline reproduces old trajectories; within-engine reset states are exact.
- [x] Only active stiffness/damping change; direct external torque delivery is verified.
- [x] Ground, timing, episode, velocity, frame and replay checks pass.
- [x] Analysis includes absolute/relative errors and individual conditions.
- [x] One subagent completes independent review; no blocking issue, causal wording narrowed, full result saved to `review.md`.
- [x] Final protected hash verification and analysis/source provenance check; concise user conclusion follows in the conversation.

## Evidence-driven refinement

Cross-engine nominal reset angles differed by up to 6.2854e-6 rad in passive joints, exceeding the original 1e-6 rad check. Do not treat nominal reset as bitwise equal. Supplement with two MuJoCo direct-torque runs using `--initial-state-from isaac-formal/evidence.json`; copy the measured 26 named joint positions/velocities and root height/normalized orientation at reset, before any physics step. Keep absolute x/y at each scene's intentional origin, with no ground and uniform gravity. Prove the matched quantities are equal, force delivery is equal, shadow-Jacobian conversion is correct, and observer-free replay remains exact. Quantify the change from nominal-init direct torque before concluding about its contribution.

The final analysis selects current-probe hashes only. Three initial MuJoCo runs precede added Jacobian observations; three later runs precede matched-state support. All intermediate evidence is retained. There are 14 completed physics subprocesses and eight final engine/mode combinations (six main plus two supplementary), not 14 independent repeated trials.
