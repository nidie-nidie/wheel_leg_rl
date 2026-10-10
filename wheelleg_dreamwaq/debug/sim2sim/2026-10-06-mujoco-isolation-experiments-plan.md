# MuJoCo Isolation Experiments Implementation Plan

> **For agentic workers:** Implement each task in order and keep every change inside `debug/sim2sim/`.

**Goal:** Add debug-only fixed-base, gravity-free suspended, and grounded pitch-response experiments without changing the training environment, formal MuJoCo runtime, frozen model, manifests, or checkpoints.

**Architecture:** A small isolation runner loads the frozen policy contract and reuses the existing mixed position/velocity controller. Debug-specific model mapping and reset logic support models with or without `base_free`. Each run writes a compact `MujocoIsolationTraceV1` NPZ plus metadata under `artifacts/debug/sim2sim/`.

**Tech Stack:** Python 3.11, MuJoCo 3.14, NumPy, pytest, MJCF.

---

### Task 1: Define isolation scenarios

**Files:**
- Create: `debug/sim2sim/isolation_scenarios.py`
- Test: `debug/sim2sim/tests/test_isolation_scenarios.py`

- [x] Define fixed-base pulse, suspended pulse, and grounded pitch-response scenario contracts.
- [x] Generate deterministic six-dimensional action sequences.
- [x] Validate pitch, action, channel, and timing arguments.
- [x] Test sequence shape, pulse timing, and canonical wheel common action.

### Task 2: Add debug-only MJCF variants

**Files:**
- Create: `debug/sim2sim/models/wheel_leg_urdf4_fixed_base_debug.xml`
- Create: `debug/sim2sim/models/wheel_leg_urdf4_suspended_debug.xml`
- Test: `debug/sim2sim/tests/test_isolation_models.py`

- [x] Derive both files from the frozen formal XML.
- [x] Keep masses, inertias, joints, constraints, actuators, and meshes unchanged.
- [x] Remove ground contact from both models.
- [x] Remove `base_free` only in the fixed-base model and adjust its reset keyframe.
- [x] Set gravity to zero in both isolation models.
- [x] Verify model topology and actuator mapping with MuJoCo compilation tests.

### Task 3: Implement the isolation collector

**Files:**
- Create: `debug/sim2sim/collect_mujoco_isolation_trace.py`
- Test: `debug/sim2sim/tests/test_mujoco_isolation_collector.py`

- [x] Implement a debug model map that supports optional free joints.
- [x] Apply control-frame pitch to the free-base reset and align wheel contact with the floor.
- [x] Reuse the frozen `MixedActionController` without modifying it.
- [x] Record action, native/canonical targets and states, substep torque metrics, base motion, contact force, virtual-leg state, and all eight closure residuals.
- [x] Write trace, metadata, and file hashes atomically into a new output directory.
- [x] Test fixed-base, suspended, and grounded pitch runs headlessly.

### Task 4: Add analysis and documentation

**Files:**
- Create: `debug/sim2sim/analyze_mujoco_isolation.py`
- Modify: `debug/sim2sim/README.md`
- Test: `debug/sim2sim/tests/test_analyze_mujoco_isolation.py`

- [x] Summarize initial/final pitch, pitch-rate peak, base velocity, wheel speed, torque, contact, and closure error.
- [x] Compare positive and negative wheel commands against the zero-action baseline without ranking transient crossings as a controller.
- [x] Document exact commands and interpretation limits.

### Task 5: Execute the experiments

- [x] Run all debug isolation tests.
- [x] Run fixed-base pulses for all six action channels.
- [x] Run suspended wheel common and differential pulses.
- [x] Run grounded `+5 deg` and `-5 deg` pitch with negative, zero, and positive wheel common actions.
- [x] Save the reviewed rerun under `artifacts/debug/sim2sim/mujoco-isolation-20261006-v2/`.
- [x] Review the implementation with a separate review agent before reporting conclusions.
