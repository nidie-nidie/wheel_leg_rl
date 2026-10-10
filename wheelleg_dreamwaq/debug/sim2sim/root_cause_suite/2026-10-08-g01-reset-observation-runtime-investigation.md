# G01 Reset Observation Runtime Investigation

**Date:** 2026-10-08  
**Scope:** debug-only runtime investigation under `debug/sim2sim/root_cause_suite`  
**Frozen design:** `RootCauseSuiteCoreV1.17`  
**Formal training/model files modified:** none  
**Status:** G01 blocker explained; full Core V1 run remains blocked

## 1. Question

The approved P10_A smoke produced three physically identical Isaac traces, but repetition 0 had a different `reset_returned_policy_hash` and therefore a different `repeatability_key` from repetitions 1 and 2. This investigation asks whether the difference is:

1. physical reset nondeterminism;
2. RNG, command, action, or history contamination;
3. a stale observation/cache returned by the reset API.

The frozen design treats `reset_returned_policy_hash` as the authoritative first input returned to the Actor. It cannot be replaced by the post-forward physical-state hash merely to make G01 pass.

## 2. Baseline Smoke Evidence

Run:

`runs/verification-20261008-review18-isaac-p10-a-smoke`

The three repetitions had identical:

- pre-forward initial-condition hash: `BB2AE1BFEC8B2FB07F59DE2E8F2F0ECEC9AFC1926ECB2B8929F237BC3E648B39`;
- post-forward state hash: `A063A5BB86BD754598617CBF5FCFDA4D6F305969CC920F7F58AAFDC77A56A5FC`;
- excitation and profile identity;
- complete physical trace, including time, all 26 hinge positions/velocities, base motion, closure residual, commanded input, and torque-equivalent input.

The only changing identity was the returned policy state:

| repetition | `reset_returned_policy_hash` | `repeatability_key` |
|---:|---|---|
| 0 | `D01A74F826E72AC358ABEC0CD7A34F2A3E367AEC4E825E5ACC2D85588A36DD50` | `B90B4E411CE4A6F572CB95037C13215FEB9DB9333F740A66D14681FFE85EADCA` |
| 1 | `3E46EDBC5F3F7F9DC924C62547F0B3DF99D15976FB6E5B30598401A61CD4B442` | `14C4F0E161517335037A80CC4535201367F6E61A01169686D6A395AA4E1B2FF7` |
| 2 | `3E46EDBC5F3F7F9DC924C62547F0B3DF99D15976FB6E5B30598401A61CD4B442` | `14C4F0E161517335037A80CC4535201367F6E61A01169686D6A395AA4E1B2FF7` |

Artifact identities:

- baseline `result.json`: `20933E6A51E95C2C799AA1737293A3A8294F82B8DAAAFCF50B39E8928E7AB905`;
- baseline `trace.npz`: `FD42D4ADC980CB103D1E525BD82A48D63849BCA8EEAF08111BCF95A05D77B82A`.

## 3. Component Isolation

Run:

`runs/diagnostic-20261008-p10-returned-policy-components-03`

This non-evidentiary run used an in-memory wrapper around `_repeatability_returned_policy_state()` and did not modify source files. It captured the raw `<f4` actor observation and independent previous-action buffer at every identity construction.

Results:

- all four captured previous-action arrays were bitwise identical zeros;
- actor observation calls for repetition 0 were identical to one another;
- repetitions 1 and 2 were bitwise identical to one another;
- the only differing actor indices were `0:3`, the frozen `ActorObsV1.ANGULAR_VELOCITY` slice;
- all actor indices `3:25` were bitwise identical.

Exact returned values:

| repetition | returned actor indices `0:3` |
|---:|---|
| 0 | `[0.0, 0.0, 0.0]` |
| 1 | `[-1.1614577e-05, 3.3813692e-05, -1.0379885e-05]` |
| 2 | `[-1.1614577e-05, 3.3813692e-05, -1.0379885e-05]` |

For repetitions 1 and 2, those three values are bitwise equal to the preceding rollout's terminal base angular velocity multiplied by the frozen normalization scale `0.25`:

```text
terminal base angular velocity
= [-4.6458306e-05, 1.3525477e-04, -4.1519539e-05]

terminal * 0.25
= [-1.1614577e-05, 3.3813692e-05, -1.0379885e-05]
```

At the same reset sample, the direct post-forward physical angular velocity is exactly zero for all three repetitions. Therefore the changed Actor input is not a changed PhysX reset state.

The diagnostic stdout SHA256 is `6AA1F4E8B32D19E2494A829D045622A84AB8217301C057BC9B34E80CFA486387`.

## 4. Data-Flow Cause

The relevant production path is:

1. `WheelLegFlatEnv._reset_idx()` writes the nominal root velocity and joint state, then immediately calls `_read_state()` and stores that result in `self._state`.
2. `_read_state()` obtains Actor angular velocity from `robot.data.root_link_ang_vel_b`.
3. Isaac Lab's `write_root_com_velocity_to_sim()` updates `root_com_vel_w`, but it does not invalidate every derived root-link velocity cache consumed by `root_link_ang_vel_b`.
4. `DirectRLEnv.reset()` subsequently calls `scene.write_data_to_sim()` and `sim.forward()`, but does not call `scene.update(dt=physics_dt)` before `_get_observations()`.
5. `_get_observations()` calls `_current_state()`, which returns the already cached `self._state` instead of rebuilding it from refreshed articulation data.

Relevant source locations:

- production reset and state cache: `source/wheelleg_dreamwaq/wheelleg_dreamwaq/tasks/direct/wheelleg_flat/env.py:653-700,852-918`;
- upstream reset order and normal per-step cache refresh: `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/envs/direct_rl_env.py:292-331,369-410`;
- upstream root velocity write behavior: `dependencies/IsaacLab-v2.3.2/source/isaaclab/isaaclab/assets/articulation/articulation.py:485-525`;
- debug reset phase capture: `debug/sim2sim/isaac_debug_env.py:279-311`.

This is a reset observation/cache-coherency defect. It is not solver nondeterminism and it is not a changed physical trajectory.

## 5. Counterfactual Tests

Three in-memory, debug-only counterfactuals were run. None changed on-disk source.

### 5.1 Re-read `_state` only

Run:

`runs/diagnostic-20261008-p10-post-forward-state-refresh-01`

Calling `_capture_state()` after forward did not fix the identity. The underlying articulation data cache had not been invalidated.

### 5.2 `scene.update(dt=0.0)` then re-read

Run:

`runs/diagnostic-20261008-p10-scene-update-refresh-01`

This also did not fix the identity because the Isaac Lab data timestamp did not advance.

### 5.3 `scene.update(dt=physics_dt)` then re-read

Run:

`runs/diagnostic-20261008-p10-scene-update-physics-dt-01`

This made all three repetitions identical:

- returned policy hash for repetitions 0, 1, and 2: `B17F8A84A22C031DA1B9CF8C58A4C75C343B0D36F228BD4C01339DA570CEF601`;
- repeatability key for repetitions 0, 1, and 2: `12CAAA613549800365F5C6FAB476CB844F7FDE3A07979A6B3900704FD14CAA56`;
- counterfactual `result.json`: `0A684EE176FF240C2E03DCCCD31AC9EF77DF6912E1B52AD25D9008489D28BF8F`.

The complete physical trace remained byte-identical to the unmodified baseline:

`FD42D4ADC980CB103D1E525BD82A48D63849BCA8EEAF08111BCF95A05D77B82A`

This positive control and the two negative controls show that advancing the Isaac Lab scene-data timestamp and rebuilding the task state are jointly necessary and sufficient for this observed identity split.

## 6. Formal Environment Confirmation

Run:

`runs/diagnostic-20261008-formal-reset-observation-01`

The suite's existing `instrumentation-probe --instrumentation-mode formal` instantiated `WheelLegFlatEnv`, not the debug environment. Across five resets under `NominalEvaluationProfileV1`:

- repetition 0 returned actor angular velocity `[0.0, 0.0, 0.0]`;
- repetition 1 returned `[0.00020298079, 0.26674849, 0.00031355838]`;
- later repetitions also returned nonzero angular velocity despite nominal reset root velocity being zero;
- each returned Actor angular-velocity slice equaled the contemporaneous cached `robot.data.root_link_ang_vel_b` transformed to ControlFrame and multiplied by `0.25`.

The formal probe kept the frozen config unchanged:

`formal_config_hash_before == formal_config_hash_after == AD92A93B61E3436DB443081E2A521926B73944E13B017FEF4F656C320100A02`

Formal trace SHA256: `89E7A7040004586051A40C2DE04CAB80F1ED81BF4DCB0265E4AD3EB3F4D79DF7`.

Therefore the defect is present on the production `WheelLegFlatEnv` reset API path and is not introduced by the debug observer.

## 7. Classification and Limits

### Proven

1. Repeated Isaac resets can return an Actor observation whose angular-velocity slice comes from stale articulation/task cache state rather than the post-forward reset state.
2. The stale slice changes the initialized five-frame history because the history is defined as five copies of the returned current observation.
3. The production `WheelLegFlatEnv` path is affected.
4. The current G01 P10_A key split is fully explained by this cache-coherency defect.

### Not proven

1. This is not yet certified as the sole cause of the observed first-episode Isaac-to-MuJoCo failure.
2. No P10-P60 physical or C70 checkpoint verdict is valid while the reset/history hard gate is failing.
3. The current DreamWaQ checkpoint may have learned under cross-episode reset-observation contamination, but this run does not quantify how much that changed the trained policy.

### Frozen-design outcome

Per `RootCauseSuiteCoreV1.17` section 7.3, a history/observation hard-gate failure is classified as:

`SIM2SIM_ADAPTER_BUG`

The narrower sub-cause supported by this investigation is:

`isaac_reset_returned_observation_stale_root_angular_velocity_cache`

The full Core V1 run must remain blocked. Continuing by dropping `reset_returned_policy_hash`, substituting the post-forward physical hash, or silently refreshing only the diagnostic worker would violate the frozen identity contract and could produce a false physical/checkpoint verdict.

## 8. Required Next Decision

The minimal production correction would need to make Isaac Lab articulation data current after reset writes/forward and then rebuild `self._state` before the reset observation is returned. That change belongs to the formal environment path and requires its own tests, code review, checkpoint compatibility decision, and retraining decision.

This investigation does not apply that correction because the current scope explicitly forbids modifying formal training code.
