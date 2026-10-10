# Cross-Engine Pitch Torque Experiment Plan

> **Scope:** Keep every code change inside `debug/sim2sim/`. Do not modify the
> training environment, Actor, checkpoint, USD, formal MJCF, or deployment code.

**Goal:** Determine whether Isaac Sim and MuJoCo produce a materially different
early whole-body response to the same body-local pitch-torque pulse while the
six policy actions and commands are held at zero.

**Experiment contract:** Use the frozen run-01 Actor contract and nominal reset,
50 Hz control, each engine's frozen physics step, and a signed pitch torque in
the shared control frame. The pulse interval is end-exclusive. Record the
applied torque, pitch response, virtual-leg state, wheel state, actuator torque,
and wheel-ground contact force at every physics substep.

## Tasks

- [x] Add one shared, validated pitch-torque pulse definition.
- [x] Add regression tests for pulse timing and control-frame direction.
- [x] Apply the pulse to MuJoCo through `xfrc_applied` at the base COM, rotating
      the body-local control-frame torque into the current engine world frame.
- [x] Apply the same pulse to Isaac through the articulation permanent-wrench
      composer in body-local coordinates.
- [x] Repair the debug-only Isaac wheel contact sensors, record world-frame
      normal force, and explicitly mark unavailable friction without changing
      the formal environment.
- [x] Verify sensor-off and sensor-on Isaac trajectories remain equivalent
      before using instrumented contact data.
- [x] Collect zero, `+4 Nm`, and `-4 Nm` zero-action runs in both engines.
- [x] Compare the first 100-200 ms after pulse onset and record the evidence,
      interpretation limits, and next decision in this document.

## Decision Rule

The experiment does not tune MuJoCo to stand. It only checks for a gross model
or adapter error. If torque direction, early angular response, wheel reaction,
and contact-force scale are coherent, further simulator fitting stops and the
remaining failure is treated primarily as policy robustness. If a sign error,
missing reaction, or order-of-magnitude contact anomaly appears, that specific
model boundary is investigated before retraining.

## Execution Record

Completed on 2026-10-06. All source changes are under `debug/sim2sim/`; the
training environment, Actor, checkpoint, USD, and formal MJCF were not changed.

### Instrumentation

- `BasePitchTorquePulse` defines one signed, end-exclusive control-tick pulse.
- MuJoCo applies the torque at the base COM with `data.xfrc_applied` after
  rotating the ControlFrameV1 body torque into the current world frame.
- Isaac applies the same body-local torque through the articulation permanent
  wrench composer. Control-frame `+pitch` maps to the USD base body's local
  `+x` axis, consistent with the existing interactive perturbation path.
- Both collectors record the commanded control-frame torque, engine-world
  torque, pitch state, virtual-leg state, wheel state, PD effort estimate, and
  wheel normal contact force/impulse at every physics substep.
- Isaac GPU PhysX does not support the attempted collider-filtered contact
  sensor against `/World/Ground`. The debug collector therefore uses each
  wheel sensor's unfiltered `net_forces_w`. Isaac Lab defines this as the net
  normal-force vector, excluding friction. Its vector norm is the normal-force
  magnitude. This is valid for this frozen task because only the two wheel
  collisions are enabled. GPU friction force is unavailable and stored as
  `NaN`, not zero.

### Observer Equivalence Gate

The same zero-action reset was collected for 10 control ticks with the Isaac
contact observer disabled and enabled:

- `artifacts/debug/sim2sim/pitch-torque-sensor-gate-off-v3`
- `artifacts/debug/sim2sim/pitch-torque-sensor-gate-on-v3`

The maximum absolute difference was exactly zero for active joint position and
velocity, base COM position, base linear and angular velocity, projected
gravity, virtual-leg length, `phi0`, and the host-side Isaac PD torque estimate.
The contact observer is therefore non-perturbing for this deterministic gate.

### Matched Experiment

Artifact root:
`artifacts/debug/sim2sim/pitch-torque-response-20261006-v3`

Only `v3` is authoritative. The `v1` and `v2` directories are retained as
intermediate debug evidence; `v2` used an incorrect temporary interpretation
of `net_forces_w` and must not be used for contact-force conclusions.

The six runs use zero ActionV1, command `[0, 0, 0.20]`, 20 control ticks, and a
`+4 Nm`, `0 Nm`, or `-4 Nm` pitch torque during ticks `[5, 10)`, corresponding
to `0.10-0.20 s`. The analyzer uses the odd response
`(positive - negative) / 2` so reset bias and the common nonlinear trajectory
do not masquerade as torque response.

Post-collection verification found exactly zero action error and exactly zero
control-frame torque error in all six traces. Engine-world torque norm error
was at most `2.39e-7 Nm` in Isaac and `7.11e-15 Nm` in MuJoCo.

| Time after onset | Engine | Pitch | Pitch rate | Forward velocity | Left/right phi0 |
| --- | --- | ---: | ---: | ---: | ---: |
| 20 ms | Isaac | 0.5689 deg | 0.5919 rad/s | 0.0906 m/s | 0.1762 / 0.1731 deg |
| 20 ms | MuJoCo | 0.5502 deg | 0.7604 rad/s | 0.0880 m/s | 0.2198 / 0.2193 deg |
| 100 ms | Isaac | 7.8870 deg | 2.1621 rad/s | 0.4385 m/s | 0.5105 / 0.4861 deg |
| 100 ms | MuJoCo | 8.1404 deg | 2.6424 rad/s | 0.4855 m/s | 0.6227 / 0.6209 deg |
| 200 ms | Isaac | 19.5401 deg | 2.1352 rad/s | 0.4116 m/s | -0.1452 / -0.1251 deg |
| 200 ms | MuJoCo | 24.0641 deg | 3.3985 rad/s | 0.6405 m/s | -0.0036 / -0.0035 deg |

During the pulse, the zero-torque wheel normal impulses were
`[2.1451, 2.1663] Ns` in Isaac and `[2.1507, 2.1447] Ns` in MuJoCo. The odd
normal-impulse response was `[-0.1716, -0.1424] Ns` in Isaac and
`[-0.1895, -0.1890] Ns` in MuJoCo. Wheel velocity also agreed closely at
20 ms: Isaac `[-0.1908, -0.1834] rad/s`, MuJoCo
`[-0.1930, -0.1926] rad/s`.

The `+4 Nm` trajectory reaches the tilt termination in both engines; MuJoCo
stops after 17 ticks and Isaac after the final twentieth tick. The 200 ms row
is consequently a large-angle, nonlinear fall regime and is not a calibration
target. The 5 ms substep transient also differs because the solvers integrate
contact differently; the policy observes at 20 ms intervals.

### Decision

The experiment found no pitch-torque sign error, missing wheel reaction,
`phi0` ordering error, or order-of-magnitude normal-contact discrepancy. At
20-100 ms the pitch angle, forward velocity, wheel velocity, and normal impulse
are coherent. MuJoCo's pitch-rate and `phi0` response is approximately
20-25 percent stronger, which is a real residual dynamics/solver difference,
but not evidence that the adapter chain is broken. Tangential friction was not
directly comparable because the Isaac GPU sensor path cannot expose it with
the required ground filter, so this experiment does not prove full contact
equivalence.

Combined with the previous same-action replay, the current MuJoCo standing
failure is consistent with insufficient robustness of the frozen run-01
checkpoint to a moderate cross-engine dynamics/contact gap. Do not tune one
MuJoCo parameter set merely until this checkpoint stands. The next training
iteration should cover the measured gap with bounded dynamics randomization
and pitch disturbances, then repeat the frozen sim2sim evaluation. If the new
policy still fails, the next diagnostic should obtain directly comparable
tangential contact/slip evidence before changing the formal model.

Machine-readable results and input hashes are in
`artifacts/debug/sim2sim/pitch-torque-response-20261006-v3/analysis/`.
