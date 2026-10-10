# WheelLeg Pitch-Torque Playback Design

## Scope

Add an interactive pitch-disturbance control to native Isaac Sim PPO playback without changing the environment, training contract, checkpoint, USD, collision policy, observations, actions, rewards, or actuator configuration.

## Behavior

- The feature is available only with `scripts/play.py --keyboard`.
- Holding `J` applies `+4.0 Nm`; holding `L` applies `-4.0 Nm`.
- Releasing the key immediately clears its contribution. Pressing both keys produces zero net torque.
- `Space`, `R`, `Esc`, and playback cleanup clear the disturbance state.
- `--pitch-torque-nm` may override the positive magnitude and defaults to `4.0`.
- Keyboard playback prints live diagnostics at 5 Hz in canonical six-joint order: raw action, action clipping and limit masks, applied actuator torque and effort-limit mask, leg position and soft-limit mask, and reset state. `P` toggles this output.
- No duration limit is imposed.

## Physics Semantics

- Resolve the unique `base_link` articulation body once after environment creation.
- Apply a pure local-frame torque with no external force and no application-position offset.
- The torque vector is `[T, 0, 0]` in the USD `base_link` frame. USD local `+X` maps to ControlFrameV1 `+Y`, so it is a pitch disturbance.
- Write the wrench before `env.step()`. Isaac Lab then writes the existing joint actuator commands and the external wrench during each decimated physics step.
- The disturbance adds to the rigid-body equations of motion; it does not replace policy actions, joint targets, or actuator torques.

## Safety And Cleanup

- Reject a non-positive `--pitch-torque-nm` value.
- Clear the wrench on reset and before closing playback.
- Keep the existing action clipping, actuator limits, terminations, and checkpoint validation unchanged.

## Verification

- Parse/compile `scripts/play.py`.
- Run the existing unit tests.
- Run bounded headless playback to verify non-keyboard behavior is unchanged.
- Start native keyboard playback and verify `J`/`L` press and release update and clear the external torque without disabling policy control.
