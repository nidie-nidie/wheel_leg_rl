# WheelLeg Keyboard Playback Design

## Scope

Add native Isaac Sim keyboard control to Phase 1 PPO playback without changing the environment, policy, observations, rewards, USD, or checkpoints.

## Design

- Add a project-local Isaac Sim Experience copied from the known-working Isaac Lab GUI Experience with only `isaacsim.sensors.rtx` removed.
- Select that Experience before `AppLauncher` starts.
- Add an optional `--keyboard` mode to `scripts/play.py`.
- Map `W/S` to X velocity, `A/D` to yaw rate, `Q/E` to base height, `Space` to zero planar commands, `R` to reset, and `Esc` to exit.
- Clamp every command to the training ranges and overwrite the environment command after every step so command resampling cannot replace keyboard input.
- Require one environment and a native GUI for keyboard mode; preserve fixed-command and video playback behavior.

## Verification

- Confirm the custom Experience does not depend on `isaacsim.sensors.rtx`.
- Run existing unit tests.
- Start native GUI playback for a bounded number of steps and confirm clean startup and shutdown.
