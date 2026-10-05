# MuJoCo Stand Tuning Log

Scope: tune only `mujoco_control_extract` and root `MJCF`; do not edit `rm_test_dev`.

## Goal

- Stand mode keeps the white thin l0 auxiliary line close to parallel with the world z axis.
- The robot stays near one point instead of drifting or falling.
- Only the wheel group should maintain body balance; leg joints should hold the virtual-leg geometry, not pin the robot through ground contact.

## Current Reference

- Command baseline:
  ```bash
  ./mujoco_control_extract/build/mujoco_bridge --headless --ground-init --debug-leg-axis --leg-length 0.20 --init-phi0 2.75 --time 6 MJCF/env.xml
  ```
- Initial geometry is good with `leg-length=0.20` and `init-phi0=2.75`: l0 axis starts nearly vertical, about 0.2 deg from z in the x-z projection.
- After switching to `SAFE`, current stand tuning still falls backward/sideways, so the wheel balancing loop and/or contact setup still needs work.

## Observations

- `--zero-wheels` still falls, which is expected for a two-wheel inverted-pendulum pose, but useful diagnostically: joint holding alone should not be treated as balance.
- Root `MJCF/robot.xml` currently allows several non-wheel link meshes to collide with the floor (`class="touch"`): `GH/OP`, `BE/EC/CF`, `JM/MK/KN`, plus wheels and base.
- If any lower links touch the ground during stand, the controller is fighting contact constraints instead of clean wheel-only balancing.
- I changed the non-wheel leg meshes in root `MJCF/robot.xml` to `class="untouch"` so stand tuning uses wheel-only floor contact.
- The visual-aligned reversed joint map gives a good initial white l0-axis pose, but the stand dynamics still tip over after torque control starts.
- The original `rm_test_dev`-style joint map can be selected with `--original-joint-map`; however, with the current XML geometry and initial-pose solver it starts with the two sides badly mismatched, so it is not yet usable as-is.
- The strongest suspicion now is not a simple PID gain issue. The MuJoCo XML joint/sign/mirror convention and the controller's front/rear torque convention still need to be reconciled before stand gains will tune cleanly.

## Added Tuning Knobs

`mujoco_bridge` now accepts MuJoCo-only stand override gains:

```bash
--stand-joint-kp <value>
--stand-joint-kd <value>
--stand-joint-limit <value>
--stand-pitch-kp <value>
--stand-pitch-kd <value>
--stand-pos-kp <value>
--stand-vel-kd <value>
--stand-wheel-limit <value>
```

This lets us sweep stand parameters without rebuilding every time.

## Useful Commands

Visual-aligned map, current stand override:

```bash
./mujoco_control_extract/build/mujoco_bridge --ground-init --debug-leg-axis --leg-length 0.20 --init-phi0 2.75 MJCF/env.xml
```

Original RM-style map comparison:

```bash
./mujoco_control_extract/build/mujoco_bridge --ground-init --debug-leg-axis --original-joint-map --leg-length 0.20 --init-phi0 2.75 MJCF/env.xml
```

Disable the MuJoCo stand override and use inner controller output directly:

```bash
./mujoco_control_extract/build/mujoco_bridge --ground-init --debug-leg-axis --no-stand-override MJCF/env.xml
```
