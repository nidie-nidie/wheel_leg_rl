# Serial Leg RL

Clean research scaffold for the wheel-leg robot described by
`../wheel_leg_urdf4/urdf/wheel_leg_urdf4.urdf`.

This project starts with model and kinematics validation before any RL
training. The original URDF and the original kinematics HTML are treated as
read-only source material.

## Phase 0 Scope

- keep the original URDF untouched
- document the robot, actuators, actions, observations, and experiment plan
- provide an offline URDF audit script
- provide an offline offset closed-chain kinematics module
- test FK, IK, and Jacobian consistency without Isaac Sim

## Later Phases

- convert the URDF into simulator assets under `assets/generated/`
- build a Manager-Based Isaac Lab standing task
- train Pure Joint PPO first
- add asymmetric actor-critic and CTS variants after the baseline works
- export the final Student policy
- add MuJoCo Sim2Sim after a usable policy exists

## Quick Checks

From this repository root:

```bash
python3 serial_leg_rl/scripts/inspect_urdf.py
PYTHONPATH=serial_leg_rl/source/serial_leg_rl python3 serial_leg_rl/scripts/check_kinematics_geometry.py
PYTHONPATH=serial_leg_rl/source/serial_leg_rl python3 serial_leg_rl/scripts/phase0_report.py
PYTHONPATH=serial_leg_rl/source/serial_leg_rl python3 -m unittest discover serial_leg_rl/tests
```
