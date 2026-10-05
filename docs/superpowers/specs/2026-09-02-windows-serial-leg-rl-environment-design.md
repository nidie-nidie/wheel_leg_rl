# Windows Serial Leg RL Environment Design

## Goal

Provide a reproducible native-Windows Python environment for the `serial_leg_rl` Isaac Lab task, using the workspace USD asset and a single command to start training.

## Constraints

- Workspace: `E:/wheel_leg_rl-main`.
- Python: 3.10, because the checked-in Isaac Sim 4.5 and Isaac Lab 2.0.1 stack requires it.
- Isaac Sim: `4.5.0` with the Windows pip package and cached extensions.
- Isaac Lab: the checked-in `wheel_leg/WheelLeg_RL_IsaacLab/IsaacLab` source tree, version `2.0.1`.
- PyTorch: `2.5.1` CUDA 11.8 build, matching the checked-in Isaac Lab setup metadata.
- USD: `wheel_leg_urdf4_usd (1)/wheel_leg_urdf4/wheel_leg_urdf4.usd`, already referenced by the serial-leg task.
- Installation must not modify the original URDF or USD sources.

## Architecture

The environment is a standard workspace-local `.venv` created with Python 3.10. Isaac Sim and Isaac Lab packages are installed into this environment; pip cache and temporary files are redirected to an E-drive cache directory. The Isaac Lab Windows launcher is adjusted only as needed to prefer an active `VIRTUAL_ENV`, so `isaaclab.bat -p` invokes the same interpreter that owns the installed packages.

The setup entry point checks Python, disk, package installation, and task registration before reporting success. The training entry point resolves its own workspace root, invokes the checked-in Isaac Lab launcher, runs `train_standing_rsl_rl.py`, defaults to headless operation and a conservative environment count, and forwards extra command-line arguments.

The generated flat-terrain USD is stored in a workspace-local cache instead of the POSIX `/tmp` path currently used by the source. The robot USD remains the checked-in asset path.

## Error Handling

- Stop before installation if Python 3.10 cannot be located.
- Stop with an actionable message if Isaac Sim 4.5 or the required Isaac Lab/RSL-RL imports are missing.
- Stop if the serial-leg task is not registered.
- Keep installation idempotent: rerunning setup upgrades/repairs packages without deleting the environment.
- Do not claim that training is ready until the one-environment, one-iteration headless smoke test exits successfully.

## Verification

1. `.venv/Scripts/python.exe` reports Python 3.10 and `sys.prefix != sys.base_prefix`.
2. `import isaacsim`, `import isaaclab`, `import isaaclab_rl`, and `import rsl_rl` succeed.
3. Gym registration resolves `SerialLeg-Standing-Direct-v0`.
4. Isaac Lab launches the serial-leg script with `--num_envs 1 --max_iterations 1 --headless` and exits with code 0.
5. The final documented command starts the regular training run and writes logs under `logs/rsl_rl/serial_leg_standing_direct`.
