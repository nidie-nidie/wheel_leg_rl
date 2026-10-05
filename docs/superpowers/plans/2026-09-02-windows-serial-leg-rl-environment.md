# Windows Serial Leg RL Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install the pinned Isaac Sim/Isaac Lab stack into a workspace-local Windows Python 3.10 environment and provide a one-command serial-leg training entry point.

**Architecture:** Use `E:/wheel_leg_rl-main/.venv` as a standard Python venv. Install Isaac Sim 4.5.0 from NVIDIA's index, the checked-in Isaac Lab 2.0.1 source extensions, RSL-RL, and the editable `serial_leg_rl` package. A Windows PowerShell setup script owns prerequisites and installation; a batch training wrapper activates the venv, sets Isaac runtime variables, and calls the existing training script through `isaaclab.bat`.

**Tech Stack:** Windows PowerShell, Python 3.10, Isaac Sim 4.5.0, Isaac Lab 2.0.1, PyTorch 2.5.1 CUDA 11.8, Gymnasium, RSL-RL.

---

### Task 1: Make the serial-leg asset paths Windows-safe

**Files:**
- Modify: `serial_leg_rl/source/serial_leg_rl/serial_leg_rl/assets/sim_urdf.py`

- [ ] **Step 1: Replace the POSIX-only generated asset directory**

Use a workspace-relative path derived from `PROJECT_ROOT`, preserving the existing generated file names and public constants:

```python
GENERATED_ASSET_DIR = PROJECT_ROOT / "serial_leg_rl" / "assets" / "generated" / "runtime"
```

- [ ] **Step 2: Run the offline asset preparation check**

Run: `C:\Users\changba01\Anaconda3\envs\aidog\python.exe serial_leg_rl\scripts\prepare_isaac_assets.py`
Expected: two generated paths are printed below `serial_leg_rl/assets/generated/runtime` and no traceback occurs.

### Task 2: Add native Windows setup and training wrappers

**Files:**
- Create: `setup_serial_leg_rl.ps1`
- Create: `train_serial_leg_rl.bat`
- Create: `check_serial_leg_rl_env.ps1`

- [ ] **Step 1: Add the idempotent setup script**

The script must locate `aidog`'s Python 3.10 when no standalone `python3.10.exe` is on PATH, create `.venv`, set `PIP_CACHE_DIR` and `TEMP`/`TMP` to `E:\pip-cache\serial_leg_rl`, install `torch==2.5.1` from the CUDA 11.8 index, install `isaacsim[all,extscache]==4.5.0` from `https://pypi.nvidia.com`, install the three Isaac Lab source extensions plus `isaaclab_rl[rsl-rl]`, and install `serial_leg_rl` editable.

- [ ] **Step 2: Add the environment check script**

It must print the interpreter path/version, import `isaacsim`, `isaaclab`, `isaaclab_rl`, and `rsl_rl`, then resolve `gym.spec('SerialLeg-Standing-Direct-v0')`; exit nonzero on any failure.

- [ ] **Step 3: Add the one-command training wrapper**

The batch file must resolve its own directory, require `.venv\Scripts\python.exe`, set `OMNI_KIT_ACCEPT_EULA=YES`, change to the workspace root, and invoke `wheel_leg\WheelLeg_RL_IsaacLab\IsaacLab\isaaclab.bat -p serial_leg_rl\scripts\train_standing_rsl_rl.py --task SerialLeg-Standing-Direct-v0 --headless --num_envs 512`, forwarding user arguments after the defaults.

### Task 3: Ensure Isaac Lab launcher uses the active venv

**Files:**
- Modify: `wheel_leg/WheelLeg_RL_IsaacLab/IsaacLab/isaaclab.bat`

- [ ] **Step 1: Prefer `%VIRTUAL_ENV%\Scripts\python.exe` in `:extract_python_exe`**

The resolution order must be active standard venv, active Conda environment, `_isaac_sim\python.bat`, then the existing pip fallback. Keep all existing error messages and batch quoting intact.

- [ ] **Step 2: Check batch syntax and help output**

Run: `cmd /c wheel_leg\WheelLeg_RL_IsaacLab\IsaacLab\isaaclab.bat --help`
Expected: usage text and exit code 0.

### Task 4: Install and verify the environment

**Files:**
- Modify: `serial_leg_rl/docs/TRAINING.md`
- Modify: `serial_leg_rl/docs/train.md`

- [ ] **Step 1: Run setup**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_serial_leg_rl.ps1`
Expected: `.venv\Scripts\python.exe` is created and all package installation commands exit 0.

- [ ] **Step 2: Run package/task verification**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File .\check_serial_leg_rl_env.ps1`
Expected: Python 3.10, all four imports pass, and task id is `SerialLeg-Standing-Direct-v0`.

- [ ] **Step 3: Run the Isaac smoke test**

Run: `cmd /c train_serial_leg_rl.bat --max_iterations 1 --num_envs 1`
Expected: Isaac Sim starts headless, the task is created, one PPO iteration completes, and the process exits 0.

- [ ] **Step 4: Document the final commands**

Document setup, environment check, smoke test, and regular training in both training guides using Windows paths and the workspace-local wrappers.
