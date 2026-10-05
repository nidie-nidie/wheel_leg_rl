@echo off
setlocal EnableExtensions

set "WORKSPACE=%~dp0"
set "PYTHON=%WORKSPACE%.venv\Scripts\python.exe"
set "ISAACLAB=%WORKSPACE%wheel_leg\WheelLeg_RL_IsaacLab\IsaacLab\isaaclab.bat"
if not exist "%PYTHON%" (
    echo [ERROR] Missing %PYTHON%
    echo [ERROR] Run setup_serial_leg_rl.ps1 first.
    exit /b 1
)
if not exist "%ISAACLAB%" (
    echo [ERROR] Missing %ISAACLAB%
    exit /b 1
)

set "VIRTUAL_ENV=%WORKSPACE%.venv"
set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"
set "PYTHONPATH=%VIRTUAL_ENV%\Lib\site-packages;%PYTHONPATH%"
set "PYTHONEXECUTABLE=%PYTHON%"
set "PYTHONHOME="
set "CONDA_PREFIX="
set "CONDA_DEFAULT_ENV="
set "CONDA_PROMPT_MODIFIER="
set "CONDA_EXE="
set "CONDA_PYTHON_EXE="
set "CONDA_SHLVL="
set "_CONDA_EXE="
set "_CONDA_ROOT="
set "PYTHONNOUSERSITE=1"
set "OMNI_KIT_ACCEPT_EULA=YES"

pushd "%WORKSPACE%"
call "%ISAACLAB%" -p "serial_leg_rl\scripts\play_standing_rsl_rl.py" --task SerialLeg-Standing-Direct-v0 --load_run 2026-09-03_19-41-23 --checkpoint model_15998.pt --num_envs 1 --num_steps 3000 %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
