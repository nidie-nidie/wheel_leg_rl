@echo off
setlocal EnableExtensions

set "WORKSPACE=%~dp0"
set "PYTHON=%WORKSPACE%.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [ERROR] Missing %PYTHON%
    echo [ERROR] Run setup_serial_leg_rl.ps1 first.
    exit /b 1
)

set "VIRTUAL_ENV=%WORKSPACE%.venv"
set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"
set "PYTHONNOUSERSITE=1"
set "CONDA_PREFIX="
set "CONDA_DEFAULT_ENV="
set "CONDA_EXE="
set "CONDA_PYTHON_EXE="
set "CONDA_SHLVL="
set "_CONDA_EXE="
set "_CONDA_ROOT="

pushd "%WORKSPACE%"
"%PYTHON%" -c "import sys; from pip._vendor import pkg_resources; sys.modules['pkg_resources']=pkg_resources; from tensorboard.main import run_main; run_main()" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
