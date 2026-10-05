$ErrorActionPreference = "Stop"

$Workspace = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
$Python = Join-Path $Workspace ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python. Run setup_serial_leg_rl.ps1 first."
}

$env:VIRTUAL_ENV = Join-Path $Workspace ".venv"
$env:PATH = (Join-Path $env:VIRTUAL_ENV "Scripts") + ";" + $env:PATH
$env:PYTHONPATH = (Join-Path $env:VIRTUAL_ENV "Lib\site-packages") + ";" + $env:PYTHONPATH
$env:CONDA_PREFIX = $null
$env:CONDA_DEFAULT_ENV = $null
$env:CONDA_PROMPT_MODIFIER = $null
$env:CONDA_EXE = $null
$env:CONDA_PYTHON_EXE = $null
$env:CONDA_SHLVL = $null
$env:_CONDA_EXE = $null
$env:_CONDA_ROOT = $null
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:PYTHONNOUSERSITE = "1"

Push-Location $Workspace
try {
    & $Python (Join-Path $Workspace "check_serial_leg_rl_env.py")
    if ($LASTEXITCODE -ne 0) { throw "Environment verification failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
