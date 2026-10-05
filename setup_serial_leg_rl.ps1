$ErrorActionPreference = "Stop"

$Workspace = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
$VenvPath = Join-Path $Workspace ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$CacheRoot = "E:\pip-cache\serial_leg_rl"
$TorchWheel = Join-Path $CacheRoot "torch-2.5.1+cu118-cp310-cp310-win_amd64.whl"
$IsaacLabRoot = Join-Path $Workspace "wheel_leg\WheelLeg_RL_IsaacLab\IsaacLab"

function Invoke-Checked {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    Write-Host ("[RUN] " + $Executable + " " + ($Arguments -join " "))
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable"
    }
}

function Get-Python310 {
    $candidates = @(
        (Get-Command python3.10.exe -ErrorAction SilentlyContinue).Source,
        (Get-Command python3.10 -ErrorAction SilentlyContinue).Source,
        "C:\Users\changba01\Anaconda3\envs\aidog\python.exe",
        "C:\Users\changba01\AppData\Local\Programs\Python\Python310\python.exe",
        "C:\Python310\python.exe"
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique

    foreach ($candidate in $candidates) {
        $version = (& $candidate -c "import sys; print('.'.join(map(str, sys.version_info[:2])))").Trim()
        if ($version -eq "3.10") {
            return $candidate
        }
    }
    throw "Python 3.10 was not found. Install Python 3.10 or make the existing 3.10 interpreter available."
}

if (-not (Test-Path -LiteralPath (Join-Path $IsaacLabRoot "isaaclab.bat"))) {
    throw "Isaac Lab checkout not found under $IsaacLabRoot"
}

New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null
$env:PIP_CACHE_DIR = $CacheRoot
$env:TEMP = Join-Path $CacheRoot "tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$env:OMNI_KIT_ACCEPT_EULA = "YES"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $BootstrapPython = Get-Python310
    Write-Host "[INFO] Creating virtual environment with $BootstrapPython"
    Invoke-Checked $BootstrapPython @("-m", "venv", $VenvPath)
}

$venvVersion = (& $VenvPython -c "import sys; print('.'.join(map(str, sys.version_info[:2])))").Trim()
if ($venvVersion -ne "3.10") {
    throw ".venv uses Python $venvVersion; expected Python 3.10. Remove .venv and rerun setup if it was created from the wrong interpreter."
}

Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
if ((Test-Path -LiteralPath $TorchWheel) -and ((Get-Item -LiteralPath $TorchWheel).Length -gt 2GB)) {
    Write-Host "[INFO] Installing cached Torch wheel: $TorchWheel"
    Invoke-Checked $VenvPython @("-m", "pip", "install", $TorchWheel)
}
else {
    Invoke-Checked $VenvPython @("-m", "pip", "install", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu118")
}
Invoke-Checked $VenvPython @("-m", "pip", "install", "isaacsim[all]==4.5.0", "--extra-index-url", "https://pypi.nvidia.com")

Invoke-Checked $VenvPython @("-m", "pip", "install", "--no-build-isolation", "--editable", (Join-Path $IsaacLabRoot "source\isaaclab"))
Invoke-Checked $VenvPython @("-m", "pip", "install", "--no-build-isolation", "--editable", (Join-Path $IsaacLabRoot "source\isaaclab_assets"))
Invoke-Checked $VenvPython @("-m", "pip", "install", "--no-build-isolation", "--editable", (Join-Path $IsaacLabRoot "source\isaaclab_tasks"))
$IsaacLabRlSpec = (Join-Path $IsaacLabRoot "source\isaaclab_rl") + "[rsl-rl]"
Invoke-Checked $VenvPython @("-m", "pip", "install", "--no-build-isolation", "--editable", $IsaacLabRlSpec)
Invoke-Checked $VenvPython @("-m", "pip", "install", "--no-build-isolation", "--editable", (Join-Path $Workspace "serial_leg_rl"))

Write-Host "[INFO] Installation completed. Run check_serial_leg_rl_env.ps1 before training."
