param([string]$Python = "py")
$ErrorActionPreference = "Stop"
$PaceRoot = Split-Path $PSScriptRoot -Parent
$PaceVenv = Join-Path $PaceRoot ".venv"
$PacePython = Join-Path $PaceVenv "Scripts\python.exe"
if (!(Test-Path -LiteralPath $PacePython)) {
    if ($Python -eq "py") { & $Python -3.11 -m venv $PaceVenv }
    else { & $Python -m venv $PaceVenv }
    if ($LASTEXITCODE -ne 0) { throw "Could not create the isolated PACE Python environment." }
}
& $PacePython -c "import sys; assert sys.version_info[:2] == (3, 11), 'Use Python 3.11 for requirements.lock.txt'"
if ($LASTEXITCODE -ne 0) { throw "PACE requires Python 3.11." }
& $PacePython -m pip install -r (Join-Path $PaceRoot "requirements.lock.txt")
if ($LASTEXITCODE -ne 0) { throw "PACE dependency installation failed." }
Write-Output "PACE Python: $PacePython"
