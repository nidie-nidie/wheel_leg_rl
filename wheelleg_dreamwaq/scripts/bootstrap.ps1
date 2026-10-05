$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$IsaacLabRoot = Join-Path $ProjectRoot "dependencies\IsaacLab-v2.3.2"
$ExpectedCommit = "37ddf626871758333d6ed89cf64ad702aef127d0"

if (-not (Test-Path -LiteralPath (Join-Path $IsaacLabRoot ".git"))) {
    git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git $IsaacLabRoot
}

$ActualCommit = (git -C $IsaacLabRoot rev-parse HEAD).Trim()
if ($ActualCommit -ne $ExpectedCommit) {
    throw "Isaac Lab commit mismatch: expected $ExpectedCommit, got $ActualCommit"
}

Push-Location $ProjectRoot
try {
    uv sync --all-groups
    uv run python scripts/check_runtime.py
}
finally {
    Pop-Location
}
