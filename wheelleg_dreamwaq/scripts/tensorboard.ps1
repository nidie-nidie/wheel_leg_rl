param(
    [string]$LogDir = "",
    [int]$Port = 6006
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $LogDir) {
    $LogDir = Join-Path $ProjectRoot "logs\rsl_rl\wheelleg_flat_ppo"
}
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $Python -m tensorboard.main --logdir $LogDir --host 127.0.0.1 --port $Port
