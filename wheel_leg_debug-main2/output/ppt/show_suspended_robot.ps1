$ErrorActionPreference = 'Stop'

$modelWindows = Join-Path $PSScriptRoot 'assets_v2\mujoco_suspended.xml'
if (-not (Test-Path -LiteralPath $modelWindows)) {
    throw "Model file not found: $modelWindows"
}

$modelUnixInput = $modelWindows.Replace('\', '/')
$modelWsl = (& wsl.exe wslpath -a $modelUnixInput).Trim()
$simulate = '/home/shun/MuJoCoBin/mujoco-3.3.0/bin/simulate'

Write-Output "Opening: $modelWsl"
& wsl.exe $simulate $modelWsl
