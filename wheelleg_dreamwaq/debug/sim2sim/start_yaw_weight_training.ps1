param([switch]$Validation)
$ErrorActionPreference = 'Stop'
$projectDirectory = 'E:\wheel_leg_rl-main\wheelleg_dreamwaq'
$pythonExecutable = Join-Path $projectDirectory '.venv\Scripts\python.exe'
$taskProfile = 'stop_reverse_raw_split_straight_yaw1_headingoff_v1'
$runLabel = 'baseline-yaw1-headingoff-env6144-fresh1000-20261017'
$launchLabel = 'baseline-yaw1-headingoff-env6144-fresh1000-seed41882943-20261017-launch'
$environmentCount = 6144
$targetIterations = 1000
if ($Validation) {
    $runLabel = 'validation-yaw1-headingoff-env8-update1-20261017-v2'
    $launchLabel = $runLabel + '-launch'
    $environmentCount = 8
    $targetIterations = 1
}
$launchDirectory = Join-Path $projectDirectory ('debug\sim2sim\runs\' + $launchLabel)
if (Test-Path -LiteralPath (Join-Path $launchDirectory 'launch.json')) { throw '启动记录已存在，禁止重复启动。' }
$existing = @(Get-ChildItem -LiteralPath (Join-Path $projectDirectory 'logs\rsl_rl\wheelleg_flat_dreamwaq') -Directory | Where-Object { $_.Name.EndsWith('_' + $runLabel) })
if ($existing.Count -ne 0) { throw '目标运行目录已存在，禁止重复启动。' }
$active = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -like ('*' + $runLabel + '*') })
if ($active.Count -ne 0) { throw '同名训练进程已存在。' }
New-Item -ItemType Directory -Path $launchDirectory -Force | Out-Null
$previousLaunch = Join-Path $projectDirectory 'debug\sim2sim\runs\baseline-headingoff-env6144-fresh1000-seed41882943-20261017-launch'
$previousFrozen = Get-Content -Raw -LiteralPath (Join-Path $previousLaunch 'source_before.json') | ConvertFrom-Json
$additional = @(
    (Join-Path $projectDirectory 'tests\unit\test_yaw_weight_profile.py'),
    (Join-Path $projectDirectory 'debug\sim2sim\2026-10-17-yaw-weight-one-training.md'),
    (Join-Path $projectDirectory 'debug\sim2sim\inspect_yaw_weight_training.py'),
    (Join-Path $projectDirectory 'debug\sim2sim\verify_yaw_weight_mujoco.py'),
    (Join-Path $projectDirectory 'debug\sim2sim\start_yaw_weight_training.ps1')
)
$frozenPaths = @(@($previousFrozen | ForEach-Object { $_.path }) + $additional | Sort-Object -Unique)
$frozen = @($frozenPaths | ForEach-Object {
    @{ path = $_; sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash }
})
$frozen | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $launchDirectory 'source_before.json') -Encoding utf8
$arguments = @('-u', 'training/scripts/train_dreamwaq.py', '--profile', 'rtx5070', '--num-envs', [string]$environmentCount,
    '--max-iterations', [string]$targetIterations, '--seed', '41882943', '--run-name', $runLabel,
    '--task-profile', $taskProfile, '--headless', '--device', 'cuda:0')
$comparisonManifest = Join-Path $projectDirectory 'logs\rsl_rl\wheelleg_flat_dreamwaq\2026-10-17_12-38-59_rtx5070_fudan-v1_baseline-headingoff-env6144-fresh1000-20261017\run_manifest.json'
$launch = [ordered]@{
    created_at = (Get-Date).ToString('o'); python = $pythonExecutable; arguments = $arguments
    working_directory = $projectDirectory; initialization = 'fresh'; num_envs = $environmentCount
    target_iterations = $targetIterations; seed = 41882943; task_profile = $taskProfile
    heading = @{ enabled = $false; kp = 9.0; kd = 0.0 }; yaw_weights = @{ tracking_yaw_rate = 1.0; tracking_yaw_rate_enhance = 1.0 }
    run_name = $runLabel; validation_only = [bool]$Validation; frozen_file_count = $frozen.Count
    comparison_manifest = $comparisonManifest; stdout = (Join-Path $launchDirectory 'stdout.log')
    stderr = (Join-Path $launchDirectory 'stderr.log'); run_dir = $null
}
$launchPath = Join-Path $launchDirectory 'launch.json'
$launch | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $launchPath -Encoding utf8
$env:PYTHONPATH = $projectDirectory + ';' + (Join-Path $projectDirectory 'training') + ';' + (Join-Path $projectDirectory 'sim2sim\mujoco')
$trainingProcess = Start-Process -FilePath $pythonExecutable -ArgumentList $arguments -WorkingDirectory $projectDirectory -WindowStyle Hidden -RedirectStandardOutput $launch.stdout -RedirectStandardError $launch.stderr -PassThru
$launch['process_id'] = $trainingProcess.Id
$launch | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $launchPath -Encoding utf8
$launch | ConvertTo-Json -Depth 8
