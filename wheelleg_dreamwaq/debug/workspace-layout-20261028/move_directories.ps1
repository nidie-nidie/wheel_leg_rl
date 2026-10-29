param([switch]$Verify)

$ErrorActionPreference = 'Stop'
$layoutRoot = $PSScriptRoot
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $layoutRoot '..\..'))
$manifestPath = Join-Path $layoutRoot 'relocations.json'

function Get-ProjectPath([string]$RelativePath) {
    $absolutePath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $RelativePath))
    if (-not $absolutePath.StartsWith($projectRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "目录超出当前工程：$absolutePath"
    }
    return $absolutePath
}

if ($Verify) {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $checkedFiles = 0
    foreach ($move in $manifest.moves) {
        foreach ($file in $move.files) {
            $currentPath = Get-ProjectPath $file.to
            if (-not (Test-Path -LiteralPath $currentPath -PathType Leaf)) { throw "迁移文件缺失：$currentPath" }
            if (-not $move.editable -and (Get-FileHash -LiteralPath $currentPath -Algorithm SHA256).Hash -ne $file.sha256) {
                throw "归档文件内容改变：$currentPath"
            }
            $checkedFiles++
        }
    }
    foreach ($evidence in $manifest.evidence) {
        if ((Get-FileHash -LiteralPath (Get-ProjectPath $evidence.path) -Algorithm SHA256).Hash -ne $evidence.sha256) {
            throw "历史模型或证据内容改变：$($evidence.path)"
        }
    }
    [pscustomobject]@{ verified_files = $checkedFiles; immutable_evidence = $manifest.evidence.Count } | ConvertTo-Json
    exit
}

if (Test-Path -LiteralPath $manifestPath) { throw '迁移清单已经存在，请使用 -Verify 检查，禁止重复移动。' }
$moves = [System.Collections.Generic.List[object]]::new()
$moves.Add(@{ from = 'source\wheelleg_dreamwaq\wheelleg_dreamwaq'; to = 'training\wheelleg_dreamwaq'; editable = $true })
$moves.Add(@{ from = 'scripts'; to = 'training\scripts'; editable = $true })
foreach ($item in Get-ChildItem -LiteralPath (Get-ProjectPath 'recovery') -Force) {
    if ($item.Extension -eq '.py' -or $item.Name -in @('tests', 'README.md', 'geometry_fp64_bound.md')) {
        $moves.Add(@{ from = "recovery\$($item.Name)"; to = "training\recovery\$($item.Name)"; editable = $true })
    }
}
$moves.Add(@{ from = 'source\wheelleg_dreamwaq\wheelleg_dreamwaq.egg-info'; to = 'debug\archive\build-metadata\wheelleg_dreamwaq.egg-info'; editable = $false })
$moves.Add(@{ from = 'recovery\__pycache__'; to = 'debug\cache\recovery-before-layout'; editable = $false })
$moves.Add(@{ from = 'artifacts\phase2_dreamwaq\baseline-heading-implementation-20261016\source'; to = 'debug\archive\source_snapshots\baseline-heading-20261016\source'; editable = $false })
$moves.Add(@{ from = 'artifacts\phase2_dreamwaq\leg-reversal-trial-20261011-v1\source-before'; to = 'debug\archive\source_snapshots\leg-reversal-20261011\source-before'; editable = $false })
$moves.Add(@{ from = 'artifacts\phase2_dreamwaq\split-action-smooth-20261013-v1\before\source'; to = 'debug\archive\source_snapshots\split-action-smooth-20261013\source'; editable = $false })
$moves.Add(@{ from = 'artifacts\phase2_dreamwaq\split-action-smooth-20261013-v1\before\sim2sim'; to = 'debug\archive\source_snapshots\split-action-smooth-20261013\sim2sim'; editable = $false })
$moves.Add(@{ from = 'artifacts\debug\sim2sim\open-beni-reward-review-20261011\source'; to = 'debug\archive\source_snapshots\open-beni-reward-review-20261011\source'; editable = $false })
$moves.Add(@{ from = 'artifacts\debug\sim2sim\remove-artificial-speed-limits-20261011-v1\source-before'; to = 'debug\archive\source_snapshots\remove-speed-limits-20261011\source-before'; editable = $false })
$moves.Add(@{ from = 'PACE\build\github-upload'; to = 'debug\archive\pace-github-upload'; editable = $false })
$moves.Add(@{ from = 'wheel_leg_rl-mainwheelleg_dreamwaqdebugsim2simroot_cause_suiterunstask7-test-v116bruntime_cachepytest'; to = 'debug\cache\task7-test-v116b'; editable = $false })
$moves.Add(@{ from = 'wheel_leg_rl-mainwheelleg_dreamwaqdebugsim2simroot_cause_suiterunstask7-test-v116runtime_cachepytest'; to = 'debug\cache\task7-test-v116'; editable = $false })

foreach ($move in $moves) {
    $sourcePath = Get-ProjectPath $move.from
    $targetPath = Get-ProjectPath $move.to
    if (-not (Test-Path -LiteralPath $sourcePath)) { throw "源路径不存在：$sourcePath" }
    if (Test-Path -LiteralPath $targetPath) { throw "目标路径已经存在：$targetPath" }
    $move.files = @(foreach ($file in Get-ChildItem -LiteralPath $sourcePath -Recurse -File -Force) {
        $suffix = $file.FullName.Substring($sourcePath.Length)
        @{ from = $move.from + $suffix; to = $move.to + $suffix; sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash }
    })
}
$evidence = @(foreach ($relativePath in @(
    'recovery\artifacts\unclipped-actions-20261024\training4096x2000\model_2000.pt',
    'recovery\artifacts\unclipped-actions-20261024\replay-model2000-isaac-poses\first_episode_trace.jsonl',
    'recovery\artifacts\unclipped-actions-20261024\replay-model2000-isaac-poses\replay_report.json'
)) { @{ path = $relativePath; sha256 = (Get-FileHash -LiteralPath (Get-ProjectPath $relativePath) -Algorithm SHA256).Hash } })
@{ moves = @($moves.ToArray()); evidence = $evidence } | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $manifestPath -Encoding utf8
foreach ($move in $moves) {
    $sourcePath = Get-ProjectPath $move.from
    $targetPath = Get-ProjectPath $move.to
    New-Item -ItemType Directory -Path (Split-Path -Parent $targetPath) -Force | Out-Null
    Move-Item -LiteralPath $sourcePath -Destination $targetPath
}
foreach ($relativePath in @('source\wheelleg_dreamwaq', 'source')) {
    $emptyPath = Get-ProjectPath $relativePath
    if (@(Get-ChildItem -LiteralPath $emptyPath -Force).Count -eq 0) { Remove-Item -LiteralPath $emptyPath }
}
"已移动 $($moves.Count) 个目录或文件；完整路径及内容摘要保存在 $manifestPath"
