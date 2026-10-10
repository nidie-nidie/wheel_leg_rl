[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = 'Stop'

function Exit-WithCode {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,
        [Parameter(Mandatory = $true)]
        [int]$Code
    )
    [Console]::Error.WriteLine($Message)
    exit $Code
}

if ($Command -notin @('run', 'verify', 'report')) {
    Exit-WithCode -Message "Invalid command: '$Command'" -Code 2
}

$forwardArgs = @()
if ($null -ne $RemainingArgs) {
    $forwardArgs = @($RemainingArgs)
}
$suiteRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $suiteRoot '..\..\..')).Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Exit-WithCode -Message "Frozen project interpreter is missing: $python" -Code 2
}

$runId = $null
$resumeIndex = -1
$isFreshRun = $false
if ($Command -eq 'run') {
    $resumeIndex = [Array]::IndexOf($forwardArgs, '--resume')
    if ($resumeIndex -ge 0) {
        if ($resumeIndex + 1 -ge $forwardArgs.Count) {
            Exit-WithCode -Message '--resume requires a run id' -Code 2
        }
        $runId = $forwardArgs[$resumeIndex + 1]
    } else {
        $isFreshRun = $true
        $runId = 'root-cause-core-v1-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
    }
} else {
    if ($forwardArgs.Count -lt 1) {
        Exit-WithCode -Message "$Command requires a run id" -Code 2
    }
    $runId = $forwardArgs[0]
}

if ($runId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
    Exit-WithCode -Message "Invalid run id: '$runId'" -Code 2
}
$runsRoot = [System.IO.Path]::GetFullPath((Join-Path $suiteRoot 'runs'))
$runRoot = [System.IO.Path]::GetFullPath((Join-Path $runsRoot $runId))
$runsPrefix = $runsRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (-not $runRoot.StartsWith($runsPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    Exit-WithCode -Message "Run root escapes the suite runs directory: $runRoot" -Code 2
}

function Assert-PlainDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )
    try {
        $attributes = [System.IO.File]::GetAttributes($Path)
    } catch [System.IO.FileNotFoundException] {
        throw "Missing ${Label}: $Path"
    } catch [System.IO.DirectoryNotFoundException] {
        throw "Missing ${Label}: $Path"
    }
    if (($attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "${Label} is a link or reparse point: $Path"
    }
    if (($attributes -band [System.IO.FileAttributes]::Directory) -eq 0) {
        throw "${Label} is not a directory: $Path"
    }
}

function Assert-PlainFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )
    try {
        $attributes = [System.IO.File]::GetAttributes($Path)
    } catch [System.IO.FileNotFoundException] {
        throw "Missing ${Label}: $Path"
    } catch [System.IO.DirectoryNotFoundException] {
        throw "Missing ${Label}: $Path"
    }
    if (($attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "${Label} is a link or reparse point: $Path"
    }
    if (($attributes -band [System.IO.FileAttributes]::Directory) -ne 0) {
        throw "${Label} is not a file: $Path"
    }
}

function Assert-PlainDirectoryIfPresent {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )
    if ([System.IO.Directory]::Exists($Path) -or [System.IO.File]::Exists($Path)) {
        Assert-PlainDirectory -Path $Path -Label $Label
    }
}

$runtimeCache = Join-Path $runRoot 'runtime_cache'
$cacheNames = @('pycache', 'pytest-cache', 'pytest-tmp')
$manifestPath = Join-Path $runRoot 'run_manifest.json'
try {
    if ($isFreshRun) {
        Assert-PlainDirectoryIfPresent -Path $runRoot -Label 'run root'
        Assert-PlainDirectoryIfPresent -Path $runtimeCache -Label 'runtime cache'
        foreach ($relative in $cacheNames) {
            Assert-PlainDirectoryIfPresent -Path (Join-Path $runtimeCache $relative) -Label "runtime cache child $relative"
        }
        foreach ($relative in $cacheNames) {
            New-Item -ItemType Directory -Force -Path (Join-Path $runtimeCache $relative) | Out-Null
        }
        Assert-PlainDirectory -Path $runRoot -Label 'run root'
        Assert-PlainDirectory -Path $runtimeCache -Label 'runtime cache'
        foreach ($relative in $cacheNames) {
            Assert-PlainDirectory -Path (Join-Path $runtimeCache $relative) -Label "runtime cache child $relative"
        }
    } else {
        Assert-PlainDirectory -Path $runRoot -Label 'run root'
        Assert-PlainFile -Path $manifestPath -Label 'run manifest'
        Assert-PlainDirectory -Path $runtimeCache -Label 'runtime cache'
        foreach ($relative in $cacheNames) {
            Assert-PlainDirectory -Path (Join-Path $runtimeCache $relative) -Label "runtime cache child $relative"
        }
    }
} catch {
    Exit-WithCode -Message $_.Exception.Message -Code 5
}

$env:WHEELLEG_ROOT_CAUSE_BOOTSTRAP = '1'
$env:WHEELLEG_ROOT_CAUSE_RUN_ID = $runId
$env:WHEELLEG_ROOT_CAUSE_RUN_ROOT = $runRoot
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPYCACHEPREFIX = Join-Path $runtimeCache 'pycache'

$arguments = @('-B', '-m', 'debug.sim2sim.root_cause_suite', $Command) + $forwardArgs
Push-Location $projectRoot
try {
    & $python @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
