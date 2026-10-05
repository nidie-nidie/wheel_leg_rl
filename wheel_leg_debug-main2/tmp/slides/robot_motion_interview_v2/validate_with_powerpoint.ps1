$ErrorActionPreference = 'Stop'

$taskDir = (Resolve-Path (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
$source = (Get-ChildItem -LiteralPath $taskDir -Filter '*.pptx' |
    Where-Object { $_.Name -notmatch '^deck_v2_' } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1).FullName
$sourceAscii = Join-Path $taskDir 'deck_v2_input.pptx'
$validatedAscii = Join-Path $taskDir 'deck_v2_validated.pptx'
$exportDir = Join-Path $taskDir 'powerpoint_export'

New-Item -ItemType Directory -Force -Path $exportDir | Out-Null
Get-ChildItem -LiteralPath $exportDir -File -ErrorAction SilentlyContinue | Remove-Item -Force
Copy-Item -LiteralPath $source -Destination $sourceAscii -Force

$powerPoint = New-Object -ComObject PowerPoint.Application
$powerPoint.DisplayAlerts = 1
try {
    $presentation = $powerPoint.Presentations.Open($sourceAscii, $false, $false, $false)
    $slideCount = $presentation.Slides.Count
    $presentation.SaveAs($validatedAscii, 24)
    $presentation.Close()

    $check = $powerPoint.Presentations.Open($validatedAscii, $false, $false, $false)
    if ($check.Slides.Count -ne $slideCount) {
        throw "Slide count changed after PowerPoint save: $slideCount -> $($check.Slides.Count)"
    }
    $check.Export($exportDir, 'PNG', 1600, 900)
    $check.Close()
    Write-Output "Validated: $validatedAscii"
    Write-Output "Slides: $slideCount"
    Write-Output "Export: $exportDir"
}
finally {
    $powerPoint.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerPoint) | Out-Null
}
