$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) '..\..\..')).Path
$old = (Get-ChildItem -LiteralPath (Join-Path $root 'output\ppt') -Filter '*.pptx' | Sort-Object Length | Select-Object -First 1).FullName
$ascii = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'old_test.pptx'
Copy-Item -LiteralPath $old -Destination $ascii -Force
$app = New-Object -ComObject PowerPoint.Application
try {
  $p = $app.Presentations.Open($ascii, $false, $false, $false)
  Write-Output "old slides: $($p.Slides.Count)"
  $p.Close()
}
finally {
  $app.Quit()
  [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($app) | Out-Null
}
