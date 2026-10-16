param([string]$Bundle = '', [int]$Port = 7860)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'project_paths.ps1')
Set-Location -LiteralPath $projectRoot
if (-not $Bundle) {
    $activeDemo = Join-Path $dataRoot 'outputs/active_demo.json'
    if (Test-Path -LiteralPath $activeDemo) {
        $Bundle = (Get-Content -LiteralPath $activeDemo -Raw -Encoding UTF8 | ConvertFrom-Json).bundle
    } else { $Bundle = 'data/outputs/fruit1_final' }
}
$env:PYTHONIOENCODING = 'utf-8'
$Bundle = Resolve-ProjectPath $Bundle
& $gsPython -m food3d serve --bundle $Bundle --port $Port
if ($LASTEXITCODE -ne 0) { throw 'Demo failed; review the displayed error.' }
