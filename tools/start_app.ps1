param(
    [int]$Port = 8000,
    [string]$DataDir = 'data/pansense'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'project_paths.ps1')
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath $gsPython)) {
    throw '运行环境不存在，请检查 configs/paths.json 或参照 environments/README.md。'
}
$resolvedDataDir = Resolve-ProjectPath $DataDir
Write-Output "PanSense: http://127.0.0.1:$Port/ ; storage: $resolvedDataDir"
& $gsPython -m pansense --port $Port --data-dir $resolvedDataDir
if ($LASTEXITCODE -ne 0) { throw "PanSense exited with code $LASTEXITCODE" }
