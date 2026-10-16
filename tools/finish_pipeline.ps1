param(
    [string]$Scene = 'data/scenes/fruit1_refined',
    [string]$Training = 'data/outputs/fruit1_refined_training',
    [string]$Segmentation = 'data/outputs/fruit1_refined_seg',
    [string]$Bundle = 'data/outputs/fruit1_final',
    [string]$Evaluation = 'data/outputs/fruit1_final_evaluation',
    [ValidateRange(3,120)][int]$MaxViews = 40,
    [ValidateSet('expected','surface')][string]$DepthReference = 'surface'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'project_paths.ps1')
Set-Location -LiteralPath $projectRoot
$Scene = Resolve-ProjectPath $Scene
$Training = Resolve-ProjectPath $Training
$Segmentation = Resolve-ProjectPath $Segmentation
$Bundle = Resolve-ProjectPath $Bundle
$Evaluation = Resolve-ProjectPath $Evaluation
$env:PYTHONIOENCODING = 'utf-8'
$latest = Get-Content -LiteralPath "$Training/latest_training.json" -Raw -Encoding UTF8 | ConvertFrom-Json
& $gsPython -m food3d export --checkpoint $latest.checkpoint --scene $Scene --output $Bundle --transform $latest.dataparser_transform
if ($LASTEXITCODE -ne 0) { throw 'Export failed.' }
& $gsPython -m food3d fuse --bundle $Bundle --segmentation $Segmentation --max-views $MaxViews --width 480 --depth-reference $DepthReference
if ($LASTEXITCODE -ne 0) { throw 'Fusion failed.' }
& $gsPython -m food3d evaluate --bundle $Bundle --output $Evaluation --width 640
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed.' }
