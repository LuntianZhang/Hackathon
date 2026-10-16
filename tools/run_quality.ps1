param([string]$Name='fruit1_quality_rerun', [switch]$SkipTraining)
$ErrorActionPreference='Stop'
if ($Name -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Name contains invalid characters.' }
. (Join-Path $PSScriptRoot 'project_paths.ps1')
Set-Location -LiteralPath $projectRoot
$env:PYTHONIOENCODING='utf-8'
$training="$dataRoot/outputs/${Name}_training"
$bundle="$dataRoot/outputs/${Name}_bundle"
$evaluation="$dataRoot/outputs/${Name}_evaluation"
if (-not $SkipTraining) {
    & $recognitionPythonExe -m food3d train --scene data/scenes/fruit1_refined --output $training --iterations 15000 --config configs/fruit1_quality.yaml
    if ($LASTEXITCODE -ne 0) { throw 'Quality training failed.' }
}
& tools/finish_pipeline.ps1 -Training $training -Bundle $bundle -Evaluation $evaluation -MaxViews 108 -DepthReference expected
$cleaned="$dataRoot/outputs/${Name}_cleaned"
$cleanedEvaluation="$dataRoot/outputs/${Name}_cleaned_evaluation"
& $gsPython tools/cleanup_floaters.py --bundle $bundle --output $cleaned
if ($LASTEXITCODE -ne 0) { throw 'Cleanup failed.' }
& $gsPython -m food3d fuse --bundle $cleaned --segmentation data/outputs/fruit1_refined_seg --max-views 108 --width 480 --depth-reference surface
if ($LASTEXITCODE -ne 0) { throw 'Refusion failed.' }
& $gsPython -m food3d evaluate --bundle $cleaned --output $cleanedEvaluation --width 640
if ($LASTEXITCODE -ne 0) { throw 'Final evaluation failed.' }
& $gsPython tools/compare_quality.py --candidate-bundle $cleaned --candidate-report $cleanedEvaluation --output "$dataRoot/outputs/${Name}_comparison"
if ($LASTEXITCODE -ne 0) { throw 'Comparison failed.' }
