param([string]$RunName = 'fruit1_rerun')
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must contain only letters, numbers, underscores or hyphens.' }
. (Join-Path $PSScriptRoot 'project_paths.ps1')
Set-Location -LiteralPath $projectRoot
$env:PYTHONIOENCODING = 'utf-8'
$initial = "$dataRoot/scenes/${RunName}_initial"
$refined = "$dataRoot/scenes/${RunName}_refined"
$segmentation = "$dataRoot/outputs/${RunName}_seg"
$refinedSegmentation = "$dataRoot/outputs/${RunName}_refined_seg"
$filtered = "$dataRoot/outputs/${RunName}_filtered.db"
$training = "$dataRoot/outputs/${RunName}_training"
$bundle = "$dataRoot/outputs/${RunName}_bundle"
$evaluation = "$dataRoot/outputs/${RunName}_evaluation"
foreach ($target in @($initial,$refined,$segmentation,$refinedSegmentation,$filtered,$training,$bundle,$evaluation)) {
    if (Test-Path -LiteralPath $target) { throw "Output exists: $target. Use a new RunName." }
}
function Invoke-Recognition {
    & $recognitionPythonExe @args
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $args" }
}
Invoke-Recognition -m food3d photos --input 'data/raw/photos/水果1' --scene $initial --config configs/fruit1_bootstrap.yaml --source-kind rendered
Invoke-Recognition -m food3d colmap --scene $initial --config configs/fruit1_bootstrap.yaml
Invoke-Recognition -m food3d segment --scene $initial --output $segmentation --reference-frame 000077
Invoke-Recognition tools/filter_sfm_matches.py --source "$initial/colmap/database.db" --output $filtered --segmentation $segmentation --focal 1152
$config = "$dataRoot/outputs/${RunName}_refined.yaml"
Invoke-Recognition -c "from food3d.io import load_config; import yaml; from pathlib import Path; c=load_config(Path('configs/fruit1_refined.yaml')); c['colmap']['reuse_database']='$filtered'; Path('$config').write_text(yaml.safe_dump(c),encoding='utf-8')"
Invoke-Recognition -m food3d photos --input 'data/raw/photos/水果1' --scene $refined --config $config --source-kind rendered
Invoke-Recognition -m food3d colmap --scene $refined --config $config
Invoke-Recognition -m food3d inspect --scene $refined
Invoke-Recognition tools/rebind_segmentation.py --segmentation $segmentation --scene $refined --output $refinedSegmentation
Invoke-Recognition -m food3d train --scene $refined --output $training --iterations 7000
& tools/finish_pipeline.ps1 -Scene $refined -Training $training -Segmentation $refinedSegmentation -Bundle $bundle -Evaluation $evaluation
Write-Output "Ready: tools/start_demo.ps1 -Bundle $bundle"
