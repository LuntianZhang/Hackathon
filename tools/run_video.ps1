param(
    [Parameter(Mandatory=$true)][string]$Video,
    [string]$Name = 'video_run',
    [string]$Config = 'configs/fruit2_video.yaml',
    [string]$ReferenceFrame = '000015',
    [ValidateRange(1,100000)][int]$Iterations = 15000
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'project_paths.ps1')
Set-Location -LiteralPath $projectRoot
$Video = Resolve-ProjectPath $Video
$Config = Resolve-ProjectPath $Config
if ($Name -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Name contains invalid characters.' }
if (-not (Test-Path -LiteralPath $Video -PathType Leaf)) { throw "Video not found: $Video" }
$env:PYTHONIOENCODING = 'utf-8'
$scene = "$dataRoot/scenes/$Name"
$segmentation = "$dataRoot/outputs/${Name}_seg"
$training = "$dataRoot/outputs/${Name}_training"
$bundle = "$dataRoot/outputs/${Name}_bundle"
$evaluation = "$dataRoot/outputs/${Name}_evaluation"
foreach ($target in @($scene,$segmentation,$training,$bundle,$evaluation)) {
    if (Test-Path -LiteralPath $target) { throw "Output exists: $target; choose a new Name." }
}
function Invoke-Stage {
    & $recognitionPythonExe @args
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $args" }
}
Invoke-Stage -m food3d extract --video $Video --scene $scene --config $Config
Invoke-Stage -m food3d colmap --scene $scene --config $Config
Invoke-Stage -m food3d inspect --scene $scene
Invoke-Stage -m food3d segment --scene $scene --output $segmentation --reference-frame $ReferenceFrame
Invoke-Stage -m food3d train --scene $scene --output $training --iterations $Iterations --config $Config
& tools/finish_pipeline.ps1 -Scene $scene -Training $training -Segmentation $segmentation -Bundle $bundle -Evaluation $evaluation -MaxViews 108 -DepthReference surface
& $gsPython -m food3d volume --bundle $bundle
if ($LASTEXITCODE -ne 0) { throw 'Volume estimate failed.' }
Write-Output "Review outputs, then open with: tools/start_demo.ps1 -Bundle $bundle"
