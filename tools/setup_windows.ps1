param([string]$RecognitionPython = '3.13', [string]$ReconstructionPython = '3.10')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'project_paths.ps1')
Set-Location -LiteralPath $projectRoot
$env:UV_CACHE_DIR = Join-Path $cacheRoot 'uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $toolsRoot 'python'
$env:SAM2_BUILD_CUDA = '0'
$env:PYTHONIOENCODING = 'utf-8'
$uvExecutable = Join-Path $toolsRoot 'uv/uv.exe'
if (-not (Test-Path -LiteralPath $uvExecutable)) {
    throw '请将 uv.exe 放入项目 .tools/uv 目录。'
}
function Invoke-Uv {
    & $uvExecutable @args
    if ($LASTEXITCODE -ne 0) { throw "uv failed: $args" }
}
if (-not (Test-Path -LiteralPath $recognitionPythonExe)) {
    Invoke-Uv venv --python $RecognitionPython $recognitionEnv
}
if (-not (Test-Path -LiteralPath $gsPython)) {
    Invoke-Uv venv --python $ReconstructionPython $reconstructionEnv
}
Invoke-Uv pip install --python $recognitionPythonExe 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128
Invoke-Uv pip install --python $recognitionPythonExe -e '.[dev,sfm,recognition]' 'hydra-core>=1.3.2' 'iopath>=0.1.10' -c environments/recognition-constraints.txt
Invoke-Uv pip install --python $gsPython 'torch==2.4.0' 'torchvision==0.19.0' --index-url https://download.pytorch.org/whl/cu124
Invoke-Uv pip install --python $gsPython 'gsplat==1.4.0+pt24cu124' --index-url https://docs.gsplat.studio/whl/pt24cu124
Invoke-Uv pip install --python $gsPython -e '.[app,viewer]' 'nerfstudio==1.1.5' -c environments/reconstruction-constraints.txt
$upstreamDirectory = Join-Path $cacheRoot 'upstream'
$samDirectory = Join-Path $upstreamDirectory 'sam2-main'
$samArchive = Join-Path $upstreamDirectory 'sam2-main.zip'
if (-not (Test-Path -LiteralPath "$samDirectory/setup.py")) {
    New-Item -ItemType Directory -Force -Path $upstreamDirectory | Out-Null
    Invoke-WebRequest 'https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip' -OutFile $samArchive
    Expand-Archive -LiteralPath $samArchive -DestinationPath $upstreamDirectory
}
Invoke-Uv pip install --python $recognitionPythonExe -e $samDirectory --no-deps
$weightsDirectory = Join-Path $toolsRoot 'weights'
$samCheckpoint = Join-Path $weightsDirectory 'sam2.1_hiera_small.pt'
if (-not (Test-Path -LiteralPath $samCheckpoint)) {
    New-Item -ItemType Directory -Force -Path $weightsDirectory | Out-Null
    Invoke-WebRequest 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt' -OutFile $samCheckpoint
}
Write-Output 'Environments ready. Run tools/gpu_smoke.py with the reconstruction Python.'
