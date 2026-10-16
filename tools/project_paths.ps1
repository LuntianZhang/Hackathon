# Shared resource paths and Python environment settings. Dot-source this file.
$projectRoot = Split-Path -Parent $PSScriptRoot
$configurationPath = Join-Path $projectRoot 'configs/paths.json'
$pathSettings = if (Test-Path -LiteralPath $configurationPath) {
    Get-Content -LiteralPath $configurationPath -Raw -Encoding UTF8 | ConvertFrom-Json
} else { [pscustomobject]@{} }

function Resolve-ResourceRoot([string]$Name, [string]$Default) {
    $value = $pathSettings.$Name
    if (-not $value) { $value = $Default }
    $resolved = if ([IO.Path]::IsPathRooted($value)) { [IO.Path]::GetFullPath($value) } else {
        [IO.Path]::GetFullPath((Join-Path $projectRoot $value))
    }
    if (-not $resolved.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "资源目录必须位于工程根目录内：$Name"
    }
    return $resolved
}

$dataRoot = Resolve-ResourceRoot 'data_root' 'data'
$dataRoot = $dataRoot.Replace('\', '/')
$cacheRoot = Resolve-ResourceRoot 'cache_root' '.cache'
$toolsRoot = Resolve-ResourceRoot 'tools_root' '.tools'
$recognitionEnv = Resolve-ResourceRoot 'recognition_env' '.venv'
$reconstructionEnv = Resolve-ResourceRoot 'reconstruction_env' '.venv-gs'
$recognitionPythonExe = Join-Path $recognitionEnv 'Scripts/python.exe'
$gsPython = Join-Path $reconstructionEnv 'Scripts/python.exe'

function Resolve-ProjectPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    $parts = $Value -split '[/\\]', 2
    $resourceRoots = @{ data=$dataRoot; '.cache'=$cacheRoot; '.tools'=$toolsRoot;
        '.venv'=$recognitionEnv; '.venv-gs'=$reconstructionEnv }
    if ($resourceRoots.ContainsKey($parts[0])) {
        $root = $resourceRoots[$parts[0]]
        if ($parts.Count -eq 1) { return $root }
        return [IO.Path]::GetFullPath((Join-Path $root $parts[1]))
    }
    return [IO.Path]::GetFullPath((Join-Path $projectRoot $Value))
}

$sourcePath = Join-Path $projectRoot 'src'
$sourceEntries = @($sourcePath) + @($env:PYTHONPATH -split [regex]::Escape([string][IO.Path]::PathSeparator) | Where-Object { $_ })
$env:PYTHONPATH = ($sourceEntries | Select-Object -Unique) -join [IO.Path]::PathSeparator
$env:PYTHONUTF8 = '0'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONNOUSERSITE = '1'
$env:UV_CACHE_DIR = Join-Path $cacheRoot 'uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $toolsRoot 'python'
$toolEntries = @((Join-Path $toolsRoot 'uv'), (Join-Path $toolsRoot 'ffmpeg/bin'))
$env:PATH = (($toolEntries + @($env:PATH -split [regex]::Escape([string][IO.Path]::PathSeparator))) | Select-Object -Unique) -join [IO.Path]::PathSeparator
$env:HF_HOME = Join-Path $cacheRoot 'models/huggingface'
$env:TORCH_HOME = Join-Path $cacheRoot 'models/torch'
