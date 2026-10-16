param([string]$Bundle = '', [int]$Port = 7860)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'project_paths.ps1')
Set-Location -LiteralPath $projectRoot
if (-not $Bundle) {
    $activeDemo = Join-Path $dataRoot 'outputs/active_demo.json'
    if (Test-Path -LiteralPath $activeDemo) {
        $Bundle = (Get-Content -LiteralPath $activeDemo -Raw -Encoding UTF8 | ConvertFrom-Json).bundle
        if ([string]::IsNullOrWhiteSpace($Bundle)) {
            throw "演示配置缺少 bundle 路径：$activeDemo。请使用 -Bundle 指定完整模型目录。"
        }
    } else { $Bundle = 'data/outputs/fruit1_final' }
}
$env:PYTHONIOENCODING = 'utf-8'
$Bundle = Resolve-ProjectPath $Bundle
$requiredFiles = @('model.json', 'gaussians.npz', 'scene.json',
    'semantics/semantics.npz', 'semantics/fusion.json', 'semantics/instances.json')
$missingFiles = @($requiredFiles | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $Bundle $_) -PathType Leaf)
})
if ($missingFiles.Count -gt 0) {
    throw @"
三维演示模型缺失或不完整：$Bundle
缺少文件：$($missingFiles -join ', ')
start_demo.ps1 只加载已完成重建和语义融合的模型，不会自动生成模型。
如果要打开盘感 App（上传照片或视频），请运行：& tools/start_app.ps1
如果已有完整模型，请运行：& tools/start_demo.ps1 -Bundle '模型目录'
如果只有原始素材，请先按 README.md 的流水线生成模型，或从备份恢复完整结果目录；不要创建空的 model.json。
"@
}
if (-not (Test-Path -LiteralPath $gsPython -PathType Leaf)) {
    throw '运行环境不存在，请检查 configs/paths.json 或参照 environments/README.md。'
}
Write-Output "Food3D: http://127.0.0.1:$Port/ ; bundle: $Bundle"
& $gsPython -m food3d serve --bundle $Bundle --port $Port
if ($LASTEXITCODE -ne 0) { throw 'Demo failed; review the displayed error.' }
