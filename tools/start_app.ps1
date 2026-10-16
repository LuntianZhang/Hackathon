param(
    [ValidateRange(1,65535)][int]$Port = 8000,
    [string]$DataDir = 'data/pansense'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'project_paths.ps1')
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath $gsPython)) {
    throw '运行环境不存在，请检查 configs/paths.json 或参照 environments/README.md。'
}
$resolvedDataDir = Resolve-ProjectPath $DataDir
$appUrl = "http://127.0.0.1:$Port"
$portProbe = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
$portProbe.Server.ExclusiveAddressUse = $true
$portInUse = $false
try {
    $portProbe.Start()
} catch {
    if ($_.Exception.InnerException.SocketErrorCode -eq [Net.Sockets.SocketError]::AddressAlreadyInUse) {
        $portInUse = $true
    } else { throw }
} finally {
    $portProbe.Stop()
}
if ($portInUse) {
    $health = $null
    try {
        $health = Invoke-RestMethod -Uri "$appUrl/api/health" -TimeoutSec 3
    } catch { }
    if ($health.status -eq 'ok' -and $health.service -eq 'PanSense P0') {
        Write-Output "盘感 App 已在运行，请直接打开：$appUrl/"
        Write-Output '本次未启动新服务，已有服务的端口和数据目录保持原设置。若需应用新参数，请先在原服务终端按 Ctrl+C 停止，再重新启动。'
        return
    }
    throw "端口 $Port 已被其他服务占用或已有服务暂未就绪。请检查原服务，或使用 -Port 指定其他空闲端口。"
}
Write-Output "PanSense: $appUrl/ ; storage: $resolvedDataDir"
& $gsPython -m pansense --port $Port --data-dir $resolvedDataDir
if ($LASTEXITCODE -ne 0) { throw "PanSense exited with code $LASTEXITCODE" }
