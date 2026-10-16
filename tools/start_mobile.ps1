param(
    [string]$LanHost = '',
    [ValidateRange(1,65535)][int]$Port = 8000,
    [ValidateRange(1,65535)][int]$HttpsPort = 8443,
    [ValidateRange(1,65535)][int]$SetupPort = 8081,
    [string]$DataDir = 'data/pansense'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'project_paths.ps1')
Set-Location -LiteralPath $projectRoot
$arguments = @('-m','pansense','--mobile','--port',"$Port",'--https-port',"$HttpsPort",'--setup-port',"$SetupPort",'--data-dir',(Resolve-ProjectPath $DataDir))
if ($LanHost) { $arguments += @('--lan-host',$LanHost) }
Write-Output "电脑主界面：http://127.0.0.1:$Port/#scan"
Write-Output '手机与电脑需连接同一 Wi-Fi；首次在手机安装本机相机证书。已有服务占用端口时，请先停止原服务。'
& $gsPython @arguments
if ($LASTEXITCODE -ne 0) { throw '手机拍摄服务未能正常启动，请检查端口、局域网地址和运行环境。' }
