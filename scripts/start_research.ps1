param(
    [string]$BindAddress = '127.0.0.1',
    [ValidateRange(1, 65535)][int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$projectDirectory = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonExecutable = Join-Path $projectDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw '请先建立项目 .venv 并安装 requirements-api.txt。'
}
if (-not (Test-Path -LiteralPath (Join-Path $projectDirectory 'frontend\dist\index.html') -PathType Leaf)) {
    throw '请先在 frontend 目录运行 npm ci 和 npm run build。'
}
Push-Location -LiteralPath $projectDirectory
try {
    $env:PYTHONIOENCODING = 'utf-8'
    & $pythonExecutable -m uvicorn refmind.web.api:app --host $BindAddress --port $Port --workers 1
    if ($LASTEXITCODE -ne 0) { throw "服务退出，代码 $LASTEXITCODE。" }
} finally {
    Pop-Location
}
