$ErrorActionPreference = "Stop"

$env:HF_HOME = "E:\hf-cache"
$env:MODELSCOPE_CACHE = "E:\modelscope-cache"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"
$env:PYTHONUTF8 = "1"

New-Item -ItemType Directory -Force $env:HF_HOME, $env:MODELSCOPE_CACHE | Out-Null

$venv = "E:\xiaodie_envs\xiaodie-qwenasr"
if (!(Test-Path "$venv\Scripts\python.exe")) {
    throw "Python venv not found: $venv"
}

$env:VIRTUAL_ENV = $venv
$env:PATH = "$venv\Scripts;$env:PATH"

Write-Host "XiaoDie environment activated."
Write-Host "Python: $venv\Scripts\python.exe"
Write-Host "HF_HOME: $env:HF_HOME"
Write-Host "MODELSCOPE_CACHE: $env:MODELSCOPE_CACHE"
