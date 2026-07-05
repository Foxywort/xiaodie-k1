$ErrorActionPreference = "Stop"

Set-Location "E:\Duanwu"

$env:HF_HOME = "E:\Duanwu\cache\hf"
$env:MODELSCOPE_CACHE = "E:\Duanwu\cache\modelscope"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

New-Item -ItemType Directory -Force $env:HF_HOME, $env:MODELSCOPE_CACHE, "E:\Duanwu\logs" | Out-Null

$venv = "E:\xiaodie_envs\xiaodie-qwenasr"
if (!(Test-Path "$venv\Scripts\python.exe")) {
    throw "Python venv not found: $venv"
}

$env:VIRTUAL_ENV = $venv
$env:PATH = "$venv\Scripts;$env:PATH"

Write-Host "Duanwu round-2 environment activated."
Write-Host "Python: $venv\Scripts\python.exe"
Write-Host "HF_HOME: $env:HF_HOME"
Write-Host "MODELSCOPE_CACHE: $env:MODELSCOPE_CACHE"
