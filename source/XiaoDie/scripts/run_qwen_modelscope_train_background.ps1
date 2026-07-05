$ErrorActionPreference = "Stop"

Set-Location "D:\Spacemit\XiaoDie"

$env:HF_HOME = "E:\hf-cache"
$env:MODELSCOPE_CACHE = "E:\modelscope-cache"
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:TRANSFORMERS_VERBOSITY = "info"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:http_proxy = ""
$env:https_proxy = ""
$env:all_proxy = ""
$env:NO_PROXY = "*"
$env:no_proxy = $env:NO_PROXY

New-Item -ItemType Directory -Force $env:HF_HOME, $env:MODELSCOPE_CACHE, "E:\xiaodie_models", "outputs\logs" | Out-Null

$venv = "E:\xiaodie_envs\xiaodie-qwenasr"
$env:VIRTUAL_ENV = $venv
$env:PATH = "$venv\Scripts;$env:PATH"
$python = "$venv\Scripts\python.exe"
$modelDir = "E:\xiaodie_models\Qwen3-4B"

Write-Host "Step 1/2: downloading Qwen/Qwen3-4B from ModelScope to $modelDir"
& $python -u scripts\download_model.py --provider modelscope --model Qwen/Qwen3-4B --output $modelDir --max-workers 1

Write-Host "Step 2/2: training QLoRA from local model directory $modelDir"
& $python -u scripts\train_sft_lora.py --config configs\train.yaml --model $modelDir
