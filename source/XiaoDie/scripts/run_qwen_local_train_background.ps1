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

New-Item -ItemType Directory -Force $env:HF_HOME, $env:MODELSCOPE_CACHE, "outputs\logs" | Out-Null

$venv = "E:\xiaodie_envs\xiaodie-qwenasr"
$env:VIRTUAL_ENV = $venv
$env:PATH = "$venv\Scripts;$env:PATH"
$python = "$venv\Scripts\python.exe"
$modelDir = "E:\xiaodie_models\Qwen3-4B"

Write-Host "Training Qwen3-4B QLoRA from local model directory $modelDir"
& $python -u scripts\train_sft_lora.py --config configs\train.yaml --model $modelDir --max-seq-length 256 --max-steps 2000 --output-dir outputs\story-qwen-lora
