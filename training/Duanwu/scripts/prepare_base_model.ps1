$ErrorActionPreference = "Stop"

$source = "E:\xiaodie_models\Qwen3-4B"
$dest = "E:\Duanwu\models\Qwen3-4B"

if (!(Test-Path $source)) {
    throw "Source Qwen3-4B model not found: $source"
}

New-Item -ItemType Directory -Force "E:\Duanwu\models" | Out-Null

if (!(Test-Path $dest)) {
    Write-Host "Copying Qwen3-4B base model into Duanwu workspace..."
    robocopy $source $dest /E /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed with code $LASTEXITCODE"
    }
} else {
    Write-Host "Base model already exists: $dest"
}
