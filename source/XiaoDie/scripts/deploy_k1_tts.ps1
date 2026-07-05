$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "board\xiaodie_tts.py"

if (!(Test-Path $src)) {
    throw "Missing source file: $src"
}

ssh k1 "mkdir -p ~/xiaodie/app ~/xiaodie/audio"
scp $src k1:/home/vicky/xiaodie/app/xiaodie_tts.py
ssh k1 "chmod +x ~/xiaodie/app/xiaodie_tts.py && python3 ~/xiaodie/app/xiaodie_tts.py --help | head -40"
