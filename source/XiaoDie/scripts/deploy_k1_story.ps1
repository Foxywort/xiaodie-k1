$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$tts = Join-Path $root "board\xiaodie_tts.py"
$story = Join-Path $root "board\xiaodie_story.py"

if (!(Test-Path $tts)) {
    throw "Missing source file: $tts"
}
if (!(Test-Path $story)) {
    throw "Missing source file: $story"
}

ssh k1 "mkdir -p ~/xiaodie/app ~/xiaodie/audio ~/xiaodie/stories"
scp $tts k1:/home/vicky/xiaodie/app/xiaodie_tts.py
scp $story k1:/home/vicky/xiaodie/app/xiaodie_story.py
ssh k1 "chmod +x ~/xiaodie/app/xiaodie_tts.py ~/xiaodie/app/xiaodie_story.py && python3 ~/xiaodie/app/xiaodie_story.py --help | head -60"
