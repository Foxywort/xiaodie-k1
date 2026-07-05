# K1 Board Notes

K1 is a SpacemiT RISC-V board used as the XiaoDie edge target.

Known board info:

- User: `vicky`
- SSH alias: `k1`
- OS: `Bianbu 2.3.3`
- Kernel: `Linux 6.6.63`
- Architecture: `riscv64`
- Hostname: `spacemit`

## SSH

```powershell
ssh k1 "whoami && hostname && uname -a"
```

Local Windows SSH config:

```text
Host k1
  HostName 192.168.214.221
  User vicky
  IdentityFile C:/xiaodie_keys/k1_ed25519
  IdentitiesOnly yes
```

The IP may change when the hotspot or network changes. If it changes, update `HostName` in:

```powershell
$env:USERPROFILE\.ssh\config
```

Common development commands:

```powershell
ssh k1 "command"
scp local_file k1:/home/vicky/
scp k1:/home/vicky/remote_file .
```

## System Probe

Useful commands after login:

```bash
uname -a
cat /etc/os-release
lscpu
free -h
df -h
aplay -l
aplay -L
python3 --version
which python3
```

## Current Audio Status

Detected playback devices:

```text
card 0: sndes8326 [snd-es8326], device 0
card 1: sndhdmi [snd-hdmi], device 0
```

PipeWire owns the hardware device, so direct playback with `plughw:CARD=sndes8326,DEV=0` can report `Device or resource busy`. Use the default output path:

```bash
aplay -D default ~/xiaodie/audio/hello_test.wav
pw-play ~/xiaodie/audio/hello_test.wav
```

## First Deployable Prototype

```text
stdin text -> local story assistant -> TTS wav -> aplay/headphone
```

Current prototype:

```powershell
.\scripts\deploy_k1_tts.ps1
ssh k1 'python3 ~/xiaodie/app/xiaodie_tts.py "你好，我是小蝶。我们开始讲故事吧。"'
```

The default engine is now Sherpa ONNX with the `vits-piper-zh_CN-huayan-medium` Chinese voice model. It is much clearer than the original `espeak-ng` prototype and still runs offline on K1.

Interactive mode:

```powershell
ssh -t k1 "python3 ~/xiaodie/app/xiaodie_tts.py"
```

On the K1 local terminal, run the same command without SSH:

```bash
python3 ~/xiaodie/app/xiaodie_tts.py
```

After entering text, wait a few seconds while it prints `正在用清晰中文模型合成语音，请稍等...`.

On K1, the script path is:

```bash
~/xiaodie/app/xiaodie_tts.py
```

It generates the latest WAV here:

```bash
~/xiaodie/audio/xiaodie_last.wav
```

Useful local K1 commands:

```bash
python3 ~/xiaodie/app/xiaodie_tts.py "你好，我是小蝶。"
python3 ~/xiaodie/app/xiaodie_tts.py --speed 0.85 "我说得慢一点，会更清楚。"
python3 ~/xiaodie/app/xiaodie_tts.py --engine espeak "如果模型不可用，会使用旧的兜底语音。"
wpctl set-volume @DEFAULT_AUDIO_SINK@ 85%
```

Do not use Ubuntu's `apt install piper` for TTS on this board. That package is a GTK gaming-device configuration app from `libratbag`, not Piper TTS.

## Story Machine

Keyword-to-story voice output is deployed at:

```bash
~/xiaodie/app/xiaodie_story.py
```

Run a complete micro story from keywords:

```bash
python3 ~/xiaodie/app/xiaodie_story.py 星星 勇气
```

Interactive mode:

```bash
python3 ~/xiaodie/app/xiaodie_story.py
```

Then type keywords such as:

```text
月亮 分享
```

Length options:

```bash
python3 ~/xiaodie/app/xiaodie_story.py 星星 勇气 --length mini
python3 ~/xiaodie/app/xiaodie_story.py 星星 勇气 --length short
python3 ~/xiaodie/app/xiaodie_story.py 星星 勇气 --length medium
```

`mini` is the default because the clear Chinese TTS model is slower on K1. For a faster but less natural voice:

```bash
python3 ~/xiaodie/app/xiaodie_story.py 星星 勇气 --engine espeak
```

Latest generated story text:

```bash
~/xiaodie/stories/xiaodie_last_story.txt
```

Suggested K1 directory:

```bash
mkdir -p ~/xiaodie/{app,models,logs,audio}
```

## Useful References

- SpacemiT K1 official documents: https://www.spacemit.com/community/document/info?lang=en&nodepath=hardware%2Fkey_stone%2Fk1
- SpacemiT AI SDK: https://github.com/spacemit-com/ai-sdk
- K1 datasheet mirror: https://docs.banana-pi.org/en/BPI-F3/SpacemiT_K1_datasheet
