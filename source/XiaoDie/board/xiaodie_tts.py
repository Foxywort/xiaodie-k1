#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_OUTPUT = Path.home() / "xiaodie" / "audio" / "xiaodie_last.wav"
DEFAULT_SHERPA_ROOT = Path.home() / "xiaodie" / "models" / "sherpa-onnx"
DEFAULT_SHERPA_RUNTIME = DEFAULT_SHERPA_ROOT / "sherpa-onnx-v1.13.2-linux-riscv64-spacemit-shared"
DEFAULT_SHERPA_MODEL = DEFAULT_SHERPA_ROOT / "vits-piper-zh_CN-huayan-medium"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XiaoDie text-to-speech prototype for K1.")
    parser.add_argument("text", nargs="*", help="Text to speak. If omitted, enter interactive mode.")
    parser.add_argument("--engine", choices=["auto", "sherpa", "espeak"], default="auto")
    parser.add_argument("--speed", type=float, default=0.9, help="Sherpa speech speed. Larger is faster.")
    parser.add_argument("--threads", type=int, default=8, help="Sherpa ONNX CPU threads.")
    parser.add_argument("--espeak-voice", default="cmn", help="Fallback espeak-ng voice.")
    parser.add_argument("--espeak-speed", type=int, default=145, help="Fallback espeak-ng speed.")
    parser.add_argument("--pitch", type=int, default=45, help="Fallback espeak-ng pitch.")
    parser.add_argument("--amplitude", type=int, default=150, help="Fallback espeak-ng volume.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output wav path.")
    parser.add_argument("--player", choices=["auto", "aplay", "pw-play"], default="auto")
    parser.add_argument("--no-play", action="store_true", help="Only generate wav, do not play it.")
    return parser


def require_command(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"Missing command: {name}")
    return path


def sherpa_paths() -> dict[str, Path]:
    return {
        "bin": DEFAULT_SHERPA_RUNTIME / "bin" / "sherpa-onnx-offline-tts",
        "lib": DEFAULT_SHERPA_RUNTIME / "lib",
        "model": DEFAULT_SHERPA_MODEL / "zh_CN-huayan-medium.onnx",
        "tokens": DEFAULT_SHERPA_MODEL / "tokens.txt",
        "data": DEFAULT_SHERPA_MODEL / "espeak-ng-data",
    }


def sherpa_available() -> bool:
    paths = sherpa_paths()
    return all(path.exists() for path in paths.values())


def synthesize(text: str, args: argparse.Namespace) -> Path:
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.engine == "sherpa" or (args.engine == "auto" and sherpa_available()):
        print("[xiaodie] 正在用清晰中文模型合成语音，请稍等...", flush=True)
        return synthesize_sherpa(text, args, output)
    print("[xiaodie] 未找到清晰中文模型，使用 espeak-ng 兜底语音。", flush=True)
    return synthesize_espeak(text, args, output)


def synthesize_sherpa(text: str, args: argparse.Namespace, output: Path) -> Path:
    paths = sherpa_paths()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing Sherpa TTS files:\n" + "\n".join(missing))

    env = os.environ.copy()
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{paths['lib']}:{old_ld}" if old_ld else str(paths["lib"])

    cmd = [
        str(paths["bin"]),
        f"--vits-model={paths['model']}",
        f"--vits-tokens={paths['tokens']}",
        f"--vits-data-dir={paths['data']}",
        f"--output-filename={output}",
        f"--speed={args.speed}",
        f"--num-threads={args.threads}",
        "--print-args=false",
        text,
    ]
    result = subprocess.run(
        cmd,
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.stderr:
        for line in result.stderr.splitlines():
            if line.startswith("Elapsed seconds:") or line.startswith("Audio duration:"):
                print(f"[xiaodie] {line}", flush=True)
    return output


def synthesize_espeak(text: str, args: argparse.Namespace, output: Path) -> Path:
    espeak = require_command("espeak-ng")
    cmd = [
        espeak,
        "-v",
        args.espeak_voice,
        "-s",
        str(args.espeak_speed),
        "-p",
        str(args.pitch),
        "-a",
        str(args.amplitude),
        "-w",
        str(output),
        text,
    ]
    subprocess.run(cmd, check=True)
    return output


def play(path: Path) -> None:
    players = []
    if args_player := getattr(play, "_preferred", None):
        players.append(args_player)
    else:
        players.extend(["aplay", "pw-play"])

    last_error = None
    for player in players:
        if player == "aplay" and shutil.which("aplay"):
            cmd = ["aplay", "-D", "default", str(path)]
        elif player == "pw-play" and shutil.which("pw-play"):
            cmd = ["pw-play", str(path)]
        else:
            continue

        print(f"[xiaodie] 正在播放: {' '.join(cmd)}", flush=True)
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"[xiaodie] 播放失败，尝试下一个播放器: {exc}", file=sys.stderr, flush=True)

    if last_error:
        raise last_error
    raise SystemExit("Missing playback command: aplay or pw-play")


def set_player_preference(args: argparse.Namespace) -> None:
    if args.player == "aplay":
        play._preferred = "aplay"
        return
    if args.player == "pw-play":
        play._preferred = "pw-play"
        return
    play._preferred = None


def speak(text: str, args: argparse.Namespace) -> None:
    text = text.strip()
    if not text:
        return
    wav = synthesize(text, args)
    print(f"[xiaodie] 已生成: {wav}", flush=True)
    if not args.no_play:
        play(wav)
        print("[xiaodie] 播放完成。", flush=True)
    else:
        print("[xiaodie] 已跳过播放。", flush=True)


def interactive(args: argparse.Namespace) -> None:
    print("小蝶 TTS 已启动。输入文字后按回车播放，输入 q 退出。", flush=True)
    print("提示：清晰中文模型首次/长句合成可能需要等待几秒。", flush=True)
    while True:
        try:
            text = input("xiaodie> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.strip().lower() in {"q", "quit", "exit"}:
            return
        speak(text, args)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    set_player_preference(args)
    text = " ".join(args.text)
    if text:
        speak(text, args)
    else:
        interactive(args)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"command failed: {exc}", file=sys.stderr)
        raise SystemExit(exc.returncode)
