#!/usr/bin/env python3
"""Low-latency Chaowen TTS runner for K1.

This script is intentionally offline-only. It calls the local sherpa-onnx
Chaowen full model on the board and can play each generated sentence
immediately.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def split_sentences(text: str) -> list[str]:
    parts = re.findall(r"[^。！？!?；;]+[。！？!?；;]?", text.strip())
    merged: list[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(buf) + len(part) < 28:
            buf += part
            continue
        if buf:
            merged.append(buf)
        buf = part
    if buf:
        merged.append(buf)
    return merged


def run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output-dir", default="/home/vicky/xiaodie/tts/stream_out")
    parser.add_argument("--base", default="/home/vicky/xiaodie/tts")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--max-sentences", type=int, default=1)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--device", default="hw:1,0")
    args = parser.parse_args()

    base = Path(args.base)
    bin_path = base / "sherpa-onnx-v1.13.3-linux-riscv64-spacemit-shared/bin/sherpa-onnx-offline-tts"
    model_dir = base / "vits-piper-zh_CN-chaowen-medium"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = Path(args.text_file).read_text(encoding="utf-8").strip()
    chunks = split_sentences(text)
    env = os.environ.copy()
    lib_dir = base / "sherpa-onnx-v1.13.3-linux-riscv64-spacemit-shared/lib"
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads)
    env["ORT_NUM_THREADS"] = str(args.threads)

    print(f"chunks={len(chunks)} chars={len(text)}")
    total_synth = 0.0
    total_audio = 0.0
    first_ready = None
    all_start = time.monotonic()
    for i, chunk in enumerate(chunks, 1):
        out = output_dir / f"chunk_{i:03d}.wav"
        cmd = [
            str(bin_path),
            "--print-args=false",
            f"--num-threads={args.threads}",
            f"--tts-max-num-sentences={args.max_sentences}",
            f"--vits-model={model_dir / 'zh_CN-chaowen-medium.onnx'}",
            f"--vits-tokens={model_dir / 'tokens.txt'}",
            f"--vits-lexicon={model_dir / 'lexicon.txt'}",
            f"--tts-rule-fsts={model_dir / 'number.fst'},{model_dir / 'date.fst'},{model_dir / 'phone.fst'}",
            "--vits-length-scale=1.0",
            "--vits-noise-scale=0.667",
            "--vits-noise-scale-w=0.8",
            f"--output-filename={out}",
            chunk,
        ]
        start = time.monotonic()
        proc = run(cmd, env)
        elapsed = time.monotonic() - start
        if proc.returncode:
            print(proc.stdout, file=sys.stderr)
            return proc.returncode
        if first_ready is None:
            first_ready = time.monotonic() - all_start
        m = re.search(r"Audio duration:\s*([0-9.]+)", proc.stdout)
        audio_s = float(m.group(1)) if m else 0.0
        total_synth += elapsed
        total_audio += audio_s
        print(f"chunk={i} chars={len(chunk)} synth_s={elapsed:.2f} audio_s={audio_s:.2f} text={chunk}")
        if args.play:
            subprocess.run(["aplay", "-q", "-D", args.device, str(out)], check=False)

    total_wall = time.monotonic() - all_start
    print(
        f"summary chunks={len(chunks)} first_ready_s={first_ready:.2f} "
        f"total_wall_s={total_wall:.2f} total_synth_s={total_synth:.2f} "
        f"total_audio_s={total_audio:.2f} rtf={total_synth / total_audio:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
