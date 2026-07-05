#!/usr/bin/env bash
set -euo pipefail

BASE="${XIAODIE_BASE:-/home/vicky/xiaodie}"
TTS_BASE="${XIAODIE_TTS_BASE:-$BASE/tts}"
SRC="${1:-$BASE/asr/xiaodie_asr_daemon.c}"
OUT="${2:-$BASE/asr/bin/xiaodie_asr_daemon}"

mkdir -p "$(dirname "$OUT")"
gcc -O3 -pipe -pthread \
  -I"$TTS_BASE/fast/include" \
  -L"$TTS_BASE/sherpa-onnx-v1.13.3-linux-riscv64-spacemit-shared/lib" \
  -Wl,-rpath,"$TTS_BASE/sherpa-onnx-v1.13.3-linux-riscv64-spacemit-shared/lib" \
  -o "$OUT" "$SRC" \
  -lsherpa-onnx-c-api -lm

echo "ASR daemon built: $OUT"
