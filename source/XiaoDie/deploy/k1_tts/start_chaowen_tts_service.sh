#!/usr/bin/env bash
set -euo pipefail

BASE="${XIAODIE_TTS_BASE:-/home/vicky/xiaodie/tts}"
FIFO="${XIAODIE_TTS_FIFO:-$BASE/tts_input.fifo}"
LOG="${XIAODIE_TTS_LOG:-$BASE/tts_service.log}"
PIDFILE="${XIAODIE_TTS_PIDFILE:-$BASE/tts_service.pid}"
DEVICE="${XIAODIE_TTS_DEVICE:-hw:CARD=sndes8326,DEV=0}"

mkdir -p "$BASE"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "TTS service already running: pid=$(cat "$PIDFILE") fifo=$FIFO"
  exit 0
fi

rm -f "$FIFO"
mkfifo "$FIFO"

export LD_LIBRARY_PATH="$BASE/sherpa-onnx-v1.13.3-linux-riscv64-spacemit-shared/lib:${LD_LIBRARY_PATH:-}"
export XIAODIE_TTS_THREADS="${XIAODIE_TTS_THREADS:-8}"
export XIAODIE_TTS_SENTENCES="${XIAODIE_TTS_SENTENCES:-5}"
export XIAODIE_TTS_CHUNK_MIN_CHARS="${XIAODIE_TTS_CHUNK_MIN_CHARS:-40}"
export XIAODIE_TTS_CHUNK_MAX_CHARS="${XIAODIE_TTS_CHUNK_MAX_CHARS:-140}"
NICE_BIN="$(command -v nice || true)"

(
  while true; do
    cat "$FIFO"
  done
) | ${NICE_BIN:+$NICE_BIN -n ${XIAODIE_TTS_NICE:--5}} "$BASE/fast/bin/chaowen_tts_daemon" --device "$DEVICE" >>"$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "TTS service started: pid=$(cat "$PIDFILE") fifo=$FIFO log=$LOG"
