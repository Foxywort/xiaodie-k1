#!/usr/bin/env bash
set -euo pipefail

BASE="${XIAODIE_TTS_BASE:-/home/vicky/xiaodie/tts}"
FIFO="${XIAODIE_TTS_FIFO:-$BASE/tts_input.fifo}"
PIDFILE="${XIAODIE_TTS_PIDFILE:-$BASE/tts_service.pid}"

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    sleep 0.5
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi

pkill -TERM -f 'chaowen_tts_daemon|ffmpeg.*s16le|aplay.*sndes8326|cat.*/home/vicky/xiaodie/tts/tts_input.fifo' 2>/dev/null || true
sleep 0.2
pkill -KILL -f 'chaowen_tts_daemon|ffmpeg.*s16le|aplay.*sndes8326|cat.*/home/vicky/xiaodie/tts/tts_input.fifo' 2>/dev/null || true

rm -f "$FIFO"
echo "TTS service stopped"
