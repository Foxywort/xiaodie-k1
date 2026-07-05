#!/usr/bin/env bash
set -euo pipefail

BASE="${XIAODIE_BASE:-/home/vicky/xiaodie}"
TTS_BASE="${XIAODIE_TTS_BASE:-$BASE/tts}"
FIFO="${XIAODIE_TTS_FIFO:-$TTS_BASE/tts_input.fifo}"

if [ ! -p "$FIFO" ] || ! pgrep -f "chaowen_tts_daemon" >/dev/null 2>&1; then
  "$TTS_BASE/start_chaowen_tts_service.sh"
fi

python3 "$BASE/llm/xiaodie_rag_ollama_stream.py" \
  --model "${XIAODIE_OLLAMA_MODEL:-xiaodie-story-1.5b:latest}" \
  --tts-fifo "$FIFO" \
  "$@"
