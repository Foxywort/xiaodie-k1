#!/usr/bin/env bash
set -euo pipefail

BASE="${XIAODIE_BASE:-/home/vicky/xiaodie}"
TTS_BASE="${XIAODIE_TTS_BASE:-$BASE/tts}"
MODEL="${XIAODIE_OLLAMA_MODEL:-xiaodie-story-1.5b:latest}"
WARM_FRANCHISES="${XIAODIE_WARM_FRANCHISES:-peppa_pig}"

export XIAODIE_TTS_THREADS="${XIAODIE_TTS_THREADS:-4}"
export XIAODIE_TTS_SENTENCES="${XIAODIE_TTS_SENTENCES:-4}"
export XIAODIE_TTS_NICE="${XIAODIE_TTS_NICE:-8}"

if ! pgrep -f "chaowen_tts_daemon" >/dev/null 2>&1; then
  "$TTS_BASE/start_chaowen_tts_service.sh"
else
  echo "TTS service already running."
fi

python3 - <<PY
import json
import urllib.request

payload = {
    "model": "${MODEL}",
    "prompt": "请只回复：小蝶准备好了。",
    "stream": False,
    "keep_alive": "60m",
    "options": {
        "num_ctx": 1024,
        "num_predict": 8,
        "temperature": 0.1,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    },
}
req = urllib.request.Request(
    "http://127.0.0.1:11434/api/generate",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=None) as resp:
    data = json.loads(resp.read().decode("utf-8", errors="replace"))
print("LLM warmup:", data.get("response", "").strip())
PY

ollama ps || true
for franchise in $WARM_FRANCHISES; do
  echo "warming RAG prefix: $franchise"
  python3 "$BASE/llm/xiaodie_rag_ollama_stream.py" \
    --model "$MODEL" \
    --franchise "$franchise" \
    --query "整理玩具学会分享" \
    --age "4-6岁" \
    --style "睡前安抚" \
    --target-chars 20 \
    --max-new-tokens 8 \
    --ctx-size 1024 \
    --top-k 2 \
    --card-chars 80 \
    --output "$BASE/reports/runtime_warm_${franchise}.md" \
    >/tmp/xiaodie_warm_${franchise}.out \
    2>/tmp/xiaodie_warm_${franchise}.err || true
  cat /tmp/xiaodie_warm_${franchise}.err || true
done
echo "XiaoDie runtime is ready: model=$MODEL fifo=$TTS_BASE/tts_input.fifo"
