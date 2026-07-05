#!/usr/bin/env bash
set -euo pipefail

BASE="/home/vicky/xiaodie/app"
PORT="${XIAODIE_WEB_PORT:-8765}"
LOG="/home/vicky/xiaodie/reports/xiaodie_web_app.log"
mkdir -p /home/vicky/xiaodie/reports

if ! pgrep -f "xiaodie_web_app.py --port $PORT" >/dev/null 2>&1; then
  nohup python3 "$BASE/xiaodie_web_app.py" --port "$PORT" >>"$LOG" 2>&1 &
fi

for _ in $(seq 1 40); do
  if python3 - "$PORT" <<'PY' >/dev/null 2>&1
import http.client, sys
port = int(sys.argv[1])
conn = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
conn.request("GET", "/api/status")
print(conn.getresponse().status)
PY
  then
    break
  fi
  sleep 0.2
done

BROWSER="$(command -v chromium-browser || command -v chromium || true)"
if [ -z "$BROWSER" ]; then
  exec xdg-open "http://127.0.0.1:$PORT/"
fi

exec "$BROWSER" \
  --app="http://127.0.0.1:$PORT/" \
  --window-size=1040,640 \
  --class=XiaoDieStory \
  --no-first-run \
  --disable-features=Translate \
  >/dev/null 2>&1
