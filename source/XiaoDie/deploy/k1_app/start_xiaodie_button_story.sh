#!/usr/bin/env bash
set -euo pipefail

BASE="${XIAODIE_BASE:-/home/vicky/xiaodie}"
GPIO="${XIAODIE_BUTTON_GPIO:-35}"
ACTIVE="${XIAODIE_BUTTON_ACTIVE:-high}"

ACTIVE_ARG="--active-high"
if [ "$ACTIVE" = "low" ]; then
  ACTIVE_ARG="--active-low"
fi

exec python3 "$BASE/app/xiaodie_button_story.py" \
  --gpio "$GPIO" \
  "$ACTIVE_ARG" \
  "$@"
