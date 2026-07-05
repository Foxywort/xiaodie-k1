#!/usr/bin/env bash
set -euo pipefail

TERMINAL="$(command -v x-terminal-emulator || command -v qterminal || true)"
if [ -z "$TERMINAL" ]; then
  echo "Cannot find terminal emulator."
  exit 1
fi

exec "$TERMINAL" -e bash -lc 'sudo -n /usr/local/bin/xiaodie-start; code=$?; echo; echo "小蝶已退出，退出码=$code"; read -r -p "按回车关闭窗口..."'
