#!/usr/bin/env bash
set -euo pipefail

export LLAMA_CLI="${LLAMA_CLI:-/usr/bin/llama-cli}"
export LD_LIBRARY_PATH="/usr/lib/riscv64-linux-gnu:/home/vicky/xiaodie/llama/lib:${LD_LIBRARY_PATH:-}"

python3 /home/vicky/xiaodie/llm/xiaodie_rag_llm.py "$@"
