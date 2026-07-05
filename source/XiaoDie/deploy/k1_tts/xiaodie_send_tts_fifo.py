#!/usr/bin/env python3
"""Send streaming LLM text to the already-running local Chaowen TTS service."""

from __future__ import annotations

import argparse
import os
import re
import sys


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
END_PUNCT = "。！？!?……"


def normalize_piece(piece: str) -> str:
    piece = ANSI_RE.sub("", piece)
    piece = piece.replace("\r", "")
    piece = piece.replace("\n", "。")
    return piece


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fifo", default="/home/vicky/xiaodie/tts/tts_input.fifo")
    parser.add_argument("--chunk-chars", type=int, default=8)
    parser.add_argument("--no-flush", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.fifo):
        print(f"FIFO does not exist: {args.fifo}", file=sys.stderr)
        return 2

    buf = ""
    with open(args.fifo, "w", encoding="utf-8", buffering=1) as fifo:
        while True:
            ch = sys.stdin.read(1)
            if ch == "":
                break
            ch = normalize_piece(ch)
            if not ch:
                continue
            buf += ch
            if len(buf) >= args.chunk_chars or any(p in buf for p in END_PUNCT):
                fifo.write(buf + "\n")
                fifo.flush()
                buf = ""
        if buf:
            fifo.write(buf + "\n")
        if not args.no_flush:
            fifo.write("::flush\n")
        fifo.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
