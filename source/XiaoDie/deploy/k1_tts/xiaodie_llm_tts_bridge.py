#!/usr/bin/env python3
"""Bridge streaming LLM text into the offline Chaowen TTS daemon.

No cloud API is used. This script only starts the local C daemon and forwards
text coming from stdin. Sentence segmentation is deliberately handled by the
daemon so normal Chinese sentences are never split by this bridge.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
END_PUNCT = "。！？!?……"


def pump_stderr(proc: subprocess.Popen[str]) -> None:
    assert proc.stderr is not None
    for line in proc.stderr:
        sys.stderr.write(line)
        sys.stderr.flush()


def normalize_piece(piece: str) -> str:
    piece = ANSI_RE.sub("", piece)
    piece = piece.replace("\r", "")
    # Paragraph breaks from the LLM should be audible pauses. If the previous
    # character is not sentence punctuation, a Chinese full stop is the safest
    # child-story default.
    piece = piece.replace("\n", "。")
    return piece


def send_piece(proc: subprocess.Popen[str], piece: str) -> None:
    if not piece or proc.stdin is None:
        return
    proc.stdin.write(piece + "\n")
    proc.stdin.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/vicky/xiaodie/tts")
    parser.add_argument("--device", default="hw:1,0")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--max-sentences", type=int, default=4)
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--chunk-chars", type=int, default=8)
    args = parser.parse_args()

    base = Path(args.base)
    daemon = base / "fast/bin/chaowen_tts_daemon"
    cmd = [str(daemon), "--device", args.device]
    if args.no_play:
        cmd.append("--no-play")

    env = os.environ.copy()
    lib = base / "sherpa-onnx-v1.13.3-linux-riscv64-spacemit-shared/lib"
    env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["XIAODIE_TTS_THREADS"] = str(args.threads)
    env["XIAODIE_TTS_SENTENCES"] = str(args.max_sentences)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )
    stderr_thread = threading.Thread(target=pump_stderr, args=(proc,), daemon=True)
    stderr_thread.start()

    buf = ""
    try:
        while True:
            ch = sys.stdin.read(1)
            if ch == "":
                break
            ch = normalize_piece(ch)
            if not ch:
                continue
            buf += ch
            if len(buf) >= args.chunk_chars or any(p in buf for p in END_PUNCT):
                send_piece(proc, buf)
                buf = ""
        if buf:
            send_piece(proc, buf)
        send_piece(proc, "::flush")
        send_piece(proc, "::quit")
    except KeyboardInterrupt:
        send_piece(proc, "::reset")
        send_piece(proc, "::quit")
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
