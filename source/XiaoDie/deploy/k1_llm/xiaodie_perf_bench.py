#!/usr/bin/env python3
"""Benchmark XiaoDie's K1-side LLM/RAG/TTS path.

The goal is to produce repeatable numbers for the competition demo:
TTFT, generated tokens/s, prompt-eval tokens/s, and optional TTS FIFO handoff.
No cloud API is used.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


SYSTEM = (
    "你是小蝶，一个端侧中文儿童故事助手。"
    "只输出适合幼儿园儿童朗读的中文故事正文。"
    "故事要温柔、安全、自然。句子短一些，适合语音播放。"
)


def call_ollama_stream(url: str, payload: dict) -> tuple[str, dict, float, float]:
    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    first_token_s: float | None = None
    parts: list[str] = []
    stats: dict = {}
    with urllib.request.urlopen(req, timeout=None) as resp:
        for raw in resp:
            if not raw.strip():
                continue
            item = json.loads(raw.decode("utf-8", errors="replace"))
            piece = item.get("response", "")
            if piece and first_token_s is None:
                first_token_s = time.time() - started
            if piece:
                parts.append(piece)
            if item.get("done"):
                stats = item
                break
    elapsed = time.time() - started
    return "".join(parts).strip(), stats, first_token_s or elapsed, elapsed


def make_chat_prompt(user_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        + SYSTEM
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + user_prompt
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def warmup(url: str, model: str, ctx_size: int) -> None:
    payload = {
        "model": model,
        "prompt": make_chat_prompt("请只回复：小蝶准备好了。"),
        "stream": False,
        "keep_alive": "60m",
        "options": {
            "num_ctx": ctx_size,
            "num_predict": 8,
            "temperature": 0.1,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        },
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=None) as resp:
        resp.read()


def run_case(url: str, model: str, name: str, prompt: str, args: argparse.Namespace) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "60m",
        "options": {
            "num_ctx": args.ctx_size,
            "num_predict": args.max_new_tokens,
            "temperature": args.temp,
            "top_p": args.top_p,
            "repeat_penalty": args.repeat_penalty,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        },
    }
    text, stats, ttft_s, elapsed_s = call_ollama_stream(url, payload)
    eval_count = int(stats.get("eval_count") or 0)
    eval_duration = int(stats.get("eval_duration") or 0)
    prompt_eval_count = int(stats.get("prompt_eval_count") or 0)
    prompt_eval_duration = int(stats.get("prompt_eval_duration") or 0)
    row = {
        "case": name,
        "ttft_s": round(ttft_s, 3),
        "elapsed_s": round(elapsed_s, 3),
        "eval_tokens": eval_count,
        "eval_tps": round(eval_count / (eval_duration / 1e9), 3) if eval_count and eval_duration else 0.0,
        "prompt_tokens": prompt_eval_count,
        "prompt_tps": round(prompt_eval_count / (prompt_eval_duration / 1e9), 3)
        if prompt_eval_count and prompt_eval_duration
        else 0.0,
        "chars": len(text),
        "text": text,
    }
    print(
        f"{name}: ttft={row['ttft_s']}s total={row['elapsed_s']}s "
        f"eval_tps={row['eval_tps']} prompt_tps={row['prompt_tps']} chars={row['chars']}",
        flush=True,
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xiaodie-story-1.5b:latest")
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--ctx-size", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temp", type=float, default=0.28)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repeat-penalty", type=float, default=1.14)
    parser.add_argument("--output", default="/home/vicky/xiaodie/reports/perf_bench_1_5b.json")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    if not args.no_warmup:
        print("warming model...", flush=True)
        warmup(args.ollama_url, args.model, args.ctx_size)

    cases = [
        (
            "pure_llm_short",
            make_chat_prompt("请写一个约80字的中文睡前小故事，关键词：月亮、分享。只输出故事正文。"),
        ),
        (
            "pure_llm_medium",
            make_chat_prompt("请写一个约160字的中文儿童故事，关键词：玩具、轮流、朋友。只输出故事正文。"),
        ),
    ]
    rows = [run_case(args.ollama_url, args.model, name, prompt, args) for name, prompt in cases]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model": args.model, "ctx_size": args.ctx_size, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written={output}")
    subprocess.run(["ollama", "ps"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
