#!/usr/bin/env python3
"""DeepSeek API story generation streamed sentence-by-sentence to Chaowen TTS.

The LLM runs in the cloud through the DeepSeek OpenAI-compatible API, while
RAG retrieval and Chaowen Full TTS stay on the K1 board. Only complete Chinese
sentences are sent to the TTS FIFO, so normal speech is not cut mid-sentence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from xiaodie_rag_ollama_stream import (
    CLOSERS,
    SENTENCE_ENDINGS,
    build_prompt,
    load_jsonl,
    normalize_text,
    retrieve,
)


PLAY_WRITE_RE = re.compile(r"play_write .*?\baudio_s=([0-9.]+).*?\btext=(.*)$")


SYSTEM_PROMPT = (
    "你是小蝶，一个给幼儿园小朋友讲故事的中文语音 AI 助手。"
    "你必须写安全、温柔、适合儿童朗读的故事。"
    "不要输出成人、暴力、色情、仇恨、政治宣传、医疗建议、隐私信息或危险模仿动作。"
    "不要输出表情符号、项目符号、Markdown 标题或解释性说明。"
    "为了语音播放流畅，每句话尽量短，一句话只表达一个动作或想法。"
    "请直接开始讲故事。"
)


def ui_progress(message: str) -> None:
    print(f"[xiaodie_progress] {message}", flush=True)


def ui_played(text: str) -> None:
    text = normalize_text(text).strip()
    if text:
        print(f"[xiaodie_played] {text}", flush=True)


def ui_tts_done() -> None:
    print("[xiaodie_tts_done]", flush=True)


def file_size(path: str) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def wait_tts_playback(log_path: str, marker: str, offset: int, timeout_s: float) -> None:
    """Follow TTS logs and emit UI text after each audio chunk should finish.

    The daemon writes chunks to an ffmpeg/aplay pipe as fast as the pipe accepts
    them. We keep a playback clock from each chunk's audio duration, so the UI
    follows the heard story instead of the much faster LLM text stream.
    """
    path = Path(log_path)
    deadline = time.time() + timeout_s
    pending: list[tuple[float, str]] = []
    play_cursor = time.time()
    saw_marker = False
    last_wait_hint = 0.0

    def emit_due(force: bool = False) -> None:
        nonlocal pending
        now = time.time()
        while pending and (force or pending[0][0] <= now):
            _, text = pending.pop(0)
            ui_played(text)

    while time.time() < deadline:
        if not path.exists():
            time.sleep(0.2)
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            while time.time() < deadline:
                emit_due()
                line = f.readline()
                if not line:
                    if saw_marker and not pending:
                        ui_tts_done()
                        return
                    now = time.time()
                    if now - last_wait_hint > 6.0:
                        ui_progress("小蝶正在把故事变成声音，请再等一下。")
                        last_wait_hint = now
                    time.sleep(0.1)
                    continue
                offset = f.tell()
                m = PLAY_WRITE_RE.search(line)
                if m:
                    try:
                        audio_s = float(m.group(1))
                    except ValueError:
                        audio_s = 0.0
                    text = m.group(2).strip()
                    play_cursor = max(play_cursor, time.time()) + max(0.0, audio_s)
                    pending.append((play_cursor, text))
                if f"mark_done" in line and f"token={marker}" in line:
                    saw_marker = True
    emit_due(force=True)
    ui_tts_done()


def read_api_key(path: str | None) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    if path:
        p = Path(path).expanduser()
        if p.exists():
            key = p.read_text(encoding="utf-8").strip()
            if key:
                return key
    raise SystemExit(
        "Missing DeepSeek API key. Set DEEPSEEK_API_KEY or create the key file."
    )


def deepseek_stream(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> Any:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "thinking": {"type": "disabled"},
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    item = json.loads(data)
                except json.JSONDecodeError:
                    continue
                yield item
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"DeepSeek HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"DeepSeek network error: {exc}") from exc


def extract_delta(item: dict[str, Any]) -> str:
    choices = item.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


class GroupedSentenceTtsWriter:
    def __init__(
        self,
        fifo: str | None,
        *,
        min_chars: int = 30,
        max_chars: int = 90,
        min_sentences: int = 2,
        max_sentences: int = 3,
    ) -> None:
        self.fifo_path = fifo
        self.min_chars = min_chars
        self.max_chars = max(max_chars, min_chars)
        self.min_sentences = max(1, min_sentences)
        self.max_sentences = max(self.min_sentences, max_sentences)
        self.buf = ""
        self.sentences: list[str] = []
        self.fifo = None
        if fifo:
            if not os.path.exists(fifo):
                raise FileNotFoundError(f"TTS FIFO does not exist: {fifo}")
            self.fifo = open(fifo, "w", encoding="utf-8", buffering=1)

    def close(self, marker: str | None = None) -> None:
        if self.fifo:
            self._extract_complete_sentences(force=True)
            self._emit(force=True)
            self.fifo.write("::flush\n")
            if marker:
                self.fifo.write(f"::mark {marker}\n")
            self.fifo.flush()
            self.fifo.close()

    def write(self, piece: str) -> None:
        if not self.fifo:
            return
        for ch in normalize_text(piece):
            if ch == "\r":
                continue
            if ch == "\n":
                if self.buf and self.buf[-1] not in SENTENCE_ENDINGS + " \t":
                    self.buf += "。"
                continue
            if ch in "“”‘’":
                continue
            self.buf += ch
        self._extract_complete_sentences(force=False)
        self._emit(force=False)

    def _extract_complete_sentences(self, force: bool) -> None:
        while True:
            end = self._sentence_end(force=force)
            if end <= 0:
                return
            sentence = self.buf[:end].strip()
            self.buf = self.buf[end:].strip()
            if not sentence:
                continue
            sentence = sentence.replace("故事开始：", "").replace("故事开始:", "").strip()
            sentence = sentence.replace("故事结束。", "").strip()
            if sentence:
                self.sentences.append(sentence)

    def _sentence_end(self, force: bool) -> int:
        for i, ch in enumerate(self.buf):
            if ch in SENTENCE_ENDINGS:
                end = i + 1
                while end < len(self.buf) and self.buf[end] in CLOSERS:
                    end += 1
                return end
        if force and self.buf.strip():
            if self.buf[-1] not in SENTENCE_ENDINGS:
                self.buf += "。"
            return len(self.buf)
        return 0

    def _emit(self, force: bool) -> None:
        if not self.fifo:
            return
        while self.sentences:
            count, chars = self._select_chunk(force=force)
            if count <= 0:
                return
            chunk = "".join(self.sentences[:count]).strip()
            del self.sentences[:count]
            if not chunk:
                continue
            self.fifo.write(chunk + "\n")
            self.fifo.flush()
            ui_progress("小蝶正在准备下一段声音。")
            print(
                f"[tts_chunk sentences={count} chars={chars}]",
                file=sys.stderr,
            )

    def _select_chunk(self, force: bool) -> tuple[int, int]:
        total = 0
        best_count = 0
        best_chars = 0
        for idx, sentence in enumerate(self.sentences, start=1):
            sentence_len = len(sentence)
            if idx > 1 and total + sentence_len > self.max_chars:
                break
            total += sentence_len
            if idx <= self.max_sentences:
                best_count = idx
                best_chars = total
            if idx >= self.max_sentences:
                break

        if force:
            return best_count, best_chars
        if best_count >= self.max_sentences:
            return best_count, best_chars
        if best_count >= self.min_sentences and best_chars >= self.min_chars:
            return best_count, best_chars
        if best_count >= 1 and best_chars >= self.max_chars:
            return best_count, best_chars
        return 0, 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="/home/vicky/xiaodie/rag")
    parser.add_argument("--api-key-file", default="/home/vicky/.config/xiaodie/deepseek_api_key")
    parser.add_argument("--endpoint", default=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/chat/completions"))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--franchise", default=None)
    parser.add_argument("--query", required=True)
    parser.add_argument("--age", default="4-6岁")
    parser.add_argument("--style", default="睡前安抚")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--card-chars", type=int, default=260)
    parser.add_argument("--target-chars", type=int, default=650)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--tts-fifo", default=None)
    parser.add_argument("--tts-min-chars", type=int, default=30)
    parser.add_argument("--tts-max-chars", type=int, default=90)
    parser.add_argument("--tts-min-sentences", type=int, default=2)
    parser.add_argument("--tts-max-sentences", type=int, default=3)
    parser.add_argument("--tts-log", default="/home/vicky/xiaodie/tts/tts_service.log")
    parser.add_argument("--wait-tts-playback", action="store_true")
    parser.add_argument("--tts-wait-timeout", type=float, default=360.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--print-cards", action="store_true")
    args = parser.parse_args()

    api_key = read_api_key(args.api_key_file)
    rag_dir = Path(args.rag_dir)
    cards = load_jsonl(rag_dir / "ip_knowledge_cards.jsonl")
    index = json.loads((rag_dir / "ip_rag_index.json").read_text(encoding="utf-8"))
    selected = retrieve(cards, index, args.query, args.top_k, args.franchise)
    if not selected:
        raise SystemExit("No RAG cards retrieved.")

    user_prompt, contexts = build_prompt(
        selected,
        args.query,
        args.age,
        args.style,
        args.target_chars,
        args.card_chars,
    )

    started = time.time()
    first_token_s: float | None = None
    story_parts: list[str] = []
    usage: dict[str, Any] | None = None
    marker = f"story_{int(started * 1000)}_{os.getpid()}" if args.wait_tts_playback and args.tts_fifo else None
    tts_log_offset = file_size(args.tts_log) if marker else 0
    raw_stdout = not bool(args.tts_fifo)
    tts = GroupedSentenceTtsWriter(
        args.tts_fifo,
        min_chars=args.tts_min_chars,
        max_chars=args.tts_max_chars,
        min_sentences=args.tts_min_sentences,
        max_sentences=args.tts_max_sentences,
    )
    ui_progress("小蝶正在想一个适合你的故事，请稍等。")
    try:
        for item in deepseek_stream(
            endpoint=args.endpoint,
            api_key=api_key,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
        ):
            if item.get("usage"):
                usage = item.get("usage")
            piece = normalize_text(extract_delta(item))
            if not piece:
                continue
            if first_token_s is None:
                first_token_s = time.time() - started
                print(f"\n[deepseek_ttft_s={first_token_s:.2f} model={args.model}]", file=sys.stderr)
                ui_progress("小蝶已经想好开头，正在准备讲出来。")
            story_parts.append(piece)
            if raw_stdout:
                sys.stdout.write(piece)
                sys.stdout.flush()
            tts.write(piece)
    finally:
        tts.close(marker)

    story = normalize_text("".join(story_parts)).strip()
    if raw_stdout:
        print("", flush=True)
    elif marker:
        ui_progress("小蝶已经写好故事，正在继续播放。")
        wait_tts_playback(args.tts_log, marker, tts_log_offset, args.tts_wait_timeout)
    elapsed = time.time() - started
    if usage:
        print(
            f"[done elapsed_s={elapsed:.1f} prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')} total_tokens={usage.get('total_tokens')}]",
            file=sys.stderr,
        )
    else:
        print(f"[done elapsed_s={elapsed:.1f}]", file=sys.stderr)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        body = "# XiaoDie DeepSeek API RAG Streaming Story Test\n\n" + story + "\n"
        if args.print_cards:
            body += "\n## Retrieved Cards\n\n" + contexts + "\n"
        output.write_text(body, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
