#!/usr/bin/env python3
"""K1-side RAG story generation with sentence-level streaming to Chaowen TTS.

This runner is offline-only. It calls the local Ollama service on the K1 board,
streams generated text, and forwards only complete Chinese sentences to the
already-running Chaowen FIFO service. Normal sentences are never cut in half.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, TextIO


ZH_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U0001F1E6-\U0001F1FF"
    "\U0000200D"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)

SENTENCE_ENDINGS = "。！？!?……"
CLOSERS = "”’）)]】》」』\"'"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text: str) -> list[str]:
    text = (text or "").lower()
    toks = WORD_RE.findall(text)
    compact_zh = "".join(ZH_RE.findall(text))
    toks.extend(compact_zh[i : i + 2] for i in range(max(0, len(compact_zh) - 1)))
    return [t for t in toks if t.strip()]


def score_bm25(query_tokens: list[str], doc_counts: dict[str, int], idf: dict[str, float], avgdl: float) -> float:
    k1 = 1.5
    b = 0.75
    dl = sum(doc_counts.values()) or 1
    score = 0.0
    for term in Counter(query_tokens):
        tf = doc_counts.get(term, 0)
        if tf <= 0:
            continue
        score += idf.get(term, 0.0) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1)))
    return score


def retrieve(cards: list[dict[str, Any]], index: dict[str, Any], query: str, top_k: int, franchise: str | None) -> list[dict[str, Any]]:
    q_tokens = tokens(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    doc_tokens = index["doc_tokens"]
    for idx, card in enumerate(cards):
        if franchise and card.get("franchise") != franchise:
            continue
        score = score_bm25(q_tokens, doc_tokens[idx], index["idf"], index["avgdl"])
        score *= float(card.get("weight", 1.0))
        if score > 0 or card.get("source_type") in {"safety_policy", "alias_policy"}:
            scored.append((score, card))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [card for _, card in scored[:top_k]]
    priority = (
        [
            card
            for card in cards
            if card.get("franchise") == franchise
            and card.get("source_type") in {"curated_fact", "safety_policy", "alias_policy"}
        ]
        if franchise
        else []
    )
    ordered: list[dict[str, Any]] = []
    for card in priority + selected:
        if card not in ordered:
            ordered.append(card)
    return ordered[: max(top_k, len(priority))]


def build_prompt(cards: list[dict[str, Any]], query: str, age: str, style: str, target_chars: int, card_chars: int) -> tuple[str, str]:
    contexts = []
    for idx, card in enumerate(cards, start=1):
        contexts.append(
            f"[卡片{idx}] 动画={card.get('franchise_zh')} 来源={card.get('source')} 类型={card.get('source_type')}\n"
            f"标题：{card.get('title')}\n内容：{str(card.get('text', ''))[:card_chars]}"
        )
    user_prompt = (
        "下面是可使用的RAG知识卡片。请基于卡片写一个中文儿童故事。如果卡片没有提到的官方设定，不要补编。\n"
        "必须紧扣用户点名的人物、物品和主题，不要把用户指定的内容换成别的东西。\n"
        "第一段必须直接出现用户指定的人物和事件。不要把整理玩具、分享、轮流这类主题换成电脑、机器、维修、陌生动物或无关冒险。\n"
        "核心事实卡的优先级最高。人物关系、职业身份、地点设定不得改写。\n"
        "不要把官方朋友写成宠物，不要让角色给官方角色重新取名，不要凭空增加危险任务或反派冲突。\n"
        "如果用户说整理、分享、轮流、道歉或合作，故事必须围绕这些日常行为解决问题。\n"
        "如果知识卡片说角色会变形，意思是角色把自己的身体变成某种形状来帮忙，不是凭空制造物品，也不是使用魔法棒。\n"
        "不要使用任何表情符号、emoji、颜文字或装饰图案。自然结尾即可。\n"
        "不要写“故事开始”“故事结束”“这个故事告诉我们”这类说明句。\n"
        "为了语音播放流畅，请使用较短的中文句子，每句话表达一个清楚动作或想法。\n\n"
        + "\n\n".join(contexts)
        + "\n\n本次用户想听："
        + query
        + f"\n年龄：{age}\n风格：{style}\n篇幅：请写成约{target_chars}个中文字符的完整故事，分成3到5个自然段。"
    )
    return user_prompt, "\n\n".join(contexts)


def normalize_text(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    text = text.replace("\r", "")
    text = text.replace("**", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    return text


class SentenceTtsWriter:
    def __init__(self, fifo: str | None, min_chars: int = 18) -> None:
        self.fifo_path = fifo
        self.min_chars = min_chars
        self.buf = ""
        self.fifo: TextIO | None = None
        if fifo:
            if not os.path.exists(fifo):
                raise FileNotFoundError(f"TTS FIFO does not exist: {fifo}")
            self.fifo = open(fifo, "w", encoding="utf-8", buffering=1)

    def close(self) -> None:
        if self.fifo:
            self.flush(force=True)
            self.fifo.write("::flush\n")
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
        self.flush(force=False)

    def flush(self, force: bool) -> None:
        if not self.fifo:
            return
        while True:
            end = self._ready_end(force=force)
            if end <= 0:
                return
            segment = self.buf[:end].strip()
            self.buf = self.buf[end:].strip()
            if segment:
                segment = re.sub(r"[。！？!?]{2,}", lambda m: m.group(0)[0], segment)
                segment = re.sub(r"^故事开始[:：。]?", "", segment).strip()
                if segment in {"故事结束。", "故事结束", "这个故事结束了。"}:
                    continue
                self.fifo.write(segment + "\n")
                self.fifo.flush()

    def _ready_end(self, force: bool) -> int:
        if not self.buf:
            return 0
        chosen = 0
        i = 0
        while i < len(self.buf):
            ch = self.buf[i]
            if ch in SENTENCE_ENDINGS:
                end = i + 1
                while end < len(self.buf) and self.buf[end] in CLOSERS:
                    end += 1
                if len(self.buf[:end].strip()) >= self.min_chars:
                    chosen = end
                    break
            i += 1
        if chosen:
            return chosen
        if force:
            return len(self.buf)
        return 0


def ollama_stream(url: str, payload: dict[str, Any]):
    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=None) as resp:
        for raw in resp:
            if not raw.strip():
                continue
            yield json.loads(raw.decode("utf-8", errors="replace"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="/home/vicky/xiaodie/rag")
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default="xiaodie-story-1.5b:latest")
    parser.add_argument("--franchise", default=None)
    parser.add_argument("--query", required=True)
    parser.add_argument("--age", default="4-6岁")
    parser.add_argument("--style", default="睡前安抚")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--card-chars", type=int, default=260)
    parser.add_argument("--target-chars", type=int, default=650)
    parser.add_argument("--max-new-tokens", type=int, default=650)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--temp", type=float, default=0.18)
    parser.add_argument("--top-p", type=float, default=0.72)
    parser.add_argument("--repeat-penalty", type=float, default=1.16)
    parser.add_argument("--tts-fifo", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--print-cards", action="store_true")
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)
    cards = load_jsonl(rag_dir / "ip_knowledge_cards.jsonl")
    index = json.loads((rag_dir / "ip_rag_index.json").read_text(encoding="utf-8"))
    selected = retrieve(cards, index, args.query, args.top_k, args.franchise)
    if not selected:
        raise SystemExit("No RAG cards retrieved.")

    prompt, contexts = build_prompt(selected, args.query, args.age, args.style, args.target_chars, args.card_chars)
    payload = {
        "model": args.model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "num_ctx": args.ctx_size,
            "num_predict": args.max_new_tokens,
            "temperature": args.temp,
            "top_p": args.top_p,
            "repeat_penalty": args.repeat_penalty,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        },
    }

    started = time.time()
    first_token_s: float | None = None
    story_parts: list[str] = []
    tts = SentenceTtsWriter(args.tts_fifo)
    try:
        for item in ollama_stream(args.ollama_url, payload):
            piece = normalize_text(item.get("response", ""))
            if piece:
                if first_token_s is None:
                    first_token_s = time.time() - started
                    print(f"\n[ttft_s={first_token_s:.2f}]", file=sys.stderr)
                story_parts.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
                tts.write(piece)
            if item.get("done"):
                stats = item
                break
        else:
            stats = {}
    finally:
        tts.close()

    story = normalize_text("".join(story_parts)).strip()
    print("", flush=True)
    elapsed = time.time() - started
    eval_count = int(stats.get("eval_count") or 0)
    eval_duration = int(stats.get("eval_duration") or 0)
    prompt_eval_count = int(stats.get("prompt_eval_count") or 0)
    prompt_eval_duration = int(stats.get("prompt_eval_duration") or 0)
    tps = eval_count / (eval_duration / 1e9) if eval_count and eval_duration else 0.0
    prompt_tps = (
        prompt_eval_count / (prompt_eval_duration / 1e9)
        if prompt_eval_count and prompt_eval_duration
        else 0.0
    )
    print(
        f"[done elapsed_s={elapsed:.1f} tokens={eval_count} tps={tps:.2f} "
        f"prompt_tokens={prompt_eval_count} prompt_tps={prompt_tps:.2f} prompt_chars={len(prompt)}]",
        file=sys.stderr,
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        body = "# XiaoDie K1 1.5B RAG Streaming Story Test\n\n" + story + "\n"
        if args.print_cards:
            body += "\n## Retrieved Cards\n\n" + contexts + "\n"
        output.write_text(body, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
