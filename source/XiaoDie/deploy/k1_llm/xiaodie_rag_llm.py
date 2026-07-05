#!/usr/bin/env python3
"""K1-side local RAG + GGUF story generation runner for XiaoDie."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ZH_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]")
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

SYSTEM = (
    "你是小蝶，一个端侧中文儿童故事助手。"
    "你必须先阅读提供的RAG知识卡片，再写故事。"
    "只能使用知识卡片中出现的人物、地点、关系和设定；不要编造动画官方设定。"
    "故事必须适合幼儿园儿童朗读，温柔、安全、自然，有开头、发展、结尾。"
    "禁止武器、打败怪物、恐吓、伤害、危险模仿动作、拉耳朵、拉尾巴、抓身体、推搡、抢夺、成人议题和不适合儿童的内容。"
    "不要使用表情符号、颜文字、装饰符号或重复图案。"
    "不要输出解释、Markdown、资料来源列表或训练说明，只输出故事正文。"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cards.append(json.loads(line))
    return cards


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
    q_counts = Counter(query_tokens)
    for term in q_counts:
        tf = doc_counts.get(term, 0)
        if tf <= 0:
            continue
        score += idf.get(term, 0.0) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1)))
    return score


def retrieve(
    cards: list[dict[str, Any]],
    index: dict[str, Any],
    query: str,
    top_k: int,
    franchise: str | None,
) -> list[dict[str, Any]]:
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

    priority_cards = (
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
    for card in priority_cards + selected:
        if card not in ordered:
            ordered.append(card)
    return ordered[: max(top_k, len(priority_cards))]


def clean_story(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = text.replace("**", "").replace("---", "")
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"([。！？!?，,、])\1{2,}", r"\1", text)
    text = re.sub(r"(.{2,12})\1{4,}", r"\1", text)
    text = re.sub(r"^(当然可以|好的|以下是|下面是)[^\n]*\n+", "", text.strip())
    text = re.sub(r"资料来源[:：].*$", "", text, flags=re.S)
    text = re.sub(r"如果需要.*$", "", text, flags=re.S)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def build_prompt(
    selected: list[dict[str, Any]],
    query: str,
    age: str,
    style: str,
    target_chars: int,
    card_chars: int,
) -> tuple[str, str, str]:
    contexts = []
    for idx, card in enumerate(selected, start=1):
        contexts.append(
            f"[卡片{idx}] 动画={card.get('franchise_zh')} 来源={card.get('source')} 类型={card.get('source_type')}\n"
            f"标题：{card.get('title')}\n内容：{str(card.get('text', ''))[:card_chars]}"
        )

    user = (
        f"用户想听：{query}\n"
        f"年龄：{age}\n"
        f"风格：{style}\n"
        f"篇幅：请写成约{target_chars}个中文字符的完整故事，分成6到9个自然段，不要太短。\n"
        "下面是可使用的RAG知识卡片。请基于卡片写一个中文儿童故事。如果卡片没有提到的官方设定，不要补编。\n"
        "必须紧扣用户想听的关键词，不要把用户指定的物品或主题换成别的东西。"
        "核心事实卡的优先级最高。人物关系、职业身份和地点设定不得改写。"
        "如果知识卡片说角色会变形，意思是角色把自己的身体变成某种形状来帮忙，不是凭空制造物品，也不是使用魔法棒。"
        "不要使用任何表情符号、emoji、颜文字或装饰图案。故事自然结束即可，不要用重复符号凑长度。"
        "故事结构必须清楚：先出现用户点名的人物和物品，再出现一个很小的日常问题，然后通过轮流、分享、合作或道歉解决，最后温柔收尾。\n\n"
        + "\n\n".join(contexts)
    )
    prompt = (
        "<|im_start|>system\n"
        + SYSTEM
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + user
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )
    return prompt, user, "\n\n".join(contexts)


def find_llama_cli(explicit: str | None) -> str:
    candidates = [
        explicit,
        os.environ.get("LLAMA_CLI"),
        # Prefer the SpacemiT-packaged llama.cpp build; it is compiled with RVV on K1.
        "/usr/bin/llama-cli",
        "/home/vicky/llama.cpp/build-scalar/bin/llama-cli",
        "/home/vicky/llama.cpp/build/bin/llama-cli",
        "/home/vicky/xiaodie/llama/bin/llama-cli",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("Cannot find llama-cli. Set --llama-cli or LLAMA_CLI.")


def normalize_ollama_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        return "http://127.0.0.1:11434"
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


def run_ollama_api(args: argparse.Namespace, prompt: str, user_prompt: str) -> subprocess.CompletedProcess[str]:
    payload: dict[str, Any] = {
        "model": args.ollama_model,
        "prompt": prompt if args.ollama_raw else user_prompt,
        "stream": False,
        "keep_alive": args.keep_alive,
        "options": {
            "num_ctx": args.ctx_size,
            "num_predict": args.max_new_tokens,
            "temperature": args.temp,
            "top_p": args.top_p,
            "repeat_penalty": args.repeat_penalty,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        },
    }
    if args.ollama_raw:
        payload["raw"] = True

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        normalize_ollama_url(args.ollama_url) + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=args.ollama_timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))

    elapsed = time.time() - started
    eval_count = int(data.get("eval_count") or 0)
    eval_duration = int(data.get("eval_duration") or 0)
    prompt_eval_count = int(data.get("prompt_eval_count") or 0)
    prompt_eval_duration = int(data.get("prompt_eval_duration") or 0)
    tokens_per_sec = eval_count / (eval_duration / 1e9) if eval_count and eval_duration else 0.0
    prompt_tokens_per_sec = (
        prompt_eval_count / (prompt_eval_duration / 1e9) if prompt_eval_count and prompt_eval_duration else 0.0
    )
    stderr = (
        f"[ollama] elapsed={elapsed:.1f}s eval_tokens={eval_count} eval_tps={tokens_per_sec:.3f} "
        f"prompt_tokens={prompt_eval_count} prompt_tps={prompt_tokens_per_sec:.3f}\n"
    )
    return subprocess.CompletedProcess(args=["ollama-api"], returncode=0, stdout=data.get("response", ""), stderr=stderr)


def write_tts_fifo(path: str | None, story: str) -> None:
    if not path:
        return
    fifo = Path(path)
    if not fifo.exists():
        print(f"[warn] TTS fifo not found: {fifo}", file=sys.stderr)
        return
    with fifo.open("w", encoding="utf-8") as f:
        f.write(story.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/vicky/xiaodie/models/qwen3-4b-xiaodie-story-Q4_K_M.gguf")
    parser.add_argument("--engine", choices=["ollama", "llama-cli"], default=os.environ.get("XIAODIE_LLM_ENGINE", "ollama"))
    parser.add_argument("--ollama-model", default=os.environ.get("XIAODIE_OLLAMA_MODEL", "xiaodie-story:latest"))
    parser.add_argument("--ollama-bin", default=os.environ.get("OLLAMA_BIN", "ollama"))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--ollama-timeout", type=int, default=1800)
    parser.add_argument("--ollama-raw", action="store_true")
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument("--rag-dir", default="/home/vicky/xiaodie/rag")
    parser.add_argument("--llama-cli", default=None)
    parser.add_argument("--franchise", default=None)
    parser.add_argument("--query", required=True)
    parser.add_argument("--age", default="4-6岁")
    parser.add_argument("--style", default="睡前安抚")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--card-chars", type=int, default=450)
    parser.add_argument("--target-chars", type=int, default=650)
    parser.add_argument("--max-new-tokens", type=int, default=650)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--temp", type=float, default=0.35)
    parser.add_argument("--top-p", type=float, default=0.82)
    parser.add_argument("--repeat-penalty", type=float, default=1.16)
    parser.add_argument("--output", default=None)
    parser.add_argument("--tts-fifo", default=None)
    parser.add_argument("--print-cards", action="store_true")
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)
    cards = load_jsonl(rag_dir / "ip_knowledge_cards.jsonl")
    index = json.loads((rag_dir / "ip_rag_index.json").read_text(encoding="utf-8"))
    selected = retrieve(cards, index, args.query, args.top_k, args.franchise)
    if not selected:
        raise SystemExit("No RAG cards retrieved.")

    prompt, user_prompt, contexts = build_prompt(
        selected,
        args.query,
        args.age,
        args.style,
        args.target_chars,
        max(120, min(args.card_chars, 900)),
    )
    env = os.environ.copy()
    if args.engine == "ollama":
        try:
            result = run_ollama_api(args, prompt, user_prompt)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"[warn] Ollama API failed, falling back to ollama run: {exc}", file=sys.stderr)
            cmd = [args.ollama_bin, "run", args.ollama_model, user_prompt]
            result = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", capture_output=True, env=env)
    else:
        llama_cli = find_llama_cli(args.llama_cli)
        cmd = [
            llama_cli,
            "-m",
            args.model,
            "-p",
            prompt,
            "-n",
            str(args.max_new_tokens),
            "-c",
            str(args.ctx_size),
            "--temp",
            str(args.temp),
            "--top-p",
            str(args.top_p),
            "--repeat-penalty",
            str(args.repeat_penalty),
            "--no-display-prompt",
        ]
        env["LD_LIBRARY_PATH"] = (
            "/home/vicky/llama.cpp/build-scalar/bin:/home/vicky/llama.cpp/build/bin:/home/vicky/xiaodie/llama/lib:"
            + env.get("LD_LIBRARY_PATH", "")
        )
        result = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", capture_output=True, env=env)
        if result.returncode != 0 and "--no-display-prompt" in cmd:
            cmd.remove("--no-display-prompt")
            result = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", capture_output=True, env=env)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)

    story = clean_story(result.stdout)
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    print(story)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        body = "# XiaoDie K1 RAG Story Test\n\n" + story + "\n"
        if args.print_cards:
            body += "\n## Retrieved Cards\n\n" + contexts + "\n"
        output.write_text(body, encoding="utf-8")
    write_tts_fifo(args.tts_fifo, story)


if __name__ == "__main__":
    main()
