#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ZH_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]")

SYSTEM = (
    "你是小蝶，一个端侧中文儿童故事助手。"
    "你必须先阅读提供的RAG知识卡片，再写故事。"
    "只能使用知识卡片中出现的人物、地点、关系和设定；不要编造动画官方设定。"
    "故事必须适合幼儿园儿童朗读，温柔、安全、自然，有开头、发展、结尾。"
    "禁止武器、打败怪物、恐吓、伤害、危险模仿动作、拉耳朵、拉尾巴、抓身体、推搡、抢夺、成人议题和不适合儿童的内容。"
    "不要使用表情符号、颜文字、装饰符号或重复图案。"
    "不要输出解释、Markdown、资料来源列表或训练说明，只输出故事正文。"
)

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
    q_counts = Counter(query_tokens)
    for term, qf in q_counts.items():
        tf = doc_counts.get(term, 0)
        if tf <= 0:
            continue
        score += idf.get(term, 0.0) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1)))
    return score


def retrieve(cards: list[dict[str, Any]], index: dict[str, Any], query: str, top_k: int, franchise: str | None = None) -> list[dict[str, Any]]:
    q_tokens = tokens(query)
    scored = []
    doc_tokens = index["doc_tokens"]
    for idx, card in enumerate(cards):
        if franchise and card.get("franchise") != franchise:
            continue
        score = score_bm25(q_tokens, doc_tokens[idx], index["idf"], index["avgdl"])
        score *= float(card.get("weight", 1.0))
        if score > 0 or card.get("source_type") in {"safety_policy", "alias_policy"}:
            scored.append((score, card))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [card for score, card in scored[:top_k]]
    priority_cards = (
        [
            c
            for c in cards
            if c.get("franchise") == franchise and c.get("source_type") in {"curated_fact", "safety_policy", "alias_policy"}
        ]
        if franchise
        else []
    )
    ordered = []
    for card in priority_cards + selected:
        if card not in selected:
            ordered.append(card)
        elif card not in ordered:
            ordered.append(card)
    return ordered[: max(top_k, len(priority_cards))]


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = text.replace("**", "").replace("---", "")
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"([🌸✨🌱❤❤️]+)", "", text)
    text = re.sub(r"([。！？!?，,、])\1{2,}", r"\1", text)
    text = re.sub(r"(.{2,12})\1{4,}", r"\1", text)
    text = re.sub(r"^(当然可以|好的|以下是|下面是)[^\n]*\n+", "", text.strip())
    text = re.sub(r"资料来源[:：].*$", "", text, flags=re.S)
    text = re.sub(r"如果需要.*$", "", text, flags=re.S)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def resolve_base_model(train_cfg: dict[str, Any]) -> Path:
    base = Path(train_cfg["model"]["base_model"])
    if base.exists():
        return base
    return Path(train_cfg["model"]["fallback_base_model"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-config", default="E:/Duanwu/configs/ip_rag_sources.yaml")
    parser.add_argument("--train-config", default="E:/Duanwu/configs/train.yaml")
    parser.add_argument("--adapter", default="E:/Duanwu/outputs/best_story_adapter_round2", help="LoRA adapter path, or 'none' to use the base model only.")
    parser.add_argument("--franchise", default=None, help="barbapapa, paw_patrol, octonauts, my_little_pony, peppa_pig")
    parser.add_argument("--query", required=True)
    parser.add_argument("--age", default="4-6岁")
    parser.add_argument("--style", default="睡前安抚")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-new-tokens", type=int, default=260)
    parser.add_argument("--max-new-tokens", type=int, default=650)
    parser.add_argument("--target-chars", type=int, default=650)
    parser.add_argument("--output", default="E:/Duanwu/reports/ip_rag_story_test.md")
    args = parser.parse_args()

    rag_cfg = yaml.safe_load(Path(args.rag_config).read_text(encoding="utf-8"))
    train_cfg = yaml.safe_load(Path(args.train_config).read_text(encoding="utf-8"))
    rag_dir = Path(rag_cfg["paths"]["rag_dir"])
    cards = load_jsonl(rag_dir / "ip_knowledge_cards.jsonl")
    index = json.loads((rag_dir / "ip_rag_index.json").read_text(encoding="utf-8"))
    top_k = args.top_k or int(rag_cfg["defaults"].get("top_k", 8))
    selected = retrieve(cards, index, args.query, top_k, args.franchise)
    if not selected:
        raise SystemExit("No RAG cards retrieved. Run collect_ip_corpus.py and build_ip_rag_kb.py first.")

    context = []
    for idx, card in enumerate(selected, start=1):
        context.append(
            f"[卡片{idx}] 动画={card.get('franchise_zh')} 来源={card.get('source')} 类型={card.get('source_type')}\n"
            f"标题：{card.get('title')}\n内容：{card.get('text')[:900]}"
        )
    user = (
        f"用户想听：{args.query}\n"
        f"年龄：{args.age}\n"
        f"风格：{args.style}\n"
        f"篇幅：请写成约{args.target_chars}个中文字符的完整故事，分成6到9个自然段，不要太短。\n"
        "下面是可使用的RAG知识卡片。请基于卡片写一个中文儿童故事。如果卡片没有提到的官方设定，不要补编。\n"
        "必须紧扣用户想听的关键词，不要把用户指定的物品或主题换成别的东西。"
        "例如用户说分享玩具，就围绕玩具分享，不要改成硬币、剑、怪物或战斗。\n\n"
        "核心事实卡的优先级最高。人物关系、职业身份和地点设定不得改写；例如卡片说乔治是弟弟，就不能写成哥哥或普通朋友。\n\n"
        "如果知识卡片说角色会变形，意思是角色把自己的身体变成某种形状来帮忙，不是凭空制造物品，也不是使用魔法棒。\n\n"
        "不要使用任何表情符号、emoji、颜文字或装饰图案。故事自然结束即可，不要用重复符号凑长度。\n\n"
        "故事结构必须清楚：先出现用户点名的人物和物品，再出现一个很小的日常问题，然后通过轮流、分享、合作或道歉解决，最后温柔收尾。\n\n"
        + "\n\n".join(context)
    )

    base = resolve_base_model(train_cfg)
    use_adapter = args.adapter.lower() not in {"none", "base", "no_adapter"}
    adapter = Path(args.adapter) if use_adapter else None
    if use_adapter and adapter and not adapter.exists():
        adapter = Path(train_cfg["model"]["best_adapter_output"])
    if use_adapter and adapter and not adapter.exists():
        adapter = Path(train_cfg["model"]["adapter_output"])

    tokenizer_source = adapter if (adapter and (adapter / "tokenizer_config.json").exists()) else base
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(str(base), trust_remote_code=True, quantization_config=quant, device_map="auto", torch_dtype=torch.float16)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
    with torch.no_grad():
        min_new_tokens = max(0, min(args.min_new_tokens, args.max_new_tokens - 32))
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            temperature=0.35,
            top_p=0.82,
            repetition_penalty=1.16,
            no_repeat_ngram_size=8,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    story = clean(tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("# IP RAG Story Test\n\n" + story + "\n\n## Retrieved Cards\n\n" + "\n\n".join(context), encoding="utf-8")
    print(story)
    print(f"\n[written] {output}")


if __name__ == "__main__":
    main()
