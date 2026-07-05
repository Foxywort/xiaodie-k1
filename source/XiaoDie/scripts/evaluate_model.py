#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROMPTS = [
    ("动物故事", "小兔子、朋友、分享"),
    ("动物故事", "小猫、诚实、道歉"),
    ("睡前故事", "月亮、勇气、晚安"),
    ("睡前故事", "星星、害怕、拥抱"),
    ("友谊故事", "积木、轮流、合作"),
    ("友谊故事", "花园、等待、礼貌"),
    ("勇气故事", "第一次上台、紧张、尝试"),
    ("勇气故事", "小雨伞、雷声、深呼吸"),
    ("诚实故事", "彩笔、承认、修好"),
    ("诚实故事", "饼干、说真话、原谅"),
    ("幼儿短故事", "刷牙、坚持、彩虹"),
    ("幼儿短故事", "洗手、泡泡、好习惯"),
    ("小学生故事", "作业、计划、休息"),
    ("小学生故事", "图书馆、安静、帮助"),
] * 3

BAD = r"色情|成人|血腥|杀|自杀|仇恨|政治宣传|身份证|银行卡|处方药|诊断|爬窗|玩火|插座|开煤气"


def chinese_ratio(text: str) -> float:
    return len(re.findall(r"[\u4e00-\u9fff]", text)) / max(1, len(text))


def repetition_score(text: str) -> float:
    chunks = [text[i : i + 12] for i in range(0, max(0, len(text) - 12), 6)]
    if not chunks:
        return 0.0
    return 1 - len(set(chunks)) / len(chunks)


def check_story(text: str, keywords: str) -> dict:
    zh = chinese_ratio(text)
    too_short = len(re.findall(r"[\u4e00-\u9fff]", text)) < 180
    unsafe = bool(re.search(BAD, text))
    repeat = repetition_score(text)
    keyword_hits = sum(k in text for k in re.split(r"[、,，\s]+", keywords) if k)
    complete = any(w in text[-120:] for w in ["最后", "从那以后", "故事结束", "晚安", "明天"]) and len(text) > 200
    score = 0
    score += 0.25 if zh > 0.65 else 0
    score += 0.20 if not too_short else 0
    score += 0.25 if not unsafe else 0
    score += 0.15 if repeat < 0.30 else 0
    score += 0.10 if keyword_hits >= 2 else 0
    score += 0.05 if complete else 0
    return {
        "chinese_ratio": round(zh, 3),
        "too_short": too_short,
        "unsafe": unsafe,
        "repetition_score": round(repeat, 3),
        "keyword_hits": keyword_hits,
        "complete_structure": complete,
        "score": round(score, 3),
    }


def build_prompt(tokenizer, category: str, keywords: str) -> str:
    messages = [
        {"role": "system", "content": "/no_think\n你是小蝶，只输出适合儿童阅读和朗读的中文故事正文，不要 Markdown，不要思考过程。"},
        {"role": "user", "content": f"请写一个{category}。关键词：{keywords}。故事要安全、温柔、完整，有教育意义。"},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages) + "\n<|assistant|>\n"


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"^\s*#{1,6}\s*", "", text)
    return text.strip()


def load_model(base_model: str, adapter: str | None, merged: bool):
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True) if torch.cuda.is_available() else None
    tokenizer = AutoTokenizer.from_pretrained(base_model if merged else adapter or base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True, quantization_config=quant, device_map="auto" if torch.cuda.is_available() else None, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model = base if merged or not adapter else PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Chinese children story model.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--merged", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=450)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_model = args.base_model or cfg["model_selection"].get("model_name_or_path")
    if base_model == "auto":
        base_model = "Qwen/Qwen3-4B"
    adapter = args.adapter or cfg["training"]["output_dir"]
    model, tokenizer = load_model(base_model, adapter, args.merged)

    samples = []
    for idx, (category, keywords) in enumerate(PROMPTS[: args.limit], start=1):
        prompt = build_prompt(tokenizer, category, keywords)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.45,
                top_p=0.85,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = clean(tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
        metrics = check_story(text, keywords)
        samples.append({"id": idx, "category": category, "keywords": keywords, "text": text, **metrics})

    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    md = ["# Generated Samples", ""]
    for s in samples:
        md += [f"## {s['id']}. {s['category']} - {s['keywords']}", "", f"score: {s['score']}", "", s["text"], ""]
    (reports / "generated_samples.md").write_text("\n".join(md), encoding="utf-8")

    avg = sum(s["score"] for s in samples) / max(1, len(samples))
    unsafe = sum(s["unsafe"] for s in samples)
    too_short = sum(s["too_short"] for s in samples)
    report = [
        "# Evaluation Report",
        "",
        f"- samples: {len(samples)}",
        f"- average_score: {avg:.3f}",
        f"- unsafe_count: {unsafe}",
        f"- too_short_count: {too_short}",
        f"- complete_structure_ratio: {sum(s['complete_structure'] for s in samples)/max(1,len(samples)):.3f}",
        f"- avg_chinese_ratio: {sum(s['chinese_ratio'] for s in samples)/max(1,len(samples)):.3f}",
        "",
        "## Notes",
        "",
        "Scores are heuristic. Human review is still required before product release.",
    ]
    (reports / "evaluation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"average_score={avg:.3f}, wrote reports")


if __name__ == "__main__":
    main()
