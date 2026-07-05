#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "你是小蝶，一个温柔的中文儿童故事助手。"
    "只输出适合儿童朗读的中文故事正文。"
    "不要输出解释、提纲、Markdown、英文、训练说明或家长建议。"
    "故事必须安全、温柔、情节完整，有开头、发展和结尾。"
)


def build_messages(keywords: str, age: str, style: str, minutes: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"请直接写一个适合{age}小朋友听的中文儿童故事。"
                f"关键词：{keywords}。"
                f"风格：{style}。"
                f"长度约{minutes}分钟。"
                "故事里可以出现小蝶，但不要解释你在做什么。"
                "不要写“当然可以”“以下是”“这个故事可以用来”。"
                "最后自然地结束。"
            ),
        },
    ]


def clean_story(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = text.replace("**", "").replace("---", "")
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.M)
    text = re.sub(r"^(当然可以|好的|以下是|下面是)[^\n]*\n+", "", text.strip())
    text = re.sub(r"这个故事可以.*$", "", text, flags=re.S)
    text = re.sub(r"如果需要.*$", "", text, flags=re.S)
    text = re.sub(r"希望.*喜欢.*$", "", text, flags=re.S)
    replacements = {
        "rabbit": "小兔子",
        "maybe": "也许",
        "very careful": "很小心",
        "very slow": "很慢",
    }
    for src, dst in replacements.items():
        text = re.sub(src, dst, text, flags=re.I)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"^(适合|内容积极|语言简单|亲子阅读|练习中文)", stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Chinese children story with the trained Qwen LoRA.")
    parser.add_argument("--base-model", default=r"E:\xiaodie_models\Qwen3-4B")
    parser.add_argument("--adapter", default=r"outputs\best_story_adapter")
    parser.add_argument("--keywords", required=True)
    parser.add_argument("--age", default="4-5岁")
    parser.add_argument("--style", default="睡前安抚")
    parser.add_argument("--minutes", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=520)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--output", default="reports/manual_story_test_cleaned.md")
    args = parser.parse_args()

    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if torch.cuda.is_available()
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        quantization_config=quant,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    try:
        prompt = tokenizer.apply_chat_template(
            build_messages(args.keywords, args.age, args.style, args.minutes),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            build_messages(args.keywords, args.age, args.style, args.minutes),
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.82,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    story = clean_story(raw)
    Path(args.output).write_text("# Cleaned Manual Story Test\n\n" + story + "\n", encoding="utf-8")
    print(story)


if __name__ == "__main__":
    main()
