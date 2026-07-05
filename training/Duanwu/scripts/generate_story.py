#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM = (
    "你是小蝶，一个温柔的中文儿童故事助手。"
    "只输出故事正文，不要解释，不要标题装饰，不要Markdown，不要英文。"
    "故事要安全、自然、完整，适合儿童朗读。"
)


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = text.replace("**", "").replace("---", "")
    text = re.sub(r"^(当然可以|好的|以下是|下面是)[^\n]*\n+", "", text.strip())
    text = re.sub(r"这个故事可以.*$", "", text, flags=re.S)
    text = re.sub(r"如果需要.*$", "", text, flags=re.S)
    text = re.sub(r"\b(rabbit|maybe|careful|slow)\b", "", text, flags=re.I)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="E:/Duanwu/configs/train.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--keywords", required=True)
    parser.add_argument("--style", default="睡前安抚")
    parser.add_argument("--age", default="4-5岁")
    parser.add_argument("--max-new-tokens", type=int, default=520)
    parser.add_argument("--output", default="E:/Duanwu/reports/manual_story_test.md")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = Path(cfg["model"]["base_model"])
    if not base.exists():
        base = Path(cfg["model"]["fallback_base_model"])
    adapter = Path(args.adapter or cfg["model"]["best_adapter_output"])
    if not adapter.exists():
        adapter = Path(cfg["model"]["adapter_output"])

    tokenizer_source = adapter if (adapter / "tokenizer_config.json").exists() else base
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(str(base), trust_remote_code=True, quantization_config=quant, device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"请写一个适合{args.age}小朋友听的中文儿童故事。关键词：{args.keywords}。风格：{args.style}。请直接开始讲故事，最后自然结束。"},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.42,
            top_p=0.86,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    story = clean(tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
    Path(args.output).write_text("# Manual Story Test\n\n" + story + "\n", encoding="utf-8")
    print(story)


if __name__ == "__main__":
    main()
