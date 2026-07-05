#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_SYSTEM = (
    "你是“小蝶”，一个运行在端侧设备上的幼儿园故事 AI 助手。"
    "请根据关键词生成适合 3-6 岁儿童收听的中文原创故事，语言温柔、句子短、适合 TTS 朗读。"
)


def read_system_prompt(path: str | None) -> str:
    if not path:
        return DEFAULT_SYSTEM
    prompt_path = Path(path)
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM


def build_prompt(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": "/no_think\n" + system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"


def clean_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<\|im_end\|>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*(故事正文[:：]\s*)", "", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a XiaoDie story with the trained LoRA adapter.")
    parser.add_argument("--base-model", default="E:/xiaodie_models/SmolLM3-3B")
    parser.add_argument("--adapter", default="llm/outputs/smollm3-3b-story-qlora")
    parser.add_argument("--merged", action="store_true", help="Load --base-model as an already merged model.")
    parser.add_argument("--system-prompt", default="llm/prompts/story_system_prompt.md")
    parser.add_argument("--keywords", default="月亮、勇气、分享")
    parser.add_argument("--style", default="睡前安抚")
    parser.add_argument("--age", default="4-5 岁")
    parser.add_argument("--minutes", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=360)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer_path = args.base_model if args.merged else args.adapter
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = base_model if args.merged else PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    user_prompt = (
        f"请给 {args.age} 小朋友讲一个{args.style}风格的原创中文故事。"
        f"关键词：{args.keywords}。"
        f"时长约 {args.minutes} 分钟，适合语音朗读，结尾要温暖。"
        "不要输出 Markdown 标记。不要出现打开冰箱、钥匙、爬高、用电、药物、独自外出等危险模仿情节。"
    )
    prompt = build_prompt(tokenizer, read_system_prompt(args.system_prompt), user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[-1] :]
    print(clean_output(tokenizer.decode(generated, skip_special_tokens=True)))


if __name__ == "__main__":
    main()
