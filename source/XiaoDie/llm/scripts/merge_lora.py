#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge XiaoDie LoRA adapter into the base model.")
    parser.add_argument("--base-model", default="E:/xiaodie_models/SmolLM3-3B")
    parser.add_argument("--adapter", default="llm/outputs/smollm3-3b-story-qlora")
    parser.add_argument("--output", default="llm/outputs/smollm3-3b-story-merged-fp16")
    parser.add_argument("--device-map", default="cpu", choices=["cpu", "auto"])
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(output)
    print(f"merged model saved to {output}")


if __name__ == "__main__":
    main()
