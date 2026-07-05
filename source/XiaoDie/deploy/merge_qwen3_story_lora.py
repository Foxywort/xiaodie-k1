#!/usr/bin/env python3
"""Merge XiaoDie's Qwen3-4B story LoRA into a deployable HF model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="E:/Duanwu/models/Qwen3-4B")
    parser.add_argument("--adapter", default="E:/Duanwu/outputs/best_story_adapter_round2")
    parser.add_argument(
        "--output",
        default="D:/Spacemit/XiaoDie/deploy_artifacts/models/qwen3-4b-xiaodie-story-merged-fp16",
    )
    parser.add_argument("--max-shard-size", default="2GB")
    args = parser.parse_args()

    base = Path(args.base_model)
    adapter = Path(args.adapter)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"[merge] base={base}")
    print(f"[merge] adapter={adapter}")
    print(f"[merge] output={output}")

    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.eval()

    model.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output)

    metadata = {
        "base_model": str(base),
        "adapter": str(adapter),
        "output": str(output),
        "dtype": "float16",
        "format": "huggingface_merged",
    }
    (output / "xiaodie_merge_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[merge] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
