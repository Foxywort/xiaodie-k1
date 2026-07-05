#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def apply_template(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    text = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
    return text + ("\n<|assistant|>\n" if add_generation_prompt else "")


class AssistantOnlyDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        messages = self.rows[idx]["messages"]
        prompt_text = apply_template(self.tokenizer, messages[:2], add_generation_prompt=True)
        full_text = apply_template(self.tokenizer, messages, add_generation_prompt=False)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        enc = self.tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=self.max_length)
        labels = enc["input_ids"].copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if all(x == -100 for x in labels):
            labels[-1] = enc["input_ids"][-1]
        enc["labels"] = labels
        return enc


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([x + [-100] * (max_len - len(x)) for x in labels], dtype=torch.long)
        return batch


def resolve_base_model(cfg: dict[str, Any]) -> str:
    base = Path(cfg["model"]["base_model"])
    fallback = Path(cfg["model"]["fallback_base_model"])
    if base.exists():
        return str(base)
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError(f"No base model found: {base} or {fallback}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="E:/Duanwu/configs/train.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    if args.max_seq_length is not None:
        cfg["data"]["max_seq_length"] = args.max_seq_length
    if args.output_dir is not None:
        cfg["model"]["adapter_output"] = args.output_dir
    if args.smoke:
        cfg["training"]["max_steps"] = min(5, int(cfg["training"]["max_steps"]))

    base_model = resolve_base_model(cfg)
    output_dir = Path(cfg["model"]["adapter_output"])
    print(f"base_model={base_model}")
    print(f"output_dir={output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg["training"].get("gradient_checkpointing", False))
    lora = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    train_rows = read_jsonl(Path(cfg["data"]["train_file"]), 60 if args.smoke else None)
    eval_rows = read_jsonl(Path(cfg["data"]["eval_file"]), 20 if args.smoke else None)
    if not train_rows or not eval_rows:
        raise SystemExit("Missing SFT train/eval files. Run collect, clean, and build_sft first.")

    max_len = int(cfg["data"]["max_seq_length"])
    train_ds = AssistantOnlyDataset(train_rows, tokenizer, max_len)
    eval_ds = AssistantOnlyDataset(eval_rows, tokenizer, max_len)
    t = cfg["training"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        max_steps=t["max_steps"],
        warmup_ratio=t["warmup_ratio"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=t.get("keep_last_checkpoints", 4),
        fp16=t.get("fp16", True),
        bf16=False,
        optim=t.get("optim", "adamw_torch"),
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=t.get("gradient_checkpointing", False),
        seed=cfg["project"]["seed"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=Collator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "round2_training_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved adapter to {output_dir}")


if __name__ == "__main__":
    main()
