#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def select_model(cfg: dict[str, Any], smoke: bool, override: str | None) -> tuple[str, list[dict[str, Any]]]:
    if override:
        return override, []
    ms = cfg["model_selection"]
    if smoke:
        return ms["smoke_model_name_or_path"], []
    if ms.get("model_name_or_path") and ms["model_name_or_path"] != "auto":
        return ms["model_name_or_path"], []

    preferred = ms.get("preferred_family", "").lower()
    vram = float(ms.get("vram_gb", 8))
    ranked = []
    for candidate in ms["candidates"]:
        if candidate.get("role") != "generative":
            candidate = {**candidate, "rejected_reason": f"role={candidate.get('role')} is not generative"}
            ranked.append(candidate)
            continue
        if candidate.get("license") not in {"apache-2.0", "mit", "cc-by-4.0"}:
            candidate = {**candidate, "rejected_reason": "license not preferred for release"}
            ranked.append(candidate)
            continue
        stability = 1.0 if candidate.get("estimated_qlora_vram_gb", 99) <= vram + 0.3 else 0.35
        preference = 0.15 if preferred and preferred in candidate["name"].lower() else 0.0
        score = (
            0.35 * candidate.get("chinese_score", 0)
            + 0.25 * candidate.get("story_score", 0)
            + 0.15 * candidate.get("context_length_score", 0)
            + 0.20 * stability
            + preference
        )
        ranked.append({**candidate, "selection_score": round(score, 4), "rejected_reason": ""})
    valid = [c for c in ranked if not c.get("rejected_reason")]
    valid.sort(key=lambda c: c["selection_score"], reverse=True)
    if not valid:
        raise RuntimeError("No valid generative model candidate found.")
    return valid[0]["name"], ranked


def write_model_selection_report(ranked: list[dict[str, Any]], selected: str) -> None:
    if not ranked:
        return
    lines = ["# Model Selection", "", f"Selected model: `{selected}`", "", "| Model | Role | License | Score | Decision | Notes |", "|---|---|---|---:|---|---|"]
    for row in sorted(ranked, key=lambda x: x.get("selection_score", -1), reverse=True):
        decision = row.get("rejected_reason") or ("selected" if row["name"] == selected else "candidate")
        lines.append(
            f"| {row['name']} | {row.get('role','')} | {row.get('license','')} | {row.get('selection_score','')} | {decision} | {row.get('notes','')} |"
        )
    Path("reports/model_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def format_messages(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages) + "\n<|end|>"


class SftDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: AutoTokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        text = format_messages(self.tokenizer, self.rows[index]["messages"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False)
        enc["labels"] = enc["input_ids"].copy()
        return enc


class CausalCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([x + [-100] * (max_len - len(x)) for x in labels], dtype=torch.long)
        return batch


def build_model(model_name: str, cfg: dict[str, Any], smoke: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = torch.cuda.is_available()
    quant = None
    if use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quant,
        device_map="auto" if use_4bit else None,
        torch_dtype=torch.float16 if use_4bit else torch.float32,
    )
    model.config.use_cache = False
    if use_4bit:
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
    return model, tokenizer


def save_loss_curve(output_dir: Path) -> None:
    state_path = output_dir / "trainer_state.json"
    if not state_path.exists():
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    points = [(x["step"], x["loss"]) for x in state.get("log_history", []) if "loss" in x]
    if not points:
        return
    xs, ys = zip(*points)
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("SFT loss")
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT LoRA/QLoRA training for Chinese children stories.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="Use first 50 samples and tiny model by default.")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    if args.max_steps is not None:
        cfg["training"]["smoke_max_steps" if args.smoke else "max_steps"] = args.max_steps
    if args.max_seq_length is not None:
        cfg["data"]["max_seq_length"] = args.max_seq_length
    if args.output_dir is not None:
        cfg["training"]["smoke_output_dir" if args.smoke else "output_dir"] = args.output_dir
    model_name, ranked = select_model(cfg, args.smoke, args.model)
    write_model_selection_report(ranked, model_name)
    print(f"selected model: {model_name}")

    data_cfg = cfg["data"]
    train_rows = read_jsonl(Path(data_cfg["train_file"]), 50 if args.smoke else None)
    eval_rows = read_jsonl(Path(data_cfg["eval_file"]), 10 if args.smoke else None)
    if not train_rows or not eval_rows:
        raise SystemExit("Missing train/eval rows. Run download, clean, deduplicate and build_jsonl first.")

    model, tokenizer = build_model(model_name, cfg, args.smoke)
    output_dir = Path(cfg["training"]["smoke_output_dir"] if args.smoke else cfg["training"]["output_dir"])
    max_steps = cfg["training"]["smoke_max_steps"] if args.smoke else cfg["training"]["max_steps"]
    max_seq_length = min(data_cfg["max_seq_length"], 256) if args.smoke else data_cfg["max_seq_length"]

    train_ds = SftDataset(train_rows, tokenizer, max_seq_length)
    eval_ds = SftDataset(eval_rows, tokenizer, max_seq_length)

    t = cfg["training"]
    args_train = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        num_train_epochs=t["epochs"],
        max_steps=max_steps,
        warmup_ratio=t["warmup_ratio"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy="steps",
        save_total_limit=3,
        fp16=t["fp16"] and torch.cuda.is_available(),
        bf16=t["bf16"] and torch.cuda.is_available(),
        optim=t["optim"],
        gradient_checkpointing=t["gradient_checkpointing"],
        report_to="none",
        remove_unused_columns=False,
        seed=cfg["project"]["seed"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    trainer = Trainer(model=model, args=args_train, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=CausalCollator(tokenizer))
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    save_loss_curve(output_dir)
    print(f"saved adapter to {output_dir}")


if __name__ == "__main__":
    main()
