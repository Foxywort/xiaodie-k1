#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from pathlib import Path

import yaml


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="E:/Duanwu/configs/train.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(cfg["model"]["adapter_output"])
    best_dir = Path(cfg["model"]["best_adapter_output"])
    best = None
    best_loss = None
    for cp in sorted(output_dir.glob("checkpoint-*"), key=checkpoint_step):
        state = cp / "trainer_state.json"
        if not state.exists():
            continue
        obj = json.loads(state.read_text(encoding="utf-8"))
        evals = [x for x in obj.get("log_history", []) if "eval_loss" in x]
        if not evals:
            continue
        loss = evals[-1]["eval_loss"]
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best = cp
    if best is None:
        best = output_dir
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best, best_dir)
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
    ]:
        src = output_dir / name
        dst = best_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    report = (
        "# Best Checkpoint Report\n\n"
        f"- source: {best}\n"
        f"- eval_loss: {best_loss}\n"
        f"- copied_to: {best_dir}\n"
    )
    Path("E:/Duanwu/reports/best_checkpoint.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
