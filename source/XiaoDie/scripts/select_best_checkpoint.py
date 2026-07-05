#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


def find_best(output_dir: Path) -> tuple[Path, float | None]:
    best = None
    best_loss = None
    for checkpoint in output_dir.glob("checkpoint-*"):
        state = checkpoint / "trainer_state.json"
        if not state.exists():
            continue
        obj = json.loads(state.read_text(encoding="utf-8"))
        losses = [x["eval_loss"] for x in obj.get("log_history", []) if "eval_loss" in x]
        if not losses:
            continue
        loss = losses[-1]
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best = checkpoint
    if best is None:
        # Fall back to final adapter directory.
        return output_dir, None
    return best, best_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best LoRA checkpoint by eval_loss.")
    parser.add_argument("--output-dir", default="outputs/story-qwen-lora")
    parser.add_argument("--best-dir", default="outputs/best_story_adapter")
    args = parser.parse_args()

    source, loss = find_best(Path(args.output_dir))
    dest = Path(args.best_dir)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
    report = Path("reports/evaluation_report.md")
    with report.open("a", encoding="utf-8") as handle:
        handle.write("\n## Best Checkpoint\n\n")
        handle.write(f"- source: {source}\n")
        handle.write(f"- eval_loss: {loss}\n")
        handle.write(f"- copied_to: {dest}\n")
    print(f"best checkpoint: {source} -> {dest}")


if __name__ == "__main__":
    main()
