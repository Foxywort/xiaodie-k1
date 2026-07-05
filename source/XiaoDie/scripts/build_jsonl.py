#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import yaml


SYSTEM_PROMPT = (
    "创建一个具有教育意义的中文儿童故事，重点针对对世界和人际交往零知识的5岁儿童。"
    "故事应该使用简单的术语，包含日常行为和常见物品的使用。"
    "请直接开始撰写故事，不要输出除了故事以外的内容。"
)


def make_instruction(row: dict) -> str:
    tags = "、".join(row.get("tags", [])[:4])
    return (
        "请写一个适合儿童阅读的中文故事，内容是："
        f"标题《{row['title']}》，主题标签：{tags}。"
        "要求语言自然、情节完整、有教育意义、适合朗读。"
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_build_stats(final_rows: list[dict], train_rows: list[dict], eval_rows: list[dict]) -> None:
    report = Path("reports/data_stats.md")
    total_chars = sum(sum("\u4e00" <= c <= "\u9fff" for c in row["text"]) for row in final_rows)
    avg_chars = total_chars / max(len(final_rows), 1)
    high_quality = sum(1 for row in final_rows if row.get("quality_score", 0) >= 0.75)
    text = (
        "\n## Final Dataset\n\n"
        f"- final_samples: {len(final_rows)}\n"
        f"- train_samples: {len(train_rows)}\n"
        f"- eval_samples: {len(eval_rows)}\n"
        f"- final_total_chinese_chars: {total_chars}\n"
        f"- final_avg_chinese_chars: {avg_chars:.1f}\n"
        f"- final_high_quality_ratio: {high_quality / max(len(final_rows), 1):.3f}\n"
    )
    previous = report.read_text(encoding="utf-8") if report.exists() else "# Data Stats\n"
    if "## Final Dataset" in previous:
        previous = previous.split("## Final Dataset", 1)[0].rstrip() + "\n"
    report.write_text(previous + text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final story JSONL and SFT files.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--input", default="data/processed/deduped_stories.jsonl")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    random.seed(cfg["project"]["seed"])
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["id"] = f"{row['source']}_{idx:06d}"

    processed = Path(cfg["data"]["processed_file"])
    write_jsonl(processed, rows)

    sft_rows = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_instruction(row)},
                {"role": "assistant", "content": row["text"]},
            ],
            "metadata": {
                "id": row["id"],
                "source": row["source"],
                "license": row["license"],
                "quality_score": row["quality_score"],
            },
        }
        for row in rows
    ]
    random.shuffle(sft_rows)
    eval_count = max(1, int(len(sft_rows) * cfg["data"].get("eval_ratio", 0.08)))
    eval_rows = sft_rows[:eval_count]
    train_rows = sft_rows[eval_count:]
    write_jsonl(Path(cfg["data"]["train_file"]), train_rows)
    write_jsonl(Path(cfg["data"]["eval_file"]), eval_rows)
    append_build_stats(rows, train_rows, eval_rows)
    print(f"wrote {len(rows)} final rows -> {processed}")
    print(f"wrote train={len(train_rows)}, eval={len(eval_rows)}")


if __name__ == "__main__":
    main()
