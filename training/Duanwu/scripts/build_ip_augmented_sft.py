#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml


SYSTEM = (
    "你是小蝶的RAG知识整理模块。"
    "你的任务是把授权动画知识卡片整理成儿童故事生成可使用的事实约束。"
    "不要编造卡片外的设定。"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-config", default="E:/Duanwu/configs/ip_rag_sources.yaml")
    parser.add_argument("--output", default="E:/Duanwu/data/processed/ip_augmented_sft.jsonl")
    parser.add_argument("--max-samples", type=int, default=600)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.rag_config).read_text(encoding="utf-8"))
    rag_dir = Path(cfg["paths"]["rag_dir"])
    cards = load_jsonl(rag_dir / "ip_knowledge_cards.jsonl")
    cards = [c for c in cards if c.get("source_type") not in {"safety_policy", "alias_policy"}]
    rng = random.Random(cfg["project"]["seed"])
    rng.shuffle(cards)
    rows = []
    for card in cards[: args.max_samples]:
        prompt = (
            f"请把下面授权来源的动画知识卡片整理成故事生成约束，动画：{card['franchise_zh']}。\n"
            f"来源：{card['source']}；许可证/授权：{card['license']}。\n"
            f"卡片内容：\n{card['text'][:900]}"
        )
        answer = (
            f"动画：{card['franchise_zh']}\n"
            f"可用事实：{card['text'][:700]}\n"
            "生成约束：只使用以上事实；不补编官方设定；故事面向儿童时要改写为温和、合作、分享、解决误会的情节；避免危险动作和暴力冲突。"
        )
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "source": card["source"],
                    "franchise": card["franchise"],
                    "license": card["license"],
                    "url": card["url"],
                    "training_purpose": "rag_grounding_fact_constraints",
                },
            }
        )
    write_jsonl(Path(args.output), rows)
    report = f"# IP Augmented SFT Report\n\n- samples: {len(rows)}\n- output: {args.output}\n"
    Path(cfg["paths"]["reports_dir"], "ip_augmented_sft_report.md").write_text(report, encoding="utf-8")
    print(f"samples={len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
