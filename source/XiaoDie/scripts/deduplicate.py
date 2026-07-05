#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def normalize_for_dedup(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
    return text


def shingles(text: str, n: int = 8) -> set[str]:
    text = normalize_for_dedup(text)
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(0, len(text) - n + 1, 2)}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def main() -> None:
    parser = argparse.ArgumentParser(description="Near-deduplicate cleaned stories.")
    parser.add_argument("--input", default="data/processed/cleaned_stories.jsonl")
    parser.add_argument("--output", default="data/processed/deduped_stories.jsonl")
    parser.add_argument("--threshold", type=float, default=0.88)
    args = parser.parse_args()

    kept = []
    kept_shingles = []
    dropped = 0
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sig = shingles(row["text"])
            if any(jaccard(sig, old) >= args.threshold for old in kept_shingles):
                dropped += 1
                continue
            kept.append(row)
            kept_shingles.append(sig)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = Path("reports/data_stats.md")
    with report.open("a", encoding="utf-8") as handle:
        handle.write("\n## Deduplication\n\n")
        handle.write(f"- input_samples: {len(kept) + dropped}\n")
        handle.write(f"- output_samples: {len(kept)}\n")
        handle.write(f"- dropped_near_duplicates: {dropped}\n")
        handle.write(f"- dedup_ratio: {dropped / max(1, len(kept) + dropped):.3f}\n")
    print(f"deduped {len(kept)} rows, dropped {dropped} -> {out}")


if __name__ == "__main__":
    main()
