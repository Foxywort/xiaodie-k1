#!/usr/bin/env python3
import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

import yaml


BAD_PATTERNS = {
    "adult": r"色情|成人|裸聊|约炮|做爱|性爱",
    "violence": r"杀人|砍死|血腥|自杀|虐待|毒打",
    "hate": r"仇恨|种族灭绝|纳粹",
    "politics": r"政治宣传|颠覆|竞选|政党",
    "medical": r"处方药|诊断|治疗方案|吃药就会好",
    "privacy": r"身份证|手机号|家庭住址|银行卡",
}


def chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def normalize(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"版权所有|广告|导航|点击下载|扫码关注", "", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def quality_score(text: str) -> float:
    zh = chinese_chars(text)
    length_score = min(1.0, zh / 900)
    paragraph_score = min(1.0, max(0, text.count("\n")) / 4)
    dialogue_score = 0.15 if "“" in text and "”" in text else 0.0
    child_words = sum(word in text for word in ["朋友", "勇敢", "分享", "小蝶", "幼儿园", "慢慢", "办法"])
    child_score = min(0.25, child_words * 0.04)
    return round(min(1.0, 0.45 * length_score + 0.15 * paragraph_score + dialogue_score + child_score + 0.25), 4)


def reject_reason(row: dict, text: str, min_chinese_chars: int) -> str | None:
    if chinese_chars(text) < min_chinese_chars:
        return "too_short_or_not_chinese"
    if chinese_chars(text) / max(1, len(text)) < 0.45:
        return "non_chinese_main_text"
    for reason, pattern in BAD_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return f"unsafe_{reason}"
    return None


def iter_raw_records(raw_dir: Path):
    for path in raw_dir.glob("*/records.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw Chinese children stories.")
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--output", default="data/processed/cleaned_stories.jsonl")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    min_chars = cfg["defaults"].get("min_chinese_chars", 100)
    raw_dir = Path(cfg["output"]["raw_dir"])
    reports_dir = Path(cfg["output"]["reports_dir"])

    kept = []
    filters = Counter()
    source_counter = Counter()
    for row in iter_raw_records(raw_dir):
        text = normalize(row.get("text", ""))
        title = normalize(row.get("title", ""))[:80]
        reason = reject_reason(row, text, min_chars)
        if reason:
            filters[reason] += 1
            continue
        item = {
            "id": row.get("id", f"unknown_{len(kept)+1:06d}"),
            "source": row.get("source", "unknown"),
            "license": row.get("license", "unknown"),
            "title": title or "未命名故事",
            "text": text,
            "age_level": row.get("age_level", "unknown"),
            "tags": sorted(set(row.get("tags", []) + ["children_story", "chinese"])),
            "quality_score": quality_score(text),
        }
        kept.append(item)
        source_counter[item["source"]] += 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in kept:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_chars = sum(chinese_chars(x["text"]) for x in kept)
    avg_len = total_chars / max(1, len(kept))
    lines = [
        "# Data Stats",
        "",
        f"- cleaned_samples: {len(kept)}",
        f"- total_chinese_chars: {total_chars}",
        f"- avg_chinese_chars: {avg_len:.1f}",
        f"- high_quality_ratio: {sum(x['quality_score'] >= cfg['defaults'].get('high_quality_threshold', 0.7) for x in kept) / max(1, len(kept)):.3f}",
        "",
        "## Source Distribution",
        "",
    ]
    for source, count in source_counter.most_common():
        lines.append(f"- {source}: {count}")
    lines += ["", "## Filtered Samples", ""]
    for reason, count in filters.most_common():
        lines.append(f"- {reason}: {count}")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "data_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"kept {len(kept)} -> {out}")


if __name__ == "__main__":
    main()
