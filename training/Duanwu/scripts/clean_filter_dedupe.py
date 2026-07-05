#!/usr/bin/env python3
import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ZH_RE = re.compile(r"[\u4e00-\u9fff]")
EN_WORD_RE = re.compile(r"\b[A-Za-z]{3,}\b")

BAD_PATTERNS = {
    "adult": r"色情|成人|裸聊|约炮|做爱|性爱|性侵|强奸|猥亵|结婚|婚礼",
    "violence": r"砍死|杀死|血腥|自杀|虐待|毒打|枪杀|斩首|尸体|复仇|报仇|打猎|猎人|长矛|矛|厄运|饥荒|吞掉|吞了|吞下|一口把|咬死|死去|杀|士兵|逃离|躲了起来|躲在|带走了|惩罚|该死|划伤|抓起来|迷路了",
    "hate": r"仇恨|种族灭绝|纳粹|屠杀",
    "politics": r"政治宣传|颠覆|政党口号|革命宣传",
    "medical": r"处方药|诊断|治疗方案|吃药就会好|手术建议|皮肤病|研究所|治病|疾病|病人|手术|用药|药物剂量|生了一场大病",
    "privacy": r"身份证|银行卡|手机号|家庭住址|密码",
    "unsafe_child": r"爬窗|玩火|插座|煤气|跳楼|离家出走|点火|火堆|爬树|放一把火|烧起来|火突然|火焰|融化|高速驶过|撞到|撞了|街道|尖锐|冰柱|摔倒|滑了一跤|跑出去玩|打碎|碎片|玻璃",
}

NON_STORY_PATTERNS = re.compile(
    r"课程单元|学习目标|教学目标|教学过程|引言\s*\n|定义与内涵|案例分析|本单元旨在|"
    r"^[一二三四五六七八九十]、|^\s*\d+\.\s*(定义|特征|目标|背景|意义)",
    flags=re.M,
)

BOILERPLATE = [
    r"版权.*所有",
    r"点击.*下载",
    r"扫码.*关注",
    r"Project Gutenberg.*",
    r"End of the Project Gutenberg",
]


def zh_count(text: str) -> int:
    return len(ZH_RE.findall(text or ""))


def normalize(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<script.*?</script>", "", text, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"(?m)^\s*---+\s*$", "", text)
    text = re.sub(r"(?m)^\s*(故事正文|故事开始)[:：]\s*", "", text)
    for pattern in BOILERPLATE:
        text = re.sub(pattern, "", text, flags=re.S | re.I)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([。！？])\1+", r"\1", text)
    return text.strip()


def reject_reason(text: str, cfg: dict[str, Any]) -> str | None:
    zhc = zh_count(text)
    if zhc < int(cfg["defaults"].get("min_chinese_chars", 120)):
        return "too_short_or_not_chinese"
    if zhc > int(cfg["defaults"].get("max_chinese_chars", 1800)):
        return "too_long"
    if zhc / max(1, len(text)) < 0.55:
        return "non_chinese_main_text"
    if len(EN_WORD_RE.findall(text)) > 3:
        return "too_much_english"
    if NON_STORY_PATTERNS.search(text):
        return "non_story_instructional_text"
    for reason, pattern in BAD_PATTERNS.items():
        if re.search(pattern, text, flags=re.I):
            return f"unsafe_{reason}"
    return None


def quality_score(text: str) -> float:
    zhc = zh_count(text)
    length_score = min(1.0, zhc / 700)
    dialogue_score = 0.12 if re.search(r"[“「].{1,40}[”」]", text) else 0.0
    ending_score = 0.12 if any(x in text[-120:] for x in ["最后", "从那以后", "故事结束", "晚安", "明天"]) else 0.0
    child_terms = sum(x in text for x in ["朋友", "小朋友", "勇气", "分享", "诚实", "礼貌", "帮助", "睡觉", "幼儿园", "妈妈", "爸爸"])
    child_score = min(0.22, child_terms * 0.035)
    repetition_penalty = min(0.20, repetition_score(text) * 0.40)
    return round(max(0.0, min(1.0, 0.30 + 0.34 * length_score + dialogue_score + ending_score + child_score - repetition_penalty)), 4)


def shingles(text: str, width: int = 18) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) <= width:
        return {compact}
    return {compact[i : i + width] for i in range(0, len(compact) - width + 1, 6)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def repetition_score(text: str) -> float:
    chunks = [text[i : i + 16] for i in range(0, max(0, len(text) - 16), 8)]
    if not chunks:
        return 0.0
    return 1 - len(set(chunks)) / max(1, len(chunks))


def iter_raw(raw_dir: Path, enabled_sources: set[str]):
    for path in raw_dir.glob("*/records.jsonl"):
        if path.parent.name not in enabled_sources:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="E:/Duanwu/configs/datasets.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = Path(cfg["output"]["raw_dir"])
    processed_dir = Path(cfg["output"]["processed_dir"])
    reports_dir = Path(cfg["output"]["reports_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    filters = Counter()
    source_counter = Counter()
    enabled_sources = {source["name"] for source in cfg["sources"] if source.get("enabled", False)}
    for item in iter_raw(raw_dir, enabled_sources):
        text = normalize(item.get("text", ""))
        title = normalize(item.get("title", ""))[:100] or "未命名故事"
        reason = reject_reason(text, cfg)
        if reason:
            filters[reason] += 1
            continue
        row = {
            "id": item.get("id") or f"raw_{len(kept)+1:07d}",
            "source": item.get("source", "unknown"),
            "license": item.get("license", "unknown"),
            "url": item.get("url", ""),
            "title": title,
            "text": text,
            "age_level": item.get("age_level", "unknown"),
            "tags": sorted(set(item.get("tags", []) + ["children_story", "chinese"])),
            "quality_score": quality_score(text),
        }
        kept.append(row)
        source_counter[row["source"]] += 1

    kept.sort(key=lambda x: x["quality_score"], reverse=True)
    deduped = []
    source_shingles: list[set[str]] = []
    dropped_dupes = 0
    for item in kept:
        sig = shingles(item["text"])
        if any(jaccard(sig, old) >= 0.82 for old in source_shingles[-2500:]):
            dropped_dupes += 1
            continue
        item["id"] = f"{item['source']}_{len(deduped)+1:07d}"
        deduped.append(item)
        source_shingles.append(sig)

    final_path = processed_dir / "chinese_children_stories.jsonl"
    with final_path.open("w", encoding="utf-8") as handle:
        for item in deduped:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_chars = sum(zh_count(x["text"]) for x in deduped)
    avg_chars = total_chars / max(1, len(deduped))
    high_quality = sum(x["quality_score"] >= cfg["defaults"].get("high_quality_threshold", 0.72) for x in deduped)
    lines = [
        "# Data Stats - Duanwu Round 2",
        "",
        f"- raw_kept_before_dedup: {len(kept)}",
        f"- final_samples: {len(deduped)}",
        f"- dropped_near_duplicates: {dropped_dupes}",
        f"- total_chinese_chars: {total_chars}",
        f"- avg_chinese_chars: {avg_chars:.1f}",
        f"- high_quality_ratio: {high_quality / max(1, len(deduped)):.3f}",
        "",
        "## Source Distribution",
        "",
    ]
    final_source_counts = Counter(x["source"] for x in deduped)
    for source, count in final_source_counts.most_common():
        lines.append(f"- {source}: {count}")
    lines += ["", "## Filtered Samples", ""]
    for reason, count in filters.most_common():
        lines.append(f"- {reason}: {count}")
    (reports_dir / "data_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(deduped)} rows -> {final_path}")


if __name__ == "__main__":
    main()
