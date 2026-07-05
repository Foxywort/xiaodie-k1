#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ZH_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokens(text: str) -> list[str]:
    text = (text or "").lower()
    toks = WORD_RE.findall(text)
    compact_zh = "".join(ZH_RE.findall(text))
    toks.extend(compact_zh[i : i + 2] for i in range(max(0, len(compact_zh) - 1)))
    return [t for t in toks if t.strip()]


def split_chunks(text: str, min_chars: int, max_chars: int) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n{1,}|(?<=[。！？.!?])\s+", text or "") if p.strip()]
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(current) + len(part) + 1 <= max_chars:
            current = (current + "\n" + part).strip()
        else:
            if len(current) >= min_chars:
                chunks.append(current)
            current = part
    if len(current) >= min_chars:
        chunks.append(current)
    return chunks


def source_priority(source_type: str) -> float:
    if source_type == "user_authorized_file":
        return 1.35
    if source_type == "official_public_web":
        return 1.2
    if source_type == "wikipedia":
        return 0.95
    return 1.0


def is_bad_chunk(chunk: str) -> bool:
    lower = chunk.lower()
    bad_markers = [
        "pages for logged out editors",
        "learn more",
        "contributions",
        "talk",
        "contents move to sidebar",
        "hide",
        "appearance",
        "references",
        "external links",
        "further reading",
        "television portal",
        "categories",
        "articles with",
        "archived from the original",
        "retrieved",
        "netflix original",
        "spin-off",
        "dvd release",
        "broadcast",
        "ratings",
        "box office",
        "important notice",
        "shop is no longer available",
        "by submitting my email",
        "marketing emails",
        "privacy policy",
        "corporate contact us",
        "where to buy",
        "all audio, visual and textual",
        "protected by trademarks",
        "received a variety of awards",
        "began airing",
        "interview with",
        "hollywood reporter",
        "acquired entertainment one",
        "renewed until",
        "theme music",
        "running time",
        "production companies",
        "executive producers",
        "country of origin",
        "original language",
        "no. of episodes",
        "theme parks",
        "peppa pig world",
        "world of play",
        "cinema experiences",
        "live-action host",
        "anniversary",
        "awards",
        "nominations",
        "grand prize",
        "festival",
        "tiktok",
        "douyin",
        "adult content",
        "criminal",
        "mobster",
        "tattoos",
        "adult humour",
        "people's liberation army",
        "weapons manufacturer",
        "social media posts",
        "linguistics experts",
        "likely exaggerated",
        "popularity in china",
        "box office",
        "partnership with",
        "exclusive licensor",
    ]
    if any(marker in lower for marker in bad_markers):
        return True
    if len(re.findall(r"\(\d{4}", chunk)) >= 5:
        return True
    if len(re.findall(r"\[\s*\d+\s*\]", chunk)) >= 5:
        return True
    return False


def has_alias(chunk: str, aliases: list[str], zh_name: str) -> bool:
    lower = chunk.lower()
    if zh_name and zh_name in chunk:
        return True
    for alias in aliases:
        alias = str(alias).strip()
        if not alias:
            continue
        if re.search(r"[\u4e00-\u9fff]", alias):
            if alias in chunk:
                return True
        elif alias.lower() in lower:
            return True
    return False


def build_index(cards: list[dict[str, Any]]) -> dict[str, Any]:
    doc_tokens = []
    df = Counter()
    for card in cards:
        toks = tokens(" ".join([card.get("franchise", ""), card.get("franchise_zh", ""), card.get("title", ""), card.get("text", "")]))
        counts = Counter(toks)
        doc_tokens.append(counts)
        df.update(counts.keys())
    n = max(1, len(cards))
    avgdl = sum(sum(c.values()) for c in doc_tokens) / n
    return {
        "avgdl": avgdl,
        "doc_tokens": [dict(c) for c in doc_tokens],
        "idf": {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()},
    }


def add_policy_cards(cfg: dict[str, Any], cards: list[dict[str, Any]]) -> None:
    for franchise in cfg["franchises"]:
        fid = franchise["id"]
        for idx, fact in enumerate(franchise.get("facts", []), start=1):
            cards.append(
                {
                    "id": f"{fid}_curated_fact_{idx:03d}",
                    "franchise": fid,
                    "franchise_zh": franchise["zh_name"],
                    "source": fact.get("source", "ip_rag_sources.yaml"),
                    "source_type": "curated_fact",
                    "license": fact.get("license", "source-derived-metadata"),
                    "url": "E:/Duanwu/configs/ip_rag_sources.yaml",
                    "title": f"{franchise['zh_name']} core facts",
                    "text": fact["text"],
                    "weight": 1.6,
                }
            )
        alias_text = "、".join(franchise.get("aliases", []))
        cards.append(
            {
                "id": f"{fid}_alias_policy",
                "franchise": fid,
                "franchise_zh": franchise["zh_name"],
                "source": "ip_rag_sources.yaml",
                "source_type": "alias_policy",
                "license": "project-generated-metadata",
                "url": "E:/Duanwu/configs/ip_rag_sources.yaml",
                "title": f"{franchise['zh_name']} aliases",
                "text": f"这个知识库条目用于识别用户说的动画片名称和角色别名：{alias_text}。生成故事时必须以检索到的来源文本为准，不要编造官方设定。",
                "weight": 0.85,
            }
        )
        cards.append(
            {
                "id": f"{fid}_safety_policy",
                "franchise": fid,
                "franchise_zh": franchise["zh_name"],
                "source": "xiaodie_child_safety_policy",
                "source_type": "safety_policy",
                "license": "project-generated-safety-policy",
                "url": "E:/Duanwu/configs/ip_rag_sources.yaml",
                "title": f"{franchise['zh_name']} child-safe adaptation policy",
                "text": "小蝶给幼儿园小朋友讲故事时，只能使用温和、安全、适合朗读的情节。避免武器、打败怪物、可怕追逐、伤害、恐吓、成人议题和危险模仿动作。不要写拉耳朵、拉尾巴、抓身体、推搡或抢夺。优先讲合作、分享、礼貌、勇气、倾听、解决误会和一起完成小任务。",
                "weight": 1.1,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="E:/Duanwu/configs/ip_rag_sources.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = Path(cfg["paths"]["raw_dir"])
    rag_dir = Path(cfg["paths"]["rag_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    rag_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(raw_dir / "ip_web_records.jsonl")
    min_chars = int(cfg["defaults"].get("chunk_min_chars", 120))
    max_chars = int(cfg["defaults"].get("chunk_max_chars", 900))
    franchise_aliases = {f["id"]: f.get("aliases", []) for f in cfg["franchises"]}
    franchise_zh = {f["id"]: f.get("zh_name", "") for f in cfg["franchises"]}

    cards: list[dict[str, Any]] = []
    for record in records:
        chunks = split_chunks(record["text"], min_chars, max_chars)
        priority = source_priority(record.get("source_type", "web"))
        for idx, chunk in enumerate(chunks, start=1):
            if len(tokens(chunk)) < 20:
                continue
            if is_bad_chunk(chunk):
                continue
            if record.get("source_type") == "wikipedia" and not has_alias(
                chunk, franchise_aliases.get(record["franchise"], []), franchise_zh.get(record["franchise"], "")
            ):
                continue
            cards.append(
                {
                    "id": f"{record['id']}_chunk_{idx:03d}",
                    "franchise": record["franchise"],
                    "franchise_zh": record["franchise_zh"],
                    "source": record["source"],
                    "source_type": record.get("source_type", "web"),
                    "license": record.get("license", "unknown"),
                    "url": record.get("url", ""),
                    "title": record.get("title", ""),
                    "text": chunk,
                    "weight": priority,
                }
            )

    add_policy_cards(cfg, cards)
    index = build_index(cards)
    write_jsonl(rag_dir / "ip_knowledge_cards.jsonl", cards)
    (rag_dir / "ip_rag_index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    by_franchise = Counter(card["franchise"] for card in cards)
    by_source = Counter(f"{card['franchise']}::{card['source']}" for card in cards)
    lines = ["# IP RAG KB Report", "", f"- cards: {len(cards)}", "", "## Franchise Distribution", ""]
    for key, count in by_franchise.most_common():
        lines.append(f"- {key}: {count}")
    lines += ["", "## Source Distribution", ""]
    for key, count in by_source.most_common():
        lines.append(f"- {key}: {count}")
    (reports_dir / "ip_kb_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"cards={len(cards)} -> {rag_dir / 'ip_knowledge_cards.jsonl'}")


if __name__ == "__main__":
    main()
