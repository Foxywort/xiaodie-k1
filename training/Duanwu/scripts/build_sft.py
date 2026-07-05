#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path

import yaml


SYSTEM_PROMPT = (
    "你是小蝶，一个温柔的中文儿童故事助手。"
    "请只输出适合儿童朗读的中文故事正文。"
    "不要输出解释、提纲、Markdown、英文、训练说明或家长建议。"
    "故事必须安全、温柔、情节完整，有开头、发展和结尾。"
)


PROMPT_TEMPLATES = [
    "请写一个适合{age}小朋友听的中文儿童故事。{title_clause}关键词：{tags}。请直接开始讲故事。",
    "请根据这些词写儿童故事：{tags}。{title_clause}语言要自然，适合朗读。",
    "小蝶，请讲一个安全、温柔、有教育意义的中文故事。{title_clause}适合{age}孩子。",
    "请写睡前故事正文，不要解释。{title_clause}可以包含这些元素：{tags}。",
]

BAD_TITLES = {"GPT-4", "baike", "Cosmopedia", "TinyStories", "TinyStories Chinese"}
THEME_WORDS = [
    "月亮", "太阳", "星星", "森林", "花园", "幼儿园", "小兔子", "小猫", "小狗", "小熊", "小鸟",
    "朋友", "妈妈", "爸爸", "老师", "玩具", "积木", "彩笔", "图书馆", "睡前", "分享",
    "勇气", "诚实", "礼貌", "合作", "等待", "倾听", "帮助", "道歉", "感谢", "好奇心",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_title(title: str) -> str:
    title = re.sub(r"^(故事题目|故事标题|标题)[:：]\s*", "", (title or "").strip())
    title = title.strip("《》“”\"' ")
    if not title or title in BAD_TITLES:
        return ""
    if re.fullmatch(r"(GPT-?\d+|baike|Cosmopedia\s*\d*|TinyStories.*|\d+)", title, flags=re.I):
        return ""
    if len(re.findall(r"[\u4e00-\u9fff]", title)) < 2:
        return ""
    return title[:40]


def infer_tags(item: dict) -> str:
    text = item.get("text", "")
    tags = []
    for word in THEME_WORDS:
        if word in text and word not in tags:
            tags.append(word)
    for tag in item.get("tags", []):
        tag = str(tag)
        if tag in {"chinese", "children_story", "tinystories", "cosmopedia", "storybooks"}:
            continue
        if tag and tag not in tags:
            tags.append(tag)
    if not tags:
        tags = ["儿童故事", "朋友", "成长"]
    return "、".join(tags[:6])


def age_label(age_level: str) -> str:
    if age_level == "preschool":
        return "4-6岁"
    if age_level == "primary":
        return "小学生"
    return "4-8岁"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="E:/Duanwu/configs/train.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rng = random.Random(cfg["project"]["seed"])
    rows = read_jsonl(Path(cfg["data"]["processed_file"]))
    excluded_licenses = set(cfg.get("data", {}).get("exclude_licenses", []))
    rows = [row for row in rows if row.get("license") not in excluded_licenses]
    min_quality = float(cfg.get("data", {}).get("min_quality_score", 0.0))
    rows = [row for row in rows if float(row.get("quality_score", 0.0)) >= min_quality]
    rows.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    sft = []
    for item in rows:
        tags = infer_tags(item)
        title = clean_title(item.get("title", ""))
        title_clause = f"标题：《{title}》。" if title else ""
        prompt = rng.choice(PROMPT_TEMPLATES).format(
            title_clause=title_clause,
            tags=tags,
            age=age_label(item.get("age_level", "unknown")),
        )
        sft.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": item["text"]},
                ],
                "metadata": {
                    "id": item["id"],
                    "source": item["source"],
                    "license": item["license"],
                    "url": item.get("url", ""),
                    "quality_score": item.get("quality_score", 0),
                },
            }
        )

    rng.shuffle(sft)
    eval_count = max(20, int(len(sft) * cfg["data"].get("eval_ratio", 0.04)))
    eval_rows = sft[:eval_count]
    train_rows = sft[eval_count:]
    write_jsonl(Path(cfg["data"]["train_file"]), train_rows)
    write_jsonl(Path(cfg["data"]["eval_file"]), eval_rows)
    report = (
        "# SFT Build Report\n\n"
        f"- total_sft_samples: {len(sft)}\n"
        f"- train_samples: {len(train_rows)}\n"
        f"- eval_samples: {len(eval_rows)}\n"
        "- label_policy: assistant_only_loss\n"
    )
    Path("E:/Duanwu/reports/sft_build_report.md").write_text(report, encoding="utf-8")
    print(f"train={len(train_rows)} eval={len(eval_rows)}")


if __name__ == "__main__":
    main()
