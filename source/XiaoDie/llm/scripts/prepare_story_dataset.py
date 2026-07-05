#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Iterable


DEFAULT_SYSTEM = (
    "你是“小蝶”，一个运行在端侧设备上的幼儿园故事 AI 助手。"
    "请根据关键词生成适合 3-6 岁儿童收听的中文原创故事，语言温柔、句子短、适合 TTS 朗读。"
)

KEYWORDS = [
    "月亮",
    "星星",
    "小火车",
    "积木",
    "彩虹",
    "小书包",
    "花园",
    "云朵",
    "小雨伞",
    "纸飞机",
    "勇气",
    "分享",
    "合作",
    "礼貌",
    "耐心",
    "想象力",
    "刷牙",
    "整理玩具",
    "午睡",
    "过马路",
]

SETTINGS = [
    "幼儿园的图书角",
    "铺着软垫的活动室",
    "窗边的小小舞台",
    "有风铃的午睡房",
    "种着向日葵的小花园",
    "放满彩色积木的小桌旁",
]

FRIENDS = [
    "愿意认真听别人说话的安安",
    "喜欢画画的朵朵",
    "总想跑得很快的乐乐",
    "有一点害羞的米米",
    "喜欢问为什么的小宇",
    "会慢慢想办法的小航",
]

HELPERS = [
    "小蝶用轻轻的声音提醒大家",
    "老师蹲下来，和大家一起想办法",
    "好朋友伸出手，邀请大家轮流试一试",
    "大家围成一个小圆圈，把想法一个一个说出来",
]

PROBLEMS = [
    "一开始，事情没有马上成功。",
    "一个小小的问题悄悄出现了。",
    "大家的想法不太一样，活动室里安静了一小会儿。",
    "小朋友有一点着急，手里的动作也变快了。",
]

LESSONS = [
    "原来，勇敢不是很大声，而是愿意再试一次。",
    "原来，把想法说出来，大家就能一起把事情变好。",
    "原来，分享以后，快乐会变成两份、三份、好多份。",
    "原来，慢慢来、认真听，心里就会亮起小灯。",
    "原来，遇到困难时，先停一停、想一想，就能找到新的办法。",
]

ENDINGS = [
    "故事讲完了，小蝶轻轻说：你今天也很棒，明天我们还可以继续想象。",
    "最后，大家把笑容收进口袋里，准备做一个甜甜的梦。",
    "小蝶把声音放得更轻：愿这个小故事陪你安安稳稳地休息。",
    "窗外的风慢慢停了，故事里的温暖还留在每个人心里。",
]

STYLES = [
    "睡前安抚",
    "幼儿园晨间分享",
    "情绪陪伴",
    "习惯养成",
    "想象冒险",
]


def read_system_prompt(path: Path | None) -> str:
    if path and path.exists():
        return path.read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM


def make_user_prompt(keywords: list[str], style: str, age: str, minutes: int) -> str:
    return (
        f"请给 {age} 小朋友讲一个{style}风格的原创中文故事。"
        f"关键词：{'、'.join(keywords)}。"
        f"时长约 {minutes} 分钟，适合语音朗读，结尾要温暖。"
    )


def make_story(rng: random.Random, keywords: list[str], style: str) -> str:
    subject = keywords[0]
    setting = rng.choice(SETTINGS)
    friend = rng.choice(FRIENDS)
    helper = rng.choice(HELPERS)
    problem = rng.choice(PROBLEMS)
    lesson = rng.choice(LESSONS)
    ending = rng.choice(ENDINGS)
    title = f"{subject}和小小办法"
    keyword_sentence = "、".join(keywords)

    paragraphs = [
        f"《{title}》",
        f"今天，小蝶收到几个关键词：{keyword_sentence}。她把它们放进一个温柔的故事里，准备讲给小朋友听。",
        f"在{setting}，{friend}发现了和{subject}有关的一件小事。大家都很好奇，想一起把它做好。",
        f"{problem}有人皱起眉头，有人低下头，还有人想马上换一个办法。",
        f"{helper}。大家先深呼吸，再轮流说出自己的主意。每个主意都被认真听见了。",
        f"他们把{keyword_sentence}一个一个连起来，试了第一次，又试了第二次。慢慢地，新的办法出现了。",
        f"{lesson}这个办法不只帮大家完成了小任务，也让每个人心里都暖暖的。",
        f"{ending}",
    ]

    if style == "情绪陪伴":
        paragraphs.insert(4, "小蝶先说：有一点难过也没关系，我们可以慢慢说，慢慢想。")
    elif style == "习惯养成":
        paragraphs.insert(5, "安安把东西放回原位，发现整齐以后，找东西也变得更容易。")
    elif style == "想象冒险":
        paragraphs.insert(3, f"忽然，{subject}像一盏小灯一样亮起来，带大家走进想象的小路。")

    return "\n".join(paragraphs)


def synthetic_records(count: int, seed: int, system_prompt: str) -> Iterable[dict]:
    rng = random.Random(seed)
    for index in range(count):
        keywords = rng.sample(KEYWORDS, rng.randint(2, 4))
        style = rng.choice(STYLES)
        age = rng.choice(["3-4 岁", "4-5 岁", "5-6 岁"])
        minutes = rng.choice([1, 2, 3])
        user_prompt = make_user_prompt(keywords, style, age, minutes)
        story = make_story(rng, keywords, style)
        yield {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": story},
            ],
            "metadata": {
                "source": "xiaodie_synthetic_generator",
                "license": "project-generated",
                "seed": seed,
                "index": index,
                "style": style,
                "keywords": keywords,
            },
        }


def normalize_local_record(record: dict, system_prompt: str, require_license: bool) -> dict | None:
    if "messages" in record:
        metadata = record.setdefault("metadata", {})
        if require_license and not metadata.get("license"):
            return None
        return record

    instruction = record.get("instruction") or record.get("prompt")
    output = record.get("output") or record.get("story") or record.get("response")
    if not instruction or not output:
        return None

    license_name = record.get("license") or record.get("licence")
    if require_license and not license_name:
        return None

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(instruction)},
            {"role": "assistant", "content": str(output)},
        ],
        "metadata": {
            "source": record.get("source", "local_jsonl"),
            "license": license_name or "missing",
        },
    }


def read_jsonl(path: Path, system_prompt: str, require_license: bool) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            normalized = normalize_local_record(json.loads(line), system_prompt, require_license)
            if normalized is not None:
                yield normalized


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare XiaoDie story SFT JSONL data.")
    parser.add_argument("--count", type=int, default=5000, help="Synthetic Chinese story instruction count.")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--output", default="llm/data/processed/story_sft_train.jsonl")
    parser.add_argument("--eval-output", default="llm/data/processed/story_sft_eval.jsonl")
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--system-prompt", default="llm/prompts/story_system_prompt.md")
    parser.add_argument("--local-jsonl", action="append", default=[], help="Licensed local JSONL to append.")
    parser.add_argument("--allow-missing-license", action="store_true")
    args = parser.parse_args()

    system_prompt = read_system_prompt(Path(args.system_prompt))
    require_license = not args.allow_missing_license

    train_records = list(synthetic_records(args.count, args.seed, system_prompt))
    eval_records = list(synthetic_records(args.eval_count, args.seed + 1, system_prompt))

    for local_path in args.local_jsonl:
        train_records.extend(read_jsonl(Path(local_path), system_prompt, require_license))

    train_written = write_jsonl(Path(args.output), train_records)
    eval_written = write_jsonl(Path(args.eval_output), eval_records)
    print(f"wrote train: {train_written} -> {args.output}")
    print(f"wrote eval: {eval_written} -> {args.eval_output}")


if __name__ == "__main__":
    main()
