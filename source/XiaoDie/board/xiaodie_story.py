#!/usr/bin/env python3
import argparse
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import xiaodie_tts


APP_DIR = Path.home() / "xiaodie" / "app"
STORY_DIR = Path.home() / "xiaodie" / "stories"


OPENINGS = [
    "清晨的阳光照进窗台，小蝶轻轻地说：今天，我们要讲一个新的故事。",
    "午后的风很轻，小蝶把声音放得软软的，准备开始一个温暖的故事。",
    "夜灯亮起来的时候，小蝶翻开想象的小书，一页一页地讲给小朋友听。",
]

PLACES = [
    "幼儿园的图书角",
    "安静的小院子",
    "亮晶晶的故事屋",
    "有彩色积木的小桌旁",
    "窗边的小小舞台",
]

HELPERS = [
    "会认真倾听的好朋友",
    "喜欢画画的小伙伴",
    "总是慢慢想办法的老师",
    "愿意分享积木的同伴",
    "心里装着温柔办法的小蝶",
]

CHALLENGES = [
    "可是，事情一开始并不顺利。",
    "不过，一个小小的问题悄悄出现了。",
    "这时，大家发现还差一个好办法。",
    "忽然，原本简单的事情变得有一点点难。",
]

LESSONS = [
    "原来，遇到困难时，先停一停、想一想，就会找到新的办法。",
    "原来，把自己的想法说出来，大家就能一起把事情变好。",
    "原来，勇敢不是很大声，而是愿意再试一次。",
    "原来，分享以后，快乐会变成两份、三份、好多份。",
]

GOOD_NIGHT = [
    "故事讲完了，小蝶轻轻地说：你也很棒，明天我们还可以继续想象。",
    "最后，小蝶把故事合上，温柔地说：今天的你，也学会了一个小小的好办法。",
    "故事的最后，大家都笑了。小蝶说：愿你今晚做一个甜甜的梦。",
]


def pick_lesson(keywords: list[str], rng: random.Random) -> str:
    joined = "".join(keywords)
    if "勇" in joined:
        return "原来，勇敢不是很大声，而是愿意再试一次。"
    if "分享" in joined:
        return "原来，分享以后，快乐会变成两份、三份、好多份。"
    if "朋友" in joined or "合作" in joined or "一起" in joined:
        return "原来，把自己的想法说出来，大家就能一起把事情变好。"
    return rng.choice(LESSONS)


@dataclass
class Story:
    title: str
    paragraphs: list[str]

    @property
    def full_text(self) -> str:
        return "\n".join([f"《{self.title}》", *self.paragraphs])


def split_keywords(raw: str) -> list[str]:
    parts = re.split(r"[,\s，、。；;|/]+", raw.strip())
    keywords = []
    for item in parts:
        item = item.strip()
        if item and item not in keywords:
            keywords.append(item)
    return keywords[:6]


def pick_keywords_text(keywords: list[str]) -> str:
    if not keywords:
        return "想象"
    if len(keywords) == 1:
        return keywords[0]
    return "、".join(keywords[:-1]) + "和" + keywords[-1]


def make_seed(keywords: list[str], length: str) -> int:
    text = "|".join(keywords) + "|" + length
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text))


def generate_story(keywords: list[str], length: str = "mini") -> Story:
    if not keywords:
        keywords = ["星星", "勇气"]

    rng = random.Random(make_seed(keywords, length))
    subject = keywords[0]
    keyword_text = pick_keywords_text(keywords)
    place = rng.choice(PLACES)
    helper = rng.choice(HELPERS)
    title = f"{subject}的小小故事"

    opening = f"{rng.choice(OPENINGS)}这一次，小朋友给小蝶的关键词是：{keyword_text}。"
    setting = f"在{place}，有一个叫安安的小朋友，发现了和{subject}有关的一件小事。安安心里有点好奇，也有一点点紧张。"
    challenge = f"{rng.choice(CHALLENGES)}安安先试了一次，没有成功；又试了一次，还是差一点点。安安没有哭闹，而是深深吸了一口气。"
    solution = f"这时，{helper}走过来，陪安安一起想办法。大家把{keyword_text}放进故事里，一边想，一边试，慢慢找到了新的方向。"
    lesson = pick_lesson(keywords, rng)
    memory = f"后来，安安把这个故事记在心里。每当想起{keyword_text}，安安就会记得：慢慢来，认真听，勇敢试一试。"
    ending = rng.choice(GOOD_NIGHT)

    extras = [
        f"安安发现，原来每一个关键词都像一盏小灯。{keyword_text}连在一起，就能照出一条新的小路。",
        "大家轮流说出自己的想法，有的人说得快，有的人说得慢，可每一个想法都被认真听见了。",
        "安安把办法轻轻说出来，又邀请身边的小伙伴一起完成。事情没有一下子变简单，可大家的心都变得更安定。",
        f"小蝶也悄悄加入进来，把{subject}变成故事里的小小线索，提醒大家不要着急。",
    ]

    if length == "mini":
        paragraphs = [
            (
                f"小蝶收到关键词：{keyword_text}。"
                f"安安在{place}发现了{subject}。"
                f"一开始没做好，安安有点着急。"
                f"{helper}陪安安一起想办法，大家终于完成了小小任务。"
                f"{lesson}"
            )
        ]
    elif length == "short":
        paragraphs = [
            opening,
            setting + challenge,
            solution,
            lesson + memory,
            ending,
        ]
    elif length == "medium":
        paragraphs = [opening, setting, challenge, solution, *extras[:2], lesson, memory, ending]
    else:
        paragraphs = [opening, setting, challenge, solution, *extras, lesson, memory, ending]

    return Story(title=title, paragraphs=paragraphs)


def build_tts_args(args: argparse.Namespace, output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        engine=args.engine,
        speed=args.speed,
        threads=args.threads,
        espeak_voice="cmn",
        espeak_speed=145,
        pitch=45,
        amplitude=150,
        output=str(output),
        player=args.player,
        no_play=args.no_play,
    )


def save_story(story: Story) -> Path:
    STORY_DIR.mkdir(parents=True, exist_ok=True)
    path = STORY_DIR / "xiaodie_last_story.txt"
    path.write_text(story.full_text + "\n", encoding="utf-8")
    return path


def chunk_paragraphs(story: Story, chunk_size: int) -> list[str]:
    chunks = []
    intro = f"今天的故事叫做，{story.title}。{story.paragraphs[0]}"
    chunks.append(intro)

    rest = story.paragraphs[1:]
    for start in range(0, len(rest), chunk_size):
        chunks.append("".join(rest[start : start + chunk_size]))
    return chunks


def speak_story(story: Story, args: argparse.Namespace) -> None:
    xiaodie_tts.set_player_preference(build_tts_args(args, Path.home() / "xiaodie" / "audio" / "dummy.wav"))
    segments = chunk_paragraphs(story, args.chunk_paragraphs)

    for index, segment in enumerate(segments, start=1):
        output = Path.home() / "xiaodie" / "audio" / f"story_segment_{index:02d}.wav"
        tts_args = build_tts_args(args, output)
        xiaodie_tts.set_player_preference(tts_args)
        print(f"\n[xiaodie-story] 第 {index}/{len(segments)} 段：{segment}", flush=True)
        xiaodie_tts.speak(segment, tts_args)


def run_once(raw_keywords: str, args: argparse.Namespace) -> None:
    keywords = split_keywords(raw_keywords)
    if not keywords:
        print("[xiaodie-story] 没有收到关键词，使用默认关键词：星星、勇气。", flush=True)
    story = generate_story(keywords, args.length)
    story_path = save_story(story)

    print("\n[xiaodie-story] 已生成完整故事：", flush=True)
    print(story.full_text, flush=True)
    print(f"\n[xiaodie-story] 故事文本已保存: {story_path}", flush=True)

    if args.text_only:
        return
    speak_story(story, args)


def interactive(args: argparse.Namespace) -> None:
    print("小蝶故事机已启动。输入关键词，例如：星星 勇气 分享。输入 q 退出。", flush=True)
    while True:
        try:
            raw = input("关键词> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if raw.strip().lower() in {"q", "quit", "exit"}:
            return
        run_once(raw, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XiaoDie keyword-to-story voice prototype for K1.")
    parser.add_argument("keywords", nargs="*", help="Story keywords. If omitted, enter interactive mode.")
    parser.add_argument("--length", choices=["mini", "short", "medium", "long"], default="mini")
    parser.add_argument("--speed", type=float, default=0.92, help="TTS speed. Larger is faster.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--engine", choices=["auto", "sherpa", "espeak"], default="auto")
    parser.add_argument("--player", choices=["auto", "aplay", "pw-play"], default="auto")
    parser.add_argument("--chunk-paragraphs", type=int, default=2, help="Paragraphs per spoken chunk.")
    parser.add_argument("--text-only", action="store_true", help="Generate story text without speaking.")
    parser.add_argument("--no-play", action="store_true", help="Generate audio files without playback.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.keywords:
        run_once(" ".join(args.keywords), args)
    else:
        interactive(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[xiaodie-story] 出错了: {exc}", file=sys.stderr, flush=True)
        raise
