#!/usr/bin/env python3
import argparse
import html
import io
import json
import random
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

import yaml
from huggingface_hub import hf_hub_download


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def chinese_count(text: str) -> int:
    return len(CHINESE_RE.findall(text or ""))


def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def row(source: dict[str, Any], idx: int, title: str, text: str, tags: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": f"{source['name']}_{idx:07d}",
        "source": source["name"],
        "license": source.get("license", "unknown"),
        "url": source.get("url", ""),
        "title": normalize_text(title)[:120] or "未命名故事",
        "text": normalize_text(text),
        "age_level": "unknown",
        "tags": sorted(set(["children_story", "chinese"] + (tags or []))),
    }


def extract_text_from_json(obj: Any, text_fields: list[str] | None = None) -> tuple[str, str]:
    text_fields = text_fields or ["text", "story", "content", "article", "response", "translation", "zh", "output"]
    if isinstance(obj, str):
        return "", obj
    if isinstance(obj, dict):
        title = str(obj.get("title") or obj.get("name") or obj.get("source") or "")
        for field in text_fields:
            value = obj.get(field)
            if isinstance(value, str) and chinese_count(value) >= 20:
                return title, value
        strings = [v for v in obj.values() if isinstance(v, str)]
        strings.sort(key=chinese_count, reverse=True)
        if strings:
            return title, strings[0]
    return "", ""


def collect_tinystories(source: dict[str, Any], raw_dir: Path, seed: int) -> tuple[int, str]:
    out = raw_dir / source["name"] / "records.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=source["dataset_id"],
        filename=source["file"],
        repo_type="dataset",
        local_dir=str(out.parent / "hf_download"),
        local_dir_use_symlinks=False,
    )
    rng = random.Random(seed)
    limit = int(source.get("sample_limit", 6000))
    reservoir: list[dict[str, Any]] = []
    seen = 0

    def maybe_add(title: str, text: str) -> None:
        nonlocal seen
        text = normalize_text(text)
        if chinese_count(text) < 80:
            return
        seen += 1
        item = row(source, seen, title or f"TinyStories Chinese {seen}", text, ["tinystories"])
        if len(reservoir) < limit:
            reservoir.append(item)
        else:
            j = rng.randint(0, seen - 1)
            if j < limit:
                reservoir[j] = item

    with tarfile.open(local, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.lower()
            if not any(name.endswith(ext) for ext in [".jsonl", ".json", ".txt"]):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            if name.endswith(".txt"):
                text = extracted.read().decode("utf-8", errors="ignore")
                for part in re.split(r"\n\s*\n", text):
                    maybe_add(Path(member.name).stem, part)
                continue
            wrapper = io.TextIOWrapper(extracted, encoding="utf-8", errors="ignore")
            for line in wrapper:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    title, text = extract_text_from_json(obj)
                    maybe_add(title, text)
                except Exception:
                    maybe_add(Path(member.name).stem, line)

    count = write_jsonl(out, reservoir)
    return count, str(out)


def collect_cosmopedia(source: dict[str, Any], raw_dir: Path, seed: int) -> tuple[int, str]:
    out = raw_dir / source["name"] / "records.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    limit = int(source.get("sample_limit", 1200))
    text_fields = source.get("text_fields") or ["text", "content", "article", "response"]
    rows: list[dict[str, Any]] = []
    try:
        from datasets import load_dataset

        dataset = load_dataset(source["dataset_id"], split=source.get("split", "train"), streaming=True)
        for idx, item in enumerate(dataset, start=1):
            title, text = extract_text_from_json(item, text_fields)
            text = normalize_text(text)
            if chinese_count(text) >= 120 and any(k in text for k in ["故事", "孩子", "朋友", "学习", "动物", "想象"]):
                rows.append(row(source, len(rows) + 1, title or f"Cosmopedia {idx}", text, ["cosmopedia"]))
            if len(rows) >= limit:
                break
            if idx >= limit * 40:
                break
    except Exception as exc:
        (out.parent / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    count = write_jsonl(out, rows)
    return count, str(out)


def download_zip(url: str, target: Path) -> None:
    with urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())


def extract_storybooks_zh_page(path: Path) -> tuple[str, str, str]:
    page = path.read_text(encoding="utf-8", errors="ignore")
    title_match = re.search(r'<h1>.*?<span class="def">(.*?)</span>', page, flags=re.S)
    if not title_match:
        title_match = re.search(r"<title>(.*?)</title>", page, flags=re.S | re.I)
    title = normalize_text(title_match.group(1) if title_match else path.parent.name)
    title = re.sub(r"\s*-\s*中文故事集\s*$", "", title)

    level_match = re.search(r"/level([1-5])", page)
    level = level_match.group(1) if level_match else ""

    paragraphs: list[str] = []
    for match in re.finditer(
        r'<div class="[^"]*\blevel\d+-txt\b[^"]*\bdef\b[^"]*">\s*<h3>(.*?)</h3>\s*</div>',
        page,
        flags=re.S,
    ):
        text = normalize_text(match.group(1))
        if text and chinese_count(text) > 0:
            paragraphs.append(text)

    if not paragraphs:
        for match in re.finditer(
            r'<div class="[^"]*\blevel\d+-txt\b[^"]*\bl1\b[^"]*">\s*<h3>(.*?)</h3>\s*</div>',
            page,
            flags=re.S,
        ):
            text = normalize_text(match.group(1))
            if text and chinese_count(text) > 0:
                paragraphs.append(text)

    deduped: list[str] = []
    for paragraph in paragraphs:
        if not deduped or deduped[-1] != paragraph:
            deduped.append(paragraph)
    return title, "\n".join(deduped), level


def collect_storybooks(source: dict[str, Any], raw_dir: Path, seed: int) -> tuple[int, str]:
    repo_dir = raw_dir / source["name"] / "repo"
    out = raw_dir / source["name"] / "records.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        if shutil.which("git"):
            subprocess.run(["git", "clone", "--depth", "1", source["repo"], str(repo_dir)], check=True)
        else:
            zip_path = out.parent / "storybooks.zip"
            download_zip(source["repo"].rstrip(".git") + "/archive/refs/heads/master.zip", zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(out.parent)
            candidates = [p for p in out.parent.iterdir() if p.is_dir() and p.name.startswith("storybooks")]
            if candidates:
                candidates[0].rename(repo_dir)

    rows: list[dict[str, Any]] = []
    limit = int(source.get("sample_limit", 1200))
    zh_story_dir = repo_dir / "stories" / "zh"
    story_pages = sorted(zh_story_dir.glob("[0-9][0-9][0-9][0-9]/index.html"))
    for path in story_pages:
        if len(rows) >= limit:
            break
        try:
            title, text, level = extract_storybooks_zh_page(path)
        except Exception:
            continue
        if chinese_count(text) < 40:
            continue
        age_level = "preschool" if level in {"1", "2"} else "primary"
        item = row(source, len(rows) + 1, title, text, ["storybooks", f"level{level}" if level else "level_unknown"])
        item["age_level"] = age_level
        item["url"] = f"{source.get('url', '').rstrip('/')}/tree/master/stories/zh/{path.parent.name}"
        rows.append(item)
    count = write_jsonl(out, rows)
    return count, str(out)


GUTENBERG_ADAPTATIONS = [
    ("竹林里的小灯", "从前有个孩子在竹林边迷了路。他看见一盏小灯，就想自己跑过去。可是风把叶子吹得沙沙响，他有一点害怕。孩子停下来，先深深吸一口气，再回头叫朋友。朋友听见了，牵着他的手一起走。他们发现，小灯不是怪东西，而是邻居挂在门口给路人看的灯。孩子学会了，害怕时可以求助，勇敢不是一个人硬撑，而是愿意把心里的话说出来。"),
    ("会说谢谢的小碗", "一个小碗每天陪孩子吃饭。孩子有时吃得很快，忘了说谢谢。一天，小碗轻轻碰了一下勺子，好像在提醒他慢一点。孩子看见妈妈把饭菜端来，也看见爸爸把桌子擦干净。他忽然明白，一顿饭里有很多人的照顾。于是他说：谢谢你们。小碗安安静静地亮了一下，好像也在微笑。"),
    ("月光下的纸船", "小女孩折了一只纸船，想把它放进小溪。朋友也想玩，可是只有一张纸。小女孩想了想，说：我们可以一起折，一起给纸船起名字。朋友点点头。他们把纸船叫作月亮号，让它在浅浅的水边慢慢走。纸船走不远，但两个孩子都很开心，因为分享让一个小玩具变成了两个心里的故事。"),
]


def collect_gutenberg(source: dict[str, Any], raw_dir: Path, seed: int) -> tuple[int, str]:
    out_dir = raw_dir / source["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, url in enumerate(source.get("urls", []), start=1):
        try:
            with urlopen(url, timeout=120) as response:
                text = response.read().decode("utf-8", errors="ignore")
            (out_dir / f"gutenberg_{idx}.txt").write_text(text, encoding="utf-8")
        except Exception as exc:
            (out_dir / f"gutenberg_{idx}.error.txt").write_text(str(exc), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    out = out_dir / "records.jsonl"
    count = write_jsonl(out, rows)
    return count, str(out)


def collect_safety_seed(source: dict[str, Any], raw_dir: Path, seed: int) -> tuple[int, str]:
    rng = random.Random(seed)
    names = ["安安", "朵朵", "米米", "可可", "乐乐", "阳阳"]
    objects = ["月亮", "小雨伞", "积木", "彩笔", "小火车", "牙刷", "图书馆"]
    themes = ["分享", "勇气", "诚实", "礼貌", "合作", "等待", "认真倾听"]
    places = ["幼儿园的图书角", "窗边的小桌旁", "种着花的小院子", "安静的睡前房间"]
    rows = []
    limit = int(source.get("sample_limit", 200))
    for idx in range(1, limit + 1):
        name = rng.choice(names)
        obj = rng.choice(objects)
        theme = rng.choice(themes)
        place = rng.choice(places)
        title = f"{obj}和{theme}的小故事"
        text = (
            f"《{title}》\n"
            f"今天，{name}在{place}看见了{obj}。{obj}安安静静的，好像在等一个温柔的小故事。\n"
            f"{name}想和朋友一起玩，可是心里有一点点紧张。小蝶轻轻说：先吸一口气，再把心里的话慢慢说出来。\n"
            f"{name}照着做了，对朋友说：我想一起玩，也想学习{theme}。朋友认真听完，点点头说：我们可以轮流说，也可以一起想办法。\n"
            f"他们没有抢，也没有着急，而是把{obj}放在中间，一人说一句，一人试一次。慢慢地，小小的问题变成了新的游戏。\n"
            f"故事结束时，{name}明白了，{theme}不是很大的道理，而是每天都可以练习的小办法。小蝶说：愿你带着这个温暖的办法安心睡觉，晚安。"
        )
        rows.append(row(source, idx, title, text, ["safety_seed"]))
    out = raw_dir / source["name"] / "records.jsonl"
    count = write_jsonl(out, rows)
    return count, str(out)


def write_license_report(rows: list[dict[str, Any]], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# License Report - Duanwu Round 2",
        "",
        "| Source | URL | License | Used Rows | Status | Notes |",
        "|---|---|---|---:|---|---|",
    ]
    for item in rows:
        lines.append(
            f"| {item['name']} | {item.get('url','')} | {item.get('license','unknown')} | "
            f"{item.get('rows',0)} | {item.get('status','')} | {item.get('notes','')} |"
        )
    (reports_dir / "license_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="E:/Duanwu/configs/datasets.yaml")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    raw_dir = Path(cfg["output"]["raw_dir"])
    reports_dir = Path(cfg["output"]["reports_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["defaults"].get("seed", 20260618))
    report_rows = []
    for source in cfg["sources"]:
        if not source.get("enabled", False):
            report_rows.append({**source, "rows": 0, "status": "disabled"})
            continue
        try:
            if source["type"] == "hf_tar_dataset":
                count, path = collect_tinystories(source, raw_dir, seed)
            elif source["type"] == "hf_streaming_dataset":
                count, path = collect_cosmopedia(source, raw_dir, seed)
            elif source["type"] == "git_text_repo":
                count, path = collect_storybooks(source, raw_dir, seed)
            elif source["type"] == "gutenberg_text":
                count, path = collect_gutenberg(source, raw_dir, seed)
            elif source["type"] == "local_safety_seed":
                count, path = collect_safety_seed(source, raw_dir, seed)
            else:
                count, path = 0, ""
            report_rows.append({**source, "rows": count, "status": f"ok: {path}"})
            print(f"{source['name']}: {count} rows")
        except Exception as exc:
            report_rows.append({**source, "rows": 0, "status": f"error: {type(exc).__name__}: {exc}"})
            print(f"{source['name']}: ERROR {type(exc).__name__}: {exc}")
    write_license_report(report_rows, reports_dir)


if __name__ == "__main__":
    main()
