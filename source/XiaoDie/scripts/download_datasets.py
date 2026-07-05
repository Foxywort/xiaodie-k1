#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download


THEMES = [
    ("月亮", "勇气", "分享"),
    ("星星", "朋友", "诚实"),
    ("积木", "合作", "耐心"),
    ("小雨伞", "礼貌", "等待"),
    ("彩虹", "想象", "帮助"),
    ("刷牙", "习惯", "坚持"),
    ("花园", "爱护", "责任"),
    ("小火车", "轮流", "规则"),
    ("云朵", "害怕", "安慰"),
    ("纸飞机", "尝试", "失败"),
]

NAMES = ["安安", "朵朵", "乐乐", "米米", "小宇", "甜甜", "阳阳", "可可"]
PLACES = ["幼儿园的图书角", "铺着软垫的活动室", "安静的午睡房", "窗边的小桌旁", "种着花的小院子"]
ENDINGS = [
    "小蝶轻轻说：今天的你也很棒，愿你带着这个温暖的办法安心睡觉。",
    "故事结束时，大家把笑容收进口袋里，准备迎接明天新的发现。",
    "从那以后，大家知道了，慢慢说、认真听，心里就会亮起小灯。",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_story(index: int, rng: random.Random, source: dict) -> dict:
    theme = THEMES[index % len(THEMES)]
    name = rng.choice(NAMES)
    place = rng.choice(PLACES)
    title = f"{theme[0]}里的小小办法"
    text = (
        f"《{title}》\n"
        f"今天，{name}在{place}看见了{theme[0]}。{theme[0]}安安静静的，好像在等一个小故事。\n"
        f"{name}想和朋友一起玩，可是心里有一点点紧张。朋友拿着玩具走过来，问：“我们可以一起玩吗？”"
        f"{name}先没有回答，只是低头看着自己的手。\n"
        f"小蝶用很轻的声音说：“紧张的时候，可以先吸一口气，再把心里的话慢慢说出来。”"
        f"{name}照着做了。{name}说：“我想一起玩，也想把我的{theme[0]}故事分享给你。”\n"
        f"朋友认真听完，没有笑话{name}，还说：“谢谢你告诉我。我们可以轮流说，也可以一起想办法。”"
        f"他们把{theme[1]}和{theme[2]}放进游戏里。第一次没有做好，大家没有着急；第二次，他们先商量，再动手。\n"
        f"慢慢地，小小的问题变成了新的游戏。{name}发现，勇敢不是很大声，分享也不是把东西全给别人，"
        f"而是愿意说出自己的想法，也愿意听见别人的想法。\n"
        f"{rng.choice(ENDINGS)}"
    )
    return {
        "id": f"{source['name']}_{index + 1:06d}",
        "source": source["name"],
        "license": source["license"],
        "title": title,
        "text": text,
        "age_level": source.get("age_level", "preschool"),
        "tags": ["children_story", "chinese", "project_seed"],
        "url": source.get("url", "local://"),
    }


def generate_project_seed(raw_dir: Path, source: dict, sample_count: int, seed: int) -> tuple[str, int]:
    rng = random.Random(seed)
    rows = [make_story(i, rng, source) for i in range(sample_count)]
    out = raw_dir / source["name"] / "records.jsonl"
    write_jsonl(out, rows)
    return str(out), len(rows)


def download_hf_metadata(raw_dir: Path, source: dict) -> tuple[str, int]:
    api = HfApi()
    info = api.dataset_info(source["dataset_id"])
    out_dir = raw_dir / source["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset_id": source["dataset_id"],
        "url": source["url"],
        "license": source["license"],
        "tags": info.tags or [],
        "siblings": [s.rfilename for s in info.siblings],
        "card_data": str(info.card_data),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        readme = hf_hub_download(source["dataset_id"], "README.md", repo_type="dataset")
        (out_dir / "README.md").write_text(Path(readme).read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    except Exception as exc:
        (out_dir / "README.error.txt").write_text(str(exc), encoding="utf-8")
    return str(out_dir), 0


def download_hf_files(raw_dir: Path, source: dict, allow_large: bool) -> tuple[str, int]:
    out_dir = raw_dir / source["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for filename in source.get("files", []):
        if not allow_large:
            (out_dir / f"{Path(filename).name}.skipped.txt").write_text(
                "Skipped by default. Re-run with --download-large to fetch this file.\n",
                encoding="utf-8",
            )
            continue
        local = hf_hub_download(source["dataset_id"], filename, repo_type="dataset")
        target = out_dir / Path(filename).name
        if not target.exists():
            target.write_bytes(Path(local).read_bytes())
        downloaded += 1
    return str(out_dir), downloaded


def write_license_report(rows: list[dict], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# License Report",
        "",
        "| Dataset | URL | License | Status | Notes |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row.get('url','')} | {row.get('license','unknown')} | "
            f"{row.get('status','')} | {row.get('notes','')} |"
        )
    (reports_dir / "license_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/generate legal raw story datasets.")
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--download-large", action="store_true", help="Allow multi-GB downloads.")
    parser.add_argument("--seed", type=int, default=20260618)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    raw_dir = Path(cfg["output"]["raw_dir"])
    reports_dir = Path(cfg["output"]["reports_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    license_rows = []
    for source in cfg["sources"]:
        if not source.get("enabled", False):
            license_rows.append({**source, "status": "disabled", "notes": source.get("notes", "")})
            continue
        try:
            if source["type"] == "project_generated":
                _, count = generate_project_seed(raw_dir, source, source.get("sample_count", 100), args.seed)
                status = f"generated {count} rows"
            elif source["type"] == "huggingface_metadata":
                download_hf_metadata(raw_dir, source)
                status = "metadata downloaded"
            elif source["type"] == "huggingface_file":
                _, count = download_hf_files(raw_dir, source, args.download_large)
                status = f"downloaded {count} files" if count else "large files skipped"
            else:
                status = "supported in config, disabled for safe default"
            license_rows.append({**source, "status": status, "notes": source.get("notes", "")})
        except Exception as exc:
            license_rows.append({**source, "status": f"error: {type(exc).__name__}", "notes": str(exc)})

    write_license_report(license_rows, reports_dir)
    print(f"wrote {reports_dir / 'license_report.md'}")


if __name__ == "__main__":
    main()
