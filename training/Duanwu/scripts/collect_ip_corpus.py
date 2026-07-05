#!/usr/bin/env python3
import argparse
import html
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

import yaml


SCRIPT_RE = re.compile(r"<(script|style|noscript|svg).*?</\1>", re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"[ \t\f\v]+")
ZH_RE = re.compile(r"[\u4e00-\u9fff]")


def normalize(text: str) -> str:
    text = html.unescape(text or "")
    text = SCRIPT_RE.sub(" ", text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|li|h1|h2|h3|section|article)>", "\n", text, flags=re.I)
    text = TAG_RE.sub(" ", text)
    text = text.replace("\xa0", " ")
    text = SPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    seen = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower() in seen:
            continue
        seen.add(line.lower())
        lines.append(line)
    return "\n".join(lines).strip()


def title_from_html(raw: str, fallback: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.I | re.S)
    if match:
        title = normalize(match.group(1))
        title = re.sub(r"\s+", " ", title).strip()
        if title:
            return title[:160]
    return fallback


def fetch_url(url: str, timeout: int) -> tuple[str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "XiaoDie-RAG-Builder/1.0 (+authorized research corpus)",
            "Accept": "text/html,application/xhtml+xml,text/plain,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    text = raw.decode(charset, errors="ignore")
    return title_from_html(text, url), normalize(text)


def wikipedia_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = parsed.path.rsplit("/", 1)[-1]
    return unquote(name).replace("_", " ")


def fetch_wikipedia_extract(url: str, timeout: int) -> tuple[str, str]:
    title = wikipedia_title_from_url(url)
    api = (
        "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1"
        f"&redirects=1&format=json&titles={quote(title.replace(' ', '_'))}"
    )
    request = Request(
        api,
        headers={
            "User-Agent": "XiaoDie-RAG-Builder/1.0 (+authorized research corpus)",
            "Accept": "application/json,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        obj = json.loads(response.read().decode("utf-8", errors="ignore"))
    pages = obj.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract", "")
        page_title = page.get("title") or title
        if extract:
            return page_title, normalize(extract)
    return title, ""


def read_authorized_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".srt", ".vtt"}:
        text = re.sub(r"(?m)^\d+\s*$", "", text)
        text = re.sub(r"(?m)^\d\d:\d\d:\d\d[,.]\d+\s+-->\s+\d\d:\d\d:\d\d[,.]\d+.*$", "", text)
        text = re.sub(r"(?m)^WEBVTT.*$", "", text)
    if path.suffix.lower() in {".html", ".htm"}:
        return normalize(text)
    if path.suffix.lower() == ".json":
        try:
            obj = json.loads(text)
            return normalize(json.dumps(obj, ensure_ascii=False))
        except Exception:
            return normalize(text)
    return normalize(text)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="E:/Duanwu/configs/ip_rag_sources.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw_dir = Path(cfg["paths"]["raw_dir"])
    auth_dir = Path(cfg["paths"]["authorized_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    auth_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    timeout = int(cfg["defaults"].get("request_timeout_sec", 45))
    min_chars = int(cfg["defaults"].get("min_text_chars", 180))
    rows: list[dict[str, Any]] = []
    report_lines = [
        "# IP RAG Source Report",
        "",
        "| Franchise | Source | URL/File | Type | License | Status | Text Chars |",
        "|---|---|---|---|---|---|---:|",
    ]

    for franchise in cfg["franchises"]:
        fid = franchise["id"]
        for source in franchise.get("sources", []):
            status = "ok"
            title = source["name"]
            text = ""
            try:
                if source.get("source_type") == "wikipedia":
                    title, text = fetch_wikipedia_extract(source["url"], timeout)
                else:
                    title, text = fetch_url(source["url"], timeout)
                time.sleep(0.8)
                if len(text) < min_chars:
                    status = "too_short_after_fetch"
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                status = f"fetch_error:{type(exc).__name__}"
            if text and len(text) >= min_chars:
                rows.append(
                    {
                        "id": f"{fid}_web_{len(rows)+1:06d}",
                        "franchise": fid,
                        "franchise_zh": franchise["zh_name"],
                        "source": source["name"],
                        "source_type": source.get("source_type", "web"),
                        "license": source.get("license", "unknown"),
                        "url": source["url"],
                        "title": title,
                        "text": text,
                        "authorization": franchise.get("authorization", "unknown"),
                    }
                )
            report_lines.append(
                f"| {fid} | {source['name']} | {source['url']} | {source.get('source_type','web')} | "
                f"{source.get('license','unknown')} | {status} | {len(text)} |"
            )

        franchise_auth_dir = auth_dir / fid
        if franchise_auth_dir.exists():
            for path in sorted(franchise_auth_dir.rglob("*")):
                if path.is_dir() or path.suffix.lower() not in {".txt", ".md", ".html", ".htm", ".json", ".jsonl", ".srt", ".vtt"}:
                    continue
                try:
                    text = read_authorized_file(path)
                    status = "ok" if len(text) >= min_chars else "too_short"
                except Exception as exc:
                    text = ""
                    status = f"read_error:{type(exc).__name__}"
                if text and len(text) >= min_chars:
                    rows.append(
                        {
                            "id": f"{fid}_authorized_{len(rows)+1:06d}",
                            "franchise": fid,
                            "franchise_zh": franchise["zh_name"],
                            "source": path.name,
                            "source_type": "user_authorized_file",
                            "license": "user-provided-authorized",
                            "url": str(path),
                            "title": path.stem,
                            "text": text,
                            "authorization": franchise.get("authorization", "unknown"),
                        }
                    )
                report_lines.append(
                    f"| {fid} | {path.name} | {path} | user_authorized_file | user-provided-authorized | {status} | {len(text)} |"
                )

    out = raw_dir / "ip_web_records.jsonl"
    write_jsonl(out, rows)
    (reports_dir / "ip_source_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} records -> {out}")
    if not rows:
        print("No IP source text was collected.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
