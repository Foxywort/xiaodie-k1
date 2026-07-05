#!/usr/bin/env python3
"""Local Chromium web app controller for XiaoDie on K1."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


START_CMD = ["sudo", "-n", "/usr/local/bin/xiaodie-start"]
STOP_CMD = ["sudo", "-n", "/usr/local/bin/xiaodie-stop"]
APP_DIR = Path("/home/vicky/xiaodie/app")
ASSET_DIR = APP_DIR / "assets"


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>小蝶故事机</title>
  <style>
    :root {
      --ink: #18202f;
      --muted: #697386;
      --panel: rgba(255,255,255,0.84);
      --panel-strong: rgba(255,255,255,0.96);
      --blue: #2563eb;
      --cyan: #0891b2;
      --green: #15935f;
      --red: #dc2626;
      --line: rgba(24,32,47,0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Noto Sans CJK SC", "Microsoft YaHei", system-ui, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 10%, rgba(37,99,235,0.22), transparent 32%),
        radial-gradient(circle at 88% 18%, rgba(20,184,166,0.20), transparent 28%),
        linear-gradient(135deg, #eef4ff 0%, #f8fbff 42%, #edf7f4 100%);
      overflow: hidden;
    }
    .app {
      height: 100vh;
      display: grid;
      grid-template-columns: 330px 1fr;
      gap: 18px;
      padding: 18px;
    }
    .side, .main {
      border: 1px solid var(--line);
      background: var(--panel);
      backdrop-filter: blur(18px);
      box-shadow: 0 24px 60px rgba(35,48,73,0.16);
    }
    .side {
      border-radius: 22px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 18px;
    }
    .spacemitLogo {
      width: 168px;
      max-width: 100%;
      height: auto;
      display: block;
      border-radius: 10px;
      background: #fff;
      padding: 8px 10px;
      box-shadow: 0 10px 24px rgba(24,32,47,0.12);
    }
    .brandText { margin-top: 16px; }
    h1 {
      margin: 0;
      font-size: 27px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .sub {
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .statusCard {
      margin: 16px 0 18px;
      border-radius: 18px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
      padding: 18px;
    }
    .statusTop {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }
    .statusLabel {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .pill {
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
      font-weight: 800;
      color: white;
      background: #667085;
      white-space: nowrap;
    }
    .pill.running { background: var(--green); }
    .pill.starting { background: #b7791f; }
    .pill.recording { background: #7c3aed; }
    .pill.speaking { background: var(--cyan); }
    .meter {
      height: 10px;
      border-radius: 999px;
      background: #dbe3ef;
      overflow: hidden;
    }
    .meter > div {
      height: 100%;
      width: 34%;
      border-radius: 999px;
      background: linear-gradient(90deg, #2563eb, #14b8a6);
      animation: pulsebar 1.6s ease-in-out infinite;
    }
    @keyframes pulsebar {
      0%, 100% { transform: translateX(-40%); width: 30%; }
      50% { transform: translateX(160%); width: 48%; }
    }
    .buttons {
      display: grid;
      gap: 11px;
    }
    button {
      height: 46px;
      border: 0;
      border-radius: 12px;
      font-size: 15px;
      font-weight: 800;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 9px;
      transition: transform 0.12s ease, box-shadow 0.12s ease, opacity 0.12s ease;
    }
    button:active { transform: translateY(1px); }
    button:disabled { opacity: 0.45; cursor: default; }
    .primary {
      color: #fff;
      background: linear-gradient(135deg, #2563eb, #0891b2);
      box-shadow: 0 12px 24px rgba(37,99,235,0.22);
    }
    .danger { color: #fff; background: var(--red); }
    .soft { color: #253044; background: #e8edf6; }
    .hint {
      margin-top: auto;
      padding: 16px;
      border-radius: 16px;
      background: rgba(37,99,235,0.08);
      color: #33415c;
      font-size: 13px;
      line-height: 1.55;
    }
    .main {
      min-width: 0;
      border-radius: 22px;
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }
    .mainHead {
      padding: 20px 22px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 18px;
      background: rgba(255,255,255,0.52);
    }
    .contestLogo {
      height: 56px;
      max-width: min(340px, 100%);
      object-fit: contain;
      border-radius: 10px;
      background: white;
      padding: 6px 10px;
      box-shadow: 0 8px 18px rgba(24,32,47,0.10);
    }
    .mainTitle {
      font-size: 18px;
      font-weight: 900;
    }
    .chips {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-start;
      width: 100%;
    }
    .chip {
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(24,32,47,0.07);
      color: #3b465c;
      font-size: 12px;
      font-weight: 800;
    }
    .logWrap {
      min-height: 0;
      padding: 16px;
      background: rgba(248,250,252,0.56);
    }
    #log {
      height: 100%;
      margin: 0;
      overflow: auto;
      padding: 18px;
      border-radius: 16px;
      color: #d9e8ff;
      background: #101828;
      border: 1px solid rgba(16,24,40,0.2);
      font: 13px/1.55 "Cascadia Mono", "Consolas", monospace;
      white-space: pre-wrap;
      word-break: break-word;
    }
    @media (max-width: 760px) {
      .app { grid-template-columns: 1fr; overflow: auto; }
      body { overflow: auto; }
      .side { min-height: 410px; }
      .main { min-height: 460px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="side">
      <div class="brand">
        <div>
          <img class="spacemitLogo" src="/assets/spacemit-logo.png" alt="进迭时空 SPACEMIT" />
          <div class="brandText">
          <h1>小蝶故事机</h1>
          <div class="sub">端侧语音故事助手</div>
          </div>
        </div>
      </div>
      <div class="statusCard">
        <div class="statusTop">
          <div>
            <div class="statusLabel">当前状态</div>
            <div id="detail" class="sub">等待启动</div>
          </div>
          <div id="pill" class="pill">未启动</div>
        </div>
        <div class="meter"><div id="bar"></div></div>
      </div>
      <div class="buttons">
        <button id="start" class="primary">启动小蝶</button>
        <button id="stop" class="danger">停止服务</button>
        <button id="clear" class="soft">清空日志</button>
      </div>
      <div class="hint">
        启动后等待 ASR ready。之后按住实体按钮说话，松开后小蝶会识别并讲故事。运行中再次按下按钮会打断当前任务。
      </div>
    </aside>
    <main class="main">
      <header class="mainHead">
        <div>
          <div class="mainTitle">实时运行日志</div>
          <div class="sub">ASR、故事生成和 TTS 状态会显示在这里</div>
        </div>
        <img class="contestLogo" src="/assets/contest-logo.jpg" alt="AI赋能设计，设计点亮AI" />
        <div class="chips">
          <span class="chip">GPIO 35</span>
          <span class="chip">Chaowen Full</span>
          <span class="chip">本地服务</span>
        </div>
      </header>
      <section class="logWrap">
        <pre id="log"></pre>
      </section>
    </main>
  </div>
  <script>
    const logEl = document.getElementById("log");
    const pill = document.getElementById("pill");
    const detail = document.getElementById("detail");
    const startBtn = document.getElementById("start");
    const stopBtn = document.getElementById("stop");
    const clearBtn = document.getElementById("clear");
    let lastSeq = 0;

    function setStatus(status, text) {
      pill.className = "pill";
      if (status === "running") pill.classList.add("running");
      if (status === "starting") pill.classList.add("starting");
      if (status === "recording") pill.classList.add("recording");
      if (status === "speaking") pill.classList.add("speaking");
      pill.textContent = text || status;
      detail.textContent = text || status;
      startBtn.disabled = ["running", "starting", "recording", "speaking"].includes(status);
    }
    async function post(path) {
      const res = await fetch(path, { method: "POST" });
      return await res.json();
    }
    startBtn.onclick = () => post("/api/start").then(refresh);
    stopBtn.onclick = () => post("/api/stop").then(refresh);
    clearBtn.onclick = () => { logEl.textContent = ""; };

    async function refresh() {
      try {
        const res = await fetch(`/api/status?after=${lastSeq}`);
        const data = await res.json();
        setStatus(data.status, data.status_text);
        for (const item of data.logs) {
          lastSeq = Math.max(lastSeq, item.seq);
          logEl.textContent += `[${item.time}] ${item.text}\n`;
        }
        logEl.scrollTop = logEl.scrollHeight;
      } catch (err) {
        setStatus("stopped", "连接中断");
      }
    }
    setInterval(refresh, 700);
    refresh();
  </script>
</body>
</html>
"""


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None
        self.seq = 0
        self.logs: deque[dict[str, Any]] = deque(maxlen=500)
        self.status = "stopped"
        self.status_text = "未启动"

    def add_log(self, text: str) -> None:
        text = text.rstrip()
        if not text:
            return
        with self.lock:
            self.seq += 1
            self.logs.append(
                {
                    "seq": self.seq,
                    "time": time.strftime("%H:%M:%S"),
                    "text": text,
                }
            )
            if "ASR daemon starting" in text:
                self.status, self.status_text = "starting", "加载 ASR"
            elif "[asr_daemon] ready" in text or "ready: hold GPIO" in text:
                self.status, self.status_text = "running", "运行中"
            elif "recording..." in text:
                self.status, self.status_text = "recording", "录音中"
            elif "DeepSeek+TTS started" in text:
                self.status, self.status_text = "speaking", "讲故事中"

    def snapshot(self, after: int) -> dict[str, Any]:
        with self.lock:
            proc = self.proc
            if proc is not None and proc.poll() is not None:
                self.proc = None
                self.status, self.status_text = "stopped", "已停止"
            return {
                "status": self.status,
                "status_text": self.status_text,
                "logs": [x for x in self.logs if x["seq"] > after],
            }


STATE = State()


def reader_thread(proc: subprocess.Popen[str]) -> None:
    if not proc.stdout:
        return
    for line in proc.stdout:
        STATE.add_log(line)
    code = proc.wait()
    STATE.add_log(f"后台服务退出，退出码 {code}")
    with STATE.lock:
        if STATE.proc is proc:
            STATE.proc = None
            STATE.status, STATE.status_text = "stopped", "已停止"


def start_service() -> None:
    with STATE.lock:
        if STATE.proc is not None and STATE.proc.poll() is None:
            STATE.add_log("小蝶已经在运行")
            return
        STATE.status, STATE.status_text = "starting", "启动中"
    STATE.add_log("正在启动小蝶后台服务")
    proc = subprocess.Popen(
        START_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    with STATE.lock:
        STATE.proc = proc
    threading.Thread(target=reader_thread, args=(proc,), daemon=True).start()


def stop_service() -> None:
    with STATE.lock:
        proc = STATE.proc
        STATE.proc = None
        STATE.status, STATE.status_text = "stopped", "已停止"
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
    subprocess.run(STOP_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    STATE.add_log("小蝶服务已停止")


class Handler(BaseHTTPRequestHandler):
    def _json(self, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path)
        if path.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.path.startswith("/assets/"):
            name = Path(path.path).name
            asset = ASSET_DIR / name
            if not asset.exists() or asset.parent != ASSET_DIR:
                self.send_error(404)
                return
            ctype = "image/png" if asset.suffix.lower() == ".png" else "image/jpeg"
            body = asset.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.path == "/api/status":
            after = 0
            if path.query.startswith("after="):
                try:
                    after = int(path.query.split("=", 1)[1])
                except ValueError:
                    after = 0
            self._json(STATE.snapshot(after))
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/start":
            try:
                start_service()
                self._json({"ok": True})
            except Exception as exc:
                STATE.add_log(f"启动失败：{exc}")
                self._json({"ok": False, "error": str(exc)})
            return
        if path == "/api/stop":
            stop_service()
            self._json({"ok": True})
            return
        self.send_error(404)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print("xiaodie_web_app_ok")
        return 0
    STATE.add_log("小蝶图形前端已启动")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    finally:
        stop_service()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
