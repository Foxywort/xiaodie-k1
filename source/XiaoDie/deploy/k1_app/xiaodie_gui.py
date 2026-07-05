#!/usr/bin/env python3
"""Graphical launcher for XiaoDie on the K1 board."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk, Pango  # noqa: E402


APP_CMD = ["sudo", "-n", "/usr/local/bin/xiaodie-start"]
STOP_CMD = [
    "sudo",
    "-n",
    "/usr/local/bin/xiaodie-stop",
]


class XiaoDieWindow(Gtk.Window):
    def __init__(self) -> None:
        super().__init__(title="小蝶故事机")
        self.set_default_size(900, 560)
        self.set_border_width(0)
        self.proc: subprocess.Popen[str] | None = None
        self.started_at: float | None = None

        self._build_css()
        self._build_ui()
        GLib.timeout_add(500, self._tick)
        self.connect("destroy", self._on_destroy)

    def _build_css(self) -> None:
        css = b"""
        window { background: #f6f7fb; }
        #sidebar { background: #20283a; color: #ffffff; }
        #title { font-size: 28px; font-weight: 700; color: #ffffff; }
        #subtitle { color: #cfd7e6; font-size: 13px; }
        #statusPill { border-radius: 18px; padding: 8px 14px; background: #667085; color: #ffffff; font-weight: 700; }
        #statusRunning { border-radius: 18px; padding: 8px 14px; background: #18875f; color: #ffffff; font-weight: 700; }
        #statusStarting { border-radius: 18px; padding: 8px 14px; background: #b7791f; color: #ffffff; font-weight: 700; }
        #statusStopped { border-radius: 18px; padding: 8px 14px; background: #667085; color: #ffffff; font-weight: 700; }
        button { border-radius: 8px; padding: 10px 16px; font-weight: 700; }
        #primaryBtn { background: #2f80ed; color: #ffffff; }
        #dangerBtn { background: #d92d20; color: #ffffff; }
        #softBtn { background: #e7ebf3; color: #263043; }
        #logView { font-family: Monospace; font-size: 12px; background: #101828; color: #d6e4ff; }
        #hint { color: #3b465c; }
        """
        provider = Gtk.CssProvider()
        provider.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(
            self.get_screen(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def _build_ui(self) -> None:
        root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        self.add(root)

        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18)
        sidebar.set_name("sidebar")
        sidebar.set_size_request(280, -1)
        sidebar.set_border_width(24)
        root.pack_start(sidebar, False, False, 0)

        title = Gtk.Label(label="小蝶故事机", xalign=0)
        title.set_name("title")
        sidebar.pack_start(title, False, False, 0)

        subtitle = Gtk.Label(
            label="按住按钮说话，松开后小蝶讲故事。运行中再次按下按钮会立即打断。",
            xalign=0,
            wrap=True,
        )
        subtitle.set_name("subtitle")
        sidebar.pack_start(subtitle, False, False, 0)

        self.status = Gtk.Label(label="未启动")
        self.status.set_name("statusStopped")
        self.status.set_xalign(0.5)
        sidebar.pack_start(self.status, False, False, 4)

        self.start_btn = Gtk.Button(label="启动小蝶")
        self.start_btn.set_name("primaryBtn")
        self.start_btn.connect("clicked", lambda _b: self.start_service())
        sidebar.pack_start(self.start_btn, False, False, 0)

        self.stop_btn = Gtk.Button(label="停止")
        self.stop_btn.set_name("dangerBtn")
        self.stop_btn.connect("clicked", lambda _b: self.stop_service())
        sidebar.pack_start(self.stop_btn, False, False, 0)

        self.clear_btn = Gtk.Button(label="清空日志")
        self.clear_btn.set_name("softBtn")
        self.clear_btn.connect("clicked", lambda _b: self.log_buffer.set_text(""))
        sidebar.pack_start(self.clear_btn, False, False, 0)

        hint = Gtk.Label(
            label="启动后等待 ASR ready。之后只需要操作实体按钮。",
            xalign=0,
            wrap=True,
        )
        hint.set_name("subtitle")
        sidebar.pack_end(hint, False, False, 0)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.set_border_width(20)
        root.pack_start(content, True, True, 0)

        header = Gtk.Label(label="实时运行日志", xalign=0)
        header.modify_font(Pango.FontDescription("Sans Bold 16"))
        content.pack_start(header, False, False, 0)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        content.pack_start(scroller, True, True, 0)

        self.log_view = Gtk.TextView()
        self.log_view.set_name("logView")
        self.log_view.set_editable(False)
        self.log_view.set_cursor_visible(False)
        self.log_view.set_monospace(True)
        self.log_buffer = self.log_view.get_buffer()
        scroller.add(self.log_view)

        self.append_log("图形界面已就绪。点击“启动小蝶”。")

    def append_log(self, text: str) -> None:
        end = self.log_buffer.get_end_iter()
        stamp = time.strftime("%H:%M:%S")
        self.log_buffer.insert(end, f"[{stamp}] {text.rstrip()}\n")
        mark = self.log_buffer.create_mark(None, self.log_buffer.get_end_iter(), False)
        self.log_view.scroll_to_mark(mark, 0.0, True, 0.0, 1.0)

    def set_status(self, text: str, kind: str) -> None:
        self.status.set_text(text)
        if kind == "running":
            self.status.set_name("statusRunning")
        elif kind == "starting":
            self.status.set_name("statusStarting")
        else:
            self.status.set_name("statusStopped")

    def start_service(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.append_log("小蝶已经在运行。")
            return
        self.set_status("启动中", "starting")
        self.append_log("正在启动小蝶后台服务...")
        try:
            self.proc = subprocess.Popen(
                APP_CMD,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            self.set_status("启动失败", "stopped")
            self.append_log(f"启动失败：{exc}")
            return

        self.started_at = time.time()
        if self.proc.stdout:
            GLib.io_add_watch(self.proc.stdout, GLib.IO_IN | GLib.IO_HUP, self._read_output)

    def stop_service(self) -> None:
        self.append_log("正在停止小蝶...")
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except Exception:
                self.proc.terminate()
        subprocess.run(STOP_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        self.proc = None
        self.set_status("已停止", "stopped")
        self.append_log("小蝶已停止。")

    def _read_output(self, source: object, condition: int) -> bool:
        if condition & GLib.IO_HUP:
            return False
        line = source.readline()
        if not line:
            return False
        clean = line.rstrip()
        if clean:
            self.append_log(clean)
            if "ready: hold GPIO" in clean or "[asr_daemon] ready" in clean:
                self.set_status("运行中", "running")
            elif "ASR daemon starting" in clean:
                self.set_status("加载 ASR", "starting")
            elif "recording..." in clean:
                self.set_status("录音中", "running")
            elif "DeepSeek+TTS started" in clean:
                self.set_status("讲故事中", "running")
        return True

    def _tick(self) -> bool:
        if self.proc:
            code = self.proc.poll()
            if code is not None:
                self.append_log(f"后台进程已退出，退出码 {code}。")
                self.proc = None
                self.set_status("已停止", "stopped")
        return True

    def _on_destroy(self, _widget: Gtk.Widget) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except Exception:
                pass
        Gtk.main_quit()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print("xiaodie_gui_ok")
        return 0

    win = XiaoDieWindow()
    win.show_all()
    Gtk.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
