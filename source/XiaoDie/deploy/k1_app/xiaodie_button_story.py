#!/usr/bin/env python3
"""Button controlled XiaoDie story assistant for the K1 board.

Hold the GPIO button to record from the USB microphone. Release it to run local
ASR, then stream the recognized request to DeepSeek and Chaowen Full TTS.
Pressing the button at any point interrupts the current task and starts a new
recording.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


def log(message: str) -> None:
    print(f"[xiaodie] {message}", flush=True)


def run_quiet(cmd: list[str], timeout: float = 10.0) -> None:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        log(f"command timed out and was ignored: {' '.join(cmd)}")


def repair_pcm_wav_header(path: Path) -> None:
    """Fix arecord WAV headers left with placeholder sizes after interruption."""
    size = path.stat().st_size
    if size < 44:
        return
    with path.open("r+b") as f:
        head = f.read(44)
        if len(head) < 44 or head[0:4] != b"RIFF" or head[8:12] != b"WAVE":
            return
        data_offset = head.find(b"data")
        if data_offset < 0 or data_offset + 8 > len(head):
            return
        riff_size = size - 8
        data_size = size - (data_offset + 8)
        old_riff = struct.unpack_from("<I", head, 4)[0]
        old_data = struct.unpack_from("<I", head, data_offset + 4)[0]
        if old_riff == riff_size and old_data == data_size:
            return
        f.seek(4)
        f.write(struct.pack("<I", riff_size))
        f.seek(data_offset + 4)
        f.write(struct.pack("<I", data_size))
    log(f"repaired WAV header: {path} data_bytes={data_size}")


class SysfsButton:
    def __init__(self, gpio: int, active_low: bool, debounce_ms: int) -> None:
        self.gpio = gpio
        self.active_low = active_low
        self.debounce_s = debounce_ms / 1000.0
        self.path = Path(f"/sys/class/gpio/gpio{gpio}")
        self.value_path = self.path / "value"
        self._export()

    def _export(self) -> None:
        if not self.path.exists():
            try:
                Path("/sys/class/gpio/export").write_text(str(self.gpio), encoding="ascii")
            except OSError as exc:
                raise SystemExit(
                    f"Cannot export GPIO {self.gpio}. Please run this program with sudo."
                ) from exc
            time.sleep(0.2)
        direction = self.path / "direction"
        if direction.exists():
            try:
                direction.write_text("in", encoding="ascii")
            except OSError:
                pass

    def raw_pressed(self) -> bool:
        value = self.value_path.read_text(encoding="ascii").strip()
        pressed = value == "1"
        return not pressed if self.active_low else pressed

    def stable_state(self, previous: bool | None) -> bool:
        first = self.raw_pressed()
        if previous is not None and first == previous:
            return previous
        time.sleep(self.debounce_s)
        second = self.raw_pressed()
        return second if second == first else (previous if previous is not None else second)


class AsrDaemon:
    def __init__(self, binary: str, model_dir: str, threads: int, tail_pad_ms: int, nice: int) -> None:
        self.binary = binary
        self.model_dir = model_dir
        self.threads = threads
        self.tail_pad_ms = tail_pad_ms
        self.nice = nice
        self.proc: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()

    def start(self) -> None:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return
            cmd = [
                self.binary,
                "--model-dir",
                self.model_dir,
                "--threads",
                str(self.threads),
                "--tail-pad-ms",
                str(self.tail_pad_ms),
            ]
            if self.nice != 0:
                cmd = ["nice", "-n", str(self.nice)] + cmd
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            log("ASR daemon starting; first load can take a little while.")

    def stop(self) -> None:
        with self.lock:
            proc = self.proc
            self.proc = None
        if not proc:
            return
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write("::quit\n")
                    proc.stdin.flush()
            except OSError:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)

    def restart(self) -> None:
        self.stop()
        self.start()

    def recognize(self, wav_path: str) -> dict[str, Any]:
        self.start()
        proc = self.proc
        if not proc or not proc.stdin or not proc.stdout or proc.poll() is not None:
            raise RuntimeError("ASR daemon is not running")
        proc.stdin.write(wav_path + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("ASR daemon returned no result")
        return json.loads(line)


class XiaoDieButtonStory:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.button = SysfsButton(args.gpio, args.active_low, args.debounce_ms)
        self.asr = AsrDaemon(
            args.asr_bin,
            args.asr_model_dir,
            args.asr_threads,
            args.asr_tail_pad_ms,
            args.asr_nice,
        )
        self.lock = threading.Lock()
        self.generation = 0
        self.record_proc: subprocess.Popen[Any] | None = None
        self.story_proc: subprocess.Popen[str] | None = None
        self.worker: threading.Thread | None = None
        self.record_path: Path | None = None
        self.running = True
        self.last_state: bool | None = None
        self.asr_busy = False
        self.tts_may_be_playing = False

    def start_services(self) -> None:
        Path(self.args.record_dir).mkdir(parents=True, exist_ok=True)
        self.restart_tts()
        self.asr.start()

    def stop_services(self) -> None:
        self.interrupt_current("shutdown", restart_tts=False)
        self.asr.stop()
        if self.args.stop_tts_on_exit:
            run_quiet([self.args.tts_stop], timeout=5)

    def restart_tts(self) -> None:
        run_quiet([self.args.tts_stop], timeout=5)
        run_quiet(["pkill", "-TERM", "-f", "chaowen_tts_daemon|ffmpeg.*s16le|aplay.*sndes8326"], timeout=3)
        run_quiet([self.args.tts_start], timeout=10)

    def ensure_tts_running(self) -> None:
        run_quiet([self.args.tts_start], timeout=5)

    def interrupt_current(self, reason: str, restart_tts: bool = True) -> int:
        with self.lock:
            self.generation += 1
            token = self.generation
            record_proc = self.record_proc
            story_proc = self.story_proc
            self.record_proc = None
            self.story_proc = None
        log(f"interrupt: {reason}")
        self._terminate_proc(record_proc, sigint=True)
        self._terminate_proc(story_proc, sigint=False)
        if self.asr_busy:
            self.asr.restart()
        hard_tts_reset = restart_tts and self.tts_may_be_playing
        if hard_tts_reset:
            self.restart_tts()
            self.tts_may_be_playing = False
        elif restart_tts:
            self.ensure_tts_running()
        return token

    @staticmethod
    def _terminate_proc(proc: subprocess.Popen[Any] | None, sigint: bool) -> None:
        if not proc or proc.poll() is not None:
            return
        try:
            if sigint:
                proc.send_signal(signal.SIGINT)
            else:
                proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
        except OSError:
            pass

    def start_recording(self) -> None:
        token = self.interrupt_current("button pressed")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        wav = Path(self.args.record_dir) / f"record_{stamp}_{token}.wav"
        cmd = [
            "arecord",
            "-D",
            self.args.mic_device,
            "-f",
            "S16_LE",
            "-c",
            "1",
            "-r",
            "16000",
            "-t",
            "wav",
            str(wav),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        with self.lock:
            if token == self.generation:
                self.record_proc = proc
                self.record_path = wav
        log(f"recording... release button to stop. wav={wav}")

    def stop_recording_and_run(self) -> None:
        with self.lock:
            token = self.generation
            proc = self.record_proc
            wav = self.record_path
            self.record_proc = None
        self._terminate_proc(proc, sigint=True)
        if not wav or not wav.exists() or wav.stat().st_size < 4096:
            log("recording too short; ignored.")
            return
        repair_pcm_wav_header(wav)
        log(f"recording stopped: {wav}")
        worker = threading.Thread(target=self._asr_then_story, args=(token, wav), daemon=True)
        with self.lock:
            self.worker = worker
        worker.start()

    def _asr_then_story(self, token: int, wav: Path) -> None:
        try:
            log("recognizing speech")
            self.asr_busy = True
            result = self.asr.recognize(str(wav))
        except Exception as exc:
            log(f"ASR failed: {exc}")
            self.asr.restart()
            return
        finally:
            self.asr_busy = False
        if not self._is_current(token):
            return
        text = str(result.get("text") or "").strip()
        log(f"ASR: {text or '<empty>'} elapsed={result.get('elapsed_s')}s rtf={result.get('rtf')}")
        if len(text) < self.args.min_asr_chars:
            self._say(token, "小蝶没有听清楚，可以再说一遍吗？")
            return
        log("thinking story")
        self._run_story(token, text)

    def _say(self, token: int, text: str) -> None:
        if not self._is_current(token):
            return
        fifo = Path(self.args.tts_fifo)
        if fifo.exists():
            try:
                self.tts_may_be_playing = True
                fifo.write_text(text + "\n::flush\n", encoding="utf-8")
            except OSError as exc:
                log(f"TTS fallback failed: {exc}")

    def _run_story(self, token: int, query: str) -> None:
        franchise = self._infer_franchise(query)
        output = Path(self.args.report_dir) / f"button_story_{time.strftime('%Y%m%d_%H%M%S')}.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.args.deepseek_runner,
            "--query",
            query,
            "--age",
            self.args.age,
            "--style",
            self.args.style,
            "--target-chars",
            str(self.args.target_chars),
            "--max-tokens",
            str(self.args.max_tokens),
            "--top-k",
            str(self.args.top_k),
            "--card-chars",
            str(self.args.card_chars),
            "--tts-min-chars",
            str(self.args.tts_min_chars),
            "--tts-max-chars",
            str(self.args.tts_max_chars),
            "--tts-min-sentences",
            str(self.args.tts_min_sentences),
            "--tts-max-sentences",
            str(self.args.tts_max_sentences),
            "--tts-log",
            self.args.tts_log,
            "--wait-tts-playback",
            "--output",
            str(output),
        ]
        if franchise:
            cmd.extend(["--franchise", franchise])
        log("DeepSeek+TTS started. Press button again to interrupt.")
        self.tts_may_be_playing = True
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        with self.lock:
            if token == self.generation:
                self.story_proc = proc
        if not proc.stdout:
            return
        for line in proc.stdout:
            if not self._is_current(token):
                self._terminate_proc(proc, sigint=False)
                return
            print(line, end="", flush=True)
        proc.wait()
        if self._is_current(token):
            self.tts_may_be_playing = False
            log(f"story done. report={output}")

    def _is_current(self, token: int) -> bool:
        with self.lock:
            return token == self.generation

    @staticmethod
    def _infer_franchise(query: str) -> str | None:
        rules = [
            ("peppa_pig", ("佩奇", "乔治", "小猪佩奇", "猪爸爸", "猪妈妈")),
            ("barbapapa", ("巴巴爸爸", "巴巴妈妈", "巴巴祖", "巴巴拉拉")),
            ("paw_patrol", ("汪汪队", "莱德", "阿奇", "毛毛", "天天", "灰灰")),
            ("octonauts", ("海底小纵队", "巴克队长", "呱唧", "皮医生", "突突兔")),
            ("my_little_pony", ("小马宝莉", "紫悦", "云宝", "柔柔", "苹果嘉儿", "珍奇", "碧琪")),
        ]
        for franchise, keys in rules:
            if any(k in query for k in keys):
                return franchise
        return None

    def loop(self) -> None:
        log(
            f"ready: hold GPIO {self.args.gpio} to record, release to ask XiaoDie. "
            "Press again any time to interrupt."
        )
        while self.running:
            state = self.button.stable_state(self.last_state)
            if self.last_state is None:
                self.last_state = state
            elif state != self.last_state:
                self.last_state = state
                if state:
                    self.start_recording()
                else:
                    self.stop_recording_and_run()
            time.sleep(self.args.poll_ms / 1000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpio", type=int, default=35)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--active-high", dest="active_low", action="store_false")
    group.add_argument("--active-low", dest="active_low", action="store_true")
    parser.set_defaults(active_low=False)
    parser.add_argument("--poll-ms", type=int, default=25)
    parser.add_argument("--debounce-ms", type=int, default=60)
    parser.add_argument("--base", default="/home/vicky/xiaodie")
    parser.add_argument("--mic-device", default="plughw:CARD=Device,DEV=0")
    parser.add_argument("--record-dir", default="/home/vicky/xiaodie/records")
    parser.add_argument("--report-dir", default="/home/vicky/xiaodie/reports/button")
    parser.add_argument("--asr-bin", default="/home/vicky/xiaodie/asr/bin/xiaodie_asr_daemon")
    parser.add_argument(
        "--asr-model-dir",
        default="/home/vicky/xiaodie/asr/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05",
    )
    parser.add_argument("--asr-threads", type=int, default=4)
    parser.add_argument("--asr-tail-pad-ms", type=int, default=500)
    parser.add_argument("--asr-nice", type=int, default=-5)
    parser.add_argument("--deepseek-runner", default="/home/vicky/xiaodie/llm/run_deepseek_story_tts.sh")
    parser.add_argument("--tts-start", default="/home/vicky/xiaodie/tts/start_chaowen_tts_service.sh")
    parser.add_argument("--tts-stop", default="/home/vicky/xiaodie/tts/stop_chaowen_tts_service.sh")
    parser.add_argument("--tts-fifo", default="/home/vicky/xiaodie/tts/tts_input.fifo")
    parser.add_argument("--tts-log", default="/home/vicky/xiaodie/tts/tts_service.log")
    parser.add_argument("--stop-tts-on-exit", action="store_true")
    parser.add_argument("--age", default="4-6岁")
    parser.add_argument("--style", default="睡前安抚")
    parser.add_argument("--target-chars", type=int, default=650)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--card-chars", type=int, default=160)
    parser.add_argument("--tts-min-chars", type=int, default=40)
    parser.add_argument("--tts-max-chars", type=int, default=140)
    parser.add_argument("--tts-min-sentences", type=int, default=2)
    parser.add_argument("--tts-max-sentences", type=int, default=4)
    parser.add_argument("--min-asr-chars", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = XiaoDieButtonStory(args)

    def shutdown(_signum: int, _frame: object) -> None:
        app.running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    app.start_services()
    try:
        app.loop()
    finally:
        app.stop_services()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
