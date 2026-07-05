#!/usr/bin/env python3
"""Deploy XiaoDie GGUF model and local RAG assets to the K1 board."""

from __future__ import annotations

import argparse
import os
import stat
import time
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[2]
LLM_DIR = ROOT / "deploy" / "k1_llm"
TTS_DIR = ROOT / "deploy" / "k1_tts"
RAG_DIR = Path("E:/Duanwu/data/ip_rag")
DEFAULT_MODEL = ROOT / "deploy_artifacts/gguf/qwen3-4b-xiaodie-story-Q4_K_M.gguf"


def remote_exec(client: paramiko.SSHClient, command: str, timeout: int = 60) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    stdin.close()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def ensure_dir(client: paramiko.SSHClient, path: str) -> None:
    code, out, err = remote_exec(client, f"mkdir -p '{path}'")
    if code != 0:
        raise RuntimeError(f"mkdir failed for {path}: {out}{err}")


def remote_size(sftp: paramiko.SFTPClient, remote: str) -> int | None:
    try:
        return sftp.stat(remote).st_size
    except FileNotFoundError:
        return None
    except OSError:
        return None


def upload_file(sftp: paramiko.SFTPClient, local: Path, remote: str, mode: int | None = None) -> None:
    local_size = local.stat().st_size
    existing = remote_size(sftp, remote)
    if existing == local_size:
        print(f"[skip] {remote} already has {local_size} bytes")
        if mode is not None:
            sftp.chmod(remote, mode)
        return

    print(f"[upload] {local} -> {remote} ({local_size / 1024 / 1024:.1f} MiB)")
    started = time.time()
    last_report = 0.0

    def progress(done: int, total: int) -> None:
        nonlocal last_report
        now = time.time()
        if now - last_report < 5 and done < total:
            return
        last_report = now
        elapsed = max(now - started, 0.1)
        speed = done / elapsed / 1024 / 1024
        pct = 100.0 * done / max(total, 1)
        print(f"  {pct:5.1f}% {done / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MiB, {speed:.2f} MiB/s")

    tmp = remote + ".part"
    sftp.put(str(local), tmp, callback=progress)
    try:
        sftp.posix_rename(tmp, remote)
    except (AttributeError, OSError):
        try:
            sftp.remove(remote)
        except OSError:
            pass
        sftp.rename(tmp, remote)
    if mode is not None:
        sftp.chmod(remote, mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.205.221")
    parser.add_argument("--user", default="vicky")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--remote-root", default="/home/vicky/xiaodie")
    args = parser.parse_args()

    model = Path(args.model)
    if not model.exists():
        raise SystemExit(f"Model not found: {model}")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=args.password, timeout=20, banner_timeout=20, auth_timeout=20)
    sftp = client.open_sftp()
    try:
        for subdir in ("llm", "rag", "models", "reports"):
            ensure_dir(client, f"{args.remote_root}/{subdir}")

        upload_file(sftp, LLM_DIR / "xiaodie_rag_llm.py", f"{args.remote_root}/llm/xiaodie_rag_llm.py", 0o755)
        upload_file(sftp, LLM_DIR / "run_xiaodie_story.sh", f"{args.remote_root}/llm/run_xiaodie_story.sh", 0o755)
        upload_file(sftp, LLM_DIR / "Modelfile.xiaodie-story", f"{args.remote_root}/llm/Modelfile.xiaodie-story")
        stream_runner = LLM_DIR / "xiaodie_rag_ollama_stream.py"
        if stream_runner.exists():
            upload_file(sftp, stream_runner, f"{args.remote_root}/llm/xiaodie_rag_ollama_stream.py", 0o755)
        stream_shell = LLM_DIR / "run_xiaodie_story_1_5b_tts.sh"
        if stream_shell.exists():
            upload_file(sftp, stream_shell, f"{args.remote_root}/llm/run_xiaodie_story_1_5b_tts.sh", 0o755)
        deepseek_runner = LLM_DIR / "xiaodie_deepseek_tts_stream.py"
        if deepseek_runner.exists():
            upload_file(sftp, deepseek_runner, f"{args.remote_root}/llm/xiaodie_deepseek_tts_stream.py", 0o755)
        deepseek_shell = LLM_DIR / "run_deepseek_story_tts.sh"
        if deepseek_shell.exists():
            upload_file(sftp, deepseek_shell, f"{args.remote_root}/llm/run_deepseek_story_tts.sh", 0o755)
        perf_bench = LLM_DIR / "xiaodie_perf_bench.py"
        if perf_bench.exists():
            upload_file(sftp, perf_bench, f"{args.remote_root}/llm/xiaodie_perf_bench.py", 0o755)
        runtime_script = LLM_DIR / "start_xiaodie_runtime.sh"
        if runtime_script.exists():
            upload_file(sftp, runtime_script, f"{args.remote_root}/llm/start_xiaodie_runtime.sh", 0o755)
        modelfile_15b = LLM_DIR / "Modelfile.xiaodie-story-1.5b"
        if modelfile_15b.exists():
            upload_file(sftp, modelfile_15b, f"{args.remote_root}/llm/Modelfile.xiaodie-story-1.5b")
        start_tts = TTS_DIR / "start_chaowen_tts_service.sh"
        if start_tts.exists():
            upload_file(sftp, start_tts, f"{args.remote_root}/tts/start_chaowen_tts_service.sh", 0o755)
        tts_daemon_c = TTS_DIR / "chaowen_tts_daemon.c"
        if tts_daemon_c.exists():
            upload_file(sftp, tts_daemon_c, f"{args.remote_root}/tts/fast/chaowen_tts_daemon.c")
        perf_report = ROOT / "reports" / "k1_edge_ai_performance_report.md"
        if perf_report.exists():
            upload_file(sftp, perf_report, f"{args.remote_root}/reports/k1_edge_ai_performance_report.md")
        upload_file(sftp, RAG_DIR / "ip_knowledge_cards.jsonl", f"{args.remote_root}/rag/ip_knowledge_cards.jsonl")
        upload_file(sftp, RAG_DIR / "ip_rag_index.json", f"{args.remote_root}/rag/ip_rag_index.json")
        remote_model = f"{args.remote_root}/models/{model.name}"
        upload_file(sftp, model, remote_model)

        code, out, err = remote_exec(
            client,
            "uname -a && df -h /home/vicky && ls -lh /home/vicky/xiaodie/models /home/vicky/xiaodie/rag /home/vicky/xiaodie/llm",
            timeout=60,
        )
        print(out)
        if err:
            print(err)
        if code != 0:
            raise SystemExit(code)
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    main()
