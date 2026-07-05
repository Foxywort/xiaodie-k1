#!/usr/bin/env python3
"""Download an asset on the Windows PC, then push it to the K1 board.

The K1 board often downloads large model files slowly. This helper keeps the
network-heavy step on the desktop, optionally through a local proxy, then uses
SFTP/SSH to place and extract the asset on the board.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import shlex
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import paramiko


def run_curl(url: str, output: Path, proxy: str | None, retries: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl.exe",
        "-L",
        "--connect-timeout",
        "20",
        "--retry",
        str(retries),
        "--retry-delay",
        "2",
        "-C",
        "-",
        "-o",
        str(output),
    ]
    if proxy:
        cmd[1:1] = ["--proxy", proxy]
    cmd.append(url)
    print("[download]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


def connect(host: str, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        username=user,
        password=password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def ssh(client: paramiko.SSHClient, command: str, timeout: int = 600) -> None:
    print(f"[ssh] {command}")
    stdin, stdout, stderr = client.exec_command(command, timeout=10)
    stdin.close()
    channel = stdout.channel
    started = time.time()
    while not channel.exit_status_ready():
        if channel.recv_ready():
            sys.stdout.buffer.write(channel.recv(65536))
            sys.stdout.buffer.flush()
        if channel.recv_stderr_ready():
            sys.stderr.buffer.write(channel.recv_stderr(65536))
            sys.stderr.buffer.flush()
        if time.time() - started > timeout:
            channel.close()
            raise TimeoutError(command)
        time.sleep(0.1)
    while channel.recv_ready():
        sys.stdout.buffer.write(channel.recv(65536))
    while channel.recv_stderr_ready():
        sys.stderr.buffer.write(channel.recv_stderr(65536))
    code = channel.recv_exit_status()
    if code:
        raise RuntimeError(f"remote command failed with {code}: {command}")


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = [p for p in remote_dir.split("/") if p]
    cur = "/"
    for part in parts:
        cur = posixpath.join(cur, part)
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def upload_if_needed(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    local_size = local.stat().st_size
    try:
        remote_size = sftp.stat(remote).st_size
    except FileNotFoundError:
        remote_size = -1
    if remote_size == local_size:
        print(f"[sftp] skip existing {remote} ({remote_size} bytes)")
        return

    sent = 0

    def progress(done: int, total: int) -> None:
        nonlocal sent
        if done == total or done - sent >= 16 * 1024 * 1024:
            sent = done
            pct = done * 100.0 / max(total, 1)
            print(f"[sftp] {pct:5.1f}% {done}/{total} bytes")

    print(f"[sftp] upload {local} -> {remote}")
    sftp.put(str(local), remote, callback=progress)


def default_name_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise SystemExit("Cannot infer filename from URL; pass --local-file")
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--local-dir", default="downloads/assets")
    parser.add_argument("--local-file")
    parser.add_argument("--proxy", default=os.environ.get("XIAODIE_PROXY", "http://127.0.0.1:7890"))
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--host", default="192.168.205.221")
    parser.add_argument("--user", default="vicky")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--extract-dir")
    parser.add_argument("--extract-kind", choices=["none", "tar-bz2", "tar-gz", "zip"], default="none")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    local = Path(args.local_file) if args.local_file else Path(args.local_dir) / default_name_from_url(args.url)
    local = local.resolve()
    if args.force_download or not local.exists() or local.stat().st_size == 0:
        run_curl(args.url, local, None if args.no_proxy else args.proxy, args.retries)
    else:
        print(f"[download] skip existing {local} ({local.stat().st_size} bytes)")

    client = connect(args.host, args.user, args.password)
    try:
        sftp = client.open_sftp()
        try:
            ensure_remote_dir(sftp, posixpath.dirname(args.remote_path))
            upload_if_needed(sftp, local, args.remote_path)
        finally:
            sftp.close()

        if args.extract_kind != "none":
            if not args.extract_dir:
                raise SystemExit("--extract-dir is required when --extract-kind is set")
            remote_q = shlex.quote(args.remote_path)
            extract_q = shlex.quote(args.extract_dir)
            if args.extract_kind == "tar-bz2":
                cmd = f"mkdir -p {extract_q} && tar -xjf {remote_q} -C {extract_q}"
            elif args.extract_kind == "tar-gz":
                cmd = f"mkdir -p {extract_q} && tar -xzf {remote_q} -C {extract_q}"
            else:
                cmd = f"mkdir -p {extract_q} && unzip -o {remote_q} -d {extract_q}"
            ssh(client, cmd, timeout=1800)
    finally:
        client.close()

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
