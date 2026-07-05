#!/usr/bin/env python3
"""Run one command on the K1 board through Paramiko."""

from __future__ import annotations

import argparse
import sys
import time

import paramiko


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.205.221")
    parser.add_argument("--user", default="vicky")
    parser.add_argument("--password", default="123456")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = " ".join(args.command).strip()
    if not command:
        parser.error("command is required")

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=args.password, timeout=20, banner_timeout=20, auth_timeout=20)
    try:
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
            if time.time() - started > args.timeout:
                channel.close()
                raise SystemExit(124)
            time.sleep(0.1)
        while channel.recv_ready():
            sys.stdout.buffer.write(channel.recv(65536))
            sys.stdout.buffer.flush()
        while channel.recv_stderr_ready():
            sys.stderr.buffer.write(channel.recv_stderr(65536))
            sys.stderr.buffer.flush()
        raise SystemExit(channel.recv_exit_status())
    finally:
        client.close()


if __name__ == "__main__":
    main()
