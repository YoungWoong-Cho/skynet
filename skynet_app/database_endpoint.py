"""Connect to the central DB through a private, process-owned SSH socket."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from psycopg.conninfo import make_conninfo


class SSHEndpoint:
    def __init__(self, config):
        self.config = dict(config)
        host = str(config["ssh_host"])
        self.transport = config.get("transport", "ssh-unix")
        remote = str(config.get("remote_socket", ""))
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", host):
            raise ValueError("Invalid database SSH host")
        if self.transport == "ssh-unix":
            if not remote.startswith("/") or any(
                c in remote for c in ("\n", "\r", ":", "\x00")
            ):
                raise ValueError("Invalid database socket path")
            self.forward_target = remote
        elif self.transport == "ssh-tcp":
            if (
                config.get("remote_address") != "127.0.0.1"
                or type(config.get("remote_port")) is not int
                or not 1 <= config["remote_port"] <= 65535
            ):
                raise ValueError(
                    "Central DB TCP forwarding must target a loopback port"
                )
            self.forward_target = "127.0.0.1:" + str(config["remote_port"])
            if not config.get("password_file"):
                raise ValueError("Central DB password file is not configured")
        else:
            raise ValueError("Unknown database SSH transport")
        self._lock = threading.Lock()
        self._process = None
        self._directory = None
        self._pid = os.getpid()
        atexit.register(self.close)

    def connection_string(self):
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._close()
                # Unix socket paths have a short OS limit. Do not place these in
                # the potentially long repository path or expose a TCP listener.
                self._directory = Path(
                    tempfile.mkdtemp(prefix="skynet-db-", dir="/tmp")
                )
                self._directory.chmod(0o700)
                local = self._directory / ".s.PGSQL.55432"
                args = [
                    "ssh",
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=8",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-o",
                    "StreamLocalBindUnlink=no",
                    "-o",
                    "ServerAliveInterval=10",
                    "-o",
                    "ServerAliveCountMax=2",
                    "-L",
                    str(local) + ":" + self.forward_target,
                    self.config["ssh_host"],
                    "cat >/dev/null",
                ]
                # The pipe's EOF ends the remote command if this app exits,
                # including abnormal termination. No background tunnel is left.
                self._process = subprocess.Popen(
                    args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                deadline = time.monotonic() + 10
                while not local.exists():
                    if self._process.poll() is not None or time.monotonic() >= deadline:
                        self._close()
                        raise ConnectionError(
                            "Central database SSH connection failed; check access to "
                            + self.config["ssh_host"]
                        )
                    time.sleep(0.05)
            credentials = {}
            if self.transport == "ssh-tcp":
                password_file = Path(self.config["password_file"])
                if password_file.stat().st_mode & 0o077:
                    raise ValueError("Central DB password file must be private (0600)")
                credentials["password"] = password_file.read_text().strip()
            return make_conninfo(
                **credentials,
                host=str(self._directory),
                port=55432,
                dbname=self.config.get("database", "skynet"),
                user=self.config["user"],
            )

    def _close(self):
        if self._process:
            if self._process.stdin:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2)
            self._process = None
        if self._directory:
            shutil.rmtree(self._directory)
            self._directory = None

    def close(self):
        if os.getpid() != self._pid:
            return
        with self._lock:
            self._close()


_ENDPOINTS = {}
_ENDPOINT_LOCK = threading.Lock()


def load_endpoint(path: Path):
    config = json.loads(path.read_text())
    if config.get("backend") != "postgresql" or config.get("transport") not in {
        "ssh-unix",
        "ssh-tcp",
    }:
        raise ValueError("Unknown central database configuration")
    if config.get("password_file"):
        password_file = Path(config["password_file"])
        if not password_file.is_absolute():
            config["password_file"] = str(path.parent / password_file)
    key = (
        os.getpid(),
        hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
    )
    with _ENDPOINT_LOCK:
        if key not in _ENDPOINTS:
            _ENDPOINTS[key] = SSHEndpoint(config)
        return _ENDPOINTS[key]
