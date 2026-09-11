"""Content-addressed cluster storage for imported headset recordings."""
from __future__ import annotations

import hashlib
import io
from pathlib import PurePosixPath
import re
import shlex
import subprocess

from .cluster_config import CLUSTER
from .cluster_runtime import ClusterClient, ClusterError, WORK_ROOT
from .remote_artifacts import RemoteArtifact

MAX_CAPTURE_BYTES = 512 * 1024 * 1024


class MemoryCapture:
    """The existing validators/converter can reopen bytes without a disk cache."""
    def __init__(self, data: bytes):
        self.data = data

    def open(self, mode="r"):
        stream = io.BytesIO(self.data)
        return stream if "b" in mode else io.TextIOWrapper(stream, encoding="utf-8")


class CaptureStorage:
    gateway = "sky2"

    def __init__(self, cluster=None):
        self.cluster = cluster or ClusterClient()
        self.root = f"{CLUSTER.paths.datasets}/raw/visionpro-local"

    def path(self, digest):
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid recording checksum")
        return f"{self.root}/{digest}/capture.jsonl"

    def publish(self, data: bytes, digest: str) -> str:
        if not 0 < len(data) <= MAX_CAPTURE_BYTES or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Recording size or checksum changed before upload")
        self.cluster.candidates(self.gateway)
        path = self.path(digest)
        parent, target = shlex.quote(str(PurePosixPath(path).parent)), shlex.quote(path)
        command = (
            f"set -eu; umask 077; mkdir -p {parent}; tmp=$(mktemp {parent}/.upload-XXXXXX); "
            'trap \'rm -f "$tmp"\' EXIT; cat > "$tmp"; '
            f'printf "%s  %s\\n" {digest} "$tmp" | sha256sum --check --status; '
            f'if test -e {target}; then test ! -L {target}; cmp -s "$tmp" {target}; '
            f'else ln "$tmp" {target} || cmp -s "$tmp" {target}; fi'
        )
        try:
            result = subprocess.run(
                ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                 "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=1", self.gateway, command],
                input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ClusterError("Recording transfer to sky2 failed. Retry the import; no local copy was saved.") from error
        if result.returncode:
            raise ClusterError("Recording transfer or checksum verification failed on sky2. Retry the import.")
        self.verify(digest, len(data))
        return path

    def verify(self, digest, size):
        target = shlex.quote(self.path(digest))
        command = (
            f"set -eu; test -f {target}; test ! -L {target}; "
            f'test "$(wc -c < {target})" -eq {int(size)}; '
            f'printf "%s  %s\\n" {digest} {target} | sha256sum --check --status'
        )
        self.cluster.ssh(self.gateway, command, timeout=60)

    def artifact(self, digest, size):
        return RemoteArtifact(self.cluster, self.gateway, self.path(digest), max_bytes=min(size, MAX_CAPTURE_BYTES))

    def stage(self, digest, size, run_id, gateway):
        if gateway != self.gateway:
            raise ValueError("Imported recording processing must run on sky2; raw data stays on cluster storage")
        run_id = self.cluster._run_id(run_id)
        self.verify(digest, size)
        source = shlex.quote(self.path(digest))
        destination = f"{WORK_ROOT}/jobs/runs/{run_id}/original.jsonl"
        parent = shlex.quote(str(PurePosixPath(destination).parent))
        target = shlex.quote(destination)
        command = (
            f"set -eu; umask 077; mkdir -p {parent}; tmp=$(mktemp {parent}/.capture-XXXXXX); "
            'trap \'rm -f "$tmp"\' EXIT; '
            f'cp -- {source} "$tmp"; printf "%s  %s\\n" {digest} "$tmp" | sha256sum --check --status; '
            f'if test -e {target}; then test ! -L {target}; cmp -s "$tmp" {target}; '
            f'else ln "$tmp" {target} || cmp -s "$tmp" {target}; fi'
        )
        self.cluster.ssh(gateway, command, timeout=60)
        return destination
