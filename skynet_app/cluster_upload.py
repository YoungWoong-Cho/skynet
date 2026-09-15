"""Stream immutable job files to the cluster with checksum verification."""

from pathlib import Path
import re
import shlex
import subprocess

from .cluster_runtime import ClusterError, WORK_ROOT


def upload_capture(
    cluster,
    path,
    run_id,
    digest,
    gateway,
    *,
    relative_path="original.jsonl",
    timeout=120,
):
    """Bounded-memory upload; immutable destination and verified original bytes."""
    cluster.candidates(gateway)
    run_id = cluster._run_id(run_id)
    if not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError("Invalid capture checksum")
    relative_path = cluster._relative_path(relative_path)
    destination = f"{WORK_ROOT}/jobs/runs/{run_id}/{relative_path}"
    parent = shlex.quote(str(Path(destination).parent))
    dest = shlex.quote(destination)
    command = (
        f"set -eu; umask 077; mkdir -p {parent}; tmp=$(mktemp {parent}/.capture-XXXXXX); "
        'trap \'rm -f "$tmp"\' EXIT; cat > "$tmp"; '
        f'printf "%s  %s\\n" {shlex.quote(digest)} "$tmp" | sha256sum --check --status; '
        f'if test -e {dest}; then cmp -s "$tmp" {dest}; else ln "$tmp" {dest}; fi'
    )
    with path.open("rb") as stream:
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=6",
                    "-o",
                    "ServerAliveInterval=5",
                    "-o",
                    "ServerAliveCountMax=1",
                    gateway,
                    command,
                ],
                stdin=stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise ClusterError(
                "Capture upload timed out; no job was submitted"
            ) from error
    if result.returncode:
        raise ClusterError(
            "Capture upload or checksum verification failed; no job was submitted"
        )
    return destination
