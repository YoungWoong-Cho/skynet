"""Bounded, execution-scoped reads of a shared JSONL training log.

This module is also sent as a standalone stdlib-only script to the gateway.
The launch wrapper records the boundary before starting the training process.
"""

import hashlib
import json
import sys
from pathlib import Path

BOUNDARY_NOT_READY = "Training progress launch boundary is not ready"


def read_execution_log(path, boundary_path, *, job_id, restart_count,
                       required=True, lines=500, max_bytes=1_000_000, contains=None):
    path = Path(path)
    boundary_path = Path(boundary_path)
    boundary = None
    if boundary_path.exists():
        boundary = json.loads(boundary_path.read_text())
        if (boundary.get("schema_version") != "skynet.training-progress-start/v1"
                or str(boundary.get("job_id")) != str(job_id)
                or boundary.get("restart_count") != restart_count
                or boundary.get("path") != str(path)):
            raise ValueError("Training progress boundary does not match this execution")
    elif required:
        # A queued/preparing/resumed execution has not yet published its boundary.
        # Never claim another execution's shared log as its progress.
        raise FileNotFoundError(BOUNDARY_NOT_READY)
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stat = __import__("os").fstat(stream.fileno())
        start = 0
        partial = False
        if boundary is not None:
            start = boundary["start_byte_offset"]
            if not isinstance(start, int) or isinstance(start, bool) or start < 0:
                raise ValueError("Invalid training progress boundary offset")
            # The producer runs on a compute node and this reader may run on a
            # gateway. Shared filesystem device/inode values are not portable
            # across those clients; only the saved content boundary identifies
            # the prefix that must not be attributed to the new execution.
            same_prefix = stat.st_size >= start
            if start and same_prefix:
                prefix_bytes = boundary["prefix_bytes"]
                if prefix_bytes != min(64, start):
                    raise ValueError("Invalid training progress boundary prefix length")
                stream.seek(start - prefix_bytes)
                same_prefix = hashlib.sha256(stream.read(prefix_bytes)).hexdigest() == boundary["prefix_sha256"]
            if not same_prefix:
                # Writers may replace/truncate their own file at startup. Its
                # old prefix no longer exists; the new file belongs to this launch.
                start = 0
            else:
                partial = not boundary.get("at_line_boundary", True)
        offset = max(start, stat.st_size - max_bytes)
        if offset > start:
            stream.seek(offset - 1)
            partial = stream.read(1) != b"\n"
        stream.seek(offset)
        content = stream.read(max_bytes)
    if partial:
        newline = content.find(b"\n")
        content = content[newline + 1:] if newline >= 0 else b""
    # A producer can still be writing the final line. Only complete rows count.
    content = content[:content.rfind(b"\n") + 1]
    rows = content.decode("utf-8", errors="replace").splitlines(keepends=True)
    if contains is not None:
        rows = [row for row in rows if contains in row]
    return "".join(rows[-lines:])


if __name__ == "__main__":
    sys.stdout.write(read_execution_log(**json.loads(sys.argv[1])))
