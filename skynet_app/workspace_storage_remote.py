"""Cluster-side first-use validation, ownership and idempotent initialization."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import sys

MARKER = ".skynet-workspace.json"
DIRECTORIES = (
    "workspace", "repos/shared", "envs", "datasets", "artifacts", "logs", "jobs",
    "eval/catalogs", "eval/datasets", "eval/assets", "eval/runs", "mlflow/db",
    "mlflow/artifacts", ".cache/uv", ".cache/huggingface", ".cache/torch",
)


def initialize(root: Path, owner_id: str) -> dict:
    if not root.is_absolute() or root == Path('/') or '..' in root.parts:
        raise ValueError("Use an absolute base directory.")
    for path in (root, *root.parents):
        if path.is_symlink():
            raise ValueError("Base path must not contain symbolic links.")
        if path != root and (path / MARKER).exists():
            raise ValueError("Choose a directory outside an existing workspace.")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
        raise ValueError("Base path must be a writable directory.")
    # Lock the directory itself so simultaneous first-use requests cannot both claim it.
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        marker = root / MARKER
        identity = {"owner_id": owner_id, "work_root": str(root)}
        if marker.exists() or marker.is_symlink():
            if marker.is_symlink() or json.loads(marker.read_text()) != identity:
                raise ValueError("This directory belongs to another workspace.")
        else:
            if next(root.iterdir(), None) is not None:
                raise ValueError("Base path is not empty. Choose a new or empty directory.")
            with marker.open('x') as stream:
                os.chmod(marker, 0o600)
                json.dump(identity, stream)
                stream.flush()
                os.fsync(stream.fileno())
        for relative in DIRECTORIES:
            target = root / relative
            for path in (target, *target.parents):
                if path == root:
                    break
                if path.is_symlink():
                    raise ValueError("Workspace directories must not be symbolic links.")
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not os.access(target, os.W_OK | os.X_OK):
                raise ValueError("Workspace directories must be writable.")
        return {**identity, "ready": True}
    finally:
        os.close(descriptor)


if __name__ == '__main__':
    try:
        os.umask(0o077)
        print(json.dumps(initialize(Path(sys.argv[1]), sys.argv[2])))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
