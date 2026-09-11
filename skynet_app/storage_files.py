"""Bounded cluster inventory and deletion with a second filesystem check."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import time
from pathlib import Path, PurePosixPath

# This module also runs remotely using its own source; no application imports.
MAX_ENTRIES = 100_000
CANONICAL = frozenset(
    {
        "jobs",
        "eval",
        "datasets",
        "artifacts",
        "logs",
        "workspace",
        "repos",
        "envs",
        "services",
        ".cache",
        ".skynet-workspace.json",
    }
)
PROTECTED = frozenset({"services", "envs", "repos", "workspace", ".cache"})


def overlaps(a, b):
    a, b = PurePosixPath(a), PurePosixPath(b)
    return a == b or a in b.parents or b in a.parents


def checked_path(root, value):
    root, path = Path(root), Path(value)
    if (
        not root.is_absolute()
        or not path.is_absolute()
        or ".." in path.parts
        or root not in path.parents
    ):
        raise ValueError("Cleanup path must be below the workspace base path")
    if path.relative_to(root).parts[0] in PROTECTED:
        raise ValueError("Service, runtime and source directories are protected")
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError("Symbolic links cannot be cleaned")
    return path


def describe(path):
    """Fingerprint names and inode metadata without reading large recordings."""
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "fingerprint": "missing",
            "size_bytes": 0,
            "files": 0,
        }
    digest = hashlib.sha256()
    size = count = newest = 0
    todo = [path]
    while todo:
        entry = todo.pop()
        info = entry.lstat()
        if not (
            stat.S_ISREG(info.st_mode)
            or stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
        ):
            raise ValueError("Special files cannot be cleaned")
        count += 1
        if count > MAX_ENTRIES:
            raise ValueError("Directory is too large for one cleanup operation")
        digest.update(
            json.dumps(
                [
                    str(entry.relative_to(path)),
                    info.st_dev,
                    info.st_ino,
                    info.st_mode,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                ]
            ).encode()
        )
        newest = max(newest, info.st_mtime)
        if stat.S_ISDIR(info.st_mode):
            todo.extend(sorted(entry.iterdir(), reverse=True))
        else:
            # Internal links belong to this directory; never walk their targets.
            # checked_path still rejects a selected link or a linked ancestor.
            if stat.S_ISLNK(info.st_mode):
                digest.update(str(entry.readlink()).encode())
            size += info.st_size
    return {
        "path": str(path),
        "exists": True,
        "fingerprint": digest.hexdigest(),
        "size_bytes": size,
        "files": count,
        "modified_at": newest,
    }


def execute(request):
    root = request["root"]
    operation = request["operation"]
    if operation in {"scan", "scan_objects"}:
        protected = request.get("protected", [])
        candidates = []
        base = Path(root)
        # Enumerate only output boundaries, never recursively crawl live data.
        for relative in (
            ()
            if operation == "scan_objects"
            else ("jobs/runs", "eval/runs", "datasets/prepared", "logs", "artifacts")
        ):
            folder = base / relative
            checked_path(root, str(folder))
            if folder.is_dir():
                candidates.extend(
                    (p, "Unreferenced output") for p in sorted(folder.iterdir())
                )
        if operation == "scan_objects" and base.is_dir():
            for prefix in sorted(base.iterdir()):
                if not re.fullmatch(r"[a-f0-9]{2}", prefix.name) or prefix.is_symlink():
                    continue
                for directory in sorted(prefix.iterdir()):
                    if (
                        not re.fullmatch(r"[a-f0-9]{64}", directory.name)
                        or directory.is_symlink()
                    ):
                        continue
                    candidates.extend(
                        (path, "Unreferenced metadata body")
                        for path in sorted(directory.iterdir())
                    )
        elif base.is_dir():
            candidates.extend(
                (p, "Outside the storage layout")
                for p in sorted(base.iterdir())
                if p.name not in CANONICAL
            )
        result = []
        for path, reason in candidates:
            if any(overlaps(str(path), ref) for ref in protected):
                continue
            try:
                checked_path(root, str(path))
                item = describe(path)
                item.update(
                    reason=reason,
                    selectable=time.time() - item.get("modified_at", time.time())
                    >= 86400,
                )
                if not item["selectable"]:
                    item["reason"] = "Recent file; protected for 24 hours"
            except (ValueError, OSError) as error:
                item = {
                    "path": str(path),
                    "reason": str(error),
                    "selectable": False,
                    "size_bytes": None,
                }
            result.append(item)
            if len(result) >= 2000:
                raise ValueError(
                    "Too many unreferenced paths; narrow the storage tree before scanning"
                )
        return {"items": result, "root": root}
    paths = [checked_path(root, item["path"]) for item in request["items"]]
    if len(paths) > 500 or any(
        a != b and overlaps(str(a), str(b)) for a in paths for b in paths
    ):
        raise ValueError("Use at most 500 separate cleanup paths")
    snapshots = [describe(path) for path in paths]
    if operation == "inspect":
        return {"items": snapshots}
    if operation != "delete":
        raise ValueError("Unknown storage operation")
    # Validate the entire batch before removing anything; missing paths make retries idempotent.
    for item, current in zip(request["items"], snapshots):
        if current["exists"] and item["fingerprint"] != current["fingerprint"]:
            raise ValueError(
                "Files changed since preview. Inspect again before deleting: "
                + current["path"]
            )
    for path in paths:
        checked_path(root, str(path))
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        # Content-addressed object directories have no meaning once empty.
        relative = path.relative_to(root)
        if (
            len(relative.parts) == 3
            and re.fullmatch(r"[a-f0-9]{2}", relative.parts[0])
            and re.fullmatch(r"[a-f0-9]{64}", relative.parts[1])
        ):
            for parent in (path.parent, path.parent.parent):
                try:
                    parent.rmdir()
                except OSError:
                    break
    return {"deleted": [str(p) for p in paths]}


if __name__ == "__main__":
    import sys

    try:
        print(json.dumps(execute(json.load(sys.stdin))))
    except Exception as error:
        print(json.dumps({"error": str(error)}))
        sys.exit(1)
