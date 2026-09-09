"""Remove only generated dataset paths; shared by local and cluster cleanup."""

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from uuid import UUID


def cleanup(root, *, jobs=(), prepared=(), cluster=False):
    root = Path(root)
    paths = []
    for identifier in jobs:
        if str(UUID(identifier)) != identifier:
            raise ValueError("Invalid preparation identifier")
        paths.append(root / "jobs/runs" / identifier if cluster else root / identifier)
    for checksum in prepared:
        if not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise ValueError("Invalid dataset checksum")
    paths.extend(root / "datasets/prepared" / sha for sha in prepared)
    # Validate the entire plan before deleting anything. A linked ancestor must
    # never redirect deletion outside these generated directories.
    for path in paths:
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Dataset cleanup cannot follow symbolic links")
        if path.parent.name == "prepared" and path.exists():
            manifest = path / "manifest.json"
            if manifest.is_symlink() or not manifest.is_file():
                raise ValueError("Prepared dataset manifest is missing")
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != path.name:
                raise ValueError("Prepared dataset manifest has changed")
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    return {"removed": True}


if __name__ == "__main__":
    print(json.dumps(cleanup(**json.loads(sys.argv[1]))))
