"""Integrity checks for externally prepared native dataset inventories."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def relative_file(value):
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in (".", "..") for part in path.parts)
        or "\\" in value
    ):
        raise ValueError("Unsafe dataset file path")
    return path


def verify(root, expected):
    root = Path(root)
    if digest(root / "manifest.json") != expected:
        raise ValueError("Dataset manifest changed; restore the pinned version")
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("files"):
        raise ValueError("Dataset manifest has no files")
    for name, receipt in manifest["files"].items():
        path = root / relative_file(name)
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or not path.is_file()
        ):
            raise ValueError("Dataset file is missing or unsafe: " + name)
        if (
            path.stat().st_size != receipt["size_bytes"]
            or digest(path) != receipt["sha256"]
        ):
            raise ValueError("Dataset file failed verification: " + name)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("manifest_sha")
    args = parser.parse_args()
    result = verify(args.root, args.manifest_sha)
    print(
        json.dumps(
            dict(
                verified=True,
                manifest_sha256=args.manifest_sha,
                episodes=len(result["episodes"]),
                steps=result["steps"],
            )
        )
    )
