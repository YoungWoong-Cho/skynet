"""Portable dataset integrity checks, shared by conversion, transfer and training."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


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


def pack(root):
    root = Path(root)
    archive = root.parent / "dataset.zip.part"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(
        archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as zipped:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(root))
    archive.replace(root.parent / "dataset.zip")


def materialize(archive, destination, expected, archive_sha):
    """Publish a verified immutable directory; incomplete copies stay private."""
    import fcntl

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / ("." + expected + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            return verify(destination, expected)
        if digest(archive) != archive_sha:
            raise ValueError("Transferred archive failed verification")
        staging = Path(tempfile.mkdtemp(prefix=".preparing-", dir=destination.parent))
        try:
            with zipfile.ZipFile(archive) as zipped:
                manifest_bytes = zipped.read("manifest.json")
                if hashlib.sha256(manifest_bytes).hexdigest() != expected:
                    raise ValueError("Archive contains a different manifest")
                manifest = json.loads(manifest_bytes)
                allowed = {
                    **manifest["files"],
                    "manifest.json": {"size_bytes": len(manifest_bytes)},
                }
                names = zipped.namelist()
                if len(names) != len(set(names)) or set(names) != set(allowed):
                    raise ValueError("Archive entries differ from the manifest")
                for entry in zipped.infolist():
                    relative_file(entry.filename)
                    if entry.file_size != allowed[entry.filename]["size_bytes"]:
                        raise ValueError("Archive size differs from its manifest")
                    target = staging / entry.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(entry) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
            verify(staging, expected)
            os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("manifest_sha")
    parser.add_argument("--archive")
    parser.add_argument("--archive-sha")
    args = parser.parse_args()
    result = (
        materialize(args.archive, args.root, args.manifest_sha, args.archive_sha)
        if args.archive
        else verify(args.root, args.manifest_sha)
    )
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
