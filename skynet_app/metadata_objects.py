"""Immutable, checksum-addressed bodies stored beside the central database."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import PurePosixPath

MAX_BYTES = 16 * 1024 * 1024
_REMOTE = r"""
import base64, hashlib, json, os, pathlib, sys, tempfile

def handle(request):
    root = pathlib.Path(request["root"])
    digest = request["sha256"]
    path = root / digest[:2] / digest / request["name"]
    if not root.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Invalid metadata storage path")
    if request["operation"] == "put":
        content = base64.b64decode(request["content"], validate=True)
        if len(content) > 16 * 1024 * 1024 or hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Invalid metadata object")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".upload-")
        try:
            with os.fdopen(descriptor, "wb") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            os.unlink(temporary)
    content = path.read_bytes()
    if len(content) > 16 * 1024 * 1024 or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Metadata object checksum mismatch")
    return {"sha256": digest, "size": len(content), "content":
        base64.b64encode(content).decode() if request["operation"] == "get" else None}
request = json.load(sys.stdin)
print(json.dumps([handle(item) for item in request] if isinstance(request, list) else handle(request)))
"""


class MetadataObjects:
    def __init__(self, database):
        config = getattr(database.url, "config", None) or database.endpoint_config
        self.host = config.get("ssh_host", "")
        root = config.get("object_store_root", "")
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", self.host):
            raise ValueError("Configure the central metadata SSH host")
        if not root.startswith("/") or ".." in PurePosixPath(root).parts or root == "/":
            raise ValueError("Configure an absolute central metadata directory")
        self.root = PurePosixPath(root)

    def path(self, digest, name="content"):
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid metadata checksum")
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,120}", name):
            raise ValueError("Invalid metadata filename")
        return str(self.root / digest[:2] / digest / name)

    def _exchange(self, request):
        result = subprocess.run(
            [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                "-o",
                "LogLevel=ERROR",
                self.host,
                shlex.join(["python3", "-c", _REMOTE]),
            ],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
            # Uvicorn handles terminal signals; keep the in-flight transfer
            # alive until its caller finishes or subprocess.run cleans it up.
            start_new_session=True,
        )
        if result.returncode:
            status = (
                f"SSH terminated by signal {-result.returncode}"
                if result.returncode < 0
                else f"SSH exited with status {result.returncode}"
            )
            detail = result.stderr.strip()[-500:]
            raise OSError(
                f"Central metadata transfer failed ({self.host}; {status})"
                + (f": {detail}" if detail else "")
            )
        return json.loads(result.stdout)

    def _request(self, operation, digest, content=None, *, name="content"):
        self.path(digest, name)
        request = {
            "operation": operation,
            "root": str(self.root),
            "sha256": digest,
            "name": name,
        }
        if content is not None:
            request["content"] = base64.b64encode(content).decode()
        return self._exchange(request)

    def put(self, content: bytes, *, name="content"):
        if len(content) > MAX_BYTES:
            raise ValueError("Supporting metadata file exceeds 16 MiB")
        digest = hashlib.sha256(content).hexdigest()
        path = self.path(digest, name)
        result = self._request("put", digest, content, name=name)
        if result["sha256"] != digest or result["size"] != len(content):
            raise OSError("Central metadata verification failed")
        return path

    def put_many(self, contents):
        """Deduplicated and bounded bulk transfer for an offline migration."""
        unique = {hashlib.sha256(content).hexdigest(): content for content in contents}
        values = list(unique.items())
        for start in range(0, len(values), 16):
            batch = values[start : start + 16]
            if any(len(content) > MAX_BYTES for _, content in batch):
                raise ValueError("Supporting metadata file exceeds 16 MiB")
            result = self._exchange(
                [
                    {
                        "operation": "put",
                        "root": str(self.root),
                        "sha256": digest,
                        "name": "body",
                        "content": base64.b64encode(content).decode(),
                    }
                    for digest, content in batch
                ]
            )
            if len(result) != len(batch) or any(
                item["sha256"] != digest or item["size"] != len(content)
                for item, (digest, content) in zip(result, batch)
            ):
                raise OSError("Incomplete metadata bulk transfer")
        return {digest: self.path(digest, "body") for digest in unique}

    def read(self, path, digest):
        name = PurePosixPath(path).name
        if path != self.path(digest, name):
            raise ValueError("Saved metadata path does not match its checksum")
        result = self._request("get", digest, name=name)
        content = base64.b64decode(result["content"], validate=True)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(
                "Saved submission script changed: central metadata checksum mismatch"
                if name.endswith(".sbatch")
                else "Central metadata checksum mismatch"
            )
        return content

    def read_many(self, references):
        output = []
        for start in range(0, len(references), 16):
            batch = references[start : start + 16]
            for ref in batch:
                if ref["path"] != self.path(ref["sha256"], "body"):
                    raise ValueError("Invalid metadata object path")
            values = self._exchange(
                [
                    {
                        "operation": "get",
                        "root": str(self.root),
                        "sha256": ref["sha256"],
                        "name": "body",
                    }
                    for ref in batch
                ]
            )
            if len(values) != len(batch):
                raise ValueError("Incomplete metadata object batch")
            for value, ref in zip(values, batch):
                content = base64.b64decode(value["content"], validate=True)
                if (
                    len(content) != ref["size"]
                    or hashlib.sha256(content).hexdigest() != ref["sha256"]
                ):
                    raise ValueError("Metadata object checksum mismatch")
                output.append(content)
        return output
