"""Immutable, checksum-addressed supporting files stored beside the central DB.

Only small manifests and submission receipts use this store. Recordings, trained
checkpoints and rollout videos continue to use their registered cluster paths.
"""

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
request = json.load(sys.stdin)
root = pathlib.Path(request["root"])
digest = request["sha256"]
path = root / digest[:2] / digest / request["name"]
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
    finally:
        os.unlink(temporary)
content = path.read_bytes()
if len(content) > 16 * 1024 * 1024 or hashlib.sha256(content).hexdigest() != digest:
    raise ValueError("Metadata object checksum mismatch")
print(json.dumps({"sha256": digest, "size": len(content), "content":
    base64.b64encode(content).decode() if request["operation"] == "get" else None}))
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

    def _request(self, operation, digest, content=None, *, name="content"):
        request = {
            "operation": operation,
            "root": str(self.root),
            "sha256": digest,
            "name": name,
        }
        if content is not None:
            request["content"] = base64.b64encode(content).decode()
        result = subprocess.run(
            [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=8",
                self.host,
                shlex.join(["python3", "-c", _REMOTE]),
            ],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode:
            raise OSError(
                "Central metadata transfer failed: " + result.stderr.strip()[-500:]
            )
        return json.loads(result.stdout)

    def put(self, content: bytes, *, name="content"):
        if len(content) > MAX_BYTES:
            raise ValueError("Supporting metadata file exceeds 16 MiB")
        digest = hashlib.sha256(content).hexdigest()
        path = self.path(digest, name)
        result = self._request("put", digest, content, name=name)
        if result["sha256"] != digest or result["size"] != len(content):
            raise OSError("Central metadata verification failed")
        return path

    def read(self, path, digest):
        name = PurePosixPath(path).name
        if path != self.path(digest, name):
            raise ValueError("Saved metadata path does not match its checksum")
        result = self._request("get", digest, name=name)
        content = base64.b64decode(result["content"], validate=True)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(
                "Saved submission script changed: central metadata checksum mismatch"
            )
        return content
