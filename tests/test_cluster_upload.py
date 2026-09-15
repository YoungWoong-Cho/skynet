import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from skynet_app import cluster_upload
from skynet_app.cluster_runtime import ClusterClient, ClusterError


@pytest.fixture
def local_upload(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "cluster"
    monkeypatch.setattr(cluster_upload, "WORK_ROOT", str(root))
    original_run = subprocess.run
    transfers = []

    def local_ssh(argv, **kwargs):
        assert argv[0] == "ssh" and argv[-2] == "sky2"
        assert "stdin" in kwargs and "input" not in kwargs and "text" not in kwargs
        transfers.append(argv)
        command = argv[-1]
        if sys.platform == "darwin":
            # The remote Linux receipt uses GNU flags; macOS shasum accepts them.
            command = 'sha256sum() { /usr/bin/shasum -a 256 "$@"; }\n' + command
        return original_run(["/bin/bash", "-c", command], **kwargs)

    monkeypatch.setattr(cluster_upload.subprocess, "run", local_ssh)
    return ClusterClient(("sky2",)), root, transfers


def test_upload_preserves_binary_bytes_and_reuses_immutable_destination(local_upload, tmp_path):
    cluster, root, transfers = local_upload
    source = tmp_path / "dataset.zip"
    content = b"PK\x00\xff\xc3\xa9\r\n$(not a command)\n"
    source.write_bytes(content)
    checksum = hashlib.sha256(content).hexdigest()
    target = cluster_upload.upload_capture(cluster, source, "run-1", checksum, "sky2", relative_path="archive/dataset.zip")
    assert Path(target).read_bytes() == content
    assert cluster_upload.upload_capture(cluster, source, "run-1", checksum, "sky2", relative_path="archive/dataset.zip") == target
    assert Path(target).read_bytes() == content and len(transfers) == 2
    assert source.read_bytes() == content
    assert not list(root.rglob(".capture-*"))


def test_upload_failure_preserves_existing_destination_and_source(local_upload, tmp_path):
    cluster, root, _ = local_upload
    source = tmp_path / "dataset.zip"
    source.write_bytes(b"new dataset")
    target = root / "jobs/runs/run-1/dataset.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"immutable prior bytes")
    with pytest.raises(ClusterError, match="verification failed"):
        cluster_upload.upload_capture(cluster, source, "run-1", hashlib.sha256(source.read_bytes()).hexdigest(), "sky2", relative_path="dataset.zip")
    assert target.read_bytes() == b"immutable prior bytes"
    assert source.read_bytes() == b"new dataset"
    with pytest.raises(ClusterError, match="verification failed"):
        cluster_upload.upload_capture(cluster, source, "run-2", "a" * 64, "sky2", relative_path="dataset.zip")
    assert not (root / "jobs/runs/run-2/dataset.zip").exists()
    assert not list(root.rglob(".capture-*"))


@pytest.mark.parametrize("changes", [
    {"run_id": "../outside"}, {"digest": "invalid"},
    {"relative_path": "../outside"}, {"relative_path": "/outside"},
    {"gateway": "unknown"},
])
def test_upload_rejects_unsafe_destination_before_open_or_transfer(tmp_path, monkeypatch, changes):
    monkeypatch.setattr(cluster_upload.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected transfer"))
    args = dict(cluster=ClusterClient(("sky2",)), path=tmp_path / "not-present", run_id="run-1", digest="a" * 64, gateway="sky2", relative_path="dataset.zip")
    args.update(changes)
    with pytest.raises(ValueError):
        cluster_upload.upload_capture(**args)


def test_upload_timeout_is_explicit_and_preserves_source(tmp_path, monkeypatch):
    source = tmp_path / "dataset.zip"
    source.write_bytes(b"saved original")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(cluster_upload.subprocess, "run", timeout)
    with pytest.raises(ClusterError, match="timed out"):
        cluster_upload.upload_capture(ClusterClient(("sky2",)), source, "run-1", hashlib.sha256(source.read_bytes()).hexdigest(), "sky2")
    assert source.read_bytes() == b"saved original"
