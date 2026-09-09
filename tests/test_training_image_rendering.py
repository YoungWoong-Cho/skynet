import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_images", ROOT / "ops/xr/native_session.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_image_receipts_require_original_and_image_checksums(tmp_path):
    folder = tmp_path / "recordings/live"
    folder.mkdir(parents=True)
    image = folder / "images.hdf5"
    image.write_bytes(b"test images")
    original = {"recordings/live/episode.pkl": dict(sha256="a" * 64, steps=3)}
    receipt = dict(path="recordings/live/images.hdf5", source_sha256="a" * 64, sha256=hashlib.sha256(image.read_bytes()).hexdigest(), size_bytes=image.stat().st_size, steps=3)
    path = tmp_path / "image-receipts.json"
    path.write_text(json.dumps({next(iter(original)): receipt}))
    worker.attach_rendered_images(tmp_path, original)
    assert next(iter(original.values()))["images"] == receipt
    receipt["source_sha256"] = "b" * 64
    path.write_text(json.dumps({next(iter(original)): receipt}))
    with pytest.raises(ValueError, match="different original"):
        worker.attach_rendered_images(tmp_path, original)
    receipt["source_sha256"] = "a" * 64
    path.write_text(json.dumps({next(iter(original)): receipt}))
    image.write_bytes(b"bad! images")
    with pytest.raises(ValueError, match="incomplete or changed"):
        worker.attach_rendered_images(tmp_path, original)


def test_render_process_has_no_xr_runtime_and_requires_ready_receipt(tmp_path, monkeypatch):
    args = []
    def start(argv, **kwargs):
        assert "XR_RUNTIME_JSON" not in kwargs["env"]
        assert "render_images.py" in argv[1]
        args.append(argv)
        (tmp_path / "image-progress.json").write_text(json.dumps(dict(state="READY", completed=1, total=1)))
        return SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(worker.subprocess, "Popen", start)
    states = []
    worker.prepare_training_images(tmp_path, tmp_path / "profile.json", tmp_path, tmp_path,
        {"XR_RUNTIME_JSON": "headset-runtime", "PATH": "/bin"}, lambda state, **details: states.append(state))
    assert args and states == ["RENDERING_IMAGES", "RENDERING_IMAGES"]


def test_render_timeout_terminates_child_and_never_marks_ready(tmp_path, monkeypatch):
    calls = []
    process = SimpleNamespace(poll=lambda: None, terminate=lambda: calls.append("terminate"), wait=lambda **k: 0)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **k: process)
    with pytest.raises(TimeoutError, match="original recordings are preserved"):
        worker.prepare_training_images(tmp_path, tmp_path / "profile.json", tmp_path, tmp_path, {}, lambda *a, **k: None, timeout=0)
    assert calls == ["terminate"]


def test_receipt_hashing_runs_without_python_311_file_digest(tmp_path, monkeypatch):
    import io
    monkeypatch.delattr(worker.hashlib, "file_digest", raising=False)
    data = b"original recording image bytes" * 100_000
    assert worker.stream_sha256(io.BytesIO(data)) == hashlib.sha256(data).hexdigest()
    test_image_receipts_require_original_and_image_checksums(tmp_path)
