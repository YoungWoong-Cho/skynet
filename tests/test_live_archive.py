"""Real local subprocesses emulate both SSH endpoints; no network or live data."""

import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from skynet_app import live_xr_archive as module
from skynet_app.database import canonical_json
from skynet_app.live_xr_archive_remote import archive_control


class Transport:
    def candidates(self, host):
        assert host in {"sky2", "bonjour-test"}
        return (host,)

    def _remote_path(self, path):
        return path


@pytest.fixture
def archives(tmp_path, monkeypatch):
    identifier = "c94e16e6-6db0-4e66-b351-184aec09e0d2"
    workstation = tmp_path / "bonjour"
    root = workstation / "sessions" / identifier
    root.mkdir(parents=True)
    profile = dict(execution="workstation", gateway="bonjour-test", work_root=str(workstation))
    unit = f"skynet-live-{identifier}.service"
    original = dict(id=identifier, profile=profile, root=str(root), gateway="bonjour-test", job_id=unit,
                    state="CAPTURED", scheduler_final=True, recordings=["recordings/live/20.pkl"])
    files = {
        "request.json": canonical_json(profile).encode(),
        "runner.py": b"# immutable submitted worker\n",
        "stdout.log": b"Collection finished\n",
        "stderr.log": b"",
        "output/status.json": canonical_json(dict(job_id=unit, state="CAPTURED")).encode(),
        "output/recordings/live/20.pkl": bytes(range(256)) * 9000,
        "output/recordings/live/20.hdf5": b"training images",
        "output/recordings/live/20.mp4": b"saved preview video",
        "output/receipts.jsonl": b'{"recorded": true}\n',
    }
    original["worker_sha256"] = hashlib.sha256(files["runner.py"]).hexdigest()
    for name, content in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (root / "output/empty").mkdir()
    jobs = {identifier: original}
    updates = []

    def update(key, **changes):
        jobs[key].update(copy.deepcopy(changes))
        updates.append(copy.deepcopy(changes))

    transport = Transport()
    live = SimpleNamespace(cluster=transport, get=lambda key: copy.deepcopy(jobs[key]),
                           update=update, list=lambda: copy.deepcopy(list(jobs.values())),
                           refresh=lambda key: copy.deepcopy(jobs[key]), transport=lambda job: transport)
    monkeypatch.setattr(module, "CLUSTER", SimpleNamespace(paths=SimpleNamespace(datasets=str(tmp_path / "sky2/datasets"))))
    # The real subprocess runs the exact shipped worker. Only SSH is replaced;
    # a fixture systemctl reports a stopped/garbage-collected owned service.
    binary = tmp_path / "bin"
    binary.mkdir()
    executable = binary / "systemctl"
    executable.write_text("#!/bin/sh\nprintf 'LoadState=not-found\\n'\n")
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary) + ":/usr/bin:/bin")
    popen = subprocess.Popen
    launches = []

    def launch(args, **kwargs):
        assert args[0] == "ssh", "Only the bounded SSH entry points should launch locally"
        launches.append(args)
        command = shlex.split(args[-1])
        assert command[:2] == ["python3", "-c"]
        return popen([sys.executable, "-c", command[2]], **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    service = module.LiveArchiveService(live)
    yield SimpleNamespace(service=service, live=live, identifier=identifier, root=root, files=files,
                          jobs=jobs, updates=updates, launches=launches, executable=executable)
    service.stop()


def test_large_control_request_survives_slow_reader_and_full_output_pipes(archives, monkeypatch):
    a = archives
    program = (
        "import json,sys,time\n"
        "time.sleep(.5)\n"
        "sys.stderr.write('diagnostic' * 30000);sys.stderr.flush()\n"
        "sys.stdout.write(' ' * 200000);sys.stdout.flush()\n"
        "request=json.load(sys.stdin)\n"
        "print(json.dumps({'length': len(request['manifest']), 'operation': request['operation']}))\n"
    )
    monkeypatch.setattr(a.service, "_program", lambda: program)
    # A broken partial-input retry must fail promptly instead of hanging the suite.
    deadline = threading.Timer(5, a.service.stopping.set)
    deadline.start()
    try:
        result = a.service._call(a.live.cluster, "sky2", "verify", manifest="x" * 200000)
        assert result == {"length": 200000, "operation": "verify"}
    finally:
        deadline.cancel()
        deadline.join()


def test_control_request_can_stop_while_large_input_is_blocked(archives, monkeypatch):
    a = archives
    monkeypatch.setattr(a.service, "_program", lambda: "import time;time.sleep(30)")
    stopping = threading.Timer(.3, a.service.stopping.set)
    started = time.monotonic()
    stopping.start()
    try:
        with pytest.raises(RuntimeError, match="app shutdown"):
            a.service._call(a.live.cluster, "sky2", "verify", manifest="x" * 200000)
    finally:
        stopping.cancel()
        stopping.join()
    assert time.monotonic() - started < 5
    assert a.root.exists()


def test_control_request_reports_remote_failure_before_large_input_is_read(archives, monkeypatch):
    a = archives
    monkeypatch.setattr(a.service, "_program", lambda: "import sys;sys.stderr.write('verification refused');sys.exit(9)")
    with pytest.raises(module.ClusterError, match="verification refused"):
        a.service._call(a.live.cluster, "sky2", "verify", manifest="x" * 200000)
    assert a.root.exists()


def test_complete_tree_streamed_verified_before_exact_source_removal(archives):
    a = archives
    result = a.service.archive(a.identifier)
    assert result["state"] == "VERIFIED" and not result["source_removed"] and a.root.exists()
    copied = Path(result["root"]).parent
    assert {item["path"] for item in result["manifest"]["files"]} == a.files.keys()
    assert (copied / "output/empty").is_dir()
    for name, content in a.files.items():
        assert (copied / name).read_bytes() == content
    assert len(a.launches) == 3  # inventory + source stream + destination receive
    sibling = a.root.parent / "another-session"
    sibling.mkdir()
    (sibling / "keep").write_text("untouched")
    completed = a.service.cleanup_source(a.identifier)
    assert completed["state"] == "READY" and completed["source_removed"]
    assert not a.root.exists() and (sibling / "keep").read_text() == "untouched"
    assert [change["archive"]["state"] for change in a.updates][:3] == ["COPYING", "VERIFIED", "CLEANUP_PENDING"]
    assert a.service.cleanup_source(a.identifier) == completed
    # Reconciliation verifies the same archive without retransferring payload.
    assert a.service.archive(a.identifier)["state"] == "READY"


def test_busy_consumer_defers_cleanup_but_can_resolve_verified_source(archives):
    a = archives
    a.service.archive(a.identifier)
    a.service.busy = lambda key: True
    result = a.service.archive(a.identifier, cleanup=True)
    assert result["state"] == "CLEANUP_PENDING" and a.root.exists()
    transport, gateway, path = a.service.resolve(a.live.get(a.identifier), "recordings/live/20.pkl")
    assert gateway == "sky2" and Path(path).read_bytes() == a.files["output/recordings/live/20.pkl"]
    assert "/derived/" in a.service.derived_root(a.live.get(a.identifier))
    with pytest.raises(ValueError, match="not part"):
        a.service.resolve(a.live.get(a.identifier), "unknown.pkl")
    with pytest.raises(ValueError, match="Invalid"):
        a.service.resolve(a.live.get(a.identifier), "../request.json")


def test_active_review_defers_copy_without_creating_a_snapshot(archives):
    a = archives
    a.service.busy = lambda key: True
    assert a.service.archive(a.identifier)["state"] == "QUEUED"
    assert a.root.exists() and not a.launches


def test_changed_source_is_preserved_after_transfer(archives):
    a = archives
    a.service.archive(a.identifier)
    (a.root / "output/recordings/live/20.pkl").write_bytes(b"new recording bytes")
    with pytest.raises(Exception, match="source session changed"):
        a.service.cleanup_source(a.identifier)
    assert a.root.exists() and a.live.get(a.identifier)["archive"]["state"] == "CLEANUP_PENDING"


def test_damaged_destination_blocks_deletion_and_revokes_archive_availability(archives):
    a = archives
    result = a.service.archive(a.identifier)
    Path(result["root"], "recordings/live/20.pkl").write_bytes(b"incomplete")
    with pytest.raises(Exception, match="incomplete or changed"):
        a.service.cleanup_source(a.identifier)
    assert a.root.exists() and not module.is_archived(a.live.get(a.identifier))
    assert a.live.get(a.identifier)["archive"]["state"] == "VERIFY_FAILED"


@pytest.mark.parametrize("inside", [False, True])
def test_symlinked_source_tree_is_not_archived(archives, inside):
    a = archives
    if inside:
        (a.root / "output/link").symlink_to(a.root / "request.json")
    else:
        moved = a.root.with_name("real-tree")
        a.root.rename(moved)
        a.root.symlink_to(moved, target_is_directory=True)
    with pytest.raises(Exception, match="link|owned regular files"):
        a.service.archive(a.identifier, cleanup=True)
    assert a.root.exists()


def test_running_or_foreign_service_prevents_snapshot(archives):
    a = archives
    a.executable.write_text("#!/bin/sh\nprintf 'LoadState=loaded\\nActiveState=active\\nSubState=running\\nMainPID=42\\nEnvironment=SKYNET_LIVE_SESSION_ID=" + a.identifier + "\\n'\n")
    with pytest.raises(Exception, match="still running"):
        a.service.archive(a.identifier, cleanup=True)
    a.executable.write_text("#!/bin/sh\nprintf 'LoadState=loaded\\nActiveState=inactive\\nMainPID=0\\nEnvironment=OTHER_SESSION=1\\n'\n")
    with pytest.raises(Exception, match="ownership"):
        a.service.archive(a.identifier, cleanup=True)
    assert a.root.exists()


def test_profile_mismatch_prevents_snapshot(archives):
    a = archives
    (a.root / "request.json").write_text('{"different": true}')
    with pytest.raises(Exception, match="profile no longer matches"):
        a.service.archive(a.identifier, cleanup=True)
    assert a.root.exists()


def test_partial_failed_launch_preserves_diagnostics_then_cleans(archives):
    a = archives
    a.jobs[a.identifier].update(state="FAILED", job_id=None, scheduler_final=False, recordings=[])
    (a.root / "output/status.json").unlink()
    (a.root / "request.json").unlink()
    result = a.service.archive(a.identifier, cleanup=True)
    assert result["state"] == "READY" and not a.root.exists()
    assert (Path(result["root"]).parent / "stdout.log").read_bytes() == a.files["stdout.log"]


def test_absent_empty_session_has_explicit_empty_receipt(archives):
    import shutil
    a = archives
    shutil.rmtree(a.root)
    a.jobs[a.identifier].update(state="FAILED", job_id=None, scheduler_final=False, recordings=[])
    result = a.service.archive(a.identifier, cleanup=True)
    assert result["state"] == "EMPTY" and result["source_missing"]
    assert not result.get("manifest_sha256") and not module.is_archived(a.live.get(a.identifier))


def test_missing_recorded_source_never_becomes_empty_or_ready(archives):
    import shutil
    a = archives
    shutil.rmtree(a.root)
    with pytest.raises(Exception):
        a.service.archive(a.identifier, cleanup=True)
    assert a.live.get(a.identifier)["archive"]["state"] == "FAILED"


def test_disabled_service_never_queues_cleanup_and_shutdown_stops_new_dispatch(archives):
    a = archives
    a.service.ensure(a.identifier)
    assert not a.service.active and not a.launches
    a.service.start(enabled=True, cleanup_enabled=False)
    deadline = time.monotonic() + 5
    while not module.is_archived(a.live.get(a.identifier)) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert module.is_archived(a.live.get(a.identifier)) and a.root.exists()
    a.service.stop()
    count = len(a.launches)
    a.service.dispatch(a.identifier, cleanup=True)
    assert len(a.launches) == count and a.root.exists()


def test_interrupted_receive_resumes_verified_files_without_overwriting_them(archives, monkeypatch):
    import io
    import tarfile
    a = archives
    report = a.service._call(a.live.transport(a.jobs[a.identifier]), "bonjour-test", "manifest",
                             **a.service._source(a.jobs[a.identifier]))
    request = dict(operation="receive", session_id=a.identifier, datasets_root=module.CLUSTER.paths.datasets,
                   **report)

    def stream(names):
        target = io.BytesIO()
        with tarfile.open(fileobj=target, mode="w") as archive:
            for name in names:
                metadata = tarfile.TarInfo(name)
                metadata.size = len(a.files[name])
                archive.addfile(metadata, io.BytesIO(a.files[name]))
        return SimpleNamespace(buffer=io.BytesIO(target.getvalue()))

    first = "request.json"
    monkeypatch.setattr(sys, "stdin", stream([first]))
    with pytest.raises(ValueError, match="complete session"):
        archive_control(request)
    base = Path(module.CLUSTER.paths.datasets) / "raw/dexverse-live" / a.identifier
    stage = base / (".incoming-" + report["manifest_sha256"])
    identity = (stage / first).stat().st_ino
    assert not (base / report["manifest_sha256"]).exists()
    monkeypatch.setattr(sys, "stdin", stream(a.files))
    result = archive_control(request)
    assert result["verified"]
    final = Path(result["root"]).parent
    assert (final / first).stat().st_ino == identity
    assert a.root.exists()


def test_transfer_rejects_unmanifested_traversal_entry(archives, monkeypatch):
    import io
    import tarfile
    a = archives
    report = a.service._call(a.live.transport(a.jobs[a.identifier]), "bonjour-test", "manifest",
                             **a.service._source(a.jobs[a.identifier]))
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode="w") as archive:
        metadata = tarfile.TarInfo("../../unrelated")
        metadata.size = 4
        archive.addfile(metadata, io.BytesIO(b"bad!"))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(target.getvalue())))
    with pytest.raises(ValueError, match="relative archive path"):
        archive_control(dict(operation="receive", session_id=a.identifier,
                             datasets_root=module.CLUSTER.paths.datasets, **report))
    assert a.root.exists()


def test_interrupted_source_deletion_resumes_only_verified_quarantine(archives, monkeypatch):
    import shutil
    a = archives
    result = a.service.archive(a.identifier)
    request = dict(operation="delete", **a.service._source(a.jobs[a.identifier]),
                   manifest=result["manifest"], manifest_sha256=result["manifest_sha256"])
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout="LoadState=not-found\n", stderr=""))
    remove = shutil.rmtree

    def interrupted(path):
        (path / "stdout.log").unlink()
        raise OSError("Synthetic interrupted deletion")

    interrupted.avoids_symlink_attacks = True
    monkeypatch.setattr(shutil, "rmtree", interrupted)
    with pytest.raises(OSError, match="interrupted deletion"):
        archive_control(request)
    assert not a.root.exists()
    quarantine = a.root.with_name(f".archived-{a.identifier}-{result['manifest_sha256']}")
    assert quarantine.is_dir() and Path(result["root"]).is_dir()
    monkeypatch.setattr(shutil, "rmtree", remove)
    extra = quarantine / "unverified-extra"
    extra.write_text("Must never be silently removed")
    with pytest.raises(ValueError, match="Unverified data appeared"):
        archive_control(request)
    assert extra.exists()
    extra.unlink()
    assert archive_control(request)["removed"] and not quarantine.exists()
    assert archive_control(request)["removed"]


def test_operator_cache_namespace_uses_separate_verified_tree(archives, monkeypatch):
    import io
    import tarfile
    a = archives
    manifest = dict(schema="skynet.live-archive/v1", session_id=a.identifier,
                    directories=["policy-exports"], files=[dict(path="policy-exports/failure.log", size_bytes=4,
                        sha256=hashlib.sha256(b"fail").hexdigest())])
    request = dict(operation="receive", namespace="operator-cache", session_id=a.identifier,
                   datasets_root=module.CLUSTER.paths.datasets, manifest=manifest,
                   manifest_sha256=hashlib.sha256(canonical_json(manifest).encode()).hexdigest())
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("policy-exports/failure.log")
        info.size = 4
        archive.addfile(info, io.BytesIO(b"fail"))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(buffer.getvalue())))
    result = archive_control(request)
    assert "/raw/operator-cache/" in result["root"] and result["verified"]
    assert archive_control(dict(request, operation="verify"))["verified"]
    with pytest.raises(ValueError, match="namespace"):
        archive_control(dict(request, namespace="../../elsewhere"))


def test_failed_destination_does_not_leave_source_pipe_blocked(archives, monkeypatch):
    a = archives
    monkeypatch.setattr(a.service, "_program", lambda mode="json":
        "import sys;sys.exit(17)" if mode == "receive" else
        "import sys;sys.stdin.read();sys.stdout.buffer.write(b'x'*10000000)")
    started = time.monotonic()
    with pytest.raises(Exception, match="transfer"):
        a.service._transfer(a.jobs[a.identifier], {}, "0" * 64)
    assert time.monotonic() - started < 3 and a.root.exists()


def test_failed_unit_with_live_cgroup_descendants_blocks_archive(archives, monkeypatch):
    a = archives
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr="",
        stdout=f"LoadState=loaded\nActiveState=failed\nSubState=failed\nMainPID=0\nControlGroup=/archive-test\nEnvironment=SKYNET_LIVE_SESSION_ID={a.identifier}\n"))
    exists, read = Path.exists, Path.read_text
    group = Path("/sys/fs/cgroup/archive-test")
    monkeypatch.setattr(Path, "exists", lambda path: True if path in {group, group / "cgroup.events"} else exists(path))
    monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs:
        "populated 1\n" if path == group / "cgroup.events" else read(path, *args, **kwargs))
    with pytest.raises(ValueError, match="descendants are still running"):
        archive_control(dict(operation="manifest", **a.service._source(a.jobs[a.identifier])))
    assert a.root.exists()
