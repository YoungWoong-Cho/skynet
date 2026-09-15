"""Actual selected-file removal, archive verification and unchanged sibling replay."""

import base64
import hashlib
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from skynet_app.database import canonical_json
from skynet_app.live_xr_archive_remote import archive_control
from skynet_app.live_xr_review import LiveReviewService
from skynet_app.recording_file_deletion import (
    RecordingFileMaintenance,
    recording_identity,
)
from tests.test_recording_deletion import recording  # noqa: F401


def identity(session, path):
    return session + ":" + base64.urlsafe_b64encode(path.encode()).decode().rstrip("=")


@pytest.fixture
def files(recording, monkeypatch):  # noqa: F811 - imported pytest fixture
    parent, live, job, folder, review, root = recording
    parent.reviews.slot = LiveReviewService.slot
    parent.videos.lock = threading.RLock()
    parent.videos.generations = {}
    service = RecordingFileMaintenance(
        parent.db,
        live,
        parent.reviews,
        parent.videos,
        parent.previews,
        conversion_root=parent.conversion_root,
    )
    old = Path(job["archive"]["root"]).parent
    # Three native files, each with a camera sidecar and independent cached review.
    names = ["recordings/one.pkl", "recordings/two.pkl", "recordings/three.pkl"]
    manifest = job["archive"]["manifest"]
    for i, name in enumerate(names):
        path = old / "output" / name
        path.write_bytes(f"recording-{i}".encode())
        path.with_suffix(".hdf5").write_bytes(f"images-{i}".encode())
        local = parent.reviews.root / job["id"] / str(i)
        local.mkdir(parents=True, exist_ok=True)
        (local / "keep.txt").write_text(name)
    manifest["files"] = [
        {
            "path": p.relative_to(old).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(old.rglob("*"))
        if p.is_file()
    ]
    manifest["directories"] = sorted(
        p.relative_to(old).as_posix() for p in old.rglob("*") if p.is_dir()
    )
    checksum = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    final = old.with_name(checksum)
    old.rename(final)
    archive = dict(
        job["archive"],
        manifest=manifest,
        manifest_sha256=checksum,
        root=str(final / "output"),
    )
    live.update(
        job["id"],
        recordings=names,
        archive=archive,
        recording_checksums={
            n: hashlib.sha256((final / "output" / n).read_bytes()).hexdigest()
            for n in names
        },
        recording_images={
            n: {"path": str(Path(n).with_suffix(".hdf5"))} for n in names
        },
    )
    (folder / "manifests").mkdir()
    (folder / "manifests" / (checksum + ".json")).write_text(canonical_json(manifest))
    import skynet_app.cluster_config as config

    monkeypatch.setattr(
        config,
        "CLUSTER",
        type(
            "Cluster",
            (),
            {"paths": type("Paths", (), {"datasets": str(root / "datasets")})()},
        )(),
    )
    calls = []

    def call(transport, host, operation, **request):
        assert host == "sky2"
        calls.append(operation)
        return archive_control(dict(request, operation=operation))

    monkeypatch.setattr(live.archive, "_call", call)
    return service, live, job["id"], names, final, calls


def test_remove_middle_then_first_preserves_sibling_bytes_slots_and_archive(files):
    service, live, session, names, final, calls = files
    original = {n: (final / "output" / n).read_bytes() for n in names}
    selector = identity(session, names[1])
    plan = service.preview("recording-file", selector)
    assert not plan["blockers"]
    assert plan["counts"]["recordings"] == 1
    assert not any(
        item["path"] == str(final / "output" / names[0]) for item in plan["files"]
    )
    service.delete("recording-file", selector, plan["token"])
    current = live.get(session)
    assert current["recordings"] == [names[0], names[2]]
    assert current["recording_slots"] == {names[0]: 0, names[2]: 2}
    assert not (final / "output" / names[1]).exists()
    assert not (final / "output" / names[1]).with_suffix(".hdf5").exists()
    assert not (service.reviews.root / session / "1").exists()
    assert (service.reviews.root / session / "2" / "keep.txt").read_text() == names[2]
    assert (final / "output" / names[2]).read_bytes() == original[names[2]]
    assert current["archive"]["root"] == str(final / "output")
    reader = object.__new__(LiveReviewService)
    reader.live, reader.root = live, service.reviews.root
    assert reader.source(session, 1)[1] == str(final / "output" / names[2])
    assert reader.directory(session, 1) == service.reviews.root / session / "2"
    assert "/reviews/2/" in reader.remote_location(session, 1, "review.json").path
    from skynet_app.policy_exports import PolicyExportService
    state_only = dict(current, recording_images={})
    sources = PolicyExportService.sources(SimpleNamespace(reviews=reader), state_only, require_images=False)
    assert [(source["index"], source["path"]) for source in sources] == [(0, names[0]), (1, names[2])]

    assert all(
        item["path"] != "output/" + names[1]
        for item in current["archive"]["manifest"]["files"]
    )
    assert live.archive._verify(current, current["archive"])["verified"]
    assert service.delete("recording-file", selector, plan["token"])["already_deleted"]
    # A stale confirmation can never select the sibling that moved into index 1.
    assert (final / "output" / names[2]).exists()
    for name in (names[0], names[2]):
        selector = identity(session, name)
        plan = service.preview("recording-file", selector)
        service.delete("recording-file", selector, plan["token"])
    assert live.get(session)["recordings"] == []
    current = live.get(session)
    assert live.archive._verify(current, current["archive"])["verified"]
    assert calls.count("revise") == 3


def test_failed_inventory_publish_blocks_new_work_and_can_retry(files, monkeypatch):
    service, live, session, names, final, calls = files
    selector = identity(session, names[1])
    plan = service.preview("recording-file", selector)
    saved = live.archive._call
    monkeypatch.setattr(
        live.archive,
        "_call",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("connection lost")),
    )
    with pytest.raises(OSError, match="connection lost"):
        service.delete("recording-file", selector, plan["token"])
    assert names[1] in live.get(session)["recordings"]
    with pytest.raises(ValueError, match="being deleted"):
        with live.recording_guard(session):
            pytest.fail("must not start processing a partially deleted session")
    other = service.preview("recording-file", identity(session, names[0]))
    assert any(item["id"] == selector for item in other["blockers"])
    monkeypatch.setattr(live.archive, "_call", saved)
    plan = service.preview("recording-file", selector)
    assert plan["retry"] and not plan["blockers"]
    service.delete("recording-file", selector, plan["token"])
    assert live.archive._verify(live.get(session), live.get(session)["archive"])[
        "verified"
    ]


def test_dataset_dependency_and_shared_camera_files_block_removal(files):
    service, live, session, names, final, _ = files
    resource = service.db.create_data_resource(
        category="dataset",
        provider="collection",
        namespace="datasets",
        name="Training",
        kind="demonstrations",
        metadata={"session_id": session},
    )
    selector = identity(session, names[0])
    plan = service.preview("recording-file", selector)
    assert any(
        b["kind"] == "dataset" and b["id"] == resource["id"] for b in plan["blockers"]
    )
    with pytest.raises(ValueError, match="dependencies"):
        service.delete("recording-file", selector, plan["token"])
    assert (final / "output" / names[0]).exists()


@pytest.mark.parametrize(
    "value",
    [
        "not-a-session:YXNk",
        "6a257afb-642c-41dd-b77e-e7cc92ee1b28:L3RtcC9vdGhlci5wa2w",
        "6a257afb-642c-41dd-b77e-e7cc92ee1b28:Li4vZmlsZS5wa2w",
    ],
)
def test_reject_invalid_recording_identity(value):
    with pytest.raises(ValueError):
        recording_identity(value)
