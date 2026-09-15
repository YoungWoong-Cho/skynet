"""Delete real isolated files through the same token/intent engine as run history."""

import copy
import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from skynet_app.database import Database, canonical_json
from skynet_app.live_xr import LiveXRService
from skynet_app.live_xr_archive import LiveArchiveService
from skynet_app.recording_deletion import RecordingMaintenance
from tests.test_maintenance import LocalCluster


@pytest.fixture
def recording(tmp_path, monkeypatch):
    root = tmp_path / "cluster"
    paths = SimpleNamespace(datasets=str(root / "datasets"))
    for module in ("recording_deletion", "live_xr_archive"):
        monkeypatch.setattr(
            "skynet_app." + module + ".CLUSTER", SimpleNamespace(paths=paths)
        )
    monkeypatch.setattr("skynet_app.recording_deletion.WORK_ROOT", str(root))
    cluster = LocalCluster()
    cluster.candidates = lambda host: (host,)
    cluster._remote_path = lambda path: path
    db = Database(tmp_path / "test.db")
    live = LiveXRService(db, cluster=cluster, root=tmp_path)
    live.archive = LiveArchiveService(live, cluster)
    reviews = SimpleNamespace(root=tmp_path / "reviews", active=set())
    videos = SimpleNamespace(active=set())
    previews = SimpleNamespace(states={}, lock=threading.RLock())
    conversions = SimpleNamespace(root=tmp_path / "conversions", active=set())
    service = RecordingMaintenance(
        db.for_workspace("legacy"), live, reviews, videos, previews, conversions
    )
    identifier = str(uuid4())
    manifest = dict(schema="skynet.live-archive/v1", session_id=identifier, files=[])
    checksum = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    folder = root / "datasets/raw/dexverse-live" / identifier
    original = folder / checksum / "output/recordings/one.pkl"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"original")
    derived = folder / "derived" / "viewer.json"
    derived.parent.mkdir(parents=True)
    derived.write_bytes(b"preview")
    review = reviews.root / identifier / "0"
    review.mkdir(parents=True)
    (review / "status.json").write_text('{"state":"READY"}')
    job = dict(
        id=identifier,
        state="STOPPED",
        scheduler_final=True,
        root="/bonjour/sessions/" + identifier,
        gateway="bonjour",
        job_id="stopped",
        profile=dict(task_name="Cube", execution="workstation"),
        recordings=["recordings/one.pkl"],
        created_at="2026-09-14T00:00:00Z",
        archive=dict(
            state="READY",
            source_removed=True,
            gateway="sky2",
            root=str(folder / checksum / "output"),
            manifest=manifest,
            manifest_sha256=checksum,
        ),
    )
    with db.transaction() as c:
        c.execute(
            "INSERT INTO live_xr_sessions VALUES (?,?)",
            (identifier, canonical_json(job)),
        )
    yield service, live, job, folder, review, root
    live.executor.shutdown(wait=True)
    live.archive.executor.shutdown(wait=True)


def test_recording_delete_cascades_files_and_records_and_is_idempotent(recording):
    service, live, job, folder, review, root = recording
    other = root / "datasets/raw/keep.pkl"
    other.write_bytes(b"keep")
    identifier = job["id"]
    conversion_id = str(uuid4())
    conversion = dict(
        id=conversion_id,
        session_id=identifier,
        state="FAILED",
        root=str(root / "jobs/runs" / conversion_id),
        dataset_root=str(
            root / "datasets/derivatives/dexverse-live" / identifier / conversion_id
        ),
    )
    for path in (
        conversion["root"],
        conversion["dataset_root"],
        service.conversions.root / conversion_id,
    ):
        Path(path).mkdir(parents=True)
        (Path(path) / "data").write_bytes(b"temporary")
    with service.db.transaction() as c:
        c.execute(
            "INSERT INTO live_conversions VALUES (?,?)",
            (conversion_id, canonical_json(conversion)),
        )
    plan = service.preview("recording", identifier)
    assert not plan["blockers"]
    assert folder.exists() and live.get(identifier)
    assert plan["counts"] == {"live_xr_sessions": 1, "live_conversions": 1}
    assert service.delete("recording", identifier, plan["token"])["deleted"]
    assert not folder.exists() and not review.parent.exists()
    assert not Path(conversion["root"]).exists()
    assert other.read_bytes() == b"keep"
    with service.db.connection() as c:
        for table in ("live_xr_sessions", "live_conversions", "maintenance_operations"):
            assert not c.execute("SELECT * FROM " + table).fetchall()
    assert service.delete("recording", identifier, plan["token"])["already_deleted"]


def test_dataset_dependency_uses_all_recording_memberships(recording):
    service, live, job, folder, _, _ = recording
    resource = service.db.create_data_resource(
        category="dataset",
        provider="collection",
        namespace="datasets",
        name="Combined",
        kind="demonstrations",
        metadata={"session_id": job["id"]},
    )
    plan = service.preview("recording", job["id"])
    assert any(
        x["id"] == resource["id"] and x["kind"] == "dataset" for x in plan["blockers"]
    )
    assert not plan["files"]
    with pytest.raises(ValueError, match="dependencies"):
        service.delete("recording", job["id"], plan["token"])
    assert folder.exists()


def test_changed_files_and_new_dependencies_invalidate_confirmation(recording):
    service, _, job, folder, _, _ = recording
    plan = service.preview("recording", job["id"])
    (folder / "new.json").write_text("new")
    with pytest.raises(ValueError, match="changed"):
        service.delete("recording", job["id"], plan["token"])
    plan = service.preview("recording", job["id"])
    service.db.create_data_resource(
        category="dataset",
        provider="collection",
        namespace="datasets",
        name="New",
        kind="demonstrations",
        metadata={"session_id": job["id"]},
    )
    with pytest.raises(ValueError, match="dependencies"):
        service.delete("recording", job["id"], plan["token"])
    assert folder.exists()


def test_interrupted_delete_retains_intent_blocks_consumers_and_retries(
    recording, monkeypatch
):
    service, live, job, folder, review, _ = recording
    plan = service.preview("recording", job["id"])
    original = service.remote

    def fail(operation, *args, **kwargs):
        if operation == "delete":
            raise OSError("offline")
        return original(operation, *args, **kwargs)

    monkeypatch.setattr(service, "remote", fail)
    with pytest.raises(OSError, match="offline"):
        service.delete("recording", job["id"], plan["token"])
    assert live.get(job["id"])
    with pytest.raises(ValueError, match="being deleted"):
        with live.recording_guard(job["id"]):
            pytest.fail("must not start consuming a deleting recording")
    with pytest.raises(ValueError, match="being deleted"):
        service.db.create_data_resource(
            category="dataset",
            provider="collection",
            namespace="datasets",
            name="Race",
            kind="demonstrations",
            metadata={"session_id": job["id"]},
        )
    monkeypatch.setattr(service, "remote", original)
    retry = service.preview("recording", job["id"])
    assert retry["retry"]
    assert service.delete("recording", job["id"], retry["token"])["deleted"]
    assert not folder.exists() and not review.parent.exists()


def test_shared_consumer_leases_block_deletion_across_hosts(recording):
    service, live, job, folder, _, _ = recording
    other = LiveXRService(live.database, cluster=live.cluster, root=live.root)
    entered, release = threading.Event(), threading.Event()

    def reading():
        with other.recording_guard(job["id"]):
            entered.set()
            release.wait(10)

    thread = threading.Thread(target=reading)
    thread.start()
    try:
        assert entered.wait(3)
        with live.recording_guard(job["id"]):
            with live.recording_guard(job["id"]):
                assert folder.exists()  # Concurrent and nested consumers are allowed.
        plan = service.preview("recording", job["id"])
        assert any(
            "processing" in x["reason"] or "work to finish" in x["reason"]
            for x in plan["blockers"]
        )
        with pytest.raises(ValueError, match="active"):
            service.delete("recording", job["id"], plan["token"])
    finally:
        release.set()
        thread.join(5)
        other.executor.shutdown(wait=True)
    assert not service.preview("recording", job["id"])["blockers"]


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "COLLECTING"},
        {"scheduler_final": False},
        {"archive": {"state": "COPYING"}},
        {"archive": {"state": "READY", "source_removed": False}},
    ],
)
def test_active_or_unarchived_recordings_cannot_be_deleted(recording, changes):
    service, live, job, folder, _, _ = recording
    live.update(job["id"], **changes)
    assert service.preview("recording", job["id"])["blockers"]
    assert folder.exists()


def test_recording_deletion_rejects_unsafe_archive_and_symlink(recording):
    service, live, job, folder, review, root = recording
    original = copy.deepcopy(job["archive"])
    live.update(job["id"], archive={**original, "root": str(root)})
    with pytest.raises(ValueError, match="location"):
        service.preview("recording", job["id"])
    live.update(job["id"], archive=original)
    (review / "status.json").unlink()
    review.rmdir()
    review.parent.rmdir()
    review.parent.symlink_to(folder)
    with pytest.raises(ValueError, match="symbolic"):
        service.preview("recording", job["id"])
    assert folder.exists()


def test_recording_deletion_api_uses_shared_preview_and_confirm(recording, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from skynet_app import maintenance_api

    service, live, job, folder, _, _ = recording
    monkeypatch.setattr(maintenance_api, "recording_manager", lambda kind="recording": service)
    app = FastAPI()
    app.include_router(maintenance_api.router)
    path = "/api/maintenance/history/recording/" + job["id"]
    with TestClient(app) as client:
        response = client.get(path)
        assert response.status_code == 200
        plan = response.json()
        assert folder.exists()
        assert (
            client.request("DELETE", path, json={"token": "0" * 64}).status_code == 409
        )
        response = client.request(
            "DELETE", path, json={"token": plan["token"], "gateway": "sky1"}
        )
        assert response.status_code == 200 and response.json()["deleted"]
        assert client.get(path).status_code == 404
        assert client.request("DELETE", path, json={"token": plan["token"]}).json()[
            "already_deleted"
        ]
    assert not folder.exists()


def test_pinned_variant_blocks_recording_deletion_with_preset_name(recording):
    service, _, job, folder, _, _ = recording
    project = service.db.create_project("Project")
    experiment = service.db.create_experiment(
        project_id=project["id"], name="Cube preset", requested_spec={}
    )
    service.db.create_variant(
        experiment["latest_revision"]["id"],
        name="Variant",
        parameters={},
        resolved_spec={"recording_id": job["id"]},
    )
    plan = service.preview("recording", job["id"])
    assert any(
        item["kind"] == "experiment"
        and item["id"] == experiment["id"]
        and item["label"] == "Cube preset"
        for item in plan["blockers"]
    )
    assert folder.exists()
