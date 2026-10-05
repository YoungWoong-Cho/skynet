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
    cluster._remote_path = lambda path: path
    db = Database(tmp_path / "test.db")
    live = LiveXRService(db, cluster=cluster, root=tmp_path)
    live.archive = LiveArchiveService(live, cluster)
    reviews = SimpleNamespace(root=tmp_path / "reviews", active=set())
    videos = SimpleNamespace(active=set())
    previews = SimpleNamespace(states={}, lock=threading.RLock())
    service = RecordingMaintenance(
        db.for_workspace("legacy"), live, reviews, videos, previews,
        conversion_root=tmp_path / "conversions",
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
        service.conversion_root / conversion_id,
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


def test_dataset_dependency_uses_exact_output_recording_membership(recording):
    service, live, job, folder, _, _ = recording
    resource = service.db.create_data_resource(
        category="dataset",
        provider="collection",
        namespace="datasets",
        source_key="Combined",
        kind="demonstrations",
        metadata={"session_id": job["id"]},
    )
    first = service.db.create_data_resource_version(
        resource["id"], revision="first", format="skynet.recording-dataset/v1", path="/prepared/first",
        manifest_sha256="a" * 64, metadata={"sources": [{"session_id": job["id"]}]},
    )
    second = service.db.create_data_resource_version(
        resource["id"], revision="second", format="skynet.recording-dataset/v1", path="/prepared/second",
        manifest_sha256="b" * 64, metadata={"sources": [{"session_id": "another-recording"}]},
    )
    service.db.update_dataset(first["id"], display_name="Renamed dataset", archived=True)
    plan = service.preview("recording", job["id"])
    assert any(
        x["id"] == first["id"] and x["kind"] == "dataset" and x["label"] == "Renamed dataset"
        for x in plan["blockers"]
    )
    assert not any(x["id"] in {resource["id"], second["id"]} for x in plan["blockers"])
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
    resource = service.db.create_data_resource(
        category="dataset",
        provider="collection",
        namespace="datasets",
        source_key="New",
        kind="demonstrations",
        metadata={"session_id": job["id"]},
    )
    service.db.create_data_resource_version(
        resource["id"], revision="new", format="skynet.recording-dataset/v1", path="/prepared/new",
        manifest_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="dependencies"):
        service.delete("recording", job["id"], plan["token"])
    assert folder.exists()


def raw_source(service, job):
    resource = service.db.create_data_resource(
        category="dataset", provider="collection", namespace="datasets", source_key=job["id"],
        kind="demonstrations", metadata={"session_id": job["id"]},
    )
    version = service.db.create_data_resource_version(
        resource["id"], revision="raw", format="skynet.episodes/v1", path="/metadata/raw-manifest",
        manifest_sha256="d" * 64, metadata={"sources": [{"session_id": job["id"], "path": job["recordings"][0]}]},
    )
    service.db.record_data_location(version["id"], kind="cluster", host="sky2",
        path=job["archive"]["root"], manifest_sha256=version["manifest_sha256"])
    return resource, version


def test_unused_raw_provenance_is_removed_with_original_recording(recording):
    service, _, job, folder, _, _ = recording
    resource, version = raw_source(service, job)
    plan = service.preview("recording", job["id"])
    assert not plan["blockers"]
    assert plan["records"]["data_resource_versions"] == [version["id"]]
    assert plan["records"]["data_resources"] == [resource["id"]]
    assert service.delete("recording", job["id"], plan["token"])["deleted"]
    assert not folder.exists()
    assert service.db.get_data_resource_version(version["id"]) is None
    assert service.db.get_data_resource(resource["id"]) is None


@pytest.mark.parametrize("consumer", ["experiment", "bundle", "preparation"])
def test_raw_provenance_consumers_still_block_recording_removal(recording, consumer):
    service, _, job, folder, _, _ = recording
    _, version = raw_source(service, job)
    if consumer == "experiment":
        service.db.create_experiment(name="Pinned source", requested_spec={"version_id": version["id"]})
    elif consumer == "bundle":
        service.db.create_data_bundle(name="Pinned source", version="1",
            assignments=[{"role": "training_data", "version_id": version["id"]}])
    else:
        identifier = str(uuid4())
        with service.db.transaction() as c:
            c.execute("INSERT INTO policy_exports VALUES (?,?)", (identifier, canonical_json(
                dict(id=identifier, state="FAILED", source_version_id=version["id"], name="Failed conversion"))))
    plan = service.preview("recording", job["id"])
    assert plan["blockers"]
    assert not plan["files"]
    with pytest.raises(ValueError, match="dependencies"):
        service.delete("recording", job["id"], plan["token"])
    assert folder.exists()
    assert service.db.get_data_resource_version(version["id"])


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
            source_key="Race",
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


def test_retired_conversion_history_still_blocks_deleting_unfinished_work(recording):
    service, _, job, folder, _, root = recording
    identifier = str(uuid4())
    historical = dict(id=identifier, session_id=job['id'], state='RUNNING',
                      root=str(root / 'jobs/runs' / identifier))
    with service.db.transaction() as connection:
        connection.execute('INSERT INTO live_conversions VALUES (?,?)',
                           (identifier, canonical_json(historical)))
    plan = service.preview('recording', job['id'])
    assert any('conversion' in blocker['reason'] for blocker in plan['blockers'])
    assert folder.exists()
    with service.db.connection() as connection:
        assert connection.execute('SELECT payload_json FROM live_conversions WHERE id=?', (identifier,)).fetchone()


def observation(recording, *, source_path="recordings/one.pkl", label="rgb", inputs=(), producer_state=None):
    """Persist an immutable observation and real bytes for deletion tests."""
    from skynet_app.database import utc_now
    from skynet_app.observation_contracts import content_digest
    service, live, job, _, _, root = recording
    spec = dict(schema="skynet.observation-artifact/v1", source_sha256="a"*64,
                modality="point_cloud" if label == "cloud" else "depth" if label == "depth" else "rgb",
                camera_id="scene_front", label=label, dependencies=list(inputs))
    key = content_digest(spec)
    path = root / "datasets/recordings" / spec["source_sha256"] / spec["modality"] / spec["camera_id"] / key
    path.mkdir(parents=True)
    (path/"data.bin").write_bytes(b"immutable observation")
    now, producer = utc_now(), str(uuid4()) if producer_state else None
    with service.db.transaction() as c:
        if producer:
            c.execute("INSERT INTO observation_producers(id,attempt_token,state,payload_json,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                      (producer,str(uuid4()),producer_state,"{}",now,now))
        c.execute("INSERT INTO observation_artifacts(artifact_key,spec_json,state,producer_id,path,manifest_sha256,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                  (key,canonical_json(spec),"RUNNING" if producer else "READY",producer,str(path),"b"*64,now,now))
        c.execute("INSERT INTO observation_sources VALUES (?,?,?)",(key,job["id"],source_path))
        for parent in inputs:
            c.execute("INSERT INTO observation_artifact_inputs VALUES (?,?)",(key,parent))
    return key,path


def test_unreferenced_observations_delete_outputs_before_inputs(recording):
    service,live,job,folder,_,_ = recording
    depth,depth_path = observation(recording,label="depth")
    cloud,cloud_path = observation(recording,label="cloud",inputs=[depth])
    plan=service.preview("recording",job["id"])
    assert not plan["blockers"]
    assert plan["records"]["observation_artifacts"] == [cloud,depth]
    assert {str(depth_path),str(cloud_path)} <= {item["path"] for item in plan["files"]}
    service.delete("recording",job["id"],plan["token"])
    assert not folder.exists() and not depth_path.exists() and not cloud_path.exists()
    with service.db.connection() as c:
        assert not c.execute("SELECT * FROM observation_sources").fetchall()
        assert not c.execute("SELECT * FROM observation_artifacts").fetchall()


def test_shared_observation_keeps_bytes_and_other_source_alias(recording):
    import json
    service,live,job,folder,_,_ = recording
    key,path=observation(recording)
    other=dict(id=str(uuid4()), profile={"task_name":"Other"}, recordings=["recordings/copy.pkl"])
    with service.db.transaction() as c:
        c.execute("INSERT INTO live_xr_sessions VALUES (?,?)",(other["id"],canonical_json(other)))
        c.execute("INSERT INTO observation_sources VALUES (?,?,?)",(key,other["id"],"recordings/copy.pkl"))
        # A consumer is safe because another original with the same content remains.
        export=str(uuid4())
        c.execute("INSERT INTO policy_exports VALUES (?,?)",(export,canonical_json(dict(id=export,state="READY"))))
        c.execute("INSERT INTO observation_job_inputs VALUES (?,?)",(export,key))
    plan=service.preview("recording",job["id"])
    assert not plan["blockers"]
    assert "observation_artifacts" not in plan["records"]
    service.delete("recording",job["id"],plan["token"])
    assert path.exists() and not folder.exists()
    with service.db.connection() as c:
        assert [row[0] for row in c.execute("SELECT session_id FROM observation_sources WHERE artifact_key=?",(key,))] == [other["id"]]
        assert c.execute("SELECT 1 FROM observation_job_inputs WHERE artifact_key=?",(key,)).fetchone()


@pytest.mark.parametrize("consumer",["job","version","producer"])
def test_observation_consumers_block_before_original_file_deletion(recording,consumer):
    service,live,job,folder,_,_ = recording
    key,path=observation(recording,producer_state="RUNNING" if consumer=="producer" else None)
    if consumer=="job":
        with service.db.transaction() as c:
            export=str(uuid4())
            c.execute("INSERT INTO policy_exports VALUES (?,?)",(export,canonical_json(dict(id=export,state="RUNNING"))))
            c.execute("INSERT INTO observation_job_inputs VALUES (?,?)",(export,key))
    if consumer=="version":
        resource=service.db.create_data_resource(category="dataset",provider="test",namespace="test",source_key="consumer",kind="demonstrations")
        version=service.db.create_data_resource_version(resource["id"],revision="r",format="test",path="/dataset",manifest_sha256="c"*64)
        with service.db.transaction() as c:
            c.execute("INSERT INTO observation_version_inputs VALUES (?,?)",(version["id"],key))
    plan=service.preview("recording",job["id"])
    assert plan["blockers"] and not plan["files"]
    with pytest.raises(ValueError,match="dependencies"):
        service.delete("recording",job["id"],plan["token"])
    assert folder.exists() and path.exists()


def test_new_observation_consumer_invalidates_deletion_preview(recording):
    service,live,job,folder,_,_ = recording
    key,path=observation(recording)
    plan=service.preview("recording",job["id"])
    with service.db.transaction() as c:
        export=str(uuid4())
        c.execute("INSERT INTO policy_exports VALUES (?,?)",(export,canonical_json(dict(id=export,state="QUEUED"))))
        c.execute("INSERT INTO observation_job_inputs VALUES (?,?)",(export,key))
    with pytest.raises(ValueError,match="dependencies"):
        service.delete("recording",job["id"],plan["token"])
    assert folder.exists() and path.exists()


def observation_producer(recording, keys, *, state):
    from skynet_app.database import utc_now
    service, _, job, _, _, root = recording
    identifier, token, now = str(uuid4()), str(uuid4()), utc_now()
    capsule = root / 'jobs/runs' / identifier
    payload = dict(id=identifier, attempt_token=token, root=str(capsule / 'observations' / token),
                   request=dict(requests=[dict(artifact_key=key) for key in keys],
                                sources=[dict(session_id=job['id'], path='recordings/one.pkl')],
                                worker_files={'worker.py': 'frozen worker source'}))
    capsule.mkdir(parents=True)
    (capsule / 'request.json').write_text(canonical_json(payload))
    with service.db.transaction() as connection:
        connection.execute('INSERT INTO observation_producers(id,attempt_token,state,payload_json,created_at,updated_at) VALUES (?,?,?,?,?,?)',
                           (identifier, token, state, canonical_json(payload), now, now))
        for key in keys:
            connection.execute('UPDATE observation_artifacts SET producer_id=? WHERE artifact_key=?', (identifier, key))
    return identifier, capsule


@pytest.mark.parametrize('state', ['READY', 'FAILED'])
def test_exclusive_observation_cleanup_removes_terminal_producer_capsules_and_rows(recording, state):
    service, _, job, _, _, _ = recording
    depth, depth_path = observation(recording, label='depth')
    cloud, cloud_path = observation(recording, label='cloud', inputs=[depth])
    locks = [path.with_name('.' + path.name + '.publish.lock') for path in (depth_path, cloud_path)]
    for lock in locks:
        lock.write_bytes(b'')
    # A previous failed attempt has the same requested artifacts but no current
    # ownership FKs after retry; its historical capsule must be cleaned too.
    previous, previous_capsule = observation_producer(recording, [depth, cloud], state='FAILED')
    producer, capsule = observation_producer(recording, [depth, cloud], state=state)
    plan = service.preview('recording', job['id'])
    assert not plan['blockers']
    assert set(plan['records']['observation_producers']) == {producer, previous}
    assert {str(capsule), str(previous_capsule)} <= {item['path'] for item in plan['files']}
    service.delete('recording', job['id'], plan['token'])
    assert not capsule.exists() and not previous_capsule.exists()
    assert not depth_path.exists() and not cloud_path.exists()
    assert all(not lock.exists() for lock in locks)
    with service.db.connection() as connection:
        assert not connection.execute('SELECT * FROM observation_artifacts').fetchall()
        assert not connection.execute('SELECT * FROM observation_producers').fetchall()


@pytest.mark.parametrize('state', ['READY', 'FAILED', 'RUNNING'])
def test_retained_recording_artifact_keeps_its_entire_producer(recording, state):
    service, _, job, folder, _, _ = recording
    exclusive, exclusive_path = observation(recording, label='first-view')
    shared, shared_path = observation(recording, label='other-view')
    exclusive_lock = exclusive_path.with_name('.' + exclusive_path.name + '.publish.lock')
    shared_lock = shared_path.with_name('.' + shared_path.name + '.publish.lock')
    exclusive_lock.write_bytes(b'')
    shared_lock.write_bytes(b'')
    producer, capsule = observation_producer(recording, [exclusive, shared], state=state)
    other_id = str(uuid4())
    with service.db.transaction() as connection:
        connection.execute('INSERT INTO live_xr_sessions VALUES (?,?)',
                           (other_id, canonical_json(dict(id=other_id, recordings=['recordings/copy.pkl']))))
        connection.execute('INSERT INTO observation_sources VALUES (?,?,?)', (shared, other_id, 'recordings/copy.pkl'))
    plan = service.preview('recording', job['id'])
    assert 'observation_producers' not in plan['records']
    if state == 'RUNNING':
        assert plan['blockers'] and not plan['files']
        with pytest.raises(ValueError, match='dependencies'):
            service.delete('recording', job['id'], plan['token'])
        assert exclusive_path.exists() and exclusive_lock.exists() and folder.exists()
    else:
        assert not plan['blockers']
        assert str(capsule) not in {item['path'] for item in plan['files']}
        service.delete('recording', job['id'], plan['token'])
        assert not exclusive_path.exists() and not exclusive_lock.exists() and not folder.exists()
    assert capsule.exists() and shared_path.exists() and shared_lock.exists()
    with service.db.connection() as connection:
        assert connection.execute('SELECT state FROM observation_producers WHERE id=?', (producer,)).fetchone()[0] == state
        assert connection.execute('SELECT producer_id FROM observation_artifacts WHERE artifact_key=?', (shared,)).fetchone()[0] == producer
        assert connection.execute('SELECT 1 FROM observation_sources WHERE session_id=? AND artifact_key=?', (other_id, shared)).fetchone()
