import hashlib
import json
from types import SimpleNamespace

import pytest

from skynet_app.cluster_runtime import WORK_ROOT
from skynet_app.database import canonical_json
from test_live_conversion import LocalCluster, conversion, metadata


@pytest.fixture
def archived_conversion(conversion, tmp_path, monkeypatch):
    service, session, calls = conversion
    cluster = LocalCluster(tmp_path / "cluster")
    archive_root = WORK_ROOT + "/datasets/raw/dexverse-live/session/archive/output"
    for index, relative in enumerate(session["recordings"]):
        raw = f"saved recording {index}".encode()
        cluster.write(archive_root + "/" + relative, raw)
        session["recording_checksums"][relative] = hashlib.sha256(raw).hexdigest()
    session["archive"] = {"state": "VERIFIED", "root": archive_root}
    ensured = []
    service.live.archive = SimpleNamespace(
        ensure=ensured.append,
        resolve=lambda job, relative: (cluster, "sky2", archive_root + "/" + relative),
    )
    service.cluster = cluster

    def local_payload_forbidden(*args, **kwargs):
        pytest.fail("Conversion must not request a Mac recording copy")

    monkeypatch.setattr(service.reviews, "create", local_payload_forbidden)
    monkeypatch.setattr(service.reviews, "artifact", local_payload_forbidden)
    return service, session, cluster, ensured


def test_archive_location_does_not_change_conversion_identity(archived_conversion):
    service, session, _, _ = archived_conversion
    session.pop("archive")
    before = service.create("session", "Cube")
    session["archive"] = {"state": "READY", "root": "/new/storage/output"}
    after = service.create("session", "Cube")
    assert after["id"] == before["id"]
    assert after["sources"] == before["sources"]
    assert after["sources"][0]["path"].startswith(session["root"] + "/output/")


def test_unarchived_conversion_waits_without_submitting_or_caching(archived_conversion):
    service, session, cluster, ensured = archived_conversion
    session["archive"] = {"state": "COPYING"}
    job = service.create("session", "Cube")
    assert service.launch(job, cluster) is False
    assert ensured == ["session"]
    assert cluster.submissions == []
    assert service.get(job["id"])["state"] == "PREPARING"
    assert "finish moving" in service.get(job["id"])["detail"]
    assert not service.reviews.root.exists()


def test_archived_raw_is_verified_and_staged_only_on_cluster(archived_conversion):
    service, session, cluster, _ = archived_conversion
    job = service.create("session", "Cube")
    assert service.launch(job, cluster) is True
    assert cluster.submissions == [job["id"]]
    for item in job["sources"]:
        staged = cluster.path(f"{job['root']}/recordings/{item['index']}.pkl")
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == item["sha256"]
    request = json.loads(cluster.path(job["root"] + "/request.json").read_text())
    assert request["sources"] == job["sources"]
    assert service.get(job["id"])["staged"] is True
    assert not service.reviews.root.exists()
    assert not list(service.root.rglob("*.pkl"))


@pytest.mark.parametrize("corrupt", ["archive", "staged"])
def test_changed_cluster_recording_never_submits_or_overwrites(archived_conversion, corrupt):
    service, session, cluster, _ = archived_conversion
    job = service.create("session", "Cube")
    target = session["archive"]["root"] + "/" + session["recordings"][0]
    if corrupt == "staged":
        target = job["root"] + "/recordings/0.pkl"
    cluster.write(target, b"changed bytes")
    with pytest.raises(ValueError, match="checksum changed"):
        service.launch(job, cluster)
    assert cluster.submissions == []
    assert cluster.path(target).read_bytes() == b"changed bytes"
    assert not list(cluster.root.rglob(".stage-*"))


def prepared_result(service, cluster):
    job = service.create("session", "Cube")
    raw = b"\x89HDF\r\n\x1a\n" + b"complete dataset"
    value = metadata(job, raw)
    manifest = canonical_json(value).encode()
    files = {"manifest.json": manifest, "dataset.hdf5": raw,
             "source-manifest.json": canonical_json(job["sources"]).encode()}
    for name, data in files.items():
        cluster.write(job["dataset_root"] + "/" + name, data)
    result = {"metadata": value, "manifest": {
        "sha256": hashlib.sha256(manifest).hexdigest(), "size_bytes": len(manifest),
    }}
    return job, result, files


@pytest.mark.parametrize("name", ["dataset.hdf5", "manifest.json", "source-manifest.json"])
def test_remote_corruption_blocks_registration_and_leaves_local_payload_absent(archived_conversion, name):
    service, _, cluster, _ = archived_conversion
    job, result, files = prepared_result(service, cluster)
    data = files[name]
    cluster.write(job["dataset_root"] + "/" + name, b"x" * len(data))
    with pytest.raises(ValueError, match="checksum changed"):
        service.finish(job, result, cluster)
    assert not service.database.list_data_resources()
    assert not (service.root / job["id"] / "dataset.hdf5").exists()


def test_legacy_local_cleanup_verifies_cluster_and_preserves_capsules(archived_conversion):
    service, _, cluster, _ = archived_conversion
    job, result, files = prepared_result(service, cluster)
    service.finish(job, result, cluster)
    directory = service.root / job["id"]
    for name in ("dataset.hdf5", "manifest.json"):
        (directory / name).write_bytes(files[name])
    # Old jobs do not yet persist the result's manifest receipt.
    service.update(job["id"], manifest=None)
    service.remove_local_copies(job["id"])
    service.remove_local_copies(job["id"])
    assert not (directory / "dataset.hdf5").exists()
    assert not (directory / "manifest.json").exists()
    assert (directory / "request.json").is_file()
    assert (directory / "worker-sources.json").is_file()
    assert cluster.path(job["dataset_root"] + "/dataset.hdf5").read_bytes() == files["dataset.hdf5"]
    assert service.get(job["id"])["state"] == "READY"


@pytest.mark.parametrize("corrupt", ["remote", "local"])
def test_cleanup_preserves_every_local_file_when_any_copy_differs(archived_conversion, corrupt):
    service, _, cluster, _ = archived_conversion
    job, result, files = prepared_result(service, cluster)
    service.finish(job, result, cluster)
    directory = service.root / job["id"]
    for name in ("dataset.hdf5", "manifest.json"):
        (directory / name).write_bytes(files[name])
    if corrupt == "remote":
        cluster.write(job["dataset_root"] + "/dataset.hdf5", b"changed")
    else:
        (directory / "dataset.hdf5").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed|differs"):
        service.remove_local_copies(job["id"])
    assert (directory / "dataset.hdf5").is_file()
    assert (directory / "manifest.json").is_file()


def test_remote_artifact_access_does_not_restart_or_recreate_local_copy(archived_conversion):
    service, _, cluster, _ = archived_conversion
    job, result, _ = prepared_result(service, cluster)
    service.finish(job, result, cluster)
    before = service.get(job["id"])
    artifact = service.artifact(job["id"], "dataset.hdf5")
    assert artifact.transport is cluster
    assert artifact.path == job["dataset_root"] + "/dataset.hdf5"
    assert service.get(job["id"]) == before
    assert not (service.root / job["id"] / "dataset.hdf5").exists()
