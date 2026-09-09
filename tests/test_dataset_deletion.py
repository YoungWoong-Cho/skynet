"""Deletion exercises generated files, registry references and recoverable failures."""

import shlex
import shutil
import sqlite3
import subprocess

import pytest
from test_dataset_preparation import prepared
from test_policy_exports import (
    setup as setup,  # noqa: PLC0414 (pytest fixture re-export)
)

from skynet_app import dataset_cleanup, policy_exports
from skynet_app.cluster_runtime import ClusterError


def cluster_copy(service, job, tmp_path, monkeypatch):
    root = tmp_path / "cluster"
    monkeypatch.setattr(policy_exports, "WORK_ROOT", str(root))
    version = service.database.get_data_resource_version(job["version_id"])
    destination = root / "datasets/prepared" / version["manifest_sha256"]
    shutil.copytree(service.root / job["id"] / "output", destination)
    capsule = root / "jobs/runs" / job["id"]
    capsule.mkdir(parents=True)
    (capsule / "dataset.zip").write_bytes(b"generated archive")
    location = service.database.record_data_location(
        version["id"],
        kind="cluster",
        host="skynet",
        path=str(destination),
        manifest_sha256=version["manifest_sha256"],
    )
    bundle = service.bundle(job, location)
    service.update(
        job["id"], target="cluster", bundle_id=bundle["id"], training_ready=True
    )

    class Cluster:
        def candidates(self, gateway):
            return ["test"]

        def resolve_gateway(self, gateway):
            return gateway

        def ssh(self, gateway, command, timeout):
            result = subprocess.run(
                shlex.split(command), capture_output=True, text=True, check=True
            )
            return result.stdout

    service.cluster = Cluster()
    return destination, capsule, bundle


def test_complete_delete_preserves_recordings_and_does_not_reappear(
    setup, tmp_path, monkeypatch
):
    service, session, source, job = prepared(setup)
    before = {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    destination, capsule, bundle = cluster_copy(service, job, tmp_path, monkeypatch)
    assert service.delete_dataset(job["resource_id"])["deleted"]
    assert not destination.exists() and not capsule.exists()
    assert not (service.root / job["id"]).exists()
    assert service.database.get_data_resource(job["resource_id"]) is None
    assert service.database.get_data_bundle(bundle["id"]) is None
    assert service.list() == []
    assert service.options()["sessions"][0]["resource_id"] is None
    assert {str(p): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    new = service.create(session["id"], "dp", "Prepared again")
    assert new["resource_id"] != job["resource_id"]
    # Immutable registry operations still reject direct deletion.
    with (
        service.database.transaction() as c,
        pytest.raises(sqlite3.IntegrityError, match="immutable"),
    ):
        c.execute("DELETE FROM data_resource_versions")


@pytest.mark.parametrize("resolved", [False, True])
def test_used_dataset_is_blocked_before_removing_any_files(
    setup, tmp_path, monkeypatch, resolved
):
    service, _, _, job = prepared(setup)
    destination, capsule, bundle = cluster_copy(service, job, tmp_path, monkeypatch)
    db = service.database
    project = db.create_project("test")
    spec = {"data": {"bundle": db.data_bundle_snapshot(bundle["id"])}}
    experiment = db.create_experiment(
        project_id=project["id"],
        name="uses data",
        requested_spec={} if resolved else spec,
    )
    if resolved:
        db.create_variant(
            experiment["latest_revision"]["id"],
            name="resolved",
            parameters={},
            resolved_spec={
                "native": {"path": str(destination / "dataset/demonstrations.zarr")}
            },
        )
    with pytest.raises(ValueError, match="used by an experiment"):
        service.delete_dataset(job["resource_id"])
    assert destination.exists() and capsule.exists()
    assert service.artifact(job["id"], "dataset.zip").exists()


def test_cluster_failure_retains_registry_for_delete_retry(
    setup, tmp_path, monkeypatch
):
    service, _, _, job = prepared(setup)
    destination, _, bundle = cluster_copy(service, job, tmp_path, monkeypatch)
    good = service.cluster.ssh

    def fail(*args, **kwargs):
        raise ClusterError("offline")

    monkeypatch.setattr(service.cluster, "ssh", fail)
    with pytest.raises(ValueError, match="Retry Delete dataset"):
        service.delete_dataset(job["resource_id"])
    assert service.get(job["id"])["state"] == "DELETE_FAILED"
    assert service.options()["exports"][0]["training_ready"] is False
    assert service.database.get_data_bundle(bundle["id"])["archived_at"]
    with pytest.raises(ValueError, match="deletion"):
        service.retry(job["id"])
    monkeypatch.setattr(service.cluster, "ssh", good)
    assert service.delete_dataset(job["resource_id"])["deleted"]
    assert not destination.exists()


def test_manual_bundle_blocks_deletion(setup):
    service, _, _, job = prepared(setup)
    db = service.database
    db.create_data_bundle(
        name="manual",
        version="1",
        assignments=[{"role": "training_data", "version_id": job["version_id"]}],
    )
    with pytest.raises(ValueError, match="bundle"):
        service.delete_dataset(job["resource_id"])
    assert service.artifact(job["id"], "dataset.zip").exists()


def test_cleanup_rejects_redirected_paths_and_keeps_outside_files(tmp_path):
    root = tmp_path / "exports"
    root.mkdir()
    outside = tmp_path / "recordings"
    outside.mkdir()
    (outside / "original.pkl").write_bytes(b"original")
    identifier = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    (root / identifier).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        dataset_cleanup.cleanup(root, jobs=[identifier])
    assert (outside / "original.pkl").read_bytes() == b"original"
    with pytest.raises(ValueError):
        dataset_cleanup.cleanup(root, jobs=["../recordings"])


def test_delete_unused_format_keeps_training_dataset(setup, tmp_path, monkeypatch):
    service, session, source, dp = prepared(setup)
    destination, _, bundle = cluster_copy(service, dp, tmp_path, monkeypatch)
    db = service.database
    project = db.create_project('uses-dp')
    db.create_experiment(project_id=project['id'], name='keep-training', requested_spec={'data': {'bundle': db.data_bundle_snapshot(bundle['id'])}})
    act = service.create(session['id'], 'act', 'Hand demonstrations')
    service.prepare(act['id'])
    act = service.get(act['id'])
    assert act['state'] == 'READY'
    assert service.delete_dataset(act['resource_id'], act['id'])['deleted']
    assert not (service.root / act['id']).exists()
    assert db.get_data_resource_version(act['version_id']) is None
    assert destination.exists() and db.get_data_bundle(bundle['id'])
    assert {v['id'] for v in db.get_data_resource(dp['resource_id'])['versions']} == {dp['version_id'], dp['source_version_id']}
    assert len(list(source.rglob('*.pkl'))) == 2
    with pytest.raises(ValueError, match='used by an experiment'):
        service.delete_dataset(dp['resource_id'], dp['id'])
