"""Regression coverage for preparation lifecycle, lineage and training handoff."""

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from test_policy_exports import setup as setup
from skynet_app.adapters import builtin_adapter_manifests
from skynet_app.pipeline_api import PipelineService
from skynet_app.cluster_runtime import ClusterError
from ops.datasets.artifacts import materialize, digest


def prepared(setup, kind="dp"):
    service, session, source = setup
    job = service.create(session["id"], kind, "Hand demonstrations")
    service.prepare(job["id"])
    job = service.get(job["id"])
    assert job["state"] == "READY", job.get("error")
    return service, session, source, job


def test_formats_share_dataset_revision_split_and_source_cache(setup):
    service, session, source, first = prepared(setup)
    cache = {p.name: p.stat().st_mtime_ns for p in (service.root / "sources").iterdir()}
    second = service.create(
        session["id"], "act", "Hand demonstrations", first["resource_id"]
    )
    service.prepare(second["id"])
    second = service.get(second["id"])
    assert second["state"] == "READY", second.get("error")
    assert second["source_version_id"] == first["source_version_id"]
    assert second["resource_id"] == first["resource_id"]
    assert second["split"] == first["split"]
    assert cache == {
        p.name: p.stat().st_mtime_ns for p in (service.root / "sources").iterdir()
    }
    assert (
        len(service.database.get_data_resource(first["resource_id"])["versions"]) == 3
    )
    overview = service.options()
    assert len(overview["exports"]) == 2


def test_normalization_excludes_validation_episodes(setup):
    service, _, _, job = prepared(setup)
    import zarr

    root = service.root / job["id"] / "output"
    data = zarr.open_group(str(root / "dataset/demonstrations.zarr"), mode="r")
    normalizer = json.loads((root / "normalization.json").read_text())
    ends = data["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    for key in ("state", "action"):
        values = np.concatenate(
            [data["data/" + key][starts[i] : ends[i]] for i in job["split"]["train"]]
        )
        np.testing.assert_allclose(
            normalizer["statistics"][key]["mean"], values.mean(axis=0)
        )
        assert not np.allclose(
            normalizer["statistics"][key]["mean"], data["data/" + key][:].mean(axis=0)
        )


def test_transfer_keeps_content_version_immutable_and_binds_verified_location(
    setup, tmp_path
):
    service, _, _, job = prepared(setup)
    db = service.database
    before = db.get_data_resource_version(job["version_id"])
    location = db.record_data_location(
        before["id"],
        kind="cluster",
        host="skynet",
        path="/cluster/prepared/dataset",
        manifest_sha256=before["manifest_sha256"],
    )
    again = db.record_data_location(
        before["id"],
        kind="cluster",
        host="skynet",
        path=location["path"],
        manifest_sha256=before["manifest_sha256"],
    )
    assert again["id"] == location["id"]
    bundle = service.bundle(job, location)
    assert bundle["assignments"][0]["config"]["location"]["path"] == location["path"]
    canonical = {"data": {"bundle": db.data_bundle_snapshot(bundle["id"])}}
    adapter = next(m for m in builtin_adapter_manifests() if m.slug == "xpolicylab-dp")
    bound = PipelineService._apply_manifest_data_bindings(
        copy.deepcopy(canonical), adapter
    )
    assert bound["native"]["config"]["dataset_path"] == location["path"]
    assert (
        bound["native"]["config"]["dataset_manifest_sha256"]
        == before["manifest_sha256"]
    )
    after = db.get_data_resource_version(before["id"])
    assert after["path"] == before["path"] and after["status"] == "LOCAL"
    assert after["used_by_bundles"][0]["id"] == bundle["id"]
    invalid = copy.deepcopy(canonical)
    invalid["data"]["bundle"]["assignments"][0]["version"]["metadata"]["contract"] = (
        "another-policy"
    )
    with pytest.raises(ValueError, match="data contract"):
        PipelineService._apply_manifest_data_bindings(invalid, adapter)
    invalid = copy.deepcopy(canonical)
    invalid["data"]["bundle"]["assignments"][0]["config"]["location"]["status"] = (
        "REMOVED"
    )
    with pytest.raises(ValueError, match="verified training-cluster"):
        PipelineService._apply_manifest_data_bindings(invalid, adapter)


def test_bundle_rejects_forged_location_and_only_copy_cannot_be_removed(setup):
    service, _, _, job = prepared(setup)
    with pytest.raises(ValueError, match="location receipt"):
        service.database.create_data_bundle(
            name="unsafe",
            version="1",
            assignments=[
                dict(
                    role="training_data",
                    version_id=job["version_id"],
                    config={"location": {"status": "AVAILABLE"}},
                )
            ],
        )
    with pytest.raises(ValueError, match="at least one verified"):
        service.remove_local_copy(job["id"])


def test_archive_prevents_new_preparation_without_deleting_files(setup):
    service, session, _, job = prepared(setup)
    service.database.update_data_resource(job["resource_id"], archived=True)
    with pytest.raises(ValueError, match="active collection dataset"):
        service.create(session["id"], "act", "Archived", job["resource_id"])
    with pytest.raises(ValueError, match="Restore"):
        service.retry(job["id"])
    assert service.artifact(job["id"], "dataset.zip").exists()
    assert not service.options()["sessions"][0]["eligible"]


@pytest.mark.parametrize("outcome", ["success", "timeout", "pinned"])
def test_local_copy_verification_keeps_catalog_responsive_and_copies_safe(
    setup, monkeypatch, outcome
):
    service, session, _, job = prepared(setup)
    version = service.database.get_data_resource_version(job["version_id"])
    service.database.record_data_location(
        version["id"], kind="cluster", host="skynet", path="/cluster/prepared",
        manifest_sha256=version["manifest_sha256"],
    )
    service.update(job["id"], gateway="sky2")
    started, release = threading.Event(), threading.Event()
    usage = []
    monkeypatch.setattr(service.database, "data_version_usage", lambda _: usage)

    def verify(command, *, gateway, timeout):
        assert gateway == "sky2" and timeout == 60
        assert "verify-dataset.py" in command
        started.set()
        assert release.wait(5)
        if outcome == "timeout":
            raise ClusterError("Verification timed out")
        return gateway, "verified"

    service.cluster = SimpleNamespace(run_with_fallback=verify)
    with ThreadPoolExecutor(max_workers=2) as pool:
        removal = pool.submit(service.remove_local_copy, job["id"])
        try:
            assert started.wait(3)
            overview = pool.submit(service.options).result(timeout=2)
            assert overview["sessions"][0]["eligible"]
            assert overview["sessions"][0]["id"] == session["id"]
            with pytest.raises(ValueError, match="preparation to finish"):
                service.delete_dataset(job["resource_id"])
            assert service.artifact(job["id"], "dataset.zip").is_file()
            if outcome == "pinned":
                usage.append({"experiment_id": "new-experiment"})
        finally:
            release.set()
        if outcome == "success":
            assert removal.result(timeout=3)["local_removed"]
        else:
            with pytest.raises((ClusterError, ValueError)):
                removal.result(timeout=3)
    assert job["id"] not in service.active
    assert (service.root / job["id"] / "dataset.zip").exists() == (outcome != "success")
    locations = service.database.get_data_resource_version(version["id"])["locations"]
    assert next(l for l in locations if l["kind"] == "cluster")["status"] == "AVAILABLE"


def test_catalog_reads_do_not_wait_for_dataset_mutations(setup):
    service, _, _, _ = prepared(setup)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with service.lock:
            pending = pool.submit(service.options)
            # The catalog reads committed registry snapshots; deletion holds this
            # mutation lock while cleaning remote files and must not hide the form.
            assert pending.result(timeout=2)["sessions"][0]["eligible"]


def test_verified_archive_is_atomic_and_content_addressed(setup, tmp_path):
    service, _, _, job = prepared(setup)
    archive = service.artifact(job["id"], "dataset.zip")
    destination = tmp_path / "cluster/dataset"
    receipt = materialize(archive, destination, job["manifest_sha256"], digest(archive))
    assert receipt["split"] == job["split"]
    materialize(archive, destination, job["manifest_sha256"], digest(archive))
    (destination / "normalization.json").write_text("corrupted")
    with pytest.raises(ValueError, match="failed verification"):
        materialize(archive, destination, job["manifest_sha256"], digest(archive))
    assert not list(destination.parent.glob(".preparing-*"))


def test_failed_transfer_retries_verified_conversion_without_rewriting(
    setup, monkeypatch
):
    service, session, _, job = prepared(setup)
    root = service.root / job["id"] / "output"
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        service,
        "transfer",
        lambda _: (_ for _ in ()).throw(ClusterError("gateway unavailable")),
    )
    service.retry(job["id"], "cluster")
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "FAILED"
    assert service.artifact(job["id"], "dataset.zip").is_file()

    def copied(j):
        return service.database.record_data_location(
            j["version_id"],
            kind="cluster",
            host="skynet",
            path="/cluster/exact",
            manifest_sha256=j["manifest_sha256"],
        )

    monkeypatch.setattr(service, "transfer", copied)
    service.retry(job["id"])
    service.prepare(job["id"])
    finished = service.get(job["id"])
    assert finished["training_ready"] and finished["bundle_id"]
    assert before == {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


def test_selection_rejects_duplicates_and_dp_without_validation(setup):
    service, session, _ = setup
    with pytest.raises(ValueError, match="unique recording"):
        service.create(
            None,
            "dp",
            "Duplicate",
            selections=[dict(session_id=session["id"], indices=[0, 0])],
        )
    with pytest.raises(ValueError, match="at least two"):
        service.create(
            None,
            "dp",
            "Too small",
            selections=[dict(session_id=session["id"], indices=[0])],
        )
    with pytest.raises(ValueError, match="non-zero"):
        service.create(session["id"], "dp", "No validation", validation_percent=0)


def test_other_session_resource_is_rejected(setup):
    service, session, _ = setup
    resource = service.database.create_data_resource(
        provider="collection",
        namespace="sessions",
        name="another",
        kind="demonstrations",
        metadata={"session_id": "another"},
    )
    with pytest.raises(ValueError, match="belonging to the selected session"):
        service.create(session["id"], "dp", "Invalid resource", resource["id"])


def test_archived_dp_adapter_leaves_export_available(setup):
    from skynet_app.dataset_formats import catalog

    service, _, _ = setup
    entry = next(p for p in catalog(service.database) if p["id"] == "dp")
    assert entry["available"]
    assert not entry["trainable"]
    assert "training_setup" not in entry


def test_declarative_dp_plan_includes_frozen_training_files():
    from test_experiments import make_spec
    from skynet_app.adapters import ManifestAdapter
    from skynet_app.dataset_formats import XPL_REPOSITORY, XPL_COMMIT

    manifest = next(m for m in builtin_adapter_manifests() if m.slug == "xpolicylab-dp")
    spec = make_spec(
        source={
            "repository": XPL_REPOSITORY,
            "revision": XPL_COMMIT,
            "adapter": "xpolicylab-dp",
        },
        resources={
            "account": "rl2-lab",
            "partition": "rl2-lab",
            "gpu": {"mode": "explicit", "count": 1, "type": "l40s"},
        },
        train={
            "checkpoint":{"auto_resume":False},
            "batch": {"value": 8, "declared_semantics": "per_device"},
            "learning_rate": 0.0001,
            "seed": 42,
        },
        native={
            "config": {
                "dataset_path": "/cluster/prepared/example",
                "dataset_manifest_sha256": "a" * 64,
                "epochs": 1,
            }
        },
    )
    document = spec.model_dump(mode="python")
    PipelineService._apply_training_preset(document, manifest)
    PipelineService._apply_manifest_input_defaults(document, manifest)
    spec = type(spec).model_validate(document)
    plan = ManifestAdapter(manifest).resolve(spec)
    assert not plan.blockers
    assert plan.capsule_files == manifest.train.capsule_files
    assert "def main():" in plan.capsule_files["adapter-support/skynet_dp_training.py"]
