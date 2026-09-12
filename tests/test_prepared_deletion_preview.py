from types import SimpleNamespace

import pytest

from test_dataset_preparation import prepared
from test_policy_exports import setup as setup
from skynet_app import prepared_deletion, data_selection


def test_preview_is_read_only_and_delete_rechecks_new_dependencies(setup):
    service, _, source, job = prepared(setup)
    workspace = SimpleNamespace(owns=lambda *args: True)
    plan = prepared_deletion.preview(service, workspace, "dataset", job["resource_id"])
    assert plan["blockers"] == []
    assert (service.root / job["id"] / "output").exists()
    assert str(service.root / job["id"]) in {f["path"] for f in plan["files"]}
    db = service.database
    project = db.create_project("preview")
    spec = {"data": {"bundle": data_selection.snapshot(db, [{"version_id": job["version_id"]}])}}
    db.create_experiment(project_id=project["id"], name="Uses this data", requested_spec=spec)
    revised = prepared_deletion.preview(service, workspace, "dataset", job["resource_id"])
    assert any(b["label"] == "Uses this data" for b in revised["blockers"])
    with pytest.raises(ValueError, match="dependencies changed"):
        prepared_deletion.delete(service, workspace, "dataset", job["resource_id"], plan["token"])
    assert source.exists() and (service.root / job["id"] / "output").exists()


def test_shared_preview_token_deletes_only_the_prepared_result(setup):
    service, _, source, job = prepared(setup)
    workspace = SimpleNamespace(owns=lambda *args: True)
    plan = prepared_deletion.preview(service, workspace, "prepared", job["id"])
    assert not plan["blockers"]
    assert prepared_deletion.delete(service, workspace, "prepared", job["id"], plan["token"])["deleted"]
    assert source.exists()
    assert service.database.get_data_resource_version(job["version_id"]) is None


def test_local_copy_shared_preview_requires_verified_cluster_copy(setup, monkeypatch):
    service, _, source, job = prepared(setup)
    workspace = SimpleNamespace(owns=lambda *args: True)
    blocked = prepared_deletion.preview(service, workspace, "local-copy", job["id"])
    assert any("cluster copy" in item["reason"] for item in blocked["blockers"])
    version = service.database.get_data_resource_version(job["version_id"])
    service.database.record_data_location(version["id"], kind="cluster", host="skynet", path="/cluster/prepared",
                                          manifest_sha256=version["manifest_sha256"])
    plan = prepared_deletion.preview(service, workspace, "local-copy", job["id"])
    assert not plan["blockers"]
    assert all(str(service.root) in file["path"] for file in plan["files"])
    verified = []
    service.cluster = SimpleNamespace(run_with_fallback=lambda *args, **kwargs: verified.append(args[0]))
    with pytest.raises(ValueError, match="dependencies changed"):
        prepared_deletion.delete(service, workspace, "local-copy", job["id"], blocked["token"])
    prepared_deletion.delete(service, workspace, "local-copy", job["id"], plan["token"])
    assert len(verified) == 1 and "verify-dataset.py" in verified[0]
    assert source.exists() and service.database.get_data_resource_version(version["id"])
    assert not (service.root / job["id"] / "output").exists()
