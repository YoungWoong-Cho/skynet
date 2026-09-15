"""Catalog reads stay bounded, fresh and equivalent to immutable input receipts."""
import hashlib

import pytest

from skynet_app import data_selection
from skynet_app.database import Database, canonical_json, content_sha256
from skynet_app.db_backend import PostgresConnection
from skynet_app.workspaces import WorkspaceDirectory


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "catalog")


@pytest.fixture
def reads(monkeypatch):
    statements = []
    original = PostgresConnection.execute
    def execute(connection, statement, parameters=None):
        if statement.lstrip().startswith(("SELECT", "WITH")):
            statements.append(statement)
        return original(connection, statement, parameters)
    monkeypatch.setattr(PostgresConnection, "execute", execute)
    return statements


def dataset(db, index=0):
    resource = db.create_data_resource(
        category="dataset", provider="collection", namespace="datasets", source_key=f"Data {index}",
        kind="demonstrations", display_name=f"Label {index}", metadata={"session_id": f"recording-{index}"},
    )
    digest = hashlib.sha256(str(index).encode()).hexdigest()
    prepared = db.create_data_resource_version(
        resource["id"], revision="prepared", format="zarr", path=f"/cluster/data/{index}",
        manifest_sha256=digest, status="ON_CLUSTER", size_bytes=123,
        source_uri="collection:source", metadata={"episodes": 3, "nested": {"key": [1, 2]}},
    )
    locations = [db.record_data_location(
        prepared["id"], kind="cluster", host=host, path=f"/cluster/data/{index}", manifest_sha256=digest,
    ) for host in ("sky2", "sky1")]
    source = db.create_data_resource_version(
        resource["id"], revision="source", format="skynet.episodes/v1", path=f"/cluster/source/{index}",
        manifest_sha256="f" * 64, status="RECORDED", metadata={"sources": [{"session_id": f"source-{index}"}]},
    )
    return resource, prepared, locations, source


@pytest.mark.parametrize("count", [1, 9])
def test_catalog_and_choices_use_three_reads_regardless_of_size(db, reads, count):
    records = [dataset(db, i) for i in range(count)]
    reads.clear()
    resources = db.list_data_resources(include_versions=True)
    assert len(reads) == 3
    assert len(resources) == count
    assert all(resource["version_count"] == 2 for resource in resources)
    assert all(resource["latest_version"] == resource["versions"][0] for resource in resources)
    assert [loc["host"] for loc in resources[0]["versions"][1]["locations"]] == ["sky1", "sky2"]
    assert set(resources[0]["recording_ids"]) == {"recording-0", "source-0"}
    reads.clear()
    choices = data_selection.choices(db)
    assert len(reads) == 3
    assert len(choices) == count
    assert {item["id"] for item in choices} == {item[1]["id"] for item in records}
    reads.clear()
    defaults = db.list_data_resources()
    assert len(reads) == 3
    assert defaults == [{key: value for key, value in row.items() if key != "versions"} for row in resources]


def test_choices_keep_exact_frozen_receipt_and_location_selection(db):
    resource, version, locations, _ = dataset(db)
    location = min(locations, key=lambda loc: loc["id"])
    expected_assignment = {
        "role": "training_data", "position": 0, "mount_path": None, "required": True,
        "config": {"location_id": location["id"], "location": location},
        "resource": {**{key: resource[key] for key in ("provider", "namespace", "kind")}, "name": resource["source_key"]},
        "version": {key: version[key] for key in (
            "revision", "format", "path", "source_uri", "manifest_sha256", "status", "size_bytes", "metadata",
        )},
    }
    expected_assignment["version"]["metadata"] = {**version["metadata"], "registered_version_id": version["id"]}
    expected = {
        "schema_version": "skynet.data-bundle/v1", "name": resource["source_key"],
        "version": "experiment-inputs", "metadata": {"direct_selection": True},
        "assignments": [expected_assignment],
    }
    choice = data_selection.choices(db)[0]
    assert choice["assignments"] == [expected_assignment]
    assert choice["manifest_sha256"] == content_sha256(expected)
    frozen = data_selection.snapshot(db, [choice["selection"]])
    assert frozen == {"id": "selection:" + content_sha256(expected), **expected, "manifest_sha256": content_sha256(expected)}
    assert choice["name"] == "Label 0"


def test_catalog_snapshot_is_coherent_and_next_read_is_fresh(db, monkeypatch):
    resource, version, locations, _ = dataset(db)
    original = PostgresConnection.execute
    changed = False
    def execute(connection, statement, parameters=None):
        nonlocal changed
        cursor = original(connection, statement, parameters)
        if not changed and statement.startswith("SELECT * FROM data_resources WHERE archived_at"):
            changed = True
            with db.transaction() as writer:
                writer.execute("UPDATE data_locations SET status='MISSING' WHERE version_id=?", (version["id"],))
        return cursor
    monkeypatch.setattr(PostgresConnection, "execute", execute)
    first = data_selection.choices(db)
    assert changed and first[0]["selection"]["version_id"] == version["id"]
    assert first[0]["assignments"][0]["config"]["location"]["status"] == "AVAILABLE"
    assert data_selection.choices(db) == []
    db.record_data_location(version["id"], kind="cluster", host=locations[0]["host"], path=locations[0]["path"], manifest_sha256=version["manifest_sha256"])
    assert len(data_selection.choices(db)) == 1
    db.update_data_resource(resource["id"], archived=True)
    assert data_selection.choices(db) == []
    assert db.list_data_resources() == []
    assert db.list_data_resources(include_archived=True)[0]["archived_at"]


def test_bulk_versions_preserve_lineage_and_bundle_links_with_five_reads(db, reads):
    resource, version, locations, source = dataset(db)
    derivation = db.create_data_derivation(
        output_version_id=version["id"], inputs=[{"version_id": source["id"]}],
        converter_repository="https://example.test/converter", converter_commit="a" * 40,
    )
    bundle = db.create_data_bundle(name="Frozen", version="1", assignments=[{
        "role": "training_data", "position": 0, "version_id": version["id"],
    }])
    reads.clear()
    versions = db.get_data_resource_versions([version["id"], source["id"], "missing"])
    assert len(reads) == 5
    assert set(versions) == {version["id"], source["id"]}
    assert versions[version["id"]]["derivation_id"] == derivation["id"]
    assert versions[source["id"]]["derivation_id"] is None
    loaded_resource = versions[version["id"]]["resource"]
    assert loaded_resource["id"] == resource["id"]
    assert loaded_resource["metadata"] == resource["metadata"]
    assert set(loaded_resource) == set(resource) - {"recording_ids", "version_count", "latest_version", "versions"}
    assert versions[version["id"]]["used_by_bundles"] == [{"id": bundle["id"], "name": "Frozen", "version": "1", "role": "training_data", "position": 0}]
    assert db.get_data_resource_version(version["id"]) == versions[version["id"]]
    reads.clear()
    assert db.get_data_resource_versions([]) == {} and reads == []


def test_usage_batch_redacts_other_workspaces_and_reflects_new_revisions(db, reads):
    _, version, _, _ = dataset(db)
    directory = WorkspaceDirectory(db)
    alice = db.for_workspace(directory.open("alice@example.com")[0]["id"])
    bob = db.for_workspace(directory.open("bob@example.com")[0]["id"])
    spec = {"data": {"bundle": {"assignments": [{"version": {"manifest_sha256": version["manifest_sha256"]}}]}}}
    own = alice.create_experiment(name="Alice input", requested_spec=spec)
    bob.create_experiment(name="Secret experiment", requested_spec=spec)
    reads.clear()
    usages = db.data_version_usage_many([version["manifest_sha256"], "missing"], workspace_id=alice.workspace_id)
    assert len(reads) == 1 and usages["missing"] == []
    assert usages[version["manifest_sha256"]] == [
        {"experiment_id": own["id"], "name": "Alice input", "revision_number": 1, "run_id": None, "run_status": None},
        {"other_workspace": True},
    ]
    assert "Secret" not in canonical_json(usages)
    alice.create_experiment_revision(own["id"], requested_spec=spec)
    assert len(db.data_version_usage_many([version["manifest_sha256"]], workspace_id=alice.workspace_id)[version["manifest_sha256"]]) == 3
    assert len(db.data_version_usage(version["manifest_sha256"])) == 3


def test_adapter_catalog_decodes_only_latest_and_preserves_legacy_seed_permissions(db, monkeypatch, reads):
    directory = WorkspaceDirectory(db)
    alice = db.for_workspace(directory.open("alice@example.com")[0]["id"])
    seed = db.upsert_seed_adapter(seed_key="seed", name="Legacy seed", manifest={"slug": "seed"})
    own = alice.create_adapter(name="Zulu", manifest={"slug": "own"})
    for index in range(5):
        own = alice.edit_adapter(own["id"], name="Bravo", manifest={"slug": "own", "version": index})
    decoded = []
    original = Database._decode
    def decode(row):
        if row is not None and "adapter_key" in row.keys():
            decoded.append(row["id"])
        return original(row)
    monkeypatch.setattr(Database, "_decode", staticmethod(decode))
    reads.clear()
    catalog = alice.list_adapter_registry(include_editable=True)
    assert len(reads) == 1
    assert len(decoded) == len(catalog) == 2
    records = {row["id"]: row for row in catalog}
    assert records[own["id"]]["editable"]
    assert records[own["id"]]["version_count"] == 6
    assert not records[seed["id"]]["editable"]
    assert not records[seed["id"]]["shared"]
    assert catalog[0]["id"] == own["id"]
    alice.archive_adapter(own["id"])
    assert own["id"] not in {row["id"] for row in alice.list_adapter_registry()}
    assert own["id"] in {row["id"] for row in alice.list_adapter_registry(include_archived=True)}


def test_http_catalog_exposes_versions_and_adapter_editability_without_detail_reads(db, reads, monkeypatch):
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from skynet_app import pipeline_api
    resource, _, _, _ = dataset(db)
    adapter = db.create_adapter(name="Adapter", manifest={"slug": "adapter"})
    monkeypatch.setattr(pipeline_api, "service", SimpleNamespace(database=db))
    monkeypatch.setattr(db, "owns", lambda *args, **kwargs: pytest.fail("Catalog did per-item ownership read"))
    app = FastAPI()
    app.include_router(pipeline_api.router)
    with TestClient(app) as client:
        reads.clear()
        response = client.get("/api/data/resources?include_versions=true")
        assert response.status_code == 200
        assert len(reads) == 3
        item = response.json()["resources"][0]
        assert item["id"] == resource["id"] and len(item["versions"]) == 2
        assert "versions" not in client.get("/api/data/resources").json()["resources"][0]
        reads.clear()
        response = client.get("/api/adapters")
        assert response.status_code == 200 and len(reads) == 1
        assert response.json()["adapters"][0]["id"] == adapter["id"]
        assert response.json()["adapters"][0]["editable"] is True
