"""Resource labels can change without changing source identity or frozen receipts."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skynet_app import data_selection, pipeline_api
from skynet_app.database import Database, canonical_json
from skynet_app.db_backend import INTEGRITY_ERRORS
from skynet_app.workspaces import WorkspaceDirectory


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def database(tmp_path):
    return Database(tmp_path / "resource-naming")


def dataset(database, source_key="recording:overfit:3", **overrides):
    values = dict(
        category="dataset", kind="demonstrations", provider="collection",
        namespace="datasets", source_key=source_key, display_name="Shadow cube demonstrations",
        description="Live DexVerse · Shadow · right hand · Pick up cube",
        metadata={"recording_session_id": "recording", "managed_dataset": True},
    )
    return database.create_data_resource(**(values | overrides))


def prepared(database, resource):
    return database.create_data_resource_version(
        resource["id"], revision="prepared-v1", format="zarr", path="/cluster/prepared/cube",
        manifest_sha256="a" * 64,
        metadata={"display_name": "Historical preparation label", "episodes": 5},
    )


def stored_rows(database, tables):
    with database.connection() as connection:
        return {
            table: [row[0] for row in connection.execute(
                f"SELECT row_to_json(t)::text FROM {table} t ORDER BY id"
            )]
            for table in tables
        }


def test_display_names_are_mutable_nonunique_and_identity_stays_unique(database):
    first = dataset(database)
    second = dataset(database, source_key="another-recording")
    assert first["id"] != second["id"]
    assert first["display_name"] == second["display_name"]
    assert "name" not in first and "display_name" not in first["metadata"]
    with pytest.raises(INTEGRITY_ERRORS):
        dataset(database, display_name="Different label, same source")
    original = copy.deepcopy(first)
    edited = database.update_data_resource(first["id"], display_name="Renamed dataset")
    for key in ("id", "provider", "namespace", "source_key", "kind", "category", "description", "metadata", "created_at"):
        assert edited[key] == original[key]
    assert edited["display_name"] == "Renamed dataset"
    assert edited["updated_at"] > original["updated_at"]
    assert database.list_data_resources()[0]["display_name"] in {"Renamed dataset", second["display_name"]}
    for field, value in (("id", "replacement"), ("provider", "replacement"),
                         ("namespace", "replacement"), ("source_key", "replacement")):
        with pytest.raises(INTEGRITY_ERRORS), database.transaction() as connection:
            connection.execute(f"UPDATE data_resources SET {field}=? WHERE id=?", (value, first["id"]))


def test_rename_preserves_v1_receipts_and_legacy_links_in_each_workspace(database):
    resource = dataset(database)
    version = prepared(database, resource)
    bundle = database.create_data_bundle(name="Frozen bundle", version="v1", assignments=[{
        "role": "training_data", "version_id": version["id"],
    }])
    selection = data_selection.snapshot(database, [{"version_id": version["id"]}])
    assert selection["assignments"][0]["resource"] == {
        "provider": "collection", "namespace": "datasets", "name": resource["source_key"],
        "kind": "demonstrations",
    }
    assert selection["name"] == resource["source_key"]
    directory = WorkspaceDirectory(database)
    alice = database.for_workspace(directory.open("alice@example.com")[0]["id"])
    bob = database.for_workspace(directory.open("bob@example.com")[0]["id"])
    # A legacy v1 bundle has no registered_version_id; links resolve the immutable
    # resource.name within its receipt against the current source_key column.
    legacy = {"data": {"bundle": bundle["manifest"]}}
    alice_experiment = alice.create_experiment(name="Alice preset", requested_spec=legacy)
    bob.create_experiment(name="Bob private preset", requested_spec=legacy)
    immutable_tables = ("data_resource_versions", "data_bundles", "experiment_revisions")
    before = stored_rows(database, immutable_tables)
    database.update_data_resource(resource["id"], display_name="New visible label", description="New description")
    assert stored_rows(database, immutable_tables) == before
    assert data_selection.snapshot(database, [{"version_id": version["id"]}]) == selection
    choice = data_selection.choices(database)[0]
    assert choice["name"] == "New visible label"
    assert choice["manifest_sha256"] == selection["manifest_sha256"]
    assert choice["assignments"] == selection["assignments"]
    assert alice.dataset_preset_links() == [{
        "experiment_id": alice_experiment["id"], "experiment_name": "Alice preset",
        "revision_number": 1, "resource_id": resource["id"], "version_id": version["id"],
    }]
    usage = alice.data_version_usage_many([version["manifest_sha256"]], workspace_id=alice.workspace_id)
    assert {"other_workspace": True} in usage[version["manifest_sha256"]]
    assert "Bob private preset" not in canonical_json(usage)


def test_resource_patch_updates_requested_fields_and_rejects_identity_changes(database, monkeypatch):
    resource = dataset(database)
    app = FastAPI()
    app.include_router(pipeline_api.router)
    app.dependency_overrides[pipeline_api.require_workspace_records] = lambda: None
    monkeypatch.setattr(pipeline_api, "service", SimpleNamespace(database=database))
    with TestClient(app) as client:
        response = client.patch(f"/api/data/resources/{resource['id']}", json={"display_name": "Edited name"})
        assert response.status_code == 200
        edited = response.json()["resource"]
        assert edited["display_name"] == "Edited name"
        assert edited["description"] == resource["description"]
        assert edited["metadata"] == resource["metadata"]
        assert edited["source_key"] == resource["source_key"] and "name" not in edited
        response = client.patch(f"/api/data/resources/{resource['id']}", json={"description": ""})
        assert response.status_code == 200
        assert response.json()["resource"]["display_name"] == "Edited name"
        assert response.json()["resource"]["description"] == ""
        for field in ("id", "provider", "namespace", "source_key", "name"):
            response = client.patch(f"/api/data/resources/{resource['id']}", json={field: "different"})
            assert response.status_code == 422, (field, response.text)
        for label in ("", "   ", None):
            response = client.patch(f"/api/data/resources/{resource['id']}", json={"display_name": label})
            assert response.status_code == 422, (label, response.text)
        response = client.patch(f"/api/data/resources/{resource['id']}", json={"metadata": {"display_name": "Hidden duplicate"}})
        assert response.status_code == 422
        catalog = client.get("/api/data/resources").json()["resources"]
        assert catalog[0]["display_name"] == "Edited name"
        assert "display_name" not in catalog[0]["metadata"]
        assert "name" not in catalog[0]
        payload = {"category": "file", "kind": "model", "provider": "fixture",
                   "namespace": "files", "source_key": "weights"}
        response = client.post("/api/data/resources", json=payload)
        assert response.status_code == 201, response.text
        created = response.json()["resource"]
        assert created["display_name"] == "files/weights"
        assert created["source_key"] == "weights" and "name" not in created
        response = client.post("/api/data/resources", json=payload | {
            "source_key": "other", "metadata": {"display_name": "Hidden duplicate"},
        })
        assert response.status_code == 422
        legacy_payload = {key: value for key, value in payload.items() if key != "source_key"}
        response = client.post("/api/data/resources", json=legacy_payload | {"name": "old-key"})
        assert response.status_code == 422


def test_migrations_and_rollback_preserve_labels_sources_and_immutable_history(database):
    resource = dataset(database)
    version = prepared(database, resource)
    bundle = database.create_data_bundle(name="Frozen", version="v1", assignments=[{
        "role": "training_data", "version_id": version["id"],
    }])
    database.create_experiment(name="Historical reference", requested_spec={"data": {"bundle": bundle["manifest"]}})
    immutable_tables = ("data_resource_versions", "data_bundles", "experiment_revisions")
    history = stored_rows(database, immutable_tables)
    rollback = (ROOT / "deploy/rollback-resource-naming.sql").read_text()
    cases = [
        ("named", "datasets", {"display_name": "Existing label", "keep": [1, 2]}, "Existing label"),
        ("missing", "datasets", {"keep": True}, "datasets/missing"),
        ("empty", "datasets", {"display_name": ""}, "datasets/empty"),
        ("blank", "datasets", {"display_name": "   "}, "datasets/blank"),
        ("number", "datasets", {"display_name": 123}, "datasets/number"),
        ("null", "datasets", {"display_name": None}, "datasets/null"),
        ("unicode", "datasets", {"display_name": "손 · 큐브"}, "손 · 큐브"),
        ("no-namespace", "", {}, "no-namespace"),
        ("duplicate", "datasets", {"display_name": "Existing label"}, "Existing label"),
    ]
    with database.connection() as connection:
        connection.executescript(rollback)
        assert connection.execute("SELECT count(*) FROM skynet_schema_migrations WHERE version IN (12,13)").fetchone()[0] == 0
        for source_key, namespace, metadata, _ in cases:
            connection.execute(
                "INSERT INTO data_resources (id,provider,namespace,name,category,kind,description,metadata_json,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("legacy-" + source_key, "fixture", namespace, source_key, "dataset", "dataset", "Keep description", canonical_json(metadata), "2026-01-01", "2026-01-02"),
            )
        count = connection.execute("SELECT count(*) FROM data_resources").fetchone()[0]
    database.backend.initialize()
    assert stored_rows(database, immutable_tables) == history
    with database.connection() as connection:
        assert connection.execute("SELECT count(*) FROM data_resources").fetchone()[0] == count
    for source_key, namespace, metadata, expected_label in cases:
        row = database.get_data_resource("legacy-" + source_key)
        assert row["source_key"] == source_key and row["namespace"] == namespace
        assert row["display_name"] == expected_label
        assert row["metadata"] == {key: value for key, value in metadata.items() if key != "display_name"}
        assert row["description"] == "Keep description"
        assert row["created_at"] == "2026-01-01" and row["updated_at"] == "2026-01-02"
        assert "name" not in row
    database.update_data_resource(resource["id"], display_name="Edited after migration")
    with database.connection() as connection:
        connection.executescript(rollback)
        row = connection.execute("SELECT name, metadata_json FROM data_resources WHERE id=?", (resource["id"],)).fetchone()
        assert row["name"] == resource["source_key"]
        assert json.loads(row["metadata_json"])["display_name"] == "Edited after migration"
        assert connection.execute("SELECT count(*) FROM data_resources").fetchone()[0] == count
    assert stored_rows(database, immutable_tables) == history
    database.backend.initialize()
    assert database.get_data_resource(resource["id"])["display_name"] == "Edited after migration"
    assert stored_rows(database, immutable_tables) == history


def test_display_name_changes_notify_after_commit_and_rolled_back_changes_stay_hidden(database):
    resource = dataset(database)
    listener = database.backend.connect()
    listener.raw.execute("LISTEN skynet_changes")
    try:
        with database.connection() as writer:
            with writer.raw.transaction():
                writer.execute("UPDATE data_resources SET display_name=? WHERE id=?", ("Committed label", resource["id"]))
                assert list(listener.raw.notifies(timeout=0.05)) == []
            events = [json.loads(event.payload) for event in listener.raw.notifies(timeout=0.1)]
            assert events == [{"v": 1, "scope": "*", "topics": ["data", "exports"]}]
            with pytest.raises(RuntimeError), writer.raw.transaction():
                writer.execute("UPDATE data_resources SET display_name=? WHERE id=?", ("Rolled back label", resource["id"]))
                raise RuntimeError("Rollback fixture")
        assert list(listener.raw.notifies(timeout=0.1)) == []
        assert database.get_data_resource(resource["id"])["display_name"] == "Committed label"
    finally:
        listener.close()
