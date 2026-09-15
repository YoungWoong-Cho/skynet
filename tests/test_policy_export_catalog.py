"""Catalog reads stay bounded as the recording library grows."""

import copy
import hashlib

import pytest

from skynet_app.cluster_config import CLUSTER
from skynet_app.cluster_runtime import ClusterClient
from skynet_app.database import canonical_json
from skynet_app.db_backend import PostgresConnection
from skynet_app.live_xr import LiveXRService
from skynet_app.live_xr_archive import LiveArchiveService
from test_policy_exports import setup


def persisted_sessions(service):
    """Use the real bulk read/public projection without starting live workers."""
    live = service.live
    live.public = LiveXRService.public
    live.list = LiveXRService.list.__get__(live)
    live.get = LiveXRService.get.__get__(live)
    return live


def save_sessions(service, sessions):
    with service.database.transaction() as connection:
        connection.execute("DELETE FROM live_xr_sessions")
        for session in sessions:
            connection.execute(
                "INSERT INTO live_xr_sessions VALUES (?, ?)",
                (session["id"], canonical_json(session)),
            )


def test_catalog_queries_do_not_scale_with_sessions_or_recordings(setup, monkeypatch):
    service, original, _ = setup
    live = persisted_sessions(service)
    original["recording_images"] = {}
    queries = []
    execute = PostgresConnection.execute

    def counted(connection, statement, parameters=None):
        queries.append(statement)
        return execute(connection, statement, parameters)

    monkeypatch.setattr(PostgresConnection, "execute", counted)
    monkeypatch.setattr(live, "get", lambda _: pytest.fail("Catalog reread a session"))
    counts = []
    for session_count, episode_count in [(1, 2), (8, 250)]:
        sessions = []
        for index in range(session_count):
            session = copy.deepcopy(original)
            session["id"] = f"session-{index}"
            session["recordings"] = [f"recordings/live/{i}.pkl" for i in range(episode_count)]
            session["recording_checksums"] = {name: "a" * 64 for name in session["recordings"]}
            sessions.append(session)
        save_sessions(service, sessions)
        queries.clear()
        options = service.options()
        assert len(options["sessions"]) == session_count
        assert all(item["eligible"] and item["episodes"] == episode_count for item in options["sessions"])
        assert sum("FROM live_xr_sessions" in query for query in queries) == 1
        assert sum("FROM data_resources" in query for query in queries) == 1
        counts.append(len(queries))
    assert counts[0] == counts[1]


def test_catalog_preserves_dataset_links_and_archived_eligibility(setup, monkeypatch):
    service, original, _ = setup
    sessions = [dict(original, id=f"session-{i}") for i in range(3)]
    monkeypatch.setattr(service.live, "list", lambda **_: sessions)
    archived = service.dataset(sessions[0])
    service.database.update_data_resource(archived["id"], archived=True)
    overfit = service.dataset(dict(sessions[1], recordings=sessions[1]["recordings"][:1]), overfit_episode=0)
    options = {item["id"]: item for item in service.options()["sessions"]}
    assert options["session-0"]["resource_id"] == archived["id"]
    assert options["session-0"]["reason"] == "Dataset is archived"
    assert not options["session-0"]["eligible"]
    assert options["session-1"]["resource_id"] == overfit["id"]
    assert options["session-1"]["eligible"]
    assert options["session-2"]["resource_id"] is None
    assert options["session-2"]["eligible"]


@pytest.mark.parametrize("mutation, reason", [
    (None, None),
    ("missing_manifest", "saved archive manifest is invalid"),
    ("unlisted", "not part of the verified session archive"),
    ("path", "Invalid saved recording path"),
    ("source_removed", "must be verified before its files can be used"),
])
def test_catalog_keeps_archive_and_recording_validation(setup, mutation, reason):
    service, session, _ = setup
    live = persisted_sessions(service)
    session["recording_images"] = {}
    manifest = dict(
        schema="skynet.live-archive/v1", session_id=session["id"],
        files=[{"path": "output/" + name} for name in session["recordings"]],
    )
    if mutation == "unlisted":
        manifest["files"] = []
    checksum = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    session["archive"] = dict(
        state="READY", gateway="sky2", manifest=manifest, manifest_sha256=checksum,
        root=f"{CLUSTER.paths.datasets}/raw/dexverse-live/{session['id']}/{checksum}/output",
    )
    if mutation == "missing_manifest":
        session["archive"].pop("manifest")
    elif mutation == "path":
        session["recordings"][0] = "recordings/../outside.pkl"
    elif mutation == "source_removed":
        session["archive"].update(state="VERIFY_FAILED", source_removed=True)
    # resolve() remains the real archive validator; no remote operation is needed.
    live.archive = object.__new__(LiveArchiveService)
    live.archive.live, live.archive.cluster = live, ClusterClient()
    save_sessions(service, [session])
    assert "manifest" not in live.list()[0]["archive"]
    options = service.options()["sessions"][0]
    assert options["eligible"] is (reason is None)
    if reason:
        assert reason in options["reason"]
    else:
        assert options["reason"] is None


def test_preparation_options_reads_only_selected_recording(setup, monkeypatch):
    service, session, _ = setup
    live = persisted_sessions(service)
    save_sessions(service, [session, dict(session, id="unrelated")])
    resource = service.dataset(session, "Existing dataset")
    monkeypatch.setattr(live, "list", lambda **_: pytest.fail("Read all recording sessions"))
    monkeypatch.setattr(service, "list", lambda: pytest.fail("Read all conversion jobs"))
    monkeypatch.setattr(service.database, "list_data_resources", lambda **_: pytest.fail("Expanded the entire dataset registry"))
    monkeypatch.setattr(service.database, "get_data_resource", lambda _: pytest.fail("Expanded dataset versions"))
    queries = []
    execute = PostgresConnection.execute
    def counted(connection, statement, parameters=None):
        queries.append(statement)
        return execute(connection, statement, parameters)
    monkeypatch.setattr(PostgresConnection, "execute", counted)
    result = service.preparation_options(session["id"])
    assert result["session"]["eligible"]
    assert result["session"]["episodes"] == 2
    assert result["resource"]["id"] == resource["id"]
    assert result["resource"]["metadata"]["display_name"] == "Existing dataset"
    assert "versions" not in result["resource"]
    assert len(queries) == 3, queries
    assert "WHERE id=?" in queries[0]
    assert all("policy_exports" not in query for query in queries)


def test_preparation_options_uses_current_source_and_archival_state(setup, monkeypatch):
    service, session, _ = setup
    result = service.preparation_options(session["id"])
    assert result["session"]["eligible"] and result["resource"] is None
    resource = service.dataset(session)
    service.database.update_data_resource(resource["id"], archived=True)
    result = service.preparation_options(session["id"])
    assert not result["session"]["eligible"]
    assert result["session"]["reason"] == "Dataset is archived"
    session["state"] = "RUNNING"
    result = service.preparation_options(session["id"])
    assert not result["session"]["eligible"]
    assert "End the collection session" in result["session"]["reason"]


def test_bulk_adapter_catalog_preserves_versions_order_and_workspace_visibility(setup, monkeypatch):
    service, _, _ = setup
    database = service.database
    first = database.create_adapter(name="Zulu", manifest={"slug": "first"})
    database.edit_adapter(first["id"], name="Bravo", manifest={"slug": "first-new"})
    archived = database.create_adapter(name="Alpha", manifest={"slug": "archived"})
    database.archive_adapter(archived["id"])
    from skynet_app.workspaces import WorkspaceDirectory
    directory = WorkspaceDirectory(database)
    alice = database.for_workspace(directory.open("alice@example.com")[0]["id"])
    bob = database.for_workspace(directory.open("bob@example.com")[0]["id"])
    private = alice.create_adapter(name="Alice private", manifest={"slug": "private"})
    bob.create_adapter(name="Bob private", manifest={"slug": "other"})
    for scoped in (database, alice, bob):
        for include_archived in (False, True):
            # Compare complete public bundles to the previous key-and-detail path.
            from skynet_app.workspace_schema import visible_sql
            with scoped.connection() as connection:
                keys = connection.execute(
                    f"SELECT adapter_key, min(lower(name)) AS sort_name FROM adapters WHERE {visible_sql('adapters')} AND (? = 1 OR archived_at IS NULL) GROUP BY adapter_key ORDER BY sort_name, adapter_key",
                    (int(include_archived),),
                ).fetchall()
                expected = [scoped._adapter_bundle(scoped._adapter_rows(connection, row["adapter_key"])) for row in keys]
            queries = []
            execute = PostgresConnection.execute
            def counted(connection, statement, parameters=None):
                queries.append(statement)
                return execute(connection, statement, parameters)
            with monkeypatch.context() as patch:
                patch.setattr(PostgresConnection, "execute", counted)
                actual = scoped.list_adapter_registry(include_archived=include_archived)
            assert actual == expected
            assert len(queries) == 1
    assert private["id"] not in {item["id"] for item in bob.list_adapter_registry()}


def test_focused_dataset_lookup_preserves_overfit_identity(setup):
    service, session, _ = setup
    single = dict(session, recordings=session["recordings"][:1])
    resource = service.dataset(single, overfit_episode=0)
    assert service.database.find_collection_dataset(session["id"])["id"] == resource["id"]
    another = dict(session, id="multi-recording-session")
    subset = service.dataset(another, overfit_episode=1)
    assert service.database.find_collection_dataset(another["id"]) is None
    assert service.database.find_collection_dataset(another["id"], f"{another['id']}:overfit:1")["id"] == subset["id"]


def test_focused_preparation_endpoint_and_missing_session(setup, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from skynet_app import policy_exports_api
    service, session, _ = setup
    persisted_sessions(service)
    save_sessions(service, [session])
    monkeypatch.setattr(policy_exports_api, "service", service)
    app = FastAPI()
    app.include_router(policy_exports_api.router)
    with TestClient(app) as client:
        response = client.get(f"/api/data/exports/options/{session['id']}")
        assert response.status_code == 200
        assert response.json()["session"]["id"] == session["id"]
        assert "exports" not in response.json()
        assert client.get("/api/data/exports/options/missing").status_code == 404
