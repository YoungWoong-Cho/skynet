"""Settings use one fresh scoped database snapshot without changing credential policy."""

import json
from collections import Counter
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skynet_app import pipeline_api
from skynet_app.credential_store import StoredCredential
from skynet_app.database import Database
from skynet_app.db_backend import PostgresConnection
from skynet_app.pipeline_api import PipelineService, router
from skynet_app.tracking import SessionCredentialStore
from skynet_app.workspaces import WorkspaceMiddleware, WorkspaceServices, session_router


@pytest.fixture
def reads(monkeypatch):
    counts = Counter()
    original = PostgresConnection.execute

    def execute(connection, statement, parameters=None):
        normalized = " ".join(statement.lower().split())
        if normalized.startswith("select "):
            for table in ("workspace_storage", "tracking_connections"):
                if "from " + table + " " in normalized + " ":
                    counts[table] += 1
        return original(connection, statement, parameters)

    monkeypatch.setattr(PostgresConnection, "execute", execute)
    return counts


def set_root(database, value):
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO workspace_storage(owner_id,work_root) VALUES (?,?) "
            "ON CONFLICT(owner_id) DO UPDATE SET work_root=excluded.work_root",
            (database.workspace_id, value),
        )


def test_settings_reads_once_and_sees_later_changes_in_the_correct_workspace(tmp_path, monkeypatch, reads):
    services = WorkspaceServices(PipelineService(
        Database(tmp_path / "settings.db"), object(), session_credentials=SessionCredentialStore(),
    ))
    monkeypatch.setattr(pipeline_api, "service", services)
    app = FastAPI()
    app.add_middleware(WorkspaceMiddleware, services=services)
    app.include_router(session_router(services.directory))
    app.include_router(router)
    with TestClient(app) as client:
        assert client.get("/api/settings").status_code == 401
        session = client.post("/api/workspace/session", json={"email": "alice@example.com"}).json()["workspace"]
        alice = services.for_workspace(session["id"])
        reads.clear()
        first = client.get("/api/settings")
        assert first.status_code == 200
        assert first.json()["storage"]["configured"] is False
        assert first.json()["paths"]["work_root"] is None
        assert first.json()["cluster"]["paths"]["jobs"] is None
        assert reads == {"workspace_storage": 1, "tracking_connections": 1}

        set_root(alice.database, "/team/alice")
        alice.database.upsert_tracking_connection(
            "mlflow", endpoint="https://alice.example.test", config={"authentication": "none"},
        )
        reads.clear()
        current = client.get("/api/settings").json()
        assert reads == {"workspace_storage": 1, "tracking_connections": 1}
        assert current["storage"]["work_root"] == current["paths"]["work_root"] == "/team/alice"
        assert current["paths"]["evaluation_root"] == current["cluster"]["paths"]["evaluation"] == "/team/alice/eval"
        assert current["tracking"]["connections"]["mlflow"]["tracking_uri"] == "https://alice.example.test"

        client.post("/api/workspace/session", json={"email": "bob@example.com"})
        reads.clear()
        other = client.get("/api/settings").json()
        assert reads == {"workspace_storage": 1, "tracking_connections": 1}
        assert other["storage"]["configured"] is False
        assert other["paths"]["work_root"] is None
        assert other["tracking"]["connections"]["mlflow"]["tracking_uri"] is None
        assert client.get("/api/settings", headers={"X-Skynet-Workspace": session["id"]}).status_code == 409


def test_tracking_restores_both_credentials_with_one_read_and_keeps_endpoint_binding(tmp_path, monkeypatch, reads):
    monkeypatch.setattr(PipelineService, "_tracking_environment", lambda self: {})
    database = Database(tmp_path / "credentials.db")
    secrets = {"wandb": "wandb-fixture-secret", "mlflow": "mlflow-fixture-secret"}
    endpoints = {"wandb": "https://wandb.example.test", "mlflow": "https://mlflow.example.test"}
    for provider, authentication in (("wandb", "api_key"), ("mlflow", "token")):
        database.upsert_tracking_connection(
            provider, endpoint=endpoints[provider],
            config={"authentication": authentication, "credential_source": "credential_store"},
        )
    store = Mock()
    store.load.side_effect = lambda provider: StoredCredential(
        provider, endpoints[provider],
        {"api_key" if provider == "wandb" else "token": secrets[provider]},
    )
    service = PipelineService(database, object(), credential_store=store, session_credentials=SessionCredentialStore())
    reads.clear()
    first = service.tracking_connections()
    assert reads == {"tracking_connections": 1}
    assert all(item["connected"] for item in first["connections"].values())
    assert all(item["credential_source"] == "credential_store" for item in first["connections"].values())
    assert store.load.call_count == 2
    assert not any(secret in json.dumps(first) for secret in secrets.values())

    database.upsert_tracking_connection(
        "wandb", endpoint="https://changed.example.test",
        config={"authentication": "api_key", "credential_source": "credential_store"},
    )
    reads.clear()
    second = service.tracking_connections()
    assert reads == {"tracking_connections": 1}
    wandb = second["connections"]["wandb"]
    assert wandb["base_url"] == "https://changed.example.test"
    assert wandb["connected"] is False
    assert wandb["credential_source"] is None
    assert second["connections"]["mlflow"]["connected"] is True
    assert store.load.call_count == 2
    assert not any(secret in json.dumps(second) for secret in secrets.values())


def test_tracking_uses_one_coherent_read_when_another_writer_changes_the_endpoint(tmp_path, monkeypatch, reads):
    monkeypatch.setattr(PipelineService, "_tracking_environment", lambda self: {})
    database = Database(tmp_path / "concurrent.db")
    database.upsert_tracking_connection(
        "mlflow", endpoint="https://before.example.test", config={"authentication": "none"},
    )
    service = PipelineService(database, object(), session_credentials=SessionCredentialStore())
    original = database.list_tracking_connections
    changed = False

    def read_then_change():
        nonlocal changed
        snapshot = original()
        if not changed:
            changed = True
            database.upsert_tracking_connection(
                "mlflow", endpoint="https://after.example.test", config={"authentication": "token", "credential_source": "session"},
            )
        return snapshot

    monkeypatch.setattr(database, "list_tracking_connections", read_then_change)
    first = service.tracking_connections()["connections"]["mlflow"]
    assert first["tracking_uri"] == "https://before.example.test"
    assert first["connected"] is True
    reads.clear()
    second = service.tracking_connections()["connections"]["mlflow"]
    assert reads == {"tracking_connections": 1}
    assert second["tracking_uri"] == "https://after.example.test"
    assert second["connected"] is False
