"""Retired workflows stay unreachable while the current collection API remains usable."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from skynet_app.db_backend import PostgresConnection


def test_application_routes_keep_current_workflows_and_remove_retired_ones():
    from skynet_app import main

    app, _owner = main._create_cluster_application()
    registered = {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if method in {"get", "post", "put", "patch", "delete", "head", "options"}
    }
    retired_prefixes = (
        "/api/collection/local/",
        "/api/collection/processing/",
        "/api/collection/live/conversions/",
    )
    assert not any(path.startswith(retired_prefixes) for _, path in registered)
    assert (
        "POST", "/api/collection/live/sessions/{identifier}/conversions"
    ) not in registered
    assert {
        ("GET", "/api/collection/live"),
        ("POST", "/api/collection/live/sessions"),
        ("GET", "/api/collection/sessions"),
        ("GET", "/api/collection/live/sessions/{identifier}/recordings/{index}/video"),
        ("POST", "/api/collection/live/sessions/{identifier}/recordings/{index}/video"),
        ("GET", "/api/data/exports/options/{session_id}"),
        ("POST", "/api/data/exports"),
        ("GET", "/api/maintenance/history/{kind}/{identifier}"),
    } <= registered


def test_live_overview_no_longer_reads_retired_conversion_history(monkeypatch, tmp_path):
    from skynet_app import live_xr_api
    from skynet_app.database import Database
    from skynet_app.live_xr import LiveXRService

    # API modules are cached, while the PostgreSQL fixture is per test.
    monkeypatch.setattr(
        live_xr_api, "service", LiveXRService(Database(tmp_path / "current-collection"))
    )

    statements = []
    execute = PostgresConnection.execute

    def recorded(connection, statement, parameters=None):
        statements.append(statement)
        return execute(connection, statement, parameters)

    monkeypatch.setattr(PostgresConnection, "execute", recorded)
    monkeypatch.setattr(
        live_xr_api.service,
        "profile",
        lambda: {"execution": "slurm", "gateway": "sky2", "duration_minutes": 30},
    )
    app = FastAPI()
    app.include_router(live_xr_api.router)
    with TestClient(app) as client:
        response = client.get("/api/collection/live")
        assert response.status_code == 200
        assert {"sessions", "catalog", "license", "target"} == response.json().keys()
        assert any("live_xr_sessions" in statement for statement in statements)
        assert not any("live_conversions" in statement for statement in statements)
        for method, path in (
            ("GET", "/api/collection/live/conversions/old-id"),
            ("GET", "/api/collection/live/conversions/old-id/logs"),
            ("GET", "/api/collection/live/conversions/old-id/dataset.hdf5"),
            ("POST", "/api/collection/live/sessions/old-id/conversions"),
        ):
            assert client.request(method, path).status_code == 404
