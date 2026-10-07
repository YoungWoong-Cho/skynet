"""The browser's current collection and maintenance routes stay registered and lean."""

import re

from fastapi import FastAPI
from fastapi.testclient import TestClient

from skynet_app.db_backend import PostgresConnection


def test_application_registers_current_collection_export_and_maintenance_routes():
    from skynet_app import main

    app, _owner = main._create_cluster_application()
    registered = {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if method in {"get", "post", "put", "patch", "delete", "head", "options"}
    }
    assert {
        ("GET", "/api/collection/live"),
        ("POST", "/api/collection/live/sessions"),
        ("GET", "/api/collection/sessions"),
        ("GET", "/api/collection/live/sessions/{identifier}/recordings/{index}/video"),
        ("GET", "/api/data/exports/options/{session_id}"),
        ("POST", "/api/data/exports"),
        ("GET", "/api/maintenance/history/{kind}/{identifier}"),
    } <= registered


def test_live_overview_reads_only_live_sessions_and_consent(monkeypatch, tmp_path):
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
    tables = {
        name.lower()
        for statement in statements
        for name in re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", statement)
    }
    assert tables == {"live_xr_sessions", "live_xr_consent"}
