"""Offline boot/recovery regressions; fake-runtime tests also run with --noconftest."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import anyio
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import pytest


class Owner:
    def __init__(self):
        self.starts = 0
        self.stops = 0
        self.started = threading.Event()

    def start(self):
        self.starts += 1
        self.started.set()

    def stop(self):
        self.stops += 1


@pytest.fixture
def local_app(monkeypatch):
    from skynet_app import main

    # Patch the instance already mounted in main.app; replacing the global alone
    # would leave requests going to the original instance.
    runtime = main.cluster_application
    monkeypatch.setattr(runtime, "application", None)
    monkeypatch.setattr(runtime, "owner", None)
    monkeypatch.setattr(runtime, "status", "starting")
    monkeypatch.setattr(runtime, "retry_interval", 0.01)
    return main, runtime


def test_import_main_does_not_initialize_cluster_services(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = r'''
import importlib.abc
import subprocess
import sys

blocked = {
    "skynet_app.pipeline_api", "skynet_app.collection_api",
    "skynet_app.live_xr_api",
    "skynet_app.policy_exports_api",
}
class NoControllers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("Eager cluster service import: " + fullname)

sys.meta_path.insert(0, NoControllers())
def no_process(*args, **kwargs):
    raise AssertionError("Import attempted to start an external process")
subprocess.Popen = no_process
from skynet_app import main
assert not blocked.intersection(sys.modules)
assert main.cluster_application.application is None
assert main.index().status_code == 200
assert main.health().status_code == 503
print("Local application imports without cluster access")
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(root),
            "SKYNET_DATABASE_URL": "host=/nonexistent/skynet-offline-test dbname=skynet connect_timeout=1",
            "SKYNET_DATA_ROOT": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "without cluster access" in result.stdout


def test_local_page_and_assets_respond_while_cluster_connection_blocks(local_app, monkeypatch):
    main, runtime = local_app
    entered, release = threading.Event(), threading.Event()

    def loader():
        entered.set()
        assert release.wait(5), "Test did not release the blocked cluster connection"
        raise ConnectionError("private SSH connection detail")

    monkeypatch.setattr(runtime, "loader", loader)
    with TestClient(main.app) as client:
        assert entered.wait(2)
        try:
            with ThreadPoolExecutor(max_workers=1) as requests:
                responses = requests.submit(
                    lambda: [
                        client.get("/"),
                        client.get("/static/email-workspace.js"),
                        client.get("/api/workspace/session"),
                        client.get("/api/health"),
                    ]
                )
                try:
                    page, asset, session, health = responses.result(timeout=2)
                finally:
                    release.set()
            assert page.status_code == asset.status_code == 200
            for response in (session, health):
                assert response.status_code == 503
                assert response.json()["code"] == "starting"
                assert "initializing" in response.json()["detail"]
                assert "VPN" not in response.json()["detail"]
                assert response.headers["retry-after"] == "5"
                assert response.headers["cache-control"] == "no-store"
                assert "private SSH" not in response.text
        finally:
            release.set()
    assert runtime.owner is None


def test_recovery_keeps_cookie_and_does_not_replay_failed_mutations(local_app, monkeypatch):
    main, runtime = local_app
    second_attempt, release = threading.Event(), threading.Event()
    owner = Owner()
    attempts, mutations = [], []
    api = FastAPI()

    @api.get("/api/workspace/session")
    def session(request: Request):
        return {"token": request.cookies.get("skynet_workspace_session")}

    @api.post("/api/changes")
    def mutate():
        mutations.append("explicit request")
        return {"ok": True}

    def loader():
        attempts.append(True)
        if len(attempts) == 1:
            raise ConnectionError("VPN unavailable")
        second_attempt.set()
        assert release.wait(5), "Test did not permit cluster recovery"
        return api, owner

    monkeypatch.setattr(runtime, "loader", loader)
    with TestClient(main.app) as client:
        client.cookies.set("skynet_workspace_session", "existing-session")
        assert second_attempt.wait(2)
        try:
            unavailable = client.post("/api/changes", json={"intent": "once"})
            assert unavailable.status_code == 503
            assert mutations == []
            assert client.cookies.get("skynet_workspace_session") == "existing-session"
            assert owner.starts == 0
            release.set()
            assert owner.started.wait(2)
            deadline = time.monotonic() + 2
            while True:
                response = client.get("/api/workspace/session")
                if response.status_code == 200 or time.monotonic() >= deadline:
                    break
                time.sleep(0.01)
            assert response.status_code == 200
            assert response.json() == {"token": "existing-session"}
            assert client.get("/api/health").json()["ok"] is True
            assert mutations == [], "An operation rejected during outage was replayed"
            assert client.post("/api/changes").status_code == 200
            assert mutations == ["explicit request"]
            assert len(attempts) == 2
            assert owner.starts == 1
        finally:
            release.set()
    assert owner.stops == 1


@pytest.mark.parametrize("kind", ["connection", "postgres", "interface", "timeout", "cluster", "lock"])
def test_expected_connection_failures_retry_until_ready(kind):
    from psycopg import InterfaceError, OperationalError
    from psycopg.errors import LockNotAvailable
    from skynet_app.availability import ClusterApplication
    from skynet_app.cluster_runtime import ClusterError

    errors = {
        "connection": ConnectionError("connection failed"),
        "postgres": OperationalError("connection failed"),
        "interface": InterfaceError("connection failed"),
        "timeout": subprocess.TimeoutExpired("ssh", 1),
        "cluster": ClusterError("gateway failed"),
        "lock": LockNotAvailable("private database lock diagnostic"),
    }
    owner, api = Owner(), FastAPI()
    attempts, retry_responses = [], []

    def loader():
        attempts.append(True)
        if len(attempts) == 1:
            raise errors[kind]
        retry_responses.append(runtime.pending_response())
        return api, owner

    runtime = ClusterApplication(loader, retry_interval=0)
    anyio.run(runtime.connect)
    assert len(attempts) == 2
    assert runtime.application is api
    response = retry_responses[0]
    body = json.loads(response.body)
    assert response.status_code == 503
    assert body["code"] == ("startup_busy" if kind == "lock" else "cluster_unavailable")
    if kind == "lock":
        assert "earlier database operation" in body["detail"]
        assert "VPN" not in body["detail"]
        assert "private database" not in body["detail"]
    else:
        assert "Cannot access the Skynet cluster" in body["detail"]
    assert runtime.status == "ready"
    assert owner.starts == 1
    runtime.stop()
    assert owner.stops == 1


def test_configuration_failure_is_not_reported_as_a_connection_outage():
    from skynet_app.availability import ClusterApplication

    def loader():
        raise ValueError("Invalid local configuration")

    runtime = ClusterApplication(loader, retry_interval=0)
    with pytest.raises(ValueError, match="Invalid local configuration"):
        anyio.run(runtime.connect)
    assert runtime.application is None
    assert runtime.owner is None


def test_real_loader_builds_session_protected_api_and_docs(request, monkeypatch):
    # --noconftest intentionally exercises only tests that need no database.
    # An environment variable alone does not establish test DB isolation.
    if not getattr(request.config, "_skynet_bootstrap", None):
        pytest.skip("Requires the isolated PostgreSQL test plugin")
    from skynet_app import main, pipeline_api
    from skynet_app.database import Database
    from skynet_app.workspaces import WorkspaceServices

    # Another test may have imported the module-level coordinator against its
    # own disposable database. Build this test's coordinator while the current
    # PostgreSQL fixture is active instead of reusing that closed database.
    services = WorkspaceServices(pipeline_api.PipelineService(Database()))
    monkeypatch.setattr(pipeline_api, "service", services)

    api, owner = main._create_cluster_application()
    assert owner.database is services.system.database
    assert owner.thread is None, "Building API routes must not start background workers"
    with TestClient(api) as client:
        assert client.get("/api/data/resources").status_code == 401
        response = client.post("/api/workspace/session", json={"email": "offline-test@example.com"})
        assert response.status_code == 200
        assert client.get("/api/workspace/session").json()["workspace"]["email"] == "offline-test@example.com"
        assert client.get("/api/docs").status_code == 200
        schema = client.get("/openapi.json")
        assert schema.status_code == 200
        assert "/api/cluster" in schema.json()["paths"]
        assert "/api/workspace/session" in schema.json()["paths"]
    assert owner.thread is None


@pytest.mark.parametrize("timeout", [False, True])
def test_both_gateway_failures_return_friendly_cluster_error(monkeypatch, timeout):
    from skynet_app import main

    attempted = []

    def offline(host, command, **kwargs):
        attempted.append(host)
        if timeout:
            raise subprocess.TimeoutExpired("ssh internal details", 1)
        raise main.ClusterUnavailable("private SSH diagnostic")

    monkeypatch.setattr(main, "_ssh", offline)
    response = main.cluster("auto")
    assert attempted == list(main.SSH_HOSTS)
    assert response.status_code == 503
    assert b"Cannot access the Skynet cluster" in response.body
    assert b"private SSH" not in response.body
    assert b"internal details" not in response.body


def test_workspace_lookup_outage_does_not_block_local_page(local_app, monkeypatch):
    from types import SimpleNamespace
    from psycopg import OperationalError
    from skynet_app.workspaces import WorkspaceMiddleware

    main, runtime = local_app
    entered, release = threading.Event(), threading.Event()
    owner, api = Owner(), FastAPI()

    def lookup(token):
        entered.set()
        assert release.wait(5), "Test did not release unavailable workspace lookup"
        raise OperationalError("private database diagnostic")

    api.add_middleware(
        WorkspaceMiddleware,
        services=SimpleNamespace(directory=SimpleNamespace(resolve=lookup)),
    )
    monkeypatch.setattr(runtime, "loader", lambda: (api, owner))
    with TestClient(main.app) as client:
        assert owner.started.wait(2)
        client.cookies.set("skynet_workspace_session", "existing-session")
        with ThreadPoolExecutor(max_workers=2) as requests:
            private_request = requests.submit(client.get, "/api/private")
            try:
                assert entered.wait(2)
                public_request = requests.submit(client.get, "/")
                assert public_request.result(timeout=2).status_code == 200
            finally:
                release.set()
            response = private_request.result(timeout=2)
            assert response.status_code == 503
            assert response.json()["code"] == "database_unavailable"
            assert "Skynet cluster" in response.json()["detail"]
            assert "private database" not in response.text
            assert client.cookies.get("skynet_workspace_session") == "existing-session"
    assert owner.stops == 1


def test_cluster_query_uses_sky2_when_sky1_is_unavailable(monkeypatch):
    from skynet_app import main

    attempted = []
    monkeypatch.setattr(main, "SSH_HOSTS", ("sky1", "sky2"))

    def fallback(host, command, **kwargs):
        attempted.append(host)
        if host == "sky1":
            raise main.ClusterUnavailable("sky1 is unreachable")
        return "\n__SKYNET_JOBS__\n\n__SKYNET_USAGE__\n\n__SKYNET_USER_USAGE__\n"

    monkeypatch.setattr(main, "_ssh", fallback)
    result = main.cluster("auto")
    assert attempted == ["sky1", "sky2"]
    assert result["gateway"] == "sky2"
    assert result["jobs"] == []
    assert result["account_usage"] == []
