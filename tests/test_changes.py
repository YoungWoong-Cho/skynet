"""Committed invalidations and SSE recovery against isolated PostgreSQL fixtures."""

import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from skynet_app.changes import CHANNEL, ChangeFeed, change_router
from skynet_app.database import Database
from skynet_app.db_backend import lock_key
from skynet_app.workspaces import COOKIE, WorkspaceDirectory, WorkspaceMiddleware


@pytest.fixture
def database(tmp_path):
    return Database(tmp_path / "changes.db")


@contextmanager
def listening(database):
    connection = database.backend.connect()
    connection.raw.execute("LISTEN " + CHANNEL)
    try:
        yield connection.raw
    finally:
        connection.close()


def notifications(listener):
    return [json.loads(item.payload) for item in listener.notifies(timeout=0.1)]


def export_write(database, state="QUEUED"):
    with database.connection() as connection:
        connection.execute(
            "INSERT INTO policy_exports VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload_json=excluded.payload_json",
            ("export-fixture", json.dumps({"state": state, "updated_at": "one", "checked_at": "one"})),
        )


def test_notify_is_commit_only_and_rollback_noops_do_not_trigger_refreshes(database):
    with listening(database) as listener, database.connection() as writer:
        with writer.raw.transaction():
            writer.execute("INSERT INTO policy_exports VALUES (?,?)", ("job", '{"state":"QUEUED","updated_at":"one","checked_at":"one"}'))
            assert notifications(listener) == []
        assert notifications(listener) == [{"v": 1, "scope": "*", "topics": ["exports"]}]
        with pytest.raises(RuntimeError):
            with writer.raw.transaction():
                writer.execute("UPDATE policy_exports SET payload_json=? WHERE id='job'", ('{"state":"FAILED"}',))
                raise RuntimeError("rollback fixture")
        assert notifications(listener) == []
        writer.execute("UPDATE policy_exports SET payload_json=? WHERE id='job'", ('{"state":"QUEUED","updated_at":"two","checked_at":"two"}',))
        assert notifications(listener) == []
        writer.execute("UPDATE policy_exports SET payload_json=payload_json WHERE id='job'")
        assert notifications(listener) == []
        with writer.raw.transaction():
            writer.execute("UPDATE policy_exports SET payload_json=? WHERE id='job'", ('{"state":"RUNNING"}',))
            writer.execute("UPDATE policy_exports SET payload_json=? WHERE id='job'", ('{"state":"READY"}',))
        # Multiple same-topic writes in one transaction are one invalidation.
        assert notifications(listener) == [{"v": 1, "scope": "*", "topics": ["exports"]}]
        writer.execute("DELETE FROM policy_exports WHERE id='job'")
        assert notifications(listener) == [{"v": 1, "scope": "*", "topics": ["exports"]}]


def test_notification_scopes_cover_private_shared_legacy_seed_and_session(database):
    directory = WorkspaceDirectory(database)
    alice, token = directory.open("alice@example.com")
    bob, _ = directory.open("bob@example.com")
    private = database.for_workspace(alice["id"])
    legacy = database.for_workspace("legacy")
    with listening(database) as listener:
        private.upsert_tracking_connection("mlflow", endpoint="https://fixture.test", config={"authentication": "none"})
        assert notifications(listener) == [{"v": 1, "scope": alice["id"], "topics": ["settings"]}]
        with private.connection() as writer:
            writer.execute("UPDATE tracking_connections SET updated_at='clock-only' WHERE provider='mlflow'")
        assert notifications(listener) == []
        private.create_adapter(name="private", manifest={})
        assert notifications(listener) == [{"v": 1, "scope": alice["id"], "topics": ["adapters"]}]
        shared = database.create_adapter(name="shared", manifest={})
        assert notifications(listener) == [{"v": 1, "scope": "legacy", "topics": ["adapters"]}]
        with database.connection() as writer:
            writer.execute("UPDATE adapters SET owner_id=NULL WHERE adapter_key=?", (shared["id"],))
        assert {event["scope"] for event in notifications(listener)} == {"legacy", "*"}
        with database.connection() as writer:
            writer.execute("UPDATE adapters SET enabled=0 WHERE adapter_key=?", (shared["id"],))
        assert notifications(listener) == [{"v": 1, "scope": "*", "topics": ["adapters"]}]
        legacy.upsert_seed_adapter(seed_key="fixture", name="legacy seed", manifest={})
        assert notifications(listener) == [{"v": 1, "scope": "*", "topics": ["adapters"]}]
        legacy.create_adapter(name="legacy personal", manifest={})
        assert notifications(listener) == [{"v": 1, "scope": "legacy", "topics": ["adapters"]}]
        with database.connection() as writer:
            writer.execute("INSERT INTO live_xr_sessions VALUES (?,?)", ("live", '{"state":"RUNNING"}'))
        assert notifications(listener) == [{"v": 1, "scope": "*", "topics": ["recordings"]}]
        directory.close(token)
        assert notifications(listener) == [{"v": 1, "scope": alice["id"], "topics": ["session"]}]
        assert directory.resolve(token) is None
        assert bob["id"] != alice["id"]


def test_rollback_and_reapply_only_remove_notification_objects(database):
    root = Path(__file__).resolve().parents[1]
    rollback = (root / "deploy/rollback-change-notifications.sql").read_text()
    assert str(lock_key("schema")) in rollback
    export_write(database)
    with listening(database) as listener, database.connection() as connection:
        before = connection.execute("SELECT payload_json FROM policy_exports WHERE id='export-fixture'").fetchone()[0]
        connection.executescript(rollback)
        assert connection.execute("SELECT to_regprocedure('skynet_notify_change()')").fetchone()[0] is None
        assert connection.execute("SELECT count(*) FROM skynet_schema_migrations WHERE version=11").fetchone()[0] == 0
        assert connection.execute("SELECT payload_json FROM policy_exports WHERE id='export-fixture'").fetchone()[0] == before
        export_write(database, "RUNNING")
        assert notifications(listener) == []
        database.backend.initialize()
        assert connection.execute("SELECT count(*) FROM skynet_schema_migrations WHERE version=11").fetchone()[0] == 1
        export_write(database, "READY")
        assert notifications(listener) == [{"v": 1, "scope": "*", "topics": ["exports"]}]
        assert json.loads(connection.execute("SELECT payload_json FROM policy_exports WHERE id='export-fixture'").fetchone()[0])["state"] == "READY"


def test_feed_filters_workspace_and_overflow_forces_resync(database):
    async def scenario():
        feed = ChangeFeed(database)
        with pytest.raises(HTTPException) as error:
            feed.subscribe("alice")
        assert error.value.status_code == 503
        feed.ready.set()
        alice, bob = feed.subscribe("alice"), feed.subscribe("bob")
        assert (await alice.queue.get())["event"] == "resync"
        assert (await bob.queue.get())["event"] == "resync"
        feed._notification(json.dumps({"v": 1, "scope": "alice", "topics": ["settings"]}))
        await asyncio.sleep(0)
        assert (await alice.queue.get()) == {"event": "change", "data": {"v": 1, "topics": ["settings"]}}
        assert bob.queue.empty()
        feed._notification(json.dumps({"v": 1, "scope": "*", "topics": ["exports"]}))
        await asyncio.sleep(0)
        assert (await alice.queue.get())["data"]["topics"] == ["exports"]
        assert (await bob.queue.get())["data"]["topics"] == ["exports"]
        feed._notification(json.dumps({"v": 1, "scope": "bob", "topics": ["session"]}))
        await asyncio.sleep(0)
        assert (await bob.queue.get())["event"] == "revalidate"
        assert alice.queue.empty()
        for _ in range(17):
            alice.push({"event": "change", "data": {"v": 1, "topics": ["data"]}})
        assert alice.queue.qsize() == 1
        assert (await alice.queue.get())["event"] == "resync"
        for _ in range(16):
            alice.push({"event": "change", "data": {"v": 1, "topics": ["data"]}})
        alice.push({"event": "unavailable", "data": {"v": 1}})
        assert (await alice.queue.get())["event"] == "unavailable"
        feed.unsubscribe(alice)
        alice.push({"event": "resync", "data": {"v": 1}})
        assert alice.queue.empty()
        assert alice not in feed.subscriptions
        feed.unsubscribe(bob)
    asyncio.run(scenario())


@pytest.mark.parametrize("payload", ["[]", "null", "1", '"text"', "{", '{"v":2,"scope":"alice","topics":["data"]}', '{"v":1,"scope":"alice","topics":["unknown"]}'])
def test_unknown_envelope_resyncs_without_crashing_or_forwarding_contents(database, payload):
    async def scenario():
        feed = ChangeFeed(database)
        feed.ready.set()
        subscription = feed.subscribe("alice")
        await subscription.queue.get()
        try:
            feed._notification(payload)
            await asyncio.sleep(0)
            assert subscription.queue.get_nowait() == {"event": "resync", "data": {"v": 1}}
        finally:
            feed.unsubscribe(subscription)
    asyncio.run(scenario())


def test_real_listener_recovers_with_new_subscription_after_database_disconnect(database, monkeypatch):
    async def scenario():
        feed = ChangeFeed(database, reconnect_delay=0.05, heartbeat=0.1)
        connections = []
        original = database.backend.connect

        def connect():
            connection = original()
            connections.append(connection.raw.info.backend_pid)
            return connection

        monkeypatch.setattr(database.backend, "connect", connect)
        async def wait_for(predicate):
            async def poll():
                while not predicate():
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(poll(), timeout=5)

        feed.start()
        try:
            await wait_for(feed.ready.is_set)
            first = feed.subscribe("alice")
            assert (await first.queue.get())["event"] == "resync"
            await asyncio.to_thread(export_write, database)
            assert (await asyncio.wait_for(first.queue.get(), 3))["data"]["topics"] == ["exports"]
            pid = connections[0]
            with database.connection() as control:
                assert control.execute("SELECT pg_terminate_backend(?)", (pid,)).fetchone()[0]
            assert (await asyncio.wait_for(first.queue.get(), 3))["event"] == "unavailable"
            feed.unsubscribe(first)
            await wait_for(lambda: feed.ready.is_set() and len(connections) >= 2)
            recovered = feed.subscribe("alice")
            assert (await recovered.queue.get())["event"] == "resync"
            await asyncio.to_thread(export_write, database, "READY")
            assert (await asyncio.wait_for(recovered.queue.get(), 3))["data"]["topics"] == ["exports"]
            feed.unsubscribe(recovered)
        finally:
            await asyncio.to_thread(feed.stop)
        assert not feed.ready.is_set()
        assert not feed.thread.is_alive()
        assert not feed.subscriptions
    asyncio.run(scenario())


def test_router_rejects_missing_or_changed_workspace_without_opening_stream(database):
    directory = WorkspaceDirectory(database)
    workspace, token = directory.open("alice@example.com")
    feed = ChangeFeed(database)
    feed.ready.set()
    app = FastAPI()
    app.add_middleware(WorkspaceMiddleware, services=SimpleNamespace(directory=directory))
    app.include_router(change_router(feed, directory))
    with TestClient(app) as client:
        assert client.get("/api/changes?expected_workspace=" + workspace["id"]).status_code == 401
        client.cookies.set(COOKIE, token)
        assert client.get("/api/changes").status_code == 422
        assert client.get("/api/changes?expected_workspace=other").status_code == 409
        feed.ready.clear()
        assert client.get("/api/changes?expected_workspace=" + workspace["id"]).status_code == 503
    assert not feed.subscriptions


@pytest.mark.parametrize("ending", ["disconnect", "cancel", "unavailable", "revoked"])
def test_stream_resync_and_all_exit_paths_release_subscription(database, ending):
    async def scenario():
        directory = WorkspaceDirectory(database)
        workspace, token = directory.open("alice@example.com")
        feed = ChangeFeed(database)
        feed.ready.set()
        disconnected = False

        async def receive():
            return {"type": "http.disconnect"} if disconnected else {"type": "http.request", "body": b"", "more_body": False}

        request = Request({
            "type": "http", "http_version": "1.1", "method": "GET", "scheme": "http",
            "path": "/api/changes", "query_string": b"", "server": ("testserver", 80),
            "headers": [(b"cookie", f"{COOKIE}={token}".encode())], "state": {"workspace": workspace},
        }, receive=receive)
        endpoint = next(route.endpoint for route in change_router(feed, directory).routes if route.path == "/api/changes")
        response = await endpoint(request, workspace["id"])
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "private, no-store"
        iterator = response.body_iterator
        assert await anext(iterator) == 'event: resync\ndata: {"v":1}\n\n'
        assert len(feed.subscriptions) == 1
        if ending == "disconnect":
            disconnected = True
        elif ending == "cancel":
            pending = asyncio.create_task(anext(iterator))
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        elif ending == "unavailable":
            feed._unavailable()
            assert await anext(iterator) == 'event: unavailable\ndata: {"v":1}\n\n'
        else:
            directory.close(token)
            feed._notification(json.dumps({"v": 1, "scope": workspace["id"], "topics": ["session"]}))
            assert await anext(iterator) == 'event: unavailable\ndata: {"v":1}\n\n'
        if ending != "cancel":
            with pytest.raises(StopAsyncIteration):
                await anext(iterator)
        assert not feed.subscriptions
    asyncio.run(scenario())


@pytest.mark.parametrize("failed_message", ["http.response.start", "http.response.body"])
def test_asgi_send_failure_releases_subscription_before_or_after_first_event(database, failed_message):
    from starlette.requests import ClientDisconnect

    async def scenario():
        directory = WorkspaceDirectory(database)
        workspace, token = directory.open("alice@example.com")
        feed = ChangeFeed(database)
        feed.ready.set()

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            if message["type"] == failed_message:
                raise OSError("client connection closed")

        scope = {
            "type": "http", "http_version": "1.1", "asgi": {"spec_version": "2.4"},
            "method": "GET", "scheme": "http", "path": "/api/changes", "query_string": b"",
            "server": ("testserver", 80), "headers": [(b"cookie", f"{COOKIE}={token}".encode())],
            "state": {"workspace": workspace},
        }
        request = Request(scope, receive=receive)
        endpoint = next(route.endpoint for route in change_router(feed, directory).routes if route.path == "/api/changes")
        response = await endpoint(request, workspace["id"])
        try:
            with pytest.raises(ClientDisconnect):
                await response(scope, receive, send)
            assert not feed.subscriptions
        finally:
            await response.body_iterator.aclose()
            for subscription in tuple(feed.subscriptions):
                feed.unsubscribe(subscription)
    asyncio.run(scenario())


def test_standby_app_keeps_its_change_feed_without_starting_job_services(database):
    from skynet_app.background_owner import BackgroundOwner

    async def scenario():
        first_feed = ChangeFeed(database, reconnect_delay=0.05)
        second_feed = ChangeFeed(database, reconnect_delay=0.05)
        calls = []
        first = BackgroundOwner(database, lambda: calls.append("first start"), lambda: calls.append("first stop"), interval=0.01, companions=(first_feed,))
        second = BackgroundOwner(database, lambda: calls.append("second start"), lambda: calls.append("second stop"), interval=0.01, companions=(second_feed,))

        async def wait_for(predicate):
            async def poll():
                while not predicate():
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(poll(), 5)

        first.start()
        try:
            await wait_for(lambda: first.state == "active" and first_feed.ready.is_set())
            second.start()
            await wait_for(lambda: second.state == "standby" and second_feed.ready.is_set())
            assert calls == ["first start"]
            subscriptions = [(first_feed, first_feed.subscribe("alice")), (second_feed, second_feed.subscribe("alice"))]
            try:
                for _, subscription in subscriptions:
                    assert (await subscription.queue.get())["event"] == "resync"
                await asyncio.to_thread(export_write, database)
                for _, subscription in subscriptions:
                    assert (await asyncio.wait_for(subscription.queue.get(), 3))["data"]["topics"] == ["exports"]
            finally:
                for feed, subscription in subscriptions:
                    feed.unsubscribe(subscription)
        finally:
            await asyncio.to_thread(second.stop)
            await asyncio.to_thread(first.stop)
        assert calls == ["first start", "first stop"]
        assert not first_feed.thread.is_alive()
        assert not second_feed.thread.is_alive()
    asyncio.run(scenario())


def test_private_experiment_usage_invalidates_shared_catalog_without_private_body(database):
    directory = WorkspaceDirectory(database)
    workspace, _ = directory.open("alice@example.com")
    private = database.for_workspace(workspace["id"])
    project = private.create_project("private project fixture")
    with listening(database) as listener:
        experiment = private.create_experiment(
            name="private experiment fixture", project_id=project["id"],
            requested_spec={"private_parameter": "private-spec-body"},
        )
        assert notifications(listener) == [{"v": 1, "scope": "*", "topics": ["data", "exports"]}]
        variant = private.create_variant(
            experiment["latest_revision"]["id"], name="private variant fixture",
            parameters={"private_parameter": 42}, resolved_spec={"private_parameter": "private-resolved-body"},
        )
        assert notifications(listener) == [{"v": 1, "scope": "*", "topics": ["data", "exports"]}]
        run = private.create_run(
            variant["id"], seed=42, adapter_name="fixture", adapter_version="1",
            run_directory="/private-fixture/jobs/run", status="PENDING",
        )
        assert notifications(listener) == [{"v": 1, "scope": "*", "topics": ["exports"]}]
        private.update_run(run["id"], status="RUNNING")
        assert notifications(listener) == [{"v": 1, "scope": "*", "topics": ["exports"]}]
        private.update_run(run["id"], status="RUNNING")
        assert notifications(listener) == []
        private.update_experiment(experiment["id"], name="renamed private fixture")
        assert notifications(listener) == [{"v": 1, "scope": "*", "topics": ["data", "exports"]}]
