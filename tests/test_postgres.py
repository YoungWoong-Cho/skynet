"""Run against real PostgreSQL with SKYNET_TEST_POSTGRES_ADMIN configured."""

import concurrent.futures
import json
import os
import threading
import time
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from skynet_app.background_owner import BackgroundOwner
from skynet_app.database import Database
from skynet_app.db_backend import INTEGRITY_ERRORS
from skynet_app.postgres_migration import import_snapshot


@pytest.fixture
def pg(tmp_path):
    admin = os.environ.get("SKYNET_TEST_POSTGRES_ADMIN")
    if not admin:
        pytest.skip("Set SKYNET_TEST_POSTGRES_ADMIN to run real PostgreSQL checks")
    name = "skynet_test_" + uuid.uuid4().hex
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    url = make_conninfo(admin, dbname=name)
    try:
        yield Database(url=url, data_root=tmp_path), url
    finally:
        with psycopg.connect(admin, autocommit=True) as c:
            c.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
            )


def test_postgres_workspace_visibility_and_write_guards(pg):
    db, url = pg
    with db.transaction() as c:
        c.execute(
            "INSERT INTO workspaces(id,email) VALUES ('alice','alice@example.com'),('bob','bob@example.com')"
        )
    alice, bob = db.for_workspace("alice"), db.for_workspace("bob")
    project = alice.create_project("Shared name")
    bob.create_project("Shared name")
    assert [p["id"] for p in alice.list_projects()] == [project["id"]]
    assert project["id"] not in {p["id"] for p in bob.list_projects()}
    with pytest.raises(INTEGRITY_ERRORS):
        with bob.transaction() as c:
            c.execute(
                "UPDATE projects SET description=? WHERE id=?",
                ("escape", project["id"]),
            )
    assert alice.list_projects()[0]["description"] == ""


def test_postgres_rolls_back_full_transaction(pg):
    db, _ = pg
    with pytest.raises(INTEGRITY_ERRORS):
        with db.transaction() as c:
            c.execute(
                "INSERT INTO projects(id,name,created_at) VALUES ('first','first','now')"
            )
            c.execute(
                "INSERT INTO projects(id,name,created_at) VALUES ('second','first','now')"
            )
    assert db.list_projects() == []


def test_postgres_json_queries_and_literal_parameters(pg):
    db, _ = pg
    with db.connection() as c:
        row = c.execute(
            "SELECT json_extract(?, '$.source.revision') AS revision, '?' AS literal, '80%' AS progress",
            (json.dumps({"source": {"revision": "it's ? 50%"}}),),
        ).fetchone()
        assert row["revision"] == "it's ? 50%"
        assert row[1] == "?"
        assert row[2] == "80%"
        assert (
            list(
                c.execute(
                    "SELECT value FROM json_each(?, ?)", ('["running", {"a":2}]', "$")
                )
            )[0][0]
            == "running"
        )


def test_postgres_connections_see_committed_changes(pg):
    db, url = pg
    second = Database(url=url)
    project = db.create_project("Across hosts")
    assert second.list_projects()[0]["id"] == project["id"]


def test_postgres_operation_lock_excludes_other_clients_and_is_reentrant(pg):
    db, url = pg
    lock = db.operation_lock("test-submit")
    other = Database(url=url).operation_lock("test-submit")
    with lock:
        with lock:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                assert pool.submit(other.acquire, False).result() is False
    assert other.acquire(False)
    other.release()


def test_postgres_transactions_serialize_read_modify_write(pg):
    db, url = pg
    with db.transaction() as c:
        c.execute("CREATE TABLE test_counter(value INTEGER)")
        c.execute("INSERT INTO test_counter VALUES (0)")
    clients = [Database(url=url) for _ in range(4)]

    def increment(client):
        for _ in range(4):
            with client.transaction() as c:
                value = c.execute("SELECT value FROM test_counter").fetchone()[0]
                time.sleep(0.01)
                c.execute("UPDATE test_counter SET value=?", (value + 1,))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(increment, clients))
    with db.connection() as c:
        assert c.execute("SELECT value FROM test_counter").fetchone()[0] == 16


def eventually(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    assert predicate()


def test_only_one_background_owner_and_standby_takes_over(pg):
    db, url = pg
    active = set()
    guard = threading.Lock()
    overlap = []

    def start(name):
        with guard:
            if active:
                overlap.append(True)
            active.add(name)

    def stop(name):
        with guard:
            active.discard(name)

    owners = [
        BackgroundOwner(
            Database(url=url), lambda n=n: start(n), lambda n=n: stop(n), interval=0.03
        )
        for n in range(2)
    ]
    try:
        for owner in owners:
            owner.start()
        eventually(lambda: len(active) == 1)
        first = next(iter(active))
        owners[first].stop()
        eventually(lambda: active == {1 - first})
        assert not overlap
    finally:
        for owner in owners:
            owner.stop()
    assert not active


def test_postgres_import_roundtrip_and_nonempty_refusal(pg, tmp_path):
    db, url = pg
    source = Database(tmp_path / "source.db")
    source.create_project("Preserved")
    # Full import intentionally rejects a partial/mismatched source schema.
    from skynet_app.collection import CollectionStore
    from skynet_app.source_metadata_cache import SourceMetadataStore

    CollectionStore(source)
    SourceMetadataStore(source)
    report = import_snapshot(source.path, url)
    assert report["tables"]["projects"]["rows"] == 1
    assert db.list_projects()[0]["name"] == "Preserved"
    with pytest.raises(ValueError, match="not empty"):
        import_snapshot(source.path, url)


def test_no_fallback_when_postgres_unreachable(tmp_path):
    with pytest.raises(psycopg.OperationalError):
        Database(
            url="host=/nonexistent/skynet-postgres dbname=skynet connect_timeout=1",
            data_root=tmp_path,
        )
    assert not (tmp_path / "skynet.db").exists()


def test_tracking_queue_and_cursor_are_shared_between_hosts(pg, tmp_path):
    from skynet_app.tracking import WandBBridge, WandBSettings
    from skynet_app.tracking_journal import TrackingJournal

    db, url = pg
    first = WandBBridge(
        tmp_path / "host-a",
        WandBSettings(auto_flush=False),
        journal=TrackingJournal(db, "shared-test"),
    )
    second = WandBBridge(
        tmp_path / "host-b",
        WandBSettings(auto_flush=False),
        journal=TrackingJournal(Database(url=url), "shared-test"),
    )
    first._enqueue("test", {"idempotency_key": "one", "metric": 3})
    second._enqueue("test", {"idempotency_key": "one", "metric": 3})
    assert first.pending_count() == second.pending_count() == 1
    with first._locked():
        state = first._load_state_unlocked()
        state["acked_through"] = 1
        first._write_state_unlocked(state)
    assert second.pending_count() == 0
    assert not (tmp_path / "host-a" / "wandb-state.json").exists()


def test_background_workers_stop_when_ownership_connection_is_lost(pg):
    db, url = pg
    started = threading.Event()
    stopped = threading.Event()
    owner = BackgroundOwner(db, started.set, stopped.set, interval=0.03)
    from skynet_app.db_backend import lock_key

    key = lock_key("background-owner") & ((1 << 64) - 1)
    try:
        owner.start()
        assert started.wait(5)
        with psycopg.connect(url, autocommit=True) as connection:
            pid = connection.execute(
                "SELECT pid FROM pg_locks WHERE locktype='advisory' AND classid=%s AND objid=%s AND granted",
                (key >> 32, key & 0xFFFFFFFF),
            ).fetchone()[0]
            connection.execute("SELECT pg_terminate_backend(%s)", (pid,))
        assert stopped.wait(5)
        eventually(lambda: owner.state == "lost")
    finally:
        owner.stop()


def test_tracking_journal_cannot_attach_another_workspace_run(pg):
    db, _ = pg
    with db.transaction() as connection:
        connection.execute(
            "INSERT INTO workspaces(id,email) VALUES ('alice','alice@example.com'),('bob','bob@example.com')"
        )
    # The FK ownership guard is also enforced when a system caller explicitly
    # chooses a journal owner. No cross-workspace write can be smuggled through.
    with (
        pytest.raises(INTEGRITY_ERRORS),
        db.for_workspace("bob").transaction() as connection,
    ):
        connection.execute(
            "INSERT INTO tracking_journals(owner_id,scope,filename,payload) VALUES ('alice','test','wandb-state.json',?)",
            (b"{}",),
        )


def test_endpoint_rejects_invalid_configuration_before_ssh(tmp_path, monkeypatch):
    import subprocess

    from skynet_app.database_endpoint import load_endpoint

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid configuration must not start SSH")

    monkeypatch.setattr(subprocess, "Popen", unexpected)
    config = tmp_path / "database.json"
    config.write_text(
        json.dumps(
            {
                "backend": "postgresql",
                "transport": "ssh-unix",
                "ssh_host": "-oProxyCommand=bad",
                "remote_socket": "/safe/socket",
                "user": "user",
            }
        )
    )
    with pytest.raises(ValueError):
        load_endpoint(config)
