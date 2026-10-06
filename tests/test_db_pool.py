"""Closed connections return to a pool and come back reset, never carrying session state across callers."""
import time

import psycopg

from skynet_app import db_backend
from test_postgres import pg  # noqa: F401

from skynet_app.db_backend import PostgresBackend, close_pools


def test_closed_connections_are_reused_and_reset(pg, monkeypatch):
    db, url = pg
    backend = PostgresBackend(url)
    first = backend.connect()
    raw = first.raw
    first.execute("LISTEN skynet_pool_test")
    first.execute("SET statement_timeout = '7s'")
    first.execute("SELECT pg_advisory_lock(424242)")
    first.raw.execute("BEGIN")  # an open transaction is rolled back on return
    first.close()
    first.close()  # idempotent

    second = backend.connect()
    assert second.raw is raw, "the same server session is reused"
    assert second.raw.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
    assert second.execute("SHOW statement_timeout").fetchone()[0] == "0"
    assert second.execute("SHOW lock_timeout").fetchone()[0] == "30s", "startup options survive the reset"
    assert second.execute("SELECT count(*) FROM pg_listening_channels()").fetchone()[0] == 0
    other = backend._open()
    try:
        assert other.execute("SELECT pg_try_advisory_lock(424242)").fetchone()[0], "the session lock was released"
        other.execute("SELECT pg_advisory_unlock_all()")
    finally:
        other.close()
    second.close()


def test_workspace_follows_the_caller_not_the_pooled_session(pg):
    db, url = pg
    alice = PostgresBackend(url, workspace_id="alice")
    everyone = PostgresBackend(url)
    connection = alice.connect()
    raw = connection.raw
    assert connection.execute("SELECT current_setting('skynet.workspace_id', true)").fetchone()[0] == "alice"
    connection.close()
    connection = everyone.connect()
    assert connection.raw is raw
    assert connection.execute("SELECT current_setting('skynet.workspace_id', true)").fetchone()[0] == ""
    connection.close()


def test_broken_or_stale_connections_are_not_reused(pg, monkeypatch):
    db, url = pg
    backend = PostgresBackend(url)
    connection = backend.connect()
    raw = connection.raw
    connection.close()
    now = time.monotonic
    monkeypatch.setattr(db_backend.time, "monotonic", lambda: now() + db_backend.POOL_IDLE_SECONDS + 1)
    fresh = backend.connect()
    assert fresh.raw is not raw and raw.closed, "an idle connection past its limit is closed, not reused"
    broken = fresh.raw
    fresh.close()
    broken.close()  # simulate the server dropping the session
    assert backend.connect().raw is not broken
    assert len(backend.pool.idle) <= db_backend.POOL_IDLE_CONNECTIONS
    close_pools()
    assert backend.pool.idle == []
