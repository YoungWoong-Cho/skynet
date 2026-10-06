"""PostgreSQL transport for the repository's parameterized SQL.

Only placeholder syntax is adapted here; schema and transaction differences are
explicit. The application requires PostgreSQL and never creates a host-local database.
"""

from __future__ import annotations

import atexit
import hashlib
import re
import threading
import time
from pathlib import Path

import psycopg
from .cluster_config import CLUSTER

INTEGRITY_ERRORS = (psycopg.IntegrityError,)
DATABASE_ERRORS = (psycopg.Error,)


class Record:
    """A database row with both column-name and positional access."""

    def __init__(self, names, values, payload_store=None):
        self._values = tuple(values)
        self._data = dict(zip(names, values))
        self.payload_store = payload_store

    def __getitem__(self, key):
        value = self._values[key] if isinstance(key, (int, slice)) else self._data[key]
        return self.payload_store.read(value) if self.payload_store and not isinstance(key, slice) else value

    def keys(self):
        return self._data.keys()

    def __iter__(self):
        return (self[index] for index in range(len(self._values)))

    def __len__(self):
        return len(self._data)


def record_factory(cursor):
    names = [column.name for column in cursor.description] if cursor.description else []
    return lambda values: Record(names, values)


# Preserve question marks and percent signs in quoted SQL, comments and dollar
# strings. Values are always bound separately by psycopg.
_SQL_TOKENS = re.compile(
    r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|--[^\n]*(?:\n|$)|/\*.*?\*/|\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$.*?\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$)",
    re.DOTALL,
)


def bind_sql(statement: str) -> str:
    parts = _SQL_TOKENS.split(statement)
    return "".join(
        part.replace("%", "%%")
        if index % 2
        else part.replace("%", "%%").replace("?", "%s")
        for index, part in enumerate(parts)
    )


class PostgresConnection:
    dialect = "postgresql"

    def __init__(self, connection):
        self.raw = connection
        self.payload_store = None
        self.pool = None
        self._released = False

    def execute(self, statement, parameters=None):
        parameters = (
            tuple(
                int(value) if isinstance(value, bool) else value for value in parameters
            )
            if parameters is not None
            else None
        )
        cursor = self.raw.execute(
            bind_sql(statement) if parameters is not None else statement, parameters
        )
        return PayloadCursor(cursor, self.payload_store) if self.payload_store else cursor

    def executemany(self, statement, parameters):
        cursor = self.raw.cursor()
        cursor.executemany(bind_sql(statement), parameters)
        return cursor

    def executescript(self, statement):
        # PostgreSQL multi-statement execution remains in the caller's transaction.
        return self.raw.execute(statement, prepare=False)

    def commit(self):
        self.raw.commit()

    def rollback(self):
        self.raw.rollback()

    def close(self):
        """Return the session to its pool (or close an unpooled one); later calls do nothing."""
        if self._released:
            return
        self._released = True
        pool, self.pool = self.pool, None
        if pool is not None:
            pool.give(self.raw)
        else:
            self.raw.close()

    def discard(self):
        """Close the session for good: after an interrupted transaction it is not reused."""
        self._released = True
        self.pool = None
        self.raw.close()


class PayloadCursor:
    def __init__(self, cursor, store):
        self.cursor, self.store = cursor, store

    @property
    def rowcount(self):
        return self.cursor.rowcount

    def fetchone(self):
        row = self.cursor.fetchone()
        if row is not None:
            row.payload_store = self.store
        return row

    def fetchall(self):
        rows = self.cursor.fetchall()
        self.store.prefetch(value for row in rows for value in row._values)
        for row in rows:
            row.payload_store = self.store
        return rows

    def __iter__(self):
        return iter(self.fetchall())


def lock_key(name: str) -> int:
    return int.from_bytes(
        hashlib.sha256(("skynet:" + name).encode()).digest()[:8], "big", signed=True
    )


# Opening a connection through the tunnel costs a few hundred milliseconds and
# a query a few, so closed connections return to a per-database pool. A returned
# connection is reset to its startup state (RESET ALL restores the -c options
# below); one kept idle longer than this is closed instead of reused.
POOL_IDLE_SECONDS = 60
POOL_IDLE_CONNECTIONS = 8
_POOLS: dict = {}
_POOLS_LOCK = threading.Lock()


def _close_quietly(raw):
    try:
        raw.close()
    except Exception:
        pass


class _ConnectionPool:
    def __init__(self):
        self.idle = []  # (connection, returned_at), most recently returned last
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            while self.idle:
                raw, returned_at = self.idle.pop()
                if raw.closed or raw.broken:
                    continue
                if time.monotonic() - returned_at > POOL_IDLE_SECONDS:
                    _close_quietly(raw)
                    continue
                return raw
        return None

    def give(self, raw):
        if raw.closed or raw.broken:
            _close_quietly(raw)
            return
        try:
            if raw.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
                raw.rollback()
            # Session locks, listeners and settings must not outlive the caller.
            raw.execute("UNLISTEN *; RESET ALL", prepare=False)
            raw.execute("SELECT pg_advisory_unlock_all()")
        except Exception:
            _close_quietly(raw)
            return
        with self.lock:
            if len(self.idle) >= POOL_IDLE_CONNECTIONS:
                _close_quietly(raw)
                return
            self.idle.append((raw, time.monotonic()))

    def close(self):
        with self.lock:
            idle, self.idle = self.idle, []
        for raw, _ in idle:
            _close_quietly(raw)


def close_pools():
    """Close every idle pooled connection (process exit, test database teardown)."""
    with _POOLS_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        pool.close()


atexit.register(close_pools)


class PostgresBackend:
    def __init__(self, url: str, workspace_id=None):
        self.url, self.workspace_id = url, workspace_id

    @property
    def pool(self):
        # One pool per database; workspace-scoped backends share their database's.
        key = self.url if isinstance(self.url, str) else id(self.url)
        with _POOLS_LOCK:
            pool = _POOLS.get(key)
            if pool is None:
                pool = _POOLS[key] = _ConnectionPool()
        return pool

    def _open(self):
        url = (
            self.url.connection_string()
            if hasattr(self.url, "connection_string")
            else self.url
        )
        return psycopg.connect(
            url,
            autocommit=True,
            row_factory=record_factory,
            connect_timeout=CLUSTER.ssh.tunnel.connect_timeout_seconds,
            application_name="skynet",
            # A disconnected SSH client can leave the server transaction open.
            # Bound that idle lifetime so its repository write lock cannot
            # indefinitely block unrelated requests. Active SQL is unaffected.
            options=(
                "-c timezone=UTC -c lock_timeout=30000 "
                "-c idle_in_transaction_session_timeout=120000"
            ),
        )

    def connect(self):
        pool = self.pool
        raw = pool.take()
        if raw is None:
            raw = self._open()
        try:
            raw.execute(
                "SELECT set_config('skynet.workspace_id', %s, false)",
                (self.workspace_id or "",),
            )
        except BaseException:
            _close_quietly(raw)
            raise
        connection = PostgresConnection(raw)
        connection.pool = pool
        return connection

    def initialize(self):
        connection = self.connect()
        try:
            with connection.raw.transaction():
                connection.execute(
                    "SELECT pg_advisory_xact_lock(?)", (lock_key("schema"),)
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS skynet_schema_migrations(version INTEGER PRIMARY KEY, checksum TEXT NOT NULL)"
                )
                for path in sorted(
                    (Path(__file__).parent / "migrations" / "postgresql").glob("*.sql")
                ):
                    version = int(path.name.split("_", 1)[0])
                    content = path.read_text()
                    checksum = hashlib.sha256(content.encode()).hexdigest()
                    row = connection.execute(
                        "SELECT checksum FROM skynet_schema_migrations WHERE version=?",
                        (version,),
                    ).fetchone()
                    if row:
                        if row[0] != checksum:
                            raise RuntimeError(
                                f"PostgreSQL migration {version} checksum changed"
                            )
                    else:
                        connection.executescript(content)
                        connection.execute(
                            "INSERT INTO skynet_schema_migrations VALUES (?,?)",
                            (version, checksum),
                        )
        finally:
            connection.close()


class DistributedRLock:
    """Reentrant process/thread lock backed by a PostgreSQL session lock.

    The server releases a lock when its connection closes. A failed acquire
    never permits the protected operation to proceed.
    """

    def __init__(self, backend, name):
        self.backend, self.key = backend, lock_key(name)
        self.local = threading.RLock()
        self.state = threading.local()

    def acquire(self, blocking=True, timeout=-1):
        acquired = (
            self.local.acquire(blocking)
            if timeout == -1
            else self.local.acquire(blocking, timeout)
        )
        if not acquired:
            return False
        depth = getattr(self.state, "depth", 0)
        if depth:
            self.state.depth = depth + 1
            return True
        connection = None
        try:
            connection = self.backend.connect()
            if blocking:
                # Operation-lock waiting follows acquire(timeout), not the
                # separate 30-second limit for SQL row/write-lock contention.
                connection.execute(
                    "SELECT set_config('lock_timeout', ?, false)",
                    (str(max(1, int(timeout * 1000))) if timeout >= 0 else "0",),
                )
                connection.execute("SELECT pg_advisory_lock(?)", (self.key,))
                connection.execute("SELECT set_config('lock_timeout', '30000', false)")
            elif not connection.execute(
                "SELECT pg_try_advisory_lock(?)", (self.key,)
            ).fetchone()[0]:
                connection.close()
                self.local.release()
                return False
            self.state.connection, self.state.depth = connection, 1
            return True
        except BaseException:
            if connection:
                connection.close()
            self.local.release()
            raise

    def release(self):
        depth = getattr(self.state, "depth", 0)
        if not depth:
            raise RuntimeError("cannot release an unowned lock")
        self.state.depth -= 1
        try:
            if depth == 1:
                connection = self.state.connection
                try:
                    connection.execute("SELECT pg_advisory_unlock(?)", (self.key,))
                finally:
                    connection.close()
                    del self.state.connection
        finally:
            self.local.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, error, traceback):
        try:
            self.release()
        except Exception as cleanup_error:
            if error is None:
                raise
            error.add_note(f"Database lock release also failed: {cleanup_error}")
