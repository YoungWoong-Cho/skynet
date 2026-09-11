"""SQLite ownership and sessions for trusted-team email workspaces."""

from __future__ import annotations

import re
import sqlite3


LEGACY_WORKSPACE = "legacy"
PRIVATE_TABLES = frozenset({
    "projects", "experiments", "experiment_revisions", "variants", "runs",
    "workflow_stages", "job_attempts", "checkpoints", "evaluations",
    "evaluation_episodes", "metrics", "training_progress_samples", "artifacts",
    "manifests", "events", "adapters", "adapter_validations", "runtime_profiles",
    "tracking_bindings",
})
SHARED_DEFAULTS = frozenset({"adapters", "runtime_profiles"})
PARENTS = {
    "experiments": ("projects", "project_id"),
    "experiment_revisions": ("experiments", "experiment_id"),
    "variants": ("experiment_revisions", "experiment_revision_id"),
    "runs": ("variants", "variant_id"),
    "workflow_stages": ("runs", "run_id"),
    "job_attempts": ("workflow_stages", "stage_id"),
    "checkpoints": ("runs", "run_id"),
    "evaluations": ("runs", "run_id"),
    "evaluation_episodes": ("evaluations", "evaluation_id"),
    "metrics": ("runs", "run_id"),
    "training_progress_samples": ("runs", "run_id"),
    "artifacts": ("runs", "run_id"),
    "manifests": ("runs", "run_id"),
}


def visible_sql(table: str, alias: str = "") -> str:
    if table not in PRIVATE_TABLES:
        return "1 = 1"
    column = f"{alias}.owner_id" if alias else "owner_id"
    shared = f" OR {column} IS NULL" if table in SHARED_DEFAULTS else ""
    return f"(current_workspace_id() IS NULL OR {column} = current_workspace_id(){shared})"


def normalize_email(value: str) -> str:
    email = value.strip().casefold()
    if len(email) > 254 or not re.fullmatch(r"[^\s@<>\x00-\x1f]+@[^\s@<>.]+(?:\.[^\s@<>.]+)+", email):
        raise ValueError("Enter a valid email address")
    return email


def _rebuild(connection: sqlite3.Connection, table: str, replacements: dict[str, str]) -> None:
    """Preserve rows, IDs, indexes and triggers while changing unique constraints."""
    ddl = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
    extras = connection.execute("SELECT sql FROM sqlite_master WHERE tbl_name=? AND type IN ('index','trigger') AND sql IS NOT NULL", (table,)).fetchall()
    for old, new in replacements.items():
        if old == "__append_constraint__":
            ddl = ddl.rstrip().removesuffix(")") + ", " + new + ")"
            continue
        if old not in ddl:
            raise RuntimeError(f"Cannot migrate {table}: expected constraint is missing")
        ddl = ddl.replace(old, new)
    temporary = f"{table}_workspace_migration"
    ddl = re.sub(rf'CREATE TABLE(?: IF NOT EXISTS)?\s+"?{table}"?', f'CREATE TABLE {temporary}', ddl, count=1, flags=re.I)
    connection.execute(ddl)
    columns = ",".join(f'"{row[1]}"' for row in connection.execute(f"PRAGMA table_info({table})"))
    connection.execute(f"INSERT INTO {temporary} ({columns}) SELECT {columns} FROM {table}")
    connection.execute(f"DROP TABLE {table}")
    connection.execute(f"ALTER TABLE {temporary} RENAME TO {table}")
    for row in extras:
        connection.execute(row[0])


def migrate_workspaces(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY, email TEXT UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        INSERT OR IGNORE INTO workspaces(id) VALUES ('legacy');
        CREATE TABLE IF NOT EXISTS workspace_sessions (
            token_hash TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            expires_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS workspace_session_expiry ON workspace_sessions(expires_at);
        CREATE TABLE IF NOT EXISTS workspace_migrations (version INTEGER PRIMARY KEY);
    """)
    if connection.execute("SELECT 1 FROM workspace_migrations WHERE version=1").fetchone():
        _scope_adapter_names(connection)
        return
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute("BEGIN IMMEDIATE")
    try:
        for table in PRIVATE_TABLES | {"tracking_connections"}:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN owner_id TEXT DEFAULT 'legacy' REFERENCES workspaces(id)")
            connection.execute(f"CREATE INDEX idx_{table}_owner ON {table}(owner_id)")
        _rebuild(connection, "projects", {"name TEXT NOT NULL UNIQUE": "name TEXT NOT NULL", "__append_constraint__": "UNIQUE(owner_id, name)"})
        _rebuild(connection, "runtime_profiles", {"name TEXT NOT NULL UNIQUE": "name TEXT NOT NULL", "__append_constraint__": "UNIQUE(owner_id, name)"})
        _rebuild(connection, "adapters", {"UNIQUE(name, version)": "UNIQUE(owner_id, name, version)"})
        _rebuild(connection, "tracking_connections", {"provider TEXT PRIMARY KEY": "provider TEXT NOT NULL", "CHECK(provider IN": "PRIMARY KEY(owner_id, provider), CHECK(provider IN"})
        for table in PRIVATE_TABLES:
            _rebuild(connection, table, {"DEFAULT 'legacy'": "DEFAULT (coalesce(current_workspace_id(), 'legacy'))"})
        connection.execute("UPDATE adapters SET owner_id=NULL WHERE seed_key IS NOT NULL")
        for table in PRIVATE_TABLES:
            for operation in ("UPDATE", "DELETE"):
                connection.execute(f"""CREATE TRIGGER workspace_{table}_{operation.lower()}
                    BEFORE {operation} ON {table}
                    WHEN current_workspace_id() IS NOT NULL
                      AND OLD.owner_id IS NOT current_workspace_id()
                    BEGIN SELECT RAISE(ABORT, 'Record is outside this workspace'); END""")
            parent = PARENTS.get(table)
            owner = f"(SELECT owner_id FROM {parent[0]} WHERE id=NEW.{parent[1]})" if parent else "COALESCE(current_workspace_id(), NEW.owner_id)"
            if table == "experiments":
                owner = f"COALESCE({owner}, current_workspace_id(), NEW.owner_id)"
            connection.execute(f"""CREATE TRIGGER workspace_{table}_insert_guard
                BEFORE INSERT ON {table}
                WHEN current_workspace_id() IS NOT NULL AND {owner} IS NOT current_workspace_id()
                BEGIN SELECT RAISE(ABORT, 'Record is outside this workspace'); END""")
            # Generic INSERTs and upserts inherit the owner even when they do not
            # use Database._insert (progress samples and evaluation episodes).
            connection.execute(f"""CREATE TRIGGER workspace_{table}_insert_owner
                AFTER INSERT ON {table} WHEN NEW.owner_id IS NOT {owner}
                BEGIN UPDATE {table} SET owner_id={owner} WHERE id=NEW.id; END""")
        if connection.execute("PRAGMA foreign_key_check").fetchone():
            raise RuntimeError("Workspace migration failed foreign-key validation")
        connection.execute("INSERT INTO workspace_migrations VALUES (1)")
        _scope_adapter_names(connection)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys=ON")


def _scope_adapter_names(connection: sqlite3.Connection) -> None:
    """Allow the same custom adapter name in different personal workspaces."""
    for operation in ("INSERT", "UPDATE"):
        name = f"adapter_registry_name_{operation.lower()}"
        connection.execute(f"DROP TRIGGER IF EXISTS {name}")
        connection.execute(f"""CREATE TRIGGER {name} BEFORE {operation} ON adapters
            WHEN NEW.adapter_key IS NOT NULL AND EXISTS (
                SELECT 1 FROM adapters WHERE lower(name)=lower(NEW.name)
                AND adapter_key <> NEW.adapter_key AND owner_id IS NEW.owner_id
            )
            BEGIN SELECT RAISE(ABORT, 'adapter name already exists'); END""")
