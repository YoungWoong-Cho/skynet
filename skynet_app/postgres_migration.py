"""Verified, offline SQLite-to-PostgreSQL import. Never merges live databases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import psycopg
from psycopg import sql

from .db_backend import PostgresBackend, lock_key


def table_digest(connection, table, columns, *, postgres=False):
    statement = (
        "SELECT " + ",".join('"' + c + '"' for c in columns) + ' FROM "' + table + '"'
    )
    rows = connection.execute(statement).fetchall()
    # Sort row digests instead of locale-dependent database ORDER BY values.
    digests = sorted(
        hashlib.sha256(
            json.dumps(
                list(row), ensure_ascii=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).digest()
        for row in rows
    )
    return hashlib.sha256(b"".join(digests)).hexdigest(), len(rows)


def import_snapshot(source_path: Path, target_url: str):
    source = sqlite3.connect(source_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        if (
            source.execute("PRAGMA integrity_check").fetchone()[0] != "ok"
            or source.execute("PRAGMA foreign_key_check").fetchall()
        ):
            raise ValueError("Source SQLite integrity check failed")
        tables = [
            row[0]
            for row in source.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        backend = PostgresBackend(target_url)
        backend.initialize()
        report = {}
        with psycopg.connect(target_url, autocommit=True) as target:
            with target.transaction():
                target.execute(
                    "SELECT pg_advisory_xact_lock(%s)", (lock_key("schema"),)
                )
                target.execute(
                    "SELECT pg_advisory_xact_lock(%s)", (lock_key("repository-write"),)
                )
                for table in tables:
                    count = target.execute(
                        sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
                    ).fetchone()[0]
                    if (
                        count
                        and not (
                            table == "workspaces"
                            and target.execute(
                                "SELECT id,email FROM workspaces"
                            ).fetchall()
                            == [("legacy", None)]
                        )
                        and not (
                            table == "workspace_migrations"
                            and target.execute(
                                "SELECT version FROM workspace_migrations"
                            ).fetchall()
                            == [(1,)]
                        )
                    ):
                        raise ValueError("Destination is not empty: " + table)
                # All import operations, trigger states, and validation share one
                # transaction. No partially imported database is ever published.
                for table in tables:
                    target.execute(
                        sql.SQL("ALTER TABLE {} DISABLE TRIGGER USER").format(
                            sql.Identifier(table)
                        )
                    )
                target.execute("DELETE FROM workspace_migrations")
                target.execute("DELETE FROM workspaces")
                pending = set(tables)
                order = []
                while pending:
                    ready = sorted(
                        t
                        for t in pending
                        if all(
                            r[2] == t or r[2] not in pending
                            for r in source.execute(
                                'PRAGMA foreign_key_list("' + t + '")'
                            )
                        )
                    )
                    if not ready:
                        raise ValueError(
                            "Cyclic table dependencies require an explicit migration"
                        )
                    order.extend(ready)
                    pending.difference_update(ready)
                for table in order:
                    columns = [
                        row[1]
                        for row in source.execute('PRAGMA table_info("' + table + '")')
                    ]
                    target_columns = [
                        row[0]
                        for row in target.execute(
                            "SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name=%s ORDER BY ordinal_position",
                            (table,),
                        )
                    ]
                    if set(columns) != set(target_columns):
                        raise ValueError("Schema mismatch: " + table)
                    # Self-referencing rows (retries) are inserted with ancestors
                    # first. Other FKs are satisfied by table order.
                    rows = source.execute('SELECT * FROM "' + table + '"').fetchall()
                    self_refs = [
                        columns.index(row[3])
                        for row in source.execute(
                            'PRAGMA foreign_key_list("' + table + '")'
                        )
                        if row[2] == table
                    ]
                    if self_refs:
                        by_id = {row[columns.index("id")]: row for row in rows}
                        rows = []
                        done = set()
                        while by_id:
                            ready = [
                                key
                                for key, row in by_id.items()
                                if all(
                                    row[i] is None or row[i] in done for i in self_refs
                                )
                            ]
                            if not ready:
                                raise ValueError("Cyclic self-reference: " + table)
                            for key in ready:
                                rows.append(by_id.pop(key))
                                done.add(key)
                    statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                        sql.Identifier(table),
                        sql.SQL(",").join(map(sql.Identifier, columns)),
                        sql.SQL(",").join(sql.Placeholder() for _ in columns),
                    )
                    with target.cursor() as cursor:
                        cursor.executemany(statement, rows)
                    before, count = table_digest(source, table, columns)
                    after, actual = table_digest(target, table, columns, postgres=True)
                    if (before, count) != (after, actual):
                        raise ValueError("Imported row verification failed: " + table)
                    report[table] = {"rows": count, "sha256": after}
                target.execute(
                    "SELECT setval(pg_get_serial_sequence('notification_outbox','id'), coalesce((SELECT max(id) FROM notification_outbox),1), EXISTS(SELECT 1 FROM notification_outbox))"
                )
                for table in tables:
                    target.execute(
                        sql.SQL("ALTER TABLE {} ENABLE TRIGGER USER").format(
                            sql.Identifier(table)
                        )
                    )
                target.execute(
                    "CREATE TABLE IF NOT EXISTS skynet_import_receipts(source_sha256 TEXT PRIMARY KEY, imported_at TIMESTAMPTZ NOT NULL DEFAULT now(), report_json TEXT NOT NULL)"
                )
                checksum = hashlib.sha256(source_path.read_bytes()).hexdigest()
                target.execute(
                    "INSERT INTO skynet_import_receipts(source_sha256,report_json) VALUES (%s,%s)",
                    (checksum, json.dumps(report, sort_keys=True)),
                )
        return {
            "source_sha256": checksum,
            "verified_tables": len(report),
            "tables": report,
        }
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--target-env",
        default="SKYNET_DATABASE_URL",
        help="Environment variable holding the destination connection string",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import os

    url = os.environ.get(args.target_env)
    if not url:
        parser.error("Destination connection is not configured")
    report = import_snapshot(args.source, url)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Verified {report['verified_tables']} imported tables. Report: {args.report}"
    )


if __name__ == "__main__":
    main()
