"""Run existing repository contract tests on PostgreSQL.

Usage: SKYNET_TEST_POSTGRES_ADMIN=... pytest -p tests.postgres_backend_plugin ...
Each test's SQLite-style temporary filenames identify isolated PostgreSQL DBs.
This adapter exists only in tests; application configuration never does this.
"""

import hashlib
import os
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from skynet_app.database import Database


@pytest.fixture(autouse=True)
def postgres_repository_contract(monkeypatch, request, tmp_path):
    if request.node.name in {
        "test_legacy_schema_migration_preserves_records_and_claims_configured_owner",
        "test_upgrade_skips_previously_queued_unconfirmed_errors",
    }:
        pytest.skip(
            "SQLite file migration is covered separately; PostgreSQL import has its own roundtrip test"
        )
    if request.node.path.name == "test_postgres.py":
        yield
        return
    from skynet_app.metadata_objects import MetadataObjects

    def local_request(self, operation, digest, content=None, *, name="content"):
        path = Path(self.path(digest, name))
        path.parent.mkdir(parents=True, exist_ok=True)
        if content is not None:
            if path.exists():
                assert path.read_bytes() == content
            else:
                path.write_bytes(content)
        import base64

        value = path.read_bytes()
        return {
            "sha256": hashlib.sha256(value).hexdigest(),
            "size": len(value),
            "content": base64.b64encode(value).decode() if operation == "get" else None,
        }

    def configure_objects(self, database):
        self.root = tmp_path / "shared-objects"
        self.host = "test"

    monkeypatch.setattr(MetadataObjects, "__init__", configure_objects)
    monkeypatch.setattr(MetadataObjects, "_request", local_request)
    admin = os.environ["SKYNET_TEST_POSTGRES_ADMIN"]
    original = Database.__init__
    names = {}

    def initialize(self, path=None, **kwargs):
        if kwargs.get("url"):
            original(self, path, **kwargs)
            self.path = next(
                (
                    key
                    for key, name in names.items()
                    if kwargs["url"] == make_conninfo(admin, dbname=name)
                ),
                path,
            )
            return
        key = str(path or "default")
        if key not in names:
            name = "skynet_contract_" + uuid.uuid4().hex
            with psycopg.connect(admin, autocommit=True) as c:
                c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
            names[key] = name
        kwargs["url"] = make_conninfo(admin, dbname=names[key])
        if path is not None:
            kwargs["data_root"] = Path(path).parent
        original(self, None, **kwargs)
        # Legacy tests reopen their same fixture by filename. This does not
        # represent a SQLite file, and is never used by the transport.
        self.path = path

    monkeypatch.setattr(Database, "__init__", initialize)
    try:
        yield
    finally:
        with psycopg.connect(admin, autocommit=True) as c:
            for name in names.values():
                c.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                        sql.Identifier(name)
                    )
                )
