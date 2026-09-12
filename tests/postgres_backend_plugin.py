"""Isolated PostgreSQL databases and cluster-file doubles for repository tests.

SKYNET_TEST_POSTGRES_ADMIN must point to a disposable test server. A bootstrap
DB isolates module-level app initialization; every test gets separate DBs. Path
arguments below are fixture keys only, never application database files.
"""

import hashlib
import os
import tempfile
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from skynet_app import database as database_module
from skynet_app.database import Database


def pytest_configure(config):
    admin = os.environ.get("SKYNET_TEST_POSTGRES_ADMIN")
    if not admin:
        raise pytest.UsageError("Set SKYNET_TEST_POSTGRES_ADMIN to an isolated PostgreSQL test server. Tests never use the app database.")
    name = "skynet_bootstrap_" + uuid.uuid4().hex
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    # Module-level app imports must never read the installation's endpoint or
    # object-store configuration, even when tests run in the deployed checkout.
    test_root = tempfile.TemporaryDirectory(prefix="skynet-test-config-")
    previous = {key: os.environ.get(key) for key in ("SKYNET_DATABASE_URL", "SKYNET_DATA_ROOT")}
    config._skynet_bootstrap = (admin, name, previous, database_module.APP_ROOT, test_root)
    database_module.APP_ROOT = Path(test_root.name)
    os.environ["SKYNET_DATA_ROOT"] = str(Path(test_root.name) / "data")
    os.environ["SKYNET_DATABASE_URL"] = make_conninfo(admin, dbname=name)


def pytest_unconfigure(config):
    bootstrap = getattr(config, "_skynet_bootstrap", None)
    if not bootstrap:
        return
    admin, name, previous, app_root, test_root = bootstrap
    try:
        with psycopg.connect(admin, autocommit=True) as c:
            c.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
    finally:
        database_module.APP_ROOT = app_root
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        test_root.cleanup()


@pytest.fixture(autouse=True)
def postgres_repository_contract(monkeypatch, request, tmp_path):
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
            original(self, **kwargs)
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
        original(self, **kwargs)
        # Tests can reopen a named fixture without sharing it across tests.
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
