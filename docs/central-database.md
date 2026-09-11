# One database across app hosts

Skynet supports one PostgreSQL 16 database shared by multiple app hosts. SQLite
remains available for isolated development and as an offline migration source.
A failed PostgreSQL connection is an error; it never switches to a local DB.

On every app host, install locked dependencies with `uv sync --group dev` and
create `config/database.json` (ignored by Git):

```json
{
  "backend": "postgresql",
  "transport": "ssh-unix",
  "ssh_host": "sky2",
  "remote_socket": "/run/user/3712043/skynet-postgres/.s.PGSQL.55432",
  "database": "skynet",
  "user": "ycho420",
  "object_store_root": "/coc/flash7/ycho420/services/skynet/objects"
}
```

The host needs working SSH access to the selected cluster user. No database TCP
port is exposed. Each app creates a private Unix-socket SSH tunnel and closes it
on exit. A direct PostgreSQL connection can instead be provided using
`SKYNET_DATABASE_URL`; supporting file storage still requires the SSH endpoint
configuration. Remove `SKYNET_DATABASE_PATH` from existing launch scripts before
using the central configuration. An explicit SQLite path selects SQLite.

Only one app owns background reconciliation and notification delivery, using a
PostgreSQL advisory lock. Other instances serve requests against the same DB and
can take ownership when the owner exits. The owner stops its workers if its DB
session fails; restart that app after connectivity is restored. Submission and
state mutations are serialized across app hosts. Immutable records and workspace
ownership guards are enforced in PostgreSQL as well as in the application.

Run history, configurations, collection/evaluation metadata, notification queues,
and tracking delivery cursors share this database. Small submission scripts and
source manifests use immutable, checksum-addressed files in `object_store_root`.
Original recordings, prepared datasets, checkpoints and videos retain their
registered cluster locations. App-side working files are disposable caches.
Credentials remain in each app host's credential store; they are not migrated or
shared by this database feature. The background owner needs its own configured
credentials to deliver notifications and tracking events.

## Install the database without sudo

Install PostgreSQL binaries in a private user directory. With user lingering
already enabled, run `deploy/install-postgres-user.py --root <private-db-root>
--bin <postgres-bin-directory>` on the cluster host, alongside
`deploy/backup-postgres.py`. It creates `skynet-postgres.service`, private Unix
socket authentication using the OS user, and a backup timer every six hours.
It does not install system packages or alter firewall rules.

The current deployment uses a hard-mounted NFS filesystem. It depends on that
storage and sky2 being available; it is not a high-availability database. Keep
`fsync`, `full_page_writes` and `synchronous_commit` enabled. Never expose the
underlying PostgreSQL data directory as a SQLite-style shared file.

## Migrate and verify

1. Stop all old app writers. Existing Slurm jobs continue running.
2. Take a consistent SQLite backup, retain the original, and verify its integrity.
3. Import into an empty PostgreSQL database with
   `python -m skynet_app.postgres_migration --source <snapshot> --report <report>`;
   provide the connection through `SKYNET_DATABASE_URL`. The importer verifies
   every table's row count and content hash in one transaction and refuses to
   overwrite existing records.
4. Migrate indexed supporting files and sanitized tracking journals. Verify
   checksums before changing location references; retain immutable execution
   snapshots as historical evidence.
5. Install identical app code and central configuration on each host. Verify both
   clients see the same rows before starting the application.
6. Make a custom-format backup and restore it to a separate test database. Compare
   table counts before deleting the verification database.

Backups are private files under `<private-db-root>/backups`; the newest 28
successful dumps are retained. Failed backups do not replace valid backups.
This is a point-in-time snapshot schedule, not continuous WAL archiving. Copy
backups to a separate storage system if protection against loss of the cluster
filesystem is required.

Do not resume an old SQLite copy after the central app has made changes. Stop all
writers first and restore a verified central backup; old local snapshots are
historical rollback material, not an automatically synchronized second DB.

## Regression tests

Set `SKYNET_TEST_POSTGRES_ADMIN` to an isolated PostgreSQL admin connection and run
`pytest tests/test_postgres.py`. Existing repository contracts can also run on
PostgreSQL with `pytest -p tests.postgres_backend_plugin <test files>`. The test
plugin creates and drops isolated test databases and stubs cluster file transfers.
No production training or evaluation jobs are submitted by these tests.
