# One database across app hosts

Skynet requires one PostgreSQL 16 database shared by all app hosts. Missing
configuration or a failed connection is an error; the app never creates a local DB.

On every app host, install locked dependencies with `uv sync --group dev` and
create `config/database.json` (ignored by Git):

```json
{
  "backend": "postgresql",
  "transport": "ssh-tcp",
  "ssh_host": "sky2",
  "remote_address": "127.0.0.1",
  "remote_port": 55432,
  "password_file": "database-password",
  "database": "skynet",
  "user": "ycho420",
  "object_store_root": "/coc/flash7/ycho420/services/skynet/objects"
}
```

The host needs working SSH access to the selected cluster user. The database listens only on the cluster host’s loopback interface. No database
port is exposed to the network. Put the DB connection password in
`config/database-password` with permission `0600`; this file is ignored by Git. Each app creates a private Unix-socket SSH tunnel and closes it
on exit. The original `ssh-unix` transport remains available for SSH servers that
support forwarding to the private PostgreSQL Unix socket. A direct PostgreSQL connection can instead be provided using
`SKYNET_DATABASE_URL`; supporting file storage still requires the SSH endpoint
configuration. The obsolete `SKYNET_DATABASE_PATH` setting is no longer used.

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
--bin <postgres-bin-directory> --loopback` on the cluster host, alongside
`deploy/backup-postgres.py`. It creates `skynet-postgres.service`, a private Unix socket for administration and a backup timer every six hours.
The optional loopback listener requires a separately provisioned PostgreSQL SCRAM
password. Without `--loopback`, only Unix socket access is enabled.
It does not install system packages or alter firewall rules.

The current deployment uses a hard-mounted NFS filesystem. It depends on that
storage and sky2 being available; it is not a high-availability database. Keep
`fsync`, `full_page_writes` and `synchronous_commit` enabled. Never expose the
underlying PostgreSQL data directory as a shared application file.

## Backup and relocation

To add an app host, configure the same central endpoint and object store. Do not
copy or create a second application database. Verify that the host sees the same
workspace IDs and histories; the database lock selects one background coordinator.

To relocate the database itself, stop app writers, take a custom-format `pg_dump`
backup and restore it with `pg_restore` into an empty PostgreSQL database. Verify
record counts, ownership and referenced cluster files before updating all app
hosts to the new endpoint. Keep the verified backup until the cutover is complete.

Backups are private files under `<private-db-root>/backups`; the newest 28
successful dumps are retained. Failed backups do not replace valid backups.
This is a point-in-time snapshot schedule, not continuous WAL archiving. Copy
backups to a separate storage system if protection against loss of the cluster
filesystem is required.

Do not restore stale local copies over the central database. Stop all writers
before restoring a verified PostgreSQL backup.

## Regression tests

Set `SKYNET_TEST_POSTGRES_ADMIN` to a disposable PostgreSQL server with permission
to create test databases, then run `pytest`. The default test harness creates and
drops isolated databases, including module initialization, and stubs cluster file
transfers. It never selects the app's configured production database and never
submits real training/evaluation jobs. Tests fail early if the test server is not
configured. Browser-only tests continue to run through the npm scripts.
