# One database across app hosts

Skynet requires one PostgreSQL 16 database shared by all app hosts. Missing
configuration or a failed connection is an error; the app never creates a local DB.

On every app host, install locked dependencies with `uv sync --group dev` and
create `config/database.json` (ignored by Git):

```json
{
  "backend": "postgresql",
  "transport": "ssh-tcp",
  "ssh_host": "sky1",
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
on exit; its connect timeout and keepalives come from the cluster profile's `ssh.tunnel` section, sized to ride out a slow login node (`ssh.command` covers one-shot commands and transfers, and `ssh.operations` holds the per-operation budgets and Slurm deadlines; see the README's "Operator cluster profile" section). The original `ssh-unix` transport remains available for SSH servers that
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
Historical `live_conversions` rows from the retired state-only conversion workflow
stay in this database; `skynet_app/recording_deletion.py` still accounts for them
when it previews and deletes recordings.

Experiment notes (Markdown, folders and attachments) are workspace-owned rows in
this database, written only through the Notes API that the browser uses.
Attachment bodies are immutable files in `object_store_root`; each attachment row
keeps a reference to its body, so Storage cleanup removes a body only after its
attachment or note is deleted.

Large stored documents are never parsed on request or reconcile paths. Experiment
revisions, variants, evaluation stages, job attempts, adapter versions, live sessions,
policy exports and data bundles each keep a small projection in `document_projections`
(retained paths, dataset assignments, the run list's spec fields, tracking providers,
frozen evaluation targets, attempt resume pins, the compact adapter manifest without
capsule code), computed by `skynet_project_document` in PostgreSQL. A
trigger maintains it for documents stored inline; for documents offloaded to the
object store the uploader writes it in the same transaction, and
`repair_document_projections` backfills stages and attempts written before their
projection existed from the bodies' projected subtrees during reconcile. Usage, deletion and retirement
checks for evaluation targets read these projections and fail closed while any
stage still lacks one.

## Install the database without sudo

Install PostgreSQL binaries in a private user directory. With user lingering
already enabled, run `deploy/install-postgres-user.py --root <private-db-root>
--bin <postgres-bin-directory> --loopback` on the cluster host, alongside
`deploy/backup-postgres.py`. It creates `skynet-postgres.service`, a private Unix socket for administration and a backup timer every six hours.
The optional loopback listener requires a separately provisioned PostgreSQL SCRAM
password. Without `--loopback`, only Unix socket access is enabled.
It does not install system packages or alter firewall rules.
The database and backup units are restricted with `ConditionHost` to the installing
host. This is required when the home directory and DB files are shared: another
login host must never start a second PostgreSQL against the same data directory.

The current deployment uses a hard-mounted NFS filesystem. It depends on that
storage and sky1 being available; it is not a high-availability database. Keep
`fsync`, `full_page_writes` and `synchronous_commit` enabled. Never expose the
underlying PostgreSQL data directory as a shared application file.

## Move the service to another login host

The units are pinned to one host with `ConditionHost`, and the PostgreSQL binaries
are linked against that host's OS libraries. When the host is retired or its OS is
reinstalled, the database does not return on its own; the app then reports
`database_unavailable` and names the configured `ssh_host`.

Confirm the old server is stopped (`data/postmaster.pid` is absent and
`pg_controldata` reports `shut down`) and keep a cold copy of `data`. If
`ldd <bin>/postgres` reports missing libraries, rebuild the same PostgreSQL major
version on the new host. Enable lingering there, then rerun
`deploy/install-postgres-user.py --root <private-db-root> --bin <new-bin> --loopback`
on that host: it keeps the existing data directory and rewrites the units for the
new host. Finally set `ssh_host` in `config/database.json` on every app host and
restart the app.

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
to create test databases, then run `pytest`. The default test harness
(`tests/postgres_backend_plugin.py`) migrates one template database per session,
clones every test's isolated databases from it, including the bootstrap database that
isolates module initialization, drops them afterwards, and stubs cluster file
transfers. It never selects the app's configured production database and never
submits real training/evaluation jobs. Tests fail early if the test server is not
configured. Browser-only tests continue to run through the npm scripts.
