# API reads and live updates

API list responses always come from current PostgreSQL reads. There is no shared
response cache or TTL. Multi-table dataset catalog reads use a short, read-only
repeatable-read transaction; all rows in that response come from one snapshot.
Related rows are fetched in batches rather than one query per item. Settings
helpers share values only within the current request.

The registry requests `/api/data/resources?include_versions=true` once instead of
requesting each resource separately. The default resource response is unchanged.
The conversion UI reads `/api/data/exports/jobs` for current job status, dataset
locations, usage, and training readiness. Opening Convert reads only its selected
recording's options. The older full exports endpoint remains compatible, but is
no longer polled by the UI. GET requests do not dispatch conversion work; the
existing background worker does that independently.

## Freshness contract

Migration 011 adds PostgreSQL triggers that publish small invalidation hints after
commit. Hints contain a version, workspace scope, and topics, with no row data or
private identifiers. Shared data reaches every workspace; personal settings and
adapters reach only the affected workspace. A dedicated LISTEN connection exists
in each application process, independently of the single background-job owner.

Authenticated `/api/changes?expected_workspace=...` streams deliver those hints.
Each stream is bound to its workspace and session. Session changes and periodic
authorization checks terminate revoked streams. A listener failure closes the
streams, allowing browsers to reconnect. No durable event history is required:
connect, reconnect, queue overflow, and unknown messages trigger a fresh read.

The frontend refresh coordinator marks affected consumers dirty, allows one
refresh per consumer, and performs another read if a change arrives during that
refresh. Hidden consumers retain their dirty state until shown. Focus and
visibility restoration also refresh current data. A successful mutation advances
the read generation before returning; an older in-flight GET or body cannot
replace a newer result. Sign-out and workspace switches abort reads and streams.

Conversion progress falls back to a small, visible-only jobs poll while live
updates are unavailable. Existing cluster/run/evaluation polling remains where
it also obtains external scheduler state. A DB notification cannot replace an
external status collector.

When adding or changing a write path, update its notification topics, dependent
frontend consumers, and tests. Meaningful changes must emit hints; timestamp-only
worker updates must not create refresh loops. Keep snapshots authoritative and
retain reconnect resynchronization when evolving the notification format.

## Migration and recovery

Migration 011 is additive: a trigger function and triggers only. It does not
rewrite or delete application data. The migration runner applies it in a
transaction and rejects changed checksums for already-applied migrations.

Before applying schema changes, take a consistent backup and verify restoration
in a separate database. A migration file alone does not recover deleted data.

To reverse 011, stop application writers, restore the prior application source,
then execute `deploy/rollback-change-notifications.sql` against the intended DB.
It removes only these triggers/function and migration 011's ledger entry under
the schema advisory lock. Restart the prior application afterward. The rollback
and subsequent reapplication are tested to preserve existing rows. Restoring an
entire older backup is a separate recovery operation that can discard newer
writes; it is not needed to remove these additive triggers.

Relevant tests: `test_batched_catalog.py`, `test_export_job_reads.py`,
`test_request_scoped_settings.py`, `test_changes.py`, `api_freshness_ui.mjs`,
`policy_exports_ui.mjs`, and `email_workspaces_ui.mjs`.
