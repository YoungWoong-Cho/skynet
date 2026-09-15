# Resource names and identity

`data_resources` stores four distinct fields:

| Field | Purpose | Editable |
| --- | --- | --- |
| `id` | Registry primary key and API path identifier | No |
| `source_key` | Provider/namespace-scoped source identity | No |
| `display_name` | Human-readable catalog and editor name | Yes |
| `description` | Additional explanation | Yes |

Display names may repeat. The unique identity remains `(provider, namespace,
source_key)`. A collection source key can be a session ID or a derived selection
key; it is not a universal recording foreign key. Recording provenance remains
in the existing explicit session/source references.

Current resource API requests and responses use these fields. Creation accepts
an optional `display_name`, defaulting to `namespace/source_key`. PATCH accepts
`display_name` and `description` independently; editing a label does not replace
metadata. Resource `metadata.display_name` is rejected to prevent competing
values. Immutable version metadata may still contain its historical label.

Datasets sort by Updated descending. Saving a name updates that timestamp and
refreshes the catalog; the existing commit notifications invalidate other open
clients. Display names are not inputs to selection identity or content hashes.

## Compatibility

Migration 012 copies existing string display labels from resource metadata,
using the existing catalog fallback for missing/blank labels, then removes the
duplicate metadata field. Migration 013 renames the live registry identity column
and guards identity changes. Neither migration rewrites version metadata,
experiment snapshots, bundle manifests, or their hashes.

The versioned `skynet.data-bundle/v1` receipt still serializes the stable source key
as `resource.name`. Other versioned collection/import contracts retain their
existing identity fields. These boundary mappings preserve existing receipt
bytes and hashes; they are not mutable display-name aliases.

## Deployment and rollback

Apply migrations and matching backend/frontend source as one coordinated release.
The migration runner applies pending migrations in one transaction. Restart all
app instances against the new schema; refresh already-open browser pages so they
load the matching API client. There is no permanent dual-write or response cache.

Before deployment, take and restore-test a database backup. Test migration,
rollback, and reapplication against the restored copy, checking resource IDs,
dates, source keys, labels, and immutable table contents.

To reverse migrations 012 and 013:

1. Stop all app writers.
2. Run `deploy/rollback-resource-naming.sql` against the intended database.
3. Restore the matching pre-012 source and restart the application.

Rollback copies the latest edited labels back into resource metadata and restores
the `name` identity column. Other metadata values and immutable records remain
unchanged. Restoring the full backup is only needed for an exact pre-deployment
snapshot; doing so discards subsequent writes, whereas the schema rollback keeps
them. The rollback is for a database with both naming migrations applied and must
not be used after later schema changes without reviewing those dependencies.
