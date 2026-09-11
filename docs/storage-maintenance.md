# Storage and history maintenance

The PostgreSQL database on sky2 is the canonical index. Large adapter manifests,
execution snapshots, resolved stage configurations and tracking journal bodies
live in checksum-addressed cluster objects. SQL retains searchable identity,
state, relationships and object references. Reads verify content hashes. Journal
appends reuse previous chunks instead of copying the entire spool. Local compiled
capsules are not persisted when using the central database.

## Delete history

Experiments, Training Runs and Evaluations each have **Delete**. The shared dialog
lists dependencies and files before confirmation. Evaluation deletion includes
its rollouts, result artifacts, execution stage, attempts, events, metrics and
references in the retained run's tracking delivery journal. Run deletion includes
its attempts, checkpoints, artifacts, metrics, notifications and tracking state.
Experiment deletion includes its revisions and variants.

Evaluations must be deleted before their training run. Restarted runs and other
checkpoint/stage consumers must also be removed first. Training runs must be
deleted before their experiment. Active work cannot be deleted. Shared recordings,
prepared datasets, adapters, runtimes and independent histories are retained.

Deletion is scoped to the current email workspace. A committed intent protects
partially deleted records against new references, including direct PostgreSQL
writes. Files are rechecked before removal; SQL records are removed only after
file cleanup succeeds. If disconnected, open Delete again to finish. Successful
operations remove their intent as well. External W&B/MLflow history and database
backups follow their own retention; this action does not delete those services.

## Inspect files

Settings → Cluster storage → **Inspect files** scans output boundaries under the
workspace base path, plus the central metadata object store. Registered paths in
all workspaces are protected. Unreferenced outputs and entries outside the storage
layout are listed for explicit selection. New files are protected for 24 hours;
selected symlinks, special files, services, database files, environments and source
trees cannot be removed here. Links inside an owned output directory are removed
with that directory without following their targets. The whole selection is
revalidated before deletion.

Canonical boundaries:

- `jobs/runs/<run-id>/`: submitted capsules, attempt receipts and checkpoints.
- `eval/runs/<stage-id>/`: evaluation results, progress and rollout videos.
- `datasets/raw/`: retained collection archives.
- `datasets/prepared/<manifest-sha256>/`: registered prepared formats.
- `logs/` and `artifacts/`: generated outputs.
- `services/skynet/objects/<hash-prefix>/<sha256>/`: supporting bodies.
- `services/skynet/database/`: central PostgreSQL and backups, always protected.
- `repos/`, `workspace/`, `envs/`, `.cache/`: reusable sources, runtimes and caches,
  protected from this cleanup feature.

This is an explicit cleanup tool, not a background garbage collector. It does not
crawl or erase the entire filesystem. Files outside supported boundaries require
separate relocation. Database import/export and synchronization UI remain deferred.

## Existing database relocation

Stop application writers, take a verified PostgreSQL backup, then run:

```sh
python -m skynet_app.payload_migration
```

Relocation uploads and independently reads back all bodies before replacing any
SQL contents. Original receipt hashes remain immutable. The transaction is atomic
and repeatable. Install the same code version on every app host before restarting
an app; an old client cannot decode the new object references.
