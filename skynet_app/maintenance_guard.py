"""Do not attach new work to an item whose file deletion is in progress."""

import json

HISTORY_TABLES = frozenset(
    {
        "experiments",
        "experiment_revisions",
        "variants",
        "runs",
        "workflow_stages",
        "job_attempts",
        "checkpoints",
        "evaluations",
        "evaluation_episodes",
        "artifacts",
        "manifests",
        "metrics",
        "training_progress_samples",
        "events",
        "tracking_bindings",
        "stage_dependencies",
    }
)
ID_FIELDS = frozenset(
    {
        "id",
        "run_id",
        "stage_id",
        "experiment_id",
        "experiment_revision_id",
        "variant_id",
        "checkpoint_id",
        "resume_checkpoint_id",
        "restarted_from_run_id",
        "evaluation_id",
        "entity_id",
        "scope_id",
        "depends_on_stage_id",
    }
)


def guard_write(connection, table, values):
    if table not in HISTORY_TABLES:
        return
    identifiers = {
        value
        for key, value in values.items()
        if key in ID_FIELDS and isinstance(value, str)
    }
    if not identifiers:
        return
    for row in connection.execute(
        "SELECT plan_json FROM maintenance_operations"
    ).fetchall():
        plan = json.loads(row[0])
        deleting = {
            identifier for ids in plan["records"].values() for identifier in ids
        }
        if identifiers & deleting:
            raise ValueError(
                "This history item is being deleted. Finish its pending deletion before using it"
            )
