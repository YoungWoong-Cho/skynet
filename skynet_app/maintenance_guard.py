"""Do not attach new work to an item whose file deletion is in progress."""

import json

HISTORY_TABLES = frozenset(
    {
        "adapters",
        "adapter_validations",
        "evaluation_suites",
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
        "adapter_key",
        "adapter_version_id",
        "source_adapter_key",
        "evaluation_suite_id",
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
    for row in connection.execute(
        "SELECT plan_json FROM maintenance_operations"
    ).fetchall():
        plan = json.loads(row[0])
        deleting = {
            identifier for ids in plan["records"].values() for identifier in ids
        }
        deleting.add(plan["id"])
        registry_dependency = False
        if plan["kind"] in {"adapter", "suite"}:
            from .registry_dependencies import consumer_matcher
            registry_table = "adapters" if plan["kind"] == "adapter" else "evaluation_suites"
            record_ids = plan["records"][registry_table]
            placeholders = ",".join("?" for _ in record_ids)
            records = [dict(record) for record in connection.execute(
                f"SELECT * FROM {registry_table} WHERE id IN ({placeholders})", record_ids
            ).fetchall()]
            matcher = consumer_matcher(plan["kind"], records)
            documents = [json.loads(value) for key, value in values.items()
                         if key.endswith("_json") and isinstance(value, str)]
            registry_dependency = any(matcher(document) for document in documents)
            if table == "evaluation_suites" and plan["kind"] == "suite":
                registry_dependency |= any(values.get("name") == record["name"]
                                           and values.get("evaluator_adapter") == record["evaluator_adapter"]
                                           for record in records)
            if table == "adapters" and plan["kind"] == "adapter":
                registry_dependency |= any(values.get("adapter_key") == record["adapter_key"]
                                           or values.get("source_adapter_key") == record["adapter_key"]
                                           for record in records)
        if identifiers & deleting or registry_dependency:
            raise ValueError(
                "This item is being deleted. Finish its pending deletion before using it"
            )
