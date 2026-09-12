"""Find actual consumers of versioned adapters and evaluation suites."""

import json


def suite_removal_notices(connection, target, workspace_id):
    """Supported suites are optional capabilities, not deletion dependencies."""
    from .evaluation_compatibility import supports_recorded_suite
    adapters = connection.execute("""
        SELECT * FROM (
            SELECT DISTINCT ON (adapter_key) * FROM adapters
            ORDER BY adapter_key, version_number DESC
        ) latest
        WHERE enabled=1 AND archived_at IS NULL
        ORDER BY lower(name), adapter_key
    """).fetchall()
    labels = []
    for row in adapters:
        manifest = json.loads(row["manifest_json"] or "{}")
        if not supports_recorded_suite(manifest, target) and not any(
            entry.get("environment") == target["evaluator_adapter"]
            and (target["name"] in entry.get("suites", []) or not entry.get("suites"))
            and entry.get("command")
            for entry in manifest.get("evaluations", [])
        ):
            continue
        visible = workspace_id is None or row["owner_id"] in (None, workspace_id)
        label = row["name"] if visible else "Another workspace's adapter"
        if label not in labels:
            labels.append(label)
    if not labels:
        return []
    return [
        "This suite will no longer be available for evaluation with: "
        + ", ".join(labels)
        + ". Training runs and checkpoints will be retained."
    ]


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def consumer_matcher(kind, records):
    ids = {row["id"] for row in records}
    if kind == "adapter":
        ids.update(row["adapter_key"] for row in records)
        aliases = set()
        for row in records:
            manifest = json.loads(row["manifest_json"] or "{}")
            aliases.update(filter(None, [manifest.get("slug"), row.get("seed_key"), *manifest.get("aliases", [])]))
    else:
        aliases = {row["name"] for row in records}

    def matches(value):
        if isinstance(value, list):
            return any(matches(child) for child in value)
        if not isinstance(value, dict):
            return False
        if ids.intersection(strings(value)):
            return True
        if kind == "adapter":
            source = value.get("source")
            if isinstance(source, dict) and not any(source.get(key) for key in ("adapter_id", "adapter_version_id")):
                if isinstance(source.get("adapter"), str) and source["adapter"] in aliases:
                    return True
        else:
            evaluation = value.get("evaluation")
            if isinstance(evaluation, dict) and aliases.intersection(strings(evaluation.get("suites", []))):
                return True
        return any(matches(child) for child in value.values() if isinstance(child, (dict, list)))

    return matches


def extend_graph(connection, kind, target, graph, block, fetch):
    if kind == "adapter":
        records = fetch(connection, "adapters", "adapter_key=?", (target["adapter_key"],))
        graph["adapters"] = records
        validations = fetch(connection, "adapter_validations", "adapter_key=?", (target["adapter_key"],))
        graph["adapter_validations"] = validations
        for row in validations:
            if row.get("owner_id") != target.get("owner_id"):
                block("adapter_validation", row, "Remove the other workspace's validation first")
        for row in fetch(connection, "adapters", "source_adapter_key=? AND adapter_key<>?", (target["adapter_key"], target["adapter_key"])):
            block("adapter", {**row, "id": row["adapter_key"]}, "Delete this derived adapter first")
    else:
        records = fetch(connection, "evaluation_suites", "evaluator_adapter=? AND name=?", (target["evaluator_adapter"], target["name"]))
        graph["evaluation_suites"] = records
        ids = {row["id"] for row in records}
        versions = {row["suite_version"] for row in records}
        for row in fetch(connection, "evaluations"):
            if row["evaluation_suite_id"] in ids or (
                not row["evaluation_suite_id"] and row["suite_name"] == target["name"]
                and row["evaluator_adapter"] == target["evaluator_adapter"] and row["suite_version"] in versions
            ):
                block("evaluation", row, "Delete this evaluation first")

    matches = consumer_matcher(kind, records)
    if kind == "suite":
        for row in fetch(connection, "adapters"):
            if matches(json.loads(row["manifest_json"])):
                block("adapter", {**row, "id": row["adapter_key"]}, "Delete this adapter with a pinned suite default first")
    experiments = {row["id"]: row for row in fetch(connection, "experiments")}
    revisions = {row["id"]: row for row in fetch(connection, "experiment_revisions")}
    runs = {row["id"]: row for row in fetch(connection, "runs")}
    evaluations = fetch(connection, "evaluations")
    stages = {row["id"]: row for row in fetch(connection, "workflow_stages")}

    def block_stage(stage):
        consumers = [row for row in evaluations if row["stage_id"] == stage["id"]]
        if consumers:
            for row in consumers:
                block("evaluation", row, "Delete this evaluation first")
        elif stage["run_id"] in runs:
            block("run", runs[stage["run_id"]], "Delete this training run first")

    for revision in revisions.values():
        if matches(json.loads(revision["requested_spec_json"])):
            block("experiment", experiments[revision["experiment_id"]], "Delete this experiment and its revisions first")
    for variant in fetch(connection, "variants"):
        if matches(json.loads(variant["resolved_spec_json"])):
            revision = revisions[variant["experiment_revision_id"]]
            block("experiment", experiments[revision["experiment_id"]], "Delete this experiment and its pinned variants first")
            for run in runs.values():
                if run["variant_id"] == variant["id"]:
                    block("run", run, "Delete this training run first")
    for stage in stages.values():
        if matches(json.loads(stage["resolved_config_json"])):
            block_stage(stage)
    for attempt in fetch(connection, "job_attempts"):
        if matches(json.loads(attempt.get("execution_snapshot_json") or "{}")):
            block_stage(stages[attempt["stage_id"]])
