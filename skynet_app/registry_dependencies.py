"""Find actual consumers of versioned adapters and evaluation suites."""

import json

from .registry_reference_match import registry_reference_match, suite_reference
from .payload_store import ImmutableProjectionCache

_INLINE_REFERENCES = ImmutableProjectionCache()


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


def consumer_ids(kind, records):
    ids = {row["id"] for row in records}
    if kind == "adapter":
        ids.update(row["adapter_key"] for row in records)
    else:
        # Experiment specs and adapter defaults name a suite version, not its ID.
        ids.update(suite_reference(row["evaluator_adapter"], row["name"], row["suite_version"]) for row in records)
    return ids


def consumer_matcher(kind, records):
    ids = consumer_ids(kind, records)
    return lambda value: registry_reference_match(value, ids)


def consumer_flags(connection, rows, column, kind, records):
    ids = consumer_ids(kind, records)
    documents = [json.loads(row.get(column) or "{}") for row in rows]
    store = getattr(connection, "payload_store", None)
    if store:
        return store.registry_matches(documents, ids)
    return [registry_reference_match(document, ids) for document in documents]


def inline_consumers(connection, table, kind, records):
    """Check inline specifications on the database, returning compact identities.

    Specifications embed adapter code. Downloading all revisions and variants
    just to find references transfers gigabytes during a registry preview.
    JSON paths select exact identity references.
    """
    columns = {"experiment_revisions": "requested_spec_json", "variants": "resolved_spec_json"}
    column = columns[table]
    digest_column = "requested_spec_sha256" if table == "experiment_revisions" else "resolved_spec_sha256"
    parent_column = "experiment_id" if table == "experiment_revisions" else "experiment_revision_id"
    ids = consumer_ids(kind, records)
    suites = [{"adapter": row["evaluator_adapter"], "suite": row["name"], "suite_version": row["suite_version"]}
              for row in records] if kind == "suite" else []
    variables = json.dumps({"ids": sorted(ids), "suites": suites})
    identity_path = 'strict $.** ? (@.type() == "string" && @ == $ids[*])'
    fallback_path = 'strict $.** ? (@.type() == "object").** ? (@.type() == "string" && @ == $ids[*])'
    # Records of one suite share its evaluator and name, so matching each field
    # against any record is the same as matching one record's version.
    named_path = ('strict $.** ? (@.type() == "object" && @.adapter == $suites[*].adapter '
                  '&& @.suite == $suites[*].suite && @.suite_version == $suites[*].suite_version)')
    headers = [dict(row) for row in connection.execute(
        f"SELECT id, owner_id, {parent_column}, {digest_column} AS digest FROM {table}"
    ).fetchall()]
    info = connection.raw.info
    namespace = (info.host, info.port, info.dbname, info.user, tuple(sorted(ids)))
    references = {(*namespace, row['digest']): row['digest'] for row in headers}

    def load(digests):
        matched = {row['digest']: row['referenced'] for row in connection.execute(f"""
            SELECT digest,
                   jsonb_path_exists(body, CASE WHEN jsonb_typeof(body)='object' THEN ?::jsonpath ELSE ?::jsonpath END, ?::jsonb)
                   OR jsonb_path_exists(body, ?::jsonpath, ?::jsonb) AS referenced
            FROM (SELECT DISTINCT ON ({digest_column}) {digest_column} AS digest,
                         {column}::jsonb AS body FROM {table} WHERE {digest_column}=ANY(?)) documents
        """, (identity_path, fallback_path, variables, named_path, variables, digests)).fetchall()}
        if set(digests) != set(matched):
            raise ValueError('Registry dependencies changed; review deletion again')
        return [matched[digest] for digest in digests]

    flags = _INLINE_REFERENCES.load(references, load)
    return [{**row, 'referenced': flags[(*namespace, row['digest'])]} for row in headers]


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
    revisions = {row["id"]: row for row in inline_consumers(connection, "experiment_revisions", kind, records)}
    runs = {row["id"]: row for row in fetch(connection, "runs")}
    evaluations = fetch(connection, "evaluations")
    # Dependency checks need identities and booleans, not executable capsules.
    # Read immutable references without hydrating each multi-megabyte body.
    stages = {row["id"]: row for row in fetch(connection, "workflow_stages", raw_payloads=True)}

    def block_stage(stage):
        consumers = [row for row in evaluations if row["stage_id"] == stage["id"]]
        if consumers:
            for row in consumers:
                block("evaluation", row, "Delete this evaluation first")
        elif stage["run_id"] in runs:
            block("run", runs[stage["run_id"]], "Delete this training run first")

    for revision in revisions.values():
        if revision["referenced"]:
            block("experiment", experiments[revision["experiment_id"]], "Delete this experiment and its revisions first")
    for variant in inline_consumers(connection, "variants", kind, records):
        if variant["referenced"]:
            revision = revisions[variant["experiment_revision_id"]]
            block("experiment", experiments[revision["experiment_id"]], "Delete this experiment and its pinned variants first")
            for run in runs.values():
                if run["variant_id"] == variant["id"]:
                    block("run", run, "Delete this training run first")
    for stage, matched in zip(stages.values(), consumer_flags(connection, list(stages.values()), "resolved_config_json", kind, records)):
        if matched:
            block_stage(stage)
    attempts = fetch(connection, "job_attempts", raw_payloads=True)
    for attempt, matched in zip(attempts, consumer_flags(connection, attempts, "execution_snapshot_json", kind, records)):
        if matched:
            block_stage(stages[attempt["stage_id"]])
