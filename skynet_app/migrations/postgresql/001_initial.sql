-- Initial PostgreSQL schema: preserves the complete SQLite workspace model.
-- Historical JSON stays TEXT so signed manifests retain identical bytes.
CREATE FUNCTION current_workspace_id() RETURNS TEXT LANGUAGE SQL STABLE AS $$
    SELECT nullif(current_setting('skynet.workspace_id', true), '');
$$;
CREATE FUNCTION json_extract(document TEXT, path TEXT) RETURNS TEXT
LANGUAGE SQL IMMUTABLE STRICT AS $$
    SELECT document::jsonb #>> string_to_array(substr(path, 3), '.');
$$;
CREATE FUNCTION json_each(document TEXT, path TEXT DEFAULT '$') RETURNS TABLE(value TEXT)
LANGUAGE SQL IMMUTABLE STRICT AS $$
    SELECT jsonb_array_elements_text(CASE WHEN path='$' THEN document::jsonb
        ELSE document::jsonb #> string_to_array(substr(path, 3), '.') END);
$$;

CREATE TABLE capture_processing_jobs (
                id TEXT PRIMARY KEY, request_sha256 TEXT UNIQUE NOT NULL, capture_sha256 TEXT NOT NULL,
                state TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);

CREATE TABLE cluster_snapshots (
    id TEXT PRIMARY KEY,
    gateway TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    gpu_usage_json TEXT NOT NULL,
    nodes_json TEXT NOT NULL,
    queue_json TEXT NOT NULL,
    sha256 TEXT NOT NULL
);

CREATE TABLE collection_adapters (
    id TEXT PRIMARY KEY,
    adapter_key TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    manifest_json TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived_at TEXT
);

CREATE TABLE data_bundles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    manifest_json TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    archived_at TEXT,
    UNIQUE(name, version)
);

CREATE TABLE data_resources (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    namespace TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived_at TEXT,
    UNIQUE(provider, namespace, name)
);

CREATE TABLE evaluation_suites (
    id TEXT PRIMARY KEY,
    evaluator_adapter TEXT NOT NULL,
    evaluator_version TEXT NOT NULL,
    name TEXT NOT NULL,
    suite_version TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    catalog_path TEXT,
    config_json TEXT NOT NULL,
    enabled BIGINT NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(evaluator_adapter, evaluator_version, name, suite_version)
);

CREATE TABLE live_conversions (id TEXT PRIMARY KEY, payload_json TEXT NOT NULL);

CREATE TABLE live_xr_consent (url TEXT PRIMARY KEY, accepted_at TEXT NOT NULL);

CREATE TABLE live_xr_sessions (id TEXT PRIMARY KEY, payload_json TEXT NOT NULL);

CREATE TABLE local_captures (
                sha256 TEXT PRIMARY KEY, provider TEXT NOT NULL, session_id TEXT NOT NULL,
                filename TEXT NOT NULL, size_bytes BIGINT NOT NULL, summary_json TEXT NOT NULL,
                version_id TEXT NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(provider, session_id));

CREATE TABLE policy_exports (id TEXT PRIMARY KEY, payload_json TEXT NOT NULL);

CREATE TABLE source_metadata_cache (
                    cache_key TEXT PRIMARY KEY,
                    cache_kind TEXT NOT NULL,
                    repository_url TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

CREATE TABLE source_repository_selections(owner_id TEXT NOT NULL, repository_url TEXT NOT NULL, branch_name TEXT NOT NULL, commits_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(owner_id, repository_url));

CREATE TABLE tensor_trace_cache (cache_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL, created_at TEXT NOT NULL);

CREATE TABLE workspace_migrations (version BIGINT PRIMARY KEY);

CREATE TABLE workspaces (
            id TEXT PRIMARY KEY, email TEXT UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

CREATE TABLE "adapters" (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    repository_url TEXT,
    capabilities_json TEXT NOT NULL,
    schema_json TEXT NOT NULL,
    enabled BIGINT NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, adapter_key TEXT, version_number BIGINT, description TEXT NOT NULL DEFAULT '', manifest_json TEXT, manifest_sha256 TEXT, archived_at TEXT, created_by TEXT, seed_key TEXT, source_adapter_key TEXT, source_version_number BIGINT, change_note TEXT NOT NULL DEFAULT '', owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id),
    UNIQUE(owner_id, name, version)
);

CREATE TABLE data_resource_versions (
    id TEXT PRIMARY KEY,
    resource_id TEXT NOT NULL REFERENCES data_resources(id) ON DELETE RESTRICT,
    revision TEXT NOT NULL,
    format TEXT NOT NULL,
    path TEXT NOT NULL,
    source_uri TEXT,
    manifest_sha256 TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'READY',
    size_bytes BIGINT,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(resource_id, revision, format)
);

CREATE TABLE "events" (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    old_status TEXT,
    new_status TEXT,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id));

CREATE TABLE "projects" (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    archived_at TEXT
, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id), UNIQUE(owner_id, name));

CREATE TABLE "runtime_profiles" (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    backend TEXT NOT NULL,
    version BIGINT NOT NULL DEFAULT 1,
    config_json TEXT NOT NULL,
    enabled BIGINT NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id), UNIQUE(owner_id, name));

CREATE TABLE slack_notifications (
            owner_id TEXT PRIMARY KEY REFERENCES workspaces(id),
            enabled BIGINT NOT NULL DEFAULT 0,
            events_json TEXT NOT NULL DEFAULT '["submitted","running","cancelled","failed","completed"]',
            app_url TEXT NOT NULL DEFAULT '',
            last_error TEXT, last_sent_at TEXT, updated_at TEXT NOT NULL
        );

CREATE TABLE "tracking_bindings" (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    remote_id TEXT,
    remote_url TEXT,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id),
    UNIQUE(provider, scope_type, scope_id),
    CHECK(provider IN ('mlflow', 'wandb')),
    CHECK(scope_type IN ('experiment', 'run'))
);

CREATE TABLE "tracking_connections" (
    provider TEXT NOT NULL,
    endpoint TEXT,
    workspace TEXT,
    config_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL, owner_id TEXT DEFAULT 'legacy' REFERENCES workspaces(id),
    PRIMARY KEY(owner_id, provider), CHECK(provider IN ('mlflow', 'wandb'))
);

CREATE TABLE workspace_sessions (
            token_hash TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id),
            expires_at BIGINT NOT NULL
        );

CREATE TABLE workspace_storage (
            owner_id TEXT PRIMARY KEY NOT NULL REFERENCES workspaces(id),
            work_root TEXT NOT NULL
        );

CREATE TABLE "adapter_validations" (
    id TEXT PRIMARY KEY,
    adapter_key TEXT NOT NULL,
    adapter_version_id TEXT NOT NULL REFERENCES adapters(id) ON DELETE RESTRICT,
    status TEXT NOT NULL,
    repository_url TEXT,
    source_revision TEXT,
    evidence_json TEXT NOT NULL,
    errors_json TEXT NOT NULL,
    resolved_runtime_json TEXT NOT NULL,
    created_by TEXT,
    created_at TEXT NOT NULL
, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id));

CREATE TABLE collection_sessions (
    id TEXT PRIMARY KEY,
    adapter_id TEXT NOT NULL REFERENCES collection_adapters(id) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    adapter_snapshot_json TEXT NOT NULL,
    config_snapshot_json TEXT NOT NULL,
    software_snapshot_json TEXT NOT NULL,
    calibration_snapshot_json TEXT NOT NULL,
    storage_snapshot_json TEXT NOT NULL,
    resources_snapshot_json TEXT NOT NULL,
    canonical_manifest_json TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    sbatch_text TEXT,
    sbatch_sha256 TEXT,
    gateway TEXT,
    slurm_job_id TEXT,
    sbatch_path TEXT,
    stdout_path TEXT,
    stderr_path TEXT,
    error_json TEXT,
    restart_of_session_id TEXT REFERENCES collection_sessions(id) ON DELETE RESTRICT,
    registered_resource_id TEXT REFERENCES data_resources(id) ON DELETE RESTRICT,
    registered_version_id TEXT REFERENCES data_resource_versions(id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    preflighted_at TEXT,
    prepared_at TEXT,
    submitted_at TEXT,
    started_at TEXT,
    captured_at TEXT,
    completed_at TEXT
, capture_snapshot_json TEXT NOT NULL DEFAULT '{}', capabilities_snapshot_json TEXT NOT NULL DEFAULT '{}');

CREATE TABLE data_bundle_assignments (
    bundle_id TEXT NOT NULL REFERENCES data_bundles(id) ON DELETE RESTRICT,
    role TEXT NOT NULL,
    position BIGINT NOT NULL,
    version_id TEXT NOT NULL REFERENCES data_resource_versions(id) ON DELETE RESTRICT,
    mount_path TEXT,
    required BIGINT NOT NULL DEFAULT 1,
    config_json TEXT NOT NULL,
    PRIMARY KEY(bundle_id, role, position)
);

CREATE TABLE data_derivations (
    id TEXT PRIMARY KEY,
    output_version_id TEXT NOT NULL UNIQUE REFERENCES data_resource_versions(id) ON DELETE RESTRICT,
    converter_repository TEXT NOT NULL,
    converter_commit TEXT NOT NULL,
    converter_config_json TEXT NOT NULL,
    runtime_lock_sha256 TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE data_imports (
    id TEXT PRIMARY KEY,
    resource_id TEXT NOT NULL REFERENCES data_resources(id) ON DELETE RESTRICT,
    state TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT,
    gateway TEXT,
    slurm_job_id TEXT,
    slurm_state TEXT,
    exit_code TEXT,
    node_list TEXT,
    run_directory TEXT,
    script_path TEXT,
    result_path TEXT,
    stdout_path TEXT,
    stderr_path TEXT,
    version_id TEXT REFERENCES data_resource_versions(id) ON DELETE RESTRICT,
    bundle_id TEXT REFERENCES data_bundles(id) ON DELETE RESTRICT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE data_locations (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL REFERENCES data_resource_versions(id) ON DELETE RESTRICT,
    kind TEXT NOT NULL,
    host TEXT NOT NULL,
    path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    verified_at TEXT NOT NULL,
    UNIQUE(version_id, host, path)
);

CREATE TABLE "experiments" (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'DRAFT',
    mlflow_experiment_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id),
    UNIQUE(project_id, name)
);

CREATE TABLE collection_session_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES collection_sessions(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    old_status TEXT,
    new_status TEXT,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE data_derivation_inputs (
    derivation_id TEXT NOT NULL REFERENCES data_derivations(id) ON DELETE RESTRICT,
    input_version_id TEXT NOT NULL REFERENCES data_resource_versions(id) ON DELETE RESTRICT,
    role TEXT NOT NULL DEFAULT 'input',
    position BIGINT NOT NULL,
    PRIMARY KEY(derivation_id, role, position),
    UNIQUE(derivation_id, input_version_id, role)
);

CREATE TABLE "experiment_revisions" (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    revision_number BIGINT NOT NULL,
    spec_schema_version TEXT NOT NULL,
    requested_spec_json TEXT NOT NULL,
    requested_spec_sha256 TEXT NOT NULL,
    created_by TEXT,
    created_at TEXT NOT NULL, submitted_at TEXT, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id),
    UNIQUE(experiment_id, revision_number)
);

CREATE TABLE "variants" (
    id TEXT PRIMARY KEY,
    experiment_revision_id TEXT NOT NULL REFERENCES experiment_revisions(id) ON DELETE CASCADE,
    variant_index BIGINT NOT NULL,
    name TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    resolved_spec_json TEXT NOT NULL,
    resolved_spec_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id),
    UNIQUE(experiment_revision_id, variant_index),
    UNIQUE(experiment_revision_id, resolved_spec_sha256)
);

CREATE TABLE "runs" (
                    id TEXT PRIMARY KEY,
                    variant_id TEXT NOT NULL REFERENCES variants(id) ON DELETE RESTRICT,
                    seed BIGINT NOT NULL,
                    run_number BIGINT NOT NULL DEFAULT 1 CHECK(run_number >= 1),
                    restarted_from_run_id TEXT REFERENCES runs(id) ON DELETE RESTRICT,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    adapter_name TEXT NOT NULL,
                    adapter_version TEXT NOT NULL,
                    source_commit TEXT,
                    runtime_profile TEXT,
                    run_directory TEXT NOT NULL,
                    mlflow_run_id TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id),
                    UNIQUE(variant_id, seed, run_number),
                    CHECK(restarted_from_run_id IS NULL OR restarted_from_run_id <> id)
                );

CREATE TABLE "workflow_stages" (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage_type TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    resolved_config_json TEXT NOT NULL,
    auto_resume BIGINT NOT NULL DEFAULT 0,
    max_attempts BIGINT NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id),
    UNIQUE(run_id, name)
);

CREATE TABLE "job_attempts" (
    id TEXT PRIMARY KEY,
    stage_id TEXT NOT NULL REFERENCES workflow_stages(id) ON DELETE CASCADE,
    attempt_number BIGINT NOT NULL,
    slurm_job_id TEXT,
    slurm_array_job_id TEXT,
    slurm_array_task_id TEXT,
    gateway TEXT,
    account TEXT,
    partition_name TEXT,
    node_list TEXT,
    gpu_type TEXT,
    gpu_count BIGINT,
    cpu_count BIGINT,
    memory_mb BIGINT,
    time_limit_seconds BIGINT,
    status TEXT NOT NULL DEFAULT 'CREATED',
    slurm_state TEXT,
    slurm_reason TEXT,
    exit_code TEXT,
    restart_count BIGINT NOT NULL DEFAULT 0,
    sbatch_path TEXT,
    stdout_path TEXT,
    stderr_path TEXT,
    resume_checkpoint_id TEXT,
    cluster_snapshot_id TEXT,
    submitted_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, execution_snapshot_json TEXT, execution_snapshot_sha256 TEXT, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id),
    UNIQUE(stage_id, attempt_number)
);

CREATE TABLE notification_outbox (
            id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
            owner_id TEXT NOT NULL REFERENCES workspaces(id),
            stage_id TEXT NOT NULL REFERENCES workflow_stages(id) ON DELETE CASCADE,
            attempt_key TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL, job_status TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (to_char(clock_timestamp() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')),
            status TEXT NOT NULL DEFAULT 'pending', attempts BIGINT NOT NULL DEFAULT 0,
            due_at DOUBLE PRECISION NOT NULL DEFAULT 0, lease_token TEXT,
            last_error TEXT, delivered_at TEXT,
            UNIQUE(owner_id, stage_id, attempt_key, category)
        );

CREATE TABLE stage_dependencies (
    stage_id TEXT NOT NULL REFERENCES workflow_stages(id) ON DELETE CASCADE,
    depends_on_stage_id TEXT NOT NULL REFERENCES workflow_stages(id) ON DELETE CASCADE,
    dependency_type TEXT NOT NULL DEFAULT 'AFTER_OK',
    created_at TEXT NOT NULL,
    PRIMARY KEY(stage_id, depends_on_stage_id),
    CHECK(stage_id <> depends_on_stage_id)
);

CREATE TABLE "checkpoints" (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    produced_by_attempt_id TEXT REFERENCES job_attempts(id) ON DELETE SET NULL,
    training_step BIGINT,
    checkpoint_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    size_bytes BIGINT,
    is_resumable BIGINT NOT NULL DEFAULT 0,
    is_selected_for_inference BIGINT NOT NULL DEFAULT 0,
    validation_metric TEXT,
    validation_metric_value DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'AVAILABLE',
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    pruned_at TEXT, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id),
    UNIQUE(run_id, path)
);

CREATE TABLE "manifests" (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    attempt_id TEXT REFERENCES job_attempts(id) ON DELETE SET NULL,
    manifest_type TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id),
    UNIQUE(run_id, manifest_type, path)
);

CREATE TABLE "training_progress_samples" (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    attempt_id TEXT NOT NULL REFERENCES job_attempts(id) ON DELETE CASCADE,
    restart_count BIGINT NOT NULL DEFAULT 0,
    completed BIGINT NOT NULL CHECK(completed >= 0),
    total BIGINT CHECK(total > 0),
    unit TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    created_at TEXT NOT NULL, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id),
    UNIQUE(attempt_id, restart_count, completed)
);

CREATE TABLE "evaluations" (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage_id TEXT REFERENCES workflow_stages(id) ON DELETE SET NULL,
    checkpoint_id TEXT REFERENCES checkpoints(id) ON DELETE RESTRICT,
    evaluation_suite_id TEXT REFERENCES evaluation_suites(id) ON DELETE RESTRICT,
    evaluator_adapter TEXT NOT NULL,
    evaluator_version TEXT NOT NULL,
    suite_name TEXT NOT NULL,
    suite_version TEXT NOT NULL,
    task_selection_json TEXT NOT NULL,
    seeds_json TEXT NOT NULL,
    episodes_per_task BIGINT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    progress_completed BIGINT NOT NULL DEFAULT 0,
    progress_total BIGINT NOT NULL DEFAULT 0,
    result_path TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL
, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id));

CREATE TABLE "artifacts" (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage_id TEXT REFERENCES workflow_stages(id) ON DELETE SET NULL,
    evaluation_id TEXT REFERENCES evaluations(id) ON DELETE SET NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    size_bytes BIGINT,
    retention_policy TEXT,
    mlflow_artifact_uri TEXT,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    deleted_at TEXT, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id),
    UNIQUE(run_id, path)
);

CREATE TABLE "evaluation_episodes" (
    id TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    task TEXT NOT NULL,
    seed BIGINT NOT NULL,
    episode_index BIGINT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    attempt_count BIGINT NOT NULL DEFAULT 0,
    success BIGINT,
    reward DOUBLE PRECISION,
    episode_length BIGINT,
    failure_reason TEXT,
    metrics_json TEXT NOT NULL,
    video_path TEXT,
    raw_result_path TEXT,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id),
    UNIQUE(evaluation_id, task, seed, episode_index)
);

CREATE TABLE "metrics" (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    evaluation_id TEXT REFERENCES evaluations(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    scope TEXT NOT NULL,
    step BIGINT,
    value DOUBLE PRECISION NOT NULL,
    unit TEXT,
    sample_count BIGINT,
    recorded_at TEXT NOT NULL
, owner_id TEXT DEFAULT (coalesce(current_workspace_id(), 'legacy')) REFERENCES workspaces(id));

CREATE UNIQUE INDEX idx_adapter_key_version ON adapters(adapter_key, version_number);

CREATE INDEX idx_adapter_validations_owner ON adapter_validations(owner_id);

CREATE INDEX idx_adapter_validations_version ON adapter_validations(adapter_version_id, created_at DESC);

CREATE INDEX idx_adapters_owner ON adapters(owner_id);

CREATE INDEX idx_adapters_registry_state ON adapters(adapter_key, archived_at, version_number DESC);

CREATE INDEX idx_adapters_seed_key ON adapters(seed_key);

CREATE INDEX idx_artifacts_owner ON artifacts(owner_id);

CREATE INDEX idx_artifacts_run_type ON artifacts(run_id, artifact_type, created_at);

CREATE UNIQUE INDEX idx_attempt_common_hyperparameter_receipt
ON events(entity_type, entity_id, event_type)
WHERE entity_type = 'job_attempt'
  AND event_type = 'COMMON_HYPERPARAMETERS_ENRICHED_V1';

CREATE INDEX idx_attempts_slurm_job ON job_attempts(slurm_job_id);

CREATE INDEX idx_attempts_stage_status ON job_attempts(stage_id, status, attempt_number);

CREATE INDEX idx_checkpoints_owner ON checkpoints(owner_id);

CREATE INDEX idx_checkpoints_run_step ON checkpoints(run_id, training_step, created_at);

CREATE INDEX idx_collection_adapters_state
ON collection_adapters(archived_at, adapter_key);

CREATE INDEX idx_collection_events_session
ON collection_session_events(session_id, created_at);

CREATE INDEX idx_collection_sessions_adapter
ON collection_sessions(adapter_id, created_at DESC);

CREATE INDEX idx_collection_sessions_job
ON collection_sessions(slurm_job_id);

CREATE INDEX idx_collection_sessions_state
ON collection_sessions(status, created_at DESC);

CREATE INDEX idx_data_bundle_assignments_version ON data_bundle_assignments(version_id);

CREATE INDEX idx_data_bundles_state ON data_bundles(archived_at, name, version);

CREATE INDEX idx_data_derivation_inputs_version ON data_derivation_inputs(input_version_id);

CREATE INDEX idx_data_imports_resource ON data_imports(resource_id, created_at DESC);

CREATE INDEX idx_data_imports_state ON data_imports(state, created_at DESC);

CREATE INDEX idx_data_resources_identity ON data_resources(provider, namespace, name);

CREATE INDEX idx_data_resources_kind ON data_resources(kind, archived_at);

CREATE INDEX idx_data_versions_resource ON data_resource_versions(resource_id, created_at DESC);

CREATE INDEX idx_data_versions_status ON data_resource_versions(status, format);

CREATE INDEX idx_episodes_evaluation_status ON evaluation_episodes(evaluation_id, status, task);

CREATE INDEX idx_evaluation_episodes_owner ON evaluation_episodes(owner_id);

CREATE INDEX idx_evaluations_owner ON evaluations(owner_id);

CREATE INDEX idx_evaluations_run_status ON evaluations(run_id, status, created_at);

CREATE INDEX idx_events_entity ON events(entity_type, entity_id, created_at);

CREATE INDEX idx_events_owner ON events(owner_id);

CREATE INDEX idx_experiment_revisions_owner ON experiment_revisions(owner_id);

CREATE INDEX idx_experiments_owner ON experiments(owner_id);

CREATE INDEX idx_experiments_project ON experiments(project_id, created_at);

CREATE INDEX idx_job_attempts_owner ON job_attempts(owner_id);

CREATE INDEX idx_manifests_owner ON manifests(owner_id);

CREATE INDEX idx_manifests_run_type ON manifests(run_id, manifest_type, created_at);

CREATE INDEX idx_metrics_owner ON metrics(owner_id);

CREATE INDEX idx_metrics_run_name ON metrics(run_id, name, step);

CREATE INDEX idx_projects_owner ON projects(owner_id);

CREATE INDEX idx_revisions_experiment ON experiment_revisions(experiment_id, revision_number);

CREATE INDEX idx_runs_owner ON runs(owner_id);

CREATE INDEX idx_runs_restarted_from ON runs(restarted_from_run_id);

CREATE INDEX idx_runs_variant_status ON runs(variant_id, status, created_at);

CREATE INDEX idx_runtime_profiles_owner ON runtime_profiles(owner_id);

CREATE INDEX idx_snapshots_captured ON cluster_snapshots(captured_at);

CREATE INDEX idx_stages_run_status ON workflow_stages(run_id, status, created_at);

CREATE INDEX idx_tracking_bindings_owner ON tracking_bindings(owner_id);

CREATE INDEX idx_tracking_bindings_scope
ON tracking_bindings(scope_type, scope_id);

CREATE INDEX idx_tracking_connections_owner ON tracking_connections(owner_id);

CREATE INDEX idx_training_progress_samples_owner ON training_progress_samples(owner_id);

CREATE INDEX idx_variants_owner ON variants(owner_id);

CREATE INDEX idx_variants_revision ON variants(experiment_revision_id, variant_index);

CREATE INDEX idx_workflow_stages_owner ON workflow_stages(owner_id);

CREATE INDEX notification_outbox_delivery
        ON notification_outbox(owner_id, status, id);

CREATE INDEX source_metadata_cache_lookup
                ON source_metadata_cache(cache_kind, repository_url, updated_at)
                ;

CREATE INDEX workspace_session_expiry ON workspace_sessions(expires_at);

INSERT INTO workspaces(id) VALUES ('legacy');
INSERT INTO workspace_migrations VALUES (1);

CREATE FUNCTION workspace_adapter_validations_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := COALESCE(current_workspace_id(), NEW.owner_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_adapter_validations_guard BEFORE INSERT OR UPDATE OR DELETE ON adapter_validations
FOR EACH ROW EXECUTE FUNCTION workspace_adapter_validations_guard();

CREATE FUNCTION workspace_adapters_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := COALESCE(current_workspace_id(), NEW.owner_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_adapters_guard BEFORE INSERT OR UPDATE OR DELETE ON adapters
FOR EACH ROW EXECUTE FUNCTION workspace_adapters_guard();

CREATE FUNCTION workspace_artifacts_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := (SELECT owner_id FROM runs WHERE id=NEW.run_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_artifacts_guard BEFORE INSERT OR UPDATE OR DELETE ON artifacts
FOR EACH ROW EXECUTE FUNCTION workspace_artifacts_guard();

CREATE FUNCTION workspace_checkpoints_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := (SELECT owner_id FROM runs WHERE id=NEW.run_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_checkpoints_guard BEFORE INSERT OR UPDATE OR DELETE ON checkpoints
FOR EACH ROW EXECUTE FUNCTION workspace_checkpoints_guard();

CREATE FUNCTION workspace_evaluation_episodes_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := (SELECT owner_id FROM evaluations WHERE id=NEW.evaluation_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_evaluation_episodes_guard BEFORE INSERT OR UPDATE OR DELETE ON evaluation_episodes
FOR EACH ROW EXECUTE FUNCTION workspace_evaluation_episodes_guard();

CREATE FUNCTION workspace_evaluations_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := (SELECT owner_id FROM runs WHERE id=NEW.run_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_evaluations_guard BEFORE INSERT OR UPDATE OR DELETE ON evaluations
FOR EACH ROW EXECUTE FUNCTION workspace_evaluations_guard();

CREATE FUNCTION workspace_events_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := COALESCE(current_workspace_id(), NEW.owner_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_events_guard BEFORE INSERT OR UPDATE OR DELETE ON events
FOR EACH ROW EXECUTE FUNCTION workspace_events_guard();

CREATE FUNCTION workspace_experiment_revisions_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := (SELECT owner_id FROM experiments WHERE id=NEW.experiment_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_experiment_revisions_guard BEFORE INSERT OR UPDATE OR DELETE ON experiment_revisions
FOR EACH ROW EXECUTE FUNCTION workspace_experiment_revisions_guard();

CREATE FUNCTION workspace_experiments_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := COALESCE((SELECT owner_id FROM projects WHERE id=NEW.project_id), current_workspace_id(), NEW.owner_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_experiments_guard BEFORE INSERT OR UPDATE OR DELETE ON experiments
FOR EACH ROW EXECUTE FUNCTION workspace_experiments_guard();

CREATE FUNCTION workspace_job_attempts_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := (SELECT owner_id FROM workflow_stages WHERE id=NEW.stage_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_job_attempts_guard BEFORE INSERT OR UPDATE OR DELETE ON job_attempts
FOR EACH ROW EXECUTE FUNCTION workspace_job_attempts_guard();

CREATE FUNCTION workspace_manifests_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := (SELECT owner_id FROM runs WHERE id=NEW.run_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_manifests_guard BEFORE INSERT OR UPDATE OR DELETE ON manifests
FOR EACH ROW EXECUTE FUNCTION workspace_manifests_guard();

CREATE FUNCTION workspace_metrics_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := (SELECT owner_id FROM runs WHERE id=NEW.run_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_metrics_guard BEFORE INSERT OR UPDATE OR DELETE ON metrics
FOR EACH ROW EXECUTE FUNCTION workspace_metrics_guard();

CREATE FUNCTION workspace_projects_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := COALESCE(current_workspace_id(), NEW.owner_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_projects_guard BEFORE INSERT OR UPDATE OR DELETE ON projects
FOR EACH ROW EXECUTE FUNCTION workspace_projects_guard();

CREATE FUNCTION workspace_runs_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := (SELECT owner_id FROM variants WHERE id=NEW.variant_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_runs_guard BEFORE INSERT OR UPDATE OR DELETE ON runs
FOR EACH ROW EXECUTE FUNCTION workspace_runs_guard();

CREATE FUNCTION workspace_runtime_profiles_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := COALESCE(current_workspace_id(), NEW.owner_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_runtime_profiles_guard BEFORE INSERT OR UPDATE OR DELETE ON runtime_profiles
FOR EACH ROW EXECUTE FUNCTION workspace_runtime_profiles_guard();

CREATE FUNCTION workspace_tracking_bindings_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := COALESCE(current_workspace_id(), NEW.owner_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_tracking_bindings_guard BEFORE INSERT OR UPDATE OR DELETE ON tracking_bindings
FOR EACH ROW EXECUTE FUNCTION workspace_tracking_bindings_guard();

CREATE FUNCTION workspace_training_progress_samples_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := (SELECT owner_id FROM runs WHERE id=NEW.run_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_training_progress_samples_guard BEFORE INSERT OR UPDATE OR DELETE ON training_progress_samples
FOR EACH ROW EXECUTE FUNCTION workspace_training_progress_samples_guard();

CREATE FUNCTION workspace_variants_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := (SELECT owner_id FROM experiment_revisions WHERE id=NEW.experiment_revision_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_variants_guard BEFORE INSERT OR UPDATE OR DELETE ON variants
FOR EACH ROW EXECUTE FUNCTION workspace_variants_guard();

CREATE FUNCTION workspace_workflow_stages_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE inherited_owner TEXT;
BEGIN
    IF TG_OP='INSERT' THEN
        inherited_owner := (SELECT owner_id FROM runs WHERE id=NEW.run_id);
        IF current_workspace_id() IS NOT NULL AND inherited_owner IS DISTINCT FROM current_workspace_id() THEN
            RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
        END IF;
        NEW.owner_id := inherited_owner;
        RETURN NEW;
    END IF;
    IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
        RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' THEN RETURN NEW; END IF;
    RETURN OLD;
END; $$;
CREATE TRIGGER workspace_workflow_stages_guard BEFORE INSERT OR UPDATE OR DELETE ON workflow_stages
FOR EACH ROW EXECUTE FUNCTION workspace_workflow_stages_guard();

CREATE FUNCTION adapter_registry_name_insert_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF NEW.adapter_key IS NOT NULL AND EXISTS (
                SELECT 1 FROM adapters WHERE lower(name)=lower(NEW.name)
                AND adapter_key <> NEW.adapter_key AND owner_id IS NOT DISTINCT FROM NEW.owner_id
            ) THEN
RAISE EXCEPTION 'adapter name already exists' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER adapter_registry_name_insert BEFORE INSERT ON adapters
FOR EACH ROW EXECUTE FUNCTION adapter_registry_name_insert_fn();

CREATE FUNCTION adapter_registry_name_update_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF NEW.adapter_key IS NOT NULL AND EXISTS (
                SELECT 1 FROM adapters WHERE lower(name)=lower(NEW.name)
                AND adapter_key <> NEW.adapter_key AND owner_id IS NOT DISTINCT FROM NEW.owner_id
            ) THEN
RAISE EXCEPTION 'adapter name already exists' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER adapter_registry_name_update BEFORE UPDATE ON adapters
FOR EACH ROW EXECUTE FUNCTION adapter_registry_name_update_fn();

CREATE FUNCTION adapter_registry_seed_insert_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF NEW.seed_key IS NOT NULL AND EXISTS (
                    SELECT 1 FROM adapters
                    WHERE seed_key = NEW.seed_key AND adapter_key <> NEW.adapter_key
                ) THEN
RAISE EXCEPTION 'adapter seed key already exists' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER adapter_registry_seed_insert BEFORE INSERT ON adapters
FOR EACH ROW EXECUTE FUNCTION adapter_registry_seed_insert_fn();

CREATE FUNCTION adapter_registry_seed_update_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF NEW.seed_key IS NOT NULL AND EXISTS (
                    SELECT 1 FROM adapters
                    WHERE seed_key = NEW.seed_key AND adapter_key <> NEW.adapter_key
                ) THEN
RAISE EXCEPTION 'adapter seed key already exists' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER adapter_registry_seed_update BEFORE UPDATE OF seed_key, adapter_key ON adapters
FOR EACH ROW EXECUTE FUNCTION adapter_registry_seed_update_fn();

CREATE FUNCTION adapter_versions_immutable_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
RAISE EXCEPTION 'adapter versions are immutable' USING ERRCODE='23514';
RETURN NEW;
END; $$;
CREATE TRIGGER adapter_versions_immutable BEFORE UPDATE OF adapter_key, version_number, manifest_json, manifest_sha256,
                                 name, description, repository_url, capabilities_json, schema_json,
                                 created_by, change_note, created_at, source_adapter_key,
                                 source_version_number ON adapters
FOR EACH ROW EXECUTE FUNCTION adapter_versions_immutable_fn();

CREATE FUNCTION attempt_common_hyperparameter_receipt_no_delete_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF OLD.entity_type = 'job_attempt'
 AND OLD.event_type = 'COMMON_HYPERPARAMETERS_ENRICHED_V1' THEN
RAISE EXCEPTION 'attempt common hyperparameter receipts are immutable' USING ERRCODE='23514';
END IF;
RETURN OLD;
END; $$;
CREATE TRIGGER attempt_common_hyperparameter_receipt_no_delete BEFORE DELETE ON events
FOR EACH ROW EXECUTE FUNCTION attempt_common_hyperparameter_receipt_no_delete_fn();

CREATE FUNCTION attempt_common_hyperparameter_receipt_no_update_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF OLD.entity_type = 'job_attempt'
 AND OLD.event_type = 'COMMON_HYPERPARAMETERS_ENRICHED_V1' THEN
RAISE EXCEPTION 'attempt common hyperparameter receipts are immutable' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER attempt_common_hyperparameter_receipt_no_update BEFORE UPDATE ON events
FOR EACH ROW EXECUTE FUNCTION attempt_common_hyperparameter_receipt_no_update_fn();

CREATE FUNCTION attempt_snapshot_immutable_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
RAISE EXCEPTION 'attempt execution snapshots are immutable' USING ERRCODE='23514';
RETURN NEW;
END; $$;
CREATE TRIGGER attempt_snapshot_immutable BEFORE UPDATE OF execution_snapshot_json, execution_snapshot_sha256 ON job_attempts
FOR EACH ROW EXECUTE FUNCTION attempt_snapshot_immutable_fn();

CREATE FUNCTION collection_session_snapshots_immutable_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
RAISE EXCEPTION 'collection session snapshots are immutable' USING ERRCODE='23514';
RETURN NEW;
END; $$;
CREATE TRIGGER collection_session_snapshots_immutable BEFORE UPDATE OF
                    adapter_id,
                    adapter_snapshot_json,
                    config_snapshot_json,
                    software_snapshot_json,
                    calibration_snapshot_json,
                    storage_snapshot_json,
                    resources_snapshot_json,
                    capture_snapshot_json,
                    capabilities_snapshot_json,
                    canonical_manifest_json,
                    manifest_sha256,
                    restart_of_session_id ON collection_sessions
FOR EACH ROW EXECUTE FUNCTION collection_session_snapshots_immutable_fn();

CREATE FUNCTION data_bundle_assignments_no_delete_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF coalesce(current_setting('skynet.allow_dataset_delete',true),'off') <> 'on' THEN
RAISE EXCEPTION 'data bundle assignments are immutable' USING ERRCODE='23514';
END IF;
RETURN OLD;
END; $$;
CREATE TRIGGER data_bundle_assignments_no_delete BEFORE DELETE ON data_bundle_assignments
FOR EACH ROW EXECUTE FUNCTION data_bundle_assignments_no_delete_fn();

CREATE FUNCTION data_bundle_assignments_no_update_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
RAISE EXCEPTION 'data bundle assignments are immutable' USING ERRCODE='23514';
RETURN NEW;
END; $$;
CREATE TRIGGER data_bundle_assignments_no_update BEFORE UPDATE ON data_bundle_assignments
FOR EACH ROW EXECUTE FUNCTION data_bundle_assignments_no_update_fn();

CREATE FUNCTION data_derivation_inputs_no_delete_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF coalesce(current_setting('skynet.allow_dataset_delete',true),'off') <> 'on' THEN
RAISE EXCEPTION 'data derivation inputs are immutable' USING ERRCODE='23514';
END IF;
RETURN OLD;
END; $$;
CREATE TRIGGER data_derivation_inputs_no_delete BEFORE DELETE ON data_derivation_inputs
FOR EACH ROW EXECUTE FUNCTION data_derivation_inputs_no_delete_fn();

CREATE FUNCTION data_derivation_inputs_no_update_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
RAISE EXCEPTION 'data derivation inputs are immutable' USING ERRCODE='23514';
RETURN NEW;
END; $$;
CREATE TRIGGER data_derivation_inputs_no_update BEFORE UPDATE ON data_derivation_inputs
FOR EACH ROW EXECUTE FUNCTION data_derivation_inputs_no_update_fn();

CREATE FUNCTION data_derivations_no_delete_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF coalesce(current_setting('skynet.allow_dataset_delete',true),'off') <> 'on' THEN
RAISE EXCEPTION 'data derivations are immutable' USING ERRCODE='23514';
END IF;
RETURN OLD;
END; $$;
CREATE TRIGGER data_derivations_no_delete BEFORE DELETE ON data_derivations
FOR EACH ROW EXECUTE FUNCTION data_derivations_no_delete_fn();

CREATE FUNCTION data_derivations_no_update_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
RAISE EXCEPTION 'data derivations are immutable' USING ERRCODE='23514';
RETURN NEW;
END; $$;
CREATE TRIGGER data_derivations_no_update BEFORE UPDATE ON data_derivations
FOR EACH ROW EXECUTE FUNCTION data_derivations_no_update_fn();

CREATE FUNCTION data_resource_versions_no_delete_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF coalesce(current_setting('skynet.allow_dataset_delete',true),'off') <> 'on' THEN
RAISE EXCEPTION 'data resource versions are immutable' USING ERRCODE='23514';
END IF;
RETURN OLD;
END; $$;
CREATE TRIGGER data_resource_versions_no_delete BEFORE DELETE ON data_resource_versions
FOR EACH ROW EXECUTE FUNCTION data_resource_versions_no_delete_fn();

CREATE FUNCTION data_resource_versions_no_update_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
RAISE EXCEPTION 'data resource versions are immutable' USING ERRCODE='23514';
RETURN NEW;
END; $$;
CREATE TRIGGER data_resource_versions_no_update BEFORE UPDATE ON data_resource_versions
FOR EACH ROW EXECUTE FUNCTION data_resource_versions_no_update_fn();

CREATE FUNCTION evaluation_suite_versions_immutable_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
RAISE EXCEPTION 'evaluation suite versions are immutable' USING ERRCODE='23514';
RETURN NEW;
END; $$;
CREATE TRIGGER evaluation_suite_versions_immutable BEFORE UPDATE OF evaluator_adapter, evaluator_version, name, suite_version,
                 description, catalog_path, config_json, created_at ON evaluation_suites
FOR EACH ROW EXECUTE FUNCTION evaluation_suite_versions_immutable_fn();

CREATE FUNCTION experiment_revision_spec_immutable_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
RAISE EXCEPTION 'experiment revision specifications are immutable' USING ERRCODE='23514';
RETURN NEW;
END; $$;
CREATE TRIGGER experiment_revision_spec_immutable BEFORE UPDATE OF experiment_id, revision_number, spec_schema_version,
                             requested_spec_json, requested_spec_sha256,
                             created_by, created_at ON experiment_revisions
FOR EACH ROW EXECUTE FUNCTION experiment_revision_spec_immutable_fn();

CREATE FUNCTION notification_outbox_owner_delete_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF current_workspace_id() IS NOT NULL
                    AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
END IF;
RETURN OLD;
END; $$;
CREATE TRIGGER notification_outbox_owner_delete BEFORE DELETE ON notification_outbox
FOR EACH ROW EXECUTE FUNCTION notification_outbox_owner_delete_fn();

CREATE FUNCTION notification_outbox_owner_immutable_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF NEW.owner_id IS DISTINCT FROM OLD.owner_id THEN
RAISE EXCEPTION 'Notification ownership is immutable' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER notification_outbox_owner_immutable BEFORE UPDATE OF owner_id ON notification_outbox
FOR EACH ROW EXECUTE FUNCTION notification_outbox_owner_immutable_fn();

CREATE FUNCTION notification_outbox_owner_insert_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF current_workspace_id() IS NOT NULL
                    AND NEW.owner_id IS DISTINCT FROM current_workspace_id() THEN
RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER notification_outbox_owner_insert BEFORE INSERT ON notification_outbox
FOR EACH ROW EXECUTE FUNCTION notification_outbox_owner_insert_fn();

CREATE FUNCTION notification_outbox_owner_update_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF current_workspace_id() IS NOT NULL
                    AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER notification_outbox_owner_update BEFORE UPDATE ON notification_outbox
FOR EACH ROW EXECUTE FUNCTION notification_outbox_owner_update_fn();

CREATE FUNCTION notify_accounted_start_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF NEW.started_at IS NOT NULL AND OLD.started_at IS NULL
            AND NEW.slurm_job_id IS NOT NULL AND NEW.slurm_job_id!=''
            AND NEW.slurm_state IN ('RUNNING','COMPLETED','FAILED','OUT_OF_MEMORY','TIMEOUT') THEN
INSERT INTO notification_outbox(owner_id,stage_id,attempt_key,category,job_status)
            SELECT r.owner_id,NEW.stage_id,NEW.id,'running','RUNNING'
            FROM workflow_stages s JOIN runs r ON r.id=s.run_id
            JOIN slack_notifications n ON n.owner_id=r.owner_id
            WHERE s.id=NEW.stage_id AND n.enabled=1
                AND EXISTS (SELECT 1 FROM json_each(n.events_json) WHERE value='running') ON CONFLICT DO NOTHING;
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER notify_accounted_start AFTER UPDATE OF started_at ON job_attempts
FOR EACH ROW EXECUTE FUNCTION notify_accounted_start_fn();

CREATE FUNCTION notify_job_submitted_insert_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF NEW.slurm_job_id IS NOT NULL AND NEW.slurm_job_id != '' THEN
INSERT INTO notification_outbox(owner_id,stage_id,attempt_key,category,job_status)
                SELECT r.owner_id, NEW.stage_id, NEW.id, 'submitted', 'SUBMITTED'
                FROM workflow_stages s JOIN runs r ON r.id=s.run_id
                JOIN slack_notifications n ON n.owner_id=r.owner_id
                WHERE s.id=NEW.stage_id AND n.enabled=1
                    AND EXISTS (SELECT 1 FROM json_each(n.events_json) WHERE value='submitted') ON CONFLICT DO NOTHING;
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER notify_job_submitted_insert AFTER INSERT ON job_attempts
FOR EACH ROW EXECUTE FUNCTION notify_job_submitted_insert_fn();

CREATE FUNCTION notify_job_submitted_update_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF NEW.slurm_job_id IS NOT NULL AND NEW.slurm_job_id != '' AND NEW.slurm_job_id IS DISTINCT FROM OLD.slurm_job_id THEN
INSERT INTO notification_outbox(owner_id,stage_id,attempt_key,category,job_status)
                SELECT r.owner_id, NEW.stage_id, NEW.id, 'submitted', 'SUBMITTED'
                FROM workflow_stages s JOIN runs r ON r.id=s.run_id
                JOIN slack_notifications n ON n.owner_id=r.owner_id
                WHERE s.id=NEW.stage_id AND n.enabled=1
                    AND EXISTS (SELECT 1 FROM json_each(n.events_json) WHERE value='submitted') ON CONFLICT DO NOTHING;
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER notify_job_submitted_update AFTER UPDATE OF slurm_job_id ON job_attempts
FOR EACH ROW EXECUTE FUNCTION notify_job_submitted_update_fn();

CREATE FUNCTION notify_workflow_status_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF NEW.status IS DISTINCT FROM OLD.status AND NEW.status IN ('RUNNING','CANCELLED','SUCCEEDED','FAILED','SUBMISSION_FAILED','BLOCKED','RETRY_PENDING') THEN
INSERT INTO notification_outbox(owner_id,stage_id,attempt_key,category,job_status)
            SELECT r.owner_id, NEW.id,
                COALESCE((SELECT id FROM job_attempts WHERE stage_id=NEW.id ORDER BY attempt_number DESC LIMIT 1),''),
                CASE NEW.status WHEN 'RUNNING' THEN 'running' WHEN 'CANCELLED' THEN 'cancelled' WHEN 'SUCCEEDED' THEN 'completed' WHEN 'FAILED' THEN 'failed' WHEN 'SUBMISSION_FAILED' THEN 'failed' WHEN 'BLOCKED' THEN 'failed' WHEN 'RETRY_PENDING' THEN 'failed' END, NEW.status
            FROM runs r JOIN slack_notifications n ON n.owner_id=r.owner_id
            WHERE r.id=NEW.run_id AND n.enabled=1
                AND EXISTS (SELECT 1 FROM job_attempts a
                    WHERE a.id=(SELECT id FROM job_attempts WHERE stage_id=NEW.id ORDER BY attempt_number DESC LIMIT 1)
                        AND a.slurm_job_id IS NOT NULL AND a.slurm_job_id!='')
                AND EXISTS (SELECT 1 FROM json_each(n.events_json) WHERE value=CASE NEW.status WHEN 'RUNNING' THEN 'running' WHEN 'CANCELLED' THEN 'cancelled' WHEN 'SUCCEEDED' THEN 'completed' WHEN 'FAILED' THEN 'failed' WHEN 'SUBMISSION_FAILED' THEN 'failed' WHEN 'BLOCKED' THEN 'failed' WHEN 'RETRY_PENDING' THEN 'failed' END) ON CONFLICT DO NOTHING;
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER notify_workflow_status AFTER UPDATE OF status ON workflow_stages
FOR EACH ROW EXECUTE FUNCTION notify_workflow_status_fn();

CREATE FUNCTION run_identity_immutable_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
RAISE EXCEPTION 'run identity is immutable' USING ERRCODE='23514';
RETURN NEW;
END; $$;
CREATE TRIGGER run_identity_immutable BEFORE UPDATE OF variant_id, seed, run_number, restarted_from_run_id,
                             adapter_name, adapter_version, source_commit,
                             runtime_profile, created_at ON runs
FOR EACH ROW EXECUTE FUNCTION run_identity_immutable_fn();

CREATE FUNCTION slack_notifications_owner_delete_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF current_workspace_id() IS NOT NULL
                    AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
END IF;
RETURN OLD;
END; $$;
CREATE TRIGGER slack_notifications_owner_delete BEFORE DELETE ON slack_notifications
FOR EACH ROW EXECUTE FUNCTION slack_notifications_owner_delete_fn();

CREATE FUNCTION slack_notifications_owner_immutable_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF NEW.owner_id IS DISTINCT FROM OLD.owner_id THEN
RAISE EXCEPTION 'Notification ownership is immutable' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER slack_notifications_owner_immutable BEFORE UPDATE OF owner_id ON slack_notifications
FOR EACH ROW EXECUTE FUNCTION slack_notifications_owner_immutable_fn();

CREATE FUNCTION slack_notifications_owner_insert_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF current_workspace_id() IS NOT NULL
                    AND NEW.owner_id IS DISTINCT FROM current_workspace_id() THEN
RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER slack_notifications_owner_insert BEFORE INSERT ON slack_notifications
FOR EACH ROW EXECUTE FUNCTION slack_notifications_owner_insert_fn();

CREATE FUNCTION slack_notifications_owner_update_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF current_workspace_id() IS NOT NULL
                    AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER slack_notifications_owner_update BEFORE UPDATE ON slack_notifications
FOR EACH ROW EXECUTE FUNCTION slack_notifications_owner_update_fn();

CREATE FUNCTION variants_immutable_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
RAISE EXCEPTION 'experiment variants are immutable' USING ERRCODE='23514';
RETURN NEW;
END; $$;
CREATE TRIGGER variants_immutable BEFORE UPDATE ON variants
FOR EACH ROW EXECUTE FUNCTION variants_immutable_fn();

CREATE FUNCTION workspace_storage_delete_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id() THEN
RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
END IF;
RETURN OLD;
END; $$;
CREATE TRIGGER workspace_storage_delete BEFORE DELETE ON workspace_storage
FOR EACH ROW EXECUTE FUNCTION workspace_storage_delete_fn();

CREATE FUNCTION workspace_storage_insert_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF current_workspace_id() IS NOT NULL AND NEW.owner_id IS DISTINCT FROM current_workspace_id() THEN
RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER workspace_storage_insert BEFORE INSERT ON workspace_storage
FOR EACH ROW EXECUTE FUNCTION workspace_storage_insert_fn();

CREATE FUNCTION workspace_storage_update_fn() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
IF NEW.owner_id IS DISTINCT FROM OLD.owner_id OR
            (current_workspace_id() IS NOT NULL AND OLD.owner_id IS DISTINCT FROM current_workspace_id()) THEN
RAISE EXCEPTION 'Record is outside this workspace' USING ERRCODE='23514';
END IF;
RETURN NEW;
END; $$;
CREATE TRIGGER workspace_storage_update BEFORE UPDATE ON workspace_storage
FOR EACH ROW EXECUTE FUNCTION workspace_storage_update_fn();
