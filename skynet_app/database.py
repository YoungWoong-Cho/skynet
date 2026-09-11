from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .data_paths import validate_mount_path
from .notification_schema import migrate_notifications
from .workspace_schema import PRIVATE_TABLES, LEGACY_WORKSPACE, migrate_workspaces, visible_sql
from .training_metrics import is_scalar


APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE_PATH = APP_ROOT / "data" / "skynet.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'DRAFT',
    mlflow_experiment_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, name)
);

CREATE TABLE IF NOT EXISTS experiment_revisions (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL,
    spec_schema_version TEXT NOT NULL,
    requested_spec_json TEXT NOT NULL,
    requested_spec_sha256 TEXT NOT NULL,
    created_by TEXT,
    created_at TEXT NOT NULL,
    submitted_at TEXT,
    UNIQUE(experiment_id, revision_number)
);

CREATE TABLE IF NOT EXISTS variants (
    id TEXT PRIMARY KEY,
    experiment_revision_id TEXT NOT NULL REFERENCES experiment_revisions(id) ON DELETE CASCADE,
    variant_index INTEGER NOT NULL,
    name TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    resolved_spec_json TEXT NOT NULL,
    resolved_spec_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(experiment_revision_id, variant_index),
    UNIQUE(experiment_revision_id, resolved_spec_sha256)
);

CREATE TABLE IF NOT EXISTS runtime_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    backend TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    config_json TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS adapters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    repository_url TEXT,
    capabilities_json TEXT NOT NULL,
    schema_json TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    adapter_key TEXT,
    version_number INTEGER,
    description TEXT NOT NULL DEFAULT '',
    manifest_json TEXT,
    manifest_sha256 TEXT,
    archived_at TEXT,
    created_by TEXT,
    change_note TEXT NOT NULL DEFAULT '',
    seed_key TEXT,
    source_adapter_key TEXT,
    source_version_number INTEGER,
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS adapter_validations (
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
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    variant_id TEXT NOT NULL REFERENCES variants(id) ON DELETE RESTRICT,
    seed INTEGER NOT NULL,
    run_number INTEGER NOT NULL DEFAULT 1 CHECK(run_number >= 1),
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
    updated_at TEXT NOT NULL,
    UNIQUE(variant_id, seed, run_number),
    CHECK(restarted_from_run_id IS NULL OR restarted_from_run_id <> id)
);

CREATE TABLE IF NOT EXISTS workflow_stages (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage_type TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    resolved_config_json TEXT NOT NULL,
    auto_resume INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(run_id, name)
);

CREATE TABLE IF NOT EXISTS stage_dependencies (
    stage_id TEXT NOT NULL REFERENCES workflow_stages(id) ON DELETE CASCADE,
    depends_on_stage_id TEXT NOT NULL REFERENCES workflow_stages(id) ON DELETE CASCADE,
    dependency_type TEXT NOT NULL DEFAULT 'AFTER_OK',
    created_at TEXT NOT NULL,
    PRIMARY KEY(stage_id, depends_on_stage_id),
    CHECK(stage_id <> depends_on_stage_id)
);

CREATE TABLE IF NOT EXISTS job_attempts (
    id TEXT PRIMARY KEY,
    stage_id TEXT NOT NULL REFERENCES workflow_stages(id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    slurm_job_id TEXT,
    slurm_array_job_id TEXT,
    slurm_array_task_id TEXT,
    gateway TEXT,
    account TEXT,
    partition_name TEXT,
    node_list TEXT,
    gpu_type TEXT,
    gpu_count INTEGER,
    cpu_count INTEGER,
    memory_mb INTEGER,
    time_limit_seconds INTEGER,
    status TEXT NOT NULL DEFAULT 'CREATED',
    slurm_state TEXT,
    slurm_reason TEXT,
    exit_code TEXT,
    restart_count INTEGER NOT NULL DEFAULT 0,
    sbatch_path TEXT,
    stdout_path TEXT,
    stderr_path TEXT,
    resume_checkpoint_id TEXT,
    cluster_snapshot_id TEXT,
    execution_snapshot_json TEXT,
    execution_snapshot_sha256 TEXT,
    submitted_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(stage_id, attempt_number)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    produced_by_attempt_id TEXT REFERENCES job_attempts(id) ON DELETE SET NULL,
    training_step INTEGER,
    checkpoint_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    size_bytes INTEGER,
    is_resumable INTEGER NOT NULL DEFAULT 0,
    is_selected_for_inference INTEGER NOT NULL DEFAULT 0,
    validation_metric TEXT,
    validation_metric_value REAL,
    status TEXT NOT NULL DEFAULT 'AVAILABLE',
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    pruned_at TEXT,
    UNIQUE(run_id, path)
);

CREATE TABLE IF NOT EXISTS evaluation_suites (
    id TEXT PRIMARY KEY,
    evaluator_adapter TEXT NOT NULL,
    evaluator_version TEXT NOT NULL,
    name TEXT NOT NULL,
    suite_version TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    catalog_path TEXT,
    config_json TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(evaluator_adapter, evaluator_version, name, suite_version)
);

CREATE TRIGGER IF NOT EXISTS evaluation_suite_versions_immutable
BEFORE UPDATE OF evaluator_adapter, evaluator_version, name, suite_version,
                 description, catalog_path, config_json, created_at
ON evaluation_suites
BEGIN
    SELECT RAISE(ABORT, 'evaluation suite versions are immutable');
END;

CREATE TABLE IF NOT EXISTS evaluations (
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
    episodes_per_task INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    progress_completed INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    result_path TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_episodes (
    id TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    task TEXT NOT NULL,
    seed INTEGER NOT NULL,
    episode_index INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    success INTEGER,
    reward REAL,
    episode_length INTEGER,
    failure_reason TEXT,
    metrics_json TEXT NOT NULL,
    video_path TEXT,
    raw_result_path TEXT,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(evaluation_id, task, seed, episode_index)
);

CREATE TABLE IF NOT EXISTS metrics (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    evaluation_id TEXT REFERENCES evaluations(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    scope TEXT NOT NULL,
    step INTEGER,
    value REAL NOT NULL,
    unit TEXT,
    sample_count INTEGER,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_progress_samples (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    attempt_id TEXT NOT NULL REFERENCES job_attempts(id) ON DELETE CASCADE,
    restart_count INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL CHECK(completed >= 0),
    total INTEGER CHECK(total > 0),
    unit TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(attempt_id, restart_count, completed)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage_id TEXT REFERENCES workflow_stages(id) ON DELETE SET NULL,
    evaluation_id TEXT REFERENCES evaluations(id) ON DELETE SET NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    size_bytes INTEGER,
    retention_policy TEXT,
    mlflow_artifact_uri TEXT,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    deleted_at TEXT,
    UNIQUE(run_id, path)
);

CREATE TABLE IF NOT EXISTS tracking_connections (
    provider TEXT PRIMARY KEY,
    endpoint TEXT,
    workspace TEXT,
    config_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    CHECK(provider IN ('mlflow', 'wandb'))
);

CREATE TABLE IF NOT EXISTS tracking_bindings (
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
    updated_at TEXT NOT NULL,
    UNIQUE(provider, scope_type, scope_id),
    CHECK(provider IN ('mlflow', 'wandb')),
    CHECK(scope_type IN ('experiment', 'run'))
);

CREATE INDEX IF NOT EXISTS idx_tracking_bindings_scope
ON tracking_bindings(scope_type, scope_id);

CREATE TABLE IF NOT EXISTS manifests (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    attempt_id TEXT REFERENCES job_attempts(id) ON DELETE SET NULL,
    manifest_type TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, manifest_type, path)
);

CREATE TABLE IF NOT EXISTS cluster_snapshots (
    id TEXT PRIMARY KEY,
    gateway TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    gpu_usage_json TEXT NOT NULL,
    nodes_json TEXT NOT NULL,
    queue_json TEXT NOT NULL,
    sha256 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    old_status TEXT,
    new_status TEXT,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_resources (
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

CREATE TABLE IF NOT EXISTS data_resource_versions (
    id TEXT PRIMARY KEY,
    resource_id TEXT NOT NULL REFERENCES data_resources(id) ON DELETE RESTRICT,
    revision TEXT NOT NULL,
    format TEXT NOT NULL,
    path TEXT NOT NULL,
    source_uri TEXT,
    manifest_sha256 TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'READY',
    size_bytes INTEGER,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(resource_id, revision, format)
);

CREATE TABLE IF NOT EXISTS data_locations (
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

CREATE TABLE IF NOT EXISTS data_derivations (
    id TEXT PRIMARY KEY,
    output_version_id TEXT NOT NULL UNIQUE REFERENCES data_resource_versions(id) ON DELETE RESTRICT,
    converter_repository TEXT NOT NULL,
    converter_commit TEXT NOT NULL,
    converter_config_json TEXT NOT NULL,
    runtime_lock_sha256 TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_derivation_inputs (
    derivation_id TEXT NOT NULL REFERENCES data_derivations(id) ON DELETE RESTRICT,
    input_version_id TEXT NOT NULL REFERENCES data_resource_versions(id) ON DELETE RESTRICT,
    role TEXT NOT NULL DEFAULT 'input',
    position INTEGER NOT NULL,
    PRIMARY KEY(derivation_id, role, position),
    UNIQUE(derivation_id, input_version_id, role)
);

CREATE TABLE IF NOT EXISTS data_bundles (
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

CREATE TABLE IF NOT EXISTS data_bundle_assignments (
    bundle_id TEXT NOT NULL REFERENCES data_bundles(id) ON DELETE RESTRICT,
    role TEXT NOT NULL,
    position INTEGER NOT NULL,
    version_id TEXT NOT NULL REFERENCES data_resource_versions(id) ON DELETE RESTRICT,
    mount_path TEXT,
    required INTEGER NOT NULL DEFAULT 1,
    config_json TEXT NOT NULL,
    PRIMARY KEY(bundle_id, role, position)
);

CREATE TABLE IF NOT EXISTS data_imports (
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

CREATE INDEX IF NOT EXISTS idx_experiments_project ON experiments(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_revisions_experiment ON experiment_revisions(experiment_id, revision_number);
CREATE INDEX IF NOT EXISTS idx_variants_revision ON variants(experiment_revision_id, variant_index);
CREATE INDEX IF NOT EXISTS idx_runs_variant_status ON runs(variant_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_stages_run_status ON workflow_stages(run_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_attempts_stage_status ON job_attempts(stage_id, status, attempt_number);
CREATE INDEX IF NOT EXISTS idx_attempts_slurm_job ON job_attempts(slurm_job_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run_step ON checkpoints(run_id, training_step, created_at);
CREATE INDEX IF NOT EXISTS idx_evaluations_run_status ON evaluations(run_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_evaluation_status ON evaluation_episodes(evaluation_id, status, task);
CREATE INDEX IF NOT EXISTS idx_metrics_run_name ON metrics(run_id, name, step);
CREATE INDEX IF NOT EXISTS idx_artifacts_run_type ON artifacts(run_id, artifact_type, created_at);
CREATE INDEX IF NOT EXISTS idx_manifests_run_type ON manifests(run_id, manifest_type, created_at);
CREATE INDEX IF NOT EXISTS idx_snapshots_captured ON cluster_snapshots(captured_at);
CREATE INDEX IF NOT EXISTS idx_events_entity ON events(entity_type, entity_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_attempt_common_hyperparameter_receipt
ON events(entity_type, entity_id, event_type)
WHERE entity_type = 'job_attempt'
  AND event_type = 'COMMON_HYPERPARAMETERS_ENRICHED_V1';
CREATE INDEX IF NOT EXISTS idx_data_resources_identity ON data_resources(provider, namespace, name);
CREATE INDEX IF NOT EXISTS idx_data_resources_kind ON data_resources(kind, archived_at);
CREATE INDEX IF NOT EXISTS idx_data_versions_resource ON data_resource_versions(resource_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_versions_status ON data_resource_versions(status, format);
CREATE INDEX IF NOT EXISTS idx_data_derivation_inputs_version ON data_derivation_inputs(input_version_id);
CREATE INDEX IF NOT EXISTS idx_data_bundles_state ON data_bundles(archived_at, name, version);
CREATE INDEX IF NOT EXISTS idx_data_bundle_assignments_version ON data_bundle_assignments(version_id);
CREATE INDEX IF NOT EXISTS idx_data_imports_state ON data_imports(state, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_imports_resource ON data_imports(resource_id, created_at DESC);

CREATE TRIGGER IF NOT EXISTS data_resource_versions_no_update
BEFORE UPDATE ON data_resource_versions
BEGIN
    SELECT RAISE(ABORT, 'data resource versions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS data_resource_versions_no_delete
BEFORE DELETE ON data_resource_versions
BEGIN
    SELECT RAISE(ABORT, 'data resource versions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS data_derivations_no_update
BEFORE UPDATE ON data_derivations
BEGIN
    SELECT RAISE(ABORT, 'data derivations are immutable');
END;

CREATE TRIGGER IF NOT EXISTS data_derivations_no_delete
BEFORE DELETE ON data_derivations
BEGIN
    SELECT RAISE(ABORT, 'data derivations are immutable');
END;

CREATE TRIGGER IF NOT EXISTS data_derivation_inputs_no_update
BEFORE UPDATE ON data_derivation_inputs
BEGIN
    SELECT RAISE(ABORT, 'data derivation inputs are immutable');
END;

CREATE TRIGGER IF NOT EXISTS data_derivation_inputs_no_delete
BEFORE DELETE ON data_derivation_inputs
BEGIN
    SELECT RAISE(ABORT, 'data derivation inputs are immutable');
END;

CREATE TRIGGER IF NOT EXISTS data_bundle_assignments_no_update
BEFORE UPDATE ON data_bundle_assignments
BEGIN
    SELECT RAISE(ABORT, 'data bundle assignments are immutable');
END;

CREATE TRIGGER IF NOT EXISTS data_bundle_assignments_no_delete
BEFORE DELETE ON data_bundle_assignments
BEGIN
    SELECT RAISE(ABORT, 'data bundle assignments are immutable');
END;

CREATE TRIGGER IF NOT EXISTS attempt_common_hyperparameter_receipt_no_update
BEFORE UPDATE ON events
WHEN OLD.entity_type = 'job_attempt'
 AND OLD.event_type = 'COMMON_HYPERPARAMETERS_ENRICHED_V1'
BEGIN
    SELECT RAISE(ABORT, 'attempt common hyperparameter receipts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS attempt_common_hyperparameter_receipt_no_delete
BEFORE DELETE ON events
WHEN OLD.entity_type = 'job_attempt'
 AND OLD.event_type = 'COMMON_HYPERPARAMETERS_ENRICHED_V1'
BEGIN
    SELECT RAISE(ABORT, 'attempt common hyperparameter receipts are immutable');
END;
"""


JSON_COLUMNS = frozenset(
    {
        "requested_spec_json",
        "parameters_json",
        "resolved_spec_json",
        "resolved_config_json",
        "task_selection_json",
        "seeds_json",
        "metrics_json",
        "metadata_json",
        "gpu_usage_json",
        "nodes_json",
        "queue_json",
        "details_json",
        "config_json",
        "capabilities_json",
        "schema_json",
        "manifest_json",
        "evidence_json",
        "errors_json",
        "resolved_runtime_json",
        "converter_config_json",
        "execution_snapshot_json",
        "request_json",
        "result_json",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_id() -> str:
    return str(uuid.uuid4())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_sha256(value: Any) -> str:
    payload = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Database:
    """Thread-safe SQLite persistence with one connection per operation."""

    def __init__(self, path: str | os.PathLike[str] | None = None, *, workspace_id: str | None = None) -> None:
        self.workspace_id = workspace_id
        configured = path if path is not None else os.environ.get("SKYNET_DATABASE_PATH")
        self.path = Path(configured or DEFAULT_DATABASE_PATH).expanduser().resolve()
        self._write_lock = threading.RLock()
        if workspace_id is None:
            self.initialize()
        else:
            # Schema upgrades run once through the unscoped coordinator, never
            # during a login while another workspace is using the database.
            with self.connection() as connection:
                if not connection.execute("SELECT 1 FROM workspaces WHERE id=?", (workspace_id,)).fetchone():
                    raise ValueError("Unknown workspace")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.create_function("current_workspace_id", 0, lambda: self.workspace_id)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = NORMAL")
                connection.executescript(SCHEMA)
                self._migrate_runs(connection)
                self._migrate_experiment_revision_lifecycle(connection)
                self._migrate_adapter_registry(connection)
                self._migrate_job_attempt_snapshots(connection)
                self._ensure_execution_immutability_triggers(connection)
                migrate_workspaces(connection)
                migrate_notifications(connection)
            finally:
                connection.close()

    @staticmethod
    def _migrate_runs(connection: sqlite3.Connection) -> None:
        """Add independent run numbering and restart lineage to legacy databases."""

        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }
        unique_indexes: set[tuple[str, ...]] = set()
        for index in connection.execute("PRAGMA index_list(runs)").fetchall():
            if not index["unique"]:
                continue
            index_name = str(index["name"]).replace('"', '""')
            unique_indexes.add(tuple(
                row["name"]
                for row in connection.execute(
                    f'PRAGMA index_info("{index_name}")'
                ).fetchall()
            ))
        current = (
            {"run_number", "restarted_from_run_id"} <= columns
            and ("variant_id", "seed", "run_number") in unique_indexes
            and ("variant_id", "seed") not in unique_indexes
        )
        if current:
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_restarted_from "
                "ON runs(restarted_from_run_id)"
            )
            return

        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("cannot migrate runs while foreign-key violations exist")
        foreign_keys = int(connection.execute("PRAGMA foreign_keys").fetchone()[0])
        connection.execute("PRAGMA foreign_keys = OFF")
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DROP TABLE IF EXISTS _skynet_runs_migration_v2")
            connection.execute("""
                CREATE TABLE _skynet_runs_migration_v2 (
                    id TEXT PRIMARY KEY,
                    variant_id TEXT NOT NULL REFERENCES variants(id) ON DELETE RESTRICT,
                    seed INTEGER NOT NULL,
                    run_number INTEGER NOT NULL DEFAULT 1 CHECK(run_number >= 1),
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
                    updated_at TEXT NOT NULL,
                    UNIQUE(variant_id, seed, run_number),
                    CHECK(restarted_from_run_id IS NULL OR restarted_from_run_id <> id)
                )
            """)
            run_number = "COALESCE(run_number, 1)" if "run_number" in columns else "1"
            restarted_from = (
                "restarted_from_run_id"
                if "restarted_from_run_id" in columns
                else "NULL"
            )
            connection.execute(f"""
                INSERT INTO _skynet_runs_migration_v2 (
                    id, variant_id, seed, run_number, restarted_from_run_id,
                    status, adapter_name, adapter_version, source_commit,
                    runtime_profile, run_directory, mlflow_run_id, created_at,
                    started_at, completed_at, updated_at
                )
                SELECT id, variant_id, seed, {run_number}, {restarted_from},
                       status, adapter_name, adapter_version, source_commit,
                       runtime_profile, run_directory, mlflow_run_id, created_at,
                       started_at, completed_at, updated_at
                FROM runs
            """)
            connection.execute("DROP TABLE runs")
            connection.execute(
                "ALTER TABLE _skynet_runs_migration_v2 RENAME TO runs"
            )
            connection.execute(
                "CREATE INDEX idx_runs_variant_status "
                "ON runs(variant_id, status, created_at)"
            )
            connection.execute(
                "CREATE INDEX idx_runs_restarted_from ON runs(restarted_from_run_id)"
            )
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise RuntimeError("run migration produced foreign-key violations")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.execute(f"PRAGMA foreign_keys = {foreign_keys}")

    @staticmethod
    def _migrate_experiment_revision_lifecycle(
        connection: sqlite3.Connection,
    ) -> None:
        """Add and backfill the one-way revision submission lock."""

        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(experiment_revisions)"
            ).fetchall()
        }
        connection.execute("BEGIN IMMEDIATE")
        try:
            if "submitted_at" not in columns:
                connection.execute(
                    "ALTER TABLE experiment_revisions ADD COLUMN submitted_at TEXT"
                )
            connection.execute("""
                UPDATE experiment_revisions
                SET submitted_at = created_at
                WHERE submitted_at IS NULL
                  AND EXISTS (
                      SELECT 1
                      FROM variants v
                      JOIN runs r ON r.variant_id = v.id
                      WHERE v.experiment_revision_id = experiment_revisions.id
                        AND (
                            r.status <> 'DRAFT'
                            OR EXISTS (
                                SELECT 1
                                FROM workflow_stages s
                                JOIN job_attempts a ON a.stage_id = s.id
                                WHERE s.run_id = r.id
                            )
                        )
                  )
            """)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    @staticmethod
    def _ensure_execution_immutability_triggers(
        connection: sqlite3.Connection,
    ) -> None:
        connection.executescript("""
            CREATE TRIGGER IF NOT EXISTS experiment_revision_spec_immutable
            BEFORE UPDATE OF experiment_id, revision_number, spec_schema_version,
                             requested_spec_json, requested_spec_sha256,
                             created_by, created_at
            ON experiment_revisions
            BEGIN
                SELECT RAISE(ABORT, 'experiment revision specifications are immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS variants_immutable
            BEFORE UPDATE ON variants
            BEGIN
                SELECT RAISE(ABORT, 'experiment variants are immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS run_identity_immutable
            BEFORE UPDATE OF variant_id, seed, run_number, restarted_from_run_id,
                             adapter_name, adapter_version, source_commit,
                             runtime_profile, created_at
            ON runs
            BEGIN
                SELECT RAISE(ABORT, 'run identity is immutable');
            END;

            CREATE TRIGGER IF NOT EXISTS attempt_snapshot_immutable
            BEFORE UPDATE OF execution_snapshot_json, execution_snapshot_sha256
            ON job_attempts
            BEGIN
                SELECT RAISE(ABORT, 'attempt execution snapshots are immutable');
            END;
        """)

    @staticmethod
    def _migrate_job_attempt_snapshots(connection: sqlite3.Connection) -> None:
        """Add immutable per-attempt execution provenance to existing databases."""
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(job_attempts)").fetchall()
        }
        additions = {
            "execution_snapshot_json": "TEXT",
            "execution_snapshot_sha256": "TEXT",
        }
        connection.execute("BEGIN IMMEDIATE")
        try:
            for name, declaration in additions.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE job_attempts ADD COLUMN {name} {declaration}"
                    )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    @staticmethod
    def _migrate_adapter_registry(connection: sqlite3.Connection) -> None:
        """Add registry metadata to legacy adapter rows without replacing them."""
        additions = {
            "adapter_key": "TEXT",
            "version_number": "INTEGER",
            "description": "TEXT NOT NULL DEFAULT ''",
            "manifest_json": "TEXT",
            "manifest_sha256": "TEXT",
            "archived_at": "TEXT",
            "created_by": "TEXT",
            "change_note": "TEXT NOT NULL DEFAULT ''",
            "seed_key": "TEXT",
            "source_adapter_key": "TEXT",
            "source_version_number": "INTEGER",
        }
        connection.execute("BEGIN IMMEDIATE")
        try:
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(adapters)").fetchall()
            }
            for trigger in (
                "adapter_versions_immutable",
                "adapter_registry_name_insert",
                "adapter_registry_name_update",
                "adapter_registry_seed_insert",
                "adapter_registry_seed_update",
            ):
                connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            for name, declaration in additions.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE adapters ADD COLUMN {name} {declaration}")

            rows = connection.execute(
                "SELECT * FROM adapters ORDER BY lower(name), created_at, version, id"
            ).fetchall()
            grouped: dict[str, list[sqlite3.Row]] = {}
            needs_backfill = any(
                row["adapter_key"] is None
                or row["version_number"] is None
                or row["manifest_json"] is None
                or row["manifest_sha256"] is None
                for row in rows
            )
            if needs_backfill:
                for row in rows:
                    grouped.setdefault(str(row["name"]).casefold(), []).append(row)

            for group in grouped.values():
                adapter_key = next(
                    (str(row["adapter_key"]) for row in group if row["adapter_key"]),
                    str(group[0]["id"]),
                )
                used_versions = {
                    int(row["version_number"])
                    for row in group
                    if row["version_number"] is not None
                }
                next_version = 1
                for row in group:
                    version_number = row["version_number"]
                    if version_number is None:
                        while next_version in used_versions:
                            next_version += 1
                        version_number = next_version
                        used_versions.add(next_version)
                        next_version += 1

                    manifest_json = row["manifest_json"]
                    if not manifest_json:
                        try:
                            capabilities = json.loads(row["capabilities_json"] or "{}")
                        except (TypeError, json.JSONDecodeError):
                            capabilities = {}
                        try:
                            parameter_schema = json.loads(row["schema_json"] or "{}")
                        except (TypeError, json.JSONDecodeError):
                            parameter_schema = {}
                        manifest_json = canonical_json({
                            "schema_version": "skynet.adapter/v1",
                            "legacy_adapter_version": row["version"],
                            "repository": {"url": row["repository_url"]},
                            "capabilities": capabilities,
                            "parameter_schema": parameter_schema,
                        })
                    manifest_sha256 = row["manifest_sha256"] or content_sha256(manifest_json)
                    archived_at = row["archived_at"]
                    if not bool(row["enabled"]) and archived_at is None:
                        archived_at = row["updated_at"] or row["created_at"] or utc_now()
                    connection.execute(
                        """
                        UPDATE adapters
                        SET adapter_key = ?, version_number = ?, description = COALESCE(description, ''),
                            manifest_json = ?, manifest_sha256 = ?, archived_at = ?,
                            created_by = COALESCE(created_by, '__migration__')
                        WHERE id = ?
                        """,
                        (
                            adapter_key,
                            version_number,
                            manifest_json,
                            manifest_sha256,
                            archived_at,
                            row["id"],
                        ),
                    )
                latest_legacy_row = group[-1]
                group_archived_at = None
                if not bool(latest_legacy_row["enabled"]):
                    group_archived_at = (
                        latest_legacy_row["archived_at"]
                        or latest_legacy_row["updated_at"]
                        or latest_legacy_row["created_at"]
                        or utc_now()
                    )
                connection.execute(
                    "UPDATE adapters SET enabled = ?, archived_at = ? WHERE adapter_key = ?",
                    (int(group_archived_at is None), group_archived_at, adapter_key),
                )

            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_adapter_key_version "
                "ON adapters(adapter_key, version_number)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_adapters_registry_state "
                "ON adapters(adapter_key, archived_at, version_number DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_adapters_seed_key ON adapters(seed_key)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_adapter_validations_version "
                "ON adapter_validations(adapter_version_id, created_at DESC)"
            )
            connection.execute("""
                CREATE TRIGGER adapter_versions_immutable
                BEFORE UPDATE OF adapter_key, version_number, manifest_json, manifest_sha256,
                                 name, description, repository_url, capabilities_json, schema_json,
                                 created_by, change_note, created_at, source_adapter_key,
                                 source_version_number
                ON adapters
                BEGIN
                    SELECT RAISE(ABORT, 'adapter versions are immutable');
                END
            """)
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS adapter_registry_name_insert
                BEFORE INSERT ON adapters
                WHEN NEW.adapter_key IS NOT NULL AND EXISTS (
                    SELECT 1 FROM adapters
                    WHERE lower(name) = lower(NEW.name) AND adapter_key <> NEW.adapter_key
                )
                BEGIN
                    SELECT RAISE(ABORT, 'adapter name already exists');
                END
            """)
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS adapter_registry_name_update
                BEFORE UPDATE OF name, adapter_key ON adapters
                WHEN NEW.adapter_key IS NOT NULL AND EXISTS (
                    SELECT 1 FROM adapters
                    WHERE lower(name) = lower(NEW.name) AND adapter_key <> NEW.adapter_key
                )
                BEGIN
                    SELECT RAISE(ABORT, 'adapter name already exists');
                END
            """)
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS adapter_registry_seed_insert
                BEFORE INSERT ON adapters
                WHEN NEW.seed_key IS NOT NULL AND EXISTS (
                    SELECT 1 FROM adapters
                    WHERE seed_key = NEW.seed_key AND adapter_key <> NEW.adapter_key
                )
                BEGIN
                    SELECT RAISE(ABORT, 'adapter seed key already exists');
                END
            """)
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS adapter_registry_seed_update
                BEFORE UPDATE OF seed_key, adapter_key ON adapters
                WHEN NEW.seed_key IS NOT NULL AND EXISTS (
                    SELECT 1 FROM adapters
                    WHERE seed_key = NEW.seed_key AND adapter_key <> NEW.adapter_key
                )
                BEGIN
                    SELECT RAISE(ABORT, 'adapter seed key already exists');
                END
            """)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for key in JSON_COLUMNS.intersection(result):
            raw = result[key]
            if raw is not None:
                result[key] = json.loads(raw)
        for key in tuple(result):
            if key.startswith("is_") or key in {"auto_resume", "enabled", "success", "required"}:
                if result[key] is not None:
                    result[key] = bool(result[key])
        return result

    @classmethod
    def _decode_many(cls, rows: Sequence[sqlite3.Row]) -> list[dict[str, Any]]:
        return [decoded for row in rows if (decoded := cls._decode(row)) is not None]

    @staticmethod
    def _insert(connection: sqlite3.Connection, table: str, values: Mapping[str, Any]) -> None:
        if table in PRIVATE_TABLES:
            values = dict(values)
            values["owner_id"] = connection.execute("SELECT current_workspace_id()").fetchone()[0] or LEGACY_WORKSPACE
            if table == "adapters" and values.get("seed_key"):
                values["owner_id"] = None
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        connection.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )

    @classmethod
    def _row_by_id(
        cls, connection: sqlite3.Connection, table: str, entity_id: str
    ) -> dict[str, Any] | None:
        return cls._decode(connection.execute(f"SELECT * FROM {table} WHERE id = ? AND {visible_sql(table)}", (entity_id,)).fetchone())

    @staticmethod
    def _encode_updates(
        fields: Mapping[str, Any], *, allowed: frozenset[str], json_fields: frozenset[str] = frozenset()
    ) -> dict[str, Any]:
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unsupported fields: {', '.join(sorted(unknown))}")
        return {
            key: canonical_json(value) if key in json_fields else int(value) if isinstance(value, bool) else value
            for key, value in fields.items()
        }

    @classmethod
    def _update(
        cls,
        connection: sqlite3.Connection,
        table: str,
        entity_id: str,
        fields: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not fields:
            current = cls._row_by_id(connection, table, entity_id)
            if current is None:
                raise KeyError(f"{table} entity not found: {entity_id}")
            return current
        if cls._row_by_id(connection, table, entity_id) is None:
            raise KeyError(f"{table} entity not found: {entity_id}")
        assignments = ", ".join(f"{key} = ?" for key in fields)
        cursor = connection.execute(
            f"UPDATE {table} SET {assignments} WHERE id = ?",
            (*fields.values(), entity_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"{table} entity not found: {entity_id}")
        updated = cls._row_by_id(connection, table, entity_id)
        assert updated is not None
        return updated

    def owns(self, table: str, identifier: str, *, writable: bool = False) -> bool:
        if table not in PRIVATE_TABLES:
            raise ValueError("Unknown personal record type")
        with self.connection() as connection:
            key = "(id = ? OR adapter_key = ?)" if table == "adapters" else "id = ?"
            params = (identifier, identifier) if table == "adapters" else (identifier,)
            row = connection.execute(f"SELECT owner_id FROM {table} WHERE {key} AND {visible_sql(table)} LIMIT 1", params).fetchone()
        return row is not None and (not writable or self.workspace_id is None or row["owner_id"] == self.workspace_id)

    def _visible_ids(self, table: str, identifiers: list[str]) -> list[str]:
        if self.workspace_id is None or not identifiers:
            return identifiers
        allowed = set()
        with self.connection() as connection:
            for start in range(0, len(identifiers), 400):
                batch = identifiers[start:start + 400]
                placeholders = ",".join("?" for _ in batch)
                allowed.update(row[0] for row in connection.execute(
                    f"SELECT id FROM {table} WHERE id IN ({placeholders}) AND {visible_sql(table)}", batch
                ))
        return [identifier for identifier in identifiers if identifier in allowed]

    def create_project(self, name: str, description: str = "") -> dict[str, Any]:
        values = {"id": new_id(), "name": name, "description": description, "created_at": utc_now(), "archived_at": None}
        with self.transaction() as connection:
            self._insert(connection, "projects", values)
        return dict(values)

    def list_projects(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        where = f"WHERE {visible_sql('projects')}" + ("" if include_archived else " AND archived_at IS NULL")
        with self.connection() as connection:
            return self._decode_many(connection.execute(f"SELECT * FROM projects {where} ORDER BY name").fetchall())

    @staticmethod
    def _attach_tracking_links(
        connection: sqlite3.Connection,
        records: list[dict[str, Any]],
        scope_type: str,
    ) -> None:
        by_id = {str(record["id"]): record for record in records}
        for record in records:
            record["tracking_links"] = []
        if not by_id:
            return
        placeholders = ",".join("?" for _ in by_id)
        rows = connection.execute(
            f"""
            SELECT * FROM tracking_bindings
            WHERE scope_type = ? AND scope_id IN ({placeholders})
            ORDER BY provider
            """,
            (scope_type, *by_id),
        ).fetchall()
        labels = {"mlflow": "MLflow", "wandb": "Weights & Biases"}
        for row in rows:
            item = dict(row)
            record = by_id.get(str(item["scope_id"]))
            if record is None:
                continue
            record["tracking_links"].append({
                "provider": item["provider"],
                "label": labels.get(str(item["provider"]), str(item["provider"])),
                "url": item["remote_url"],
                "remote_id": item["remote_id"],
                "status": item["status"],
                "last_error": item["last_error"],
            })

    def upsert_tracking_connection(
        self,
        provider: str,
        *,
        endpoint: str | None,
        workspace: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if provider not in {"mlflow", "wandb"}:
            raise ValueError(f"unsupported tracking provider: {provider}")
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO tracking_connections(owner_id, provider, endpoint, workspace, config_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, provider) DO UPDATE SET
                    endpoint = excluded.endpoint,
                    workspace = excluded.workspace,
                    config_json = excluded.config_json,
                    updated_at = excluded.updated_at
                """,
                (self.workspace_id or LEGACY_WORKSPACE, provider, endpoint, workspace, canonical_json(config or {}), now),
            )
            result = self._decode(connection.execute(
                "SELECT * FROM tracking_connections WHERE owner_id = ? AND provider = ?", (self.workspace_id or LEGACY_WORKSPACE, provider)
            ).fetchone())
        assert result is not None
        return result

    def get_tracking_connection(self, provider: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            return self._decode(connection.execute(
                "SELECT * FROM tracking_connections WHERE owner_id = ? AND provider = ?", (self.workspace_id or LEGACY_WORKSPACE, provider)
            ).fetchone())

    def delete_tracking_connection(self, provider: str) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM tracking_connections WHERE owner_id = ? AND provider = ?", (self.workspace_id or LEGACY_WORKSPACE, provider))

    def upsert_tracking_binding(
        self,
        provider: str,
        scope_type: str,
        scope_id: str,
        *,
        remote_id: str | None,
        remote_url: str | None,
        status: str,
        metadata: Mapping[str, Any] | None = None,
        last_error: str | None = None,
    ) -> dict[str, Any]:
        if provider not in {"mlflow", "wandb"}:
            raise ValueError(f"unsupported tracking provider: {provider}")
        if scope_type not in {"experiment", "run"}:
            raise ValueError(f"unsupported tracking scope: {scope_type}")
        table = "experiments" if scope_type == "experiment" else "runs"
        if self.workspace_id is not None and not self.owns(table, scope_id):
            raise KeyError("Record not found in this workspace")
        now = utc_now()
        binding_id = new_id()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO tracking_bindings(
                    id, provider, scope_type, scope_id, remote_id, remote_url,
                    status, metadata_json, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, scope_type, scope_id) DO UPDATE SET
                    remote_id = COALESCE(excluded.remote_id, tracking_bindings.remote_id),
                    remote_url = COALESCE(excluded.remote_url, tracking_bindings.remote_url),
                    status = excluded.status,
                    metadata_json = excluded.metadata_json,
                    last_error = excluded.last_error,
                    updated_at = excluded.updated_at
                """,
                (
                    binding_id, provider, scope_type, scope_id, remote_id, remote_url,
                    status, canonical_json(metadata or {}), last_error, now, now,
                ),
            )
            result = self._decode(connection.execute(
                """SELECT * FROM tracking_bindings
                   WHERE provider = ? AND scope_type = ? AND scope_id = ?""",
                (provider, scope_type, scope_id),
            ).fetchone())
        assert result is not None
        return result

    def list_tracking_bindings(
        self, scope_type: str, scope_id: str
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return self._decode_many(connection.execute(
                f"""SELECT * FROM tracking_bindings
                   WHERE scope_type = ? AND scope_id = ? AND {visible_sql('tracking_bindings')} ORDER BY provider""",
                (scope_type, scope_id),
            ).fetchall())

    def list_tracking_bindings_for_provider(
        self,
        provider: str,
        *,
        scope_type: str | None = None,
        statuses: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        clauses = ["provider = ?", visible_sql("tracking_bindings")]
        parameters: list[Any] = [provider]
        if scope_type:
            clauses.append("scope_type = ?")
            parameters.append(scope_type)
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            parameters.extend(statuses)
        with self.connection() as connection:
            return self._decode_many(connection.execute(
                f"SELECT * FROM tracking_bindings WHERE {' AND '.join(clauses)} "
                "ORDER BY updated_at",
                parameters,
            ).fetchall())

    def create_experiment(
        self,
        *,
        name: str,
        requested_spec: Mapping[str, Any],
        project_id: str | None = None,
        description: str = "",
        status: str = "DRAFT",
        spec_schema_version: str = "skynet.rl2/v1",
        created_by: str | None = None,
        mlflow_experiment_id: str | None = None,
    ) -> dict[str, Any]:
        if self.workspace_id is not None and project_id is not None and not self.owns("projects", project_id):
            raise KeyError("Record not found in this workspace")
        now = utc_now()
        experiment_id = new_id()
        revision_id = new_id()
        spec_json = canonical_json(requested_spec)
        with self.transaction() as connection:
            self._insert(connection, "experiments", {
                "id": experiment_id, "project_id": project_id, "name": name,
                "description": description, "status": status,
                "mlflow_experiment_id": mlflow_experiment_id, "created_at": now, "updated_at": now,
            })
            self._insert(connection, "experiment_revisions", {
                "id": revision_id, "experiment_id": experiment_id, "revision_number": 1,
                "spec_schema_version": spec_schema_version, "requested_spec_json": spec_json,
                "requested_spec_sha256": content_sha256(spec_json), "created_by": created_by,
                "created_at": now,
            })
        result = self.get_experiment(experiment_id)
        assert result is not None
        return result

    def create_experiment_revision(
        self,
        experiment_id: str,
        requested_spec: Mapping[str, Any],
        *,
        spec_schema_version: str = "skynet.rl2/v1",
        created_by: str | None = None,
    ) -> dict[str, Any]:
        if self.workspace_id is not None and experiment_id is not None and not self.owns("experiments", experiment_id):
            raise KeyError("Record not found in this workspace")
        spec_json = canonical_json(requested_spec)
        with self.transaction() as connection:
            exists = connection.execute("SELECT 1 FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
            if not exists:
                raise KeyError(f"Experiment not found: {experiment_id}")
            revision_number = connection.execute(
                "SELECT COALESCE(MAX(revision_number), 0) + 1 FROM experiment_revisions WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()[0]
            values = {
                "id": new_id(), "experiment_id": experiment_id, "revision_number": revision_number,
                "spec_schema_version": spec_schema_version, "requested_spec_json": spec_json,
                "requested_spec_sha256": content_sha256(spec_json), "created_by": created_by,
                "created_at": utc_now(),
            }
            self._insert(connection, "experiment_revisions", values)
            connection.execute("UPDATE experiments SET updated_at = ? WHERE id = ?", (utc_now(), experiment_id))
            result = self._row_by_id(connection, "experiment_revisions", values["id"])
        assert result is not None
        return result

    def discard_unsubmitted_experiment_revision(
        self,
        experiment_id: str,
        revision_id: str,
        *,
        delete_experiment_if_empty: bool = False,
    ) -> bool:
        """Rollback a draft graph only while it has no execution history."""
        if self.workspace_id is not None and experiment_id is not None and not self.owns("experiments", experiment_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and revision_id is not None and not self.owns("experiment_revisions", revision_id):
            raise KeyError("Record not found in this workspace")

        with self.transaction() as connection:
            revision = connection.execute(
                """
                SELECT submitted_at FROM experiment_revisions
                WHERE id = ? AND experiment_id = ?
                """,
                (revision_id, experiment_id),
            ).fetchone()
            if revision is None or revision["submitted_at"] is not None:
                return False
            unsafe = connection.execute(
                """
                SELECT 1
                FROM variants v
                LEFT JOIN runs r ON r.variant_id = v.id
                LEFT JOIN workflow_stages s ON s.run_id = r.id
                LEFT JOIN job_attempts a ON a.stage_id = s.id
                WHERE v.experiment_revision_id = ?
                  AND (
                    COALESCE(r.status, 'DRAFT') <> 'DRAFT'
                    OR COALESCE(s.status, 'DRAFT') <> 'DRAFT'
                    OR a.id IS NOT NULL
                  )
                LIMIT 1
                """,
                (revision_id,),
            ).fetchone()
            if unsafe is not None:
                return False
            connection.execute(
                """
                DELETE FROM runs
                WHERE variant_id IN (
                    SELECT id FROM variants WHERE experiment_revision_id = ?
                )
                """,
                (revision_id,),
            )
            connection.execute(
                "DELETE FROM variants WHERE experiment_revision_id = ?",
                (revision_id,),
            )
            connection.execute(
                "DELETE FROM experiment_revisions WHERE id = ? AND experiment_id = ?",
                (revision_id, experiment_id),
            )
            if delete_experiment_if_empty:
                connection.execute(
                    """
                    DELETE FROM experiments
                    WHERE id = ?
                      AND NOT EXISTS (
                        SELECT 1 FROM experiment_revisions
                        WHERE experiment_id = experiments.id
                      )
                    """,
                    (experiment_id,),
                )
            return True

    def claim_experiment_revision_submission(
        self, experiment_id: str, revision_id: str
    ) -> dict[str, Any] | None:
        """Atomically lock a draft revision before its first submission."""
        if self.workspace_id is not None and experiment_id is not None and not self.owns("experiments", experiment_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and revision_id is not None and not self.owns("experiment_revisions", revision_id):
            raise KeyError("Record not found in this workspace")

        now = utc_now()
        with self.transaction() as connection:
            claimed = connection.execute(
                """
                UPDATE experiment_revisions
                SET submitted_at = ?
                WHERE id = ? AND experiment_id = ? AND submitted_at IS NULL
                """,
                (now, revision_id, experiment_id),
            )
            if claimed.rowcount != 1:
                return None
            return self._row_by_id(connection, "experiment_revisions", revision_id)

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        if self.workspace_id is not None and experiment_id is not None and not self.owns("experiments", experiment_id):
            return None
        with self.connection() as connection:
            experiment = self._decode(connection.execute(
                f"SELECT e.*, p.name AS project_name FROM experiments e LEFT JOIN projects p ON p.id = e.project_id WHERE e.id = ? AND {visible_sql('experiments', 'e')}",
                (experiment_id,),
            ).fetchone())
            if experiment is None:
                return None
            experiment["revisions"] = self._decode_many(connection.execute(
                "SELECT * FROM experiment_revisions WHERE experiment_id = ? ORDER BY revision_number DESC",
                (experiment_id,),
            ).fetchall())
            experiment["latest_revision"] = experiment["revisions"][0] if experiment["revisions"] else None
            self._attach_tracking_links(connection, [experiment], "experiment")
            return experiment

    def list_experiments(
        self, *, project_id: str | None = None, status: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        if self.workspace_id is not None and project_id is not None and not self.owns("projects", project_id):
            return []
        clauses: list[str] = [visible_sql("experiments", "e")]
        parameters: list[Any] = []
        if project_id is not None:
            clauses.append("e.project_id = ?")
            parameters.append(project_id)
        if status is not None:
            clauses.append("e.status = ?")
            parameters.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT e.*, p.name AS project_name, r.id AS latest_revision_id,
                   r.revision_number AS latest_revision_number,
                   r.requested_spec_sha256 AS latest_spec_sha256,
                   r.submitted_at AS latest_revision_submitted_at,
                   json_extract(r.requested_spec_json, '$.source.revision') AS git_revision,
                   json_extract(r.requested_spec_json, '$.source.repository') AS repository,
                   json_extract(r.requested_spec_json, '$.source.project_subdirectory') AS project_subdirectory
            FROM experiments e
            LEFT JOIN projects p ON p.id = e.project_id
            LEFT JOIN experiment_revisions r ON r.id = (
                SELECT id FROM experiment_revisions WHERE experiment_id = e.id
                ORDER BY revision_number DESC LIMIT 1
            )
            {where} ORDER BY e.created_at DESC LIMIT ? OFFSET ?
        """
        parameters.extend((limit, offset))
        with self.connection() as connection:
            results = self._decode_many(connection.execute(query, parameters).fetchall())
            self._attach_tracking_links(connection, results, "experiment")
            return results

    def update_experiment(self, experiment_id: str, **fields: Any) -> dict[str, Any]:
        if self.workspace_id is not None and experiment_id is not None and not self.owns("experiments", experiment_id):
            raise KeyError("Record not found in this workspace")
        encoded = self._encode_updates(fields, allowed=frozenset({"name", "description", "status", "mlflow_experiment_id", "project_id"}))
        encoded["updated_at"] = utc_now()
        with self.transaction() as connection:
            return self._update(connection, "experiments", experiment_id, encoded)

    def create_variant(
        self,
        experiment_revision_id: str,
        *,
        name: str,
        parameters: Mapping[str, Any],
        resolved_spec: Mapping[str, Any],
        variant_index: int | None = None,
    ) -> dict[str, Any]:
        if self.workspace_id is not None and experiment_revision_id is not None and not self.owns("experiment_revisions", experiment_revision_id):
            raise KeyError("Record not found in this workspace")
        resolved_json = canonical_json(resolved_spec)
        with self.transaction() as connection:
            if variant_index is None:
                variant_index = connection.execute(
                    "SELECT COALESCE(MAX(variant_index), -1) + 1 FROM variants WHERE experiment_revision_id = ?",
                    (experiment_revision_id,),
                ).fetchone()[0]
            values = {
                "id": new_id(), "experiment_revision_id": experiment_revision_id,
                "variant_index": variant_index, "name": name,
                "parameters_json": canonical_json(parameters), "resolved_spec_json": resolved_json,
                "resolved_spec_sha256": content_sha256(resolved_json), "created_at": utc_now(),
            }
            self._insert(connection, "variants", values)
            result = self._row_by_id(connection, "variants", values["id"])
        assert result is not None
        return result

    def list_variants(self, experiment_revision_id: str) -> list[dict[str, Any]]:
        if self.workspace_id is not None and experiment_revision_id is not None and not self.owns("experiment_revisions", experiment_revision_id):
            return []
        with self.connection() as connection:
            return self._decode_many(connection.execute(
                "SELECT * FROM variants WHERE experiment_revision_id = ? ORDER BY variant_index", (experiment_revision_id,)
            ).fetchall())

    def create_run(
        self,
        variant_id: str,
        *,
        seed: int,
        adapter_name: str,
        adapter_version: str,
        run_directory: str,
        status: str = "PENDING",
        source_commit: str | None = None,
        runtime_profile: str | None = None,
        mlflow_run_id: str | None = None,
        run_number: int | None = None,
        restarted_from_run_id: str | None = None,
    ) -> dict[str, Any]:
        if self.workspace_id is not None and variant_id is not None and not self.owns("variants", variant_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and restarted_from_run_id is not None and not self.owns("runs", restarted_from_run_id):
            raise KeyError("Record not found in this workspace")
        now = utc_now()
        with self.transaction() as connection:
            if restarted_from_run_id is not None:
                source = connection.execute(
                    "SELECT variant_id, seed FROM runs WHERE id = ?",
                    (restarted_from_run_id,),
                ).fetchone()
                if source is None:
                    raise KeyError(f"Run not found: {restarted_from_run_id}")
                if source["variant_id"] != variant_id or int(source["seed"]) != seed:
                    raise ValueError(
                        "a restarted run must preserve its source variant and seed"
                    )
            if run_number is None:
                run_number = int(connection.execute(
                    """
                    SELECT COALESCE(MAX(run_number), 0) + 1
                    FROM runs WHERE variant_id = ? AND seed = ?
                    """,
                    (variant_id, seed),
                ).fetchone()[0])
            if run_number < 1:
                raise ValueError("run_number must be at least 1")
            values = {
                "id": new_id(), "variant_id": variant_id, "seed": seed,
                "run_number": run_number,
                "restarted_from_run_id": restarted_from_run_id,
                "status": status, "adapter_name": adapter_name,
                "adapter_version": adapter_version,
                "source_commit": source_commit, "runtime_profile": runtime_profile,
                "run_directory": run_directory, "mlflow_run_id": mlflow_run_id,
                "created_at": now, "started_at": None, "completed_at": None,
                "updated_at": now,
            }
            self._insert(connection, "runs", values)
            result = self._row_by_id(connection, "runs", values["id"])
        assert result is not None
        return result

    def materialize_empty_revision_runs(
        self,
        revision_id: str,
        planned_runs: Sequence[Mapping[str, Any]],
        *,
        run_directory_root: str,
    ) -> bool:
        """Restore a saved definition only on explicit Submit after history cleanup.

        The entire draft graph is committed once across independent service
        instances. Existing execution records always win over rematerialization.
        """
        if self.workspace_id is not None and revision_id is not None and not self.owns("experiment_revisions", revision_id):
            raise KeyError("Record not found in this workspace")
        with self.transaction() as connection:
            revision = connection.execute(
                "SELECT id FROM experiment_revisions WHERE id = ?", (revision_id,)
            ).fetchone()
            if revision is None:
                raise KeyError("Experiment revision not found")
            existing = connection.execute(
                """
                SELECT 1 FROM runs r
                JOIN variants v ON v.id = r.variant_id
                WHERE v.experiment_revision_id = ? LIMIT 1
                """,
                (revision_id,),
            ).fetchone()
            if existing is not None:
                return False
            variants = {
                row["id"]: row
                for row in connection.execute(
                    """
                    SELECT id, resolved_spec_sha256 FROM variants
                    WHERE experiment_revision_id = ?
                    """,
                    (revision_id,),
                )
            }
            if (
                not variants
                or len(planned_runs) != len(variants)
                or {item["variant_id"] for item in planned_runs} != set(variants)
            ):
                raise ValueError(
                    "The saved experiment variants are incomplete; "
                    "load its configuration and save a new revision"
                )
            for item in planned_runs:
                saved = variants[item["variant_id"]]
                if item["resolved_spec_sha256"] != saved["resolved_spec_sha256"]:
                    raise ValueError(
                        "The saved variant changed before its new run was created"
                    )
            for item in planned_runs:
                identifier, timestamp = new_id(), utc_now()
                self._insert(
                    connection,
                    "runs",
                    {
                        "id": identifier,
                        "variant_id": item["variant_id"],
                        "seed": item["seed"],
                        "run_number": 1,
                        "status": "DRAFT",
                        "adapter_name": item["adapter_name"],
                        "adapter_version": item["adapter_version"],
                        "source_commit": item["source_commit"],
                        "runtime_profile": item["runtime_profile"],
                        "run_directory": run_directory_root.rstrip("/") + "/" + identifier,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                )
                self._insert(
                    connection,
                    "workflow_stages",
                    {
                        "id": new_id(),
                        "run_id": identifier,
                        "stage_type": "TRAIN",
                        "name": "train",
                        "status": "DRAFT",
                        "resolved_config_json": canonical_json(item["resolved_config"]),
                        "auto_resume": int(item["auto_resume"]),
                        "max_attempts": item["max_attempts"],
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                )
            return True

    def list_runs(
        self,
        *,
        experiment_id: str | None = None,
        experiment_revision_id: str | None = None,
        variant_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if self.workspace_id is not None and experiment_id is not None and not self.owns("experiments", experiment_id):
            return []
        if self.workspace_id is not None and experiment_revision_id is not None and not self.owns("experiment_revisions", experiment_revision_id):
            return []
        if self.workspace_id is not None and variant_id is not None and not self.owns("variants", variant_id):
            return []
        clauses: list[str] = [visible_sql("runs", "r")]
        parameters: list[Any] = []
        if experiment_id is not None:
            clauses.append("er.experiment_id = ?")
            parameters.append(experiment_id)
        if experiment_revision_id is not None:
            clauses.append("er.id = ?")
            parameters.append(experiment_revision_id)
        if variant_id is not None:
            clauses.append("r.variant_id = ?")
            parameters.append(variant_id)
        if status is not None:
            clauses.append("r.status = ?")
            parameters.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT r.*, v.name AS variant_name, er.experiment_id,
                   er.id AS experiment_revision_id,
                   er.revision_number AS experiment_revision_number,
                   e.name AS experiment_name
            FROM runs r
            JOIN variants v ON v.id = r.variant_id
            JOIN experiment_revisions er ON er.id = v.experiment_revision_id
            JOIN experiments e ON e.id = er.experiment_id
            {where} ORDER BY r.created_at DESC LIMIT ? OFFSET ?
        """
        parameters.extend((limit, offset))
        with self.connection() as connection:
            results = self._decode_many(connection.execute(query, parameters).fetchall())
            self._attach_tracking_links(connection, results, "run")
            return results

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            return None
        with self.connection() as connection:
            run = self._decode(connection.execute(f"""
                SELECT r.*, v.name AS variant_name, v.parameters_json, v.resolved_spec_json,
                       er.experiment_id, er.id AS experiment_revision_id,
                       er.revision_number AS experiment_revision_number,
                       e.name AS experiment_name
                FROM runs r
                JOIN variants v ON v.id = r.variant_id
                JOIN experiment_revisions er ON er.id = v.experiment_revision_id
                JOIN experiments e ON e.id = er.experiment_id
                WHERE r.id = ? AND {visible_sql('runs', 'r')}
            """, (run_id,)).fetchone())
            if run is None:
                return None
            run["stages"] = self._decode_many(connection.execute(
                "SELECT * FROM workflow_stages WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall())
            run["attempts"] = self._decode_many(connection.execute("""
                SELECT a.* FROM job_attempts a JOIN workflow_stages s ON s.id = a.stage_id
                WHERE s.run_id = ? ORDER BY a.created_at
            """, (run_id,)).fetchall())
            run["checkpoints"] = self._decode_many(connection.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY training_step, created_at", (run_id,)
            ).fetchall())
            run["evaluations"] = self._decode_many(connection.execute(
                "SELECT * FROM evaluations WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall())
            run["artifacts"] = self._decode_many(connection.execute(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall())
            run["manifests"] = self._decode_many(connection.execute(
                "SELECT * FROM manifests WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall())
            run["events"] = self._decode_many(connection.execute(
                "SELECT * FROM events WHERE entity_id = ? ORDER BY created_at DESC LIMIT 100", (run_id,)
            ).fetchall())
            self._attach_tracking_links(connection, [run], "run")
            return run

    def run_progress_evidence(
        self, run_ids: Sequence[str]
    ) -> dict[str, dict[str, Any]]:
        """Return persisted, provider-neutral training progress evidence in batches."""
        identifiers = list(dict.fromkeys(str(run_id) for run_id in run_ids if run_id))
        identifiers = self._visible_ids("runs", identifiers)
        evidence = {
            run_id: {
                "resolved_spec_json": None,
                "attempts": [],
                "checkpoints": [],
                "metrics": [],
                "progress_samples": [],
            }
            for run_id in identifiers
        }
        if not identifiers:
            return evidence
        with self.connection() as connection:
            for start in range(0, len(identifiers), 400):
                batch = identifiers[start:start + 400]
                placeholders = ",".join("?" for _ in batch)
                variants = self._decode_many(connection.execute(f"""
                    SELECT r.id AS run_id, v.resolved_spec_json
                    FROM runs r JOIN variants v ON v.id = r.variant_id
                    WHERE r.id IN ({placeholders})
                """, batch).fetchall())
                for row in variants:
                    evidence[row["run_id"]]["resolved_spec_json"] = row.get("resolved_spec_json")

                attempts = self._decode_many(connection.execute(f"""
                    SELECT s.run_id, a.*
                    FROM job_attempts a
                    JOIN workflow_stages s ON s.id = a.stage_id
                    WHERE s.stage_type = 'TRAIN' AND s.run_id IN ({placeholders})
                    ORDER BY s.run_id, a.attempt_number, a.created_at
                """, batch).fetchall())
                for row in attempts:
                    evidence[row["run_id"]]["attempts"].append(row)

                checkpoints = self._decode_many(connection.execute(f"""
                    SELECT * FROM checkpoints
                    WHERE run_id IN ({placeholders}) AND training_step IS NOT NULL
                    ORDER BY run_id, created_at
                """, batch).fetchall())
                for row in checkpoints:
                    evidence[row["run_id"]]["checkpoints"].append(row)

                metrics = self._decode_many(connection.execute(f"""
                    SELECT * FROM metrics
                    WHERE run_id IN ({placeholders}) AND evaluation_id IS NULL
                    ORDER BY run_id, recorded_at, step
                """, batch).fetchall())
                for row in metrics:
                    evidence[row["run_id"]]["metrics"].append(row)
                progress_samples = self._decode_many(connection.execute(f"""
                    SELECT * FROM training_progress_samples
                    WHERE run_id IN ({placeholders})
                    ORDER BY run_id, recorded_at, completed
                """, batch).fetchall())
                for row in progress_samples:
                    evidence[row["run_id"]]["progress_samples"].append(row)
        return evidence

    def update_run(self, run_id: str, **fields: Any) -> dict[str, Any]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            raise KeyError("Record not found in this workspace")
        encoded = self._encode_updates(fields, allowed=frozenset({
            "status", "run_directory", "mlflow_run_id",
            "started_at", "completed_at",
        }))
        encoded["updated_at"] = utc_now()
        with self.transaction() as connection:
            return self._update(connection, "runs", run_id, encoded)

    def create_stage(
        self,
        run_id: str,
        *,
        stage_type: str,
        name: str,
        resolved_config: Mapping[str, Any] | None = None,
        auto_resume: bool = False,
        max_attempts: int = 1,
        status: str = "PENDING",
    ) -> dict[str, Any]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            raise KeyError("Record not found in this workspace")
        now = utc_now()
        values = {
            "id": new_id(), "run_id": run_id, "stage_type": stage_type, "name": name,
            "status": status, "resolved_config_json": canonical_json(resolved_config or {}),
            "auto_resume": int(auto_resume), "max_attempts": max_attempts, "created_at": now,
            "started_at": None, "completed_at": None, "updated_at": now,
        }
        with self.transaction() as connection:
            self._insert(connection, "workflow_stages", values)
            result = self._row_by_id(connection, "workflow_stages", values["id"])
        assert result is not None
        return result

    create_workflow_stage = create_stage

    def list_stages(self, run_id: str) -> list[dict[str, Any]]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            return []
        with self.connection() as connection:
            return self._decode_many(connection.execute(
                "SELECT * FROM workflow_stages WHERE run_id = ? ORDER BY created_at", (run_id,)
            ).fetchall())

    def update_stage(self, stage_id: str, **fields: Any) -> dict[str, Any]:
        if self.workspace_id is not None and stage_id is not None and not self.owns("workflow_stages", stage_id):
            raise KeyError("Record not found in this workspace")
        encoded = self._encode_updates(
            fields,
            allowed=frozenset({"status", "resolved_config_json", "auto_resume", "max_attempts", "started_at", "completed_at"}),
            json_fields=frozenset({"resolved_config_json"}),
        )
        encoded["updated_at"] = utc_now()
        with self.transaction() as connection:
            return self._update(connection, "workflow_stages", stage_id, encoded)

    def claim_stage_for_submission(self, stage_id: str) -> dict[str, Any] | None:
        """Atomically claim one pending stage so concurrent dispatchers cannot submit it twice."""
        if self.workspace_id is not None and stage_id is not None and not self.owns("workflow_stages", stage_id):
            raise KeyError("Record not found in this workspace")

        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE workflow_stages
                SET status = 'SUBMITTING', updated_at = ?
                WHERE id = ? AND status IN ('PENDING', 'RETRY_PENDING')
                """,
                (now, stage_id),
            )
            if cursor.rowcount != 1:
                return None
            return self._row_by_id(connection, "workflow_stages", stage_id)

    def add_stage_dependency(
        self, stage_id: str, depends_on_stage_id: str, dependency_type: str = "AFTER_OK"
    ) -> dict[str, Any]:
        if self.workspace_id is not None and stage_id is not None and not self.owns("workflow_stages", stage_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and depends_on_stage_id is not None and not self.owns("workflow_stages", depends_on_stage_id):
            raise KeyError("Record not found in this workspace")
        values = {
            "stage_id": stage_id, "depends_on_stage_id": depends_on_stage_id,
            "dependency_type": dependency_type, "created_at": utc_now(),
        }
        with self.transaction() as connection:
            self._insert(connection, "stage_dependencies", values)
        return values

    def create_job_attempt(
        self,
        stage_id: str,
        *,
        attempt_number: int | None = None,
        status: str = "CREATED",
        **fields: Any,
    ) -> dict[str, Any]:
        if self.workspace_id is not None and stage_id is not None and not self.owns("workflow_stages", stage_id):
            raise KeyError("Record not found in this workspace")
        allowed = frozenset({
            "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id", "gateway", "account",
            "partition_name", "node_list", "gpu_type", "gpu_count", "cpu_count", "memory_mb",
            "time_limit_seconds", "slurm_state", "slurm_reason", "exit_code", "restart_count",
            "sbatch_path", "stdout_path", "stderr_path", "resume_checkpoint_id", "cluster_snapshot_id",
            "execution_snapshot_json", "execution_snapshot_sha256",
            "submitted_at", "started_at", "finished_at",
        })
        values_extra = self._encode_updates(
            fields,
            allowed=allowed,
            json_fields=frozenset({"execution_snapshot_json"}),
        )
        now = utc_now()
        with self.transaction() as connection:
            if attempt_number is None:
                attempt_number = connection.execute(
                    "SELECT COALESCE(MAX(attempt_number), 0) + 1 FROM job_attempts WHERE stage_id = ?", (stage_id,)
                ).fetchone()[0]
            values: dict[str, Any] = {
                "id": new_id(), "stage_id": stage_id, "attempt_number": attempt_number,
                "status": status, "created_at": now, "updated_at": now,
            }
            values.update(values_extra)
            self._insert(connection, "job_attempts", values)
            result = self._row_by_id(connection, "job_attempts", values["id"])
        assert result is not None
        return result

    def list_job_attempts(
        self, *, stage_id: str | None = None, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        if self.workspace_id is not None and stage_id is not None and not self.owns("workflow_stages", stage_id):
            return []
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            return []
        if stage_id is None and run_id is None:
            raise ValueError("stage_id or run_id is required")
        if stage_id is not None:
            query = "SELECT * FROM job_attempts WHERE stage_id = ? ORDER BY attempt_number"
            parameters = (stage_id,)
        else:
            query = """
                SELECT a.* FROM job_attempts a JOIN workflow_stages s ON s.id = a.stage_id
                WHERE s.run_id = ? ORDER BY a.created_at
            """
            parameters = (run_id,)
        with self.connection() as connection:
            return self._decode_many(connection.execute(query, parameters).fetchall())

    def update_job_attempt(self, attempt_id: str, **fields: Any) -> dict[str, Any]:
        if self.workspace_id is not None and attempt_id is not None and not self.owns("job_attempts", attempt_id):
            raise KeyError("Record not found in this workspace")
        encoded = self._encode_updates(fields, allowed=frozenset({
            "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id", "gateway", "account",
            "partition_name", "node_list", "gpu_type", "gpu_count", "cpu_count", "memory_mb",
            "time_limit_seconds", "status", "slurm_state", "slurm_reason", "exit_code", "restart_count",
            "sbatch_path", "stdout_path", "stderr_path", "resume_checkpoint_id", "cluster_snapshot_id",
            "submitted_at", "started_at", "finished_at",
        }))
        encoded["updated_at"] = utc_now()
        with self.transaction() as connection:
            return self._update(connection, "job_attempts", attempt_id, encoded)

    def claim_stage_and_create_job_attempt(
        self,
        stage_id: str,
        *,
        status: str = "SUBMITTING",
        **fields: Any,
    ) -> dict[str, Any] | None:
        """Atomically claim a pending stage and create its next attempt."""
        if self.workspace_id is not None and stage_id is not None and not self.owns("workflow_stages", stage_id):
            raise KeyError("Record not found in this workspace")

        values_extra = self._encode_updates(
            fields,
            allowed=frozenset({
                "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id", "gateway", "account",
                "partition_name", "node_list", "gpu_type", "gpu_count", "cpu_count", "memory_mb",
                "time_limit_seconds", "slurm_state", "slurm_reason", "exit_code", "restart_count",
                "sbatch_path", "stdout_path", "stderr_path", "resume_checkpoint_id", "cluster_snapshot_id",
                "execution_snapshot_json", "execution_snapshot_sha256",
                "submitted_at", "started_at", "finished_at",
            }),
            json_fields=frozenset({"execution_snapshot_json"}),
        )
        now = utc_now()
        with self.transaction() as connection:
            claimed = connection.execute(
                """
                UPDATE workflow_stages
                SET status = 'SUBMITTING', updated_at = ?
                WHERE id = ? AND status IN ('DRAFT', 'PENDING', 'RETRY_PENDING')
                """,
                (now, stage_id),
            )
            if claimed.rowcount != 1:
                return None
            attempt_number = connection.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) + 1 FROM job_attempts WHERE stage_id = ?",
                (stage_id,),
            ).fetchone()[0]
            values: dict[str, Any] = {
                "id": new_id(),
                "stage_id": stage_id,
                "attempt_number": attempt_number,
                "status": status,
                "created_at": now,
                "updated_at": now,
            }
            values.update(values_extra)
            self._insert(connection, "job_attempts", values)
            connection.execute(
                """
                UPDATE runs
                SET status = 'SUBMITTING', completed_at = NULL, updated_at = ?
                WHERE id = (
                    SELECT run_id
                    FROM workflow_stages
                    WHERE id = ? AND stage_type != 'EVALUATE'
                )
                """,
                (now, stage_id),
            )
            connection.execute(
                """
                UPDATE experiment_revisions
                SET submitted_at = COALESCE(submitted_at, ?)
                WHERE id = (
                    SELECT v.experiment_revision_id
                    FROM workflow_stages s
                    JOIN runs r ON r.id = s.run_id
                    JOIN variants v ON v.id = r.variant_id
                    WHERE s.id = ? AND s.stage_type != 'EVALUATE'
                )
                """,
                (now, stage_id),
            )
            connection.execute(
                """
                UPDATE experiments
                SET status = 'ACTIVE', updated_at = ?
                WHERE id = (
                    SELECT er.experiment_id
                    FROM workflow_stages s
                    JOIN runs r ON r.id = s.run_id
                    JOIN variants v ON v.id = r.variant_id
                    JOIN experiment_revisions er
                      ON er.id = v.experiment_revision_id
                    WHERE s.id = ? AND s.stage_type != 'EVALUATE'
                )
                """,
                (now, stage_id),
            )
            return self._row_by_id(connection, "job_attempts", values["id"])

    def claim_workflow_cancellation(
        self,
        stage_id: str,
        *,
        entity_type: str,
        entity_id: str,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically persist one idempotent cancellation intent for a workflow stage."""
        if self.workspace_id is not None and stage_id is not None and not self.owns("workflow_stages", stage_id):
            raise KeyError("Record not found in this workspace")

        cancellable_states = {
            "CREATED", "PENDING", "RETRY_PENDING", "SUBMITTING", "SUBMITTED",
            "PENDING_SLURM", "RUNNING", "REQUEUED",
        }
        inflight_stage_states = {
            "SUBMITTING", "SUBMITTED", "PENDING_SLURM", "RUNNING", "REQUEUED",
        }
        inflight_attempt_states = {
            "SUBMITTING", "SUBMITTED", "PENDING", "RUNNING", "REQUEUED",
            "CANCELLING",
        }
        if entity_type not in {"run", "evaluation"}:
            raise ValueError("cancellation entity_type must be run or evaluation")

        now = utc_now()
        with self.transaction() as connection:
            stage = self._row_by_id(connection, "workflow_stages", stage_id)
            if stage is None:
                raise KeyError("Workflow stage not found")
            if entity_type == "run":
                if stage["run_id"] != entity_id:
                    raise ValueError("Workflow stage does not belong to this run")
                entity = self._row_by_id(connection, "runs", entity_id)
                if entity is None:
                    raise KeyError("Run not found")
                entity_table = "runs"
            else:
                entity = self._row_by_id(connection, "evaluations", entity_id)
                if entity is None:
                    raise KeyError("Evaluation not found")
                if entity["stage_id"] != stage_id or entity["run_id"] != stage["run_id"]:
                    raise ValueError("Workflow stage does not belong to this evaluation")
                entity_table = "evaluations"

            attempt_rows = connection.execute(
                """
                SELECT * FROM job_attempts
                WHERE stage_id = ?
                ORDER BY attempt_number DESC, created_at DESC
                """,
                (stage_id,),
            ).fetchall()
            attempts = [dict(row) for row in attempt_rows]
            active_attempts = [
                attempt for attempt in attempts
                if str(attempt.get("status") or "").upper() in inflight_attempt_states
            ]
            stage_state = str(stage.get("status") or "").upper()
            if stage_state in {"CANCELLING", "CANCELLED"}:
                return {
                    "claimed": False,
                    "status": stage_state,
                    "stage": stage,
                    "entity": entity,
                    "active_attempts": active_attempts,
                    "event": None,
                }
            if stage_state not in cancellable_states:
                raise ValueError(
                    f"Workflow stage cannot be cancelled from {stage_state or 'UNKNOWN'}"
                )

            has_inflight_work = bool(active_attempts) or stage_state in inflight_stage_states
            target_status = "CANCELLING" if has_inflight_work else "CANCELLED"
            completed_at = None if has_inflight_work else now
            cursor = connection.execute(
                """
                UPDATE workflow_stages
                SET status = ?, completed_at = ?, updated_at = ?
                WHERE id = ? AND status = ?
                """,
                (target_status, completed_at, now, stage_id, stage["status"]),
            )
            if cursor.rowcount != 1:
                current = self._row_by_id(connection, "workflow_stages", stage_id)
                return {
                    "claimed": False,
                    "status": str((current or {}).get("status") or "UNKNOWN").upper(),
                    "stage": current,
                    "entity": entity,
                    "active_attempts": active_attempts,
                    "event": None,
                }
            connection.execute(
                f"UPDATE {entity_table} SET status = ?, completed_at = ?, updated_at = ? WHERE id = ?",
                (target_status, completed_at, now, entity_id),
            )
            if target_status == "CANCELLING":
                for attempt in active_attempts:
                    connection.execute(
                        "UPDATE job_attempts SET status = 'CANCELLING', updated_at = ? WHERE id = ?",
                        (now, attempt["id"]),
                    )
                    attempt["status"] = "CANCELLING"
            event_details = {
                "stage_id": stage_id,
                "stage_type": stage["stage_type"],
                "attempt_ids": [attempt["id"] for attempt in active_attempts],
                "job_ids": [
                    attempt["slurm_job_id"]
                    for attempt in active_attempts
                    if attempt.get("slurm_job_id")
                ],
                **dict(details or {}),
            }
            event_values = {
                "id": new_id(),
                "entity_type": entity_type,
                "entity_id": entity_id,
                "event_type": "CANCEL_REQUESTED",
                "old_status": entity["status"],
                "new_status": target_status,
                "details_json": canonical_json(event_details),
                "created_at": now,
            }
            self._insert(connection, "events", event_values)
            if target_status == "CANCELLED":
                self._insert(
                    connection,
                    "events",
                    {
                        "id": new_id(),
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "event_type": "JOB_CANCELLED",
                        "old_status": target_status,
                        "new_status": "CANCELLED",
                        "details_json": canonical_json({
                            **event_details,
                            "before_submission": True,
                        }),
                        "created_at": now,
                    },
                )
            return {
                "claimed": True,
                "status": target_status,
                "stage": self._row_by_id(connection, "workflow_stages", stage_id),
                "entity": self._row_by_id(connection, entity_table, entity_id),
                "active_attempts": active_attempts,
                "event": self._row_by_id(connection, "events", event_values["id"]),
            }

    def transition_workflow_state(
        self,
        *,
        stage_id: str,
        stage_updates: Mapping[str, Any],
        attempt_id: str | None = None,
        attempt_updates: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        run_updates: Mapping[str, Any] | None = None,
        evaluation_id: str | None = None,
        evaluation_updates: Mapping[str, Any] | None = None,
        event: Mapping[str, Any] | None = None,
        expected_stage_statuses: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Commit one attempt/stage/run/evaluation transition and its event together."""
        if self.workspace_id is not None and stage_id is not None and not self.owns("workflow_stages", stage_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and attempt_id is not None and not self.owns("job_attempts", attempt_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and evaluation_id is not None and not self.owns("evaluations", evaluation_id):
            raise KeyError("Record not found in this workspace")

        now = utc_now()
        encoded_stage = self._encode_updates(
            stage_updates,
            allowed=frozenset({
                "status", "resolved_config_json", "auto_resume", "max_attempts", "started_at", "completed_at",
            }),
            json_fields=frozenset({"resolved_config_json"}),
        )
        encoded_stage["updated_at"] = now
        encoded_attempt = None
        if attempt_id is not None:
            encoded_attempt = self._encode_updates(
                attempt_updates or {},
                allowed=frozenset({
                    "slurm_job_id", "slurm_array_job_id", "slurm_array_task_id", "gateway", "account",
                    "partition_name", "node_list", "gpu_type", "gpu_count", "cpu_count", "memory_mb",
                    "time_limit_seconds", "status", "slurm_state", "slurm_reason", "exit_code", "restart_count",
                    "sbatch_path", "stdout_path", "stderr_path", "resume_checkpoint_id", "cluster_snapshot_id",
                    "execution_snapshot_json", "execution_snapshot_sha256",
                    "submitted_at", "started_at", "finished_at",
                }),
                json_fields=frozenset({"execution_snapshot_json"}),
            )
            encoded_attempt["updated_at"] = now
        encoded_run = None
        if run_id is not None and run_updates is not None:
            encoded_run = self._encode_updates(
                run_updates,
                allowed=frozenset({
                    "status", "source_commit", "runtime_profile", "run_directory", "mlflow_run_id",
                    "started_at", "completed_at",
                }),
            )
            encoded_run["updated_at"] = now
        encoded_evaluation = None
        if evaluation_id is not None and evaluation_updates is not None:
            encoded_evaluation = self._encode_updates(
                evaluation_updates,
                allowed=frozenset({
                    "status", "progress_completed", "progress_total", "result_path", "started_at", "completed_at",
                    "task_selection_json", "seeds_json", "episodes_per_task",
                }),
                json_fields=frozenset({"task_selection_json", "seeds_json"}),
            )
            encoded_evaluation["updated_at"] = now

        with self.transaction() as connection:
            if expected_stage_statuses is not None:
                current_stage = self._row_by_id(connection, "workflow_stages", stage_id)
                if current_stage is None:
                    raise KeyError("Workflow stage not found")
                expected = {str(value).upper() for value in expected_stage_statuses}
                if str(current_stage.get("status") or "").upper() not in expected:
                    return {"applied": False, "stage": current_stage}
            result: dict[str, Any] = {
                "stage": self._update(connection, "workflow_stages", stage_id, encoded_stage)
            }
            if attempt_id is not None and encoded_attempt is not None:
                result["attempt"] = self._update(
                    connection, "job_attempts", attempt_id, encoded_attempt
                )
            if run_id is not None and encoded_run is not None:
                result["run"] = self._update(connection, "runs", run_id, encoded_run)
            if evaluation_id is not None and encoded_evaluation is not None:
                result["evaluation"] = self._update(
                    connection, "evaluations", evaluation_id, encoded_evaluation
                )
            if event is not None:
                event_values = {
                    "id": new_id(),
                    "entity_type": str(event["entity_type"]),
                    "entity_id": str(event["entity_id"]),
                    "event_type": str(event["event_type"]),
                    "old_status": event.get("old_status"),
                    "new_status": event.get("new_status"),
                    "details_json": canonical_json(event.get("details") or {}),
                    "created_at": now,
                }
                self._insert(connection, "events", event_values)
                result["event"] = self._row_by_id(connection, "events", event_values["id"])
            if expected_stage_statuses is not None:
                result["applied"] = True
            return result

    def repair_pre_submission_orphans(
        self, experiment_id: str | None = None
    ) -> dict[str, Any]:
        """Restore false submission state for revisions with no attempt history."""
        if self.workspace_id is not None and experiment_id is not None and not self.owns("experiments", experiment_id):
            raise KeyError("Record not found in this workspace")

        now = utc_now()
        repaired = 0
        revision_ids: set[str] = set()
        experiment_ids: set[str] = set()
        with self.transaction() as connection:
            rows = connection.execute(
                f"""
                SELECT er.id AS revision_id, er.experiment_id,
                       er.submitted_at, e.status AS experiment_status,
                       (
                           SELECT latest.id
                           FROM experiment_revisions latest
                           WHERE latest.experiment_id = er.experiment_id
                           ORDER BY latest.revision_number DESC
                           LIMIT 1
                       ) AS latest_revision_id
                FROM experiment_revisions er
                JOIN experiments e ON e.id = er.experiment_id
                WHERE {visible_sql("experiment_revisions", "er")} AND (? IS NULL OR er.experiment_id = ?)
                  AND NOT EXISTS (
                      SELECT 1
                      FROM variants v
                      JOIN runs r ON r.variant_id = v.id
                      JOIN workflow_stages s ON s.run_id = r.id
                      JOIN job_attempts a ON a.stage_id = s.id
                      WHERE v.experiment_revision_id = er.id
                  )
                  AND (
                      er.submitted_at IS NOT NULL
                      OR EXISTS (
                          SELECT 1
                          FROM variants v
                          JOIN runs r ON r.variant_id = v.id
                          WHERE v.experiment_revision_id = er.id
                            AND UPPER(COALESCE(r.status, '')) != 'DRAFT'
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM variants v
                          JOIN runs r ON r.variant_id = v.id
                          JOIN workflow_stages s ON s.run_id = r.id
                          WHERE v.experiment_revision_id = er.id
                            AND UPPER(COALESCE(s.status, '')) != 'DRAFT'
                      )
                      OR (
                          er.id = (
                              SELECT latest.id
                              FROM experiment_revisions latest
                              WHERE latest.experiment_id = er.experiment_id
                              ORDER BY latest.revision_number DESC
                              LIMIT 1
                          )
                          AND UPPER(COALESCE(e.status, '')) != 'DRAFT'
                      )
                  )
                ORDER BY er.experiment_id, er.revision_number
                """,
                (experiment_id, experiment_id),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE workflow_stages
                    SET status = 'DRAFT', started_at = NULL,
                        completed_at = NULL, updated_at = ?
                    WHERE run_id IN (
                        SELECT r.id
                        FROM runs r
                        JOIN variants v ON v.id = r.variant_id
                        WHERE v.experiment_revision_id = ?
                    )
                    """,
                    (now, row["revision_id"]),
                )
                connection.execute(
                    """
                    UPDATE runs
                    SET status = 'DRAFT', started_at = NULL,
                        completed_at = NULL, updated_at = ?
                    WHERE variant_id IN (
                        SELECT id FROM variants
                        WHERE experiment_revision_id = ?
                    )
                    """,
                    (now, row["revision_id"]),
                )
                connection.execute(
                    """
                    UPDATE experiment_revisions
                    SET submitted_at = NULL
                    WHERE id = ?
                    """,
                    (row["revision_id"],),
                )
                if row["revision_id"] == row["latest_revision_id"]:
                    connection.execute(
                        """
                        UPDATE experiments
                        SET status = 'DRAFT', updated_at = ?
                        WHERE id = ?
                        """,
                        (now, row["experiment_id"]),
                    )
                self._insert(connection, "events", {
                    "id": new_id(),
                    "entity_type": "experiment",
                    "entity_id": row["experiment_id"],
                    "event_type": "PRE_SUBMISSION_ORPHAN_REPAIRED",
                    "old_status": row["experiment_status"],
                    "new_status": "DRAFT",
                    "details_json": canonical_json({
                        "experiment_revision_id": row["revision_id"],
                        "submitted_at": row["submitted_at"],
                    }),
                    "created_at": now,
                })
                repaired += 1
                revision_ids.add(row["revision_id"])
                experiment_ids.add(row["experiment_id"])
        return {
            "repaired": repaired,
            "revision_ids": sorted(revision_ids),
            "experiment_ids": sorted(experiment_ids),
        }

    def repair_workflow_state_invariants(self) -> dict[str, Any]:
        """Repair interrupted terminal transitions using their latest attempt as evidence."""

        draft_repairs = self.repair_pre_submission_orphans()
        active_stage_states = (
            "SUBMITTING", "SUBMITTED", "PENDING_SLURM", "RUNNING", "CANCELLING",
        )
        terminal_attempt_states = (
            "SUBMISSION_FAILED", "FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL",
            "BOOT_FAIL", "CANCELLED", "PREEMPTED", "DEADLINE", "SPECIAL_EXIT",
        )
        transient_states = {"PREEMPTED", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL"}
        now = utc_now()
        repaired = 0
        run_ids: set[str] = set()
        experiment_ids: set[str] = set(draft_repairs["experiment_ids"])
        with self.transaction() as connection:
            placeholders_stage = ",".join("?" for _ in active_stage_states)
            placeholders_attempt = ",".join("?" for _ in terminal_attempt_states)
            rows = connection.execute(
                f"""
                SELECT s.id AS stage_id, s.run_id, s.stage_type, s.status AS stage_status,
                       s.auto_resume, s.max_attempts,
                       a.id AS attempt_id, a.attempt_number, a.status AS attempt_status,
                       a.slurm_job_id, a.slurm_reason, e.id AS evaluation_id,
                       er.experiment_id,
                       (
                           SELECT COUNT(*) FROM job_attempts budget_attempt
                           WHERE budget_attempt.stage_id = s.id
                             AND UPPER(COALESCE(budget_attempt.status, '')) != 'CANCELLED'
                       ) AS budget_attempt_count
                FROM workflow_stages s
                JOIN job_attempts a ON a.stage_id = s.id
                JOIN runs r ON r.id = s.run_id
                JOIN variants v ON v.id = r.variant_id
                JOIN experiment_revisions er ON er.id = v.experiment_revision_id
                LEFT JOIN evaluations e ON e.stage_id = s.id
                WHERE {visible_sql("workflow_stages", "s")} AND s.status IN ({placeholders_stage})
                  AND a.status IN ({placeholders_attempt})
                  AND NOT EXISTS (
                      SELECT 1 FROM job_attempts active
                      WHERE active.stage_id = a.stage_id
                        AND active.slurm_job_id IS NOT NULL
                        AND active.status IN ('SUBMITTED', 'PENDING', 'RUNNING', 'REQUEUED', 'CANCELLING')
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM job_attempts newer
                      WHERE newer.stage_id = a.stage_id
                        AND newer.attempt_number > a.attempt_number
                  )
                """,
                (*active_stage_states, *terminal_attempt_states),
            ).fetchall()
            for row in rows:
                cancellation_requested = row["stage_status"] == "CANCELLING"
                can_retry = (
                    not cancellation_requested
                    and
                    row["attempt_status"] in transient_states
                    and bool(row["auto_resume"])
                    and int(row["budget_attempt_count"]) < int(row["max_attempts"])
                )
                if cancellation_requested or row["attempt_status"] == "CANCELLED":
                    target = "CANCELLED"
                else:
                    target = "RETRY_PENDING" if can_retry else "FAILED"
                completed_at = None if can_retry else now
                connection.execute(
                    """
                    UPDATE workflow_stages
                    SET status = ?, completed_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (target, completed_at, now, row["stage_id"]),
                )
                if row["stage_type"] in {"TRAIN", "UTILITY"}:
                    connection.execute(
                        """
                        UPDATE runs
                        SET status = ?, completed_at = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (target, completed_at, now, row["run_id"]),
                    )
                elif row["evaluation_id"]:
                    connection.execute(
                        """
                        UPDATE evaluations
                        SET status = ?, completed_at = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (target, completed_at, now, row["evaluation_id"]),
                    )
                event_values = {
                    "id": new_id(),
                    "entity_type": "evaluation" if row["evaluation_id"] else "run",
                    "entity_id": row["evaluation_id"] or row["run_id"],
                    "event_type": "WORKFLOW_STATE_REPAIRED",
                    "old_status": "ACTIVE",
                    "new_status": target,
                    "details_json": canonical_json({
                        "attempt_id": row["attempt_id"],
                        "attempt_status": row["attempt_status"],
                        "slurm_job_id": row["slurm_job_id"],
                        "reason": row["slurm_reason"],
                    }),
                    "created_at": now,
                }
                self._insert(connection, "events", event_values)
                repaired += 1
                run_ids.add(row["run_id"])
                experiment_ids.add(row["experiment_id"])
        return {
            "repaired": repaired,
            "draft_graphs_repaired": draft_repairs["repaired"],
            "revision_ids": draft_repairs["revision_ids"],
            "run_ids": sorted(run_ids),
            "experiment_ids": sorted(experiment_ids),
        }

    def create_checkpoint(
        self,
        run_id: str,
        *,
        checkpoint_type: str,
        path: str,
        produced_by_attempt_id: str | None = None,
        training_step: int | None = None,
        sha256: str | None = None,
        size_bytes: int | None = None,
        is_resumable: bool = False,
        is_selected_for_inference: bool = False,
        validation_metric: str | None = None,
        validation_metric_value: float | None = None,
        status: str = "AVAILABLE",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a checkpoint once, including concurrent finalization retries.

        Replaying the same receipt preserves its ID and evaluation references.
        A different receipt at an existing path is still an integrity error.
        """
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and produced_by_attempt_id is not None and not self.owns("job_attempts", produced_by_attempt_id):
            raise KeyError("Record not found in this workspace")
        values = {
            "id": new_id(), "run_id": run_id, "produced_by_attempt_id": produced_by_attempt_id,
            "training_step": training_step, "checkpoint_type": checkpoint_type, "path": path,
            "sha256": sha256, "size_bytes": size_bytes, "is_resumable": int(is_resumable),
            "is_selected_for_inference": int(is_selected_for_inference),
            "validation_metric": validation_metric, "validation_metric_value": validation_metric_value,
            "status": status, "metadata_json": canonical_json(metadata or {}), "created_at": utc_now(),
            "pruned_at": None,
        }
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? AND path = ?",
                (run_id, path),
            ).fetchone()
            if existing is not None:
                receipt_fields = values.keys() - {
                    "id", "created_at", "is_selected_for_inference",
                }
                if any(existing[key] != values[key] for key in receipt_fields):
                    raise sqlite3.IntegrityError(
                        "checkpoint path is already registered with a different receipt"
                    )
                values["id"] = existing["id"]
            if is_selected_for_inference:
                connection.execute(
                    """
                    UPDATE checkpoints
                    SET is_selected_for_inference = 0
                    WHERE run_id = ? AND is_selected_for_inference <> 0
                    """,
                    (run_id,),
                )
            if existing is None:
                self._insert(connection, "checkpoints", values)
            elif is_selected_for_inference:
                connection.execute(
                    "UPDATE checkpoints SET is_selected_for_inference = 1 WHERE id = ?",
                    (existing["id"],),
                )
            result = self._row_by_id(connection, "checkpoints", values["id"])
        assert result is not None
        return result

    def list_checkpoints(
        self, run_id: str, *, available_only: bool = False, resumable_only: bool = False
    ) -> list[dict[str, Any]]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            return []
        clauses = ["run_id = ?"]
        parameters: list[Any] = [run_id]
        if available_only:
            clauses.append("pruned_at IS NULL")
            clauses.append("status = 'AVAILABLE'")
        if resumable_only:
            clauses.append("is_resumable = 1")
        with self.connection() as connection:
            return self._decode_many(connection.execute(
                f"SELECT * FROM checkpoints WHERE {' AND '.join(clauses)} ORDER BY training_step DESC, created_at DESC",
                parameters,
            ).fetchall())

    def update_checkpoint(self, checkpoint_id: str, **fields: Any) -> dict[str, Any]:
        if self.workspace_id is not None and checkpoint_id is not None and not self.owns("checkpoints", checkpoint_id):
            raise KeyError("Record not found in this workspace")
        encoded = self._encode_updates(
            fields,
            allowed=frozenset({
                "sha256", "size_bytes", "is_resumable", "is_selected_for_inference", "validation_metric",
                "validation_metric_value", "status", "metadata_json", "pruned_at",
            }),
            json_fields=frozenset({"metadata_json"}),
        )
        with self.transaction() as connection:
            return self._update(connection, "checkpoints", checkpoint_id, encoded)

    def checkpoint_prune_candidates(self, run_id: str, *, keep_last: int = 3) -> list[dict[str, Any]]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            raise KeyError("Record not found in this workspace")
        if keep_last < 0:
            raise ValueError("keep_last must be non-negative")
        with self.connection() as connection:
            return self._decode_many(connection.execute("""
                SELECT * FROM checkpoints
                WHERE run_id = ? AND is_resumable = 1 AND is_selected_for_inference = 0 AND pruned_at IS NULL
                ORDER BY COALESCE(training_step, -1) DESC, created_at DESC
                LIMIT -1 OFFSET ?
            """, (run_id, keep_last)).fetchall())

    def register_evaluation_suite(
        self,
        *,
        evaluator_adapter: str,
        evaluator_version: str,
        name: str,
        suite_version: str,
        config: Mapping[str, Any],
        description: str = "",
        catalog_path: str | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        now = utc_now()
        serialized_config = canonical_json(config)
        with self.transaction() as connection:
            existing = connection.execute("""
                SELECT * FROM evaluation_suites
                WHERE evaluator_adapter = ? AND evaluator_version = ? AND name = ? AND suite_version = ?
            """, (evaluator_adapter, evaluator_version, name, suite_version)).fetchone()
            if existing:
                decoded = self._decode(existing)
                assert decoded is not None
                if (
                    decoded["description"] != description
                    or decoded.get("catalog_path") != catalog_path
                    or canonical_json(decoded["config_json"]) != serialized_config
                ):
                    raise ValueError(
                        "evaluation suite version already exists with different content; "
                        "publish a new suite_version"
                    )
                suite_id = decoded["id"]
            else:
                suite_id = new_id()
                self._insert(connection, "evaluation_suites", {
                    "id": suite_id, "evaluator_adapter": evaluator_adapter,
                    "evaluator_version": evaluator_version, "name": name, "suite_version": suite_version,
                    "description": description, "catalog_path": catalog_path,
                    "config_json": serialized_config, "enabled": int(enabled),
                    "created_at": now, "updated_at": now,
                })
            if enabled:
                connection.execute(
                    """
                    UPDATE evaluation_suites
                    SET enabled = 0, updated_at = ?
                    WHERE evaluator_adapter = ? AND name = ? AND id <> ? AND enabled <> 0
                    """,
                    (now, evaluator_adapter, name, suite_id),
                )
            connection.execute(
                "UPDATE evaluation_suites SET enabled = ?, updated_at = ? WHERE id = ?",
                (int(enabled), now, suite_id),
            )
            result = self._row_by_id(connection, "evaluation_suites", suite_id)
        assert result is not None
        return result

    create_evaluation_suite = register_evaluation_suite

    def list_evaluation_suites(
        self, *, evaluator_adapter: str | None = None, enabled_only: bool = True
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if evaluator_adapter is not None:
            clauses.append("evaluator_adapter = ?")
            parameters.append(evaluator_adapter)
        if enabled_only:
            clauses.append("enabled = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connection() as connection:
            return self._decode_many(connection.execute(
                f"SELECT * FROM evaluation_suites {where} ORDER BY evaluator_adapter, name, suite_version",
                parameters,
            ).fetchall())

    def create_evaluation(
        self,
        run_id: str,
        *,
        evaluator_adapter: str,
        evaluator_version: str,
        suite_name: str,
        suite_version: str,
        tasks: Sequence[str],
        seeds: Sequence[int],
        episodes_per_task: int,
        stage_id: str | None = None,
        checkpoint_id: str | None = None,
        evaluation_suite_id: str | None = None,
        status: str = "PENDING",
        result_path: str | None = None,
    ) -> dict[str, Any]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and stage_id is not None and not self.owns("workflow_stages", stage_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and checkpoint_id is not None and not self.owns("checkpoints", checkpoint_id):
            raise KeyError("Record not found in this workspace")
        now = utc_now()
        values = {
            "id": new_id(), "run_id": run_id, "stage_id": stage_id, "checkpoint_id": checkpoint_id,
            "evaluation_suite_id": evaluation_suite_id, "evaluator_adapter": evaluator_adapter,
            "evaluator_version": evaluator_version, "suite_name": suite_name,
            "suite_version": suite_version, "task_selection_json": canonical_json(list(tasks)),
            "seeds_json": canonical_json(list(seeds)), "episodes_per_task": episodes_per_task,
            "status": status, "progress_completed": 0,
            "progress_total": len(tasks) * len(seeds) * episodes_per_task,
            "result_path": result_path, "created_at": now, "started_at": None,
            "completed_at": None, "updated_at": now,
        }
        with self.transaction() as connection:
            self._insert(connection, "evaluations", values)
            result = self._row_by_id(connection, "evaluations", values["id"])
        assert result is not None
        return result

    def list_evaluations(
        self, *, run_id: str | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            return []
        clauses: list[str] = [visible_sql("evaluations", "")]
        parameters: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            parameters.append(run_id)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connection() as connection:
            evaluations = self._decode_many(connection.execute(
                f"SELECT * FROM evaluations {where} ORDER BY created_at DESC", parameters
            ).fetchall())
            # Fetch result summaries in batches; list rows must not depend on
            # opening each result detail (or issue one query per evaluation).
            by_id = {item["id"]: item for item in evaluations}
            identifiers = list(by_id)
            for start in range(0, len(identifiers), 400):
                batch = identifiers[start:start + 400]
                placeholders = ",".join("?" for _ in batch)
                artifacts = self._decode_many(connection.execute(f"""
                    SELECT evaluation_id, metadata_json FROM artifacts
                    WHERE artifact_type = 'EVALUATION_RESULT'
                      AND evaluation_id IN ({placeholders})
                    ORDER BY created_at
                """, batch).fetchall())
                for artifact in artifacts:
                    by_id[artifact["evaluation_id"]]["aggregate"] = (
                        artifact.get("metadata_json") or {}
                    ).get("aggregate", [])
            return evaluations

    @staticmethod
    def _unfinished_evaluation_episode_state(status: str) -> tuple[str, str] | None:
        status = str(status).upper()
        if status not in {"SUCCEEDED", "FAILED", "CANCELLED", "BLOCKED", "TIMEOUT", "SUBMISSION_FAILED"}:
            return None
        return (
            "CANCELLED" if status == "CANCELLED" else "NOT_COMPLETED",
            f"Evaluation ended ({status.lower()}) before this episode completed.",
        )

    def get_evaluation(self, evaluation_id: str) -> dict[str, Any] | None:
        if self.workspace_id is not None and evaluation_id is not None and not self.owns("evaluations", evaluation_id):
            return None
        with self.connection() as connection:
            evaluation = self._row_by_id(connection, "evaluations", evaluation_id)
            if evaluation is None:
                return None
            evaluation["episodes"] = self._decode_many(connection.execute(
                "SELECT * FROM evaluation_episodes WHERE evaluation_id = ? ORDER BY task, seed, episode_index",
                (evaluation_id,),
            ).fetchall())
            # Older terminal records may predate episode finalization. Present
            # their effective state without changing the stored historical row.
            terminal = self._unfinished_evaluation_episode_state(evaluation["status"])
            if terminal:
                for episode in evaluation["episodes"]:
                    if episode["status"] in {"PENDING", "RUNNING"}:
                        episode["status"] = terminal[0]
                        episode["failure_reason"] = episode.get("failure_reason") or terminal[1]
            return evaluation

    def evaluation_progress_evidence(
        self, evaluation_ids: Sequence[str]
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Return episode and current-stage attempt evidence without per-row queries."""
        identifiers = list(dict.fromkeys(
            str(evaluation_id) for evaluation_id in evaluation_ids if evaluation_id
        ))
        identifiers = self._visible_ids("evaluations", identifiers)
        evidence: dict[str, dict[str, list[dict[str, Any]]]] = {
            evaluation_id: {"attempts": [], "episodes": []}
            for evaluation_id in identifiers
        }
        if not identifiers:
            return evidence
        with self.connection() as connection:
            for start in range(0, len(identifiers), 400):
                batch = identifiers[start:start + 400]
                placeholders = ",".join("?" for _ in batch)
                attempts = self._decode_many(connection.execute(f"""
                    SELECT e.id AS evaluation_id, a.*
                    FROM evaluations e
                    JOIN job_attempts a ON a.stage_id = e.stage_id
                    WHERE e.id IN ({placeholders})
                    ORDER BY e.id, a.attempt_number, a.created_at
                """, batch).fetchall())
                for row in attempts:
                    evidence[row["evaluation_id"]]["attempts"].append(row)
                episodes = self._decode_many(connection.execute(f"""
                    SELECT * FROM evaluation_episodes
                    WHERE evaluation_id IN ({placeholders})
                    ORDER BY evaluation_id, task, seed, episode_index
                """, batch).fetchall())
                for row in episodes:
                    evidence[row["evaluation_id"]]["episodes"].append(row)
        return evidence

    def update_evaluation(self, evaluation_id: str, **fields: Any) -> dict[str, Any]:
        if self.workspace_id is not None and evaluation_id is not None and not self.owns("evaluations", evaluation_id):
            raise KeyError("Record not found in this workspace")
        encoded = self._encode_updates(fields, allowed=frozenset({
            "status", "progress_completed", "progress_total", "result_path", "started_at", "completed_at",
            "task_selection_json", "seeds_json", "episodes_per_task",
        }), json_fields=frozenset({"task_selection_json", "seeds_json"}))
        encoded["updated_at"] = utc_now()
        with self.transaction() as connection:
            result = self._update(connection, "evaluations", evaluation_id, encoded)
            status = str(fields.get("status", "")).upper()
            terminal = self._unfinished_evaluation_episode_state(status)
            if terminal:
                connection.execute(
                    """UPDATE evaluation_episodes
                       SET status = ?, failure_reason = COALESCE(failure_reason, ?),
                           completed_at = COALESCE(completed_at, ?), updated_at = ?
                       WHERE evaluation_id = ? AND status IN ('PENDING', 'RUNNING')""",
                    (*terminal,
                     encoded["updated_at"], encoded["updated_at"], evaluation_id),
                )
            elif status in {"PENDING", "SUBMITTED"}:
                connection.execute(
                    """UPDATE evaluation_episodes SET status = 'PENDING',
                       failure_reason = NULL, completed_at = NULL, updated_at = ?
                       WHERE evaluation_id = ? AND status IN ('CANCELLED', 'NOT_COMPLETED')""",
                    (encoded["updated_at"], evaluation_id),
                )
            return result

    def upsert_evaluation_episode(
        self,
        evaluation_id: str,
        *,
        task: str,
        seed: int,
        episode_index: int,
        status: str = "PENDING",
        attempt_count: int = 0,
        success: bool | None = None,
        reward: float | None = None,
        episode_length: int | None = None,
        failure_reason: str | None = None,
        metrics: Mapping[str, Any] | None = None,
        video_path: str | None = None,
        raw_result_path: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        expected_parent_states: Sequence[str] | None = None,
    ) -> dict[str, Any] | None:
        if self.workspace_id is not None and evaluation_id is not None and not self.owns("evaluations", evaluation_id):
            raise KeyError("Record not found in this workspace")
        now = utc_now()
        with self.transaction() as connection:
            if expected_parent_states is not None:
                parent = connection.execute("SELECT status FROM evaluations WHERE id = ?", (evaluation_id,)).fetchone()
                if parent is None or parent["status"] not in expected_parent_states:
                    return None
            existing = connection.execute("""
                SELECT id FROM evaluation_episodes
                WHERE evaluation_id = ? AND task = ? AND seed = ? AND episode_index = ?
            """, (evaluation_id, task, seed, episode_index)).fetchone()
            values = {
                "status": status, "attempt_count": attempt_count,
                "success": None if success is None else int(success), "reward": reward,
                "episode_length": episode_length, "failure_reason": failure_reason,
                "metrics_json": canonical_json(metrics or {}), "video_path": video_path,
                "raw_result_path": raw_result_path, "started_at": started_at,
                "completed_at": completed_at, "updated_at": now,
            }
            if existing:
                episode_id = existing[0]
                self._update(connection, "evaluation_episodes", episode_id, values)
            else:
                episode_id = new_id()
                self._insert(connection, "evaluation_episodes", {
                    "id": episode_id, "evaluation_id": evaluation_id, "task": task,
                    "seed": seed, "episode_index": episode_index, **values,
                })
            progress = connection.execute("""
                SELECT COUNT(*) FROM evaluation_episodes
                WHERE evaluation_id = ? AND status IN ('SUCCEEDED', 'FAILED', 'TIMEOUT')
            """, (evaluation_id,)).fetchone()[0]
            connection.execute(
                "UPDATE evaluations SET progress_completed = ?, updated_at = ? WHERE id = ?",
                (progress, now, evaluation_id),
            )
            result = self._row_by_id(connection, "evaluation_episodes", episode_id)
        assert result is not None
        return result

    def record_metric(
        self,
        run_id: str,
        *,
        name: str,
        value: float,
        scope: str,
        evaluation_id: str | None = None,
        step: int | None = None,
        unit: str | None = None,
        sample_count: int | None = None,
        recorded_at: str | None = None,
    ) -> dict[str, Any]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and evaluation_id is not None and not self.owns("evaluations", evaluation_id):
            raise KeyError("Record not found in this workspace")
        values = {
            "id": new_id(), "run_id": run_id, "evaluation_id": evaluation_id, "name": name,
            "scope": scope, "step": step, "value": value, "unit": unit,
            "sample_count": sample_count, "recorded_at": recorded_at or utc_now(),
        }
        with self.transaction() as connection:
            self._insert(connection, "metrics", values)
        return values

    def record_training_progress_sample(
        self,
        run_id: str,
        attempt_id: str,
        *,
        restart_count: int,
        completed: int,
        total: int | None,
        source_kind: str,
        evidence: Mapping[str, Any] | None = None,
        recorded_at: str | None = None,
        unit: str = "step",
    ) -> dict[str, Any]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and attempt_id is not None and not self.owns("job_attempts", attempt_id):
            raise KeyError("Record not found in this workspace")
        if restart_count < 0 or completed < 0 or (total is not None and total < 1):
            raise ValueError("invalid training progress sample")
        now = utc_now()
        values = {
            "id": new_id(),
            "run_id": run_id,
            "attempt_id": attempt_id,
            "restart_count": restart_count,
            "completed": completed,
            "total": total,
            "unit": unit,
            "source_kind": source_kind,
            "evidence_json": canonical_json(evidence or {}),
            "recorded_at": recorded_at or now,
            "created_at": now,
        }
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO training_progress_samples (
                    id, run_id, attempt_id, restart_count, completed, total, unit,
                    source_kind, evidence_json, recorded_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attempt_id, restart_count, completed) DO NOTHING
                """,
                tuple(values[key] for key in (
                    "id", "run_id", "attempt_id", "restart_count", "completed",
                    "total", "unit", "source_kind", "evidence_json", "recorded_at",
                    "created_at",
                )),
            )
            result = self._decode(connection.execute(
                """
                SELECT * FROM training_progress_samples
                WHERE attempt_id = ? AND restart_count = ? AND completed = ?
                """,
                (attempt_id, restart_count, completed),
            ).fetchone())
        assert result is not None
        return result

    def enrich_training_progress_metrics(
        self, sample_id: str, metrics: Mapping[str, float | int]
    ) -> bool:
        """Add missing metrics without changing the sample's original observations."""
        additions = {
            key: value
            for key, value in metrics.items()
            if isinstance(key, str) and is_scalar(value)
        }
        if not additions:
            return False
        with self.transaction() as connection:
            row = connection.execute(
                f"SELECT evidence_json FROM training_progress_samples WHERE id = ? AND {visible_sql('training_progress_samples')}",
                (sample_id,),
            ).fetchone()
            if row is None:
                return False
            evidence = json.loads(row["evidence_json"])
            if not isinstance(evidence, dict):
                return False
            existing = evidence.get("metrics", {})
            if not isinstance(existing, dict):
                return False
            additions = {key: value for key, value in additions.items() if key not in existing}
            if not additions:
                return False
            evidence["metrics"] = {**existing, **additions}
            connection.execute(
                "UPDATE training_progress_samples SET evidence_json = ? WHERE id = ?",
                (canonical_json(evidence), sample_id),
            )
        return True

    def list_training_progress_samples(
        self, run_id: str, *, attempt_id: str | None = None
    ) -> list[dict[str, Any]]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            return []
        if self.workspace_id is not None and attempt_id is not None and not self.owns("job_attempts", attempt_id):
            return []
        clauses = ["run_id = ?"]
        parameters: list[Any] = [run_id]
        if attempt_id is not None:
            clauses.append("attempt_id = ?")
            parameters.append(attempt_id)
        with self.connection() as connection:
            return self._decode_many(connection.execute(
                f"SELECT * FROM training_progress_samples WHERE {' AND '.join(clauses)} "
                "ORDER BY recorded_at, completed",
                parameters,
            ).fetchall())

    def list_metrics(
        self, run_id: str, *, evaluation_id: str | None = None, name: str | None = None
    ) -> list[dict[str, Any]]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            return []
        if self.workspace_id is not None and evaluation_id is not None and not self.owns("evaluations", evaluation_id):
            return []
        clauses = ["run_id = ?"]
        parameters: list[Any] = [run_id]
        if evaluation_id is not None:
            clauses.append("evaluation_id = ?")
            parameters.append(evaluation_id)
        if name is not None:
            clauses.append("name = ?")
            parameters.append(name)
        with self.connection() as connection:
            return self._decode_many(connection.execute(
                f"SELECT * FROM metrics WHERE {' AND '.join(clauses)} ORDER BY recorded_at, step", parameters
            ).fetchall())

    def create_artifact(
        self,
        run_id: str,
        *,
        artifact_type: str,
        path: str,
        stage_id: str | None = None,
        evaluation_id: str | None = None,
        sha256: str | None = None,
        size_bytes: int | None = None,
        retention_policy: str | None = None,
        mlflow_artifact_uri: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and stage_id is not None and not self.owns("workflow_stages", stage_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and evaluation_id is not None and not self.owns("evaluations", evaluation_id):
            raise KeyError("Record not found in this workspace")
        values = {
            "id": new_id(), "run_id": run_id, "stage_id": stage_id,
            "evaluation_id": evaluation_id, "artifact_type": artifact_type, "path": path,
            "sha256": sha256, "size_bytes": size_bytes, "retention_policy": retention_policy,
            "mlflow_artifact_uri": mlflow_artifact_uri, "metadata_json": canonical_json(metadata or {}),
            "created_at": utc_now(), "deleted_at": None,
        }
        with self.transaction() as connection:
            self._insert(connection, "artifacts", values)
            result = self._row_by_id(connection, "artifacts", values["id"])
        assert result is not None
        return result

    def list_artifacts(self, run_id: str, *, artifact_type: str | None = None) -> list[dict[str, Any]]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            return []
        query = "SELECT * FROM artifacts WHERE run_id = ?"
        parameters: list[Any] = [run_id]
        if artifact_type is not None:
            query += " AND artifact_type = ?"
            parameters.append(artifact_type)
        query += " ORDER BY created_at"
        with self.connection() as connection:
            return self._decode_many(connection.execute(query, parameters).fetchall())

    def create_manifest(
        self,
        run_id: str,
        *,
        manifest_type: str,
        schema_version: str,
        path: str,
        sha256: str,
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        if self.workspace_id is not None and run_id is not None and not self.owns("runs", run_id):
            raise KeyError("Record not found in this workspace")
        if self.workspace_id is not None and attempt_id is not None and not self.owns("job_attempts", attempt_id):
            raise KeyError("Record not found in this workspace")
        values = {
            "id": new_id(), "run_id": run_id, "attempt_id": attempt_id,
            "manifest_type": manifest_type, "schema_version": schema_version,
            "path": path, "sha256": sha256, "created_at": utc_now(),
        }
        with self.transaction() as connection:
            self._insert(connection, "manifests", values)
            result = self._row_by_id(connection, "manifests", values["id"])
        assert result is not None
        return result

    def create_cluster_snapshot(
        self,
        *,
        gateway: str,
        gpu_usage: Any,
        nodes: Any,
        queue: Any,
        captured_at: str | None = None,
    ) -> dict[str, Any]:
        payload = {"gateway": gateway, "gpu_usage": gpu_usage, "nodes": nodes, "queue": queue}
        values = {
            "id": new_id(), "gateway": gateway, "captured_at": captured_at or utc_now(),
            "gpu_usage_json": canonical_json(gpu_usage), "nodes_json": canonical_json(nodes),
            "queue_json": canonical_json(queue), "sha256": content_sha256(payload),
        }
        with self.transaction() as connection:
            self._insert(connection, "cluster_snapshots", values)
            result = self._row_by_id(connection, "cluster_snapshots", values["id"])
        assert result is not None
        return result

    def record_event(
        self,
        *,
        entity_type: str,
        entity_id: str,
        event_type: str,
        old_status: str | None = None,
        new_status: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        values = {
            "id": new_id(), "entity_type": entity_type, "entity_id": entity_id,
            "event_type": event_type, "old_status": old_status, "new_status": new_status,
            "details_json": canonical_json(details or {}), "created_at": utc_now(),
        }
        with self.transaction() as connection:
            self._insert(connection, "events", values)
            result = self._row_by_id(connection, "events", values["id"])
        assert result is not None
        return result

    def record_attempt_common_hyperparameter_receipt(
        self, attempt_id: str, details: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Insert the v1 enrichment receipt once; retries return the original receipt."""
        if self.workspace_id is not None and attempt_id is not None and not self.owns("job_attempts", attempt_id):
            raise KeyError("Record not found in this workspace")

        event_type = "COMMON_HYPERPARAMETERS_ENRICHED_V1"
        encoded_details = canonical_json(dict(details))
        with self.transaction() as connection:
            existing = connection.execute(
                """
                SELECT * FROM events
                WHERE entity_type = 'job_attempt'
                  AND entity_id = ?
                  AND event_type = ?
                LIMIT 1
                """,
                (attempt_id, event_type),
            ).fetchone()
            if existing is not None:
                decoded = self._decode(existing)
                assert decoded is not None
                return decoded
            values = {
                "id": new_id(),
                "entity_type": "job_attempt",
                "entity_id": attempt_id,
                "event_type": event_type,
                "old_status": None,
                "new_status": None,
                "details_json": encoded_details,
                "created_at": utc_now(),
            }
            self._insert(connection, "events", values)
            result = self._row_by_id(connection, "events", values["id"])
        assert result is not None
        return result

    def list_events(
        self, *, entity_type: str | None = None, entity_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        clauses: list[str] = [visible_sql("events", "")]
        parameters: list[Any] = []
        if entity_type is not None:
            clauses.append("entity_type = ?")
            parameters.append(entity_type)
        if entity_id is not None:
            clauses.append("entity_id = ?")
            parameters.append(entity_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self.connection() as connection:
            return self._decode_many(connection.execute(
                f"SELECT * FROM events {where} ORDER BY created_at DESC LIMIT ?", parameters
            ).fetchall())

    def register_runtime_profile(
        self, *, name: str, backend: str, config: Mapping[str, Any], version: int = 1, enabled: bool = True
    ) -> dict[str, Any]:
        now = utc_now()
        with self.transaction() as connection:
            existing = connection.execute(f"SELECT id FROM runtime_profiles WHERE name = ? AND {visible_sql('runtime_profiles')}", (name,)).fetchone()
            if existing:
                profile_id = existing[0]
                self._update(connection, "runtime_profiles", profile_id, {
                    "backend": backend, "version": version, "config_json": canonical_json(config),
                    "enabled": int(enabled), "updated_at": now,
                })
            else:
                profile_id = new_id()
                self._insert(connection, "runtime_profiles", {
                    "id": profile_id, "name": name, "backend": backend, "version": version,
                    "config_json": canonical_json(config), "enabled": int(enabled),
                    "created_at": now, "updated_at": now,
                })
            result = self._row_by_id(connection, "runtime_profiles", profile_id)
        assert result is not None
        return result

    @staticmethod
    def _repository_url_from_manifest(manifest: Mapping[str, Any]) -> str | None:
        repository = manifest.get("repository")
        if isinstance(repository, Mapping):
            url = repository.get("url")
            return str(url).strip() if url else None
        return None

    @classmethod
    def _adapter_version_payload(cls, row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
        decoded = cls._decode(row) if isinstance(row, sqlite3.Row) else dict(row)
        assert decoded is not None
        return {
            "id": decoded["id"],
            "adapter_id": decoded["adapter_key"],
            "version_number": decoded["version_number"],
            "legacy_version": decoded["version"],
            "name": decoded["name"],
            "description": decoded.get("description") or "",
            "repository_url": decoded.get("repository_url"),
            "manifest": decoded.get("manifest_json") or {},
            "manifest_sha256": decoded.get("manifest_sha256"),
            "created_by": decoded.get("created_by"),
            "change_note": decoded.get("change_note") or "",
            "created_at": decoded["created_at"],
            "source_adapter_id": decoded.get("source_adapter_key"),
            "source_version_number": decoded.get("source_version_number"),
        }

    @classmethod
    def _adapter_bundle(
        cls,
        rows: Sequence[sqlite3.Row],
        *,
        selected_version: int | None = None,
        include_versions: bool = False,
    ) -> dict[str, Any] | None:
        if not rows:
            return None
        decoded = [cls._decode(row) for row in rows]
        versions = sorted(
            (row for row in decoded if row is not None),
            key=lambda row: int(row["version_number"]),
            reverse=True,
        )
        latest = versions[0]
        selected = latest
        if selected_version is not None:
            selected = next(
                (row for row in versions if int(row["version_number"]) == selected_version),
                None,
            )
            if selected is None:
                return None
        result = {
            "id": latest["adapter_key"],
            "name": latest["name"],
            "description": latest.get("description") or "",
            "repository_url": latest.get("repository_url"),
            "archived_at": latest.get("archived_at"),
            "enabled": bool(latest.get("enabled")),
            "seed_key": latest.get("seed_key"),
            "shared": latest.get("owner_id") is None,
            "created_at": min(str(row["created_at"]) for row in versions),
            "updated_at": latest["updated_at"],
            "latest_version_number": int(latest["version_number"]),
            "version_count": len(versions),
            "latest_version": cls._adapter_version_payload(latest),
            "selected_version": cls._adapter_version_payload(selected),
        }
        if include_versions:
            result["versions"] = [cls._adapter_version_payload(row) for row in versions]
        return result

    @staticmethod
    def _adapter_rows(
        connection: sqlite3.Connection, adapter_id: str
    ) -> list[sqlite3.Row]:
        return connection.execute(
            f"""
            SELECT * FROM adapters
            WHERE {visible_sql("adapters")} AND adapter_key = COALESCE(
                (SELECT adapter_key FROM adapters WHERE adapter_key = ? LIMIT 1),
                (SELECT adapter_key FROM adapters WHERE id = ? LIMIT 1)
            )
            ORDER BY version_number DESC
            """,
            (adapter_id, adapter_id),
        ).fetchall()

    @classmethod
    def _insert_adapter_version(
        cls,
        connection: sqlite3.Connection,
        *,
        adapter_key: str,
        version_number: int,
        name: str,
        manifest: Mapping[str, Any],
        description: str,
        repository_url: str | None,
        created_by: str | None,
        change_note: str | None = None,
        archived_at: str | None = None,
        seed_key: str | None = None,
        source_adapter_key: str | None = None,
        source_version_number: int | None = None,
        legacy_version: str | None = None,
    ) -> str:
        manifest_json = canonical_json(manifest)
        capabilities = manifest.get("capabilities")
        parameter_schema = manifest.get("parameter_schema", manifest.get("schema"))
        now = utc_now()
        version_id = new_id()
        stored_version = str(version_number)
        if legacy_version:
            stored_version = (
                legacy_version
                if version_number == 1
                else f"{legacy_version}+registry.{version_number}"
            )
        cls._insert(connection, "adapters", {
            "id": version_id,
            "name": name,
            "version": stored_version,
            "repository_url": repository_url,
            "capabilities_json": canonical_json(capabilities if isinstance(capabilities, Mapping) else {}),
            "schema_json": canonical_json(parameter_schema if isinstance(parameter_schema, Mapping) else {}),
            "enabled": int(archived_at is None),
            "created_at": now,
            "updated_at": now,
            "adapter_key": adapter_key,
            "version_number": version_number,
            "description": description,
            "manifest_json": manifest_json,
            "manifest_sha256": content_sha256(manifest_json),
            "archived_at": archived_at,
            "created_by": created_by,
            "change_note": change_note or "",
            "seed_key": seed_key,
            "source_adapter_key": source_adapter_key,
            "source_version_number": source_version_number,
        })
        return version_id

    def list_adapter_registry(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        with self.connection() as connection:
            keys = connection.execute(
                f"""
                SELECT adapter_key, lower(name) AS sort_name
                FROM adapters
                WHERE {visible_sql("adapters")} AND (? = 1 OR archived_at IS NULL)
                GROUP BY adapter_key
                ORDER BY sort_name, adapter_key
                """,
                (int(include_archived),),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for key in keys:
                bundle = self._adapter_bundle(self._adapter_rows(connection, key["adapter_key"]))
                if bundle is not None:
                    result.append(bundle)
            return result

    def get_adapter(
        self,
        adapter_id: str,
        *,
        version_number: int | None = None,
        include_versions: bool = True,
    ) -> dict[str, Any] | None:
        if self.workspace_id is not None and adapter_id is not None and not self.owns("adapters", adapter_id):
            return None
        with self.connection() as connection:
            return self._adapter_bundle(
                self._adapter_rows(connection, adapter_id),
                selected_version=version_number,
                include_versions=include_versions,
            )

    def create_adapter(
        self,
        *,
        name: str,
        manifest: Mapping[str, Any],
        description: str = "",
        repository_url: str | None = None,
        created_by: str | None = None,
        change_note: str | None = None,
    ) -> dict[str, Any]:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Adapter name must not be empty")
        if not isinstance(manifest, Mapping):
            raise ValueError("Adapter manifest must be an object")
        adapter_key = new_id()
        resolved_url = repository_url or self._repository_url_from_manifest(manifest)
        with self.transaction() as connection:
            self._insert_adapter_version(
                connection,
                adapter_key=adapter_key,
                version_number=1,
                name=normalized_name,
                manifest=manifest,
                description=description,
                repository_url=resolved_url,
                created_by=created_by,
                change_note=change_note,
            )
            result = self._adapter_bundle(
                self._adapter_rows(connection, adapter_key), include_versions=True
            )
        assert result is not None
        return result

    def edit_adapter(
        self,
        adapter_id: str,
        *,
        manifest: Mapping[str, Any],
        name: str | None = None,
        description: str | None = None,
        repository_url: str | None = None,
        created_by: str | None = None,
        change_note: str | None = None,
        expected_latest_version: int | None = None,
    ) -> dict[str, Any]:
        if self.workspace_id is not None and adapter_id is not None and not self.owns("adapters", adapter_id, writable=True):
            raise KeyError("Record not found in this workspace")
        if not isinstance(manifest, Mapping):
            raise ValueError("Adapter manifest must be an object")
        with self.transaction() as connection:
            rows = self._adapter_rows(connection, adapter_id)
            current = self._adapter_bundle(rows)
            if current is None:
                raise KeyError(f"Adapter not found: {adapter_id}")
            latest_number = int(current["latest_version_number"])
            if expected_latest_version is not None and expected_latest_version != latest_number:
                raise RuntimeError(
                    f"Adapter version conflict: expected {expected_latest_version}, found {latest_number}"
                )
            normalized_name = (name if name is not None else current["name"]).strip()
            if not normalized_name:
                raise ValueError("Adapter name must not be empty")
            inferred_url = self._repository_url_from_manifest(manifest)
            resolved_url = repository_url or inferred_url or current.get("repository_url")
            latest = current["latest_version"]
            self._insert_adapter_version(
                connection,
                adapter_key=current["id"],
                version_number=latest_number + 1,
                name=normalized_name,
                manifest=manifest,
                description=current["description"] if description is None else description,
                repository_url=resolved_url,
                created_by=created_by,
                change_note=change_note,
                archived_at=current["archived_at"],
                seed_key=current.get("seed_key"),
                source_adapter_key=latest.get("source_adapter_id"),
                source_version_number=latest.get("source_version_number"),
            )
            result = self._adapter_bundle(
                self._adapter_rows(connection, current["id"]), include_versions=True
            )
        assert result is not None
        return result

    def clone_adapter(
        self,
        adapter_id: str,
        *,
        name: str,
        version_number: int | None = None,
        description: str | None = None,
        created_by: str | None = None,
        change_note: str | None = None,
    ) -> dict[str, Any]:
        if self.workspace_id is not None and adapter_id is not None and not self.owns("adapters", adapter_id):
            raise KeyError("Record not found in this workspace")
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Adapter name must not be empty")
        with self.transaction() as connection:
            source = self._adapter_bundle(
                self._adapter_rows(connection, adapter_id), selected_version=version_number
            )
            if source is None:
                raise KeyError(f"Adapter or version not found: {adapter_id}")
            selected = source["selected_version"]
            clone_key = new_id()
            self._insert_adapter_version(
                connection,
                adapter_key=clone_key,
                version_number=1,
                name=normalized_name,
                manifest=selected["manifest"],
                description=source["description"] if description is None else description,
                repository_url=selected.get("repository_url"),
                created_by=created_by,
                change_note=change_note,
                source_adapter_key=source["id"],
                source_version_number=int(selected["version_number"]),
            )
            result = self._adapter_bundle(
                self._adapter_rows(connection, clone_key), include_versions=True
            )
        assert result is not None
        return result

    def archive_adapter(self, adapter_id: str) -> dict[str, Any]:
        if self.workspace_id is not None and adapter_id is not None and not self.owns("adapters", adapter_id, writable=True):
            raise KeyError("Record not found in this workspace")
        with self.transaction() as connection:
            rows = self._adapter_rows(connection, adapter_id)
            current = self._adapter_bundle(rows)
            if current is None:
                raise KeyError(f"Adapter not found: {adapter_id}")
            archived_at = current["archived_at"] or utc_now()
            connection.execute(
                "UPDATE adapters SET enabled = 0, archived_at = ?, updated_at = ? WHERE adapter_key = ?",
                (archived_at, utc_now(), current["id"]),
            )
            result = self._adapter_bundle(
                self._adapter_rows(connection, current["id"]), include_versions=True
            )
        assert result is not None
        return result

    def restore_adapter(self, adapter_id: str) -> dict[str, Any]:
        if self.workspace_id is not None and adapter_id is not None and not self.owns("adapters", adapter_id, writable=True):
            raise KeyError("Record not found in this workspace")
        with self.transaction() as connection:
            rows = self._adapter_rows(connection, adapter_id)
            current = self._adapter_bundle(rows)
            if current is None:
                raise KeyError(f"Adapter not found: {adapter_id}")
            connection.execute(
                "UPDATE adapters SET enabled = 1, archived_at = NULL, updated_at = ? WHERE adapter_key = ?",
                (utc_now(), current["id"]),
            )
            result = self._adapter_bundle(
                self._adapter_rows(connection, current["id"]), include_versions=True
            )
        assert result is not None
        return result

    def upsert_seed_adapter(
        self,
        *,
        seed_key: str,
        name: str,
        manifest: Mapping[str, Any],
        description: str = "",
        repository_url: str | None = None,
        legacy_version: str | None = None,
        change_note: str | None = None,
    ) -> dict[str, Any]:
        normalized_seed = seed_key.strip()
        normalized_name = name.strip()
        if not normalized_seed or not normalized_name:
            raise ValueError("Seed key and adapter name must not be empty")
        manifest_json = canonical_json(manifest)
        manifest_sha256 = content_sha256(manifest_json)
        resolved_url = repository_url or self._repository_url_from_manifest(manifest)
        with self.transaction() as connection:
            seeded = connection.execute(
                "SELECT adapter_key FROM adapters WHERE seed_key = ? LIMIT 1",
                (normalized_seed,),
            ).fetchone()
            if seeded is None:
                same_name = connection.execute(
                    "SELECT adapter_key FROM adapters WHERE lower(name) = lower(?) LIMIT 1",
                    (normalized_name,),
                ).fetchone()
                if same_name is not None:
                    adapter_key = same_name["adapter_key"]
                    connection.execute(
                        "UPDATE adapters SET seed_key = ? WHERE adapter_key = ? AND seed_key IS NULL",
                        (normalized_seed, adapter_key),
                    )
                else:
                    adapter_key = new_id()
                    self._insert_adapter_version(
                        connection,
                        adapter_key=adapter_key,
                        version_number=1,
                        name=normalized_name,
                        manifest=manifest,
                        description=description,
                        repository_url=resolved_url,
                        created_by="__seed__",
                        change_note=change_note,
                        seed_key=normalized_seed,
                        legacy_version=legacy_version,
                    )
                    result = self._adapter_bundle(
                        self._adapter_rows(connection, adapter_key), include_versions=True
                    )
                    assert result is not None
                    return result
            else:
                adapter_key = seeded["adapter_key"]

            current = self._adapter_bundle(self._adapter_rows(connection, adapter_key))
            assert current is not None
            latest = current["latest_version"]
            if latest["manifest_sha256"] == manifest_sha256:
                result = self._adapter_bundle(
                    self._adapter_rows(connection, adapter_key), include_versions=True
                )
                assert result is not None
                return result

            # Canonical seeds may advance seed or schema-migration output, but never
            # supersede a user-authored adapter version.
            if latest.get("created_by") not in {"__seed__", "__migration__"}:
                result = self._adapter_bundle(
                    self._adapter_rows(connection, adapter_key), include_versions=True
                )
                assert result is not None
                return result

            self._insert_adapter_version(
                connection,
                adapter_key=adapter_key,
                version_number=int(current["latest_version_number"]) + 1,
                name=normalized_name,
                manifest=manifest,
                description=description,
                repository_url=resolved_url,
                created_by="__seed__",
                change_note=change_note,
                archived_at=current["archived_at"],
                seed_key=normalized_seed,
                legacy_version=legacy_version,
            )
            result = self._adapter_bundle(
                self._adapter_rows(connection, adapter_key), include_versions=True
            )
        assert result is not None
        return result

    @staticmethod
    def _adapter_validation_payload(row: sqlite3.Row) -> dict[str, Any]:
        decoded = dict(row)
        return {
            "id": decoded["id"],
            "adapter_id": decoded["adapter_key"],
            "adapter_version_id": decoded["adapter_version_id"],
            "version_number": decoded.get("version_number"),
            "status": decoded["status"],
            "repository_url": decoded.get("repository_url"),
            "source_revision": decoded.get("source_revision"),
            "evidence": json.loads(decoded["evidence_json"]),
            "errors": json.loads(decoded["errors_json"]),
            "resolved_runtime": json.loads(decoded["resolved_runtime_json"]),
            "created_by": decoded.get("created_by"),
            "created_at": decoded["created_at"],
        }

    def record_adapter_validation(
        self,
        adapter_id: str,
        *,
        status: str,
        version_number: int | None = None,
        repository_url: str | None = None,
        source_revision: str | None = None,
        evidence: Mapping[str, Any] | None = None,
        errors: Sequence[Any] | None = None,
        resolved_runtime: Mapping[str, Any] | None = None,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        if self.workspace_id is not None and adapter_id is not None and not self.owns("adapters", adapter_id):
            raise KeyError("Record not found in this workspace")
        normalized_status = status.strip().upper()
        if normalized_status not in {"PASSED", "FAILED", "WARNING"}:
            raise ValueError("Validation status must be PASSED, FAILED, or WARNING")
        with self.transaction() as connection:
            adapter = self._adapter_bundle(
                self._adapter_rows(connection, adapter_id), selected_version=version_number
            )
            if adapter is None:
                raise KeyError(f"Adapter or version not found: {adapter_id}")
            selected = adapter["selected_version"]
            validation_id = new_id()
            self._insert(connection, "adapter_validations", {
                "id": validation_id,
                "adapter_key": adapter["id"],
                "adapter_version_id": selected["id"],
                "status": normalized_status,
                "repository_url": repository_url or selected.get("repository_url"),
                "source_revision": source_revision,
                "evidence_json": canonical_json(evidence or {}),
                "errors_json": canonical_json(list(errors or [])),
                "resolved_runtime_json": canonical_json(resolved_runtime or {}),
                "created_by": created_by,
                "created_at": utc_now(),
            })
            row = connection.execute(
                """
                SELECT v.*, a.version_number
                FROM adapter_validations v
                JOIN adapters a ON a.id = v.adapter_version_id
                WHERE v.id = ?
                """,
                (validation_id,),
            ).fetchone()
        assert row is not None
        return self._adapter_validation_payload(row)

    def list_adapter_validations(
        self,
        adapter_id: str,
        *,
        version_number: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if self.workspace_id is not None and adapter_id is not None and not self.owns("adapters", adapter_id):
            return []
        if limit < 1:
            raise ValueError("limit must be positive")
        with self.connection() as connection:
            rows = self._adapter_rows(connection, adapter_id)
            adapter = self._adapter_bundle(rows, selected_version=version_number)
            if adapter is None:
                raise KeyError(f"Adapter or version not found: {adapter_id}")
            parameters: list[Any] = [adapter["id"]]
            version_clause = ""
            if version_number is not None:
                version_clause = " AND a.version_number = ?"
                parameters.append(version_number)
            parameters.append(limit)
            validations = connection.execute(
                f"""
                SELECT v.*, a.version_number
                FROM adapter_validations v
                JOIN adapters a ON a.id = v.adapter_version_id
                WHERE v.adapter_key = ?{version_clause} AND {visible_sql('adapter_validations', 'v')}
                ORDER BY v.created_at DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            return [self._adapter_validation_payload(row) for row in validations]

    @classmethod
    def _public_data_resource(cls, row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
        result = cls._decode(row) if isinstance(row, sqlite3.Row) else dict(row)
        assert result is not None
        result["metadata"] = result.pop("metadata_json", {})
        return result

    @classmethod
    def _public_data_version(
        cls,
        connection: sqlite3.Connection,
        row: sqlite3.Row | Mapping[str, Any],
        *,
        include_resource: bool = False,
    ) -> dict[str, Any]:
        result = cls._decode(row) if isinstance(row, sqlite3.Row) else dict(row)
        assert result is not None
        result["metadata"] = result.pop("metadata_json", {})
        result["locations"] = [dict(item) for item in connection.execute("SELECT * FROM data_locations WHERE version_id=? ORDER BY kind, host", (result["id"],)).fetchall()]
        if include_resource:
            resource_row = connection.execute(
                "SELECT * FROM data_resources WHERE id = ?", (result["resource_id"],)
            ).fetchone()
            if resource_row is None:
                raise KeyError(f"Data resource not found: {result['resource_id']}")
            result["resource"] = cls._public_data_resource(resource_row)
        return result

    @classmethod
    def _data_resource_payload(
        cls,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        include_versions: bool,
    ) -> dict[str, Any]:
        result = cls._public_data_resource(row)
        version_rows = connection.execute(
            "SELECT * FROM data_resource_versions WHERE resource_id = ? "
            "ORDER BY created_at DESC, id DESC",
            (result["id"],),
        ).fetchall()
        versions = [
            cls._public_data_version(connection, version_row)
            for version_row in version_rows
        ]
        result["version_count"] = len(versions)
        result["latest_version"] = versions[0] if versions else None
        if include_versions:
            result["versions"] = versions
        return result

    def _insert_data_resource(
        self,
        connection: sqlite3.Connection,
        *,
        provider: str,
        namespace: str,
        name: str,
        kind: str,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        values = {
            "id": new_id(),
            "provider": provider,
            "namespace": namespace,
            "name": name,
            "kind": kind,
            "description": description,
            "metadata_json": canonical_json(dict(metadata or {})),
            "created_at": now,
            "updated_at": now,
            "archived_at": None,
        }
        self._insert(connection, "data_resources", values)
        row = connection.execute(
            "SELECT * FROM data_resources WHERE id = ?", (values["id"],)
        ).fetchone()
        assert row is not None
        return self._data_resource_payload(connection, row, include_versions=True)

    def create_data_resource(
        self,
        *,
        provider: str,
        namespace: str,
        name: str,
        kind: str,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            return self._insert_data_resource(
                connection, provider=provider, namespace=namespace, name=name,
                kind=kind, description=description, metadata=metadata,
            )

    def list_data_resources(
        self,
        *,
        provider: str | None = None,
        namespace: str | None = None,
        kind: str | None = None,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if not include_archived:
            clauses.append("archived_at IS NULL")
        for column, value in (("provider", provider), ("namespace", namespace), ("kind", kind)):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM data_resources {where} "
                "ORDER BY lower(provider), lower(namespace), lower(name)",
                parameters,
            ).fetchall()
            return [
                self._data_resource_payload(connection, row, include_versions=False)
                for row in rows
            ]

    def get_data_resource(self, resource_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM data_resources WHERE id = ?", (resource_id,)
            ).fetchone()
            return (
                self._data_resource_payload(connection, row, include_versions=True)
                if row is not None
                else None
            )

    def update_data_resource(
        self,
        resource_id: str,
        *,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {"updated_at": utc_now()}
        if description is not None:
            fields["description"] = description
        if metadata is not None:
            fields["metadata_json"] = canonical_json(dict(metadata))
        if archived is not None:
            fields["archived_at"] = utc_now() if archived else None
        with self.transaction() as connection:
            self._update(connection, "data_resources", resource_id, fields)
        result = self.get_data_resource(resource_id)
        assert result is not None
        return result

    def _insert_data_resource_version(
        self,
        connection: sqlite3.Connection,
        resource_id: str,
        *,
        revision: str,
        format: str,
        path: str,
        manifest_sha256: str,
        status: str = "READY",
        size_bytes: int | None = None,
        source_uri: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resource = connection.execute(
            "SELECT * FROM data_resources WHERE id = ?", (resource_id,)
        ).fetchone()
        if resource is None:
            raise KeyError(f"Data resource not found: {resource_id}")
        if resource["archived_at"] is not None:
            raise ValueError("cannot add a version to an archived data resource")
        values = {
            "id": new_id(),
            "resource_id": resource_id,
            "revision": revision,
            "format": format,
            "path": path,
            "source_uri": source_uri,
            "manifest_sha256": manifest_sha256.lower(),
            "status": status.upper(),
            "size_bytes": size_bytes,
            "metadata_json": canonical_json(dict(metadata or {})),
            "created_at": utc_now(),
        }
        self._insert(connection, "data_resource_versions", values)
        row = connection.execute(
            "SELECT * FROM data_resource_versions WHERE id = ?", (values["id"],)
        ).fetchone()
        assert row is not None
        return self._public_data_version(connection, row, include_resource=True)

    def create_data_resource_version(
        self,
        resource_id: str,
        *,
        revision: str,
        format: str,
        path: str,
        manifest_sha256: str,
        status: str = "READY",
        size_bytes: int | None = None,
        source_uri: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            return self._insert_data_resource_version(
                connection, resource_id, revision=revision, format=format, path=path,
                manifest_sha256=manifest_sha256, status=status, size_bytes=size_bytes,
                source_uri=source_uri, metadata=metadata,
            )

    def record_data_location(self, version_id, *, kind, host, path, manifest_sha256, status="AVAILABLE"):
        version = self.get_data_resource_version(version_id)
        if version is None or version["manifest_sha256"] != manifest_sha256:
            raise ValueError("Location must reference the exact registered dataset")
        with self.transaction() as connection:
            connection.execute("""INSERT INTO data_locations VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(version_id, host, path) DO UPDATE SET
                status=excluded.status, verified_at=excluded.verified_at""",
                (new_id(), version_id, kind, host, path, manifest_sha256, status, utc_now()))
            return dict(connection.execute("SELECT * FROM data_locations WHERE version_id=? AND host=? AND path=?", (version_id, host, path)).fetchone())

    def delete_prepared_dataset(self, resource_id, cleanup, *, identifier=None):
        """Delete a managed dataset only after its generated copies are removed.

        One write transaction prevents an experiment pin racing cleanup. Immutable
        delete triggers are restored in that same transaction; rollback restores
        them too. Ordinary registry operations cannot use this exception.
        """
        failure = None
        with self.transaction() as c:
            resource = c.execute(
                "SELECT * FROM data_resources WHERE id=?", (resource_id,)
            ).fetchone()
            if resource is None:
                raise KeyError("Dataset not found")
            if (resource["provider"], resource["namespace"]) != (
                "collection",
                "datasets",
            ):
                raise ValueError(
                    "Only prepared collection datasets can be deleted here"
                )
            versions = [
                dict(v)
                for v in c.execute(
                    "SELECT * FROM data_resource_versions WHERE resource_id=?",
                    (resource_id,),
                )
            ]
            jobs = [
                json.loads(r[0])
                for r in c.execute(
                    "SELECT payload_json FROM policy_exports WHERE json_extract(payload_json, '$.resource_id')=?",
                    (resource_id,),
                )
            ]
            if not json.loads(resource["metadata_json"]).get("managed_dataset"):
                # Older collections can predate the display metadata flag. The
                # preparation ledger and immutable version backlinks establish
                # ownership without trusting a label or widening file access.
                owned_versions = {
                    (job["id"], job.get("version_id")) for job in jobs
                }
                if not jobs or any(
                    version["format"] != "skynet.episodes/v1"
                    and (
                        json.loads(version["metadata_json"]).get("export_id"),
                        version["id"],
                    ) not in owned_versions
                    for version in versions
                ):
                    raise ValueError(
                        "Only prepared collection datasets can be deleted here"
                    )
            if identifier is not None:
                selected = next((j for j in jobs if j["id"] == identifier), None)
                if selected is None:
                    raise KeyError("Prepared format not found")
                version_id = selected.get("version_id")
                versions = [v for v in versions if v["id"] == version_id]
                jobs = [
                    j
                    for j in jobs
                    if j["id"] == identifier
                    or (version_id and j.get("version_id") == version_id)
                ]
            if any(
                j["state"] not in {"READY", "FAILED", "DELETE_FAILED"} for j in jobs
            ):
                raise ValueError(
                    "Wait for dataset preparation to finish before deleting it"
                )
            version_ids = {v["id"] for v in versions}
            checksums = {v["manifest_sha256"] for v in versions}
            for v in c.execute(
                "SELECT id, manifest_sha256 FROM data_resource_versions",
            ):
                if v["id"] not in version_ids and v["manifest_sha256"] in checksums:
                    raise ValueError(
                        "These files are shared with another registered dataset"
                    )
            bundles = []
            for row in c.execute(
                "SELECT DISTINCT b.* FROM data_bundles b JOIN data_bundle_assignments a ON a.bundle_id=b.id JOIN data_resource_versions v ON v.id=a.version_id WHERE v.resource_id=?",
                (resource_id,),
            ):
                assignments = list(
                    c.execute(
                        "SELECT version_id FROM data_bundle_assignments WHERE bundle_id=?",
                        (row["id"],),
                    )
                )
                if not any(a[0] in version_ids for a in assignments):
                    continue
                metadata = json.loads(row["metadata_json"])
                if (
                    not metadata.get("prepared_dataset")
                    or metadata.get("resource_id") != resource_id
                    or any(a[0] not in version_ids for a in assignments)
                ):
                    raise ValueError(
                        "This dataset is used by a bundle; remove that reference first"
                    )
                bundles.append(dict(row))
            references = (
                version_ids
                | checksums
                | ({resource_id} if identifier is None else set())
                | {b["id"] for b in bundles}
                | {b["manifest_sha256"] for b in bundles}
            )
            references.update(v["path"] for v in versions)
            locations = [
                dict(r)
                for r in c.execute(
                    "SELECT l.* FROM data_locations l JOIN data_resource_versions v ON v.id=l.version_id WHERE v.resource_id=?",
                    (resource_id,),
                )
            ]
            locations = [l for l in locations if l["version_id"] in version_ids]
            references.update(l["path"] for l in locations)

            # Check both requested and resolved inputs, including direct paths.
            def references_dataset(value):
                if isinstance(value, str):
                    return value in references or any(
                        value.startswith(path.rstrip("/") + "/")
                        for path in references
                        if path.startswith("/")
                    )
                if isinstance(value, dict):
                    return any(references_dataset(v) for v in value.values())
                return isinstance(value, list) and any(
                    references_dataset(v) for v in value
                )

            for table, column in [
                ("experiment_revisions", "requested_spec_json"),
                ("variants", "resolved_spec_json"),
            ]:
                for row in c.execute(f"SELECT {column} FROM {table}"):
                    if references_dataset(json.loads(row[0])):
                        raise ValueError(
                            "This dataset is used by an experiment and cannot be deleted"
                        )
            for row in c.execute(
                "SELECT i.input_version_id, d.output_version_id FROM data_derivation_inputs i JOIN data_derivations d ON d.id=i.derivation_id"
            ):
                if row[0] in version_ids and row[1] not in version_ids:
                    raise ValueError("Another dataset was derived from this dataset")
            if any(
                (identifier is None and row["resource_id"] == resource_id)
                or row["version_id"] in version_ids
                or row["bundle_id"] in {b["id"] for b in bundles}
                for row in c.execute(
                    "SELECT resource_id, version_id, bundle_id FROM data_imports"
                )
            ):
                raise ValueError("This dataset is referenced by an import job")
            try:
                cleanup(jobs, versions, locations)
            except Exception as exc:
                # Filesystem deletions cannot roll back. Keep records for retry,
                # but make every affected copy and bundle unavailable to training.
                failure = exc
                for job in jobs:
                    job.update(
                        state="DELETE_FAILED",
                        training_ready=False,
                        error=str(exc),
                        updated_at=utc_now(),
                    )
                    c.execute(
                        "UPDATE policy_exports SET payload_json=? WHERE id=?",
                        (canonical_json(job), job["id"]),
                    )
                for location in locations:
                    c.execute(
                        "UPDATE data_locations SET status='REMOVED' WHERE id=?",
                        (location["id"],),
                    )
                for bundle in bundles:
                    c.execute(
                        "UPDATE data_bundles SET archived_at=? WHERE id=?",
                        (utc_now(), bundle["id"]),
                    )
            else:
                trigger_names = [
                    f"{table}_no_delete"
                    for table in (
                        "data_resource_versions",
                        "data_derivations",
                        "data_derivation_inputs",
                        "data_bundle_assignments",
                    )
                ]
                triggers = [
                    c.execute(
                        "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
                        (name,),
                    ).fetchone()[0]
                    for name in trigger_names
                ]
                for name in trigger_names:
                    c.execute(f"DROP TRIGGER {name}")
                for bundle in bundles:
                    c.execute(
                        "DELETE FROM data_bundle_assignments WHERE bundle_id=?",
                        (bundle["id"],),
                    )
                    c.execute("DELETE FROM data_bundles WHERE id=?", (bundle["id"],))
                for v in versions:
                    c.execute(
                        "DELETE FROM data_derivation_inputs WHERE derivation_id IN (SELECT id FROM data_derivations WHERE output_version_id=?)",
                        (v["id"],),
                    )
                for v in versions:
                    c.execute(
                        "DELETE FROM data_derivations WHERE output_version_id=?",
                        (v["id"],),
                    )
                    c.execute(
                        "DELETE FROM data_locations WHERE version_id=?", (v["id"],)
                    )
                for version_id in version_ids:
                    c.execute(
                        "DELETE FROM data_resource_versions WHERE id=?", (version_id,)
                    )
                for job in jobs:
                    c.execute("DELETE FROM policy_exports WHERE id=?", (job["id"],))
                if identifier is None:
                    c.execute("DELETE FROM data_resources WHERE id=?", (resource_id,))
                for trigger in triggers:
                    c.execute(trigger)
        if failure:
            raise ValueError(
                f"Dataset deletion is incomplete. Retry Delete dataset. {failure}"
            ) from failure
        return {"deleted": True, "resource_id": resource_id}

    def data_version_usage(self, manifest_sha256):
        with self.connection() as connection:
            return [dict(row) for row in connection.execute("""
                SELECT DISTINCT e.id AS experiment_id, e.name, er.revision_number, r.id AS run_id, r.status AS run_status
                FROM experiment_revisions er JOIN experiments e ON e.id=er.experiment_id
                LEFT JOIN variants v ON v.experiment_revision_id=er.id
                LEFT JOIN runs r ON r.variant_id=v.id
                JOIN json_each(er.requested_spec_json, '$.data.bundle.assignments') assignment
                WHERE json_extract(assignment.value, '$.version.manifest_sha256')=?
                ORDER BY e.name, er.revision_number
            """, (manifest_sha256,)).fetchall()]

    def get_data_resource_version(self, version_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM data_resource_versions WHERE id = ?", (version_id,)
            ).fetchone()
            if row is None:
                return None
            result = self._public_data_version(connection, row, include_resource=True)
            derivation = connection.execute(
                "SELECT id FROM data_derivations WHERE output_version_id = ?", (version_id,)
            ).fetchone()
            result["derivation_id"] = derivation["id"] if derivation is not None else None
            result["used_by_bundles"] = self._decode_many(connection.execute(
                """
                SELECT b.id, b.name, b.version, a.role, a.position
                FROM data_bundle_assignments a
                JOIN data_bundles b ON b.id = a.bundle_id
                WHERE a.version_id = ?
                ORDER BY b.name, b.version, a.role, a.position
                """,
                (version_id,),
            ).fetchall())
            return result

    @classmethod
    def _public_data_import(cls, row: sqlite3.Row) -> dict[str, Any]:
        result = cls._decode(row)
        assert result is not None
        result["request"] = result.pop("request_json", {})
        result["result"] = result.pop("result_json", None)
        return result

    def create_data_import(
        self, resource_id: str, *, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        now = utc_now()
        with self.transaction() as connection:
            resource = connection.execute(
                "SELECT archived_at FROM data_resources WHERE id = ?", (resource_id,)
            ).fetchone()
            if resource is None:
                raise KeyError(f"Data resource not found: {resource_id}")
            if resource["archived_at"] is not None:
                raise ValueError("cannot import into an archived data resource")
            # Resource budgets and gateway choices do not change the immutable
            # output identity. Claim it in the same transaction as insertion.
            identity_fields = (
                "revision", "subset", "format", "role", "bundle_name", "bundle_version",
            )
            identity = tuple(request.get(key) for key in identity_fields)
            active = connection.execute(
                """SELECT request_json FROM data_imports WHERE resource_id = ?
                   AND state IN ('SUBMITTING', 'SUBMITTED', 'PENDING', 'RUNNING', 'FINALIZING', 'CANCELLING')""",
                (resource_id,),
            ).fetchall()
            for row in active:
                existing_request = json.loads(row["request_json"])
                if tuple(existing_request.get(key) for key in identity_fields) == identity:
                    raise ValueError("the same immutable Hugging Face import is already active")
            values = {
                "id": new_id(),
                "resource_id": resource_id,
                "state": "SUBMITTING",
                "request_json": canonical_json(dict(request)),
                "result_json": None,
                "gateway": None,
                "slurm_job_id": None,
                "slurm_state": None,
                "exit_code": None,
                "node_list": None,
                "run_directory": None,
                "script_path": None,
                "result_path": None,
                "stdout_path": None,
                "stderr_path": None,
                "version_id": None,
                "bundle_id": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
            self._insert(connection, "data_imports", values)
            row = connection.execute(
                "SELECT * FROM data_imports WHERE id = ?", (values["id"],)
            ).fetchone()
            assert row is not None
            return self._public_data_import(row)

    def update_data_import(
        self, import_id: str, *, expected_states: Sequence[str] | None = None, **changes: Any
    ) -> dict[str, Any] | None:
        """Update atomically; conditional callers receive None if another transition won."""
        allowed = {
            "state", "result", "gateway", "slurm_job_id", "slurm_state", "exit_code",
            "node_list", "run_directory", "script_path", "result_path", "stdout_path",
            "stderr_path", "version_id", "bundle_id", "error",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unsupported data import fields: {', '.join(sorted(unknown))}")
        fields = {key: value for key, value in changes.items() if key != "result"}
        if "result" in changes:
            fields["result_json"] = (
                canonical_json(dict(changes["result"])) if changes["result"] is not None else None
            )
        fields["updated_at"] = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM data_imports WHERE id = ?", (import_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Data import not found: {import_id}")
            if expected_states is not None and row["state"] not in expected_states:
                return None
            if row["state"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                return self._public_data_import(row)
            # Scheduler observations can lag a cancellation or finalization.
            # A conditional transition can explicitly roll back a failed cancel.
            if (
                expected_states is None
                and row["state"] in {"CANCELLING", "FINALIZING"}
                and fields.get("state") in {"SUBMITTING", "SUBMITTED", "PENDING", "RUNNING"}
            ):
                fields.pop("state")
            self._update(connection, "data_imports", import_id, fields)
            row = connection.execute(
                "SELECT * FROM data_imports WHERE id = ?", (import_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Data import not found: {import_id}")
            return self._public_data_import(row)

    def get_data_import(self, import_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM data_imports WHERE id = ?", (import_id,)
            ).fetchone()
            return self._public_data_import(row) if row is not None else None

    def list_data_imports(self, *, states: Sequence[str] | None = None) -> list[dict[str, Any]]:
        parameters: list[Any] = []
        where = ""
        if states:
            normalized = [str(state).upper() for state in states]
            where = f"WHERE state IN ({','.join('?' for _ in normalized)})"
            parameters.extend(normalized)
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM data_imports {where} ORDER BY created_at DESC", parameters
            ).fetchall()
            return [self._public_data_import(row) for row in rows]

    @classmethod
    def _data_derivation_payload(
        cls, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        result = cls._decode(row)
        assert result is not None
        result["converter_config"] = result.pop("converter_config_json", {})
        output_row = connection.execute(
            "SELECT * FROM data_resource_versions WHERE id = ?", (result["output_version_id"],)
        ).fetchone()
        assert output_row is not None
        result["output_version"] = cls._public_data_version(
            connection, output_row, include_resource=True
        )
        input_rows = connection.execute(
            """
            SELECT i.*, v.*
            FROM data_derivation_inputs i
            JOIN data_resource_versions v ON v.id = i.input_version_id
            WHERE i.derivation_id = ?
            ORDER BY i.role, i.position
            """,
            (result["id"],),
        ).fetchall()
        inputs: list[dict[str, Any]] = []
        for input_row in input_rows:
            version_row = connection.execute(
                "SELECT * FROM data_resource_versions WHERE id = ?", (input_row["input_version_id"],)
            ).fetchone()
            assert version_row is not None
            inputs.append({
                "version_id": input_row["input_version_id"],
                "role": input_row["role"],
                "position": input_row["position"],
                "version": cls._public_data_version(
                    connection, version_row, include_resource=True
                ),
            })
        result["inputs"] = inputs
        return result

    def create_data_derivation(
        self,
        *,
        output_version_id: str,
        inputs: Sequence[Mapping[str, Any]],
        converter_repository: str,
        converter_commit: str,
        converter_config: Mapping[str, Any] | None = None,
        runtime_lock_sha256: str | None = None,
    ) -> dict[str, Any]:
        if not inputs:
            raise ValueError("a data derivation requires at least one input")
        with self.transaction() as connection:
            if connection.execute(
                "SELECT 1 FROM data_resource_versions WHERE id = ?", (output_version_id,)
            ).fetchone() is None:
                raise KeyError(f"Data resource version not found: {output_version_id}")
            normalized_inputs: list[dict[str, Any]] = []
            next_positions: dict[str, int] = {}
            seen: set[tuple[str, int]] = set()
            for item in inputs:
                version_id = str(item["version_id"])
                role = str(item.get("role") or "input")
                position = item.get("position")
                if position is None:
                    position = next_positions.get(role, 0)
                position = int(position)
                next_positions[role] = max(next_positions.get(role, 0), position + 1)
                if (role, position) in seen:
                    raise ValueError(f"duplicate derivation input position: {role}[{position}]")
                seen.add((role, position))
                if version_id == output_version_id:
                    raise ValueError("a derivation output cannot also be its input")
                if connection.execute(
                    "SELECT 1 FROM data_resource_versions WHERE id = ?", (version_id,)
                ).fetchone() is None:
                    raise KeyError(f"Data resource version not found: {version_id}")
                cycle = connection.execute(
                    """
                    WITH RECURSIVE ancestors(version_id) AS (
                        SELECT ?
                        UNION
                        SELECT i.input_version_id
                        FROM ancestors a
                        JOIN data_derivations d ON d.output_version_id = a.version_id
                        JOIN data_derivation_inputs i ON i.derivation_id = d.id
                    )
                    SELECT 1 FROM ancestors WHERE version_id = ? LIMIT 1
                    """,
                    (version_id, output_version_id),
                ).fetchone()
                if cycle is not None:
                    raise ValueError("data derivation would create a lineage cycle")
                normalized_inputs.append({
                    "version_id": version_id,
                    "role": role,
                    "position": position,
                })
            values = {
                "id": new_id(),
                "output_version_id": output_version_id,
                "converter_repository": converter_repository,
                "converter_commit": converter_commit,
                "converter_config_json": canonical_json(dict(converter_config or {})),
                "runtime_lock_sha256": runtime_lock_sha256.lower() if runtime_lock_sha256 else None,
                "created_at": utc_now(),
            }
            self._insert(connection, "data_derivations", values)
            for item in normalized_inputs:
                self._insert(connection, "data_derivation_inputs", {
                    "derivation_id": values["id"],
                    "input_version_id": item["version_id"],
                    "role": item["role"],
                    "position": item["position"],
                })
            row = connection.execute(
                "SELECT * FROM data_derivations WHERE id = ?", (values["id"],)
            ).fetchone()
            assert row is not None
            return self._data_derivation_payload(connection, row)

    def list_data_derivations(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM data_derivations ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._data_derivation_payload(connection, row) for row in rows]

    def get_data_derivation(self, derivation_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM data_derivations WHERE id = ?", (derivation_id,)
            ).fetchone()
            return self._data_derivation_payload(connection, row) if row is not None else None

    @classmethod
    def _data_bundle_payload(
        cls, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        result = cls._decode(row)
        assert result is not None
        result["metadata"] = result.pop("metadata_json", {})
        result["manifest"] = result.pop("manifest_json")
        assignment_rows = connection.execute(
            "SELECT * FROM data_bundle_assignments WHERE bundle_id = ? "
            "ORDER BY role, position",
            (result["id"],),
        ).fetchall()
        assignments: list[dict[str, Any]] = []
        for assignment_row in assignment_rows:
            assignment = cls._decode(assignment_row)
            assert assignment is not None
            assignment["config"] = assignment.pop("config_json", {})
            frozen = next((item for item in result["manifest"]["assignments"] if item["role"] == assignment["role"] and item["position"] == assignment["position"]), None)
            if frozen is not None:
                assignment["config"] = frozen.get("config", {})
            version_row = connection.execute(
                "SELECT * FROM data_resource_versions WHERE id = ?", (assignment["version_id"],)
            ).fetchone()
            assert version_row is not None
            assignment["version"] = cls._public_data_version(
                connection, version_row, include_resource=True
            )
            assignments.append(assignment)
        result["assignments"] = assignments
        result["assignment_count"] = len(assignments)
        return result

    @classmethod
    def _bundle_manifest_assignment(
        cls,
        connection: sqlite3.Connection,
        assignment: Mapping[str, Any],
    ) -> dict[str, Any]:
        version_row = connection.execute(
            "SELECT * FROM data_resource_versions WHERE id = ?", (assignment["version_id"],)
        ).fetchone()
        if version_row is None:
            raise KeyError(f"Data resource version not found: {assignment['version_id']}")
        version = cls._public_data_version(connection, version_row, include_resource=True)
        resource = version.pop("resource")
        config = dict(assignment.get("config") or {})
        if config.get("location_id"):
            location = connection.execute("SELECT * FROM data_locations WHERE id=? AND version_id=?", (config["location_id"], assignment["version_id"])).fetchone()
            if location is None or location["status"] != "AVAILABLE" or location["kind"] != "cluster":
                raise ValueError("Dataset copy is not verified on the training cluster")
            config["location"] = dict(location)
        elif config.get("location"):
            raise ValueError("Choose a registered location_id; a location receipt cannot be supplied manually")
        return {
            "role": assignment["role"],
            "position": assignment["position"],
            "mount_path": assignment.get("mount_path"),
            "required": bool(assignment.get("required", True)),
            "config": config,
            "resource": {
                key: resource[key]
                for key in ("provider", "namespace", "name", "kind")
            },
            "version": {
                key: version[key]
                for key in (
                    "revision", "format", "path", "source_uri", "manifest_sha256",
                    "status", "size_bytes", "metadata",
                )
            },
        }

    def create_data_bundle(
        self,
        *,
        name: str,
        version: str,
        assignments: Sequence[Mapping[str, Any]],
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not assignments:
            raise ValueError("a data bundle requires at least one role assignment")
        with self.transaction() as connection:
            normalized: list[dict[str, Any]] = []
            next_positions: dict[str, int] = {}
            seen: set[tuple[str, int]] = set()
            for item in assignments:
                role = str(item["role"])
                position = item.get("position")
                if position is None:
                    position = next_positions.get(role, 0)
                position = int(position)
                next_positions[role] = max(next_positions.get(role, 0), position + 1)
                if (role, position) in seen:
                    raise ValueError(f"duplicate bundle role position: {role}[{position}]")
                seen.add((role, position))
                normalized.append({
                    "role": role,
                    "position": position,
                    "version_id": str(item["version_id"]),
                    "mount_path": validate_mount_path(item.get("mount_path")),
                    "required": bool(item.get("required", True)),
                    "config": dict(item.get("config") or {}),
                })
            normalized.sort(key=lambda item: (item["role"], item["position"]))
            manifest = {
                "schema_version": "skynet.data-bundle/v1",
                "name": name,
                "version": version,
                "metadata": dict(metadata or {}),
                "assignments": [
                    self._bundle_manifest_assignment(connection, item) for item in normalized
                ],
            }
            manifest_sha256 = content_sha256(manifest)
            values = {
                "id": new_id(),
                "name": name,
                "version": version,
                "description": description,
                "manifest_json": canonical_json(manifest),
                "manifest_sha256": manifest_sha256,
                "metadata_json": canonical_json(dict(metadata or {})),
                "created_at": utc_now(),
                "archived_at": None,
            }
            self._insert(connection, "data_bundles", values)
            for item in normalized:
                self._insert(connection, "data_bundle_assignments", {
                    "bundle_id": values["id"],
                    "role": item["role"],
                    "position": item["position"],
                    "version_id": item["version_id"],
                    "mount_path": item["mount_path"],
                    "required": int(item["required"]),
                    "config_json": canonical_json(item["config"]),
                })
            row = connection.execute(
                "SELECT * FROM data_bundles WHERE id = ?", (values["id"],)
            ).fetchone()
            assert row is not None
            return self._data_bundle_payload(connection, row)

    def list_data_bundles(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        where = "" if include_archived else "WHERE archived_at IS NULL"
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM data_bundles {where} ORDER BY lower(name), version"
            ).fetchall()
            return [self._data_bundle_payload(connection, row) for row in rows]

    def get_data_bundle(self, bundle_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM data_bundles WHERE id = ?", (bundle_id,)
            ).fetchone()
            return self._data_bundle_payload(connection, row) if row is not None else None

    def archive_data_bundle(self, bundle_id: str) -> dict[str, Any]:
        with self.transaction() as connection:
            self._update(
                connection,
                "data_bundles",
                bundle_id,
                {"archived_at": utc_now()},
            )
        result = self.get_data_bundle(bundle_id)
        assert result is not None
        return result

    def data_bundle_snapshot(
        self, bundle_id: str, *, require_active: bool = False
    ) -> dict[str, Any]:
        bundle = self.get_data_bundle(bundle_id)
        if bundle is None:
            raise KeyError(f"Data bundle not found: {bundle_id}")
        if require_active and bundle["archived_at"] is not None:
            raise ValueError(f"data bundle is archived: {bundle_id}")
        return {
            "id": bundle["id"],
            **bundle["manifest"],
            "manifest_sha256": bundle["manifest_sha256"],
        }

    def register_adapter(
        self,
        *,
        name: str,
        version: str,
        capabilities: Mapping[str, Any],
        schema: Mapping[str, Any],
        repository_url: str | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        """Compatibility seed registration; never overwrites user-created versions."""
        manifest = {
            "schema_version": "skynet.adapter/v1",
            "legacy_adapter_version": version,
            "repository": {"url": repository_url},
            "capabilities": dict(capabilities),
            "parameter_schema": dict(schema),
        }
        adapter = self.upsert_seed_adapter(
            seed_key=f"builtin:{name}",
            name=name,
            manifest=manifest,
            repository_url=repository_url,
            legacy_version=version,
        )
        if not enabled and adapter["archived_at"] is None:
            adapter = self.archive_adapter(adapter["id"])
        version_id = adapter["latest_version"]["id"]
        with self.connection() as connection:
            result = self._row_by_id(connection, "adapters", version_id)
        assert result is not None
        return result

    def list_adapters(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        """Compatibility flat version listing; prefer list_adapter_registry for UI/API use."""
        where = f"WHERE {visible_sql('adapters')}" + (" AND enabled = 1" if enabled_only else "")
        with self.connection() as connection:
            return self._decode_many(connection.execute(
                f"SELECT * FROM adapters {where} ORDER BY name, version_number"
            ).fetchall())


__all__ = [
    "Database",
    "DEFAULT_DATABASE_PATH",
    "canonical_json",
    "content_sha256",
    "new_id",
    "utc_now",
]
