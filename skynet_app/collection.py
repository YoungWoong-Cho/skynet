from __future__ import annotations

import base64
import json
import re
import shlex
import sqlite3
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .cluster_config import CLUSTER
from .cluster_runtime import ClusterClient, ClusterError, HOME_ROOT, WORK_ROOT
from .database import Database, canonical_json, content_sha256, new_id, utc_now
from .experiments import format_slurm_duration, parse_slurm_duration


COLLECTION_SCHEMA_VERSION = "skynet.collection/v1"
ADAPTER_SCHEMA_VERSION = "skynet.collection-adapter/v1"
ADAPTER_SEED_ROOT = Path(__file__).resolve().parent.parent / "config" / "collection_adapters"

SESSION_STATES = (
    "DRAFT",
    "PREFLIGHTED",
    "PREFLIGHT_FAILED",
    "READY",
    "SUBMITTING",
    "SUBMITTED",
    "PENDING",
    "RUNNING",
    "CAPTURED",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
)

COLLECTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS collection_adapters (
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

CREATE TABLE IF NOT EXISTS collection_sessions (
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
    capture_snapshot_json TEXT NOT NULL,
    capabilities_snapshot_json TEXT NOT NULL,
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
);

CREATE TABLE IF NOT EXISTS collection_session_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES collection_sessions(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    old_status TEXT,
    new_status TEXT,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_collection_adapters_state
ON collection_adapters(archived_at, adapter_key);

CREATE INDEX IF NOT EXISTS idx_collection_sessions_state
ON collection_sessions(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_collection_sessions_adapter
ON collection_sessions(adapter_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_collection_sessions_job
ON collection_sessions(slurm_job_id);

CREATE INDEX IF NOT EXISTS idx_collection_events_session
ON collection_session_events(session_id, created_at);

CREATE TRIGGER IF NOT EXISTS collection_session_snapshots_immutable
BEFORE UPDATE OF
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
    restart_of_session_id
ON collection_sessions
BEGIN
    SELECT RAISE(ABORT, 'collection session snapshots are immutable');
END;
"""


class CollectionValidationError(ValueError):
    pass


class StreamDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    kind: Literal["action", "proprio", "sensor"]
    shape: list[int] | None
    dynamic: bool = False
    dtype: str = Field(min_length=1, max_length=128)
    units: str | None = Field(default=None, max_length=128)
    frame: str | None = Field(default=None, max_length=256)
    rate_hz: float | None = Field(default=None, gt=0)
    native_key: str = Field(min_length=1, max_length=512)
    selector: dict[str, Any] = Field(default_factory=dict)
    required: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and any(dimension < 1 for dimension in value):
            raise ValueError("stream dimensions must be positive integers")
        return value


class CollectionRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal[
        "config",
        "software",
        "calibration",
        "storage",
        "resources",
        "capture",
        "capabilities",
    ]
    key: str = Field(pattern=r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$")
    description: str = Field(min_length=1, max_length=1000)
    required: bool = True
    blocking: bool = True
    equals: str | int | float | bool | None = None


class EnvironmentOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Z_][A-Z0-9_]*$")
    operation: Literal["set", "prepend", "append", "unset"] = "set"
    value: str | None = None
    separator: str = Field(default=":", min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_value(self) -> "EnvironmentOperation":
        if self.operation == "unset":
            if self.value is not None:
                raise ValueError("unset environment operations cannot declare a value")
        elif self.value is None:
            raise ValueError(f"{self.operation} environment operations require a value")
        if any(character in self.separator for character in "\n\r$`{}"):
            raise ValueError("environment separator contains unsafe shell syntax")
        return self


class CollectionCapabilityDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{1,127}$")
    kind: Literal[
        "container_runtime",
        "container_daemon",
        "gpu",
        "network_ports",
        "gui",
        "stream_endpoint",
        "declared",
    ]
    scope: Literal["submission_host", "compute_node", "client", "network", "operator"]
    description: str = Field(min_length=1, max_length=2000)
    required: bool = True
    blocking: bool = True
    default_status: Literal["UNKNOWN", "ADMIN_REQUIRED"] = "UNKNOWN"
    tcp_ports: list[int] = Field(default_factory=list)
    udp_ports: list[int] = Field(default_factory=list)

    @field_validator("tcp_ports", "udp_ports")
    @classmethod
    def validate_ports(cls, value: list[int]) -> list[int]:
        if any(port < 1 or port > 65535 for port in value):
            raise ValueError("network ports must be between 1 and 65535")
        if len(value) != len(set(value)):
            raise ValueError("network ports must be unique")
        return value


class CollectionLauncherStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
    argv: list[str] = Field(min_length=1)
    cwd: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    environment_operations: list[EnvironmentOperation] = Field(default_factory=list)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: list[str]) -> list[str]:
        if any(not item or "\x00" in item or "\n" in item for item in value):
            raise ValueError("launcher argv entries must be non-empty single-line strings")
        return value

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [key for key in value if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key)]
        if invalid:
            raise ValueError(f"invalid environment variable: {invalid[0]}")
        return value


class CollectionLauncher(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[CollectionLauncherStep] = Field(min_length=1)
    operator_steps: list[CollectionLauncherStep] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    environment_operations: list[EnvironmentOperation] = Field(default_factory=list)

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [key for key in value if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key)]
        if invalid:
            raise ValueError(f"invalid environment variable: {invalid[0]}")
        return value


class CollectionAdapterManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[ADAPTER_SCHEMA_VERSION] = ADAPTER_SCHEMA_VERSION
    key: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")
    display_name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=4000)
    runnable: bool = True
    streams: list[StreamDeclaration] = Field(min_length=1)
    requirements: list[CollectionRequirement] = Field(default_factory=list)
    capabilities: list[CollectionCapabilityDeclaration] = Field(default_factory=list)
    launcher: CollectionLauncher
    defaults: dict[str, Any] = Field(default_factory=dict)
    todos: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_stream_catalog(self) -> "CollectionAdapterManifest":
        names = [stream.name for stream in self.streams]
        if len(names) != len(set(names)):
            raise ValueError("stream names must be unique")
        stream_sources = [
            (stream.native_key, canonical_json(stream.selector)) for stream in self.streams
        ]
        if len(stream_sources) != len(set(stream_sources)):
            raise ValueError("stream native_key and selector pairs must be unique")
        capability_ids = [capability.id for capability in self.capabilities]
        if len(capability_ids) != len(set(capability_ids)):
            raise ValueError("capability declaration IDs must be unique")
        if self.runnable:
            unresolved = [
                stream.name
                for stream in self.streams
                if stream.shape is None and not stream.dynamic
                or stream.dtype.lower() in {"unknown", "todo"}
                or "todo" in stream.native_key.lower()
            ]
            if unresolved:
                raise ValueError(
                    "runnable adapters require resolved shape, dtype, and native_key for: "
                    + ", ".join(unresolved)
                )
        return self


class DataRegistrationTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=128)
    namespace: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    kind: str = Field(default="demonstrations", min_length=1, max_length=128)


class CollectionStorage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: str = Field(min_length=1, max_length=4096)
    native_format: str = Field(min_length=1, max_length=128)
    registration: DataRegistrationTarget | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.lower()


class CollectionCaptureMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_name: str = Field(min_length=1, max_length=256)
    schema_version: str = Field(min_length=1, max_length=128)
    clock_source: str = Field(min_length=1, max_length=256)
    nominal_rate_hz: float = Field(gt=0)
    timestamp_unit: str = Field(min_length=1, max_length=64)
    alignment: Literal[
        "action_then_post_state",
        "observation_then_action",
        "synchronous",
        "adapter_defined",
    ]
    timestamps_recorded: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionCapabilityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["verified", "unavailable", "unknown", "admin_required"]
    scope: Literal["submission_host", "compute_node", "client", "network", "operator"]
    verified_by: str | None = Field(default=None, max_length=512)
    checked_at: str | None = Field(default=None, max_length=128)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_verification_identity(self) -> "CollectionCapabilityEvidence":
        if self.status == "verified" and (not self.verified_by or not self.checked_at):
            raise ValueError("verified capability evidence requires verified_by and checked_at")
        return self


class CollectionResources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gateway: str = "auto"
    account: str = "rl2-lab"
    partition: str = "rl2-lab"
    gpu_count: int = Field(default=1, ge=1, le=16)
    gpu_type: str | None = Field(default="l40s", pattern=r"^[A-Za-z0-9_.-]+$")
    cpu_count: int = Field(default=16, ge=1, le=256)
    memory_gb: int = Field(default=64, ge=1, le=2048)
    time_limit: str = "04:00:00"
    node: str | None = None

    @field_validator("gateway")
    @classmethod
    def validate_gateway(cls, value: str) -> str:
        if value not in {"auto", *CLUSTER.gateways}:
            raise ValueError("gateway is not configured")
        return value

    @field_validator("time_limit")
    @classmethod
    def validate_time_limit(cls, value: str) -> str:
        return format_slurm_duration(parse_slurm_duration(value))

    @field_validator("node")
    @classmethod
    def validate_node(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9_.\[\],-]+", value):
            raise ValueError("invalid Slurm node expression")
        return value

    @model_validator(mode="after")
    def validate_queue(self) -> "CollectionResources":
        queue = CLUSTER.queue_for_partition(self.partition)
        if self.account != queue.account:
            raise ValueError(
                f"partition {self.partition!r} requires account {queue.account!r}"
            )
        seconds = parse_slurm_duration(self.time_limit)
        if seconds > queue.max_time_seconds:
            raise ValueError(
                f"partition is limited to {format_slurm_duration(queue.max_time_seconds)}"
            )
        return self


def _reject_plaintext_secrets(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            sensitive = re.search(r"(?:token|password|secret|api[_-]?key)", key_text, re.I)
            reference = re.search(r"(?:_ref|_refs|reference|references)$", key_text, re.I)
            if sensitive and not reference and child not in (None, "", False):
                raise ValueError(
                    f"plaintext secret field {child_path!r} is forbidden; store a secret reference"
                )
            _reject_plaintext_secrets(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_plaintext_secrets(child, f"{path}[{index}]")


class CollectionSessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=128)
    runtime_profile: str | None = Field(
        default=None, pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$"
    )
    config: dict[str, Any] = Field(default_factory=dict)
    software: dict[str, Any] = Field(default_factory=dict)
    calibration: CollectionCalibration
    storage: CollectionStorage
    resources: CollectionResources = Field(default_factory=CollectionResources)
    capture: CollectionCaptureMetadata
    capabilities: dict[str, CollectionCapabilityEvidence] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_secrets(self) -> "CollectionSessionCreate":
        for scope, value in (
            ("config", self.config),
            ("software", self.software),
            ("calibration", self.calibration.model_dump(mode="python")),
            ("storage", self.storage.model_dump(mode="python")),
            ("capture", self.capture.model_dump(mode="python")),
            (
                "capabilities",
                {
                    key: evidence.model_dump(mode="python")
                    for key, evidence in self.capabilities.items()
                },
            ),
        ):
            _reject_plaintext_secrets(value, scope)
        return self


class CollectionGatewayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gateway: str | None = None
    remote_validate: bool = True


class CollectionCompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    revision: str | None = Field(default=None, min_length=1, max_length=512)
    size_bytes: int | None = Field(default=None, ge=0)
    resource_metadata: dict[str, Any] = Field(default_factory=dict)
    version_metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionRestartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=128)


class CollectionErrorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(default="COLLECTION_ERROR", min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    details: dict[str, Any] = Field(default_factory=dict)


class CompiledCollectionJob(BaseModel):
    script: str
    script_sha256: str
    stdout_path: str
    stderr_path: str


class CollectionStore:
    def __init__(self, database: Database) -> None:
        self.database = database
        with self.database.transaction() as connection:
            connection.executescript(COLLECTION_SCHEMA)
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(collection_sessions)"
                ).fetchall()
            }
            for column in ("capture_snapshot_json", "capabilities_snapshot_json"):
                if column not in columns:
                    connection.execute(
                        f"ALTER TABLE collection_sessions ADD COLUMN {column} "
                        "TEXT NOT NULL DEFAULT '{}'"
                    )
            connection.execute(
                "DROP TRIGGER IF EXISTS collection_session_snapshots_immutable"
            )
            connection.executescript(
                """
                CREATE TRIGGER collection_session_snapshots_immutable
                BEFORE UPDATE OF
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
                    restart_of_session_id
                ON collection_sessions
                BEGIN
                    SELECT RAISE(ABORT, 'collection session snapshots are immutable');
                END;
                """
            )

    @staticmethod
    def _adapter(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["manifest"] = json.loads(result.pop("manifest_json"))
        return result

    @staticmethod
    def _session(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for column in (
            "adapter_snapshot_json",
            "config_snapshot_json",
            "software_snapshot_json",
            "calibration_snapshot_json",
            "storage_snapshot_json",
            "resources_snapshot_json",
            "capture_snapshot_json",
            "capabilities_snapshot_json",
            "canonical_manifest_json",
            "error_json",
        ):
            value = result.pop(column)
            result[column.removesuffix("_json")] = json.loads(value) if value else None
        return result

    @staticmethod
    def _event(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["details"] = json.loads(result.pop("details_json"))
        return result

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        session_id: str,
        event_type: str,
        old_status: str | None,
        new_status: str | None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        connection.execute(
            "INSERT INTO collection_session_events "
            "(id, session_id, event_type, old_status, new_status, details_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                new_id(),
                session_id,
                event_type,
                old_status,
                new_status,
                canonical_json(dict(details or {})),
                utc_now(),
            ),
        )

    def create_adapter(self, manifest: CollectionAdapterManifest) -> dict[str, Any]:
        payload = manifest.model_dump(mode="json")
        now = utc_now()
        adapter_id = new_id()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO collection_adapters "
                "(id, adapter_key, display_name, description, manifest_json, "
                "manifest_sha256, created_at, updated_at, archived_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    adapter_id,
                    manifest.key,
                    manifest.display_name,
                    manifest.description,
                    canonical_json(payload),
                    content_sha256(payload),
                    now,
                    now,
                ),
            )
        result = self.get_adapter(adapter_id)
        assert result is not None
        return result

    def update_adapter(
        self, adapter_id: str, manifest: CollectionAdapterManifest
    ) -> dict[str, Any]:
        payload = manifest.model_dump(mode="json")
        with self.database.transaction() as connection:
            current = connection.execute(
                "SELECT * FROM collection_adapters WHERE id = ?", (adapter_id,)
            ).fetchone()
            if current is None:
                raise KeyError("Collection adapter not found")
            if current["archived_at"] is not None:
                raise ValueError("restore the collection adapter before editing it")
            if current["adapter_key"] != manifest.key:
                raise ValueError("collection adapter key is immutable")
            connection.execute(
                "UPDATE collection_adapters SET display_name = ?, description = ?, "
                "manifest_json = ?, manifest_sha256 = ?, updated_at = ? WHERE id = ?",
                (
                    manifest.display_name,
                    manifest.description,
                    canonical_json(payload),
                    content_sha256(payload),
                    utc_now(),
                    adapter_id,
                ),
            )
        result = self.get_adapter(adapter_id)
        assert result is not None
        return result

    def get_adapter(self, adapter_id: str) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM collection_adapters WHERE id = ?", (adapter_id,)
            ).fetchone()
            if row is None:
                return None
            result = self._adapter(row)
            result["session_count"] = connection.execute(
                "SELECT count(*) FROM collection_sessions WHERE adapter_id = ?",
                (adapter_id,),
            ).fetchone()[0]
            return result

    def adapter_by_key(self, adapter_key: str) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM collection_adapters WHERE adapter_key = ?", (adapter_key,)
            ).fetchone()
            return self._adapter(row) if row is not None else None

    def list_adapters(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        where = "" if include_archived else "WHERE archived_at IS NULL"
        with self.database.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM collection_adapters {where} ORDER BY lower(display_name)"
            ).fetchall()
            return [self._adapter(row) for row in rows]

    def set_adapter_archived(self, adapter_id: str, archived: bool) -> dict[str, Any]:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE collection_adapters SET archived_at = ?, updated_at = ? WHERE id = ?",
                (utc_now() if archived else None, utc_now(), adapter_id),
            )
            if cursor.rowcount != 1:
                raise KeyError("Collection adapter not found")
        result = self.get_adapter(adapter_id)
        assert result is not None
        return result

    @staticmethod
    def bundled_templates(root: Path = ADAPTER_SEED_ROOT) -> list[dict[str, Any]]:
        """Expose shipped definitions for review without replacing operator edits."""
        if not root.is_dir():
            return []
        templates = []
        for path in sorted(root.glob("*.json")):
            manifest = CollectionAdapterManifest.model_validate_json(path.read_text())
            payload = manifest.model_dump(mode="json")
            templates.append({"manifest": payload, "manifest_sha256": content_sha256(payload)})
        return templates

    def seed_from_directory(self, root: Path = ADAPTER_SEED_ROOT) -> list[dict[str, Any]]:
        seeded: list[dict[str, Any]] = []
        if not root.is_dir():
            return seeded
        for path in sorted(root.glob("*.json")):
            manifest = CollectionAdapterManifest.model_validate_json(path.read_text())
            existing = self.adapter_by_key(manifest.key)
            if existing is None:
                existing = self.create_adapter(manifest)
            seeded.append(existing)
        return seeded

    def _insert_session(
        self,
        *,
        adapter_id: str,
        name: str,
        adapter_snapshot: Mapping[str, Any],
        config: Mapping[str, Any],
        software: Mapping[str, Any],
        calibration: Mapping[str, Any],
        storage: Mapping[str, Any],
        resources: Mapping[str, Any],
        capture: Mapping[str, Any],
        capabilities: Mapping[str, Any],
        restart_of_session_id: str | None = None,
    ) -> dict[str, Any]:
        session_id = new_id()
        canonical_manifest = {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "session_id": session_id,
            "name": name,
            "adapter": dict(adapter_snapshot),
            "config": dict(config),
            "software": dict(software),
            "calibration": dict(calibration),
            "storage": dict(storage),
            "resources": dict(resources),
            "capture": dict(capture),
            "capabilities": dict(capabilities),
            "restart_of_session_id": restart_of_session_id,
        }
        manifest_sha256 = content_sha256(canonical_manifest)
        now = utc_now()
        log_prefix = f"{CLUSTER.paths.logs}/collection-{session_id}-%j"
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO collection_sessions ("
                "id, adapter_id, name, status, adapter_snapshot_json, config_snapshot_json, "
                "software_snapshot_json, calibration_snapshot_json, storage_snapshot_json, "
                "resources_snapshot_json, capture_snapshot_json, capabilities_snapshot_json, "
                "canonical_manifest_json, manifest_sha256, "
                "stdout_path, stderr_path, error_json, restart_of_session_id, created_at, updated_at"
                ") VALUES (?, ?, ?, 'DRAFT', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)",
                (
                    session_id,
                    adapter_id,
                    name,
                    canonical_json(adapter_snapshot),
                    canonical_json(config),
                    canonical_json(software),
                    canonical_json(calibration),
                    canonical_json(storage),
                    canonical_json(resources),
                    canonical_json(capture),
                    canonical_json(capabilities),
                    canonical_json(canonical_manifest),
                    manifest_sha256,
                    f"{log_prefix}.out",
                    f"{log_prefix}.err",
                    restart_of_session_id,
                    now,
                    now,
                ),
            )
            self._insert_event(
                connection,
                session_id,
                "SESSION_CREATED",
                None,
                "DRAFT",
                {"manifest_sha256": manifest_sha256},
            )
        result = self.get_session(session_id)
        assert result is not None
        return result

    def create_session(self, request: CollectionSessionCreate) -> dict[str, Any]:
        adapter = self.get_adapter(request.adapter_id)
        if adapter is None:
            raise KeyError("Collection adapter not found")
        if adapter["archived_at"] is not None:
            raise ValueError("cannot create a session from an archived adapter")
        adapter_manifest = CollectionAdapterManifest.model_validate(adapter["manifest"])
        declared_capabilities = {
            capability.id for capability in adapter_manifest.capabilities
        }
        unknown_capabilities = set(request.capabilities) - declared_capabilities
        if unknown_capabilities:
            raise ValueError(
                "undeclared capability evidence: "
                + ", ".join(sorted(unknown_capabilities))
            )
        adapter_snapshot = {
            "id": adapter["id"],
            "key": adapter["adapter_key"],
            "manifest_sha256": adapter["manifest_sha256"],
            "manifest": adapter["manifest"],
        }
        software = dict(request.software)
        if "runtime_profile" in software:
            raise ValueError("software.runtime_profile is reserved; select runtime_profile")
        if request.runtime_profile:
            software["runtime_profile"] = CLUSTER.runtime_profile_snapshot(
                request.runtime_profile
            )
        return self._insert_session(
            adapter_id=adapter["id"],
            name=request.name,
            adapter_snapshot=adapter_snapshot,
            config=request.config,
            software=software,
            calibration=request.calibration.model_dump(mode="json"),
            storage=request.storage.model_dump(mode="json"),
            resources=request.resources.model_dump(mode="json"),
            capture=request.capture.model_dump(mode="json"),
            capabilities={
                key: evidence.model_dump(mode="json")
                for key, evidence in request.capabilities.items()
            },
        )

    def restart_session(self, session_id: str, *, name: str | None = None) -> dict[str, Any]:
        original = self.get_session(session_id)
        if original is None:
            raise KeyError("Collection session not found")
        return self._insert_session(
            adapter_id=original["adapter_id"],
            name=name or f"{original['name']}-restart",
            adapter_snapshot=original["adapter_snapshot"],
            config=original["config_snapshot"],
            software=original["software_snapshot"],
            calibration=original["calibration_snapshot"],
            storage=original["storage_snapshot"],
            resources=original["resources_snapshot"],
            capture=original["capture_snapshot"],
            capabilities=original["capabilities_snapshot"],
            restart_of_session_id=session_id,
        )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM collection_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            result = self._session(row)
            result["events"] = [
                self._event(event)
                for event in connection.execute(
                    "SELECT * FROM collection_session_events WHERE session_id = ? "
                    "ORDER BY created_at, id",
                    (session_id,),
                ).fetchall()
            ]
            return result

    def list_sessions(
        self,
        *,
        adapter_id: str | None = None,
        status: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if adapter_id:
            clauses.append("adapter_id = ?")
            parameters.append(adapter_id)
        if status:
            if status.upper() not in SESSION_STATES:
                raise ValueError("invalid collection session status")
            clauses.append("status = ?")
            parameters.append(status.upper())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(max(1, min(limit, 10000)))
        with self.database.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM collection_sessions {where} "
                "ORDER BY created_at DESC LIMIT ?",
                parameters,
            ).fetchall()
            return [self._session(row) for row in rows]

    def record_preflight(
        self, session_id: str, report: Mapping[str, Any]
    ) -> dict[str, Any]:
        new_status = "PREFLIGHTED" if report.get("ok") else "PREFLIGHT_FAILED"
        now = utc_now()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT status FROM collection_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Collection session not found")
            if row["status"] not in {"DRAFT", "PREFLIGHTED", "PREFLIGHT_FAILED", "READY"}:
                raise ValueError(f"cannot preflight a session in {row['status']} state")
            error = None if report.get("ok") else canonical_json({"preflight": report})
            connection.execute(
                "UPDATE collection_sessions SET status = ?, error_json = ?, "
                "preflighted_at = ?, updated_at = ? WHERE id = ?",
                (new_status, error, now, now, session_id),
            )
            self._insert_event(
                connection,
                session_id,
                "PREFLIGHT_COMPLETED",
                row["status"],
                new_status,
                report,
            )
        result = self.get_session(session_id)
        assert result is not None
        return result

    def record_prepared(
        self,
        session_id: str,
        compiled: CompiledCollectionJob,
        *,
        gateway: str,
        validation_output: str,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT status FROM collection_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Collection session not found")
            if row["status"] not in {"PREFLIGHTED", "READY"}:
                raise ValueError(f"cannot prepare a session in {row['status']} state")
            connection.execute(
                "UPDATE collection_sessions SET status = 'READY', sbatch_text = ?, "
                "sbatch_sha256 = ?, gateway = ?, error_json = NULL, prepared_at = ?, "
                "updated_at = ? WHERE id = ?",
                (
                    compiled.script,
                    compiled.script_sha256,
                    gateway,
                    now,
                    now,
                    session_id,
                ),
            )
            self._insert_event(
                connection,
                session_id,
                "SESSION_PREPARED",
                row["status"],
                "READY",
                {
                    "sbatch_sha256": compiled.script_sha256,
                    "gateway": gateway,
                    "validation_output": validation_output,
                },
            )
        result = self.get_session(session_id)
        assert result is not None
        return result

    def claim_submission(self, session_id: str) -> dict[str, Any]:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE collection_sessions SET status = 'SUBMITTING', updated_at = ? "
                "WHERE id = ? AND status = 'READY' AND sbatch_text IS NOT NULL",
                (utc_now(), session_id),
            )
            if cursor.rowcount != 1:
                row = connection.execute(
                    "SELECT status FROM collection_sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if row is None:
                    raise KeyError("Collection session not found")
                raise ValueError(f"session is not ready for submission: {row['status']}")
            self._insert_event(
                connection, session_id, "SUBMISSION_CLAIMED", "READY", "SUBMITTING"
            )
        result = self.get_session(session_id)
        assert result is not None
        return result

    def record_submission(
        self,
        session_id: str,
        *,
        job_id: str,
        gateway: str,
        sbatch_path: str,
        stdout_path: str,
        stderr_path: str,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT status FROM collection_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Collection session not found")
            if row["status"] != "SUBMITTING":
                raise ValueError(f"unexpected submission state: {row['status']}")
            connection.execute(
                "UPDATE collection_sessions SET status = 'SUBMITTED', slurm_job_id = ?, "
                "gateway = ?, sbatch_path = ?, stdout_path = ?, stderr_path = ?, "
                "submitted_at = ?, updated_at = ? WHERE id = ?",
                (
                    job_id,
                    gateway,
                    sbatch_path,
                    stdout_path,
                    stderr_path,
                    now,
                    now,
                    session_id,
                ),
            )
            self._insert_event(
                connection,
                session_id,
                "JOB_SUBMITTED",
                "SUBMITTING",
                "SUBMITTED",
                {"job_id": job_id, "gateway": gateway, "sbatch_path": sbatch_path},
            )
        result = self.get_session(session_id)
        assert result is not None
        return result

    def update_runtime_status(
        self,
        session_id: str,
        new_status: str,
        *,
        details: Mapping[str, Any],
        error: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if new_status not in SESSION_STATES:
            raise ValueError("invalid collection session state")
        now = utc_now()
        timestamps: dict[str, str] = {}
        if new_status == "RUNNING":
            timestamps["started_at"] = now
        elif new_status == "CAPTURED":
            timestamps["captured_at"] = now
        elif new_status == "COMPLETED":
            timestamps["completed_at"] = now
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT status FROM collection_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Collection session not found")
            fields = {
                "status": new_status,
                "error_json": canonical_json(error) if error else None,
                "updated_at": now,
                **timestamps,
            }
            assignments = ", ".join(f"{key} = ?" for key in fields)
            connection.execute(
                f"UPDATE collection_sessions SET {assignments} WHERE id = ?",
                (*fields.values(), session_id),
            )
            self._insert_event(
                connection,
                session_id,
                "STATUS_CHANGED",
                row["status"],
                new_status,
                details,
            )
        result = self.get_session(session_id)
        assert result is not None
        return result

    def record_registered_capture(
        self, session_id: str, *, resource_id: str, version_id: str
    ) -> dict[str, Any]:
        now = utc_now()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT status, registered_version_id FROM collection_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError("Collection session not found")
            if row["registered_version_id"]:
                if row["registered_version_id"] != version_id:
                    raise ValueError("session is already registered to another data version")
                result = self.get_session(session_id)
                assert result is not None
                return result
            if row["status"] != "CAPTURED":
                raise ValueError("only a captured session can be completed")
            connection.execute(
                "UPDATE collection_sessions SET status = 'COMPLETED', "
                "registered_resource_id = ?, registered_version_id = ?, "
                "completed_at = ?, updated_at = ? WHERE id = ?",
                (resource_id, version_id, now, now, session_id),
            )
            self._insert_event(
                connection,
                session_id,
                "RAW_CAPTURE_REGISTERED",
                "CAPTURED",
                "COMPLETED",
                {"resource_id": resource_id, "version_id": version_id},
            )
        result = self.get_session(session_id)
        assert result is not None
        return result


_MISSING = object()
_TEMPLATE = re.compile(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}")


def _lookup(value: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = value
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _template_context(session: Mapping[str, Any]) -> dict[str, Any]:
    runtime = session["software_snapshot"].get("runtime_profile") or {}
    return {
        "session": {"id": session["id"], "name": session["name"]},
        "adapter": session["adapter_snapshot"],
        "config": session["config_snapshot"],
        "software": session["software_snapshot"],
        "calibration": session["calibration_snapshot"],
        "storage": session["storage_snapshot"],
        "resources": session["resources_snapshot"],
        "capture": session["capture_snapshot"],
        "capabilities": session["capabilities_snapshot"],
        "runtime": runtime,
    }


def _render_template(value: str, context: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        resolved = _lookup(context, match.group(1))
        if resolved is _MISSING:
            raise CollectionValidationError(f"unresolved launcher token: {match.group(1)}")
        if isinstance(resolved, (dict, list)):
            return canonical_json(resolved)
        if resolved is None:
            return ""
        return str(resolved)

    rendered = _TEMPLATE.sub(replace, value)
    if "{{" in rendered or "}}" in rendered:
        raise CollectionValidationError(f"invalid launcher template: {value}")
    return rendered


def _environment_operation_lines(
    operations: Sequence[EnvironmentOperation], context: Mapping[str, Any]
) -> list[str]:
    lines: list[str] = []
    for operation in operations:
        name = operation.name
        if operation.operation == "unset":
            lines.append(f"unset {name}")
            continue
        assert operation.value is not None
        value = shlex.quote(_render_template(operation.value, context))
        if operation.operation == "set":
            lines.append(f"export {name}={value}")
        elif operation.operation == "prepend":
            lines.append(
                f"export {name}={value}\"${{{name}:+{operation.separator}}}${{{name}:-}}\""
            )
        else:
            lines.append(
                f"export {name}=\"${{{name}:-}}${{{name}:+{operation.separator}}}\"{value}"
            )
    return lines


def preflight_collection_session(session: Mapping[str, Any]) -> dict[str, Any]:
    manifest = CollectionAdapterManifest.model_validate(
        session["adapter_snapshot"]["manifest"]
    )
    context = _template_context(session)
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, message: str, *, blocking: bool = True) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if ok else "FAIL" if blocking else "WARN",
                "blocking": blocking,
                "message": message,
            }
        )

    check(
        "adapter.runnable",
        manifest.runnable,
        "adapter is runnable"
        if manifest.runnable
        else "adapter is marked non-runnable; resolve its TODOs and publish an edit",
    )
    for requirement in manifest.requirements:
        scope = context[requirement.scope]
        value = _lookup(scope, requirement.key)
        present = value is not _MISSING and value not in (None, "", [], {})
        matches = present and (
            requirement.equals is None or value == requirement.equals
        )
        ok = matches if requirement.required or requirement.equals is not None else True
        check(
            f"requirement.{requirement.scope}.{requirement.key}",
            ok,
            requirement.description,
            blocking=requirement.blocking,
        )

    output_path = str(session["storage_snapshot"].get("output_path") or "")
    candidate = PurePosixPath(output_path)
    root = PurePosixPath(WORK_ROOT)
    path_ok = candidate.is_absolute() and (candidate == root or root in candidate.parents)
    check(
        "storage.output_path",
        path_ok,
        f"raw output path must be an absolute path below {WORK_ROOT}",
    )

    resources_ok = True
    try:
        CollectionResources.model_validate(session["resources_snapshot"])
    except ValueError:
        resources_ok = False
    check("resources.slurm", resources_ok, "Slurm resources and queue policy are valid")

    for step in [*manifest.launcher.steps, *manifest.launcher.operator_steps]:
        try:
            for argument in step.argv:
                _render_template(argument, context)
            if step.cwd:
                _render_template(step.cwd, context)
            for value in {**manifest.launcher.environment, **step.environment}.values():
                _render_template(value, context)
        except CollectionValidationError as error:
            check(f"launcher.{step.name}", False, str(error))
        else:
            check(f"launcher.{step.name}", True, "all launcher tokens resolve")

    evidence_catalog = session["capabilities_snapshot"]
    for capability in manifest.capabilities:
        raw_evidence = evidence_catalog.get(capability.id)
        status = capability.default_status
        message = capability.description
        if raw_evidence is not None:
            evidence = CollectionCapabilityEvidence.model_validate(raw_evidence)
            if evidence.scope != capability.scope:
                status = "UNKNOWN"
                message = (
                    f"{capability.description} Evidence was collected on "
                    f"{evidence.scope}, not required scope {capability.scope}."
                )
            elif evidence.status == "verified":
                status = "PASS"
                suffix = ""
                if capability.kind == "network_ports":
                    suffix = " Configuration evidence is recorded; end-to-end reachability was not probed."
                message = f"{capability.description}{suffix}"
            elif evidence.status == "unavailable":
                status = "FAIL"
            elif evidence.status == "admin_required":
                status = "ADMIN_REQUIRED"
            else:
                status = "UNKNOWN"
        checks.append(
            {
                "name": f"capability.{capability.id}",
                "status": status,
                "blocking": capability.blocking,
                "message": message,
                "kind": capability.kind,
                "scope": capability.scope,
                "tcp_ports": capability.tcp_ports,
                "udp_ports": capability.udp_ports,
                "evidence": raw_evidence,
            }
        )

    blocking_failures = [
        item
        for item in checks
        if item["blocking"]
        and item["status"] in {"FAIL", "UNKNOWN", "ADMIN_REQUIRED"}
    ]
    return {
        "ok": not blocking_failures,
        "session_id": session["id"],
        "manifest_sha256": session["manifest_sha256"],
        "checks": checks,
        "todos": manifest.todos,
    }


def compile_collection_sbatch(session: Mapping[str, Any]) -> CompiledCollectionJob:
    report = preflight_collection_session(session)
    if not report["ok"]:
        messages = [
            item["message"]
            for item in report["checks"]
            if item["blocking"] and item["status"] == "FAIL"
        ]
        raise CollectionValidationError("; ".join(messages))

    manifest = CollectionAdapterManifest.model_validate(
        session["adapter_snapshot"]["manifest"]
    )
    resources = CollectionResources.model_validate(session["resources_snapshot"])
    context = _template_context(session)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", session["name"]).strip("-.")
    job_name = f"collect-{safe_name[:36]}-{session['id'][:8]}"
    stdout_path = f"{CLUSTER.paths.logs}/collection-{session['id']}-%j.out"
    stderr_path = f"{CLUSTER.paths.logs}/collection-{session['id']}-%j.err"
    gpu = (
        f"gpu:{resources.gpu_type}:{resources.gpu_count}"
        if resources.gpu_type
        else f"gpu:{resources.gpu_count}"
    )
    directives = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={resources.account}",
        f"#SBATCH --partition={resources.partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resources.cpu_count}",
        f"#SBATCH --mem={resources.memory_gb}G",
        f"#SBATCH --gres={gpu}",
        f"#SBATCH --time={resources.time_limit}",
        f"#SBATCH --chdir={CLUSTER.paths.workspace}",
        f"#SBATCH --output={stdout_path}",
        f"#SBATCH --error={stderr_path}",
        "#SBATCH --signal=B:TERM@60",
    ]
    if resources.node:
        directives.append(f"#SBATCH --nodelist={resources.node}")

    output_path = str(session["storage_snapshot"]["output_path"])
    control_path = f"{WORK_ROOT}/jobs/runs/{session['id']}"
    encoded_manifest = base64.b64encode(
        canonical_json(session["canonical_manifest"]).encode("utf-8")
    ).decode("ascii")
    lines = [
        *directives,
        "",
        "set -euo pipefail",
        "umask 027",
        f"export HOME={shlex.quote(HOME_ROOT)}",
        f"export WORK_ROOT={shlex.quote(WORK_ROOT)}",
        'export UV_CACHE_DIR="$WORK_ROOT/.cache/uv"',
        'export HF_HOME="$WORK_ROOT/.cache/huggingface"',
        'export TORCH_HOME="$WORK_ROOT/.cache/torch"',
        f"export SKYNET_COLLECTION_SESSION_ID={shlex.quote(session['id'])}",
        f"export SKYNET_COLLECTION_MANIFEST_SHA256={shlex.quote(session['manifest_sha256'])}",
        f"export SKYNET_COLLECTION_OUTPUT={shlex.quote(output_path)}",
        f"export SKYNET_COLLECTION_CONTROL={shlex.quote(control_path)}",
        'mkdir -p "$WORK_ROOT"/{workspace,repos,datasets,artifacts,logs,jobs}',
        'mkdir -p "$WORK_ROOT"/.cache/{uv,huggingface,torch}',
        'mkdir -p "$SKYNET_COLLECTION_OUTPUT" "$SKYNET_COLLECTION_CONTROL"',
        f"printf %s {shlex.quote(encoded_manifest)} | base64 --decode > "
        '"$SKYNET_COLLECTION_CONTROL/session-manifest.json"',
        'mkdir -p "$SKYNET_COLLECTION_OUTPUT/.skynet"',
        'cp "$SKYNET_COLLECTION_CONTROL/session-manifest.json" '
        '"$SKYNET_COLLECTION_OUTPUT/.skynet/session-manifest.json"',
        "",
        "# Raw capture stays in its adapter-native format. Conversion is a separate lineage job.",
    ]

    runtime_snapshot = session["software_snapshot"].get("runtime_profile") or {}
    runtime_operations = [
        EnvironmentOperation.model_validate(item)
        for item in runtime_snapshot.get("environment_operations", [])
    ]
    global_operations = [
        *runtime_operations,
        *[
            EnvironmentOperation(name=key, operation="set", value=value)
            for key, value in manifest.launcher.environment.items()
        ],
        *manifest.launcher.environment_operations,
    ]
    if global_operations:
        lines.extend(["", "# Runtime and adapter environment"])
        lines.extend(_environment_operation_lines(global_operations, context))
    for step in manifest.launcher.steps:
        argv = [_render_template(item, context) for item in step.argv]
        command = " ".join(shlex.quote(item) for item in argv)
        cwd = _render_template(step.cwd, context) if step.cwd else CLUSTER.paths.workspace
        step_operations = [
            *[
                EnvironmentOperation(name=key, operation="set", value=value)
                for key, value in step.environment.items()
            ],
            *step.environment_operations,
        ]
        lines.extend(
            [
                "",
                f"# Collection launcher step: {step.name}",
                "(",
                f"  cd {shlex.quote(cwd)}",
                *[
                    f"  {line}"
                    for line in _environment_operation_lines(step_operations, context)
                ],
                f"  {command}",
                ")",
            ]
        )
    script = "\n".join(lines) + "\n"
    return CompiledCollectionJob(
        script=script,
        script_sha256=content_sha256(script),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


_SLURM_STATE_MAP = {
    "PENDING": "PENDING",
    "CONFIGURING": "PENDING",
    "RUNNING": "RUNNING",
    "COMPLETING": "RUNNING",
    "COMPLETED": "CAPTURED",
    "CANCELLED": "CANCELLED",
    "FAILED": "FAILED",
    "TIMEOUT": "FAILED",
    "OUT_OF_MEMORY": "FAILED",
    "NODE_FAIL": "FAILED",
    "PREEMPTED": "FAILED",
    "BOOT_FAIL": "FAILED",
    "DEADLINE": "FAILED",
}


class CollectionService:
    def __init__(
        self,
        database: Database | None = None,
        cluster: ClusterClient | None = None,
        *,
        seed: bool = True,
    ) -> None:
        self.database = database or Database()
        self.cluster = cluster or ClusterClient()
        self.store = CollectionStore(self.database)
        if seed:
            self.store.seed_from_directory()

    def preflight(self, session_id: str) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError("Collection session not found")
        report = preflight_collection_session(session)
        self.store.record_preflight(session_id, report)
        return report

    def prepare(
        self,
        session_id: str,
        *,
        gateway: str | None = None,
        remote_validate: bool = True,
    ) -> dict[str, Any]:
        report = self.preflight(session_id)
        if not report["ok"]:
            raise CollectionValidationError("collection preflight failed")
        session = self.store.get_session(session_id)
        assert session is not None
        compiled = compile_collection_sbatch(session)
        requested_gateway = gateway or session["resources_snapshot"]["gateway"]
        if requested_gateway not in {"auto", *CLUSTER.gateways}:
            raise CollectionValidationError("gateway is not configured")
        actual_gateway = requested_gateway
        validation_output = "remote validation skipped"
        if remote_validate:
            actual_gateway, validation_output = self.cluster.test_script(
                compiled.script, requested_gateway
            )
        prepared = self.store.record_prepared(
            session_id,
            compiled,
            gateway=actual_gateway,
            validation_output=validation_output,
        )
        return {"session": prepared, "preflight": report, "sbatch": compiled.model_dump()}

    def submit(self, session_id: str, *, gateway: str | None = None) -> dict[str, Any]:
        session = self.store.claim_submission(session_id)
        requested_gateway = gateway or session["gateway"] or session["resources_snapshot"]["gateway"]
        try:
            submission = self.cluster.submit_script(
                session["sbatch_text"], session_id, requested_gateway
            )
        except Exception as error:
            self.store.update_runtime_status(
                session_id,
                "FAILED",
                details={"phase": "submission", "message": str(error)},
                error={"code": "SUBMISSION_FAILED", "message": str(error)},
            )
            raise
        stdout_path = session["stdout_path"].replace("%j", submission.job_id)
        stderr_path = session["stderr_path"].replace("%j", submission.job_id)
        submitted = self.store.record_submission(
            session_id,
            job_id=submission.job_id,
            gateway=submission.gateway,
            sbatch_path=submission.script_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        return {"session": submitted, "submission": submission.__dict__}

    def refresh(self, session_id: str) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError("Collection session not found")
        if not session["slurm_job_id"]:
            return session
        gateway, statuses = self.cluster.job_statuses(
            [session["slurm_job_id"]], session["gateway"] or "auto"
        )
        status = statuses.get(session["slurm_job_id"])
        if not status:
            return session
        slurm_state = status.get("State", "UNKNOWN").split("+", 1)[0].upper()
        mapped = _SLURM_STATE_MAP.get(slurm_state)
        if mapped is None or mapped == session["status"]:
            return session
        error = None
        if mapped == "FAILED":
            error = {
                "code": "SLURM_JOB_FAILED",
                "state": slurm_state,
                "reason": status.get("Reason"),
                "exit_code": status.get("ExitCode"),
            }
        return self.store.update_runtime_status(
            session_id,
            mapped,
            details={"gateway": gateway, "slurm": status},
            error=error,
        )

    def read_log(self, session_id: str, stream: str, lines: int) -> str:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError("Collection session not found")
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be stdout or stderr")
        path = session[f"{stream}_path"]
        if not path or "%j" in path:
            raise ValueError("the collection job has not been submitted")
        _, content = self.cluster.read_log(
            path, session["gateway"] or "auto", lines=lines
        )
        return content

    def cancel(self, session_id: str, gateway: str | None = None) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError("Collection session not found")
        if not session["slurm_job_id"]:
            raise ValueError("the collection session has no Slurm job")
        actual_gateway = self.cluster.cancel(
            session["slurm_job_id"], gateway or session["gateway"] or "auto"
        )
        return self.store.update_runtime_status(
            session_id,
            "CANCELLED",
            details={"gateway": actual_gateway, "job_id": session["slurm_job_id"]},
        )

    def report_error(
        self, session_id: str, request: CollectionErrorRequest
    ) -> dict[str, Any]:
        return self.store.update_runtime_status(
            session_id,
            "FAILED",
            details={"reported": True, **request.details},
            error=request.model_dump(mode="json"),
        )

    def complete(
        self, session_id: str, request: CollectionCompleteRequest
    ) -> dict[str, Any]:
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError("Collection session not found")
        if session["registered_version_id"]:
            version = self.database.get_data_resource_version(
                session["registered_version_id"]
            )
            return {"session": session, "version": version, "registered": False}
        if session["status"] != "CAPTURED":
            raise ValueError("only a captured session can be completed")
        storage = CollectionStorage.model_validate(session["storage_snapshot"])
        if storage.registration is None:
            raise ValueError("storage.registration is required to register raw capture")
        target = storage.registration
        matches = self.database.list_data_resources(
            provider=target.provider,
            namespace=target.namespace,
            include_archived=True,
        )
        resource = next((item for item in matches if item["name"] == target.name), None)
        if resource is None:
            resource = self.database.create_data_resource(
                provider=target.provider,
                namespace=target.namespace,
                name=target.name,
                kind=target.kind,
                description=f"Native raw captures registered by collection sessions",
                metadata={
                    "collection_schema_version": COLLECTION_SCHEMA_VERSION,
                    **request.resource_metadata,
                },
            )
        elif resource["kind"] != target.kind:
            raise ValueError("existing data resource kind does not match registration target")
        elif resource["archived_at"] is not None:
            raise ValueError("cannot register capture to an archived data resource")

        resource = self.database.get_data_resource(resource["id"])
        assert resource is not None
        revision = request.revision or request.manifest_sha256.lower()
        version = next(
            (
                item
                for item in resource["versions"]
                if item["revision"] == revision and item["format"] == storage.native_format
            ),
            None,
        )
        if version is not None:
            if (
                version["path"] != storage.output_path
                or version["manifest_sha256"] != request.manifest_sha256.lower()
            ):
                raise ValueError("existing immutable data version does not match this capture")
        else:
            version = self.database.create_data_resource_version(
                resource["id"],
                revision=revision,
                format=storage.native_format,
                path=storage.output_path,
                manifest_sha256=request.manifest_sha256,
                size_bytes=request.size_bytes,
                source_uri=f"collection:{session_id}",
                metadata={
                    "collection_session_id": session_id,
                    "collection_manifest_sha256": session["manifest_sha256"],
                    "adapter_manifest_sha256": session["adapter_snapshot"][
                        "manifest_sha256"
                    ],
                    "streams": session["adapter_snapshot"]["manifest"]["streams"],
                    "calibration": session["calibration_snapshot"],
                    "capture": session["capture_snapshot"],
                    "native_raw_preserved": True,
                    **request.version_metadata,
                },
            )
        completed = self.store.record_registered_capture(
            session_id, resource_id=resource["id"], version_id=version["id"]
        )
        return {"session": completed, "version": version, "registered": True}


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "COLLECTION_SCHEMA_VERSION",
    "SESSION_STATES",
    "CollectionAdapterManifest",
    "CollectionCalibration",
    "CollectionCapabilityDeclaration",
    "CollectionCapabilityEvidence",
    "CollectionCaptureMetadata",
    "CollectionCompleteRequest",
    "CollectionErrorRequest",
    "CollectionGatewayRequest",
    "CollectionResources",
    "CollectionRestartRequest",
    "CollectionService",
    "CollectionSessionCreate",
    "CollectionStorage",
    "CollectionValidationError",
    "CompiledCollectionJob",
    "StreamDeclaration",
    "compile_collection_sbatch",
    "preflight_collection_session",
]
