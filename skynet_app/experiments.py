from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import re
import statistics
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
import urllib.parse
from typing import Any, Iterable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, model_serializer

from .cluster_config import CLUSTER


WORK_ROOT = CLUSTER.paths.work_root
EVAL_ROOT = CLUSTER.paths.evaluation
APP_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_CATALOG_ROOT = APP_ROOT / "config" / "evaluation_suites"
FULL_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_./:@+,-]+$")
SWEEP_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*$")


class CanonicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class AdapterName(StrEnum):
    GENERIC = "generic"
    EGOVERSE = "egoverse"
    DEXVERSE = "dexverse"
    DEXMIMICGEN = "dexmimicgen"
    GET_ZERO = "get_zero"
    GROOT = "groot"
    OPENPI = "openpi"


class IdentitySpec(CanonicalModel):
    project: str = Field(min_length=1, max_length=96)
    experiment: str = Field(min_length=1, max_length=128)
    variant: str | None = Field(default=None, max_length=128)
    tags: list[str] = Field(default_factory=list, max_length=64)

    @field_validator("project", "experiment", "variant")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("names must be non-empty single-line values")
        return value


class SourceSpec(CanonicalModel):
    repository: str = Field(min_length=1)
    revision: str = Field(min_length=1, max_length=128)
    adapter: str = Field(min_length=1, max_length=96)
    adapter_id: str | None = Field(default=None, max_length=128)
    adapter_version_id: str | None = Field(default=None, max_length=128)
    adapter_version: int = Field(default=1, ge=1)
    adapter_manifest: dict[str, Any] | None = None
    adapter_manifest_sha256: str | None = None
    dirty_policy: Literal["reject", "capture"] = "reject"
    project_subdirectory: str = "."
    include_submodules: bool = True
    include_git_lfs: bool = True

    @field_validator("repository")
    @classmethod
    def validate_repository(cls, value: str) -> str:
        value = value.strip()
        if any(character in value for character in ("\x00", "\n", "\r")):
            raise ValueError("repository must be a single-line Git URL or path")
        if re.search(r"://[^/@:]+:[^/@]+@", value):
            raise ValueError("repository URLs must not contain embedded credentials")
        return value

    @field_validator("revision")
    @classmethod
    def normalize_revision(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9._/+~-]+", value):
            raise ValueError("revision contains unsupported characters")
        return value.lower() if FULL_COMMIT_RE.fullmatch(value) else value

    @field_validator("adapter")
    @classmethod
    def validate_adapter_slug(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,95}", normalized):
            raise ValueError("adapter must be a lowercase slug")
        return normalized

    @model_validator(mode="after")
    def validate_adapter_snapshot(self) -> "SourceSpec":
        if self.adapter_manifest_sha256:
            if not SHA256_RE.fullmatch(self.adapter_manifest_sha256):
                raise ValueError("adapter_manifest_sha256 must be a SHA-256 value")
            if self.adapter_manifest is None:
                raise ValueError("adapter manifest hash requires an adapter manifest snapshot")
            if canonical_sha256(self.adapter_manifest) != self.adapter_manifest_sha256.lower():
                raise ValueError("adapter manifest snapshot hash does not match")
            self.adapter_manifest_sha256 = self.adapter_manifest_sha256.lower()
        return self

    @field_validator("project_subdirectory")
    @classmethod
    def validate_subdirectory(cls, value: str) -> str:
        value = value.strip().strip("/") or "."
        if value.startswith("../") or "/../" in value or not SAFE_PATH_RE.fullmatch(value):
            raise ValueError("project_subdirectory must stay inside the repository")
        return value


class RuntimeSourcePrerequisiteSnapshot(CanonicalModel):
    kind: Literal["git_checkout", "directory"]
    name: str
    path: str
    revision: str | None = None
    required: bool = True


class RuntimeEnvironmentOperationSnapshot(CanonicalModel):
    name: str
    operation: Literal["set", "prepend", "append", "unset"] = "set"
    value: str | None = None
    separator: str = ":"


class RuntimeGymRegistrationSnapshot(CanonicalModel):
    module: str
    ids: list[str]


class RuntimeExecutableProviderSnapshot(CanonicalModel):
    name: str
    module: str
    resolver: str


class RuntimeComputeSmokeResourcesSnapshot(CanonicalModel):
    queue_policy: str
    gpu_type: str
    gpu_count: int = 1
    cpus_per_task: int
    memory_gb: int
    time_limit: str


class RuntimeComputeSmokeSnapshot(CanonicalModel):
    suite: str
    adapter: str
    capsule_asset: str
    capsule_sha256: str
    argv: list[str]
    environment: dict[str, str] = Field(default_factory=dict)
    result_schema: str
    timeout_seconds: int = 900
    required_checks: list[str] = Field(default_factory=list)
    resources: RuntimeComputeSmokeResourcesSnapshot


class RuntimeProfileVerificationSnapshot(CanonicalModel):
    python_version: str | None = None
    distributions: dict[str, str] = Field(default_factory=dict)
    required_distributions: list[str] = Field(default_factory=list)
    python_imports: list[str] = Field(default_factory=list)
    executables: list[str] = Field(default_factory=list)
    executable_providers: list[RuntimeExecutableProviderSnapshot] = Field(
        default_factory=list
    )
    shared_libraries: list[str] = Field(default_factory=list)
    gym_registrations: list[RuntimeGymRegistrationSnapshot] = Field(default_factory=list)
    platform_system: str | None = None
    glibc_minimum: str | None = None
    requires_gpu: bool = False
    compute_attestation_path: str | None = None
    compute_smoke: RuntimeComputeSmokeSnapshot | None = None
    pip_check: bool = False


class RuntimeProfileSnapshot(CanonicalModel):
    schema_version: int = 1
    id: str
    label: str
    description: str = ""
    backend: Literal["uv", "conda", "apptainer", "existing"]
    environment_path: str | None = None
    container_image: str | None = None
    lock_file: str | None = None
    uv_executable: str | None = None
    bootstrap_uv: bool = False
    environment: dict[str, str] = Field(default_factory=dict)
    environment_operations: list[RuntimeEnvironmentOperationSnapshot] = Field(
        default_factory=list
    )
    source_prerequisites: list[RuntimeSourcePrerequisiteSnapshot] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)
    verification: RuntimeProfileVerificationSnapshot = Field(
        default_factory=RuntimeProfileVerificationSnapshot
    )


class RuntimeSpec(CanonicalModel):
    backend: Literal["auto", "uv", "conda", "apptainer", "existing"] = Field(
        default_factory=lambda: CLUSTER.defaults.runtime_backend
    )
    profile: str = "default"
    profile_id: str | None = None
    profile_snapshot: RuntimeProfileSnapshot | None = None
    profile_snapshot_sha256: str | None = None
    lock_file: str | None = None
    lock_sha256: str | None = None
    environment_path: str | None = None
    container_image: str | None = None
    container_digest: str | None = None
    uv_executable: str | None = None
    bootstrap_uv: bool = True
    uv_version: str = Field(default_factory=lambda: CLUSTER.defaults.uv_version)
    environment: dict[str, str] = Field(default_factory=dict)
    resolution: dict[str, Any] = Field(default_factory=dict)
    resolution_sha256: str | None = None

    @field_validator("lock_sha256", "container_digest")
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.removeprefix("sha256:").lower()
        if not SHA256_RE.fullmatch(normalized):
            raise ValueError("digest must be a SHA-256 value")
        return normalized

    @field_validator("lock_file", "environment_path", "container_image", "uv_executable")
    @classmethod
    def validate_runtime_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if any(character in value for character in ("\x00", "\n", "\r")):
            raise ValueError("runtime paths must be single-line values")
        return value

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"invalid environment variable name: {key}")
            if any(character in item for character in ("\x00", "\n", "\r")):
                raise ValueError(f"environment variable {key} must be single-line")
            if re.search(r"(TOKEN|PASSWORD|SECRET|API_KEY)$", key, re.IGNORECASE):
                raise ValueError(f"store {key} as a secret reference, not a literal value")
        return value

    @model_validator(mode="after")
    def validate_backend_requirements(self) -> "RuntimeSpec":
        if self.backend == "conda" and not self.environment_path:
            raise ValueError("conda runtime requires a deterministic environment_path")
        if self.backend == "apptainer" and not self.container_image:
            raise ValueError("apptainer runtime requires container_image")
        if self.resolution_sha256:
            if not SHA256_RE.fullmatch(self.resolution_sha256):
                raise ValueError("resolution_sha256 must be a SHA-256 value")
            if canonical_sha256(self.resolution) != self.resolution_sha256.lower():
                raise ValueError("runtime resolution evidence hash does not match")
            self.resolution_sha256 = self.resolution_sha256.lower()
        profile_fields = (
            self.profile_id,
            self.profile_snapshot,
            self.profile_snapshot_sha256,
        )
        if any(value is not None for value in profile_fields):
            if not all(value is not None for value in profile_fields):
                raise ValueError(
                    "named runtime profiles require profile_id, profile_snapshot, and profile_snapshot_sha256"
                )
            assert self.profile_id is not None
            assert self.profile_snapshot is not None
            assert self.profile_snapshot_sha256 is not None
            if self.profile_snapshot.id != self.profile_id:
                raise ValueError("runtime profile snapshot ID does not match profile_id")
            if self.profile_snapshot.backend != self.backend:
                raise ValueError("runtime profile snapshot backend does not match runtime backend")
            digest = self.profile_snapshot_sha256.lower()
            if not SHA256_RE.fullmatch(digest):
                raise ValueError("profile_snapshot_sha256 must be a SHA-256 value")
            if canonical_sha256(self.profile_snapshot.model_dump(mode="json")) != digest:
                raise ValueError("runtime profile snapshot hash does not match")
            self.profile_snapshot_sha256 = digest
        if self.bootstrap_uv and not re.fullmatch(r"\d+\.\d+\.\d+(?:[A-Za-z0-9.-]*)?", self.uv_version):
            raise ValueError("uv bootstrap requires a pinned uv_version")
        return self


class DatasetSpec(CanonicalModel):
    name: str
    uri: str
    revision: str
    sha256: str | None = None
    split: str | None = None
    manifest_path: str | None = None

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is not None and not SHA256_RE.fullmatch(value):
            raise ValueError("dataset sha256 must contain 64 hexadecimal characters")
        return value.lower() if value else value


class DataResourceSnapshot(CanonicalModel):
    provider: str
    namespace: str
    name: str
    kind: str


class DataVersionSnapshot(CanonicalModel):
    revision: str
    format: str
    path: str
    source_uri: str | None = None
    manifest_sha256: str
    status: str
    size_bytes: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("manifest_sha256")
    @classmethod
    def validate_manifest_sha256(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("data version manifest_sha256 must be a SHA-256 value")
        return value.lower()


class DataBundleAssignmentSnapshot(CanonicalModel):
    role: str
    position: int = Field(ge=0)
    mount_path: str | None = None
    required: bool = True
    config: dict[str, Any] = Field(default_factory=dict)
    resource: DataResourceSnapshot
    version: DataVersionSnapshot


class DataBundleSnapshot(CanonicalModel):
    id: str
    schema_version: Literal["skynet.data-bundle/v1"] = "skynet.data-bundle/v1"
    name: str
    version: str
    manifest_sha256: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    assignments: list[DataBundleAssignmentSnapshot]

    @field_validator("manifest_sha256")
    @classmethod
    def validate_manifest_sha256(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("data bundle manifest_sha256 must be a SHA-256 value")
        return value.lower()


class DataSpec(CanonicalModel):
    datasets: list[DatasetSpec] = Field(default_factory=list)
    bundle: DataBundleSnapshot | None = None
    normalization_artifact: str | None = None


class BatchSpec(CanonicalModel):
    declared_semantics: Literal[
        "per_device",
        "global_before_accumulation",
        "global_effective",
        "repository_native",
    ] = "per_device"
    value: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    expected_effective_batch: int | None = Field(default=None, ge=1)


class CheckpointPolicy(CanonicalModel):
    save_every_steps: int = Field(default=1000, ge=1)
    save_before_timeout_seconds: int = Field(default=300, ge=60, le=1800)
    keep_last: int = Field(default=3, ge=1, le=3)
    auto_resume: bool = True
    max_attempts: int = Field(default=5, ge=1, le=100)
    final_selector: Literal["best", "latest"] = "latest"
    remove_training_state_after_success: bool = True


class TrainSpec(CanonicalModel):
    learning_rate: float | None = Field(default=None, gt=0)
    batch: BatchSpec = Field(default_factory=BatchSpec)
    num_workers_per_rank: int = Field(default=4, ge=0, le=256)
    max_steps: int | None = Field(default=None, ge=1)
    max_epochs: int | None = Field(default=None, ge=1)
    seed: int = 42
    precision: Literal["bf16", "fp16", "fp32"] | None = None
    checkpoint: CheckpointPolicy = Field(default_factory=CheckpointPolicy)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_training_length(self) -> "TrainSpec":
        if self.max_steps is None and self.max_epochs is None:
            self.max_steps = 1000
        return self


class NodeSpec(CanonicalModel):
    mode: Literal["auto", "manual"] = "auto"
    name: str | None = None

    @model_validator(mode="after")
    def validate_manual_node(self) -> "NodeSpec":
        if self.mode == "manual":
            if not self.name or not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
                raise ValueError("manual node mode requires one concrete Slurm node")
        else:
            self.name = None
        return self


class GPUResourceSpec(CanonicalModel):
    mode: Literal["auto", "explicit"] = "auto"
    count: int | None = Field(default=None, ge=1, le=CLUSTER.limits.max_gpus_per_node)
    profile: Literal["minimum", "recommended", "maximum-throughput"] = Field(
        default_factory=lambda: CLUSTER.defaults.gpu_profile
    )
    gpu_type: str = Field(default_factory=lambda: CLUSTER.defaults.gpu_type, alias="type")

    @field_validator("gpu_type")
    @classmethod
    def validate_gpu_type(cls, value: str) -> str:
        if value not in CLUSTER.gpu_aliases:
            raise ValueError(f"GPU type is not configured: {value}")
        return value

    @model_validator(mode="after")
    def validate_gpu_mode(self) -> "GPUResourceSpec":
        if self.mode == "explicit" and self.count is None:
            raise ValueError("explicit GPU mode requires count")
        if self.mode == "auto" and self.count is not None:
            raise ValueError("auto GPU mode resolves count from the adapter profile")
        return self


def parse_slurm_duration(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d{1,3}):(\d{2}):(\d{2})", value.strip())
    if not match:
        raise ValueError("time limit must use HH:MM:SS or D-HH:MM:SS")
    days, hours, minutes, seconds = (int(item or 0) for item in match.groups())
    if minutes > 59 or seconds > 59 or (days and hours > 23):
        raise ValueError("invalid Slurm time limit")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_slurm_duration(total_seconds: int) -> str:
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ResourceSpec(CanonicalModel):
    gateway: str = Field(default_factory=lambda: CLUSTER.defaults.gateway)
    queue_policy: str = Field(default_factory=lambda: CLUSTER.defaults.queue_policy)
    account: str = Field(default_factory=lambda: CLUSTER.queue(CLUSTER.defaults.queue_policy).account)
    partition: str = Field(default_factory=lambda: CLUSTER.queue(CLUSTER.defaults.queue_policy).partition)
    nodes: int = Field(default=1, ge=1, le=1)
    node: NodeSpec = Field(default_factory=NodeSpec)
    gpu: GPUResourceSpec = Field(default_factory=GPUResourceSpec)
    cpus_per_task: int = Field(default_factory=lambda: CLUSTER.defaults.cpus_per_task, ge=1, le=CLUSTER.limits.max_cpus_per_task)
    memory_gb: int = Field(default_factory=lambda: CLUSTER.defaults.memory_gb, ge=1, le=CLUSTER.limits.max_memory_gb)
    time_limit: str = Field(default_factory=lambda: CLUSTER.defaults.time_limit)

    @model_validator(mode="before")
    @classmethod
    def infer_queue_policy(cls, value: Any) -> Any:
        if isinstance(value, dict) and "queue_policy" not in value:
            value = dict(value)
            pair = (value.get("partition"), value.get("account"))
            value["queue_policy"] = next(
                (name for name, queue in CLUSTER.queues.items() if pair == (queue.partition, queue.account)),
                CLUSTER.defaults.queue_policy,
            )
        return value

    @field_validator("gateway")
    @classmethod
    def validate_gateway(cls, value: str) -> str:
        if value != "auto" and value not in CLUSTER.gateways:
            raise ValueError(f"gateway is not configured: {value}")
        return value

    @field_validator("queue_policy")
    @classmethod
    def validate_queue_policy(cls, value: str) -> str:
        if value != "auto" and value not in CLUSTER.queues:
            raise ValueError(f"queue policy is not configured: {value}")
        return value

    @field_validator("time_limit")
    @classmethod
    def normalize_time_limit(cls, value: str) -> str:
        return format_slurm_duration(parse_slurm_duration(value))

    @model_validator(mode="after")
    def validate_cluster_policy(self) -> "ResourceSpec":
        matching = [
            (name, queue)
            for name, queue in CLUSTER.queues.items()
            if (self.partition, self.account) == (queue.partition, queue.account)
        ]
        if not matching:
            queues = list(CLUSTER.queues.values())
            if queues and all(queue.partition == queue.account for queue in queues):
                choices = " or ".join(f"both be {queue.account}" for queue in queues)
                raise ValueError(f"partition and account must {choices}")
            choices = " or ".join(
                f"partition {queue.partition} with account {queue.account}" for queue in queues
            )
            raise ValueError(f"partition and account must match a configured queue: {choices}")
        if self.queue_policy != "auto" and matching[0][0] != self.queue_policy:
            required = CLUSTER.queue(self.queue_policy)
            if required.partition == required.account:
                raise ValueError(
                    f"{self.queue_policy} queue policy requires account and partition {required.account}"
                )
            raise ValueError(
                f"{self.queue_policy} queue policy requires account {required.account} "
                f"and partition {required.partition}"
            )
        maximum = max(queue.max_time_seconds for queue in CLUSTER.queues.values()) if self.queue_policy == "auto" else matching[0][1].max_time_seconds
        if parse_slurm_duration(self.time_limit) > maximum:
            raise ValueError(f"{self.partition} jobs are limited to {format_slurm_duration(maximum)}")
        return self


class SweepSpec(CanonicalModel):
    strategy: Literal["grid", "zip", "explicit"] = "grid"
    axes: dict[str, list[Any]] = Field(default_factory=dict)
    variants: list[dict[str, Any]] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1)
    max_parallel: int = Field(default=2, ge=1, le=256)
    confirmation_threshold: int = Field(default=20, ge=1)

    @model_serializer(mode="wrap")
    def serialize_seed_intent(self, handler):
        payload = handler(self)
        # Preserve omission across saved-spec round trips: an automatic baseline
        # seed must not become an explicit unsupported adapter override.
        if "seeds" not in self.model_fields_set:
            payload.pop("seeds", None)
        return payload

    @field_validator("axes")
    @classmethod
    def validate_axes(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for path, choices in value.items():
            if not SWEEP_PATH_RE.fullmatch(path):
                raise ValueError(f"invalid sweep path: {path}")
            if not choices:
                raise ValueError(f"sweep axis {path} has no choices")
        return value

    @model_validator(mode="after")
    def validate_strategy(self) -> "SweepSpec":
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("sweep seeds must be unique")
        if "train.seed" in self.axes and self.seeds != [42]:
            raise ValueError("use either sweep.seeds or a train.seed axis, not both")
        if self.strategy == "explicit":
            if self.axes:
                raise ValueError("explicit sweeps use variants, not axes")
            if not self.variants:
                raise ValueError("explicit sweeps require at least one variant")
        elif self.variants:
            raise ValueError("grid and zip sweeps use axes, not variants")
        if self.strategy == "zip" and self.axes:
            lengths = {len(choices) for choices in self.axes.values()}
            if len(lengths) != 1:
                raise ValueError("all zip sweep axes must have the same length")
        return self


class TrackingProviderSpec(CanonicalModel):
    provider: Literal["mlflow", "wandb"]
    enabled: bool = True
    tracking_uri: str | None = None
    experiment: str | None = None
    base_url: str | None = None
    entity: str | None = None
    project: str | None = None
    run_name_template: str | None = None

    @model_validator(mode="after")
    def validate_provider_fields(self) -> "TrackingProviderSpec":
        if self.provider == "mlflow" and any((self.base_url, self.entity, self.project)):
            raise ValueError("MLflow provider does not accept W&B base_url, entity, or project")
        if self.provider == "wandb" and any((self.tracking_uri, self.experiment)):
            raise ValueError("W&B provider does not accept MLflow tracking_uri or experiment")
        return self


class TrackingSpec(CanonicalModel):
    providers: list[TrackingProviderSpec] = Field(default_factory=list)
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str | None = None
    native_tracking: Literal["preserve", "disable"] = "preserve"
    tags: dict[str, str] = Field(default_factory=dict)
    offline_spool: bool = True

    @field_validator("native_tracking", mode="before")
    @classmethod
    def normalize_native_tracking(cls, value: Any) -> Any:
        if isinstance(value, bool):
            return "preserve" if value else "disable"
        return value

    @model_validator(mode="after")
    def validate_unique_providers(self) -> "TrackingSpec":
        names = [item.provider for item in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("tracking providers must be unique")
        uris = [self.mlflow_tracking_uri]
        uris.extend(item.tracking_uri for item in self.providers if item.provider == "mlflow")
        for uri in (item for item in uris if item):
            parsed = urllib.parse.urlsplit(uri)
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("tracking URI credentials must be configured through Connections")
        return self


class EvaluationSpec(CanonicalModel):
    adapter: str = Field(min_length=1)
    suite: str
    suite_version: str
    tasks: list[str] = Field(default_factory=list)
    profile: str = Field(default="standard", min_length=1)
    episodes_per_task: int | None = Field(default=None, ge=1, le=10000)
    seeds: list[int] | None = None
    horizon: int | None = Field(default=None, ge=1)
    episode_timeout_seconds: int = Field(default=900, ge=1)
    render: bool = False
    video: Literal["none", "failures", "all"] = "failures"
    checkpoint_selector: Literal["best", "latest", "explicit"] = "best"
    checkpoint_path: str | None = None
    auto_resume: bool = True
    max_attempts: int = Field(default=5, ge=1, le=100)
    data_root: str = EVAL_ROOT
    native: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def apply_profile_defaults(self) -> "EvaluationSpec":
        profile_episodes = {"smoke": 2, "standard": 20, "report": 50}
        if self.episodes_per_task is None:
            self.episodes_per_task = profile_episodes.get(self.profile, profile_episodes["standard"])
        if self.seeds is None:
            self.seeds = [41, 42, 43] if self.profile == "report" else [42]
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("evaluation seeds must be unique")
        if self.checkpoint_selector == "explicit" and not self.checkpoint_path:
            raise ValueError("explicit checkpoint selection requires checkpoint_path")
        return self


class NativeSpec(CanonicalModel):
    argv: list[str] = Field(default_factory=list)
    resume_argv: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("argv", "resume_argv")
    @classmethod
    def validate_argv(cls, value: list[str]) -> list[str]:
        for argument in value:
            if any(character in argument for character in ("\x00", "\n", "\r")):
                raise ValueError("argv values must be single-line strings")
            if "submitit" in argument.casefold() or "hydra/launcher=submitit" in argument.casefold():
                raise ValueError("repository-native Submitit is disabled; use the canonical outer sbatch")
        return value


class ReproducibilitySpec(CanonicalModel):
    mode: Literal["exact-input", "deterministic-requested", "statistical-only"] = "exact-input"
    replay_policy: Literal["strict", "compatible", "fork"] = "strict"
    fail_on_drift: bool = True
    capture_system: bool = True
    capture_rng: bool = True
    capture_data_hashes: bool = True


def _contains_explicit_parameter_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_explicit_parameter_value(child) for child in value.values())
    if isinstance(value, list):
        return bool(value)
    return value is not None and value != "" and value != "adapter-default"


def explicit_train_parameter_paths(document: Mapping[str, Any]) -> list[str]:
    """Capture requested train values before model or adapter defaults are applied."""
    train = document.get("train")
    if not isinstance(train, Mapping):
        return []

    paths: set[str] = set()

    def visit(value: Any, path: str) -> None:
        if path == "train.hyperparameters":
            if _contains_explicit_parameter_value(value):
                paths.add(path)
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}")
            return
        if _contains_explicit_parameter_value(value):
            paths.add(path)

    visit(train, "train")
    return sorted(paths)


class ExperimentIntentSpec(CanonicalModel):
    explicit_parameters: list[str] = Field(default_factory=list)

    @field_validator("explicit_parameters")
    @classmethod
    def validate_explicit_parameters(cls, value: list[str]) -> list[str]:
        normalized = sorted(set(value))
        for path in normalized:
            if not SWEEP_PATH_RE.fullmatch(path) or not path.startswith("train."):
                raise ValueError(
                    f"explicit parameter intent must be a canonical train path: {path}"
                )
        return normalized

    def is_explicit(self, path: str) -> bool:
        return any(
            path == requested
            or path.startswith(f"{requested}.")
            or requested.startswith(f"{path}.")
            for requested in self.explicit_parameters
        )


class ExperimentSpec(CanonicalModel):
    api_version: Literal["skynet.rl2/v1"] = Field(default="skynet.rl2/v1", alias="apiVersion")
    kind: Literal["Experiment"] = "Experiment"
    identity: IdentitySpec
    source: SourceSpec
    intent: ExperimentIntentSpec = Field(default_factory=ExperimentIntentSpec)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    data: DataSpec = Field(default_factory=DataSpec)
    train: TrainSpec = Field(default_factory=TrainSpec)
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    sweep: SweepSpec = Field(default_factory=SweepSpec)
    tracking: TrackingSpec = Field(default_factory=TrackingSpec)
    evaluation: list[EvaluationSpec] = Field(default_factory=list)
    native: NativeSpec = Field(default_factory=NativeSpec)
    reproducibility: ReproducibilitySpec = Field(default_factory=ReproducibilitySpec)

    @model_validator(mode="before")
    @classmethod
    def infer_parameter_intent(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "intent" not in value:
            value = dict(value)
            value["intent"] = {
                "explicit_parameters": explicit_train_parameter_paths(value),
            }
        return value

    @model_validator(mode="after")
    def validate_reproducibility(self) -> "ExperimentSpec":
        if self.reproducibility.mode == "exact-input" and not FULL_COMMIT_RE.fullmatch(self.source.revision):
            raise ValueError("exact-input reproducibility requires a full 40-character Git commit")
        if self.train.checkpoint.auto_resume and self.train.checkpoint.save_before_timeout_seconds >= parse_slurm_duration(self.resources.time_limit):
            raise ValueError("checkpoint warning must occur before the job time limit")
        return self


class ResolvedVariant(CanonicalModel):
    index: int
    name: str
    seed: int
    parameters: dict[str, Any]
    resolved_spec: ExperimentSpec
    resolved_spec_sha256: str


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _set_dotted_path(document: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: dict[str, Any] = document
    for index, part in enumerate(parts[:-1]):
        if part not in current:
            if parts[:2] in (["native", "overrides"], ["train", "hyperparameters"]):
                current[part] = {}
            else:
                raise ValueError(f"sweep path does not exist: {path}")
        next_value = current[part]
        if not isinstance(next_value, dict):
            raise ValueError(f"sweep path crosses a non-object value: {path}")
        current = next_value
    final = parts[-1]
    if final not in current and parts[:2] not in (["native", "overrides"], ["train", "hyperparameters"]):
        raise ValueError(f"sweep path does not exist: {path}")
    current[final] = copy.deepcopy(value)


def _sweep_assignments(sweep: SweepSpec) -> list[dict[str, Any]]:
    if sweep.strategy == "explicit":
        return copy.deepcopy(sweep.variants)
    if not sweep.axes:
        return [{}]
    keys = list(sweep.axes)
    if sweep.strategy == "zip":
        return [dict(zip(keys, values, strict=True)) for values in zip(*(sweep.axes[key] for key in keys), strict=True)]
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(sweep.axes[key] for key in keys))]


def expand_sweep(spec: ExperimentSpec) -> list[ResolvedVariant]:
    variants: list[ResolvedVariant] = []
    index = 0
    for assignment in _sweep_assignments(spec.sweep):
        explicit_seed = "train.seed" in assignment or "seeds" in spec.sweep.model_fields_set
        seeds = ([assignment["train.seed"]] if "train.seed" in assignment
                 else spec.sweep.seeds if explicit_seed else [spec.train.seed])
        for seed in seeds:
            parameters = copy.deepcopy(assignment)
            parameters["train.seed"] = seed
            payload = spec.model_dump(mode="json", by_alias=True)
            for path, value in parameters.items():
                _set_dotted_path(payload, path, value)
            explicit_parameters = set(
                payload.setdefault("intent", {}).setdefault("explicit_parameters", [])
            )
            explicit_parameters.update(
                path for path in parameters if path.startswith("train.") and (path != "train.seed" or explicit_seed)
            )
            payload["intent"]["explicit_parameters"] = sorted(explicit_parameters)
            payload["sweep"] = SweepSpec().model_dump(mode="json")
            resolved = ExperimentSpec.model_validate(payload)
            digest = canonical_sha256(resolved)
            variants.append(
                ResolvedVariant(
                    index=index,
                    name=f"v{index:03d}-s{seed}-{digest[:8]}",
                    seed=seed,
                    parameters=parameters,
                    resolved_spec=resolved,
                    resolved_spec_sha256=digest,
                )
            )
            index += 1
    return variants


class EvaluationTaskOption(CanonicalModel):
    id: str = Field(min_length=1, max_length=512)
    label: str = Field(min_length=1, max_length=512)
    description: str = Field(default="", max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "label")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if any(character in normalized for character in ("\x00", "\n", "\r")):
            raise ValueError("task identifiers and labels must be single-line strings")
        return normalized


class EvaluationTaskCatalogProvenance(CanonicalModel):
    repository: str = Field(min_length=1, max_length=2048)
    revision: str = Field(min_length=1, max_length=256)
    source_path: str = Field(min_length=1, max_length=2048)


def evaluation_task_catalog_sha256(tasks: list[str]) -> str:
    encoded = json.dumps(
        tasks, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class EvaluationDatasetTaskBinding(CanonicalModel):
    role: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    metadata_path: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


class EvaluationRuntimeSuite(CanonicalModel):
    suite: str = Field(min_length=1)
    version: str = Field(min_length=1)


class EvaluationSuite(CanonicalModel):
    runtime_readiness_suite: EvaluationRuntimeSuite | None = None
    initial_state: Literal["single_training_episode"] | None = None
    dataset_task_binding: EvaluationDatasetTaskBinding | None = None
    dataset_episode_binding: EvaluationDatasetTaskBinding | None = None
    default_tasks: list[str] | None = None
    schema_version: Literal[2]
    evaluator: str = Field(min_length=1, max_length=96)
    suite: str = Field(min_length=1, max_length=256)
    version: str = Field(min_length=1, max_length=256)
    label: str = Field(min_length=1, max_length=512)
    task_source: str = Field(min_length=1, max_length=96)
    task_catalog_complete: bool
    task_selection_mode: Literal["subset", "single", "all_only"]
    task_selection_reason: str = Field(min_length=1, max_length=1000)
    task_catalog_provenance: EvaluationTaskCatalogProvenance
    task_catalog_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    tasks: list[str]
    task_options: list[EvaluationTaskOption]
    current: bool
    default_profile: str = Field(
        default="standard",
        min_length=1,
        max_length=96,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$",
    )
    notes: str = ""
    catalog_path: str | None = None
    catalog_sha256: str | None = None

    @model_serializer(mode="wrap")
    def serialize_without_absent_binding(self, handler):
        result = handler(self)
        if self.initial_state is None:
            result.pop("initial_state", None)
        if self.runtime_readiness_suite is None:
            result.pop("runtime_readiness_suite", None)
        # Existing immutable catalogs must retain their original content hashes.
        if self.default_tasks is None:
            result.pop("default_tasks", None)
        if self.dataset_task_binding is None:
            result.pop("dataset_task_binding", None)
        if self.dataset_episode_binding is None:
            result.pop("dataset_episode_binding", None)
        return result

    @model_validator(mode="after")
    def validate_task_catalog(self) -> "EvaluationSuite":
        selection_reason = self.task_selection_reason.strip()
        if not selection_reason:
            raise ValueError("task_selection_reason must explain the declared capability")
        normalized_tasks = [task.strip() for task in self.tasks]
        if any(
            not task or any(character in task for character in ("\x00", "\n", "\r"))
            for task in normalized_tasks
        ):
            raise ValueError("task IDs must be non-empty single-line strings")
        if len(set(normalized_tasks)) != len(normalized_tasks):
            raise ValueError("task IDs must be unique")
        option_ids = [option.id for option in self.task_options]
        if option_ids != normalized_tasks:
            raise ValueError("task_options IDs must exactly match ordered tasks")
        if self.task_catalog_complete and not normalized_tasks:
            raise ValueError("a complete task catalog cannot be empty")
        expected = evaluation_task_catalog_sha256(normalized_tasks)
        if self.task_catalog_sha256.lower() != expected:
            raise ValueError("task_catalog_sha256 does not match ordered task IDs")
        self.tasks = normalized_tasks
        self.task_catalog_sha256 = expected
        self.task_selection_reason = selection_reason
        return self


def get_evaluation_catalog(
    catalog_root: str | Path | None = None,
) -> list[EvaluationSuite]:
    root = Path(catalog_root or EVALUATION_CATALOG_ROOT).resolve()
    if not root.exists():
        return []
    suites: list[EvaluationSuite] = []
    exact_versions: set[tuple[str, str, str]] = set()
    current_versions: set[tuple[str, str]] = set()
    for candidate in sorted(root.rglob("*.json")):
        path = candidate.resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"evaluation catalog escapes configured root: {candidate}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid evaluation catalog {candidate}: {error}") from error
        if not isinstance(document, dict):
            raise ValueError(f"evaluation catalog must contain one object: {candidate}")
        reserved = {"catalog_path", "catalog_sha256"}.intersection(document)
        if reserved:
            raise ValueError(
                f"evaluation catalog contains loader-owned fields {sorted(reserved)}: {candidate}"
            )
        suite = EvaluationSuite.model_validate(document)
        identity = (suite.evaluator, suite.suite, suite.version)
        if identity in exact_versions:
            raise ValueError(f"duplicate evaluation suite version: {'/'.join(identity)}")
        exact_versions.add(identity)
        logical_identity = (suite.evaluator, suite.suite)
        if suite.current and logical_identity in current_versions:
            raise ValueError(
                f"multiple current evaluation suite versions: {'/'.join(logical_identity)}"
            )
        if suite.current:
            current_versions.add(logical_identity)
        immutable_document = suite.model_dump(
            mode="json",
            exclude={"catalog_path", "catalog_sha256", "current"},
        )
        catalog_digest = hashlib.sha256(
            json.dumps(
                immutable_document,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        try:
            catalog_path = str(path.relative_to(APP_ROOT))
        except ValueError:
            catalog_path = str(path)
        suites.append(
            suite.model_copy(
                update={
                    "catalog_path": catalog_path,
                    "catalog_sha256": catalog_digest,
                },
                deep=True,
            )
        )
    return sorted(
        suites,
        key=lambda item: (item.evaluator, item.suite, item.version, item.catalog_path or ""),
    )


class CheckpointReference(CanonicalModel):
    path: str
    sha256: str


class EvaluatorReference(CanonicalModel):
    adapter: str
    version: str


class EnvironmentReference(CanonicalModel):
    suite: str
    version: str
    task: str | None = None


class CanonicalMetric(CanonicalModel):
    metric: str
    unit: str
    mean: float
    std: float
    sample_count: int = Field(ge=1)
    task: str | None = None


class EpisodeResult(CanonicalModel):
    task: str
    seed: int
    episode_index: int = Field(ge=0)
    success: bool | None = None
    reward: float | None = None
    episode_length: int | None = Field(default=None, ge=0)
    status: Literal["SUCCEEDED", "FAILED", "TIMEOUT"] = "SUCCEEDED"
    metrics: dict[str, float] = Field(default_factory=dict)
    video_path: str | None = None
    failure_reason: str | None = None


class CanonicalResult(CanonicalModel):
    schema_version: Literal[1] = 1
    run_id: str
    checkpoint: CheckpointReference
    evaluator: EvaluatorReference
    environment: EnvironmentReference
    aggregate: list[CanonicalMetric]
    episodes: list[EpisodeResult]
    raw_metrics_path: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def build_canonical_result(
    *,
    run_id: str,
    checkpoint: CheckpointReference,
    evaluator: EvaluatorReference,
    environment: EnvironmentReference,
    episodes: Iterable[EpisodeResult],
    raw_metrics_path: str | None = None,
    artifacts: list[str] | None = None,
) -> CanonicalResult:
    episode_list = list(episodes)
    aggregate: list[CanonicalMetric] = []

    def append_metric(name: str, unit: str, values: list[float], task: str | None = None) -> None:
        if values:
            aggregate.append(
                CanonicalMetric(
                    metric=name,
                    unit=unit,
                    mean=statistics.fmean(values),
                    std=statistics.pstdev(values) if len(values) > 1 else 0.0,
                    sample_count=len(values),
                    task=task,
                )
            )

    successful = [episode for episode in episode_list if episode.status == "SUCCEEDED"]
    append_metric("success_rate", "fraction", [float(episode.success) for episode in successful if episode.success is not None])
    append_metric("reward", "repository-native", [episode.reward for episode in successful if episode.reward is not None])
    append_metric("episode_length", "steps", [float(episode.episode_length) for episode in successful if episode.episode_length is not None])
    custom_names = sorted({name for episode in successful for name in episode.metrics})
    for name in custom_names:
        append_metric(name, "repository-native", [episode.metrics[name] for episode in successful if name in episode.metrics])
    for task in sorted({episode.task for episode in episode_list}):
        task_episodes = [episode for episode in successful if episode.task == task]
        append_metric("success_rate", "fraction", [float(episode.success) for episode in task_episodes if episode.success is not None], task)
        append_metric("reward", "repository-native", [episode.reward for episode in task_episodes if episode.reward is not None], task)

    return CanonicalResult(
        run_id=run_id,
        checkpoint=checkpoint,
        evaluator=evaluator,
        environment=environment,
        aggregate=aggregate,
        episodes=episode_list,
        raw_metrics_path=raw_metrics_path,
        artifacts=artifacts or [],
    )
