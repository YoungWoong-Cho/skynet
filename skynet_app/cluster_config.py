from __future__ import annotations

import copy
import json
import os
import re
import shlex
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProfileModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClusterPaths(ProfileModel):
    home_root: str
    work_root: str
    workspace: str
    repositories: str
    shared_repositories: str
    environments: str
    datasets: str
    artifacts: str
    logs: str
    jobs: str
    evaluation: str
    uv_cache: str
    huggingface_cache: str
    torch_cache: str


class ClusterCommands(ProfileModel):
    slurm_bin: str
    gpu_usage: str
    gpu_usage_interpreter: str | None = None

    def gpu_usage_shell_command(self, option: Literal["-l", "-u"]) -> str:
        argv = [self.gpu_usage, option]
        if self.gpu_usage_interpreter:
            argv.insert(0, self.gpu_usage_interpreter)
        return shlex.join(argv)


class RuntimeSourcePrerequisite(ProfileModel):
    kind: Literal["git_checkout", "directory"]
    name: str = Field(min_length=1, max_length=128)
    path: str
    revision: str | None = None
    required: bool = True

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        value = value.strip()
        if not PurePosixPath(value).is_absolute() or any(
            character in value for character in ("\x00", "\n", "\r")
        ):
            raise ValueError("runtime prerequisite paths must be absolute single-line paths")
        return value

    @model_validator(mode="after")
    def require_git_revision(self) -> "RuntimeSourcePrerequisite":
        if self.kind == "git_checkout" and not self.revision:
            raise ValueError("git checkout prerequisites require a revision")
        return self


class RuntimeGymRegistration(ProfileModel):
    module: str = Field(min_length=1)
    ids: list[str] = Field(min_length=1)


class RuntimeExecutableProvider(ProfileModel):
    name: str = Field(min_length=1)
    module: str = Field(min_length=1)
    resolver: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


class RuntimeComputeSmokeResources(ProfileModel):
    """Per-profile defaults and minimums for its compute readiness probe."""

    queue_policy: str = Field(min_length=1)
    gpu_type: str = Field(min_length=1)
    gpu_count: int = Field(default=1, ge=1)
    cpus_per_task: int = Field(ge=1)
    memory_gb: int = Field(ge=1)
    time_limit: str = Field(min_length=1)

    @field_validator("time_limit")
    @classmethod
    def single_line_time_limit(cls, value: str) -> str:
        value = value.strip()
        if any(character in value for character in ("\x00", "\n", "\r")):
            raise ValueError("readiness time limit must be a single-line value")
        return value


class RuntimeComputeSmoke(ProfileModel):
    suite: str = Field(min_length=1, max_length=256)
    adapter: str = Field(min_length=1, max_length=96)
    capsule_asset: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9_./-]+$",
    )
    capsule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    argv: list[str] = Field(min_length=1)
    environment: dict[str, str] = Field(default_factory=dict)
    result_schema: str = Field(min_length=1)
    timeout_seconds: int = Field(default=900, ge=30, le=3600)
    required_checks: list[str] = Field(default_factory=list)
    resources: RuntimeComputeSmokeResources


class RuntimeProfileVerification(ProfileModel):
    python_version: str | None = None
    distributions: dict[str, str] = Field(default_factory=dict)
    required_distributions: list[str] = Field(default_factory=list)
    python_imports: list[str] = Field(default_factory=list)
    executables: list[str] = Field(default_factory=list)
    executable_providers: list[RuntimeExecutableProvider] = Field(default_factory=list)
    shared_libraries: list[str] = Field(default_factory=list)
    gym_registrations: list[RuntimeGymRegistration] = Field(default_factory=list)
    platform_system: str | None = None
    glibc_minimum: str | None = None
    requires_gpu: bool = False
    compute_attestation_path: str | None = None
    compute_smoke: RuntimeComputeSmoke | None = None
    pip_check: bool = False

    @field_validator("compute_attestation_path")
    @classmethod
    def absolute_compute_attestation_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not PurePosixPath(value).is_absolute() or any(
            character in value for character in ("\x00", "\n", "\r")
        ):
            raise ValueError("compute attestation paths must be absolute single-line paths")
        return value


class RuntimeEnvironmentOperation(ProfileModel):
    name: str = Field(pattern=r"^[A-Z_][A-Z0-9_]*$")
    operation: Literal["set", "prepend", "append", "unset"] = "set"
    value: str | None = None
    separator: str = Field(default=":", min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_value(self) -> "RuntimeEnvironmentOperation":
        if self.operation == "unset":
            if self.value is not None:
                raise ValueError("unset environment operations cannot declare a value")
        elif self.value is None:
            raise ValueError(f"{self.operation} environment operations require a value")
        if any(character in self.separator for character in "\n\r$`{}"):
            raise ValueError("environment separator contains unsafe shell syntax")
        return self


class RuntimeProfileConfig(ProfileModel):
    label: str = Field(min_length=1, max_length=160)
    description: str = ""
    backend: Literal["uv", "conda", "apptainer", "existing"]
    environment_path: str | None = None
    environment_operations: list[RuntimeEnvironmentOperation] = Field(default_factory=list)
    container_image: str | None = None
    lock_file: str | None = None
    uv_executable: str | None = None
    bootstrap_uv: bool = False
    environment: dict[str, str] = Field(default_factory=dict)
    source_prerequisites: list[RuntimeSourcePrerequisite] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)
    verification: RuntimeProfileVerification = Field(default_factory=RuntimeProfileVerification)

    @field_validator("environment_path")
    @classmethod
    def absolute_environment_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not PurePosixPath(value).is_absolute() or any(
            character in value for character in ("\x00", "\n", "\r")
        ):
            raise ValueError("runtime environment paths must be absolute single-line paths")
        return value

    @field_validator("container_image", "lock_file", "uv_executable")
    @classmethod
    def single_line_value(cls, value: str | None) -> str | None:
        if value is not None and any(character in value for character in ("\x00", "\n", "\r")):
            raise ValueError("runtime profile values must be single-line")
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def validate_backend_inputs(self) -> "RuntimeProfileConfig":
        if self.backend == "conda" and not self.environment_path:
            raise ValueError("conda runtime profiles require environment_path")
        if self.backend == "apptainer" and not self.container_image:
            raise ValueError("apptainer runtime profiles require container_image")
        return self

    @property
    def resolved_location(self) -> str | None:
        return self.environment_path or self.container_image or self.lock_file or self.uv_executable


class QueueProfile(ProfileModel):
    partition: str
    account: str
    max_time_seconds: int = Field(ge=60)
    preemptible: bool = False


class DashboardProfile(ProfileModel):
    gpu_usage_columns: list[str] = Field(min_length=1)
    overflow_partitions: list[str] = Field(default_factory=list)
    overflow_account_label: str = "overcap/scavenger"


class ClusterDefaults(ProfileModel):
    queue_policy: str
    gateway: str = "auto"
    runtime_backend: str = "auto"
    gpu_type: str = "any"
    gpu_profile: str = "recommended"
    cpus_per_task: int = Field(ge=1)
    memory_gb: int = Field(ge=1)
    time_limit: str
    checkpoint_save_steps: int = Field(ge=1)
    checkpoint_keep_last: int = Field(ge=1, le=3)
    checkpoint_auto_resume: bool = True
    max_attempts: int = Field(ge=1)
    uv_version: str


class ClusterLimits(ProfileModel):
    max_nodes: int = Field(default=1, ge=1)
    max_gpus_per_node: int = Field(default=16, ge=1)
    max_cpus_per_task: int = Field(default=256, ge=1)
    max_memory_gb: int = Field(default=2048, ge=1)


class EvaluationNode(ProfileModel):
    gpu_type: str
    gpu_count: int = Field(ge=1)


class IsaacEvaluationPlacement(ProfileModel):
    nodes: dict[str, EvaluationNode] = Field(min_length=1)
    default_node: str

    @model_validator(mode="after")
    def validate_nodes(self) -> "IsaacEvaluationPlacement":
        if self.default_node not in self.nodes:
            raise ValueError("Isaac evaluation default node must be in the allowed nodes")
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in self.nodes):
            raise ValueError("Isaac evaluation nodes must be concrete Slurm node names")
        return self


class ClusterProfile(ProfileModel):
    schema_version: int = 1
    id: str
    label: str
    gateways: list[str] = Field(min_length=1)
    paths: ClusterPaths
    commands: ClusterCommands
    training_environment: dict[str, str] = Field(default_factory=dict)
    runtime_profiles: dict[str, RuntimeProfileConfig] = Field(default_factory=dict)
    queues: dict[str, QueueProfile]
    gpu_aliases: dict[str, str | None]
    dashboard: DashboardProfile
    defaults: ClusterDefaults
    limits: ClusterLimits
    isaac_evaluation_placement: IsaacEvaluationPlacement | None = None

    @field_validator("gateways")
    @classmethod
    def unique_gateways(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if len(normalized) != len(set(normalized)):
            raise ValueError("cluster gateways must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_references(self) -> "ClusterProfile":
        if self.defaults.queue_policy not in self.queues:
            raise ValueError("default queue policy is not configured")
        if self.defaults.gpu_type not in self.gpu_aliases:
            raise ValueError("default GPU type is not configured")
        if self.isaac_evaluation_placement:
            for node in self.isaac_evaluation_placement.nodes.values():
                if not self.gpu_aliases.get(node.gpu_type):
                    raise ValueError("Isaac evaluation nodes require a concrete configured GPU type")
        pairs = [(queue.partition, queue.account) for queue in self.queues.values()]
        if len(pairs) != len(set(pairs)):
            raise ValueError("queue partition/account pairs must be unique")
        for profile_id in self.runtime_profiles:
            if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", profile_id):
                raise ValueError(f"invalid runtime profile ID: {profile_id}")
        return self

    def queue(self, policy: str) -> QueueProfile:
        try:
            return self.queues[policy]
        except KeyError as error:
            raise ValueError(f"unknown queue policy: {policy}") from error

    def queue_for_partition(self, partition: str) -> QueueProfile:
        for queue in self.queues.values():
            if queue.partition == partition:
                return queue
        raise ValueError(f"partition is not configured: {partition}")

    def runtime_profile(self, profile_id: str) -> RuntimeProfileConfig:
        try:
            return self.runtime_profiles[profile_id]
        except KeyError as error:
            raise ValueError(f"unknown runtime profile: {profile_id}") from error

    def runtime_profile_snapshot(self, profile_id: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "id": profile_id,
            **self.runtime_profile(profile_id).model_dump(mode="json"),
        }

    def public_runtime_profiles(self) -> list[dict[str, Any]]:
        return [
            {
                **self.runtime_profile_snapshot(profile_id),
                "resolved_location": profile.resolved_location,
                "status": "configured",
                "runtime_verified": False,
            }
            for profile_id, profile in self.runtime_profiles.items()
        ]

    def public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _profile_path() -> Path:
    configured = os.environ.get("SKYNET_CLUSTER_CONFIG")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parent.parent / "config" / "clusters" / "skynet.json"


def _apply_environment_overrides(document: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(document)
    if hosts := os.environ.get("SKYNET_SSH_HOSTS"):
        payload["gateways"] = [host.strip() for host in hosts.split(",") if host.strip()]
    path_overrides = {
        "home_root": "SKYNET_HOME_ROOT",
        "work_root": "SKYNET_WORK_ROOT",
    }
    for field, variable in path_overrides.items():
        if value := os.environ.get(variable):
            old_root = str(payload["paths"][field])
            payload["paths"][field] = value
            if field == "work_root":
                for key, path in list(payload["paths"].items()):
                    if key != field and isinstance(path, str) and path.startswith(old_root + "/"):
                        payload["paths"][key] = value.rstrip("/") + path[len(old_root):]
    command_overrides = {
        "slurm_bin": "SKYNET_SLURM_BIN",
        "gpu_usage": "SKYNET_GPU_USAGE_COMMAND",
        "gpu_usage_interpreter": "SKYNET_GPU_USAGE_INTERPRETER",
    }
    for field, variable in command_overrides.items():
        if value := os.environ.get(variable):
            payload["commands"][field] = value
    for policy in tuple(payload.get("queues", {})):
        prefix = f"SKYNET_QUEUE_{policy.upper()}_"
        queue = payload["queues"][policy]
        if value := os.environ.get(prefix + "PARTITION"):
            queue["partition"] = value
        if value := os.environ.get(prefix + "ACCOUNT"):
            queue["account"] = value
        if value := os.environ.get(prefix + "MAX_TIME_SECONDS"):
            queue["max_time_seconds"] = int(value)
    return payload


@lru_cache(maxsize=1)
def get_cluster_profile() -> ClusterProfile:
    path = _profile_path()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not load cluster profile {path}: {error}") from error
    return ClusterProfile.model_validate(_apply_environment_overrides(document))


CLUSTER = get_cluster_profile()


__all__ = [
    "CLUSTER",
    "ClusterProfile",
    "QueueProfile",
    "RuntimeProfileConfig",
    "RuntimeSourcePrerequisite",
    "get_cluster_profile",
]
