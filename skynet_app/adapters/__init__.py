from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import PurePosixPath
from typing import Any, Literal, Mapping

from pydantic import Field, computed_field, field_serializer, field_validator, model_validator

from skynet_app.experiments import AdapterName, CanonicalModel, ExperimentSpec, canonical_sha256

from .groot_isaaclab_bridge import GROOT_ISAACLAB_BRIDGE_SOURCE
from .groot_robocasa_bridge import GROOT_ROBOCASA_BRIDGE_SOURCE
from .openpi_libero_bridge import OPENPI_LIBERO_BRIDGE_SOURCE
from .openpi_dataset_bridge import OPENPI_DATASET_BRIDGE_SOURCE


RESUME_CHECKPOINT_TOKEN = "{{SKYNET_RESUME_CHECKPOINT}}"
RUN_DIR_TOKEN = "{{SKYNET_RUN_DIR}}"
SOURCE_DIR_TOKEN = "{{SKYNET_SOURCE_DIR}}"
GENERATED_ARGS_TOKEN = "{{generated_args}}"


class AdapterCapabilities(CanonicalModel):
    name: str
    version: int = 1
    runtime_backends: set[str]
    supports_multi_gpu_single_node: bool
    supports_resume: bool
    supports_checkpoint_signal: bool = False
    supports_evaluation_resume: bool = False
    minimum_gpus: int = Field(default=1, ge=1)
    recommended_gpus: int = Field(default=1, ge=1)
    maximum_gpus: int = Field(default=1, ge=1)
    evaluation_adapters: list[str] = Field(default_factory=list)

    @field_serializer("runtime_backends")
    def serialize_runtime_backends(self, value: set[str]) -> list[str]:
        return sorted(value)


AdapterBatchSemantics = Literal[
    "per_device",
    "global_before_accumulation",
    "global_effective",
]


class AdapterBatchCompatibility(CanonicalModel):
    schema_version: Literal["skynet.batch-compatibility/v1"] = (
        "skynet.batch-compatibility/v1"
    )
    allowed_semantics: list[AdapterBatchSemantics] = Field(min_length=1)
    multi_gpu_allowed_semantics: list[AdapterBatchSemantics] = Field(
        default_factory=list
    )
    supports_gradient_accumulation: bool
    batch_size_divisible_by: Literal["resolved_gpu_count"] | None = None

    @model_validator(mode="after")
    def validate_semantics(self) -> "AdapterBatchCompatibility":
        if len(self.allowed_semantics) != len(set(self.allowed_semantics)):
            raise ValueError("batch compatibility allowed_semantics must be unique")
        if len(self.multi_gpu_allowed_semantics) != len(
            set(self.multi_gpu_allowed_semantics)
        ):
            raise ValueError(
                "batch compatibility multi_gpu_allowed_semantics must be unique"
            )
        unsupported = set(self.multi_gpu_allowed_semantics) - set(
            self.allowed_semantics
        )
        if unsupported:
            raise ValueError(
                "multi-GPU batch semantics must also be present in allowed_semantics"
            )
        return self


class PreparationStep(CanonicalModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,95}$")
    argv: list[str] = Field(min_length=1)
    working_directory: str = "."
    output_globs: list[str] = Field(min_length=1)
    expected_output_sha256: dict[str, str] = Field(default_factory=dict)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: list[str]) -> list[str]:
        if any(any(character in item for character in ("\x00", "\n", "\r")) for item in value):
            raise ValueError("preparation argv must contain single-line strings")
        return value

    @staticmethod
    def _safe_relative(value: str, *, label: str) -> str:
        if not value or value.startswith("/") or "\\" in value:
            raise ValueError(f"{label} must be a non-empty relative POSIX path")
        if any(character in value for character in ("\x00", "\n", "\r")):
            raise ValueError(f"{label} must be single-line")
        if ".." in PurePosixPath(value).parts:
            raise ValueError(f"{label} must stay inside the run directory")
        return value

    @field_validator("working_directory")
    @classmethod
    def validate_working_directory(cls, value: str) -> str:
        return cls._safe_relative(value, label="preparation working_directory")

    @field_validator("output_globs")
    @classmethod
    def validate_output_globs(cls, value: list[str]) -> list[str]:
        return [cls._safe_relative(item, label="preparation output_glob") for item in value]

    @field_validator("expected_output_sha256")
    @classmethod
    def validate_expected_output_sha256(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for path, digest in value.items():
            safe_path = cls._safe_relative(path, label="preparation expected output")
            if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ValueError("preparation expected output SHA-256 must contain 64 hex characters")
            normalized[safe_path] = digest.lower()
        return dict(sorted(normalized.items()))


CheckpointCandidateKind = Literal["any", "file", "directory"]


def _validate_checkpoint_glob(value: str) -> str:
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError("checkpoint globs must be non-empty relative POSIX paths")
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise ValueError("checkpoint globs must be single-line strings")
    if ".." in PurePosixPath(value).parts:
        raise ValueError("checkpoint globs must not escape their checkpoint root")
    return value


def _validate_checkpoint_basename_regex(value: str | None) -> str | None:
    if value is None:
        return None
    if not value or len(value) > 256:
        raise ValueError("checkpoint basename regex must contain 1-256 characters")
    if any(character in value for character in ("\x00", "\n", "\r", "/", "\\")):
        raise ValueError("checkpoint basename regex must target a basename, not a path")
    try:
        re.compile(value)
    except re.error as error:
        raise ValueError(f"invalid checkpoint basename regex: {error}") from error
    return value


class TrainingProgressLogSource(CanonicalModel):
    kind: Literal["log_regex"] = "log_regex"
    stream: Literal["stdout", "stderr"] = "stderr"
    pattern: str = Field(min_length=1, max_length=1000)
    value_format: Literal["integer", "decimal_si"] = "integer"
    elapsed_format: Literal["seconds", "hms", "clock"] | None = None
    tail_lines: int = Field(default=500, ge=2, le=5000)
    poll_seconds: int = Field(default=5, ge=2, le=300)

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, value: str) -> str:
        if any(character in value for character in ("\x00", "\n", "\r")):
            raise ValueError("training progress regex must be a single-line pattern")
        try:
            compiled = re.compile(value)
        except re.error as error:
            raise ValueError(f"invalid training progress regex: {error}") from error
        missing = {"completed", "total"} - set(compiled.groupindex)
        if missing:
            raise ValueError(
                "training progress regex requires named groups: "
                + ", ".join(sorted(missing))
            )
        return value

    @model_validator(mode="after")
    def validate_elapsed_group(self) -> "TrainingProgressLogSource":
        groups = set(re.compile(self.pattern).groupindex)
        if self.elapsed_format is not None and "elapsed" not in groups:
            raise ValueError(
                "training progress regex requires an elapsed group when elapsed_format is set"
            )
        return self


class TrainingProgressContract(CanonicalModel):
    schema_version: Literal["skynet.training-progress-source/v1"] = (
        "skynet.training-progress-source/v1"
    )
    unit: Literal["step"] = "step"
    total_path: Literal["train.max_steps"] = "train.max_steps"
    source: TrainingProgressLogSource


_RESERVED_CAPSULE_FILES = {
    "argv.json",
    "attempt-snapshot.json",
    "checksums.sha256",
    "execution.json",
    "job.sbatch",
    "native-config.json",
    "preparation.json",
    "requested-spec.json",
    "resolved-spec.json",
}


def _validate_capsule_files(value: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    total_size = 0
    for raw_path, content in value.items():
        path = str(raw_path).strip()
        parsed = PurePosixPath(path)
        if (
            not path
            or path.startswith("/")
            or "\\" in path
            or any(character in path for character in ("\x00", "\n", "\r"))
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise ValueError("adapter capsule files require safe relative POSIX paths")
        if path in _RESERVED_CAPSULE_FILES:
            raise ValueError(f"adapter capsule file conflicts with a reserved run file: {path}")
        if not isinstance(content, str) or "\x00" in content:
            raise ValueError("adapter capsule file content must be NUL-free text")
        size = len(content.encode("utf-8"))
        if size > 1_000_000:
            raise ValueError(f"adapter capsule file exceeds the 1 MB limit: {path}")
        total_size += size
        normalized[path] = content
    if total_size > 4_000_000:
        raise ValueError("adapter capsule files exceed the 4 MB manifest limit")
    return dict(sorted(normalized.items()))


NativeTrackingParameter = Literal["enabled", "project", "experiment"]


class NativeTrackingIntegration(CanonicalModel):
    """Adapter-declared mapping from central tracking to native repository controls."""

    provider: Literal["wandb", "mlflow"]
    parameter_paths: dict[NativeTrackingParameter, str] = Field(default_factory=dict)
    run_id_file: str | None = None

    @field_validator("parameter_paths")
    @classmethod
    def validate_parameter_paths(
        cls, value: dict[NativeTrackingParameter, str]
    ) -> dict[NativeTrackingParameter, str]:
        for path in value.values():
            if not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*", path
            ):
                raise ValueError(f"invalid native tracking parameter path: {path}")
        if "enabled" not in value:
            raise ValueError("native tracking integration requires an enabled parameter path")
        return dict(value)

    @field_validator("run_id_file")
    @classmethod
    def validate_run_id_file(cls, value: str | None) -> str | None:
        if value is not None and (not value or any(c in value for c in ("\x00", "\n", "\r"))):
            raise ValueError("native tracking run ID file must be a non-empty single-line path")
        return value


class AdapterPlan(CanonicalModel):
    adapter: str
    adapter_version: int
    argv: list[str]
    resume_argv: list[str] = Field(default_factory=list)
    retry_clean_argv: list[str] = Field(default_factory=list)
    native_config: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
    native_tracking: list[NativeTrackingIntegration] = Field(default_factory=list)
    resolved_gpu_type: str | None = None
    preparation_steps: list[PreparationStep] = Field(default_factory=list)
    capsule_files: dict[str, str] = Field(default_factory=dict)
    checkpoint_globs: list[str] = Field(default_factory=list)
    checkpoint_candidate_kind: CheckpointCandidateKind = "any"
    checkpoint_basename_regex: str | None = None
    checkpoint_prune_globs: list[str] = Field(default_factory=list)
    training_output_prune_globs: list[str] = Field(default_factory=list)
    checkpoint_inference_required_globs: list[str] = Field(default_factory=list)
    progress: TrainingProgressContract | None = None
    blockers: list[str] = Field(default_factory=list)
    todos: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    capabilities: AdapterCapabilities
    manifest_sha256: str | None = None

    @field_validator("argv", "resume_argv", "retry_clean_argv")
    @classmethod
    def validate_argv(cls, value: list[str]) -> list[str]:
        for argument in value:
            if any(character in argument for character in ("\x00", "\n", "\r")):
                raise ValueError("adapter argv must be a list of single-line strings")
            if "submitit" in argument.casefold():
                raise ValueError("repository-native Submitit is forbidden")
        return value

    @field_validator(
        "checkpoint_globs",
        "checkpoint_prune_globs",
        "training_output_prune_globs",
        "checkpoint_inference_required_globs",
    )
    @classmethod
    def validate_checkpoint_globs(cls, value: list[str]) -> list[str]:
        return [_validate_checkpoint_glob(item) for item in value]

    @field_validator("checkpoint_basename_regex")
    @classmethod
    def validate_checkpoint_basename_regex(cls, value: str | None) -> str | None:
        return _validate_checkpoint_basename_regex(value)

    @field_validator("capsule_files")
    @classmethod
    def validate_capsule_files(cls, value: dict[str, str]) -> dict[str, str]:
        return _validate_capsule_files(value)

    @computed_field
    @property
    def runnable(self) -> bool:
        return bool(self.argv) and not self.blockers


class AdapterError(ValueError):
    pass


def _serialize_override(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _gpu_count(spec: ExperimentSpec, capabilities: AdapterCapabilities) -> int:
    gpu = spec.resources.gpu
    if gpu.mode == "explicit":
        assert gpu.count is not None
        return gpu.count
    return {
        "minimum": capabilities.minimum_gpus,
        "recommended": capabilities.recommended_gpus,
        "maximum-throughput": capabilities.maximum_gpus,
    }[gpu.profile]


class RepositoryAdapter(ABC):
    capabilities: AdapterCapabilities
    supported_canonical_fields: frozenset[str] = frozenset()

    def _capabilities_for_spec(self, spec: ExperimentSpec) -> AdapterCapabilities:
        return self.capabilities

    def _gpu_type_for_spec(self, spec: ExperimentSpec) -> str:
        return spec.resources.gpu.gpu_type

    def validate(self, spec: ExperimentSpec) -> list[str]:
        capabilities = self._capabilities_for_spec(spec)
        blockers: list[str] = []
        if spec.runtime.backend not in capabilities.runtime_backends:
            blockers.append(
                f"{capabilities.name} does not support the {spec.runtime.backend} runtime; "
                f"choose one of {sorted(capabilities.runtime_backends)}"
            )
        count = _gpu_count(spec, capabilities)
        if count > 1 and not capabilities.supports_multi_gpu_single_node:
            blockers.append(f"{capabilities.name} supports one GPU per training job")
        if count < capabilities.minimum_gpus or count > capabilities.maximum_gpus:
            blockers.append(
                f"{capabilities.name} supports {capabilities.minimum_gpus}-"
                f"{capabilities.maximum_gpus} GPUs, requested {count}"
            )
        return blockers

    def resolve(self, spec: ExperimentSpec) -> AdapterPlan:
        capabilities = self._capabilities_for_spec(spec)
        plan = self._resolve(spec, _gpu_count(spec, capabilities))
        plan.capabilities = capabilities
        plan.resolved_gpu_type = self._gpu_type_for_spec(spec)
        plan.blockers = [*self.validate(spec), *plan.blockers]
        if spec.train.checkpoint.auto_resume and not capabilities.supports_resume:
            plan.blockers.append(
                f"TODO: {capabilities.name} has no validated checkpoint resume mapping; "
                "disable auto_resume or provide native.resume_argv"
            )
        return plan

    @abstractmethod
    def _resolve(self, spec: ExperimentSpec, gpu_count: int) -> AdapterPlan:
        raise NotImplementedError

    def _base_plan(self, **kwargs: Any) -> AdapterPlan:
        return AdapterPlan(
            adapter=kwargs.pop("adapter", self.capabilities.name),
            adapter_version=kwargs.pop("adapter_version", self.capabilities.version),
            capabilities=kwargs.pop("capabilities", self.capabilities),
            **kwargs,
        )


class GenericAdapter(RepositoryAdapter):
    capabilities = AdapterCapabilities(
        name=AdapterName.GENERIC,
        runtime_backends={"uv", "conda", "apptainer", "existing"},
        supports_multi_gpu_single_node=True,
        supports_resume=True,
        minimum_gpus=1,
        recommended_gpus=1,
        maximum_gpus=16,
    )

    def _resolve(self, spec: ExperimentSpec, gpu_count: int) -> AdapterPlan:
        blockers = [] if spec.native.argv else ["generic adapter requires native.argv as a structured argument list"]
        if spec.train.checkpoint.auto_resume and not spec.native.resume_argv:
            blockers.append("generic auto-resume requires native.resume_argv")
        return self._base_plan(
            argv=list(spec.native.argv),
            resume_argv=list(spec.native.resume_argv),
            native_config=spec.native.config,
            environment={"SKYNET_ASSIGNED_GPU_COUNT": str(gpu_count)},
            checkpoint_globs=list(spec.native.config.get("checkpoint_globs", [])),
            blockers=blockers,
            todos=[],
            warnings=[],
        )


class EgoVerseAdapter(RepositoryAdapter):
    supported_canonical_fields = frozenset(
        {
            "train.learning_rate",
            "train.batch.gradient_accumulation_steps",
            "train.max_steps",
            "train.seed",
        }
    )
    capabilities = AdapterCapabilities(
        name=AdapterName.EGOVERSE,
        runtime_backends={"uv", "conda", "apptainer", "existing"},
        supports_multi_gpu_single_node=True,
        supports_resume=True,
        minimum_gpus=1,
        recommended_gpus=1,
        maximum_gpus=8,
        evaluation_adapters=["repository-native"],
    )

    def _resolve(self, spec: ExperimentSpec, gpu_count: int) -> AdapterPlan:
        if spec.native.argv:
            argv = list(spec.native.argv)
        else:
            config_name = str(spec.native.config.get("config_name", "train_zarr_cartesian"))
            argv = [
                "python",
                "egomimic/trainHydra.py",
                "--config-name",
                config_name,
                "hydra/launcher=basic",
                f"trainer.devices={gpu_count}",
                f"seed={spec.train.seed}",
                f"paths.output_dir={RUN_DIR_TOKEN}/artifacts",
                f"trainer.accumulate_grad_batches={spec.train.batch.gradient_accumulation_steps}",
            ]
            if spec.train.learning_rate is not None:
                argv.append(f"model.optimizer.lr={spec.train.learning_rate}")
            if spec.train.max_steps is not None:
                argv.append(f"trainer.max_steps={spec.train.max_steps}")
            argv.extend(f"{key}={_serialize_override(value)}" for key, value in sorted(spec.native.overrides.items()))
        blockers = []
        if any("hydra/launcher=submitit" in argument.casefold() for argument in argv):
            blockers.append("EgoVerse must use hydra/launcher=basic inside the canonical sbatch")
        return self._base_plan(
            argv=argv,
            resume_argv=list(spec.native.resume_argv) or [f"ckpt_path={RESUME_CHECKPOINT_TOKEN}"],
            native_config={
                "canonical_train": spec.train.model_dump(mode="json"),
                "hydra_overrides": spec.native.overrides,
                **spec.native.config,
            },
            environment={},
            checkpoint_globs=["**/*.ckpt"],
            blockers=blockers,
            todos=["Confirm the EgoVerse config-specific batch-size override path before exposing it canonically."],
            warnings=["EgoVerse validation is repository-native rather than an environment rollout suite."],
        )


class DexVerseAdapter(RepositoryAdapter):
    capabilities = AdapterCapabilities(
        name=AdapterName.DEXVERSE,
        runtime_backends={"conda", "apptainer", "existing"},
        supports_multi_gpu_single_node=False,
        supports_resume=True,
        minimum_gpus=1,
        recommended_gpus=1,
        maximum_gpus=1,
        evaluation_adapters=["isaac_lab", "isaac_sim"],
    )

    def _resolve(self, spec: ExperimentSpec, gpu_count: int) -> AdapterPlan:
        blockers: list[str] = []
        if not spec.native.argv:
            blockers.append(
                "TODO: DexVerse defines Isaac Lab environments but no canonical trainer; "
                "select and pin an external Isaac Lab runner in native.argv"
            )
        if spec.train.checkpoint.auto_resume and not spec.native.resume_argv:
            blockers.append("DexVerse auto-resume requires the selected external runner's native.resume_argv")
        return self._base_plan(
            argv=list(spec.native.argv),
            resume_argv=list(spec.native.resume_argv),
            native_config=spec.native.config,
            environment={"SKYNET_ASSIGNED_GPU_COUNT": str(gpu_count)},
            checkpoint_globs=list(spec.native.config.get("checkpoint_globs", ["**/*.pt"])),
            blockers=blockers,
            todos=["Pin the external Isaac Lab trainer commit independently in the source manifest."],
            warnings=[],
        )


class DexMimicGenAdapter(RepositoryAdapter):
    capabilities = AdapterCapabilities(
        name=AdapterName.DEXMIMICGEN,
        runtime_backends={"uv", "conda", "apptainer", "existing"},
        supports_multi_gpu_single_node=False,
        supports_resume=False,
        supports_evaluation_resume=True,
        minimum_gpus=1,
        recommended_gpus=1,
        maximum_gpus=1,
        evaluation_adapters=["robosuite", "mujoco"],
    )

    def _resolve(self, spec: ExperimentSpec, gpu_count: int) -> AdapterPlan:
        config_path = spec.native.config.get("training_config")
        blockers: list[str] = []
        if spec.native.argv:
            argv = list(spec.native.argv)
        elif config_path:
            argv = ["python", "-m", "robomimic.scripts.train", "--config", str(config_path)]
        else:
            argv = []
            blockers.append(
                "DexMimicGen requires native.config.training_config generated into an immutable run directory"
            )
        if not spec.native.config.get("robomimic_revision"):
            blockers.append("DexMimicGen requires a pinned native.config.robomimic_revision")
        return self._base_plan(
            argv=argv,
            resume_argv=list(spec.native.resume_argv),
            native_config=spec.native.config,
            environment={},
            checkpoint_globs=["**/models/*.pth"],
            blockers=blockers,
            todos=["Implement a tested robomimic checkpoint-resume hook before enabling training auto-resume."],
            warnings=["Rollout evaluation needs substantial CPUs and RAM despite single-GPU training."],
        )


class GetZeroAdapter(RepositoryAdapter):
    capabilities = AdapterCapabilities(
        name=AdapterName.GET_ZERO,
        runtime_backends={"conda", "apptainer", "existing"},
        supports_multi_gpu_single_node=True,
        supports_resume=True,
        supports_evaluation_resume=True,
        minimum_gpus=1,
        recommended_gpus=2,
        maximum_gpus=8,
        evaluation_adapters=["isaac_gym"],
    )

    def _resolve(self, spec: ExperimentSpec, gpu_count: int) -> AdapterPlan:
        mode = str(spec.native.config.get("mode", "rl_distill"))
        entrypoints = {
            "rl": "get_zero/rl/train.py",
            "runner": "get_zero/rl/scripts/train_leap_runner.py",
            "rl_distill": "get_zero/rl/rl_distill.py",
        }
        blockers: list[str] = []
        if spec.native.argv:
            argv = list(spec.native.argv)
        elif mode not in entrypoints:
            argv = []
            blockers.append(f"unsupported GET-Zero mode: {mode}")
        else:
            argv = ["python", entrypoints[mode]]
            scopes = spec.native.config.get("override_scopes", [])
            if mode == "rl_distill":
                if len(scopes) != 4 or not all(isinstance(scope, list) for scope in scopes):
                    blockers.append("GET-Zero rl_distill requires four ordered native.config.override_scopes")
                else:
                    for index, scope in enumerate(scopes):
                        if index:
                            argv.append("--")
                        argv.extend(str(value) for value in scope)
            else:
                argv.extend(f"{key}={_serialize_override(value)}" for key, value in sorted(spec.native.overrides.items()))
        if spec.tracking.native_tracking == "disable" and mode == "runner":
            blockers.append("GET-Zero runner uses W&B as orchestration state; native tracking cannot be disabled")
        if spec.train.checkpoint.auto_resume and not spec.native.resume_argv:
            blockers.append("GET-Zero auto-resume requires mode-specific native.resume_argv")
        return self._base_plan(
            argv=argv,
            resume_argv=list(spec.native.resume_argv),
            native_config=spec.native.config,
            environment={"SKYNET_ASSIGNED_GPU_COUNT": str(gpu_count)},
            checkpoint_globs=["**/*.pth", "**/*.pt"],
            blockers=blockers,
            todos=["Retain native W&B coordination while mirroring canonical results to MLflow."],
            warnings=["GET-Zero uses legacy Isaac Gym Preview 4, not Isaac Sim."],
        )


GROOT_GR1_MODALITY_CONFIG_SOURCE = r'''"""Pinned GR1 arms-only modality bridge for GR00T N1.6.

NVIDIA shipped this profile as ``fourier_gr1_arms_only`` with GR00T N1.5.
N1.6 retains the pretrained GR1 embodiment head but no longer registers the
matching data modality.  This capsule restores that adapter-specific contract
and validates the selected immutable dataset before registering it.
"""

import json
import os
from pathlib import Path

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


dataset_path = Path(os.environ["SKYNET_GROOT_DATASET_PATH"]).resolve()
info_path = dataset_path / "meta" / "info.json"
modality_path = dataset_path / "meta" / "modality.json"
if not info_path.is_file() or not modality_path.is_file():
    raise RuntimeError(
        "GR00T GR1 adapter requires meta/info.json and meta/modality.json in "
        f"the dataset root: {dataset_path}"
    )

info = json.loads(info_path.read_text(encoding="utf-8"))
modality = json.loads(modality_path.read_text(encoding="utf-8"))
robot_type = info.get("robot_type")
if robot_type != "GR1ArmsOnly":
    raise RuntimeError(
        "The built-in GR00T GR1 bridge supports robot_type=GR1ArmsOnly only; "
        f"the selected dataset declares {robot_type!r}. Supply a versioned "
        "custom modality config for this GR1 layout."
    )

video_keys = ["ego_view"]
state_keys = ["left_arm", "right_arm", "left_hand", "right_hand"]
action_keys = ["left_arm", "right_arm", "left_hand", "right_hand"]
language_keys = ["annotation.human.action.task_description"]

required = {
    "video": video_keys,
    "state": state_keys,
    "action": action_keys,
    "annotation": ["human.action.task_description"],
}
missing = {
    section: [key for key in keys if key not in modality.get(section, {})]
    for section, keys in required.items()
}
missing = {section: keys for section, keys in missing.items() if keys}
if missing:
    raise RuntimeError(
        "Selected dataset does not match NVIDIA's Fourier GR1 arms-only "
        f"modality profile; missing keys: {missing}"
    )

gr1_arms_only_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=video_keys),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=state_keys,
        sin_cos_embedding_keys=state_keys,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=action_keys,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in action_keys
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=language_keys),
}

register_modality_config(gr1_arms_only_config, embodiment_tag=EmbodimentTag.GR1)
'''

GROOT_N16_LAUNCHER_SOURCE = r'''"""Adapter launcher for GR00T N1.6's non-CLI data settings."""

import os
import runpy

from gr00t.configs import base_config
from gr00t.utils import video_utils


backend = os.environ.get("SKYNET_GROOT_VIDEO_BACKEND")
if not backend:
    raise RuntimeError("SKYNET_GROOT_VIDEO_BACKEND is required by the GR00T adapter")

original_load_dict = base_config.Config.load_dict
original_resolve_backend = video_utils.resolve_backend


def load_dict_with_explicit_video_backend(self, data):
    result = original_load_dict(self, data)
    self.data.video_backend = backend
    return result


def resolve_backend_without_fallback(video_path: str, requested_backend: str) -> str:
    resolved = original_resolve_backend(video_path, requested_backend)
    if resolved != requested_backend:
        raise RuntimeError(
            "GR00T video backend fallback is disabled for reproducibility: "
            f"requested {requested_backend!r}, resolved {resolved!r}. Install the "
            "requested backend or select a different explicit adapter value."
        )
    return resolved


base_config.Config.load_dict = load_dict_with_explicit_video_backend
video_utils.resolve_backend = resolve_backend_without_fallback
runpy.run_path("gr00t/experiment/launch_finetune.py", run_name="__main__")
'''


GROOT_CHECKPOINT_CONTRACT = dict(
    checkpoint_globs=["artifacts/checkpoint-*"],
    checkpoint_candidate_kind="directory",
    checkpoint_basename_regex=r"^checkpoint-[0-9]+$",
    checkpoint_prune_globs=[
        "global_step*",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state_*.pth",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
        "zero_to_fp32.py",
        "latest",
    ],
    training_output_prune_globs=[
        "artifacts/model-*.safetensors",
        "artifacts/model.safetensors",
        "artifacts/model.safetensors.index.json",
        "artifacts/optimizer.pt",
        "artifacts/scheduler.pt",
        "artifacts/rng_state*.pth",
        "artifacts/training_args.bin",
        "artifacts/config.json",
        "artifacts/processor_config.json",
        "artifacts/statistics.json",
        "artifacts/embodiment_id.json",
        "artifacts/processor",
        "artifacts/experiment_cfg",
    ],
    checkpoint_inference_required_globs=[
        "config.json",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    ],
)


class GrootAdapter(RepositoryAdapter):
    supported_canonical_fields = frozenset(
        {
            "train.learning_rate",
            "train.batch.declared_semantics",
            "train.batch.value",
            "train.batch.gradient_accumulation_steps",
            "train.num_workers_per_rank",
            "train.max_steps",
            "train.checkpoint.save_every_steps",
            "train.checkpoint.keep_last",
        }
    )
    capabilities = AdapterCapabilities(
        name=AdapterName.GROOT,
        runtime_backends={"uv", "conda", "apptainer", "existing"},
        supports_multi_gpu_single_node=True,
        supports_resume=True,
        supports_evaluation_resume=True,
        minimum_gpus=1,
        recommended_gpus=4,
        maximum_gpus=8,
        evaluation_adapters=["libero", "mujoco", "isaac_sim"],
    )

    def _resolve(self, spec: ExperimentSpec, gpu_count: int) -> AdapterPlan:
        dataset_path = spec.native.config.get("dataset_path")
        base_model_path = spec.native.config.get("base_model_path")
        embodiment_tag = spec.native.config.get("embodiment_tag")
        modality_config_path = spec.native.config.get("modality_config_path")
        video_backend = str(spec.native.config.get("video_backend") or "opencv")
        blockers: list[str] = []
        environment: dict[str, str] = {
            "SKYNET_GROOT_VIDEO_BACKEND": video_backend,
        }
        launcher_path = "adapter-support/groot-n16-launcher.py"
        capsule_files: dict[str, str] = {
            launcher_path: GROOT_N16_LAUNCHER_SOURCE,
        }
        resolved_modality_profile: str | None = None
        if video_backend not in {"opencv", "torchcodec", "decord", "ffmpeg"}:
            blockers.append(f"unsupported GR00T video backend: {video_backend}")
        if spec.native.argv:
            argv = list(spec.native.argv)
        else:
            argv = ["python", f"{RUN_DIR_TOKEN}/{launcher_path}"]
            if not base_model_path:
                blockers.append("GR00T requires native.config.base_model_path")
            else:
                argv.extend(["--base-model-path", str(base_model_path)])
            if not dataset_path:
                blockers.append("GR00T requires native.config.dataset_path")
            else:
                argv.extend(["--dataset-path", str(dataset_path)])
            if not embodiment_tag:
                blockers.append("GR00T requires native.config.embodiment_tag")
            else:
                argv.extend(["--embodiment-tag", str(embodiment_tag)])
            if modality_config_path:
                argv.extend(["--modality-config-path", str(modality_config_path)])
            elif str(embodiment_tag or "").strip().casefold() == "gr1":
                resolved_modality_profile = "nvidia-fourier-gr1-arms-only"
                support_path = "adapter-support/groot-gr1-modality.py"
                argv.extend(
                    ["--modality-config-path", f"{RUN_DIR_TOKEN}/{support_path}"]
                )
                capsule_files[support_path] = GROOT_GR1_MODALITY_CONFIG_SOURCE
                if dataset_path:
                    environment["SKYNET_GROOT_DATASET_PATH"] = str(dataset_path)
            elif str(embodiment_tag or "").strip().casefold() == "new_embodiment":
                blockers.append(
                    "GR00T NEW_EMBODIMENT requires native.config.modality_config_path"
                )
            argv.extend(["--output-dir", f"{RUN_DIR_TOKEN}/artifacts", "--num-gpus", str(gpu_count)])
            if spec.train.max_steps is not None:
                argv.extend(["--max-steps", str(spec.train.max_steps)])
            if spec.train.learning_rate is not None:
                argv.extend(["--learning-rate", str(spec.train.learning_rate)])
            batch = spec.train.batch
            if batch.declared_semantics == "per_device":
                global_batch_size = batch.value * gpu_count
            elif batch.declared_semantics == "global_effective":
                if batch.value % batch.gradient_accumulation_steps:
                    blockers.append(
                        "GR00T global_effective batch must be divisible by gradient accumulation steps"
                    )
                global_batch_size = max(1, batch.value // batch.gradient_accumulation_steps)
            else:
                global_batch_size = batch.value
            argv.extend(
                [
                    "--global-batch-size", str(global_batch_size),
                    "--gradient-accumulation-steps", str(batch.gradient_accumulation_steps),
                    "--dataloader-num-workers", str(spec.train.num_workers_per_rank),
                    "--save-steps", str(spec.train.checkpoint.save_every_steps),
                    "--save-total-limit", str(spec.train.checkpoint.keep_last),
                ]
            )
            for key, value in sorted(spec.native.overrides.items()):
                argv.extend([f"--{key.replace('_', '-')}", _serialize_override(value)])
            if gpu_count > 1:
                argv = ["torchrun", "--standalone", f"--nproc_per_node={gpu_count}", *argv[1:]]
        return self._base_plan(
            argv=argv,
            resume_argv=list(spec.native.resume_argv) or ["--resume-from-checkpoint"],
            native_config={
                "canonical_train": spec.train.model_dump(mode="json"),
                **spec.native.config,
                **(
                    {"resolved_modality_profile": resolved_modality_profile}
                    if resolved_modality_profile
                    else {}
                ),
            },
            environment=environment,
            capsule_files=capsule_files,
            **GROOT_CHECKPOINT_CONTRACT,
            blockers=blockers,
            todos=[],
            warnings=["GR00T reports nondeterministic augmentation variance even when seeds are fixed."],
        )


_OPENPI_BATCH_COMPATIBILITY = AdapterBatchCompatibility(
    allowed_semantics=[
        "per_device",
        "global_before_accumulation",
        "global_effective",
    ],
    multi_gpu_allowed_semantics=[
        "global_before_accumulation",
        "global_effective",
    ],
    supports_gradient_accumulation=False,
    batch_size_divisible_by="resolved_gpu_count",
)


def _batch_compatibility_blockers(
    spec: ExperimentSpec,
    compatibility: AdapterBatchCompatibility | None,
    *,
    adapter_slug: str,
    gpu_count: int,
) -> list[str]:
    if compatibility is None:
        return []

    batch = spec.train.batch
    blockers: list[str] = []
    batch_size_explicit = spec.intent.is_explicit("train.batch.value")
    semantics_explicit = spec.intent.is_explicit(
        "train.batch.declared_semantics"
    )
    accumulation_explicit = spec.intent.is_explicit(
        "train.batch.gradient_accumulation_steps"
    )

    if (
        accumulation_explicit
        and not compatibility.supports_gradient_accumulation
        and batch.gradient_accumulation_steps != 1
    ):
        blockers.append(
            f"{adapter_slug} training does not support gradient accumulation"
        )
    if semantics_explicit and batch.declared_semantics not in set(
        compatibility.allowed_semantics
    ):
        blockers.append(
            f"{adapter_slug} does not support {batch.declared_semantics} batch semantics"
        )
    if (
        batch_size_explicit
        and gpu_count > 1
        and batch.declared_semantics
        not in set(compatibility.multi_gpu_allowed_semantics)
    ):
        choices = " or ".join(compatibility.multi_gpu_allowed_semantics)
        if batch.declared_semantics == "per_device" and choices:
            blockers.append(
                f"{adapter_slug} --batch-size is global; choose {choices} batch "
                "semantics for multi-GPU training"
            )
        else:
            blockers.append(
                f"{adapter_slug} {batch.declared_semantics} batch semantics are not "
                f"valid for {gpu_count} resolved GPUs"
            )
    if (
        batch_size_explicit
        and compatibility.batch_size_divisible_by == "resolved_gpu_count"
        and batch.value % gpu_count
    ):
        blockers.append(
            f"{adapter_slug} batch size must be divisible by the resolved GPU count "
            f"({gpu_count})"
        )
    return blockers


class OpenPiAdapter(RepositoryAdapter):
    supported_canonical_fields = frozenset(
        {
            "train.batch.declared_semantics",
            "train.batch.value",
            "train.batch.gradient_accumulation_steps",
            "train.num_workers_per_rank",
            "train.max_steps",
            "train.precision",
            "train.seed",
            "train.checkpoint.save_every_steps",
        }
    )
    capabilities = AdapterCapabilities(
        name=AdapterName.OPENPI,
        runtime_backends={"uv", "conda", "apptainer", "existing"},
        supports_multi_gpu_single_node=True,
        supports_resume=True,
        supports_evaluation_resume=True,
        minimum_gpus=1,
        recommended_gpus=1,
        maximum_gpus=8,
        evaluation_adapters=["libero", "mujoco"],
    )

    def validate(self, spec: ExperimentSpec) -> list[str]:
        blockers = super().validate(spec)
        gpu_count = _gpu_count(spec, self.capabilities)
        blockers.extend(
            _batch_compatibility_blockers(
                spec,
                _OPENPI_BATCH_COMPATIBILITY,
                adapter_slug="openpi",
                gpu_count=gpu_count,
            )
        )
        return blockers

    def _resolve(self, spec: ExperimentSpec, gpu_count: int) -> AdapterPlan:
        config_name = spec.native.config.get("config_name")
        blockers: list[str] = []
        if spec.native.argv:
            argv = list(spec.native.argv)
        elif not config_name:
            argv = []
            blockers.append("openpi requires native.config.config_name")
        else:
            experiment_name = spec.identity.variant or spec.identity.experiment
            argv = [
                "python",
                "scripts/train.py",
                str(config_name),
                "--exp-name",
                experiment_name,
                "--checkpoint-base-dir",
                f"{RUN_DIR_TOKEN}/checkpoints",
            ]
            for key, value in sorted(spec.native.overrides.items()):
                argv.extend([f"--{key.replace('_', '-')}", _serialize_override(value)])
        dataset_path = spec.native.config.get("dataset_path")
        norm_path = spec.native.config.get("dataset_norm_stats_path")
        if dataset_path:
            if spec.native.argv:
                blockers.append("Selected OpenPI dataset input is unsupported with a custom training command; clear the custom command or remove the dataset selection")
            if not norm_path:
                blockers.append("Dataset normalization file is required when selecting OpenPI data. Enter an absolute path to norm_stats.json computed for this dataset and training config")
            for label, path in (("Dataset", dataset_path), ("Normalization", norm_path)):
                if path and not PurePosixPath(str(path)).is_absolute():
                    blockers.append(f"{label} path must be an absolute compute-node path")
            if any(str(key).replace("_", "-").startswith(("data.", "data-", "assets-base-dir")) for key in spec.native.overrides):
                blockers.append("OpenPI data/asset overrides conflict with the selected dataset; use the dataset and normalization inputs instead")
            if argv and not spec.native.argv:
                argv = ["python", f"{RUN_DIR_TOKEN}/adapter-support/openpi-dataset.py",
                        "--dataset-root", str(dataset_path), "--norm-stats", str(norm_path or ""), "--", *argv[2:]]
        elif norm_path:
            blockers.append("A normalization file requires an explicit OpenPI dataset path or bundle")
        warnings = ["openpi supports multiple GPUs on one node but not multi-node JAX training."]
        return self._base_plan(
            argv=argv,
            resume_argv=list(spec.native.resume_argv) or ["--resume"],
            native_config={"canonical_train": spec.train.model_dump(mode="json"), **spec.native.config},
            environment={"XLA_PYTHON_CLIENT_MEM_FRACTION": str(spec.native.config.get("xla_memory_fraction", 0.9))},
            checkpoint_globs=["checkpoints/*/*/*"],
            checkpoint_candidate_kind="directory",
            checkpoint_basename_regex=r"^[0-9]+$",
            checkpoint_prune_globs=["train_state"],
            checkpoint_inference_required_globs=["params", "assets"],
            blockers=blockers,
            todos=["Evaluation runs through a separately versioned policy-server/client adapter."],
            warnings=warnings,
        )


_REGISTRY: dict[tuple[str, int], RepositoryAdapter] = {}


def register_adapter(adapter: RepositoryAdapter) -> None:
    key = (str(adapter.capabilities.name), adapter.capabilities.version)
    if key in _REGISTRY:
        raise RuntimeError(f"adapter already registered: {key}")
    _REGISTRY[key] = adapter


for _adapter in (
    GenericAdapter(),
    EgoVerseAdapter(),
    DexVerseAdapter(),
    DexMimicGenAdapter(),
    GetZeroAdapter(),
    GrootAdapter(),
    OpenPiAdapter(),
):
    register_adapter(_adapter)


def get_adapter(name: AdapterName | str, version: int = 1) -> RepositoryAdapter:
    normalized = str(name)
    try:
        return _REGISTRY[(normalized, version)]
    except KeyError as error:
        raise AdapterError(f"unsupported adapter version: {normalized}@{version}") from error


def list_adapter_capabilities() -> list[AdapterCapabilities]:
    return [adapter.capabilities.model_copy(deep=True) for _, adapter in sorted(_REGISTRY.items(), key=lambda item: item[0])]


class ArgumentBinding(CanonicalModel):
    flag: str
    style: Literal["separate", "equals", "hydra", "boolean"] = "separate"
    omit_if_none: bool = True
    value_map: dict[str, str] = Field(default_factory=dict)
    false_flag: str | None = None
    companion_arguments: list[str] = Field(default_factory=list)

    @field_validator("flag", "false_flag")
    @classmethod
    def validate_flag(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or any(character in value for character in ("\x00", "\n", "\r")):
            raise ValueError("argument flags must be non-empty single-line strings")
        return value

    @field_validator("value_map")
    @classmethod
    def validate_value_map(cls, value: dict[str, str]) -> dict[str, str]:
        for source, target in value.items():
            if not source or any(character in source for character in ("\x00", "\n", "\r")):
                raise ValueError("argument value-map keys must be non-empty single-line strings")
            if any(character in target for character in ("\x00", "\n", "\r")):
                raise ValueError("argument value-map values must be single-line strings")
        return dict(sorted(value.items()))

    @field_validator("companion_arguments")
    @classmethod
    def validate_companion_arguments(cls, value: list[str]) -> list[str]:
        if any(
            not argument
            or any(character in argument for character in ("\x00", "\n", "\r"))
            for argument in value
        ):
            raise ValueError("companion arguments must be non-empty single-line strings")
        return value

    @model_validator(mode="after")
    def validate_mapping_style(self) -> "ArgumentBinding":
        if self.style == "boolean" and self.value_map:
            raise ValueError("boolean argument bindings cannot declare a value map")
        if self.false_flag is not None and self.style != "boolean":
            raise ValueError("false_flag is only valid for boolean argument bindings")
        if self.false_flag == self.flag:
            raise ValueError("boolean true and false flags must differ")
        return self


AdapterInputKind = Literal["string", "integer", "number", "boolean", "json", "string_list"]
_PROTOTYPE_LIKE_SEGMENTS = {"__proto__", "prototype", "constructor"}


def _validate_adapter_input_path(value: str) -> str:
    if value in {"native.argv", "native.resume_argv"}:
        return value
    if value.startswith("native.config."):
        suffix = value.removeprefix("native.config.")
        segments = suffix.split(".")
        if not suffix or any(
            not re.fullmatch(r"[a-z][a-z0-9_]*", segment)
            or segment.casefold() in _PROTOTYPE_LIKE_SEGMENTS
            for segment in segments
        ):
            raise ValueError("native config input paths require safe lowercase dotted segments")
        return value
    if value.startswith("native.overrides."):
        suffix = value.removeprefix("native.overrides.")
        segments = suffix.split(".")
        if not suffix or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", segment)
            or segment.casefold() in _PROTOTYPE_LIKE_SEGMENTS
            for segment in segments
        ):
            raise ValueError("native override input paths require a non-empty safe dotted suffix")
        return value
    raise ValueError(
        "adapter input path must be native.argv, native.resume_argv, "
        "native.config.<path>, or native.overrides.<key>"
    )


class RepositoryChoiceMetadataField(CanonicalModel):
    source_path: str = Field(min_length=1, max_length=255)
    canonical_path: Literal[
        "train.learning_rate",
        "train.batch.declared_semantics",
        "train.batch.value",
        "train.batch.gradient_accumulation_steps",
        "train.num_workers_per_rank",
        "train.max_steps",
        "train.max_epochs",
        "train.precision",
    ]
    value_map: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*",
            normalized,
        ):
            raise ValueError(
                "repository metadata source paths must be dotted Python identifiers"
            )
        return normalized


class RepositoryChoiceSource(CanonicalModel):
    kind: Literal["python_static_registry", "python_enum"]
    entrypoint: str = Field(min_length=1, max_length=512)
    supporting_files: list[str] = Field(default_factory=list, max_length=15)
    registry: str = Field(min_length=1, max_length=128)
    constructor: str | None = Field(default=None, min_length=1, max_length=128)
    value_keyword: str = Field(default="name", min_length=1, max_length=128)
    metadata_fields: list[RepositoryChoiceMetadataField] = Field(
        default_factory=list, max_length=16
    )

    @staticmethod
    def _validate_repository_path(value: str) -> str:
        normalized = value.strip()
        parts = normalized.split("/")
        if (
            not normalized.endswith(".py")
            or normalized.startswith("/")
            or "\\" in normalized
            or any(part in {"", ".", ".."} for part in parts)
            or any(character in normalized for character in ("\x00", "\n", "\r"))
        ):
            raise ValueError("repository choice files must be safe relative Python paths")
        return normalized

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        return cls._validate_repository_path(value)

    @field_validator("supporting_files")
    @classmethod
    def validate_supporting_files(cls, value: list[str]) -> list[str]:
        normalized = [cls._validate_repository_path(item) for item in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("repository choice supporting_files must be unique")
        return normalized

    @field_validator("registry", "value_keyword")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("repository choice registry identifiers must be Python identifiers")
        return value

    @field_validator("constructor")
    @classmethod
    def validate_optional_identifier(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("repository choice constructor must be a Python identifier")
        return value

    @model_validator(mode="after")
    def validate_file_set(self) -> "RepositoryChoiceSource":
        if self.entrypoint in self.supporting_files:
            raise ValueError("choice_source entrypoint must not be repeated in supporting_files")
        if self.kind == "python_static_registry" and self.constructor is None:
            raise ValueError("python_static_registry choice sources require constructor")
        if self.kind != "python_static_registry" and self.metadata_fields:
            raise ValueError("repository choice metadata requires python_static_registry")
        if self.kind == "python_enum" and self.value_keyword not in {"name", "value"}:
            raise ValueError("python_enum value_keyword must be name or value")
        canonical_paths = [field.canonical_path for field in self.metadata_fields]
        if len(canonical_paths) != len(set(canonical_paths)):
            raise ValueError("repository choice metadata canonical paths must be unique")
        return self


class RepositoryYamlMetadataField(RepositoryChoiceMetadataField):
    source_file: str = Field(min_length=1, max_length=512)

    @field_validator("source_file")
    @classmethod
    def validate_source_file(cls, value: str) -> str:
        normalized = value.strip()
        path = PurePosixPath(normalized)
        if (
            path.suffix not in {".yaml", ".yml"}
            or path.is_absolute()
            or "\\" in normalized
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(character in normalized for character in ("\x00", "\n", "\r"))
        ):
            raise ValueError(
                "repository YAML metadata files must be safe relative YAML paths"
            )
        return normalized


class RepositoryYamlChoiceSource(CanonicalModel):
    """Adapter-declared scalar mappings for one statically composed YAML config."""

    kind: Literal["yaml_static_mapping"] = "yaml_static_mapping"
    entrypoint: str = Field(min_length=1, max_length=512)
    supporting_files: list[str] = Field(default_factory=list, max_length=15)
    choice: str = Field(min_length=1, max_length=128)
    allow_custom: bool = False
    metadata_fields: list[RepositoryYamlMetadataField] = Field(
        min_length=1, max_length=16
    )

    @staticmethod
    def _validate_repository_path(value: str) -> str:
        normalized = value.strip()
        path = PurePosixPath(normalized)
        if (
            path.suffix not in {".yaml", ".yml"}
            or path.is_absolute()
            or "\\" in normalized
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(character in normalized for character in ("\x00", "\n", "\r"))
        ):
            raise ValueError(
                "repository YAML source files must be safe relative YAML paths"
            )
        return normalized

    @field_validator("entrypoint")
    @classmethod
    def validate_entrypoint(cls, value: str) -> str:
        return cls._validate_repository_path(value)

    @field_validator("supporting_files")
    @classmethod
    def validate_supporting_files(cls, value: list[str]) -> list[str]:
        normalized = [cls._validate_repository_path(item) for item in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("repository YAML supporting_files must be unique")
        return normalized

    @field_validator("choice")
    @classmethod
    def validate_choice(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", normalized):
            raise ValueError("repository YAML choices must be safe config names")
        return normalized

    @model_validator(mode="after")
    def validate_file_set(self) -> "RepositoryYamlChoiceSource":
        if self.entrypoint in self.supporting_files:
            raise ValueError(
                "YAML choice_source entrypoint must not be repeated in supporting_files"
            )
        files = {self.entrypoint, *self.supporting_files}
        undeclared = sorted(
            {
                field.source_file
                for field in self.metadata_fields
                if field.source_file not in files
            }
        )
        if undeclared:
            raise ValueError(
                "repository YAML metadata fields require declared source files: "
                + ", ".join(undeclared)
            )
        canonical_paths = [field.canonical_path for field in self.metadata_fields]
        if len(canonical_paths) != len(set(canonical_paths)):
            raise ValueError("repository YAML metadata canonical paths must be unique")
        return self


class RepositoryYamlCatalogMetadataField(CanonicalModel):
    source_path: str = Field(min_length=1, max_length=255)
    canonical_path: Literal[
        "train.learning_rate",
        "train.batch.declared_semantics",
        "train.batch.value",
        "train.batch.gradient_accumulation_steps",
        "train.num_workers_per_rank",
        "train.max_steps",
        "train.max_epochs",
        "train.precision",
    ]
    value_map: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        normalized = value.strip()
        segments = normalized.split(".")
        if not normalized or any(
            segment != "*" and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", segment)
            for segment in segments
        ):
            raise ValueError(
                "repository YAML catalog metadata paths must be dotted identifiers "
                "with optional whole-segment wildcards"
            )
        return normalized


class RepositoryYamlCatalogSource(CanonicalModel):
    """Discover and statically compose a bounded catalog of repository YAML configs."""

    kind: Literal["yaml_static_catalog"] = "yaml_static_catalog"
    directory: str = Field(min_length=1, max_length=512)
    filename_pattern: str = Field(min_length=1, max_length=128)
    allow_custom: bool = False
    metadata_fields: list[RepositoryYamlCatalogMetadataField] = Field(
        min_length=1, max_length=16
    )

    @field_validator("directory")
    @classmethod
    def validate_directory(cls, value: str) -> str:
        normalized = value.strip().strip("/")
        path = PurePosixPath(normalized)
        if (
            not normalized
            or path == PurePosixPath(".")
            or path.is_absolute()
            or "\\" in normalized
            or any(part in {"", ".", ".."} for part in path.parts)
            or any(character in normalized for character in ("\x00", "\n", "\r"))
        ):
            raise ValueError("repository YAML catalog directories must be safe relative paths")
        return normalized

    @field_validator("filename_pattern")
    @classmethod
    def validate_filename_pattern(cls, value: str) -> str:
        normalized = value.strip()
        if (
            "/" in normalized
            or "\\" in normalized
            or not re.fullmatch(r"[A-Za-z0-9_.?*-]+\.ya?ml", normalized)
        ):
            raise ValueError(
                "repository YAML catalog filename patterns must be safe YAML basenames"
            )
        return normalized

    @model_validator(mode="after")
    def validate_metadata_fields(self) -> "RepositoryYamlCatalogSource":
        canonical_paths = [field.canonical_path for field in self.metadata_fields]
        if len(canonical_paths) != len(set(canonical_paths)):
            raise ValueError("repository YAML catalog canonical paths must be unique")
        return self


class DataBundleInputBinding(CanonicalModel):
    role: str = Field(min_length=1, max_length=128)
    position: int = Field(default=0, ge=0)
    formats: list[str] = Field(default_factory=list, max_length=32)
    value_path: Literal["version.path", "mount_path"] = "version.path"

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", normalized):
            raise ValueError("data binding role must be a safe registry role")
        return normalized

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item or any(character in item for character in ("\x00", "\n", "\r")) for item in normalized):
            raise ValueError("data binding formats must be nonempty single-line strings")
        if len({item.casefold() for item in normalized}) != len(normalized):
            raise ValueError("data binding formats must be unique")
        return normalized


class AdapterInputField(CanonicalModel):
    path: str
    label: str = Field(min_length=1, max_length=128)
    kind: AdapterInputKind
    required: bool = False
    default: Any | None = None
    choices: list[Any] = Field(default_factory=list)
    choice_source: (
        RepositoryChoiceSource
        | RepositoryYamlChoiceSource
        | RepositoryYamlCatalogSource
        | None
    ) = None
    data_binding: DataBundleInputBinding | None = None
    help: str = Field(default="", max_length=1000)
    tutorial_value: Any | None = None
    sensitive: bool = Field(
        default=False,
        deprecated=True,
        description=(
            "Deprecated presentation-only masking hint. Skynet does not provide secret "
            "storage; submitted values remain part of the reproducibility snapshot."
        ),
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_adapter_input_path(value)

    @staticmethod
    def _matches_kind(kind: AdapterInputKind, value: Any) -> bool:
        if kind == "string":
            return isinstance(value, str)
        if kind == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if kind == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if kind == "boolean":
            return isinstance(value, bool)
        if kind == "string_list":
            return isinstance(value, list) and all(isinstance(item, str) for item in value)
        try:
            canonical_sha256(value)
        except (TypeError, ValueError):
            return False
        return True

    @model_validator(mode="after")
    def validate_values(self) -> "AdapterInputField":
        if self.__dict__.get("sensitive", False) and (
            self.default is not None
            or self.tutorial_value is not None
            or self.choices
            or self.choice_source is not None
        ):
            raise ValueError(
                "sensitive input fields cannot declare choices, default, or tutorial_value; "
                "sensitive is presentation metadata only and does not provide secret storage"
            )
        if self.choice_source is not None and self.choices:
            raise ValueError("adapter input fields cannot declare both choices and choice_source")
        if self.choice_source is not None and self.kind != "string":
            raise ValueError("repository-discovered choices currently require input kind string")
        if self.data_binding is not None and self.kind != "string":
            raise ValueError("data bundle input bindings currently require input kind string")
        if self.data_binding is not None and self.default is not None:
            raise ValueError(
                "data-bound adapter inputs cannot declare a fallback default; "
                "require a compatible bundle or an explicit user value"
            )
        for field_name in ("default", "tutorial_value"):
            value = getattr(self, field_name)
            if value is not None and not self._matches_kind(self.kind, value):
                raise ValueError(f"{field_name} must match input field kind {self.kind}")
        choice_hashes: list[str] = []
        for choice in self.choices:
            if not self._matches_kind(self.kind, choice):
                raise ValueError(f"choices must match input field kind {self.kind}")
            choice_hashes.append(canonical_sha256(choice))
        if len(choice_hashes) != len(set(choice_hashes)):
            raise ValueError("input field choices must be unique")
        allowed = set(choice_hashes)
        for field_name in ("default", "tutorial_value"):
            value = getattr(self, field_name)
            if value is not None and allowed and canonical_sha256(value) not in allowed:
                raise ValueError(f"{field_name} must be one of the declared choices")
        return self


class PreparationStepTemplate(PreparationStep):
    enabled_when: dict[str, list[Any]] = Field(default_factory=dict)
    consumer_argv: list[str] = Field(default_factory=list)

    @field_validator("enabled_when")
    @classmethod
    def validate_enabled_when(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for path, choices in value.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", path):
                raise ValueError(f"invalid preparation condition path: {path}")
            if not choices:
                raise ValueError("preparation condition choices must not be empty")
            for choice in choices:
                canonical_sha256(choice)
        return dict(sorted(value.items()))

    @field_validator("consumer_argv")
    @classmethod
    def validate_consumer_argv(cls, value: list[str]) -> list[str]:
        if any(any(character in item for character in ("\x00", "\n", "\r")) for item in value):
            raise ValueError("preparation consumer argv must contain single-line strings")
        return value


class RepositoryArgumentValidation(CanonicalModel):
    argv: list[str] = Field(min_length=1)
    timeout_seconds: int = Field(default=120, ge=1, le=300)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: list[str]) -> list[str]:
        if any(
            not argument
            or any(character in argument for character in ("\x00", "\n", "\r"))
            for argument in value
        ):
            raise ValueError(
                "repository argument-validation argv must contain non-empty "
                "single-line strings"
            )
        if value.count(GENERATED_ARGS_TOKEN) != 1 or any(
            GENERATED_ARGS_TOKEN in argument and argument != GENERATED_ARGS_TOKEN
            for argument in value
        ):
            raise ValueError(
                "repository argument-validation argv must contain exactly one "
                f"{GENERATED_ARGS_TOKEN} element"
            )
        return value


class CommandTemplate(CanonicalModel):
    argv: list[str] = Field(default_factory=list)
    static_args: list[str] = Field(default_factory=list)
    parameter_flags: dict[str, ArgumentBinding] = Field(default_factory=dict)
    resume_argv: list[str] = Field(default_factory=list)
    retry_clean_argv: list[str] = Field(default_factory=list)
    checkpoint_globs: list[str] = Field(default_factory=list)
    checkpoint_candidate_kind: CheckpointCandidateKind = "any"
    checkpoint_basename_regex: str | None = None
    checkpoint_prune_globs: list[str] = Field(default_factory=list)
    training_output_prune_globs: list[str] = Field(default_factory=list)
    checkpoint_inference_required_globs: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    required_values: list[str] = Field(default_factory=list)
    supported_canonical_fields: list[str] = Field(default_factory=list)
    input_fields: list[AdapterInputField] = Field(default_factory=list)
    preparation_steps: list[PreparationStepTemplate] = Field(default_factory=list)
    capsule_files: dict[str, str] = Field(default_factory=dict)
    batch_compatibility: AdapterBatchCompatibility | None = None
    progress: TrainingProgressContract | None = None
    argument_validation: RepositoryArgumentValidation | None = None
    native_tracking: list[NativeTrackingIntegration] = Field(default_factory=list)

    @field_validator(
        "argv", "static_args", "resume_argv", "retry_clean_argv"
    )
    @classmethod
    def validate_strings(cls, value: list[str]) -> list[str]:
        if any(any(character in item for character in ("\x00", "\n", "\r")) for item in value):
            raise ValueError("command template values must be single-line strings")
        return value

    @field_validator(
        "checkpoint_globs",
        "checkpoint_prune_globs",
        "training_output_prune_globs",
        "checkpoint_inference_required_globs",
    )
    @classmethod
    def validate_checkpoint_globs(cls, value: list[str]) -> list[str]:
        return [_validate_checkpoint_glob(item) for item in value]

    @field_validator("checkpoint_basename_regex")
    @classmethod
    def validate_checkpoint_basename_regex(cls, value: str | None) -> str | None:
        return _validate_checkpoint_basename_regex(value)

    @field_validator("capsule_files")
    @classmethod
    def validate_capsule_files(cls, value: dict[str, str]) -> dict[str, str]:
        return _validate_capsule_files(value)

    @field_validator("parameter_flags")
    @classmethod
    def validate_parameter_paths(cls, value: dict[str, ArgumentBinding]) -> dict[str, ArgumentBinding]:
        for path in value:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*", path):
                raise ValueError(f"invalid canonical parameter path: {path}")
        return value

    @model_validator(mode="after")
    def validate_native_tracking(self) -> "CommandTemplate":
        providers: set[str] = set()
        for integration in self.native_tracking:
            if integration.provider in providers:
                raise ValueError(
                    f"duplicate native tracking integration: {integration.provider}"
                )
            providers.add(integration.provider)
            missing = sorted(
                set(integration.parameter_paths.values()) - set(self.parameter_flags)
            )
            if missing:
                raise ValueError(
                    "native tracking parameter paths require parameter_flags mappings: "
                    + ", ".join(missing)
                )
        return self

    @field_validator("required_values", "supported_canonical_fields")
    @classmethod
    def validate_canonical_paths(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("canonical value paths must be unique")
        for path in value:
            if not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", path):
                raise ValueError(f"invalid canonical value path: {path}")
        return value

    @model_validator(mode="after")
    def align_required_input_fields(self) -> "CommandTemplate":
        paths = [field.path for field in self.input_fields]
        if len(paths) != len(set(paths)):
            raise ValueError("adapter input field paths must be unique")
        required_values = list(self.required_values)
        for field in self.input_fields:
            if field.required and field.path not in required_values:
                required_values.append(field.path)
        step_ids = [step.id for step in self.preparation_steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("adapter preparation step IDs must be unique")
        self.required_values = required_values
        self.supported_canonical_fields = sorted(
            set(self.supported_canonical_fields) | set(self.parameter_flags)
        )
        return self


class AdapterRuntimePolicy(CanonicalModel):
    allowed_backends: set[Literal["uv", "conda", "apptainer", "existing"]]
    recommended_backend: Literal["uv", "conda", "apptainer", "existing"] | None = None

    @field_serializer("allowed_backends")
    def serialize_allowed_backends(
        self,
        value: set[Literal["uv", "conda", "apptainer", "existing"]],
    ) -> list[str]:
        return sorted(value)

    @model_validator(mode="after")
    def validate_recommendation(self) -> "AdapterRuntimePolicy":
        if self.recommended_backend and self.recommended_backend not in self.allowed_backends:
            raise ValueError("recommended runtime must be in allowed_backends")
        return self


class EvaluationAdapterMetadata(CanonicalModel):
    environment: str
    suites: list[str] = Field(default_factory=list)
    runtime_profile_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$",
    )
    command: CommandTemplate | None = None
    result_schema: str = "skynet.evaluation-result/v1"
    enabled_when: dict[str, list[Any]] = Field(default_factory=dict)

    @field_validator("enabled_when")
    @classmethod
    def validate_enabled_when(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for field_path, allowed_values in value.items():
            if not field_path.strip():
                raise ValueError("evaluation enabled_when field paths cannot be empty")
            if not allowed_values:
                raise ValueError(
                    f"evaluation enabled_when values cannot be empty for {field_path}"
                )
        return value


class AdapterHyperparameterDefaults(CanonicalModel):
    learning_rate: float | None = Field(default=None, gt=0)
    batch_semantics: Literal[
        "per_device",
        "global_before_accumulation",
        "global_effective",
        "repository_native",
    ] | None = None
    batch_size: int | None = Field(default=None, ge=1)
    gradient_accumulation_steps: int | None = Field(default=None, ge=1)
    num_workers_per_rank: int | None = Field(default=None, ge=0)
    max_steps: int | None = Field(default=None, ge=1)
    max_epochs: int | None = Field(default=None, ge=1)
    seed: int | None = None
    precision: Literal["bf16", "fp16", "fp32"] | None = None
    values: dict[str, Any] = Field(default_factory=dict)


class AdapterGPURecommendation(CanonicalModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,95}$")
    enabled_when: dict[str, list[Any]] = Field(min_length=1)
    minimum_gpus: int = Field(ge=1)
    recommended_gpus: int = Field(ge=1)
    maximum_gpus: int = Field(ge=1)
    gpu_type: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("enabled_when")
    @classmethod
    def validate_enabled_when(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for path, choices in value.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", path):
                raise ValueError(f"invalid GPU recommendation condition path: {path}")
            if not choices:
                raise ValueError("GPU recommendation condition choices must not be empty")
            for choice in choices:
                canonical_sha256(choice)
        return dict(sorted(value.items()))

    @model_validator(mode="after")
    def validate_gpu_profiles(self) -> "AdapterGPURecommendation":
        if not self.minimum_gpus <= self.recommended_gpus <= self.maximum_gpus:
            raise ValueError(
                "GPU recommendation counts must satisfy minimum <= recommended <= maximum"
            )
        return self

    @field_validator("gpu_type")
    @classmethod
    def validate_gpu_type(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value):
            raise ValueError("recommended GPU type must be a canonical lowercase identifier")
        return value


class AdapterResourceDefaults(CanonicalModel):
    gateway: str | None = None
    queue_policy: str | None = None
    node_mode: Literal["auto", "manual"] | None = None
    node: str | None = None
    gpu_mode: Literal["auto", "explicit"] | None = None
    gpu_count: int | None = Field(default=None, ge=1)
    gpu_type: str | None = None
    gpu_profile: Literal["minimum", "recommended", "maximum-throughput"] | None = None
    cpus_per_task: int | None = Field(default=None, ge=1)
    memory_gb: int | None = Field(default=None, ge=1)
    time_limit: str | None = None
    gpu_recommendations: list[AdapterGPURecommendation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_gpu_default(self) -> "AdapterResourceDefaults":
        if self.gpu_mode == "explicit" and self.gpu_count is None:
            raise ValueError("explicit default GPU mode requires gpu_count")
        if self.node_mode == "manual" and not self.node:
            raise ValueError("manual default node mode requires node")
        rule_ids = [rule.id for rule in self.gpu_recommendations]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("GPU recommendation rule IDs must be unique")
        return self


class AdapterCheckpointDefaults(CanonicalModel):
    save_every_steps: int | None = Field(default=None, ge=1)
    save_before_timeout_seconds: int | None = Field(default=None, ge=60, le=1800)
    keep_last: int | None = Field(default=None, ge=1, le=3)
    auto_resume: bool | None = None
    max_attempts: int | None = Field(default=None, ge=1, le=100)
    final_selector: Literal["best", "latest"] | None = None
    remove_training_state_after_success: bool | None = None


class AdapterTrackingDefaults(CanonicalModel):
    enabled: bool | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str | None = None
    native_tracking: Literal["preserve", "disable"] | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    offline_spool: bool | None = None


class AdapterDefaults(CanonicalModel):
    workdir: str = "."
    project_subdirectory: str | None = None
    hyperparameters: AdapterHyperparameterDefaults = Field(
        default_factory=AdapterHyperparameterDefaults
    )
    resources: AdapterResourceDefaults = Field(default_factory=AdapterResourceDefaults)
    checkpoint: AdapterCheckpointDefaults = Field(default_factory=AdapterCheckpointDefaults)
    tracking: AdapterTrackingDefaults = Field(default_factory=AdapterTrackingDefaults)
    evaluation: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("workdir", "project_subdirectory")
    @classmethod
    def validate_workdir(cls, value: str | None) -> str | None:
        if value is None:
            return None
        raw = value.strip()
        if raw.startswith("/"):
            raise ValueError("adapter workdir must be relative to the repository")
        normalized = raw.strip("/") or "."
        if (
            normalized.startswith("../")
            or "/../" in normalized
            or any(character in normalized for character in ("\x00", "\n", "\r"))
        ):
            raise ValueError("adapter workdir must stay inside the repository")
        return normalized

    @property
    def effective_project_subdirectory(self) -> str:
        return self.project_subdirectory or self.workdir


class AdapterPrerequisite(CanonicalModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,95}$")
    kind: Literal["python", "simulator", "framework", "package", "asset"]
    name: str = Field(min_length=1, max_length=128)
    version: str | None = Field(default=None, max_length=128)
    required: bool = True
    gated: bool = False
    install_mode: Literal["manual", "editable"] = "manual"
    source_repository: str | None = None
    relative_path: str | None = None
    relationship: Literal["sibling_checkout"] | None = None
    description: str = Field(min_length=1, max_length=2000)

    @field_validator("source_repository", "relative_path", "description")
    @classmethod
    def validate_advisory_text(cls, value: str | None) -> str | None:
        if value is not None and any(character in value for character in ("\x00", "\n", "\r")):
            raise ValueError("prerequisite metadata must be single-line advisory text")
        return value


class AdapterManifest(CanonicalModel):
    schema_version: Literal["skynet.adapter/v1"] = "skynet.adapter/v1"
    slug: str = Field(min_length=1, max_length=96)
    display_name: str = Field(min_length=1, max_length=128)
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    repository_patterns: list[str] = Field(default_factory=list)
    default_repository: str | None = None
    legacy_handler: str | None = None
    runtime: AdapterRuntimePolicy
    capabilities: AdapterCapabilities
    defaults: AdapterDefaults = Field(default_factory=AdapterDefaults)
    prerequisites: list[AdapterPrerequisite] = Field(default_factory=list)
    train: CommandTemplate = Field(default_factory=CommandTemplate)
    evaluations: list[EvaluationAdapterMetadata] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    todos: list[str] = Field(default_factory=list)

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,95}", normalized):
            raise ValueError("adapter slug must contain lowercase letters, numbers, underscores, or hyphens")
        return normalized

    @model_validator(mode="after")
    def validate_consistency(self) -> "AdapterManifest":
        if self.capabilities.name != self.slug:
            raise ValueError("capabilities.name must match adapter slug")
        if set(self.capabilities.runtime_backends) != set(self.runtime.allowed_backends):
            raise ValueError("runtime.allowed_backends must match capabilities.runtime_backends")
        for recommendation in self.defaults.resources.gpu_recommendations:
            if (
                recommendation.minimum_gpus < self.capabilities.minimum_gpus
                or recommendation.maximum_gpus > self.capabilities.maximum_gpus
            ):
                raise ValueError(
                    f"GPU recommendation {recommendation.id} must remain inside adapter "
                    f"capabilities {self.capabilities.minimum_gpus}-"
                    f"{self.capabilities.maximum_gpus}"
                )
        supported = set(self.train.supported_canonical_fields)
        declared_defaults = {
            "train.learning_rate": self.defaults.hyperparameters.learning_rate,
            "train.batch.declared_semantics": (
                self.defaults.hyperparameters.batch_semantics
            ),
            "train.batch.value": self.defaults.hyperparameters.batch_size,
            "train.batch.gradient_accumulation_steps": (
                self.defaults.hyperparameters.gradient_accumulation_steps
            ),
            "train.num_workers_per_rank": (
                self.defaults.hyperparameters.num_workers_per_rank
            ),
            "train.max_steps": self.defaults.hyperparameters.max_steps,
            "train.max_epochs": self.defaults.hyperparameters.max_epochs,
            "train.seed": self.defaults.hyperparameters.seed,
            "train.precision": self.defaults.hyperparameters.precision,
        }
        unmapped_defaults = [
            path
            for path, value in declared_defaults.items()
            if value is not None and path not in supported
        ]
        if unmapped_defaults and self.legacy_handler is None:
            raise ValueError(
                "adapter defaults require canonical mappings: "
                + ", ".join(unmapped_defaults)
            )
        return self


def canonical_adapter_manifest(value: AdapterManifest | dict[str, Any]) -> dict[str, Any]:
    """Return the sole representation used for adapter snapshots and hashes."""
    manifest = value if isinstance(value, AdapterManifest) else AdapterManifest.model_validate(value)
    document = manifest.model_dump(mode="json")

    def omit_empty_companions(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                key: omit_empty_companions(child)
                for key, child in item.items()
                if not (
                    (key == "companion_arguments" and child == [])
                    or (key == "argument_validation" and child is None)
                )
            }
        if isinstance(item, list):
            return [omit_empty_companions(child) for child in item]
        return item

    return omit_empty_companions(document)


def adapter_manifest_sha256(value: AdapterManifest | dict[str, Any]) -> str:
    return canonical_sha256(canonical_adapter_manifest(value))


_TEMPLATE_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*}}")


def _path_value(document: dict[str, Any], path: str) -> Any:
    if path.startswith("native.overrides."):
        native = document.get("native")
        overrides = native.get("overrides") if isinstance(native, dict) else None
        if not isinstance(overrides, dict):
            return None
        return overrides.get(path.removeprefix("native.overrides."))
    current: Any = document
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _conditions_match(document: dict[str, Any], conditions: dict[str, list[Any]]) -> bool:
    for path, choices in conditions.items():
        actual = _path_value(document, path)
        if actual is None or not any(
            canonical_sha256(actual) == canonical_sha256(choice) for choice in choices
        ):
            return False
    return True


def _render_argument(template: str, values: dict[str, Any], blockers: list[str]) -> str:
    def substitute(match: re.Match[str]) -> str:
        path = match.group(1)
        value = _path_value(values, path)
        if value is None:
            blockers.append(f"command template value is missing: {path}")
            return match.group(0)
        return _serialize_override(value)

    return _TEMPLATE_RE.sub(substitute, template)


def _argv_has_flag(argv: list[str], flag: str) -> bool:
    return any(argument == flag or argument.startswith(f"{flag}=") for argument in argv)


def _apply_parameter_flags(
    argv: list[str],
    document: dict[str, Any],
    parameter_flags: dict[str, ArgumentBinding],
    blockers: list[str],
) -> None:
    for path, binding in sorted(parameter_flags.items()):
        if path.startswith("train."):
            explicit_parameters = _path_value(document, "intent.explicit_parameters")
            if isinstance(explicit_parameters, list) and not any(
                path == requested
                or path.startswith(f"{requested}.")
                or requested.startswith(f"{path}.")
                for requested in explicit_parameters
            ):
                continue
        value = _path_value(document, path)
        if value is None and binding.omit_if_none:
            continue
        if value is None:
            blockers.append(f"adapter flag mapping requires canonical value: {path}")
            continue

        serialized = _serialize_override(value)
        if binding.value_map:
            if serialized not in binding.value_map:
                supported = ", ".join(sorted(binding.value_map))
                blockers.append(
                    f"adapter flag mapping does not support {path}={serialized}; "
                    f"supported values: {supported}"
                )
                continue
            serialized = binding.value_map[serialized]

        arguments: list[str] = []
        if binding.style == "separate":
            arguments = [binding.flag, serialized]
        elif binding.style in {"equals", "hydra"}:
            arguments = [f"{binding.flag}={serialized}"]
        elif binding.style == "boolean":
            if not isinstance(value, bool):
                blockers.append(
                    f"adapter boolean flag mapping requires a boolean value: {path}"
                )
                continue
            if value:
                arguments = [binding.flag]
            elif binding.false_flag:
                arguments = [binding.false_flag]

        if not arguments:
            continue

        companion_flags = [
            argument.split("=", 1)[0]
            for argument in binding.companion_arguments
        ]
        conflict_flags = [binding.flag, *companion_flags]
        if binding.false_flag:
            conflict_flags.append(binding.false_flag)
        if any(_argv_has_flag(argv, flag) for flag in conflict_flags):
            blockers.append(
                f"adapter flag mapping for {path} conflicts with existing argument: "
                f"{', '.join(conflict_flags)}"
            )
            continue
        argv.extend([*arguments, *binding.companion_arguments])


def _set_document_path(document: dict[str, Any], path: str, value: Any) -> None:
    current: dict[str, Any] = document
    segments = path.split(".")
    for segment in segments[:-1]:
        child = current.get(segment)
        if child is None:
            child = {}
            current[segment] = child
        if not isinstance(child, dict):
            raise AdapterError(
                f"native tracking parameter path crosses a non-object value: {path}"
            )
        current = child
    current[segments[-1]] = value


def _resolve_native_tracking(
    spec: ExperimentSpec,
    integrations: list[NativeTrackingIntegration],
    document: dict[str, Any],
    blockers: list[str],
) -> list[NativeTrackingIntegration]:
    providers = {provider.provider: provider for provider in spec.tracking.providers}
    resolved: list[NativeTrackingIntegration] = []
    for integration in integrations:
        provider = providers.get(integration.provider)
        enabled = bool(provider and provider.enabled)
        values: dict[str, Any] = {"enabled": enabled}
        if integration.provider == "wandb":
            from skynet_app.tracking import wandb_project_slug

            values["project"] = wandb_project_slug(
                (provider.project if provider else None) or spec.identity.experiment
            )
        elif integration.provider == "mlflow":
            values["experiment"] = (
                (provider.experiment if provider else None)
                or spec.tracking.mlflow_experiment
                or spec.identity.experiment
            )
        for field, path in integration.parameter_paths.items():
            value = values.get(field)
            if enabled and value in {None, ""}:
                blockers.append(
                    f"native {integration.provider} tracking cannot resolve {field}"
                )
                continue
            _set_document_path(document, path, value)
        run_id_file = (
            _render_argument(integration.run_id_file, document, blockers)
            if integration.run_id_file
            else None
        )
        resolved.append(integration.model_copy(update={"run_id_file": run_id_file}))
    return resolved


def _resolve_preparation_steps(
    command: CommandTemplate,
    document: dict[str, Any],
    blockers: list[str],
) -> tuple[list[PreparationStep], list[str]]:
    steps: list[PreparationStep] = []
    consumer_argv: list[str] = []
    for template in command.preparation_steps:
        enabled = all(
            any(canonical_sha256(_path_value(document, path)) == canonical_sha256(choice) for choice in choices)
            for path, choices in template.enabled_when.items()
        )
        if not enabled:
            continue
        rendered = {
            "id": template.id,
            "argv": [_render_argument(item, document, blockers) for item in template.argv],
            "working_directory": _render_argument(
                template.working_directory, document, blockers
            ),
            "output_globs": [
                _render_argument(item, document, blockers) for item in template.output_globs
            ],
            "expected_output_sha256": {
                _render_argument(path, document, blockers): digest
                for path, digest in template.expected_output_sha256.items()
            },
        }
        try:
            steps.append(PreparationStep.model_validate(rendered))
        except ValueError as error:
            blockers.append(f"invalid preparation step {template.id}: {error}")
            continue
        consumer_argv.extend(
            _render_argument(item, document, blockers) for item in template.consumer_argv
        )
    return steps, consumer_argv


_CANONICAL_TRAIN_OVERRIDE_PATHS = (
    "train.learning_rate",
    "train.batch.declared_semantics",
    "train.batch.value",
    "train.batch.gradient_accumulation_steps",
    "train.num_workers_per_rank",
    "train.max_steps",
    "train.max_epochs",
    "train.seed",
    "train.precision",
    "train.checkpoint.save_every_steps",
    "train.hyperparameters",
)


def _canonical_path_is_supported(path: str, supported: set[str]) -> bool:
    return any(path == item or path.startswith(f"{item}.") for item in supported)


def _unsupported_canonical_train_overrides(
    spec: ExperimentSpec,
    manifest: AdapterManifest,
) -> list[str]:
    supported = set(manifest.train.supported_canonical_fields)
    if manifest.legacy_handler:
        supported.update(
            get_adapter(manifest.legacy_handler).supported_canonical_fields
        )

    actual = spec.model_dump(mode="python", by_alias=False)
    baseline_train = spec.train.__class__().model_dump(mode="python")
    hyper_defaults = manifest.defaults.hyperparameters
    default_values = {
        "learning_rate": hyper_defaults.learning_rate,
        "batch_size": hyper_defaults.batch_size,
        "gradient_accumulation_steps": hyper_defaults.gradient_accumulation_steps,
        "num_workers_per_rank": hyper_defaults.num_workers_per_rank,
        "max_steps": hyper_defaults.max_steps,
        "max_epochs": hyper_defaults.max_epochs,
        "seed": hyper_defaults.seed,
        "precision": hyper_defaults.precision,
    }
    if default_values["learning_rate"] is not None:
        baseline_train["learning_rate"] = default_values["learning_rate"]
    if default_values["batch_size"] is not None:
        baseline_train["batch"]["value"] = default_values["batch_size"]
    if default_values["gradient_accumulation_steps"] is not None:
        baseline_train["batch"]["gradient_accumulation_steps"] = default_values[
            "gradient_accumulation_steps"
        ]
    for field in (
        "num_workers_per_rank",
        "max_steps",
        "max_epochs",
        "seed",
        "precision",
    ):
        if default_values[field] is not None:
            baseline_train[field] = default_values[field]
    if hyper_defaults.max_epochs is not None and hyper_defaults.max_steps is None:
        baseline_train["max_steps"] = None
    if manifest.defaults.checkpoint.save_every_steps is not None:
        baseline_train["checkpoint"]["save_every_steps"] = (
            manifest.defaults.checkpoint.save_every_steps
        )
    baseline = {"train": baseline_train}

    blockers: list[str] = []
    for path in _CANONICAL_TRAIN_OVERRIDE_PATHS:
        if not spec.intent.is_explicit(path):
            continue
        value = _path_value(actual, path)
        expected = _path_value(baseline, path)
        if canonical_sha256(value) == canonical_sha256(expected):
            continue
        if _canonical_path_is_supported(path, supported):
            continue
        blockers.append(
            f"adapter {manifest.slug} does not map canonical value {path}="
            f"{_serialize_override(value)}; leave it at the declared default or add "
            "a versioned adapter mapping"
        )
    return blockers


class ManifestAdapter(RepositoryAdapter):
    def __init__(
        self,
        manifest: AdapterManifest,
        *,
        manifest_sha256: str | None = None,
    ) -> None:
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256 or adapter_manifest_sha256(manifest)
        self.capabilities = manifest.capabilities.model_copy(
            update={"version": manifest.capabilities.version, "name": manifest.slug}, deep=True
        )

    def _capabilities_for_spec(self, spec: ExperimentSpec) -> AdapterCapabilities:
        if spec.resources.gpu.mode != "auto":
            return self.capabilities
        document = spec.model_dump(mode="python", by_alias=False)
        for recommendation in self.manifest.defaults.resources.gpu_recommendations:
            if _conditions_match(document, recommendation.enabled_when):
                return self.capabilities.model_copy(
                    update={
                        "minimum_gpus": recommendation.minimum_gpus,
                        "recommended_gpus": recommendation.recommended_gpus,
                        "maximum_gpus": recommendation.maximum_gpus,
                    },
                    deep=True,
                )
        return self.capabilities

    def _gpu_type_for_spec(self, spec: ExperimentSpec) -> str:
        requested = spec.resources.gpu.gpu_type
        if spec.resources.gpu.mode != "auto" or requested != "any":
            return requested
        document = spec.model_dump(mode="python", by_alias=False)
        for recommendation in self.manifest.defaults.resources.gpu_recommendations:
            if _conditions_match(document, recommendation.enabled_when):
                return recommendation.gpu_type or requested
        return requested

    def resolve_evaluation(
        self,
        spec: ExperimentSpec,
        *,
        environment: str,
        suite: str,
        context: Mapping[str, Any],
    ) -> AdapterPlan:
        document = spec.model_dump(mode="python", by_alias=False)
        declared = [
            item
            for item in self.manifest.evaluations
            if item.environment == environment and (suite in item.suites or not item.suites)
        ]
        exact = [
            item
            for item in self.manifest.evaluations
            if item.environment == environment
            and suite in item.suites
            and _conditions_match(document, item.enabled_when)
        ]
        wildcard = [
            item
            for item in self.manifest.evaluations
            if item.environment == environment
            and not item.suites
            and _conditions_match(document, item.enabled_when)
        ]
        matches = exact or wildcard
        blockers: list[str] = []
        if not matches:
            if declared:
                blockers.append(
                    f"adapter {self.manifest.slug} declares {environment}/{suite}, but the "
                    "pinned experiment does not satisfy its explicit compatibility contract"
                )
            else:
                blockers.append(
                    f"adapter {self.manifest.slug} does not declare evaluation support for "
                    f"{environment}/{suite}"
                )
            return self._base_plan(
                adapter_version=spec.source.adapter_version,
                argv=[],
                blockers=blockers,
                capabilities=self.capabilities,
                manifest_sha256=self.manifest_sha256,
            )
        if len(matches) != 1:
            blockers.append(
                f"adapter {self.manifest.slug} has ambiguous evaluation declarations for "
                f"{environment}/{suite}"
            )
            return self._base_plan(
                adapter_version=spec.source.adapter_version,
                argv=[],
                blockers=blockers,
                capabilities=self.capabilities,
                manifest_sha256=self.manifest_sha256,
            )

        metadata = matches[0]
        command = metadata.command
        if command is None:
            blockers.append(
                f"adapter {self.manifest.slug} has no versioned command for "
                f"{environment}/{suite}"
            )
            return self._base_plan(
                adapter_version=spec.source.adapter_version,
                argv=[],
                blockers=blockers,
                capabilities=self.capabilities,
                manifest_sha256=self.manifest_sha256,
            )

        canonical_context = json.loads(json.dumps(dict(context), sort_keys=True))
        document["computed"] = {"gpu_count": _gpu_count(spec, self._capabilities_for_spec(spec))}
        document["evaluation"] = canonical_context
        document["tokens"] = {
            "run_dir": RUN_DIR_TOKEN,
            "source_dir": SOURCE_DIR_TOKEN,
            "resume_checkpoint": RESUME_CHECKPOINT_TOKEN,
        }
        for path in command.required_values:
            value = _path_value(document, path)
            if value is None or value == "" or value == []:
                blockers.append(f"adapter requires canonical evaluation value: {path}")

        argv = [_render_argument(argument, document, blockers) for argument in command.argv]
        argv.extend(_render_argument(argument, document, blockers) for argument in command.static_args)
        _apply_parameter_flags(argv, document, command.parameter_flags, blockers)
        if not argv:
            blockers.append("adapter requires a declarative evaluation command")
        preparation_steps, consumer_argv = _resolve_preparation_steps(
            command, document, blockers
        )
        argv.extend(consumer_argv)
        # Evaluation resume arguments are an argv suffix appended by the runner,
        # never a replacement for the base command.  An empty declaration means
        # the base adapter command is idempotent and owns its resume ledger.
        resume_templates = list(command.resume_argv)
        resume_argv = [
            _render_argument(argument, document, blockers)
            for argument in resume_templates
        ]
        context_path = "adapter-support/evaluation-context.json"
        capsule_files = dict(command.capsule_files)
        if context_path in capsule_files:
            blockers.append(f"adapter capsule file conflicts with canonical context: {context_path}")
        else:
            capsule_files[context_path] = json.dumps(
                canonical_context, indent=2, sort_keys=True
            ) + "\n"
        rendered_environment = {
            key: _render_argument(value, document, blockers)
            for key, value in command.environment.items()
        }
        return self._base_plan(
            adapter_version=spec.source.adapter_version,
            argv=argv,
            resume_argv=resume_argv,
            retry_clean_argv=[
                _render_argument(argument, document, blockers)
                for argument in command.retry_clean_argv
            ],
            native_config={
                "canonical_evaluation": canonical_context,
                "adapter_manifest_sha256": self.manifest_sha256,
                "evaluation_environment": environment,
                "evaluation_suite": suite,
                "evaluation_result_schema": metadata.result_schema,
                **(
                    {"evaluation_runtime_profile_id": metadata.runtime_profile_id}
                    if metadata.runtime_profile_id is not None
                    else {}
                ),
            },
            environment=rendered_environment,
            resolved_gpu_type=self._gpu_type_for_spec(spec),
            preparation_steps=preparation_steps,
            capsule_files=capsule_files,
            blockers=list(dict.fromkeys(blockers)),
            todos=list(self.manifest.todos),
            warnings=list(self.manifest.warnings),
            capabilities=self.capabilities,
            manifest_sha256=self.manifest_sha256,
        )

    def _resolve(self, spec: ExperimentSpec, gpu_count: int) -> AdapterPlan:
        command = self.manifest.train
        compatibility_blockers = [
            *_unsupported_canonical_train_overrides(spec, self.manifest),
            *_batch_compatibility_blockers(
                spec,
                command.batch_compatibility,
                adapter_slug=self.manifest.slug,
                gpu_count=gpu_count,
            ),
        ]
        document = spec.model_dump(mode="python", by_alias=False)
        document["computed"] = {"gpu_count": gpu_count}
        document["tokens"] = {
            "run_dir": RUN_DIR_TOKEN,
            "source_dir": SOURCE_DIR_TOKEN,
            "resume_checkpoint": RESUME_CHECKPOINT_TOKEN,
        }
        native_tracking = _resolve_native_tracking(
            spec, command.native_tracking, document, compatibility_blockers
        )
        if not command.argv and self.manifest.legacy_handler:
            try:
                legacy = get_adapter(self.manifest.legacy_handler).resolve(spec)
            except AdapterError as error:
                return self._base_plan(
                    argv=[], blockers=[str(error)], capabilities=self.capabilities,
                    todos=self.manifest.todos, warnings=self.manifest.warnings,
                )
            legacy.adapter = self.manifest.slug
            legacy.adapter_version = spec.source.adapter_version
            legacy.capabilities = self.capabilities
            legacy.manifest_sha256 = self.manifest_sha256
            if command.resume_argv:
                legacy.resume_argv = list(command.resume_argv)
            if command.retry_clean_argv:
                legacy.retry_clean_argv = list(command.retry_clean_argv)
            if command.checkpoint_globs:
                legacy.checkpoint_globs = list(command.checkpoint_globs)
            if command.checkpoint_candidate_kind != "any":
                legacy.checkpoint_candidate_kind = command.checkpoint_candidate_kind
            if command.checkpoint_basename_regex is not None:
                legacy.checkpoint_basename_regex = command.checkpoint_basename_regex
            if command.checkpoint_prune_globs:
                legacy.checkpoint_prune_globs = list(command.checkpoint_prune_globs)
            if command.training_output_prune_globs:
                legacy.training_output_prune_globs = list(command.training_output_prune_globs)
            if command.checkpoint_inference_required_globs:
                legacy.checkpoint_inference_required_globs = list(
                    command.checkpoint_inference_required_globs
                )
            legacy.progress = command.progress
            legacy.environment.update(command.environment)
            legacy.native_tracking = native_tracking
            legacy.native_config.update(document["native"]["config"])
            legacy.blockers.extend(compatibility_blockers)
            if (self.manifest.legacy_handler == "openpi"
                    and spec.native.config.get("dataset_path")
                    and "adapter-support/openpi-dataset.py" not in command.capsule_files):
                legacy.blockers.append("Selected dataset input is unsupported by this pinned OpenPI adapter. Choose the current adapter version to use the dataset bridge.")
            legacy.blockers = list(dict.fromkeys(legacy.blockers))
            legacy.todos = list(dict.fromkeys([*legacy.todos, *self.manifest.todos]))
            legacy.warnings = list(dict.fromkeys([*legacy.warnings, *self.manifest.warnings]))
            if not spec.native.argv:
                legacy.argv.extend(
                    _render_argument(argument, document, legacy.blockers)
                    for argument in command.static_args
                )
                _apply_parameter_flags(
                    legacy.argv, document, command.parameter_flags, legacy.blockers
                )
                preparation_steps, consumer_argv = _resolve_preparation_steps(
                    command, document, legacy.blockers
                )
                legacy.preparation_steps = preparation_steps
                legacy.argv.extend(consumer_argv)
            for path, content in command.capsule_files.items():
                existing = legacy.capsule_files.get(path)
                if existing is not None and existing != content:
                    legacy.blockers.append(
                        f"adapter capsule file conflicts with legacy handler: {path}"
                    )
                    continue
                legacy.capsule_files[path] = content
            return legacy

        blockers: list[str] = list(compatibility_blockers)
        for path in command.required_values:
            value = _path_value(document, path)
            if value is None or value == "" or value == []:
                blockers.append(f"adapter requires canonical value: {path}")
        templates = list(spec.native.argv) if spec.native.argv else [*command.argv, *command.static_args]
        argv = [_render_argument(argument, document, blockers) for argument in templates]
        if not argv:
            blockers.append("adapter requires a declarative train.argv or native.argv")
        if not spec.native.argv:
            _apply_parameter_flags(argv, document, command.parameter_flags, blockers)
        preparation_steps: list[PreparationStep] = []
        if not spec.native.argv:
            preparation_steps, consumer_argv = _resolve_preparation_steps(
                command, document, blockers
            )
            argv.extend(consumer_argv)
        resume_templates = list(spec.native.resume_argv) or list(command.resume_argv)
        resume_argv = [_render_argument(argument, document, blockers) for argument in resume_templates]
        retry_clean_argv = [
            _render_argument(argument, document, blockers)
            for argument in command.retry_clean_argv
        ]
        return self._base_plan(
            adapter_version=spec.source.adapter_version,
            argv=argv,
            resume_argv=resume_argv,
            retry_clean_argv=retry_clean_argv,
            native_config={
                "canonical_train": spec.train.model_dump(mode="json"),
                "adapter_manifest_sha256": self.manifest_sha256,
                **document["native"]["config"],
            },
            environment=dict(command.environment),
            native_tracking=native_tracking,
            preparation_steps=preparation_steps,
            checkpoint_globs=list(command.checkpoint_globs),
            checkpoint_candidate_kind=command.checkpoint_candidate_kind,
            checkpoint_basename_regex=command.checkpoint_basename_regex,
            checkpoint_prune_globs=list(command.checkpoint_prune_globs),
            training_output_prune_globs=list(command.training_output_prune_globs),
            checkpoint_inference_required_globs=list(
                command.checkpoint_inference_required_globs
            ),
            progress=command.progress,
            blockers=list(dict.fromkeys(blockers)),
            todos=list(self.manifest.todos),
            warnings=list(self.manifest.warnings),
            manifest_sha256=self.manifest_sha256,
        )


def _builtin_manifest(
    slug: str,
    display_name: str,
    repository: str | None,
    aliases: list[str],
    recommended_backend: str | None,
    project_subdirectory: str | None = None,
    prerequisites: list[AdapterPrerequisite] | None = None,
    warnings: list[str] | None = None,
    input_fields: list[AdapterInputField] | None = None,
    argv: list[str] | None = None,
    resume_argv: list[str] | None = None,
    parameter_flags: dict[str, ArgumentBinding] | None = None,
    retry_clean_argv: list[str] | None = None,
    preparation_steps: list[PreparationStepTemplate] | None = None,
    capsule_files: dict[str, str] | None = None,
    gpu_recommendations: list[AdapterGPURecommendation] | None = None,
    checkpoint_globs: list[str] | None = None,
    checkpoint_candidate_kind: CheckpointCandidateKind = "any",
    checkpoint_basename_regex: str | None = None,
    checkpoint_prune_globs: list[str] | None = None,
    training_output_prune_globs: list[str] | None = None,
    checkpoint_inference_required_globs: list[str] | None = None,
    evaluations: list[EvaluationAdapterMetadata] | None = None,
    hyperparameter_defaults: AdapterHyperparameterDefaults | None = None,
    supported_canonical_fields: list[str] | None = None,
    batch_compatibility: AdapterBatchCompatibility | None = None,
    progress: TrainingProgressContract | None = None,
    argument_validation: RepositoryArgumentValidation | None = None,
    native_tracking: list[NativeTrackingIntegration] | None = None,
) -> AdapterManifest:
    adapter = get_adapter(slug)
    capabilities = adapter.capabilities.model_copy(update={"name": slug}, deep=True)
    resolved_evaluations = evaluations or [
        EvaluationAdapterMetadata(environment=name)
        for name in capabilities.evaluation_adapters
    ]
    return AdapterManifest(
        slug=slug,
        display_name=display_name,
        description=f"Seeded compatibility adapter for {display_name}",
        aliases=aliases,
        repository_patterns=[repository] if repository else [],
        default_repository=repository,
        legacy_handler=slug,
        runtime=AdapterRuntimePolicy(
            allowed_backends=set(capabilities.runtime_backends),
            recommended_backend=recommended_backend,
        ),
        capabilities=capabilities,
        defaults=AdapterDefaults(
            workdir=project_subdirectory or ".",
            project_subdirectory=project_subdirectory,
            hyperparameters=(
                hyperparameter_defaults or AdapterHyperparameterDefaults()
            ),
            resources=AdapterResourceDefaults(
                gpu_recommendations=gpu_recommendations or [],
            ),
        ),
        prerequisites=prerequisites or [],
        train=CommandTemplate(
            argv=argv or [],
            resume_argv=resume_argv or [],
            input_fields=input_fields or [],
            parameter_flags=parameter_flags or {},
            supported_canonical_fields=(
                supported_canonical_fields
                if supported_canonical_fields is not None
                else sorted(adapter.supported_canonical_fields)
            ),
            retry_clean_argv=retry_clean_argv or [],
            preparation_steps=preparation_steps or [],
            capsule_files=capsule_files or {},
            checkpoint_globs=checkpoint_globs or [],
            checkpoint_candidate_kind=checkpoint_candidate_kind,
            checkpoint_basename_regex=checkpoint_basename_regex,
            checkpoint_prune_globs=checkpoint_prune_globs or [],
            training_output_prune_globs=training_output_prune_globs or [],
            checkpoint_inference_required_globs=(
                checkpoint_inference_required_globs or []
            ),
            batch_compatibility=batch_compatibility,
            progress=progress,
            argument_validation=argument_validation,
            native_tracking=native_tracking or [],
        ),
        evaluations=resolved_evaluations,
        warnings=warnings or [],
    )


def builtin_adapter_manifests() -> list[AdapterManifest]:
    return [
        _builtin_manifest(
            "generic", "Custom structured command", None, ["custom"], None,
            input_fields=[
                AdapterInputField(
                    path="native.argv", label="Training command", kind="string_list", required=True,
                    help="Structured argv for the repository training entrypoint; shell strings are not accepted.",
                ),
                AdapterInputField(
                    path="native.resume_argv", label="Resume arguments", kind="string_list",
                    help="Required when checkpoint auto-resume is enabled.",
                ),
            ],
        ),
        _builtin_manifest(
            "egoverse", "EgoVerse", "https://github.com/GaTech-RL2/EgoVerse", [], "uv",
            argv=[
                "python",
                "egomimic/trainHydra.py",
                "--config-name",
                "{{native.config.config_name}}",
                "hydra/launcher=basic",
                "trainer.devices={{computed.gpu_count}}",
                "seed={{train.seed}}",
                "paths.output_dir={{tokens.run_dir}}/artifacts",
            ],
            resume_argv=["ckpt_path={{tokens.resume_checkpoint}}"],
            hyperparameter_defaults=AdapterHyperparameterDefaults(
                batch_semantics="per_device",
                gradient_accumulation_steps=1,
            ),
            parameter_flags={
                "train.batch.gradient_accumulation_steps": ArgumentBinding(
                    flag="+trainer.accumulate_grad_batches",
                    style="hydra",
                ),
                "train.checkpoint.save_every_steps": ArgumentBinding(
                    flag="+callbacks.model_checkpoint.every_n_train_steps",
                    style="hydra",
                    companion_arguments=[
                        "callbacks.model_checkpoint.every_n_epochs=null",
                    ],
                ),
                "train.learning_rate": ArgumentBinding(
                    flag="model.optimizer.lr",
                    style="hydra",
                ),
                "train.max_steps": ArgumentBinding(
                    flag="+trainer.max_steps",
                    style="hydra",
                ),
            },
            input_fields=[
                AdapterInputField(
                    path="native.config.config_name", label="Hydra config name", kind="string",
                    default="train_zarr_cartesian", tutorial_value="train_zarr_cartesian",
                    help="Hydra config used by EgoVerse; the legacy resolver uses train_zarr_cartesian when omitted.",
                    choice_source=RepositoryYamlCatalogSource(
                        directory="egomimic/hydra_configs",
                        filename_pattern="train*.yaml",
                        allow_custom=False,
                        metadata_fields=[
                            RepositoryYamlCatalogMetadataField(
                                source_path="model.optimizer.lr",
                                canonical_path="train.learning_rate",
                            ),
                            RepositoryYamlCatalogMetadataField(
                                source_path=(
                                    "data.train_dataloader_params.*.batch_size"
                                ),
                                canonical_path="train.batch.value",
                            ),
                            RepositoryYamlCatalogMetadataField(
                                source_path=(
                                    "data.train_dataloader_params.*.num_workers"
                                ),
                                canonical_path="train.num_workers_per_rank",
                            ),
                            RepositoryYamlCatalogMetadataField(
                                source_path="trainer.accumulate_grad_batches",
                                canonical_path=(
                                    "train.batch.gradient_accumulation_steps"
                                ),
                            ),
                            RepositoryYamlCatalogMetadataField(
                                source_path="trainer.max_steps",
                                canonical_path="train.max_steps",
                            ),
                            RepositoryYamlCatalogMetadataField(
                                source_path="trainer.max_epochs",
                                canonical_path="train.max_epochs",
                            ),
                            RepositoryYamlCatalogMetadataField(
                                source_path="trainer.precision",
                                canonical_path="train.precision",
                            ),
                        ],
                    ),
                ),
            ],
        ),
        _builtin_manifest(
            "dexverse", "DexVerse", "https://github.com/ycyao216/DexVerse", [], "conda",
            project_subdirectory="source/dexverse",
            prerequisites=[
                AdapterPrerequisite(
                    id="python-3-11",
                    kind="python",
                    name="Python",
                    version="3.11",
                    description="Use Python 3.11 in a manually provisioned environment.",
                ),
                AdapterPrerequisite(
                    id="isaac-sim-5-1-0",
                    kind="simulator",
                    name="NVIDIA Isaac Sim",
                    version="5.1.0",
                    description="Install Isaac Sim 5.1.0 before installing or running DexVerse.",
                ),
                AdapterPrerequisite(
                    id="isaac-lab-v2-3-2",
                    kind="framework",
                    name="NVIDIA Isaac Lab",
                    version="v2.3.2",
                    source_repository="https://github.com/isaac-sim/IsaacLab",
                    relative_path="../IsaacLab",
                    relationship="sibling_checkout",
                    description="Provision Isaac Lab v2.3.2 manually as a sibling checkout of DexVerse.",
                ),
                AdapterPrerequisite(
                    id="dexverse-editable-install",
                    kind="package",
                    name="DexVerse Python package",
                    install_mode="editable",
                    relative_path="source/dexverse",
                    description="Install the source/dexverse package in editable mode into the prepared environment.",
                ),
                AdapterPrerequisite(
                    id="dexverse-huggingface-assets",
                    kind="asset",
                    name="DexVerse release assets",
                    gated=True,
                    source_repository="https://huggingface.co/datasets/dexverse/DexVerse_release",
                    description="Accept the gated Hugging Face dataset terms and provision required robot, object, scene, and demonstration assets manually.",
                ),
            ],
            warnings=[
                "Conda is the upstream recommendation, but DexVerse provides no conda-lock file; select and provision Conda manually rather than treating it as reproducible auto-detection.",
                "Isaac Lab must remain a separately versioned sibling checkout; Skynet does not clone or install that dependency automatically.",
                "Gated Hugging Face assets require authorization and explicit provisioning before environments can run.",
            ],
            input_fields=[
                AdapterInputField(
                    path="native.argv", label="Pinned Isaac Lab trainer command", kind="string_list", required=True,
                    help="DexVerse defines environments but no canonical trainer; provide a structured argv for a separately pinned runner.",
                ),
                AdapterInputField(
                    path="native.resume_argv", label="Trainer resume arguments", kind="string_list",
                    help="Required when checkpoint auto-resume is enabled for the selected external runner.",
                ),
                AdapterInputField(
                    path="native.config.checkpoint_globs", label="Checkpoint globs", kind="string_list",
                    default=["**/*.pt"], help="Native checkpoint discovery patterns used by the external runner.",
                ),
            ],
        ),
        _builtin_manifest(
            "dexmimicgen", "DexMimicGen", "https://github.com/NVlabs/dexmimicgen", [], "uv",
            input_fields=[
                AdapterInputField(
                    path="native.config.training_config", label="Generated training config", kind="string",
                    help="Required by the default robomimic launcher unless native.argv supplies a complete command.",
                ),
                AdapterInputField(
                    path="native.config.robomimic_revision", label="Robomimic revision", kind="string", required=True,
                    help="Exact pinned robomimic revision used to generate and execute the training config.",
                ),
            ],
        ),
        _builtin_manifest(
            "get_zero", "GET-Zero", "https://github.com/real-stanford/get_zero", ["get-zero"], "conda",
            input_fields=[
                AdapterInputField(
                    path="native.config.mode", label="Training mode", kind="string", default="rl_distill",
                    tutorial_value="rl_distill",
                    help="Legacy resolver mode; repository-specific support must be validated at the pinned revision.",
                ),
                AdapterInputField(
                    path="native.config.override_scopes", label="Ordered override scopes", kind="json",
                    help="The rl_distill mode requires exactly four ordered JSON lists.",
                ),
                AdapterInputField(
                    path="native.resume_argv", label="Mode-specific resume arguments", kind="string_list",
                    help="Required when checkpoint auto-resume is enabled.",
                ),
            ],
        ),
        _builtin_manifest(
            "groot", "Isaac-GR00T", "https://github.com/NVIDIA/Isaac-GR00T",
            ["isaac-groot", "isaac_gr00t"], "uv",
            hyperparameter_defaults=AdapterHyperparameterDefaults(
                batch_size=1,
                num_workers_per_rank=2,
            ),
            warnings=[
                "GR00T N1.6 retains the pretrained GR1 head but removed its built-in Fourier GR1 modality profile; the adapter supplies a pinned, metadata-validated GR1ArmsOnly bridge when no custom config is selected."
            ],
            capsule_files={
                "adapter-support/groot-gr1-modality.py": GROOT_GR1_MODALITY_CONFIG_SOURCE,
                "adapter-support/groot-n16-launcher.py": GROOT_N16_LAUNCHER_SOURCE,
            },
            **GROOT_CHECKPOINT_CONTRACT,
            evaluations=[
                EvaluationAdapterMetadata(
                    environment="mujoco",
                    suites=["groot_gr1_tabletop"],
                    enabled_when={"native.config.embodiment_tag": ["GR1"]},
                    command=CommandTemplate(
                        argv=[
                            "python",
                            "{{tokens.run_dir}}/adapter-support/groot-robocasa.py",
                            "--context",
                            "{{tokens.run_dir}}/adapter-support/evaluation-context.json",
                            "--source-dir",
                            "{{tokens.source_dir}}",
                        ],
                        environment={
                            "MUJOCO_GL": "egl",
                            "PYOPENGL_PLATFORM": "egl",
                            "SKYNET_EVAL_RESUME_GRANULARITY": "episode",
                        },
                        required_values=[
                            "evaluation.run_id",
                            "evaluation.checkpoint.path",
                            "evaluation.checkpoint.sha256",
                            "evaluation.suite.name",
                            "evaluation.suite.version",
                            "evaluation.tasks",
                            "evaluation.episodes_per_task",
                            "evaluation.seeds",
                            "evaluation.result_path",
                            "evaluation.progress_path",
                            "evaluation.video_path",
                        ],
                        capsule_files={
                            "adapter-support/groot-robocasa.py": GROOT_ROBOCASA_BRIDGE_SOURCE,
                        },
                    ),
                ),
                EvaluationAdapterMetadata(
                    environment="isaac_sim",
                    suites=["groot_gr1_isaaclab_evaltasks"],
                    runtime_profile_id="groot-isaacsim-5.0.0_isaaclab-2.2.0_py311",
                    enabled_when={"native.config.embodiment_tag": ["GR1"]},
                    command=CommandTemplate(
                        argv=[
                            "python",
                            "{{tokens.run_dir}}/adapter-support/groot-isaaclab.py",
                            "--context",
                            "{{tokens.run_dir}}/adapter-support/evaluation-context.json",
                            "--source-dir",
                            "{{tokens.source_dir}}",
                        ],
                        environment={
                            "ACCEPT_EULA": "Y",
                            "PRIVACY_CONSENT": "Y",
                            "SKYNET_EVAL_RESUME_GRANULARITY": "episode",
                        },
                        required_values=[
                            "evaluation.run_id",
                            "evaluation.checkpoint.path",
                            "evaluation.checkpoint.sha256",
                            "evaluation.suite.name",
                            "evaluation.suite.version",
                            "evaluation.tasks",
                            "evaluation.episodes_per_task",
                            "evaluation.seeds",
                            "evaluation.result_path",
                            "evaluation.progress_path",
                            "evaluation.video_path",
                            "evaluation.policy_runtime.backend",
                            "evaluation.evaluator_runtime.profile",
                            "evaluation.evaluator_runtime.backend",
                            "evaluation.evaluator_runtime.environment_path",
                            "evaluation.evaluator_runtime.python_executable",
                            "evaluation.evaluator_runtime.source_dir",
                            "evaluation.evaluator_runtime.source.repository",
                            "evaluation.evaluator_runtime.source.revision",
                            "evaluation.evaluator_runtime.versions.python",
                            "evaluation.evaluator_runtime.versions.isaac_sim",
                            "evaluation.evaluator_runtime.versions.isaac_lab",
                        ],
                        capsule_files={
                            "adapter-support/groot-isaaclab.py": GROOT_ISAACLAB_BRIDGE_SOURCE,
                        },
                    ),
                ),
            ],
            input_fields=[
                AdapterInputField(
                    path="native.config.dataset_path", label="GR00T dataset path", kind="string",
                    tutorial_value="demo_data/libero_demo",
                    data_binding=DataBundleInputBinding(
                        role="training_data",
                        formats=["groot-lerobot-v2", "groot-lerobot-v2.0"],
                    ),
                    help=(
                        "Dataset root passed to GR00T. A selected training_data bundle is used only when "
                        "its registered format declares GR00T-compatible LeRobot data including meta/modality.json."
                    ),
                ),
                AdapterInputField(
                    path="native.config.base_model_path", label="Base model path", kind="string",
                    default="nvidia/GR00T-N1.6-3B",
                    help="Hugging Face model ID or local model directory. The default matches the tested N1.6 training and evaluation bridge; other weights must match the selected repository and embodiment.",
                ),
                AdapterInputField(
                    path="native.config.embodiment_tag", label="Embodiment tag", kind="string",
                    default="LIBERO_PANDA", tutorial_value="LIBERO_PANDA",
                    choice_source=RepositoryChoiceSource(
                        kind="python_enum",
                        entrypoint="gr00t/data/embodiment_tags.py",
                        registry="EmbodimentTag",
                        value_keyword="name",
                    ),
                    help="Allowed tags are read statically from EmbodimentTag at the selected exact commit.",
                ),
                AdapterInputField(
                    path="native.config.modality_config_path", label="Custom modality config path", kind="string",
                    help=(
                        "Optional repository-relative Python config for a custom embodiment. Leave blank for "
                        "pre-registered tags such as LIBERO_PANDA."
                    ),
                ),
                AdapterInputField(
                    path="native.config.video_backend",
                    label="Video decoder backend",
                    kind="string",
                    default="opencv",
                    choices=["opencv", "torchcodec", "decord", "ffmpeg"],
                    help=(
                        "Explicit indexed-frame decoder used by GR00T. OpenCV is the portable default; "
                        "choose another backend only when it is installed and functional in the pinned runtime."
                    ),
                ),
            ],
        ),
        _builtin_manifest(
            "openpi", "openpi", "https://github.com/Physical-Intelligence/openpi", [], "uv",
            capsule_files={"adapter-support/openpi-dataset.py": OPENPI_DATASET_BRIDGE_SOURCE},
            progress=TrainingProgressContract(
                source=TrainingProgressLogSource(
                    stream="stderr",
                    pattern=(
                        r"Progress on:\s*(?P<completed>[0-9]+(?:\.[0-9]+)?[kMGT]?)it/"
                        r"(?P<total>[0-9]+(?:\.[0-9]+)?[kMGT]?)it\b.*?"
                        r"\belapsed:(?P<elapsed>(?:[0-9]+:)?[0-5][0-9]:[0-5][0-9])\b"
                    ),
                    value_format="decimal_si",
                    elapsed_format="clock",
                )
            ),
            batch_compatibility=_OPENPI_BATCH_COMPATIBILITY,
            hyperparameter_defaults=AdapterHyperparameterDefaults(
                batch_semantics="global_before_accumulation",
                gradient_accumulation_steps=1,
            ),
            warnings=[
                "OpenPI LIBERO evaluation auto-resume commits complete task+seed groups; "
                "an interrupted group is rerun in a new attempt directory."
            ],
            retry_clean_argv=["--overwrite"],
            checkpoint_globs=["checkpoints/*/*/*"],
            checkpoint_candidate_kind="directory",
            checkpoint_basename_regex=r"^[0-9]+$",
            checkpoint_prune_globs=["train_state"],
            checkpoint_inference_required_globs=["params", "assets"],
            gpu_recommendations=[
                AdapterGPURecommendation(
                    id="pi05-libero",
                    enabled_when={"native.config.config_name": ["pi05_libero"]},
                    minimum_gpus=2,
                    recommended_gpus=2,
                    maximum_gpus=8,
                    gpu_type="l40s",
                )
            ],
            preparation_steps=[
                PreparationStepTemplate(
                    id="openpi-pi05-libero-norm-stats",
                    argv=[
                        "curl",
                        "--fail",
                        "--location",
                        "--create-dirs",
                        "--output",
                        "assets/pi05_libero/physical-intelligence/libero/norm_stats.json",
                        "https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json",
                    ],
                    working_directory="artifacts/preparation/openpi/pi05-libero",
                    output_globs=[
                        "assets/pi05_libero/physical-intelligence/libero/norm_stats.json"
                    ],
                    expected_output_sha256={
                        "assets/pi05_libero/physical-intelligence/libero/norm_stats.json": (
                            "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
                        )
                    },
                    enabled_when={"native.config.config_name": ["pi05_libero"], "native.config.dataset_path": [None, ""]},
                    consumer_argv=[
                        "--assets-base-dir",
                        "{{tokens.run_dir}}/artifacts/preparation/openpi/pi05-libero/assets",
                    ],
                )
            ],
            input_fields=[
                AdapterInputField(
                    path="native.config.config_name", label="OpenPI training config", kind="string", required=True,
                    tutorial_value="debug",
                    help="Name registered by the pinned OpenPI training config module. The debug config is for tutorial smoke previews only, not a production default.",
                    choice_source=RepositoryChoiceSource(
                        kind="python_static_registry",
                        entrypoint="src/openpi/training/config.py",
                        supporting_files=[
                            "src/openpi/training/misc/roboarena_config.py",
                            "src/openpi/training/misc/polaris_config.py",
                        ],
                        registry="_CONFIGS",
                        constructor="TrainConfig",
                        value_keyword="name",
                        metadata_fields=[
                            RepositoryChoiceMetadataField(
                                source_path="lr_schedule.peak_lr",
                                canonical_path="train.learning_rate",
                            ),
                            RepositoryChoiceMetadataField(
                                source_path="batch_size",
                                canonical_path="train.batch.value",
                            ),
                            RepositoryChoiceMetadataField(
                                source_path="num_workers",
                                canonical_path="train.num_workers_per_rank",
                            ),
                            RepositoryChoiceMetadataField(
                                source_path="num_train_steps",
                                canonical_path="train.max_steps",
                            ),
                            RepositoryChoiceMetadataField(
                                source_path="pytorch_training_precision",
                                canonical_path="train.precision",
                                value_map={
                                    "bfloat16": "bf16",
                                    "float16": "fp16",
                                    "float32": "fp32",
                                },
                            ),
                        ],
                    ),
                ),
                AdapterInputField(
                    path="native.config.dataset_path", label="OpenPI dataset path", kind="string",
                    data_binding=DataBundleInputBinding(role="training_data", formats=["lerobot-v2.0", "lerobot-v2.1", "openpi-libero-lerobot-v2"]),
                    help="Optional absolute compute-node path, or select a dataset bundle. Supports LeRobot LIBERO observations: image, wrist_image, state[8], actions[7]. Requires a matching normalization file; no download fallback is allowed.",
                ),
                AdapterInputField(
                    path="native.config.dataset_norm_stats_path", label="Dataset normalization file", kind="string",
                    help="Required when selecting a dataset: absolute compute-node path to norm_stats.json computed for those files and the chosen OpenPI config. Repository default statistics are not substituted. Leave both dataset inputs blank to use the repository configuration.",
                ),
                AdapterInputField(
                    path="native.config.ema_decay",
                    label="OpenPI EMA decay",
                    kind="string",
                    choices=["0.999", "None"],
                    help=(
                        "Optional OpenPI CLI value. Leave unset to preserve the upstream "
                        "default; None disables EMA, changes training semantics, and reduces "
                        "memory use."
                    ),
                ),
            ],
            parameter_flags={
                "computed.gpu_count": ArgumentBinding(flag="--fsdp-devices"),
                "native.config.ema_decay": ArgumentBinding(flag="--ema-decay"),
                "native.config.wandb_enabled": ArgumentBinding(
                    flag="--wandb-enabled",
                    false_flag="--no-wandb-enabled",
                    style="boolean",
                ),
                "native.config.project_name": ArgumentBinding(flag="--project-name"),
                "train.batch.value": ArgumentBinding(flag="--batch-size"),
                "train.checkpoint.save_every_steps": ArgumentBinding(flag="--save-interval"),
                "train.max_steps": ArgumentBinding(flag="--num-train-steps"),
                "train.num_workers_per_rank": ArgumentBinding(flag="--num-workers"),
                "train.precision": ArgumentBinding(
                    flag="--pytorch-training-precision",
                    style="separate",
                    value_map={"bf16": "bfloat16", "fp32": "float32"},
                ),
                "train.seed": ArgumentBinding(flag="--seed"),
            },
            native_tracking=[
                NativeTrackingIntegration(
                    provider="wandb",
                    parameter_paths={
                        "enabled": "native.config.wandb_enabled",
                        "project": "native.config.project_name",
                    },
                    run_id_file=(
                        "{{tokens.run_dir}}/checkpoints/"
                        "{{native.config.config_name}}/{{identity.experiment}}/wandb_id.txt"
                    ),
                )
            ],
            evaluations=[
                EvaluationAdapterMetadata(
                    environment="libero",
                    suites=[
                        "libero_spatial",
                        "libero_object",
                        "libero_goal",
                        "libero_10",
                        "libero_90",
                    ],
                    command=CommandTemplate(
                        argv=[
                            "python",
                            "{{tokens.run_dir}}/adapter-support/openpi-libero.py",
                            "--context",
                            "{{tokens.run_dir}}/adapter-support/evaluation-context.json",
                        ],
                        resume_argv=[],
                        environment={
                            "MUJOCO_GL": "egl",
                            "MUJOCO_EGL_DEVICE_ID": "0",
                            "PYOPENGL_PLATFORM": "egl",
                            "SKYNET_EVAL_RESUME_GRANULARITY": "task_seed_group",
                        },
                        required_values=[
                            "evaluation.run_id",
                            "evaluation.checkpoint.path",
                            "evaluation.checkpoint.sha256",
                            "evaluation.suite.name",
                            "evaluation.suite.version",
                            "evaluation.policy.native_config",
                            "evaluation.episodes_per_task",
                            "evaluation.seeds",
                            "evaluation.result_path",
                            "evaluation.progress_path",
                        ],
                        capsule_files={
                            "adapter-support/openpi-libero.py": OPENPI_LIBERO_BRIDGE_SOURCE,
                        },
                    ),
                ),
                EvaluationAdapterMetadata(environment="mujoco"),
            ],
        ),
    ]


_VERSIONED_NATIVE_TRACKING_MIGRATIONS: dict[str, dict[str, Any]] = {
    # OpenPI adapter v104 predates the canonical native-tracking declaration,
    # although its immutable command contract already exposes the verified
    # W&B enable flag. The pinned upstream revision also exposes project_name
    # and the checkpoint-local wandb_id.txt resume identity. Keep this narrow:
    # an unknown manifest or source revision must never inherit current adapter
    # behavior implicitly.
    "503907585a9cb93a714271780718ce8a7aa8445fc5ae6620685fba8f4f950c79": {
        "id": "openpi-v104-native-wandb-v1",
        "adapter": "openpi",
        "source_revisions": {
            "215abfb217dbac7d5f1273282331b9b1866c0479",
        },
        "required_parameter_flags": {
            "native.config.wandb_enabled": ArgumentBinding(
                flag="--wandb-enabled",
                false_flag="--no-wandb-enabled",
                style="boolean",
            ),
        },
        "added_parameter_flags": {
            "native.config.project_name": ArgumentBinding(flag="--project-name"),
        },
        "native_tracking": [
            NativeTrackingIntegration(
                provider="wandb",
                parameter_paths={
                    "enabled": "native.config.wandb_enabled",
                    "project": "native.config.project_name",
                },
                run_id_file=(
                    "{{tokens.run_dir}}/checkpoints/"
                    "{{native.config.config_name}}/{{identity.experiment}}/wandb_id.txt"
                ),
            )
        ],
    },
}


def _versioned_adapter_manifest_for_resolution(
    spec: ExperimentSpec,
    manifest: AdapterManifest,
    manifest_sha256: str,
) -> tuple[AdapterManifest, dict[str, Any] | None]:
    """Apply an explicit compatibility contract without altering pinned evidence."""

    enabled_providers = {
        provider.provider
        for provider in spec.tracking.providers
        if provider.enabled
    }
    if spec.tracking.native_tracking == "disable" or "wandb" not in enabled_providers:
        return manifest, None
    migration = _VERSIONED_NATIVE_TRACKING_MIGRATIONS.get(manifest_sha256)
    if migration is None:
        return manifest, None
    if manifest.slug != migration["adapter"]:
        raise AdapterError(
            "versioned native-tracking migration does not match the pinned adapter"
        )
    if spec.source.revision not in migration["source_revisions"]:
        raise AdapterError(
            "pinned adapter native-tracking migration is not verified for source "
            f"revision {spec.source.revision}"
        )
    command = manifest.train
    if command.native_tracking:
        raise AdapterError(
            "versioned native-tracking migration conflicts with the pinned manifest"
        )
    for path, expected in migration["required_parameter_flags"].items():
        if command.parameter_flags.get(path) != expected:
            raise AdapterError(
                "versioned native-tracking migration prerequisite changed: " + path
            )
    parameter_flags = dict(command.parameter_flags)
    for path, binding in migration["added_parameter_flags"].items():
        existing = parameter_flags.get(path)
        if existing is not None and existing != binding:
            raise AdapterError(
                "versioned native-tracking migration conflicts with parameter: " + path
            )
        parameter_flags[path] = binding
    migrated_command = command.model_copy(
        update={
            "parameter_flags": parameter_flags,
            "native_tracking": list(migration["native_tracking"]),
        },
        deep=True,
    )
    warning = (
        "Applied explicit adapter compatibility migration "
        f"{migration['id']} to the verified pinned manifest."
    )
    migrated = AdapterManifest.model_validate(
        manifest.model_copy(
            update={
                "train": migrated_command,
                "warnings": [*manifest.warnings, warning],
            },
            deep=True,
        ).model_dump(mode="json")
    )
    return migrated, {
        "kind": "versioned_adapter_native_tracking",
        "migration_id": migration["id"],
        "adapter": manifest.slug,
        "source_revision": spec.source.revision,
        "source_manifest_sha256": manifest_sha256,
        "providers": [item.provider for item in migration["native_tracking"]],
    }


def apply_pinned_adapter_plan_compatibility(
    spec: ExperimentSpec,
    pinned_plan: AdapterPlan,
) -> tuple[AdapterPlan, dict[str, Any] | None]:
    """Overlay an allowlisted control-plane integration on an exact pinned plan."""

    raw_manifest = spec.source.adapter_manifest
    expected = spec.source.adapter_manifest_sha256
    if raw_manifest is None or not expected:
        return pinned_plan, None
    raw_digest = canonical_sha256(raw_manifest)
    if raw_digest != expected:
        raise AdapterError("adapter manifest snapshot hash does not match")
    manifest = AdapterManifest.model_validate(raw_manifest)
    migrated, transformation = _versioned_adapter_manifest_for_resolution(
        spec, manifest, raw_digest
    )
    if transformation is None:
        return pinned_plan, None
    baseline = ManifestAdapter(manifest, manifest_sha256=raw_digest).resolve(spec)
    if baseline.model_dump(mode="json") != pinned_plan.model_dump(mode="json"):
        raise AdapterError(
            "pinned adapter plan does not match its pre-migration manifest resolution"
        )
    return (
        ManifestAdapter(migrated, manifest_sha256=raw_digest).resolve(spec),
        transformation,
    )


def resolve_adapter_plan(spec: ExperimentSpec) -> AdapterPlan:
    if spec.source.adapter_manifest is not None:
        raw_manifest = spec.source.adapter_manifest
        manifest = AdapterManifest.model_validate(raw_manifest)
        if manifest.slug != spec.source.adapter:
            raise AdapterError("adapter manifest slug does not match experiment source")
        expected = spec.source.adapter_manifest_sha256
        raw_digest = canonical_sha256(raw_manifest)
        if expected and raw_digest != expected:
            raise AdapterError("adapter manifest snapshot hash does not match")
        manifest, _ = _versioned_adapter_manifest_for_resolution(
            spec, manifest, raw_digest
        )
        return ManifestAdapter(
            manifest,
            manifest_sha256=expected or raw_digest,
        ).resolve(spec)
    return get_adapter(spec.source.adapter, spec.source.adapter_version).resolve(spec)


def resolve_adapter_evaluation_plan(
    spec: ExperimentSpec,
    *,
    environment: str,
    suite: str,
    context: dict[str, Any],
    manifest: AdapterManifest | dict[str, Any] | None = None,
) -> AdapterPlan:
    if manifest is None:
        if spec.source.adapter_manifest is None:
            raise AdapterError("evaluation resolution requires a versioned adapter manifest")
        resolved_manifest = AdapterManifest.model_validate(spec.source.adapter_manifest)
        expected = spec.source.adapter_manifest_sha256
        if expected and canonical_sha256(spec.source.adapter_manifest) != expected:
            raise AdapterError("adapter manifest snapshot hash does not match")
    else:
        resolved_manifest = (
            manifest
            if isinstance(manifest, AdapterManifest)
            else AdapterManifest.model_validate(manifest)
        )
    if resolved_manifest.slug != spec.source.adapter:
        raise AdapterError("adapter manifest slug does not match experiment source")
    raw_manifest = spec.source.adapter_manifest if manifest is None else manifest
    return ManifestAdapter(
        resolved_manifest,
        manifest_sha256=spec.source.adapter_manifest_sha256
        or canonical_sha256(raw_manifest),
    ).resolve_evaluation(
        spec,
        environment=environment,
        suite=suite,
        context=context,
    )


def resolve_gpu_count(spec: ExperimentSpec, plan: AdapterPlan) -> int:
    return _gpu_count(spec, plan.capabilities)


def resolve_gpu_type(spec: ExperimentSpec, plan: AdapterPlan) -> str:
    return plan.resolved_gpu_type or spec.resources.gpu.gpu_type
