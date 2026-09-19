"""Shared, versioned training requirements and presets for every adapter.

Requirements describe the trainer, independently of where a dataset originated.
Converters advertise the contracts they produce; they never imply that every
dataset with the same file extension has the same robot or observation layout.
"""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, model_validator

from .experiments import CanonicalModel

RECORDING_DATASET_FORMAT = "skynet.recording-dataset/v1"


class RecordingLoaderValidation(CanonicalModel):
    """The selected adapter's CPU loader check."""
    runtime_profile: str
    script: str
    argv: list[str]
    schemas: list[str]
    mode: str


class RecordingDataPreset(CanonicalModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    name: str
    description: str = ""
    contract: str
    observations: list[str]
    action_representation: str | dict[str, Any]
    observation_requirements: dict[str, Any]
    training_setup: dict[str, Any]
    loader_validation: RecordingLoaderValidation
    conversion_dependencies: list[str] = Field(default_factory=list)
    minimum_episodes: int = Field(default=1, ge=1)
    validation_required: bool = False
    split_mode: Literal["episode"] = "episode"
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    supported_robots: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_recording_requirements(self):
        from .observation_contracts import validate_requirements
        self.observation_requirements = validate_requirements(self.observation_requirements)
        return self


class RecordingConversion(CanonicalModel):
    format: Literal["skynet.recording-dataset/v1"] = RECORDING_DATASET_FORMAT
    default_preset: str
    presets: list[RecordingDataPreset] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_presets(self):
        identifiers = [preset.id for preset in self.presets]
        if len(identifiers) != len(set(identifiers)) or self.default_preset not in identifiers:
            raise ValueError("Recording conversion requires unique presets and a declared default")
        return self


class RecordingSampling(CanonicalModel):
    """Experiment loader behavior; never a conversion requirement."""
    window_policy: Literal["complete", "pad"]
    require_validation: bool = False
    default_action_steps: int = Field(default=1, ge=1)


class DatasetRequirement(CanonicalModel):
    description: str
    mode: Literal["dataset", "simulation", "custom"] = "dataset"
    observations: list[str] = Field(default_factory=list)
    action_representation: str | None = None
    observation_requirements: dict[str, Any] | None = None
    recording_conversion: RecordingConversion | None = None
    recording_sampling: RecordingSampling | None = None

    @model_validator(mode="after")
    def validate_observations(self):
        if self.observation_requirements is not None:
            from .observation_contracts import validate_requirements
            self.observation_requirements = validate_requirements(self.observation_requirements)
        return self


class TrainingPreset(CanonicalModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*/v[0-9]+$")
    name: str
    source: str | None = None
    description: str = ""
    values: dict[str, Any]

    @model_validator(mode="after")
    def training_paths_only(self):
        for path in self.values:
            if not path.startswith(("train.", "native.config.", "native.overrides.")):
                raise ValueError("Presets may only set training and adapter settings")
            if any(
                p in {"__proto__", "prototype", "constructor"} for p in path.split(".")
            ):
                raise ValueError("Unsafe preset setting")
        return self


def lookup(document, path):
    value = document
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def selected_preset(document, command):
    if not command.presets:
        return None
    identifier = (
        lookup(document, "native.config.training_preset") or command.default_preset
    )
    if identifier == "custom":
        return None
    preset = next((p for p in command.presets if p.id == identifier), None)
    if preset is None:
        raise ValueError("Choose a declared training preset")
    return preset


def data_contract_error(binding, metadata, document):
    """Used for collection and imported registry versions alike."""
    accepted = binding.contracts or ([binding.contract] if binding.contract else [])
    selected = (
        lookup(document, binding.contract_selector)
        if binding.contract_selector
        else None
    )
    if selected is not None and binding.contract_choices:
        accepted = binding.contract_choices.get(str(selected), [])
        if not accepted:
            return "Choose a supported observation mode"
    if accepted and (
        metadata.get("contract") not in accepted
        or metadata.get("validation", {}).get("status") != "PASSED"
    ):
        return (
            "Dataset does not satisfy the selected data contract; prepare or import a verified "
            + " or ".join(accepted)
            + " dataset"
        )
    return None
