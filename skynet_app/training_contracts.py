"""Shared, versioned training requirements and presets for every adapter.

Requirements describe the trainer, independently of where a dataset originated.
Converters advertise the contracts they produce; they never imply that every
dataset with the same file extension has the same robot or observation layout.
"""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, model_validator

from .experiments import CanonicalModel


class DatasetRequirement(CanonicalModel):
    description: str
    mode: Literal["dataset", "simulation", "custom"] = "dataset"
    observations: list[str] = Field(default_factory=list)
    action_representation: str | None = None


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
