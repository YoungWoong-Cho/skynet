"""Skynet must not silently omit a supplied override missing its intent marker."""

import pytest

from skynet_app.adapters import (
    AdapterCapabilities, AdapterDefaults, AdapterHyperparameterDefaults,
    AdapterManifest, AdapterRuntimePolicy, ArgumentBinding, CommandTemplate,
    ManifestAdapter,
)
from skynet_app.experiments import ExperimentSpec
from skynet_app.pipeline_api import PipelineService
from test_experiments import make_spec


PATH = "train.batch.gradient_accumulation_steps"


def declared_adapter(*, strict=True, accumulation=4):
    return AdapterManifest(
        slug="parameter-intent", display_name="Parameter intent test",
        runtime=AdapterRuntimePolicy(allowed_backends={"existing"}),
        capabilities=AdapterCapabilities(name="parameter-intent", runtime_backends={"existing"},
            supports_multi_gpu_single_node=True, supports_resume=True, maximum_gpus=8),
        defaults=AdapterDefaults(hyperparameters=AdapterHyperparameterDefaults(
            gradient_accumulation_steps=accumulation, learning_rate=1e-4)),
        train=CommandTemplate(argv=["python", "train.py"], strict_canonical_inputs=strict,
            resume_argv=["--checkpoint", "{{tokens.resume_checkpoint}}"],
            supported_canonical_fields=[PATH, "train.learning_rate",
                                        "train.batch.declared_semantics", "train.batch.value"],
            parameter_flags={PATH: ArgumentBinding(flag="--gradient-accumulation"),
                             "train.learning_rate": ArgumentBinding(flag="--learning-rate")}),
    )


@pytest.mark.parametrize("strict", [True, False])
def test_supplied_implicit_override_cannot_silently_fall_back_to_declared_default(strict):
    manifest = declared_adapter(strict=strict)
    spec = make_spec(native={}, train={"batch": {"gradient_accumulation_steps": 1}},
                     intent={"explicit_parameters": []})
    plan = ManifestAdapter(manifest).resolve(spec)
    assert "--gradient-accumulation" not in plan.argv
    assert not plan.runnable
    message = next(item for item in plan.blockers if "not marked explicit" in item)
    assert PATH + "=1" in message and "inherited default 4" in message
    assert "intent.explicit_parameters" in message


@pytest.mark.parametrize("intent", [[PATH], ["train.batch"]])
def test_explicit_override_emits_exact_requested_value_including_schema_default(intent):
    spec = make_spec(native={}, train={"batch": {"gradient_accumulation_steps": 1}},
                     intent={"explicit_parameters": intent})
    plan = ManifestAdapter(declared_adapter()).resolve(spec)
    assert plan.runnable, plan.blockers
    assert plan.argv == ["python", "train.py", "--gradient-accumulation", "1"]


def test_genuinely_omitted_fields_and_matching_inherited_values_keep_native_defaults():
    manifest = declared_adapter()
    for train in ({}, {"batch": {}}, {"batch": {"gradient_accumulation_steps": 4}},
                  {"learning_rate": None}):
        plan = ManifestAdapter(manifest).resolve(make_spec(
            native={}, train=train, intent={"explicit_parameters": []}))
        assert plan.runnable, plan.blockers
        assert plan.argv == ["python", "train.py"]


def test_normalized_defaults_survive_saved_spec_round_trip_and_changed_values_are_blocked():
    manifest = declared_adapter()
    payload = make_spec(native={}).model_dump(mode="json", by_alias=True)
    payload.pop("train")
    PipelineService._apply_canonical_manifest_defaults(payload, manifest)
    normalized = ExperimentSpec.model_validate(payload)
    restored = ExperimentSpec.model_validate(normalized.model_dump(mode="json", by_alias=True))
    plan = ManifestAdapter(manifest).resolve(restored)
    assert plan.runnable, plan.blockers
    assert plan.argv == ["python", "train.py"]
    assert restored.train.batch.gradient_accumulation_steps == 4
    changed = restored.model_dump(mode="json", by_alias=True)
    changed["train"]["batch"]["gradient_accumulation_steps"] = 1
    assert not ManifestAdapter(manifest).resolve(ExperimentSpec.model_validate(changed)).runnable
    changed["intent"]["explicit_parameters"].append(PATH)
    fixed = ManifestAdapter(manifest).resolve(ExperimentSpec.model_validate(changed))
    assert fixed.runnable, fixed.blockers
    assert fixed.argv[-2:] == ["--gradient-accumulation", "1"]


def test_unknown_native_defaults_and_manual_commands_are_not_guessed():
    implicit = make_spec(native={}, train={"batch": {"gradient_accumulation_steps": 2}},
                         intent={"explicit_parameters": []})
    plan = ManifestAdapter(declared_adapter(accumulation=None)).resolve(implicit)
    assert plan.runnable, plan.blockers
    assert plan.argv == ["python", "train.py"]
    manual = implicit.model_copy(update={"native": implicit.native.model_copy(
        update={"argv": ["python", "custom.py", "--gradient-accumulation", "2"]})})
    plan = ManifestAdapter(declared_adapter()).resolve(manual)
    assert plan.runnable, plan.blockers
    assert plan.argv == manual.native.argv


def test_shared_retention_policy_needs_no_native_mapping_but_mapped_defaults_cannot_conflict():
    path = "train.checkpoint.keep_last"
    declared = declared_adapter()
    explicit = make_spec(native={}, train={"checkpoint": {"keep_last": 2}},
                         intent={"explicit_parameters": [path]})
    shared_only = ManifestAdapter(declared).resolve(explicit)
    assert shared_only.runnable, shared_only.blockers
    assert shared_only.argv == ["python", "train.py"]

    declared.defaults.checkpoint.keep_last = 3
    declared.train.parameter_flags[path] = ArgumentBinding(flag="--keep-checkpoints")
    implicit = make_spec(native={}, train={"checkpoint": {"keep_last": 2}},
                         intent={"explicit_parameters": []})
    blocked = ManifestAdapter(declared).resolve(implicit)
    assert not blocked.runnable
    assert any(path in item and "not marked explicit" in item for item in blocked.blockers)
    mapped = ManifestAdapter(declared).resolve(explicit)
    assert mapped.runnable, mapped.blockers
    assert mapped.argv[-2:] == ["--keep-checkpoints", "2"]
