"""Behavioral checks for shared adapter contracts and input defaults."""

import copy
import pytest
from test_experiments import make_spec

from skynet_app.adapters import (
    AdapterManifest,
    AdapterInputField,
    DataBundleInputBinding,
    ManifestAdapter,
    builtin_adapter_manifests,
    canonical_adapter_manifest,
)
from skynet_app.pipeline_api import (
    PipelineService,
    _resolve_effective_common_hyperparameters,
)
from skynet_app.training_contracts import data_contract_error


def act():
    return next(m for m in builtin_adapter_manifests() if m.slug == "xpolicylab-act")


def test_every_adapter_declares_data_requirements_and_rejects_ignored_common_overrides():
    for manifest in builtin_adapter_manifests():
        assert manifest.train.data_requirements is not None
        assert manifest.train.strict_canonical_inputs
        # An explicit value is rejected even when it equals the schema default.
        if "train.max_steps" not in manifest.train.supported_canonical_fields:
            spec = make_spec(train={"max_steps": 1000})
            spec.source.adapter = manifest.slug
            spec.resources.gpu.count = 1
            spec.intent.explicit_parameters = ["train.max_steps"]
            plan = ManifestAdapter(manifest).resolve(spec)
            assert any("train.max_steps" in b for b in plan.blockers)


def test_preset_applies_canonical_and_native_defaults_preserving_explicit_overrides():
    manifest = act()
    doc = {
        "intent": {"explicit_parameters": ["train.batch.value"]},
        "train": {"batch": {"value": 4}},
        "native": {"config": {"training_preset": "xpolicylab-act/v1", "epochs": 7}},
    }
    PipelineService._apply_training_preset(doc, manifest)
    PipelineService._apply_manifest_input_defaults(doc, manifest)
    assert doc["train"]["batch"]["value"] == 4
    assert doc["native"]["config"]["epochs"] == 7
    assert doc["native"]["config"]["action_steps"] == 50
    assert doc["native"]["config"]["hidden_dim"] == 512
    with pytest.raises(ValueError, match="preset"):
        PipelineService._apply_training_preset(
            {"native": {"config": {"training_preset": "typo"}}}, manifest
        )


def test_custom_training_values_survive_defaults_and_validation():
    manifest = act()
    doc = {"train": {"batch": {"value": 128}, "learning_rate": 0.002},
           "native": {"config": {"training_preset": "custom", "epochs": 7,
                       "dataset_path": "/prepared", "dataset_manifest_sha256": "a" * 64}}}
    PipelineService._apply_training_preset(doc, manifest)
    PipelineService._apply_manifest_input_defaults(doc, manifest)
    PipelineService._validate_manifest_input_fields(doc, manifest)
    assert doc["train"]["batch"]["value"] == 128
    assert doc["train"]["learning_rate"] == 0.002
    assert doc["native"]["config"]["training_preset"] == "custom"
    assert doc["native"]["config"]["epochs"] == 7


def test_input_bounds_and_unknown_settings_are_rejected_before_launch():
    manifest = act().model_copy(deep=True)
    # Generic selector and dependent-bound validation are tested explicitly;
    # these fixture settings are not claimed as ACT model configuration.
    manifest.train.input_fields.extend([
        AdapterInputField(path="native.config.fixture_mode", label="Fixture mode", kind="string", default="one", choices=["one", "two"]),
        AdapterInputField(path="native.config.fixture_window", label="Fixture window", kind="integer", default=1, maximum_path="native.config.action_steps"),
    ])
    base = {
        "native": {
            "config": {"dataset_path": "/prepared", "dataset_manifest_sha256": "a" * 64}
        }
    }
    PipelineService._apply_training_preset(base, manifest)
    PipelineService._apply_manifest_input_defaults(base, manifest)
    for key, value, message in [
        ("weight_decay", 2, "maximum"),
        ("epochs", 0, "minimum"),
        ("fixture_mode", "invalid", "choices"),
        ("ignored_parameter", 1, "Unsupported"),
        ("fixture_window", 51, "cannot exceed"),
    ]:
        doc = copy.deepcopy(base)
        doc["native"]["config"][key] = value
        with pytest.raises(ValueError, match=message):
            PipelineService._validate_manifest_input_fields(doc, manifest)


def test_same_data_contract_applies_to_imported_and_collected_versions():
    binding = DataBundleInputBinding(
        role="training_data",
        contracts=["test.recording-state/v1", "test.recording-rgb/v1"],
        contract_selector="native.config.observation_mode",
        contract_choices={"state": ["test.recording-state/v1", "test.recording-rgb/v1"],
                          "rgb": ["test.recording-rgb/v1"]},
    )
    # The validator deliberately has no collection provider/session dependency.
    for provider in ["collection", "huggingface", "filesystem"]:
        metadata = {
            "provider": provider,
            "contract": "test.recording-state/v1",
            "validation": {"status": "PASSED"},
        }
        assert (
            data_contract_error(
                binding, metadata, {"native": {"config": {"observation_mode": "state"}}}
            )
            is None
        )
        assert data_contract_error(
            binding, metadata, {"native": {"config": {"observation_mode": "rgb"}}}
        )
        metadata["contract"] = "test.recording-rgb/v1"
        assert (
            data_contract_error(
                binding, metadata, {"native": {"config": {"observation_mode": "state"}}}
            )
            is None
        )
        metadata["validation"]["status"] = "FAILED"
        assert data_contract_error(binding, metadata, {})


def test_common_receipt_does_not_report_unmapped_defaults_as_applied():
    manifest = canonical_adapter_manifest(act())
    spec = {
        "train": {"max_steps": 1000},
        "native": {"config": {"epochs": 300, "weight_decay": 1e-4}},
        "intent": {"explicit_parameters": ["train.max_steps"]},
    }
    receipt = _resolve_effective_common_hyperparameters(spec, manifest)
    assert receipt["values"]["max_steps"] is None
    assert receipt["values"]["max_epochs"] == 300
    assert receipt["adapter_settings"]["native.config.weight_decay"] == 1e-4




def test_old_pinned_manifests_do_not_gain_new_contract_fields():
    raw = canonical_adapter_manifest(act())
    for key in [
        "data_requirements",
        "presets",
        "default_preset",
        "strict_canonical_inputs",
        "strict_native_config",
    ]:
        raw["train"].pop(key)
    assert canonical_adapter_manifest(AdapterManifest.model_validate(raw)) == raw


def test_training_adapters_respect_their_declared_gpu_capabilities():
    for manifest in builtin_adapter_manifests():
        spec = make_spec()
        spec.source.adapter = manifest.slug
        count = max(manifest.capabilities.minimum_gpus, min(2, manifest.capabilities.maximum_gpus))
        spec.resources.gpu.count = count
        spec.native.argv = []
        plan = ManifestAdapter(manifest).resolve(spec)
        assert not any("one GPU" in issue or "1-1 GPUs" in issue for issue in plan.blockers), manifest.slug
        if manifest.slug == "xpolicylab-act":
            flag = plan.argv.index("--gpu-count")
            assert plan.argv[flag + 1] == str(count)
            assert "adapter-support/training_parallel.py" in plan.capsule_files


def test_dexmimicgen_keeps_old_pinned_launcher_and_versions_new_bridge():
    manifest = next(m for m in builtin_adapter_manifests() if m.slug == "dexmimicgen")
    spec = make_spec()
    spec.source.adapter = manifest.slug
    spec.native.argv = []
    spec.native.config = {"training_config": "/cluster/train.json", "robomimic_revision": "a" * 40}
    spec.resources.gpu.count = 2
    spec.source.adapter_manifest = canonical_adapter_manifest(manifest)
    plan = ManifestAdapter(manifest).resolve(spec)
    assert "robomimic_training.py" in plan.argv[1]
    assert plan.argv[plan.argv.index("--gpu-count") + 1] == "2"
    assert "adapter-support/robomimic_training.py" in plan.capsule_files
    old = manifest.model_copy(deep=True)
    old.train.capsule_files = {}
    old.capabilities.maximum_gpus = 1
    old.capabilities.supports_multi_gpu_single_node = False
    spec.source.adapter_manifest = canonical_adapter_manifest(old)
    spec.resources.gpu.count = 1
    plan = ManifestAdapter(old).resolve(spec)
    assert plan.argv == ["python", "-m", "robomimic.scripts.train", "--config", "/cluster/train.json"]
    assert not plan.capsule_files
