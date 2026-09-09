"""Behavioral checks for shared adapter contracts and state-only conversion."""

import copy
import json
import pickle

import numpy as np
import pytest
import zarr
from test_experiments import make_spec
from test_policy_exports import digest
from test_policy_exports import setup as setup

from skynet_app.adapters import (
    AdapterManifest,
    ManifestAdapter,
    builtin_adapter_manifests,
    canonical_adapter_manifest,
)
from skynet_app.pipeline_api import (
    PipelineService,
    _resolve_effective_common_hyperparameters,
)
from skynet_app.training_contracts import data_contract_error


def dp():
    return next(m for m in builtin_adapter_manifests() if m.slug == "xpolicylab-dp")


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
    manifest = dp()
    doc = {
        "intent": {"explicit_parameters": ["train.batch.value"]},
        "train": {"batch": {"value": 4}},
        "native": {"config": {"training_preset": "rgb-joints/v2", "epochs": 7}},
    }
    PipelineService._apply_training_preset(doc, manifest)
    PipelineService._apply_manifest_input_defaults(doc, manifest)
    assert doc["train"]["batch"]["value"] == 4
    assert doc["native"]["config"]["epochs"] == 7
    assert doc["native"]["config"]["observation_mode"] == "rgb"
    assert doc["native"]["config"]["inference_steps"] == 20
    with pytest.raises(ValueError, match="preset"):
        PipelineService._apply_training_preset(
            {"native": {"config": {"training_preset": "typo"}}}, manifest
        )


def test_custom_training_values_survive_defaults_and_validation():
    manifest = dp()
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
    manifest = dp()
    base = {
        "native": {
            "config": {"dataset_path": "/prepared", "dataset_manifest_sha256": "a" * 64}
        }
    }
    PipelineService._apply_training_preset(base, manifest)
    PipelineService._apply_manifest_input_defaults(base, manifest)
    for key, value, message in [
        ("ema_decay", 1, "maximum"),
        ("epochs", 0, "minimum"),
        ("action_steps", 15, "choices"),
        ("ignored_parameter", 1, "Unsupported"),
        ("inference_steps", 101, "cannot exceed"),
    ]:
        doc = copy.deepcopy(base)
        doc["native"]["config"][key] = value
        with pytest.raises(ValueError, match=message):
            PipelineService._validate_manifest_input_fields(doc, manifest)


def test_same_data_contract_applies_to_imported_and_collected_versions():
    binding = next(f.data_binding for f in dp().train.input_fields if f.data_binding)
    # The validator deliberately has no collection provider/session dependency.
    for provider in ["collection", "huggingface", "filesystem"]:
        metadata = {
            "provider": provider,
            "contract": "skynet.dp-joints/v1",
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
        metadata["contract"] = "skynet.dp-rgb-joints/v1"
        assert (
            data_contract_error(
                binding, metadata, {"native": {"config": {"observation_mode": "state"}}}
            )
            is None
        )
        metadata["validation"]["status"] = "FAILED"
        assert data_contract_error(binding, metadata, {})


def test_common_receipt_does_not_report_unmapped_defaults_as_applied():
    manifest = canonical_adapter_manifest(dp())
    spec = {
        "train": {"max_steps": 1000},
        "native": {"config": {"epochs": 300, "weight_decay": 1e-4}},
        "intent": {"explicit_parameters": ["train.max_steps"]},
    }
    receipt = _resolve_effective_common_hyperparameters(spec, manifest)
    assert receipt["values"]["max_steps"] is None
    assert receipt["values"]["max_epochs"] == 300
    assert receipt["adapter_settings"]["native.config.weight_decay"] == 1e-4


def test_state_conversion_without_images_preserves_data_and_split(setup):
    import h5py

    service, session, source = setup
    expected = []
    for index, name in enumerate(session["recordings"]):
        path = source / name
        payload = pickle.loads(path.read_bytes())
        sidecar = source / session["recording_images"][name]["path"]
        with h5py.File(sidecar) as f:
            payload["skynet_state_metadata"] = json.loads(f.attrs["metadata"])
            expected.append((f["state"][:], f["action"][:]))
        path.write_bytes(pickle.dumps(payload))
        session["recording_checksums"][name] = digest(path)
        sidecar.unlink()
    session["recording_images"] = {}
    job = service.create(session["id"], "dp-state", "State data")
    service.prepare(job["id"])
    result = service.get(job["id"])
    assert result["state"] == "READY", result
    output = service.root / job["id"] / "output"
    group = zarr.open_group(str(output / "dataset/demonstrations.zarr"), mode="r")
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(group["data"].keys()) == {"state", "action"}
    assert manifest["camera_slots"] == {}
    assert manifest["contract"] == "skynet.dp-joints/v1"
    assert manifest["split"]["train"] and manifest["split"]["validation"]
    order = manifest["policy_to_source_indices"]
    np.testing.assert_array_equal(
        group["data/action"][:][:, np.argsort(order)],
        np.concatenate([v[1] for v in expected]),
    )
    np.testing.assert_array_equal(
        group["data/state"][:][:, np.argsort(order)],
        np.concatenate([v[0] for v in expected]),
    )
    with pytest.raises(ValueError, match="no completed training images"):
        service.create(session["id"], "dp", "RGB data")


def test_old_pinned_manifests_do_not_gain_new_contract_fields():
    raw = canonical_adapter_manifest(dp())
    for key in [
        "data_requirements",
        "presets",
        "default_preset",
        "strict_canonical_inputs",
        "strict_native_config",
    ]:
        raw["train"].pop(key)
    assert canonical_adapter_manifest(AdapterManifest.model_validate(raw)) == raw


def test_all_current_training_adapters_accept_multiple_gpu_allocations():
    for manifest in builtin_adapter_manifests():
        assert manifest.capabilities.supports_multi_gpu_single_node, manifest.slug
        assert manifest.capabilities.maximum_gpus >= 2, manifest.slug
        spec = make_spec()
        spec.source.adapter = manifest.slug
        spec.resources.gpu.count = 2
        spec.native.argv = []
        plan = ManifestAdapter(manifest).resolve(spec)
        assert not any("one GPU" in issue or "1-1 GPUs" in issue for issue in plan.blockers), manifest.slug
        if manifest.slug.startswith("xpolicylab-"):
            flag = plan.argv.index("--gpu-count")
            assert plan.argv[flag + 1] == "2"
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
