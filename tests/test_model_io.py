"""I/O previews and saved receipts share model- and dataset-specific dimensions."""
from copy import deepcopy
import pytest
from skynet_app.adapters import builtin_adapter_manifests, canonical_adapter_manifest, AdapterManifest
from skynet_app.model_io import resolve_model_io, preview_spec, adapter_io_contract
from skynet_app.pipeline_api import PipelineService
from test_experiments import make_spec
from skynet_app.adapters import ManifestAdapter

@pytest.fixture(scope="module")
def manifests():
    return {m.slug: canonical_adapter_manifest(m) for m in builtin_adapter_manifests()}

@pytest.fixture
def bundle():
    return {"assignments": [{"role": "training_data", "config": {"location": {"path": "/cluster/prepared"}}, "version": {"manifest_sha256": "abc", "metadata": {"policy_to_source_indices": list(range(28)), "capture": {"cameras": {key: {"height": 256, "width": 256} for key in ("scene_front", "scene_left", "scene_right")}}}}}]}


def test_modalities_use_policy_resolution_not_capture_resolution(manifests, bundle):
    dp = manifests["xpolicylab-dp"]
    state = resolve_model_io(dp, preview_spec({}, bundle))
    assert not any("RGB" in key for key, _ in state["entries"])
    assert dict(state["entries"])["Output · Joint commands"] == "16 × 28 (steps × values)"
    rgb = resolve_model_io(dp, preview_spec({"native.config.observation_mode": "rgb", "native.config.observation_steps": 4, "native.config.action_steps": 32}, bundle))
    assert dict(rgb["entries"])["Input · RGB · scene_front"] == "4 × 240 × 320 × 3 (timesteps × height × width × channels)"
    assert dict(rgb["entries"])["Output · Joint commands"] == "32 × 28 (steps × values)"
    act = resolve_model_io(manifests["xpolicylab-act"], preview_spec({}, bundle))
    assert "480 × 640 × 3" in dict(act["entries"])["Input · RGB · scene_left"]


def test_native_flat_overrides_and_saved_dataset_resolve_together(manifests, bundle):
    spec = preview_spec({"native.config.model_overrides": {"robomimic_model.chunk_size": 25}}, bundle)
    before = deepcopy(spec)
    summary = resolve_model_io(manifests["egoverse-act"], spec)
    assert spec == before
    assert summary["resolved"]
    assert dict(summary["entries"])["Output · Joint commands"] == "25 × 28 (steps × values)"
    assert "256 × 256 × 3" in dict(summary["entries"])["Input · RGB · scene_right"]


@pytest.mark.parametrize("values", [{"native.config.dataset_path": "/another/dataset"}, {"native.config.dataset_manifest_sha256": "different"}])
def test_changed_dataset_binding_cannot_borrow_dimensions(manifests, bundle, values):
    summary = resolve_model_io(manifests["egoverse-act"], preview_spec(values, bundle))
    assert not summary["resolved"]
    assert "100 × ?" in dict(summary["entries"])["Output · Joint commands"]


def test_native_heads_are_not_assumed_to_match_state_size(manifests):
    summary = resolve_model_io(manifests["egoverse-hpt-cotrain-flow-shared-head"])
    assert "1 × 12" in dict(summary["entries"])["Input · human_bimanual · state_ee_pose"]
    assert "100 × 14" in dict(summary["entries"])["Output · human_bimanual actions"]
    pi = resolve_model_io(manifests["egoverse-pi05-bc-aria"])
    assert "100 × 32" in dict(pi["entries"])["Output · Padded action tensor"]
    for slug, manifest in manifests.items():
        summary = resolve_model_io(manifest)
        assert summary["entries"], slug
        if slug.startswith("egoverse-"):
            assert manifest["train"]["model_io"], slug


def test_legacy_receipt_is_derived_only_from_saved_settings(manifests, bundle):
    manifest = deepcopy(manifests["egoverse-act"])
    manifest["train"].pop("model_io")
    assert canonical_adapter_manifest(AdapterManifest.model_validate(manifest)) == manifest
    spec = preview_spec({"native.config.model_overrides": {"robomimic_model.chunk_size": 17}}, bundle)
    summary = resolve_model_io(manifest, spec, legacy=True)
    assert "17 × 28" in dict(summary["entries"])["Output · Joint commands"]
    assert "model_io" not in manifest["train"]


def test_new_attempt_snapshot_freezes_resolved_io(manifests):
    spec = make_spec()
    manifest = manifests["egoverse-act"]
    spec.source.adapter = "egoverse-act"
    spec.source.adapter_manifest = manifest
    spec.native.config = {"model_overrides": {"robomimic_model.chunk_size": 17}}
    plan = ManifestAdapter(AdapterManifest.model_validate(manifest)).resolve(spec)
    snapshot = PipelineService._attempt_execution_snapshot(spec, plan, {})
    assert "17 × ?" in dict(snapshot["model_io"]["entries"])["Output · Joint commands"]
    spec.native.config["model_overrides"]["robomimic_model.chunk_size"] = 99
    assert "17 × ?" in dict(snapshot["model_io"]["entries"])["Output · Joint commands"]


def test_invalid_native_size_is_not_replaced_with_a_default(manifests):
    spec = preview_spec({"native.config.model_overrides": {"robomimic_model.chunk_size": 0}})
    result = resolve_model_io(manifests["egoverse-act"], spec)
    assert dict(result["entries"])["Output · Joint commands"] == "? × ? (steps × values)"
