import copy
import inspect
import json
from types import SimpleNamespace

import pytest

from skynet_app.adapters import (
    AdapterError, ManifestAdapter, canonical_adapter_manifest,
    resolve_adapter_evaluation_plan, resolve_adapter_plan,
)
from skynet_app.adapters.egoverse_manifest import manifests, CONTRACT, REVISION
from skynet_app.adapters.egoverse_models import (
    ALGORITHMS, MODELS, model_algorithm, execution_compatibility_error,
)
from skynet_app.adapters.egoverse_runtime import (
    build_config, map_joint_model, model_settings, parser, validate_manifest,
    validate_hpt_joint_inputs,
)
from skynet_app.database import Database
from skynet_app.model_io import preview_spec, resolve_model_io
from skynet_app.pipeline_api import PipelineService
from skynet_app.training_contracts import data_contract_error
from test_experiments import make_spec
from test_pipeline import FakeCluster


def native_spec(manifest, model=None):
    fields = manifest.train.input_fields
    native = {f.path.removeprefix("native.config."): copy.deepcopy(f.default)
              for f in fields if f.default is not None}
    native.update(dataset_path="/prepared", dataset_manifest_sha256="a" * 64)
    if manifest.slug == "egoverse-pi":
        native["weights"] = "/weights"
    if model:
        native["model_preset"] = model
    return make_spec(source={
        "repository": manifest.default_repository, "revision": REVISION,
        "adapter": manifest.slug, "adapter_manifest": canonical_adapter_manifest(manifest),
    }, native={"config": native})


def test_catalog_contains_only_native_algorithms_with_complete_model_choices():
    catalog = list(manifests())
    assert [m.slug for m in catalog] == ["egoverse-act", "egoverse-hpt", "egoverse-pi"]
    choices = []
    for manifest in catalog:
        field = next(f for f in manifest.train.input_fields if f.path == "native.config.model_preset")
        choices.extend(field.choices)
        for model in field.choices:
            plan = resolve_adapter_plan(native_spec(manifest, model))
            args = parser().parse_args(plan.argv[2:])
            assert args.model == model
            assert args.algorithm == model_algorithm(model)
            assert args.algorithm == manifest.slug.removeprefix("egoverse-")
            assert not plan.blockers
    assert set(choices) == set(MODELS)
    assert len(choices) == 21
    assert "dp_joints" not in choices


def test_dataset_contract_and_model_io_follow_the_selected_preset():
    hpt = next(m for m in manifests() if m.slug == "egoverse-hpt")
    binding = next(f.data_binding for f in hpt.train.input_fields if f.data_binding)
    joint = {"contract": CONTRACT, "validation": {"status": "PASSED"}}
    assert data_contract_error(binding, joint, {"native": {"config": {"model_preset": "hpt_joints"}}}) is None
    selection = {"native": {"config": {"model_preset": "hpt_bc_flow_aria"}}}
    assert "does not satisfy" in data_contract_error(binding, joint, selection)
    native = {"contract": "egoverse.native-hpt_bc_flow_aria/v1", "validation": {"status": "PASSED"}}
    assert data_contract_error(binding, native, selection) is None
    values = resolve_model_io(canonical_adapter_manifest(hpt), preview_spec({"native.config.model_preset": "hpt_bc_flow_aria"}))
    assert any("human_bimanual" in key for key, _ in values["entries"])
    assert not any("Joint commands" in key or "scene_front" in key for key, _ in values["entries"])


@pytest.mark.parametrize("slug", ["egoverse-dp-joints", "copied-old-diffusion"])
def test_old_frozen_diffusion_manifest_cannot_train_resume_or_evaluate(slug):
    manifest = next(m for m in manifests() if m.slug == "egoverse-hpt")
    manifest.slug = slug
    manifest.train.argv.extend(["--model", "dp_joints"])
    spec = native_spec(manifest)
    before = copy.deepcopy(spec.source.adapter_manifest)
    for resumed in (False, True):
        spec.train.checkpoint.auto_resume = resumed
        with pytest.raises(AdapterError, match="synthetic EgoVerse Diffusion Policy adapter was removed"):
            resolve_adapter_plan(spec)
    with pytest.raises(AdapterError, match="historical results remain"):
        resolve_adapter_evaluation_plan(spec, environment="egoverse", suite="egoverse_held_out", context={})
    assert spec.source.adapter_manifest == before


def test_runtime_refuses_removed_model_before_loading_data_or_frameworks():
    with pytest.raises(ValueError, match="was removed"):
        validate_manifest("/unused", "a" * 64, "dp_joints")
    with pytest.raises(ValueError, match="was removed"):
        build_config(SimpleNamespace(model="dp_joints"), {})
    with pytest.raises(ValueError, match="does not belong"):
        build_config(SimpleNamespace(model="hpt_joints", algorithm="pi"), {})
    assert "diffusion" not in inspect.signature(map_joint_model).parameters
    assert "DiffusionPolicy" not in inspect.getsource(map_joint_model)


@pytest.mark.parametrize("mode", ["train", "resume", "evaluation"])
@pytest.mark.parametrize("defect", ["removed_dp", "omitted_joints"])
def test_submission_gate_blocks_retired_stored_plans_without_cluster_calls(mode, defect):
    manifest = next(m for m in manifests() if m.slug == "egoverse-hpt")
    if defect == "removed_dp":
        manifest.slug = "egoverse-dp-joints"
    else:
        manifest.train.capsule_files["adapter-support/egoverse_runtime.py"] = (
            'renames = {"state_ee_pose": "joint_positions"}'
        )
    spec = native_spec(manifest)
    frozen = ManifestAdapter(manifest).resolve(spec).model_dump(mode="json")
    document = spec.model_dump(mode="json")
    stage = {"id": "stage", "stage_type": "EVALUATE" if mode == "evaluation" else "TRAIN",
             "auto_resume": mode == "resume", "resolved_config_json": {"spec": document, "plan": frozen}}
    run = {"id": "run", "resolved_spec_json": document, "stages": [stage], "attempts": [],
           "evaluations": [{"id": "evaluation", "stage_id": "stage"}] if mode == "evaluation" else []}
    transitions = []
    service = PipelineService.__new__(PipelineService)
    service.database = SimpleNamespace(
        get_run=lambda _: copy.deepcopy(run), transition_workflow_state=lambda **kw: transitions.append(kw),
    )
    kwargs = {"manual_mode": "resume", "resume_checkpoint": "/saved.ckpt",
              "pinned_execution": {"resolved_spec": document, "plan": frozen}} if mode == "resume" else {}
    result = service._submit_stage("run", "stage", "sky2", **kwargs)
    assert result["status"] == "BLOCKED"
    assert ("was removed" if defect == "removed_dp" else "omits joint-state inputs") in result["blockers"][0]
    assert transitions[0]["stage_updates"]["status"] == "BLOCKED"
    assert transitions[0]["run_id"] is None if mode == "evaluation" else transitions[0]["run_id"] == "run"
    assert run["stages"][0]["resolved_config_json"]["plan"] == frozen


def test_legacy_hpt_mapping_cannot_be_previewed_or_used_for_evaluation():
    manifest = next(m for m in manifests() if m.slug == "egoverse-hpt")
    manifest.train.capsule_files["adapter-support/egoverse_runtime.py"] = (
        "renames = {'state_ee_pose': 'joint_positions'}"
    )
    spec = native_spec(manifest)
    with pytest.raises(AdapterError, match="latest EgoVerse HPT adapter"):
        resolve_adapter_plan(spec)
    with pytest.raises(AdapterError, match="latest EgoVerse HPT adapter"):
        resolve_adapter_evaluation_plan(spec, environment="egoverse", suite="egoverse_held_out", context={})
    for model in ("act", "hpt_bc_flow_eva", "pi0.5_bc_eva"):
        assert execution_compatibility_error(spec.source.model_dump(), {"model_preset": model}) is None


def test_native_hpt_binding_preserves_its_original_head():
    head = {"_target_": "native.FlowHead", "infer_ac_dims": {"eva_bimanual": 14},
            "model": {"act_dim": 14}, "action_horizon": 100}
    model = {"robomimic_model": {
        "shared_obs_keys": ["front_img_1"],
        "shared_stem_specs": {"front_img_1": {}}, "encoder_specs": {"front_img_1": {}},
        "stem_specs": {"eva_bimanual": {"state_ee_pose": {"input_dim": 14}}},
        "head_specs": {"eva_bimanual": head},
    }}
    mapped = map_joint_model(model, 28)["robomimic_model"]
    assert mapped["head_specs"]["skynet_joints"] is head
    assert head["_target_"] == "native.FlowHead"
    assert head["model"]["act_dim"] == 28
    assert mapped["domains"] == ["skynet_joints"]
    assert mapped["stem_specs"]["skynet_joints"]["state_joint_positions"]["input_dim"] == 28
    assert "joint_positions" not in mapped["stem_specs"]["skynet_joints"]
    validate_hpt_joint_inputs({"robomimic_model": mapped})


@pytest.mark.parametrize("stems", [{}, {"joint_positions": {"input_dim": 28}},
    {"state_joint_positions": {}, "joint_positions": {}}])
def test_hpt_rejects_missing_or_legacy_joint_inputs(stems):
    model = {"robomimic_model": {"stem_specs": {"skynet_joints": stems}}}
    with pytest.raises(ValueError, match="previous mapping omitted joint-state inputs"):
        validate_hpt_joint_inputs(model, checkpoint=True)


@pytest.mark.parametrize("override", [
    {"robomimic_model._target_": "another.Algorithm"},
    {"robomimic_model.head_specs.skynet_joints": {"_target_": "egomimic.models.diffusion_policy.DiffusionPolicy"}},
    {"robomimic_model.head_specs": {"skynet_joints": {"layers": [{"_target_": "another.Head"}]}}},
    {"optimizer.parameters": {"_target_": "another.Optimizer"}},
])
def test_model_overrides_cannot_reintroduce_removed_or_alternate_architectures(override):
    with pytest.raises(ValueError, match="without replacing architecture targets"):
        model_settings(json.dumps(override))


def test_native_numeric_and_nested_settings_remain_supported():
    override = {"robomimic_model.head_specs.skynet_joints.action_horizon": 50,
                "optimizer.lr": 1e-4, "scheduler.config": {"milestones": [5, 10]}}
    assert model_settings(json.dumps(override)) == override


def test_seeding_retires_per_config_choices_without_changing_history(tmp_path):
    database = Database(tmp_path / "registry.store")
    manifest = next(m for m in manifests() if m.slug == "egoverse-hpt")
    saved = []
    for slug in ("egoverse-hpt-joints", "egoverse-dp-joints", "egoverse-pi05-bc-aria"):
        legacy = canonical_adapter_manifest(manifest)
        legacy["slug"] = slug
        record = database.upsert_seed_adapter(seed_key=slug, name=slug, manifest=legacy)
        saved.append(copy.deepcopy(record))
    service = PipelineService(database, FakeCluster())
    visible = {r["seed_key"] for r in database.list_adapter_registry() if str(r.get("seed_key", "")).startswith("egoverse-")}
    assert visible == {"egoverse-act", "egoverse-hpt", "egoverse-pi"}
    archived = {r["id"]: r for r in database.list_adapter_registry(include_archived=True)}
    for old in saved:
        record = archived[old["id"]]
        assert record["archived_at"]
        assert record["latest_version"] == old["latest_version"]
    service._seed_registries()
    assert len([r for r in database.list_adapter_registry() if str(r.get("seed_key", "")).startswith("egoverse-")]) == 3
