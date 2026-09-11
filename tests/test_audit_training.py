"""Regression cases reproduced in the exhaustive browser audit (B14–B27/B55)."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from skynet_app.adapters import (
    CommandTemplate, ManifestAdapter, builtin_adapter_manifests,
)
from skynet_app.database import Database
from skynet_app.cluster_config import CLUSTER
from skynet_app.experiments import ExperimentSpec, EvaluationSpec, expand_sweep, get_evaluation_catalog
from skynet_app.pipeline_api import PipelineService, _sweep_from_frontend


def spec(**changes):
    value = dict(
        identity={"project": "audit", "experiment": "regression"},
        source={"repository": "https://github.com/example/test", "revision": "a" * 40, "adapter": "generic"},
        runtime={"backend": "existing"},
        native={"argv": ["python", "train.py"], "resume_argv": ["--resume"]},
    )
    value.update(changes)
    return ExperimentSpec.model_validate(value)


def test_automatic_seed_remains_implicit_after_expand_and_json_round_trip():
    generic = next(m for m in builtin_adapter_manifests() if m.slug == "generic")
    original = spec(sweep=_sweep_from_frontend(None))
    for _ in range(3):
        expanded = expand_sweep(original)[0].resolved_spec
        assert not expanded.intent.is_explicit("train.seed")
        assert not any("train.seed" in b for b in ManifestAdapter(generic).resolve(expanded).blockers)
        original = ExperimentSpec.model_validate_json(expanded.model_dump_json())
    explicit = expand_sweep(spec(sweep=_sweep_from_frontend('{"seed":[7]}')))[0].resolved_spec
    assert explicit.train.seed == 7
    assert explicit.intent.is_explicit("train.seed")
    assert any("train.seed" in b for b in ManifestAdapter(generic).resolve(explicit).blockers)


def test_unspecified_sweep_preserves_an_explicit_training_seed():
    assert expand_sweep(spec(train={"seed": 17}))[0].seed == 17


def test_null_checkpoint_warning_consumes_manifest_default_and_explicit_wins():
    manifest = next(m for m in builtin_adapter_manifests() if m.slug == "generic").model_copy(deep=True)
    manifest.defaults.checkpoint.save_before_timeout_seconds = 60
    assert PipelineService._payload_with_manifest_defaults({"checkpoint_warning_seconds": None}, manifest)["checkpoint_warning_seconds"] == 60
    assert PipelineService._payload_with_manifest_defaults({"checkpoint_warning_seconds": 120}, manifest)["checkpoint_warning_seconds"] == 120
    assert spec(train={"checkpoint": {"auto_resume": False}}, resources={"time_limit": "00:01:00"})
    assert spec(train={"checkpoint": {"auto_resume": True, "save_before_timeout_seconds": 60}}, resources={"time_limit": "00:05:00"})
    with pytest.raises(ValidationError, match="warning"):
        spec(train={"checkpoint": {"auto_resume": True}}, resources={"time_limit": "00:05:00"})


def test_registered_evaluation_metadata_uses_catalog_names_and_profiles():
    catalog = get_evaluation_catalog()
    suites = catalog.values() if isinstance(catalog, dict) else catalog
    for suite in suites:
        EvaluationSpec(adapter=suite.evaluator, suite=suite.suite, suite_version=suite.version, profile=suite.default_profile)
    assert EvaluationSpec(adapter="egoverse", suite="held_out", suite_version="1").episodes_per_task
    assert EvaluationSpec(adapter="mujoco", suite="DexJoCo", suite_version="1", profile="rand_obj").profile == "rand_obj"


def test_manual_existing_environment_accepts_no_profile_path():
    service = PipelineService.__new__(PipelineService)
    service.source_discovery = SimpleNamespace(repository_url=lambda value: value, project_subdirectory=lambda value: value)
    manifest = next(m for m in builtin_adapter_manifests() if m.slug == "generic")
    source, runtime = service._resolve_source_and_runtime(
        spec().source.model_dump(mode="python"), {"backend": "existing", "profile": None}, manifest, "sky2",
    )
    assert runtime["profile"] == "default"
    assert runtime["resolution"]["mode"] == "manual"


def test_dataset_bound_evaluation_without_bundle_reports_actionable_validation():
    from skynet_app.evaluation_contracts import bound_metadata
    with pytest.raises(ValueError, match="one registered training dataset"):
        bound_metadata({"data": {"bundle": None}}, {"role": "training_data", "metadata_path": "split.validation"})


@pytest.mark.parametrize("assignment", [None, "legacy string", {"mount_path": "../escape"}])
def test_consumed_legacy_bundle_assignments_are_validated_before_binding(assignment):
    manifest = next(m for m in builtin_adapter_manifests() if m.slug == "generic")
    with pytest.raises(ValueError, match="assignments must be objects|Mount path"):
        PipelineService._apply_manifest_data_bindings({"data": {"bundle": {"assignments": [assignment]}}}, manifest)


def test_swept_native_values_cannot_bypass_declared_input_validation():
    manifest = next(m for m in builtin_adapter_manifests() if m.slug == "generic").model_copy(deep=True)
    manifest.train = CommandTemplate(argv=["python", "train.py"], strict_native_config=True)
    with pytest.raises(ValueError, match="declared training settings"):
        PipelineService._validate_sweep_inputs(spec(sweep=_sweep_from_frontend('{"unsupported.key":[1,2]}')), manifest)
    valid = PipelineService._validate_sweep_inputs(spec(sweep=_sweep_from_frontend('{"seed":[1,2]}')), manifest)
    assert [v.seed for v in expand_sweep(valid)] == [1, 2]


@pytest.mark.parametrize("quota,expected", [("0 / 8 | 16 / 16 | 0 / 0", "rl2-lab"), ("8 / 8 | 16 / 16 | 0 / 0", "overcap")])
def test_any_gpu_queue_uses_available_columns_without_persisting_snapshot(quota, expected):
    service = PipelineService.__new__(PipelineService)
    service.cluster = SimpleNamespace(run_with_fallback=lambda *args, **kwargs: ("sky2", "| rl2-lab | " + quota + " |"))
    value = spec(resources={"queue_policy": "auto", "gpu": {"mode": "explicit", "count": 1, "type": "any"}})
    plan = ManifestAdapter(next(m for m in builtin_adapter_manifests() if m.slug == "generic")).resolve(value)
    resolved, snapshot = service._auto_queue(value, plan, "sky2", record_snapshot=False)
    assert resolved.resources.account == expected
    assert snapshot is None


def test_preview_uses_same_live_queue_resolution_as_create():
    service = PipelineService.__new__(PipelineService)
    service.storage = SimpleNamespace(work_root=CLUSTER.paths.work_root)
    value = spec(resources={"queue_policy": "auto", "gpu": {"mode": "explicit", "count": 1, "type": "a40"}})
    service.normalize_spec = lambda payload: value
    service._validate_tracking_requirements = lambda value: None
    service._repository_argument_validation = lambda *args: None
    calls = []
    def resolve(value, plan, gateway, *, record_snapshot):
        calls.append(record_snapshot)
        return value, None
    service._auto_queue = resolve
    assert service.preview({})["variant_count"] == 1
    assert calls == [False]


def test_terminal_evaluation_finalizes_only_unfinished_episodes_and_can_resume(tmp_path):
    db = Database(tmp_path / "episodes.db")
    project = db.create_project("audit")
    experiment = db.create_experiment(project_id=project["id"], name="test", requested_spec={})
    variant = db.create_variant(experiment["latest_revision"]["id"], name="test", parameters={}, resolved_spec={})
    run = db.create_run(variant["id"], seed=0, adapter_name="generic", adapter_version="1", run_directory="/tmp/audit")
    evaluation = db.create_evaluation(run["id"], evaluator_adapter="generic", evaluator_version="1", suite_name="test", suite_version="1", tasks=["task"], seeds=[0], episodes_per_task=2)
    db.upsert_evaluation_episode(evaluation["id"], task="task", seed=0, episode_index=0, status="SUCCEEDED", success=True)
    db.upsert_evaluation_episode(evaluation["id"], task="task", seed=0, episode_index=1)
    db.update_evaluation(evaluation["id"], status="CANCELLED")
    episodes = db.get_evaluation(evaluation["id"])["episodes"]
    assert [e["status"] for e in episodes] == ["SUCCEEDED", "CANCELLED"]
    assert "before this episode completed" in episodes[1]["failure_reason"]
    db.update_evaluation(evaluation["id"], status="PENDING")
    assert [e["status"] for e in db.get_evaluation(evaluation["id"])["episodes"]] == ["SUCCEEDED", "PENDING"]
    # A historical terminal parent stored by older builds is normalized on read.
    with db.transaction() as connection:
        connection.execute("UPDATE evaluations SET status = 'FAILED' WHERE id = ?", (evaluation["id"],))
    assert [e["status"] for e in db.get_evaluation(evaluation["id"])["episodes"]] == ["SUCCEEDED", "NOT_COMPLETED"]
    with db.connection() as connection:
        assert connection.execute("SELECT status FROM evaluation_episodes WHERE id = ?", (episodes[1]["id"],)).fetchone()["status"] == "PENDING"


def test_preview_validation_response_excludes_library_dump_and_input(monkeypatch):
    from skynet_app import pipeline_api
    def reject(payload):
        return spec(train={"learning_rate": 0})
    monkeypatch.setattr(pipeline_api.service, "preview", reject)
    with pytest.raises(HTTPException) as caught:
        pipeline_api.preview_experiment({})
    assert caught.value.status_code == 422
    detail = caught.value.detail
    assert detail[0]["loc"] == ("train", "learning_rate")
    assert "url" not in detail[0] and "input" not in detail[0]
