import copy

import pytest

from skynet_app.adapters import builtin_adapter_manifests, canonical_adapter_manifest, resolve_adapter_plan
from skynet_app.cluster_config import CLUSTER
from skynet_app.database import Database
from skynet_app.evaluation_placement import resolve_evaluation_resources
from skynet_app.experiments import ResourceSpec
from skynet_app.pipeline_api import EvaluationRequest
from skynet_app.runtime_readiness import render_readiness_sbatch
from skynet_app.slurm import SlurmCompileError, compile_sbatch
from test_pipeline import FakeCluster, _create_submitted_run, make_pipeline_service
from test_slurm import make_spec


@pytest.mark.parametrize("context,profile", [
    ({"environment": "isaac_sim"}, None),
    ({"environment": "isaac_lab"}, None),
    ({"suite": {"config": {"evaluator": "isaac_lab"}}}, None),
    ({"evaluator": {"adapter": "isaac_sim"}}, None),
    ({"evaluator_runtime": {"versions": {"isaac_sim": "5.1.0"}}}, None),
    ({}, "isaacsim-5.1.0_isaaclab-2.3.2_py311"),
    ({}, "groot-isaacsim-5.0.0_isaaclab-2.2.0_py311"),
])
@pytest.mark.parametrize("gpu,node", [("l40s", "grom"), ("a40", "megazord"), ("any", "grom")])
def test_compiler_enforces_policy_for_simulator_declarations_and_runtime_pins(context, profile, gpu, node):
    spec = make_spec(account="overcap")
    spec.resources.gpu.gpu_type = gpu
    plan = resolve_adapter_plan(spec)
    plan.native_config["canonical_evaluation"] = context
    if profile:
        plan.native_config["evaluation_runtime_profile_id"] = profile
    compiled = compile_sbatch(spec, plan, run_id="eval-placement", stage="eval", stage_auto_resume=True)
    assert f"#SBATCH --nodelist={node}\n" in compiled.script
    assert compiled.script.count("#SBATCH --nodelist=") == 1
    assert f"#SBATCH --gres=gpu:{'a40' if node == 'megazord' else 'l40s'}:4" in compiled.script
    assert "#SBATCH --partition=overcap" in compiled.script
    assert "#SBATCH --requeue" in compiled.script
    assert spec.resources.node.mode == "auto"  # No mutation of pinned input.


@pytest.mark.parametrize("node,gpu,count,error", [
    ("bishop", "l40s", 1, "not allowed"),
    ("baymax", "a40", 1, "not allowed"),
    ("grom", "a40", 1, "requires l40s"),
    ("megazord", "l40s", 1, "requires a40"),
    (None, "rtx_6000", 1, "no compatible"),
    (None, "a40", 9, "at most 8"),
])
def test_compiler_rejects_unsafe_or_impossible_placements(node, gpu, count, error):
    spec = make_spec(node={"mode": "manual", "name": node} if node else None)
    spec.resources.gpu.gpu_type = gpu
    spec.resources.gpu.count = count
    plan = resolve_adapter_plan(spec)
    plan.native_config["canonical_evaluation"] = {"environment": "isaac_lab"}
    with pytest.raises(SlurmCompileError, match=error):
        compile_sbatch(spec, plan, run_id="unsafe-eval", stage="eval")


@pytest.mark.parametrize("stage,environment", [
    ("train", "isaac_lab"), ("eval", "isaac_gym"),
    ("eval", "libero"), ("eval", "egoverse"),
])
def test_training_and_other_evaluators_keep_their_placement(stage, environment):
    spec = make_spec(node={"mode": "manual", "name": "bishop"})
    plan = resolve_adapter_plan(spec)
    plan.native_config["canonical_evaluation"] = {"environment": environment}
    assert "#SBATCH --nodelist=bishop" in compile_sbatch(spec, plan, run_id="unrelated", stage=stage).script


def test_isaac_policy_fails_closed_when_cluster_has_no_allowed_nodes(monkeypatch):
    monkeypatch.setattr(CLUSTER, "isaac_evaluation_placement", None)
    with pytest.raises(ValueError, match="requires configured compatible nodes"):
        resolve_evaluation_resources(make_spec().resources, {"environment": "isaac_lab"})


@pytest.fixture
def evaluation_service(tmp_path, monkeypatch):
    monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", tmp_path / "capsules")
    cluster = FakeCluster()
    service = make_pipeline_service(Database(tmp_path / "test.db"), cluster)
    run = _create_submitted_run(service, "placement")
    stage = next(s for s in run["stages"] if s["stage_type"] == "TRAIN")
    service.database.update_job_attempt(run["attempts"][0]["id"], status="SUCCEEDED", slurm_state="COMPLETED")
    service.database.update_stage(stage["id"], status="SUCCEEDED")
    service.database.update_run(run["id"], status="SUCCEEDED")
    service.database.create_checkpoint(run["id"], produced_by_attempt_id=run["attempts"][0]["id"],
        checkpoint_type="INFERENCE", path="/checkpoint", sha256="a" * 64, size_bytes=1,
        is_selected_for_inference=True)
    suite = next(s for s in service.database.list_evaluation_suites() if s["config_json"]["evaluator"] == "isaac_sim"
                 and "custom" in s["name"])
    return service, cluster, run, suite


def request_for(run, suite, gpu="l40s", node=None):
    return EvaluationRequest(run_id=run["id"], checkpoint_path="/checkpoint", suite_id=suite["id"],
        tasks=[], episodes_per_task=1, seeds=[42], argv=["python", "custom-eval.py"],
        resources=ResourceSpec.model_validate({"queue_policy": "normal", "gpu": {"mode": "explicit", "count": 1, "type": gpu},
                                              "node": {"mode": "manual", "name": node} if node else {"mode": "auto"}}))


@pytest.mark.parametrize("gpu,node", [("l40s", "grom"), ("a40", "megazord"), ("any", "grom")])
def test_validation_submission_and_snapshot_agree_even_for_manual_commands(evaluation_service, gpu, node):
    service, cluster, run, suite = evaluation_service
    request = request_for(run, suite, gpu)
    validation = service.validate_evaluation_target(run["id"], "/checkpoint", suite_id=suite["id"], tasks=request.tasks,
        episodes_per_task=1, seeds=[42], resources=request.resources, argv=request.argv)
    assert validation["valid"], validation
    assert validation["resolved_resources"]["node"]["name"] == node
    evaluation = service.create_evaluation(request)
    assert f"#SBATCH --nodelist={node}" in cluster.script
    updated = service.database.get_run(run["id"])
    attempt = next(a for a in updated["attempts"] if a["stage_id"] == evaluation["stage_id"])
    assert attempt["execution_snapshot_json"]["resolved_spec"]["resources"]["node"]["name"] == node


@pytest.mark.parametrize("old_node,expected", [(None, "grom"), ("bishop", None)])
def test_historical_retry_is_constrained_or_blocked_before_submission(evaluation_service, old_node, expected):
    service, cluster, run, suite = evaluation_service
    evaluation = service.create_evaluation(request_for(run, suite))
    updated = service.database.get_run(run["id"])
    stage = next(s for s in updated["stages"] if s["id"] == evaluation["stage_id"])
    config = copy.deepcopy(stage["resolved_config_json"])
    config["spec"]["resources"]["node"] = {"mode": "manual", "name": old_node} if old_node else {"mode": "auto"}
    # Also exercise old manual plans which omitted canonical_evaluation.
    config["plan"]["native_config"].pop("canonical_evaluation", None)
    for attempt in updated["attempts"]:
        if attempt["stage_id"] == stage["id"]:
            service.database.update_job_attempt(attempt["id"], status="FAILED", slurm_state="FAILED")
    service.database.update_stage(stage["id"], status="RETRY_PENDING", resolved_config_json=config)
    before = cluster.submit_count
    result = service._submit_stage(run["id"], stage["id"], "sky2")
    if expected:
        assert cluster.submit_count == before + 1, result
        assert f"#SBATCH --nodelist={expected}" in cluster.script
        latest = service.database.get_run(run["id"])["attempts"][-1]
        assert latest["execution_snapshot_json"]["resolved_spec"]["resources"]["node"]["name"] == expected
    else:
        assert result["status"] == "BLOCKED"
        assert "not allowed" in result["blockers"][0]
        assert cluster.submit_count == before


def test_readiness_checks_use_same_policy_and_reject_other_nodes():
    profile = "groot-isaacsim-5.0.0_isaaclab-2.2.0_py311"
    suite = "groot_gr1_isaaclab_evaltasks"
    assert "#SBATCH --nodelist=megazord" in render_readiness_sbatch(profile, suite, gpu_type="a40")
    with pytest.raises(ValueError, match="not allowed"):
        render_readiness_sbatch(profile, suite, node="bishop")


@pytest.mark.parametrize("slug,suite_name,model", [
    ("egoverse-hpt", "dexverse_training_episode", "hpt_joints"),
    ("egoverse-act", "dexverse_training_episode", "act"),
    ("xpolicylab-dp", "dexverse_training_episode", None),
    ("xpolicylab-act", "dexverse_training_episode", None),
    ("groot", "groot_gr1_isaaclab_evaltasks", None),
])
@pytest.mark.parametrize("gpu,node", [("l40s", "grom"), ("a40", "megazord")])
def test_native_evaluators_resolve_safe_resources_and_frozen_capsules(evaluation_service, monkeypatch, slug, suite_name, model, gpu, node):
    from test_experiments import make_spec as native_spec

    service, _, run, _ = evaluation_service
    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.setattr("skynet_app.pipeline_api.recorded_episode_sources", lambda *args: [{"path": "/episode.pkl", "sha256": "a" * 64}])
    manifest = next(m for m in builtin_adapter_manifests() if m.slug == slug)
    config = {f.path.removeprefix("native.config."): copy.deepcopy(f.default)
              for f in manifest.train.input_fields if f.path.startswith("native.config.") and f.default is not None}
    config.update(dataset_path="/prepared", dataset_manifest_sha256="a" * 64, embodiment_tag="GR1")
    if model:
        config["model_preset"] = model
    spec = native_spec(source={"repository": manifest.default_repository, "revision": "a" * 40,
                              "adapter": slug, "adapter_manifest": canonical_adapter_manifest(manifest)}, native={"config": config})
    run = {**run, "resolved_spec_json": spec.model_dump(mode="json", by_alias=True)}
    suite = next(s for s in service.database.list_evaluation_suites() if s["name"] == suite_name)
    checkpoint = {"id": "checkpoint", "path": "/checkpoint", "sha256": "b" * 64}
    resources = request_for(run, suite, gpu).resources
    parallelism = 8 if slug.startswith("xpolicylab-") else 1
    resolved, plan, context, *_ = service._resolve_evaluation_implementation(
        run, checkpoint, suite, environment=suite["evaluator_adapter"],
        tasks=[suite["config_json"]["tasks"][0]], seeds=list(range(parallelism)), episodes_per_task=1,
        parallelism=parallelism, headless=True, execution_key="native-placement", resources=resources,
    )
    assert plan.runnable, plan.blockers
    assert resolved.resources.node.name == node
    assert resolved.resources.gpu.count == parallelism
    assert context["worker_resources"]["node"]["name"] == node
    assert plan.native_config["canonical_evaluation"]["worker_resources"]["node"]["name"] == node
    compiled = compile_sbatch(resolved, plan, run_id="native-placement", stage="eval")
    assert f"#SBATCH --nodelist={node}" in compiled.script


def test_invalid_manual_node_is_rejected_by_validation(evaluation_service):
    service, cluster, run, suite = evaluation_service
    request = request_for(run, suite, "l40s", "bishop")
    before = cluster.submit_count
    validation = service.validate_evaluation_target(run["id"], "/checkpoint", suite_id=suite["id"],
        tasks=[], episodes_per_task=1, resources=request.resources, argv=request.argv)
    assert not validation["valid"]
    assert "not allowed" in validation["plan_message"]
    assert cluster.submit_count == before


def test_evaluation_does_not_inherit_an_unsafe_training_node(evaluation_service):
    service, _, run, suite = evaluation_service
    run = copy.deepcopy(run)
    run["resolved_spec_json"]["resources"]["node"] = {"mode": "manual", "name": "baymax"}
    resolved, plan, *_ = service._resolve_evaluation_implementation(
        run, {"id": "checkpoint", "path": "/checkpoint", "sha256": "b" * 64}, suite,
        environment="isaac_sim", tasks=[], seeds=[42], episodes_per_task=1,
        parallelism=1, headless=True, execution_key="inherited-node", manual_argv=["python", "eval.py"],
    )
    assert resolved.resources.node.name == "megazord"
    assert run["resolved_spec_json"]["resources"]["node"]["name"] == "baymax"
    assert "#SBATCH --nodelist=megazord" in compile_sbatch(resolved, plan, run_id="inherited-node", stage="eval").script


def test_eight_parallel_gpus_and_manual_any_gpu_stay_on_one_allowed_node():
    spec = make_spec(node={"mode": "manual", "name": "megazord"})
    spec.resources.gpu.gpu_type = "any"
    spec.resources.gpu.count = 8
    plan = resolve_adapter_plan(spec)
    plan.native_config["canonical_evaluation"] = {"environment": "isaac_lab", "parallelism": 8}
    compiled = compile_sbatch(spec, plan, run_id="eight-workers", stage="eval")
    assert "#SBATCH --gres=gpu:a40:8" in compiled.script
    assert "#SBATCH --nodelist=megazord" in compiled.script
