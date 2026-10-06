"""Failed evaluation planning leaves no durable stage or target-data reference."""
from types import SimpleNamespace

import pytest

from skynet_app.database import Database
from skynet_app.pipeline_api import EvaluationRequest, PipelineService
from test_data_version_retirement import migration
from test_evaluation_dataset_lifecycle import dataset, evaluation_run


def evaluation_case(db, target):
    _, created = evaluation_run(db)
    run = db.get_run(created["id"])
    checkpoint = db.create_checkpoint(run["id"], checkpoint_type="INFERENCE", path="/checkpoint", sha256="a" * 64)
    selection = dict(version_id=target["id"], manifest_sha256=target["manifest_sha256"], path=target["path"])
    suite = db.register_evaluation_suite(evaluator_adapter="isaac_lab", evaluator_version="1", name="target-lifecycle",
        suite_version="1", config={"target_dataset": selection, "tasks": ["Dexverse-PickCube-v0"]})
    service = PipelineService.__new__(PipelineService)
    service.database = db
    service._reconcile_lock = db.operation_lock("pipeline")
    service._resolve_evaluation_target = lambda *_: (dict(run_valid=True, checkpoint_valid=True, errors={}), run, checkpoint)
    service._resolve_evaluation_suite_selection = lambda *_: (suite, "isaac_lab", suite["config_json"]["tasks"], {})
    service._submit_stage = lambda *_: pytest.fail("Planning tests must not submit jobs")
    request = EvaluationRequest(run_id=run["id"], checkpoint_path="/checkpoint", suite_id=suite["id"],
        target_dataset_id=target["id"], episodes_per_task=1, seeds=[42], argv=["python", "fixture-evaluator.py"])
    return SimpleNamespace(db=db, run=run, checkpoint=checkpoint, suite=suite, target=selection,
                           service=service, request=request)


def resolve_plan(case, during_planning=None):
    def resolve(*args, execution_key, **kwargs):
        assert case.db.list_stages(case.run["id"]) == [], "A path namespace does not need a saved stage"
        assert case.db.list_evaluations(run_id=case.run["id"]) == []
        case.execution_key = execution_key
        if during_planning:
            during_planning()
        context = dict(target_dataset=case.target, checkpoint=case.checkpoint,
            suite={"name": case.suite["name"]}, result_path=f"/evaluation/{execution_key}/result.json")
        plan_document = dict(argv=["python", "fixture-evaluator.py"], resume_argv=[], blockers=["test runtime unavailable"])
        plan = SimpleNamespace(**plan_document, model_dump=lambda **_: plan_document)
        spec = SimpleNamespace(model_dump=lambda **_: {})
        return spec, plan, context, {}, {}, "registered_adapter"
    return resolve


@pytest.mark.parametrize("error", [ValueError("source collection session unavailable"), OSError("hand manifest read failed")])
def test_profile_or_plan_failure_leaves_no_stage_or_dataset_reference(tmp_path, error):
    db = Database(tmp_path / "planning-failure")
    _, target = dataset(db)
    case = evaluation_case(db, target)
    def fail():
        raise error
    case.service._resolve_evaluation_implementation = resolve_plan(case, fail)
    with pytest.raises(type(error), match=str(error)):
        case.service.create_evaluation(case.request)
    assert db.list_stages(case.run["id"]) == []
    assert db.list_evaluations(run_id=case.run["id"]) == []
    assert db.data_version_usage(target["manifest_sha256"]) == []


def test_completed_plan_is_saved_once_with_its_original_execution_key(tmp_path, monkeypatch):
    db = Database(tmp_path / "planned-stage")
    _, target = dataset(db)
    case = evaluation_case(db, target)
    case.service._resolve_evaluation_implementation = resolve_plan(case)
    create_stage = db.create_stage
    saved = []
    def save(*args, **kwargs):
        assert kwargs["resolved_config"]["context"]["target_dataset"] == case.target
        assert kwargs["resolved_config"]["plan"]["argv"]
        saved.append(kwargs)
        return create_stage(*args, **kwargs)
    monkeypatch.setattr(db, "create_stage", save)
    evaluation = case.service.create_evaluation(case.request)
    stages = db.list_stages(case.run["id"])
    assert len(saved) == len(stages) == 1
    assert evaluation["stage_id"] == stages[0]["id"] == case.execution_key
    assert evaluation["result_path"] == f"/evaluation/{case.execution_key}/result.json"
    assert evaluation["status"] == "BLOCKED"
    assert db.data_version_usage(target["manifest_sha256"])


def test_blocked_evaluation_is_created_with_settled_episodes(tmp_path):
    db = Database(tmp_path / "blocked-ledger")
    _, target = dataset(db)
    case = evaluation_case(db, target)
    case.service._resolve_evaluation_implementation = resolve_plan(case)
    evaluation = case.service.create_evaluation(case.request)
    assert evaluation["status"] == "BLOCKED"
    assert [(episode["status"], episode["failure_reason"]) for episode in evaluation["episodes"]] == [
        ("NOT_COMPLETED", "Evaluation ended (blocked) before this episode completed.")
    ]


def test_target_retired_during_planning_cannot_create_an_evaluation(migration):
    m = migration
    case = evaluation_case(m.db, m.old)
    def retire():
        plan = m.retirement.preview(m.old["id"], m.new["id"])
        assert m.retirement.retire(m.old["id"], m.new["id"], plan["token"])["state"] == "RETIRED"
    case.service._resolve_evaluation_implementation = resolve_plan(case, retire)
    with pytest.raises(ValueError, match="Evaluation target dataset changed"):
        # Exercise the final repository guard independently of the outer API
        # pipeline lock, as if another writer retired the target while planning.
        case.service._create_evaluation(case.request)
    assert m.db.list_stages(case.run["id"]) == []
    assert m.db.list_evaluations(run_id=case.run["id"]) == []
    assert len(m.calls) == 1


def test_target_archived_during_planning_cannot_create_an_evaluation(tmp_path):
    db = Database(tmp_path / "archive-during-plan")
    _, target = dataset(db)
    case = evaluation_case(db, target)
    case.service._resolve_evaluation_implementation = resolve_plan(case, lambda: db.update_dataset(target["id"], archived=True))
    with pytest.raises(ValueError, match="Evaluation target dataset changed"):
        case.service.create_evaluation(case.request)
    assert db.list_stages(case.run["id"]) == []
    assert db.list_evaluations(run_id=case.run["id"]) == []


def test_evaluation_stage_name_survives_missing_history_rows(tmp_path):
    db=Database(tmp_path/"stage-history")
    _, target=dataset(db)
    case=evaluation_case(db,target)
    def resolve(*args, execution_key, **kwargs):
        context=dict(target_dataset=case.target,checkpoint=case.checkpoint,
                     suite={"name":case.suite["name"]},result_path=f"/evaluation/{execution_key}/result.json")
        plan=SimpleNamespace(argv=["python","fixture-evaluator.py"],resume_argv=[],blockers=["test runtime unavailable"])
        plan.model_dump=lambda **_: dict(argv=plan.argv,resume_argv=[],blockers=plan.blockers)
        spec=SimpleNamespace(model_dump=lambda **_: {})
        return spec,plan,context,{}, {},"registered_adapter"
    case.service._resolve_evaluation_implementation=resolve
    # The stale/filtered run view deliberately never includes earlier evaluations.
    assert case.run["evaluations"]==[]
    one=case.service.create_evaluation(case.request)
    two=case.service.create_evaluation(case.request)
    stages=db.list_stages(case.run["id"])
    assert len(stages)==2 and len({stage["name"] for stage in stages})==2
    assert one["stage_id"]!=two["stage_id"]


def test_checkpoint_lost_during_remote_planning_is_rejected(tmp_path):
    db = Database(tmp_path / "checkpoint-during-plan")
    _, target = dataset(db)
    case = evaluation_case(db, target)
    def remove_checkpoint():
        case.service._resolve_evaluation_target = lambda *_: (
            dict(run_valid=True, checkpoint_valid=False, errors={}), case.run, None
        )
    case.service._resolve_evaluation_implementation = resolve_plan(case, remove_checkpoint)
    with pytest.raises(ValueError, match="checkpoint changed while planning"):
        case.service.create_evaluation(case.request)
    assert db.list_stages(case.run["id"]) == []
    assert db.list_evaluations(run_id=case.run["id"]) == []
