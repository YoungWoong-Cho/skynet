"""Episode rows follow their evaluation's lifecycle in the writing transaction."""

import pytest

from skynet_app import database
from skynet_app.database import Database, canonical_json, utc_now

from factories import make_run_chain

RECORDED = ["SUCCEEDED", "FAILED", "TIMEOUT"]
UNSETTLED = [*RECORDED, "RUNNING", "PENDING", "PENDING"]
NOT_COMPLETED = [*RECORDED, "NOT_COMPLETED", "NOT_COMPLETED", "NOT_COMPLETED"]


def _created(db, name="episodes", *, stage_status="RUNNING", evaluation_status="RUNNING"):
    """Return a six-episode evaluation and its stage, created with its ledger."""
    run = make_run_chain(db, project_name=name, experiment_name="episodes", variant_name="v", seed=0,
                         run_directory="/tmp/episodes").run
    stage = db.create_stage(run["id"], stage_type="EVALUATE", name="evaluate", status=stage_status)
    evaluation = db.create_evaluation(
        run["id"], evaluator_adapter="generic", evaluator_version="1", suite_name="test", suite_version="1",
        tasks=["task"], seeds=[0], episodes_per_task=6, stage_id=stage["id"], status=evaluation_status,
    )
    db.initialize_evaluation_episodes(evaluation["id"])
    return stage, evaluation


def _evaluation(db, name="episodes", **statuses):
    """Return an evaluation with recorded outcomes, one RUNNING and two PENDING episodes."""
    stage, evaluation = _created(db, name, **statuses)
    for index, status in enumerate([*RECORDED, "RUNNING"]):
        db.upsert_evaluation_episode(
            evaluation["id"], task="task", seed=0, episode_index=index, status=status,
            success=status == "SUCCEEDED", failure_reason="evaluator" if status in {"FAILED", "TIMEOUT"} else None,
            started_at="2026-09-23T00:00:00.000Z",
            completed_at=None if status == "RUNNING" else "2026-09-23T01:00:00.000Z",
        )
    return stage, evaluation


def _stored(db, evaluation_id):
    with db.connection() as connection:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM evaluation_episodes WHERE evaluation_id = ? ORDER BY episode_index", (evaluation_id,),
        ).fetchall()]


def _statuses(db, evaluation_id):
    return [row["status"] for row in _stored(db, evaluation_id)]


def _transition(db, stage, evaluation, status, **kwargs):
    return db.transition_workflow_state(
        stage_id=stage["id"], stage_updates={"status": status},
        evaluation_id=evaluation["id"], evaluation_updates={"status": status}, **kwargs,
    )


def test_update_evaluation_finalizes_only_unfinished_episodes(tmp_path):
    db = Database(tmp_path / "update.db")
    _, evaluation = _evaluation(db)
    recorded = _stored(db, evaluation["id"])[:3]
    db.update_evaluation(evaluation["id"], status="FAILED")
    episodes = _stored(db, evaluation["id"])
    assert [row["status"] for row in episodes] == NOT_COMPLETED
    assert episodes[:3] == recorded
    for episode in episodes[3:]:
        assert episode["failure_reason"] == "Evaluation ended (failed) before this episode completed."
        assert episode["completed_at"] == episode["updated_at"]
    assert [e["status"] for e in db.get_evaluation(evaluation["id"])["episodes"]] == _statuses(db, evaluation["id"])


@pytest.mark.parametrize("status,unfinished", [
    ("CANCELLED", "CANCELLED"), ("FAILED", "NOT_COMPLETED"),
    ("SUCCEEDED", "NOT_COMPLETED"), ("BLOCKED", "NOT_COMPLETED"),
])
def test_lifecycle_transition_finalizes_episodes_with_the_evaluation(tmp_path, status, unfinished):
    db = Database(tmp_path / "transition.db")
    stage, evaluation = _evaluation(db)
    recorded = _stored(db, evaluation["id"])[:3]
    _transition(db, stage, evaluation, status)
    episodes = _stored(db, evaluation["id"])
    assert [row["status"] for row in episodes] == [*RECORDED, unfinished, unfinished, unfinished]
    assert episodes[:3] == recorded
    assert {row["failure_reason"] for row in episodes[3:]} == {
        f"Evaluation ended ({status.lower()}) before this episode completed."
    }


def test_rejected_transition_and_cancellation_request_keep_unfinished_episodes(tmp_path):
    db = Database(tmp_path / "rejected.db")
    stage, evaluation = _evaluation(db)
    before = _stored(db, evaluation["id"])
    assert _transition(db, stage, evaluation, "FAILED", expected_stage_statuses=("SUBMITTING",))["applied"] is False
    # A running job keeps executing its episodes until cancellation completes.
    _transition(db, stage, evaluation, "CANCELLING")
    assert _stored(db, evaluation["id"]) == before


def test_ledger_of_an_evaluation_created_blocked_is_settled(tmp_path):
    db = Database(tmp_path / "blocked.db")
    _, evaluation = _created(db, stage_status="BLOCKED", evaluation_status="BLOCKED")
    assert {(row["status"], row["failure_reason"]) for row in _stored(db, evaluation["id"])} == {
        ("NOT_COMPLETED", "Evaluation ended (blocked) before this episode completed.")
    }


def test_cancellation_before_submission_cancels_unfinished_episodes(tmp_path):
    db = Database(tmp_path / "claim.db")
    stage, evaluation = _evaluation(db, stage_status="PENDING", evaluation_status="PENDING")
    claim = db.claim_workflow_cancellation(stage["id"], entity_type="evaluation", entity_id=evaluation["id"])
    assert claim["status"] == "CANCELLED"
    assert _statuses(db, evaluation["id"]) == [*RECORDED, "CANCELLED", "CANCELLED", "CANCELLED"]


@pytest.mark.parametrize("status", ["RETRY_PENDING", "SUBMITTING", "SUBMITTED", "PENDING", "RUNNING"])
def test_execution_reopens_only_episodes_the_evaluation_ended(tmp_path, status):
    db = Database(tmp_path / "reopen.db")
    stage, evaluation = _evaluation(db)
    _transition(db, stage, evaluation, "CANCELLED")
    recorded = _stored(db, evaluation["id"])[:3]
    _transition(db, stage, evaluation, status)
    episodes = _stored(db, evaluation["id"])
    assert [row["status"] for row in episodes] == [*RECORDED, "PENDING", "PENDING", "PENDING"]
    assert episodes[:3] == recorded
    assert all(row["failure_reason"] is None and row["completed_at"] is None for row in episodes[3:])


def test_workflow_repair_finalizes_episodes_of_the_evaluation_it_ends(tmp_path):
    db = Database(tmp_path / "repair.db")
    stage, evaluation = _evaluation(db)
    db.create_job_attempt(stage["id"], status="FAILED", slurm_job_id="4242")
    result = db.repair_workflow_state_invariants()
    assert result["repaired"] == 1
    assert result["evaluation_episodes_settled"] == 3
    assert db.get_evaluation(evaluation["id"])["status"] == "FAILED"
    assert _statuses(db, evaluation["id"]) == NOT_COMPLETED


def _write_status_without_episodes(db, evaluation, status):
    # Builds before same-transaction settlement ended evaluations like this.
    with db.transaction() as connection:
        connection.execute("UPDATE evaluations SET status = ? WHERE id = ?", (status, evaluation["id"]))


def test_workflow_repair_settles_status_written_without_episodes(tmp_path):
    system = Database(tmp_path / "stale.db")
    with system.transaction() as connection:
        connection.execute("INSERT INTO workspaces(id,email) VALUES ('alice','alice@example.com'),('bob','bob@example.com')")
    _, cancelled = _evaluation(system.for_workspace("alice"))
    _, failed = _evaluation(system.for_workspace("bob"))
    _, reopened = _evaluation(system)
    _write_status_without_episodes(system, cancelled, "CANCELLED")
    _write_status_without_episodes(system, failed, "FAILED")
    with system.transaction() as connection:
        connection.execute(
            "UPDATE evaluation_episodes SET status = 'NOT_COMPLETED' WHERE evaluation_id = ? AND status = 'PENDING'",
            (reopened["id"],),
        )
    assert [e["status"] for e in system.get_evaluation(cancelled["id"])["episodes"]] == UNSETTLED

    # A workspace repair settles its own rows and passes over other workspaces.
    assert system.for_workspace("alice").repair_workflow_state_invariants()["evaluation_episodes_settled"] == 3
    assert _statuses(system, failed["id"]) == UNSETTLED
    assert system.repair_workflow_state_invariants()["evaluation_episodes_settled"] == 5
    assert _statuses(system, cancelled["id"]) == [*RECORDED, "CANCELLED", "CANCELLED", "CANCELLED"]
    assert _statuses(system, failed["id"]) == NOT_COMPLETED
    assert _statuses(system, reopened["id"]) == UNSETTLED
    assert system.repair_workflow_state_invariants()["evaluation_episodes_settled"] == 0


def test_episode_catch_up_settles_a_bounded_batch_per_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "EPISODE_CATCH_UP_EVALUATIONS", 2)
    db = Database(tmp_path / "batch.db")
    stale = [_evaluation(db, f"stale-{index}")[1] for index in range(3)]
    for evaluation in stale:
        _write_status_without_episodes(db, evaluation, "FAILED")
    assert [db.repair_workflow_state_invariants()["evaluation_episodes_settled"] for _ in range(3)] == [6, 3, 0]
    assert [_statuses(db, evaluation["id"]) for evaluation in stale] == [NOT_COMPLETED] * 3


def test_pending_deletion_defers_only_the_episode_catch_up(tmp_path):
    db = Database(tmp_path / "deleting.db")
    _, deleting = _evaluation(db, "deleting")
    stage, ending = _evaluation(db, "ending")
    interrupted_stage, interrupted = _evaluation(db, "interrupted")
    db.create_job_attempt(interrupted_stage["id"], status="FAILED", slurm_job_id="4242")
    _write_status_without_episodes(db, deleting, "CANCELLED")
    with db.transaction() as connection:
        connection.execute(
            "INSERT INTO maintenance_operations(target_kind,target_id,owner_id,plan_json,created_at) VALUES ('evaluation',?,'legacy',?,?)",
            (deleting["id"], canonical_json({"kind": "evaluation", "id": deleting["id"], "records": {
                "evaluations": [deleting["id"]],
                "evaluation_episodes": [row["id"] for row in _stored(db, deleting["id"])],
            }, "files": []}), utc_now()),
        )
    _transition(db, stage, ending, "FAILED")
    assert _statuses(db, ending["id"]) == NOT_COMPLETED
    # Workflow repair settles the evaluation it ends; only the catch-up waits.
    result = db.repair_workflow_state_invariants()
    assert (result["repaired"], result["evaluation_episodes_settled"]) == (1, 3)
    assert _statuses(db, interrupted["id"]) == NOT_COMPLETED
    assert _statuses(db, deleting["id"]) == UNSETTLED
    with db.transaction() as connection:
        connection.execute("DELETE FROM maintenance_operations")
    assert db.repair_workflow_state_invariants()["evaluation_episodes_settled"] == 3
    assert _statuses(db, deleting["id"]) == [*RECORDED, "CANCELLED", "CANCELLED", "CANCELLED"]
