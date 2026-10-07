"""The run list fetches only the progress samples its summary can observe."""
from datetime import datetime, timedelta, timezone

from skynet_app.db_backend import PostgresConnection
from skynet_app.pipeline_api import training_progress_summary
from test_tracking_reconciliation import submitted

START = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)


def stamp(minutes, spelling="micro"):
    moment = START + timedelta(minutes=minutes)
    if spelling == "micro":
        return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if spelling == "milli":
        return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def test_reduced_samples_give_the_same_summary_as_the_full_evidence(tmp_path, monkeypatch):
    db, cluster, service, run_id = submitted(tmp_path, monkeypatch)
    run = db.get_run(run_id)
    first = run["attempts"][0]["id"]
    stage = run["stages"][0]
    # (a) one-shot backfill: 400 steps at one timestamp; (b) a back-dated monotone tail on restart 1.
    db.record_training_progress_samples(run_id, first, [
        {"restart_count": 0, "completed": step, "total": 1000, "source_kind": "jsonl", "recorded_at": stamp(5)}
        for step in range(1, 401)])
    db.record_training_progress_samples(run_id, first, [
        {"restart_count": 1, "completed": 400 + step, "total": 1000, "source_kind": "jsonl", "recorded_at": stamp(10 + step, "milli")}
        for step in range(1, 401)])
    # (c) a second attempt with a non-monotone segment, in three timestamp spellings.
    second = db.create_job_attempt(stage["id"], status="RUNNING")
    db.update_job_attempt(second["id"], started_at=stamp(60))
    for minutes, completed, spelling in ((61, 10, "micro"), (62, 30, "milli"), (63, 5, "second"), (64, 40, "micro")):
        db.record_training_progress_sample(run_id, second["id"], restart_count=0, completed=completed, total=1000,
                                           source_kind="jsonl", recorded_at=stamp(minutes, spelling))
    statements = []
    original = PostgresConnection.execute
    def recorded(self, statement, parameters=None):
        statements.append(statement)
        return original(self, statement, parameters)
    monkeypatch.setattr(PostgresConnection, "execute", recorded)
    evidence = db.run_progress_evidence([run_id])[run_id]
    monkeypatch.setattr(PostgresConnection, "execute", original)
    assert sum("training_progress_samples" in statement for statement in statements) == 1
    reduced = evidence["progress_samples"]
    by_segment = {}
    for sample in reduced:
        by_segment.setdefault((sample["attempt_id"], sample["restart_count"]), []).append(sample["completed"])
    assert by_segment == {(first, 0): [400], (first, 1): [401, 800], (second["id"], 0): [10, 5, 40]}
    full = db.list_training_progress_samples(run_id)
    assert len(full) == 804, "Run detail keeps every sample"
    attempts = db.get_run(run_id)["attempts"]
    spec = evidence["resolved_spec_json"]
    for status in ("RUNNING", "SUCCEEDED", "FAILED"):
        for checkpoints in ([], [{"id": "cp", "training_step": 350, "produced_by_attempt_id": first, "created_at": stamp(4)}]):
            summaries = [training_progress_summary({**run, "status": status}, attempts=attempts, checkpoints=checkpoints, metrics=[],
                                                   progress_samples=samples, resolved_spec=spec, now=stamp(70))
                         for samples in (full, reduced)]
            assert summaries[0] == summaries[1], (status, bool(checkpoints))
