"""Scheduler restart metadata must not relabel an append-only log's old rows."""

import json
import threading
from types import SimpleNamespace

import pytest

import skynet_app.pipeline_api as pipeline
from skynet_app.database import Database


@pytest.fixture
def progress(tmp_path):
    database = Database(tmp_path / "restart.db")
    spec = {"native": {"config": {"epochs": 10}}}
    project = database.create_project("restart")
    experiment = database.create_experiment(project_id=project["id"], name="progress", requested_spec=spec)
    variant = database.create_variant(experiment["latest_revision"]["id"], name="one", parameters={}, resolved_spec=spec)
    run = database.create_run(variant["id"], seed=0, adapter_name="xpolicylab-dp", adapter_version="1",
                              run_directory="/synthetic/run", status="RUNNING")
    stage = database.create_stage(run["id"], stage_type="TRAIN", name="train", status="RUNNING")
    attempt = database.create_job_attempt(stage["id"], status="RUNNING", gateway="synthetic",
                                          slurm_job_id="123", started_at="2026-01-01T00:00:00Z")
    log = {"text": ""}
    service = pipeline.PipelineService.__new__(pipeline.PipelineService)
    service.database = database
    service.cluster = SimpleNamespace(read_log=lambda *_args, **_kwargs: ("synthetic", log["text"]))
    return SimpleNamespace(database=database, service=service, attempt=dict(attempt), run_id=run["id"], log=log)


def stored_attempt(progress):
    return progress.database.get_run(progress.run_id)["attempts"][0]


def epoch(number, loss=None):
    return json.dumps({"epoch": number, "val_loss": 1 / (number + 1) if loss is None else loss}) + "\n"


def ingest(progress):
    progress.service._training_progress_last_reads = {}
    return progress.service._ingest_training_progress(progress.database.get_run(progress.run_id))


def test_restart_sync_updates_metadata_once_without_changing_workflow_state(progress):
    original = stored_attempt(progress)
    status = {"123": {"Restarts": "3", "Start": "2026-01-02T00:00:00Z", "State": "PENDING"}}
    progress.service._sync_attempt_restart_counts([progress.attempt], status)
    updated = stored_attempt(progress)
    assert updated["restart_count"] == progress.attempt["restart_count"] == 3
    assert updated["started_at"] == progress.attempt["started_at"] == "2026-01-02T00:00:00Z"
    assert updated["status"] == original["status"] == "RUNNING"
    assert updated["id"] == original["id"] and updated["slurm_job_id"] == original["slurm_job_id"]
    progress.service._sync_attempt_restart_counts([progress.attempt], status)
    assert stored_attempt(progress) == updated
    progress.service._sync_attempt_restart_counts([progress.attempt], {"123": {"Restarts": "2", "Start": "2027-01-01T00:00:00Z"}})
    assert stored_attempt(progress) == updated


@pytest.mark.parametrize("value", [None, "", "Unknown", "N/A", -1, "1.5", True, float("nan"), float("inf")])
def test_unknown_or_invalid_restart_metadata_does_not_clear_existing_observation(progress, value):
    progress.database.update_job_attempt(progress.attempt["id"], restart_count=2)
    progress.attempt = stored_attempt(progress)
    original = dict(progress.attempt)
    progress.service._sync_attempt_restart_counts([progress.attempt], {"123": {"Restarts": value, "Start": "2027-01-01T00:00:00Z"}})
    assert progress.attempt == original
    assert stored_attempt(progress) == original


def test_missing_scheduler_record_preserves_attempt(progress):
    before = stored_attempt(progress)
    progress.service._sync_attempt_restart_counts([progress.attempt], {})
    assert stored_attempt(progress) == before


def test_restart_count_can_advance_without_a_known_start_time(progress):
    previous = stored_attempt(progress)["started_at"]
    progress.service._sync_attempt_restart_counts([progress.attempt], {"123": {"Restarts": 1, "Start": "Unknown"}})
    assert stored_attempt(progress)["restart_count"] == 1
    assert stored_attempt(progress)["started_at"] == previous


def test_append_only_monotonic_log_retains_old_sample_identity_after_same_job_restart(progress):
    progress.log["text"] = epoch(0) + epoch(1)
    assert ingest(progress) == 2
    old = progress.database.list_training_progress_samples(progress.run_id)
    progress.service._sync_attempt_restart_counts([progress.attempt], {"123": {"Restarts": 1, "Start": "2026-01-02T00:00:00Z"}})
    progress.log["text"] += epoch(2)
    assert ingest(progress) == 1
    samples = progress.database.list_training_progress_samples(progress.run_id)
    assert len(samples) == 3
    assert {sample["id"]: sample for sample in samples if sample["restart_count"] == 0} == {sample["id"]: sample for sample in old}
    assert [(sample["completed"], sample["restart_count"]) for sample in samples if sample["restart_count"] == 1] == [(3, 1)]
    assert ingest(progress) == 0
    assert progress.database.list_training_progress_samples(progress.run_id) == samples


def test_real_counter_reset_still_assigns_the_new_segment_to_the_restart(progress):
    progress.log["text"] = epoch(0) + epoch(1)
    assert ingest(progress) == 2
    old = progress.database.list_training_progress_samples(progress.run_id)
    progress.service._sync_attempt_restart_counts([progress.attempt], {"123": {"Restarts": 1}})
    progress.log["text"] += epoch(0, 0.125)
    assert ingest(progress) == 1
    samples = progress.database.list_training_progress_samples(progress.run_id)
    assert {sample["id"] for sample in old}.issubset({sample["id"] for sample in samples})
    restarted = [sample for sample in samples if sample["restart_count"] == 1]
    assert len(restarted) == 1 and restarted[0]["completed"] == 1
    assert restarted[0]["evidence_json"]["metrics"]["val_loss"] == 0.125


def test_trimmed_tail_after_observed_reset_keeps_equal_valued_new_rows_in_current_restart(progress):
    progress.log["text"] = epoch(0) + epoch(1)
    assert ingest(progress) == 2
    progress.service._sync_attempt_restart_counts([progress.attempt], {"123": {"Restarts": 1}})
    progress.log["text"] += epoch(0, 0.125)
    assert ingest(progress) == 1
    # The bounded tail now omits the reset and first new row. Its next loss can
    # legitimately equal an old run's loss; current-segment evidence still owns it.
    progress.log["text"] = epoch(1) + epoch(2)
    assert ingest(progress) == 2
    restarted = [sample for sample in progress.database.list_training_progress_samples(progress.run_id)
                 if sample["restart_count"] == 1]
    assert sorted(sample["completed"] for sample in restarted) == [1, 2, 3]


def test_reconcile_syncs_restart_before_ingestion_and_preserves_normal_status_transition(progress, monkeypatch):
    service = progress.service
    service._reconcile_lock = threading.Lock()
    service._flush_tracking_provider = lambda *_args, **_kwargs: None
    service._repair_missing_attempt_log_paths = lambda: None
    service.reconcile_data_imports = lambda: None
    service._repair_missing_active_tracking_bindings = lambda _rows: 0
    service._publish_training_progress_tracking = lambda _run_id: 0
    service._dispatch_active_experiments = lambda: None
    service._dispatch_experiment = lambda _identifier: None
    service._refresh_experiment_status = lambda _identifier: None
    monkeypatch.setattr(service.database, "repair_workflow_state_invariants", lambda: {"repaired": 0, "draft_graphs_repaired": 0, "experiment_ids": []})
    monkeypatch.setattr(pipeline, "sync_gpu_statistics", lambda *_args: None)
    service.cluster.job_statuses = lambda _ids: ("synthetic", {"123": {"State": "RUNNING", "Restarts": "2", "Start": "2026-01-02T00:00:00Z"}})
    observed = []
    original_ingest = service._ingest_training_progress

    def inspect_ingestion(run):
        observed.append((run["attempts"][0]["restart_count"], run["attempts"][0]["started_at"]))
        return original_ingest(run)

    monkeypatch.setattr(service, "_ingest_training_progress", inspect_ingestion)
    progress.log["text"] = epoch(0)
    result = service.reconcile()
    assert result["checked"] == result["updated"] == 1
    assert observed == [(2, "2026-01-02T00:00:00Z")]
    assert stored_attempt(progress)["restart_count"] == 2
    assert stored_attempt(progress)["status"] == "RUNNING"
    assert progress.database.get_run(progress.run_id)["status"] == "RUNNING"
    assert progress.database.list_training_progress_samples(progress.run_id)[0]["restart_count"] == 2
