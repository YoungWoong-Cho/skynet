"""Scheduler restart metadata must not relabel an append-only log's old rows."""

import json
import hashlib
import threading
from types import SimpleNamespace

import pytest

import skynet_app.pipeline_api as pipeline
from skynet_app.adapters import canonical_adapter_manifest
from skynet_app.adapters.act_manifest import manifest as act_manifest
from skynet_app.database import Database
from skynet_app.training_progress_log import read_execution_log
from skynet_app.cluster_runtime import JobStatusSnapshot

from factories import make_run_chain


@pytest.fixture
def progress(tmp_path):
    database = Database(tmp_path / "restart.db")
    spec = {"source": {"adapter_manifest": canonical_adapter_manifest(act_manifest())},
            "native": {"config": {"epochs": 10}}}
    run = make_run_chain(database, project_name="restart", experiment_name="progress", requested_spec=spec, seed=0,
                         adapter_name="xpolicylab-act", run_directory="/synthetic/run", status="RUNNING").run
    stage = database.create_stage(run["id"], stage_type="TRAIN", name="train", status="RUNNING")
    attempt = database.create_job_attempt(stage["id"], status="RUNNING", gateway="synthetic",
                                          slurm_job_id="123", started_at="2026-01-01T00:00:00Z")
    log = {"text": "", "path": tmp_path / "progress.jsonl", "root": tmp_path}
    service = pipeline.PipelineService.__new__(pipeline.PipelineService)
    service.database = database
    def read_log(_path, _gateway, **kwargs):
        log["path"].write_text(log["text"])
        boundary = dict(kwargs.pop("execution_boundary"))
        boundary["boundary_path"] = tmp_path / f'{boundary["job_id"]}-{boundary["restart_count"]}.json'
        try:
            return "synthetic", read_execution_log(log["path"], **boundary, **kwargs)
        except FileNotFoundError as error:
            raise pipeline.ClusterError(str(error)) from error
    service.cluster = SimpleNamespace(read_log=read_log)
    fixture = SimpleNamespace(database=database, service=service, attempt=dict(attempt), run_id=run["id"], log=log)
    original_sync = service._sync_attempt_restart_counts
    def sync(attempts, statuses):
        previous = {row["id"]: row.get("restart_count", 0) for row in attempts}
        original_sync(attempts, statuses)
        for row in attempts:
            if (row.get("restart_count") or 0) > (previous[row["id"]] or 0):
                mark_boundary(fixture, row)
    service._sync_attempt_restart_counts = sync
    return fixture


def mark_boundary(progress, attempt):
    path = progress.log["path"]
    data = progress.log["text"].encode()
    path.write_bytes(data)
    stat = path.stat()
    receipt = {
        "schema_version": "skynet.training-progress-start/v1",
        "job_id": attempt["slurm_job_id"], "restart_count": attempt.get("restart_count") or 0,
        "path": str(path), "start_byte_offset": len(data), "device": stat.st_dev, "inode": stat.st_ino,
        "prefix_bytes": min(64, len(data)), "prefix_sha256": hashlib.sha256(data[-64:]).hexdigest(),
        "at_line_boundary": not data or data.endswith(b"\n"),
    }
    (progress.log["root"] / f'{receipt["job_id"]}-{receipt["restart_count"]}.json').write_text(json.dumps(receipt))


def new_attempt(progress):
    progress.database.update_job_attempt(progress.attempt["id"], status="TIMEOUT")
    current = progress.database.create_job_attempt(progress.attempt["stage_id"], status="RUNNING",
        gateway="synthetic", slurm_job_id="124", started_at="2026-01-02T00:00:00Z")
    mark_boundary(progress, current)
    return current


def stored_attempt(progress):
    return progress.database.get_run(progress.run_id)["attempts"][0]


def epoch(number, loss=None):
    return json.dumps({"epoch": number, "train_loss": 1 / (number + 1) if loss is None else loss}) + "\n"


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
    assert restarted[0]["evidence_json"]["metrics"]["train_loss"] == 0.125


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


def test_new_attempt_does_not_ingest_previous_execution_even_with_old_resets_and_unobserved_rows(progress):
    progress.log["text"] = epoch(0) + epoch(1) + epoch(0) + epoch(1)
    assert ingest(progress) == 2
    # This old row was written after the last poll, so DB dedup cannot identify it.
    progress.log["text"] += epoch(2)
    current = new_attempt(progress)
    assert ingest(progress) == 0
    assert progress.database.list_training_progress_samples(progress.run_id, attempt_id=current["id"]) == []
    # A replay can legitimately produce an identical loss for an identical step.
    progress.log["text"] += epoch(1)
    assert ingest(progress) == 1
    sample, = progress.database.list_training_progress_samples(progress.run_id, attempt_id=current["id"])
    assert sample["completed"] == 2
    assert sample["evidence_json"]["metrics"]["train_loss"] == 0.5
    assert ingest(progress) == 0


def test_resume_without_a_boundary_does_not_claim_shared_history(progress):
    progress.log["text"] = epoch(0) + epoch(1)
    assert ingest(progress) == 2
    current = new_attempt(progress)
    (progress.log["root"] / "124-0.json").unlink()
    assert ingest(progress) == 0
    assert progress.database.list_training_progress_samples(progress.run_id, attempt_id=current["id"]) == []


def test_unchanged_tail_does_not_requery_each_sample_for_metric_enrichment(progress, monkeypatch):
    progress.log["text"] = epoch(0) + epoch(1)
    assert ingest(progress) == 2
    calls = []
    original = progress.database.enrich_training_progress_metrics
    def enrich(identifier, metrics):
        calls.append((identifier, metrics))
        return original(identifier, metrics)
    monkeypatch.setattr(progress.database, "enrich_training_progress_metrics", enrich)
    assert ingest(progress) == 0
    assert calls == []
    progress.log["text"] = epoch(0) + json.dumps({"epoch": 1, "train_loss": 0.5, "val_loss": 0.25}) + "\n"
    assert ingest(progress) == 1
    assert len(calls) == 1 and calls[0][1]["val_loss"] == 0.25
    calls.clear()
    assert ingest(progress) == 0
    assert calls == []


def test_terminal_missing_boundary_is_retried_after_receipt_recovery(progress):
    progress.log["text"] = epoch(0) + epoch(1)
    assert ingest(progress) == 2
    current = new_attempt(progress)
    receipt = progress.log["root"] / "124-0.json"
    recovered = receipt.read_text()
    receipt.unlink()
    progress.database.update_job_attempt(current["id"], status="TIMEOUT")
    assert ingest(progress) == 0
    assert not getattr(progress.service, "_training_progress_final_reads", set())
    receipt.write_text(recovered)
    progress.log["text"] += epoch(1)
    progress.service._training_progress_final_failures = {}
    assert ingest(progress) == 1


def test_reconcile_syncs_restart_before_ingestion_and_preserves_normal_status_transition(progress, monkeypatch):
    service = progress.service
    service._reconcile_lock = threading.Lock()
    service._reconcile_scan_lock = threading.Lock()
    service._stop = threading.Event()
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
    service.cluster.job_status_snapshot = lambda _ids: JobStatusSnapshot(
        "synthetic", {"123": {"State": "RUNNING", "Restarts": "2", "Start": "2026-01-02T00:00:00Z"}})
    observed = []
    original_ingest = service._ingest_training_progress

    def inspect_ingestion(run, **kwargs):
        observed.append((run["attempts"][0]["restart_count"], run["attempts"][0]["started_at"]))
        return original_ingest(run)

    monkeypatch.setattr(service, "_ingest_training_progress", inspect_ingestion)
    result = service.reconcile()
    assert result["checked"] == result["updated"] == 1
    assert observed == []  # Optional ingestion belongs to the independent worker.
    progress.log["text"] = epoch(0)  # Written after this restart's launch boundary.
    service._tracking_reconcile_lock = threading.Lock()
    service._tracking_delivery_lock = threading.Lock()
    service._stop = threading.Event()
    service.reconcile_tracking()
    assert observed == [(2, "2026-01-02T00:00:00Z")]
    assert stored_attempt(progress)["restart_count"] == 2
    assert stored_attempt(progress)["status"] == "RUNNING"
    assert progress.database.get_run(progress.run_id)["status"] == "RUNNING"
    assert progress.database.list_training_progress_samples(progress.run_id)[0]["restart_count"] == 2


def test_ended_execution_without_a_launch_boundary_has_no_final_progress(progress):
    progress.log["text"] = epoch(0) + epoch(1)  # Rows of the earlier execution.
    progress.database.update_job_attempt(progress.attempt["id"], status="TIMEOUT")
    progress.database.create_job_attempt(
        progress.attempt["stage_id"], status="SUCCEEDED", gateway="synthetic", slurm_job_id="124",
        started_at="2026-01-02T00:00:00Z", finished_at="2026-01-02T01:00:00Z")
    progress.database.update_run(progress.run_id, status="SUCCEEDED")
    run = progress.database.get_run(progress.run_id)
    reads = []
    read_log = progress.service.cluster.read_log
    progress.service.cluster.read_log = lambda *args, **kwargs: (reads.append(kwargs), read_log(*args, **kwargs))[1]
    assert progress.service._ingest_training_progress(run, raise_on_error=True) == 0
    assert progress.service._ingest_training_progress(run, raise_on_error=True) == 0
    assert len(reads) == 1 and reads[0]["execution_boundary"]["required"] is True
    assert progress.database.list_training_progress_samples(progress.run_id) == []
