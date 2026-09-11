from concurrent.futures import ThreadPoolExecutor
import threading
import time
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from skynet_app import pipeline_api as pipeline
from skynet_app.adapters.dp_manifest import manifest
from skynet_app.cluster_runtime import ClusterError
from skynet_app.database import Database
from skynet_app.tracking import SessionCredentialStore


@pytest.fixture
def service(tmp_path, monkeypatch):
    credentials = SimpleNamespace(load=lambda _provider: None)
    result = pipeline.PipelineService(Database(tmp_path / "progress.db"), object(),
        credential_store=credentials, session_credentials=SessionCredentialStore())
    monkeypatch.setattr(result, "tracking_connections", lambda: {"connections": {}})
    monkeypatch.setattr(result, "run_tracking_actions", lambda *_args: {})
    yield result
    result.stop()


def wait_until_idle(service):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with service._progress_refresh_lock:
            if not service._progress_refresh_workers:
                return
        time.sleep(.005)
    raise AssertionError("Progress worker did not settle")


def create_run(database):
    project = database.create_project("progress")
    experiment = database.create_experiment(project_id=project["id"], name="list", requested_spec={})
    variant = database.create_variant(experiment["latest_revision"]["id"], name="one", parameters={}, resolved_spec={})
    return database.create_run(variant["id"], seed=0, adapter_name="generic", adapter_version="1",
                               run_directory="/fixture/run", status="RUNNING")


def test_lists_respond_while_progress_is_blocked_and_do_not_cache_status(service, monkeypatch):
    run = create_run(service.database)
    evaluation = service.database.create_evaluation(run["id"], evaluator_adapter="generic", evaluator_version="1",
        suite_name="fixture", suite_version="1", tasks=["task"], seeds=[0], episodes_per_task=1, status="RUNNING")
    release = threading.Event()
    started = {kind: threading.Event() for kind in ("training", "evaluation")}
    def blocked(kind):
        def read(_record):
            started[kind].set()
            assert release.wait(3)
        return read
    monkeypatch.setattr(service, "_ingest_training_progress", blocked("training"))
    monkeypatch.setattr(service, "_ingest_evaluation_progress", blocked("evaluation"))
    monkeypatch.setattr(pipeline, "service", service)
    app = FastAPI()
    app.include_router(pipeline.router)
    try:
        with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as requests:
            runs = requests.submit(client.get, "/api/runs").result(timeout=1).json()
            evaluations = requests.submit(client.get, "/api/evaluations").result(timeout=1).json()
            assert runs["runs"][0]["id"] == run["id"]
            assert evaluations["evaluations"][0]["id"] == evaluation["id"]
            assert runs["progress_refresh_pending"] and evaluations["progress_refresh_pending"]
            assert all(event.wait(1) for event in started.values())
            service.database.update_run(run["id"], status="CANCELLED")
            service.database.update_evaluation(evaluation["id"], status="CANCELLED")
            assert client.get("/api/runs").json()["runs"][0]["status"] == "CANCELLED"
            assert client.get("/api/evaluations").json()["evaluations"][0]["status"] == "CANCELLED"
            release.set()
            wait_until_idle(service)
            client.get("/api/runs")  # The changed terminal outcome gets one final enrichment.
            wait_until_idle(service)
            assert client.get("/api/runs").json()["progress_refresh_pending"] is False
            assert client.get("/api/evaluations").json()["progress_refresh_pending"] is False
    finally:
        release.set()
        wait_until_idle(service)


def test_progress_queue_coalesces_requests_and_prioritizes_active_records(service, monkeypatch):
    records = [{"id": str(i), "status": "FAILED"} for i in range(8)]
    by_id = {record["id"]: record for record in records}
    by_id["active"] = {"id": "active", "status": "RUNNING"}
    monkeypatch.setattr(service.database, "get_run", lambda identifier: by_id[identifier])
    release = threading.Event()
    two_started = threading.Event()
    calls = []
    lock = threading.Lock()
    def read(record):
        with lock:
            calls.append(record["id"])
            if len(calls) == 2:
                two_started.set()
        assert release.wait(3)
    monkeypatch.setattr(service, "_ingest_training_progress", read)
    try:
        service._queue_list_progress_refresh("training", records)
        assert two_started.wait(1)
        for _ in range(3):
            service._queue_list_progress_refresh("training", records + [by_id["active"]])
        assert len(calls) == 2, "At most two remote reads may run while the gate is closed"
        release.set()
        wait_until_idle(service)
        assert calls[2] == "active", "New active work precedes the queued historical enrichment"
        assert len(calls) == len(set(calls)) == 9, "Repeated list reads must not duplicate queued or in-flight work"
        by_id["0"]["status"] = "RUNNING"
        assert service._queue_list_progress_refresh("training", [by_id["0"]])
        wait_until_idle(service)
        assert calls[-1] == "0" and len(calls) == 10, "A resumed record bypasses its historical refresh delay"
    finally:
        release.set()
        wait_until_idle(service)


def test_missing_terminal_progress_has_bounded_retry_without_hiding_new_attempts(service, monkeypatch):
    clock = [500.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pipeline, "_training_progress_contract", lambda _run: (manifest().train.progress, "fixture"))
    calls = []
    fail = [True]
    def read(*_args, **_kwargs):
        calls.append(clock[0])
        if fail[0]:
            raise ClusterError("Historic log unavailable")
        return "fixture", ""
    service.cluster = SimpleNamespace(read_log=read)
    run = {"id": "run", "status": "FAILED", "run_directory": "/fixture/run", "resolved_spec_json": {},
        "stages": [{"id": "stage", "stage_type": "TRAIN"}],
        "attempts": [{"id": "attempt", "stage_id": "stage", "status": "FAILED", "slurm_job_id": "123",
                      "restart_count": 0, "finished_at": "first"}]}
    service._ingest_training_progress(run)
    service._ingest_training_progress(run)
    assert len(calls) == 1
    clock[0] += 61
    fail[0] = False
    service._ingest_training_progress(run)
    service._ingest_training_progress(run)
    assert len(calls) == 2, "A late log may recover, then a successful final read is retained"
    run["attempts"][0].update(restart_count=1, finished_at="second")
    service._ingest_training_progress(run)
    assert len(calls) == 3, "A newer attempt outcome bypasses the previous outcome’s final-read cache"


@pytest.mark.parametrize("cancel_during", ["read", "upsert"])
def test_late_evaluation_progress_cannot_reopen_a_cancelled_episode(service, monkeypatch, cancel_during):
    run = create_run(service.database)
    stage = service.database.create_stage(run["id"], stage_type="EVAL", name="evaluate")
    service.database.create_job_attempt(stage["id"], status="RUNNING", gateway="fixture", slurm_job_id="123")
    evaluation = service.database.create_evaluation(run["id"], stage_id=stage["id"], evaluator_adapter="generic", evaluator_version="1",
        suite_name="fixture", suite_version="1", tasks=["task"], seeds=[0], episodes_per_task=1, status="RUNNING", result_path="/fixture/result.json")
    service.database.upsert_evaluation_episode(evaluation["id"], task="task", seed=0, episode_index=0, status="PENDING")
    if cancel_during == "upsert":
        ordinary_upsert = service.database.upsert_evaluation_episode
        def cancel_before_upsert(*args, **kwargs):
            service.database.update_evaluation(evaluation["id"], status="CANCELLED")
            return ordinary_upsert(*args, **kwargs)
        monkeypatch.setattr(service.database, "upsert_evaluation_episode", cancel_before_upsert)
    def read(*_args, **_kwargs):
        if cancel_during == "read":
            service.database.update_evaluation(evaluation["id"], status="CANCELLED")
        return "fixture", json.dumps({"kind": "episode_observed", "total": 1,
            "episode": {"task": "task", "seed": 0, "episode_index": 0, "status": "RUNNING", "metrics": {}}})
    service.cluster = SimpleNamespace(read_file=read)
    service._ingest_evaluation_progress(evaluation)
    with service.database.connection() as connection:
        state = connection.execute("SELECT status FROM evaluation_episodes WHERE evaluation_id = ?", (evaluation["id"],)).fetchone()["status"]
    assert state == "CANCELLED", "A remote read finishing after cancellation must not rewrite a terminal episode"


def test_stop_discards_queued_progress_reads(service, monkeypatch):
    records = [{"id": str(i), "status": "RUNNING"} for i in range(10)]
    monkeypatch.setattr(service.database, "get_run", lambda identifier: {"id": identifier})
    release = threading.Event()
    two_started = threading.Event()
    calls = []
    lock = threading.Lock()
    def read(record):
        with lock:
            calls.append(record["id"])
            if len(calls) == 2:
                two_started.set()
        assert release.wait(3)
    monkeypatch.setattr(service, "_ingest_training_progress", read)
    try:
        service._queue_list_progress_refresh("training", records)
        assert two_started.wait(1)
        service.stop()
        release.set()
        wait_until_idle(service)
        assert len(calls) == 2, "Shutdown must not start the remaining queued remote reads"
        assert not service._progress_refresh_pending
        assert service._queue_list_progress_refresh("training", records) is False
    finally:
        release.set()
        wait_until_idle(service)
