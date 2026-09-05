from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from types import SimpleNamespace
import threading

import pytest

from skynet_app.cluster_runtime import ClusterError
from skynet_app.database import Database
from skynet_app.capture_processing.tensor_trace import (
    STEP_IDS,
    STEP_WIDTHS,
    TensorTraceService,
    TraceError,
    validate_trace,
)


def fixture():
    artifacts = {
        "state-bc.pt": {"sha256": "a" * 64, "size_bytes": 100},
        "dataset.hdf5": {"sha256": "b" * 64, "size_bytes": 200},
    }
    job = {
        "id": "cycle",
        "state": "SUCCEEDED",
        "root": "/runs/cycle",
        "gateway": "saved-host",
        "config": {"pipeline": {"runtime": "/saved-runtime"}},
        "result": {
            "dataset": {"training_policy": "skynet-state-bc/v1", "frames": 3},
            "artifacts": artifacts,
        },
    }
    trace = {
        "schema": "skynet.tensor-trace/v1",
        "policy": "skynet-state-bc/v1",
        "device": "cpu",
        "frame": 0,
        "frame_count": 3,
        "checkpoint_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "source_index": 0,
        "provenance": "Test fixture",
        "parameter_count": 24732,
        "action_mae": 0,
        "changed_action_indices": [],
        "demonstrated_action": [0.0] * 28,
        "steps": [
            {
                "id": identifier,
                "shape": [1, width],
                "dtype": "float32",
                "values": [0.0] * width,
                "title": identifier,
                "explanation": "fixture",
                "formula": "fixture",
                "parameters": [],
                "stats": {"min": 0, "max": 0, "mean": 0},
            }
            for identifier, width in zip(STEP_IDS, STEP_WIDTHS, strict=True)
        ],
    }
    return job, trace


def service(tmp_path):
    job, trace = fixture()
    calls = []

    def ssh(host, command, **kwargs):
        calls.append((host, command, kwargs))
        return json.dumps(trace)

    processing = SimpleNamespace(
        database=Database(tmp_path / "trace.db"),
        get=lambda _: deepcopy(job),
        cluster=SimpleNamespace(ssh=ssh),
    )
    return TensorTraceService(processing), job, trace, calls


def test_real_shape_contract_rejects_wrong_frame_hash_nonfinite_and_partial_steps():
    job, trace = fixture()
    validate_trace(trace, 0, job["result"]["artifacts"], 3)
    for mutate in (
        lambda t: t.update(frame=2),
        lambda t: t.update(dataset_sha256="wrong"),
        lambda t: t["steps"].pop(),
        lambda t: t["steps"][0]["values"].__setitem__(0, float("nan")),
        lambda t: t["steps"][0].update(shape=[35]),
        lambda t: t["steps"][0]["values"].pop(),
        lambda t: t.update(changed_action_indices=[28]),
    ):
        bad = deepcopy(trace)
        mutate(bad)
        with pytest.raises(TraceError, match="incomplete or incompatible"):
            validate_trace(bad, 0, job["result"]["artifacts"], 3)


def test_persistent_cache_avoids_cluster_requests_and_is_bound_to_artifacts(tmp_path):
    traces, job, data, calls = service(tmp_path)
    assert traces.get("cycle") == data
    assert traces.get("cycle") == data
    reopened = TensorTraceService(traces.processing)
    assert reopened.get("cycle") == data
    assert len(calls) == 1
    assert calls[0][0] == "saved-host"
    assert "/saved-runtime/bin/python" in calls[0][1]
    assert "weights_only=True" in calls[0][2]["stdin"]
    job["result"]["artifacts"]["state-bc.pt"]["sha256"] = "c" * 64
    data["checkpoint_sha256"] = "c" * 64
    assert traces.get("cycle") == data
    assert len(calls) == 2


@pytest.mark.parametrize("frame", [-1, 3, 0.5, True])
def test_out_of_range_is_rejected_before_remote_work(tmp_path, frame):
    traces, _, _, calls = service(tmp_path)
    with pytest.raises(TraceError, match="whole|Choose"):
        traces.get("cycle", frame)
    assert not calls


def test_unsupported_incomplete_and_missing_artifacts_are_explicit(tmp_path):
    traces, job, _, calls = service(tmp_path)
    job["state"] = "RUNNING"
    with pytest.raises(TraceError, match="completed cycle") as error:
        traces.get("cycle")
    assert error.value.status == 409
    job["state"] = "SUCCEEDED"
    job["result"]["dataset"]["training_policy"] = "pi0.5"
    with pytest.raises(TraceError, match="Unsupported model"):
        traces.get("cycle")
    job["result"]["dataset"]["training_policy"] = "skynet-state-bc/v1"
    job["result"]["artifacts"].pop("state-bc.pt")
    with pytest.raises(TraceError, match="missing"):
        traces.get("cycle")
    assert not calls


def test_network_failure_is_not_cached_and_retry_succeeds(tmp_path):
    traces, _, data, calls = service(tmp_path)
    original = traces.processing.cluster.ssh

    def fail(*args, **kwargs):
        raise ClusterError("host offline")

    traces.processing.cluster.ssh = fail
    with pytest.raises(ClusterError, match="host offline"):
        traces.get("cycle")
    traces.processing.cluster.ssh = original
    assert traces.get("cycle") == data
    assert len(calls) == 1


def test_duplicate_requests_share_one_trace(tmp_path):
    traces, _, data, calls = service(tmp_path)
    original = traces.processing.cluster.ssh
    entered, release = threading.Event(), threading.Event()

    def slow(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    traces.processing.cluster.ssh = slow
    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(traces.get, "cycle")
        assert entered.wait(3)
        second = executor.submit(traces.get, "cycle")
        release.set()
        assert first.result() == second.result() == data
    assert len(calls) == 1


def test_bad_cached_payload_is_recomputed(tmp_path):
    traces, _, data, calls = service(tmp_path)
    traces.get("cycle")
    with traces.processing.database.transaction() as connection:
        connection.execute("UPDATE tensor_trace_cache SET payload_json='{}'")
    assert traces.get("cycle") == data
    assert len(calls) == 2


def test_route_statuses_and_query_validation(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from skynet_app.capture_processing import api

    traces, job, data, _ = service(tmp_path)
    monkeypatch.setattr(api, "traces", traces)
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        path = "/api/collection/processing/jobs/cycle/tensor-trace"
        assert client.get(path).json() == data
        assert client.get(path + "?frame=-1").status_code == 422
        assert client.get(path + "?frame=0.5").status_code == 422
        assert client.get(path + "?frame=3").status_code == 422
        job["state"] = "FAILED"
        assert client.get(path).status_code == 409
