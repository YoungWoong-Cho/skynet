import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from skynet_app import gpu_metrics, gpu_tracking
from skynet_app.slurm import compile_sbatch, RUNNER_SOURCE
from skynet_app.adapters import resolve_adapter_plan
from skynet_app.tracking import TrackingRequestError, WandBSettings, WandBBridge
from test_slurm import make_spec
from test_tracking import FakeWandBBridge


CSV = "2, GPU-abc, NVIDIA A40, 0, 14, 5632, 46068, 130, 300, 43, 1740, 7000\n3, GPU-def, NVIDIA A40, 90, N/A, 5634, 46068, [Not Supported], 300, 44, 1725, 7000\n"


def test_gpu_scope_and_optional_sensors():
    assert gpu_metrics.allocated_devices({}) == []
    assert gpu_metrics.allocated_devices({"CUDA_VISIBLE_DEVICES": "-1"}) == []
    assert gpu_metrics.allocated_devices(
        {"SLURM_STEP_GPUS": "2,3", "CUDA_VISIBLE_DEVICES": "0,1"}
    ) == ["0", "1"]
    assert gpu_metrics.allocated_devices(
        {"SLURM_STEP_GPUS": "2,3,6,7", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    ) == ["0", "1", "2", "3"]
    assert gpu_metrics.allocated_devices({"CUDA_VISIBLE_DEVICES": "GPU-abc"}) == [
        "GPU-abc"
    ]
    with patch.object(
        gpu_metrics.subprocess, "run", return_value=SimpleNamespace(stdout=CSV)
    ) as call:
        record = gpu_metrics.sample({"SLURM_JOB_ID": "123", "SLURM_STEP_GPUS": "2,3"})
    assert "--id=2,3" in call.call_args.args[0]
    assert record["job_id"] == "123"
    first, second = record["devices"]
    assert first["metrics"]["gpu"] == 0  # An idle device is a valid zero measurement.
    assert first["metrics"]["memoryAllocatedBytes"] == 5632 * 1024**2
    assert first["metrics"]["memoryAllocated"] == pytest.approx(12.2254, rel=0.001)
    assert "powerWatts" not in second["metrics"]
    assert "memory" not in second["metrics"]
    assert gpu_tracking.system_metrics(record)["gpu.2.gpu"] == 0
    record["node"] = "node-a"
    assert "gpu.node-a.2.gpu" in gpu_tracking.system_metrics(record, multi_node=True)


def test_sampler_stops_and_reader_replays_complete_records(tmp_path):
    record = {"timestamp_ms": 123, "job_id": "1", "devices": []}
    with patch.object(gpu_metrics, "sample", return_value=record):
        sampler = gpu_metrics.GPUSampler(tmp_path, interval=0.01).start()
        deadline = time.monotonic() + 1
        while not list(tmp_path.glob("*.jsonl")) and time.monotonic() < deadline:
            time.sleep(0.01)
        sampler.stop()
    assert not sampler.thread.is_alive()
    first = gpu_metrics.read_records(tmp_path, {})
    assert first["records"]
    assert gpu_metrics.read_records(tmp_path, first["cursors"])["records"] == []
    path = next(tmp_path.glob("*.jsonl"))
    with path.open("ab") as f:
        f.write(b'{"timestamp_ms":456')
    partial = gpu_metrics.read_records(tmp_path, first["cursors"])
    assert partial["cursors"] == first["cursors"]
    with path.open("ab") as f:
        f.write(b"}\n")
    assert gpu_metrics.read_records(tmp_path, partial["cursors"])["records"] == [
        {"timestamp_ms": 456}
    ]


def test_sensor_failures_do_not_crash_training(tmp_path):
    with patch.object(
        gpu_metrics, "sample", side_effect=FileNotFoundError("nvidia-smi")
    ):
        gpu_metrics.GPUSampler(tmp_path).run(once=True)
    assert gpu_metrics.read_records(tmp_path, {})["records"] == []


def test_all_training_capsules_contain_sampler_with_integrity_receipt():
    spec = make_spec()
    job = compile_sbatch(spec, resolve_adapter_plan(spec), run_id="gpu-monitor-test")
    assert "gpu_metrics.py" in job.files
    assert "  gpu_metrics.py\n" in job.files["checksums.sha256"]
    assert "sampler.stop()" in RUNNER_SOURCE
    assert 'stage == "train"' in RUNNER_SOURCE


def test_wandb_system_stream_retries_without_changing_training_steps(tmp_path):
    bridge = FakeWandBBridge(
        tmp_path, WandBSettings(api_key="fake", entity="team", auto_flush=False)
    )
    bridge.ensure_run(
        local_run_id="run", entity="team", project="test", run_name="test", group="test"
    )
    bridge.log_metrics("run", {"loss": 0.2}, step=77, timestamp_ms=100000)
    assert not bridge.drain_spool().errors
    for i in range(3):
        bridge.log_system_metrics(
            "run",
            {"gpu.0.gpu": 80 + i},
            timestamp_ms=101000 + i * 15000,
            runtime_seconds=1 + i * 15,
            idempotency_key=f"gpu:{i}",
        )
    # The same persisted samples can safely be ingested after a controller restart.
    bridge.log_system_metrics(
        "run",
        {"gpu.0.gpu": 80},
        timestamp_ms=101000,
        runtime_seconds=1,
        idempotency_key="gpu:0",
    )
    with patch.object(
        bridge,
        "_append_system_rows",
        side_effect=TrackingRequestError("busy", status_code=429, retry_after=60),
    ):
        report = bridge.drain_spool()
    assert report.remaining == 3
    assert bridge.binding("run")["events_offset"] == 0
    assert bridge.binding("run")["history_offset"] == 1
    assert bridge.drain_spool().attempted == 0
    with patch("skynet_app.tracking.time.time", return_value=time.time() + 120):
        assert not bridge.drain_spool().errors
    assert len(bridge.system_rows) == 3
    assert bridge.history_rows[0]["_step"] == 77
    assert "_step" not in bridge.system_rows[0]
    assert bridge.system_rows[0]["system.gpu.0.gpu"] == 80
    assert bridge.system_rows[0]["_wandb"] is True
    bridge.ensure_run(
        local_run_id="run", entity="team", project="test", run_name="test", group="test"
    )
    bridge.drain_spool()
    assert bridge.binding("run")["events_offset"] == 3
    assert "gpu:0" in bridge.metric_idempotency_keys()


def test_wandb_upload_uses_separate_event_file(tmp_path):
    bridge = WandBBridge(tmp_path, WandBSettings(api_key="fake"))
    run = {
        "entity": "team",
        "project": "test",
        "name": "run",
        "history_offset": 12,
        "events_offset": 3,
    }
    with patch("skynet_app.tracking.urllib.request.urlopen") as request:
        bridge._append_system_rows(run, [{"system.gpu.0.gpu": 99, "_timestamp": 100}])
    body = json.loads(request.call_args.args[0].data)
    assert list(body["files"]) == ["wandb-events.jsonl"]
    assert body["files"]["wandb-events.jsonl"]["offset"] == 3
    assert run["events_offset"] == 4 and run["history_offset"] == 12


def test_existing_job_probe_is_scoped_bounded_and_does_not_edit_capsule():
    calls = []
    service = SimpleNamespace(
        cluster=SimpleNamespace(
            run_with_fallback=lambda *a, **kw: calls.append((a, kw))
        )
    )
    gpu_tracking._sample_existing(
        service,
        "/runs/run/state/gpu-stats/123",
        {"slurm_job_id": "123", "gateway": "sky2"},
        1,
    )
    command = calls[0][0][0]
    assert "--jobid=123" in command and "--overlap" in command and "--once" in command
    assert command.startswith("timeout 15 ")
    assert "scancel" not in command and "sbatch" not in command


def test_native_sdk_keeps_ownership_of_wandb_stream(tmp_path):
    service = SimpleNamespace(_native_tracking_provider_names=lambda spec: {"wandb"})
    run = {
        "id": "native",
        "run_directory": "/runs/native",
        "resolved_spec_json": make_spec().model_dump(mode="json", by_alias=True),
    }
    assert gpu_tracking.sync_gpu_statistics(service, run, tmp_path, force=True) == 0
    assert not list(tmp_path.iterdir())


def test_controller_ingests_existing_job_once_and_preserves_pending_samples(tmp_path):
    settings = WandBSettings(api_key="fake", entity="team", auto_flush=False)
    bridge = FakeWandBBridge(tmp_path / "run", settings)
    bridge.ensure_run(
        local_run_id="run", entity="team", project="test", run_name="test", group="test"
    )
    bridge.drain_spool()
    service = SimpleNamespace(
        _native_tracking_provider_names=lambda spec: set(),
        _active_tracking_providers=lambda spec: [{"provider": "wandb"}],
        _tracking_provider_value=lambda provider, key: provider[key],
        _wandb_settings=lambda provider: settings,
        _training_progress_timestamp_ms=lambda value: 100000,
        _tracking_failure=lambda *a: None,
    )
    run = {
        "id": "run",
        "run_directory": "/runs/run",
        "resolved_spec_json": make_spec().model_dump(mode="json", by_alias=True),
        "stages": [
            {"id": "train", "stage_type": "TRAIN"},
            {"id": "eval", "stage_type": "EVAL"},
        ],
        "attempts": [
            {
                "id": "attempt",
                "stage_id": "train",
                "status": "RUNNING",
                "slurm_job_id": "123",
            },
            {
                "id": "evaluation",
                "stage_id": "eval",
                "status": "RUNNING",
                "slurm_job_id": "456",
            },
        ],
    }
    record = {
        "timestamp_ms": 120000,
        "job_id": "123",
        "node": "node-a",
        "devices": gpu_metrics.parse_devices(CSV),
    }
    empty = {"records": [], "cursors": {}, "latest": 0, "oldest": 0}
    response = {
        "records": [record],
        "cursors": {"node-a.jsonl": 100},
        "latest": time.time(),
        "oldest": time.time(),
    }
    with (
        patch.object(gpu_tracking, "WandBBridge", return_value=bridge),
        patch.object(gpu_tracking, "_read", side_effect=[empty, response, response]),
        patch.object(gpu_tracking, "_sample_existing") as probe,
    ):
        assert gpu_tracking.sync_gpu_statistics(service, run, tmp_path, force=True) == 1
        assert gpu_tracking.sync_gpu_statistics(service, run, tmp_path, force=True) == 0
    assert probe.call_count == 1
    assert probe.call_args.args[2]["slurm_job_id"] == "123"
    assert len(bridge.system_rows) == 1
    assert bridge.system_rows[0]["_runtime"] == 20
    state = json.loads((tmp_path / "run/gpu-statistics.json").read_text())
    assert state["attempt"]["cursors"] == response["cursors"]
