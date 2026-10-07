"""Publish allocation-scoped GPU samples through the existing W&B bridge."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import threading
import time

from .cluster_runtime import SLURM_BIN
from .experiments import ExperimentSpec
from .tracking import sanitize

_LOCK = threading.Lock()
_LAST_POLLS = {}
_SOURCE = Path(__file__).with_name("gpu_metrics.py").read_text()
STATE_FILENAME = "gpu-statistics.json"
ERROR_FILENAME = "gpu-statistics-error.txt"


def _error_path(capsule_root, run_id):
    return Path(capsule_root) / run_id / ERROR_FILENAME


def progress_timestamp_ms(value):
    """Epoch milliseconds of a recorded ISO-8601 timestamp; now when it is absent or unreadable."""
    if not isinstance(value, str) or not value.strip():
        return int(datetime.now(timezone.utc).timestamp() * 1000)
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return int(datetime.now(timezone.utc).timestamp() * 1000)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def system_metrics(record, multi_node=False):
    metrics = {}
    node = re.sub(r"[^A-Za-z0-9_-]", "_", str(record.get("node", "unknown")))
    for device in record.get("devices", []):
        index = device.get("index")
        if not isinstance(index, int) or index < 0:
            continue
        prefix = f"gpu.{node}.{index}." if multi_node else f"gpu.{index}."
        for key, value in device.get("metrics", {}).items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and re.fullmatch(r"[A-Za-z]+", key)
            ):
                metrics[prefix + key] = value
    return metrics


def _read(service, directory, cursors, gateway):
    command = shlex.join(
        [
            "python3",
            "-c",
            _SOURCE,
            "--read-dir",
            directory,
            "--cursors",
            json.dumps(cursors),
        ]
    )
    _, output = service.cluster.run_with_fallback(command, gateway, timeout=15)
    return json.loads(output)


def _sample_existing(service, directory, attempt, nodes):
    # Old immutable capsules lack the sampler. Take a short observation inside
    # their allocation; no checkpoint, trainer, or pinned source is changed.
    command = shlex.join(
        [
            "timeout",
            "15",
            str(PurePosixPath(SLURM_BIN) / "srun"),
            "--overlap",
            "--immediate=3",
            "--jobid=" + str(attempt["slurm_job_id"]),
            "--nodes=" + str(nodes),
            "--ntasks=" + str(nodes),
            "--ntasks-per-node=1",
            "--cpus-per-task=1",
            "--cpu-bind=none",
            "python3",
            "-c",
            _SOURCE,
            "--output",
            directory,
            "--once",
        ]
    )
    service.cluster.run_with_fallback(
        command, attempt.get("gateway") or "auto", timeout=20
    )


def _sync(service, run, capsule_root, force, *, raise_on_error=False):
    run_id = str(run.get("id") or "")
    if not run_id or not run.get("run_directory"):
        return 0
    spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
    bridge = service.gpu_tracking_bridge(run, spec, capsule_root)
    if bridge is None:
        return 0
    if not bridge.binding(run_id):
        if raise_on_error:
            raise RuntimeError("W&B run binding is not ready for GPU metrics")
        return 0
    if not force and time.monotonic() - _LAST_POLLS.get(run_id, 0) < 30:
        return 0
    _LAST_POLLS[run_id] = time.monotonic()
    state_file = bridge.sidecar(STATE_FILENAME)
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    train_stages = {
        s["id"] for s in run.get("stages", []) if s.get("stage_type") == "TRAIN"
    }
    emitted = bridge.metric_idempotency_keys()
    published = 0
    for attempt in run.get("attempts", []):
        job_id = str(attempt.get("slurm_job_id") or "")
        if attempt.get("stage_id") not in train_stages or not re.fullmatch(
            r"\d+(?:_\d+)?", job_id
        ):
            continue
        prior = state.get(attempt["id"], {})
        if prior.get("finished_at") and prior["finished_at"] == attempt.get(
            "finished_at"
        ):
            continue
        directory = str(
            PurePosixPath(run["run_directory"]) / "state/gpu-stats" / job_id
        )
        gateway = attempt.get("gateway") or "auto"
        response = _read(service, directory, prior.get("cursors", {}), gateway)
        running = attempt.get("status") == "RUNNING"
        if running and (
            time.time() - response.get("oldest", 0) > 60
            or len(response["cursors"]) < spec.resources.nodes
        ):
            _sample_existing(service, directory, attempt, spec.resources.nodes)
            response = _read(service, directory, prior.get("cursors", {}), gateway)
        while response.get("backlog"):
            page = _read(service, directory, response["cursors"], gateway)
            if page["cursors"] == response["cursors"]:
                raise ValueError(
                    "GPU statistics contain an oversized or incomplete record"
                )
            response["records"].extend(page["records"])
            response.update(
                {key: value for key, value in page.items() if key != "records"}
            )
        start = progress_timestamp_ms(run.get("started_at") or attempt.get("started_at"))
        samples = []
        for record in response["records"]:
            if str(record.get("job_id")) != job_id:
                continue
            metrics = system_metrics(record, spec.resources.nodes > 1)
            if not metrics:
                continue
            timestamp = int(record["timestamp_ms"])
            identity = hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()
            ).hexdigest()
            key = "gpu-statistics:" + attempt["id"] + ":" + identity
            if key not in emitted:
                samples.append({
                    "metrics": metrics,
                    "timestamp_ms": timestamp,
                    "runtime_seconds": (timestamp - start) / 1000,
                    "idempotency_key": key,
                })
                emitted.add(key)
                published += 1
            state["latest"] = record
        bridge.log_system_metrics_batch(run_id, samples)
        # Advance the source cursor only after all samples are durably queued.
        state[attempt["id"]] = {
            "cursors": response["cursors"],
            "finished_at": attempt.get("finished_at"),
        }
    state["updated_at"] = datetime.now().astimezone().isoformat()
    state.pop("error", None)
    bridge.write_sidecar(STATE_FILENAME, json.dumps(state).encode())
    _error_path(capsule_root, run_id).unlink(missing_ok=True)
    error = service.deliver_gpu_tracking(bridge, run)
    if error is not None and raise_on_error:
        raise error
    return published


def sync_gpu_statistics(
    service, run, capsule_root, *, force=False, raise_on_error=False
):
    database = getattr(service, "database", None)
    # Different runs have independent GPU logs and delivery cursors. Keep
    # same-run writers serialized across hosts without blocking other runs.
    lock = (database.operation_lock("gpu-tracking:" + str(run.get("id")))
            if database is not None else _LOCK)
    if not lock.acquire(blocking=False):
        if raise_on_error:
            raise RuntimeError("GPU metric synchronization is already in progress")
        return 0
    try:
        try:
            return _sync(service, run, capsule_root, force, raise_on_error=raise_on_error)
        except Exception as error:
            # Telemetry must never change the training outcome. Keep an actionable
            # diagnostic alongside the local run rather than failing the trainer.
            path = _error_path(capsule_root, str(run.get("id", "")))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(sanitize(str(error)))[:2000])
            if raise_on_error:
                raise
            return 0
    finally:
        lock.release()
