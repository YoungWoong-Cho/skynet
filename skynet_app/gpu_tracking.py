"""Publish allocation-scoped GPU samples through the existing W&B bridge."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
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
from .tracking import WandBBridge, sanitize

_LOCK = threading.Lock()
_LAST_POLLS = {}
_SOURCE = Path(__file__).with_name("gpu_metrics.py").read_text()


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


def _sync(service, run, capsule_root, force):
    run_id = str(run.get("id") or "")
    if not run_id or not run.get("run_directory"):
        return 0
    spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
    # Do not compete with an adapter's native W&B SDK for the events stream.
    if "wandb" in service._native_tracking_provider_names(spec):
        return 0
    provider = next(
        (
            p
            for p in service._active_tracking_providers(spec)
            if service._tracking_provider_value(p, "provider") == "wandb"
        ),
        None,
    )
    if provider is None:
        return 0
    bridge = WandBBridge(
        Path(capsule_root) / run_id,
        replace(service._wandb_settings(provider), auto_flush=False),
    )
    if not bridge.binding(run_id):
        return 0
    if not force and time.monotonic() - _LAST_POLLS.get(run_id, 0) < 30:
        return 0
    _LAST_POLLS[run_id] = time.monotonic()
    state_path = Path(capsule_root) / run_id / "gpu-statistics.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
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
        start = service._training_progress_timestamp_ms(
            run.get("started_at") or attempt.get("started_at")
        )
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
                bridge.log_system_metrics(
                    run_id,
                    metrics,
                    timestamp_ms=timestamp,
                    runtime_seconds=(timestamp - start) / 1000,
                    idempotency_key=key,
                )
                emitted.add(key)
                published += 1
            state["latest"] = record
        state[attempt["id"]] = {
            "cursors": response["cursors"],
            "finished_at": attempt.get("finished_at"),
        }
    state["updated_at"] = datetime.now().astimezone().isoformat()
    state.pop("error", None)
    bridge._atomic_write(state_path, json.dumps(state).encode())
    state_path.with_name("gpu-statistics-error.txt").unlink(missing_ok=True)
    report = bridge.drain_spool()
    if report.errors:
        service._tracking_failure("wandb", run, RuntimeError(report.errors[0]))
    return published


def sync_gpu_statistics(service, run, capsule_root, *, force=False):
    if not _LOCK.acquire(blocking=False):
        return 0
    try:
        try:
            return _sync(service, run, capsule_root, force)
        except Exception as error:
            # Telemetry must never change the training outcome. Keep an actionable
            # diagnostic alongside the local run rather than failing the trainer.
            path = (
                Path(capsule_root) / str(run.get("id", "")) / "gpu-statistics-error.txt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(sanitize(str(error)))[:2000])
            return 0
    finally:
        _LOCK.release()
