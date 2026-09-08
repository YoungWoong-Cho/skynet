from __future__ import annotations

import math
import ipaddress
import urllib.parse

import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .adapters import (
    AdapterPlan,
    AdapterManifest,
    TrainingProgressContract,
    apply_pinned_adapter_plan_compatibility,
    adapter_manifest_sha256,
    builtin_adapter_manifests,
    canonical_adapter_manifest,
    resolve_adapter_evaluation_plan,
    resolve_adapter_plan,
    resolve_gpu_count,
)
from .cluster_config import CLUSTER
from .cluster_runtime import (
    ClusterClient,
    ClusterError,
    SLURM_BIN,
    SubmissionOutcomeUnknown,
    WORK_ROOT,
    approved_operator_environment,
)
from .data_imports import build_huggingface_import_job
from .data_preview import build_data_bundle_preview, resolve_data_bundle_preview_media
from .credential_store import (
    CredentialStore,
    CredentialStoreError,
    KeyringCredentialStore,
)
from .database import Database, canonical_json, content_sha256, utc_now
from .experiments import (
    CanonicalResult,
    EvaluationSpec,
    ExperimentSpec,
    FULL_COMMIT_RE,
    ResourceSpec,
    ResolvedVariant,
    canonical_sha256,
    explicit_train_parameter_paths,
    expand_sweep,
    get_evaluation_catalog,
    parse_slurm_duration,
)
from .slurm import (
    CompiledSlurmJob,
    SlurmCompileError,
    compile_sbatch,
    resolve_slurm_log_path,
    resolve_slurm_log_paths_from_sbatch,
)
from .source_control import SourceDiscovery, resolve_runtime
from .source_metadata_cache import SourceMetadataStore
from .source_validation import (
    CACHE_KIND as REPOSITORY_ARGUMENT_VALIDATION_CACHE_KIND,
    repository_argument_validation_cache_parameters,
    validate_repository_arguments,
    validation_failure_reason,
)
from .tracking import (
    MLflowBridge,
    SESSION_CREDENTIALS,
    SessionCredentialStore,
    TrackingRequestError,
    TrackingSettings,
    WandBBridge,
    WandBSettings,
    mlflow_experiment_url,
    mlflow_run_url,
    sanitize,
    wandb_project_slug,
    wandb_web_base,
)
from pydantic import ConfigDict, SecretStr


TRANSIENT_STATES = {"PREEMPTED", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL", "REVOKED"}
ACTIVE_STATES = {
    "PENDING",
    "CONFIGURING",
    "RUNNING",
    "COMPLETING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
}
TERMINAL_FAILURE_STATES = {"FAILED", "OUT_OF_MEMORY", "DEADLINE", "SPECIAL_EXIT"}
GPU_USAGE_LONG_COMMAND = CLUSTER.commands.gpu_usage_shell_command("-l")
LOCAL_CAPSULE_ROOT = Path(
    os.environ.get(
        "SKYNET_LOCAL_CAPSULE_ROOT",
        str(Path(__file__).resolve().parent.parent / "data" / "capsules"),
    )
).expanduser()


def _frontend_parameter_is_unset(value: Any) -> bool:
    return value is None or value == "" or value == "adapter-default"


_PROGRESS_SUCCESS_STATES = {"SUCCEEDED", "COMPLETED", "COMPLETE"}
_PROGRESS_FAILURE_STATES = {
    "FAILED", "CANCELLED", "CANCELED", "TIMEOUT", "TIMED_OUT",
    "OUT_OF_MEMORY", "DEADLINE", "SPECIAL_EXIT", "BLOCKED",
}
_PROGRESS_RUNNING_STATES = {"RUNNING", "COMPLETING"}
_PROGRESS_WAITING_STATES = {
    "DRAFT", "CREATED", "PENDING", "SUBMITTED", "QUEUED", "CONFIGURING",
    "REQUEUED", "RETRY_PENDING", "RESIZING", "SUSPENDED", "PREEMPTED", "CANCELLING",
}
_TRAINING_PROGRESS_METRIC_NAMES = {
    "training_step", "train.step", "train/global_step", "global_step",
    "progress.completed",
}
_EVALUATION_EPISODE_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "TIMEOUT"}


def _progress_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _progress_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _progress_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _progress_now(value: datetime | str | None) -> datetime:
    return _progress_timestamp(value) or datetime.now(timezone.utc)


def _latest_attempt(attempts: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    def key(attempt: Mapping[str, Any]) -> tuple[int, datetime]:
        number = _progress_integer(attempt.get("attempt_number")) or 0
        created = _progress_timestamp(attempt.get("created_at")) or datetime.min.replace(
            tzinfo=timezone.utc
        )
        return number, created

    return max(attempts, key=key) if attempts else None


def _progress_elapsed(
    started_at: datetime | None,
    now: datetime,
    finished_at: datetime | None = None,
) -> int | None:
    if started_at is None:
        return None
    end = finished_at or now
    if end < started_at:
        return None
    return int((end - started_at).total_seconds())


def _progress_payload(
    *,
    completed: int | None,
    total: int | None,
    unit: str,
    observed_at: datetime | None,
    elapsed_seconds: int | None,
    eta_seconds: int | None,
    eta_state: str,
    eta_reason: str,
) -> dict[str, Any]:
    fraction = None
    if completed is not None and total is not None and total > 0:
        fraction = max(0.0, min(1.0, completed / total))
    return {
        "completed": completed,
        "total": total,
        "unit": unit,
        "fraction": fraction,
        "observed_at": _progress_iso(observed_at),
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
        "eta_state": eta_state,
        "eta_reason": eta_reason,
    }


def _resolved_training_max_steps(value: Any) -> int | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping):
        return None
    train = value.get("train")
    if not isinstance(train, Mapping):
        return None
    total = _progress_integer(train.get("max_steps"))
    return total if total and total > 0 else None


_DECIMAL_SI_PROGRESS_RE = re.compile(r"^(?P<number>[0-9]+(?:\.[0-9]+)?)(?P<scale>[kMGT]?)$")


def _declared_progress_number(value: str, value_format: str) -> int | None:
    raw = value.strip()
    if value_format == "integer":
        return _progress_integer(raw)
    match = _DECIMAL_SI_PROGRESS_RE.fullmatch(raw)
    if not match:
        return None
    try:
        number = Decimal(match.group("number")) * {
            "": Decimal(1),
            "k": Decimal(1_000),
            "M": Decimal(1_000_000),
            "G": Decimal(1_000_000_000),
            "T": Decimal(1_000_000_000_000),
        }[match.group("scale")]
    except (InvalidOperation, KeyError):
        return None
    integral = number.to_integral_value()
    return int(integral) if number == integral and integral >= 0 else None


def _declared_elapsed_seconds(value: str, elapsed_format: str | None) -> int | None:
    if elapsed_format is None:
        return None
    if elapsed_format == "seconds":
        return _progress_integer(value)
    parts = value.split(":")
    if elapsed_format == "clock" and len(parts) == 2:
        parts.insert(0, "0")
    if len(parts) != 3:
        return None
    hours, minutes, seconds = (_progress_integer(part) for part in parts)
    if None in {hours, minutes, seconds} or minutes >= 60 or seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def parse_declared_training_progress(
    content: str, contract: TrainingProgressContract | Mapping[str, Any]
) -> list[dict[str, int | None]]:
    """Normalize bounded adapter-declared log matches; no adapter grammar lives here."""
    resolved = (
        contract
        if isinstance(contract, TrainingProgressContract)
        else TrainingProgressContract.model_validate(contract)
    )
    source = resolved.source
    pattern = re.compile(source.pattern)
    records: list[dict[str, int | None]] = []
    for line in content.splitlines():
        match = pattern.search(line[-4096:])
        if match is None:
            continue
        completed = _declared_progress_number(match.group("completed"), source.value_format)
        total = _declared_progress_number(match.group("total"), source.value_format)
        elapsed = (
            _declared_elapsed_seconds(match.group("elapsed"), source.elapsed_format)
            if source.elapsed_format is not None
            else None
        )
        if completed is None or total is None or total < 1 or completed > total:
            continue
        if source.elapsed_format is not None and elapsed is None:
            continue
        record = {"completed": completed, "total": total, "elapsed_seconds": elapsed}
        if not records or record != records[-1]:
            records.append(record)
    return records


def training_progress_summary(
    run: Mapping[str, Any],
    *,
    attempts: list[Mapping[str, Any]] | None = None,
    checkpoints: list[Mapping[str, Any]] | None = None,
    metrics: list[Mapping[str, Any]] | None = None,
    progress_samples: list[Mapping[str, Any]] | None = None,
    resolved_spec: Any = None,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    """Build an ETA only from persisted step evidence in the current attempt."""
    clock = _progress_now(now)
    status = str(run.get("status") or run.get("state") or "").upper()
    attempt_rows = list(attempts if attempts is not None else run.get("attempts") or [])
    checkpoint_rows = list(
        checkpoints if checkpoints is not None else run.get("checkpoints") or []
    )
    metric_rows = list(metrics if metrics is not None else run.get("metrics") or [])
    canonical_rows = list(
        progress_samples
        if progress_samples is not None
        else run.get("progress_samples") or []
    )
    spec = resolved_spec if resolved_spec is not None else run.get("resolved_spec_json")
    total = _resolved_training_max_steps(spec)
    current_attempt = _latest_attempt(attempt_rows)
    current_attempt_id = str(current_attempt.get("id")) if current_attempt and current_attempt.get("id") else None
    current_restart_count = _progress_integer((current_attempt or {}).get("restart_count")) or 0
    attempt_status = str(
        (current_attempt or {}).get("status") or (current_attempt or {}).get("state") or ""
    ).upper()
    started_at = _progress_timestamp((current_attempt or {}).get("started_at"))
    finished_at = _progress_timestamp(
        (current_attempt or {}).get("finished_at") or (current_attempt or {}).get("ended_at")
    )
    elapsed = _progress_elapsed(started_at, clock, finished_at)

    samples: list[tuple[datetime, int, str | None]] = []
    checkpoint_by_id: dict[str, Mapping[str, Any]] = {}
    for checkpoint in checkpoint_rows:
        checkpoint_id = checkpoint.get("id")
        if checkpoint_id:
            checkpoint_by_id[str(checkpoint_id)] = checkpoint
        step = _progress_integer(checkpoint.get("training_step"))
        recorded = _progress_timestamp(checkpoint.get("created_at"))
        if step is not None and recorded is not None:
            producer = checkpoint.get("produced_by_attempt_id")
            samples.append((recorded, step, str(producer) if producer else None))
    for sample in canonical_rows:
        step = _progress_integer(sample.get("completed"))
        recorded = _progress_timestamp(sample.get("recorded_at"))
        producer = sample.get("attempt_id")
        sample_restart_count = _progress_integer(sample.get("restart_count")) or 0
        if (
            step is not None
            and recorded is not None
            and producer
            and (
                str(producer) != current_attempt_id
                or sample_restart_count == current_restart_count
            )
        ):
            samples.append((recorded, step, str(producer)))
    for metric in metric_rows:
        if metric.get("evaluation_id") is not None:
            continue
        name = str(metric.get("name") or "").strip().lower()
        scope = str(metric.get("scope") or "").strip().upper()
        if name not in _TRAINING_PROGRESS_METRIC_NAMES or scope not in {"TRAIN", "TRAINING"}:
            continue
        step = _progress_integer(metric.get("value"))
        if step is None:
            step = _progress_integer(metric.get("step"))
        recorded = _progress_timestamp(metric.get("recorded_at"))
        if step is not None and recorded is not None:
            producer = metric.get("attempt_id")
            samples.append((recorded, step, str(producer) if producer else None))
    samples.sort(key=lambda sample: sample[0])

    baseline: tuple[datetime, int, str | None] | None = None
    baseline_observed_at: datetime | None = None
    resume_checkpoint_id = (current_attempt or {}).get("resume_checkpoint_id")
    if resume_checkpoint_id is not None:
        checkpoint = checkpoint_by_id.get(str(resume_checkpoint_id))
        baseline_step = _progress_integer((checkpoint or {}).get("training_step"))
        if checkpoint is not None and baseline_step is not None and started_at is not None:
            baseline = (started_at, baseline_step, current_attempt_id)
            baseline_observed_at = _progress_timestamp(checkpoint.get("created_at"))

    current_samples: list[tuple[datetime, int, str | None]] = []
    if current_attempt is not None:
        for sample in samples:
            recorded, _, producer = sample
            if producer is not None:
                if current_attempt_id is not None and producer == current_attempt_id:
                    current_samples.append(sample)
            elif started_at is not None and recorded >= started_at:
                current_samples.append(sample)
    elif samples:
        current_samples = samples

    observed_sample = current_samples[-1] if current_samples else baseline
    if observed_sample is None and status in _PROGRESS_SUCCESS_STATES | _PROGRESS_FAILURE_STATES:
        observed_sample = samples[-1] if samples else None
    completed = observed_sample[1] if observed_sample else None
    observed_at = (
        baseline_observed_at
        if baseline is not None and observed_sample is baseline
        else observed_sample[0] if observed_sample else None
    )

    common = {
        "completed": completed,
        "total": total,
        "unit": "step",
        "observed_at": observed_at,
        "elapsed_seconds": elapsed,
    }
    if status in _PROGRESS_SUCCESS_STATES:
        return _progress_payload(
            **common, eta_seconds=0, eta_state="complete", eta_reason="terminal_success"
        )
    if status in _PROGRESS_FAILURE_STATES:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="not_applicable",
            eta_reason="terminal_without_completion",
        )
    if completed is not None and total is not None and completed >= total:
        return _progress_payload(
            **common, eta_seconds=0, eta_state="complete", eta_reason="work_complete"
        )
    if total is None:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="unknown",
            eta_reason="resolved_max_steps_unavailable",
        )
    if current_attempt is None or started_at is None:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="waiting", eta_reason="attempt_not_started"
        )
    if status in _PROGRESS_WAITING_STATES or attempt_status in _PROGRESS_WAITING_STATES:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="waiting", eta_reason="attempt_not_running"
        )
    if attempt_status and attempt_status not in _PROGRESS_RUNNING_STATES:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="waiting", eta_reason="waiting_for_retry"
        )
    if not current_samples:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="waiting", eta_reason="waiting_for_progress"
        )

    rate_samples = ([baseline] if baseline is not None else []) + current_samples
    anchor = rate_samples[0]
    latest = rate_samples[-1]
    for candidate in rate_samples[1:-1]:
        if candidate[1] <= anchor[1]:
            anchor = candidate
    evidence_duration = (latest[0] - anchor[0]).total_seconds()
    duration = (clock - anchor[0]).total_seconds()
    advanced = latest[1] - anchor[1]
    if (
        len(rate_samples) < 2
        or advanced <= 0
        or evidence_duration <= 0
        or duration <= 0
        or latest[0] > clock
    ):
        return _progress_payload(
            **common, eta_seconds=None, eta_state="unknown",
            eta_reason="insufficient_progress_samples",
        )
    eta = math.ceil(max(0, total - latest[1]) / (advanced / duration))
    return _progress_payload(
        **common, eta_seconds=eta, eta_state="estimating",
        eta_reason="current_attempt_progress_rate",
    )


def evaluation_progress_summary(
    evaluation: Mapping[str, Any],
    *,
    attempts: list[Mapping[str, Any]] | None = None,
    episodes: list[Mapping[str, Any]] | None = None,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    """Build an evaluation ETA from its persisted ledger and current attempt only."""
    clock = _progress_now(now)
    status = str(evaluation.get("status") or evaluation.get("state") or "").upper()
    completed = _progress_integer(evaluation.get("progress_completed"))
    total = _progress_integer(evaluation.get("progress_total"))
    attempt_rows = list(attempts if attempts is not None else evaluation.get("attempts") or [])
    episode_rows = list(episodes if episodes is not None else evaluation.get("episodes") or [])
    current_attempt = _latest_attempt(attempt_rows)
    attempt_status = str(
        (current_attempt or {}).get("status") or (current_attempt or {}).get("state") or ""
    ).upper()
    started_at = _progress_timestamp((current_attempt or {}).get("started_at"))
    if started_at is None and (current_attempt is None or len(attempt_rows) <= 1):
        started_at = _progress_timestamp(evaluation.get("started_at"))
    finished_at = _progress_timestamp(
        (current_attempt or {}).get("finished_at") or (current_attempt or {}).get("ended_at")
    )
    elapsed = _progress_elapsed(started_at, clock, finished_at)

    observed_times = []
    for episode in episode_rows:
        if str(episode.get("status") or "").upper() not in _EVALUATION_EPISODE_TERMINAL_STATES:
            continue
        recorded = _progress_timestamp(episode.get("completed_at"))
        if recorded is not None:
            observed_times.append(recorded)
    observed_at = max(observed_times) if observed_times else None
    current_completed = 0
    if started_at is not None:
        current_completed = sum(recorded >= started_at for recorded in observed_times)
        if completed is not None:
            current_completed = min(current_completed, completed)

    common = {
        "completed": completed,
        "total": total,
        "unit": "episode",
        "observed_at": observed_at,
        "elapsed_seconds": elapsed,
    }
    if status in _PROGRESS_SUCCESS_STATES or (
        completed is not None and total is not None and total > 0 and completed >= total
    ):
        return _progress_payload(
            **common, eta_seconds=0, eta_state="complete",
            eta_reason="terminal_success" if status in _PROGRESS_SUCCESS_STATES else "work_complete",
        )
    if status in _PROGRESS_FAILURE_STATES:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="not_applicable",
            eta_reason="terminal_without_completion",
        )
    if total is None or total <= 0:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="unknown", eta_reason="total_work_unavailable"
        )
    if current_attempt is None or started_at is None:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="waiting", eta_reason="attempt_not_started"
        )
    if status in _PROGRESS_WAITING_STATES or attempt_status in _PROGRESS_WAITING_STATES:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="waiting", eta_reason="attempt_not_running"
        )
    if attempt_status and attempt_status not in _PROGRESS_RUNNING_STATES:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="waiting", eta_reason="waiting_for_retry"
        )
    if not completed or current_completed <= 0:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="waiting", eta_reason="waiting_for_progress"
        )
    if elapsed is None or elapsed <= 0:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="unknown",
            eta_reason="current_attempt_timing_unavailable",
        )
    remaining = max(0, total - completed)
    eta = math.ceil(remaining / (current_completed / elapsed))
    return _progress_payload(
        **common, eta_seconds=eta, eta_state="estimating",
        eta_reason="current_attempt_progress_rate",
    )


def _attach_run_progress_summaries(database: Database, runs: list[dict[str, Any]]) -> None:
    evidence = database.run_progress_evidence([str(run.get("id") or "") for run in runs])
    for run in runs:
        row = evidence.get(str(run.get("id") or ""), {})
        attempts = row.get("attempts") or []
        run["attempt_count"] = len(attempts)
        run["latest_attempt"] = _latest_attempt(attempts)
        spec = row.get("resolved_spec_json") or {}
        run["resources"] = spec.get("resources") or {}
        run["progress_summary"] = training_progress_summary(
            run,
            attempts=row.get("attempts") or [],
            checkpoints=row.get("checkpoints") or [],
            metrics=row.get("metrics") or [],
            progress_samples=row.get("progress_samples") or [],
            resolved_spec=row.get("resolved_spec_json"),
        )


def _attach_evaluation_progress_summaries(
    database: Database, evaluations: list[dict[str, Any]]
) -> None:
    evidence = database.evaluation_progress_evidence([
        str(evaluation.get("id") or "") for evaluation in evaluations
    ])
    for evaluation in evaluations:
        row = evidence.get(str(evaluation.get("id") or ""), {})
        evaluation["progress_summary"] = evaluation_progress_summary(
            evaluation,
            attempts=row.get("attempts") or [],
            episodes=row.get("episodes") or [],
        )


class GatewayRequest(BaseModel):
    gateway: str = "auto"


class TrackingConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = None
    base_url: str | None = None
    entity: str | None = None
    tracking_uri: str | None = None
    token: SecretStr | None = None
    username: str | None = None
    password: SecretStr | None = None
    verify_tls: bool = True
    remember: bool = True


class ResumeRequest(GatewayRequest):
    mode: str = "resume"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        if value != "resume":
            raise ValueError(
                "mode='retry' is deprecated; use POST /api/runs/{run_id}/rerun"
            )
        return value


class ExperimentRevisionRequest(GatewayRequest):
    spec: dict[str, Any]
    submit: bool = False


_MANUAL_ACTION_ACTIVE_STATES = frozenset({
    "SUBMITTING", "SUBMITTED", "PENDING", "PENDING_SLURM", "RUNNING", "REQUEUED",
    "RETRY_PENDING", "CANCELLING",
})

_CANCELLABLE_STAGE_STATES = frozenset({
    "CREATED", "PENDING", "RETRY_PENDING", "SUBMITTING", "SUBMITTED",
    "PENDING_SLURM", "RUNNING", "REQUEUED",
})
_RUN_CANCELLATION_STAGE_TYPES = frozenset({"TRAIN", "UTILITY"})


def manual_cancel_action(
    stage: Mapping[str, Any] | None,
    attempts: list[Mapping[str, Any]],
) -> dict[str, bool | str]:
    if stage is None:
        return {"enabled": False, "reason": "No cancellable workflow stage exists."}
    state = str(stage.get("status") or "").upper()
    if state == "CANCELLING":
        return {"enabled": False, "reason": "Cancellation has already been requested."}
    if state == "CANCELLED":
        return {"enabled": False, "reason": "This work has already been cancelled."}
    if state not in _CANCELLABLE_STAGE_STATES:
        return {
            "enabled": False,
            "reason": f"Only queued, submitting, or active work can be cancelled; current state is {state or 'UNKNOWN'}.",
        }
    stage_attempts = [
        attempt for attempt in attempts
        if not attempt.get("stage_id") or attempt.get("stage_id") == stage.get("id")
    ]
    has_job = any(attempt.get("slurm_job_id") for attempt in stage_attempts)
    return {
        "enabled": True,
        "reason": (
            "Request Slurm cancellation for the active attempt; logs, checkpoints, and history are preserved."
            if has_job
            else "Cancel this queued work before it starts; existing history is preserved."
        ),
    }


def _run_cancellation_stage(run: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates = [
        stage for stage in list(run.get("stages") or [])
        if str(stage.get("stage_type") or "").upper() in _RUN_CANCELLATION_STAGE_TYPES
    ]
    active = [
        stage for stage in candidates
        if str(stage.get("status") or "").upper()
        in (_CANCELLABLE_STAGE_STATES | {"CANCELLING"})
    ]
    return (active or candidates)[-1] if (active or candidates) else None
_MANUAL_ACTION_SUCCESS_STATES = frozenset({"SUCCEEDED", "COMPLETED"})
_MANUAL_ACTION_FAILURE_STATES = frozenset({
    "FAILED", "INTERRUPTED", "PREEMPTED", "TIMEOUT", "OUT_OF_MEMORY", "OOM", "CANCELLED",
    "NODE_FAIL", "BOOT_FAIL", "DEADLINE",
})


def _training_stage_attempts(
    run: Mapping[str, Any], stage: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    return sorted(
        (
            item for item in list(run.get("attempts") or [])
            if not item.get("stage_id") or item.get("stage_id") == stage.get("id")
        ),
        key=lambda item: (
            int(item.get("attempt_number") or item.get("number") or 0),
            str(item.get("created_at") or ""),
        ),
    )


def _evaluation_busy_reason(run: Mapping[str, Any]) -> str | None:
    active_states = {"SUBMITTING", "SUBMITTED", "PENDING_SLURM", "RUNNING", "RETRY_PENDING", "CANCELLING"}
    if any(str(stage.get("status", "")).upper() in active_states for stage in run.get("stages", [])):
        return "This run already has active training or evaluation work. Wait for it to finish, or cancel that work before starting another evaluation."
    return None


def _resumable_checkpoint(run: Mapping[str, Any]) -> Mapping[str, Any] | None:
    unusable_states = {"DELETED", "INVALID", "MISSING", "PRUNED", "UNAVAILABLE"}
    return next(
        (
            item for item in reversed(list(run.get("checkpoints") or []))
            if bool(item.get("is_resumable"))
            and bool(item.get("path"))
            and str(item.get("status") or "AVAILABLE").upper() not in unusable_states
        ),
        None,
    )


def _pinned_training_execution(
    run: Mapping[str, Any],
    stage: Mapping[str, Any],
    *,
    prefer_initial_attempt: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate and return immutable execution evidence without registry lookup."""

    attempts = _training_stage_attempts(run, stage)
    source_attempt: Mapping[str, Any] | None = attempts[0] if attempts else None
    snapshot: Mapping[str, Any] | None = None
    source: str
    repository_inputs: dict[str, Any] | None = None
    if prefer_initial_attempt and source_attempt is not None:
        candidate = source_attempt.get("execution_snapshot_json")
        if not isinstance(candidate, Mapping):
            return None, "the first attempt is missing its immutable execution snapshot"
        snapshot = candidate
        snapshot_sha256 = source_attempt.get("execution_snapshot_sha256")
        if snapshot_sha256 and snapshot_sha256 != content_sha256(snapshot):
            return None, "the first attempt execution snapshot hash failed its integrity check"
        spec_payload = snapshot.get("resolved_spec")
        plan_payload = snapshot.get("plan")
        source = "job_attempts.execution_snapshot_json"
        resolved_spec_sha256 = snapshot.get("resolved_spec_sha256")
        plan_sha256 = snapshot.get("plan_sha256")
        selected = snapshot.get("repository_inputs")
        if selected is not None:
            if not isinstance(selected, Mapping) or not isinstance(
                selected.get("selected"), Mapping
            ):
                return None, "the first attempt repository-input evidence is invalid"
            repository_inputs = copy.deepcopy(dict(selected["selected"]))
    else:
        config = stage.get("resolved_config_json")
        if not isinstance(config, Mapping):
            return None, "the training stage has no immutable resolved configuration"
        spec_payload = config.get("spec")
        plan_payload = config.get("plan")
        source = "workflow_stages.resolved_config_json"
        resolved_spec_sha256 = config.get("spec_sha256")
        plan_sha256 = config.get("plan_sha256")

    if not isinstance(spec_payload, Mapping) or not isinstance(plan_payload, Mapping):
        return None, "the pinned training spec or adapter plan is missing"
    if resolved_spec_sha256 and resolved_spec_sha256 != canonical_sha256(spec_payload):
        return None, "the pinned training spec failed its integrity check"
    raw_plan = copy.deepcopy(dict(plan_payload))
    if plan_sha256 and plan_sha256 != canonical_sha256(raw_plan):
        return None, "the pinned adapter plan failed its integrity check"
    plan_input = copy.deepcopy(raw_plan)
    plan_input.pop("runnable", None)
    try:
        spec = ExperimentSpec.model_validate(copy.deepcopy(dict(spec_payload)))
        plan = AdapterPlan.model_validate(plan_input)
    except (TypeError, ValueError) as error:
        return None, f"the pinned execution evidence is invalid: {sanitize(str(error))}"
    if plan.blockers:
        return None, "the pinned adapter plan is not runnable: " + "; ".join(plan.blockers)
    manifest = spec.source.adapter_manifest
    manifest_sha256 = spec.source.adapter_manifest_sha256
    if not isinstance(manifest, Mapping) or not manifest_sha256:
        return None, "the pinned adapter manifest or its hash is missing"
    if canonical_sha256(manifest) != manifest_sha256:
        return None, "the pinned adapter manifest hash failed its integrity check"

    if snapshot is not None:
        adapter = snapshot.get("adapter")
        if not isinstance(adapter, Mapping):
            return None, "the first attempt has no pinned adapter identity"
        expected_adapter = {
            "id": spec.source.adapter_id,
            "version_id": spec.source.adapter_version_id,
            "slug": spec.source.adapter,
            "version": spec.source.adapter_version,
            "manifest_sha256": spec.source.adapter_manifest_sha256,
        }
        for key, expected in expected_adapter.items():
            observed = adapter.get(key)
            if observed is not None and expected is not None and observed != expected:
                return None, f"the first attempt pinned adapter {key} does not match its spec"
        snapshot_manifest = adapter.get("manifest")
        if not isinstance(snapshot_manifest, Mapping) or canonical_sha256(
            snapshot_manifest
        ) != manifest_sha256:
            return None, "the first attempt pinned adapter manifest hash is invalid"
        argv = snapshot.get("argv")
        if not isinstance(argv, list) or argv != list(plan.argv):
            return None, "the first attempt command does not match its pinned adapter plan"
        argv_sha256 = snapshot.get("argv_sha256")
        if argv_sha256 and argv_sha256 != canonical_sha256(argv):
            return None, "the first attempt command failed its integrity check"
        resume_argv = snapshot.get("resume_argv")
        if not isinstance(resume_argv, list) or resume_argv != list(plan.resume_argv):
            return None, "the first attempt resume command does not match its pinned adapter plan"

    return {
        "resolved_spec": spec.model_dump(mode="json", by_alias=True),
        "plan": plan.model_dump(mode="json"),
        "repository_inputs": repository_inputs,
        "source": source,
        "source_attempt_id": source_attempt.get("id") if snapshot is not None else None,
        "source_spec_sha256": canonical_sha256(spec_payload),
        "source_plan_sha256": canonical_sha256(raw_plan),
    }, None


def manual_run_actions(
    run: Mapping[str, Any],
    *,
    current_adapter: Mapping[str, Any] | None = None,
    clean_retry_error: str | None = None,
) -> dict[str, dict[str, bool | str]]:
    """Return the authoritative eligibility and explanation for manual run actions."""
    disabled = lambda reason: {"enabled": False, "reason": reason}
    stages = list(run.get("stages") or [])
    attempts = list(run.get("attempts") or [])
    cancel_stage = _run_cancellation_stage(run)
    cancel = manual_cancel_action(cancel_stage, attempts)
    stage = next((item for item in reversed(stages) if item.get("stage_type") == "TRAIN"), None)
    if not stage:
        reason = "This run has no training stage."
        return {"resume": disabled(reason), "rerun": disabled(reason), "cancel": cancel}

    stage_state = str(stage.get("status") or "").upper()
    run_state = str(run.get("status") or "").upper()
    has_active_attempt = any(
        str(item.get("status") or "").upper() in _MANUAL_ACTION_ACTIVE_STATES
        for item in _training_stage_attempts(run, stage)
    )
    if (
        stage_state in _MANUAL_ACTION_ACTIVE_STATES
        or run_state in _MANUAL_ACTION_ACTIVE_STATES
        or has_active_attempt
    ):
        reason = "The run already has an active or queued attempt."
        actions = {"resume": disabled(reason), "rerun": disabled(reason), "cancel": cancel}
        pending = stage_state == "SUBMITTING" and run_state != "CANCELLING" and any(
            item.get("status") == "SUBMITTING" and not item.get("slurm_job_id")
            and str(item.get("slurm_reason") or "").startswith("Submission outcome unknown")
            for item in _training_stage_attempts(run, stage))
        if pending:
            actions["recover_submission"] = {"enabled": True,
                "reason": "Check and finish the same submission through the selected SSH gateway. Its original identity prevents duplicate jobs."}
        return actions

    resolved = stage.get("resolved_config_json") or {}
    stored_spec = resolved.get("spec") if isinstance(resolved, Mapping) else None
    stored_source = stored_spec.get("source") if isinstance(stored_spec, Mapping) else None
    pinned_version = (
        stored_source.get("adapter_version")
        if isinstance(stored_source, Mapping)
        else run.get("adapter_version")
    )
    pinned_label = f"v{pinned_version}" if pinned_version is not None else "unknown version"
    rerun = {
        "enabled": True,
        "reason": (
            "Start an independent checkpoint-free run from the same immutable "
            f"variant and pinned adapter {pinned_label}."
        ),
    }
    if stage_state in _MANUAL_ACTION_SUCCESS_STATES or run_state in _MANUAL_ACTION_SUCCESS_STATES:
        reason = "Successful runs cannot be resumed; start a new run instead."
        return {"resume": disabled(reason), "rerun": rerun, "cancel": cancel}
    if stage_state not in _MANUAL_ACTION_FAILURE_STATES and run_state not in _MANUAL_ACTION_FAILURE_STATES:
        reason = "Only an unsuccessful terminal run can be resumed."
        return {"resume": disabled(reason), "rerun": rerun, "cancel": cancel}

    checkpoint = _resumable_checkpoint(run)
    pinned_stage, pinned_stage_error = _pinned_training_execution(
        run, stage, prefer_initial_attempt=True
    )
    resume_argv = (pinned_stage or {}).get("plan", {}).get("resume_argv")
    if checkpoint is not None and pinned_stage is not None and resume_argv:
        resume = {
            "enabled": True,
            "reason": (
                "Resume will create a new attempt from the latest valid resumable "
                f"checkpoint using the immutable plan and pinned adapter {pinned_label}."
            ),
        }
    elif pinned_stage is None:
        resume = disabled(
            "The immutable pinned execution evidence is unavailable: "
            f"{pinned_stage_error}."
        )
    else:
        resume = {
            "enabled": True,
            "reason": (
                (f"Checkpoint resume is unsupported by pinned adapter {pinned_label}; "
                 if checkpoint is not None and not resume_argv
                 else "No usable resumable checkpoint is available; ")
                + "this action restarts training from the beginning in a new attempt "
                + f"using its exact initial pinned execution with adapter {pinned_label}."
            ),
        }
    return {"resume": resume, "rerun": rerun, "cancel": cancel}


class EvaluationRequest(BaseModel):
    run_id: str | None = None
    checkpoint_path: str | None = None
    suite_id: str
    environment: str | None = None
    tasks: list[str] = Field(default_factory=list)
    episodes_per_task: int = Field(default=20, ge=1, le=10000)
    seeds: list[int] = Field(default_factory=lambda: [42])
    parallelism: int = Field(
        default=1,
        ge=1,
        le=1,
        description=(
            "Parallel evaluation workers are not implemented by any registered evaluator; "
            "requests must use exactly one worker and are never silently serialized."
        ),
    )
    headless: bool = True
    auto_resume: bool = True
    max_attempts: int = Field(default=5, ge=1, le=100)
    gateway: str = "auto"
    resources: ResourceSpec | None = None
    argv: list[str] = Field(default_factory=list)
    resume_argv: list[str] = Field(default_factory=list)

    @field_validator("argv", "resume_argv")
    @classmethod
    def validate_structured_argv(cls, value: list[str]) -> list[str]:
        if any(not item or any(character in item for character in ("\x00", "\n", "\r")) for item in value):
            raise ValueError("argv values must be non-empty single-line strings")
        return value


class EvaluationTargetValidationRequest(BaseModel):
    run_id: str | None = None
    checkpoint_path: str | None = None
    suite_id: str | None = None
    environment: str | None = None
    tasks: list[str] = Field(default_factory=list)
    episodes_per_task: int = Field(default=20, ge=1, le=10000)
    seeds: list[int] = Field(default_factory=lambda: [42])
    parallelism: int = Field(
        default=1,
        ge=1,
        le=1,
        description=(
            "Parallel evaluation workers are not implemented by any registered evaluator; "
            "requests must use exactly one worker and are never silently serialized."
        ),
    )
    headless: bool = True
    gateway: str = "auto"
    resources: ResourceSpec | None = None
    argv: list[str] = Field(default_factory=list)
    resume_argv: list[str] = Field(default_factory=list)

    @field_validator("argv", "resume_argv")
    @classmethod
    def validate_structured_argv(cls, value: list[str]) -> list[str]:
        if any(
            not item or any(character in item for character in ("\x00", "\n", "\r"))
            for item in value
        ):
            raise ValueError("argv values must be non-empty single-line strings")
        return value


class AdapterCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    manifest: dict[str, Any]
    description: str = ""
    repository_url: str | None = None
    created_by: str | None = None
    change_note: str | None = Field(default=None, max_length=2000)


class AdapterEditRequest(BaseModel):
    manifest: dict[str, Any]
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = None
    repository_url: str | None = None
    created_by: str | None = None
    change_note: str | None = Field(default=None, max_length=2000)
    expected_latest_version: int | None = Field(default=None, ge=1)


class AdapterCloneRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    version_number: int | None = Field(default=None, ge=1)
    description: str | None = None
    created_by: str | None = None
    change_note: str | None = Field(default=None, max_length=2000)


class AdapterValidationRequest(BaseModel):
    repository_url: str | None = None
    source_revision: str = "main"
    project_subdirectory: str = "."
    gateway: str = "auto"
    version_number: int | None = Field(default=None, ge=1)
    created_by: str | None = None


class UnsavedAdapterValidationRequest(BaseModel):
    manifest: dict[str, Any]
    repository_url: str | None = None
    source_revision: str = "main"
    project_subdirectory: str = "."
    gateway: str = "auto"


def _json_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


_PROTOTYPE_LIKE_PATH_SEGMENTS = {"__proto__", "prototype", "constructor"}


def _native_config_path(key: str) -> tuple[str, ...]:
    suffix = key.removeprefix("config.")
    segments = tuple(suffix.split("."))
    if not suffix or any(
        not segment
        or not (segment[0].isalpha() or segment[0] == "_")
        or any(not (character.isalnum() or character in {"_", "-"}) for character in segment)
        or segment.casefold() in _PROTOTYPE_LIKE_PATH_SEGMENTS
        for segment in segments
    ):
        raise ValueError(f"invalid native config path: {key}")
    return segments


def _set_native_config_value(
    config: dict[str, Any], assigned: list[tuple[str, ...]], key: str, value: Any
) -> None:
    path = _native_config_path(key)
    if any(
        path[: len(existing)] == existing or existing[: len(path)] == path
        for existing in assigned
    ):
        raise ValueError(f"native config path collision: {key}")
    current = config
    for segment in path[:-1]:
        nested = current.get(segment)
        if nested is None:
            nested = {}
            current[segment] = nested
        elif not isinstance(nested, dict):
            raise ValueError(f"native config path collision: {key}")
        current = nested
    if path[-1] in current:
        raise ValueError(f"native config path collision: {key}")
    current[path[-1]] = value
    assigned.append(path)


def _parse_native(lines: list[str]) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    config: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    argv: list[str] = []
    resume_argv: list[str] = []
    assigned_config_paths: list[tuple[str, ...]] = []
    for line in lines:
        key, separator, raw = line.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"native override must use key=value: {line}")
        key = key.strip()
        value = _json_scalar(raw.strip())
        if key == "argv":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError("argv must be a JSON list of strings")
            argv = value
        elif key == "resume_argv":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError("resume_argv must be a JSON list of strings")
            resume_argv = value
        elif key.startswith("config."):
            _set_native_config_value(config, assigned_config_paths, key, value)
        else:
            if any(
                segment.casefold() in _PROTOTYPE_LIKE_PATH_SEGMENTS
                for segment in key.split(".")
            ):
                raise ValueError(f"invalid native override key: {key}")
            overrides[key] = value
    return config, overrides, argv, resume_argv


def _sweep_from_frontend(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {"strategy": "grid", "axes": {}, "seeds": [42], "max_parallel": 2}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"sweep definition must be JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise ValueError("sweep definition must be a JSON object")
    seeds = parsed.pop("seed", parsed.pop("seeds", [42]))
    if not isinstance(seeds, list):
        seeds = [seeds]
    axes: dict[str, list[Any]] = {}
    canonical_roots = {"train", "resources", "runtime", "tracking", "native", "data"}
    for key, values in parsed.items():
        if not isinstance(values, list):
            raise ValueError(f"sweep axis {key} must be a JSON list")
        path = key if key.split(".", 1)[0] in canonical_roots else f"native.overrides.{key}"
        axes[path] = values
    return {
        "strategy": "grid",
        "axes": axes,
        "seeds": [int(seed) for seed in seeds],
        "max_parallel": 2,
        "confirmation_threshold": 20,
    }


class DataResourceCreateRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=128)
    namespace: str = Field(min_length=1, max_length=255)
    name: str = Field(min_length=1, max_length=255)
    kind: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    description: str = Field(default="", max_length=4096)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", "namespace", "name", "kind")
    @classmethod
    def strip_identity(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("data resource identity values cannot be blank")
        return stripped


class DataResourceEditRequest(BaseModel):
    description: str | None = Field(default=None, max_length=4096)
    metadata: dict[str, Any] | None = None
    archived: bool | None = None


class DataVersionCreateRequest(BaseModel):
    revision: str = Field(min_length=1, max_length=512)
    format: str = Field(min_length=1, max_length=128)
    path: str = Field(min_length=1, max_length=4096)
    source_uri: str | None = Field(default=None, max_length=4096)
    manifest_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    status: str = Field(default="READY", min_length=1, max_length=64)
    size_bytes: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("revision", "format", "path", "status")
    @classmethod
    def strip_version_fields(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("data version values cannot be blank")
        return stripped

    @field_validator("status")
    @classmethod
    def canonical_status(cls, value: str) -> str:
        normalized = value.upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", normalized):
            raise ValueError("status must be an uppercase identifier")
        return normalized


class HuggingFaceImportRequest(BaseModel):
    revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    subset: str = Field(min_length=1, max_length=512)
    format: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    role: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    bundle_name: str = Field(min_length=1, max_length=255)
    bundle_version: str = Field(min_length=1, max_length=255)
    gateway: str = "auto"
    queue: str = "overcap"

    @field_validator("revision")
    @classmethod
    def normalize_revision(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("subset")
    @classmethod
    def validate_subset(cls, value: str) -> str:
        value = value.strip().strip("/")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value):
            raise ValueError("subset must be a repository-relative directory without globs")
        if any(part in {"", ".", ".."} for part in PurePosixPath(value).parts):
            raise ValueError("subset contains an unsafe path component")
        return value

    @field_validator("gateway")
    @classmethod
    def validate_gateway(cls, value: str) -> str:
        if value != "auto" and value not in CLUSTER.gateways:
            raise ValueError(f"gateway must be auto or one of: {', '.join(CLUSTER.gateways)}")
        return value

    @field_validator("queue")
    @classmethod
    def validate_queue(cls, value: str) -> str:
        if value not in CLUSTER.queues:
            raise ValueError(f"queue must be one of: {', '.join(CLUSTER.queues)}")
        return value


class DataDerivationInputRequest(BaseModel):
    version_id: str = Field(min_length=1)
    role: str = Field(default="input", min_length=1, max_length=128)
    position: int | None = Field(default=None, ge=0)


class DataDerivationCreateRequest(BaseModel):
    output_version_id: str = Field(min_length=1)
    inputs: list[DataDerivationInputRequest] = Field(min_length=1)
    converter_repository: str = Field(min_length=1, max_length=4096)
    converter_commit: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    converter_config: dict[str, Any] = Field(default_factory=dict)
    runtime_lock_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")


class DataBundleAssignmentRequest(BaseModel):
    role: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    version_id: str = Field(min_length=1)
    position: int | None = Field(default=None, ge=0)
    mount_path: str | None = Field(default=None, max_length=4096)
    required: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class DataBundleCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    version: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=4096)
    assignments: list[DataBundleAssignmentRequest] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineService:
    def __init__(
        self,
        database: Database | None = None,
        cluster: ClusterClient | None = None,
        *,
        credential_store: CredentialStore | None = None,
        session_credentials: SessionCredentialStore | None = None,
    ) -> None:
        self.database = database or Database()
        self.cluster = cluster or ClusterClient()
        self.credential_store = credential_store or KeyringCredentialStore()
        self.credentials = session_credentials or SESSION_CREDENTIALS
        self._credential_restore_lock = threading.Lock()
        self._credentials_restored = False
        self._evaluator_runtime_readiness_lock = threading.Lock()
        self._evaluator_runtime_readiness_cache: dict[
            tuple[str, str, str, str], tuple[float, dict[str, Any], list[str]]
        ] = {}
        self.source_discovery = SourceDiscovery(self.cluster)
        self.source_metadata = SourceMetadataStore(self.database)
        self._reconcile_lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seed_registries()

    def _seed_registries(self) -> None:
        for manifest in builtin_adapter_manifests():
            self.database.upsert_seed_adapter(
                seed_key=manifest.slug,
                name=manifest.display_name,
                manifest=canonical_adapter_manifest(manifest),
                description=manifest.description,
                repository_url=manifest.default_repository,
            )
        # Old database migrations created builtin:* compatibility records whose
        # payload predates AdapterManifest. Preserve them for history, but never
        # expose an invalid manifest as a runnable choice for a new experiment.
        for record in self.database.list_adapter_registry(include_archived=False):
            if not str(record.get("seed_key") or "").startswith("builtin:"):
                continue
            try:
                version = self._selected_version(record)
                AdapterManifest.model_validate(version.get("manifest"))
            except (TypeError, ValueError):
                self.database.archive_adapter(str(record["id"]))
        for suite in get_evaluation_catalog():
            suite_config = suite.model_dump(
                mode="json", exclude={"catalog_path", "current"}
            )
            self.database.register_evaluation_suite(
                evaluator_adapter=suite.evaluator,
                evaluator_version="1",
                name=suite.suite,
                suite_version=suite.version,
                description=suite.label,
                catalog_path=suite.catalog_path,
                config=suite_config,
                enabled=suite.current,
            )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="skynet-reconciler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        while not self._stop.wait(15):
            try:
                self.reconcile()
            except Exception:
                # Reconciliation errors are exposed through explicit API calls and events;
                # the daemon must remain alive for the next cluster recovery.
                continue

    @staticmethod
    def _selected_version(record: Mapping[str, Any]) -> Mapping[str, Any]:
        version = record.get("selected_version") or record.get("latest_version")
        if not isinstance(version, Mapping):
            raise ValueError("adapter record has no selected version")
        return version

    def _adapter_selection(
        self, adapter_key: str, version_number: int | None = None
    ) -> tuple[dict[str, Any], Mapping[str, Any], AdapterManifest]:
        requested = adapter_key.strip()
        matches: list[dict[str, Any]] = []
        for record in self.database.list_adapter_registry(include_archived=False):
            version = self._selected_version(record)
            raw_manifest = version.get("manifest")
            if not isinstance(raw_manifest, Mapping):
                continue
            direct_keys = {
                str(record.get("id") or ""),
                str(record.get("seed_key") or ""),
            }
            try:
                manifest = AdapterManifest.model_validate(raw_manifest)
            except ValueError as error:
                if requested in direct_keys:
                    raise ValueError(
                        f"adapter {requested} has an invalid latest manifest: {error}"
                    ) from error
                continue
            keys = {
                *direct_keys,
                manifest.slug,
                *manifest.aliases,
            }
            if requested in keys:
                matches.append(record)
        if not matches:
            raise ValueError(f"unknown or archived adapter: {requested}")
        if len(matches) > 1:
            raise ValueError(f"adapter key is ambiguous; use its registry ID: {requested}")
        detail = self.database.get_adapter(
            str(matches[0]["id"]), version_number=version_number, include_versions=True
        )
        if not detail:
            raise ValueError(f"adapter version was not found: {requested}@{version_number}")
        version = self._selected_version(detail)
        manifest = AdapterManifest.model_validate(version["manifest"])
        return detail, version, manifest

    def _snapshot_adapter(
        self, source: dict[str, Any], requested_key: str
    ) -> tuple[dict[str, Any], AdapterManifest, str | None]:
        existing_manifest = source.get("adapter_manifest")
        database = getattr(self, "database", None)
        if isinstance(existing_manifest, Mapping) and database is None:
            raw_snapshot = copy.deepcopy(dict(existing_manifest))
            manifest = AdapterManifest.model_validate(raw_snapshot)
            digest = canonical_sha256(raw_snapshot)
            expected = source.get("adapter_manifest_sha256")
            if expected and expected != digest:
                raise ValueError("adapter manifest snapshot hash does not match")
            source.update(
                {
                    "adapter": manifest.slug,
                    "adapter_version": int(source.get("adapter_version") or manifest.capabilities.version),
                    "adapter_manifest": raw_snapshot,
                    "adapter_manifest_sha256": digest,
                }
            )
            return source, manifest, source.get("adapter_id")

        version_number = source.get("adapter_version")
        requested_version_id = source.get("adapter_version_id")
        requested_adapter_id = source.get("adapter_id")
        record: dict[str, Any]
        version: dict[str, Any]
        manifest: AdapterManifest
        if database is not None and requested_adapter_id:
            registered = database.get_adapter(
                str(requested_adapter_id), include_versions=True
            )
            if not registered:
                raise ValueError(
                    f"registered adapter was not found: {requested_adapter_id}"
                )
            record = registered
            versions = list(record.get("versions") or [])
            if requested_version_id:
                selected = next(
                    (
                        item
                        for item in versions
                        if str(item.get("id")) == str(requested_version_id)
                    ),
                    None,
                )
            elif version_number is not None:
                selected = next(
                    (
                        item
                        for item in versions
                        if int(item.get("version_number") or 0) == int(version_number)
                    ),
                    None,
                )
            else:
                selected = self._selected_version(record)
            if selected is None:
                requested_version = requested_version_id or version_number or "selected"
                raise ValueError(
                    f"registered adapter version was not found: "
                    f"{requested_adapter_id}@{requested_version}"
                )
            version = selected
            manifest = AdapterManifest.model_validate(version["manifest"])
        else:
            record, version, manifest = self._adapter_selection(
                requested_key,
                int(version_number) if version_number is not None else None,
            )
            if requested_version_id and str(version.get("id")) != str(requested_version_id):
                selected = next(
                    (
                        item
                        for item in record.get("versions", [])
                        if str(item.get("id")) == str(requested_version_id)
                    ),
                    None,
                )
                if selected is None:
                    raise ValueError(
                        f"registered adapter version was not found: "
                        f"{record['id']}@{requested_version_id}"
                    )
                version = selected
                manifest = AdapterManifest.model_validate(version["manifest"])

        if version_number is not None and int(version["version_number"]) != int(version_number):
            raise ValueError("adapter version ID and version number do not match")
        registered_snapshot = copy.deepcopy(dict(version["manifest"]))
        manifest = AdapterManifest.model_validate(registered_snapshot)
        digest = canonical_sha256(registered_snapshot)
        stored_digest = version.get("manifest_sha256")
        if stored_digest and stored_digest != digest:
            raise ValueError("registered adapter manifest hash does not match its JSON")
        if isinstance(existing_manifest, Mapping):
            raw_snapshot = copy.deepcopy(dict(existing_manifest))
            AdapterManifest.model_validate(raw_snapshot)
            supplied_digest = canonical_sha256(raw_snapshot)
            expected = source.get("adapter_manifest_sha256")
            if expected and expected != supplied_digest:
                raise ValueError("adapter manifest snapshot hash does not match")
            if supplied_digest != digest:
                raise ValueError(
                    "adapter manifest snapshot does not match the immutable "
                    "registered adapter version"
                )
        source.update(
            {
                "adapter": manifest.slug,
                "adapter_id": record["id"],
                "adapter_version_id": version["id"],
                "adapter_version": int(version["version_number"]),
                "adapter_manifest": registered_snapshot,
                "adapter_manifest_sha256": digest,
            }
        )
        return source, manifest, str(record["id"])

    def run_manual_actions(
        self, run: Mapping[str, Any]
    ) -> dict[str, dict[str, bool | str]]:
        return manual_run_actions(run)

    def evaluation_manual_actions(
        self,
        evaluation: Mapping[str, Any],
        run: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, bool | str]]:
        parent = run or self.database.get_run(str(evaluation.get("run_id") or ""))
        if not parent:
            return {
                "cancel": {
                    "enabled": False,
                    "reason": "The parent Training Run is unavailable.",
                }
            }
        stage = next(
            (
                item for item in list(parent.get("stages") or [])
                if item.get("id") == evaluation.get("stage_id")
                and str(item.get("stage_type") or "").upper() == "EVALUATE"
            ),
            None,
        )
        attempts = [
            item for item in list(parent.get("attempts") or [])
            if item.get("stage_id") == evaluation.get("stage_id")
        ]
        return {"cancel": manual_cancel_action(stage, attempts)}

    @staticmethod
    def _binding_value_for_canonical(legacy_value: Any, binding: Any) -> Any:
        if not binding.value_map:
            return legacy_value
        if isinstance(legacy_value, bool):
            serialized = "true" if legacy_value else "false"
        elif isinstance(legacy_value, str):
            serialized = legacy_value
        else:
            serialized = canonical_json(legacy_value)
        if serialized in binding.value_map:
            return serialized
        candidates = [
            canonical
            for canonical, native in binding.value_map.items()
            if native == serialized
        ]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f"legacy value {serialized!r} is not represented by the adapter's canonical value map"
        )

    @classmethod
    def _migrate_bound_native_overrides(
        cls, payload: dict[str, Any], manifest: AdapterManifest
    ) -> list[dict[str, Any]]:
        """Move legacy same-name native overrides into manifest-bound train fields."""
        native = payload.get("native")
        overrides = native.get("overrides") if isinstance(native, dict) else None
        if not isinstance(overrides, dict):
            return []
        migrations: list[dict[str, Any]] = []
        for canonical_path, binding in sorted(manifest.train.parameter_flags.items()):
            if not canonical_path.startswith("train."):
                continue
            legacy_key = canonical_path.rsplit(".", 1)[-1]
            if legacy_key not in overrides:
                continue
            legacy_value = overrides[legacy_key]
            canonical_value = cls._binding_value_for_canonical(legacy_value, binding)
            present, existing = cls._manifest_input_lookup(payload, canonical_path)
            if present and existing is not None and existing != canonical_value:
                raise ValueError(
                    f"clean retry cannot migrate native.overrides.{legacy_key}: "
                    f"it conflicts with {canonical_path}"
                )
            if not present or existing is None:
                parts = canonical_path.split(".")
                target = payload
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = copy.deepcopy(canonical_value)
            intent = payload.setdefault(
                "intent",
                {"explicit_parameters": explicit_train_parameter_paths(payload)},
            )
            explicit_parameters = intent.setdefault("explicit_parameters", [])
            if canonical_path not in explicit_parameters:
                explicit_parameters.append(canonical_path)
                explicit_parameters.sort()
            del overrides[legacy_key]
            migrations.append(
                {
                    "kind": "native_override_to_canonical_binding",
                    "from": f"native.overrides.{legacy_key}",
                    "to": canonical_path,
                    "input_value": copy.deepcopy(legacy_value),
                    "resolved_value": copy.deepcopy(canonical_value),
                }
            )
        return migrations

    def _prepare_clean_retry(
        self, run: Mapping[str, Any]
    ) -> tuple[ExperimentSpec, AdapterPlan, dict[str, Any]]:
        original = copy.deepcopy(dict(run["resolved_spec_json"]))
        payload = copy.deepcopy(original)
        source = dict(payload.get("source") or {})
        pinned_adapter = {
            "id": source.get("adapter_id"),
            "version_id": source.get("adapter_version_id"),
            "version": source.get("adapter_version"),
            "slug": source.get("adapter"),
            "manifest_sha256": source.get("adapter_manifest_sha256"),
        }
        slug = str(source.get("adapter") or "")
        if not slug:
            raise ValueError("clean retry requires the stored adapter slug")
        for key in (
            "adapter_id",
            "adapter_version_id",
            "adapter_version",
            "adapter_manifest",
            "adapter_manifest_sha256",
        ):
            source.pop(key, None)
        source["adapter"] = slug
        source, manifest, _ = self._snapshot_adapter(source, slug)
        payload["source"] = source
        transformations = self._migrate_bound_native_overrides(payload, manifest)
        native_config = payload.setdefault("native", {}).setdefault("config", {})
        removed_checkpoint_fields = [
            key
            for key in ("initial_checkpoint", "initial_checkpoint_mode")
            if key in native_config
        ]
        for key in removed_checkpoint_fields:
            native_config.pop(key, None)
        payload.setdefault("train", {}).setdefault("checkpoint", {})[
            "auto_resume"
        ] = False
        if removed_checkpoint_fields:
            transformations.append(
                {
                    "kind": "clean_retry_checkpoint_reset",
                    "removed": removed_checkpoint_fields,
                }
            )
        payload = self._apply_canonical_manifest_defaults(payload, manifest)
        spec = ExperimentSpec.model_validate(payload)
        plan = resolve_adapter_plan(spec)
        current_adapter = {
            "id": spec.source.adapter_id,
            "version_id": spec.source.adapter_version_id,
            "version": spec.source.adapter_version,
            "slug": spec.source.adapter,
            "manifest_sha256": spec.source.adapter_manifest_sha256,
        }
        provenance = {
            "policy": "clean_retry_current_adapter",
            "source": "variants.resolved_spec_json",
            "source_spec_sha256": canonical_sha256(original),
            "adapter_resolution": {
                "pinned": pinned_adapter,
                "current": current_adapter,
            },
            "checkpoint": None,
            "transformations": transformations,
        }
        return spec, plan, provenance

    @staticmethod
    def _attempt_execution_snapshot(
        spec: ExperimentSpec,
        plan: AdapterPlan,
        provenance: Mapping[str, Any],
        repository_inputs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_spec = spec.model_dump(mode="json", by_alias=True)
        resolved_plan = plan.model_dump(mode="json")
        selected_repository_inputs = copy.deepcopy(dict(repository_inputs or {}))
        common_hyperparameters = _resolve_effective_common_hyperparameters(
            resolved_spec,
            spec.source.adapter_manifest,
            selected_repository_inputs,
            resolved_spec_sha256=canonical_sha256(resolved_spec),
            manifest_sha256=spec.source.adapter_manifest_sha256,
        )
        return {
            "schema_version": "skynet.job-attempt/v1",
            "resolved_spec": resolved_spec,
            "resolved_spec_sha256": canonical_sha256(resolved_spec),
            "adapter": {
                "id": spec.source.adapter_id,
                "version_id": spec.source.adapter_version_id,
                "slug": spec.source.adapter,
                "version": spec.source.adapter_version,
                "manifest": spec.source.adapter_manifest,
                "manifest_sha256": spec.source.adapter_manifest_sha256,
            },
            "plan": resolved_plan,
            "plan_sha256": canonical_sha256(resolved_plan),
            "argv": list(plan.argv),
            "argv_sha256": canonical_sha256(plan.argv),
            "resume_argv": list(plan.resume_argv),
            "repository_inputs": {
                "schema_version": "skynet.repository-input-selections/v1",
                "selected": selected_repository_inputs,
            },
            "common_hyperparameters": common_hyperparameters,
            "migration_provenance": copy.deepcopy(dict(provenance)),
        }

    @staticmethod
    def _merge_missing(target: dict[str, Any], defaults: Mapping[str, Any]) -> None:
        for key, value in defaults.items():
            if value is None:
                continue
            if key not in target:
                target[key] = copy.deepcopy(value)
            elif isinstance(target[key], dict) and isinstance(value, Mapping):
                PipelineService._merge_missing(target[key], value)

    @staticmethod
    def _canonical_manifest_defaults(manifest: AdapterManifest) -> dict[str, Any]:
        defaults = manifest.defaults
        hyper = defaults.hyperparameters
        resources = defaults.resources
        checkpoint = defaults.checkpoint.model_dump(mode="json", exclude_none=True)
        train: dict[str, Any] = {
            key: value
            for key, value in {
                "learning_rate": hyper.learning_rate,
                "num_workers_per_rank": hyper.num_workers_per_rank,
                "max_steps": hyper.max_steps,
                "max_epochs": hyper.max_epochs,
                "seed": hyper.seed,
                "precision": hyper.precision,
            }.items()
            if value is not None
        }
        batch = {
            key: value
            for key, value in {
                "declared_semantics": hyper.batch_semantics,
                "value": hyper.batch_size,
                "gradient_accumulation_steps": hyper.gradient_accumulation_steps,
            }.items()
            if value is not None
        }
        if batch:
            train["batch"] = batch
        if hyper.values:
            train["hyperparameters"] = copy.deepcopy(hyper.values)
        if checkpoint:
            train["checkpoint"] = checkpoint

        resource_document: dict[str, Any] = {
            key: value
            for key, value in {
                "gateway": resources.gateway,
                "queue_policy": resources.queue_policy,
                "cpus_per_task": resources.cpus_per_task,
                "memory_gb": resources.memory_gb,
                "time_limit": resources.time_limit,
            }.items()
            if value is not None
        }
        if resources.queue_policy and resources.queue_policy != "auto":
            queue = CLUSTER.queue(resources.queue_policy)
            resource_document.update({"account": queue.account, "partition": queue.partition})
        node = {
            key: value
            for key, value in {"mode": resources.node_mode, "name": resources.node}.items()
            if value is not None
        }
        if node:
            resource_document["node"] = node
        gpu = {
            key: value
            for key, value in {
                "mode": resources.gpu_mode,
                "count": resources.gpu_count,
                "type": resources.gpu_type,
                "profile": resources.gpu_profile,
            }.items()
            if value is not None
        }
        if gpu:
            resource_document["gpu"] = gpu
        tracking = defaults.tracking.model_dump(
            mode="json", exclude_none=True, exclude={"enabled"}
        )
        if defaults.tracking.enabled is False:
            tracking["mlflow_tracking_uri"] = None
        return {
            "source": {
                "project_subdirectory": defaults.effective_project_subdirectory
            },
            "train": train,
            "resources": resource_document,
            "tracking": tracking,
            "evaluation": copy.deepcopy(defaults.evaluation),
        }

    @classmethod
    def _apply_canonical_manifest_defaults(
        cls, canonical: dict[str, Any], manifest: AdapterManifest
    ) -> dict[str, Any]:
        cls._merge_missing(canonical, cls._canonical_manifest_defaults(manifest))
        cls._apply_manifest_input_defaults(canonical, manifest)
        cls._validate_manifest_input_fields(canonical, manifest)
        return canonical

    @staticmethod
    def _manifest_input_lookup(
        document: Mapping[str, Any], path: str
    ) -> tuple[bool, Any]:
        if path.startswith("native.overrides."):
            native = document.get("native")
            overrides = native.get("overrides") if isinstance(native, Mapping) else None
            key = path.removeprefix("native.overrides.")
            if not isinstance(overrides, Mapping) or key not in overrides:
                return False, None
            return True, overrides[key]
        current: Any = document
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return False, None
            current = current[part]
        return True, current

    @staticmethod
    def _set_missing_canonical_path(
        document: dict[str, Any], path: str, value: Any
    ) -> None:
        if path.startswith("native.overrides."):
            native = document.setdefault("native", {})
            if not isinstance(native, dict):
                return
            overrides = native.setdefault("overrides", {})
            if not isinstance(overrides, dict):
                return
            overrides.setdefault(
                path.removeprefix("native.overrides."), copy.deepcopy(value)
            )
            return
        parts = path.split(".")
        current: dict[str, Any] = document
        for part in parts[:-1]:
            existing = current.get(part)
            if existing is None:
                nested: dict[str, Any] = {}
                current[part] = nested
                current = nested
            elif isinstance(existing, dict):
                current = existing
            else:
                return
        current.setdefault(parts[-1], copy.deepcopy(value))

    @classmethod
    def _apply_manifest_input_defaults(
        cls, canonical: dict[str, Any], manifest: AdapterManifest
    ) -> dict[str, Any]:
        for field in manifest.train.input_fields:
            if field.default is not None:
                cls._set_missing_canonical_path(canonical, field.path, field.default)
        return canonical

    @classmethod
    def _apply_manifest_data_bindings(
        cls, canonical: dict[str, Any], manifest: AdapterManifest
    ) -> dict[str, Any]:
        bundle = (canonical.get("data") or {}).get("bundle")
        assignments = bundle.get("assignments") if isinstance(bundle, Mapping) else None
        if not isinstance(assignments, list):
            return canonical
        if not any(field.data_binding for field in manifest.train.input_fields):
            raise ValueError(
                "Dataset bundle selection is unsupported by this adapter version: "
                "it declares no training dataset binding. Remove the bundle and use "
                "the repository's dataset configuration, or select an adapter with "
                "a compatible training_data binding."
            )
        consumed = {(field.data_binding.role, field.data_binding.position)
                    for field in manifest.train.input_fields if field.data_binding}
        bound_roles = {role for role, _ in consumed}
        identities = [(str(item.get("role") or ""), int(item.get("position") or 0))
                      for item in assignments if isinstance(item, Mapping)]
        if len(identities) != len(set(identities)):
            raise ValueError("Dataset bundle contains duplicate role positions; choose an unambiguous bundle")
        for role, position in identities:
            if role in bound_roles and (role, position) not in consumed:
                raise ValueError(f"This adapter cannot consume {role} at position {position}. Choose a bundle with only the declared dataset inputs.")
        for field in manifest.train.input_fields:
            binding = field.data_binding
            if binding is None:
                continue
            present, value = cls._manifest_input_lookup(canonical, field.path)
            matching = [
                item for item in assignments
                if isinstance(item, Mapping) and str(item.get("role") or "") == binding.role
            ]
            matching.sort(key=lambda item: int(item.get("position") or 0))
            assignment = next(
                (item for item in matching if int(item.get("position") or 0) == binding.position),
                None,
            )
            if assignment is None:
                raise ValueError(
                    f"{field.path}: selected data bundle does not provide role "
                    f"{binding.role} at position {binding.position}"
                )
            version = assignment.get("version")
            if not isinstance(version, Mapping):
                raise ValueError(
                    f"{field.path}: selected data bundle role {binding.role} has no version snapshot"
                )
            if str(version.get("status") or "").upper() != "READY":
                raise ValueError(f"Selected {binding.role} is {version.get('status') or 'unverified'}, not ready on the cluster. Complete its import or transfer before training.")
            if (version.get("metadata") or {}).get("storage_location") == "workstation":
                raise ValueError(
                    "This dataset is stored on the collection workstation. "
                    "Transfer it to the training cluster and register that copy before submitting training."
                )
            if binding.formats and str(version.get("format") or "").casefold() not in {
                item.casefold() for item in binding.formats
            }:
                actual_format = str(version.get("format") or "undeclared")
                raise ValueError(
                    f"{field.path}: selected {binding.role} format {actual_format} is incompatible; "
                    f"accepted formats: {', '.join(binding.formats)}"
                )
            bound_value = (
                version.get("path")
                if binding.value_path == "version.path"
                else assignment.get("mount_path")
            )
            if isinstance(bound_value, str) and bound_value:
                if present and value not in (None, "", bound_value):
                    raise ValueError(
                        f"{field.path} conflicts with the selected dataset bundle. "
                        "Clear the explicit dataset path to use the bundle, or "
                        "remove the bundle to use the explicit path."
                    )
                if present and value in (None, ""):
                    if field.path.startswith("native.overrides."):
                        canonical["native"]["overrides"].pop(field.path.removeprefix("native.overrides."))
                    else:
                        parent = canonical
                        parts = field.path.split(".")
                        for part in parts[:-1]:
                            parent = parent[part]
                        parent.pop(parts[-1])
                cls._set_missing_canonical_path(canonical, field.path, bound_value)
            else:
                raise ValueError(
                    f"{field.path}: selected data bundle role {binding.role} does not provide "
                    f"{binding.value_path}"
                )
        return canonical

    @classmethod
    def _validate_manifest_input_fields(
        cls, canonical: Mapping[str, Any], manifest: AdapterManifest
    ) -> None:
        for field in manifest.train.input_fields:
            present, value = cls._manifest_input_lookup(canonical, field.path)
            missing = not present or value is None or value == "" or value == []
            if missing:
                if field.required:
                    raise ValueError(f"{field.path}: required adapter input is missing")
                continue
            if not field._matches_kind(field.kind, value):
                raise ValueError(
                    f"{field.path}: expected {field.kind}, got {type(value).__name__}"
                )
            if field.choices:
                allowed = {canonical_sha256(choice) for choice in field.choices}
                if canonical_sha256(value) not in allowed:
                    raise ValueError(
                        f"{field.path}: value is not one of the declared choices"
                    )

    @staticmethod
    def _repository_inspection_cache_parameters(
        source: Mapping[str, Any], manifest: AdapterManifest
    ) -> dict[str, Any]:
        discovery_fields = [
            {
                "path": field.path,
                "choice_source": field.choice_source.model_dump(mode="json"),
            }
            for field in manifest.train.input_fields
            if field.choice_source is not None
        ]
        return {
            "revision": str(source.get("revision") or "").lower(),
            "project_subdirectory": str(
                source.get("project_subdirectory") or "."
            ),
            "input_discovery_sha256": hashlib.sha256(
                canonical_json(discovery_fields).encode("utf-8")
            ).hexdigest(),
        }

    def _cached_repository_inspection(
        self, source: Mapping[str, Any], manifest: AdapterManifest
    ) -> dict[str, Any] | None:
        repository = str(source.get("repository") or "")
        parameters = self._repository_inspection_cache_parameters(source, manifest)
        entry = self.source_metadata.get("inspection", repository, parameters)
        if entry is None:
            legacy_parameters = {
                "revision": parameters["revision"],
                "project_subdirectory": parameters["project_subdirectory"],
                "adapter_id": str(source.get("adapter_id") or ""),
                "adapter_version_id": str(source.get("adapter_version_id") or ""),
                "adapter_manifest_sha256": str(
                    source.get("adapter_manifest_sha256") or ""
                ),
            }
            legacy_entry = self.source_metadata.get(
                "inspection", repository, legacy_parameters
            )
            if legacy_entry is not None:
                entry = self.source_metadata.put(
                    "inspection", repository, parameters, legacy_entry["payload"]
                )
        payload = entry.get("payload") if entry is not None else None
        if not isinstance(payload, Mapping):
            return None
        expected_commit = parameters["revision"]
        cached_commit = str(payload.get("commit") or "").lower()
        if not FULL_COMMIT_RE.fullmatch(cached_commit):
            raise ValueError(
                "cached repository inspection does not identify a pinned commit; "
                "explicitly re-inspect the selected source revision"
            )
        if cached_commit != expected_commit:
            raise ValueError(
                "cached repository inspection belongs to a different commit; "
                "explicitly re-inspect the selected source revision"
            )
        return copy.deepcopy(dict(payload))

    def _repository_input_options(
        self,
        source: Mapping[str, Any],
        manifest: AdapterManifest,
        gateway: str,
    ) -> dict[str, Any]:
        if not any(
            field.choice_source is not None for field in manifest.train.input_fields
        ):
            return {}
        payload = self._cached_repository_inspection(source, manifest)
        options = payload.get("input_options") if payload is not None else None
        if not isinstance(options, Mapping):
            raise ValueError(
                "pinned repository input metadata is not cached; explicitly inspect "
                "the selected source revision before preview or submission"
            )
        return copy.deepcopy(dict(options))

    def _validate_repository_input_choices(
        self,
        document: Mapping[str, Any],
        source: Mapping[str, Any],
        manifest: AdapterManifest,
        gateway: str,
    ) -> dict[str, Any]:
        options = self._repository_input_options(source, manifest, gateway)
        for field in manifest.train.input_fields:
            if field.choice_source is None:
                continue
            resolved = options.get(field.path)
            if not isinstance(resolved, Mapping):
                raise ValueError(f"{field.path}: repository choices were not resolved")
            warnings = [str(item) for item in resolved.get("warnings") or []]
            if not resolved.get("complete"):
                detail = "; ".join(warnings) or "static registry resolution was incomplete"
                raise ValueError(
                    f"{field.path}: repository choice discovery is incomplete at "
                    f"{source.get('revision')}: {detail}"
                )
            present, value = self._manifest_input_lookup(document, field.path)
            if not present or value is None or value == "":
                continue
            choices = list(resolved.get("choices") or [])
            if value not in choices and not getattr(
                field.choice_source, "allow_custom", False
            ):
                raise ValueError(
                    f"{field.path}: {value!r} is not registered at source commit "
                    f"{source.get('revision')}"
                )
        return options

    @staticmethod
    def _reject_repository_choice_sweeps(
        document: Mapping[str, Any], manifest: AdapterManifest
    ) -> None:
        choice_paths = {
            field.path
            for field in manifest.train.input_fields
            if field.choice_source is not None
        }
        if not choice_paths:
            return
        sweep = document.get("sweep")
        if not isinstance(sweep, Mapping):
            return
        parameters = sweep.get("parameters")
        if not isinstance(parameters, Mapping):
            return
        for sweep_path in parameters:
            normalized_path = str(sweep_path)
            for choice_path in choice_paths:
                if (
                    normalized_path == choice_path
                    or normalized_path.startswith(f"{choice_path}.")
                    or choice_path.startswith(f"{normalized_path}.")
                ):
                    raise ValueError(
                        f"{choice_path}: repository-discovered choice fields cannot "
                        f"be swept (conflicting sweep path: {normalized_path})"
                    )

    @staticmethod
    def _payload_with_manifest_defaults(
        payload: Mapping[str, Any], manifest: AdapterManifest
    ) -> dict[str, Any]:
        document = copy.deepcopy(dict(payload))
        defaults = manifest.defaults
        hyper = document.setdefault("hyperparameters", {})
        for field, value in {
            "learning_rate": defaults.hyperparameters.learning_rate,
            "batch_semantics": defaults.hyperparameters.batch_semantics,
            "batch_size": defaults.hyperparameters.batch_size,
            "gradient_accumulation": defaults.hyperparameters.gradient_accumulation_steps,
            "num_workers": defaults.hyperparameters.num_workers_per_rank,
            "max_steps": defaults.hyperparameters.max_steps,
            "max_epochs": defaults.hyperparameters.max_epochs,
            "seed": defaults.hyperparameters.seed,
            "precision": defaults.hyperparameters.precision,
        }.items():
            if value is not None and (
                field not in hyper or _frontend_parameter_is_unset(hyper[field])
            ):
                hyper[field] = value
        for key, value in defaults.hyperparameters.values.items():
            hyper.setdefault(key, copy.deepcopy(value))

        resources = document.setdefault("resources", {})
        for field, value in {
            "gateway": defaults.resources.gateway,
            "queue_policy": defaults.resources.queue_policy,
            "node_mode": defaults.resources.node_mode,
            "node": defaults.resources.node,
            "gpu_mode": defaults.resources.gpu_mode,
            "gpus_per_node": defaults.resources.gpu_count,
            "gpu_type": defaults.resources.gpu_type,
            "gpu_profile": defaults.resources.gpu_profile,
            "cpus_per_task": defaults.resources.cpus_per_task,
            "memory_gb": defaults.resources.memory_gb,
            "time_limit": defaults.resources.time_limit,
        }.items():
            if value is not None:
                resources.setdefault(field, value)

        checkpoint = defaults.checkpoint
        for field, value in {
            "checkpoint_save_steps": checkpoint.save_every_steps,
            "checkpoint_warning_seconds": checkpoint.save_before_timeout_seconds,
            "checkpoint_keep_last": checkpoint.keep_last,
            "auto_resume": checkpoint.auto_resume,
            "max_attempts": checkpoint.max_attempts,
            "checkpoint_final_selector": checkpoint.final_selector,
            "remove_training_state_after_success": checkpoint.remove_training_state_after_success,
        }.items():
            if value is not None:
                document.setdefault(field, value)

        tracking = document.setdefault("tracking", {})
        for field, value in {
            "enabled": defaults.tracking.enabled,
            "uri": defaults.tracking.mlflow_tracking_uri,
            "experiment": defaults.tracking.mlflow_experiment,
            "native_tracking": defaults.tracking.native_tracking,
            "tags": defaults.tracking.tags or None,
            "offline_spool": defaults.tracking.offline_spool,
        }.items():
            if value is not None:
                tracking.setdefault(field, copy.deepcopy(value))
        if "evaluation" not in document and defaults.evaluation:
            document["evaluation"] = {"canonical_specs": copy.deepcopy(defaults.evaluation)}
        return document

    def _resolve_source_and_runtime(
        self,
        source: dict[str, Any],
        runtime_input: dict[str, Any],
        manifest: AdapterManifest,
        gateway: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        repository = self.source_discovery.repository_url(
            str(source.get("repository") or manifest.default_repository or "")
        )
        if not repository:
            raise ValueError("a source repository is required")
        requested_revision = str(source.get("revision") or "main")
        project_subdirectory = self.source_discovery.project_subdirectory(
            str(source.get("project_subdirectory") or ".")
        )
        if not FULL_COMMIT_RE.fullmatch(requested_revision):
            raise ValueError(
                f"source revision {requested_revision!r} is unresolved; preview and "
                "submission require a pinned 40-character commit. Inspect and select "
                "the source revision first"
            )
        revision = requested_revision.lower()
        source["repository"] = repository
        source["revision"] = revision
        source["project_subdirectory"] = project_subdirectory
        profile_id = str(runtime_input.get("profile_id") or "").strip() or None
        configured_profile = CLUSTER.runtime_profile(profile_id) if profile_id else None
        requested_backend = str(
            runtime_input.get("backend") or runtime_input.get("type") or "auto"
        ).strip().lower()
        requested_backend = {
            "container": "apptainer",
            "adapter-default": "auto",
        }.get(requested_backend, requested_backend)
        supported = {"auto", "uv", "conda", "apptainer", "existing"}
        if requested_backend not in supported:
            raise ValueError(f"unsupported runtime backend: {requested_backend}")
        if configured_profile is not None:
            if requested_backend not in {"auto", configured_profile.backend}:
                raise ValueError(
                    f"runtime profile {profile_id} uses {configured_profile.backend}, not {requested_backend}"
                )
            requested_backend = configured_profile.backend
        allowed = set(
            manifest.runtime.model_dump(mode="json").get("allowed_backends")
            or {"uv", "conda", "apptainer", "existing"}
        )
        if requested_backend != "auto":
            if requested_backend not in allowed:
                raise ValueError(
                    f"adapter does not allow {requested_backend}; choose one of {sorted(allowed)}"
                )
            runtime = copy.deepcopy(runtime_input)
            runtime.pop("type", None)
            runtime["backend"] = requested_backend
            if configured_profile is not None:
                assert profile_id is not None
                snapshot = CLUSTER.runtime_profile_snapshot(profile_id)
                snapshot_sha256 = content_sha256(snapshot)
                request_environment = dict(runtime.get("environment") or {})
                for field in (
                    "environment_path",
                    "container_image",
                    "lock_file",
                    "uv_executable",
                    "bootstrap_uv",
                ):
                    runtime.pop(field, None)
                profile_document = configured_profile.model_dump(mode="json")
                for field in (
                    "environment_path",
                    "container_image",
                    "lock_file",
                    "uv_executable",
                    "bootstrap_uv",
                ):
                    value = profile_document.get(field)
                    if value is not None:
                        runtime[field] = value
                runtime["environment"] = {
                    **configured_profile.environment,
                    **request_environment,
                }
                runtime.update(
                    {
                        "profile": profile_id,
                        "profile_id": profile_id,
                        "profile_snapshot": snapshot,
                        "profile_snapshot_sha256": snapshot_sha256,
                    }
                )
            else:
                runtime.setdefault("profile", "default")
            if requested_backend != "uv":
                runtime.setdefault("bootstrap_uv", False)
            if not runtime.get("resolution"):
                resolution = {
                    "schema_version": 1,
                    "mode": "operator_profile" if configured_profile is not None else "manual",
                    "requested_backend": requested_backend,
                    "selected_backend": requested_backend,
                    "selected_candidate": (
                        {
                            "profile_id": profile_id,
                            "profile_snapshot_sha256": runtime.get("profile_snapshot_sha256"),
                            "resolved_location": configured_profile.resolved_location,
                            "versions": configured_profile.versions,
                            "status": "configured",
                        }
                        if configured_profile is not None
                        else None
                    ),
                    "adapter_recommendation": manifest.runtime.recommended_backend,
                    "inspection": {
                        "schema_version": 1,
                        "repository": repository,
                        "requested_revision": requested_revision,
                        "commit": revision,
                        "project_subdirectory": project_subdirectory,
                        "skipped": True,
                        "reason": (
                            "operator runtime profile explicitly selected by the request"
                            if configured_profile is not None
                            else "runtime explicitly selected by the request"
                        ),
                    },
                }
                runtime["resolution"] = resolution
                runtime["resolution_sha256"] = content_sha256(resolution)
            return source, runtime
        inspection = self._cached_repository_inspection(source, manifest)
        if inspection is None:
            raise ValueError(
                "runtime backend is unresolved and no exact pinned repository "
                f"inspection is cached for {revision}; explicitly inspect that commit "
                "or select a supported runtime backend"
            )
        runtime = resolve_runtime(
            inspection,
            runtime_input,
            manifest.runtime.model_dump(mode="json"),
        )
        if environment := runtime_input.get("environment"):
            runtime["environment"] = dict(environment)
        return source, runtime

    def runtime_profiles(self, gateway: str = "auto", *, verify: bool = False) -> list[dict[str, Any]]:
        profiles = CLUSTER.public_runtime_profiles()
        if not verify:
            return profiles
        return [self._probe_runtime_profile(profile, gateway) for profile in profiles]

    @staticmethod
    def _validate_compute_runtime_attestation(
        attestation: Mapping[str, Any],
        expected: Mapping[str, Any],
    ) -> list[str]:
        errors: list[str] = []

        def version_matches(wanted: str, found: str) -> bool:
            return found == wanted or found.startswith(wanted + ".")

        def version_tuple(value: str) -> tuple[int, ...]:
            return tuple(int(part) for part in re.findall(r"\d+", value))

        if attestation.get("schema_version") != "skynet.runtime-readiness/v1":
            errors.append(
                "compute readiness attestation has an unsupported schema_version"
            )
        if attestation.get("profile_id") != expected.get("profile_id"):
            errors.append("compute readiness attestation belongs to a different runtime profile")
        if attestation.get("profile_snapshot_sha256") != expected.get(
            "profile_snapshot_sha256"
        ):
            errors.append(
                "compute readiness attestation is stale for the configured runtime profile; "
                "rerun the one-GPU evaluator readiness smoke"
            )
        expected_suite_sha = str(expected.get("suite_contract_sha256") or "")
        if expected_suite_sha and attestation.get(
            "suite_contract_sha256"
        ) != expected_suite_sha:
            errors.append(
                "compute readiness attestation is stale for the selected evaluation suite; "
                "rerun the one-GPU evaluator readiness smoke"
            )
        if attestation.get("ready") is not True:
            errors.append("compute readiness attestation does not declare ready=true")
        for item in attestation.get("errors") or []:
            errors.append(f"compute readiness smoke reported: {item}")

        execution = attestation.get("execution")
        if not isinstance(execution, Mapping):
            execution = {}
        if not str(execution.get("slurm_job_id") or "").isdigit():
            errors.append("compute readiness attestation has no valid Slurm job ID")
        if not str(execution.get("node") or "").strip():
            errors.append("compute readiness attestation has no compute node")

        actual = attestation.get("actual")
        if not isinstance(actual, Mapping):
            actual = {}
        found_python = str(actual.get("python") or "")
        wanted_python = str(expected.get("python_version") or "")
        if wanted_python and not version_matches(wanted_python, found_python):
            errors.append(
                f"compute readiness Python {found_python or 'missing'} does not match "
                f"{wanted_python}"
            )

        distributions = actual.get("distributions")
        if not isinstance(distributions, Mapping):
            distributions = {}
        for distribution, wanted in (expected.get("distributions") or {}).items():
            found = str(distributions.get(distribution) or "")
            if not found:
                errors.append(
                    f"compute readiness did not verify distribution {distribution}"
                )
            elif not version_matches(str(wanted), found):
                errors.append(
                    f"compute readiness distribution {distribution} {found} does not match "
                    f"{wanted}"
                )
        for distribution in expected.get("required_distributions") or []:
            if not str(distributions.get(distribution) or ""):
                errors.append(
                    f"compute readiness did not verify required distribution {distribution}"
                )

        imports = actual.get("imports")
        if not isinstance(imports, Mapping):
            imports = {}
        for module in expected.get("python_imports") or []:
            if imports.get(module) is not True:
                errors.append(
                    f"compute readiness did not verify required Python import {module}"
                )

        executables = actual.get("executables")
        if not isinstance(executables, Mapping):
            executables = {}
        for executable in expected.get("executables") or []:
            if not str(executables.get(executable) or "").startswith("/"):
                errors.append(
                    f"compute readiness did not verify required executable {executable}"
                )

        providers = actual.get("executable_providers")
        if not isinstance(providers, Mapping):
            providers = {}
        for provider in expected.get("executable_providers") or []:
            record = providers.get(provider["name"])
            if not isinstance(record, Mapping) or not str(record.get("path") or "").startswith(
                "/"
            ):
                errors.append(
                    "compute readiness did not verify executable provider "
                    f"{provider['name']}"
                )

        libraries = actual.get("shared_libraries")
        if not isinstance(libraries, Mapping):
            libraries = {}
        for library in expected.get("shared_libraries") or []:
            if libraries.get(library) is not True:
                errors.append(
                    f"compute readiness did not verify required shared library {library}"
                )

        registrations = actual.get("gym_registrations")
        if not isinstance(registrations, Mapping):
            registrations = {}
        for registration in expected.get("gym_registrations") or []:
            record = registrations.get(registration["module"])
            if not isinstance(record, Mapping) or record.get("module_imported") is not True:
                errors.append(
                    "compute readiness did not import Gym registration module "
                    f"{registration['module']}"
                )
                continue
            task_ids = record.get("ids")
            if not isinstance(task_ids, Mapping):
                task_ids = {}
            for task_id in registration.get("ids") or []:
                if task_ids.get(task_id) is not True:
                    errors.append(
                        f"compute readiness did not register required Gym task {task_id}"
                    )

        platform_actual = actual.get("platform")
        if not isinstance(platform_actual, Mapping):
            platform_actual = {}
        wanted_system = str(expected.get("platform_system") or "")
        found_system = str(platform_actual.get("system") or "")
        if wanted_system and found_system != wanted_system:
            errors.append(
                f"compute readiness operating system {found_system or 'missing'} does not "
                f"match {wanted_system}"
            )
        wanted_glibc = str(expected.get("glibc_minimum") or "")
        found_glibc = str(platform_actual.get("libc_version") or "")
        if wanted_glibc and (
            platform_actual.get("libc") != "glibc"
            or not found_glibc
            or version_tuple(found_glibc) < version_tuple(wanted_glibc)
        ):
            errors.append(
                f"compute readiness glibc {found_glibc or 'missing'} does not satisfy "
                f"minimum {wanted_glibc}"
            )

        if expected.get("requires_gpu"):
            gpu = actual.get("gpu")
            if not isinstance(gpu, Mapping):
                gpu = {}
            if gpu.get("cuda_available") is not True or int(gpu.get("device_count") or 0) < 1:
                errors.append(
                    "compute readiness did not verify an NVIDIA CUDA GPU on the Slurm node"
                )
        checks = actual.get("checks")
        if not isinstance(checks, Mapping):
            checks = {}
        for check in expected.get("required_checks") or []:
            if checks.get(check) is not True:
                errors.append(
                    f"compute readiness did not pass required evaluator check {check}"
                )
        if expected.get("pip_check"):
            pip_check = actual.get("pip_check")
            if not isinstance(pip_check, Mapping) or pip_check.get("ok") is not True:
                errors.append("compute readiness pip check did not pass")
        return list(dict.fromkeys(errors))

    def _probe_runtime_profile(
        self,
        public_profile: dict[str, Any],
        gateway: str,
        *,
        operator_environment: Mapping[str, str] | None = None,
        suite_config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile_id = str(public_profile["id"])
        profile = CLUSTER.runtime_profile(profile_id)
        result = copy.deepcopy(public_profile)
        result["verification"] = {
            "checked": False,
            "status": "configured",
            "errors": [],
        }

        if profile.backend in {"conda", "existing"} and profile.environment_path:
            python_executable = f"{profile.environment_path.rstrip('/')}/bin/python"
            probe_input = {
                "profile_id": profile_id,
                "profile_snapshot_sha256": content_sha256(
                    CLUSTER.runtime_profile_snapshot(profile_id)
                ),
                "environment_path": profile.environment_path,
                "python_version": profile.verification.python_version,
                "distributions": profile.verification.distributions,
                "required_distributions": profile.verification.required_distributions,
                "python_imports": profile.verification.python_imports,
                "executables": profile.verification.executables,
                "executable_providers": [
                    item.model_dump(mode="json")
                    for item in profile.verification.executable_providers
                ],
                "shared_libraries": profile.verification.shared_libraries,
                "gym_registrations": [
                    item.model_dump(mode="json")
                    for item in profile.verification.gym_registrations
                ],
                "platform_system": profile.verification.platform_system,
                "glibc_minimum": profile.verification.glibc_minimum,
                "requires_gpu": profile.verification.requires_gpu,
                "required_checks": (
                    profile.verification.compute_smoke.required_checks
                    if profile.verification.compute_smoke
                    else []
                ),
                "pip_check": profile.verification.pip_check,
                "compute_attestation_path": (
                    profile.verification.compute_attestation_path
                ),
                "source_prerequisites": [
                    item.model_dump(mode="json") for item in profile.source_prerequisites
                ],
            }
            if suite_config is not None:
                from .runtime_readiness import suite_contract_sha256

                probe_input["suite_contract_sha256"] = suite_contract_sha256(
                    suite_config
                )
            attestation_path = str(
                profile.verification.compute_attestation_path or ""
            ).strip()
            if not attestation_path:
                result["verification"] = {
                    "checked": True,
                    "status": "configured",
                    "errors": [
                        f"runtime profile {profile_id} has no compute_attestation_path; "
                        "configure one and run the one-GPU evaluator readiness smoke"
                    ],
                }
                return result

            command_lines = [
                (
                    f"test -d {shlex.quote(profile.environment_path)} && "
                    "printf 'SKYNET_RUNTIME_ENVIRONMENT_PRESENT=1\\n' || "
                    "printf 'SKYNET_RUNTIME_ENVIRONMENT_PRESENT=0\\n'"
                ),
                (
                    f"test -x {shlex.quote(python_executable)} && "
                    "printf 'SKYNET_RUNTIME_PYTHON_PRESENT=1\\n' || "
                    "printf 'SKYNET_RUNTIME_PYTHON_PRESENT=0\\n'"
                ),
                (
                    f"if test -r {shlex.quote(attestation_path)}; then "
                    "printf 'SKYNET_RUNTIME_ATTESTATION_BEGIN\\n'; "
                    f"cat {shlex.quote(attestation_path)}; "
                    "printf '\\nSKYNET_RUNTIME_ATTESTATION_END\\n'; "
                    "else printf 'SKYNET_RUNTIME_ATTESTATION_MISSING=1\\n'; fi"
                ),
            ]
            for index, prerequisite in enumerate(profile.source_prerequisites):
                path = shlex.quote(prerequisite.path)
                command_lines.append(
                    f"test -d {path} && printf 'SKYNET_RUNTIME_SOURCE_{index}_PRESENT=1\\n' "
                    f"|| printf 'SKYNET_RUNTIME_SOURCE_{index}_PRESENT=0\\n'"
                )
                if prerequisite.kind == "git_checkout":
                    revision = shlex.quote(str(prerequisite.revision))
                    command_lines.append(
                        f"printf 'SKYNET_RUNTIME_SOURCE_{index}_HEAD=%s\\n' "
                        f'"$(git -C {path} rev-parse HEAD 2>/dev/null || true)"'
                    )
                    command_lines.append(
                        f"printf 'SKYNET_RUNTIME_SOURCE_{index}_EXPECTED=%s\\n' "
                        f'"$(git -C {path} rev-parse {revision}^{{commit}} '
                        '2>/dev/null || true)"'
                    )
            command = "\n".join(command_lines)
            try:
                host, output = self.cluster.run_with_fallback(command, gateway, timeout=30)
                errors: list[str] = []
                actual: dict[str, Any] = {"source_prerequisites": []}
                if "SKYNET_RUNTIME_ENVIRONMENT_PRESENT=1" not in output:
                    errors.append(
                        "configured evaluator environment directory is missing: "
                        f"{profile.environment_path}"
                    )
                if "SKYNET_RUNTIME_PYTHON_PRESENT=1" not in output:
                    errors.append(
                        "configured evaluator Python is missing or not executable: "
                        f"{python_executable}"
                    )

                begin = "SKYNET_RUNTIME_ATTESTATION_BEGIN\n"
                end = "\nSKYNET_RUNTIME_ATTESTATION_END"
                if begin not in output or end not in output:
                    errors.append(
                        "compute readiness attestation is missing: "
                        f"{attestation_path}; run the one-GPU evaluator readiness smoke on "
                        "a supported Slurm compute node"
                    )
                else:
                    encoded = output.split(begin, 1)[1].split(end, 1)[0]
                    try:
                        attestation = json.loads(encoded)
                    except json.JSONDecodeError as error:
                        errors.append(
                            f"compute readiness attestation is invalid JSON: {error}"
                        )
                    else:
                        actual["compute_attestation"] = attestation
                        if not isinstance(attestation, Mapping):
                            errors.append(
                                "compute readiness attestation must be a JSON object"
                            )
                        else:
                            errors.extend(
                                self._validate_compute_runtime_attestation(
                                    attestation, probe_input
                                )
                            )

                for index, prerequisite in enumerate(profile.source_prerequisites):
                    present = (
                        f"SKYNET_RUNTIME_SOURCE_{index}_PRESENT=1" in output
                    )
                    record: dict[str, Any] = {
                        "name": prerequisite.name,
                        "path": prerequisite.path,
                        "present": present,
                    }
                    if prerequisite.required and not present:
                        errors.append(
                            "required source prerequisite is missing: "
                            f"{prerequisite.path}"
                        )
                    if prerequisite.kind == "git_checkout" and present:
                        head_match = re.search(
                            rf"^SKYNET_RUNTIME_SOURCE_{index}_HEAD=(.*)$",
                            output,
                            flags=re.MULTILINE,
                        )
                        expected_match = re.search(
                            rf"^SKYNET_RUNTIME_SOURCE_{index}_EXPECTED=(.*)$",
                            output,
                            flags=re.MULTILINE,
                        )
                        head = head_match.group(1).strip() if head_match else ""
                        expected_revision = (
                            expected_match.group(1).strip()
                            if expected_match
                            else ""
                        )
                        record.update(
                            {"head": head or None, "expected_commit": expected_revision or None}
                        )
                        if not head or not expected_revision:
                            errors.append(
                                f"could not verify {prerequisite.name} Git revision "
                                f"{prerequisite.revision}"
                            )
                        elif head != expected_revision:
                            errors.append(
                                f"{prerequisite.name} checkout does not match "
                                f"{prerequisite.revision}"
                            )
                    actual["source_prerequisites"].append(record)

                errors = list(dict.fromkeys(errors))
                verified = not errors
                result.update(
                    {
                        "status": "runtime_verified" if verified else "configured",
                        "runtime_verified": verified,
                        "verification": {
                            "checked": True,
                            "status": "runtime_verified" if verified else "configured",
                            "gateway": host,
                            "actual": actual,
                            "errors": errors,
                        },
                    }
                )
            except (ClusterError, ValueError, json.JSONDecodeError, IndexError) as error:
                result["verification"] = {
                    "checked": True,
                    "status": "configured",
                    "errors": [str(error)],
                }
            return result

        if profile.backend == "apptainer" and profile.container_image:
            image = profile.container_image
            if not image.startswith("/"):
                result["verification"]["reason"] = (
                    "remote container references are configured but are not pulled during inspection"
                )
                return result
            command = (
                "command -v apptainer >/dev/null 2>&1 && "
                f"test -r {shlex.quote(image)}"
            )
            try:
                host, _ = self.cluster.run_with_fallback(command, gateway, timeout=15)
                result.update(
                    {
                        "status": "runtime_verified",
                        "runtime_verified": True,
                        "verification": {
                            "checked": True,
                            "status": "runtime_verified",
                            "gateway": host,
                            "errors": [],
                        },
                    }
                )
            except ClusterError as error:
                result["verification"] = {
                    "checked": True,
                    "status": "configured",
                    "errors": [str(error)],
                }
            return result

        result["verification"]["reason"] = "this backend cannot be safely verified without a project"
        return result

    def validate_adapter_registry_entry(
        self, adapter_id: str, request: AdapterValidationRequest
    ) -> dict[str, Any]:
        detail = self.database.get_adapter(
            adapter_id, version_number=request.version_number, include_versions=True
        )
        if not detail:
            raise KeyError("Adapter not found")
        version = self._selected_version(detail)
        errors: list[str] = []
        try:
            manifest = AdapterManifest.model_validate(version["manifest"])
        except ValueError as error:
            manifest = None
            errors.append(str(error))
        repository = request.repository_url or version.get("repository_url") or detail.get("repository_url")
        inspection: dict[str, Any] | None = None
        resolved: dict[str, Any] | None = None
        commit: str | None = None
        if manifest and not repository:
            repository = manifest.default_repository
        if not repository:
            errors.append("repository_url is required for validation")
        elif manifest:
            try:
                commit = self.source_discovery.resolve_revision(
                    str(repository), request.source_revision, request.gateway
                )
                inspection = self.source_discovery.inspect(
                    str(repository), commit, request.gateway, request.project_subdirectory
                )
                resolved = resolve_runtime(
                    inspection,
                    {"backend": "auto"},
                    manifest.runtime.model_dump(mode="json"),
                )
            except (ValueError, ClusterError) as error:
                errors.append(str(error))
        validation = self.database.record_adapter_validation(
            adapter_id,
            status="PASSED" if not errors else "FAILED",
            version_number=int(version["version_number"]),
            repository_url=str(repository) if repository else None,
            source_revision=commit or request.source_revision,
            evidence=inspection or {},
            errors=errors,
            resolved_runtime=resolved,
            created_by=request.created_by,
        )
        return {
            "validation": validation,
            "status": "VALID" if not errors else "INVALID",
            "errors": errors,
            "inspection": inspection,
            "resolved_runtime": resolved,
        }

    def validate_unsaved_adapter_manifest(
        self, request: UnsavedAdapterValidationRequest
    ) -> dict[str, Any]:
        errors: list[str] = []
        inspection: dict[str, Any] | None = None
        resolved: dict[str, Any] | None = None
        manifest: AdapterManifest | None = None
        try:
            manifest = AdapterManifest.model_validate(request.manifest)
        except ValueError as error:
            errors.append(str(error))
        repository = request.repository_url or (
            manifest.default_repository if manifest is not None else None
        )
        commit: str | None = None
        if manifest is not None and repository:
            try:
                commit = self.source_discovery.resolve_revision(
                    repository, request.source_revision, request.gateway
                )
                inspection = self.source_discovery.inspect(
                    repository, commit, request.gateway, request.project_subdirectory
                )
                resolved = resolve_runtime(
                    inspection,
                    {"backend": "auto"},
                    manifest.runtime.model_dump(mode="json"),
                )
            except (ValueError, ClusterError) as error:
                errors.append(str(error))
        normalized = canonical_adapter_manifest(manifest) if manifest is not None else None
        return {
            "status": "VALID" if not errors else "INVALID",
            "errors": errors,
            "manifest": normalized,
            "manifest_sha256": adapter_manifest_sha256(manifest) if manifest is not None else None,
            "repository": repository,
            "source_revision": commit or request.source_revision,
            "inspection": inspection,
            "resolved_runtime": resolved,
            "persisted": False,
        }

    def _normalize_data_input(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        data = copy.deepcopy(dict(payload.get("data") or {}))
        bundle_id = payload.get("data_bundle_id") or data.pop("bundle_id", None)
        if bundle_id is not None:
            if data.get("bundle") is not None:
                raise ValueError("provide either data_bundle_id or a canonical data.bundle snapshot, not both")
            data["bundle"] = self.database.data_bundle_snapshot(
                str(bundle_id), require_active=True
            )
        return data

    @staticmethod
    def _reject_tracking_secrets(value: Any, path: str = "tracking") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
                sensitive = normalized in {
                    "api_key", "token", "password", "secret", "access_key", "private_key"
                } or normalized.endswith(("_api_key", "_token", "_password", "_secret"))
                if sensitive:
                    raise ValueError(
                        f"{path}.{key} is credential-like; configure credentials through Connections"
                    )
                PipelineService._reject_tracking_secrets(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                PipelineService._reject_tracking_secrets(child, f"{path}[{index}]")

    def normalize_spec(self, payload: Mapping[str, Any]) -> ExperimentSpec:
        self._reject_tracking_secrets(payload.get("tracking"))
        if "identity" in payload and "source" in payload:
            canonical = copy.deepcopy(dict(payload))
            canonical.setdefault(
                "intent",
                {"explicit_parameters": explicit_train_parameter_paths(canonical)},
            )
            canonical["data"] = self._normalize_data_input(canonical)
            canonical.pop("data_bundle_id", None)
            canonical.setdefault("apiVersion", canonical.pop("api_version", "skynet.rl2/v1"))
            resources = canonical.setdefault("resources", {})
            source = dict(canonical["source"])
            adapter_key = str(source.get("adapter_id") or source.get("adapter") or "")
            source, manifest, _ = self._snapshot_adapter(source, adapter_key)
            source.setdefault(
                "project_subdirectory", manifest.defaults.effective_project_subdirectory
            )
            runtime_input = dict(canonical.get("runtime") or {"backend": "auto"})
            gateway = str(resources.get("gateway") or "auto")
            source, runtime = self._resolve_source_and_runtime(
                source, runtime_input, manifest, gateway
            )
            canonical["source"] = source
            canonical["runtime"] = runtime
            canonical = self._apply_manifest_data_bindings(canonical, manifest)
            canonical = self._apply_canonical_manifest_defaults(canonical, manifest)
            self._validate_repository_input_choices(canonical, source, manifest, gateway)
            self._reject_repository_choice_sweeps(canonical, manifest)
            canonical.setdefault("resources", {}).setdefault(
                "queue_policy", CLUSTER.defaults.queue_policy
            )
            return ExperimentSpec.model_validate(canonical)

        requested_hyper = dict(payload.get("hyperparameters") or {})
        explicit_parameters = {
            path
            for field, path in {
                "learning_rate": "train.learning_rate",
                "batch_semantics": "train.batch.declared_semantics",
                "batch_size": "train.batch.value",
                "gradient_accumulation": "train.batch.gradient_accumulation_steps",
                "num_workers": "train.num_workers_per_rank",
                "max_steps": "train.max_steps",
                "max_epochs": "train.max_epochs",
                "seed": "train.seed",
                "precision": "train.precision",
            }.items()
            if field in requested_hyper
            and not _frontend_parameter_is_unset(requested_hyper[field])
        }
        canonical_hyperparameter_names = {
            "learning_rate", "batch_semantics", "batch_size", "gradient_accumulation",
            "num_workers", "max_steps", "max_epochs", "seed", "precision",
        }
        if any(
            field not in canonical_hyperparameter_names
            and not _frontend_parameter_is_unset(value)
            for field, value in requested_hyper.items()
        ):
            explicit_parameters.add("train.hyperparameters")
        for field, path in {
            "checkpoint_save_steps": "train.checkpoint.save_every_steps",
            "checkpoint_warning_seconds": "train.checkpoint.save_before_timeout_seconds",
            "checkpoint_keep_last": "train.checkpoint.keep_last",
            "auto_resume": "train.checkpoint.auto_resume",
            "max_attempts": "train.checkpoint.max_attempts",
            "checkpoint_final_selector": "train.checkpoint.final_selector",
            "remove_training_state_after_success": (
                "train.checkpoint.remove_training_state_after_success"
            ),
        }.items():
            if field in payload and not _frontend_parameter_is_unset(payload[field]):
                explicit_parameters.add(path)

        source = dict(payload.get("source") or {})
        adapter_key = str(payload.get("adapter") or source.get("adapter_id") or source.get("adapter") or "")
        if not adapter_key:
            raise ValueError("an adapter registry ID is required")
        source, manifest, _ = self._snapshot_adapter(source, adapter_key)
        source.setdefault(
            "project_subdirectory", manifest.defaults.effective_project_subdirectory
        )
        payload = self._payload_with_manifest_defaults(payload, manifest)
        resource_input = dict(payload.get("resources") or {})
        gateway = str(resource_input.get("gateway") or CLUSTER.defaults.gateway)
        runtime_input = dict(payload.get("runtime") or {"backend": "auto"})
        source, runtime = self._resolve_source_and_runtime(
            source, runtime_input, manifest, gateway
        )

        hyper = dict(payload.get("hyperparameters") or {})
        native_lines = list(payload.get("native_overrides") or [])
        native_config, native_overrides, argv, resume_argv = _parse_native(native_lines)
        precision = hyper.get("precision")
        if _frontend_parameter_is_unset(precision):
            precision = None
        checkpoint_input = dict(payload.get("checkpoint") or {})
        checkpoint_mode = checkpoint_input.get("mode", "none")
        if checkpoint_mode != "none" and checkpoint_input.get("path"):
            native_config["initial_checkpoint"] = checkpoint_input["path"]
            native_config["initial_checkpoint_mode"] = checkpoint_mode

        queue_policy = str(resource_input.get("queue_policy") or CLUSTER.defaults.queue_policy)
        initial_policy = CLUSTER.defaults.queue_policy if queue_policy == "auto" else queue_policy
        queue = CLUSTER.queue(initial_policy)
        gpu_mode = str(resource_input.get("gpu_mode") or "auto")
        gpu = {
            "mode": "explicit" if gpu_mode in {"manual", "explicit"} else "auto",
            "type": CLUSTER.defaults.gpu_type if resource_input.get("gpu_type") in {None, "auto"} else resource_input["gpu_type"],
        }
        if gpu["mode"] == "explicit":
            gpu["count"] = int(resource_input.get("gpus_per_node") or 1)
        if resource_input.get("gpu_profile"):
            gpu["profile"] = str(resource_input["gpu_profile"])

        max_steps = hyper.get("max_steps")
        max_steps = None if _frontend_parameter_is_unset(max_steps) else max_steps
        max_epochs = hyper.get("max_epochs")
        max_epochs = None if _frontend_parameter_is_unset(max_epochs) else max_epochs
        if max_steps is None and max_epochs is None:
            max_steps = 1000
        learning_rate = hyper.get("learning_rate")
        if _frontend_parameter_is_unset(learning_rate):
            learning_rate = None
        batch_size = hyper.get("batch_size")
        batch_size = 1 if _frontend_parameter_is_unset(batch_size) else int(batch_size)
        gradient_accumulation = hyper.get("gradient_accumulation")
        gradient_accumulation = (
            1
            if _frontend_parameter_is_unset(gradient_accumulation)
            else int(gradient_accumulation)
        )
        batch: dict[str, Any] = {
            "value": batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
        }
        batch_semantics = hyper.get("batch_semantics")
        if not _frontend_parameter_is_unset(batch_semantics):
            batch["declared_semantics"] = batch_semantics
        num_workers = hyper.get("num_workers")
        num_workers = 4 if _frontend_parameter_is_unset(num_workers) else int(num_workers)
        seed = hyper.get("seed")
        seed = 42 if _frontend_parameter_is_unset(seed) else int(seed)

        train: dict[str, Any] = {
            "learning_rate": learning_rate,
            "batch": batch,
            "num_workers_per_rank": num_workers,
            "max_steps": max_steps,
            "max_epochs": max_epochs,
            "seed": seed,
            "precision": precision,
            "hyperparameters": {
                key: copy.deepcopy(value)
                for key, value in hyper.items()
                if key not in canonical_hyperparameter_names
            },
            "checkpoint": {
                "save_every_steps": int(payload.get("checkpoint_save_steps") or CLUSTER.defaults.checkpoint_save_steps),
                "save_before_timeout_seconds": int(payload.get("checkpoint_warning_seconds") or 300),
                "keep_last": int(payload.get("checkpoint_keep_last") or CLUSTER.defaults.checkpoint_keep_last),
                "auto_resume": bool(payload.get("auto_resume", CLUSTER.defaults.checkpoint_auto_resume)),
                "max_attempts": int(payload.get("max_attempts") or CLUSTER.defaults.max_attempts),
                "final_selector": str(payload.get("checkpoint_final_selector") or "latest"),
                "remove_training_state_after_success": (
                    True
                    if _frontend_parameter_is_unset(payload.get("remove_training_state_after_success"))
                    else bool(payload["remove_training_state_after_success"])
                ),
            },
        }

        tracking_input = dict(payload.get("tracking") or {})
        forbidden_tracking_keys = {
            "api_key", "token", "password", "secret", "access_key", "private_key"
        }
        for key in tracking_input:
            if str(key).lower() in forbidden_tracking_keys:
                raise ValueError(
                    "tracking credentials must be configured through Settings > Connections"
                )
        provider_inputs = copy.deepcopy(tracking_input.get("providers") or [])
        if not isinstance(provider_inputs, list):
            raise ValueError("tracking.providers must be a list")
        providers: list[dict[str, Any]] = []
        for provider_input in provider_inputs:
            if not isinstance(provider_input, Mapping):
                raise ValueError("each tracking provider must be an object")
            if any(str(key).lower() in forbidden_tracking_keys for key in provider_input):
                raise ValueError(
                    "tracking credentials must be configured through Settings > Connections"
                )
            provider_name = str(provider_input.get("provider") or "").lower()
            if provider_name not in {"mlflow", "wandb"}:
                raise ValueError(f"unsupported tracking provider: {provider_name or '<missing>'}")
            normalized_provider = {
                "provider": provider_name,
                "enabled": bool(provider_input.get("enabled", True)),
                "run_name_template": provider_input.get("run_name_template") or None,
            }
            if provider_name == "mlflow":
                normalized_provider.update({
                    "tracking_uri": provider_input.get("tracking_uri") or None,
                    "experiment": provider_input.get("experiment") or None,
                })
            else:
                normalized_provider.update({
                    "base_url": provider_input.get("base_url") or None,
                    "entity": provider_input.get("entity") or None,
                    "project": provider_input.get("project") or None,
                })
            providers.append(normalized_provider)
        legacy_uri = tracking_input.get("tracking_uri") or tracking_input.get("uri")
        legacy_experiment = (
            tracking_input.get("mlflow_experiment") or tracking_input.get("experiment")
        )
        if not providers and tracking_input.get("enabled", bool(legacy_uri)) and legacy_uri:
            providers.append({
                "provider": "mlflow",
                "enabled": True,
                "tracking_uri": legacy_uri,
                "experiment": legacy_experiment or str(payload.get("name") or "skynet"),
            })
        native_tracking = tracking_input.get("native_tracking", "preserve")
        if isinstance(native_tracking, bool):
            native_tracking = "preserve" if native_tracking else "disable"
        tracking = {
            "providers": providers,
            "mlflow_tracking_uri": legacy_uri if any(
                item["provider"] == "mlflow" and item["enabled"] for item in providers
            ) else None,
            "mlflow_experiment": legacy_experiment or str(payload.get("name") or "skynet"),
            "native_tracking": native_tracking,
            "tags": copy.deepcopy(tracking_input.get("tags") or {}),
            "offline_spool": bool(tracking_input.get("offline_spool", True)),
        }
        evaluation_input = dict(payload.get("evaluation") or {})
        evaluation_specs: list[dict[str, Any]] = copy.deepcopy(
            evaluation_input.get("canonical_specs") or []
        )
        if evaluation_input.get("enabled"):
            suite_ids = evaluation_input.get("suite_ids") or []
            suites = {item["id"]: item for item in self.database.list_evaluation_suites()}
            for suite_id in suite_ids:
                suite = suites.get(str(suite_id))
                if not suite:
                    raise ValueError(f"unknown evaluation suite: {suite_id}")
                config = suite["config_json"]
                evaluation_specs.append({
                    "adapter": suite["evaluator_adapter"],
                    "suite": suite["name"],
                    "suite_version": suite["suite_version"],
                    "tasks": config.get("tasks", []),
                    "profile": config.get("default_profile", "standard"),
                })

        canonical = {
            "apiVersion": "skynet.rl2/v1",
            "kind": "Experiment",
            "identity": {
                "project": str(payload.get("project") or "default"),
                "experiment": str(payload.get("name") or "experiment"),
            },
            "source": source,
            "intent": {"explicit_parameters": sorted(explicit_parameters)},
            "runtime": runtime,
            "data": self._normalize_data_input(payload),
            "train": train,
            "resources": {
                "gateway": gateway,
                "queue_policy": queue_policy,
                "account": queue.account,
                "partition": queue.partition,
                "nodes": int(resource_input.get("nodes") or 1),
                "node": {
                    "mode": resource_input.get("node_mode", "auto"),
                    "name": resource_input.get("node"),
                },
                "gpu": gpu,
                "cpus_per_task": int(resource_input.get("cpus_per_task") or CLUSTER.defaults.cpus_per_task),
                "memory_gb": int(resource_input.get("memory_gb") or CLUSTER.defaults.memory_gb),
                "time_limit": str(resource_input.get("time_limit") or CLUSTER.defaults.time_limit),
            },
            "sweep": _sweep_from_frontend((payload.get("sweep") or {}).get("definition")),
            "tracking": tracking,
            "evaluation": evaluation_specs,
            "native": {
                "argv": argv,
                "resume_argv": resume_argv,
                "config": native_config,
                "overrides": native_overrides,
            },
        }
        self._apply_manifest_data_bindings(canonical, manifest)
        self._apply_manifest_input_defaults(canonical, manifest)
        self._validate_manifest_input_fields(canonical, manifest)
        self._validate_repository_input_choices(canonical, source, manifest, gateway)
        self._reject_repository_choice_sweeps(canonical, manifest)
        return ExperimentSpec.model_validate(canonical)

    @staticmethod
    def _project(database: Database, name: str) -> dict[str, Any]:
        existing = next((item for item in database.list_projects() if item["name"] == name), None)
        if existing:
            return existing
        try:
            return database.create_project(name)
        except sqlite3.IntegrityError:
            return next(item for item in database.list_projects() if item["name"] == name)

    def _materialize_experiment_revision(
        self, revision_id: str, variants: list[Any]
    ) -> None:
        for resolved in variants:
            variant = self.database.create_variant(
                revision_id,
                variant_index=resolved.index,
                name=resolved.name,
                parameters=resolved.parameters,
                resolved_spec=resolved.resolved_spec.model_dump(mode="json", by_alias=True),
            )
            run = self.database.create_run(
                variant["id"],
                seed=resolved.seed,
                adapter_name=resolved.resolved_spec.source.adapter,
                adapter_version=str(resolved.resolved_spec.source.adapter_version),
                source_commit=resolved.resolved_spec.source.revision,
                runtime_profile=resolved.resolved_spec.runtime.profile,
                run_directory=f"{WORK_ROOT}/jobs/runs/pending",
                status="DRAFT",
            )
            self.database.update_run(
                run["id"], run_directory=f"{WORK_ROOT}/jobs/runs/{run['id']}"
            )
            plan = resolve_adapter_plan(resolved.resolved_spec)
            self.database.create_stage(
                run["id"],
                stage_type="TRAIN",
                name="train",
                status="DRAFT",
                auto_resume=resolved.resolved_spec.train.checkpoint.auto_resume,
                max_attempts=resolved.resolved_spec.train.checkpoint.max_attempts,
                resolved_config={
                    "spec": resolved.resolved_spec.model_dump(mode="json", by_alias=True),
                    "plan": plan.model_dump(mode="json"),
                    "blockers": plan.blockers,
                },
            )

    def _preflight_live_auto_queue(
        self,
        specs: list[ExperimentSpec],
        gateway: str | None = None,
    ) -> None:
        """Fail closed on unavailable live quota before creating execution records."""

        checked: set[tuple[str, str, int, str]] = set()
        for spec in specs:
            if spec.resources.queue_policy != "auto":
                continue
            plan = resolve_adapter_plan(spec)
            if plan.blockers:
                continue
            requested_gateway = gateway if gateway is not None else spec.resources.gateway
            gpu_count = resolve_gpu_count(spec, plan)
            signature = (
                requested_gateway,
                spec.resources.gpu.gpu_type,
                gpu_count,
                spec.resources.time_limit,
            )
            if signature in checked:
                continue
            self._auto_queue(
                spec,
                plan,
                requested_gateway,
                record_snapshot=False,
            )
            checked.add(signature)

    def _repository_argument_validation(
        self,
        spec: ExperimentSpec,
        plan: AdapterPlan,
        gateway: str,
    ) -> dict[str, Any] | None:
        if spec.source.adapter_manifest is None:
            return None
        manifest = AdapterManifest.model_validate(spec.source.adapter_manifest)
        repository_options = self._repository_input_options(
            spec.source.model_dump(mode="json"), manifest, gateway
        )
        parameters = repository_argument_validation_cache_parameters(
            spec, manifest, plan, repository_options
        )
        cached = self.source_metadata.get(
            REPOSITORY_ARGUMENT_VALIDATION_CACHE_KIND,
            spec.source.repository,
            parameters,
        )
        if cached is not None:
            return self.source_metadata.response(cached, cache_hit=True)
        result = validate_repository_arguments(
            spec,
            manifest,
            plan,
            repository_options,
        )
        result.update(parameters)
        stored = self.source_metadata.put(
            REPOSITORY_ARGUMENT_VALIDATION_CACHE_KIND,
            spec.source.repository,
            parameters,
            result,
        )
        return self.source_metadata.response(stored, cache_hit=False)

    def _reject_cached_repository_argument_failures(
        self,
        specs: list[ExperimentSpec],
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for spec in specs:
            plan = resolve_adapter_plan(spec)
            if plan.blockers:
                continue
            report = self._repository_argument_validation(
                spec,
                plan,
                spec.resources.gateway,
            )
            if report is None:
                continue
            reports.append(report)
            blocker = validation_failure_reason(report)
            if blocker:
                raise ValueError(blocker)
        return reports

    def _preflight_repository_arguments(
        self,
        specs: list[ExperimentSpec],
        gateway: str | None = None,
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for spec in specs:
            plan = resolve_adapter_plan(spec)
            if plan.blockers:
                continue
            selected_gateway = gateway if gateway is not None else spec.resources.gateway
            try:
                report = self._repository_argument_validation(
                    spec, plan, selected_gateway
                )
            except ValueError as error:
                raise ValueError(
                    "repository-native argument validation is unavailable: "
                    f"{error}"
                ) from error
            if report is None:
                continue
            reports.append(report)
            blocker = validation_failure_reason(report)
            if blocker:
                raise ValueError(blocker)
        return reports

    def repair_pre_submission_orphans(
        self, experiment_id: str | None = None
    ) -> dict[str, Any]:
        """Restore evidence-free submission graphs to an editable draft state."""

        with self._reconcile_lock:
            return self.database.repair_pre_submission_orphans(experiment_id)

    def create_experiment(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        spec = self.normalize_spec(payload)
        self._validate_tracking_requirements(spec)
        variants = expand_sweep(spec)
        self._reject_cached_repository_argument_failures(
            [variant.resolved_spec for variant in variants]
        )
        self._preflight_live_auto_queue(
            [variant.resolved_spec for variant in variants]
        )
        requested_spec = spec.model_dump(mode="json", by_alias=True)
        requested_hash = content_sha256(requested_spec)
        project = self._project(self.database, spec.identity.project)
        with self._reconcile_lock:
            try:
                experiment = self.database.create_experiment(
                    project_id=project["id"],
                    name=spec.identity.experiment,
                    description="Canonical Skynet experiment",
                    requested_spec=requested_spec,
                    status="DRAFT",
                    spec_schema_version=spec.api_version,
                )
            except sqlite3.IntegrityError as error:
                existing = next(
                    (
                        item
                        for item in self.database.list_experiments(
                            project_id=project["id"], limit=1000
                        )
                        if item["name"] == spec.identity.experiment
                    ),
                    None,
                )
                if existing is None:
                    raise ValueError(
                        f"experiment {spec.identity.project}/{spec.identity.experiment} "
                        "already exists"
                    ) from error
                self.database.repair_pre_submission_orphans(existing["id"])
                if (
                    existing.get("latest_spec_sha256") == requested_hash
                ):
                    return self.experiment_detail(existing["id"])
                # A natural-key collision identifies the experiment lineage. A
                # changed canonical spec is a new editable revision, regardless
                # of whether earlier revisions acquired execution evidence.
                return self.create_experiment_revision(existing["id"], payload)
            revision = experiment["latest_revision"]
            try:
                self._materialize_experiment_revision(revision["id"], variants)
            except Exception:
                rolled_back = self.database.discard_unsubmitted_experiment_revision(
                    experiment["id"],
                    revision["id"],
                    delete_experiment_if_empty=True,
                )
                if not rolled_back:
                    raise RuntimeError(
                        "experiment materialization failed and its draft graph could "
                        "not be rolled back safely"
                    )
                raise
            return self.experiment_detail(experiment["id"])

    def create_experiment_revision(
        self,
        experiment_id: str,
        payload: Mapping[str, Any],
        *,
        submit: bool = False,
        gateway: str = "auto",
    ) -> dict[str, Any]:
        spec = self.normalize_spec(payload)
        self._validate_tracking_requirements(spec)
        variants = expand_sweep(spec)
        if submit:
            self._preflight_repository_arguments(
                [variant.resolved_spec for variant in variants], gateway
            )
        else:
            self._reject_cached_repository_argument_failures(
                [variant.resolved_spec for variant in variants]
            )
        if submit:
            self._preflight_live_auto_queue(
                [variant.resolved_spec for variant in variants], gateway
            )
        requested_spec = spec.model_dump(mode="json", by_alias=True)
        requested_hash = content_sha256(requested_spec)
        with self._reconcile_lock:
            experiment = self.database.get_experiment(experiment_id)
            if not experiment:
                raise KeyError("Experiment not found")
            if (
                spec.identity.project != experiment.get("project_name")
                or spec.identity.experiment != experiment.get("name")
            ):
                raise ValueError(
                    "a revision must preserve the experiment project and name"
                )
            matching = next(
                (
                    revision
                    for revision in experiment["revisions"]
                    if revision["requested_spec_sha256"] == requested_hash
                ),
                None,
            )
            latest = experiment["latest_revision"]
            if matching is not None:
                if not (
                    submit
                    and latest
                    and matching["id"] == latest["id"]
                ):
                    raise ValueError(
                        "this exact specification already exists as experiment "
                        f"revision {matching['revision_number']}"
                    )
            else:
                revision = self.database.create_experiment_revision(
                    experiment_id,
                    requested_spec,
                    spec_schema_version=spec.api_version,
                )
                try:
                    self._materialize_experiment_revision(revision["id"], variants)
                except Exception:
                    rolled_back = self.database.discard_unsubmitted_experiment_revision(
                        experiment_id, revision["id"]
                    )
                    if not rolled_back:
                        raise RuntimeError(
                            "experiment revision materialization failed and its draft "
                            "graph could not be rolled back safely"
                        )
                    raise
        if submit:
            return self.submit_experiment(experiment_id, gateway)["experiment"]
        return self.experiment_detail(experiment_id)

    def experiment_detail(self, experiment_id: str) -> dict[str, Any]:
        experiment = self.database.get_experiment(experiment_id)
        if not experiment:
            raise KeyError("Experiment not found")
        revision = experiment["latest_revision"]
        for item in experiment["revisions"]:
            item["locked"] = bool(item.get("submitted_at"))
            item["lifecycle"] = "SUBMITTED" if item["locked"] else "DRAFT"
        variants = self.database.list_variants(revision["id"]) if revision else []
        runs = self.database.list_runs(
            experiment_revision_id=revision["id"] if revision else None,
            limit=10000,
        )
        _attach_run_progress_summaries(self.database, runs)
        experiment["variants"] = variants
        experiment["runs"] = runs
        experiment["variant_count"] = len(variants)
        experiment["run_count"] = len(runs)
        experiment["revision_count"] = len(experiment["revisions"])
        experiment["revision_number"] = revision["revision_number"] if revision else None
        experiment["locked"] = bool(revision and revision.get("submitted_at"))
        experiment["lifecycle"] = "SUBMITTED" if experiment["locked"] else "DRAFT"
        return experiment

    def preview(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        spec = self.normalize_spec(payload)
        self._validate_tracking_requirements(spec)
        variants = expand_sweep(spec)
        scripts: list[str] = []
        warnings: list[str] = []
        blockers: list[dict[str, Any]] = []
        argument_validations: list[dict[str, Any]] = []
        for variant in variants:
            resolved = self._preview_queue(variant.resolved_spec)
            plan = resolve_adapter_plan(resolved)
            warnings.extend(plan.warnings)
            warnings.extend(plan.todos)
            if plan.blockers:
                blockers.append({"variant": variant.name, "reasons": plan.blockers})
                scripts.append(
                    "# BLOCKED: " + variant.name + "\n# " + "\n# ".join(plan.blockers)
                )
                continue
            try:
                validation = self._repository_argument_validation(
                    resolved,
                    plan,
                    resolved.resources.gateway,
                )
                validation_blocker = (
                    validation_failure_reason(validation) if validation else None
                )
            except ValueError as error:
                validation = None
                validation_blocker = (
                    "repository-native argument validation is unavailable: "
                    f"{error}"
                )
            if validation is not None:
                argument_validations.append(
                    {"variant": variant.name, "result": validation}
                )
            if validation_blocker:
                blockers.append(
                    {"variant": variant.name, "reasons": [validation_blocker]}
                )
                scripts.append(
                    f"# BLOCKED: {variant.name}\n# {validation_blocker}"
                )
                continue
            compiled = compile_sbatch(resolved, plan, run_id=f"preview-{variant.index:03d}")
            scripts.append(compiled.script)
        if len(variants) > spec.sweep.confirmation_threshold:
            warnings.append(
                f"{len(variants)} variants exceed the confirmation threshold "
                f"of {spec.sweep.confirmation_threshold}"
            )
        return {
            "script": "\n\n".join(scripts),
            "scripts": scripts,
            "variant_count": len(variants),
            "warnings": list(dict.fromkeys(warnings)),
            "blockers": blockers,
            "argument_validations": argument_validations,
            "resolved_revision": spec.source.revision,
        }

    @staticmethod
    def _preview_queue(spec: ExperimentSpec) -> ExperimentSpec:
        if spec.resources.queue_policy != "auto":
            return spec
        payload = spec.model_dump(mode="json", by_alias=True)
        normal_name, normal = next(
            ((name, queue) for name, queue in CLUSTER.queues.items() if not queue.preemptible),
            (CLUSTER.defaults.queue_policy, CLUSTER.queue(CLUSTER.defaults.queue_policy)),
        )
        overflow_name, overflow = next(
            ((name, queue) for name, queue in CLUSTER.queues.items() if queue.preemptible),
            (normal_name, normal),
        )
        selected = overflow if parse_slurm_duration(spec.resources.time_limit) > normal.max_time_seconds else normal
        payload["resources"]["account"] = selected.account
        payload["resources"]["partition"] = selected.partition
        return ExperimentSpec.model_validate(payload)

    def submit_experiment(self, experiment_id: str, gateway: str = "auto") -> dict[str, Any]:
        with self._reconcile_lock:
            pre_submission_repairs = self.database.repair_pre_submission_orphans(
                experiment_id
            )
            detail = self.experiment_detail(experiment_id)
            revision = detail["latest_revision"]
            if not revision:
                raise ValueError("experiment has no revision to submit")
            revision_locked = bool(revision.get("submitted_at"))
            self._validate_tracking_requirements(
                ExperimentSpec.model_validate(revision["requested_spec_json"])
            )
            run_details: list[dict[str, Any]] = []
            candidates: list[tuple[dict[str, Any], dict[str, Any], str]] = []
            for run in detail["runs"]:
                run_detail = self.database.get_run(run["id"])
                if not run_detail:
                    raise KeyError("Run not found")
                training = [
                    stage
                    for stage in run_detail["stages"]
                    if stage["stage_type"] == "TRAIN"
                ]
                if not training:
                    raise ValueError("latest revision has no training stage")
                if not revision_locked:
                    if run_detail["attempts"] or any(
                        stage["status"] != "DRAFT" for stage in training
                    ):
                        raise ValueError("latest revision is not an unsubmitted draft")
                    candidates.extend(
                        (run_detail, stage, "initial") for stage in training
                    )
                else:
                    for stage in training:
                        stage_attempts = [
                            attempt
                            for attempt in run_detail["attempts"]
                            if attempt["stage_id"] == stage["id"]
                        ]
                        stage_attempts.sort(
                            key=lambda item: int(item.get("attempt_number") or 0)
                        )
                        latest_attempt = stage_attempts[-1] if stage_attempts else None
                        active_attempt = any(
                            str(attempt.get("status") or "").upper()
                            in {
                                "SUBMITTING",
                                "SUBMITTED",
                                "PENDING",
                                "RUNNING",
                                "REQUEUED",
                                "CANCELLING",
                            }
                            for attempt in stage_attempts
                        )
                        status = str(stage.get("status") or "").upper()
                        if status in {"DRAFT", "PENDING", "RETRY_PENDING"}:
                            if not active_attempt:
                                candidates.append((run_detail, stage, "continue"))
                        elif (
                            status == "FAILED"
                            and latest_attempt is not None
                            and str(latest_attempt.get("status") or "").upper()
                            == "SUBMISSION_FAILED"
                            and not latest_attempt.get("slurm_job_id")
                            and not latest_attempt.get("slurm_array_job_id")
                        ):
                            candidates.append(
                                (run_detail, stage, "retry_pre_slurm_failure")
                            )
                run_details.append(run_detail)

            candidate_specs: list[ExperimentSpec] = []
            candidate_run_ids: set[str] = set()
            for run_detail, _, _ in candidates:
                if run_detail["id"] in candidate_run_ids:
                    continue
                candidate_specs.append(
                    ExperimentSpec.model_validate(run_detail["resolved_spec_json"])
                )
                candidate_run_ids.add(run_detail["id"])
            self._preflight_repository_arguments(candidate_specs, gateway)
            self._preflight_live_auto_queue(candidate_specs, gateway)

            for run_detail, stage, mode in candidates:
                status = str(stage.get("status") or "").upper()
                if status in {"DRAFT", "PENDING", "RETRY_PENDING"}:
                    continue
                if mode != "retry_pre_slurm_failure":
                    continue
                old_status = stage.get("status")
                transition = self.database.transition_workflow_state(
                    stage_id=stage["id"],
                    stage_updates={"status": "RETRY_PENDING", "completed_at": None},
                    run_id=run_detail["id"],
                    run_updates={"status": "RETRY_PENDING", "completed_at": None},
                    event={
                        "entity_type": "run",
                        "entity_id": run_detail["id"],
                        "event_type": "PRE_SLURM_SUBMISSION_RETRY_QUEUED",
                        "old_status": old_status,
                        "new_status": "RETRY_PENDING",
                        "details": {
                            "gateway": gateway,
                            "experiment_revision_id": revision["id"],
                            "idempotent_retry": revision_locked,
                        },
                    },
                    expected_stage_statuses=(status,),
                )
                if transition.get("applied") is False:
                    continue
            post_submission_repairs: dict[str, Any] = {
                "repaired": 0,
                "revision_ids": [],
                "experiment_ids": [],
            }
            try:
                submitted = self._dispatch_experiment(
                    experiment_id, gateway, experiment_revision_id=revision["id"]
                )
            finally:
                # Validation, compilation, and queue resolution all precede the
                # atomic attempt claim. If none reached that claim, retain an
                # editable draft instead of a false submitted lifecycle.
                post_submission_repairs = self.database.repair_pre_submission_orphans(
                    experiment_id
                )
            repaired_revision_ids = sorted(
                {
                    *pre_submission_repairs.get("revision_ids", []),
                    *post_submission_repairs.get("revision_ids", []),
                }
            )
            return {
                "experiment_id": experiment_id,
                "experiment_revision_id": revision["id"],
                "revision_number": revision["revision_number"],
                "run_count": len(detail["runs"]),
                "submitted": submitted,
                "idempotent_retry": revision_locked,
                "pre_submission_repairs": len(repaired_revision_ids),
                "repaired_revision_ids": repaired_revision_ids,
                "experiment": self.experiment_detail(experiment_id),
            }

    def _auto_queue(
        self,
        spec: ExperimentSpec,
        plan: Any,
        gateway: str,
        *,
        record_snapshot: bool = True,
    ) -> tuple[ExperimentSpec, str | None]:
        if spec.resources.queue_policy != "auto":
            return spec, None
        gpu_count = resolve_gpu_count(spec, plan)
        normal_selection = next(
            ((name, queue) for name, queue in CLUSTER.queues.items() if not queue.preemptible),
            None,
        )
        if normal_selection is None:
            raise ValueError("auto queue selection requires a configured non-preemptible queue")
        normal_name, normal = normal_selection
        overflow_selection = next(
            ((name, queue) for name, queue in CLUSTER.queues.items() if queue.preemptible),
            None,
        )
        exceeds_normal_limit = parse_slurm_duration(spec.resources.time_limit) > normal.max_time_seconds
        if exceeds_normal_limit and overflow_selection is None:
            raise ValueError("requested duration exceeds the normal queue limit and no preemptible queue is configured")
        overflow_name = overflow_selection[0] if overflow_selection else ""
        selected_name = overflow_name if exceeds_normal_limit else normal_name
        snapshot_id: str | None = None
        command = (
            f"export PATH={shlex.quote(SLURM_BIN)}:${{PATH:-}}; "
            f"LC_ALL=C {GPU_USAGE_LONG_COMMAND}"
        )
        try:
            host, output = self.cluster.run_with_fallback(command, gateway, timeout=20)
            row = next(
                (
                    [cell.strip() for cell in line.strip().strip("|").split("|")]
                    for line in output.splitlines()
                    if line.lstrip().startswith("|") and line.strip().strip("|").split("|")[0].strip() == normal.account
                ),
                None,
            )
            if not row:
                raise ClusterError(f"gpu_usage did not return account {normal.account}")
            columns = {
                gpu_type: index + 1
                for index, gpu_type in enumerate(CLUSTER.dashboard.gpu_usage_columns)
            }
            column = columns.get(spec.resources.gpu.gpu_type)
            if column is None:
                raise ClusterError(f"gpu_usage has no column for GPU type {spec.resources.gpu.gpu_type}")
            if column >= len(row):
                raise ClusterError(f"gpu_usage row for {normal.account} is missing the requested GPU column")
            match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", row[column])
            if not match:
                raise ClusterError(f"gpu_usage returned an invalid quota value for {spec.resources.gpu.gpu_type}: {row[column]}")
            if int(match.group(1)) + gpu_count > int(match.group(2)):
                if overflow_selection is None:
                    raise ClusterError("GPU quota is exhausted and no preemptible queue is configured")
                selected_name = overflow_name
            if record_snapshot:
                snapshot = self.database.create_cluster_snapshot(
                    gateway=host,
                    gpu_usage={
                        "raw": output,
                        "selected": selected_name,
                        "requested_gpus": gpu_count,
                    },
                    nodes=[],
                    queue=[],
                )
                snapshot_id = snapshot["id"]
        except ClusterError as error:
            raise ValueError(f"auto queue selection could not verify live GPU quota: {error}") from error
        selected = CLUSTER.queue(selected_name)
        payload = spec.model_dump(mode="json", by_alias=True)
        payload["resources"]["account"] = selected.account
        payload["resources"]["partition"] = selected.partition
        return ExperimentSpec.model_validate(payload), snapshot_id

    def _dispatch_experiment(
        self,
        experiment_id: str,
        gateway: str = "auto",
        *,
        experiment_revision_id: str | None = None,
    ) -> list[dict[str, Any]]:
        experiment = self.database.get_experiment(experiment_id)
        if not experiment:
            raise KeyError("Experiment not found")
        revision = (
            next(
                (
                    item
                    for item in experiment["revisions"]
                    if item["id"] == experiment_revision_id
                ),
                None,
            )
            if experiment_revision_id
            else experiment["latest_revision"]
        )
        if not revision:
            raise KeyError("Experiment revision not found")
        spec = ExperimentSpec.model_validate(revision["requested_spec_json"])
        with self.database.connection() as connection:
            active = connection.execute(
                """
                SELECT COUNT(*) FROM workflow_stages s
                JOIN runs r ON r.id = s.run_id
                JOIN variants v ON v.id = r.variant_id
                JOIN experiment_revisions er ON er.id = v.experiment_revision_id
                WHERE er.experiment_id = ? AND er.id = ?
                  AND s.status IN ('SUBMITTING','SUBMITTED','PENDING_SLURM','RUNNING','CANCELLING')
                """,
                (experiment_id, revision["id"]),
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT s.id, s.run_id FROM workflow_stages s
                JOIN runs r ON r.id = s.run_id
                JOIN variants v ON v.id = r.variant_id
                JOIN experiment_revisions er ON er.id = v.experiment_revision_id
                WHERE er.experiment_id = ? AND er.id = ?
                  AND s.status IN ('DRAFT','PENDING','RETRY_PENDING')
                ORDER BY s.created_at
                """,
                (experiment_id, revision["id"]),
            ).fetchall()
        slots = max(0, spec.sweep.max_parallel - int(active))
        submissions: list[dict[str, Any]] = []
        for row in rows[:slots]:
            result = self._submit_stage(row["run_id"], row["id"], gateway)
            submissions.append(result)
        return submissions

    def _local_capsule(self, run_id: str, compiled: CompiledSlurmJob) -> Path:
        root = LOCAL_CAPSULE_ROOT / run_id
        root.mkdir(parents=True, exist_ok=True)
        for name, content in compiled.files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        (root / "job.sbatch").write_text(compiled.script, encoding="utf-8")
        return root

    def _save_submission_script(self, run_id: str, attempt_id: str, script: str) -> dict[str, Any]:
        """Keep the exact transport payload independently of later run capsules."""
        path = LOCAL_CAPSULE_ROOT / run_id / "submissions" / f"{attempt_id}.sbatch"
        content = script.encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as target:
                target.write(content)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError("The saved submission script differs from this attempt")
        existing = next((item for item in self.database.list_artifacts(run_id, artifact_type="SUBMISSION_SCRIPT")
                         if item.get("metadata_json", {}).get("attempt_id") == attempt_id), None)
        if existing:
            if existing.get("sha256") != digest:
                raise ValueError("The saved submission checksum differs from this attempt")
            return existing
        return self.database.create_artifact(run_id, artifact_type="SUBMISSION_SCRIPT", path=str(path),
            sha256=digest, size_bytes=len(content), retention_policy="preserve", metadata={"attempt_id": attempt_id, "location": "submission_host"})

    def recover_run_submission(self, run_id: str, gateway: str) -> dict[str, Any]:
        with self._reconcile_lock:
            run = self.database.get_run(run_id)
            if not run:
                raise KeyError("Run not found")
            stage = next((item for item in reversed(run["stages"]) if item["stage_type"] == "TRAIN"), None)
            attempts = [item for item in run["attempts"] if stage and item["stage_id"] == stage["id"]]
            attempt = max(attempts, key=lambda item: item["attempt_number"], default=None)
            if attempt and attempt.get("slurm_job_id"):
                return {"run_id": run_id, "status": run["status"], "slurm_job_id": attempt["slurm_job_id"]}
            if not attempt or stage["status"] != "SUBMITTING" or attempt["status"] != "SUBMITTING" or not str(attempt.get("slurm_reason") or "").startswith("Submission outcome unknown"):
                raise ValueError("Only an unconfirmed training submission can be recovered. Cancelled submissions cannot be restarted here.")
            selected_gateway = self.cluster.resolve_gateway(gateway) if gateway == "auto" else self.cluster.candidates(gateway)[0]
            self.database.update_job_attempt(attempt["id"], gateway=selected_gateway)
            try:
                submission = self.cluster.recover_submission(run_id, attempt["id"], selected_gateway)
                if submission is None:
                    artifact = next((item for item in self.database.list_artifacts(run_id, artifact_type="SUBMISSION_SCRIPT")
                                     if item.get("metadata_json", {}).get("attempt_id") == attempt["id"]), None)
                    if not artifact:
                        raise ValueError("No accepted job was found and this older attempt has no saved submission script. Its original script must be restored before upload recovery.")
                    path = Path(artifact["path"])
                    expected = (LOCAL_CAPSULE_ROOT / run_id / "submissions" / f"{attempt['id']}.sbatch").resolve()
                    if path.resolve() != expected:
                        raise ValueError("Saved submission path does not match this attempt")
                    content = path.read_bytes()
                    if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
                        raise ValueError("Saved submission script changed; restore the original before recovery")
                    submission = self.cluster.submit_script(content.decode("utf-8"), run_id, selected_gateway,
                                                            submission_key=attempt["id"])
            except (ClusterError, OSError, ValueError) as error:
                self.database.update_job_attempt(attempt["id"], slurm_reason=f"Submission outcome unknown: recovery did not complete: {error}")
                raise
            self._record_recovered_submission({**attempt, "stage_id": stage["id"], "stage_type": "TRAIN",
                "stage_status": stage["status"], "run_id": run_id, "run_started_at": run.get("started_at"), "evaluation_id": None}, submission)
            return {"run_id": run_id, "status": self.database.get_run(run_id)["status"], "slurm_job_id": submission.job_id, "gateway": submission.gateway}

    def _preserve_cancelling_submission(
        self,
        *,
        run_id: str,
        stage: Mapping[str, Any],
        evaluation: Mapping[str, Any] | None,
        attempt_id: str,
        attempt_updates: Mapping[str, Any],
        event_type: str,
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        is_evaluation = str(stage.get("stage_type") or "").upper() == "EVALUATE"
        return self.database.transition_workflow_state(
            attempt_id=attempt_id,
            attempt_updates=attempt_updates,
            stage_id=str(stage["id"]),
            stage_updates={"status": "CANCELLING", "completed_at": None},
            run_id=None if is_evaluation else run_id,
            run_updates=None if is_evaluation else {
                "status": "CANCELLING",
                "completed_at": None,
            },
            evaluation_id=str(evaluation["id"]) if evaluation else None,
            evaluation_updates={"status": "CANCELLING", "completed_at": None}
            if evaluation else None,
            event={
                "entity_type": "evaluation" if evaluation else "run",
                "entity_id": str(evaluation["id"]) if evaluation else run_id,
                "event_type": event_type,
                "old_status": "CANCELLING",
                "new_status": "CANCELLING",
                "details": {**dict(details), "cancellation_requested": True},
            },
        )

    def _finalize_cancelled_before_submission(
        self,
        *,
        run_id: str,
        stage: Mapping[str, Any],
        evaluation: Mapping[str, Any] | None,
        attempt_id: str,
        error: Exception,
    ) -> dict[str, Any]:
        completed = utc_now()
        is_evaluation = str(stage.get("stage_type") or "").upper() == "EVALUATE"
        return self.database.transition_workflow_state(
            attempt_id=attempt_id,
            attempt_updates={
                "status": "CANCELLED",
                "slurm_reason": f"Cancelled before Slurm accepted the job: {error}",
                "finished_at": completed,
            },
            stage_id=str(stage["id"]),
            stage_updates={"status": "CANCELLED", "completed_at": completed},
            run_id=None if is_evaluation else run_id,
            run_updates=None if is_evaluation else {
                "status": "CANCELLED",
                "completed_at": completed,
            },
            evaluation_id=str(evaluation["id"]) if evaluation else None,
            evaluation_updates={"status": "CANCELLED", "completed_at": completed}
            if evaluation else None,
            event={
                "entity_type": "evaluation" if evaluation else "run",
                "entity_id": str(evaluation["id"]) if evaluation else run_id,
                "event_type": "CANCELLED_BEFORE_SUBMISSION",
                "old_status": "CANCELLING",
                "new_status": "CANCELLED",
                "details": {"attempt_id": attempt_id, "submission_error": str(error)},
            },
        )

    def _signal_stage_cancellation(
        self,
        *,
        entity_type: str,
        entity_id: str,
        attempt: Mapping[str, Any],
        record_event: bool,
    ) -> dict[str, Any]:
        job_id = str(attempt.get("slurm_job_id") or "")
        gateway = str(attempt.get("gateway") or "auto")
        if not job_id:
            return {"ok": True, "pending_job_id": True, "gateway": gateway}
        try:
            host = self.cluster.cancel(job_id, gateway)
        except Exception as error:
            message = sanitize(str(error))
            if record_event:
                self.database.record_event(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    event_type="CANCEL_SIGNAL_FAILED",
                    old_status="CANCELLING",
                    new_status="CANCELLING",
                    details={
                        "attempt_id": attempt.get("id"),
                        "job_id": job_id,
                        "gateway": gateway,
                        "error": message,
                    },
                )
            return {
                "ok": False,
                "job_id": job_id,
                "gateway": gateway,
                "error": message,
            }
        if record_event:
            self.database.record_event(
                entity_type=entity_type,
                entity_id=entity_id,
                event_type="CANCEL_SIGNAL_SENT",
                old_status="CANCELLING",
                new_status="CANCELLING",
                details={
                    "attempt_id": attempt.get("id"),
                    "job_id": job_id,
                    "gateway": host,
                },
            )
        return {"ok": True, "job_id": job_id, "gateway": host}

    def _submit_stage(
        self,
        run_id: str,
        stage_id: str,
        gateway: str,
        *,
        manual_mode: str | None = None,
        resume_checkpoint: str | None = None,
        resume_checkpoint_id: str | None = None,
        pinned_execution: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self.database.get_run(run_id)
        if not run:
            raise KeyError("Run not found")
        stage = next(item for item in run["stages"] if item["id"] == stage_id)
        is_evaluation_stage = stage["stage_type"] == "EVALUATE"
        evaluation = next(
            (item for item in run["evaluations"] if item.get("stage_id") == stage_id),
            None,
        )
        spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
        plan: AdapterPlan | None = None
        preserve_pinned_plan = False
        pinned_repository_inputs: dict[str, Any] | None = None
        execution_provenance: dict[str, Any] = {
            "policy": "pinned_variant",
            "source": "variants.resolved_spec_json",
            "source_spec_sha256": canonical_sha256(run["resolved_spec_json"]),
            "checkpoint": None,
            "transformations": [],
        }
        if is_evaluation_stage:
            config = stage["resolved_config_json"]
            stored_spec = config.get("spec") if isinstance(config, Mapping) else None
            stored_plan = config.get("plan") if isinstance(config, Mapping) else None
            if not isinstance(stored_spec, Mapping) or not isinstance(stored_plan, Mapping):
                message = "stored evaluation adapter plan is missing"
                self.database.transition_workflow_state(
                    stage_id=stage_id,
                    stage_updates={"status": "BLOCKED", "completed_at": utc_now()},
                    evaluation_id=evaluation["id"] if evaluation else None,
                    evaluation_updates={"status": "BLOCKED", "completed_at": utc_now()}
                    if evaluation else None,
                    event={
                        "entity_type": "evaluation" if evaluation else "run",
                        "entity_id": evaluation["id"] if evaluation else run_id,
                        "event_type": "EVALUATION_PLAN_BLOCKED",
                        "new_status": "BLOCKED",
                        "details": {"error": message},
                    },
                )
                return {
                    "run_id": run_id,
                    "stage_id": stage_id,
                    "status": "BLOCKED",
                    "blockers": [message],
                }
            spec = ExperimentSpec.model_validate(stored_spec)
            plan_payload = copy.deepcopy(dict(stored_plan))
            plan_payload.pop("runnable", None)
            plan = AdapterPlan.model_validate(plan_payload)
            preserve_pinned_plan = True
            execution_provenance = {
                "policy": "strict_pinned_evaluation",
                "source": "workflow_stages.resolved_config_json",
                "source_spec_sha256": config.get("spec_sha256")
                or canonical_sha256(stored_spec),
                "adapter_resolution": {
                    "training": copy.deepcopy(config.get("training_adapter")),
                    "evaluator": copy.deepcopy(config.get("evaluator_adapter")),
                },
                "checkpoint": copy.deepcopy(config.get("checkpoint")),
                "evaluation_context_sha256": config.get("context_sha256"),
                "evaluation_plan_sha256": config.get("plan_sha256"),
                "transformations": [],
            }
            has_prior_attempt = any(
                attempt["stage_id"] == stage_id for attempt in run["attempts"]
            )
            if has_prior_attempt and stage["auto_resume"] and manual_mode != "retry":
                execution_provenance["transformations"].append(
                    {
                        "kind": "evaluation_ledger_resume",
                        "progress_path": (config.get("context") or {}).get("progress_path"),
                    }
                )
        if manual_mode not in {None, "resume", "replay_initial"}:
            raise ValueError("clean retry is deprecated; create a new run instead")
        if stage["stage_type"] == "TRAIN" and manual_mode in {
            "resume", "replay_initial",
        }:
            evidence = pinned_execution
            if not isinstance(evidence, Mapping):
                evidence, evidence_error = _pinned_training_execution(
                    run,
                    stage,
                    prefer_initial_attempt=manual_mode == "replay_initial",
                )
                if evidence is None:
                    raise ValueError(
                        "resume requires immutable pinned execution evidence: "
                        f"{evidence_error}"
                    )
            stored_spec = evidence.get("resolved_spec")
            stored_plan = evidence.get("plan")
            if not isinstance(stored_spec, Mapping) or not isinstance(stored_plan, Mapping):
                raise ValueError("resume requires a pinned spec and adapter plan")
            payload = copy.deepcopy(dict(stored_spec))
            pinned_spec = ExperimentSpec.model_validate(stored_spec)
            if manual_mode == "resume":
                payload.setdefault("native", {}).setdefault("config", {})[
                    "initial_checkpoint"
                ] = resume_checkpoint
                payload.setdefault("train", {}).setdefault("checkpoint", {})[
                    "auto_resume"
                ] = True
            spec = ExperimentSpec.model_validate(payload)
            plan_payload = copy.deepcopy(dict(stored_plan))
            # ``runnable`` is a derived API field, not adapter-plan input.
            plan_payload.pop("runnable", None)
            plan = AdapterPlan.model_validate(plan_payload)
            plan, adapter_compatibility = apply_pinned_adapter_plan_compatibility(
                pinned_spec, plan
            )
            if manual_mode == "resume":
                plan.native_config["initial_checkpoint"] = resume_checkpoint
            preserve_pinned_plan = True
            repository_inputs = evidence.get("repository_inputs")
            if repository_inputs is not None:
                if not isinstance(repository_inputs, Mapping):
                    raise ValueError("pinned repository-input evidence is invalid")
                pinned_repository_inputs = copy.deepcopy(dict(repository_inputs))
            execution_provenance = {
                "policy": (
                    "strict_pinned_resume"
                    if manual_mode == "resume"
                    else "strict_pinned_initial_replay"
                ),
                "source": evidence.get("source"),
                "source_attempt_id": evidence.get("source_attempt_id"),
                "source_spec_sha256": evidence.get("source_spec_sha256")
                or canonical_sha256(stored_spec),
                "source_plan_sha256": evidence.get("source_plan_sha256")
                or canonical_sha256(plan_payload),
                "adapter_resolution": {
                    "pinned": {
                        "id": spec.source.adapter_id,
                        "version_id": spec.source.adapter_version_id,
                        "version": spec.source.adapter_version,
                        "slug": spec.source.adapter,
                        "manifest_sha256": spec.source.adapter_manifest_sha256,
                    },
                    "current": None,
                },
                "checkpoint": (
                    {"id": resume_checkpoint_id, "path": resume_checkpoint}
                    if manual_mode == "resume"
                    else None
                ),
                "transformations": (
                    [adapter_compatibility] if adapter_compatibility else []
                ),
            }
        if plan is None:
            plan = resolve_adapter_plan(spec)
        repository_argument_validation: dict[str, Any] | None = None
        if not is_evaluation_stage and not plan.blockers:
            try:
                repository_argument_validation = self._repository_argument_validation(
                    spec, plan, gateway
                )
                native_blocker = (
                    validation_failure_reason(repository_argument_validation)
                    if repository_argument_validation
                    else None
                )
            except ValueError as error:
                native_blocker = (
                    "repository-native argument validation is unavailable: "
                    f"{error}"
                )
            if native_blocker:
                plan.blockers.append(native_blocker)
        if plan.blockers:
            completed = utc_now()
            self.database.transition_workflow_state(
                stage_id=stage_id,
                stage_updates={"status": "BLOCKED", "completed_at": completed},
                run_id=None if is_evaluation_stage else run_id,
                run_updates=None if is_evaluation_stage else {"status": "BLOCKED", "completed_at": completed},
                evaluation_id=evaluation["id"] if evaluation else None,
                evaluation_updates={"status": "BLOCKED", "completed_at": completed}
                if evaluation else None,
                event={
                    "entity_type": "evaluation" if evaluation else "run",
                    "entity_id": evaluation["id"] if evaluation else run_id,
                    "event_type": "ADAPTER_BLOCKED",
                    "new_status": "BLOCKED",
                    "details": {"blockers": plan.blockers, "todos": plan.todos},
                },
            )
            return {"run_id": run_id, "stage_id": stage_id, "status": "BLOCKED", "blockers": plan.blockers}
        spec, snapshot_id = self._auto_queue(spec, plan, gateway)
        if not preserve_pinned_plan:
            plan = resolve_adapter_plan(spec)
        if stage["stage_type"] == "TRAIN" and manual_mode == "resume" and resume_checkpoint:
            plan.native_config["initial_checkpoint"] = resume_checkpoint
        pinned_manifest = AdapterManifest.model_validate(spec.source.adapter_manifest)
        resolved_spec_document = spec.model_dump(mode="json", by_alias=True)
        if pinned_repository_inputs is None:
            repository_options = self._validate_repository_input_choices(
                resolved_spec_document,
                spec.source.model_dump(mode="json"),
                pinned_manifest,
                gateway,
            )
            selected_repository_inputs = _selected_repository_input_metadata(
                resolved_spec_document, repository_options
            )
        else:
            selected_repository_inputs = pinned_repository_inputs
        attempt_snapshot = self._attempt_execution_snapshot(
            spec, plan, execution_provenance, selected_repository_inputs
        )
        if repository_argument_validation is not None:
            attempt_snapshot["repository_argument_validation"] = (
                repository_argument_validation
            )
        if not is_evaluation_stage:
            unresolved = _required_unresolved_repository_defaults(
                attempt_snapshot["common_hyperparameters"], pinned_manifest
            )
            if unresolved:
                raise ValueError(
                    "selected repository config did not expose exact pinned defaults for: "
                    + ", ".join(unresolved)
                )
        capsule_files = dict(plan.capsule_files)
        native_tracking_runtime = (
            self._native_tracking_runtime(
                spec,
                plan,
                run_id,
                stage_id,
                continuation=manual_mode in {"resume", "replay_initial"},
            )
            if not is_evaluation_stage
            else {
                "environment": {},
                "secret_environment_files": {},
                "secret_contents": {},
                "run_ids": {},
            }
        )
        if native_tracking_runtime["run_ids"]:
            attempt_snapshot["native_tracking_runtime"] = {
                "environment": native_tracking_runtime["environment"],
                "secret_environment": sorted(
                    native_tracking_runtime["secret_environment_files"]
                ),
                "run_ids": native_tracking_runtime["run_ids"],
            }
        attempt_snapshot_sha256 = content_sha256(attempt_snapshot)
        if evaluation:
            evaluation_context = copy.deepcopy(
                stage["resolved_config_json"].get("context") or {}
            )
            evaluation_context.update(
                {
                    "evaluation_id": evaluation["id"],
                    "stage_id": stage_id,
                }
            )
            capsule_files["evaluation-context.json"] = json.dumps(
                evaluation_context, indent=2, sort_keys=True
            ) + "\n"
        capsule_files["attempt-snapshot.json"] = json.dumps(
            attempt_snapshot, indent=2, sort_keys=True
        ) + "\n"
        try:
            compiled = compile_sbatch(
                spec,
                plan,
                run_id=run_id,
                stage="eval" if stage["stage_type"] == "EVALUATE" else "train",
                capsule_files=capsule_files,
                stage_auto_resume=(
                    bool(stage["auto_resume"])
                    if manual_mode != "replay_initial"
                    else False
                ),
                runtime_environment=native_tracking_runtime["environment"],
                secret_environment_files=native_tracking_runtime[
                    "secret_environment_files"
                ],
                native_tracking_run_ids=native_tracking_runtime["run_ids"],
                native_tracking_resume=manual_mode in {"resume", "replay_initial"},
            )
        except SlurmCompileError as error:
            completed = utc_now()
            self.database.transition_workflow_state(
                stage_id=stage_id,
                stage_updates={"status": "BLOCKED", "completed_at": completed},
                run_id=None if is_evaluation_stage else run_id,
                run_updates=None if is_evaluation_stage else {"status": "BLOCKED", "completed_at": completed},
                evaluation_id=evaluation["id"] if evaluation else None,
                evaluation_updates={"status": "BLOCKED", "completed_at": completed}
                if evaluation else None,
                event={
                    "entity_type": "evaluation" if evaluation else "run",
                    "entity_id": evaluation["id"] if evaluation else run_id,
                    "event_type": "SBATCH_COMPILE_BLOCKED",
                    "new_status": "BLOCKED",
                    "details": {"error": str(error)},
                },
            )
            return {"run_id": run_id, "stage_id": stage_id, "status": "BLOCKED", "blockers": [str(error)]}

        attempt = self.database.claim_stage_and_create_job_attempt(
            stage_id,
            status="SUBMITTING",
            gateway=gateway,
            account=spec.resources.account,
            partition_name=spec.resources.partition,
            gpu_type=compiled.gpu_type,
            gpu_count=compiled.gpu_count,
            cpu_count=spec.resources.cpus_per_task,
            memory_mb=spec.resources.memory_gb * 1024,
            time_limit_seconds=parse_slurm_duration(spec.resources.time_limit),
            cluster_snapshot_id=snapshot_id,
            resume_checkpoint_id=resume_checkpoint_id,
            execution_snapshot_json=attempt_snapshot,
            execution_snapshot_sha256=attempt_snapshot_sha256,
            sbatch_path=f"{WORK_ROOT}/jobs/runs/{run_id}/job.sbatch",
            stdout_path=compiled.stdout_path_template,
            stderr_path=compiled.stderr_path_template,
        )
        if attempt is None:
            current = next(item for item in self.database.list_stages(run_id) if item["id"] == stage_id)
            return {
                "run_id": run_id,
                "stage_id": stage_id,
                "status": current["status"],
                "skipped": "stage is already claimed by another dispatcher",
            }
        selected_gateway = gateway if gateway != "auto" else spec.resources.gateway
        staged_secret_paths: list[str] = []

        def cleanup_staged_secrets(cleanup_gateway: str) -> None:
            for relative_path in staged_secret_paths:
                try:
                    self.cluster.remove_capsule_file(
                        run_id, relative_path, cleanup_gateway
                    )
                except Exception:
                    pass

        try:
            native_providers = set(native_tracking_runtime["run_ids"])
            if native_providers:
                self._start_tracking(
                    spec,
                    run,
                    LOCAL_CAPSULE_ROOT / run_id,
                    "",
                    attempt_snapshot=attempt_snapshot,
                    continuation_attempt_number=(
                        int(attempt["attempt_number"])
                        if manual_mode in {"resume", "replay_initial"}
                        else None
                    ),
                )
                self._require_native_tracking_bindings(run_id, native_providers)
            capsule = self._local_capsule(run_id, compiled)
            self._save_submission_script(run_id, attempt["id"], compiled.script)
            secret_gateway = selected_gateway
            for relative_path, content in native_tracking_runtime[
                "secret_contents"
            ].items():
                secret_gateway, _ = self.cluster.write_capsule_file(
                    run_id,
                    relative_path,
                    content,
                    secret_gateway,
                )
                staged_secret_paths.append(relative_path)
            test_gateway, test_output = self.cluster.test_script(
                compiled.script, secret_gateway
            )
            # Submit through the gateway that successfully parsed and validated this
            # exact script. Re-resolving here can select a login node whose Slurm
            # binaries respond while its shared-filesystem access is stalled.

            submission_options: dict[str, Any] = {
                "submission_key": attempt["id"]
            }
            forwarded_environment = approved_operator_environment(
                enabled=is_evaluation_stage
            )
            if forwarded_environment:
                submission_options["forwarded_environment"] = forwarded_environment
            submission = self.cluster.submit_script(
                compiled.script,
                run_id,
                test_gateway,
                **submission_options,
            )
        except SubmissionOutcomeUnknown as error:
            transition = self.database.transition_workflow_state(
                attempt_id=attempt["id"],
                attempt_updates={
                    "status": "SUBMITTING",
                    "gateway": test_gateway,
                    "slurm_reason": f"Submission outcome unknown: {error}",
                },
                stage_id=stage_id,
                stage_updates={"status": "SUBMITTING", "completed_at": None},
                run_id=None if is_evaluation_stage else run_id,
                run_updates=None if is_evaluation_stage else {
                    "status": "SUBMITTING",
                    "completed_at": None,
                },
                evaluation_id=evaluation["id"] if evaluation else None,
                evaluation_updates={"status": "SUBMITTING", "completed_at": None}
                if evaluation else None,
                event={
                    "entity_type": "evaluation" if evaluation else "run",
                    "entity_id": evaluation["id"] if evaluation else run_id,
                    "event_type": "SUBMISSION_OUTCOME_UNKNOWN",
                    "new_status": "SUBMITTING",
                    "details": {"error": str(error), "attempt_id": attempt["id"]},
                },
                expected_stage_statuses=("SUBMITTING",),
            )
            if transition.get("applied") is False:
                current_status = str(transition["stage"].get("status") or "").upper()
                if current_status == "CANCELLING":
                    self._preserve_cancelling_submission(
                        run_id=run_id,
                        stage=stage,
                        evaluation=evaluation,
                        attempt_id=attempt["id"],
                        attempt_updates={
                            "status": "CANCELLING",
                            "gateway": test_gateway,
                            "slurm_reason": f"Submission outcome unknown during cancellation: {error}",
                        },
                        event_type="SUBMISSION_OUTCOME_UNKNOWN",
                        details={"error": str(error), "attempt_id": attempt["id"]},
                    )
                    return {
                        "run_id": run_id,
                        "stage_id": stage_id,
                        "status": "CANCELLING",
                        "warning": str(error),
                    }
                return {
                    "run_id": run_id,
                    "stage_id": stage_id,
                    "status": current_status,
                    "skipped": "workflow state changed while submission was in flight",
                }
            return {
                "run_id": run_id,
                "stage_id": stage_id,
                "status": "SUBMITTING",
                "warning": str(error),
            }
        except (ClusterError, OSError, ValueError) as error:
            cleanup_staged_secrets(selected_gateway)
            completed = utc_now()
            transition = self.database.transition_workflow_state(
                attempt_id=attempt["id"],
                attempt_updates={
                    "status": "SUBMISSION_FAILED",
                    "slurm_reason": str(error),
                    "finished_at": completed,
                },
                stage_id=stage_id,
                stage_updates={"status": "FAILED", "completed_at": completed},
                run_id=None if is_evaluation_stage else run_id,
                run_updates=None if is_evaluation_stage else {"status": "FAILED", "completed_at": completed},
                evaluation_id=evaluation["id"] if evaluation else None,
                evaluation_updates={"status": "FAILED", "completed_at": completed}
                if evaluation else None,
                event={
                    "entity_type": "evaluation" if evaluation else "run",
                    "entity_id": evaluation["id"] if evaluation else run_id,
                    "event_type": "SUBMISSION_FAILED",
                    "new_status": "FAILED",
                    "details": {"error": str(error), "attempt_id": attempt["id"]},
                },
                expected_stage_statuses=("SUBMITTING",),
            )
            if transition.get("applied") is False:
                current_status = str(transition["stage"].get("status") or "").upper()
                if current_status == "CANCELLING":
                    self._finalize_cancelled_before_submission(
                        run_id=run_id,
                        stage=stage,
                        evaluation=evaluation,
                        attempt_id=attempt["id"],
                        error=error,
                    )
                    return {
                        "run_id": run_id,
                        "stage_id": stage_id,
                        "status": "CANCELLED",
                    }
                return {
                    "run_id": run_id,
                    "stage_id": stage_id,
                    "status": current_status,
                    "skipped": "workflow state changed while submission was in flight",
                }
            return {"run_id": run_id, "stage_id": stage_id, "status": "FAILED", "error": str(error)}

        stdout = resolve_slurm_log_path(
            compiled.stdout_path_template, submission.job_id
        )
        stderr = resolve_slurm_log_path(
            compiled.stderr_path_template, submission.job_id
        )
        if stdout is None or stderr is None:
            raise RuntimeError("compiled Slurm log path templates could not be resolved")
        attempt_directory = f"{submission.run_directory}/attempts/{submission.job_id}"
        archived_script_path = f"{attempt_directory}/job.sbatch"
        submitted_at = utc_now()
        submission_transition = self.database.transition_workflow_state(
            attempt_id=attempt["id"],
            attempt_updates={
                "status": "SUBMITTED",
                "slurm_job_id": submission.job_id,
                "gateway": submission.gateway,
                "sbatch_path": archived_script_path,
                "stdout_path": stdout,
                "stderr_path": stderr,
                "submitted_at": submitted_at,
                "slurm_reason": test_output.strip()[:2000],
            },
            stage_id=stage_id,
            stage_updates={"status": "SUBMITTED", "started_at": submitted_at},
            run_id=None if is_evaluation_stage else run_id,
            run_updates=None if is_evaluation_stage else {
                "status": "SUBMITTED",
                "started_at": run.get("started_at") or submitted_at,
            },
            evaluation_id=evaluation["id"] if evaluation else None,
            evaluation_updates={
                "status": "SUBMITTED",
                "started_at": submitted_at,
                "completed_at": None,
            } if evaluation else None,
            event={
                "entity_type": "evaluation" if evaluation else "run",
                "entity_id": evaluation["id"] if evaluation else run_id,
                "event_type": "JOB_SUBMITTED",
                "old_status": evaluation["status"] if evaluation else run["status"],
                "new_status": "SUBMITTED",
                "details": {
                    "job_id": submission.job_id,
                    "gateway": submission.gateway,
                    "attempt": attempt["attempt_number"],
                    "recovered": submission.recovered,
                },
            },
            expected_stage_statuses=("SUBMITTING",),
        )
        cancellation_pending = submission_transition.get("applied") is False and str(
            submission_transition["stage"].get("status") or ""
        ).upper() == "CANCELLING"
        if cancellation_pending:
            self._preserve_cancelling_submission(
                run_id=run_id,
                stage=stage,
                evaluation=evaluation,
                attempt_id=attempt["id"],
                attempt_updates={
                    "status": "CANCELLING",
                    "slurm_job_id": submission.job_id,
                    "gateway": submission.gateway,
                    "sbatch_path": archived_script_path,
                    "stdout_path": stdout,
                    "stderr_path": stderr,
                    "submitted_at": submitted_at,
                    "slurm_reason": test_output.strip()[:2000],
                },
                event_type="JOB_SUBMITTED_AFTER_CANCEL_REQUEST",
                details={
                    "job_id": submission.job_id,
                    "gateway": submission.gateway,
                    "attempt": attempt["attempt_number"],
                },
            )
        try:
            self.database.create_artifact(
                run_id,
                stage_id=stage_id,
                artifact_type="SBATCH",
                path=archived_script_path,
                sha256=compiled.script_sha256,
                retention_policy="KEEP",
            )
            self.database.create_manifest(
                run_id,
                attempt_id=attempt["id"],
                manifest_type="RESOLVED",
                schema_version="skynet.rl2/v1",
                path=f"{attempt_directory}/resolved-spec.json",
                sha256=compiled.spec_sha256,
            )
        except sqlite3.IntegrityError:
            pass
        if not is_evaluation_stage:
            self._start_tracking(
                spec,
                run,
                capsule,
                submission.job_id,
                attempt_snapshot=attempt_snapshot,
                continuation_attempt_number=(
                    int(attempt["attempt_number"])
                    if manual_mode in {"resume", "replay_initial"}
                    else None
                ),
            )
        cancellation_signal = None
        if cancellation_pending:
            cancellation_signal = self._signal_stage_cancellation(
                entity_type="evaluation" if evaluation else "run",
                entity_id=str(evaluation["id"]) if evaluation else run_id,
                attempt={
                    **attempt,
                    "slurm_job_id": submission.job_id,
                    "gateway": submission.gateway,
                },
                record_event=True,
            )
        return {
            "run_id": run_id,
            "stage_id": stage_id,
            "status": "CANCELLING" if cancellation_pending else "SUBMITTED",
            "job_id": submission.job_id,
            "gateway": submission.gateway,
            "script_path": archived_script_path,
            **({"cancellation": cancellation_signal} if cancellation_signal else {}),
        }

    @staticmethod
    def _tracking_provider_value(provider: Any, key: str, default: Any = None) -> Any:
        if isinstance(provider, Mapping):
            return provider.get(key, default)
        return getattr(provider, key, default)

    @staticmethod
    def _validate_tracking_endpoint(value: str, label: str) -> str:
        parsed = urllib.parse.urlsplit(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"{label} must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(f"{label} must not contain credentials; use Connections")
        if parsed.scheme == "http":
            hostname = (parsed.hostname or "").lower()
            try:
                is_loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                is_loopback = hostname == "localhost"
            if not is_loopback:
                raise ValueError(
                    f"{label} must use HTTPS; plaintext HTTP is allowed only for loopback"
                )
        for key, _ in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
            if re.search(r"(?i)(token|secret|password|api[_-]?key)", key):
                raise ValueError(f"{label} must not contain credential query parameters")
        return urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), parsed.query, "")
        )

    @staticmethod
    def _environment_credentials(provider: str) -> dict[str, str]:
        if provider == "wandb":
            values = {"api_key": os.environ.get("WANDB_API_KEY")}
        else:
            values = {
                "token": os.environ.get("MLFLOW_TRACKING_TOKEN"),
                "username": os.environ.get("MLFLOW_TRACKING_USERNAME"),
                "password": os.environ.get("MLFLOW_TRACKING_PASSWORD"),
            }
        return {key: value for key, value in values.items() if value}

    @staticmethod
    def _credential_authentication(provider: str, credentials: Mapping[str, str]) -> str:
        if provider == "wandb":
            return "api_key" if credentials.get("api_key") else "none"
        if credentials.get("token"):
            return "token"
        if credentials.get("username"):
            return "basic"
        return "none"

    def _ensure_tracking_credentials_restored(self) -> None:
        if self._credentials_restored:
            return
        with self._credential_restore_lock:
            if self._credentials_restored:
                return
            self._credentials_restored = True
            for provider in ("wandb", "mlflow"):
                connection = self.database.get_tracking_connection(provider)
                if not connection or self.credentials.get(provider):
                    continue
                config = connection.get("config_json") or {}
                expected_source = config.get("credential_source")
                authentication = config.get("authentication")
                environment_credentials = self._environment_credentials(provider)
                if expected_source == "environment":
                    if environment_credentials:
                        self.credentials.mark_connected(provider)
                    else:
                        self.credentials.mark_error(
                            provider,
                            "The configured environment credential is no longer available; reconnect",
                        )
                    continue
                if provider == "mlflow" and authentication == "none":
                    self.credentials.mark_connected(provider)
                    continue
                if expected_source == "session":
                    self.credentials.mark_error(
                        provider,
                        "The session credential was cleared by the backend restart; reconnect",
                    )
                    continue
                try:
                    stored = self.credential_store.load(provider)
                except CredentialStoreError as error:
                    if environment_credentials and expected_source is None:
                        self.credentials.mark_connected(provider)
                    else:
                        self.credentials.mark_error(provider, error)
                    continue
                if stored is None:
                    if environment_credentials and expected_source is None:
                        self.credentials.mark_connected(provider)
                    elif expected_source == "credential_store":
                        self.credentials.mark_error(
                            provider,
                            "The saved OS credential is no longer available; reconnect",
                        )
                    continue
                endpoint = str(connection.get("endpoint") or "")
                if stored.endpoint.rstrip("/") != endpoint.rstrip("/"):
                    self.credentials.mark_error(
                        provider,
                        "The saved credential is pinned to a different endpoint; reconnect",
                    )
                    continue
                credentials = {
                    key: value
                    for key, value in stored.credentials.items()
                    if key in ({"api_key"} if provider == "wandb" else {"token", "username", "password"})
                }
                restored_authentication = self._credential_authentication(provider, credentials)
                if restored_authentication == "none" or (
                    authentication not in (None, restored_authentication)
                ):
                    self.credentials.mark_error(
                        provider,
                        "The saved credential does not match the validated authentication mode; reconnect",
                    )
                    continue
                self.credentials.replace(
                    provider, credentials, source="credential_store"
                )
                self.credentials.mark_connected(provider)

    def _activate_validated_credentials(
        self,
        provider: str,
        endpoint: str,
        credentials: Mapping[str, str | None],
        *,
        origin: str | None,
        remember: bool,
        replace_existing: bool,
    ) -> str | None:
        values = {
            key: str(value)
            for key, value in credentials.items()
            if value is not None and str(value)
        }
        current_source = self.credentials.source(provider)
        connection = self.database.get_tracking_connection(provider) or {}
        configured_source = (connection.get("config_json") or {}).get("credential_source")
        persistent = current_source == "credential_store" or configured_source == "credential_store"
        if not values:
            if replace_existing:
                if persistent:
                    self.credential_store.delete(provider)
                self.credentials.clear(provider)
            return None
        if origin == "environment":
            return "environment"
        if remember:
            self.credential_store.save(provider, endpoint, values)
            self.credentials.replace(provider, values, source="credential_store")
            return "credential_store"
        if persistent:
            self.credential_store.delete(provider)
        self.credentials.replace(provider, values, source="session")
        return "session"

    def _mlflow_settings(self, provider: Any | None = None) -> TrackingSettings:
        self._ensure_tracking_credentials_restored()
        settings = TrackingSettings.from_env()
        connection = self.database.get_tracking_connection("mlflow") or {}
        credentials = self.credentials.get("mlflow")
        requested_uri = self._tracking_provider_value(provider, "tracking_uri") if provider else None
        trusted_uri = connection.get("endpoint") or settings.tracking_uri
        if requested_uri and trusted_uri:
            requested_uri = self._validate_tracking_endpoint(
                str(requested_uri), "MLflow tracking URI"
            )
            if requested_uri.rstrip("/") != str(trusted_uri).rstrip("/"):
                raise ValueError(
                    "experiment MLflow URI does not match the validated connection endpoint"
                )
        tracking_uri = trusted_uri or requested_uri
        return replace(
            settings,
            tracking_uri=tracking_uri,
            token=credentials.get("token") or settings.token,
            username=credentials.get("username") or settings.username,
            password=credentials.get("password") or settings.password,
            verify_tls=bool((connection.get("config_json") or {}).get("verify_tls", settings.verify_tls)),
        )

    def _wandb_settings(self, provider: Any | None = None) -> WandBSettings:
        self._ensure_tracking_credentials_restored()
        settings = WandBSettings.from_env()
        connection = self.database.get_tracking_connection("wandb") or {}
        credentials = self.credentials.get("wandb")
        requested_url = (
            self._tracking_provider_value(provider, "base_url") if provider else None
        )
        trusted_url = connection.get("endpoint") or settings.base_url
        if requested_url and trusted_url:
            requested_url = self._validate_tracking_endpoint(
                str(requested_url), "W&B base URL"
            )
            if requested_url.rstrip("/") != str(trusted_url).rstrip("/"):
                raise ValueError(
                    "experiment W&B base URL does not match the validated connection endpoint"
                )
        base_url = trusted_url or requested_url
        entity = (
            self._tracking_provider_value(provider, "entity") if provider else None
        ) or connection.get("workspace") or settings.entity
        return replace(
            settings,
            base_url=base_url,
            entity=entity,
            api_key=credentials.get("api_key") or settings.api_key,
            verify_tls=bool((connection.get("config_json") or {}).get("verify_tls", settings.verify_tls)),
        )

    def tracking_connections(self) -> dict[str, Any]:
        self._ensure_tracking_credentials_restored()
        results: dict[str, Any] = {}
        for provider in ("wandb", "mlflow"):
            runtime = self.credentials.state(provider)
            credentials = self.credentials.get(provider)
            if provider == "wandb":
                settings = self._wandb_settings()
                environment_credential = bool(os.environ.get("WANDB_API_KEY"))
                public = {
                    "provider": provider,
                    "base_url": settings.public_dict()["base_url"],
                    "entity": settings.entity,
                    "configured": settings.configured,
                }
            else:
                settings = self._mlflow_settings()
                environment_credential = bool(
                    os.environ.get("MLFLOW_TRACKING_TOKEN")
                    or os.environ.get("MLFLOW_TRACKING_USERNAME")
                    or os.environ.get("MLFLOW_TRACKING_PASSWORD")
                )
                public = {
                    "provider": provider,
                    "tracking_uri": settings.public_dict()["tracking_uri"],
                    "configured": settings.configured,
                }
            connected = bool(runtime.get("connected"))
            configured = bool(public["configured"])
            status = (
                "error" if runtime.get("last_error")
                else "connected" if connected
                else "configured" if configured
                else "not_configured"
            )
            public.update({
                "connected": connected,
                "status": status,
                "credential_source": (
                    self.credentials.source(provider) if credentials
                    else "environment" if environment_credential
                    else None
                ),
                "validated_at": runtime.get("validated_at"),
                "last_error": runtime.get("last_error"),
            })
            results[provider] = public
        return {"connections": results}

    def configure_tracking_connection(
        self, provider: str, request: TrackingConnectionRequest
    ) -> dict[str, Any]:
        self._ensure_tracking_credentials_restored()
        if provider not in {"wandb", "mlflow"}:
            raise ValueError(f"unsupported tracking provider: {provider}")
        if provider == "wandb":
            if any((request.tracking_uri, request.token, request.username, request.password)):
                raise ValueError("W&B connection accepts only api_key, base_url, entity, and verify_tls")
            submitted_key = request.api_key is not None
            current_credentials = self.credentials.get("wandb")
            environment_credentials = self._environment_credentials("wandb")
            if submitted_key:
                key = request.api_key.get_secret_value() if request.api_key else ""
                origin = "session"
            elif current_credentials.get("api_key"):
                key = current_credentials["api_key"]
                origin = self.credentials.source("wandb")
            else:
                key = environment_credentials.get("api_key", "")
                origin = "environment" if key else None
            if not key:
                raise ValueError("W&B API key is required")
            base_url = self._validate_tracking_endpoint(
                request.base_url or "https://api.wandb.ai", "W&B base URL"
            )
            settings = replace(
                WandBSettings.from_env(),
                base_url=base_url,
                api_key=key,
                entity=request.entity or None,
                verify_tls=request.verify_tls,
            )
            try:
                bridge = WandBBridge(
                    LOCAL_CAPSULE_ROOT / ".tracking-connections" / "wandb", settings
                )
                identity = bridge.validate_connection()
                entity = request.entity or str(identity["entity"])
                bridge.validate_entity(entity)
            except Exception as error:
                safe_error = str(sanitize(str(error), secrets=(key,)))
                self.credentials.mark_error("wandb", safe_error)
                raise TrackingRequestError(safe_error) from error
            try:
                credential_source = self._activate_validated_credentials(
                    "wandb",
                    base_url,
                    {"api_key": key},
                    origin=origin,
                    remember=request.remember,
                    replace_existing=submitted_key,
                )
            except CredentialStoreError as error:
                self.credentials.mark_error("wandb", error)
                raise
            self.database.upsert_tracking_connection(
                "wandb",
                endpoint=base_url,
                workspace=entity,
                config={
                    "verify_tls": request.verify_tls,
                    "authentication": "api_key",
                    "credential_source": credential_source,
                },
            )
            self.credentials.mark_connected(
                "wandb", entity=entity, username=identity.get("username")
            )
        else:
            if any((request.api_key, request.base_url, request.entity)):
                raise ValueError(
                    "MLflow connection accepts tracking_uri, token, username, password, and verify_tls"
                )
            if not request.tracking_uri:
                raise ValueError("MLflow tracking URI is required")
            tracking_uri = self._validate_tracking_endpoint(
                request.tracking_uri, "MLflow tracking URI"
            )
            submitted_authentication = any(
                value is not None for value in (request.token, request.username, request.password)
            )
            current_credentials = self.credentials.get("mlflow")
            environment_credentials = self._environment_credentials("mlflow")
            if submitted_authentication:
                credentials = {
                    "token": request.token.get_secret_value() if request.token else None,
                    "username": request.username,
                    "password": request.password.get_secret_value() if request.password else None,
                }
                origin = "session"
            elif current_credentials:
                credentials = current_credentials
                origin = self.credentials.source("mlflow")
            else:
                credentials = environment_credentials
                origin = "environment" if credentials else None
            token = credentials.get("token")
            username = credentials.get("username")
            password = credentials.get("password")
            settings = replace(
                TrackingSettings.from_env(),
                tracking_uri=tracking_uri,
                token=token,
                username=username,
                password=password,
                verify_tls=request.verify_tls,
            )
            try:
                MLflowBridge(
                    LOCAL_CAPSULE_ROOT / ".tracking-connections" / "mlflow", settings
                ).validate_connection()
            except Exception as error:
                safe_error = str(sanitize(str(error), secrets=(token, password)))
                self.credentials.mark_error("mlflow", safe_error)
                raise TrackingRequestError(safe_error) from error
            try:
                credential_source = self._activate_validated_credentials(
                    "mlflow",
                    tracking_uri,
                    credentials,
                    origin=origin,
                    remember=request.remember,
                    replace_existing=submitted_authentication,
                )
            except CredentialStoreError as error:
                self.credentials.mark_error("mlflow", error)
                raise
            self.database.upsert_tracking_connection(
                "mlflow",
                endpoint=tracking_uri,
                config={
                    "verify_tls": request.verify_tls,
                    "authentication": self._credential_authentication("mlflow", credentials),
                    "credential_source": credential_source,
                },
            )
            self.credentials.mark_connected("mlflow")
        return {
            "connection": self.tracking_connections()["connections"][provider],
            "flush": self._flush_tracking_provider(provider),
        }

    def test_tracking_connection(self, provider: str) -> dict[str, Any]:
        if provider == "wandb":
            settings = self._wandb_settings()
            bridge: Any = WandBBridge(
                LOCAL_CAPSULE_ROOT / ".tracking-connections" / "wandb", settings
            )
        elif provider == "mlflow":
            settings = self._mlflow_settings()
            bridge = MLflowBridge(
                LOCAL_CAPSULE_ROOT / ".tracking-connections" / "mlflow", settings
            )
        else:
            raise ValueError(f"unsupported tracking provider: {provider}")
        try:
            result = bridge.validate_connection()
        except Exception as error:
            secrets = tuple(self.credentials.get(provider).values())
            safe_error = str(sanitize(str(error), secrets=secrets))
            self.credentials.mark_error(provider, safe_error)
            raise TrackingRequestError(safe_error) from error
        if provider == "wandb":
            entity = str(result["entity"])
            connection = self.database.get_tracking_connection("wandb")
            bridge.validate_entity(connection.get("workspace") if connection else entity)
            self.database.upsert_tracking_connection(
                "wandb",
                endpoint=(connection or {}).get("endpoint") or settings.base_url,
                workspace=(connection or {}).get("workspace") or entity,
                config=(connection or {}).get("config_json") or {
                    "verify_tls": settings.verify_tls
                },
            )
            self.credentials.mark_connected(
                provider, entity=entity, username=result.get("username")
            )
        else:
            connection = self.database.get_tracking_connection("mlflow")
            self.database.upsert_tracking_connection(
                "mlflow",
                endpoint=(connection or {}).get("endpoint") or settings.tracking_uri,
                config=(connection or {}).get("config_json") or {
                    "verify_tls": settings.verify_tls
                },
            )
            self.credentials.mark_connected(provider)
        return {
            "connection": self.tracking_connections()["connections"][provider],
            "flush": self._flush_tracking_provider(provider),
        }

    def disconnect_tracking_connection(self, provider: str) -> dict[str, Any]:
        self._ensure_tracking_credentials_restored()
        if provider not in {"wandb", "mlflow"}:
            raise ValueError(f"unsupported tracking provider: {provider}")
        connection = self.database.get_tracking_connection(provider) or {}
        configured_source = (connection.get("config_json") or {}).get("credential_source")
        credential_source = self.credentials.source(provider) or configured_source
        if credential_source == "credential_store":
            try:
                self.credential_store.delete(provider)
            except CredentialStoreError as error:
                self.credentials.mark_error(provider, error)
                raise
        self.credentials.clear(provider)
        self.database.delete_tracking_connection(provider)
        return {"connection": self.tracking_connections()["connections"][provider]}

    def _active_tracking_providers(self, spec: ExperimentSpec) -> list[Any]:
        if spec.tracking.providers:
            return [item for item in spec.tracking.providers if item.enabled]
        if spec.tracking.mlflow_tracking_uri:
            return [{
                "provider": "mlflow",
                "enabled": True,
                "tracking_uri": spec.tracking.mlflow_tracking_uri,
                "experiment": spec.tracking.mlflow_experiment,
                "run_name_template": None,
            }]
        return []

    def run_tracking_actions(
        self,
        run: Mapping[str, Any],
        connections: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return server-authorized retroactive tracking actions for one run."""

        current_connections = connections or self.tracking_connections()["connections"]
        linked = {
            str(item.get("provider") or "").lower()
            for item in run.get("tracking_links") or []
            if item.get("remote_id") and item.get("url")
        }
        labels = {"wandb": "W&B", "mlflow": "MLflow"}
        actions: dict[str, dict[str, Any]] = {}
        for provider, label in labels.items():
            connection = current_connections.get(provider) or {}
            if provider in linked or not (
                connection.get("connected") is True
                and str(connection.get("status") or "").lower() == "connected"
            ):
                continue
            run_id = str(run["id"])
            actions[provider] = {
                "provider": provider,
                "label": f"Connect {label}",
                "enabled": True,
                "reason": "",
                "method": "POST",
                "path": (
                    f"/api/runs/{urllib.parse.quote(run_id, safe='')}"
                    f"/tracking/{provider}/attach"
                ),
            }
        return actions

    def _retroactive_tracking_provider(
        self,
        provider: str,
        spec: ExperimentSpec,
        run: Mapping[str, Any],
        connection: Mapping[str, Any],
    ) -> dict[str, Any]:
        experiment_binding = next(
            (
                item
                for item in self.database.list_tracking_bindings(
                    "experiment", str(run["experiment_id"])
                )
                if item["provider"] == provider
            ),
            {},
        )
        metadata = experiment_binding.get("metadata_json") or {}
        if provider == "wandb":
            entity = connection.get("entity")
            if not entity:
                raise TrackingRequestError(
                    "W&B connection is healthy but has no validated entity"
                )
            return {
                "provider": "wandb",
                "enabled": True,
                "base_url": connection.get("base_url"),
                "entity": entity,
                "project": metadata.get("project") or spec.identity.experiment,
                "run_name_template": None,
            }
        if provider == "mlflow":
            tracking_uri = connection.get("tracking_uri")
            if not tracking_uri:
                raise TrackingRequestError(
                    "MLflow connection is healthy but has no validated tracking URI"
                )
            return {
                "provider": "mlflow",
                "enabled": True,
                "tracking_uri": tracking_uri,
                "experiment": metadata.get("experiment") or spec.identity.experiment,
                "run_name_template": None,
            }
        raise ValueError(f"unsupported tracking provider: {provider}")

    def attach_run_tracking(self, run_id: str, provider: str) -> dict[str, Any]:
        """Attach a connected provider without mutating the immutable run spec."""

        provider = str(provider or "").lower()
        if provider not in {"wandb", "mlflow"}:
            raise ValueError(f"unsupported tracking provider: {provider}")
        with self._reconcile_lock:
            run = self.database.get_run(run_id)
            if not run:
                raise KeyError("Run not found")
            existing = next(
                (
                    item
                    for item in self.database.list_tracking_bindings("run", run_id)
                    if item["provider"] == provider
                ),
                {},
            )
            if existing.get("remote_id") and existing.get("remote_url"):
                return existing

            connection = self.tracking_connections()["connections"].get(provider) or {}
            if not (
                connection.get("connected") is True
                and str(connection.get("status") or "").lower() == "connected"
            ):
                detail = connection.get("last_error") or (
                    f"{provider} is not connected; configure and test it in Settings first"
                )
                raise TrackingRequestError(str(detail))

            spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
            configured_provider = self._retroactive_tracking_provider(
                provider, spec, run, connection
            )
            state = str(run.get("status") or "").upper()
            if state in {"SUCCEEDED", "COMPLETED"}:
                final_status = "FINISHED"
            elif state == "CANCELLED":
                final_status = "KILLED"
            elif state in TERMINAL_FAILURE_STATES | {
                "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "DEADLINE", "SPECIAL_EXIT"
            }:
                final_status = "FAILED"
            else:
                final_status = None

            self._sync_tracking_outputs(
                run_id,
                status=final_status,
                providers=[configured_provider],
                central_authoritative=True,
            )
            attached = next(
                (
                    item
                    for item in self.database.list_tracking_bindings("run", run_id)
                    if item["provider"] == provider
                ),
                {},
            )
            if not (attached.get("remote_id") and attached.get("remote_url")):
                raise TrackingRequestError(
                    str(attached.get("last_error") or f"{provider} did not return a run binding")
                )
            metadata = {
                **(attached.get("metadata_json") or {}),
                "attachment_mode": "retroactive",
                "attached_at": utc_now(),
            }
            self.database.upsert_tracking_binding(
                provider,
                "run",
                run_id,
                remote_id=attached["remote_id"],
                remote_url=attached["remote_url"],
                status=attached.get("status") or "CONNECTED",
                metadata=metadata,
                last_error=attached.get("last_error"),
            )
            self.database.record_event(
                entity_type="run",
                entity_id=run_id,
                event_type="TRACKING_ATTACHED",
                details={"provider": provider, "remote_id": attached["remote_id"]},
            )
            return next(
                item
                for item in self.database.list_tracking_bindings("run", run_id)
                if item["provider"] == provider
            )

    def _native_tracking_provider_names(
        self,
        spec: ExperimentSpec,
        plan: AdapterPlan | None = None,
    ) -> set[str]:
        integrations = plan.native_tracking if plan is not None else []
        if plan is None:
            try:
                integrations = resolve_adapter_plan(spec).native_tracking
            except Exception:
                return set()
        enabled = {
            str(self._tracking_provider_value(provider, "provider"))
            for provider in self._active_tracking_providers(spec)
        }
        return {
            integration.provider
            for integration in integrations
            if integration.provider in enabled
        }

    def _native_tracking_runtime(
        self,
        spec: ExperimentSpec,
        plan: AdapterPlan,
        run_id: str,
        stage_id: str,
        *,
        continuation: bool = False,
    ) -> dict[str, Any]:
        active = self._native_tracking_provider_names(spec, plan)
        if not active:
            return {
                "environment": {},
                "secret_environment_files": {},
                "secret_contents": {},
                "run_ids": {},
            }
        providers = {
            str(self._tracking_provider_value(provider, "provider")): provider
            for provider in self._active_tracking_providers(spec)
        }
        environment: dict[str, str] = {}
        secret_environment_files: dict[str, str] = {}
        secret_contents: dict[str, str] = {}
        run_ids: dict[str, str] = {}
        run_directory = f"{WORK_ROOT}/jobs/runs/{run_id}"
        for name in sorted(active):
            provider = providers[name]
            if name != "wandb":
                raise ValueError(
                    f"adapter declares native {name} tracking, but no secure runtime binding is registered"
                )
            settings = self._wandb_settings(provider)
            if not settings.api_key:
                raise ValueError(
                    "native W&B tracking requires the validated session/environment API key"
                )
            if not settings.entity:
                raise ValueError(
                    "native W&B tracking requires a validated W&B entity in Settings"
                )
            project = wandb_project_slug(
                self._tracking_provider_value(provider, "project")
                or spec.identity.experiment
            )
            environment.update(
                {
                    "WANDB_BASE_URL": settings.base_url,
                    "WANDB_ENTITY": settings.entity,
                    "WANDB_PROJECT": project,
                    "WANDB_RUN_ID": run_id,
                    # The control plane reserves the W&B run ID before the native
                    # SDK writes history, but that reservation is not an initialized
                    # SDK run yet. ``allow`` attaches to an existing initialized run
                    # or initializes the reserved ID on its first native write.
                    # Continuations must fail closed rather than silently create
                    # a second tracking identity when the pinned run is absent.
                    "WANDB_RESUME": "must" if continuation else "allow",
                    "WANDB_MODE": "online",
                    "WANDB_CONSOLE": "off",
                    "WANDB_DIR": run_directory,
                }
            )
            relative_path = f"state/runtime-secrets/{stage_id}/wandb-api-key"
            secret_environment_files["WANDB_API_KEY"] = (
                f"{run_directory}/{relative_path}"
            )
            secret_contents[relative_path] = settings.api_key + "\n"
            run_ids[name] = run_id
        return {
            "environment": environment,
            "secret_environment_files": secret_environment_files,
            "secret_contents": secret_contents,
            "run_ids": run_ids,
        }

    def _require_native_tracking_bindings(
        self, run_id: str, providers: set[str]
    ) -> None:
        bindings = {
            str(binding["provider"]): binding
            for binding in self.database.list_tracking_bindings("run", run_id)
        }
        for provider in sorted(providers):
            binding = bindings.get(provider)
            if (
                not binding
                or binding.get("status") != "CONNECTED"
                or not binding.get("remote_id")
            ):
                detail = (binding or {}).get("last_error") or "remote run was not created"
                raise ValueError(
                    f"native {provider} tracking binding failed before submission: {detail}"
                )
            if provider == "wandb" and str(binding["remote_id"]) != run_id:
                raise ValueError(
                    "native W&B tracking binding returned a remote run ID that does not match the Skynet run"
                )

    def _validate_tracking_requirements(self, spec: ExperimentSpec) -> None:
        for provider in self._active_tracking_providers(spec):
            name = str(self._tracking_provider_value(provider, "provider"))
            if not self.database.get_tracking_connection(name):
                raise ValueError(
                    f"{name} tracking is enabled but has no validated connection; "
                    "connect it in Settings before previewing or submitting"
                )
            settings = (
                self._wandb_settings(provider)
                if name == "wandb"
                else self._mlflow_settings(provider)
            )
            if not settings.configured:
                missing = "API key" if name == "wandb" else "tracking endpoint"
                raise ValueError(
                    f"{name} tracking is enabled but its validated {missing} is unavailable"
                )

    def _flush_tracking_provider(self, provider: str, *, limit: int = 100) -> dict[str, Any]:
        attempted = delivered = 0
        errors: list[str] = []
        bindings = self.database.list_tracking_bindings_for_provider(
            provider, scope_type="run", statuses=("QUEUED", "ERROR")
        )[:limit]
        for existing in bindings:
            run_id = str(existing["scope_id"])
            capsule = LOCAL_CAPSULE_ROOT / run_id
            if not capsule.exists():
                continue
            try:
                if provider == "wandb":
                    settings = self._wandb_settings()
                    current_endpoint = settings.base_url
                    bridge: Any = WandBBridge(capsule, settings)
                else:
                    settings = self._mlflow_settings()
                    current_endpoint = settings.tracking_uri
                    bridge = MLflowBridge(capsule, settings)
                metadata = existing.get("metadata_json") or {}
                pinned_endpoint = metadata.get("endpoint")
                if not pinned_endpoint or str(pinned_endpoint).rstrip("/") != str(
                    current_endpoint or ""
                ).rstrip("/"):
                    message = (
                        "queued tracking capsule endpoint is unpinned"
                        if not pinned_endpoint
                        else "queued tracking capsule endpoint does not match the active connection"
                    )
                    self.database.upsert_tracking_binding(
                        provider, "run", run_id,
                        remote_id=existing.get("remote_id"),
                        remote_url=existing.get("remote_url"),
                        status="BLOCKED",
                        metadata=metadata,
                        last_error=message,
                    )
                    errors.append(message)
                    continue
                report = bridge.drain_spool()
                attempted += report.attempted
                delivered += report.delivered
                errors.extend(report.errors)
                binding = bridge.binding(run_id)
                remote_id = (binding or {}).get("remote_id")
                remote_url = (binding or {}).get("url") or existing.get("remote_url")
                run = self.database.get_run(run_id) or {}
                experiment_scope_id = str(run.get("experiment_id") or "")
                experiment_binding = (
                    next(
                        (
                            item for item in self.database.list_tracking_bindings(
                                "experiment", experiment_scope_id
                            ) if item["provider"] == provider
                        ),
                        {},
                    )
                    if experiment_scope_id
                    else {}
                )
                experiment_metadata = dict(experiment_binding.get("metadata_json") or {})
                experiment_remote_id = experiment_binding.get("remote_id")
                experiment_remote_url = experiment_binding.get("remote_url")
                experiment_recovered = False
                if provider == "mlflow":
                    experiment_name = str(
                        metadata.get("experiment")
                        or experiment_metadata.get("experiment")
                        or ""
                    )
                    recovered_experiment_id = (
                        bridge.experiment_binding(experiment_name)
                        if experiment_name else None
                    )
                    if recovered_experiment_id:
                        experiment_remote_id = recovered_experiment_id
                        experiment_recovered = True
                    tracking_uri = settings.tracking_uri
                    if tracking_uri and experiment_remote_id:
                        experiment_remote_url = mlflow_experiment_url(
                            tracking_uri, str(experiment_remote_id)
                        )
                    if not remote_url and tracking_uri and experiment_remote_id and remote_id:
                        remote_url = mlflow_run_url(
                            tracking_uri, str(experiment_remote_id), str(remote_id)
                        )
                    if experiment_name:
                        experiment_metadata["experiment"] = experiment_name
                    experiment_metadata["endpoint"] = settings.public_dict()["tracking_uri"]
                    if remote_id:
                        self.database.update_run(run_id, mlflow_run_id=remote_id)
                else:
                    entity = str(
                        metadata.get("entity")
                        or experiment_metadata.get("entity")
                        or settings.entity
                        or ""
                    )
                    project = str(
                        metadata.get("project")
                        or experiment_metadata.get("project")
                        or ""
                    )
                    recovered_project = (
                        bridge.project_binding(entity, project)
                        if entity and project else None
                    )
                    if recovered_project:
                        experiment_remote_id = recovered_project.get("remote_id")
                        experiment_remote_url = recovered_project.get("url")
                        experiment_recovered = bool(experiment_remote_id)
                    if entity:
                        experiment_metadata["entity"] = entity
                    if project:
                        experiment_metadata["project"] = project
                    experiment_metadata["endpoint"] = settings.public_dict()["base_url"]
                drain_error = report.errors[0] if report.errors else None
                run_connected = bool(remote_id) and report.remaining == 0 and not drain_error
                self.database.upsert_tracking_binding(
                    provider, "run", run_id,
                    remote_id=remote_id,
                    remote_url=remote_url,
                    status=(
                        "CONNECTED" if run_connected
                        else "ERROR" if drain_error
                        else "QUEUED"
                    ),
                    metadata=metadata,
                    last_error=drain_error,
                )
                if experiment_scope_id:
                    self.database.upsert_tracking_binding(
                        provider, "experiment", experiment_scope_id,
                        remote_id=experiment_remote_id,
                        remote_url=experiment_remote_url,
                        status=(
                            "CONNECTED" if experiment_recovered
                            else "ERROR" if drain_error
                            else "QUEUED"
                        ),
                        metadata=experiment_metadata,
                        last_error=None if experiment_recovered else drain_error,
                    )
            except Exception as error:
                secrets = tuple(self.credentials.get(provider).values())
                message = str(sanitize(str(error), secrets=secrets))
                errors.append(message)
                self.database.upsert_tracking_binding(
                    provider, "run", run_id,
                    remote_id=existing.get("remote_id"),
                    remote_url=existing.get("remote_url"),
                    status="ERROR",
                    metadata=existing.get("metadata_json") or {},
                    last_error=message,
                )
        return {
            "capsules": len(bindings),
            "attempted": attempted,
            "delivered": delivered,
            "errors": errors[:10],
        }

    def _tracking_run_name(
        self, provider: Any, spec: ExperimentSpec, run: Mapping[str, Any]
    ) -> str:
        template = self._tracking_provider_value(provider, "run_name_template")
        values = {
            "experiment": spec.identity.experiment,
            "variant": str(run.get("variant_name") or run.get("variant_id") or "variant"),
            "run": str(run["id"]),
            "run_number": int(run.get("run_number") or 1),
            "seed": int(run.get("seed") or spec.train.seed),
        }
        if not template:
            template = "{experiment}/{variant}/run-{run_number}"
        try:
            return str(template).format(**values)
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid tracking run name template: {error}") from error

    def _tracking_tags(
        self, spec: ExperimentSpec, run: Mapping[str, Any], job_id: str
    ) -> dict[str, Any]:
        return {
            **spec.tracking.tags,
            "skynet.experiment_id": run.get("experiment_id"),
            "skynet.experiment_revision_id": run.get("experiment_revision_id"),
            "skynet.experiment_revision": run.get("experiment_revision_number"),
            "skynet.run_id": run["id"],
            "skynet.variant_id": run.get("variant_id"),
            "skynet.status": run.get("status"),
            "adapter": spec.source.adapter,
            "source.commit": spec.source.revision,
            "slurm.job_id": job_id,
        }

    def _tracking_params(
        self,
        spec: ExperimentSpec,
        run: Mapping[str, Any],
        attempt_snapshot: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resources = spec.resources.model_dump(mode="json")
        data = spec.data.model_dump(mode="json")
        if attempt_snapshot is not None:
            common = _common_hyperparameter_contract_from_snapshot(attempt_snapshot)
        else:
            attempts = list(run.get("attempts") or [])
            common = (
                _attempt_common_hyperparameter_contract(
                    run,
                    attempts[-1],
                    _attempt_enrichment_event(
                        self.database, str(attempts[-1]["id"])
                    ),
                )
                if attempts
                else _resolve_effective_common_hyperparameters(
                    spec.model_dump(mode="json", by_alias=True),
                    spec.source.adapter_manifest,
                )
            )
        effective = common["values"]
        common_params = {
            "learning_rate": effective["learning_rate"],
            "batch_semantics": effective["batch_semantics"],
            "batch_value": effective["batch_size"],
            "gradient_accumulation_steps": effective["gradient_accumulation"],
            "num_workers": effective["num_workers"],
            "max_steps": effective["max_steps"],
            "precision": effective["precision"],
        }
        return {
            "seed": run["seed"],
            **{key: value for key, value in common_params.items() if value is not None},
            "gpu_count": resolve_gpu_count(spec, resolve_adapter_plan(spec)),
            "gpu_type": (resources.get("gpu") or {}).get("type"),
            "account": spec.resources.account,
            "partition": spec.resources.partition,
            "source_repository": spec.source.repository,
            "source_commit": spec.source.revision,
            "adapter": spec.source.adapter,
            "adapter_version": run.get("adapter_version"),
            "data_bundle_id": data.get("bundle_id") or data.get("bundle"),
        }

    def _tracking_failure(
        self, provider: str, run: Mapping[str, Any], error: BaseException | str
    ) -> None:
        secrets = tuple(self.credentials.get(provider).values())
        message = str(sanitize(str(error), secrets=secrets))
        existing = next(
            (
                item for item in self.database.list_tracking_bindings(
                    "run", str(run["id"])
                ) if item["provider"] == provider
            ),
            {},
        )
        self.database.upsert_tracking_binding(
            provider,
            "run",
            str(run["id"]),
            remote_id=existing.get("remote_id"),
            remote_url=existing.get("remote_url"),
            status="ERROR",
            metadata=existing.get("metadata_json") or {},
            last_error=message,
        )
        experiment_id = run.get("experiment_id")
        if experiment_id:
            experiment_existing = next(
                (
                    item for item in self.database.list_tracking_bindings(
                        "experiment", str(experiment_id)
                    ) if item["provider"] == provider
                ),
                {},
            )
            self.database.upsert_tracking_binding(
                provider,
                "experiment",
                str(experiment_id),
                remote_id=experiment_existing.get("remote_id"),
                remote_url=experiment_existing.get("remote_url"),
                status="ERROR",
                metadata=experiment_existing.get("metadata_json") or {},
                last_error=message,
            )
        self.database.record_event(
            entity_type="run",
            entity_id=str(run["id"]),
            event_type="TRACKING_ERROR",
            details={"provider": provider, "error": message},
        )

    def _start_tracking(
        self,
        spec: ExperimentSpec,
        run: Mapping[str, Any],
        capsule: Path,
        job_id: str,
        *,
        attempt_snapshot: Mapping[str, Any] | None = None,
        providers: list[Any] | None = None,
        continuation_attempt_number: int | None = None,
    ) -> None:
        del capsule
        local_run_id = str(run["id"])
        tags = self._tracking_tags(spec, run, job_id)
        if continuation_attempt_number is not None:
            tags["skynet.attempt_number"] = continuation_attempt_number
        params = self._tracking_params(spec, run, attempt_snapshot)
        for provider in providers if providers is not None else self._active_tracking_providers(spec):
            name = str(self._tracking_provider_value(provider, "provider"))
            try:
                run_name = self._tracking_run_name(provider, spec, run)
                if name == "mlflow":
                    settings = self._mlflow_settings(provider)
                    bridge = MLflowBridge(LOCAL_CAPSULE_ROOT / local_run_id, settings)
                    experiment_name = (
                        self._tracking_provider_value(provider, "experiment")
                        or spec.tracking.mlflow_experiment
                        or spec.identity.experiment
                    )
                    experiment_result = bridge.ensure_experiment(experiment_name)
                    if continuation_attempt_number is None:
                        created = bridge.ensure_run(
                            experiment_name=experiment_name,
                            local_run_id=local_run_id,
                            run_name=run_name,
                            tags=tags,
                        )
                    else:
                        provider_bindings = [
                            item
                            for item in self.database.list_tracking_bindings(
                                "run", local_run_id
                            )
                            if item["provider"] == "mlflow"
                        ]
                        existing_binding = (
                            provider_bindings[0]
                            if len(provider_bindings) == 1
                            else None
                        )
                        local_binding = bridge.binding(local_run_id)
                        metadata = (
                            existing_binding.get("metadata_json")
                            if existing_binding else {}
                        ) or {}
                        if (
                            not existing_binding
                            or not existing_binding.get("remote_id")
                            or not existing_binding.get("remote_url")
                            or not local_binding
                            or local_binding.get("remote_id")
                            != existing_binding.get("remote_id")
                            or local_binding.get("url")
                            != existing_binding.get("remote_url")
                            or str(metadata.get("endpoint") or "").rstrip("/")
                            != str(settings.tracking_uri or "").rstrip("/")
                            or metadata.get("experiment") != experiment_name
                        ):
                            raise TrackingRequestError(
                                "cannot resume MLflow tracking without one exact pinned run binding"
                            )
                        created = bridge.reopen_run(
                            local_run_id, continuation_attempt_number
                        )
                    bridge.set_tags(local_run_id, tags)
                    if continuation_attempt_number is None:
                        bridge.log_params(local_run_id, params)
                    experiment_id = experiment_result.remote_id or bridge.experiment_binding(experiment_name)
                    run_binding = bridge.binding(local_run_id)
                    remote_run_id = created.remote_id or (
                        run_binding.get("remote_id") if run_binding else None
                    )
                    error = created.error or experiment_result.error or bridge.last_error
                    self.database.upsert_tracking_binding(
                        "mlflow", "experiment", str(run["experiment_id"]),
                        remote_id=experiment_id,
                        remote_url=(
                            mlflow_experiment_url(settings.tracking_uri, experiment_id)
                            if settings.tracking_uri and experiment_id else None
                        ),
                        status="CONNECTED" if experiment_id else "QUEUED",
                        metadata={
                            "experiment": experiment_name,
                            "endpoint": settings.public_dict()["tracking_uri"],
                        },
                        last_error=error,
                    )
                    self.database.upsert_tracking_binding(
                        "mlflow", "run", local_run_id,
                        remote_id=remote_run_id,
                        remote_url=(
                            mlflow_run_url(settings.tracking_uri, experiment_id, remote_run_id)
                            if settings.tracking_uri and experiment_id and remote_run_id else None
                        ),
                        status="CONNECTED" if remote_run_id else "QUEUED",
                        metadata={
                            "endpoint": settings.public_dict()["tracking_uri"],
                            "experiment": experiment_name,
                        },
                        last_error=error,
                    )
                    if remote_run_id:
                        self.database.update_run(local_run_id, mlflow_run_id=remote_run_id)
                elif name == "wandb":
                    settings = self._wandb_settings(provider)
                    entity = settings.entity
                    if not entity:
                        if not settings.configured:
                            raise TrackingRequestError(
                                "W&B is enabled but no session/environment API key and entity are configured"
                            )
                        entity = str(
                            WandBBridge(
                                LOCAL_CAPSULE_ROOT / local_run_id, settings
                            ).validate_connection()["entity"]
                        )
                        settings = replace(settings, entity=entity)
                    project = wandb_project_slug(
                        self._tracking_provider_value(provider, "project")
                        or spec.identity.experiment
                    )
                    bridge = WandBBridge(LOCAL_CAPSULE_ROOT / local_run_id, settings)
                    project_result = bridge.ensure_experiment(entity, project)
                    if continuation_attempt_number is None:
                        created = bridge.ensure_run(
                            entity=entity,
                            project=project,
                            local_run_id=local_run_id,
                            run_name=run_name,
                            group=(
                                f"{spec.identity.experiment}/revision-"
                                f"{run.get('experiment_revision_number') or 1}"
                            ),
                            tags=tags,
                            config=params,
                        )
                    else:
                        provider_bindings = [
                            item
                            for item in self.database.list_tracking_bindings(
                                "run", local_run_id
                            )
                            if item["provider"] == "wandb"
                        ]
                        existing_binding = (
                            provider_bindings[0]
                            if len(provider_bindings) == 1
                            else None
                        )
                        local_binding = bridge.binding(local_run_id)
                        metadata = (
                            existing_binding.get("metadata_json")
                            if existing_binding else {}
                        ) or {}
                        expected_url = (
                            f"{wandb_web_base(settings.base_url)}/"
                            f"{urllib.parse.quote(entity, safe='')}/"
                            f"{urllib.parse.quote(project, safe='')}/runs/"
                            f"{urllib.parse.quote(local_run_id, safe='')}"
                        )
                        if (
                            not existing_binding
                            or existing_binding.get("remote_id") != local_run_id
                            or existing_binding.get("remote_url") != expected_url
                            or not local_binding
                            or local_binding.get("remote_id") != local_run_id
                            or local_binding.get("url") != expected_url
                            or metadata.get("entity") != entity
                            or metadata.get("project") != project
                            or str(metadata.get("endpoint") or "").rstrip("/")
                            != str(settings.public_dict()["base_url"] or "").rstrip("/")
                        ):
                            raise TrackingRequestError(
                                "cannot resume W&B tracking without one exact pinned run binding"
                            )
                        created = bridge.reopen_run(
                            local_run_id, continuation_attempt_number
                        )
                    bridge.set_tags(local_run_id, tags)
                    if continuation_attempt_number is None:
                        bridge.log_params(local_run_id, params)
                    binding = bridge.binding(local_run_id)
                    error = created.error or project_result.error
                    project_url = (
                        f"{wandb_web_base(settings.base_url)}/"
                        f"{urllib.parse.quote(entity, safe='')}/"
                        f"{urllib.parse.quote(project, safe='')}"
                    )
                    self.database.upsert_tracking_binding(
                        "wandb", "experiment", str(run["experiment_id"]),
                        remote_id=f"{entity}/{project}",
                        remote_url=project_url if binding else None,
                        status="CONNECTED" if binding else "QUEUED",
                        metadata={
                            "entity": entity,
                            "project": project,
                            "endpoint": settings.public_dict()["base_url"],
                        },
                        last_error=error,
                    )
                    self.database.upsert_tracking_binding(
                        "wandb", "run", local_run_id,
                        remote_id=(binding or {}).get("remote_id"),
                        remote_url=(binding or {}).get("url"),
                        status="CONNECTED" if binding else "QUEUED",
                        metadata={
                            "entity": entity,
                            "project": project,
                            "endpoint": settings.public_dict()["base_url"],
                        },
                        last_error=error,
                    )
                else:
                    raise ValueError(f"unsupported tracking provider: {name}")
            except Exception as error:
                self._tracking_failure(name, run, error)

    @staticmethod
    def _slurm_accounting_timestamp(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value or value.upper() in {"NONE", "UNKNOWN", "N/A", "INVALID"}:
            return None
        if re.fullmatch(r"\d+", value):
            try:
                return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat(
                    timespec="milliseconds"
                ).replace("+00:00", "Z")
            except (OverflowError, OSError, ValueError):
                return None
        return value

    def submit_data_import(
        self, resource_id: str, request: HuggingFaceImportRequest
    ) -> dict[str, Any]:
        resource = self.database.get_data_resource(resource_id)
        if resource is None:
            raise KeyError(f"Data resource not found: {resource_id}")
        if resource.get("archived_at") is not None:
            raise ValueError("cannot import into an archived data resource")
        if resource.get("provider") != "huggingface":
            raise ValueError("only resources with provider=huggingface can use this importer")
        payload = request.model_dump(mode="python")
        active = self.database.list_data_imports(
            states=["SUBMITTING", "SUBMITTED", "PENDING", "RUNNING", "FINALIZING"]
        )
        if any(item["resource_id"] == resource_id and item["request"] == payload for item in active):
            raise ValueError("the same immutable Hugging Face import is already active")
        record = self.database.create_data_import(resource_id, request=payload)
        job = build_huggingface_import_job(record["id"], resource, payload)
        try:
            self.cluster.test_script(job.script, request.gateway)
            submission = self.cluster.submit_script(job.script, job.run_id, request.gateway)
        except Exception as error:
            self.database.update_data_import(record["id"], state="FAILED", error=str(error))
            raise
        stdout_path = f"{CLUSTER.paths.logs}/{job.job_name}-{submission.job_id}.out"
        stderr_path = f"{CLUSTER.paths.logs}/{job.job_name}-{submission.job_id}.err"
        metadata = dict(resource.get("metadata") or {})
        metadata.update({
            "repository_type": "dataset",
            "source_uri": (
                f"https://huggingface.co/datasets/{resource['namespace']}/{resource['name']}"
            ),
        })
        self.database.update_data_resource(resource_id, metadata=metadata)
        return self.database.update_data_import(
            record["id"],
            state="SUBMITTED",
            gateway=submission.gateway,
            slurm_job_id=submission.job_id,
            slurm_state="SUBMITTED",
            run_directory=submission.run_directory,
            script_path=submission.script_path,
            result_path=job.result_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    def _publish_data_import_result(
        self, record: Mapping[str, Any], result: Mapping[str, Any]
    ) -> dict[str, Any]:
        request = dict(record.get("request") or {})
        resource = self.database.get_data_resource(str(record["resource_id"]))
        if resource is None:
            raise KeyError(f"Data resource not found: {record['resource_id']}")
        expected = {
            "schema_version": "skynet.data-import-result/v1",
            "import_id": record["id"],
            "resource_id": record["resource_id"],
            "provider": resource["provider"],
            "namespace": resource["namespace"],
            "name": resource["name"],
            "revision": request.get("revision"),
            "subset": request.get("subset"),
            "format": request.get("format"),
            "role": request.get("role"),
            "bundle_name": request.get("bundle_name"),
            "bundle_version": request.get("bundle_version"),
        }
        mismatches = [key for key, value in expected.items() if result.get(key) != value]
        if mismatches:
            raise ValueError(
                "data import result does not match its immutable request: " + ", ".join(mismatches)
            )
        manifest_sha256 = str(result.get("manifest_sha256") or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
            raise ValueError("data import result has an invalid manifest SHA-256")
        path = PurePosixPath(str(result.get("path") or ""))
        datasets_root = PurePosixPath(CLUSTER.paths.datasets)
        if not path.is_absolute() or datasets_root not in path.parents:
            raise ValueError("data import result path is outside the canonical datasets root")
        version_revision = str(result["version_revision"])
        version = next(
            (
                item for item in resource.get("versions", [])
                if item["revision"] == version_revision and item["format"] == result["format"]
            ),
            None,
        )
        version_metadata = {
            "subset": result["subset"],
            "inventory_sha256": result.get("inventory_sha256"),
            "file_count": result.get("file_count"),
            "manifest_path": result.get("manifest_path"),
            "snapshot_path": result.get("snapshot_path"),
            "resource_uri": (
                f"resource:huggingface/{resource['namespace']}/{resource['name']}"
                f"@{result['revision']}#subset={result['subset']}"
            ),
        }
        if version is None:
            version = self.database.create_data_resource_version(
                str(record["resource_id"]),
                revision=version_revision,
                format=str(result["format"]),
                path=str(path),
                source_uri=str(result["source_uri"]),
                manifest_sha256=manifest_sha256,
                status="READY",
                size_bytes=int(result["size_bytes"]),
                metadata=version_metadata,
            )
        elif any((
            version.get("path") != str(path),
            version.get("manifest_sha256") != manifest_sha256,
            version.get("source_uri") != result.get("source_uri"),
        )):
            raise ValueError("existing immutable version conflicts with completed import")
        bundle = next(
            (
                item for item in self.database.list_data_bundles(include_archived=False)
                if item["name"] == request["bundle_name"]
                and item["version"] == request["bundle_version"]
            ),
            None,
        )
        expected_assignment = {
            "role": request["role"],
            "position": 0,
            "version_id": version["id"],
        }
        if bundle is None:
            bundle = self.database.create_data_bundle(
                name=str(request["bundle_name"]),
                version=str(request["bundle_version"]),
                description=(
                    f"Pinned Hugging Face subset {resource['namespace']}/{resource['name']}"
                    f"/{request['subset']} at {request['revision']}"
                ),
                assignments=[{**expected_assignment, "required": True}],
                metadata={"data_import_id": record["id"]},
            )
        else:
            assignments = bundle.get("assignments") or []
            if len(assignments) != 1 or any(
                assignments[0].get(key) != value for key, value in expected_assignment.items()
            ):
                raise ValueError("existing immutable bundle conflicts with completed import")
        return self.database.update_data_import(
            str(record["id"]),
            state="SUCCEEDED",
            result=dict(result),
            slurm_state="COMPLETED",
            version_id=version["id"],
            bundle_id=bundle["id"],
            error=None,
        )

    def reconcile_data_imports(self) -> list[dict[str, Any]]:
        records = self.database.list_data_imports(
            states=["SUBMITTED", "PENDING", "RUNNING", "FINALIZING"]
        )
        for record in records:
            job_id = record.get("slurm_job_id")
            if not job_id:
                continue
            try:
                gateway, statuses = self.cluster.job_statuses(
                    [str(job_id)], str(record.get("gateway") or "auto")
                )
                status = statuses.get(str(job_id))
                if status is None:
                    continue
                slurm_state = str(status.get("State") or "UNKNOWN").upper()
                common = {
                    "gateway": gateway,
                    "slurm_state": slurm_state,
                    "exit_code": status.get("ExitCode"),
                    "node_list": status.get("NodeList"),
                    "error": None,
                }
                if slurm_state in ACTIVE_STATES:
                    app_state = "RUNNING" if slurm_state in {"RUNNING", "COMPLETING"} else "PENDING"
                    self.database.update_data_import(record["id"], state=app_state, **common)
                    continue
                if slurm_state == "COMPLETED":
                    _, payload = self.cluster.read_file(str(record["result_path"]), gateway)
                    result = json.loads(payload)
                    record = self.database.update_data_import(
                        record["id"], state="FINALIZING", **common
                    )
                    self._publish_data_import_result(record, result)
                    continue
                error = f"Slurm data import ended in {slurm_state} ({status.get('ExitCode') or 'unknown exit'})"
                try:
                    _, stderr = self.cluster.read_log(str(record["stderr_path"]), gateway, lines=80)
                    if stderr.strip():
                        error += "\n" + stderr.strip()
                except ClusterError:
                    pass
                common["error"] = error
                self.database.update_data_import(record["id"], state="FAILED", **common)
            except ClusterError as error:
                self.database.update_data_import(
                    record["id"], error=f"Import status refresh failed: {error}"
                )
            except Exception as error:
                self.database.update_data_import(
                    record["id"], state="FAILED", error=f"Import reconciliation failed: {error}"
                )
        return self.database.list_data_imports()

    def _repair_missing_attempt_log_paths(self) -> int:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, gateway, slurm_job_id, sbatch_path, stdout_path, stderr_path
                FROM job_attempts
                WHERE slurm_job_id IS NOT NULL
                  AND sbatch_path IS NOT NULL
                  AND (stdout_path IS NULL OR stderr_path IS NULL)
                """
            ).fetchall()

        repaired = 0
        for row in rows:
            job_id = str(row["slurm_job_id"])
            archived_suffix = f"/attempts/{job_id}/job.sbatch"
            script_path = str(row["sbatch_path"])
            if not script_path.endswith(archived_suffix):
                continue
            try:
                _, script = self.cluster.read_file(
                    script_path,
                    row["gateway"] or "auto",
                    max_bytes=2_000_000,
                )
            except ClusterError:
                continue
            stdout_path, stderr_path = resolve_slurm_log_paths_from_sbatch(
                script, job_id
            )
            updates: dict[str, str] = {}
            if row["stdout_path"] is None and stdout_path is not None:
                updates["stdout_path"] = stdout_path
            if row["stderr_path"] is None and stderr_path is not None:
                updates["stderr_path"] = stderr_path
            if not updates:
                continue
            self.database.update_job_attempt(str(row["id"]), **updates)
            repaired += 1
        return repaired

    def _repair_missing_active_tracking_bindings(
        self, rows: list[Mapping[str, Any]]
    ) -> int:
        latest_by_run: dict[str, Mapping[str, Any]] = {}
        for raw_row in rows:
            row = dict(raw_row)
            if row["stage_type"] == "EVALUATE":
                continue
            run_id = str(row["run_id"])
            current = latest_by_run.get(run_id)
            if current is None or int(row["attempt_number"] or 0) > int(
                current["attempt_number"] or 0
            ):
                latest_by_run[run_id] = row

        repaired = 0
        for run_id, attempt in latest_by_run.items():
            run = self.database.get_run(run_id)
            if not run:
                continue
            try:
                spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
            except Exception:
                continue
            providers = self._active_tracking_providers(spec)
            if not providers:
                continue
            run_bindings = {
                str(binding["provider"])
                for binding in self.database.list_tracking_bindings("run", run_id)
            }
            experiment_id = str(run.get("experiment_id") or "")
            experiment_bindings = {
                str(binding["provider"])
                for binding in self.database.list_tracking_bindings(
                    "experiment", experiment_id
                )
            } if experiment_id else set()
            missing = [
                provider
                for provider in providers
                if (
                    str(self._tracking_provider_value(provider, "provider"))
                    not in run_bindings
                    or str(self._tracking_provider_value(provider, "provider"))
                    not in experiment_bindings
                )
            ]
            if not missing:
                continue
            snapshot = attempt.get("execution_snapshot_json")
            if isinstance(snapshot, str):
                try:
                    snapshot = json.loads(snapshot)
                except json.JSONDecodeError:
                    snapshot = None
            try:
                self._start_tracking(
                    spec,
                    run,
                    LOCAL_CAPSULE_ROOT / run_id,
                    str(attempt["slurm_job_id"]),
                    attempt_snapshot=snapshot if isinstance(snapshot, Mapping) else None,
                    providers=missing,
                )
            except Exception as error:
                for provider in missing:
                    self._tracking_failure(
                        str(self._tracking_provider_value(provider, "provider")),
                        run,
                        error,
                    )
            repaired += len(missing)
        return repaired

    def _record_recovered_submission(self, unknown: Mapping[str, Any], recovered: Any) -> None:
        submitted_at = utc_now()
        stdout_path = resolve_slurm_log_path(
            unknown["stdout_path"], recovered.job_id
        )
        stderr_path = resolve_slurm_log_path(
            unknown["stderr_path"], recovered.job_id
        )
        is_evaluation = unknown["stage_type"] == "EVALUATE"
        transition = self.database.transition_workflow_state(
            attempt_id=unknown["id"],
            attempt_updates={
                "status": "SUBMITTED",
                "slurm_job_id": recovered.job_id,
                "gateway": recovered.gateway,
                "sbatch_path": recovered.script_path,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "submitted_at": submitted_at,
                "slurm_reason": "Recovered after a lost SSH acknowledgement",
            },
            stage_id=unknown["stage_id"],
            stage_updates={"status": "SUBMITTED", "started_at": submitted_at},
            run_id=None if is_evaluation else unknown["run_id"],
            run_updates=None if is_evaluation else {
                "status": "SUBMITTED",
                "started_at": unknown["run_started_at"] or submitted_at,
                "completed_at": None,
            },
            evaluation_id=unknown["evaluation_id"] if is_evaluation else None,
            evaluation_updates={
                "status": "SUBMITTED",
                "started_at": submitted_at,
                "completed_at": None,
            } if is_evaluation and unknown["evaluation_id"] else None,
            event={
                "entity_type": "evaluation" if is_evaluation else "run",
                "entity_id": unknown["evaluation_id"] if is_evaluation else unknown["run_id"],
                "event_type": "SUBMISSION_RECOVERED",
                "new_status": "SUBMITTED",
                "details": {
                    "attempt_id": unknown["id"],
                    "job_id": recovered.job_id,
                    "gateway": recovered.gateway,
                },
            },
            expected_stage_statuses=("SUBMITTING",),
        )
        cancellation_pending = (
            str(unknown["stage_status"] or "").upper() == "CANCELLING"
            or (
                transition.get("applied") is False
                and str(transition["stage"].get("status") or "").upper()
                == "CANCELLING"
            )
        )
        if cancellation_pending:
            self._preserve_cancelling_submission(
                run_id=str(unknown["run_id"]),
                stage={"id": unknown["stage_id"], "stage_type": unknown["stage_type"]},
                evaluation={"id": unknown["evaluation_id"]}
                if is_evaluation and unknown["evaluation_id"]
                else None,
                attempt_id=str(unknown["id"]),
                attempt_updates={
                    "status": "CANCELLING",
                    "slurm_job_id": recovered.job_id,
                    "gateway": recovered.gateway,
                    "sbatch_path": recovered.script_path,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                    "submitted_at": submitted_at,
                    "slurm_reason": "Recovered after cancellation was requested",
                },
                event_type="SUBMISSION_RECOVERED_AFTER_CANCEL_REQUEST",
                details={
                    "attempt_id": unknown["id"],
                    "job_id": recovered.job_id,
                    "gateway": recovered.gateway,
                },
            )
            self._signal_stage_cancellation(
                entity_type="evaluation" if is_evaluation else "run",
                entity_id=str(unknown["evaluation_id"])
                if is_evaluation
                else str(unknown["run_id"]),
                attempt={
                    "id": unknown["id"],
                    "slurm_job_id": recovered.job_id,
                    "gateway": recovered.gateway,
                },
                record_event=True,
            )

    def reconcile(self) -> dict[str, Any]:
        if not self._reconcile_lock.acquire(blocking=False):
            return {
                "ok": True,
                "skipped": "already running",
                "draft_graphs_repaired": 0,
                "tracking_bindings_repaired": 0,
            }
        try:
            repairs = self.database.repair_workflow_state_invariants()
            recovered_submissions = 0
            with self.database.connection() as connection:
                unknown_rows = [
                    dict(row)
                    for row in connection.execute(
                    """
                    SELECT a.id, a.gateway, a.stdout_path, a.stderr_path,
                           s.id AS stage_id, s.run_id, s.stage_type,
                           s.status AS stage_status,
                           r.started_at AS run_started_at,
                           e.id AS evaluation_id,
                           er.experiment_id
                    FROM job_attempts a
                    JOIN workflow_stages s ON s.id = a.stage_id
                    JOIN runs r ON r.id = s.run_id
                    JOIN variants v ON v.id = r.variant_id
                    JOIN experiment_revisions er ON er.id = v.experiment_revision_id
                    LEFT JOIN evaluations e ON e.stage_id = s.id
                    WHERE a.status IN ('SUBMITTING', 'CANCELLING')
                      AND a.slurm_job_id IS NULL
                      AND a.slurm_reason LIKE 'Submission outcome unknown%'
                    """
                    ).fetchall()
                ]
            for unknown in unknown_rows:
                try:
                    recovered = self.cluster.recover_submission(
                        unknown["run_id"],
                        unknown["id"],
                        unknown["gateway"] or "auto",
                    )
                except ClusterError:
                    continue
                if recovered is None:
                    continue
                self._record_recovered_submission(unknown, recovered)
                recovered_submissions += 1
            self._repair_missing_attempt_log_paths()
            try:
                self.reconcile_data_imports()
            except Exception:
                pass
            with self.database.connection() as connection:
                rows = [
                    dict(row)
                    for row in connection.execute(
                    """
                    SELECT a.*, s.run_id, s.stage_type, s.auto_resume, s.max_attempts,
                           s.resolved_config_json, s.status AS stage_status,
                           er.experiment_id,
                           (
                               SELECT COUNT(*) FROM job_attempts budget_attempt
                               WHERE budget_attempt.stage_id = s.id
                                 AND UPPER(COALESCE(budget_attempt.status, '')) != 'CANCELLED'
                           ) AS budget_attempt_count
                    FROM job_attempts a
                    JOIN workflow_stages s ON s.id = a.stage_id
                    JOIN runs r ON r.id = s.run_id
                    JOIN variants v ON v.id = r.variant_id
                    JOIN experiment_revisions er ON er.id = v.experiment_revision_id
                    WHERE a.slurm_job_id IS NOT NULL
                      AND (
                          a.status IN ('SUBMITTED','PENDING','RUNNING','REQUEUED','CANCELLING')
                          OR (
                              a.status = 'SUCCEEDED'
                              AND s.status IN ('SUBMITTING','SUBMITTED','PENDING_SLURM','RUNNING','CANCELLING')
                          )
                      )
                    """
                    ).fetchall()
                ]
            tracking_bindings_repaired = self._repair_missing_active_tracking_bindings(rows)
            if not rows:
                self._dispatch_active_experiments()
                return {
                    "ok": True,
                    "checked": 0,
                    "updated": 0,
                    "repaired": repairs["repaired"],
                    "draft_graphs_repaired": repairs["draft_graphs_repaired"],
                    "recovered_submissions": recovered_submissions,
                    "tracking_bindings_repaired": tracking_bindings_repaired,
                }
            training_run_ids = {
                str(row["run_id"])
                for row in rows
                if str(row["stage_type"]).upper() == "TRAIN"
            }
            for training_run_id in training_run_ids:
                current_run = self.database.get_run(training_run_id)
                if current_run:
                    self._ingest_training_progress(current_run)
                self._publish_training_progress_tracking(training_run_id)
            _, statuses = self.cluster.job_statuses([row["slurm_job_id"] for row in rows])
            updated = 0
            experiments: set[str] = set(repairs["experiment_ids"])
            for row in rows:
                record = statuses.get(row["slurm_job_id"])
                if not record:
                    continue
                raw_state = str(record.get("StateRaw") or record["State"]).strip()
                state = re.split(r"[+\s]", str(record["State"]).strip(), maxsplit=1)[0].upper()
                reason = record.get("Reason")
                if not reason or str(reason).strip().upper() in {"NONE", "UNKNOWN", "N/A"}:
                    cancelled_by = record.get("CancelledBy")
                    if state == "CANCELLED" and cancelled_by:
                        reason = f"CANCELLED by {cancelled_by}"
                    elif raw_state.upper() != state:
                        reason = raw_state
                    else:
                        reason = None
                accounting_started = self._slurm_accounting_timestamp(record.get("Start"))
                accounting_finished = self._slurm_accounting_timestamp(record.get("End"))
                experiments.add(row["experiment_id"])
                is_evaluation_stage = row["stage_type"] == "EVALUATE"
                evaluation = (
                    self._evaluation_for_stage(row["run_id"], row["stage_id"])
                    if is_evaluation_stage
                    else None
                )
                common = {
                    "slurm_state": state,
                    "slurm_reason": reason,
                    "exit_code": record.get("ExitCode"),
                    "node_list": record.get("NodeList"),
                }
                cancellation_requested = (
                    str(row.get("stage_status") or "").upper() == "CANCELLING"
                    or str(row.get("status") or "").upper() == "CANCELLING"
                )
                if state in ACTIVE_STATES:
                    if cancellation_requested:
                        attempt_updates = {"status": "CANCELLING", **common}
                        if state == "RUNNING" and not row["started_at"]:
                            attempt_updates["started_at"] = accounting_started or utc_now()
                        self.database.transition_workflow_state(
                            attempt_id=row["id"],
                            attempt_updates=attempt_updates,
                            stage_id=row["stage_id"],
                            stage_updates={"status": "CANCELLING", "completed_at": None},
                            run_id=None if is_evaluation_stage else row["run_id"],
                            run_updates=None if is_evaluation_stage else {
                                "status": "CANCELLING",
                                "completed_at": None,
                            },
                            evaluation_id=evaluation["id"] if evaluation else None,
                            evaluation_updates={
                                "status": "CANCELLING",
                                "completed_at": None,
                            } if evaluation else None,
                        )
                        self._signal_stage_cancellation(
                            entity_type="evaluation" if evaluation else "run",
                            entity_id=str(evaluation["id"])
                            if evaluation else str(row["run_id"]),
                            attempt=row,
                            record_event=False,
                        )
                        updated += 1
                        continue
                    attempt_status = "RUNNING" if state == "RUNNING" else "PENDING"
                    stage_status = "RUNNING" if state == "RUNNING" else "PENDING_SLURM"
                    attempt_updates = {"status": attempt_status, **common}
                    if state == "RUNNING" and not row["started_at"]:
                        attempt_updates["started_at"] = accounting_started or utc_now()
                    self.database.transition_workflow_state(
                        attempt_id=row["id"],
                        attempt_updates=attempt_updates,
                        stage_id=row["stage_id"],
                        stage_updates={"status": stage_status},
                        run_id=None if is_evaluation_stage else row["run_id"],
                        run_updates=None if is_evaluation_stage else {"status": attempt_status},
                        evaluation_id=evaluation["id"] if evaluation else None,
                        evaluation_updates={"status": attempt_status} if evaluation else None,
                    )
                    updated += 1
                    continue
                if state == "COMPLETED":
                    finished = accounting_finished or utc_now()
                    attempt_success_updates = {
                        "status": "SUCCEEDED",
                        "started_at": row["started_at"] or accounting_started or row["submitted_at"] or finished,
                        "finished_at": finished,
                        **common,
                    }
                    if is_evaluation_stage:
                        ingestion_error, completed_episodes = self._ingest_evaluation_result(row)
                        if ingestion_error:
                            self.database.transition_workflow_state(
                                attempt_id=row["id"],
                                attempt_updates=attempt_success_updates,
                                stage_id=row["stage_id"],
                                stage_updates={"status": "FAILED", "completed_at": finished},
                                evaluation_id=evaluation["id"] if evaluation else None,
                                evaluation_updates={"status": "FAILED", "completed_at": finished}
                                if evaluation else None,
                                event={
                                    "entity_type": "evaluation" if evaluation else "run",
                                    "entity_id": evaluation["id"] if evaluation else row["run_id"],
                                    "event_type": "EVALUATION_RESULT_INVALID",
                                    "new_status": "FAILED",
                                    "details": {"error": ingestion_error, "job_id": row["slurm_job_id"]},
                                },
                            )
                            updated += 1
                            continue
                        self.database.transition_workflow_state(
                            attempt_id=row["id"],
                            attempt_updates=attempt_success_updates,
                            stage_id=row["stage_id"],
                            stage_updates={"status": "SUCCEEDED", "completed_at": finished},
                            evaluation_id=evaluation["id"] if evaluation else None,
                            evaluation_updates={
                                "status": "SUCCEEDED",
                                "progress_completed": completed_episodes,
                                "completed_at": finished,
                            } if evaluation else None,
                        )
                        updated += 1
                        continue
                    if row["stage_type"] == "TRAIN":
                        checkpoint_globs: list[str] = []
                        try:
                            checkpoint_globs = self._checkpoint_globs(row["resolved_config_json"])
                            self._capture_checkpoint(
                                row["run_id"],
                                row["id"],
                                required=bool(checkpoint_globs),
                                gateway=row["gateway"] or "auto",
                            )
                        except Exception as error:
                            message = sanitize(str(error))
                            self.database.transition_workflow_state(
                                attempt_id=row["id"],
                                attempt_updates=attempt_success_updates,
                                stage_id=row["stage_id"],
                                stage_updates={"status": "FAILED", "completed_at": finished},
                                run_id=row["run_id"],
                                run_updates={"status": "FAILED", "completed_at": finished},
                                event={
                                    "entity_type": "run",
                                    "entity_id": row["run_id"],
                                    "event_type": "CHECKPOINT_FINALIZATION_FAILED",
                                    "old_status": row["status"],
                                    "new_status": "FAILED",
                                    "details": {
                                        "error": message,
                                        "attempt_id": row["id"],
                                        "job_id": row["slurm_job_id"],
                                        "slurm_state": state,
                                        "exit_code": record.get("ExitCode"),
                                        "process_status": "SUCCEEDED",
                                        "checkpoint_globs": checkpoint_globs,
                                    },
                                },
                            )
                            self._finish_tracking(row["run_id"], "FAILED")
                            updated += 1
                            continue
                    self.database.transition_workflow_state(
                        attempt_id=row["id"],
                        attempt_updates=attempt_success_updates,
                        stage_id=row["stage_id"],
                        stage_updates={"status": "SUCCEEDED", "completed_at": finished},
                        run_id=None if is_evaluation_stage else row["run_id"],
                        run_updates=None if is_evaluation_stage else {"status": "SUCCEEDED", "completed_at": finished},
                    )
                    if not is_evaluation_stage:
                        self._finish_tracking(row["run_id"], "FINISHED")
                    updated += 1
                    continue
                if (
                    not cancellation_requested
                    and state in TRANSIENT_STATES
                    and row["auto_resume"]
                    and row["budget_attempt_count"] < row["max_attempts"]
                ):
                    finished = accounting_finished or utc_now()
                    self.database.transition_workflow_state(
                        attempt_id=row["id"],
                        attempt_updates={
                            "status": state,
                            "started_at": row["started_at"] or accounting_started or row["submitted_at"] or finished,
                            "finished_at": finished,
                            **common,
                        },
                        stage_id=row["stage_id"],
                        stage_updates={"status": "RETRY_PENDING", "completed_at": None},
                        run_id=None if is_evaluation_stage else row["run_id"],
                        run_updates=None if is_evaluation_stage else {"status": "RETRY_PENDING", "completed_at": None},
                        evaluation_id=evaluation["id"] if evaluation else None,
                        evaluation_updates={"status": "RETRY_PENDING", "completed_at": None}
                        if evaluation else None,
                        event={
                            "entity_type": "evaluation" if evaluation else "run",
                            "entity_id": evaluation["id"] if evaluation else row["run_id"],
                            "event_type": "AUTO_RESUME_QUEUED",
                            "old_status": state,
                            "new_status": "RETRY_PENDING",
                            "details": {"attempt": row["attempt_number"], "job_id": row["slurm_job_id"]},
                        },
                    )
                    updated += 1
                    continue
                if state in TRANSIENT_STATES | TERMINAL_FAILURE_STATES | {"CANCELLED"}:
                    finished = accounting_finished or utc_now()
                    target_status = (
                        "CANCELLED"
                        if cancellation_requested or state == "CANCELLED"
                        else "FAILED"
                    )
                    event_details = {
                                "attempt_id": row["id"],
                                "job_id": row["slurm_job_id"],
                                "state": state,
                                "state_raw": raw_state,
                                "cancelled_by": record.get("CancelledBy"),
                                "reason": reason,
                                "accounting_start_raw": record.get("Start"),
                                "accounting_end_raw": record.get("End"),
                                "started_at": row["started_at"] or accounting_started,
                                "finished_at": finished,
                                "exit_code": record.get("ExitCode"),
                                "node_list": record.get("NodeList"),
                    }
                    self.database.transition_workflow_state(
                        attempt_id=row["id"],
                        attempt_updates={
                            "status": target_status if cancellation_requested else state,
                            "started_at": row["started_at"] or accounting_started or row["submitted_at"] or finished,
                            "finished_at": finished,
                            **common,
                        },
                        stage_id=row["stage_id"],
                        stage_updates={"status": target_status, "completed_at": finished},
                        run_id=None if is_evaluation_stage else row["run_id"],
                        run_updates=None if is_evaluation_stage else {
                            "status": target_status,
                            "completed_at": finished,
                        },
                        evaluation_id=evaluation["id"] if evaluation else None,
                        evaluation_updates={"status": target_status, "completed_at": finished}
                        if evaluation else None,
                        event={
                            "entity_type": "evaluation" if evaluation else "run",
                            "entity_id": evaluation["id"] if evaluation else row["run_id"],
                            "event_type": "JOB_CANCELLED"
                            if target_status == "CANCELLED"
                            else "JOB_FAILED",
                            "old_status": row["status"],
                            "new_status": target_status,
                            "details": event_details,
                        },
                    )
                    if not is_evaluation_stage:
                        self._finish_tracking(
                            row["run_id"],
                            "KILLED" if target_status == "CANCELLED" else "FAILED",
                        )
                    updated += 1
            for experiment_id in experiments:
                self._dispatch_experiment(experiment_id)
                self._refresh_experiment_status(experiment_id)
            return {
                "ok": True,
                "checked": len(rows),
                "updated": updated,
                "repaired": repairs["repaired"],
                "draft_graphs_repaired": repairs["draft_graphs_repaired"],
                "recovered_submissions": recovered_submissions,
                "tracking_bindings_repaired": tracking_bindings_repaired,
            }
        finally:
            self._reconcile_lock.release()

    def _dispatch_active_experiments(self) -> None:
        for experiment in self.database.list_experiments(status="ACTIVE", limit=1000):
            self._dispatch_experiment(experiment["id"])
            self._refresh_experiment_status(experiment["id"])

    def _refresh_experiment_status(self, experiment_id: str) -> None:
        experiment = self.database.get_experiment(experiment_id)
        revision = experiment["latest_revision"] if experiment else None
        runs = self.database.list_runs(
            experiment_revision_id=revision["id"], limit=10000
        ) if revision else []
        states = {run["status"] for run in runs}
        if not runs or states <= {"DRAFT"}:
            self.database.update_experiment(experiment_id, status="DRAFT")
        elif states <= {"SUCCEEDED"}:
            self.database.update_experiment(experiment_id, status="SUCCEEDED")
        elif states & {
            "SUBMITTING", "RUNNING", "SUBMITTED", "PENDING", "RETRY_PENDING",
            "CANCELLING",
        }:
            self.database.update_experiment(experiment_id, status="ACTIVE")
        elif states <= {"SUCCEEDED", "CANCELLED"} and "CANCELLED" in states:
            self.database.update_experiment(experiment_id, status="CANCELLED")
        elif states & {"FAILED", "BLOCKED", "CANCELLED"}:
            self.database.update_experiment(experiment_id, status="FAILED")

    @staticmethod
    def _checkpoint_globs(resolved_config: Any) -> list[str]:
        if isinstance(resolved_config, str):
            try:
                resolved_config = json.loads(resolved_config)
            except json.JSONDecodeError as error:
                raise ValueError("stage resolved configuration is not valid JSON") from error
        if not isinstance(resolved_config, Mapping):
            raise ValueError("stage resolved configuration is missing")
        plan = resolved_config.get("plan")
        if not isinstance(plan, Mapping):
            raise ValueError("stage resolved configuration has no adapter plan")
        globs = plan.get("checkpoint_globs", [])
        if not isinstance(globs, list) or any(not isinstance(item, str) for item in globs):
            raise ValueError("adapter plan checkpoint_globs must be a list of strings")
        return [item for item in globs if item]

    def _capture_checkpoint(
        self,
        run_id: str,
        attempt_id: str,
        *,
        required: bool = False,
        gateway: str = "auto",
    ) -> dict[str, Any] | None:
        path = f"{WORK_ROOT}/jobs/runs/{run_id}/checkpoints/selected-for-inference.json"
        try:
            _, content = self.cluster.read_log(path, gateway, lines=100)
        except ClusterError as error:
            if required:
                raise RuntimeError(f"required inference checkpoint descriptor is unavailable: {error}") from error
            return None
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            if required:
                raise RuntimeError("required inference checkpoint descriptor is invalid JSON") from error
            return None
        if not isinstance(payload, Mapping):
            if required:
                raise RuntimeError("required inference checkpoint descriptor must be an object")
            return None
        checkpoint_path = payload.get("path")
        if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
            if required:
                raise RuntimeError("required inference checkpoint descriptor has no path")
            return None
        run_root = Path(f"{WORK_ROOT}/jobs/runs/{run_id}")
        checkpoint_target = Path(checkpoint_path)
        try:
            checkpoint_relative = checkpoint_target.relative_to(run_root)
        except ValueError:
            checkpoint_relative = None
        if (
            not checkpoint_target.is_absolute()
            or ".." in checkpoint_target.parts
            or checkpoint_relative in {None, Path(".")}
            or checkpoint_target.name in {"latest.json", "selected-for-inference.json"}
        ):
            if required:
                raise RuntimeError(
                    "required inference checkpoint path is outside the run namespace"
                )
            return None
        descriptor_run_id = payload.get("run_id")
        if descriptor_run_id is not None and descriptor_run_id != run_id:
            if required:
                raise RuntimeError("required inference checkpoint descriptor belongs to another run")
            return None
        size_bytes = payload.get("size_bytes")
        if (
            size_bytes is None
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            if required:
                raise RuntimeError("required inference checkpoint descriptor has invalid size_bytes")
            return None
        file_count = payload.get("file_count")
        if (
            isinstance(file_count, bool)
            or not isinstance(file_count, int)
            or file_count < 1
        ):
            if required:
                raise RuntimeError("required inference checkpoint descriptor has invalid file_count")
            return None
        if not isinstance(payload.get("is_directory"), bool):
            if required:
                raise RuntimeError("required inference checkpoint descriptor has invalid candidate type")
            return None
        if payload.get("final") is not True:
            if required:
                raise RuntimeError("required inference checkpoint descriptor is not final")
            return None
        checkpoint_sha256 = payload.get("sha256")
        if (
            not isinstance(checkpoint_sha256, str)
            or len(checkpoint_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in checkpoint_sha256)
        ):
            if required:
                raise RuntimeError("required inference checkpoint descriptor has invalid sha256")
            return None
        try:
            _, probe = self.cluster.run_with_fallback(
                "if test -e "
                + shlex.quote(checkpoint_path)
                + "; then printf '%s' SKYNET_CHECKPOINT_PRESENT; else exit 44; fi",
                gateway,
            )
        except ClusterError as error:
            if required:
                raise RuntimeError(
                    f"required inference checkpoint target is unavailable: {error}"
                ) from error
            return None
        if "SKYNET_CHECKPOINT_PRESENT" not in probe:
            if required:
                raise RuntimeError("required inference checkpoint target is unavailable")
            return None
        try:
            return self.database.create_checkpoint(
                run_id,
                produced_by_attempt_id=attempt_id,
                checkpoint_type="INFERENCE",
                path=checkpoint_path,
                sha256=checkpoint_sha256.lower(),
                size_bytes=size_bytes,
                is_resumable=False,
                is_selected_for_inference=True,
                metadata=payload,
            )
        except sqlite3.IntegrityError as error:
            if required:
                raise RuntimeError(f"required inference checkpoint could not be registered: {error}") from error
            return None

    def _evaluation_for_stage(self, run_id: str, stage_id: str) -> dict[str, Any] | None:
        return next(
            (
                evaluation
                for evaluation in self.database.list_evaluations(run_id=run_id)
                if evaluation.get("stage_id") == stage_id
            ),
            None,
        )

    @staticmethod
    def _training_progress_contract(
        run: Mapping[str, Any],
    ) -> tuple[TrainingProgressContract, str] | None:
        for stage in run.get("stages") or []:
            if str(stage.get("stage_type") or "").upper() != "TRAIN":
                continue
            resolved = stage.get("resolved_config_json") or {}
            if isinstance(resolved, str):
                try:
                    resolved = json.loads(resolved)
                except json.JSONDecodeError:
                    resolved = {}
            plan = resolved.get("plan") if isinstance(resolved, Mapping) else None
            progress = plan.get("progress") if isinstance(plan, Mapping) else None
            if progress:
                try:
                    return TrainingProgressContract.model_validate(progress), "pinned_plan"
                except Exception:
                    return None

        # Compatibility observer for active jobs created before the optional
        # progress field existed. Adapter-specific grammar remains in the
        # adapter declaration; the pipeline core still consumes one contract.
        adapter_name = str(run.get("adapter_name") or "")
        for manifest in builtin_adapter_manifests():
            if manifest.slug == adapter_name and manifest.train.progress is not None:
                return manifest.train.progress, "builtin_compatibility"
        return None

    def _ingest_training_progress(self, value: Mapping[str, Any]) -> int:
        status = str(value.get("status") or value.get("state") or "").upper()
        if status not in _PROGRESS_RUNNING_STATES:
            return 0
        run = value if value.get("stages") and value.get("attempts") else self.database.get_run(
            str(value.get("id") or value.get("run_id") or "")
        )
        if not run:
            return 0
        declared = self._training_progress_contract(run)
        if declared is None:
            return 0
        contract, declaration_origin = declared
        train_stage_ids = {
            str(stage["id"])
            for stage in run.get("stages") or []
            if str(stage.get("stage_type") or "").upper() == "TRAIN"
        }
        attempts = [
            attempt for attempt in run.get("attempts") or []
            if str(attempt.get("stage_id") or "") in train_stage_ids
        ]
        attempt = _latest_attempt(attempts)
        if not attempt or str(attempt.get("status") or "").upper() not in _PROGRESS_RUNNING_STATES:
            return 0
        attempt_id = str(attempt.get("id") or "")
        if not attempt_id:
            return 0
        source = contract.source
        path = attempt.get(f"{source.stream}_path")
        if not isinstance(path, str) or not path:
            return 0
        throttle_key = f"{run['id']}:{attempt_id}"
        monotonic_now = __import__("time").monotonic()
        last_reads = getattr(self, "_training_progress_last_reads", {})
        if monotonic_now - float(last_reads.get(throttle_key, 0.0)) < source.poll_seconds:
            return 0
        last_reads[throttle_key] = monotonic_now
        self._training_progress_last_reads = last_reads
        try:
            _, content = self.cluster.read_log(
                path,
                attempt.get("gateway") or "auto",
                lines=source.tail_lines,
                max_bytes=1_000_000,
            )
        except ClusterError:
            return 0
        records = parse_declared_training_progress(content, contract)
        expected_total = _resolved_training_max_steps(run.get("resolved_spec_json"))
        if expected_total is not None:
            records = [record for record in records if record["total"] == expected_total]
        if not records:
            return 0

        segment: list[dict[str, int | None]] = []
        for record in records:
            if segment:
                previous = segment[-1]
                reset = record["completed"] < previous["completed"]
                if record["elapsed_seconds"] is not None and previous["elapsed_seconds"] is not None:
                    reset = reset or record["elapsed_seconds"] < previous["elapsed_seconds"]
                if reset:
                    segment = []
            segment.append(record)
        latest = segment[-1]
        candidates = [latest]
        if (
            source.elapsed_format is not None
            and segment[0]["completed"] != latest["completed"]
        ):
            candidates.insert(0, segment[0])

        restart_count = _progress_integer(attempt.get("restart_count")) or 0
        existing = {
            int(sample["completed"])
            for sample in self.database.list_training_progress_samples(
                str(run["id"]), attempt_id=attempt_id
            )
            if int(sample.get("restart_count") or 0) == restart_count
        }
        observed_now = datetime.now(timezone.utc)
        latest_elapsed = latest.get("elapsed_seconds")
        inserted = 0
        for record in candidates:
            completed = int(record["completed"])
            if completed in existing:
                continue
            recorded_at = observed_now
            elapsed = record.get("elapsed_seconds")
            if latest_elapsed is not None and elapsed is not None and latest_elapsed >= elapsed:
                recorded_at -= timedelta(seconds=latest_elapsed - elapsed)
            self.database.record_training_progress_sample(
                str(run["id"]),
                attempt_id,
                restart_count=restart_count,
                completed=completed,
                total=int(record["total"]),
                source_kind=source.kind,
                evidence={
                    "declaration_origin": declaration_origin,
                    "stream": source.stream,
                    "path": path,
                    "elapsed_seconds": elapsed,
                },
                recorded_at=_progress_iso(recorded_at),
            )
            existing.add(completed)
            inserted += 1
        return inserted

    @staticmethod
    def _training_progress_timestamp_ms(value: Any) -> int:
        if not isinstance(value, str) or not value.strip():
            return int(datetime.now(timezone.utc).timestamp() * 1000)
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return int(datetime.now(timezone.utc).timestamp() * 1000)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)

    @staticmethod
    def _training_progress_metrics(sample: Mapping[str, Any]) -> dict[str, float | int]:
        completed = int(sample.get("completed") or 0)
        total = int(sample.get("total") or 0)
        metrics: dict[str, float | int] = {
            "training/completed": completed,
            "training/restart_count": int(sample.get("restart_count") or 0),
        }
        if total > 0:
            metrics["training/total"] = total
            metrics["training/progress"] = completed / total
        evidence = sample.get("evidence") or sample.get("evidence_json") or {}
        if isinstance(evidence, Mapping) and evidence.get("elapsed_seconds") is not None:
            metrics["training/elapsed_seconds"] = float(evidence["elapsed_seconds"])
        return metrics

    def _publish_training_progress_tracking(
        self,
        run_id: str,
        *,
        providers: list[Any] | None = None,
        include_native: bool = False,
    ) -> int:
        """Publish persisted progress samples without adapter-specific knowledge."""

        run = self.database.get_run(run_id)
        if not run:
            return 0
        try:
            spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
        except Exception:
            return 0
        native_providers = self._native_tracking_provider_names(spec)
        if providers is None:
            providers = self._active_tracking_providers(spec)
        if not include_native:
            providers = [
                provider
                for provider in providers
                if str(self._tracking_provider_value(provider, "provider"))
                not in native_providers
            ]
        samples = self.database.list_training_progress_samples(run_id)
        if not providers or not samples:
            return 0

        published = 0
        for provider in providers:
            name = str(self._tracking_provider_value(provider, "provider"))
            try:
                if name == "mlflow":
                    bridge: Any = MLflowBridge(
                        LOCAL_CAPSULE_ROOT / run_id, self._mlflow_settings(provider)
                    )
                elif name == "wandb":
                    bridge = WandBBridge(
                        LOCAL_CAPSULE_ROOT / run_id, self._wandb_settings(provider)
                    )
                else:
                    raise ValueError(f"unsupported tracking provider: {name}")
                if not bridge.binding(run_id):
                    continue
                emitted = bridge.metric_idempotency_keys()
                for sample in samples:
                    sample_id = str(sample.get("id") or "")
                    if not sample_id:
                        continue
                    idempotency_key = f"training-progress:{sample_id}"
                    if idempotency_key in emitted:
                        continue
                    bridge.log_metrics(
                        run_id,
                        self._training_progress_metrics(sample),
                        step=int(sample.get("completed") or 0),
                        timestamp_ms=self._training_progress_timestamp_ms(
                            sample.get("recorded_at")
                        ),
                        idempotency_key=idempotency_key,
                    )
                    emitted.add(idempotency_key)
                    published += 1
            except Exception as error:
                self._tracking_failure(name, run, error)
        return published

    def _set_evaluation_status(
        self, run_id: str, stage_id: str, status: str, *, completed: bool = False
    ) -> None:
        evaluation = self._evaluation_for_stage(run_id, stage_id)
        if evaluation:
            now = utc_now()
            started_at = evaluation.get("started_at")
            if status == "RUNNING" and not started_at:
                started_at = now
            self.database.update_evaluation(
                evaluation["id"],
                status=status,
                started_at=started_at,
                completed_at=now if completed else evaluation.get("completed_at"),
            )

    def _ingest_evaluation_progress(self, evaluation: Mapping[str, Any]) -> None:
        """Import changed canonical episode records while an evaluation is running."""
        if evaluation.get("status") not in {"PENDING", "SUBMITTED", "RUNNING"}:
            return

        # The list and detail endpoints can be polled together. Keep that from
        # causing duplicate SSH reads while still making progress visibly live.
        now = __import__("time").monotonic()
        last_reads = getattr(self, "_evaluation_progress_last_reads", {})
        if now - float(last_reads.get(evaluation["id"], 0.0)) < 4.0:
            return
        last_reads[evaluation["id"]] = now
        self._evaluation_progress_last_reads = last_reads

        run = self.database.get_run(str(evaluation["run_id"]))
        if not run:
            return
        attempts = [
            attempt
            for attempt in run.get("attempts", [])
            if attempt.get("stage_id") == evaluation.get("stage_id")
        ]
        if not attempts:
            return
        attempt = max(attempts, key=lambda item: int(item.get("attempt_number") or 0))
        gateway = attempt.get("gateway") or "auto"

        stage = next(
            (
                item
                for item in run.get("stages", [])
                if item.get("id") == evaluation.get("stage_id")
            ),
            None,
        )
        resolved = (stage or {}).get("resolved_config_json") or {}
        context = resolved.get("context") or {}
        progress_path = context.get("progress_path")
        if not isinstance(progress_path, str) or not progress_path:
            result_path = evaluation.get("result_path")
            if not isinstance(result_path, str) or "/" not in result_path:
                return
            progress_path = result_path.rsplit("/", 1)[0] + "/progress.jsonl"

        try:
            _, content = self.cluster.read_file(
                progress_path,
                gateway,
                max_bytes=20_000_000,
            )
        except ClusterError:
            # The progress file legitimately does not exist during scheduler
            # queueing and process startup.
            return

        tasks = evaluation.get("task_selection_json") or []
        seeds = evaluation.get("seeds_json") or []
        episodes_per_task = int(evaluation.get("episodes_per_task") or 0)
        expected = {
            (task, seed, episode_index)
            for task in tasks
            for seed in seeds
            for episode_index in range(episodes_per_task)
        }
        expected_total = len(expected)
        observed: dict[tuple[str, int, int], tuple[dict[str, Any], dict[str, Any]]] = {}
        for line in content.splitlines():
            try:
                record = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict) or record.get("kind") != "episode_observed":
                continue
            if record.get("total") != expected_total:
                continue
            episode = record.get("episode")
            if not isinstance(episode, dict):
                continue
            task = episode.get("task")
            seed = episode.get("seed")
            episode_index = episode.get("episode_index")
            if (
                not isinstance(task, str)
                or isinstance(seed, bool)
                or not isinstance(seed, int)
                or isinstance(episode_index, bool)
                or not isinstance(episode_index, int)
            ):
                continue
            key = (task, seed, episode_index)
            if key not in expected:
                continue
            status = episode.get("status")
            success = episode.get("success")
            metrics = episode.get("metrics")
            if status not in {"SUCCEEDED", "FAILED", "TIMEOUT"}:
                continue
            if success is not None and not isinstance(success, bool):
                continue
            if not isinstance(metrics, dict):
                continue
            observed[key] = (episode, record)

        if not observed:
            return
        current_evaluation = self.database.get_evaluation(str(evaluation["id"]))
        if not current_evaluation:
            return
        current_by_key = {
            (item["task"], item["seed"], item["episode_index"]): item
            for item in current_evaluation.get("episodes", [])
        }
        attempt_number = max(1, int(attempt.get("attempt_number") or 1))
        for key, (episode, record) in observed.items():
            current = current_by_key.get(key) or {}
            current_success = current.get("success")
            if current_success is not None:
                current_success = bool(current_success)
            desired = {
                "status": episode["status"],
                "attempt_count": attempt_number,
                "success": episode.get("success"),
                "reward": episode.get("reward"),
                "episode_length": episode.get("episode_length"),
                "failure_reason": episode.get("failure_reason"),
                "metrics_json": episode.get("metrics") or {},
                "video_path": episode.get("video_path"),
            }
            existing = {
                "status": current.get("status"),
                "attempt_count": int(current.get("attempt_count") or 0),
                "success": current_success,
                "reward": current.get("reward"),
                "episode_length": current.get("episode_length"),
                "failure_reason": current.get("failure_reason"),
                "metrics_json": current.get("metrics_json") or {},
                "video_path": current.get("video_path"),
            }
            if existing == desired:
                continue
            recorded_at = record.get("recorded_at")
            self.database.upsert_evaluation_episode(
                str(evaluation["id"]),
                task=key[0],
                seed=key[1],
                episode_index=key[2],
                status=desired["status"],
                attempt_count=attempt_number,
                success=desired["success"],
                reward=desired["reward"],
                episode_length=desired["episode_length"],
                failure_reason=desired["failure_reason"],
                metrics=desired["metrics_json"],
                video_path=desired["video_path"],
                completed_at=recorded_at if isinstance(recorded_at, str) else utc_now(),
            )

    def _ingest_evaluation_result(
        self, attempt: Mapping[str, Any]
    ) -> tuple[str | None, int]:
        evaluation = self._evaluation_for_stage(attempt["run_id"], attempt["stage_id"])
        if not evaluation:
            return "evaluation record is missing", 0
        try:
            _, content = self.cluster.read_file(
                evaluation["result_path"],
                attempt["gateway"] or "auto",
                max_bytes=20_000_000,
            )
            result = CanonicalResult.model_validate_json(content)
            if result.run_id != attempt["run_id"]:
                raise ValueError("canonical result run_id does not match the evaluated run")
            if (result.evaluator.adapter, result.evaluator.version) != (
                evaluation["evaluator_adapter"],
                evaluation["evaluator_version"],
            ):
                raise ValueError("canonical result evaluator identity does not match the request")
            if (result.environment.suite, result.environment.version) != (
                evaluation["suite_name"],
                evaluation["suite_version"],
            ):
                raise ValueError("canonical result suite identity does not match the request")
            expected = {
                (task, seed, episode_index)
                for task in evaluation["task_selection_json"]
                for seed in evaluation["seeds_json"]
                for episode_index in range(evaluation["episodes_per_task"])
            }
            received = {
                (episode.task, episode.seed, episode.episode_index) for episode in result.episodes
            }
            if len(received) != len(result.episodes):
                raise ValueError("canonical result contains duplicate episode keys")
            if received != expected:
                missing = len(expected - received)
                unexpected = len(received - expected)
                raise ValueError(
                    f"canonical result episode ledger mismatch: {missing} missing, {unexpected} unexpected"
                )
            finished = utc_now()
            for episode in result.episodes:
                self.database.upsert_evaluation_episode(
                    evaluation["id"],
                    task=episode.task,
                    seed=episode.seed,
                    episode_index=episode.episode_index,
                    status=episode.status,
                    attempt_count=1,
                    success=episode.success,
                    reward=episode.reward,
                    episode_length=episode.episode_length,
                    failure_reason=episode.failure_reason,
                    metrics=episode.metrics,
                    video_path=episode.video_path,
                    raw_result_path=result.raw_metrics_path,
                    completed_at=finished,
                )
            try:
                self.database.create_artifact(
                    attempt["run_id"],
                    stage_id=attempt["stage_id"],
                    evaluation_id=evaluation["id"],
                    artifact_type="EVALUATION_RESULT",
                    path=evaluation["result_path"],
                    retention_policy="KEEP",
                    metadata={
                        "schema_version": result.schema_version,
                        "aggregate": [metric.model_dump(mode="json") for metric in result.aggregate],
                    },
                )
            except sqlite3.IntegrityError:
                pass
            linked_outputs = [
                (result.raw_metrics_path, "EVALUATION_RAW_METRICS", "canonical raw metrics")
            ]
            linked_outputs.extend(
                (path, "EVALUATION_ARTIFACT", "canonical evaluator artifact")
                for path in result.artifacts
            )
            for path, artifact_type, description in linked_outputs:
                if not path:
                    continue
                try:
                    self.database.create_artifact(
                        attempt["run_id"],
                        stage_id=attempt["stage_id"],
                        evaluation_id=evaluation["id"],
                        artifact_type=artifact_type,
                        path=path,
                        retention_policy="KEEP",
                        metadata={
                            "description": description,
                            "reference_only": True,
                            "uploaded": False,
                        },
                    )
                except sqlite3.IntegrityError:
                    pass
            self._sync_tracking_outputs(str(attempt["run_id"]))
            return None, len(result.episodes)
        except Exception as error:
            message = sanitize(str(error))
            return message, 0

    @staticmethod
    def _flatten_tracking_metrics(
        value: Any, *, prefix: str = "result", limit: int = 200
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        def visit(item: Any, path: str) -> None:
            if len(metrics) >= limit:
                return
            if isinstance(item, bool) or item is None:
                return
            if isinstance(item, (int, float)):
                if math.isfinite(float(item)):
                    metrics[path[:250]] = float(item)
                return
            if isinstance(item, Mapping):
                metric_name = item.get("name") or item.get("key") or item.get("metric")
                metric_value = item.get("value")
                if isinstance(metric_name, str) and isinstance(metric_value, (int, float)) \
                        and not isinstance(metric_value, bool):
                    visit(metric_value, f"{prefix}/{metric_name}")
                    return
                for key, child in sorted(item.items(), key=lambda pair: str(pair[0])):
                    visit(child, f"{path}/{key}")
                return
            if isinstance(item, list):
                for index, child in enumerate(item):
                    visit(child, f"{path}/{index}")

        visit(value, prefix)
        return metrics

    def _final_tracking_payload(
        self, run: Mapping[str, Any]
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        metrics: dict[str, float] = {}
        links: list[dict[str, Any]] = []
        evaluations = {
            str(item.get("id")): item for item in run.get("evaluations") or []
        }
        for artifact in run.get("artifacts") or []:
            artifact_type = str(artifact.get("artifact_type") or "artifact").lower()
            metadata = artifact.get("metadata_json") or {}
            aggregate = metadata.get("aggregate") if isinstance(metadata, Mapping) else None
            if artifact_type == "evaluation_result" and isinstance(aggregate, list):
                evaluation = evaluations.get(str(artifact.get("evaluation_id") or ""), {})
                suite = re.sub(
                    r"[^A-Za-z0-9_.-]+", "-", str(evaluation.get("suite_name") or "suite")
                ).strip("-") or "suite"
                for metric in aggregate:
                    if not isinstance(metric, Mapping) or not metric.get("metric"):
                        continue
                    metric_name = re.sub(
                        r"[^A-Za-z0-9_.-]+", "-", str(metric["metric"])
                    ).strip("-") or "metric"
                    task = re.sub(
                        r"[^A-Za-z0-9_.-]+", "-", str(metric.get("task") or "all")
                    ).strip("-") or "all"
                    prefix = f"evaluation/{suite}/{task}/{metric_name}"
                    for statistic in ("mean", "std", "sample_count"):
                        value = metric.get(statistic)
                        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                                and math.isfinite(float(value)):
                            metrics[f"{prefix}/{statistic}"[:250]] = float(value)
            else:
                metrics.update(self._flatten_tracking_metrics(
                    metadata, prefix=f"artifact/{artifact_type}"
                ))
            if artifact.get("path"):
                links.append({
                    "name": f"{artifact_type}-{artifact.get('id', '')[:8]}",
                    "uri": str(artifact["path"]),
                    "artifact_type": artifact_type,
                    "sha256": artifact.get("sha256"),
                    "metadata": metadata,
                })
        for checkpoint in run.get("checkpoints") or []:
            if checkpoint.get("path"):
                links.append({
                    "name": f"checkpoint-step-{checkpoint.get('training_step', 'unknown')}",
                    "uri": str(checkpoint["path"]),
                    "artifact_type": "checkpoint",
                    "sha256": checkpoint.get("sha256"),
                    "metadata": {
                        "training_step": checkpoint.get("training_step"),
                        "selected_for_inference": bool(
                            checkpoint.get("is_selected_for_inference")
                        ),
                        "resumable": bool(checkpoint.get("is_resumable")),
                    },
                })
        for manifest in run.get("manifests") or []:
            if manifest.get("path"):
                links.append({
                    "name": f"manifest-{manifest.get('manifest_type', 'run')}-{manifest.get('id', '')[:8]}",
                    "uri": str(manifest["path"]),
                    "artifact_type": "manifest",
                    "sha256": manifest.get("sha256"),
                    "metadata": {"schema_version": manifest.get("schema_version")},
                })
        for attempt in run.get("attempts") or []:
            attempt_number = attempt.get("attempt_number") or "unknown"
            for field, artifact_type in (
                ("sbatch_path", "sbatch"),
                ("stdout_path", "stdout"),
                ("stderr_path", "stderr"),
            ):
                if attempt.get(field):
                    links.append({
                        "name": f"attempt-{attempt_number}-{artifact_type}",
                        "uri": str(attempt[field]),
                        "artifact_type": artifact_type,
                        "metadata": {
                            "attempt": attempt_number,
                            "slurm_job_id": attempt.get("slurm_job_id"),
                        },
                    })
        for evaluation in run.get("evaluations") or []:
            metrics.update(self._flatten_tracking_metrics(
                evaluation.get("metrics_json") or {},
                prefix=f"evaluation/{evaluation.get('id', 'unknown')}",
            ))
            if evaluation.get("result_path"):
                links.append({
                    "name": f"evaluation-{evaluation.get('id', '')[:8]}-result",
                    "uri": str(evaluation["result_path"]),
                    "artifact_type": "evaluation_result",
                    "metadata": {
                        "suite_id": evaluation.get("suite_id"),
                        "status": evaluation.get("status"),
                    },
                })
            try:
                detail = self.database.get_evaluation(str(evaluation["id"])) or {}
            except Exception:
                detail = {}
            for episode in detail.get("episodes") or []:
                metrics.update(self._flatten_tracking_metrics(
                    episode.get("metrics_json") or {},
                    prefix=(
                        f"evaluation/{evaluation.get('id', 'unknown')}/"
                        f"episode/{episode.get('id', 'unknown')}"
                    ),
                ))
                if episode.get("video_path"):
                    links.append({
                        "name": f"evaluation-{evaluation.get('id', '')[:8]}-video-{episode.get('id', '')[:8]}",
                        "uri": str(episode["video_path"]),
                        "artifact_type": "rollout_video",
                        "metadata": {
                            "task": episode.get("task"),
                            "seed": episode.get("seed"),
                            "success": episode.get("success"),
                        },
                    })
        return dict(list(metrics.items())[:200]), links

    def _sync_tracking_outputs(
        self,
        run_id: str,
        status: str | None = None,
        *,
        providers: list[Any] | None = None,
        central_authoritative: bool = False,
    ) -> None:
        run = self.database.get_run(run_id)
        if not run:
            return
        spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
        providers = (
            list(providers)
            if providers is not None
            else self._active_tracking_providers(spec)
        )
        if not providers:
            return
        native_providers = (
            set() if central_authoritative else self._native_tracking_provider_names(spec)
        )
        latest_job_id = ""
        for attempt in reversed(run.get("attempts") or []):
            if attempt.get("slurm_job_id"):
                latest_job_id = str(attempt["slurm_job_id"])
                break
        self._start_tracking(
            spec,
            run,
            LOCAL_CAPSULE_ROOT / run_id,
            latest_job_id,
            providers=providers,
        )
        self._publish_training_progress_tracking(
            run_id,
            providers=providers,
            include_native=central_authoritative,
        )
        metrics, links = self._final_tracking_payload(
            self.database.get_run(run_id) or run
        )
        for provider in providers:
            name = str(self._tracking_provider_value(provider, "provider"))
            if name in native_providers:
                # The repository process is the sole history/final-state writer for
                # a declared native provider. The central bridge still owns the
                # durable binding, tags, parameters, and app URL above.
                continue
            try:
                if name == "mlflow":
                    bridge: Any = MLflowBridge(
                        LOCAL_CAPSULE_ROOT / run_id, self._mlflow_settings(provider)
                    )
                elif name == "wandb":
                    bridge = WandBBridge(
                        LOCAL_CAPSULE_ROOT / run_id, self._wandb_settings(provider)
                    )
                else:
                    raise ValueError(f"unsupported tracking provider: {name}")
                bridge.drain_spool()
                if metrics:
                    bridge.log_metrics(run_id, metrics)
                for link in links:
                    bridge.log_artifact_link(run_id, **link)
                result = bridge.finish_run(run_id, status=status) if status else None
                binding = bridge.binding(run_id)
                existing = next(
                    (
                        item for item in self.database.list_tracking_bindings("run", run_id)
                        if item["provider"] == name
                    ),
                    {},
                )
                self.database.upsert_tracking_binding(
                    name,
                    "run",
                    run_id,
                    remote_id=(binding or {}).get("remote_id") or existing.get("remote_id"),
                    remote_url=(binding or {}).get("url") or existing.get("remote_url"),
                    status=(
                        status or "CONNECTED"
                        if not result or result.delivered
                        else "QUEUED"
                    ),
                    metadata=existing.get("metadata_json") or {},
                    last_error=result.error if result else None,
                )
            except Exception as error:
                self._tracking_failure(name, run, error)

    def _finish_tracking(self, run_id: str, status: str) -> None:
        self._sync_tracking_outputs(run_id, status=status)

    def retry_run(self, run_id: str, mode: str, gateway: str) -> dict[str, Any]:
        if mode != "resume":
            raise ValueError(
                "clean retry is deprecated; use POST /api/runs/{run_id}/rerun"
            )
        with self._reconcile_lock:
            run = self.database.get_run(run_id)
            if not run:
                raise KeyError("Run not found")
            action = self.run_manual_actions(run)["resume"]
            if not action["enabled"]:
                raise ValueError(str(action["reason"]))
            stage = next(item for item in reversed(run["stages"]) if item["stage_type"] == "TRAIN")
            checkpoint = _resumable_checkpoint(run)
            pinned_stage, pinned_stage_error = _pinned_training_execution(
                run, stage, prefer_initial_attempt=True
            )
            can_resume_checkpoint = bool(
                checkpoint
                and pinned_stage
                and (pinned_stage.get("plan") or {}).get("resume_argv")
            )
            if can_resume_checkpoint:
                strategy = "checkpoint_resume"
                pinned_execution = pinned_stage
            else:
                pinned_execution = pinned_stage
                if pinned_execution is None:
                    raise ValueError(
                        "immutable pinned execution evidence is unavailable: "
                        f"{pinned_stage_error}"
                    )
                checkpoint = None
                strategy = "pinned_initial_replay"
            self.database.transition_workflow_state(
                stage_id=stage["id"],
                stage_updates={"status": "RETRY_PENDING", "completed_at": None},
                run_id=run_id,
                run_updates={"status": "RETRY_PENDING", "completed_at": None},
                event={
                    "entity_type": "run",
                    "entity_id": run_id,
                    "event_type": "MANUAL_RESUME_QUEUED",
                    "new_status": "RETRY_PENDING",
                    "details": {
                        "gateway": gateway,
                        "strategy": strategy,
                        "checkpoint_id": checkpoint["id"] if checkpoint else None,
                        "source_attempt_id": pinned_execution.get("source_attempt_id"),
                    },
                },
            )
            result = self._submit_stage(
                run_id,
                stage["id"],
                gateway,
                manual_mode="resume" if checkpoint else "replay_initial",
                resume_checkpoint=checkpoint["path"] if checkpoint else None,
                resume_checkpoint_id=checkpoint["id"] if checkpoint else None,
                pinned_execution=pinned_execution,
            )
            return result

    def rerun_run(self, run_id: str, gateway: str) -> dict[str, Any]:
        """Start a checkpoint-free run from the exact immutable source variant."""

        with self._reconcile_lock:
            source = self.database.get_run(run_id)
            if not source:
                raise KeyError("Run not found")
            action = self.run_manual_actions(source)["rerun"]
            if not action["enabled"]:
                raise ValueError(str(action["reason"]))
            stage = next(
                item
                for item in reversed(source["stages"])
                if item["stage_type"] == "TRAIN"
            )
            rerun = self.database.create_run(
                source["variant_id"],
                seed=int(source["seed"]),
                adapter_name=source["adapter_name"],
                adapter_version=str(source["adapter_version"]),
                source_commit=source.get("source_commit"),
                runtime_profile=source.get("runtime_profile"),
                run_directory=f"{WORK_ROOT}/jobs/runs/pending",
                status="PENDING",
                restarted_from_run_id=run_id,
            )
            self.database.update_run(
                rerun["id"],
                run_directory=f"{WORK_ROOT}/jobs/runs/{rerun['id']}",
            )
            rerun_stage = self.database.create_stage(
                rerun["id"],
                stage_type="TRAIN",
                name="train",
                status="PENDING",
                auto_resume=bool(stage["auto_resume"]),
                max_attempts=int(stage["max_attempts"]),
                resolved_config=stage["resolved_config_json"],
            )
            self.database.record_event(
                entity_type="run",
                entity_id=rerun["id"],
                event_type="RERUN_CREATED",
                new_status="PENDING",
                details={
                    "restarted_from_run_id": run_id,
                    "experiment_revision_id": source["experiment_revision_id"],
                },
            )
            self.database.update_experiment(source["experiment_id"], status="ACTIVE")
            result = self._submit_stage(rerun["id"], rerun_stage["id"], gateway)
            result["restarted_from_run_id"] = run_id
            result["run"] = self.database.get_run(rerun["id"])
            return result

    def _read_attempt_log(
        self,
        attempt: Mapping[str, Any],
        stream: str,
        lines: int,
        *,
        retrieval_errors_as_text: bool = False,
    ) -> str:
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be stdout or stderr")
        path = attempt.get("stderr_path" if stream == "stderr" else "stdout_path")
        if not path:
            message = f"No {stream} path has been recorded."
            if retrieval_errors_as_text:
                return message
            raise KeyError(message)
        try:
            _, content = self.cluster.read_log(path, attempt.get("gateway") or "auto", lines=lines)
            return content
        except ClusterError as error:
            if "SKYNET_LOG_NOT_READY" in str(error):
                state = str(attempt.get("status") or attempt.get("slurm_state") or "").upper()
                if state in {
                    "CREATED", "SUBMITTED", "PENDING", "RUNNING", "REQUEUED",
                    "RETRY_PENDING", "CANCELLING",
                }:
                    return f"Waiting for Slurm to create {stream} output."
                return f"No {stream} log file was produced for this attempt."
            if retrieval_errors_as_text:
                return f"{error}\nLog path: {path}"
            raise

    def run_attempt_log(self, run_id: str, attempt_id: str, stream: str, lines: int) -> str:
        run = self.database.get_run(run_id)
        if not run:
            raise KeyError("Run not found")
        attempt = next(
            (item for item in run["attempts"] if str(item.get("id")) == attempt_id),
            None,
        )
        if attempt is None:
            raise KeyError("Attempt not found")
        return self._read_attempt_log(attempt, stream, lines)

    def run_log(self, run_id: str, stream: str, lines: int) -> str:
        run = self.database.get_run(run_id)
        if not run:
            raise KeyError("Run not found")
        attempts = run["attempts"]
        if not attempts:
            return "No Slurm attempt has been submitted."
        return self._read_attempt_log(
            attempts[-1], stream, lines, retrieval_errors_as_text=True
        )

    def _cancel_stage(
        self,
        *,
        entity_type: str,
        entity_id: str,
        run: Mapping[str, Any],
        stage: Mapping[str, Any],
    ) -> dict[str, Any]:
        claim = self.database.claim_workflow_cancellation(
            str(stage["id"]),
            entity_type=entity_type,
            entity_id=entity_id,
        )
        signals: list[dict[str, Any]] = []
        if claim.get("claimed") and str(claim["status"]).upper() == "CANCELLING":
            for attempt in claim.get("active_attempts") or []:
                signals.append(
                    self._signal_stage_cancellation(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        attempt=attempt,
                        record_event=bool(claim.get("claimed")),
                    )
                )
        if entity_type == "run":
            refreshed = self.database.get_run(entity_id)
            if refreshed:
                refreshed["manual_actions"] = self.run_manual_actions(refreshed)
                self._refresh_experiment_status(str(refreshed["experiment_id"]))
            entity: Mapping[str, Any] | None = refreshed
        else:
            refreshed_evaluation = self.database.get_evaluation(entity_id)
            parent = self.database.get_run(str(run["id"]))
            if refreshed_evaluation:
                refreshed_evaluation["attempts"] = [
                    attempt for attempt in list((parent or {}).get("attempts") or [])
                    if attempt.get("stage_id") == refreshed_evaluation.get("stage_id")
                ]
                refreshed_evaluation["manual_actions"] = self.evaluation_manual_actions(
                    refreshed_evaluation, parent
                )
            entity = refreshed_evaluation
        latest_attempt = (claim.get("active_attempts") or [None])[-1]
        latest_signal = signals[-1] if signals else (latest_attempt or {})
        return {
            "ok": all(bool(signal.get("ok")) for signal in signals),
            "status": str(claim["status"]).upper(),
            "stage_id": str(stage["id"]),
            "job_id": latest_signal.get("job_id"),
            "gateway": latest_signal.get("gateway"),
            "signals": signals,
            entity_type: entity,
        }

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        run = self.database.get_run(run_id)
        if not run:
            raise KeyError("Run not found")
        stage = _run_cancellation_stage(run)
        if stage is None:
            raise ValueError("Run has no cancellable training or utility stage")
        state = str(stage.get("status") or "").upper()
        action = manual_cancel_action(stage, list(run.get("attempts") or []))
        if not action["enabled"] and state not in {"CANCELLING", "CANCELLED"}:
            raise ValueError(f"Training cancellation is unavailable: {action['reason']}")
        return self._cancel_stage(
            entity_type="run",
            entity_id=run_id,
            run=run,
            stage=stage,
        )

    def cancel_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        evaluation = self.database.get_evaluation(evaluation_id)
        if not evaluation:
            raise KeyError("Evaluation not found")
        run = self.database.get_run(str(evaluation["run_id"]))
        if not run:
            raise KeyError("Parent run not found")
        stage = next(
            (
                item for item in list(run.get("stages") or [])
                if item.get("id") == evaluation.get("stage_id")
                and str(item.get("stage_type") or "").upper() == "EVALUATE"
            ),
            None,
        )
        if stage is None:
            raise ValueError("Evaluation has no matching evaluation stage")
        state = str(stage.get("status") or "").upper()
        attempts = [
            attempt for attempt in list(run.get("attempts") or [])
            if attempt.get("stage_id") == stage.get("id")
        ]
        action = manual_cancel_action(stage, attempts)
        if not action["enabled"] and state not in {"CANCELLING", "CANCELLED"}:
            raise ValueError(str(action["reason"]))
        return self._cancel_stage(
            entity_type="evaluation",
            entity_id=evaluation_id,
            run=run,
            stage=stage,
        )

    @staticmethod
    def _adapter_identity(spec: ExperimentSpec) -> dict[str, Any]:
        return {
            "id": spec.source.adapter_id,
            "version_id": spec.source.adapter_version_id,
            "slug": spec.source.adapter,
            "version": spec.source.adapter_version,
            "manifest": copy.deepcopy(spec.source.adapter_manifest),
            "manifest_sha256": spec.source.adapter_manifest_sha256,
        }

    @staticmethod
    def _resolve_dual_evaluation_runtimes(
        policy_runtime: Mapping[str, Any],
        runtime_profile_id: str,
        suite_config: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Pin policy and simulator runtimes without conflating their environments."""
        resolved: dict[str, Any] = {
            "policy_runtime": copy.deepcopy(dict(policy_runtime)),
        }
        blockers: list[str] = []
        try:
            profile = CLUSTER.runtime_profile(runtime_profile_id)
            profile_snapshot = CLUSTER.runtime_profile_snapshot(runtime_profile_id)
        except ValueError as error:
            blockers.append(
                f"evaluation runtime profile {runtime_profile_id} is not configured: {error}"
            )
            return resolved, blockers

        backend = str(profile.backend)
        environment_path = str(profile.environment_path or "").rstrip("/")
        if backend not in {"conda", "existing"}:
            blockers.append(
                f"evaluation runtime profile {runtime_profile_id} uses {backend}; "
                "dual-runtime evaluators require a conda or existing Python environment"
            )
        if not environment_path or not environment_path.startswith("/"):
            blockers.append(
                f"evaluation runtime profile {runtime_profile_id} must declare an absolute "
                "environment_path"
            )

        provenance = suite_config.get("task_catalog_provenance")
        if not isinstance(provenance, Mapping):
            provenance = {}
        source_repository = str(provenance.get("repository") or "").strip()
        source_revision = str(provenance.get("revision") or "").strip()
        prerequisites = list(profile_snapshot.get("source_prerequisites") or [])
        source_prerequisite = next(
            (
                item
                for item in prerequisites
                if isinstance(item, Mapping)
                and bool(item.get("required", True))
                and str(item.get("revision") or "").strip() == source_revision
            ),
            None,
        )
        source_dir = str((source_prerequisite or {}).get("path") or "").rstrip("/")
        if not source_repository or not source_revision:
            blockers.append(
                "evaluation suite must declare exact task_catalog_provenance repository and revision"
            )
        elif source_prerequisite is None:
            blockers.append(
                f"evaluation runtime profile {runtime_profile_id} does not declare a required "
                f"evaluator checkout at suite revision {source_revision}"
            )
        elif not source_dir.startswith("/"):
            blockers.append(
                f"evaluation runtime profile {runtime_profile_id} evaluator checkout path "
                "must be absolute"
            )

        versions = copy.deepcopy(dict(profile.versions))
        for version_name in ("python", "isaac_sim", "isaac_lab"):
            if not str(versions.get(version_name) or "").strip():
                blockers.append(
                    f"evaluation runtime profile {runtime_profile_id} must pin {version_name}"
                )
        runtime_environment = copy.deepcopy(dict(profile.environment))
        import os

        operator_environment: dict[str, str] = {}
        if os.environ.get("OMNI_KIT_ACCEPT_EULA") == "YES":
            operator_environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
        else:
            blockers.append(
                "NVIDIA Isaac Sim EULA acceptance is not configured. After reviewing and "
                "accepting the NVIDIA Isaac Sim EULA, set OMNI_KIT_ACCEPT_EULA=YES in the "
                "Skynet server operator environment; Skynet never accepts it by default"
            )
        resolved["evaluator_runtime"] = {
            "profile": runtime_profile_id,
            "profile_id": runtime_profile_id,
            "profile_snapshot": profile_snapshot,
            "profile_snapshot_sha256": content_sha256(profile_snapshot),
            "backend": backend,
            "environment_path": environment_path or None,
            "python_executable": (
                f"{environment_path}/bin/python" if environment_path else None
            ),
            "environment": runtime_environment,
            "operator_environment": operator_environment,
            "source_dir": source_dir or None,
            "source": {
                "repository": source_repository or None,
                "revision": source_revision or None,
            },
            "source_prerequisites": prerequisites,
            "versions": versions,
            "readiness": {
                "status": "configured" if not blockers else "blocked",
                "required_paths": [
                    path
                    for path in (environment_path, source_dir)
                    if path
                ],
            },
        }
        return resolved, list(dict.fromkeys(blockers))

    def _verify_evaluator_runtime_readiness(
        self,
        runtime_profile_id: str,
        gateway: str,
        *,
        operator_environment: Mapping[str, str] | None = None,
        suite_config: Mapping[str, Any] | None = None,
        refresh: bool = False,
    ) -> tuple[dict[str, Any], list[str]]:
        public_profile = next(
            (
                item
                for item in CLUSTER.public_runtime_profiles()
                if item.get("id") == runtime_profile_id
            ),
            None,
        )
        if public_profile is None:
            return {}, [
                f"evaluation runtime profile {runtime_profile_id} is not configured"
            ]
        from .runtime_readiness import suite_contract_sha256

        cache_key = (
            runtime_profile_id,
            str(gateway or "auto"),
            content_sha256(CLUSTER.runtime_profile_snapshot(runtime_profile_id)),
            content_sha256(approved_operator_environment(operator_environment)),
            suite_contract_sha256(suite_config) if suite_config is not None else "",
        )
        cache = getattr(self, "_evaluator_runtime_readiness_cache", None)
        cache_lock = getattr(self, "_evaluator_runtime_readiness_lock", None)
        if cache is None or cache_lock is None:
            cache = {}
            cache_lock = threading.Lock()
            self._evaluator_runtime_readiness_cache = cache
            self._evaluator_runtime_readiness_lock = cache_lock
        now = __import__("time").monotonic()
        if not refresh:
            with cache_lock:
                cached = cache.get(cache_key)
            if cached is not None and now - cached[0] < (60.0 if not cached[2] else 10.0):
                return copy.deepcopy(cached[1]), list(cached[2])
        readiness = self._probe_runtime_profile(
            public_profile,
            gateway,
            operator_environment=operator_environment,
            suite_config=suite_config,
        )
        verification = copy.deepcopy(readiness.get("verification") or {})
        if readiness.get("runtime_verified"):
            blockers: list[str] = []
        else:
            readiness_errors = list(verification.get("errors") or [])
            detail = "; ".join(str(item) for item in readiness_errors)
            blockers = [
                f"evaluation runtime profile {runtime_profile_id} is not ready"
                + (f": {detail}" if detail else "")
            ]
        with cache_lock:
            cache[cache_key] = (now, copy.deepcopy(verification), list(blockers))
        return verification, blockers

    def _resolve_evaluation_suite_selection(
        self,
        suite_id: str | None,
        environment: str | None,
        tasks: list[str],
    ) -> tuple[dict[str, Any] | None, str | None, list[str], dict[str, str]]:
        errors: dict[str, str] = {}
        normalized_suite_id = str(suite_id or "").strip()
        suite = next(
            (
                item
                for item in self.database.list_evaluation_suites()
                if item["id"] == normalized_suite_id
            ),
            None,
        )
        if suite is None:
            errors["suite_id"] = "Evaluation suite was not found."
            return None, None, [], errors

        canonical_environment = str(suite["evaluator_adapter"])
        requested_environment = str(environment or "").strip()
        if requested_environment and requested_environment != canonical_environment:
            errors["environment"] = (
                f"Evaluation suite {suite['name']} requires environment "
                f"{canonical_environment}, not {requested_environment}."
            )

        suite_config = suite["config_json"]
        catalog_tasks = list(
            dict.fromkeys(
                str(task).strip()
                for task in (suite_config.get("tasks") or [])
                if str(task).strip()
            )
        )
        requested_tasks = list(
            dict.fromkeys(task.strip() for task in tasks if task.strip())
        )
        catalog_complete = bool(suite_config.get("task_catalog_complete"))
        selection_mode = str(suite_config.get("task_selection_mode") or "subset")
        selection_reason = str(suite_config.get("task_selection_reason") or "").strip()
        if catalog_complete:
            if not catalog_tasks:
                errors["tasks"] = "The selected suite has an invalid empty complete task catalog."
            catalog_set = set(catalog_tasks)
            unknown_tasks = [task for task in requested_tasks if task not in catalog_set]
            if unknown_tasks:
                errors["tasks"] = (
                    "Tasks are not members of the selected evaluation suite: "
                    + ", ".join(unknown_tasks)
                )
            resolved_tasks = (
                []
                if selection_mode == "single" and not requested_tasks
                else requested_tasks or catalog_tasks
            )
        else:
            resolved_tasks = requested_tasks

        if "tasks" not in errors:
            reason_suffix = f" Reason: {selection_reason}" if selection_reason else ""
            if selection_mode == "single":
                if not requested_tasks:
                    errors["tasks"] = (
                        "This suite requires exactly one explicit task; an empty selection is not "
                        f"valid in single-task mode.{reason_suffix}"
                    )
                elif len(resolved_tasks) != 1:
                    errors["tasks"] = (
                        "This suite accepts exactly one task; select one task instead of "
                        f"{len(resolved_tasks)}.{reason_suffix}"
                    )
            elif selection_mode == "all_only" and requested_tasks:
                if not catalog_complete or set(requested_tasks) != set(catalog_tasks):
                    errors["tasks"] = (
                        "This suite only supports evaluating all tasks; leave Tasks empty."
                        f"{reason_suffix}"
                    )
                else:
                    resolved_tasks = catalog_tasks
            elif selection_mode not in {"subset", "single", "all_only"}:
                errors["tasks"] = (
                    f"The suite declares unsupported task selection mode {selection_mode!r}."
                )
        return suite, canonical_environment, resolved_tasks, errors

    def _resolve_evaluation_implementation(
        self,
        run: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
        suite: Mapping[str, Any],
        *,
        environment: str,
        tasks: list[str],
        seeds: list[int],
        episodes_per_task: int,
        parallelism: int,
        headless: bool,
        execution_key: str,
        resources: ResourceSpec | None = None,
        manual_argv: list[str] | None = None,
        manual_resume_argv: list[str] | None = None,
        verify_evaluator_runtime: bool = False,
        evaluator_runtime_gateway: str = "auto",
        refresh_evaluator_runtime: bool = False,
    ) -> tuple[ExperimentSpec, AdapterPlan, dict[str, Any], dict[str, Any], dict[str, Any], str]:
        training_spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
        training_adapter = self._adapter_identity(training_spec)
        training_document = training_spec.model_dump(mode="json", by_alias=True)

        evaluator_document = copy.deepcopy(training_document)
        evaluator_source = evaluator_document["source"]
        for key in (
            "adapter_id",
            "adapter_version_id",
            "adapter_version",
            "adapter_manifest",
            "adapter_manifest_sha256",
        ):
            evaluator_source.pop(key, None)
        evaluator_source["adapter"] = training_spec.source.adapter
        evaluator_document["source"], evaluator_manifest, _ = self._snapshot_adapter(
            evaluator_source, training_spec.source.adapter
        )
        if resources is not None:
            evaluator_document["resources"] = resources.model_dump(mode="json", by_alias=True)
        evaluator_spec = ExperimentSpec.model_validate(evaluator_document)
        evaluator_adapter = self._adapter_identity(evaluator_spec)

        evaluation_root = f"{WORK_ROOT}/eval/runs/{execution_key}"
        source_document = training_document["source"]
        suite_config = copy.deepcopy(suite["config_json"])
        context = {
            "schema_version": "skynet.evaluation-context/v1",
            "run_id": run["id"],
            "checkpoint": {
                key: checkpoint.get(key)
                for key in ("id", "path", "sha256", "checkpoint_type", "training_step")
            },
            "suite": {
                "id": suite["id"],
                "name": suite["name"],
                "version": suite["suite_version"],
                "evaluator_adapter": suite["evaluator_adapter"],
                "evaluator_version": suite["evaluator_version"],
                "catalog_path": suite.get("catalog_path"),
                "catalog_sha256": suite_config.get("catalog_sha256"),
                "config": suite_config,
                "config_sha256": content_sha256(suite_config),
                "task_source": suite_config.get("task_source"),
                "task_catalog_complete": bool(
                    suite_config.get("task_catalog_complete")
                ),
                "task_selection_mode": suite_config.get(
                    "task_selection_mode", "subset"
                ),
                "task_selection_reason": suite_config.get(
                    "task_selection_reason", ""
                ),
                "task_catalog_provenance": copy.deepcopy(
                    suite_config.get("task_catalog_provenance")
                ),
                "task_catalog_sha256": suite_config.get("task_catalog_sha256"),
                "tasks": copy.deepcopy(suite_config.get("tasks") or []),
                "task_options": copy.deepcopy(
                    suite_config.get("task_options") or []
                ),
            },
            "evaluator": {
                "adapter": suite["evaluator_adapter"],
                "version": suite["evaluator_version"],
            },
            "environment": environment,
            "tasks": list(tasks),
            "seeds": list(seeds),
            "episodes_per_task": episodes_per_task,
            "parallelism": parallelism,
            "headless": headless,
            "result_path": f"{evaluation_root}/result.json",
            "progress_path": f"{evaluation_root}/progress.jsonl",
            "video_path": f"{evaluation_root}/videos",
            "policy": {
                "adapter": training_spec.source.adapter,
                "adapter_version": training_spec.source.adapter_version,
                "manifest_sha256": training_spec.source.adapter_manifest_sha256,
                "native_config": copy.deepcopy(
                    training_document.get("native", {}).get("config", {})
                ),
                "source": {
                    key: source_document.get(key)
                    for key in ("repository", "revision", "project_subdirectory")
                },
            },
        }
        plan = resolve_adapter_evaluation_plan(
            evaluator_spec,
            environment=environment,
            suite=str(suite["name"]),
            context=context,
            manifest=evaluator_manifest,
        )
        evaluator_runtime_blockers: list[str] = []
        runtime_profile_id = str(
            plan.native_config.get("evaluation_runtime_profile_id") or ""
        ).strip()
        if runtime_profile_id:
            dual_runtime_context, evaluator_runtime_blockers = (
                self._resolve_dual_evaluation_runtimes(
                    training_spec.runtime.model_dump(mode="json"),
                    runtime_profile_id,
                    suite_config,
                )
            )
            context.update(dual_runtime_context)
            if (
                verify_evaluator_runtime
                and "evaluator_runtime" in context
                and not evaluator_runtime_blockers
            ):
                readiness, readiness_blockers = (
                    self._verify_evaluator_runtime_readiness(
                        runtime_profile_id,
                        evaluator_runtime_gateway,
                        operator_environment=context["evaluator_runtime"].get(
                            "operator_environment"
                        ),
                        suite_config=suite_config,
                        refresh=refresh_evaluator_runtime,
                    )
                )
                context["evaluator_runtime"]["readiness"] = readiness
                evaluator_runtime_blockers.extend(readiness_blockers)
            plan = resolve_adapter_evaluation_plan(
                evaluator_spec,
                environment=environment,
                suite=str(suite["name"]),
                context=context,
                manifest=evaluator_manifest,
            )
        plan_source = "registered_adapter"
        if manual_argv:
            plan.argv = list(manual_argv)
            plan.resume_argv = list(manual_resume_argv or [])
            plan.blockers = list(evaluator_runtime_blockers)
            plan_source = "manual_override"
        elif not plan.argv and not plan.blockers:
            plan.blockers.append("resolved evaluation plan has no executable command")
        else:
            plan.blockers = list(
                dict.fromkeys([*plan.blockers, *evaluator_runtime_blockers])
            )
        return (
            evaluator_spec,
            plan,
            context,
            training_adapter,
            evaluator_adapter,
            plan_source,
        )

    def _resolve_evaluation_target(
        self, run_id: str | None, checkpoint_path: str | None
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        normalized_run_id = str(run_id or "").strip()
        normalized_checkpoint_path = str(checkpoint_path or "").strip()
        errors: dict[str, str] = {}
        run: dict[str, Any] | None = None
        checkpoint: dict[str, Any] | None = None

        if not normalized_run_id:
            errors["run_id"] = "Training run ID is required."
        else:
            run = self.database.get_run(normalized_run_id)
            if run is None:
                errors["run_id"] = "Training run was not found."

        available_checkpoints: list[dict[str, Any]] = []
        if run is not None:
            available_checkpoints = [
                item
                for item in run.get("checkpoints", [])
                if item.get("status") == "AVAILABLE" and not item.get("pruned_at")
            ]
            if normalized_checkpoint_path:
                checkpoint = next(
                    (
                        item
                        for item in available_checkpoints
                        if item.get("path") == normalized_checkpoint_path
                    ),
                    None,
                )
                if checkpoint is None:
                    errors["checkpoint_path"] = (
                        "Checkpoint path is not a registered AVAILABLE checkpoint for this run."
                    )
            else:
                checkpoint = next(
                    (
                        item
                        for item in reversed(available_checkpoints)
                        if item.get("is_selected_for_inference")
                    ),
                    None,
                )
                if checkpoint is None:
                    errors["checkpoint_path"] = (
                        "This run has no selected AVAILABLE inference checkpoint."
                    )
        else:
            errors["checkpoint_path"] = (
                "Enter a valid training run before selecting a checkpoint."
            )

        public_checkpoints = [
            {
                key: item.get(key)
                for key in (
                    "id",
                    "path",
                    "status",
                    "checkpoint_type",
                    "is_selected_for_inference",
                    "is_resumable",
                    "sha256",
                    "training_step",
                )
            }
            for item in available_checkpoints
        ]
        response = {
            "valid": run is not None and checkpoint is not None,
            "run_valid": run is not None,
            "checkpoint_valid": checkpoint is not None,
            "errors": errors,
            "resolved_checkpoint_path": checkpoint.get("path") if checkpoint else None,
            "resolved_checkpoint_id": checkpoint.get("id") if checkpoint else None,
            "run_state": run.get("status") if run else None,
            "available_checkpoints": public_checkpoints,
        }
        return response, run, checkpoint

    def validate_evaluation_target(
        self,
        run_id: str | None,
        checkpoint_path: str | None,
        *,
        suite_id: str | None = None,
        environment: str | None = None,
        tasks: list[str] | None = None,
        episodes_per_task: int = 20,
        seeds: list[int] | None = None,
        parallelism: int = 1,
        headless: bool = True,
        resources: ResourceSpec | None = None,
        argv: list[str] | None = None,
        resume_argv: list[str] | None = None,
        gateway: str = "auto",
    ) -> dict[str, Any]:
        validation, run, checkpoint = self._resolve_evaluation_target(
            run_id, checkpoint_path
        )
        busy_reason = _evaluation_busy_reason(run) if run is not None else None
        if busy_reason:
            validation.update({"valid": False, "plan_valid": False, "plan_message": busy_reason, "plan_blockers": [busy_reason]})
            return validation
        if not str(suite_id or "").strip():
            return validation

        suite, resolved_environment, resolved_tasks, suite_errors = (
            self._resolve_evaluation_suite_selection(
                suite_id, environment, list(tasks or [])
            )
        )
        suite_config = suite["config_json"] if suite is not None else {}
        validation["errors"].update(suite_errors)
        validation.update(
            {
                "suite_valid": suite is not None and not suite_errors,
                "plan_valid": False,
                "evaluator": None,
                "plan_blockers": [],
                "plan_message": None,
                "plan_source": None,
                "resolved_environment": resolved_environment,
                "resolved_tasks": resolved_tasks,
                "task_selection_mode": suite_config.get(
                    "task_selection_mode", "subset"
                ),
                "task_selection_reason": suite_config.get(
                    "task_selection_reason", ""
                ),
            }
        )
        if not validation["valid"]:
            validation["plan_message"] = (
                "Resolve the training run and checkpoint before checking evaluator readiness."
            )
        elif suite is None or suite_errors:
            validation["plan_message"] = next(iter(suite_errors.values()), None)
        else:
            assert run is not None
            assert checkpoint is not None
            validation_key = "validate-" + canonical_sha256(
                {
                    "run_id": run["id"],
                    "checkpoint_id": checkpoint["id"],
                    "suite_id": suite["id"],
                    "tasks": resolved_tasks,
                    "seeds": list(seeds or [42]),
                    "resources": resources.model_dump(mode="json", by_alias=True) if resources else None,
                }
            )[:20]
            try:
                evaluator_spec, plan, _, _, evaluator_adapter, plan_source = (
                    self._resolve_evaluation_implementation(
                        run,
                        checkpoint,
                        suite,
                        environment=str(resolved_environment),
                        tasks=resolved_tasks,
                        seeds=list(seeds or [42]),
                        episodes_per_task=episodes_per_task,
                        parallelism=parallelism,
                        headless=headless,
                        execution_key=validation_key,
                        resources=resources,
                        manual_argv=list(argv or []),
                        manual_resume_argv=list(resume_argv or []),
                        verify_evaluator_runtime=True,
                        evaluator_runtime_gateway=gateway,
                    )
                )
                public_evaluator = {
                    key: evaluator_adapter.get(key)
                    for key in (
                        "id",
                        "version_id",
                        "slug",
                        "version",
                        "manifest_sha256",
                    )
                }
                blockers = list(plan.blockers)
                validation.update(
                    {
                        "resolved_resources": evaluator_spec.resources.model_dump(mode="json", by_alias=True) if evaluator_spec is not None else None,
                        "plan_valid": not blockers and bool(plan.argv),
                        "evaluator": public_evaluator,
                        "plan_blockers": blockers,
                        "plan_source": plan_source,
                        "plan_message": None if not blockers else "; ".join(blockers),
                    }
                )
            except Exception as error:
                message = sanitize(str(error))
                validation["plan_blockers"] = [message]
                validation["plan_message"] = message
        validation["valid"] = bool(
            validation["valid"]
            and validation["suite_valid"]
            and validation["plan_valid"]
        )
        return validation

    def create_evaluation(self, request: EvaluationRequest) -> dict[str, Any]:
        target, run, checkpoint = self._resolve_evaluation_target(
            request.run_id, request.checkpoint_path
        )
        if not target["run_valid"]:
            message = target["errors"]["run_id"]
            if str(request.run_id or "").strip():
                raise KeyError(message)
            raise ValueError(message)
        if not target["checkpoint_valid"]:
            raise ValueError(target["errors"]["checkpoint_path"])
        assert run is not None
        assert checkpoint is not None
        busy_reason = _evaluation_busy_reason(run)
        if busy_reason:
            raise ValueError(busy_reason)
        suite, canonical_environment, tasks, suite_errors = (
            self._resolve_evaluation_suite_selection(
                request.suite_id, request.environment, request.tasks
            )
        )
        if suite_errors:
            raise ValueError(next(iter(suite_errors.values())))
        assert suite is not None
        assert canonical_environment is not None

        checkpoint_id = checkpoint["id"]
        checkpoint_path = checkpoint["path"]
        stage = self.database.create_stage(
            run["id"],
            stage_type="EVALUATE",
            name=f"eval-{suite['name']}-{len(run['evaluations']) + 1}",
            status="BLOCKED",
            auto_resume=request.auto_resume,
            max_attempts=request.max_attempts,
            resolved_config={
                "schema_version": "skynet.evaluation-stage/v1",
                "blocker": "evaluation plan resolution is pending",
            },
        )
        (
            evaluator_spec,
            plan,
            context,
            training_adapter,
            evaluator_adapter,
            plan_source,
        ) = self._resolve_evaluation_implementation(
            run,
            checkpoint,
            suite,
            environment=canonical_environment,
            tasks=tasks,
            seeds=request.seeds,
            episodes_per_task=request.episodes_per_task,
            parallelism=request.parallelism,
            headless=request.headless,
            execution_key=stage["id"],
            resources=request.resources,
            manual_argv=request.argv,
            manual_resume_argv=request.resume_argv,
            verify_evaluator_runtime=True,
            evaluator_runtime_gateway=request.gateway,
            refresh_evaluator_runtime=True,
        )
        resolved_spec = evaluator_spec.model_dump(mode="json", by_alias=True)
        resolved_plan = plan.model_dump(mode="json")
        blockers = list(plan.blockers)
        status = "BLOCKED" if blockers or not plan.argv else "PENDING"
        blocker_message = "; ".join(blockers) if blockers else None
        suite_snapshot = copy.deepcopy(context["suite"])
        checkpoint_snapshot = copy.deepcopy(context["checkpoint"])
        stage = self.database.update_stage(
            stage["id"],
            status=status,
            resolved_config_json={
                "schema_version": "skynet.evaluation-stage/v1",
                "training_adapter": training_adapter,
                "evaluator_adapter": evaluator_adapter,
                "suite": suite_snapshot,
                "checkpoint": checkpoint_snapshot,
                "context": context,
                "context_sha256": canonical_sha256(context),
                "spec": resolved_spec,
                "spec_sha256": canonical_sha256(resolved_spec),
                "plan": resolved_plan,
                "plan_sha256": canonical_sha256(resolved_plan),
                "plan_source": plan_source,
                "argv": list(plan.argv),
                "resume_argv": list(plan.resume_argv),
                "checkpoint_path": checkpoint_path,
                "suite_id": request.suite_id,
                "environment": canonical_environment,
                "tasks": tasks,
                "task_selection": {
                    "mode": suite_snapshot.get("task_selection_mode", "subset"),
                    "reason": suite_snapshot.get("task_selection_reason", ""),
                },
                "parallelism": request.parallelism,
                "headless": request.headless,
                "blocker": blocker_message,
            },
        )
        evaluation = self.database.create_evaluation(
            run["id"],
            stage_id=stage["id"],
            checkpoint_id=checkpoint_id,
            evaluation_suite_id=suite["id"],
            evaluator_adapter=suite["evaluator_adapter"],
            evaluator_version=suite["evaluator_version"],
            suite_name=suite["name"],
            suite_version=suite["suite_version"],
            tasks=tasks,
            seeds=request.seeds,
            episodes_per_task=request.episodes_per_task,
            status=status,
            result_path=context["result_path"],
        )
        for task in tasks:
            for seed in request.seeds:
                for episode_index in range(request.episodes_per_task):
                    self.database.upsert_evaluation_episode(
                        evaluation["id"], task=task, seed=seed, episode_index=episode_index
                    )
        submission = None
        if status == "PENDING":
            submission = self._submit_stage(run["id"], stage["id"], request.gateway)
        evaluation = self.database.get_evaluation(evaluation["id"])
        assert evaluation is not None
        evaluation["submission"] = submission
        evaluation["blocker"] = blocker_message
        evaluation["evaluator_implementation"] = {
            key: evaluator_adapter.get(key)
            for key in ("id", "version_id", "slug", "version", "manifest_sha256")
        }
        return evaluation


service = PipelineService()
router = APIRouter(prefix="/api")


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, KeyError):
        return HTTPException(status_code=404, detail=str(error).strip("'"))
    if isinstance(error, sqlite3.IntegrityError):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, (ValueError, SlurmCompileError)):
        return HTTPException(status_code=422, detail=str(error))
    if isinstance(error, ClusterError):
        return HTTPException(status_code=503, detail=str(error))
    return HTTPException(status_code=500, detail=str(error))


@router.get("/data/resources")
def list_data_resources(
    provider: str | None = Query(default=None),
    namespace: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    include_archived: bool = Query(default=False),
) -> dict[str, Any]:
    return {
        "resources": service.database.list_data_resources(
            provider=provider,
            namespace=namespace,
            kind=kind,
            include_archived=include_archived,
        )
    }


@router.post("/data/resources", status_code=201)
def create_data_resource(request: DataResourceCreateRequest) -> dict[str, Any]:
    try:
        return {
            "resource": service.database.create_data_resource(
                **request.model_dump(mode="python")
            )
        }
    except Exception as error:
        raise _http_error(error) from error


@router.get("/data/resources/{resource_id}")
def get_data_resource(resource_id: str) -> dict[str, Any]:
    resource = service.database.get_data_resource(resource_id)
    if resource is None:
        raise HTTPException(status_code=404, detail="Data resource not found")
    return {"resource": resource}


@router.patch("/data/resources/{resource_id}")
def edit_data_resource(
    resource_id: str, request: DataResourceEditRequest
) -> dict[str, Any]:
    try:
        return {
            "resource": service.database.update_data_resource(
                resource_id,
                **request.model_dump(mode="python", exclude_unset=True),
            )
        }
    except Exception as error:
        raise _http_error(error) from error


@router.delete("/data/resources/{resource_id}")
def archive_data_resource(resource_id: str) -> dict[str, Any]:
    try:
        return {
            "resource": service.database.update_data_resource(resource_id, archived=True)
        }
    except Exception as error:
        raise _http_error(error) from error


@router.post("/data/resources/{resource_id}/versions", status_code=201)
def create_data_resource_version(
    resource_id: str, request: DataVersionCreateRequest
) -> dict[str, Any]:
    try:
        return {
            "version": service.database.create_data_resource_version(
                resource_id, **request.model_dump(mode="python")
            )
        }
    except Exception as error:
        raise _http_error(error) from error


@router.post("/data/resources/{resource_id}/imports", status_code=202)
def submit_data_import(
    resource_id: str, request: HuggingFaceImportRequest
) -> dict[str, Any]:
    try:
        return {"import": service.submit_data_import(resource_id, request)}
    except Exception as error:
        raise _http_error(error) from error


@router.get("/data/imports")
def list_data_imports() -> dict[str, Any]:
    try:
        return {"imports": service.reconcile_data_imports()}
    except Exception as error:
        raise _http_error(error) from error


@router.get("/data/imports/{import_id}/logs")
def get_data_import_log(
    import_id: str,
    stream: str = Query(default="stdout", pattern=r"^(stdout|stderr)$"),
    lines: int = Query(default=500, ge=1, le=5000),
) -> PlainTextResponse:
    record = service.database.get_data_import(import_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Data import not found")
    path = record.get(f"{stream}_path")
    if not path:
        raise HTTPException(status_code=404, detail=f"Data import {stream} path is not available")
    try:
        gateway, content = service.cluster.read_log(
            str(path), str(record.get("gateway") or "auto"), lines=lines
        )
    except Exception as error:
        raise _http_error(error) from error
    return PlainTextResponse(content, headers={"X-Skynet-Gateway": gateway})


@router.get("/data/versions/{version_id}")
def get_data_resource_version(version_id: str) -> dict[str, Any]:
    version = service.database.get_data_resource_version(version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Data resource version not found")
    return {"version": version}


@router.get("/data/derivations")
def list_data_derivations(
    limit: int = Query(default=1000, ge=1, le=10000),
) -> dict[str, Any]:
    return {"derivations": service.database.list_data_derivations(limit=limit)}


@router.post("/data/derivations", status_code=201)
def create_data_derivation(request: DataDerivationCreateRequest) -> dict[str, Any]:
    try:
        payload = request.model_dump(mode="python")
        payload["inputs"] = [item.model_dump(mode="python") for item in request.inputs]
        return {"derivation": service.database.create_data_derivation(**payload)}
    except Exception as error:
        raise _http_error(error) from error


@router.get("/data/derivations/{derivation_id}")
def get_data_derivation(derivation_id: str) -> dict[str, Any]:
    derivation = service.database.get_data_derivation(derivation_id)
    if derivation is None:
        raise HTTPException(status_code=404, detail="Data derivation not found")
    return {"derivation": derivation}


@router.get("/data/bundles")
def list_data_bundles(
    include_archived: bool = Query(default=False),
) -> dict[str, Any]:
    return {
        "bundles": service.database.list_data_bundles(
            include_archived=include_archived
        )
    }


@router.post("/data/bundles", status_code=201)
def create_data_bundle(request: DataBundleCreateRequest) -> dict[str, Any]:
    try:
        payload = request.model_dump(mode="python")
        payload["assignments"] = [
            item.model_dump(mode="python") for item in request.assignments
        ]
        return {"bundle": service.database.create_data_bundle(**payload)}
    except Exception as error:
        raise _http_error(error) from error


@router.get("/data/bundles/{bundle_id}")
def get_data_bundle(bundle_id: str) -> dict[str, Any]:
    bundle = service.database.get_data_bundle(bundle_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Data bundle not found")
    return {"bundle": bundle}


@router.get("/data/bundles/{bundle_id}/preview")
def preview_data_bundle(
    bundle_id: str,
    gateway: str = Query(default="auto", min_length=1, max_length=64),
) -> dict[str, Any]:
    bundle = service.database.get_data_bundle(bundle_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Data bundle not found")
    try:
        return {
            "preview": build_data_bundle_preview(bundle, service.cluster, gateway)
        }
    except Exception as error:
        raise _http_error(error) from error


@router.get("/data/bundles/{bundle_id}/preview/media")
def get_data_bundle_preview_media(
    bundle_id: str,
    request: Request,
    role: str = Query(min_length=1, max_length=128),
    position: int = Query(default=0, ge=0, le=10000),
    path: str = Query(min_length=1, max_length=2048),
    gateway: str = Query(default="auto", min_length=1, max_length=64),
) -> StreamingResponse:
    bundle = service.database.get_data_bundle(bundle_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Data bundle not found")
    try:
        media_source = resolve_data_bundle_preview_media(
            bundle,
            service.cluster,
            role=role,
            position=position,
            relative_path=path,
            gateway=gateway,
        )
    except Exception as error:
        raise _http_error(error) from error

    if media_source["kind"] == "provider":
        return RedirectResponse(
            url=media_source["url"],
            status_code=307,
            headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
            },
        )

    host = media_source["host"]
    media_path = media_source["path"]
    media_type = media_source["media_type"]
    try:
        _, size = service.cluster.file_size(media_path, host)
    except Exception as error:
        raise _http_error(error) from error
    if size < 1:
        raise HTTPException(status_code=404, detail="Preview media file is empty")

    start = 0
    end = size - 1
    partial = False
    range_header = request.headers.get("range")
    if range_header:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match or (not match.group(1) and not match.group(2)):
            raise HTTPException(
                status_code=416,
                detail="Unsupported media byte range",
                headers={"Content-Range": f"bytes */{size}"},
            )
        if match.group(1):
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else size - 1
            if start >= size or end < start:
                raise HTTPException(
                    status_code=416,
                    detail="Media byte range is outside the file",
                    headers={"Content-Range": f"bytes */{size}"},
                )
            end = min(end, size - 1)
        else:
            suffix_size = int(match.group(2))
            if suffix_size < 1:
                raise HTTPException(
                    status_code=416,
                    detail="Media suffix range must be positive",
                    headers={"Content-Range": f"bytes */{size}"},
                )
            start = max(0, size - suffix_size)
        partial = True

    length = end - start + 1
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Cache-Control": "private, max-age=3600",
        "Content-Disposition": "inline",
        "X-Content-Type-Options": "nosniff",
    }
    if partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(
        service.cluster.stream_file_range(
            media_path, host, start=start, end=end
        ),
        status_code=206 if partial else 200,
        media_type=media_type,
        headers=headers,
    )


@router.delete("/data/bundles/{bundle_id}")
def archive_data_bundle(bundle_id: str) -> dict[str, Any]:
    try:
        return {"bundle": service.database.archive_data_bundle(bundle_id)}
    except Exception as error:
        raise _http_error(error) from error


@router.get("/adapters")
def adapters(include_archived: bool = Query(default=False)) -> dict[str, Any]:
    records = []
    for row in service.database.list_adapter_registry(include_archived=include_archived):
        version = row.get("latest_version") or {}
        manifest = version.get("manifest") or {}
        manifest_error = None
        try:
            normalized_manifest = AdapterManifest.model_validate(manifest).model_dump(mode="json")
        except (TypeError, ValueError) as error:
            normalized_manifest = {}
            manifest_error = str(error)
        capabilities = normalized_manifest.get("capabilities") or {}
        records.append({
            **row,
            "slug": normalized_manifest.get("slug") or row.get("seed_key") or row["id"],
            "label": normalized_manifest.get("display_name") or row["name"],
            "version": version.get("version_number"),
            "status": "archived" if row.get("archived_at") else "ready" if not manifest_error else "invalid",
            "manifest_valid": manifest_error is None,
            "manifest_error": manifest_error,
            "runtime": sorted((normalized_manifest.get("runtime") or {}).get("allowed_backends") or []),
            "repository": version.get("repository_url") or row.get("repository_url"),
            "capabilities": capabilities,
        })
    return {"adapters": records}


@router.post("/adapters")
def create_adapter(request: AdapterCreateRequest) -> dict[str, Any]:
    try:
        manifest = AdapterManifest.model_validate(request.manifest)
        record = service.database.create_adapter(
            name=request.name,
            manifest=manifest.model_dump(mode="json"),
            description=request.description,
            repository_url=request.repository_url or manifest.default_repository,
            created_by=request.created_by,
            change_note=request.change_note,
        )
        return {"adapter": record}
    except Exception as error:
        raise _http_error(error) from error


@router.post("/adapters/validate")
def validate_unsaved_adapter(request: UnsavedAdapterValidationRequest) -> dict[str, Any]:
    try:
        return {"report": service.validate_unsaved_adapter_manifest(request)}
    except Exception as error:
        raise _http_error(error) from error


@router.get("/adapters/{adapter_id}")
def adapter_detail(
    adapter_id: str,
    version_number: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    record = service.database.get_adapter(
        adapter_id, version_number=version_number, include_versions=True
    )
    if not record:
        raise HTTPException(status_code=404, detail="Adapter not found")
    return {"adapter": record}


@router.put("/adapters/{adapter_id}")
@router.patch("/adapters/{adapter_id}")
def edit_adapter(adapter_id: str, request: AdapterEditRequest) -> dict[str, Any]:
    try:
        manifest = AdapterManifest.model_validate(request.manifest)
        record = service.database.edit_adapter(
            adapter_id,
            manifest=manifest.model_dump(mode="json"),
            name=request.name,
            description=request.description,
            repository_url=request.repository_url or manifest.default_repository,
            created_by=request.created_by,
            expected_latest_version=request.expected_latest_version,
            change_note=request.change_note,
        )
        return {"adapter": record}
    except Exception as error:
        raise _http_error(error) from error


@router.post("/adapters/{adapter_id}/clone")
def clone_adapter(adapter_id: str, request: AdapterCloneRequest) -> dict[str, Any]:
    try:
        record = service.database.clone_adapter(
            adapter_id,
            name=request.name,
            version_number=request.version_number,
            description=request.description,
            created_by=request.created_by,
            change_note=request.change_note,
        )
        return {"adapter": record}
    except Exception as error:
        raise _http_error(error) from error


@router.delete("/adapters/{adapter_id}")
@router.post("/adapters/{adapter_id}/archive")
def archive_adapter(adapter_id: str) -> dict[str, Any]:
    try:
        return {"adapter": service.database.archive_adapter(adapter_id)}
    except Exception as error:
        raise _http_error(error) from error


@router.post("/adapters/{adapter_id}/restore")
def restore_adapter(adapter_id: str) -> dict[str, Any]:
    try:
        return {"adapter": service.database.restore_adapter(adapter_id)}
    except Exception as error:
        raise _http_error(error) from error


@router.post("/adapters/{adapter_id}/validate")
def validate_adapter(adapter_id: str, request: AdapterValidationRequest) -> dict[str, Any]:
    try:
        return service.validate_adapter_registry_entry(adapter_id, request)
    except Exception as error:
        raise _http_error(error) from error


@router.get("/adapters/{adapter_id}/validations")
def adapter_validations(
    adapter_id: str,
    version_number: int | None = Query(default=None, ge=1),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    try:
        return {
            "validations": service.database.list_adapter_validations(
                adapter_id, version_number=version_number, limit=limit
            )
        }
    except Exception as error:
        raise _http_error(error) from error


@router.get("/source/branches")
def source_branches(
    repo_url: str = Query(min_length=1, max_length=2048),
    gateway: str = Query(default="auto", min_length=1, max_length=64),
    refresh: bool = Query(default=False),
) -> dict[str, Any]:
    try:
        repository = service.source_discovery.repository_url(repo_url)
        if refresh:
            service.source_discovery.clear_cache()
        entry = None if refresh else service.source_metadata.get("branches", repository, {})
        cache_hit = entry is not None
        if entry is None:
            payload = service.source_discovery.branches(repository, gateway)
            entry = service.source_metadata.put("branches", repository, {}, payload)
        result = service.source_metadata.response(entry, cache_hit=cache_hit)
        result["selection"] = service.source_metadata.get_selection(repository)
        return result
    except Exception as error:
        raise _http_error(error) from error


@router.get("/source/commits")
def source_commits(
    repo_url: str = Query(min_length=1, max_length=2048),
    branch: str = Query(min_length=1, max_length=255),
    limit: int = Query(default=50, ge=1, le=100),
    gateway: str = Query(default="auto", min_length=1, max_length=64),
    refresh: bool = Query(default=False),
) -> dict[str, Any]:
    try:
        repository = service.source_discovery.repository_url(repo_url)
        branch_name = service.source_discovery.branch_name(branch)
        parameters = {"branch": branch_name, "limit": limit}
        if refresh:
            service.source_discovery.clear_cache()
        entry = None if refresh else service.source_metadata.get(
            "commits", repository, parameters
        )
        cache_hit = entry is not None
        if entry is None:
            payload = service.source_discovery.commits(
                repository, branch_name, limit, gateway
            )
            entry = service.source_metadata.put(
                "commits", repository, parameters, payload
            )
        return service.source_metadata.response(entry, cache_hit=cache_hit)
    except Exception as error:
        raise _http_error(error) from error


@router.put("/source/selection")
def save_source_selection(
    repo_url: str = Query(min_length=1, max_length=2048),
    branch: str = Query(min_length=1, max_length=255),
    commit: str = Query(default="", max_length=40),
) -> dict[str, Any]:
    try:
        repository = service.source_discovery.repository_url(repo_url)
        branch_name = service.source_discovery.branch_name(branch)
        return {
            "repository": repository,
            "selection": service.source_metadata.save_selection(
                repository, branch_name, commit
            ),
        }
    except Exception as error:
        raise _http_error(error) from error


@router.get("/source/inspect")
def inspect_source(
    repo_url: str = Query(min_length=1, max_length=2048),
    revision: str = Query(min_length=1, max_length=128),
    project_subdirectory: str = Query(default=".", min_length=1, max_length=512),
    gateway: str = Query(default="auto", min_length=1, max_length=64),
    adapter_id: str | None = Query(default=None, min_length=1, max_length=128),
    adapter_version_id: str | None = Query(default=None, min_length=1, max_length=128),
    refresh: bool = Query(default=False),
) -> dict[str, Any]:
    try:
        repository = service.source_discovery.repository_url(repo_url)
        requested_revision = revision.strip()
        subdirectory = service.source_discovery.project_subdirectory(
            project_subdirectory
        )
        input_fields: list[dict[str, Any]] = []
        resolved_adapter_version_id = ""
        manifest_hash = ""
        if adapter_version_id and not adapter_id:
            raise ValueError("adapter_version_id requires adapter_id")
        if adapter_id:
            record = service.database.get_adapter(adapter_id, include_versions=True)
            if not record:
                raise ValueError(f"adapter was not found: {adapter_id}")
            if adapter_version_id:
                version = next(
                    (
                        item for item in record.get("versions", [])
                        if str(item.get("id")) == adapter_version_id
                    ),
                    None,
                )
                if version is None:
                    raise ValueError(
                        f"adapter version was not found: {adapter_id}@{adapter_version_id}"
                    )
            else:
                version = service._selected_version(record)
            try:
                manifest = AdapterManifest.model_validate(version["manifest"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "selected adapter version is not a valid canonical manifest; "
                    "repair or archive it in the Adapter Registry"
                ) from error
            input_fields = [
                field.model_dump(mode="json") for field in manifest.train.input_fields
            ]
            resolved_adapter_version_id = str(version["id"])
            manifest_hash = str(
                version.get("manifest_sha256") or adapter_manifest_sha256(manifest)
            )

        discovery_fields = [
            {
                "path": field["path"],
                "choice_source": field["choice_source"],
            }
            for field in input_fields
            if isinstance(field.get("choice_source"), dict)
        ]
        discovery_sha256 = hashlib.sha256(
            canonical_json(discovery_fields).encode("utf-8")
        ).hexdigest()
        normalized_revision = (
            requested_revision.lower()
            if FULL_COMMIT_RE.fullmatch(requested_revision)
            else requested_revision
        )
        parameters = {
            "revision": normalized_revision,
            "project_subdirectory": subdirectory,
            "input_discovery_sha256": discovery_sha256,
        }
        if refresh:
            service.source_discovery.clear_cache()
        entry = None if refresh else service.source_metadata.get(
            "inspection", repository, parameters
        )
        if entry is None and not refresh:
            legacy_parameters = {
                "revision": normalized_revision,
                "project_subdirectory": subdirectory,
                "adapter_id": adapter_id or "",
                "adapter_version_id": resolved_adapter_version_id,
                "adapter_manifest_sha256": manifest_hash,
            }
            legacy_entry = service.source_metadata.get(
                "inspection", repository, legacy_parameters
            )
            if legacy_entry is not None:
                entry = service.source_metadata.put(
                    "inspection", repository, parameters, legacy_entry["payload"]
                )
        if entry is not None:
            cached_payload = entry.get("payload")
            cached_commit = (
                str(cached_payload.get("commit") or "").lower()
                if isinstance(cached_payload, Mapping)
                else ""
            )
            if FULL_COMMIT_RE.fullmatch(cached_commit):
                pinned_parameters = {**parameters, "revision": cached_commit}
                if (
                    pinned_parameters != parameters
                    and service.source_metadata.get(
                        "inspection", repository, pinned_parameters
                    )
                    is None
                ):
                    service.source_metadata.put(
                        "inspection", repository, pinned_parameters, cached_payload
                    )
            return service.source_metadata.response(entry, cache_hit=True)

        commit = service.source_discovery.resolve_revision(
            repository, requested_revision, gateway
        )
        result = service.source_discovery.inspect(
            repository, commit, gateway, subdirectory
        )
        result["input_options"] = {}
        if input_fields:
            result["input_options"] = service.source_discovery.input_options(
                repository,
                commit,
                input_fields,
                gateway,
                subdirectory,
            )
        for candidate in result.get("candidates", []):
            candidate.setdefault("type", candidate.get("backend"))
            candidate.setdefault("confidence", candidate.get("strength"))
        result["runtime_profiles"] = service.runtime_profiles(gateway, verify=True)
        entry = service.source_metadata.put(
            "inspection", repository, parameters, result
        )
        pinned_parameters = {**parameters, "revision": commit.lower()}
        if pinned_parameters != parameters:
            service.source_metadata.put(
                "inspection", repository, pinned_parameters, result
            )
        return service.source_metadata.response(entry, cache_hit=False)
    except Exception as error:
        raise _http_error(error) from error


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    return {
        "adapter_manifest_schema": AdapterManifest.model_json_schema(),
        "experiment_schema": ExperimentSpec.model_json_schema(by_alias=True),
        "runtime_backends": ["uv", "conda", "apptainer", "existing"],
        "runtime_profiles": CLUSTER.public_runtime_profiles(),
        "cluster": CLUSTER.public_dict(),
        "safety": {"multi_node": False, "arbitrary_adapter_code": False, "readme_execution": False},
    }


@router.get("/runtime-profiles")
def runtime_profiles(
    gateway: str = Query(default="auto", min_length=1, max_length=64),
    verify: bool = Query(default=False),
) -> dict[str, Any]:
    try:
        return {"runtime_profiles": service.runtime_profiles(gateway, verify=verify)}
    except Exception as error:
        raise _http_error(error) from error


@router.get("/evaluation-suites")
def evaluation_suites(
    run_id: str | None = Query(default=None, min_length=1, max_length=128),
) -> dict[str, Any]:
    evaluator_manifest: AdapterManifest | None = None
    evaluator_identity: dict[str, Any] | None = None
    spec_document: dict[str, Any] | None = None
    if run_id is not None:
        run = service.database.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Training run was not found.")
        try:
            training_spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
            evaluator_document = copy.deepcopy(
                training_spec.model_dump(mode="json", by_alias=True)
            )
            evaluator_source = evaluator_document.get("source")
            if not isinstance(evaluator_source, dict):
                raise ValueError("the run has no canonical source configuration")
            for key in (
                "adapter_id",
                "adapter_version_id",
                "adapter_version",
                "adapter_manifest",
                "adapter_manifest_sha256",
            ):
                evaluator_source.pop(key, None)
            evaluator_source["adapter"] = training_spec.source.adapter
            (
                evaluator_document["source"],
                evaluator_manifest,
                _,
            ) = service._snapshot_adapter(
                evaluator_source, training_spec.source.adapter
            )
            evaluator_spec = ExperimentSpec.model_validate(evaluator_document)
            spec_document = evaluator_spec.model_dump(mode="python", by_alias=True)
            evaluator_identity = service._adapter_identity(evaluator_spec)
            evaluator_identity.pop("manifest", None)
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Training run cannot resolve its current evaluation adapter: "
                    f"{sanitize(str(error))}"
                ),
            ) from error

    missing = object()

    def dotted_value(document: Mapping[str, Any], path: str) -> Any:
        current: Any = document
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return missing
            current = current[part]
        return current

    def declaration_enabled(entry: Any) -> bool:
        conditions = getattr(entry, "enabled_when", {}) or {}
        if not isinstance(conditions, Mapping) or spec_document is None:
            return not conditions
        for path, choices in conditions.items():
            actual = dotted_value(spec_document, str(path))
            if actual is missing:
                return False
            allowed = choices if isinstance(choices, (list, tuple, set)) else [choices]
            if not any(actual == choice for choice in allowed):
                return False
        return True

    def runnable_for_current_evaluator(row: Mapping[str, Any]) -> bool:
        if evaluator_manifest is None:
            return True
        return any(
            entry.environment == row["evaluator_adapter"]
            and row["name"] in entry.suites
            and entry.command is not None
            and declaration_enabled(entry)
            for entry in evaluator_manifest.evaluations
        )

    suites = []
    for row in service.database.list_evaluation_suites():
        if not runnable_for_current_evaluator(row):
            continue
        config = row["config_json"]
        suites.append({
            **row,
            "slug": row["id"],
            "label": row["description"] or row["name"],
            "evaluator": row["evaluator_adapter"],
            "version": row["suite_version"],
            "tasks": config.get("tasks", []),
            "task_options": config.get("task_options", []),
            "task_source": config.get("task_source"),
            "task_catalog_complete": bool(config.get("task_catalog_complete")),
            "task_selection_mode": config.get("task_selection_mode", "subset"),
            "task_selection_reason": config.get("task_selection_reason", ""),
            "task_catalog_provenance": config.get("task_catalog_provenance"),
            "task_catalog_sha256": config.get("task_catalog_sha256"),
            "catalog_sha256": config.get("catalog_sha256"),
            "current": bool(row["enabled"]),
            **(
                {"evaluator_implementation": copy.deepcopy(evaluator_identity)}
                if evaluator_identity is not None
                else {}
            ),
        })
    return {"suites": suites, "evaluation_suites": suites}


@router.post("/experiments/preview")
def preview_experiment(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return service.preview(payload)
    except Exception as error:
        raise _http_error(error) from error


@router.post("/experiments")
def create_experiment(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        experiment = service.create_experiment(payload)
        return {"experiment": experiment, "id": experiment["id"]}
    except Exception as error:
        raise _http_error(error) from error


@router.get("/experiments")
def list_experiments() -> dict[str, Any]:
    records = service.database.list_experiments(limit=1000)
    for record in records:
        runs = service.database.list_runs(
            experiment_revision_id=record["latest_revision_id"], limit=10000
        )
        record["run_count"] = len(runs)
        record["variant_count"] = len({run["variant_id"] for run in runs})
        record["adapter"] = runs[0]["adapter_name"] if runs else None
        record["revision_number"] = record["latest_revision_number"]
        record["locked"] = bool(record.get("latest_revision_submitted_at"))
        record["lifecycle"] = "SUBMITTED" if record["locked"] else "DRAFT"
    return {"experiments": records}


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict[str, Any]:
    try:
        return {"experiment": service.experiment_detail(experiment_id)}
    except Exception as error:
        raise _http_error(error) from error


@router.post("/experiments/{experiment_id}/submit")
def submit_experiment(experiment_id: str, request: GatewayRequest) -> dict[str, Any]:
    try:
        return service.submit_experiment(experiment_id, request.gateway)
    except Exception as error:
        raise _http_error(error) from error


@router.post("/experiments/{experiment_id}/revisions")
def create_experiment_revision(
    experiment_id: str, request: ExperimentRevisionRequest
) -> dict[str, Any]:
    try:
        experiment = service.create_experiment_revision(
            experiment_id,
            request.spec,
            submit=request.submit,
            gateway=request.gateway,
        )
        return {
            "experiment": experiment,
            "revision": experiment["latest_revision"],
            "id": experiment["latest_revision"]["id"],
        }
    except Exception as error:
        raise _http_error(error) from error


@router.get("/runs")
def list_runs(
    experiment_id: str | None = Query(default=None),
    experiment_revision_id: str | None = Query(default=None),
    variant_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
) -> dict[str, Any]:
    runs = service.database.list_runs(
        experiment_id=experiment_id,
        experiment_revision_id=experiment_revision_id,
        variant_id=variant_id,
        status=status,
        limit=1000,
    )
    connections = service.tracking_connections()["connections"]
    for run in runs:
        service._ingest_training_progress(run)
        run["tracking_actions"] = service.run_tracking_actions(run, connections)
    _attach_run_progress_summaries(service.database, runs)
    return {"runs": runs}


_COMMON_HYPERPARAMETER_FIELDS = {
    "learning_rate": ("train.learning_rate", "learning_rate"),
    "batch_semantics": ("train.batch.declared_semantics", "batch_semantics"),
    "batch_size": ("train.batch.value", "batch_size"),
    "gradient_accumulation": (
        "train.batch.gradient_accumulation_steps",
        "gradient_accumulation_steps",
    ),
    "num_workers": ("train.num_workers_per_rank", "num_workers_per_rank"),
    "max_steps": ("train.max_steps", "max_steps"),
    "max_epochs": ("train.max_epochs", "max_epochs"),
    "precision": ("train.precision", "precision"),
}

_HISTORICAL_OPENPI_METADATA_RULE = {
    "schema_version": "skynet.repository-choice-enrichment/v1",
    "id": "openpi-train-config-common-hyperparameters-v1",
    "match": {
        "path": "native.config.config_name",
        "kind": "python_static_registry",
        "entrypoint": "src/openpi/training/config.py",
        "supporting_files": [
            "src/openpi/training/misc/roboarena_config.py",
            "src/openpi/training/misc/polaris_config.py",
        ],
        "registry": "_CONFIGS",
        "constructor": "TrainConfig",
        "value_keyword": "name",
    },
    "metadata_fields": [
        {
            "source_path": "lr_schedule.peak_lr",
            "canonical_path": "train.learning_rate",
            "value_map": {},
        },
        {
            "source_path": "batch_size",
            "canonical_path": "train.batch.value",
            "value_map": {},
        },
        {
            "source_path": "num_workers",
            "canonical_path": "train.num_workers_per_rank",
            "value_map": {},
        },
        {
            "source_path": "num_train_steps",
            "canonical_path": "train.max_steps",
            "value_map": {},
        },
        {
            "source_path": "pytorch_training_precision",
            "canonical_path": "train.precision",
            "value_map": {
                "bfloat16": "bf16",
                "float16": "fp16",
                "float32": "fp32",
            },
        },
    ],
    "fixed_values": {
        "train.batch.declared_semantics": "global_before_accumulation",
        "train.batch.gradient_accumulation_steps": 1,
    },
}


def _choice_source_matches_historical_rule(
    path: str, choice_source: Mapping[str, Any], rule: Mapping[str, Any]
) -> bool:
    match = rule.get("match")
    if not isinstance(match, Mapping) or path != match.get("path"):
        return False
    return all(
        choice_source.get(key) == expected
        for key, expected in match.items()
        if key != "path"
    )


def _repository_enrichment_fields(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = manifest.get("train")
    raw_fields = train.get("input_fields") if isinstance(train, Mapping) else []
    fields: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    for raw_field in raw_fields or []:
        if not isinstance(raw_field, Mapping):
            continue
        path = str(raw_field.get("path") or "")
        raw_source = raw_field.get("choice_source")
        if not isinstance(raw_source, Mapping):
            continue
        choice_source = copy.deepcopy(dict(raw_source))
        metadata_fields = choice_source.get("metadata_fields")
        origin = "adapter_manifest"
        rule_document: dict[str, Any]
        if not metadata_fields and _choice_source_matches_historical_rule(
            path, choice_source, _HISTORICAL_OPENPI_METADATA_RULE
        ):
            metadata_fields = copy.deepcopy(
                _HISTORICAL_OPENPI_METADATA_RULE["metadata_fields"]
            )
            choice_source["metadata_fields"] = metadata_fields
            origin = "versioned_historical_rule"
            rule_document = copy.deepcopy(_HISTORICAL_OPENPI_METADATA_RULE)
        elif metadata_fields:
            rule_document = {
                "schema_version": "skynet.repository-choice-enrichment/v1",
                "id": f"manifest:{path}",
                "path": path,
                "choice_source": copy.deepcopy(choice_source),
            }
        else:
            continue
        fields.append({"path": path, "choice_source": choice_source})
        rules.append(
            {
                "path": path,
                "origin": origin,
                "rule": rule_document,
                "rule_sha256": canonical_sha256(rule_document),
            }
        )
    return fields, rules


def _attempt_pinned_enrichment_context(
    run: Mapping[str, Any], attempt: Mapping[str, Any]
) -> dict[str, Any]:
    snapshot = attempt["execution_snapshot_json"]
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    resolved_spec = snapshot.get("resolved_spec") or run.get("resolved_spec_json")
    if not isinstance(resolved_spec, Mapping):
        raise ValueError("attempt has no pinned resolved specification")
    source = resolved_spec.get("source")
    source = source if isinstance(source, Mapping) else {}
    adapter = snapshot.get("adapter")
    adapter = adapter if isinstance(adapter, Mapping) else {}
    manifest = adapter.get("manifest") or source.get("adapter_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("attempt has no pinned adapter manifest")
    if manifest.get("slug") != source.get("adapter"):
        raise ValueError("attempt pinned adapter manifest does not match its source adapter")
    repository = str(source.get("repository") or "")
    commit = str(source.get("revision") or "").lower()
    if not repository:
        raise ValueError("attempt has no pinned repository")
    if not FULL_COMMIT_RE.fullmatch(commit):
        raise ValueError("historical enrichment requires a pinned 40-character commit")
    project_subdirectory = str(source.get("project_subdirectory") or ".")
    fields, rules = _repository_enrichment_fields(manifest)
    if not fields:
        raise ValueError("pinned adapter has no safe repository enrichment rule")
    discovery_fields = [
        {"path": field["path"], "choice_source": field["choice_source"]}
        for field in fields
    ]
    discovery_sha256 = hashlib.sha256(
        canonical_json(discovery_fields).encode("utf-8")
    ).hexdigest()
    resolved_spec_sha256 = canonical_sha256(resolved_spec)
    recorded_spec_sha256 = snapshot.get("resolved_spec_sha256")
    if recorded_spec_sha256 and recorded_spec_sha256 != resolved_spec_sha256:
        raise ValueError("attempt pinned resolved specification hash does not match")
    adapter_manifest_sha256 = canonical_sha256(manifest)
    recorded_manifest_sha256 = adapter.get("manifest_sha256") or source.get(
        "adapter_manifest_sha256"
    )
    if recorded_manifest_sha256 and recorded_manifest_sha256 != adapter_manifest_sha256:
        raise ValueError("attempt pinned adapter manifest hash does not match")
    execution_snapshot_sha256 = attempt.get("execution_snapshot_sha256")
    if (
        execution_snapshot_sha256
        and snapshot
        and execution_snapshot_sha256 != content_sha256(snapshot)
    ):
        raise ValueError("attempt execution snapshot hash does not match")
    return {
        "spec": copy.deepcopy(dict(resolved_spec)),
        "manifest": copy.deepcopy(dict(manifest)),
        "repository": repository,
        "commit": commit,
        "project_subdirectory": project_subdirectory,
        "fields": fields,
        "rules": rules,
        "parameters": {
            "revision": commit,
            "project_subdirectory": project_subdirectory,
            "input_discovery_sha256": discovery_sha256,
        },
        "resolved_spec_sha256": resolved_spec_sha256,
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "execution_snapshot_sha256": execution_snapshot_sha256,
    }


def _validated_cached_repository_options(
    context: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    payload = entry.get("payload")
    input_options = payload.get("input_options") if isinstance(payload, Mapping) else None
    if not isinstance(input_options, Mapping):
        raise ValueError("cached inspection has no repository input metadata")
    validated: dict[str, Any] = {}
    for field in context["fields"]:
        path = str(field["path"])
        option = input_options.get(path)
        if not isinstance(option, Mapping) or not option.get("complete"):
            raise ValueError(f"cached repository metadata is incomplete for {path}")
        source = option.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"cached repository metadata has no evidence for {path}")
        expected_rule_sha256 = canonical_sha256(field["choice_source"])
        if source.get("commit") != context["commit"]:
            raise ValueError(f"cached repository metadata commit does not match for {path}")
        if source.get("rule_sha256") != expected_rule_sha256:
            raise ValueError(f"cached repository metadata rule does not match for {path}")
        files = source.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"cached repository metadata has no pinned files for {path}")
        pinned_files = {
            str(item.get("path")): str(item.get("sha256"))
            for item in files
            if isinstance(item, Mapping)
            and isinstance(item.get("path"), str)
            and isinstance(item.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256")))
        }
        if len(pinned_files) != len(files):
            raise ValueError(f"cached repository metadata file evidence is invalid for {path}")
        required_files = {
            str(field["choice_source"].get("entrypoint") or ""),
            *{
                str(item)
                for item in field["choice_source"].get("supporting_files") or []
            },
        }
        if not required_files <= set(pinned_files):
            raise ValueError(f"cached repository metadata file evidence is incomplete for {path}")
        present, choice = _mapping_path(context["spec"], path)
        metadata = option.get("metadata")
        if (
            not present
            or not isinstance(metadata, Mapping)
            or not isinstance(metadata.get(str(choice)), Mapping)
        ):
            raise ValueError(f"cached repository metadata lacks selected choice for {path}")
        selected = metadata[str(choice)]
        values = selected.get("values")
        evidence = selected.get("evidence")
        allowed_fields = {
            str(item.get("canonical_path")): (
                str(item.get("source_path")),
                str(item.get("source_file")) if item.get("source_file") else None,
            )
            for item in field["choice_source"].get("metadata_fields") or []
            if isinstance(item, Mapping)
        }
        if not isinstance(values, Mapping) or not isinstance(evidence, Mapping):
            raise ValueError(f"cached selected metadata is malformed for {path}")
        if any(canonical_path not in allowed_fields for canonical_path in values):
            raise ValueError(f"cached selected metadata contains undeclared fields for {path}")
        for canonical_path in values:
            field_evidence = evidence.get(canonical_path)
            declared_source_path, declared_source_file = allowed_fields[canonical_path]
            if (
                not isinstance(field_evidence, Mapping)
                or field_evidence.get("source_path") != declared_source_path
                or field_evidence.get("file") not in pinned_files
                or (
                    declared_source_file is not None
                    and field_evidence.get("file") != declared_source_file
                )
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(field_evidence.get("expression_sha256") or ""),
                )
            ):
                raise ValueError(
                    f"cached selected metadata evidence is invalid for {canonical_path}"
                )
        validated[path] = copy.deepcopy(dict(option))
    return validated


def _pinned_repository_metadata_cache_entry(
    pipeline_service: PipelineService, context: Mapping[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    for cache_kind in ("common_hyperparameters", "inspection"):
        entry = pipeline_service.source_metadata.get(
            cache_kind, context["repository"], context["parameters"]
        )
        if entry is None:
            continue
        try:
            options = _validated_cached_repository_options(context, entry)
        except ValueError:
            continue
        return cache_kind, entry, options
    return None


def _stored_attempt_common_hyperparameter_contract(
    snapshot: Mapping[str, Any]
) -> dict[str, Any] | None:
    stored = snapshot.get("common_hyperparameters")
    if (
        isinstance(stored, Mapping)
        and stored.get("schema_version") == "skynet.common-hyperparameters/v1"
        and isinstance(stored.get("values"), Mapping)
        and isinstance(stored.get("provenance"), Mapping)
    ):
        return copy.deepcopy(dict(stored))
    return None


def _validated_attempt_enrichment_receipt(
    run: Mapping[str, Any],
    attempt: Mapping[str, Any],
    event: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(event, Mapping):
        return None
    details = event.get("details_json")
    if not isinstance(details, Mapping):
        return None
    receipt = copy.deepcopy(dict(details))
    receipt_sha256 = receipt.pop("receipt_sha256", None)
    if receipt_sha256 != canonical_sha256(receipt):
        return None
    try:
        context = _attempt_pinned_enrichment_context(run, attempt)
    except ValueError:
        return None
    pinned = receipt.get("pinned")
    if not isinstance(pinned, Mapping):
        return None
    expected = {
        "attempt_id": attempt.get("id"),
        "execution_snapshot_sha256": context["execution_snapshot_sha256"],
        "resolved_spec_sha256": context["resolved_spec_sha256"],
        "adapter_manifest_sha256": context["adapter_manifest_sha256"],
        "repository": context["repository"],
        "commit": context["commit"],
        "project_subdirectory": context["project_subdirectory"],
        "input_discovery_sha256": context["parameters"]["input_discovery_sha256"],
    }
    if any(pinned.get(key) != value for key, value in expected.items()):
        return None
    contract = receipt.get("common_hyperparameters")
    if (
        not isinstance(contract, Mapping)
        or contract.get("schema_version") != "skynet.common-hyperparameters/v1"
        or not isinstance(contract.get("values"), Mapping)
        or not isinstance(contract.get("provenance"), Mapping)
    ):
        return None
    return copy.deepcopy(dict(contract))


def _mapping_path(
    document: Mapping[str, Any] | None, path: str
) -> tuple[bool, Any]:
    current: Any = document
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _path_is_explicit(spec: Mapping[str, Any], path: str) -> bool:
    intent = spec.get("intent")
    requested_paths = (
        intent.get("explicit_parameters") if isinstance(intent, Mapping) else []
    )
    return any(
        isinstance(requested, str)
        and (
            path == requested
            or path.startswith(f"{requested}.")
            or requested.startswith(f"{path}.")
        )
        for requested in requested_paths or []
    )


def _canonical_path_supported(path: str, supported: set[str]) -> bool:
    return any(
        path == candidate
        or path.startswith(f"{candidate}.")
        or candidate.startswith(f"{path}.")
        for candidate in supported
    )


def _selected_repository_input_metadata(
    spec: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for input_path, raw_option in options.items():
        if not isinstance(raw_option, Mapping):
            continue
        present, choice = _mapping_path(spec, str(input_path))
        metadata = raw_option.get("metadata")
        record = metadata.get(str(choice)) if isinstance(metadata, Mapping) else None
        if not present or not isinstance(record, Mapping):
            continue
        selected[str(input_path)] = {
            "value": copy.deepcopy(choice),
            "values": copy.deepcopy(dict(record.get("values") or {})),
            "evidence": copy.deepcopy(dict(record.get("evidence") or {})),
            "source": copy.deepcopy(dict(raw_option.get("source") or {})),
        }
    return selected


def _with_historical_rule_values(
    selected: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    enriched = copy.deepcopy(dict(selected))
    for rule in context.get("rules") or []:
        if not isinstance(rule, Mapping) or rule.get("origin") != "versioned_historical_rule":
            continue
        document = rule.get("rule")
        fixed_values = document.get("fixed_values") if isinstance(document, Mapping) else None
        if not isinstance(fixed_values, Mapping) or not fixed_values:
            continue
        rule_sha256 = str(rule.get("rule_sha256") or canonical_sha256(document))
        key = f"historical_rule:{rule.get('path')}"
        enriched[key] = {
            "value": document.get("id"),
            "provenance_source": "historical_enrichment_rule",
            "values": copy.deepcopy(dict(fixed_values)),
            "evidence": {
                str(path): {
                    "origin": "versioned_historical_rule",
                    "rule_sha256": rule_sha256,
                }
                for path in fixed_values
            },
            "source": {
                "kind": "versioned_historical_rule",
                "commit": context["commit"],
                "rule_sha256": rule_sha256,
            },
        }
    return enriched


def _resolve_effective_common_hyperparameters(
    spec: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    repository_inputs: Mapping[str, Any] | None = None,
    *,
    resolved_spec_sha256: str | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    spec = spec if isinstance(spec, Mapping) else {}
    manifest = manifest if isinstance(manifest, Mapping) else {}
    defaults = manifest.get("defaults")
    defaults = defaults if isinstance(defaults, Mapping) else {}
    hyper_defaults = defaults.get("hyperparameters")
    hyper_defaults = hyper_defaults if isinstance(hyper_defaults, Mapping) else {}
    train_manifest = manifest.get("train")
    train_manifest = train_manifest if isinstance(train_manifest, Mapping) else {}
    supported = {
        str(path) for path in train_manifest.get("supported_canonical_fields") or []
    }
    source = spec.get("source")
    source = source if isinstance(source, Mapping) else {}
    repository_values: dict[str, tuple[Any, str, dict[str, Any]]] = {}
    for input_path, selection in (repository_inputs or {}).items():
        if not isinstance(selection, Mapping):
            continue
        selected_values = selection.get("values")
        selected_evidence = selection.get("evidence")
        if not isinstance(selected_values, Mapping):
            continue
        for canonical_path, value in selected_values.items():
            field_evidence = (
                selected_evidence.get(canonical_path)
                if isinstance(selected_evidence, Mapping)
                else {}
            )
            repository_values[str(canonical_path)] = (
                copy.deepcopy(value),
                str(selection.get("provenance_source") or "repository_inspection"),
                {
                    "input_path": str(input_path),
                    "choice": copy.deepcopy(selection.get("value")),
                    "repository": source.get("repository"),
                    **copy.deepcopy(dict(selection.get("source") or {})),
                    **copy.deepcopy(dict(field_evidence or {})),
                },
            )

    values: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for key, (canonical_path, manifest_field) in _COMMON_HYPERPARAMETER_FIELDS.items():
        present, spec_value = _mapping_path(spec, canonical_path)
        if (
            _path_is_explicit(spec, canonical_path)
            and present
            and spec_value is not None
        ):
            values[key] = copy.deepcopy(spec_value)
            provenance[key] = {
                "status": "resolved",
                "source": "explicit_spec",
                "canonical_path": canonical_path,
                "evidence": {"resolved_spec_sha256": resolved_spec_sha256},
            }
        elif canonical_path in repository_values:
            value, provenance_source, evidence = repository_values[canonical_path]
            values[key] = value
            provenance[key] = {
                "status": "resolved",
                "source": provenance_source,
                "canonical_path": canonical_path,
                "evidence": evidence,
            }
        elif (
            manifest_field in hyper_defaults
            and hyper_defaults[manifest_field] is not None
        ):
            values[key] = copy.deepcopy(hyper_defaults[manifest_field])
            provenance[key] = {
                "status": "resolved",
                "source": "adapter_manifest_default",
                "canonical_path": canonical_path,
                "evidence": {"adapter_manifest_sha256": manifest_sha256},
            }
        else:
            is_supported = _canonical_path_supported(canonical_path, supported)
            values[key] = None
            provenance[key] = {
                "status": "unresolved" if is_supported else "not_applicable",
                "source": "unresolved" if is_supported else "not_applicable",
                "canonical_path": canonical_path,
                "evidence": {
                    "reason": (
                        "no exact value exists in the pinned spec, adapter manifest, "
                        "or repository metadata"
                        if is_supported
                        else "adapter does not declare canonical support for this field"
                    )
                },
            }
    return {
        "schema_version": "skynet.common-hyperparameters/v1",
        "values": values,
        "provenance": provenance,
    }


def _required_unresolved_repository_defaults(
    contract: Mapping[str, Any], manifest: AdapterManifest
) -> list[str]:
    declared_repository_paths = {
        metadata.canonical_path
        for field in manifest.train.input_fields
        if field.choice_source is not None
        and not getattr(field.choice_source, "allow_custom", False)
        for metadata in field.choice_source.metadata_fields
    }
    provenance = contract.get("provenance")
    if not isinstance(provenance, Mapping):
        return sorted(declared_repository_paths)
    return sorted(
        str(record["canonical_path"])
        for record in provenance.values()
        if isinstance(record, Mapping)
        and record.get("canonical_path") in declared_repository_paths
        and record.get("status") == "unresolved"
    )


def _common_hyperparameter_contract_from_snapshot(
    snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    stored = _stored_attempt_common_hyperparameter_contract(snapshot)
    if stored is not None:
        return stored
    resolved_spec = snapshot.get("resolved_spec")
    adapter = snapshot.get("adapter")
    adapter = adapter if isinstance(adapter, Mapping) else {}
    repository = snapshot.get("repository_inputs")
    selected = repository.get("selected") if isinstance(repository, Mapping) else {}
    return _resolve_effective_common_hyperparameters(
        resolved_spec if isinstance(resolved_spec, Mapping) else None,
        adapter.get("manifest")
        if isinstance(adapter.get("manifest"), Mapping)
        else None,
        selected if isinstance(selected, Mapping) else None,
        resolved_spec_sha256=snapshot.get("resolved_spec_sha256"),
        manifest_sha256=adapter.get("manifest_sha256"),
    )


def _attempt_common_hyperparameter_contract(
    run: Mapping[str, Any],
    attempt: Mapping[str, Any],
    receipt_event: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = attempt.get("execution_snapshot_json")
    if isinstance(snapshot, Mapping):
        stored = _stored_attempt_common_hyperparameter_contract(snapshot)
        if stored is not None:
            return stored
    receipt = _validated_attempt_enrichment_receipt(run, attempt, receipt_event)
    if receipt is not None:
        return receipt
    if isinstance(snapshot, Mapping):
        return _common_hyperparameter_contract_from_snapshot(snapshot)
    resolved_spec = run.get("resolved_spec_json")
    source = (
        resolved_spec.get("source") if isinstance(resolved_spec, Mapping) else None
    )
    manifest = source.get("adapter_manifest") if isinstance(source, Mapping) else None
    return _resolve_effective_common_hyperparameters(
        resolved_spec if isinstance(resolved_spec, Mapping) else None,
        manifest if isinstance(manifest, Mapping) else None,
        resolved_spec_sha256=(
            canonical_sha256(resolved_spec)
            if isinstance(resolved_spec, Mapping)
            else None
        ),
        manifest_sha256=(
            source.get("adapter_manifest_sha256")
            if isinstance(source, Mapping)
            else None
        ),
    )


def _attempt_enrichment_event(
    database: Database, attempt_id: str
) -> dict[str, Any] | None:
    return next(
        (
            event
            for event in database.list_events(
                entity_type="job_attempt", entity_id=attempt_id, limit=5
            )
            if event.get("event_type") == "COMMON_HYPERPARAMETERS_ENRICHED_V1"
        ),
        None,
    )


def _attempt_common_hyperparameter_resolution(
    pipeline_service: PipelineService,
    run: Mapping[str, Any],
    attempt: Mapping[str, Any],
    receipt_event: Mapping[str, Any] | None,
) -> dict[str, Any]:
    snapshot = attempt.get("execution_snapshot_json")
    if isinstance(snapshot, Mapping):
        if _stored_attempt_common_hyperparameter_contract(snapshot) is not None:
            return {"status": "snapshotted", "action": None}
    if _validated_attempt_enrichment_receipt(run, attempt, receipt_event) is not None:
        return {"status": "enriched_receipt", "action": None}
    try:
        context = _attempt_pinned_enrichment_context(run, attempt)
    except ValueError as error:
        return {"status": "unavailable", "reason": str(error), "action": None}
    cache_ready = (
        _pinned_repository_metadata_cache_entry(pipeline_service, context) is not None
    )
    return {
        "status": "available_from_cache" if cache_ready else "requires_explicit_fetch",
        "action": {
            "method": "POST",
            "path": (
                f"/api/runs/{run['id']}/attempts/{attempt['id']}"
                "/common-hyperparameters/resolve"
            ),
            "body": {
                "allow_network": not cache_ready,
                "gateway": str(attempt.get("gateway") or "auto"),
            },
        },
    }


class ResolveAttemptCommonHyperparametersRequest(BaseModel):
    allow_network: bool = False
    gateway: str = Field(default="auto", min_length=1, max_length=64)


@router.post(
    "/runs/{run_id}/attempts/{attempt_id}/common-hyperparameters/resolve"
)
def resolve_attempt_common_hyperparameters(
    run_id: str,
    attempt_id: str,
    request: ResolveAttemptCommonHyperparametersRequest = Body(
        default_factory=ResolveAttemptCommonHyperparametersRequest
    ),
) -> dict[str, Any]:
    run = service.database.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    attempt = next(
        (item for item in run.get("attempts") or [] if item.get("id") == attempt_id),
        None,
    )
    if attempt is None:
        raise HTTPException(status_code=404, detail="Attempt not found for this run")
    snapshot = attempt.get("execution_snapshot_json")
    if isinstance(snapshot, Mapping):
        stored = _stored_attempt_common_hyperparameter_contract(snapshot)
        if stored is not None:
            return {
                "attempt_id": attempt_id,
                "common_hyperparameters": stored["values"],
                "common_hyperparameter_provenance": stored["provenance"],
                "common_hyperparameters_schema_version": stored["schema_version"],
                "resolution": {"status": "snapshotted", "action": None},
            }
    existing_event = _attempt_enrichment_event(service.database, attempt_id)
    existing = _validated_attempt_enrichment_receipt(run, attempt, existing_event)
    if existing is not None:
        return {
            "attempt_id": attempt_id,
            "common_hyperparameters": existing["values"],
            "common_hyperparameter_provenance": existing["provenance"],
            "common_hyperparameters_schema_version": existing["schema_version"],
            "resolution": {"status": "enriched_receipt", "action": None},
        }
    try:
        context = _attempt_pinned_enrichment_context(run, attempt)
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "COMMON_HYPERPARAMETERS_UNRESOLVABLE",
                "message": str(error),
                "can_retry_with_network": False,
            },
        ) from error

    cached = _pinned_repository_metadata_cache_entry(service, context)
    if cached is None and not request.allow_network:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "COMMON_HYPERPARAMETER_CACHE_MISS",
                "message": (
                    "No exact pinned repository metadata is cached. Retry this explicit "
                    "operation with allow_network=true to inspect the pinned commit."
                ),
                "can_retry_with_network": True,
                "required_commit": context["commit"],
            },
        )
    if cached is None:
        try:
            inspection = service.source_discovery.inspect(
                context["repository"],
                context["commit"],
                request.gateway,
                context["project_subdirectory"],
            )
            inspection["input_options"] = service.source_discovery.input_options(
                context["repository"],
                context["commit"],
                context["fields"],
                request.gateway,
                context["project_subdirectory"],
            )
            entry = service.source_metadata.put(
                "common_hyperparameters",
                context["repository"],
                context["parameters"],
                inspection,
            )
            options = _validated_cached_repository_options(context, entry)
            cached = ("common_hyperparameters", entry, options)
        except Exception as error:
            raise _http_error(error) from error
    cache_kind, cache_entry, options = cached
    selected_repository_inputs = _with_historical_rule_values(
        _selected_repository_input_metadata(context["spec"], options), context
    )
    contract = _resolve_effective_common_hyperparameters(
        context["spec"],
        context["manifest"],
        selected_repository_inputs,
        resolved_spec_sha256=context["resolved_spec_sha256"],
        manifest_sha256=context["adapter_manifest_sha256"],
    )
    pinned = {
        "attempt_id": attempt_id,
        "execution_snapshot_sha256": context["execution_snapshot_sha256"],
        "resolved_spec_sha256": context["resolved_spec_sha256"],
        "adapter_manifest_sha256": context["adapter_manifest_sha256"],
        "repository": context["repository"],
        "commit": context["commit"],
        "project_subdirectory": context["project_subdirectory"],
        "input_discovery_sha256": context["parameters"]["input_discovery_sha256"],
    }
    receipt_body = {
        "schema_version": "skynet.attempt-common-hyperparameter-receipt/v1",
        "pinned": pinned,
        "enrichment_rules": copy.deepcopy(context["rules"]),
        "source_cache": {
            "kind": cache_kind,
            "key": cache_entry["cache_key"],
            "updated_at": cache_entry["updated_at"],
            "payload_sha256": canonical_sha256(cache_entry["payload"]),
        },
        "repository_inputs": selected_repository_inputs,
        "common_hyperparameters": contract,
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": canonical_sha256(receipt_body),
    }
    event = service.database.record_attempt_common_hyperparameter_receipt(
        attempt_id, receipt
    )
    stored = _validated_attempt_enrichment_receipt(run, attempt, event)
    if stored is None:
        raise HTTPException(
            status_code=409,
            detail="Stored common hyperparameter receipt failed immutable binding validation",
        )
    return {
        "attempt_id": attempt_id,
        "common_hyperparameters": stored["values"],
        "common_hyperparameter_provenance": stored["provenance"],
        "common_hyperparameters_schema_version": stored["schema_version"],
        "resolution": {
            "status": "enriched_receipt",
            "receipt_event_id": event["id"],
            "action": None,
        },
    }
@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = service.database.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    service._ingest_training_progress(run)
    for attempt in run.get("attempts") or []:
        receipt_event = _attempt_enrichment_event(
            service.database, str(attempt["id"])
        )
        common = _attempt_common_hyperparameter_contract(
            run, attempt, receipt_event
        )
        attempt["common_hyperparameters"] = common["values"]
        attempt["common_hyperparameter_provenance"] = common["provenance"]
        attempt["common_hyperparameters_schema_version"] = common["schema_version"]
        attempt["common_hyperparameters_resolution"] = (
            _attempt_common_hyperparameter_resolution(
                service, run, attempt, receipt_event
            )
        )
    run["latest_checkpoint"] = run["checkpoints"][-1]["path"] if run["checkpoints"] else None
    run["manual_actions"] = service.run_manual_actions(run)
    run["evaluation_blocker"] = _evaluation_busy_reason(run)
    run["tracking_actions"] = service.run_tracking_actions(run)
    _attach_run_progress_summaries(service.database, [run])
    return {"run": run}


@router.post("/runs/{run_id}/tracking/{provider}/attach")
def attach_run_tracking(run_id: str, provider: str) -> dict[str, Any]:
    try:
        service.attach_run_tracking(run_id, provider)
        return get_run(run_id)
    except Exception as error:
        raise _http_error(error) from error


@router.get("/runs/{run_id}/logs", response_class=PlainTextResponse)
def get_run_log(
    run_id: str,
    stream: str = Query(default="stderr", pattern="^(stdout|stderr)$"),
    lines: int = Query(default=500, ge=1, le=5000),
) -> str:
    try:
        return service.run_log(run_id, stream, lines)
    except Exception as error:
        raise _http_error(error) from error


@router.get("/runs/{run_id}/attempts/{attempt_id}/logs", response_class=PlainTextResponse)
def get_run_attempt_log(
    run_id: str,
    attempt_id: str,
    stream: str = Query(default="stderr", pattern="^(stdout|stderr)$"),
    lines: int = Query(default=500, ge=1, le=5000),
) -> str:
    try:
        return service.run_attempt_log(run_id, attempt_id, stream, lines)
    except Exception as error:
        raise _http_error(error) from error


@router.post("/runs/{run_id}/recover-submission")
def recover_run_submission(run_id: str, request: GatewayRequest) -> dict[str, Any]:
    try:
        return service.recover_run_submission(run_id, request.gateway)
    except Exception as error:
        raise _http_error(error) from error


@router.post("/runs/{run_id}/resume")
def resume_run(run_id: str, request: ResumeRequest) -> dict[str, Any]:
    try:
        return service.retry_run(run_id, request.mode, request.gateway)
    except Exception as error:
        raise _http_error(error) from error


@router.post("/runs/{run_id}/rerun")
def rerun_run(run_id: str, request: GatewayRequest) -> dict[str, Any]:
    try:
        return service.rerun_run(run_id, request.gateway)
    except Exception as error:
        raise _http_error(error) from error


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, Any]:
    try:
        return service.cancel_run(run_id)
    except Exception as error:
        raise _http_error(error) from error


@router.get("/evaluations")
def list_evaluations() -> dict[str, Any]:
    evaluations = service.database.list_evaluations()
    for evaluation in evaluations:
        service._ingest_evaluation_progress(evaluation)
    evaluations = service.database.list_evaluations()
    _attach_evaluation_progress_summaries(service.database, evaluations)
    return {"evaluations": evaluations}


@router.get("/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: str) -> dict[str, Any]:
    evaluation = service.database.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    service._ingest_evaluation_progress(evaluation)
    evaluation = service.database.get_evaluation(evaluation_id)
    assert evaluation is not None
    run = service.database.get_run(evaluation["run_id"])
    evaluation["attempts"] = [
        attempt
        for attempt in (run["attempts"] if run else [])
        if attempt.get("stage_id") == evaluation.get("stage_id")
    ]
    artifacts = [
        artifact
        for artifact in (run["artifacts"] if run else [])
        if artifact.get("evaluation_id") == evaluation_id
    ]
    evaluation["artifacts"] = artifacts
    result_artifact = next(
        (artifact for artifact in artifacts if artifact["artifact_type"] == "EVALUATION_RESULT"),
        None,
    )
    evaluation["aggregate"] = (
        result_artifact["metadata_json"].get("aggregate", []) if result_artifact else []
    )
    evaluation["progress_summary"] = evaluation_progress_summary(evaluation)
    evaluation["manual_actions"] = service.evaluation_manual_actions(evaluation, run)
    return {"evaluation": evaluation}


@router.post("/evaluations/{evaluation_id}/cancel")
def cancel_evaluation(evaluation_id: str) -> dict[str, Any]:
    try:
        return service.cancel_evaluation(evaluation_id)
    except Exception as error:
        raise _http_error(error) from error


@router.get("/evaluations/{evaluation_id}/episodes/{episode_id}/video")
def get_evaluation_episode_video(
    evaluation_id: str,
    episode_id: str,
    request: Request,
) -> StreamingResponse:
    evaluation = service.database.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    episode = next(
        (item for item in evaluation.get("episodes", []) if item.get("id") == episode_id),
        None,
    )
    if not episode or not episode.get("video_path"):
        raise HTTPException(status_code=404, detail="Rollout video not found")

    result_path = evaluation.get("result_path")
    if not isinstance(result_path, str) or not result_path:
        raise HTTPException(status_code=409, detail="Evaluation result path is unavailable")
    video_path = PurePosixPath(str(episode["video_path"]))
    video_root = PurePosixPath(result_path).parent / "videos"
    if (
        not video_path.is_absolute()
        or video_root not in video_path.parents
        or video_path.suffix.lower() != ".mp4"
        or ".." in video_path.parts
    ):
        raise HTTPException(status_code=409, detail="Registered rollout video path is invalid")

    run = service.database.get_run(evaluation["run_id"])
    attempts = [
        attempt
        for attempt in ((run or {}).get("attempts") or [])
        if attempt.get("stage_id") == evaluation.get("stage_id")
    ]
    gateway = (
        max(attempts, key=lambda item: int(item.get("attempt_number") or 0)).get("gateway")
        if attempts
        else "auto"
    ) or "auto"
    try:
        host, size = service.cluster.file_size(str(video_path), gateway)
    except Exception as error:
        raise _http_error(error) from error
    if size < 1:
        raise HTTPException(status_code=404, detail="Rollout video is empty")

    range_header = request.headers.get("range")
    start = 0
    end = size - 1
    partial = False
    if range_header:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if not match or (not match.group(1) and not match.group(2)):
            raise HTTPException(
                status_code=416,
                detail="Unsupported video byte range",
                headers={"Content-Range": f"bytes */{size}"},
            )
        if match.group(1):
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else size - 1
            if start >= size or end < start:
                raise HTTPException(
                    status_code=416,
                    detail="Video byte range is outside the file",
                    headers={"Content-Range": f"bytes */{size}"},
                )
            end = min(end, size - 1)
        else:
            suffix_size = int(match.group(2))
            if suffix_size < 1:
                raise HTTPException(
                    status_code=416,
                    detail="Video suffix range must be positive",
                    headers={"Content-Range": f"bytes */{size}"},
                )
            start = max(0, size - suffix_size)
        partial = True

    length = end - start + 1
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Cache-Control": "private, max-age=3600",
        "Content-Disposition": "inline",
        "X-Content-Type-Options": "nosniff",
    }
    if partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(
        service.cluster.stream_file_range(
            str(video_path),
            host,
            start=start,
            end=end,
        ),
        status_code=206 if partial else 200,
        media_type="video/mp4",
        headers=headers,
    )


@router.post("/evaluations/validate-target")
def validate_evaluation_target(
    request: EvaluationTargetValidationRequest,
) -> dict[str, Any]:
    return service.validate_evaluation_target(
        request.run_id,
        request.checkpoint_path,
        suite_id=request.suite_id,
        environment=request.environment,
        tasks=request.tasks,
        episodes_per_task=request.episodes_per_task,
        seeds=request.seeds,
        parallelism=request.parallelism,
        headless=request.headless,
        gateway=request.gateway,
        resources=request.resources,
        argv=request.argv,
        resume_argv=request.resume_argv,
    )


@router.post("/evaluations")
def create_evaluation(request: EvaluationRequest) -> dict[str, Any]:
    try:
        return {"evaluation": service.create_evaluation(request)}
    except Exception as error:
        raise _http_error(error) from error


@router.post("/reconcile")
def reconcile() -> dict[str, Any]:
    try:
        return service.reconcile()
    except Exception as error:
        raise _http_error(error) from error


@router.get("/tracking/connections")
def list_tracking_connections() -> dict[str, Any]:
    return service.tracking_connections()


@router.post("/tracking/connections/{provider}/connect")
@router.post("/tracking/connections/{provider}", include_in_schema=False)
def configure_tracking_connection(
    provider: str, request: TrackingConnectionRequest
) -> dict[str, Any]:
    try:
        return service.configure_tracking_connection(provider, request)
    except CredentialStoreError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except TrackingRequestError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/tracking/connections/{provider}/test")
def test_tracking_connection(provider: str) -> dict[str, Any]:
    try:
        return service.test_tracking_connection(provider)
    except TrackingRequestError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/tracking/connections/{provider}/disconnect")
def disconnect_tracking_connection(provider: str) -> dict[str, Any]:
    try:
        return service.disconnect_tracking_connection(provider)
    except CredentialStoreError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.delete("/tracking/connections/{provider}", include_in_schema=False)
def delete_tracking_connection(provider: str) -> dict[str, Any]:
    return disconnect_tracking_connection(provider)


@router.get("/settings")
def settings() -> dict[str, Any]:
    connections = service.tracking_connections()["connections"]
    tracking = {
        "providers": list(connections.values()),
        "connections": connections,
        "secrets_persisted": False,
    }
    return {
        "paths": {
            "work_root": WORK_ROOT,
            "database": str(service.database.path),
            "local_capsules": str(LOCAL_CAPSULE_ROOT),
            "evaluation_root": CLUSTER.paths.evaluation,
        },
        "cluster": {
            **CLUSTER.public_dict(),
            "multi_node": False,
            "multi_gpu_single_node": True,
        },
        "runtime_profiles": CLUSTER.public_runtime_profiles(),
        "checkpoint": {
            "save_every_steps": CLUSTER.defaults.checkpoint_save_steps,
            "keep_last": CLUSTER.defaults.checkpoint_keep_last,
            "auto_resume": CLUSTER.defaults.checkpoint_auto_resume,
            "max_attempts": CLUSTER.defaults.max_attempts,
        },
        "tracking": tracking,
        "reproducibility": {
            "modes": ["exact-input", "deterministic-requested", "statistical-only"],
            "secrets_persisted": False,
        },
    }


__all__ = ["PipelineService", "router", "service"]
