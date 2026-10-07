from __future__ import annotations

from skynet_app.evaluation_contracts import bind_suite_to_dataset
from skynet_app.evaluation_targets import attach_evaluation_target, evaluation_target_contract
from skynet_app.recorded_evaluation import recorded_episode_sources
from skynet_app import data_selection
from skynet_app.gpu_tracking import progress_timestamp_ms, sync_gpu_statistics
from skynet_app.model_io import resolve_model_io, preview_spec

import math
import logging
import ipaddress
import urllib.parse

import copy
import functools
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
import re
import shlex
from .db_backend import INTEGRITY_ERRORS, DATABASE_ERRORS
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

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
    resolve_gpu_type,
    resolve_training_progress_contract,
)
from .adapters.dataset_inputs import evaluation_context_json
from .gpu_preflight import (
    GPU_MISSING_EXIT_CODE,
    GPU_MISSING_MESSAGE,
    GPU_MISSING_STATE,
    INTERRUPTION_RECEIPT,
    TIME_LIMIT_EXIT_CODE,
    gpu_missing_receipt,
    time_limit_receipt,
)
from .gpu_quota import account_gpu_quota, idle_partition_quota
from .preparation_states import ATTEMPT_FAILURE_STATES, EXECUTING_STATES, TRANSIENT_STATES
from .workflow_states import ACTIVE_STAGE_STATES, CANCELLABLE_STAGE_STATES, SLURM_BOUND_ATTEMPT_STATES, sql_list
from .cluster_config import CLUSTER
from .evaluation_placement import resolve_evaluation_resources, uses_isaac_sim
from .evaluation_compatibility import inspect_compatibility, compose_evaluator, declaration_matches, policy_loader, dataset_metadata
from .cluster_runtime import (
    ACTIVE_STATES,
    ClusterClient,
    ClusterError,
    SLURM_BIN,
    SubmissionOutcomeUnknown,
    approved_operator_environment,
)
from .data_resource_policy import (
    RESOURCE_TYPES, validate_resource_edit, validate_resource_metadata, validate_resource_type,
)
from .data_import_states import (
    DATA_IMPORT_IN_FLIGHT_STATES,
    DATA_IMPORT_SCHEDULED_STATES,
    DATA_IMPORT_SETTLING_STATES,
    DATA_IMPORT_TERMINAL_STATES,
)
from .data_imports import HuggingFaceImportRequest, build_huggingface_import_job
from .data_paths import validate_mount_path
from .credential_store import (
    CredentialStore,
    CredentialStoreError,
    KeyringCredentialStore,
)
from .database import Database, canonical_json, content_sha256, new_id, utc_now
from .workspace_schema import LEGACY_WORKSPACE, visible_sql
from .workspace_storage import WorkspaceStorage, paths_for_root, validate_work_root, evaluation_execution_directory
from .slack_notifications import SlackNotifications
from .job_status import SUBMISSION_UNKNOWN_PREFIX, attach_attempt_display_status, attach_job_display_status
from .lazy_service import LazyService
from .metadata_objects import MetadataObjects
from .workspaces import BACKGROUND_POLL_INTERVAL_SECONDS, WorkspaceServices, require_workspace_records
from .experiments import (
    CanonicalResult,
    DEFAULT_EVALUATION_EPISODES,
    DEFAULT_EVALUATION_PROFILE,
    DEFAULT_EVALUATION_SEEDS,
    ExperimentSpec,
    FULL_COMMIT_RE,
    MAX_EVALUATION_EPISODES,
    ResourceSpec,
    SPEC_API_VERSION,
    canonical_sha256,
    explicit_train_parameter_paths,
    expand_sweep,
    get_evaluation_catalog,
    parse_slurm_duration,
)
from .slurm import (
    SlurmCompileError,
    compile_sbatch,
    resolve_slurm_log_path,
    resolve_slurm_log_paths_from_sbatch,
)
from .source_control import SourceDiscovery, resolve_runtime
from .source_metadata_cache import SourceMetadataStore
from .training_metrics import INTERNAL_PROGRESS_METRICS, is_scalar, recorded_scalar_metrics
from .training_progress_log import BOUNDARY_NOT_READY
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
    TRACKING_PROVIDERS,
    TrackingRequestError,
    TrackingSettings,
    WandBBridge,
    WANDB_DEFAULT_BASE_URL,
    WandBSettings,
    compact_tracking_parameters,
    mlflow_experiment_url,
    mlflow_run_url,
    sanitize,
    wandb_project_slug,
    wandb_web_base,
)
from pydantic import ConfigDict, SecretStr


# The gateway's NFS view can trail a job's end, so what a finished job left behind
# (an evaluation's result.json, a training launch boundary) is read again for a
# short window before its absence is final.
RESULT_READ_GRACE = timedelta(minutes=10)
# A job the scheduler no longer describes is settled from the exit record its own batch
# script wrote ({run}/attempts/{job}/final.json); without one it is held, never guessed.
# Rows a list endpoint returns at most; the UI filters within this window.
LIST_RESPONSE_LIMIT = 1000
EXIT_RECORD_SOURCE = "exit-record"
EXIT_RECORD_REASON = "Taken from the job's exit record; Slurm no longer has a record of this job"
FORGOTTEN_JOB_CANCELLED = "Cancellation requested; Slurm no longer lists this job"
FORGOTTEN_JOB_HELD = (
    "Slurm no longer lists this job and it left no exit record. "
    "Its outcome is unknown: cancel it, or wait for Slurm accounting."
)
HELD_RECHECK_SECONDS = 600
# Runs one tracking delivery pass flushes per provider, and spooled events one drain
# sends per run (the delivery pass, the progress publisher and the terminal sync alike).
TRACKING_FLUSH_RUN_LIMIT = 10
TRACKING_FLUSH_EVENT_LIMIT = 100
# Queued runs one provider flush replays outside the delivery pass, as when a
# tracking connection is saved.
TRACKING_CONNECTION_FLUSH_RUN_LIMIT = 100
# Seconds before the final progress read of an ended attempt is retried after it
# failed or found no evidence yet.
FINAL_PROGRESS_RETRY_SECONDS = 60
# Seconds between remote reads of one running evaluation's episode progress; the
# list and detail endpoints polled together share one read.
EVALUATION_PROGRESS_READ_INTERVAL_SECONDS = 4.0
# List-driven remote progress refreshes queued at most, and threads draining the queue.
PROGRESS_REFRESH_QUEUE_LIMIT = 2000
PROGRESS_REFRESH_WORKERS = 2
# Seconds an evaluator runtime readiness probe is reused: longer when it found no
# blockers, shorter when it did.
RUNTIME_READINESS_CACHE_SECONDS = 60.0
RUNTIME_READINESS_BLOCKED_CACHE_SECONDS = 10.0
# Seconds allowed for the live GPU usage query and the idle-quota verification that
# automatic queue selection runs over SSH.
GPU_USAGE_QUERY_TIMEOUT_SECONDS = 20
IDLE_QUOTA_VERIFICATION_TIMEOUT_SECONDS = 55
# Characters of sbatch's own output kept on an attempt as its submission reason.
SUBMISSION_OUTPUT_LIMIT = 2000
GPU_USAGE_LONG_COMMAND = CLUSTER.commands.gpu_usage_shell_command("-l")


def _within_result_read_grace(finished_at: str | None) -> bool:
    if not finished_at:
        return False
    try:
        finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if finished.tzinfo is None:
        return False
    return datetime.now(timezone.utc) - finished < RESULT_READ_GRACE
# v1 acknowledged completion before the last data event in older journals.
# A distinct reconciliation marker migrates those already-FINISHED bindings;
# finish event identities remain stable and include their data watermark.
WANDB_TERMINAL_SYNC_PROTOCOL = "file-stream-watermark-v1"

LOCAL_CAPSULE_ROOT = Path(
    os.environ.get(
        "SKYNET_LOCAL_CAPSULE_ROOT",
        str(Path(__file__).resolve().parent.parent / "data" / "capsules"),
    )
).expanduser()


def _frontend_parameter_is_unset(value: Any) -> bool:
    return value is None or value == "" or value == "adapter-default"


_UNSET: Any = object()

# Frontend submission field -> (canonical train path, manifest defaults attribute).
# Hyperparameter rows arrive under payload["hyperparameters"], checkpoint rows at
# the payload's top level; everything that translates between the frontend and
# canonical shapes (explicit-intent detection, manifest defaults) reads this table.
FRONTEND_TRAIN_FIELDS: dict[str, tuple[str, str]] = {
    "learning_rate": ("train.learning_rate", "hyperparameters.learning_rate"),
    "batch_semantics": ("train.batch.declared_semantics", "hyperparameters.batch_semantics"),
    "batch_size": ("train.batch.value", "hyperparameters.batch_size"),
    "gradient_accumulation": (
        "train.batch.gradient_accumulation_steps", "hyperparameters.gradient_accumulation_steps",
    ),
    "num_workers": ("train.num_workers_per_rank", "hyperparameters.num_workers_per_rank"),
    "max_steps": ("train.max_steps", "hyperparameters.max_steps"),
    "max_epochs": ("train.max_epochs", "hyperparameters.max_epochs"),
    "seed": ("train.seed", "hyperparameters.seed"),
    "precision": ("train.precision", "hyperparameters.precision"),
    "checkpoint_save_steps": ("train.checkpoint.save_every_steps", "checkpoint.save_every_steps"),
    "checkpoint_warning_seconds": (
        "train.checkpoint.save_before_timeout_seconds", "checkpoint.save_before_timeout_seconds",
    ),
    "checkpoint_keep_last": ("train.checkpoint.keep_last", "checkpoint.keep_last"),
    "auto_resume": ("train.checkpoint.auto_resume", "checkpoint.auto_resume"),
    "max_attempts": ("train.checkpoint.max_attempts", "checkpoint.max_attempts"),
    "checkpoint_final_selector": ("train.checkpoint.final_selector", "checkpoint.final_selector"),
    "remove_training_state_after_success": (
        "train.checkpoint.remove_training_state_after_success",
        "checkpoint.remove_training_state_after_success",
    ),
}
FRONTEND_HYPERPARAMETER_FIELDS = frozenset(
    field for field, (_, attribute) in FRONTEND_TRAIN_FIELDS.items()
    if attribute.startswith("hyperparameters.")
)


def _frontend_train_field_holder(
    field: str, payload: Mapping[str, Any], hyperparameters: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Where a table field sits in a frontend payload: its hyperparameters or top level."""
    return hyperparameters if field in FRONTEND_HYPERPARAMETER_FIELDS else payload


def _manifest_default(manifest: AdapterManifest, attribute: str) -> Any:
    value: Any = manifest.defaults
    for part in attribute.split("."):
        value = getattr(value, part)
    return value


_PROGRESS_SUCCESS_STATES = {"SUCCEEDED", "COMPLETED", "COMPLETE"}
_PROGRESS_FAILURE_STATES = {
    "FAILED", "CANCELLED", "CANCELED", "TIMEOUT", "TIMED_OUT",
    "OUT_OF_MEMORY", "DEADLINE", "SPECIAL_EXIT", "BLOCKED",
}
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


def _checkpoint_training_step(value: Any) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("Checkpoint training step must be a nonnegative integer")
    return value


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


def _resolved_training_total(value: Any, path: str) -> int | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping):
        return None
    for key in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    total = _progress_integer(value)
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
    content: str, contract: TrainingProgressContract | Mapping[str, Any],
    *, resolved_spec: Any = None,
) -> list[dict[str, Any]]:
    """Normalize bounded adapter-declared log matches; no adapter grammar lives here."""
    resolved = resolve_training_progress_contract(contract, resolved_spec)
    source = resolved.source
    if source.kind == "jsonl":
        total = _resolved_training_total(resolved_spec, resolved.total_path)
        if total is None:
            return []
        records = []
        for line in content.splitlines(keepends=True):
            # A live writer may still be writing the final record.
            if not line.endswith("\n"):
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(row, dict) or source.required_key not in row:
                continue
            completed = _progress_integer(row.get(source.completed_key))
            if completed is None:
                continue
            completed += source.completed_offset
            if completed > total:
                continue
            metrics = recorded_scalar_metrics(row, source.metrics)
            records.append({
                "completed": completed, "total": total,
                "elapsed_seconds": None, "metrics": metrics,
            })
        return records
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
                return resolve_training_progress_contract(progress, run.get("resolved_spec_json")), "pinned_plan"
            except Exception:
                return None

    spec = run.get("resolved_spec_json") or {}
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except ValueError:
            spec = {}
    source = spec.get("source") if isinstance(spec, Mapping) else None
    manifest = source.get("adapter_manifest") if isinstance(source, Mapping) else None
    train = manifest.get("train") if isinstance(manifest, Mapping) else None
    progress = train.get("progress") if isinstance(train, Mapping) else None
    if progress:
        try:
            return resolve_training_progress_contract(progress, spec), "pinned_manifest"
        except ValueError:
            return None
    return None


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
    """Build an ETA from persisted completed-work evidence in this attempt."""
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
    declared = _training_progress_contract({**run, "resolved_spec_json": spec})
    contract = declared[0] if declared else None
    unit = contract.unit if contract else "step"
    total = _resolved_training_total(spec, contract.total_path if contract else "train.max_steps")
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
    for checkpoint in checkpoint_rows if unit == "step" else []:
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
            and sample.get("unit", "step") == unit
            and (
                str(producer) != current_attempt_id
                or sample_restart_count == current_restart_count
            )
        ):
            samples.append((recorded, step, str(producer)))
    for metric in metric_rows if unit == "step" else []:
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
    previous_attempt = (
        len(attempt_rows) > 1
        or (_progress_integer((current_attempt or {}).get("attempt_number")) or 0) > 1
    )
    pinned_resume = bool((current_attempt or {}).get("has_initial_checkpoint"))
    for document, paths in (
        ((current_attempt or {}).get("execution_snapshot_json"), (
            "plan.native_config.initial_checkpoint", "migration_provenance.checkpoint",
            "resolved_spec.native.config.initial_checkpoint",
        )),
        (spec, ("native.config.initial_checkpoint",)),
    ):
        if isinstance(document, str):
            try:
                document = json.loads(document)
            except ValueError:
                document = None
        pinned_resume = pinned_resume or any(
            bool(_mapping_path(document, path)[1]) for path in paths
        )
    if resume_checkpoint_id is not None:
        checkpoint = checkpoint_by_id.get(str(resume_checkpoint_id))
        baseline_step = _progress_integer((checkpoint or {}).get("training_step"))
        if checkpoint is not None and baseline_step is not None and started_at is not None:
            baseline = (started_at, baseline_step, current_attempt_id)
            baseline_observed_at = _progress_timestamp(checkpoint.get("created_at"))
    elif (
        contract and contract.starts_at_zero and started_at
        and current_restart_count == 0 and not previous_attempt and not pinned_resume
    ):
        # starts_at_zero describes a fresh loop, not an automatic retry that
        # restores a checkpoint without populating resume_checkpoint_id. Unknown
        # retry baselines require two observations from the current attempt.
        baseline = (started_at, 0, current_attempt_id)
        baseline_observed_at = started_at

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
        "unit": unit,
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
    if attempt_status and attempt_status not in EXECUTING_STATES:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="waiting", eta_reason="waiting_for_retry"
        )
    if not current_samples:
        return _progress_payload(
            **common, eta_seconds=None, eta_state="waiting", eta_reason="waiting_for_progress"
        )

    # A JSONL backfill can attach one poll timestamp to hundreds of old steps.
    # It provides one timing observation at the highest observed step, not a
    # measured interval starting at the first row in that batch.
    timing_observations: dict[datetime, tuple[datetime, int, str | None]] = {}
    for sample in ([baseline] if baseline is not None else []) + current_samples:
        previous = timing_observations.get(sample[0])
        if previous is None or sample[1] > previous[1]:
            timing_observations[sample[0]] = sample
    rate_samples = sorted(timing_observations.values(), key=lambda sample: sample[0])
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
    if current_attempt is None and status in ({"RUNNING"} | _PROGRESS_SUCCESS_STATES | _PROGRESS_FAILURE_STATES):
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
    if attempt_status and attempt_status not in EXECUTING_STATES:
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
        attach_job_display_status(run, attempts)
        # Detail responses have separately loaded (and enriched) attempt objects.
        for attempt in run.get("attempts") or []:
            attach_attempt_display_status(attempt)
        spec = row.get("resolved_spec_json") or {}
        run["resources"] = spec.get("resources") or {}
        run["training_data"] = data_selection.describe(spec)
        run["progress_summary"] = training_progress_summary(
            run,
            attempts=row.get("attempts") or [],
            checkpoints=row.get("checkpoints") or [],
            metrics=row.get("metrics") or [],
            progress_samples=row.get("progress_samples") or [],
            resolved_spec=row.get("resolved_spec_json"),
        )


    data_selection.attach_links(database, runs)


def _attach_evaluation_progress_summaries(
    database: Database, evaluations: list[dict[str, Any]]
) -> None:
    evidence = database.evaluation_progress_evidence([
        str(evaluation.get("id") or "") for evaluation in evaluations
    ])
    for evaluation in evaluations:
        row = evidence.get(str(evaluation.get("id") or ""), {})
        evaluation["latest_attempt"] = _latest_attempt(row.get("attempts") or [])
        attach_job_display_status(evaluation, row.get("attempts") or [])
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


class ExperimentRevisionRequest(GatewayRequest):
    spec: dict[str, Any]
    submit: bool = False


_MANUAL_ACTION_ACTIVE_STATES = frozenset({
    "SUBMITTING", "SUBMITTED", "PENDING", "PENDING_SLURM", "RUNNING", "REQUEUED",
    "RETRY_PENDING", "CANCELLING",
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
    if state not in CANCELLABLE_STAGE_STATES:
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
        in (CANCELLABLE_STAGE_STATES | {"CANCELLING"})
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


def _stage_attempts(run: Mapping[str, Any] | None, stage_id: Any) -> list[Mapping[str, Any]]:
    """Attempts recorded for one stage; unlike training lookups, a stage-less attempt never qualifies."""
    return [
        attempt for attempt in list((run or {}).get("attempts") or [])
        if attempt.get("stage_id") == stage_id
    ]


def _latest_stage_attempt(
    run: Mapping[str, Any] | None, stage_id: Any
) -> Mapping[str, Any] | None:
    return _latest_attempt(_stage_attempts(run, stage_id))


def _evaluation_busy_reason(run: Mapping[str, Any]) -> str | None:
    active_states = {"SUBMITTING", "SUBMITTED", "PENDING_SLURM", "RUNNING", "RETRY_PENDING", "CANCELLING"}
    for stage in run.get("stages", []):
        if str(stage.get("status", "")).upper() not in active_states:
            continue
        context = (stage.get("resolved_config_json") or {}).get("context") or {}
        if stage.get("stage_type") == "EVALUATE" and context.get("execution_key") == stage.get("id") and stage.get("id"):
            continue
        return "This run already has active training or an evaluation using shared execution files. Wait for it to finish, or cancel that work before starting another evaluation."
    return None


_MANUAL_RESUME_CHECKPOINT_PROBE = r'''import fnmatch, hashlib, json, os, re, stat, sys
from pathlib import Path
request = json.loads(sys.argv[1])
root = Path(request["root"])
receipt_path = root / "checkpoints/latest.json"
def require(condition, message):
    if not condition:
        raise RuntimeError(message)
def checked_stat(path):
    try:
        value = path.lstat()
    except FileNotFoundError:
        return None
    require(not stat.S_ISLNK(value.st_mode), "Checkpoint search contains a symlink")
    return value
def strict_glob(base, pattern):
    # pathlib glob/rglob may suppress directory access errors. Only actual
    # missing paths mean no candidate; unreadable storage must fail closed.
    parts = Path(pattern).parts
    require(parts and not Path(pattern).is_absolute() and ".." not in parts,
            "Checkpoint glob must stay relative to its namespace")
    require(all("**" not in part or part == "**" for part in parts),
            "Recursive checkpoint glob must use a complete path component")
    def walk(path, remaining):
        value = checked_stat(path)
        if value is None:
            return
        if not remaining:
            if not pattern.endswith("/") or stat.S_ISDIR(value.st_mode):
                yield path
            return
        if not stat.S_ISDIR(value.st_mode):
            return
        part, rest = remaining[0], remaining[1:]
        if part == "**":
            yield from walk(path, rest)
            with os.scandir(path) as entries:
                children = [Path(entry.path) for entry in entries]
            for child in children:
                child_stat = checked_stat(child)
                if child_stat is not None and stat.S_ISDIR(child_stat.st_mode):
                    yield from walk(child, remaining)
        elif not any(character in part for character in "*?["):
            yield from walk(path / part, rest)
        else:
            with os.scandir(path) as entries:
                children = [Path(entry.path) for entry in entries if fnmatch.fnmatchcase(entry.name, part)]
            for child in children:
                yield from walk(child, rest)
    yield from walk(base, parts)
def read_receipt(path):
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "Resume receipt escapes the run namespace")
    value = checked_stat(path)
    if value is None:
        return None
    require(stat.S_ISREG(value.st_mode) and value.st_size <= 131072, "Invalid resume receipt file")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "Resume receipt must be an object")
    return value
contract = request["contract"]
patterns = contract["candidate_globs"]
require(isinstance(patterns, list), "Checkpoint candidate globs must be a list")
for pattern in patterns:
    require(isinstance(pattern, str) and bool(pattern), "Checkpoint candidate glob must be non-empty")
    require(not Path(pattern).is_absolute() and ".." not in Path(pattern).parts,
            "Checkpoint candidate glob must stay relative to the run namespace")
candidates = {path for pattern in patterns for path in strict_glob(root, pattern)}
receipt = read_receipt(receipt_path)
if receipt is None:
    require(not candidates, "Checkpoint files exist but latest.json is missing; recover the runner receipt before resuming")
    print(json.dumps({"present": False}))
    sys.exit(0)
require(isinstance(receipt, dict), "Resume receipt must be an object")
require(receipt.get("schema_version") == 2 and receipt.get("run_id") == request["run_id"], "Resume receipt identity differs from the run")
require(receipt.get("final") is False and receipt.get("resumable") is True, "Checkpoint is not resumable training state")
cleanup = receipt.get("cleanup")
require(isinstance(cleanup, dict) and cleanup.get("performed") is False and not cleanup.get("removed") and not cleanup.get("removed_outputs"), "Checkpoint training state was pruned")
require(receipt.get("contract") == contract, "Resume receipt differs from the pinned checkpoint contract")
value = receipt.get("path")
require(isinstance(value, str) and value, "Checkpoint path is missing")
target = Path(value)
require(target.is_absolute() and ".." not in target.parts, "Checkpoint path is invalid")
relative = target.relative_to(root)
require(relative != Path(".") and target.resolve().is_relative_to(root.resolve()), "Checkpoint escapes the run namespace")
require(target in candidates, "Checkpoint path does not match its pinned glob")
pattern = contract["basename_regex"]
require(not pattern or re.fullmatch(pattern, target.name), "Checkpoint basename differs from its pinned contract")
require(not target.is_symlink() and (target.is_file() or target.is_dir()), "Checkpoint target is missing or unsafe")
require(type(receipt.get("is_directory")) is bool and receipt["is_directory"] == target.is_dir(), "Checkpoint type differs from its receipt")
kind = contract["candidate_kind"]
require(kind == "any" or (kind == "directory" and target.is_dir()) or (kind == "file" and target.is_file()), "Checkpoint type differs from its pinned contract")
require(isinstance(receipt.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"]), "Checkpoint SHA-256 is invalid")
for key in ("size_bytes", "file_count"):
    require(type(receipt.get(key)) is int and receipt[key] > 0, "Checkpoint size or file count is invalid")
step = receipt.get("training_step")
require(step is None or (type(step) is int and step >= 0), "Checkpoint training step is invalid")
def fingerprint():
    paths = [target, *sorted(strict_glob(target, "**/*"))] if target.is_dir() else [target]
    rows = []
    for path in paths:
        require(not path.is_symlink(), "Checkpoint contains a symlink")
        s = path.stat()
        require(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode), "Checkpoint contains a special file")
        rows.append((str(path), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))
    return rows
def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
before = fingerprint()
if target.is_file():
    identity = dict(sha256=file_sha(target), size_bytes=target.stat().st_size, file_count=1, is_directory=False)
else:
    # This is the existing runner schema-2 directory identity wire format.
    # Fixed stdlib code avoids importing/executing the remote runtime wrapper.
    records, size, count = [], 0, 0
    for child in sorted(strict_glob(target, "**/*"), key=lambda p: p.relative_to(target).as_posix()):
        relative = child.relative_to(target).as_posix()
        if child.is_file():
            child_size = child.stat().st_size
            records.append(dict(path=relative, type="file", sha256=file_sha(child), size_bytes=child_size))
            size += child_size
            count += 1
        else:
            records.append(dict(path=relative, type="directory"))
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity = dict(sha256=digest, size_bytes=size, file_count=count, is_directory=True)
require(before == fingerprint(), "Checkpoint changed during verification")
require(identity == {key: receipt[key] for key in identity}, "Checkpoint SHA-256/size verification failed")
require(receipt == read_receipt(receipt_path), "Resume receipt changed during verification")
# A run-level receipt does not identify its producer. Only a single historical
# training attempt plus that attempt's identical receipt proves that mapping.
producer = None
attempts = request["attempts"]
if len(attempts) == 1:
    item = attempts[0]
    job = item["job_id"]
    require(re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job), "Invalid training attempt job ID")
    if read_receipt(root / "attempts" / job / "checkpoints/latest.json") == receipt:
        producer = item["id"]
print(json.dumps({"present": True, "receipt": receipt, "identity": identity, "producer": producer}))
'''


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
            and str(item.get("slurm_reason") or "").startswith(SUBMISSION_UNKNOWN_PREFIX)
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
    target_dataset_id: str | None = None
    unseen_embodiment: bool = False
    suite_id: str
    environment: str | None = None
    tasks: list[str] = Field(default_factory=list)
    episodes_per_task: int = Field(default=DEFAULT_EVALUATION_EPISODES, ge=1, le=MAX_EVALUATION_EPISODES)
    seeds: list[int] = Field(default_factory=lambda: list(DEFAULT_EVALUATION_SEEDS))
    parallelism: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Concurrent evaluation workers; checked against the evaluator capability.",
    )
    headless: bool = True
    auto_resume: bool = True
    max_attempts: int = Field(default_factory=lambda: CLUSTER.defaults.max_attempts, ge=1, le=100)
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
    target_dataset_id: str | None = None
    unseen_embodiment: bool = False
    suite_id: str | None = None
    environment: str | None = None
    tasks: list[str] = Field(default_factory=list)
    episodes_per_task: int = Field(default=DEFAULT_EVALUATION_EPISODES, ge=1, le=MAX_EVALUATION_EPISODES)
    seeds: list[int] = Field(default_factory=lambda: list(DEFAULT_EVALUATION_SEEDS))
    parallelism: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Concurrent evaluation workers; checked against the evaluator capability.",
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
    """Grid axes from the frontend's JSON; SweepSpec supplies every other default."""
    if not raw:
        return {"strategy": "grid", "axes": {}}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"sweep definition must be JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise ValueError("sweep definition must be a JSON object")
    explicit_seeds = "seed" in parsed or "seeds" in parsed
    seeds = parsed.pop("seed", parsed.pop("seeds", None))
    if explicit_seeds and not isinstance(seeds, list):
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
        **({"seeds": seeds} if explicit_seeds else {}),
    }


class DataResourceCreateRequest(BaseModel):
    model_config = {"extra": "forbid"}

    category: Literal["dataset", "file"]
    provider: str = Field(min_length=1, max_length=128)
    namespace: str = Field(min_length=1, max_length=255)
    source_key: str = Field(min_length=1, max_length=255)
    display_name: str | None = Field(default=None, min_length=1)
    kind: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    description: str = Field(default="", max_length=4096)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str | None) -> str:
        if value is None or not value.strip():
            raise ValueError("display_name cannot be blank or null")
        return value.strip()

    @field_validator("provider", "namespace", "source_key", "kind")
    @classmethod
    def strip_identity(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("data resource identity values cannot be blank")
        return stripped

    @model_validator(mode="after")
    def validate_classification(self):
        validate_resource_type(self.category, self.kind)
        validate_resource_metadata(self.category, self.metadata)
        return self


class DataResourceEditRequest(BaseModel):
    model_config = {"extra": "forbid"}

    display_name: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, max_length=4096)
    metadata: dict[str, Any] | None = None
    archived: bool | None = None

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str | None) -> str:
        if value is None or not value.strip():
            raise ValueError("display_name cannot be blank or null")
        return value.strip()


class DatasetEditRequest(BaseModel):
    model_config = {"extra": "forbid"}

    display_name: str | None = Field(default=None, min_length=1, max_length=512)
    description: str | None = Field(default=None, max_length=4096)
    archived: bool | None = None

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value):
        if value is None or not value.strip():
            raise ValueError("display_name cannot be blank or null")
        return value.strip()


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
        self.storage = WorkspaceStorage(self.database)
        self.cluster = cluster or ClusterClient()
        if isinstance(self.cluster, ClusterClient) and self.database.workspace_id is not None:
            self.cluster = self.cluster.with_storage(self.storage)
        self.credential_store = credential_store or KeyringCredentialStore()
        self.credentials = session_credentials or SESSION_CREDENTIALS
        self.notifications = SlackNotifications(self.database, self.credential_store)
        self._credential_restore_lock = threading.Lock()
        self._tracking_connection_lock = self.database.operation_lock("tracking:" + (self.database.workspace_id or "system"))
        self._tracking_connection_revisions = dict.fromkeys(TRACKING_PROVIDERS, 0)
        self._credentials_restored = False
        self._evaluator_runtime_readiness_lock = threading.Lock()
        self._evaluator_runtime_readiness_cache: dict[
            tuple[str, str, str, str], tuple[float, dict[str, Any], list[str]]
        ] = {}
        self.source_discovery = SourceDiscovery(self.cluster)
        self.source_metadata = SourceMetadataStore(self.database)
        self._reconcile_lock = self.database.operation_lock("pipeline")
        self._reconcile_scan_lock = self.database.operation_lock("pipeline-scan")
        self._tracking_reconcile_lock = self.database.operation_lock("tracking-reconcile")
        self._tracking_delivery_lock = self.database.operation_lock("tracking-delivery")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracking_thread: threading.Thread | None = None
        self._tracking_delivery_thread: threading.Thread | None = None
        self._progress_refresh_lock = threading.Lock()
        self._progress_refresh_pending: dict[tuple[str, str], int] = {}
        self._progress_refresh_inflight: set[tuple[str, str]] = set()
        self._progress_refresh_due: dict[tuple[str, str], float] = {}
        self._progress_refresh_versions: dict[tuple[str, str], tuple[Any, ...]] = {}
        self._progress_refresh_workers = 0
        self._seed_registries()

    def _tracking_journal(self, capsule):
        from .tracking_journal import TrackingJournal
        path = Path(capsule)
        scope = path.name if re.fullmatch(r"[a-f0-9-]{36}", path.name) else str(path.resolve().relative_to(LOCAL_CAPSULE_ROOT.resolve()))
        return {"journal": TrackingJournal(self.database, scope)}

    def _wandb_bridge(self, capsule, settings=None):
        return WandBBridge(capsule, settings, **self._tracking_journal(capsule))

    def _mlflow_bridge(self, capsule, settings=None):
        return MLflowBridge(capsule, settings, **self._tracking_journal(capsule))

    @property
    def work_root(self) -> str:
        return self.storage.require_root()

    def _run_directory(self, run_id: str) -> str:
        return self.storage.run_directory(run_id)

    def _run_paths(self, run_id: str):
        return paths_for_root(self.storage.root_for_run(run_id))

    def _seed_registries(self) -> None:
        if self.database.workspace_id is not None:
            return
        with self.database.operation_lock("registry-seeding"):
            self._seed_registries_unlocked()

    def _seed_registries_unlocked(self) -> None:
        from .registry_policy import suite_key
        with self.database.connection() as connection:
            exclusions = {(row["kind"], row["seed_key"]) for row in connection.execute("SELECT kind,seed_key FROM registry_exclusions").fetchall()}
        for manifest in builtin_adapter_manifests():
            if ("adapter", manifest.slug) in exclusions:
                continue
            self.database.upsert_seed_adapter(
                seed_key=manifest.slug,
                name=manifest.display_name,
                manifest=canonical_adapter_manifest(manifest),
                description=manifest.description,
                repository_url=manifest.default_repository,
                materialize_result=False,
            )
        # The plain EgoVerse launcher is retired in favour of the native EgoVerse
        # manifests. Keep its seed archived; saved versions remain resolvable.
        for record in self.database.list_adapter_registry(include_archived=False, manifests="none"):
            if record.get("seed_key") == "egoverse":
                self.database.archive_adapter(str(record["id"]))
        for suite in get_evaluation_catalog():
            if ("suite", suite_key(suite.evaluator, suite.suite)) in exclusions:
                continue
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

    def prepare_background_restart(self) -> None:
        if any(thread and thread.is_alive() for thread in (self._thread, self._tracking_thread, self._tracking_delivery_thread)):
            raise RuntimeError("Previous pipeline workers have not stopped")
        self._stop.clear()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.prepare_background_restart()
        self._thread = threading.Thread(target=self._loop, name="skynet-reconciler", daemon=True)
        self._tracking_thread = threading.Thread(target=self._tracking_loop, name="skynet-tracking", daemon=True)
        self._thread.start()
        self._tracking_thread.start()
        self._tracking_delivery_thread = threading.Thread(target=self._tracking_delivery_loop, name="skynet-tracking-delivery", daemon=True)
        self._tracking_delivery_thread.start()

    def request_stop(self) -> None:
        self._stop.set()
        with self._progress_refresh_lock:
            self._progress_refresh_pending.clear()

    def stop(self) -> None:
        self.request_stop()
        for thread in (self._thread, self._tracking_thread, self._tracking_delivery_thread):
            if thread:
                thread.join()

    def _loop(self) -> None:
        while not self._stop.wait(BACKGROUND_POLL_INTERVAL_SECONDS):
            try:
                self.reconcile()
            except Exception:
                # The next pass retries after cluster/database recovery.
                logging.getLogger(__name__).exception("Reconciliation failed")

    def _tracking_loop(self) -> None:
        while not self._stop.wait(BACKGROUND_POLL_INTERVAL_SECONDS):
            try:
                self.reconcile_tracking()
            except Exception:
                logging.getLogger(__name__).exception("Tracking reconciliation failed")

    def _tracking_delivery_loop(self) -> None:
        while not self._stop.wait(BACKGROUND_POLL_INTERVAL_SECONDS):
            try:
                self.flush_tracking()
            except Exception:
                logging.getLogger(__name__).exception("Tracking delivery failed")

    def flush_tracking(self) -> dict[str, Any]:
        # Durable uploads must not wait for telemetry collection of every run.
        # This cross-process lock and each journal's cursor preserve exactly the
        # existing delivery/recovery path; no alternate W&B IDs are generated.
        if not self._tracking_delivery_lock.acquire(blocking=False):
            return {"ok": True, "skipped": "already running"}
        try:
            reports = {}
            for provider in TRACKING_PROVIDERS:
                if self._stop.is_set():
                    break
                reports[provider] = self._flush_tracking_provider(
                    provider, limit=TRACKING_FLUSH_RUN_LIMIT, event_limit=TRACKING_FLUSH_EVENT_LIMIT
                )
            return {"ok": True, "providers": reports}
        finally:
            self._tracking_delivery_lock.release()

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
        # Key matching reads the compact projections; the match below is re-read in full.
        for record in self.database.list_adapter_registry(include_archived=False, manifests="projected"):
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
        version = self.database.get_adapter_version(
            str(matches[0]["id"]), version_number=version_number,
        )
        if version is None:
            raise ValueError(f"adapter version was not found: {requested}@{version_number}")
        detail = {**matches[0], "selected_version": version, "versions": [version]}
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
            version = database.get_adapter_version(
                str(requested_adapter_id),
                version_id=str(requested_version_id) if requested_version_id else None,
                version_number=int(version_number) if version_number is not None else None,
            )
            if version is None:
                raise ValueError(
                    f"registered adapter version was not found: "
                    f"{requested_adapter_id}@{requested_version_id or version_number or 'latest'}"
                )
            record = {"id": version["adapter_id"]}
            manifest = AdapterManifest.model_validate(version["manifest"])
        else:
            record, version, manifest = self._adapter_selection(
                requested_key,
                int(version_number) if version_number is not None else None,
            )
            if requested_version_id and str(version.get("id")) != str(requested_version_id):
                selected = database.get_adapter_version(
                    record["id"], version_id=str(requested_version_id),
                    version_number=int(version_number) if version_number is not None else None,
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
        parent = run or self.database.get_run(str(evaluation.get("run_id") or ""), include_payloads=False)
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
        attempts = _stage_attempts(parent, evaluation.get("stage_id"))
        latest = _latest_attempt(attempts) or {}
        submission_failed = latest.get("status") == "SUBMISSION_FAILED" and not latest.get("slurm_job_id")
        execution_failed = bool(latest.get("slurm_job_id")) and latest.get("status") in ATTEMPT_FAILURE_STATES
        retryable = bool(
            stage and stage.get("status") == "FAILED"
            and evaluation.get("status") == "FAILED"
            and (submission_failed or execution_failed)
            and not any(item.get("status") in (EXECUTING_STATES |
                        (_PROGRESS_WAITING_STATES - TRANSIENT_STATES) | {"SUBMISSION_UNCONFIRMED"})
                        for item in attempts if item.get("id") != latest.get("id"))
        )
        # The job finished its episodes; only reading result.json failed.
        rereadable = bool(
            stage and stage.get("status") == "FAILED"
            and evaluation.get("status") == "FAILED"
            and latest.get("status") == "SUCCEEDED"
            and evaluation.get("result_path")
        )
        return {
            "cancel": manual_cancel_action(stage, attempts),
            "retry_submission": {
                "enabled": retryable,
                "label": "Retry evaluation" if execution_failed else "Retry submission",
                "reason": "" if retryable else "Only a failed evaluation with no active attempt can be retried.",
            },
            "reread_result": {
                "enabled": rereadable,
                "label": "Re-read result",
                "reason": "" if rereadable else
                "Only a failed evaluation whose Slurm job succeeded can re-read its result.",
            },
        }

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
            "model_io": resolve_model_io(spec.source.adapter_manifest, resolved_spec),
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
        resources = defaults.resources
        document: dict[str, Any] = {"train": {}}
        for path, attribute in FRONTEND_TRAIN_FIELDS.values():
            value = _manifest_default(manifest, attribute)
            if value is not None:
                PipelineService._set_missing_canonical_path(document, path, value)
        train: dict[str, Any] = document["train"]
        if defaults.hyperparameters.values:
            train["hyperparameters"] = copy.deepcopy(defaults.hyperparameters.values)

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
            mode="json", exclude_none=True,
            exclude={"enabled", "mlflow_tracking_uri", "mlflow_experiment"},
        )
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
    def _apply_training_preset(cls, canonical, manifest):
        from .training_contracts import selected_preset
        preset = selected_preset(canonical, manifest.train)
        if preset is None:
            return canonical
        explicit = set((canonical.get("intent") or {}).get("explicit_parameters") or [])
        cls._set_missing_canonical_path(canonical, "native.config.training_preset", preset.id)
        for path, value in preset.values.items():
            present, current = cls._manifest_input_lookup(canonical, path)
            if path.startswith("train."):
                if any(path == item or path.startswith(item + ".") for item in explicit):
                    continue
                parent = canonical
                parts = path.split(".")
                for part in parts[:-1]:
                    parent = parent.setdefault(part, {})
                parent[parts[-1]] = copy.deepcopy(value)
            elif not present or current is None:
                cls._set_missing_canonical_path(canonical, path, value)
        return canonical

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
        from .adapters.dataset_inputs import resolve_data_selections, validate_selection_sources
        from .training_contracts import data_contract_error

        cls._apply_training_preset(canonical, manifest)
        cls._apply_manifest_input_defaults(canonical, manifest)
        bundle = (canonical.get("data") or {}).get("bundle")
        assignments = bundle.get("assignments") if isinstance(bundle, Mapping) else None
        if not isinstance(assignments, list):
            return canonical
        fields = [field for field in manifest.train.input_fields if field.data_binding]
        if not fields:
            raise ValueError(
                "Dataset bundle selection is unsupported by this adapter version: "
                "it declares no training dataset binding. Remove the bundle and use "
                "the repository's dataset configuration, or select an adapter with "
                "a compatible training_data binding."
            )
        by_role = {}
        for assignment in assignments:
            if not isinstance(assignment, Mapping):
                raise ValueError("Dataset bundle assignments must be objects with a role and prepared version.")
            validate_mount_path(assignment.get("mount_path"))
            position = assignment.get("position", 0)
            if type(position) is not int or position < 0:
                raise ValueError("Dataset inputs require nonnegative integer positions")
            by_role.setdefault(str(assignment.get("role") or ""), []).append(assignment)
        consumed = {(field.data_binding.role, field.data_binding.position)
                    for field in fields if field.data_binding.cardinality == "one"}
        many_roles = {field.data_binding.role for field in fields if field.data_binding.cardinality == "many"}
        if many_roles & {role for role, _ in consumed}:
            raise ValueError("An adapter cannot declare both one and many inputs for the same role")
        bound_roles = many_roles | {role for role, _ in consumed}
        for role, selected in by_role.items():
            positions = [item.get("position", 0) for item in selected]
            if len(positions) != len(set(positions)):
                raise ValueError("Dataset bundle contains duplicate role positions; choose an unambiguous bundle")
            selected.sort(key=lambda item: item.get("position", 0))
            if role in many_roles and sorted(positions) != list(range(len(selected))):
                raise ValueError("Multiple dataset inputs require consecutive positions starting at zero")
            if role in bound_roles and role not in many_roles:
                for position in positions:
                    if (role, position) not in consumed:
                        raise ValueError(f"This adapter cannot consume {role} at position {position}. Choose a bundle with only the declared dataset inputs.")
            if role in bound_roles:
                validate_selection_sources([dict(
                    version_id=(item.get("version", {}).get("metadata") or {}).get("registered_version_id"),
                    manifest_sha256=item.get("version", {}).get("manifest_sha256"),
                    metadata=item.get("version", {}).get("metadata") or {},
                ) for item in selected])
        for field in fields:
            binding = field.data_binding
            present, value = cls._manifest_input_lookup(canonical, field.path)
            selected = by_role.get(binding.role, [])
            if binding.cardinality == "one":
                selected = [item for item in selected if item.get("position", 0) == binding.position]
            if not selected:
                raise ValueError(f"{field.path}: selected data bundle does not provide role {binding.role} at position {binding.position}")
            for assignment in selected:
                version = assignment.get("version")
                if not isinstance(version, Mapping):
                    raise ValueError(f"{field.path}: selected data bundle role {binding.role} has no version snapshot")
                location = (assignment.get("config") or {}).get("location", {})
                error = data_contract_error(binding, version.get("metadata") or {}, canonical)
                if error:
                    raise ValueError(f"{field.path}: {error}")
                location_ready = (location.get("status") == "AVAILABLE" and location.get("kind") == "cluster" and location.get("manifest_sha256") == version.get("manifest_sha256"))
                if binding.value_path in {"location.path", "selection"} and not location_ready:
                    raise ValueError("Choose a verified training-cluster copy of every dataset")
                if str(version.get("status") or "").upper() != "READY" and not location_ready:
                    raise ValueError(f"Selected {binding.role} is {version.get('status') or 'unverified'}, not ready on the cluster. Complete its import or transfer before training.")
                if (version.get("metadata") or {}).get("storage_location") == "workstation":
                    raise ValueError("This dataset is stored on the collection workstation. Transfer it to the training cluster and register that copy before submitting training.")
                if binding.formats and str(version.get("format") or "").casefold() not in {item.casefold() for item in binding.formats}:
                    raise ValueError(f"{field.path}: selected {binding.role} format {version.get('format') or 'undeclared'} is incompatible; accepted formats: {', '.join(binding.formats)}")
            if binding.value_path == "selection":
                selections = resolve_data_selections(canonical, binding.role, runtime=True)
                bound_value = selections if binding.cardinality == "many" else next(item for item in selections if item["position"] == binding.position)
                if present and value != bound_value:
                    # Normalize only an exact, previously derived full selection.
                    # Explicit changes to paths, hashes or metadata still fail.
                    complete = resolve_data_selections(canonical, binding.role)
                    previous = complete if binding.cardinality == "many" else next(item for item in complete if item["position"] == binding.position)
                    if value == previous:
                        parent = canonical
                        parts = field.path.split(".")
                        for part in parts[:-1]:
                            parent = parent[part]
                        parent[parts[-1]] = copy.deepcopy(bound_value)
                        value = bound_value
            else:
                assignment = selected[0]
                version = assignment["version"]
                location = (assignment.get("config") or {}).get("location", {})
                bound_value = {"version.path": version.get("path"), "mount_path": assignment.get("mount_path"),
                               "location.path": location.get("path"), "version.manifest_sha256": version.get("manifest_sha256")}[binding.value_path]
                if not isinstance(bound_value, str) or not bound_value:
                    raise ValueError(f"{field.path}: selected data bundle role {binding.role} does not provide {binding.value_path}")
            if present and value is not None and value != "" and value != [] and value != bound_value:
                raise ValueError(f"{field.path} conflicts with the selected dataset bundle. Clear the explicit dataset input to use the bundle.")
            if present and (value is None or value == "" or value == []):
                if field.path.startswith("native.overrides."):
                    canonical["native"]["overrides"].pop(field.path.removeprefix("native.overrides."))
                else:
                    parent = canonical
                    parts = field.path.split(".")
                    for part in parts[:-1]:
                        parent = parent[part]
                    parent.pop(parts[-1])
            cls._set_missing_canonical_path(canonical, field.path, bound_value)
        if "training_data" in many_roles:
            from .model_io import resolve_model_io
            io = resolve_model_io(manifest.model_dump(mode="json"), canonical)
            if io.get("compatible") is False:
                raise ValueError(io["note"])
        return canonical

    @classmethod
    def _validate_manifest_input_fields(
        cls, canonical: Mapping[str, Any], manifest: AdapterManifest
    ) -> None:
        if manifest.train.strict_native_config:
            declared = {field.path.removeprefix("native.config.").split(".")[0] for field in manifest.train.input_fields if field.path.startswith("native.config.")}
            unknown = set((canonical.get("native") or {}).get("config") or {}) - declared
            runner_checkpoint_keys = unknown & {"initial_checkpoint", "initial_checkpoint_mode"}
            if runner_checkpoint_keys:
                # These keys were previously rejected. Admit only a verified
                # full-state runner contract, without changing declared model
                # inputs or any non-strict adapter's existing behavior.
                native = canonical.get("native") or {}
                config = native.get("config") or {}
                path = config.get("initial_checkpoint")
                if (
                    not isinstance(path, str) or not path.strip() or path != path.strip()
                    or not PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
                    or any(character in path for character in ("\x00", "\n", "\r"))
                ):
                    raise ValueError("Initial checkpoint requires a nonempty absolute path")
                if config.get("initial_checkpoint_mode", "resume") != "resume":
                    raise ValueError("This adapter does not declare this checkpoint mode; use its declared weights-only initialization settings")
                if not manifest.capabilities.supports_resume:
                    raise ValueError("This adapter does not support full-state checkpoint resume")
                policy = (canonical.get("train") or {}).get("checkpoint") or {}
                default_auto_resume = manifest.defaults.checkpoint.auto_resume
                if policy.get("auto_resume", True if default_auto_resume is None else default_auto_resume) is not True:
                    raise ValueError("Full-state checkpoint resume requires auto_resume=true; otherwise the runner would ignore it")
                resume_argv = native.get("resume_argv") or manifest.train.resume_argv
                if not resume_argv and manifest.legacy_handler:
                    from .adapters import ManifestAdapter
                    resume_argv = ManifestAdapter(manifest).resolve(
                        ExperimentSpec.model_validate(canonical)
                    ).resume_argv
                if not isinstance(resume_argv, (list, tuple)) or not any(
                    isinstance(argument, str) and any(token in argument for token in (
                        "{{tokens.resume_checkpoint}}", "{{SKYNET_RESUME_CHECKPOINT}}",
                    )) for argument in resume_argv
                ):
                    raise ValueError("Full-state checkpoint resume requires resume_argv that consumes the checkpoint path")
                unknown -= runner_checkpoint_keys
            if (canonical.get("native") or {}).get("overrides"):
                raise ValueError("This adapter only accepts its declared training settings")
            if unknown:
                raise ValueError("Unsupported adapter settings: " + ", ".join(sorted(unknown)))
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
            if field.minimum is not None and value < field.minimum:
                raise ValueError(f"{field.label}: minimum is {field.minimum:g}")
            if field.maximum is not None and value > field.maximum:
                raise ValueError(f"{field.label}: maximum is {field.maximum:g}")
            if field.maximum_path:
                _, limit = cls._manifest_input_lookup(canonical, field.maximum_path)
                if limit is not None and value > limit:
                    raise ValueError(f"{field.label} cannot exceed {field.maximum_path}")
            if field.choices:
                allowed = {canonical_sha256(choice) for choice in field.choices}
                if field.path == "native.config.training_preset" and manifest.train.presets:
                    allowed.add(canonical_sha256("custom"))
                if canonical_sha256(value) not in allowed:
                    raise ValueError(
                        f"{field.path}: value is not one of the declared choices"
                    )

    @classmethod
    def _validate_sweep_inputs(cls, spec: ExperimentSpec, manifest: AdapterManifest) -> ExperimentSpec:
        from .recording_sampling import experiment_sampling
        for variant in expand_sweep(spec):
            document = variant.resolved_spec.model_dump(mode="python")
            cls._apply_manifest_data_bindings(document, manifest)
            cls._validate_manifest_input_fields(
                document, manifest
            )
            experiment_sampling(document, manifest)
        return spec

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
        for field, (_, attribute) in FRONTEND_TRAIN_FIELDS.items():
            holder = _frontend_train_field_holder(field, document, hyper)
            value = _manifest_default(manifest, attribute)
            if value is not None and (
                field not in holder or _frontend_parameter_is_unset(holder[field])
            ):
                holder[field] = value
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

        tracking = document.setdefault("tracking", {})
        for field, value in {
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
        runtime_input = copy.deepcopy(runtime_input)
        runtime_input["environment"] = {
            **CLUSTER.training_environment,
            **(runtime_input.get("environment") or {}),
        }
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
                if not runtime.get("profile"):
                    runtime["profile"] = "default"
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
        selections = payload.get("data_selections")
        if selections is not None:
            if payload.get("data_bundle_id") or data.get("bundle") or data.get("bundle_id"):
                raise ValueError("Choose direct data inputs or an existing input snapshot, not both")
            data["bundle"] = data_selection.snapshot(self.database, selections)
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
            canonical.pop("data_selections", None)
            canonical.setdefault("apiVersion", SPEC_API_VERSION)
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
            return self._validate_sweep_inputs(ExperimentSpec.model_validate(canonical), manifest)

        requested_hyper = dict(payload.get("hyperparameters") or {})
        explicit_parameters = {
            path
            for field, (path, _) in FRONTEND_TRAIN_FIELDS.items()
            if not _frontend_parameter_is_unset(
                _frontend_train_field_holder(field, payload, requested_hyper).get(field)
            )
        }
        if any(
            field not in FRONTEND_HYPERPARAMETER_FIELDS
            and not _frontend_parameter_is_unset(value)
            for field, value in requested_hyper.items()
        ):
            explicit_parameters.add("train.hyperparameters")

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
                if key not in FRONTEND_HYPERPARAMETER_FIELDS
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
        provider_inputs = copy.deepcopy(tracking_input.get("providers") or [])
        if not isinstance(provider_inputs, list):
            raise ValueError("tracking.providers must be a list")
        providers: list[dict[str, Any]] = []
        for provider_input in provider_inputs:
            if not isinstance(provider_input, Mapping):
                raise ValueError("each tracking provider must be an object")
            provider_name = str(provider_input.get("provider") or "").lower()
            if provider_name not in TRACKING_PROVIDERS:
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
        native_tracking = tracking_input.get("native_tracking", "preserve")
        if isinstance(native_tracking, bool):
            native_tracking = "preserve" if native_tracking else "disable"
        tracking = {
            "providers": providers,
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
                    "profile": config.get("default_profile", DEFAULT_EVALUATION_PROFILE),
                })

        canonical = {
            "apiVersion": SPEC_API_VERSION,
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
        return self._validate_sweep_inputs(ExperimentSpec.model_validate(canonical), manifest)

    @staticmethod
    def _project(database: Database, name: str) -> dict[str, Any]:
        existing = next((item for item in database.list_projects() if item["name"] == name), None)
        if existing:
            return existing
        try:
            return database.create_project(name)
        except INTEGRITY_ERRORS:
            return next(item for item in database.list_projects() if item["name"] == name)

    def _materialize_experiment_revision(
        self, revision_id: str, variants: list[Any]
    ) -> None:
        work_root = self.work_root
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
                run_directory=f"{work_root}/jobs/runs/pending",
                status="DRAFT",
            )
            self.database.update_run(
                run["id"], run_directory=f"{work_root}/jobs/runs/{run['id']}"
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
            signature = self._auto_queue_signature(spec, plan, requested_gateway)
            if signature in checked:
                continue
            self._auto_queue(
                spec,
                plan,
                requested_gateway,
                record_snapshot=False,
            )
            checked.add(signature)

    @staticmethod
    def _auto_queue_signature(spec: ExperimentSpec, plan: Any, gateway: str) -> tuple[str, str, int, str]:
        """What the live quota probe depends on; equal signatures resolve to the same queue."""
        return (gateway, spec.resources.gpu.gpu_type, resolve_gpu_count(spec, plan), spec.resources.time_limit)

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
            except INTEGRITY_ERRORS as error:
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
            experiment_revision_id=revision["id"], limit=10000,
        ) if revision else []
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
        from .recording_sampling import experiment_sampling
        spec = self.normalize_spec(payload)
        self._validate_tracking_requirements(spec)
        variants = expand_sweep(spec)
        scripts: list[str] = []
        warnings: list[str] = []
        blockers: list[dict[str, Any]] = []
        argument_validations: list[dict[str, Any]] = []
        sampling_summaries = []
        queue_choices: dict[tuple[str, str, int, str], tuple[str, str]] = {}
        for variant in variants:
            resolved = variant.resolved_spec
            sampling = experiment_sampling(resolved.model_dump(mode="python"))
            if sampling:
                sampling_summaries.append(dict(variant=variant.name, **sampling))
                excluded = sum(len(value["excluded_episodes"]) for value in sampling["splits"].values())
                if excluded:
                    warnings.append(f"{variant.name}: {excluded} recording(s) are too short for the selected frequency and action chunk and will be excluded.")
            plan = resolve_adapter_plan(resolved)
            if not plan.blockers:
                # One live quota probe per distinct queue signature across the sweep.
                signature = self._auto_queue_signature(resolved, plan, resolved.resources.gateway)
                if resolved.resources.queue_policy == "auto" and signature in queue_choices:
                    payload = resolved.model_dump(mode="json", by_alias=True)
                    payload["resources"]["account"], payload["resources"]["partition"] = queue_choices[signature]
                    resolved = ExperimentSpec.model_validate(payload)
                else:
                    resolved, _ = self._auto_queue(resolved, plan, resolved.resources.gateway, record_snapshot=False)
                    if resolved.resources.queue_policy == "auto":
                        queue_choices[signature] = (resolved.resources.account, resolved.resources.partition)
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
            compiled = compile_sbatch(resolved, plan, run_id=f"preview-{variant.index:03d}", work_root=self.work_root)
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
            "recording_sampling": sampling_summaries,
            "resolved_revision": spec.source.revision,
        }

    def _materialize_cleaned_revision(
        self,
        revision: Mapping[str, Any],
        variants: list[dict[str, Any]],
    ) -> None:
        """Rebuild draft runs from preserved variants after an explicit Submit."""
        planned_runs = []
        for variant in variants:
            spec = ExperimentSpec.model_validate(variant["resolved_spec_json"])
            plan = resolve_adapter_plan(spec)
            planned_runs.append(
                {
                    "variant_id": variant["id"],
                    "resolved_spec_sha256": variant["resolved_spec_sha256"],
                    "seed": spec.train.seed,
                    "adapter_name": spec.source.adapter,
                    "adapter_version": str(spec.source.adapter_version),
                    "source_commit": spec.source.revision,
                    "runtime_profile": spec.runtime.profile,
                    "resolved_config": {
                        "spec": spec.model_dump(mode="json", by_alias=True),
                        "plan": plan.model_dump(mode="json"),
                        "blockers": plan.blockers,
                    },
                    "auto_resume": spec.train.checkpoint.auto_resume,
                    "max_attempts": spec.train.checkpoint.max_attempts,
                }
            )
        self.database.materialize_empty_revision_runs(
            revision["id"], planned_runs, run_directory_root=f"{self.work_root}/jobs/runs"
        )

    def submit_experiment(self, experiment_id: str, gateway: str = "auto") -> dict[str, Any]:
        with self._reconcile_lock:
            detail = self.experiment_detail(experiment_id)
            revision = detail["latest_revision"]
            if not revision:
                raise ValueError("experiment has no revision to submit")
            data_selection.assert_available(self.database, revision["requested_spec_json"])
            pre_submission_repairs = self.database.repair_pre_submission_orphans(
                experiment_id
            )
            if pre_submission_repairs.get("revision_ids"):
                detail = self.experiment_detail(experiment_id)
                revision = detail["latest_revision"]
            revision_locked = bool(revision.get("submitted_at"))
            self._validate_tracking_requirements(
                ExperimentSpec.model_validate(revision["requested_spec_json"])
            )
            if not detail["runs"]:
                self._materialize_cleaned_revision(revision, detail["variants"])
                detail = self.experiment_detail(experiment_id)
            # Selection needs lifecycle state and specs, not stage documents or snapshots.
            run_details_by_id = self.database.list_run_details([run["id"] for run in detail["runs"]])
            candidates: list[tuple[dict[str, Any], dict[str, Any], str]] = []
            for run in detail["runs"]:
                run_detail = run_details_by_id.get(run["id"])
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

            candidate_specs: list[ExperimentSpec] = []
            candidate_run_ids: set[str] = set()
            for run_detail, _, _ in candidates:
                if run_detail["id"] in candidate_run_ids:
                    continue
                candidate_specs.append(
                    ExperimentSpec.model_validate(run_detail["resolved_spec_json"])
                )
                candidate_run_ids.add(run_detail["id"])
            from .recording_sampling import experiment_sampling
            for candidate in candidate_specs:
                experiment_sampling(candidate.model_dump(mode="python"))
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
        guaranteed = CLUSTER.guaranteed_queues()
        if not guaranteed:
            raise ValueError("auto queue selection requires a configured non-preemptible queue")
        normal_name, normal = guaranteed[0]
        overflow_selection = CLUSTER.preemptible_queue()
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
            host, output = self.cluster.run_with_fallback(command, gateway, timeout=GPU_USAGE_QUERY_TIMEOUT_SECONDS)
            requested_type = resolve_gpu_type(spec, plan)
            requested_columns = (CLUSTER.dashboard.gpu_usage_columns if requested_type == "any"
                                 else [requested_type])
            quotas = account_gpu_quota(output, normal.account, requested_columns,
                                       fallback_columns=CLUSTER.dashboard.gpu_usage_columns)
            idle_receipt = None
            if quotas is None:
                import inspect
                program = inspect.getsource(idle_partition_quota) + (
                    "\nimport json,sys\ntry:\n result=idle_partition_quota(sys.argv[1], sys.argv[2])"
                    "\nexcept Exception as error:\n result={'error':str(error)}\nprint(json.dumps(result))"
                )
                idle_command = (
                    f"export PATH={shlex.quote(SLURM_BIN)}:${{PATH:-}}; LC_ALL=C python3 -c "
                    + shlex.quote(program) + " " + shlex.quote(normal.account) + " " + shlex.quote(normal.partition)
                )
                host, raw = self.cluster.run_with_fallback(idle_command, host, timeout=IDLE_QUOTA_VERIFICATION_TIMEOUT_SECONDS)
                idle_receipt = json.loads(raw)
                if not isinstance(idle_receipt, dict):
                    raise ValueError("Slurm returned an invalid quota verification response")
                if idle_receipt.get("error"):
                    raise ValueError(str(idle_receipt["error"]))
                if (idle_receipt.get("account") != normal.account
                        or idle_receipt.get("partition") != normal.partition
                        or idle_receipt.get("active_allocations") != 0):
                    raise ValueError("Slurm did not confirm an idle normal account")
                limits = idle_receipt.get("limits") or {}
                quotas = {}
                for gpu_type in requested_columns:
                    candidates = [limits[key] for key in ("gpu", "gpu:" + gpu_type) if key in limits]
                    if not candidates or any(type(value) is not int or value < 0 for value in candidates):
                        raise ValueError(f"Slurm did not return an explicit GPU limit for {gpu_type}")
                    quotas[gpu_type] = (0, min(candidates))
            available = False
            for usage, limit in quotas.values():
                available = available or usage + gpu_count <= limit
            if not available:
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
                        "idle_quota_verification": idle_receipt,
                    },
                    nodes=[],
                    queue=[],
                )
                snapshot_id = snapshot["id"]
        except (ClusterError, ValueError) as error:
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
        with self.database.connection() as connection:
            candidate = connection.execute(
                f"""SELECT s.id FROM workflow_stages s
                JOIN runs r ON r.id=s.run_id JOIN variants v ON v.id=r.variant_id
                JOIN experiment_revisions er ON er.id=v.experiment_revision_id
                WHERE er.experiment_id=? AND {visible_sql('workflow_stages', 's')}
                  AND er.id=COALESCE(?, (SELECT id FROM experiment_revisions
                    WHERE experiment_id=? ORDER BY revision_number DESC LIMIT 1))
                  AND s.status IN ('DRAFT','PENDING','RETRY_PENDING') LIMIT 1""",
                (experiment_id, experiment_revision_id, experiment_id),
            ).fetchone()
        if candidate is None:
            return []
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
                f"""
                SELECT COUNT(*) FROM workflow_stages s
                JOIN runs r ON r.id = s.run_id
                JOIN variants v ON v.id = r.variant_id
                JOIN experiment_revisions er ON er.id = v.experiment_revision_id
                WHERE er.experiment_id = ? AND er.id = ?
                  AND s.status IN ({sql_list(ACTIVE_STAGE_STATES)})
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

    def _save_submission_script(self, run_id: str, attempt_id: str, script: str) -> dict[str, Any]:
        """Keep the exact transport payload independently of later run capsules."""
        content = script.encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        existing = next((item for item in self.database.list_artifacts(run_id, artifact_type="SUBMISSION_SCRIPT")
                         if item.get("metadata_json", {}).get("attempt_id") == attempt_id), None)
        if existing:
            if existing.get("sha256") != digest:
                raise ValueError("The saved submission checksum differs from this attempt")
            return existing
        path = MetadataObjects(self.database).put(content, name=f"{attempt_id}.sbatch")
        location = "cluster_objects"
        return self.database.create_artifact(run_id, artifact_type="SUBMISSION_SCRIPT", path=str(path),
            sha256=digest, size_bytes=len(content), retention_policy="preserve", metadata={"attempt_id": attempt_id, "location": location})

    def recover_run_submission(self, run_id: str, gateway: str) -> dict[str, Any]:
        with self._reconcile_lock:
            run = self.database.get_run(run_id)
            if not run:
                raise KeyError("Run not found")
            stage = next((item for item in reversed(run["stages"]) if item["stage_type"] == "TRAIN"), None)
            attempts = [item for item in run["attempts"] if stage and item["stage_id"] == stage["id"]]
            attempt = max(attempts, key=lambda item: item["attempt_number"], default=None)
            if attempt and attempt.get("slurm_job_id"):
                self._repair_recovered_submission_tracking({**attempt, "run_id": run_id, "stage_type": "TRAIN"})
                return {"run_id": run_id, "status": run["status"], "slurm_job_id": attempt["slurm_job_id"]}
            if not attempt or stage["status"] != "SUBMITTING" or attempt["status"] != "SUBMITTING" or not str(attempt.get("slurm_reason") or "").startswith(SUBMISSION_UNKNOWN_PREFIX):
                raise ValueError("Only an unconfirmed training submission can be recovered. Cancelled submissions cannot be restarted here.")
            selected_gateway = self.cluster.resolve_gateway(gateway) if gateway == "auto" else self.cluster.candidates(gateway)[0]
            self.database.update_job_attempt(attempt["id"], gateway=selected_gateway)
            try:
                submission = self.cluster.recover_submission(run_id, attempt["id"], selected_gateway)
                if submission is None:
                    artifact = next((item for item in self.database.list_artifacts(run_id, artifact_type="SUBMISSION_SCRIPT")
                                     if item.get("metadata_json", {}).get("attempt_id") == attempt["id"]
                                     and item.get("metadata_json", {}).get("location") == "cluster_objects"), None)
                    if not artifact:
                        raise ValueError("No accepted job was found and this older attempt has no saved submission script. Its original script must be restored before upload recovery.")
                    content = MetadataObjects(self.database).read(artifact["path"], artifact["sha256"])
                    if hashlib.sha256(content).hexdigest() != artifact["sha256"]:
                        raise ValueError("Saved submission script changed; restore the original before recovery")
                    submission = self.cluster.submit_script(content.decode("utf-8"), run_id, selected_gateway,
                                                            submission_key=attempt["id"])
            except (ClusterError, OSError, ValueError) as error:
                self.database.update_job_attempt(attempt["id"], slurm_reason=f"{SUBMISSION_UNKNOWN_PREFIX}: recovery did not complete: {error}")
                raise
            self._record_recovered_submission({**attempt, "stage_id": stage["id"], "stage_type": "TRAIN",
                "stage_status": stage["status"], "run_id": run_id, "run_started_at": run.get("started_at"), "evaluation_id": None}, submission)
            return {"run_id": run_id, "status": self.database.get_run(run_id)["status"], "slurm_job_id": submission.job_id, "gateway": submission.gateway}

    def _transition_stage(
        self,
        *,
        stage_id: str,
        stage_type: str | None,
        run_id: str | None,
        evaluation_id: str | None,
        status: str,
        stage_status: str | None = None,
        completed_at: Any = _UNSET,
        stage_updates: Mapping[str, Any] | None = None,
        run_updates: Mapping[str, Any] | None = None,
        evaluation_updates: Mapping[str, Any] | None = None,
        attempt_id: str | None = None,
        attempt_updates: Mapping[str, Any] | None = None,
        event_type: str | None = None,
        old_status: str | None = None,
        details: Mapping[str, Any] | None = None,
        expected_stage_statuses: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Move a stage and the entity that owns it to ``status`` in one commit.

        An evaluation stage carries its evaluation and leaves the run untouched;
        every other stage carries its run. ``completed_at`` is written only when
        given, so a transition that must keep a completion time omits it. The
        event, when ``event_type`` is set, names the evaluation if there is one
        and the run otherwise.
        """
        is_evaluation = str(stage_type or "").upper() == "EVALUATE"
        lifecycle: dict[str, Any] = {"status": status}
        if completed_at is not _UNSET:
            lifecycle["completed_at"] = completed_at
        event = None
        if event_type:
            event = {
                "entity_type": "evaluation" if evaluation_id else "run",
                "entity_id": evaluation_id if evaluation_id else run_id,
                "event_type": event_type,
                "old_status": old_status,
                "new_status": status,
                "details": details or {},
            }
        return self.database.transition_workflow_state(
            stage_id=stage_id,
            stage_updates={**lifecycle, "status": stage_status or status, **(stage_updates or {})},
            attempt_id=attempt_id,
            attempt_updates=attempt_updates,
            run_id=None if is_evaluation else run_id,
            run_updates=None if is_evaluation else {**lifecycle, **(run_updates or {})},
            evaluation_id=evaluation_id,
            evaluation_updates={**lifecycle, **(evaluation_updates or {})} if evaluation_id else None,
            event=event,
            expected_stage_statuses=expected_stage_statuses,
        )

    def _block_stage(
        self,
        *,
        stage_id: str,
        stage_type: str | None,
        run_id: str,
        evaluation_id: str | None,
        event_type: str,
        details: Mapping[str, Any],
        blockers: list[str],
    ) -> dict[str, Any]:
        self._transition_stage(
            stage_id=stage_id,
            stage_type=stage_type,
            run_id=run_id,
            evaluation_id=evaluation_id,
            status="BLOCKED",
            completed_at=utc_now(),
            event_type=event_type,
            details=details,
        )
        return {"run_id": run_id, "stage_id": stage_id, "status": "BLOCKED", "blockers": blockers}

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
        return self._transition_stage(
            stage_id=str(stage["id"]),
            stage_type=stage.get("stage_type"),
            run_id=run_id,
            evaluation_id=str(evaluation["id"]) if evaluation else None,
            status="CANCELLING",
            completed_at=None,
            attempt_id=attempt_id,
            attempt_updates=attempt_updates,
            event_type=event_type,
            old_status="CANCELLING",
            details={**dict(details), "cancellation_requested": True},
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
        return self._transition_stage(
            stage_id=str(stage["id"]),
            stage_type=stage.get("stage_type"),
            run_id=run_id,
            evaluation_id=str(evaluation["id"]) if evaluation else None,
            status="CANCELLED",
            completed_at=completed,
            attempt_id=attempt_id,
            attempt_updates={
                "status": "CANCELLED",
                "slurm_reason": f"Cancelled before Slurm accepted the job: {error}",
                "finished_at": completed,
            },
            event_type="CANCELLED_BEFORE_SUBMISSION",
            old_status="CANCELLING",
            details={"attempt_id": attempt_id, "submission_error": str(error)},
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
        run = self.database.get_run(run_id, execution_stage_ids=[stage_id])
        if not run:
            raise KeyError("Run not found")
        stage = next(item for item in run["stages"] if item["id"] == stage_id)
        data_selection.assert_available(self.database, run["resolved_spec_json"],
                                        stage["resolved_config_json"], pinned_execution)
        is_evaluation_stage = stage["stage_type"] == "EVALUATE"
        evaluation = next(
            (item for item in run["evaluations"] if item.get("stage_id") == stage_id),
            None,
        )
        stage_entity = {
            "stage_id": stage_id,
            "stage_type": stage["stage_type"],
            "run_id": run_id,
            "evaluation_id": evaluation["id"] if evaluation else None,
        }
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
                return self._block_stage(
                    **stage_entity,
                    event_type="EVALUATION_PLAN_BLOCKED",
                    details={"error": message},
                    blockers=[message],
                )
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
        if not is_evaluation_stage:
            from .recording_sampling import experiment_sampling
            experiment_sampling(spec.model_dump(mode="python"))
        from .adapters.egoverse_models import execution_compatibility_error
        compatibility_error = execution_compatibility_error(spec.source.model_dump(mode="python"), spec.native.config)
        if compatibility_error:
            if plan is None:
                from .adapters import AdapterCapabilities
                plan = AdapterPlan(
                    adapter=spec.source.adapter, adapter_version=spec.source.adapter_version,
                    argv=[], capabilities=AdapterCapabilities(
                        name=spec.source.adapter, runtime_backends={"existing"},
                        supports_multi_gpu_single_node=False, supports_resume=False,
                    ),
                )
            if compatibility_error not in plan.blockers:
                plan.blockers.append(compatibility_error)
        if plan is None:
            plan = resolve_adapter_plan(spec)
        policy_resources = spec.resources.with_gpu_cpu_policy(resolve_gpu_count(spec, plan))
        if policy_resources is not spec.resources:
            execution_provenance["transformations"].append({
                "kind": "cluster_cpu_per_gpu_policy",
                "before": spec.resources.cpus_per_task,
                "after": policy_resources.cpus_per_task,
            })
            spec = spec.model_copy(update={"resources": policy_resources})
        if is_evaluation_stage:
            # Reapply current cluster safety policy to historical pinned plans,
            # including automatic ledger resumes, before snapshots or submission.
            evaluation_context = copy.deepcopy(stage["resolved_config_json"].get("context") or {})
            if not evaluation_context:
                evaluation_context = copy.deepcopy(plan.native_config.get("canonical_evaluation") or {})
            if evaluation_context.get("worker_resources"):
                evaluation_context["worker_resources"]["cpus_per_task"] = CLUSTER.defaults.cpus_per_gpu
            evaluation_context.setdefault("environment", stage["resolved_config_json"].get("environment"))
            plan.native_config["canonical_evaluation"] = evaluation_context
            try:
                placed_resources = resolve_evaluation_resources(
                    spec.resources, evaluation_context,
                    runtime_profile_id=plan.native_config.get("evaluation_runtime_profile_id"),
                    runtime=spec.runtime.model_dump(mode="json"),
                    gpu_count=resolve_gpu_count(spec, plan), gpu_type=resolve_gpu_type(spec, plan),
                )
                if placed_resources != spec.resources:
                    execution_provenance["transformations"].append({
                        "kind": "isaac_evaluation_placement",
                        "before": spec.resources.model_dump(mode="json", by_alias=True),
                        "after": placed_resources.model_dump(mode="json", by_alias=True),
                    })
                if placed_resources is not spec.resources:
                    spec = spec.model_copy(update={"resources": placed_resources})
                    plan.resolved_gpu_type = placed_resources.gpu.gpu_type
            except ValueError as error:
                plan.blockers.append(str(error))
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
            return self._block_stage(
                **stage_entity,
                event_type="ADAPTER_BLOCKED",
                details={"blockers": plan.blockers, "todos": plan.todos},
                blockers=plan.blockers,
            )
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
        attempt_snapshot["storage"] = {
            "work_root": self.storage.root_for_run(run_id),
            "run_directory": evaluation_execution_directory(
                self._run_directory(run_id),
                evaluation_context.get("execution_key") if is_evaluation_stage else None,
            ),
        }
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
            capsule_files["adapter-support/evaluation-context.json"] = evaluation_context_json(evaluation_context)
        capsule_files["attempt-snapshot.json"] = json.dumps(
            attempt_snapshot, separators=(",", ":"), sort_keys=True
        ) + "\n"
        try:
            compiled = compile_sbatch(
                spec,
                plan,
                run_id=run_id,
                work_root=self.storage.root_for_run(run_id),
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
            return self._block_stage(
                **stage_entity,
                event_type="SBATCH_COMPILE_BLOCKED",
                details={"error": str(error)},
                blockers=[str(error)],
            )

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
            sbatch_path=f"{self._run_directory(run_id)}/job.sbatch",
            stdout_path=compiled.stdout_path_template,
            stderr_path=compiled.stderr_path_template,
        )
        if attempt is None:
            current = next(item for item in self.database.list_stages(run_id, include_payloads=False)
                           if item["id"] == stage_id)
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
                    "",
                    attempt_snapshot=attempt_snapshot,
                    continuation_attempt_number=(
                        int(attempt["attempt_number"])
                        if manual_mode in {"resume", "replay_initial"}
                        else None
                    ),
                )
                self._require_native_tracking_bindings(run_id, native_providers)
            self._save_submission_script(run_id, attempt["id"], compiled.script)
            # Keep scheduler scripts small regardless of dataset count. Files
            # are pinned by their checksum manifest and verified before sbatch.
            secret_gateway, _ = self.cluster.write_capsule_files(
                run_id, compiled.upload_files, selected_gateway, immutable=True,
            )
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
            transition = self._transition_stage(
                **stage_entity,
                status="SUBMITTING",
                completed_at=None,
                attempt_id=attempt["id"],
                attempt_updates={
                    "status": "SUBMITTING",
                    "gateway": test_gateway,
                    "slurm_reason": f"{SUBMISSION_UNKNOWN_PREFIX}: {error}",
                },
                event_type="SUBMISSION_OUTCOME_UNKNOWN",
                details={"error": str(error), "attempt_id": attempt["id"]},
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
                            "slurm_reason": f"{SUBMISSION_UNKNOWN_PREFIX} during cancellation: {error}",
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
            transition = self._transition_stage(
                **stage_entity,
                status="FAILED",
                completed_at=completed,
                attempt_id=attempt["id"],
                attempt_updates={
                    "status": "SUBMISSION_FAILED",
                    "slurm_reason": str(error),
                    "finished_at": completed,
                },
                event_type="SUBMISSION_FAILED",
                details={"error": str(error), "attempt_id": attempt["id"]},
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
            compiled.stdout_path_template, submission.job_id, logs_root=self._run_paths(run_id).logs
        )
        stderr = resolve_slurm_log_path(
            compiled.stderr_path_template, submission.job_id, logs_root=self._run_paths(run_id).logs
        )
        if stdout is None or stderr is None:
            raise RuntimeError("compiled Slurm log path templates could not be resolved")
        attempt_directory = f"{submission.run_directory}/attempts/{submission.job_id}"
        archived_script_path = f"{attempt_directory}/job.sbatch"
        submitted_at = utc_now()
        submission_transition = self._transition_stage(
            **stage_entity,
            status="SUBMITTED",
            stage_updates={"started_at": submitted_at},
            run_updates={"started_at": run.get("started_at") or submitted_at},
            evaluation_updates={"started_at": submitted_at, "completed_at": None},
            attempt_id=attempt["id"],
            attempt_updates={
                "status": "SUBMITTED",
                "slurm_job_id": submission.job_id,
                "gateway": submission.gateway,
                "sbatch_path": archived_script_path,
                "stdout_path": stdout,
                "stderr_path": stderr,
                "submitted_at": submitted_at,
                "slurm_reason": test_output.strip()[:SUBMISSION_OUTPUT_LIMIT],
            },
            event_type="JOB_SUBMITTED",
            old_status=evaluation["status"] if evaluation else run["status"],
            details={
                "job_id": submission.job_id,
                "gateway": submission.gateway,
                "attempt": attempt["attempt_number"],
                "recovered": submission.recovered,
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
                    "slurm_reason": test_output.strip()[:SUBMISSION_OUTPUT_LIMIT],
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
                schema_version=SPEC_API_VERSION,
                path=f"{attempt_directory}/resolved-spec.json",
                sha256=compiled.spec_sha256,
            )
        except INTEGRITY_ERRORS:
            pass
        if not is_evaluation_stage:
            self._start_tracking(
                spec,
                run,
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

    def _tracking_environment(self):
        return os.environ if self.database.workspace_id in (None, LEGACY_WORKSPACE) else {}

    def _environment_credentials(self, provider: str) -> dict[str, str]:
        if provider == "wandb":
            values = {"api_key": self._tracking_environment().get("WANDB_API_KEY")}
        else:
            values = {
                "token": self._tracking_environment().get("MLFLOW_TRACKING_TOKEN"),
                "username": self._tracking_environment().get("MLFLOW_TRACKING_USERNAME"),
                "password": self._tracking_environment().get("MLFLOW_TRACKING_PASSWORD"),
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

    def _environment_credentials_for_endpoint(self, provider: str, endpoint: str | None) -> dict[str, str]:
        settings = WandBSettings.from_env(self._tracking_environment()) if provider == "wandb" else TrackingSettings.from_env(self._tracking_environment())
        expected = settings.base_url if provider == "wandb" else settings.tracking_uri
        if not expected or not endpoint or endpoint.rstrip("/") != expected.rstrip("/"):
            return {}
        return self._credentials_for_mode(provider, self._environment_credentials(provider))

    @classmethod
    def _credentials_for_mode(cls, provider: str, credentials: Mapping[str, str]) -> dict[str, str]:
        mode = cls._credential_authentication(provider, credentials)
        fields = {"api_key": ("api_key",), "token": ("token",), "basic": ("username", "password")}.get(mode, ())
        return {key: credentials[key] for key in fields if credentials.get(key)}

    def _bound_tracking_credentials(
        self, provider: str, endpoint: str | None, *,
        connections: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, str], str | None]:
        with self._tracking_connection_lock:
            connection = (connections.get(provider) if connections is not None
                          else self.database.get_tracking_connection(provider)) or {}
            config = connection.get("config_json") or {}
            if config.get("authentication") == "none":
                return {}, None
            current = self.credentials.get(provider)
            # An endpoint lives with the in-memory secret. A failed database write
            # must never attach that secret to the previous connection's host.
            bound_endpoint = self.credentials.endpoint(provider) or connection.get("endpoint")
            if config.get("credential_source") != "environment" and current and endpoint and bound_endpoint and endpoint.rstrip("/") == str(bound_endpoint).rstrip("/"):
                mode = self._credential_authentication(provider, current)
                if config.get("authentication") in (None, mode):
                    return self._credentials_for_mode(provider, current), self.credentials.source(provider)
            if config.get("credential_source") not in (None, "environment"):
                return {}, None
            environment = self._environment_credentials_for_endpoint(provider, endpoint)
            if config.get("authentication") not in (None, self._credential_authentication(provider, environment)):
                return {}, None
            return environment, "environment" if environment else None

    def _ensure_tracking_credentials_restored(
        self, connections: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if self._credentials_restored:
            return
        with self._credential_restore_lock:
            if self._credentials_restored:
                return
            self._credentials_restored = True
            for provider in TRACKING_PROVIDERS:
                connection = (connections.get(provider) if connections is not None
                              else self.database.get_tracking_connection(provider))
                if not connection or self.credentials.get(provider):
                    continue
                config = connection.get("config_json") or {}
                expected_source = config.get("credential_source")
                authentication = config.get("authentication")
                environment_credentials = self._environment_credentials_for_endpoint(provider, connection.get("endpoint"))
                if expected_source == "environment":
                    if environment_credentials and authentication in (None, self._credential_authentication(provider, environment_credentials)):
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
                    provider, credentials, source="credential_store", endpoint=endpoint
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
        if not values or origin == "environment":
            if persistent:
                self.credential_store.delete(provider)
            self.credentials.clear(provider)
            return "environment" if values else None
        if remember:
            self.credential_store.save(provider, endpoint, values)
            self.credentials.replace(provider, values, source="credential_store", endpoint=endpoint)
            return "credential_store"
        if persistent:
            self.credential_store.delete(provider)
        self.credentials.replace(provider, values, source="session", endpoint=endpoint)
        return "session"

    def _tracking_connection_snapshot(self, provider: str) -> tuple[TrackingSettings | WandBSettings, int]:
        with self._tracking_connection_lock:
            settings = self._wandb_settings() if provider == "wandb" else self._mlflow_settings()
            return settings, self._tracking_connection_revisions[provider]

    def _commit_tracking_connection(
        self, provider: str, endpoint: str, credentials: Mapping[str, str | None],
        *, origin: str | None, remember: bool, verify_tls: bool,
        expected_settings: TrackingSettings | WandBSettings,
        expected_revision: int,
        workspace: str | None = None, public_state: Mapping[str, Any] | None = None,
    ) -> None:
        # Only the local commit is serialized; remote verification does not block readers.
        with self._tracking_connection_lock:
            current_settings = self._wandb_settings() if provider == "wandb" else self._mlflow_settings()
            if current_settings != expected_settings or self._tracking_connection_revisions[provider] != expected_revision:
                raise ValueError("The tracking connection changed while it was connecting; review the current connection and try again")
            self._tracking_connection_revisions[provider] += 1
            try:
                source = self._activate_validated_credentials(
                    provider, endpoint, credentials, origin=origin, remember=remember,
                )
                self.database.upsert_tracking_connection(
                    provider, endpoint=endpoint, workspace=workspace,
                    config={"verify_tls": verify_tls,
                            "authentication": self._credential_authentication(provider, credentials),
                            "credential_source": source},
                )
                self.credentials.mark_connected(provider, **(public_state or {}))
            except Exception as error:
                self.credentials.mark_error(provider, sanitize(str(error), secrets=tuple(credentials.values())))
                raise

    def _mlflow_settings(
        self, provider: Any | None = None, *,
        connections: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> TrackingSettings:
        with self._tracking_connection_lock:
            self._ensure_tracking_credentials_restored(connections)
            settings = TrackingSettings.from_env(self._tracking_environment())
            connection = (connections.get("mlflow") if connections is not None
                          else self.database.get_tracking_connection("mlflow")) or {}
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
            credentials, _ = self._bound_tracking_credentials("mlflow", tracking_uri, connections=connections)
            return replace(
                settings,
                tracking_uri=tracking_uri,
                token=credentials.get("token"),
                username=credentials.get("username"),
                password=credentials.get("password"),
                verify_tls=bool((connection.get("config_json") or {}).get("verify_tls", settings.verify_tls)),
            )

    def _wandb_settings(
        self, provider: Any | None = None, *,
        connections: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> WandBSettings:
        with self._tracking_connection_lock:
            self._ensure_tracking_credentials_restored(connections)
            settings = WandBSettings.from_env(self._tracking_environment())
            connection = (connections.get("wandb") if connections is not None
                          else self.database.get_tracking_connection("wandb")) or {}
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
            credentials, _ = self._bound_tracking_credentials("wandb", base_url, connections=connections)
            entity = (
                self._tracking_provider_value(provider, "entity") if provider else None
            ) or connection.get("workspace") or settings.entity
            return replace(
                settings,
                base_url=base_url,
                entity=entity,
                api_key=credentials.get("api_key"),
                verify_tls=bool((connection.get("config_json") or {}).get("verify_tls", settings.verify_tls)),
            )

    def tracking_connections(self) -> dict[str, Any]:
        with self._tracking_connection_lock:
            # A fresh scoped read is shared only within this response, under the
            # same lock as credential restoration and endpoint binding.
            connections = self.database.list_tracking_connections()
            self._ensure_tracking_credentials_restored(connections)
            results: dict[str, Any] = {}
            for provider in TRACKING_PROVIDERS:
                runtime = self.credentials.state(provider)
                if provider == "wandb":
                    settings = self._wandb_settings(connections=connections)
                    public = {
                        "provider": provider,
                        "base_url": settings.public_dict()["base_url"],
                        "entity": settings.entity,
                        "configured": settings.configured,
                    }
                else:
                    settings = self._mlflow_settings(connections=connections)
                    public = {
                        "provider": provider,
                        "tracking_uri": settings.public_dict()["tracking_uri"],
                        "username": settings.username,
                        "configured": settings.configured,
                    }
                effective_credentials, credential_source = self._bound_tracking_credentials(provider, settings.base_url if provider == "wandb" else settings.tracking_uri, connections=connections)
                expected_mode = ((connections.get(provider) or {}).get("config_json") or {}).get("authentication")
                connected = bool(runtime.get("connected")) and (expected_mode in (None, "none") or bool(effective_credentials))
                configured = bool(public["configured"])
                status = (
                    "error" if runtime.get("last_error")
                    else "connected" if connected
                    else "configured" if configured
                    else "not_configured"
                )
                public.update({
                    "verify_tls": settings.verify_tls,
                    "connected": connected,
                    "status": status,
                    "credential_source": credential_source,
                    "validated_at": runtime.get("validated_at"),
                    "last_error": runtime.get("last_error"),
                })
                results[provider] = public
            return {"connections": results}

    def configure_tracking_connection(
        self, provider: str, request: TrackingConnectionRequest
    ) -> dict[str, Any]:
        self._ensure_tracking_credentials_restored()
        if provider not in TRACKING_PROVIDERS:
            raise ValueError(f"unsupported tracking provider: {provider}")
        if provider == "wandb":
            if any((request.tracking_uri, request.token, request.username, request.password)):
                raise ValueError("W&B connection accepts only api_key, base_url, entity, and verify_tls")
            saved_settings, saved_revision = self._tracking_connection_snapshot("wandb")
            base_url = self._validate_tracking_endpoint(
                request.base_url or saved_settings.base_url or WANDB_DEFAULT_BASE_URL, "W&B base URL"
            )
            submitted_key = request.api_key is not None
            if not submitted_key and base_url.rstrip("/") != saved_settings.base_url.rstrip("/"):
                raise ValueError("Enter a new W&B API key when changing the base URL; saved credentials stay bound to their original endpoint")
            if submitted_key:
                key = request.api_key.get_secret_value() if request.api_key else ""
                origin = "session"
            else:
                current_credentials, origin = self._bound_tracking_credentials("wandb", base_url)
                key = current_credentials.get("api_key", "")
            if not key:
                raise ValueError("W&B API key is required")
            settings = replace(
                WandBSettings.from_env(self._tracking_environment()),
                base_url=base_url,
                api_key=key,
                entity=request.entity or None,
                verify_tls=request.verify_tls,
            )
            try:
                bridge = self._wandb_bridge(
                    LOCAL_CAPSULE_ROOT / ".tracking-connections" / (self.database.workspace_id or LEGACY_WORKSPACE) / "wandb", settings
                )
                identity = bridge.validate_connection()
                entity = request.entity or str(identity["entity"])
                bridge.validate_entity(entity)
            except Exception as error:
                safe_error = str(sanitize(str(error), secrets=(key,)))
                with self._tracking_connection_lock:
                    if self._wandb_settings() == saved_settings and self._tracking_connection_revisions["wandb"] == saved_revision:
                        self.credentials.mark_error("wandb", safe_error)
                raise TrackingRequestError(safe_error) from error
            self._commit_tracking_connection(
                "wandb", base_url, {"api_key": key}, origin=origin,
                remember=request.remember, verify_tls=request.verify_tls,
                expected_settings=saved_settings,
                expected_revision=saved_revision,
                workspace=entity, public_state={"entity": entity, "username": identity.get("username")},
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
            saved_settings, saved_revision = self._tracking_connection_snapshot("mlflow")
            same_endpoint = tracking_uri.rstrip("/") == (saved_settings.tracking_uri or "").rstrip("/")
            old_credentials, old_origin = self._bound_tracking_credentials("mlflow", saved_settings.tracking_uri)
            connection = self.database.get_tracking_connection("mlflow") or {}
            old_mode = (connection.get("config_json") or {}).get("authentication")
            authenticated = bool(old_credentials) or old_mode in {"token", "basic"}
            if not same_endpoint and authenticated and not (request.token or request.password):
                raise ValueError("Enter new MLflow credentials when changing the tracking URI, or disconnect first to use an unauthenticated endpoint")
            reusable = old_credentials if same_endpoint else {}
            if request.token and (request.username or request.password):
                raise ValueError("Choose either token authentication or username and password")
            if submitted_authentication:
                same_username = request.username == reusable.get("username")
                credentials = {
                    "token": request.token.get_secret_value() if request.token else None,
                    "username": request.username,
                    "password": request.password.get_secret_value() if request.password else (
                        reusable.get("password") if same_username and not request.token else None
                    ),
                }
                origin = "session"
            else:
                credentials = reusable
                origin = old_origin if credentials else None
            token = credentials.get("token")
            username = credentials.get("username")
            password = credentials.get("password")
            settings = replace(
                TrackingSettings.from_env(self._tracking_environment()),
                tracking_uri=tracking_uri,
                token=token,
                username=username,
                password=password,
                verify_tls=request.verify_tls,
            )
            try:
                self._mlflow_bridge(
                    LOCAL_CAPSULE_ROOT / ".tracking-connections" / (self.database.workspace_id or LEGACY_WORKSPACE) / "mlflow", settings
                ).validate_connection()
            except Exception as error:
                safe_error = str(sanitize(str(error), secrets=(token, password)))
                with self._tracking_connection_lock:
                    if self._mlflow_settings() == saved_settings and self._tracking_connection_revisions["mlflow"] == saved_revision:
                        self.credentials.mark_error("mlflow", safe_error)
                raise TrackingRequestError(safe_error) from error
            self._commit_tracking_connection(
                "mlflow", tracking_uri, credentials, origin=origin,
                remember=request.remember, verify_tls=request.verify_tls,
                expected_settings=saved_settings,
                expected_revision=saved_revision,
            )
        return {
            "connection": self.tracking_connections()["connections"][provider],
            "flush": self._flush_tracking_provider(provider),
        }

    def disconnect_tracking_connection(self, provider: str) -> dict[str, Any]:
        with self._tracking_connection_lock:
            self._ensure_tracking_credentials_restored()
            if provider not in TRACKING_PROVIDERS:
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
            self._tracking_connection_revisions[provider] += 1
            self.credentials.clear(provider)
            self.database.delete_tracking_connection(provider)
            return {"connection": self.tracking_connections()["connections"][provider]}

    def _active_tracking_providers(self, spec: ExperimentSpec) -> list[Any]:
        return [item for item in spec.tracking.providers if item.enabled]

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
        if provider not in TRACKING_PROVIDERS:
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
            elif state in ATTEMPT_FAILURE_STATES:
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

    def gpu_tracking_bridge(
        self, run: Mapping[str, Any], spec: ExperimentSpec, capsule_root: Path
    ) -> WandBBridge | None:
        """The W&B bridge a run's GPU statistics publish through, or None when they must not.

        An adapter's native W&B SDK owns the run's events stream, so the console
        never competes with it; a run without an active W&B provider has nowhere
        to publish.
        """
        if "wandb" in self._native_tracking_provider_names(spec):
            return None
        provider = next(
            (
                p
                for p in self._active_tracking_providers(spec)
                if self._tracking_provider_value(p, "provider") == "wandb"
            ),
            None,
        )
        if provider is None:
            return None
        return self._wandb_bridge(
            Path(capsule_root) / str(run["id"]), replace(self._wandb_settings(provider), auto_flush=False)
        )

    def deliver_gpu_tracking(self, bridge: WandBBridge, run: Mapping[str, Any]) -> RuntimeError | None:
        """Drain the GPU samples queued on the bridge ``gpu_tracking_bridge`` returned.

        A delivery error is recorded on the run's W&B binding and returned rather
        than raised, so the caller alone decides whether it fails anything.
        """
        report = bridge.drain_spool(limit=TRACKING_FLUSH_EVENT_LIMIT)
        if not report.errors:
            return None
        error = RuntimeError(report.errors[0])
        self._tracking_failure("wandb", run, error)
        return error

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
        run_directory = self._run_directory(run_id)
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

    def _flush_tracking_provider(self, provider: str, *, limit: int = TRACKING_CONNECTION_FLUSH_RUN_LIMIT, event_limit: int | None = None, run_ids: set[str] | None = None) -> dict[str, Any]:
        attempted = delivered = 0
        errors: list[str] = []
        bindings = [binding for binding in self.database.list_tracking_bindings_for_provider(
            provider, scope_type="run", statuses=("QUEUED", "ERROR")
        ) if run_ids is None or str(binding["scope_id"]) in run_ids][:limit]
        # Experiment ids and bindings are read once per flush, not once per run.
        experiment_ids = self.database.run_experiment_ids([str(binding["scope_id"]) for binding in bindings])
        experiment_bindings = {
            str(item["scope_id"]): item
            for item in self.database.list_tracking_bindings_for_provider(provider, scope_type="experiment")
        }
        for existing in bindings:
            run_id = str(existing["scope_id"])
            capsule = LOCAL_CAPSULE_ROOT / run_id
            try:
                if provider == "wandb":
                    settings = self._wandb_settings()
                    current_endpoint = settings.base_url
                    bridge: Any = self._wandb_bridge(capsule, settings)
                else:
                    settings = self._mlflow_settings()
                    current_endpoint = settings.tracking_uri
                    bridge = self._mlflow_bridge(capsule, settings)
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
                report = bridge.drain_spool(limit=event_limit)
                attempted += report.attempted
                delivered += report.delivered
                errors.extend(report.errors)
                binding = bridge.binding(run_id)
                remote_id = (binding or {}).get("remote_id")
                remote_url = (binding or {}).get("url") or existing.get("remote_url")
                experiment_scope_id = experiment_ids.get(run_id, "")
                experiment_binding = experiment_bindings.get(experiment_scope_id, {}) if experiment_scope_id else {}
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
                    experiment_bindings[experiment_scope_id] = next(
                        (item for item in self.database.list_tracking_bindings("experiment", experiment_scope_id)
                         if item["provider"] == provider), {})
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
        return compact_tracking_parameters({
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
        })

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
        job_id: str,
        *,
        attempt_snapshot: Mapping[str, Any] | None = None,
        providers: list[Any] | None = None,
        continuation_attempt_number: int | None = None,
        auto_flush: bool | None = None,
    ) -> None:
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
                    if auto_flush is not None:
                        settings = replace(settings, auto_flush=auto_flush)
                    bridge = self._mlflow_bridge(LOCAL_CAPSULE_ROOT / local_run_id, settings)
                    experiment_name = (
                        self._tracking_provider_value(provider, "experiment")
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
                elif name == "wandb":
                    settings = self._wandb_settings(provider)
                    if auto_flush is not None:
                        settings = replace(settings, auto_flush=auto_flush)
                    entity = settings.entity
                    if not entity:
                        if not settings.configured:
                            raise TrackingRequestError(
                                "W&B is enabled but no session/environment API key and entity are configured"
                            )
                        entity = str(
                            self._wandb_bridge(
                                LOCAL_CAPSULE_ROOT / local_run_id, settings
                            ).validate_connection()["entity"]
                        )
                        settings = replace(settings, entity=entity)
                    project = wandb_project_slug(
                        self._tracking_provider_value(provider, "project")
                        or spec.identity.experiment
                    )
                    bridge = self._wandb_bridge(LOCAL_CAPSULE_ROOT / local_run_id, settings)
                    project_result = bridge.ensure_experiment(entity, project)
                    existing = next((
                        item for item in self.database.list_tracking_bindings("run", local_run_id)
                        if item["provider"] == "wandb"
                    ), {})
                    local_binding = bridge.binding(local_run_id)
                    metadata = existing.get("metadata_json") or {}
                    expected_url = (
                        f"{wandb_web_base(settings.base_url)}/"
                        f"{urllib.parse.quote(entity, safe='')}/"
                        f"{urllib.parse.quote(project, safe='')}/runs/"
                        f"{urllib.parse.quote(local_run_id, safe='')}"
                    )
                    # A failed submission can have an attempt number without ever
                    # creating a tracking run. Missing records are not mismatches.
                    comparisons = [
                        (existing.get("remote_id"), local_run_id),
                        (existing.get("remote_url"), expected_url),
                        (metadata.get("entity"), entity),
                        (metadata.get("project"), project),
                        (str(metadata.get("endpoint") or "").rstrip("/"),
                         str(settings.public_dict()["base_url"] or "").rstrip("/")),
                        ((local_binding or {}).get("remote_id"), local_run_id),
                        ((local_binding or {}).get("url"), expected_url),
                    ]
                    if any(saved and saved != expected for saved, expected in comparisons):
                        raise TrackingRequestError(
                            "This run uses a different W&B account or project. Restore its original W&B connection."
                        )
                    if not local_binding:
                        if existing.get("remote_id"):
                            raise TrackingRequestError(
                                "W&B history is missing from this computer. Reconnect W&B for this run."
                            )
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
                    elif continuation_attempt_number is not None:
                        created = bridge.reopen_run(local_run_id, continuation_attempt_number)
                    else:
                        created = project_result
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
                f"https://huggingface.co/datasets/{resource['namespace']}/{resource['source_key']}"
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

    def cancel_data_import(self, import_id: str) -> dict[str, Any]:
        record = self.database.get_data_import(import_id)
        if record is None:
            raise KeyError("Data import not found")
        if record["state"] in DATA_IMPORT_TERMINAL_STATES | DATA_IMPORT_SETTLING_STATES:
            return record
        if not record.get("slurm_job_id"):
            raise ValueError("Import submission is still in progress; refresh before cancelling")
        claimed = self.database.update_data_import(
            import_id, expected_states=DATA_IMPORT_SCHEDULED_STATES,
            state="CANCELLING", error=None,
        )
        if claimed is None:
            return self.database.get_data_import(import_id)
        try:
            gateway = self.cluster.cancel(str(record["slurm_job_id"]), record.get("gateway") or "auto")
        except Exception as error:
            self.database.update_data_import(
                import_id, expected_states=["CANCELLING"], state=record["state"],
                error=f"Import cancellation failed: {error}",
            )
            raise
        self.database.update_data_import(
            import_id, expected_states=["CANCELLING"], gateway=gateway, error=None,
        )
        return self.database.get_data_import(import_id)

    # Background reconciliation and browser refresh may finalize together.
    # Share the local publication lock across service instances; remote reads
    # happen before entering it, and no network work belongs in publication.
    _data_import_publication_lock = threading.RLock()

    def _update_data_import_observation(self, import_id: str, **changes: Any) -> dict[str, Any]:
        with self._data_import_publication_lock:
            return self.database.update_data_import(import_id, **changes)

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
            "name": resource["source_key"],
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
                f"resource:huggingface/{resource['namespace']}/{resource['source_key']}"
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
        return self.database.update_data_import(
            str(record["id"]),
            state="SUCCEEDED",
            result=dict(result),
            slurm_state="COMPLETED",
            version_id=version["id"],
            bundle_id=None,
            error=None,
        )

    def reconcile_data_imports(self) -> list[dict[str, Any]]:
        records = [
            record
            for record in self.database.list_data_imports(
                states=DATA_IMPORT_IN_FLIGHT_STATES
            )
            if record.get("slurm_job_id")
        ]
        # One accounting round trip per gateway answers for every in-flight import;
        # a failed lookup settles each of its imports exactly as a direct one would.
        job_ids_by_gateway: dict[str, list[str]] = {}
        for record in records:
            job_ids_by_gateway.setdefault(
                str(record.get("gateway") or "auto"), []
            ).append(str(record["slurm_job_id"]))
        snapshots: dict[str, tuple[str, dict[str, dict[str, str]]] | Exception] = {}
        for requested_gateway, job_ids in job_ids_by_gateway.items():
            try:
                snapshots[requested_gateway] = self.cluster.job_statuses(job_ids, requested_gateway)
            except Exception as error:
                snapshots[requested_gateway] = error
        for record in records:
            job_id = record["slurm_job_id"]
            try:
                snapshot = snapshots[str(record.get("gateway") or "auto")]
                if isinstance(snapshot, Exception):
                    raise snapshot
                gateway, statuses = snapshot
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
                    app_state = "RUNNING" if slurm_state in EXECUTING_STATES else "PENDING"
                    self._update_data_import_observation(record["id"], state=app_state, **common)
                    continue
                if slurm_state.startswith("CANCELLED"):
                    self._update_data_import_observation(record["id"], state="CANCELLED", **common)
                    continue
                if slurm_state == "COMPLETED":
                    _, payload = self.cluster.read_file(str(record["result_path"]), gateway)
                    result = json.loads(payload)
                    with self._data_import_publication_lock:
                        claimed = self.database.update_data_import(
                            record["id"], state="FINALIZING", **common,
                            expected_states=DATA_IMPORT_IN_FLIGHT_STATES,
                        )
                        if claimed is None or claimed["state"] != "FINALIZING":
                            continue
                        try:
                            self._publish_data_import_result(claimed, result)
                        except Exception as error:
                            # Settle a failed local publication before another
                            # refresh may claim the same unfinished import.
                            self.database.update_data_import(
                                record["id"], state="FAILED",
                                error=f"Import reconciliation failed: {error}",
                            )
                    continue
                error = f"Slurm data import ended in {slurm_state} ({status.get('ExitCode') or 'unknown exit'})"
                try:
                    _, stderr = self.cluster.read_log(str(record["stderr_path"]), gateway, lines=80)
                    if stderr.strip():
                        error += "\n" + stderr.strip()
                except ClusterError:
                    pass
                common["error"] = error
                self._update_data_import_observation(record["id"], state="FAILED", **common)
            except ClusterError as error:
                self._update_data_import_observation(
                    record["id"], error=f"Import status refresh failed: {error}"
                )
            except Exception as error:
                self._update_data_import_observation(
                    record["id"], state="FAILED", error=f"Import reconciliation failed: {error}"
                )
        return self.database.list_data_imports()

    def _repair_missing_attempt_log_paths(self) -> int:
        with self.database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT id, gateway, slurm_job_id, sbatch_path, stdout_path, stderr_path,
                       (SELECT run_id FROM workflow_stages WHERE id=job_attempts.stage_id) AS run_id
                FROM job_attempts
                WHERE {visible_sql("job_attempts")} AND slurm_job_id IS NOT NULL
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
                script, job_id, logs_root=self._run_paths(str(row["run_id"])).logs
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
            try:
                run = self.database.get_run(run_id, include_details=False)
                if not run:
                    continue
                spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
            except Exception:
                logging.getLogger(__name__).exception("Tracking repair could not read run: %s", run_id)
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
                # Fetch immutable execution bodies only when a binding is actually
                # missing. Scanning healthy runs must not download every capsule.
                if not isinstance(snapshot, Mapping):
                    run = self.database.get_run(run_id) or run
                self._start_tracking(
                    spec,
                    run,
                    str(attempt["slurm_job_id"]),
                    attempt_snapshot=snapshot if isinstance(snapshot, Mapping) else None,
                    providers=missing,
                    auto_flush=False,
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

    def _repair_recovered_submission_tracking(self, attempt: Mapping[str, Any]) -> None:
        # The scheduler receipt is already durable. Optional telemetry cannot
        # invalidate it, and reconciliation retries a missing/error binding.
        try:
            self._repair_missing_active_tracking_bindings([attempt])
        except Exception:
            logging.getLogger(__name__).exception(
                "Recovered submission tracking remains pending: %s", attempt["run_id"]
            )

    def _record_recovered_submission(self, unknown: Mapping[str, Any], recovered: Any) -> None:
        submitted_at = utc_now()
        stdout_path = resolve_slurm_log_path(
            unknown["stdout_path"], recovered.job_id, logs_root=self._run_paths(str(unknown["run_id"])).logs
        )
        stderr_path = resolve_slurm_log_path(
            unknown["stderr_path"], recovered.job_id, logs_root=self._run_paths(str(unknown["run_id"])).logs
        )
        is_evaluation = unknown["stage_type"] == "EVALUATE"
        transition = self._transition_stage(
            stage_id=unknown["stage_id"],
            stage_type=unknown["stage_type"],
            run_id=unknown["run_id"],
            evaluation_id=unknown["evaluation_id"] if is_evaluation else None,
            status="SUBMITTED",
            stage_updates={"started_at": submitted_at},
            run_updates={"started_at": unknown["run_started_at"] or submitted_at, "completed_at": None},
            evaluation_updates={"started_at": submitted_at, "completed_at": None},
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
            event_type="SUBMISSION_RECOVERED",
            details={
                "attempt_id": unknown["id"],
                "job_id": recovered.job_id,
                "gateway": recovered.gateway,
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

        elif transition.get("applied") and not is_evaluation:
            self._repair_recovered_submission_tracking({**unknown, "slurm_job_id": recovered.job_id})

    def _interruption_receipt(self, attempt: Mapping[str, Any]) -> dict[str, Any] | None:
        """The receipt our batch script left for this attempt's deliberate exit, if any.

        It distinguishes the time-limit warning exit and the GPU preflight exit
        from an arbitrary trainer failure; a receipt bound to another job or run
        does not count.
        """
        path = (
            f"{self._run_directory(attempt['run_id'])}/attempts/"
            f"{attempt['slurm_job_id']}/state/{INTERRUPTION_RECEIPT}"
        )
        # Only a gateway that answers can say the receipt is missing.
        _, content = self.cluster.read_optional_file(path, attempt.get("gateway") or "auto", max_bytes=65_536)
        try:
            receipt = json.loads(content or "")
        except ValueError:
            return None
        if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
            return None
        if receipt.get("job_id") != str(attempt["slurm_job_id"]) or receipt.get("run_id") != attempt["run_id"]:
            return None
        return receipt

    @staticmethod
    def _accounting_state_and_reason(record: Mapping[str, Any]) -> tuple[str, str | None]:
        raw = str(record.get("StateRaw") or record["State"]).strip()
        state = re.split(r"[+\s]", str(record["State"]).strip(), maxsplit=1)[0].upper()
        reason = record.get("Reason")
        if not reason or str(reason).strip().upper() in {"NONE", "UNKNOWN", "N/A"}:
            cancelled_by = record.get("CancelledBy")
            reason = (f"CANCELLED by {cancelled_by}" if state == "CANCELLED" and cancelled_by
                      else raw if raw.upper() != state else None)
        return state, reason

    @classmethod
    def _active_accounting_unchanged(cls, row: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
        """A read-only scheduler snapshot needs no lifecycle lock or write.

        Any actual change takes the locked, fresh-read path below. A concurrent
        cancellation after this snapshot is handled by its own operation or the
        next scan; this fast path never writes stale state over that operation.
        """
        state, reason = cls._accounting_state_and_reason(record)
        if state not in ACTIVE_STATES:
            return False
        status = "RUNNING" if state in EXECUTING_STATES else "PENDING"
        parent_status = row.get("evaluation_status") if row["stage_type"] == "EVALUATE" else row.get("run_status")
        restarts = _progress_integer(record.get("Restarts"))
        return (
            row["status"] == status and parent_status == status
            and row["stage_status"] == ("RUNNING" if status == "RUNNING" else "PENDING_SLURM")
            and (status != "RUNNING" or bool(row.get("started_at")))
            and (restarts is None or restarts <= int(row.get("restart_count") or 0))
            and all(row.get(key) == value for key, value in {
                "slurm_state": state, "slurm_reason": reason,
                "exit_code": record.get("ExitCode"), "node_list": record.get("NodeList"),
            }.items())
        )

    def reconcile(self) -> dict[str, Any]:
        if not self._reconcile_scan_lock.acquire(blocking=False):
            return {
                "ok": True,
                "skipped": "already running",
                "draft_graphs_repaired": 0,
                "tracking_bindings_repaired": 0,
            }
        try:
            with self._reconcile_lock:
                repairs = self.database.repair_workflow_state_invariants()
                repairs["document_projections_repaired"] = self.database.repair_document_projections()
            recovered_submissions = 0
            with self.database.connection() as connection:
                unknown_rows = [
                    dict(row)
                    for row in connection.execute(
                    f"""
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
                    WHERE {visible_sql('job_attempts', 'a')} AND a.status IN ('SUBMITTING', 'CANCELLING')
                      AND a.slurm_job_id IS NULL
                      AND a.slurm_reason LIKE '{SUBMISSION_UNKNOWN_PREFIX}%'
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
                with self._reconcile_lock:
                    self._record_recovered_submission(unknown, recovered)
                recovered_submissions += 1
            self._repair_missing_attempt_log_paths()
            try:
                if self.database.workspace_id in (None, LEGACY_WORKSPACE):
                    self.reconcile_data_imports()
            except Exception:
                pass
            with self.database.connection() as connection:
                rows = [
                    dict(row)
                    for row in connection.execute(
                    f"""
                    SELECT a.id, a.stage_id, a.slurm_job_id, a.status, a.gateway,
                           a.started_at, a.finished_at, a.submitted_at, a.restart_count,
                           a.attempt_number, a.stdout_path, a.stderr_path,
                           a.slurm_state, a.slurm_reason, a.exit_code, a.node_list,
                           s.run_id, s.stage_type, s.auto_resume, s.max_attempts,
                           s.status AS stage_status, r.status AS run_status,
                           (SELECT status FROM evaluations WHERE stage_id=s.id) AS evaluation_status,
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
                    WHERE {visible_sql("job_attempts", "a")} AND a.slurm_job_id IS NOT NULL
                      AND (
                          a.status IN ({sql_list(SLURM_BOUND_ATTEMPT_STATES)})
                          OR (
                              a.status = 'SUCCEEDED'
                              AND s.status IN ('SUBMITTING','SUBMITTED','PENDING_SLURM','RUNNING','CANCELLING')
                          )
                      )
                    """
                    ).fetchall()
                ]
            tracking_bindings_repaired = 0  # Repaired by the independent tracking worker.
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
            statuses = self._scheduler_statuses(rows)
            updated = 0
            experiments: set[str] = set(repairs["experiment_ids"])
            for row in rows:
                if self._stop.is_set():
                    break
                record = statuses.get(row["slurm_job_id"])
                if not record or self._active_accounting_unchanged(row, record):
                    continue
                with self._reconcile_lock:
                    with self.database.connection() as connection:
                        current = connection.execute(
                            """SELECT a.status, a.slurm_job_id, a.restart_count,
                               a.started_at, s.status AS stage_status,
                               (SELECT id FROM job_attempts WHERE stage_id=s.id
                                ORDER BY attempt_number DESC LIMIT 1) AS latest_id
                            FROM job_attempts a JOIN workflow_stages s ON s.id=a.stage_id
                            WHERE a.id=?""", (row["id"],),
                        ).fetchone()
                    if (current is None or current["latest_id"] != row["id"]
                            or current["slurm_job_id"] != row["slurm_job_id"]
                            or current["stage_status"] not in
                                {'SUBMITTING','SUBMITTED','PENDING_SLURM','RUNNING','CANCELLING'}):
                        continue
                    row.update(dict(current))
                    self._sync_attempt_restart_counts([row], statuses)
                    record = statuses.get(row["slurm_job_id"])
                    if not record:
                        continue
                    state, reason = self._accounting_state_and_reason(record)
                    accounting_started = self._slurm_accounting_timestamp(record.get("Start"))
                    accounting_finished = self._slurm_accounting_timestamp(record.get("End"))
                    experiments.add(row["experiment_id"])
                    is_evaluation_stage = row["stage_type"] == "EVALUATE"
                    evaluation = (
                        self._evaluation_for_stage(row["run_id"], row["stage_id"])
                        if is_evaluation_stage
                        else None
                    )
                    stage_entity = {
                        "stage_id": row["stage_id"],
                        "stage_type": row["stage_type"],
                        "run_id": row["run_id"],
                        "evaluation_id": evaluation["id"] if evaluation else None,
                    }
                    common = {
                        "slurm_state": state,
                        "slurm_reason": reason,
                        "exit_code": record.get("ExitCode"),
                        "node_list": record.get("NodeList"),
                    }
                    receipt: dict[str, Any] | None = None
                    exit_code = record.get("ExitCode")
                    time_limit_exit = exit_code == f"{TIME_LIMIT_EXIT_CODE}:0"
                    if state == "FAILED" and (
                        # An exit record cannot tell a time limit from a cancellation
                        # that followed its warning, so it never queues a new attempt;
                        # the GPU preflight's receipt is unambiguous.
                        (time_limit_exit and record.get("Source") != EXIT_RECORD_SOURCE)
                        or exit_code == f"{GPU_MISSING_EXIT_CODE}:0"
                    ):
                        try:
                            receipt = self._interruption_receipt(row)
                        except ClusterError:
                            continue  # The gateway failed, not the receipt.
                    if time_limit_exit and time_limit_receipt(receipt):
                        state = "TIMEOUT"
                        common["slurm_reason"] = "Stopped at the Slurm time-limit warning; checkpoint preserved if available"
                    elif gpu_missing_receipt(receipt):
                        state = GPU_MISSING_STATE
                        common["slurm_reason"] = GPU_MISSING_MESSAGE
                    cancellation_requested = (
                        str(row.get("stage_status") or "").upper() == "CANCELLING"
                        or str(row.get("status") or "").upper() == "CANCELLING"
                    )
                    if state in ACTIVE_STATES:
                        if cancellation_requested:
                            attempt_updates = {"status": "CANCELLING", **common}
                            if state in EXECUTING_STATES and not row["started_at"]:
                                attempt_updates["started_at"] = accounting_started or utc_now()
                            self._transition_stage(
                                **stage_entity,
                                status="CANCELLING",
                                completed_at=None,
                                attempt_id=row["id"],
                                attempt_updates=attempt_updates,
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
                        attempt_status = "RUNNING" if state in EXECUTING_STATES else "PENDING"
                        stage_status = "RUNNING" if state in EXECUTING_STATES else "PENDING_SLURM"
                        attempt_updates = {"status": attempt_status, **common}
                        if state in EXECUTING_STATES and not row["started_at"]:
                            attempt_updates["started_at"] = accounting_started or utc_now()
                        self._transition_stage(
                            **stage_entity,
                            status=attempt_status,
                            stage_status=stage_status,
                            attempt_id=row["id"],
                            attempt_updates=attempt_updates,
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
                            try:
                                ingestion_error, completed_episodes = self._ingest_evaluation_result(row)
                            except ClusterError:
                                continue  # The gateway failed, not the result.
                            if ingestion_error and _within_result_read_grace(accounting_finished):
                                # Keep the attempt open; the next cycle reads the result again.
                                continue
                            if ingestion_error:
                                self._transition_stage(
                                    **stage_entity,
                                    status="FAILED",
                                    completed_at=finished,
                                    attempt_id=row["id"],
                                    attempt_updates=attempt_success_updates,
                                    event_type="EVALUATION_RESULT_INVALID",
                                    details={"error": ingestion_error, "job_id": row["slurm_job_id"]},
                                )
                                updated += 1
                                continue
                            self._transition_stage(
                                **stage_entity,
                                status="SUCCEEDED",
                                completed_at=finished,
                                evaluation_updates={"progress_completed": completed_episodes},
                                attempt_id=row["id"],
                                attempt_updates=attempt_success_updates,
                            )
                            updated += 1
                            continue
                        if row["stage_type"] == "TRAIN":
                            with self.database.connection() as connection:
                                row["resolved_config_json"] = connection.execute(
                                    "SELECT resolved_config_json FROM workflow_stages WHERE id=?",
                                    (row["stage_id"],),
                                ).fetchone()[0]
                            checkpoint_globs: list[str] = []
                            try:
                                checkpoint_globs = self._checkpoint_globs(row["resolved_config_json"])
                                self._capture_checkpoint(
                                    row["run_id"],
                                    row["id"],
                                    required=bool(checkpoint_globs),
                                    gateway=row["gateway"] or "auto",
                                )
                            except ClusterError:
                                continue  # The gateway failed, not the checkpoint.
                            except (*DATABASE_ERRORS, ConnectionError):
                                raise  # Nor did the database's failure change what the job produced.
                            except Exception as error:
                                message = sanitize(str(error))
                                self._transition_stage(
                                    **stage_entity,
                                    status="FAILED",
                                    completed_at=finished,
                                    attempt_id=row["id"],
                                    attempt_updates=attempt_success_updates,
                                    event_type="CHECKPOINT_FINALIZATION_FAILED",
                                    old_status=row["status"],
                                    details={
                                        "error": message,
                                        "attempt_id": row["id"],
                                        "job_id": row["slurm_job_id"],
                                        "slurm_state": state,
                                        "exit_code": record.get("ExitCode"),
                                        "process_status": "SUCCEEDED",
                                        "checkpoint_globs": checkpoint_globs,
                                    },
                                )
                                updated += 1
                                continue
                        self._transition_stage(
                            **stage_entity,
                            status="SUCCEEDED",
                            completed_at=finished,
                            attempt_id=row["id"],
                            attempt_updates=attempt_success_updates,
                        )
                        updated += 1
                        continue
                    if (
                        not cancellation_requested
                        and state in TRANSIENT_STATES
                        and row["auto_resume"]
                        and row["budget_attempt_count"] < row["max_attempts"]
                    ):
                        finished = accounting_finished or utc_now()
                        self._transition_stage(
                            **stage_entity,
                            status="RETRY_PENDING",
                            completed_at=None,
                            attempt_id=row["id"],
                            attempt_updates={
                                "status": state,
                                "started_at": row["started_at"] or accounting_started or row["submitted_at"] or finished,
                                "finished_at": finished,
                                **common,
                            },
                            event_type="AUTO_RESUME_QUEUED",
                            old_status=state,
                            details={"attempt": row["attempt_number"], "job_id": row["slurm_job_id"],
                                     "reason": common["slurm_reason"]},
                        )
                        updated += 1
                        continue
                    if state in ATTEMPT_FAILURE_STATES | {"CANCELLED"}:
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
                                    "state_raw": str(record.get("StateRaw") or record["State"]).strip(),
                                    "cancelled_by": record.get("CancelledBy"),
                                    "reason": reason,
                                    "accounting_start_raw": record.get("Start"),
                                    "accounting_end_raw": record.get("End"),
                                    "started_at": row["started_at"] or accounting_started,
                                    "finished_at": finished,
                                    "exit_code": record.get("ExitCode"),
                                    "node_list": record.get("NodeList"),
                                    "source": record.get("Source"),
                        }
                        self._transition_stage(
                            **stage_entity,
                            status=target_status,
                            completed_at=finished,
                            attempt_id=row["id"],
                            attempt_updates={
                                "status": target_status if cancellation_requested else state,
                                "started_at": row["started_at"] or accounting_started or row["submitted_at"] or finished,
                                "finished_at": finished,
                                **common,
                            },
                            event_type="JOB_CANCELLED" if target_status == "CANCELLED" else "JOB_FAILED",
                            old_status=row["status"],
                            details=event_details,
                        )
                        updated += 1
            for experiment_id in experiments:
                if self._stop.is_set():
                    break
                with self._reconcile_lock:
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
            self._reconcile_scan_lock.release()

    # Jobs the last scan found missing from the scheduler, and its condition then.
    _forgotten_jobs: frozenset[str] = frozenset()
    _scheduler_condition: str | None = None
    # When each held job's evidence is read again; nothing new appears between scans.
    _held_recheck: dict[str, float] | None = None

    def _scheduler_statuses(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Scheduler records for the active attempts in ``rows``.

        Accounting can be down and the controller forgets a finished job, so a job
        that both have lost is settled from its own exit record. An unreachable
        scheduler leaves every attempt as it is; dispatch still runs.
        """
        try:
            snapshot = self.cluster.job_status_snapshot([row["slurm_job_id"] for row in rows])
        except ClusterError as error:
            self._note_scheduler_condition(f"Slurm cannot be queried ({error})")
            self._forgotten_jobs = frozenset()
            return {}
        self._note_scheduler_condition(
            f"Slurm accounting is unavailable ({snapshot.accounting_error}); using the live controller"
            if snapshot.accounting_error else None
        )
        statuses = dict(snapshot.statuses)
        if snapshot.controller_error is not None:
            self._forgotten_jobs = frozenset()
            return statuses
        forgotten = frozenset(
            str(row["slurm_job_id"]) for row in rows if str(row["slurm_job_id"]) not in statuses
        )
        # One scan can race a submission's first appearance; act on the second.
        confirmed, self._forgotten_jobs = forgotten & self._forgotten_jobs, forgotten
        if self._held_recheck is None:
            self._held_recheck = {}
        for job in set(self._held_recheck) - forgotten:
            del self._held_recheck[job]
        for row in rows:
            if self._stop.is_set():
                break
            job = str(row["slurm_job_id"])
            if job not in confirmed:
                continue
            cancelling = "CANCELLING" in {
                str(row.get("status") or "").upper(), str(row.get("stage_status") or "").upper()}
            if not cancelling and time.monotonic() < self._held_recheck.get(job, 0.0):
                continue
            record = self._forgotten_job_status(row, cancelling)
            if record:
                statuses[job] = record
        return statuses

    def _note_scheduler_condition(self, condition: str | None) -> None:
        if condition == self._scheduler_condition:
            return
        logger = logging.getLogger(__name__)
        if condition:
            logger.warning("Job status is degraded: %s", sanitize(condition))
        else:
            logger.info("Job status is back to normal")
        self._scheduler_condition = condition

    def _forgotten_job_status(self, row: Mapping[str, Any], cancelling: bool) -> dict[str, Any] | None:
        """A scheduler record for a job that neither accounting nor the controller describes.

        The job's own exit record settles it. With none, a requested cancellation has
        taken effect; otherwise the attempt is held and says why.
        """
        try:
            record = self._exit_record_status(row)
        except ClusterError:
            return None  # Unreadable is not absent; the next scan reads it again.
        if record:
            return record
        if cancelling:
            return {"State": "CANCELLED", "StateRaw": "CANCELLED", "Reason": FORGOTTEN_JOB_CANCELLED,
                    "Source": "absent"}
        if row.get("slurm_reason") != FORGOTTEN_JOB_HELD:
            self.database.update_job_attempt(row["id"], slurm_reason=FORGOTTEN_JOB_HELD)
        self._held_recheck[str(row["slurm_job_id"])] = time.monotonic() + HELD_RECHECK_SECONDS
        return None

    def _exit_record_status(self, row: Mapping[str, Any]) -> dict[str, Any] | None:
        """The scheduler record a finished job left in its own capsule, if it can be trusted.

        A clean exit is a completion, which the normal path still validates against its
        checkpoint or result. Any other exit is a failure of unknown Slurm cause. A gateway
        that cannot read the capsule raises ClusterError.
        """
        capsule = f"{self._run_directory(row['run_id'])}/attempts/{row['slurm_job_id']}"
        gateway = row.get("gateway") or "auto"
        _, content = self.cluster.read_optional_file(f"{capsule}/final.json", gateway, max_bytes=65_536)
        try:
            receipt = json.loads(content) if content else None
            finished = datetime.fromisoformat(receipt["finished_at"])
            exit_code, restarts = receipt["exit_code"], receipt["restart_count"]
            proven = (
                receipt.get("schema_version") == 1
                and str(receipt.get("job_id")) == str(row["slurm_job_id"])
                and type(exit_code) is int and type(restarts) is int
                # A requeued job keeps the record of its previous execution until it exits again.
                and restarts >= int(row.get("restart_count") or 0)
                and finished.tzinfo is not None
            )
        except (KeyError, TypeError, ValueError):
            proven = False
        if not proven:
            return None
        state = "COMPLETED" if exit_code == 0 else "FAILED"
        record = {
            "State": state, "StateRaw": state, "ExitCode": f"{exit_code}:0",
            "Reason": EXIT_RECORD_REASON, "NodeList": receipt.get("node_list"),
            "End": str(int(finished.timestamp())), "Restarts": str(restarts),
            "Source": EXIT_RECORD_SOURCE,
        }
        # The exit record has no start time; the launch record of the same execution does.
        _, manifest = self.cluster.read_optional_file(
            f"{capsule}/system-manifest.json", gateway, max_bytes=1_000_000)
        try:
            started = datetime.fromisoformat(json.loads(manifest or "")["captured_at"])
            if started.tzinfo is not None and started <= finished:
                record["Start"] = str(int(started.timestamp()))
        except (KeyError, TypeError, ValueError):
            pass
        return record

    def reconcile_tracking(self) -> dict[str, Any]:
        """Sync optional telemetry independently of scheduler state transitions.

        Terminal runs with unfinished bindings are the durable work queue. A crash
        after recording SUCCEEDED but before enqueueing finish_run is retried from
        those persisted rows; no in-memory completion notification is required.
        """
        if not self._tracking_reconcile_lock.acquire(blocking=False):
            return {"ok": True, "skipped": "already running"}
        try:
            with self.database.connection() as connection:
                rows = [dict(row) for row in connection.execute(
                    f"""
                    SELECT DISTINCT ON (s.run_id)
                           a.id, a.attempt_number, a.slurm_job_id, a.stage_id,
                           s.run_id, s.stage_type,
                           er.experiment_id, r.status AS run_status
                    FROM job_attempts a
                    JOIN workflow_stages s ON s.id = a.stage_id
                    JOIN runs r ON r.id = s.run_id
                    JOIN variants v ON v.id = r.variant_id
                    JOIN experiment_revisions er ON er.id = v.experiment_revision_id
                    WHERE {visible_sql('job_attempts', 'a')}
                      AND a.slurm_job_id IS NOT NULL AND s.stage_type = 'TRAIN'
                      AND (
                        r.status IN ('SUBMITTED','PENDING','RUNNING','RETRY_PENDING','CANCELLING')
                        OR (r.status IN ('SUCCEEDED','FAILED','CANCELLED') AND (
                            EXISTS (
                                SELECT 1 FROM tracking_bindings b
                                WHERE b.scope_type='run' AND b.scope_id=r.id
                                  AND b.status <> 'BLOCKED'
                                  AND (b.status <> CASE r.status
                                      WHEN 'SUCCEEDED' THEN 'FINISHED'
                                      WHEN 'CANCELLED' THEN 'KILLED' ELSE 'FAILED' END
                                      OR (b.provider='wandb' AND COALESCE(
                                          b.metadata_json::jsonb ->> 'terminal_sync_protocol', '') <> '{WANDB_TERMINAL_SYNC_PROTOCOL}'))
                            ) OR EXISTS (
                                SELECT 1 FROM document_projections vp
                                CROSS JOIN LATERAL jsonb_array_elements(COALESCE(
                                    vp.projection_json -> 'tracking_providers', '[]'::jsonb)) AS requested(value)
                                WHERE vp.table_name='variants' AND vp.record_id=v.id
                                  AND COALESCE((requested.value ->> 'enabled')::boolean, true)
                                  AND NOT EXISTS (
                                      SELECT 1 FROM tracking_bindings b
                                      WHERE b.scope_type='run' AND b.scope_id=r.id
                                        AND b.provider=(requested.value ->> 'provider')
                                  )
                            )
                        ))
                      )
                    ORDER BY s.run_id, a.attempt_number DESC
                    """
                ).fetchall()]
            repaired = self._repair_missing_active_tracking_bindings(rows)
            synced = 0
            for row in rows:
                if self._stop.is_set():
                    break
                run_id = str(row["run_id"])
                try:
                    run = self.database.get_run(run_id, execution_stage_ids=[row["stage_id"]])
                    if not run:
                        continue
                    self._ingest_training_progress(run, raise_on_error=run["status"] == "SUCCEEDED")
                    status = {"SUCCEEDED": "FINISHED", "FAILED": "FAILED", "CANCELLED": "KILLED"}.get(run["status"])
                    if status:
                        spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
                        bindings = {binding["provider"]: binding for binding in self.database.list_tracking_bindings("run", run_id)}
                        pending = [provider for provider in self._active_tracking_providers(spec)
                                   if self._tracking_terminal_pending(
                                       bindings.get(str(self._tracking_provider_value(provider, "provider")), {}),
                                       status, str(self._tracking_provider_value(provider, "provider")))]
                        if pending:
                            self._sync_tracking_outputs(run_id, status=status, providers=pending)
                    else:
                        sync_gpu_statistics(self, run, LOCAL_CAPSULE_ROOT)
                        self._publish_training_progress_tracking(run_id)
                    synced += 1
                except Exception:
                    # Persisted run/binding state remains eligible for the next pass.
                    logging.getLogger(__name__).exception("Run tracking reconciliation failed: %s", run_id)
            self.flush_tracking()
            return {"ok": True, "synced": synced, "tracking_bindings_repaired": repaired}
        finally:
            self._tracking_reconcile_lock.release()

    def _sync_attempt_restart_counts(
        self, attempts: list[dict[str, Any]], statuses: Mapping[str, Mapping[str, Any]]
    ) -> None:
        """Refresh same-job restart metadata before assigning progress samples."""
        for attempt in attempts:
            record = statuses.get(str(attempt["slurm_job_id"]), {})
            count = _progress_integer(record.get("Restarts"))
            previous = int(attempt.get("restart_count") or 0)
            if count is None or count <= previous:
                continue
            updates: dict[str, Any] = {"restart_count": count}
            started = self._slurm_accounting_timestamp(record.get("Start"))
            if started:
                updates["started_at"] = started
            self.database.update_job_attempt(attempt["id"], **updates)
            attempt.update(updates)

    def _dispatch_active_experiments(self) -> None:
        with self.database.connection() as connection:
            experiments = connection.execute(
                f"SELECT id FROM experiments WHERE status='ACTIVE' AND {visible_sql('experiments', '')} LIMIT 1000"
            ).fetchall()
        for experiment in experiments:
            if self._stop.is_set():
                break
            with self._reconcile_lock:
                self._dispatch_experiment(experiment["id"])
                self._refresh_experiment_status(experiment["id"])

    def _refresh_experiment_status(self, experiment_id: str) -> None:
        with self.database.connection() as connection:
            runs = connection.execute(
                f"""SELECT r.status FROM runs r JOIN variants v ON v.id=r.variant_id
                WHERE v.experiment_revision_id=(SELECT id FROM experiment_revisions
                    WHERE experiment_id=? ORDER BY revision_number DESC LIMIT 1)
                  AND {visible_sql('runs', 'r')}""", (experiment_id,),
            ).fetchall()
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
        path = f"{self._run_directory(run_id)}/checkpoints/selected-for-inference.json"
        try:
            _, content = self.cluster.read_optional_file(path, gateway, max_bytes=1_000_000)
        except ClusterError:
            if required:
                raise  # Unreadable is not absent: the caller reads again later.
            return None
        if content is None:
            if required:
                raise RuntimeError("required inference checkpoint descriptor is missing")
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
        run_root = Path(self._run_directory(run_id))
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
                + "; then printf '%s' SKYNET_CHECKPOINT_PRESENT; else printf '%s' SKYNET_CHECKPOINT_MISSING; fi",
                gateway,
            )
        except ClusterError:
            if required:
                raise  # Unreadable is not absent: the caller probes again later.
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
                training_step=_checkpoint_training_step(payload.get("training_step")),
                path=checkpoint_path,
                sha256=checkpoint_sha256.lower(),
                size_bytes=size_bytes,
                is_resumable=False,
                is_selected_for_inference=True,
                metadata=payload,
            )
        except INTEGRITY_ERRORS as error:
            if required:
                raise RuntimeError(f"required inference checkpoint could not be registered: {error}") from error
            return None

    def _evaluation_for_stage(self, run_id: str, stage_id: str) -> dict[str, Any] | None:
        # Internal lifecycle operations need the ledger identity, not the UI's
        # enriched dataset/capsule context for every evaluation of this run.
        with self.database.connection() as connection:
            return self.database._decode(connection.execute(
                f"SELECT * FROM evaluations WHERE run_id=? AND stage_id=? AND {visible_sql('evaluations', '')}",
                (run_id, stage_id),
            ).fetchone())

    def _queue_list_progress_refresh(self, kind: str, records: list[dict[str, Any]]) -> bool:
        """Keep optional remote progress reads off the list response path."""
        now = time.monotonic()
        active_states = EXECUTING_STATES if kind == "training" else {"RUNNING"}
        eligible = active_states | (_PROGRESS_SUCCESS_STATES | _PROGRESS_FAILURE_STATES if kind == "training" else set())
        keys = {(kind, str(record["id"])) for record in records}
        with self._progress_refresh_lock:
            if self._stop.is_set():
                return False
            for record in records:
                key = (kind, str(record["id"]))
                status = str(record.get("status") or record.get("state") or "").upper()
                version = (status, record.get("updated_at"), record.get("completed_at"))
                if (status not in eligible or key in self._progress_refresh_inflight
                        or (version == self._progress_refresh_versions.get(key)
                            and now < self._progress_refresh_due.get(key, 0))):
                    continue
                if key in self._progress_refresh_pending or len(self._progress_refresh_pending) < PROGRESS_REFRESH_QUEUE_LIMIT:
                    self._progress_refresh_pending[key] = 0 if status in active_states else 1
                    self._progress_refresh_versions[key] = version
            while self._progress_refresh_pending and self._progress_refresh_workers < PROGRESS_REFRESH_WORKERS:
                self._progress_refresh_workers += 1
                threading.Thread(target=self._refresh_list_progress, name="skynet-list-progress", daemon=True).start()
            return bool(keys & (self._progress_refresh_pending.keys() | self._progress_refresh_inflight))

    def _list_progress_refresh_pending(self, kind: str, records: list[dict[str, Any]]) -> bool:
        keys = {(kind, str(record["id"])) for record in records}
        with self._progress_refresh_lock:
            return bool(keys & (self._progress_refresh_pending.keys() | self._progress_refresh_inflight))

    def _refresh_list_progress(self) -> None:
        while True:
            with self._progress_refresh_lock:
                if self._stop.is_set() or not self._progress_refresh_pending:
                    self._progress_refresh_workers -= 1
                    return
                key = min(self._progress_refresh_pending, key=self._progress_refresh_pending.get)
                priority = self._progress_refresh_pending.pop(key)
                self._progress_refresh_inflight.add(key)
            try:
                kind, identifier = key
                # Re-read identity/state when work begins; queued observations
                # must not revive a job that was cancelled in the meantime.
                record = self.database.get_run(identifier) if kind == "training" else self.database.get_evaluation(identifier)
                if record:
                    if kind == "training":
                        self._ingest_training_progress(record)
                    else:
                        self._ingest_evaluation_progress(record)
            except Exception:
                # Progress is optional enrichment. Keep the last recorded values
                # and permit a later retry without failing the list itself.
                pass
            finally:
                with self._progress_refresh_lock:
                    self._progress_refresh_inflight.discard(key)
                    self._progress_refresh_due[key] = time.monotonic() + (4 if priority == 0 else 60)

    def _ingest_training_progress(self, value: Mapping[str, Any], *, raise_on_error: bool = False) -> int:
        status = str(value.get("status") or value.get("state") or "").upper()
        terminal_states = _PROGRESS_SUCCESS_STATES | _PROGRESS_FAILURE_STATES
        if status not in EXECUTING_STATES | terminal_states:
            return 0
        run = value if value.get("stages") and value.get("attempts") else self.database.get_run(
            str(value.get("id") or value.get("run_id") or "")
        )
        if not run:
            return 0
        declared = _training_progress_contract(run)
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
        if not attempt:
            return 0
        attempt_status = str(attempt.get("status") or "").upper()
        final_read = attempt_status in terminal_states and bool(attempt.get("slurm_job_id"))
        if attempt_status not in EXECUTING_STATES and not final_read:
            return 0
        attempt_id = str(attempt.get("id") or "")
        if not attempt_id:
            return 0
        source = contract.source
        jsonl = source.kind == "jsonl"
        if final_read and not jsonl:
            return 0
        path = (
            str(PurePosixPath(run["run_directory"]) / source.path)
            if jsonl and run.get("run_directory")
            else attempt.get(f"{source.stream}_path") if not jsonl else None
        )
        if not isinstance(path, str) or not path:
            return 0
        restart_count = _progress_integer(attempt.get("restart_count")) or 0
        execution_boundary = None
        if jsonl and attempt.get("slurm_job_id") and run.get("run_directory"):
            job_id = str(attempt["slurm_job_id"])
            if not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id):
                raise ValueError("Invalid training progress Slurm job id")
            execution_boundary = {
                "boundary_path": str(PurePosixPath(run["run_directory"]) / "attempts" / job_id
                                     / "state" / f"training-progress-start-{restart_count}.json"),
                "job_id": job_id,
                "restart_count": restart_count,
                # Old first attempts predate launch receipts and have no previous
                # execution's log. Resumes require a proven launch boundary.
                "required": restart_count > 0 or len(attempts) > 1
                            or (_progress_integer(attempt.get("attempt_number")) or 1) > 1,
            }
        throttle_key = f"{run['id']}:{attempt_id}"
        final_key = (throttle_key, attempt.get("restart_count"), attempt.get("finished_at"))
        final_reads = getattr(self, "_training_progress_final_reads", set())
        if final_read and final_key in final_reads:
            return 0
        monotonic_now = time.monotonic()
        last_reads = getattr(self, "_training_progress_last_reads", {})
        final_failures = getattr(self, "_training_progress_final_failures", {})
        if final_read and monotonic_now < final_failures.get(final_key, 0):
            if raise_on_error:
                raise ClusterError("Final training progress is awaiting a read retry")
            return 0
        if not final_read and monotonic_now - float(last_reads.get(throttle_key, 0.0)) < source.poll_seconds:
            return 0
        last_reads[throttle_key] = monotonic_now
        self._training_progress_last_reads = last_reads
        try:
            _, content = self.cluster.read_log(
                path,
                attempt.get("gateway") or "auto",
                lines=source.tail_lines,
                max_bytes=1_000_000,
                **({"contains": json.dumps(source.required_key)} if jsonl else {}),
                **({"execution_boundary": execution_boundary} if execution_boundary else {}),
            )
        except ClusterError as error:
            ended = _progress_timestamp(attempt.get("finished_at"))
            if (final_read and BOUNDARY_NOT_READY in str(error)
                    and ended is not None and datetime.now(timezone.utc) - ended >= RESULT_READ_GRACE):
                # The launcher publishes the boundary before it starts the trainer. An
                # execution that ended without one left no row that is provably its own.
                logging.getLogger(__name__).warning(
                    "Run %s: the execution that ended at %s published no training progress "
                    "boundary; its progress is final with no new rows", run["id"], attempt.get("finished_at"))
                final_reads.add(final_key)
                self._training_progress_final_reads = final_reads
                final_failures.pop(final_key, None)
                return 0
            if final_read:
                final_failures[final_key] = time.monotonic() + FINAL_PROGRESS_RETRY_SECONDS
                self._training_progress_final_failures = final_failures
            if raise_on_error:
                raise
            return 0
        records = parse_declared_training_progress(
            content, contract, resolved_spec=run.get("resolved_spec_json")
        )
        expected_total = _resolved_training_total(run.get("resolved_spec_json"), contract.total_path)
        if expected_total is not None:
            records = [record for record in records if record["total"] == expected_total]
        if not records:
            if final_read and attempt_status in _PROGRESS_SUCCESS_STATES:
                # An empty/temporarily unavailable tail is not completion evidence.
                final_failures[final_key] = time.monotonic() + FINAL_PROGRESS_RETRY_SECONDS
                self._training_progress_final_failures = final_failures
                if raise_on_error:
                    raise ClusterError("Successful training has no valid final progress evidence yet")
            elif final_read:
                final_reads.add(final_key)
                self._training_progress_final_reads = final_reads
                final_failures.pop(final_key, None)
            return 0

        segment: list[dict[str, Any]] = []
        reset_detected = False
        for record in records:
            if segment:
                previous = segment[-1]
                reset = record["completed"] < previous["completed"]
                if record["elapsed_seconds"] is not None and previous["elapsed_seconds"] is not None:
                    reset = reset or record["elapsed_seconds"] < previous["elapsed_seconds"]
                if reset:
                    segment = []
                    reset_detected = True
            segment.append(record)
        latest = segment[-1]
        candidates = segment if jsonl else [latest]
        if (
            not jsonl and source.elapsed_format is not None
            and segment[0]["completed"] != latest["completed"]
        ):
            candidates.insert(0, segment[0])

        samples = self.database.list_training_progress_samples(
            str(run["id"]), attempt_id=attempt_id
        )
        existing = {
            int(sample["completed"]): sample
            for sample in samples
            if int(sample.get("restart_count") or 0) == restart_count
        }
        prior = {
            int(sample["completed"]): sample
            for sample in sorted(samples, key=lambda item: int(item.get("restart_count") or 0))
            if int(sample.get("restart_count") or 0) < restart_count
        }
        observed_now = datetime.now(timezone.utc)
        latest_elapsed = latest.get("elapsed_seconds")
        changed = 0
        current_segment_started = execution_boundary is not None or reset_detected or any(
            completed <= candidates[0]["completed"] for completed in existing
        )

        pending_samples: list[dict[str, Any]] = []

        def enrich_metrics(sample: Mapping[str, Any], metrics: Mapping[str, Any]) -> int:
            if sample.get("_pending"):
                known = sample["evidence_json"].setdefault("metrics", {})
                additions = {key: value for key, value in metrics.items() if key not in known}
                known.update(additions)
                return int(bool(additions))
            # All persisted evidence was fetched above. Avoid a transaction for
            # every unchanged row of an append-only tail on every progress poll.
            known = (sample.get("evidence_json") or {}).get("metrics", {})
            additions = {key: value for key, value in metrics.items() if key not in known}
            return int(bool(additions) and self.database.enrich_training_progress_metrics(sample["id"], additions))

        for record in candidates:
            completed = int(record["completed"])
            if completed in existing:
                current_segment_started = True
                if jsonl:
                    changed += enrich_metrics(existing[completed], record["metrics"])
                continue
            if jsonl and not current_segment_started and completed in prior:
                previous = prior[completed]
                previous_metrics = (previous.get("evidence_json") or {}).get("metrics", {})
                if all(
                    record["metrics"][key] == value
                    for key, value in previous_metrics.items() if key in record["metrics"]
                ):
                    # Append-only logs retain the previous execution's prefix.
                    # Enrich those observations instead of relabeling them as a restart.
                    changed += enrich_metrics(previous, record["metrics"])
                    continue
            current_segment_started = True
            recorded_at = observed_now
            elapsed = record.get("elapsed_seconds")
            if latest_elapsed is not None and elapsed is not None and latest_elapsed >= elapsed:
                recorded_at -= timedelta(seconds=latest_elapsed - elapsed)
            sample = dict(
                restart_count=restart_count,
                completed=completed,
                total=int(record["total"]),
                unit=contract.unit,
                source_kind=source.kind,
                evidence={
                    "declaration_origin": declaration_origin,
                    "stream": "file" if jsonl else source.stream,
                    "path": path,
                    "elapsed_seconds": elapsed,
                    **({"execution_boundary": execution_boundary["boundary_path"]}
                       if execution_boundary else {}),
                    **({"metrics": record["metrics"]} if jsonl else {}),
                },
                recorded_at=_progress_iso(recorded_at),
            )
            pending_samples.append(sample)
            existing[completed] = {"_pending": True, "evidence_json": sample["evidence"]}
            changed += 1
        self.database.record_training_progress_samples(str(run["id"]), attempt_id, pending_samples)
        if final_read:
            # Mark a terminal source consumed only after every sample is persisted.
            final_reads.add(final_key)
            self._training_progress_final_reads = final_reads
            final_failures.pop(final_key, None)
        return changed

    @staticmethod
    def _training_progress_metrics(sample: Mapping[str, Any]) -> dict[str, float | int]:
        evidence = sample.get("evidence") or sample.get("evidence_json") or {}
        if not isinstance(evidence, Mapping) or not isinstance(evidence.get("metrics"), Mapping):
            return {}
        # Scheduling bookkeeping stays in Skynet; tracking charts contain the
        # actual metrics emitted by the training code.
        return {
            str(key): value for key, value in evidence["metrics"].items()
            if is_scalar(value) and key not in INTERNAL_PROGRESS_METRICS
        }

    _training_tracking_locks_guard = threading.Lock()
    _training_tracking_locks: dict[str, Any] = {}

    def _publish_training_progress_tracking(
        self,
        run_id: str,
        *,
        providers: list[Any] | None = None,
        include_native: bool = False,
        raise_on_error: bool = False,
    ) -> int:
        # Reconciliation and manual sync may overlap. Serialize the read/delta/
        # append sequence per run, including calls made by another service instance.
        with self._training_tracking_locks_guard:
            lock = self._training_tracking_locks.setdefault((self.database.url is not None, run_id), self.database.operation_lock("training-metrics:" + run_id))
        with lock:
            return self._publish_training_progress_tracking_locked(
                run_id, providers=providers, include_native=include_native, raise_on_error=raise_on_error
            )

    def _publish_training_progress_tracking_locked(
        self,
        run_id: str,
        *,
        providers: list[Any] | None = None,
        include_native: bool = False,
        raise_on_error: bool = False,
    ) -> int:
        """Publish persisted progress samples without adapter-specific knowledge."""

        run = self.database.get_run(run_id, include_details=False)
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
                    bridge: Any = self._mlflow_bridge(
                        LOCAL_CAPSULE_ROOT / run_id, self._mlflow_settings(provider)
                    )
                elif name == "wandb":
                    bridge = self._wandb_bridge(
                        LOCAL_CAPSULE_ROOT / run_id, replace(self._wandb_settings(provider), auto_flush=False)
                    )
                else:
                    raise ValueError(f"unsupported tracking provider: {name}")
                if not bridge.binding(run_id):
                    if raise_on_error:
                        raise TrackingRequestError(f"{name} run binding is not ready for final metrics")
                    continue
                emitted: dict[str, set[str]] = {}
                for key, names in bridge.metric_names_by_idempotency_key().items():
                    sample_key = key.split(":metrics:", 1)[0]
                    emitted.setdefault(sample_key, set()).update(names)
                pending_metrics: list[dict[str, Any]] = []
                for sample in samples:
                    sample_id = str(sample.get("id") or "")
                    if not sample_id:
                        continue
                    idempotency_key = f"training-progress:{sample_id}"
                    prior_names = emitted.get(idempotency_key, set())
                    metrics = {
                        key: value
                        for key, value in self._training_progress_metrics(sample).items()
                        if key not in prior_names
                    }
                    if not metrics:
                        continue
                    sample_key = idempotency_key
                    if prior_names:
                        # Enriched samples append only previously omitted metrics,
                        # retaining their original step/time and existing curves.
                        idempotency_key += ":metrics:" + content_sha256(canonical_json(metrics))
                    pending_metrics.append({
                        "metrics": metrics,
                        "step": int(sample.get("completed") or 0),
                        "timestamp_ms": progress_timestamp_ms(sample.get("recorded_at")),
                        "idempotency_key": idempotency_key,
                    })
                    emitted.setdefault(sample_key, set()).update(metrics)
                    published += 1
                if name == "wandb":
                    bridge.log_metrics_batch(run_id, pending_metrics)
                    report = bridge.drain_spool(limit=TRACKING_FLUSH_EVENT_LIMIT)
                    if report.errors:
                        raise TrackingRequestError(report.errors[0])
                else:
                    for sample in pending_metrics:
                        bridge.log_metrics(run_id, **sample)
            except Exception as error:
                self._tracking_failure(name, run, error)
                if raise_on_error:
                    raise
        return published

    def _ingest_evaluation_progress(self, evaluation: Mapping[str, Any]) -> None:
        """Import changed canonical episode records while an evaluation is running."""
        if evaluation.get("status") != "RUNNING":
            return

        # The list and detail endpoints can be polled together. Keep that from
        # causing duplicate SSH reads while still making progress visibly live.
        now = time.monotonic()
        last_reads = getattr(self, "_evaluation_progress_last_reads", {})
        if now - float(last_reads.get(evaluation["id"], 0.0)) < EVALUATION_PROGRESS_READ_INTERVAL_SECONDS:
            return
        last_reads[evaluation["id"]] = now
        self._evaluation_progress_last_reads = last_reads

        run = self.database.get_run(str(evaluation["run_id"]), execution_stage_ids=[evaluation["stage_id"]])
        if not run:
            return
        attempt = _latest_stage_attempt(run, evaluation.get("stage_id"))
        if attempt is None:
            return
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
            if status not in {"RUNNING", "SUCCEEDED", "FAILED", "TIMEOUT"}:
                continue
            if success is not None and not isinstance(success, bool):
                continue
            if not isinstance(metrics, dict):
                continue
            observed[key] = (episode, record)

        if not observed:
            return
        current_evaluation = self.database.get_evaluation(str(evaluation["id"]))
        if not current_evaluation or current_evaluation.get("status") not in {"PENDING", "SUBMITTED", "RUNNING"}:
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
                started_at=current.get("started_at") or (recorded_at if isinstance(recorded_at, str) else utc_now()),
                completed_at=(recorded_at if isinstance(recorded_at, str) else utc_now())
                    if desired["status"] != "RUNNING" else None,
                expected_parent_states=("PENDING", "SUBMITTED", "RUNNING"),
            )

    def _ingest_evaluation_result(
        self, attempt: Mapping[str, Any]
    ) -> tuple[str | None, int]:
        evaluation = self._evaluation_for_stage(attempt["run_id"], attempt["stage_id"])
        if not evaluation:
            return "evaluation record is missing", 0
        # Only a gateway that answers can say the result is missing; a failed
        # read raises ClusterError and the caller reads again later.
        _, content = self.cluster.read_optional_file(
            evaluation["result_path"],
            attempt["gateway"] or "auto",
            max_bytes=20_000_000,
        )
        if content is None:
            return "canonical result file is missing", 0
        try:
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
            except INTEGRITY_ERRORS:
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
                except INTEGRITY_ERRORS:
                    pass
            # Optional telemetry is reconciled by its independent worker. A
            # busy GPU-metric lock or unavailable provider cannot invalidate a
            # fully checked and persisted episode ledger.
            run_id = str(attempt["run_id"])
            run = self.database.get_run(run_id)
            if run:
                spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
                active = {
                    str(self._tracking_provider_value(provider, "provider"))
                    for provider in self._active_tracking_providers(spec)
                }
                for binding in self.database.list_tracking_bindings("run", run_id):
                    if binding["provider"] not in active or binding["status"] == "BLOCKED":
                        continue
                    self.database.upsert_tracking_binding(
                        binding["provider"], "run", run_id,
                        remote_id=binding.get("remote_id"),
                        remote_url=binding.get("remote_url"),
                        status="QUEUED", metadata=binding.get("metadata_json") or {},
                        last_error=binding.get("last_error"),
                    )
            return None, len(result.episodes)
        except INTEGRITY_ERRORS as error:
            return sanitize(str(error)), 0
        except (*DATABASE_ERRORS, ConnectionError):
            raise  # The database failed, not the result: the caller ingests it again.
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

    @staticmethod
    def _tracking_terminal_pending(binding: Mapping[str, Any], status: str, provider: str) -> bool:
        if binding.get("status") == "BLOCKED":
            return False
        return (binding.get("status") != status or (
            provider == "wandb" and
            (binding.get("metadata_json") or {}).get("terminal_sync_protocol") != WANDB_TERMINAL_SYNC_PROTOCOL
        ))

    def _sync_tracking_outputs(
        self,
        run_id: str,
        status: str | None = None,
        *,
        providers: list[Any] | None = None,
        central_authoritative: bool = False,
    ) -> None:
        # Evaluation stages can hold large immutable plans, and their attempts
        # must never replace the training job identity in training telemetry.
        with self.database.connection() as connection:
            training_stage_ids = [row[0] for row in connection.execute(
                "SELECT id FROM workflow_stages WHERE run_id=? AND stage_type='TRAIN'", (run_id,)
            ).fetchall()]
        run = self.database.get_run(run_id, execution_stage_ids=training_stage_ids)
        if not run:
            return
        if status is None:
            status = {"SUCCEEDED": "FINISHED", "FAILED": "FAILED", "CANCELLED": "KILLED"}.get(run["status"])
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
            if attempt.get("stage_id") in training_stage_ids and attempt.get("slurm_job_id"):
                latest_job_id = str(attempt["slurm_job_id"])
                break
        self._start_tracking(
            spec,
            run,
            latest_job_id,
            providers=providers,
            auto_flush=False,
        )
        self._publish_training_progress_tracking(
            run_id,
            providers=providers,
            include_native=central_authoritative,
            raise_on_error=bool(status),
        )
        sync_gpu_statistics(self, run, LOCAL_CAPSULE_ROOT, force=True, raise_on_error=run["status"] == "SUCCEEDED")
        metrics, links = self._final_tracking_payload(
            self.database.get_run(run_id, execution_stage_ids=training_stage_ids) or run
        )
        final_key = f"final:{run_id}:{latest_job_id}:{status}" if status else None
        for provider in providers:
            name = str(self._tracking_provider_value(provider, "provider"))
            if name in native_providers:
                # The trainer owns history and final state remotely. Only drain
                # the central metadata queue here; never issue a second finish.
                if status:
                    try:
                        if name == "wandb":
                            bridge = self._wandb_bridge(LOCAL_CAPSULE_ROOT / run_id, replace(self._wandb_settings(provider), auto_flush=False))
                        else:
                            bridge = self._mlflow_bridge(LOCAL_CAPSULE_ROOT / run_id, replace(self._mlflow_settings(provider), auto_flush=False))
                        report = bridge.drain_spool(limit=TRACKING_FLUSH_EVENT_LIMIT)
                        existing = next((item for item in self.database.list_tracking_bindings("run", run_id) if item["provider"] == name), {})
                        self.database.upsert_tracking_binding(
                            name, "run", run_id,
                            remote_id=existing.get("remote_id"), remote_url=existing.get("remote_url"),
                            status=status if not report.errors and report.remaining == 0 else "QUEUED",
                            metadata={**(existing.get("metadata_json") or {}),
                                      **({"terminal_sync_protocol": WANDB_TERMINAL_SYNC_PROTOCOL, "terminal_state_owner": "native"}
                                         if name == "wandb" and not report.errors and report.remaining == 0 else {})},
                            last_error=report.errors[0] if report.errors else None,
                        )
                    except Exception as error:
                        self._tracking_failure(name, run, error)
                continue
            try:
                if name == "mlflow":
                    bridge: Any = self._mlflow_bridge(
                        LOCAL_CAPSULE_ROOT / run_id, replace(self._mlflow_settings(provider), auto_flush=False)
                    )
                elif name == "wandb":
                    bridge = self._wandb_bridge(
                        LOCAL_CAPSULE_ROOT / run_id, replace(self._wandb_settings(provider), auto_flush=False)
                    )
                else:
                    raise ValueError(f"unsupported tracking provider: {name}")
                if metrics:
                    digest = ":metrics:" + content_sha256(canonical_json(metrics))
                    # The key names the job that was latest when the run ended. The same final
                    # metrics published earlier under another job are not a new history row.
                    if not (final_key and any(
                        key.startswith(f"final:{run_id}:") and key.endswith(digest)
                        for key in bridge.metric_idempotency_keys()
                    )):
                        bridge.log_metrics(
                            run_id, metrics, **({"idempotency_key": final_key + digest} if final_key else {}))
                artifact_links = [{**link,
                    **({"idempotency_key": final_key + ":artifact:" + content_sha256(canonical_json(link))} if final_key else {})}
                    for link in links]
                if name == "wandb":
                    bridge.log_artifact_links(run_id, artifact_links)
                else:
                    for link in artifact_links:
                        bridge.log_artifact_link(run_id, **link)
                if status:
                    bridge.finish_run(run_id, status=status, idempotency_key=final_key + (
                        ":finish:file-stream-v1" if name == "wandb" else ":finish"))
                report = bridge.drain_spool(limit=TRACKING_FLUSH_EVENT_LIMIT)
                delivered = not report.errors and report.remaining == 0
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
                        if delivered
                        else "QUEUED"
                    ),
                    metadata={**(existing.get("metadata_json") or {}),
                              **({"terminal_sync_protocol": WANDB_TERMINAL_SYNC_PROTOCOL, "terminal_state_owner": "skynet"}
                                 if name == "wandb" and status and delivered else {})},
                    last_error=report.errors[0] if report.errors else None,
                )
            except Exception as error:
                self._tracking_failure(name, run, error)

    def _finish_tracking(self, run_id: str, status: str) -> None:
        self._sync_tracking_outputs(run_id, status=status)

    def _recover_manual_resume_checkpoint(
        self, run: Mapping[str, Any], stage: Mapping[str, Any],
        pinned: Mapping[str, Any], gateway: str,
    ) -> Mapping[str, Any] | None:
        """Register verified runner evidence only during explicit manual resume.

        Automatic retries already read latest.json directly. Missing receipts
        permit initial replay; invalid or unavailable evidence must not.
        """
        plan = pinned["plan"]
        request = {
            "run_id": str(run["id"]), "root": str(self._run_directory(str(run["id"]))),
            "contract": {
                "candidate_globs": list(plan.get("checkpoint_globs") or []),
                "candidate_kind": plan.get("checkpoint_candidate_kind", "any"),
                "basename_regex": plan.get("checkpoint_basename_regex"),
                "inference_required_globs": list(plan.get("checkpoint_inference_required_globs") or []),
            },
            "attempts": [
                {"id": str(item["id"]), "job_id": str(item.get("slurm_job_id") or "")}
                for item in _training_stage_attempts(run, stage)
            ],
        }
        try:
            _, output = self.cluster.run_with_fallback(
                "python3 - " + shlex.quote(canonical_json(request))
                + " # SKYNET_MANUAL_RESUME_CHECKPOINT", gateway,
                stdin=_MANUAL_RESUME_CHECKPOINT_PROBE, timeout=900,
            )
            evidence = json.loads(output)
        except (ClusterError, ValueError) as error:
            raise ValueError(f"Cannot verify manual resume checkpoint: {sanitize(str(error))}") from error
        if not isinstance(evidence, Mapping) or type(evidence.get("present")) is not bool:
            raise ValueError("Invalid manual resume checkpoint verification response")
        if not evidence["present"]:
            return None
        payload = evidence.get("receipt")
        if not isinstance(payload, Mapping) or evidence.get("identity") != {
            key: payload.get(key) for key in ("sha256", "size_bytes", "file_count", "is_directory")
        }:
            raise ValueError("Manual resume checkpoint verification is incomplete")
        producer = evidence.get("producer")
        if producer is not None and producer not in {item["id"] for item in request["attempts"]}:
            raise ValueError("Manual resume checkpoint has an unknown producing attempt")
        return self.database.create_checkpoint(
            str(run["id"]), checkpoint_type="FULL_RESUME", path=payload["path"],
            produced_by_attempt_id=producer, training_step=payload.get("training_step"),
            sha256=payload["sha256"], size_bytes=payload["size_bytes"], is_resumable=True,
            is_selected_for_inference=False, status="AVAILABLE", metadata=payload,
        )

    def retry_run(self, run_id: str, gateway: str) -> dict[str, Any]:
        with self._reconcile_lock:
            run = self.database.get_run(run_id)
            if not run:
                raise KeyError("Run not found")
            data_selection.assert_available(self.database, run)
            action = self.run_manual_actions(run)["resume"]
            if not action["enabled"]:
                raise ValueError(str(action["reason"]))
            stage = next(item for item in reversed(run["stages"]) if item["stage_type"] == "TRAIN")
            checkpoint = _resumable_checkpoint(run)
            pinned_stage, pinned_stage_error = _pinned_training_execution(
                run, stage, prefer_initial_attempt=True
            )
            if checkpoint is None and pinned_stage and (pinned_stage.get("plan") or {}).get("resume_argv"):
                checkpoint = self._recover_manual_resume_checkpoint(run, stage, pinned_stage, gateway)
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
            data_selection.assert_available(self.database, source)
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
                run_directory=f"{self.work_root}/jobs/runs/pending",
                status="PENDING",
                restarted_from_run_id=run_id,
            )
            self.database.update_run(
                rerun["id"],
                run_directory=f"{self.work_root}/jobs/runs/{rerun['id']}",
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
        run = self.database.get_run(run_id, include_payloads=False)
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
        run = self.database.get_run(run_id, include_payloads=False)
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
                refreshed_evaluation["attempts"] = _stage_attempts(
                    parent, refreshed_evaluation.get("stage_id")
                )
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

    def retry_evaluation_submission(self, evaluation_id: str, gateway: str) -> dict[str, Any]:
        """Retry a frozen evaluation after verifying submission or terminal accounting."""
        evaluation = self.database.get_evaluation(evaluation_id)
        if not evaluation:
            raise KeyError("Evaluation not found")
        run = self.database.get_run(evaluation["run_id"], execution_stage_ids=[evaluation["stage_id"]])
        action = self.evaluation_manual_actions(evaluation, run)["retry_submission"]
        if not action["enabled"]:
            raise ValueError(str(action["reason"]))
        latest = _latest_stage_attempt(run, evaluation["stage_id"])
        selected_gateway = latest["gateway"] or gateway
        recovered = None
        job_id = latest.get("slurm_job_id")
        if job_id:
            # Accepted jobs require fresh terminal accounting. A stale FAILED
            # database row is not permission to duplicate a live/requeued job.
            snapshot = self.cluster.job_status_snapshot([str(job_id)], selected_gateway)
            record = snapshot.statuses.get(str(job_id))
            if record is None and snapshot.controller_error is None:
                record = self._exit_record_status({**latest, "run_id": run["id"]})
            state = str((record or {}).get("State") or "UNKNOWN").upper()
            if state not in ATTEMPT_FAILURE_STATES:
                raise ValueError(f"Retry requires a confirmed failed Slurm job; {job_id} is {state}")
        else:
            # A failed lookup must never be interpreted as no Slurm receipt.
            recovered = self.cluster.recover_submission(run["id"], latest["id"], selected_gateway)
        with self._reconcile_lock:
            current = self.database.get_evaluation(evaluation_id)
            parent = self.database.get_run(run["id"], include_payloads=False)
            newest = _latest_stage_attempt(parent, evaluation["stage_id"])
            action = self.evaluation_manual_actions(current, parent)["retry_submission"]
            if not action["enabled"] or newest["id"] != latest["id"]:
                return {"evaluation": current, "skipped": "Submission state changed while checking the receipt"}
            transition = self.database.transition_workflow_state(
                stage_id=evaluation["stage_id"],
                stage_updates={"status": "SUBMITTING" if recovered else "RETRY_PENDING", "completed_at": None},
                evaluation_id=evaluation_id,
                evaluation_updates={"status": "SUBMITTING" if recovered else "RETRY_PENDING", "completed_at": None},
                expected_stage_statuses=("FAILED",),
                event={"entity_type": "evaluation", "entity_id": evaluation_id,
                       "event_type": "EVALUATION_RUNTIME_RETRY_REQUESTED" if job_id else "EVALUATION_SUBMISSION_RETRY_REQUESTED", "old_status": "FAILED",
                       "new_status": "SUBMITTING" if recovered else "RETRY_PENDING",
                       "details": {"previous_attempt_id": latest["id"], "receipt_recovered": bool(recovered)}},
            )
            if transition.get("applied") is False:
                return {"evaluation": self.database.get_evaluation(evaluation_id), "skipped": "Evaluation state changed"}
            if recovered:
                self._record_recovered_submission({**latest, "run_id": run["id"],
                    "stage_type": "EVALUATE", "stage_status": "SUBMITTING",
                    "evaluation_id": evaluation_id, "run_started_at": run.get("started_at")}, recovered)
        if not recovered:
            # Reuse the same stage, model, dataset, budget, and episode ledger.
            self._submit_stage(run["id"], evaluation["stage_id"], selected_gateway)
        return {"evaluation": self.database.get_evaluation(evaluation_id)}

    def reread_evaluation_result(self, evaluation_id: str) -> dict[str, Any]:
        """Validate and ingest result.json of a job that already succeeded.

        No episode is rerun: the same canonical result file and checks as the
        automatic completion path decide the outcome."""
        evaluation = self.database.get_evaluation(evaluation_id)
        if not evaluation:
            raise KeyError("Evaluation not found")
        stage_ids = [evaluation["stage_id"]]
        run = self.database.get_run(evaluation["run_id"], execution_stage_ids=stage_ids)
        action = self.evaluation_manual_actions(evaluation, run)["reread_result"]
        if not action["enabled"]:
            raise ValueError(str(action["reason"]))
        with self._reconcile_lock:
            current = self.database.get_evaluation(evaluation_id)
            parent = self.database.get_run(run["id"], execution_stage_ids=stage_ids)
            if not self.evaluation_manual_actions(current, parent)["reread_result"]["enabled"]:
                return {"evaluation": current, "skipped": "Evaluation state changed"}
            latest = _latest_stage_attempt(parent, evaluation["stage_id"])
            error, completed = self._ingest_evaluation_result({**latest, "run_id": parent["id"]})
            if error:
                raise ValueError(f"The evaluation result is still unreadable: {error}")
            transition = self.database.transition_workflow_state(
                stage_id=evaluation["stage_id"],
                stage_updates={"status": "SUCCEEDED"},
                evaluation_id=evaluation_id,
                evaluation_updates={"status": "SUCCEEDED", "progress_completed": completed},
                expected_stage_statuses=("FAILED",),
                event={"entity_type": "evaluation", "entity_id": evaluation_id,
                       "event_type": "EVALUATION_RESULT_REREAD", "old_status": "FAILED",
                       "new_status": "SUCCEEDED",
                       "details": {"attempt_id": latest["id"], "job_id": latest.get("slurm_job_id"),
                                   "completed_episodes": completed}},
            )
            if transition.get("applied") is False:
                return {"evaluation": self.database.get_evaluation(evaluation_id),
                        "skipped": "Evaluation state changed"}
        return {"evaluation": self.database.get_evaluation(evaluation_id)}

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

    def _evaluation_source(self, training_spec: ExperimentSpec):
        document = training_spec.model_dump(mode="json", by_alias=True)
        source = copy.deepcopy(document["source"])
        # Independent policy loaders use the training receipt. Updating the
        # trainer registry cannot change how an old checkpoint is interpreted.
        if policy_loader(document) and source.get("adapter_manifest"):
            return source, AdapterManifest.model_validate(source["adapter_manifest"])
        for key in ("adapter_id", "adapter_version_id", "adapter_version", "adapter_manifest", "adapter_manifest_sha256"):
            source.pop(key, None)
        source, manifest, _ = self._snapshot_adapter(source, training_spec.source.adapter)
        return source, manifest

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
        uses_isaac = uses_isaac_sim({"suite": {"config": suite_config}}, runtime=profile_snapshot)
        for version_name in (("python", "isaac_sim", "isaac_lab") if uses_isaac else ("python",)):
            if not str(versions.get(version_name) or "").strip():
                blockers.append(
                    f"evaluation runtime profile {runtime_profile_id} must pin {version_name}"
                )
        runtime_environment = copy.deepcopy(dict(profile.environment))
        import os

        operator_environment: dict[str, str] = {}
        if uses_isaac and os.environ.get("OMNI_KIT_ACCEPT_EULA") == "YES":
            operator_environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
        elif uses_isaac:
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
        now = time.monotonic()
        if not refresh:
            with cache_lock:
                cached = cache.get(cache_key)
            if cached is not None and now - cached[0] < (
                RUNTIME_READINESS_BLOCKED_CACHE_SECONDS if cached[2] else RUNTIME_READINESS_CACHE_SECONDS
            ):
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
        run: Mapping[str, Any] | None = None,
        target_dataset_id: str | None = None,
        unseen_embodiment: bool = False,
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

        try:
            suite = attach_evaluation_target(self.database, suite, (run or {}).get("resolved_spec_json") or {}, target_dataset_id, unseen_embodiment)
            suite = bind_suite_to_dataset(suite, (run or {}).get("resolved_spec_json") or {})
        except ValueError as error:
            errors["tasks"] = str(error)
            return suite, str(suite["evaluator_adapter"]), [], errors

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
                else requested_tasks or suite_config.get("default_tasks") or catalog_tasks
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
        episode_limit = suite.get("config_json", {}).get("maximum_episodes_per_task")
        if episode_limit is not None and episodes_per_task > episode_limit:
            raise ValueError(f"The dataset has {episode_limit} available episodes; choose at most {episode_limit}")
        training_spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
        training_adapter = self._adapter_identity(training_spec)
        training_document = training_spec.model_dump(mode="json", by_alias=True)

        evaluator_document = copy.deepcopy(training_document)
        evaluator_document["source"], evaluator_manifest = self._evaluation_source(training_spec)
        compatibility, _ = inspect_compatibility(training_document, evaluator_manifest, suite, checkpoint)
        if manual_argv:
            compatibility.update(status="unknown", label="Custom command: unverified", ready=False,
                                 runtime_verification="custom_command")
            compatibility["messages"] = ["Custom commands are responsible for their own model and environment compatibility checks."]
        evaluator_manifest = compose_evaluator(training_document, evaluator_manifest, suite)
        compatibility["implementation_sha256"] = adapter_manifest_sha256(evaluator_manifest)
        if resources is not None:
            evaluator_document["resources"] = resources.model_dump(mode="json", by_alias=True)
        # Each evaluation worker owns one GPU. Apply the cluster CPU policy
        # before multiplying worker resources into the Slurm allocation.
        evaluator_document["resources"] = ResourceSpec.model_validate(
            evaluator_document["resources"]
        ).with_gpu_cpu_policy(1).model_dump(mode="json", by_alias=True)
        # Evaluation uses its episode ledger, not the training checkpoint signal policy.
        evaluator_document["train"]["checkpoint"]["auto_resume"] = False
        worker_resources = copy.deepcopy(evaluator_document["resources"])
        supports_workers = any(
            entry.environment == environment and suite["name"] in entry.suites
            and (entry.maximum_parallelism or 1) > 1
            for entry in evaluator_manifest.evaluations
        )
        if supports_workers:
            total_resources = evaluator_document["resources"]
            total_resources["gpu"].update(mode="explicit", count=parallelism)
            for key in ("cpus_per_task", "memory_gb"):
                total_resources[key] = worker_resources[key] * parallelism
        evaluator_spec = ExperimentSpec.model_validate(evaluator_document)
        evaluator_adapter = self._adapter_identity(evaluator_spec)

        evaluation_root = f"{self._run_paths(str(run['id'])).evaluation}/runs/{execution_key}"
        source_document = training_document["source"]
        suite_config = copy.deepcopy(suite["config_json"])
        context = {
            "schema_version": "skynet.evaluation-context/v1",
            "execution_key": execution_key,
            "compatibility": compatibility,
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
            "worker_resources": worker_resources,
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
        if suite_config.get("target_dataset"):
            from .simulation_profiles import freeze_target_simulation_profile
            context["target_dataset"] = copy.deepcopy(suite_config["target_dataset"])
            context["unseen_embodiment"] = bool(suite_config.get("unseen_embodiment"))
            data_selection.assert_available(self.database, context["target_dataset"])
            context["target_simulation_profile"] = freeze_target_simulation_profile(
                self.database, self.cluster, context["target_dataset"], Path(__file__).resolve().parent.parent
            )
        if suite_config.get("initial_state") == "single_training_episode":
            context["recorded_episode_sources"] = recorded_episode_sources(
                self.database, self.cluster, training_document
            )
        elif environment == "isaac_lab" and not context.get("target_dataset"):
            # A reference demonstration is independent of the simulator reset policy.
            # Only an unambiguous, verified single training recording is selected.
            try:
                context["recorded_episode_sources"] = recorded_episode_sources(
                    self.database, self.cluster, training_document
                )
            except ValueError as error:
                context["demonstration_unavailable"] = str(error)
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
        if uses_isaac_sim(context, runtime_profile_id=runtime_profile_id or None,
                          runtime=evaluator_spec.runtime.model_dump(mode="json")):
            placement_resources = evaluator_spec.resources
            if resources is None:
                # A training-node pin is not a user-selected evaluation node.
                placement_resources = placement_resources.model_copy(update={
                    "node": placement_resources.node.model_copy(update={"mode": "auto", "name": None}),
                })
            placed_resources = resolve_evaluation_resources(
                placement_resources, context,
                runtime_profile_id=runtime_profile_id or None,
                runtime=evaluator_spec.runtime.model_dump(mode="json"),
                gpu_count=resolve_gpu_count(evaluator_spec, plan),
                gpu_type=resolve_gpu_type(evaluator_spec, plan),
                node_inventory=(self.cluster.node_names(evaluator_runtime_gateway)
                                if placement_resources.node.mode == "auto"
                                and not placement_resources.node.eligible_names else None),
            )
            evaluator_spec = evaluator_spec.model_copy(update={"resources": placed_resources})
            worker_resources["node"] = placed_resources.node.model_dump(mode="json")
            worker_resources["gpu"]["type"] = placed_resources.gpu.gpu_type
        if runtime_profile_id:
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
        # Manual commands still carry the selected simulator's placement policy,
        # even when the adapter has no registered evaluator command.
        plan.native_config["canonical_evaluation"] = copy.deepcopy(context)
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
        if not manual_argv:
            plan.blockers = list(dict.fromkeys([*plan.blockers, *compatibility["messages"]]))
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
            run = self.database.get_run(normalized_run_id, execution_stage_ids=[])
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
        target_dataset_id: str | None = None,
        unseen_embodiment: bool = False,
        environment: str | None = None,
        tasks: list[str] | None = None,
        episodes_per_task: int = DEFAULT_EVALUATION_EPISODES,
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
                suite_id, environment, list(tasks or []), run, target_dataset_id, unseen_embodiment
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
                    "target_dataset": suite_config.get("target_dataset"),
                    "unseen_embodiment": suite_config.get("unseen_embodiment", False),
                    "tasks": resolved_tasks,
                    "seeds": list(seeds or DEFAULT_EVALUATION_SEEDS),
                    "resources": resources.model_dump(mode="json", by_alias=True) if resources else None,
                }
            )[:20]
            try:
                if not argv:
                    training_spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
                    _, manifest = self._evaluation_source(training_spec)
                    compatibility, _ = inspect_compatibility(
                        training_spec.model_dump(mode="json", by_alias=True), manifest, suite, checkpoint
                    )
                    if not compatibility["ready"]:
                        validation.update(valid=False, compatibility=compatibility,
                                          plan_blockers=compatibility["messages"],
                                          plan_message="; ".join(compatibility["messages"]))
                        return validation
                evaluator_spec, plan, context, _, evaluator_adapter, plan_source = (
                    self._resolve_evaluation_implementation(
                        run,
                        checkpoint,
                        suite,
                        environment=str(resolved_environment),
                        tasks=resolved_tasks,
                        seeds=list(seeds or DEFAULT_EVALUATION_SEEDS),
                        episodes_per_task=episodes_per_task,
                        parallelism=parallelism,
                        headless=headless,
                        execution_key=validation_key,
                        resources=resources,
                        manual_argv=list(argv or []),
                        manual_resume_argv=list(resume_argv or []),
                        verify_evaluator_runtime=False,
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
                        "compatibility": context.get("compatibility"),
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
        evaluation = self._create_evaluation(request, dispatch=False)
        if evaluation["status"] == "PENDING":
            # The durable stage and complete ledger now exist. Submission owns
            # the stage through claim_stage_and_create_job_attempt; SSH/Slurm
            # acknowledgement must not hold the workspace-wide planning lock.
            submission = self._submit_stage(evaluation["run_id"], evaluation["stage_id"], request.gateway)
            refreshed = self.database.get_evaluation(evaluation["id"])
            assert refreshed is not None
            for key in ("blocker", "evaluator_implementation"):
                refreshed[key] = evaluation[key]
            refreshed["submission"] = submission
            return refreshed
        return evaluation

    def _create_evaluation(self, request: EvaluationRequest, *, dispatch: bool = True) -> dict[str, Any]:
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
        data_selection.assert_available(self.database, run)
        busy_reason = _evaluation_busy_reason(run)
        if busy_reason:
            raise ValueError(busy_reason)
        suite, canonical_environment, tasks, suite_errors = (
            self._resolve_evaluation_suite_selection(
                request.suite_id, request.environment, request.tasks, run, request.target_dataset_id, request.unseen_embodiment
            )
        )
        if suite_errors:
            raise ValueError(next(iter(suite_errors.values())))
        assert suite is not None
        assert canonical_environment is not None

        if not request.argv:
            training_spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
            _, evaluator_manifest = self._evaluation_source(training_spec)
            compatibility, _ = inspect_compatibility(
                training_spec.model_dump(mode="json", by_alias=True), evaluator_manifest, suite, checkpoint
            )
            if not compatibility["ready"]:
                raise ValueError("; ".join(compatibility["messages"]))

        checkpoint_id = checkpoint["id"]
        checkpoint_path = checkpoint["path"]
        # Planning needs a stable output namespace, not a durable stage. A
        # failed profile/plan must not leave an invisible target-data reference.
        stage_id = new_id()
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
            execution_key=stage_id,
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
        # Runtime/cluster probes above are read-only and must not serialize all
        # workspace submissions. Revalidate mutable prerequisites immediately
        # before publishing the stage and episode ledger under the lifecycle lock.
        with self._reconcile_lock:
            current_target, current_run, current_checkpoint = self._resolve_evaluation_target(
                run["id"], checkpoint_path
            )
            if not current_target["run_valid"] or current_run is None:
                raise ValueError("Training run changed while planning evaluation")
            if not current_target["checkpoint_valid"] or current_checkpoint is None or current_checkpoint["id"] != checkpoint_id:
                raise ValueError("Evaluation checkpoint changed while planning evaluation")
            data_selection.assert_available(self.database, current_run)
            busy_reason = _evaluation_busy_reason(current_run)
            if busy_reason:
                raise ValueError(busy_reason)
            stage = self.database.create_stage(
                run["id"],
                stage_id=stage_id,
                stage_type="EVALUATE",
                name=f"eval-{suite['name']}-{stage_id}",
                status=status,
                auto_resume=request.auto_resume,
                max_attempts=request.max_attempts,
                resolved_config={
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
            self.database.initialize_evaluation_episodes(evaluation["id"])
        submission = None
        if status == "PENDING" and dispatch:
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


service = LazyService(lambda: WorkspaceServices(PipelineService()))
router = APIRouter(prefix="/api", dependencies=[Depends(require_workspace_records)])


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, KeyError):
        return HTTPException(status_code=404, detail=str(error).strip("'"))
    if isinstance(error, INTEGRITY_ERRORS):
        message = str(error)
        identities = {
            "data_resources_provider_namespace_source_key_key": "A resource with this provider, namespace, and source key already exists. Open that resource or choose another source key.",
            "data_resource_versions.resource_id": "This revision and format are already registered for the resource. Use the existing immutable version or choose a new revision.",
            "data_bundles.name": "This bundle name and version already exist. Open the existing bundle or choose a new version.",
        }
        detail = next((text for key, text in identities.items() if key in message), "This record conflicts with an existing record. Refresh and check its identity before retrying.")
        return HTTPException(status_code=409, detail=detail)
    if isinstance(error, (ValueError, SlurmCompileError)):
        return HTTPException(status_code=422, detail=str(error))
    if isinstance(error, ClusterError):
        return HTTPException(status_code=503, detail=str(error))
    return HTTPException(status_code=500, detail=str(error))


def _display_payload(value, *, metadata=False):
    """A screen projection, never an execution/validation input or stored receipt.

    Full bodies remain available in the existing entity detail endpoints. Keep
    manifest IDs/hashes so a summary always identifies its authoritative source.
    """
    if isinstance(value, list):
        return [_display_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == "capsule_files" or (metadata and key == "shared_artifacts"):
            continue
        if metadata and key == "episodes" and isinstance(item, list):
            result[key] = len(item)
            continue
        result[key] = _display_payload(item, metadata=key in {"metadata", "metadata_json"})
    return result


@router.get("/data/datasets")
def list_datasets(include_archived: bool = Query(default=False)) -> dict[str, Any]:
    return {"datasets": _display_payload(service.database.list_datasets(include_archived=include_archived))}


@router.get("/data/datasets/{version_id}")
def get_dataset(version_id: str) -> dict[str, Any]:
    dataset = service.database.get_dataset(version_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"dataset": dataset}


@router.patch("/data/datasets/{version_id}")
def edit_dataset(version_id: str, request: DatasetEditRequest) -> dict[str, Any]:
    try:
        return {"dataset": service.database.update_dataset(
            version_id, **request.model_dump(mode="python", exclude_unset=True))}
    except Exception as error:
        raise _http_error(error) from error


@router.get("/data/datasets/{version_id}/preview")
def dataset_input_preview(version_id: str) -> dict[str, Any]:
    from .dataset_previews import available_dataset, describe
    try:
        return describe(service.database, available_dataset(service.database, version_id))
    except Exception as error:
        raise _http_error(error) from error


@router.get("/data/datasets/{version_id}/preview/episodes/{episode_index}/frames")
def dataset_input_preview_frames(
    version_id: str, episode_index: int,
    start: int = Query(default=0, ge=0), count: int = Query(default=60, ge=1, le=120),
    gateway: str = Query(default="auto"),
):
    from .dataset_previews import available_dataset, previews
    from fastapi.responses import Response
    try:
        dataset = available_dataset(service.database, version_id)
        payload = previews.frames(service.database, service.cluster, dataset, episode_index, start, count, gateway)
        return Response(payload, media_type="application/json", headers={"Cache-Control": "private, no-store"})
    except Exception as error:
        raise _http_error(error) from error


@router.get("/data/resources")
def list_data_resources(
    category: str | None = Query(default=None, pattern=r"^(dataset|file)$"),
    provider: str | None = Query(default=None),
    namespace: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    include_archived: bool = Query(default=False),
    include_versions: bool = Query(default=False),
) -> dict[str, Any]:
    return {
        "resource_types": RESOURCE_TYPES,
        "resources": _display_payload(service.database.list_data_resources(
            category=category,
            provider=provider,
            namespace=namespace,
            kind=kind,
            include_archived=include_archived,
            include_versions=include_versions,
        ))
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
    resource["experiment_presets"] = [link for link in service.database.dataset_preset_links()
                                      if link["resource_id"] == resource_id]
    return {"resource": resource}


@router.patch("/data/resources/{resource_id}")
def edit_data_resource(
    resource_id: str, request: DataResourceEditRequest
) -> dict[str, Any]:
    try:
        resource = service.database.get_data_resource(resource_id)
        if resource is None:
            raise KeyError("Data resource not found")
        fields = request.model_dump(mode="python", exclude_unset=True)
        validate_resource_edit(resource["category"], fields)
        return {"resource": service.database.update_data_resource(resource_id, **fields)}
    except Exception as error:
        raise _http_error(error) from error


@router.delete("/data/resources/{resource_id}")
def archive_data_resource(resource_id: str) -> dict[str, Any]:
    try:
        resource = service.database.get_data_resource(resource_id)
        if resource is None:
            raise KeyError("Data resource not found")
        fields = {"archived": True}
        validate_resource_edit(resource["category"], fields)
        return {"resource": service.database.update_data_resource(resource_id, **fields)}
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


@router.post("/data/imports/{import_id}/cancel")
def cancel_data_import(import_id: str) -> dict[str, Any]:
    try:
        return {"import": service.cancel_data_import(import_id)}
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


@router.get("/data/derivations")
def list_data_derivations(
    limit: int = Query(default=1000, ge=1, le=10000),
) -> dict[str, Any]:
    return {"derivations": _display_payload(service.database.list_data_derivations(limit=limit))}


@router.post("/data/derivations", status_code=201)
def create_data_derivation(request: DataDerivationCreateRequest) -> dict[str, Any]:
    try:
        payload = request.model_dump(mode="python")
        payload["inputs"] = [item.model_dump(mode="python") for item in request.inputs]
        return {"derivation": service.database.create_data_derivation(**payload)}
    except Exception as error:
        raise _http_error(error) from error


@router.post("/model-io/preview")
def preview_model_io(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        bundle_id = payload.get("bundle_id")
        bundle = (data_selection.snapshot(service.database, payload["data_selections"])
                  if payload.get("data_selections") else service.database.get_data_bundle(bundle_id) if bundle_id else None)
        return resolve_model_io(payload.get("manifest") or {}, preview_spec(payload.get("values"), bundle))
    except (TypeError, ValueError) as error:
        raise _http_error(error) from error


@router.get("/adapters")
def adapters(include_archived: bool = Query(default=False)) -> dict[str, Any]:
    records = []
    # The list carries each adapter's compact projected manifest (no capsule code);
    # its validity is judged on that projection, the detail route on the full body.
    for row in service.database.list_adapter_registry(include_archived=include_archived, include_editable=True,
                                                      manifests="projected"):
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
            "editable": row["editable"],
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
    return {"adapters": _display_payload(records)}


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
    record["editable"] = service.database.owns("adapters", adapter_id, writable=True)
    return {"adapter": record}


@router.put("/adapters/{adapter_id}")
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
        if adapter_version_id and not adapter_id:
            raise ValueError("adapter_version_id requires adapter_id")
        if adapter_id:
            version = service.database.get_adapter_version(
                adapter_id, version_id=adapter_version_id,
            )
            if version is None:
                raise ValueError(
                    f"adapter version was not found: {adapter_id}@{adapter_version_id or 'latest'}"
                )
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
            result = service.source_metadata.response(entry, cache_hit=True)
            # Repository contents are pinned, but operator runtime profiles can change.
            result["runtime_profiles"] = service.runtime_profiles(gateway)
            return result

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


@router.get("/evaluation-suites")
def evaluation_suites(
    run_id: str | None = Query(default=None, min_length=1, max_length=128),
    target_dataset_id: str | None = Query(default=None, min_length=1, max_length=128),
    unseen_embodiment: bool = Query(default=False),
) -> dict[str, Any]:
    evaluator_manifest: AdapterManifest | None = None
    evaluator_identity: dict[str, Any] | None = None
    spec_document: dict[str, Any] | None = None
    if run_id is not None:
        run = service.database.get_run(run_id, execution_stage_ids=[])
        if run is None:
            raise HTTPException(status_code=404, detail="Training run was not found.")
        try:
            training_spec = ExperimentSpec.model_validate(run["resolved_spec_json"])
            evaluator_document = copy.deepcopy(
                training_spec.model_dump(mode="json", by_alias=True)
            )
            evaluator_document["source"], evaluator_manifest = service._evaluation_source(training_spec)
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

    suites = []
    unavailable_suites = []
    checkpoint = next((c for c in (run.get("checkpoints", []) if run_id else [])
                       if c.get("is_selected_for_inference") and c.get("status") == "AVAILABLE" and not c.get("pruned_at")), None)
    target_config = {}
    target_error = None
    if run_id is not None:
        try:
            target_config = attach_evaluation_target(service.database, {"config_json": {}},
                spec_document or {}, target_dataset_id, unseen_embodiment)["config_json"]
        except ValueError as error:
            target_error = str(error)

    @functools.cache
    def recorded_episode_problem() -> str | None:
        # The recorded training episode belongs to the run, not to a suite: look it
        # up once per request, and only when a ready suite starts from it.
        try:
            recorded_episode_sources(service.database, service.cluster, spec_document or {})
        except ValueError as error:
            return sanitize(str(error))
        return None

    for original in service.database.list_evaluation_suites():
        row = original
        compatibility = None
        resolved_manifest = evaluator_manifest
        if run_id is not None:
            row = copy.deepcopy(row)
            row["config_json"].update(target_config)
            if target_error:
                # Resolve the same target once for the entire picker.
                row["config_json"]["target_error"] = target_error
            compatibility, row = inspect_compatibility(spec_document or {}, evaluator_manifest, row, checkpoint)
            if row["config_json"].get("target_error"):
                compatibility.update(status="incompatible", label="Incompatible", ready=False)
                compatibility["messages"].append(row["config_json"].pop("target_error"))
            resolved_manifest = compose_evaluator(spec_document or {}, evaluator_manifest, row)
            if compatibility["ready"] and row["config_json"].get("initial_state") == "single_training_episode":
                problem = recorded_episode_problem()
                if problem is not None:
                    compatibility.update(status="unknown", label="Missing information", ready=False)
                    compatibility["messages"].append(problem)
                    compatibility["checks"].append({"field": "recording", "status": "unknown", "message": problem})
            if not compatibility["ready"]:
                unavailable_suites.append({"id": row["id"], "reason": "; ".join(compatibility["messages"])})
        config = row["config_json"]
        isaac_evaluation = uses_isaac_sim({
            "environment": row["evaluator_adapter"], "suite": {"config": config},
        })
        placement = CLUSTER.isaac_evaluation_placement
        allowed_gpu_types = sorted({node.gpu_type for node in placement.nodes.values()}) if placement else []
        suites.append({
            **row,
            "config_json": {key: value for key, value in config.items() if key != "target_dataset"},
            "can_delete": service.database.workspace_id in (None, LEGACY_WORKSPACE),
            "is_default": bool(compatibility and compatibility["ready"] and config.get("initial_state") == "single_training_episode"),
            "compatibility": compatibility,
            "allowed_gpu_types": allowed_gpu_types if isaac_evaluation else None,
            "maximum_episodes_per_task": config.get("maximum_episodes_per_task"),
            "label": row["description"] or row["name"],
            "requires_target_dataset": bool(evaluation_target_contract(spec_document or {})),
            "target_dataset_contract": evaluation_target_contract(spec_document or {}),
            "tasks": config.get("tasks", []), "task_options": config.get("task_options", []),
            "default_tasks": config.get("default_tasks", []),
            "maximum_parallelism": max((entry.maximum_parallelism or 1
                for entry in (resolved_manifest.evaluations if resolved_manifest else [])
                if declaration_matches(entry, spec_document or {}, row)), default=1),
            "task_source": config.get("task_source"),
            "task_catalog_complete": bool(config.get("task_catalog_complete")),
            "task_selection_mode": config.get("task_selection_mode", "subset"),
            "task_selection_reason": config.get("task_selection_reason", ""),
            "task_catalog_provenance": config.get("task_catalog_provenance"),
            "task_catalog_sha256": config.get("task_catalog_sha256"),
            "catalog_sha256": config.get("catalog_sha256"),
            **({"evaluator_implementation": copy.deepcopy(evaluator_identity)} if evaluator_identity is not None else {}),
        })
    return {"suites": suites, "unavailable_suites": unavailable_suites}


@router.post("/experiments/preview")
def preview_experiment(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return service.preview(payload)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors(include_url=False, include_input=False, include_context=False)) from error
    except Exception as error:
        raise _http_error(error) from error


@router.post("/experiments")
def create_experiment(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        experiment = service.create_experiment(payload)
        return {"experiment": experiment, "id": experiment["id"]}
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors(include_url=False, include_input=False, include_context=False)) from error
    except Exception as error:
        raise _http_error(error) from error


@router.get("/experiments")
def list_experiments() -> dict[str, Any]:
    records = service.database.list_experiments(limit=LIST_RESPONSE_LIMIT)
    dataset_links = service.database.dataset_preset_links()
    datasets_by_experiment: dict[str, set[str]] = {}
    for link in dataset_links:
        datasets_by_experiment.setdefault(link["experiment_id"], set()).add(link["version_id"])
    for record in records:
        record["dataset_ids"] = sorted(datasets_by_experiment.get(record["id"], ()))
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
    refresh_progress: bool = Query(default=True),
) -> dict[str, Any]:
    runs = service.database.list_runs(
        experiment_id=experiment_id,
        experiment_revision_id=experiment_revision_id,
        variant_id=variant_id,
        status=status,
        limit=LIST_RESPONSE_LIMIT,
    )
    connections = service.tracking_connections()["connections"]
    progress_refresh_pending = (service._queue_list_progress_refresh("training", runs) if refresh_progress
                                else service._list_progress_refresh_pending("training", runs))
    for run in runs:
        run["tracking_actions"] = service.run_tracking_actions(run, connections)
    _attach_run_progress_summaries(service.database, runs)
    return {"runs": runs, "progress_refresh_pending": progress_refresh_pending}


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
        native_field = next((f for f in train_manifest.get("input_fields") or [] if f.get("canonical_path") == canonical_path), None)
        if native_field and _mapping_path(spec, native_field["path"])[0]:
            values[key] = copy.deepcopy(_mapping_path(spec, native_field["path"])[1])
            provenance[key] = {"status": "resolved", "source": "adapter_input", "canonical_path": canonical_path, "evidence": {"input_path": native_field["path"]}}
        elif (
            (not train_manifest.get("strict_canonical_inputs") or _canonical_path_supported(canonical_path, supported))
            and _path_is_explicit(spec, canonical_path)
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
        elif (preset := next((p for p in train_manifest.get("presets") or [] if p.get("id") == (spec.get("native", {}).get("config", {}).get("training_preset") or train_manifest.get("default_preset"))), None)) and canonical_path in preset.get("values", {}):
            values[key] = copy.deepcopy(spec_value if present else preset["values"][canonical_path])
            provenance[key] = {"status":"resolved", "source":"training_preset", "canonical_path":canonical_path,
                               "evidence":{"preset":preset["id"], "adapter_manifest_sha256":manifest_sha256}}
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
        "adapter_settings": {f["path"]: _mapping_path(spec, f["path"])[1] for f in train_manifest.get("input_fields") or [] if not f.get("data_binding") and _mapping_path(spec, f["path"])[0]},
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


ENRICHMENT_EVENT_TYPE = "COMMON_HYPERPARAMETERS_ENRICHED_V1"


def _attempt_enrichment_events(
    database: Database, attempt_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Each attempt's latest enrichment receipt, in one query."""
    return database.latest_events(
        entity_type="job_attempt", entity_ids=attempt_ids, event_type=ENRICHMENT_EVENT_TYPE
    )


def _attempt_enrichment_event(
    database: Database, attempt_id: str
) -> dict[str, Any] | None:
    return _attempt_enrichment_events(database, [attempt_id]).get(attempt_id)


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
def get_run(run_id: str, include_payloads: bool = True) -> dict[str, Any]:
    run = service.database.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    service._queue_list_progress_refresh("training", [run])
    receipt_events = _attempt_enrichment_events(
        service.database, [str(attempt["id"]) for attempt in run.get("attempts") or []]
    )
    for attempt in run.get("attempts") or []:
        receipt_event = receipt_events.get(str(attempt["id"]))
        common = _attempt_common_hyperparameter_contract(
            run, attempt, receipt_event
        )
        attempt["model_io"] = (attempt.get("execution_snapshot_json") or {}).get("model_io")
        attempt["adapter_settings"] = common.get("adapter_settings", {})
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
    if include_payloads:
        return {"run": run}
    display = _display_payload(run)
    # These repeat the same pinned code/data in several forms; the UI consumes
    # the derived attempt contract above and the top-level training snapshot.
    for attempt in display.get("attempts", []):
        snapshot = attempt.get("execution_snapshot_json") or {}
        attempt["execution_snapshot_json"] = {
            key: snapshot[key] for key in ("adapter", "resolved_spec", "schema_version") if key in snapshot
        }
    for stage in display.get("stages", []):
        stage.pop("resolved_config_json", None)
    return {"run": display}


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
def resume_run(run_id: str, request: GatewayRequest) -> dict[str, Any]:
    try:
        return service.retry_run(run_id, request.gateway)
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
def list_evaluations(refresh_progress: bool = Query(default=True)) -> dict[str, Any]:
    evaluations = service.database.list_evaluations(limit=LIST_RESPONSE_LIMIT)
    progress_refresh_pending = (service._queue_list_progress_refresh("evaluation", evaluations) if refresh_progress
                                else service._list_progress_refresh_pending("evaluation", evaluations))
    _attach_evaluation_progress_summaries(service.database, evaluations)
    return {"evaluations": evaluations, "progress_refresh_pending": progress_refresh_pending}


@router.get("/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: str) -> dict[str, Any]:
    evaluation = service.database.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    service._queue_list_progress_refresh("evaluation", [evaluation])
    run = service.database.get_run(evaluation["run_id"], include_payloads=False)
    evaluation["attempts"] = _stage_attempts(run, evaluation.get("stage_id"))
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
    attach_job_display_status(evaluation, evaluation["attempts"])
    return {"evaluation": evaluation}


@router.post("/evaluations/{evaluation_id}/cancel")
def cancel_evaluation(evaluation_id: str) -> dict[str, Any]:
    try:
        return service.cancel_evaluation(evaluation_id)
    except Exception as error:
        raise _http_error(error) from error


@router.post("/evaluations/{evaluation_id}/retry-submission")
def retry_evaluation_submission(evaluation_id: str, request: GatewayRequest) -> dict[str, Any]:
    try:
        return service.retry_evaluation_submission(evaluation_id, request.gateway)
    except Exception as error:
        raise _http_error(error) from error


@router.post("/evaluations/{evaluation_id}/reread-result")
def reread_evaluation_result(evaluation_id: str) -> dict[str, Any]:
    try:
        return service.reread_evaluation_result(evaluation_id)
    except Exception as error:
        raise _http_error(error) from error


@router.get(
    "/evaluations/{evaluation_id}/episodes/{episode_id}/logs",
    response_class=PlainTextResponse,
)
def get_evaluation_episode_log(
    evaluation_id: str,
    episode_id: str,
    stream: Literal["stdout", "stderr"] = "stdout",
) -> str:
    evaluation = service.database.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    episode = next(
        (item for item in evaluation.get("episodes", []) if item["id"] == episode_id),
        None,
    )
    if not episode:
        raise HTTPException(status_code=404, detail="Rollout not found")
    run = service.database.get_run(evaluation["run_id"], include_payloads=False)
    attempts = _stage_attempts(run, evaluation.get("stage_id"))
    if not attempts:
        return "No Slurm attempt recorded."
    metrics = episode.get("metrics_json") or {}
    job = _progress_integer(metrics.get("slurm_job_id"))
    attempt = next(
        (a for a in attempts if job and str(a.get("slurm_job_id")) == str(job)), None
    )
    attempt = attempt or _latest_attempt(attempts)
    worker = metrics.get("worker_index")
    if worker is None:
        return service._read_attempt_log(
            attempt, stream, 500, retrieval_errors_as_text=True
        )
    if (
        isinstance(worker, bool)
        or not isinstance(worker, (int, float))
        or not math.isfinite(worker)
        or worker != int(worker)
        or not 0 <= worker <= 100000
    ):
        raise HTTPException(status_code=409, detail="Invalid rollout worker identity")
    result_path = evaluation.get("result_path")
    if not result_path:
        raise HTTPException(
            status_code=409, detail="Evaluation result location is unavailable"
        )
    path = str(
        PurePosixPath(result_path).parent
        / "workers"
        / str(int(worker))
        / (stream + ".log")
    )
    return service._read_attempt_log(
        {**attempt, stream + "_path": path}, stream, 500, retrieval_errors_as_text=True
    )


@router.get("/data/selections")
def get_training_data_selections():
    return {"datasets": _display_payload(data_selection.choices(service.database))}


@router.get("/evaluations/{evaluation_id}/episodes/{episode_id}/viewer")
def get_evaluation_episode_viewer(evaluation_id: str, episode_id: str):
    evaluation = service.database.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    episode = next((item for item in evaluation.get("episodes", []) if item["id"] == episode_id), None)
    if not episode:
        raise HTTPException(status_code=404, detail="Rollout not found")
    run = service.database.get_run(evaluation["run_id"], execution_stage_ids=[evaluation["stage_id"]]) or {}
    gateway = (_latest_stage_attempt(run, evaluation.get("stage_id")) or {}).get("gateway") or "auto"
    spec = run.get("resolved_spec_json") or {}
    stage = next((item for item in run.get("stages", []) if item.get("id") == evaluation.get("stage_id")), {})
    target = ((stage.get("resolved_config_json") or {}).get("context") or {}).get("target_dataset")
    metadata = target.get("metadata", {}) if target else dataset_metadata(spec)
    robot = (metadata.get("capture") or {}).get("robot")
    missing = {"state": "UNAVAILABLE", "robot": robot,
               "detail": "This episode has no saved replay trace. Camera separation and keypoints are available for new recorded-simulator evaluations."}
    if not episode.get("video_path"):
        return missing
    video = PurePosixPath(episode["video_path"])
    root = PurePosixPath(evaluation["result_path"]).parent / "videos"
    if not video.is_absolute() or root not in video.parents or ".." in video.parts or video.suffix.lower() != ".mp4":
        raise HTTPException(status_code=409, detail="Invalid registered rollout path")
    path = str(video.with_suffix(".review.json"))
    from .adapters.episode_geometry import MAX_VIEWER_BYTES
    program = (
        "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); "
        f"print(p.read_text() if p.is_file() and p.stat().st_size <= {MAX_VIEWER_BYTES} "
        "else json.dumps({'preview_error': 'The saved replay exceeds the interactive viewer size limit.'}) "
        "if p.is_file() else 'null')"
    )
    try:
        _, text = service.cluster.run_with_fallback("python3 -c " + shlex.quote(program) + " " + shlex.quote(path), gateway, timeout=30)
        viewer = json.loads(text)
        if isinstance(viewer, dict) and viewer.get("preview_error"):
            return {"state": "UNAVAILABLE", "robot": robot, "detail": viewer["preview_error"]}
    except Exception as error:
        raise _http_error(error) from error
    try:
        from .rollout_preview import enrich_demonstration
        if not target:
            viewer = enrich_demonstration(service.database, service.cluster, spec, viewer, gateway,
                                          task=episode.get("task"), duration=(episode.get("episode_length") or 0) * float((metadata.get("capture") or {}).get("step_dt", 0)))
    except (ValueError, OSError, ClusterError) as error:
        if viewer:
            viewer.setdefault("warnings", []).append("Demonstration could not be restored: " + str(error))
        else:
            missing["detail"] += " Original demonstration could not be restored: " + str(error)
    if viewer is None:
        return missing
    if viewer.get("schema") != "skynet.episode-viewer/v1":
        raise HTTPException(status_code=409, detail="Unsupported episode replay format")
    return {"state": "READY", "viewer": viewer}


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

    run = service.database.get_run(evaluation["run_id"], include_payloads=False)
    gateway = (_latest_stage_attempt(run, evaluation.get("stage_id")) or {}).get("gateway") or "auto"
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
        target_dataset_id=request.target_dataset_id,
        unseen_embodiment=request.unseen_embodiment,
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


@router.get("/tracking/connections")
def list_tracking_connections() -> dict[str, Any]:
    return service.tracking_connections()


@router.post("/tracking/connections/{provider}/connect")
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


@router.post("/tracking/connections/{provider}/disconnect")
def disconnect_tracking_connection(provider: str) -> dict[str, Any]:
    try:
        return service.disconnect_tracking_connection(provider)
    except CredentialStoreError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/settings")
def settings() -> dict[str, Any]:
    connections = service.tracking_connections()["connections"]
    storage = service.storage.snapshot()
    paths = storage["paths"]
    tracking = {
        "providers": list(connections.values()),
        "connections": connections,
        "secrets_persisted": False,
    }
    return {
        "paths": {
            "work_root": storage["settings"]["work_root"],
            "database": "central-postgresql",
            "local_capsules": str(LOCAL_CAPSULE_ROOT),
            "evaluation_root": paths["evaluation"],
        },
        "cluster": {
            **CLUSTER.public_dict(),
            "paths": paths,
            "multi_node": CLUSTER.limits.max_nodes > 1,
            "multi_gpu_single_node": CLUSTER.limits.max_gpus_per_node > 1,
        },
        "storage": storage["settings"],
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


class WorkspaceStorageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    work_root: str = Field(min_length=2, max_length=512)
    expected_work_root: str | None = Field(default=None, min_length=2, max_length=512)

    @field_validator("work_root")
    @classmethod
    def valid_root(cls, value: str) -> str:
        return validate_work_root(value)


@router.put("/workspace/storage")
def update_workspace_storage(payload: WorkspaceStorageRequest, gateway: str = Query(default="auto")) -> dict[str, Any]:
    try:
        return service.storage.configure(payload.work_root, payload.expected_work_root, service.cluster, gateway)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ClusterError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


__all__ = ["PipelineService", "router", "service"]
