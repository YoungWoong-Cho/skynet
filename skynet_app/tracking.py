from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - the application runs on Linux/macOS.
    fcntl = None  # type: ignore[assignment]


SPOOL_SCHEMA_VERSION = 1
SPOOL_FILENAME = "mlflow-spool.jsonl"
STATE_FILENAME = "mlflow-state.json"
LOCK_FILENAME = ".mlflow-spool.lock"
CURSOR_TAG = "skynet.spool.cursor"
LOCAL_RUN_TAG = "skynet.local_run_id"
RUN_STATUS_TAG = "skynet.status"
ATTEMPT_NUMBER_TAG = "skynet.attempt_number"
WANDB_SPOOL_FILENAME = "wandb-spool.jsonl"
WANDB_STATE_FILENAME = "wandb-state.json"
WANDB_LOCK_FILENAME = ".wandb-spool.lock"
WANDB_TAG_MAX_LENGTH = 64
WANDB_TAG_HASH_LENGTH = 12
WANDB_TAG_METADATA_CONFIG_KEY = "skynet_tag_metadata"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _milliseconds_now() -> int:
    return int(time.time() * 1000)


def _validated_attempt_number(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("attempt_number must be a positive integer")
    return value


def _environment_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _is_sensitive_key(key: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
    exact = {
        "authorization",
        "cookie",
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "access_key",
        "secret_key",
        "private_key",
        "client_secret",
    }
    return (
        normalized in exact
        or normalized.endswith("_password")
        or normalized.endswith("_passwd")
        or normalized.endswith("_secret")
        or normalized.endswith("_token")
        or normalized.endswith("_api_key")
        or normalized.endswith("_access_key")
        or normalized.endswith("_private_key")
    )


def _sanitize_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return value
    if parsed.scheme not in {"http", "https", "s3", "gs", "ftp"}:
        return value
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username is not None or parsed.password is not None:
        netloc = f"[REDACTED]@{netloc}"
    query = []
    for key, item in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
        query.append((key, "[REDACTED]" if _is_sensitive_key(key) else item))
    return urllib.parse.urlunsplit(
        (parsed.scheme, netloc, parsed.path, urllib.parse.urlencode(query), parsed.fragment)
    )


def sanitize(value: Any, *, secrets: Sequence[str | None] = ()) -> Any:
    """Return a JSON-safe value with credentials and secret-like fields redacted."""

    runtime_secrets = tuple(secret for secret in secrets if secret)
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _is_sensitive_key(key) else sanitize(item, secrets=runtime_secrets)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item, secrets=runtime_secrets) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        sanitized = _sanitize_url(value)
        for secret in runtime_secrets:
            sanitized = sanitized.replace(secret, "[REDACTED]")
        sanitized = re.sub(
            r"(?i)(password|passwd|secret|access[_-]?token|api[_-]?key)=([^&\s]+)",
            r"\1=[REDACTED]",
            sanitized,
        )
        return sanitized
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


@dataclass(frozen=True)
class TrackingSettings:
    tracking_uri: str | None = field(default=None, repr=False)
    token: str | None = field(default=None, repr=False)
    username: str | None = field(default=None, repr=False)
    password: str | None = field(default=None, repr=False)
    timeout_seconds: float = 5.0
    enabled: bool = True
    auto_flush: bool = True
    verify_tls: bool = True

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "TrackingSettings":
        environment = os.environ if environ is None else environ
        timeout_text = environment.get("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
        try:
            timeout = max(0.1, float(timeout_text))
        except ValueError:
            timeout = 5.0
        return cls(
            tracking_uri=environment.get("MLFLOW_TRACKING_URI") or None,
            token=environment.get("MLFLOW_TRACKING_TOKEN") or None,
            username=environment.get("MLFLOW_TRACKING_USERNAME") or None,
            password=environment.get("MLFLOW_TRACKING_PASSWORD") or None,
            timeout_seconds=timeout,
            enabled=_environment_bool(environment.get("SKYNET_MLFLOW_ENABLED"), True),
            auto_flush=_environment_bool(environment.get("SKYNET_MLFLOW_AUTO_FLUSH"), True),
            verify_tls=_environment_bool(environment.get("MLFLOW_TRACKING_SERVER_CERT_IGNORE_TLS"), False)
            is False,
        )

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.tracking_uri)

    @property
    def redacted_tracking_uri(self) -> str | None:
        return _sanitize_url(self.tracking_uri) if self.tracking_uri else None

    def public_dict(self) -> dict[str, Any]:
        return {
            "tracking_uri": self.redacted_tracking_uri,
            "configured": self.configured,
            "timeout_seconds": self.timeout_seconds,
            "enabled": self.enabled,
            "auto_flush": self.auto_flush,
            "verify_tls": self.verify_tls,
            "authentication": "token" if self.token else "basic" if self.username else "none",
        }


@dataclass(frozen=True)
class TrackingResult:
    event_id: str
    sequence: int
    delivered: bool
    queued: bool
    remote_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class DrainReport:
    attempted: int
    delivered: int
    remaining: int
    online: bool
    errors: tuple[str, ...] = ()


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        from email.utils import parsedate_to_datetime
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return None


class TrackingRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class MLflowBridge:
    """Offline-first, dependency-free MLflow REST bridge for one run capsule."""

    def __init__(
        self,
        run_capsule: str | os.PathLike[str],
        settings: TrackingSettings | None = None,
    ) -> None:
        self.run_capsule = Path(run_capsule).expanduser().resolve()
        self.settings = settings or TrackingSettings.from_env()
        self.spool_path = self.run_capsule / SPOOL_FILENAME
        self.state_path = self.run_capsule / STATE_FILENAME
        self.lock_path = self.run_capsule / LOCK_FILENAME
        self._thread_lock = threading.RLock()
        self._last_error: str | None = None
        self.run_capsule.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if not self.spool_path.exists():
                self._atomic_write(self.spool_path, b"")
            if not self.state_path.exists():
                self._write_state_unlocked(self._empty_state())

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _runtime_secrets(self) -> tuple[str | None, ...]:
        return (self.settings.token, self.settings.username, self.settings.password)

    def _safe_error(self, error: BaseException) -> str:
        return str(sanitize(str(error), secrets=self._runtime_secrets()))

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "schema_version": SPOOL_SCHEMA_VERSION,
            "acked_through": 0,
            "inflight_sequence": None,
            "experiments": {},
            "runs": {},
            "updated_at": _utc_now(),
        }

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _load_state_unlocked(self) -> dict[str, Any]:
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            loaded = self._empty_state()
        state = self._empty_state()
        if isinstance(loaded, Mapping):
            state.update(loaded)
        if not isinstance(state.get("experiments"), dict):
            state["experiments"] = {}
        if not isinstance(state.get("runs"), dict):
            state["runs"] = {}
        return state

    def _write_state_unlocked(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        payload["schema_version"] = SPOOL_SCHEMA_VERSION
        payload["updated_at"] = _utc_now()
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        self._atomic_write(self.state_path, (serialized + "\n").encode("utf-8"))

    def _read_events_unlocked(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        try:
            lines = self.spool_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return events
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid tracking spool record at line {line_number}") from error
            if not isinstance(event, dict) or not isinstance(event.get("sequence"), int):
                raise ValueError(f"Invalid tracking spool record at line {line_number}")
            events.append(event)
        return events

    def _enqueue(self, operation: str, payload: Mapping[str, Any]) -> TrackingResult:
        safe_payload = sanitize(payload, secrets=self._runtime_secrets())
        with self._locked():
            events = self._read_events_unlocked()
            idempotency_key = safe_payload.get("idempotency_key")
            event = next(
                (
                    item
                    for item in events
                    if idempotency_key
                    and item.get("operation") == operation
                    and isinstance(item.get("payload"), Mapping)
                    and item["payload"].get("idempotency_key") == idempotency_key
                ),
                None,
            )
            if event is None:
                sequence = max((int(item["sequence"]) for item in events), default=0) + 1
                event = {
                    "schema_version": SPOOL_SCHEMA_VERSION,
                    "id": str(uuid.uuid4()),
                    "sequence": sequence,
                    "operation": operation,
                    "payload": safe_payload,
                    "created_at": _utc_now(),
                }
                existing = self.spool_path.read_bytes() if self.spool_path.exists() else b""
                line = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
                self._atomic_write(self.spool_path, existing + line)
            else:
                sequence = int(event["sequence"])

        report = self.drain_spool() if self.settings.auto_flush else DrainReport(0, 0, self.pending_count(), False)
        with self._locked():
            state = self._load_state_unlocked()
            delivered = int(state.get("acked_through", 0)) >= sequence
            remote_id = self._remote_id_for_event(event, state)
        error = report.errors[0] if report.errors else None
        return TrackingResult(
            event_id=event["id"],
            sequence=sequence,
            delivered=delivered,
            queued=not delivered,
            remote_id=remote_id,
            error=error,
        )

    @staticmethod
    def _remote_id_for_event(event: Mapping[str, Any], state: Mapping[str, Any]) -> str | None:
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            return None
        if event.get("operation") == "ensure_experiment":
            return state.get("experiments", {}).get(payload.get("name"))
        local_run_id = payload.get("local_run_id")
        return state.get("runs", {}).get(local_run_id) if local_run_id else None

    def pending_count(self) -> int:
        with self._locked():
            state = self._load_state_unlocked()
            acknowledged = int(state.get("acked_through", 0))
            return sum(event["sequence"] > acknowledged for event in self._read_events_unlocked())

    def pending_events(self) -> list[dict[str, Any]]:
        with self._locked():
            state = self._load_state_unlocked()
            acknowledged = int(state.get("acked_through", 0))
            return [event for event in self._read_events_unlocked() if event["sequence"] > acknowledged]

    def metric_idempotency_keys(self) -> set[str]:
        with self._locked():
            return {
                str(payload["idempotency_key"])
                for event in self._read_events_unlocked()
                if event.get("operation") == "log_batch"
                and isinstance((payload := event.get("payload")), Mapping)
                and payload.get("idempotency_key")
            }

    def metric_names_by_idempotency_key(self) -> dict[str, set[str]]:
        """Include queued and delivered fields so enrichment cannot duplicate them."""
        with self._locked():
            return {
                str(payload["idempotency_key"]): {
                    str(metric["key"]) for metric in payload.get("metrics", [])
                }
                for event in self._read_events_unlocked()
                if event.get("operation") == "log_batch"
                and isinstance((payload := event.get("payload")), Mapping)
                and payload.get("idempotency_key")
            }

    def ensure_experiment(
        self,
        name: str,
        *,
        artifact_location: str | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> TrackingResult:
        return self._enqueue("ensure_experiment", {
            "name": name,
            "artifact_location": artifact_location,
            "tags": tags or {},
        })

    get_or_create_experiment = ensure_experiment

    def ensure_run(
        self,
        *,
        experiment_name: str,
        local_run_id: str,
        run_name: str,
        tags: Mapping[str, Any] | None = None,
        start_time_ms: int | None = None,
    ) -> TrackingResult:
        return self._enqueue("ensure_run", {
            "experiment_name": experiment_name,
            "local_run_id": local_run_id,
            "run_name": run_name,
            "tags": tags or {},
            "start_time": start_time_ms or _milliseconds_now(),
        })

    get_or_create_run = ensure_run

    def log_params(self, local_run_id: str, params: Mapping[str, Any]) -> TrackingResult:
        return self._enqueue("log_batch", {
            "local_run_id": local_run_id,
            "params": params,
            "metrics": [],
            "tags": {},
        })

    def log_metrics(
        self,
        local_run_id: str,
        metrics: Mapping[str, float | int],
        *,
        step: int = 0,
        timestamp_ms: int | None = None,
        idempotency_key: str | None = None,
    ) -> TrackingResult:
        timestamp = timestamp_ms if timestamp_ms is not None else _milliseconds_now()
        records = [
            {"key": str(key), "value": float(value), "step": step, "timestamp": timestamp}
            for key, value in metrics.items()
        ]
        return self._enqueue("log_batch", {
            "local_run_id": local_run_id,
            "params": {},
            "metrics": records,
            "tags": {},
            **({"idempotency_key": idempotency_key} if idempotency_key else {}),
        })

    def set_tags(self, local_run_id: str, tags: Mapping[str, Any]) -> TrackingResult:
        return self._enqueue("log_batch", {
            "local_run_id": local_run_id,
            "params": {},
            "metrics": [],
            "tags": tags,
        })

    def log_artifact_link(
        self,
        local_run_id: str,
        *,
        name: str,
        uri: str,
        artifact_type: str | None = None,
        sha256: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrackingResult:
        return self._enqueue("artifact_link", {
            "local_run_id": local_run_id,
            "name": name,
            "uri": uri,
            "artifact_type": artifact_type,
            "sha256": sha256,
            "metadata": metadata or {},
        })

    def finish_run(
        self,
        local_run_id: str,
        *,
        status: str = "FINISHED",
        end_time_ms: int | None = None,
    ) -> TrackingResult:
        return self._enqueue("finish_run", {
            "local_run_id": local_run_id,
            "status": status.upper(),
            "end_time": end_time_ms or _milliseconds_now(),
        })

    def reopen_run(self, local_run_id: str, attempt_number: int) -> TrackingResult:
        attempt = _validated_attempt_number(attempt_number)
        return self._enqueue("reopen_run", {
            "local_run_id": local_run_id,
            "attempt_number": attempt,
            "idempotency_key": f"reopen:{local_run_id}:attempt:{attempt}",
        })

    def get_experiment(self, name: str) -> dict[str, Any] | None:
        if not self.settings.configured:
            return None
        try:
            return self._get_remote_experiment(name)
        except Exception as error:
            self._last_error = self._safe_error(error)
            return None

    def get_run(self, run_id: str, *, local: bool = True) -> dict[str, Any] | None:
        if not self.settings.configured:
            return None
        remote_run_id = run_id
        if local:
            with self._locked():
                remote_run_id = self._load_state_unlocked().get("runs", {}).get(run_id)
            if not remote_run_id:
                return None
        try:
            response = self._request(
                "GET", "/api/2.0/mlflow/runs/get", query={"run_id": remote_run_id}
            )
            run = response.get("run")
            return run if isinstance(run, dict) else None
        except Exception as error:
            self._last_error = self._safe_error(error)
            return None

    def validate_connection(self) -> dict[str, Any]:
        """Verify the configured tracking server without creating remote state."""

        if not self.settings.configured:
            raise TrackingRequestError("MLflow tracking URI is not configured")
        response = self._request(
            "GET",
            "/api/2.0/mlflow/experiments/search",
            query={"max_results": 1},
        )
        experiments = response.get("experiments", [])
        if not isinstance(experiments, list):
            raise TrackingRequestError(
                "MLflow experiment search returned an unexpected response"
            )
        return {"server_version": response.get("server_version")}

    def binding(self, local_run_id: str) -> dict[str, str] | None:
        with self._locked():
            state = self._load_state_unlocked()
            remote_run_id = state.get("runs", {}).get(local_run_id)
            if not remote_run_id:
                return None
            return {"remote_id": str(remote_run_id)}

    def experiment_binding(self, experiment_name: str) -> str | None:
        with self._locked():
            value = self._load_state_unlocked().get("experiments", {}).get(experiment_name)
            return str(value) if value else None

    def drain_spool(self, *, limit: int | None = None) -> DrainReport:
        if not self.settings.configured:
            return DrainReport(0, 0, self.pending_count(), False)

        attempted = 0
        delivered = 0
        errors: list[str] = []
        with self._locked():
            try:
                events = self._read_events_unlocked()
                state = self._load_state_unlocked()
                acknowledged = int(state.get("acked_through", 0))
                pending = [event for event in events if int(event["sequence"]) > acknowledged]
                if limit is not None:
                    pending = pending[: max(0, limit)]
                for event in pending:
                    attempted += 1
                    sequence = int(event["sequence"])
                    was_inflight = state.get("inflight_sequence") == sequence
                    state["inflight_sequence"] = sequence
                    self._write_state_unlocked(state)
                    try:
                        self._deliver_event(event, state, retry=was_inflight)
                    except Exception as error:
                        message = self._safe_error(error)
                        self._last_error = message
                        errors.append(message)
                        if event.get("operation") != "artifact_link":
                            break
                        state.setdefault("warnings", []).append({
                            "sequence": sequence,
                            "operation": "artifact_link",
                            "error": message,
                        })
                        state["acked_through"] = sequence
                        state["inflight_sequence"] = None
                        self._write_state_unlocked(state)
                        continue
                    state["acked_through"] = sequence
                    state["inflight_sequence"] = None
                    self._write_state_unlocked(state)
                    delivered += 1
                remaining = sum(
                    int(event["sequence"]) > int(state.get("acked_through", 0)) for event in events
                )
            except Exception as error:
                message = self._safe_error(error)
                self._last_error = message
                errors.append(message)
                remaining = self.pending_count() if not isinstance(error, ValueError) else 0
        return DrainReport(
            attempted=attempted,
            delivered=delivered,
            remaining=remaining,
            online=not errors,
            errors=tuple(errors),
        )

    def _deliver_event(self, event: Mapping[str, Any], state: dict[str, Any], *, retry: bool) -> None:
        operation = event.get("operation")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("Tracking event payload is invalid")
        sequence = int(event["sequence"])

        if operation == "ensure_experiment":
            experiment_id = self._ensure_remote_experiment(payload)
            state["experiments"][str(payload["name"])] = experiment_id
            return

        local_run_id = str(payload.get("local_run_id", ""))
        if not local_run_id:
            raise ValueError("Run-scoped tracking event has no local run id")
        remote_run_id = state["runs"].get(local_run_id)

        if operation == "ensure_run":
            remote_run_id = self._ensure_remote_run(payload, state, sequence)
            state["runs"][local_run_id] = remote_run_id
            return
        if not remote_run_id:
            raise TrackingRequestError(f"No remote MLflow run exists for local run {local_run_id}")
        if retry and self._remote_cursor(remote_run_id) >= sequence:
            return

        if operation == "log_batch":
            self._deliver_log_batch(remote_run_id, payload, sequence)
        elif operation == "artifact_link":
            self._deliver_artifact_link(remote_run_id, payload, sequence)
        elif operation == "finish_run":
            self._request("POST", "/api/2.0/mlflow/runs/update", payload={
                "run_id": remote_run_id,
                "status": str(payload.get("status", "FINISHED")),
                "end_time": int(payload.get("end_time", _milliseconds_now())),
            })
            self._deliver_log_batch(remote_run_id, {"params": {}, "metrics": [], "tags": {}}, sequence)
        elif operation == "reopen_run":
            attempt = _validated_attempt_number(payload.get("attempt_number"))
            self._request("POST", "/api/2.0/mlflow/runs/update", payload={
                "run_id": remote_run_id,
                "status": "RUNNING",
            })
            self._deliver_log_batch(remote_run_id, {
                "params": {},
                "metrics": [],
                "tags": {
                    RUN_STATUS_TAG: "running",
                    ATTEMPT_NUMBER_TAG: attempt,
                },
            }, sequence)
        else:
            raise ValueError(f"Unsupported tracking operation: {operation}")

    def _ensure_remote_experiment(self, payload: Mapping[str, Any]) -> str:
        name = str(payload["name"])
        existing = self._get_remote_experiment(name)
        if existing:
            return str(existing["experiment_id"])
        body: dict[str, Any] = {"name": name}
        if payload.get("artifact_location"):
            body["artifact_location"] = payload["artifact_location"]
        tags = payload.get("tags", {})
        if isinstance(tags, Mapping) and tags:
            body["tags"] = self._tag_records(tags)
        try:
            response = self._request("POST", "/api/2.0/mlflow/experiments/create", payload=body)
            return str(response["experiment_id"])
        except TrackingRequestError as error:
            if error.status_code not in {400, 409}:
                raise
            raced = self._get_remote_experiment(name)
            if not raced:
                raise
            return str(raced["experiment_id"])

    def _get_remote_experiment(self, name: str) -> dict[str, Any] | None:
        try:
            response = self._request(
                "GET", "/api/2.0/mlflow/experiments/get-by-name", query={"experiment_name": name}
            )
        except TrackingRequestError as error:
            if error.status_code == 404:
                return None
            raise
        experiment = response.get("experiment")
        return experiment if isinstance(experiment, dict) else None

    def _ensure_remote_run(
        self, payload: Mapping[str, Any], state: dict[str, Any], sequence: int
    ) -> str:
        experiment_name = str(payload["experiment_name"])
        experiment_id = state["experiments"].get(experiment_name)
        if not experiment_id:
            experiment_id = self._ensure_remote_experiment({
                "name": experiment_name,
                "artifact_location": None,
                "tags": {},
            })
            state["experiments"][experiment_name] = experiment_id
        local_run_id = str(payload["local_run_id"])
        existing = self._find_remote_run(str(experiment_id), local_run_id)
        if existing:
            return existing
        tags = dict(payload.get("tags", {})) if isinstance(payload.get("tags"), Mapping) else {}
        tags[LOCAL_RUN_TAG] = local_run_id
        tags[CURSOR_TAG] = str(sequence)
        tags.setdefault("mlflow.runName", str(payload.get("run_name", local_run_id)))
        response = self._request("POST", "/api/2.0/mlflow/runs/create", payload={
            "experiment_id": str(experiment_id),
            "run_name": str(payload.get("run_name", local_run_id)),
            "start_time": int(payload.get("start_time", _milliseconds_now())),
            "tags": self._tag_records(tags),
        })
        try:
            return str(response["run"]["info"]["run_id"])
        except (KeyError, TypeError) as error:
            raise TrackingRequestError("MLflow create-run response did not include a run id") from error

    def _find_remote_run(self, experiment_id: str, local_run_id: str) -> str | None:
        escaped = local_run_id.replace("\\", "\\\\").replace("'", "\\'")
        response = self._request("POST", "/api/2.0/mlflow/runs/search", payload={
            "experiment_ids": [experiment_id],
            "filter": f"tags.`{LOCAL_RUN_TAG}` = '{escaped}'",
            "max_results": 1,
        })
        if "runs" not in response or not isinstance(response["runs"], list):
            raise TrackingRequestError("MLflow search-runs response did not contain a runs list")
        runs = response["runs"]
        if not runs:
            return None
        try:
            return str(runs[0]["info"]["run_id"])
        except (KeyError, TypeError) as error:
            raise TrackingRequestError("MLflow search-runs response contained a run without an id") from error

    def _remote_cursor(self, remote_run_id: str) -> int:
        response = self._request("GET", "/api/2.0/mlflow/runs/get", query={"run_id": remote_run_id})
        try:
            tags = response["run"]["data"]["tags"]
        except (KeyError, TypeError) as error:
            raise TrackingRequestError("MLflow get-run response did not contain tags") from error
        if not isinstance(tags, list):
            raise TrackingRequestError("MLflow get-run tags must be a list")
        for tag in tags:
            if isinstance(tag, Mapping) and tag.get("key") == CURSOR_TAG:
                try:
                    return int(tag.get("value", 0))
                except (TypeError, ValueError) as error:
                    raise TrackingRequestError("MLflow tracking cursor is not an integer") from error
        return 0

    @staticmethod
    def _string_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def _tag_records(cls, tags: Mapping[str, Any]) -> list[dict[str, str]]:
        return [{"key": str(key), "value": cls._string_value(value)} for key, value in tags.items()]

    def _deliver_log_batch(
        self, remote_run_id: str, payload: Mapping[str, Any], sequence: int
    ) -> None:
        params_value = payload.get("params", {})
        params = [
            {"key": str(key), "value": self._string_value(value)}
            for key, value in params_value.items()
        ] if isinstance(params_value, Mapping) else []
        metrics_value = payload.get("metrics", [])
        metrics = list(metrics_value) if isinstance(metrics_value, list) else []
        tags_value = payload.get("tags", {})
        tags = dict(tags_value) if isinstance(tags_value, Mapping) else {}
        tags[CURSOR_TAG] = str(sequence)
        self._request("POST", "/api/2.0/mlflow/runs/log-batch", payload={
            "run_id": remote_run_id,
            "metrics": metrics,
            "params": params,
            "tags": self._tag_records(tags),
        })

    def _deliver_artifact_link(
        self, remote_run_id: str, payload: Mapping[str, Any], sequence: int
    ) -> None:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(payload.get("name", "artifact"))).strip("-")
        link = {
            "uri": payload.get("uri"),
            "type": payload.get("artifact_type"),
            "sha256": payload.get("sha256"),
            "metadata": payload.get("metadata", {}),
        }
        manifest_path = self.run_capsule / "tracking-artifact-links.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            manifest = {"schema_version": 1, "artifacts": {}}
        artifacts = manifest.setdefault("artifacts", {})
        artifacts[f"{sequence:08d}-{slug or 'artifact'}"] = sanitize(link)
        self._atomic_write(
            manifest_path,
            (json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode(),
        )
        manifest_reference = json.dumps(
            {"uri": str(manifest_path), "entries": len(artifacts)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        self._deliver_log_batch(remote_run_id, {
            "params": {},
            "metrics": [],
            "tags": {"skynet.artifact_manifest": manifest_reference},
        }, sequence)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.settings.tracking_uri:
            raise TrackingRequestError("MLflow tracking URI is not configured")
        parsed = urllib.parse.urlsplit(self.settings.tracking_uri.rstrip("/"))
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        netloc = hostname
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        base_uri = urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))
        request_uri = f"{base_uri}{path}"
        if query:
            request_uri = f"{request_uri}?{urllib.parse.urlencode(query)}"

        headers = {"Accept": "application/json"}
        token = self.settings.token
        username = self.settings.username or (urllib.parse.unquote(parsed.username) if parsed.username else None)
        password = self.settings.password or (urllib.parse.unquote(parsed.password) if parsed.password else None)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        elif username:
            credentials = base64.b64encode(f"{username}:{password or ''}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {credentials}"
        body = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(request_uri, data=body, headers=headers, method=method)
        context = None
        if parsed.scheme == "https" and not self.settings.verify_tls:
            context = ssl._create_unverified_context()
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.timeout_seconds, context=context
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            try:
                detail = error.read().decode("utf-8", errors="replace")[:2000]
            except Exception:
                detail = str(error.reason)
            raise TrackingRequestError(
                f"MLflow returned HTTP {error.code}: {detail}", status_code=error.code
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise TrackingRequestError(f"MLflow transport unavailable: {error}") from error
        if not raw:
            return {}
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TrackingRequestError("MLflow returned an invalid JSON response") from error
        if not isinstance(result, dict):
            raise TrackingRequestError("MLflow returned an unexpected response")
        return result


class SessionCredentialStore:
    """Process-local credentials. Values are never serialized or exposed."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._credentials: dict[str, dict[str, str]] = {}
        self._credential_sources: dict[str, str] = {}
        self._credential_endpoints: dict[str, str] = {}
        self._connection_state: dict[str, dict[str, Any]] = {}

    def replace(
        self,
        provider: str,
        credentials: Mapping[str, str | None],
        *,
        source: str = "session",
        endpoint: str | None = None,
    ) -> None:
        values = {
            str(key): str(value)
            for key, value in credentials.items()
            if value is not None and str(value)
        }
        with self._lock:
            self._credentials[provider] = values
            if values and endpoint:
                self._credential_endpoints[provider] = endpoint.rstrip("/")
            else:
                self._credential_endpoints.pop(provider, None)
            if values:
                self._credential_sources[provider] = source
            else:
                self._credential_sources.pop(provider, None)

    def get(self, provider: str) -> dict[str, str]:
        with self._lock:
            return dict(self._credentials.get(provider, {}))

    def clear(self, provider: str) -> None:
        with self._lock:
            self._credentials.pop(provider, None)
            self._credential_sources.pop(provider, None)
            self._credential_endpoints.pop(provider, None)
            self._connection_state.pop(provider, None)

    def source(self, provider: str) -> str | None:
        with self._lock:
            return self._credential_sources.get(provider)

    def endpoint(self, provider: str) -> str | None:
        with self._lock:
            return self._credential_endpoints.get(provider)

    def mark_connected(self, provider: str, **public_state: Any) -> None:
        with self._lock:
            self._connection_state[provider] = {
                "connected": True,
                "last_error": None,
                "validated_at": _utc_now(),
                **sanitize(public_state),
            }

    def mark_error(self, provider: str, error: BaseException | str) -> None:
        with self._lock:
            self._connection_state[provider] = {
                "connected": False,
                "last_error": str(sanitize(str(error), secrets=tuple(
                    self._credentials.get(provider, {}).values()
                ))),
                "validated_at": _utc_now(),
            }

    def state(self, provider: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._connection_state.get(provider, {}))


SESSION_CREDENTIALS = SessionCredentialStore()


@dataclass(frozen=True)
class WandBSettings:
    base_url: str = "https://api.wandb.ai"
    api_key: str | None = field(default=None, repr=False)
    entity: str | None = None
    timeout_seconds: float = 8.0
    enabled: bool = True
    auto_flush: bool = True
    verify_tls: bool = True

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "WandBSettings":
        environment = os.environ if environ is None else environ
        timeout_text = environment.get("WANDB_HTTP_TIMEOUT", "8")
        try:
            timeout = max(0.1, float(timeout_text))
        except ValueError:
            timeout = 8.0
        return cls(
            base_url=(environment.get("WANDB_BASE_URL") or "https://api.wandb.ai").rstrip("/"),
            api_key=environment.get("WANDB_API_KEY") or None,
            entity=environment.get("WANDB_ENTITY") or None,
            timeout_seconds=timeout,
            enabled=_environment_bool(environment.get("SKYNET_WANDB_ENABLED"), True),
            auto_flush=_environment_bool(environment.get("SKYNET_WANDB_AUTO_FLUSH"), True),
            verify_tls=_environment_bool(environment.get("WANDB_VERIFY_TLS"), True),
        )

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.api_key and self.base_url)

    def public_dict(self) -> dict[str, Any]:
        return {
            "base_url": _sanitize_url(self.base_url),
            "entity": self.entity,
            "configured": self.configured,
            "enabled": self.enabled,
            "auto_flush": self.auto_flush,
            "verify_tls": self.verify_tls,
            "authentication": "api_key" if self.api_key else "none",
        }


def wandb_project_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    if not slug:
        raise ValueError("W&B project name resolves to an empty slug")
    return slug[:128]


def normalize_wandb_tag(value: Any) -> str:
    """Return a stable tag accepted by W&B's 1..64 character constraint."""

    tag = str(value)
    if not tag:
        return "_"
    if len(tag) <= WANDB_TAG_MAX_LENGTH:
        return tag
    digest = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:WANDB_TAG_HASH_LENGTH]
    prefix_length = WANDB_TAG_MAX_LENGTH - WANDB_TAG_HASH_LENGTH - 1
    return f"{tag[:prefix_length]}~{digest}"


def wandb_tag_label(key: Any, value: Any) -> str:
    """Encode structured metadata as a deterministic, W&B-safe label."""

    safe_value = sanitize(value)
    if isinstance(safe_value, str):
        value_text = safe_value
    else:
        value_text = json.dumps(
            safe_value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    return normalize_wandb_tag(f"{key}:{value_text}")


def wandb_web_base(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.rstrip("/"))
    hostname = (parsed.hostname or "").lower()
    if hostname == "api.wandb.ai":
        return "https://wandb.ai"
    path = re.sub(r"/api/?$", "", parsed.path.rstrip("/"))
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def mlflow_experiment_url(tracking_uri: str, experiment_id: str) -> str:
    return f"{tracking_uri.rstrip('/')}/#/experiments/{urllib.parse.quote(experiment_id, safe='')}"


def mlflow_run_url(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    return (
        f"{tracking_uri.rstrip('/')}/#/experiments/"
        f"{urllib.parse.quote(experiment_id, safe='')}/runs/{urllib.parse.quote(run_id, safe='')}"
    )


class WandBBridge:
    """Offline-first W&B GraphQL bridge with stable Skynet run identities."""

    def __init__(
        self,
        run_capsule: str | os.PathLike[str],
        settings: WandBSettings | None = None,
    ) -> None:
        self.run_capsule = Path(run_capsule).expanduser().resolve()
        self.settings = settings or WandBSettings.from_env()
        self.spool_path = self.run_capsule / WANDB_SPOOL_FILENAME
        self.state_path = self.run_capsule / WANDB_STATE_FILENAME
        self.lock_path = self.run_capsule / WANDB_LOCK_FILENAME
        self._thread_lock = threading.RLock()
        self._last_error: str | None = None
        self.run_capsule.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if not self.spool_path.exists():
                self._atomic_write(self.spool_path, b"")
            if not self.state_path.exists():
                self._write_state_unlocked(self._empty_state())

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "schema_version": SPOOL_SCHEMA_VERSION,
            "acked_through": 0,
            "projects": {},
            "runs": {},
            "updated_at": _utc_now(),
        }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _load_state_unlocked(self) -> dict[str, Any]:
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            loaded = {}
        state = self._empty_state()
        if isinstance(loaded, Mapping):
            state.update(loaded)
        for key in ("projects", "runs"):
            if not isinstance(state.get(key), dict):
                state[key] = {}
        return state

    def _write_state_unlocked(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        payload["updated_at"] = _utc_now()
        self._atomic_write(
            self.state_path,
            (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode(),
        )

    def _events_unlocked(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.spool_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid W&B spool record at line {line_number}") from error
            if not isinstance(event, dict) or not isinstance(event.get("sequence"), int):
                raise ValueError(f"Invalid W&B spool record at line {line_number}")
            events.append(event)
        return events

    def _enqueue(self, operation: str, payload: Mapping[str, Any]) -> TrackingResult:
        safe_payload = sanitize(payload, secrets=(self.settings.api_key,))
        with self._locked():
            events = self._events_unlocked()
            idempotency_key = safe_payload.get("idempotency_key")
            event = next(
                (
                    item
                    for item in events
                    if idempotency_key
                    and item.get("operation") == operation
                    and isinstance(item.get("payload"), Mapping)
                    and item["payload"].get("idempotency_key") == idempotency_key
                ),
                None,
            )
            if event is None:
                sequence = max((int(item["sequence"]) for item in events), default=0) + 1
                event = {
                    "schema_version": SPOOL_SCHEMA_VERSION,
                    "id": str(uuid.uuid4()),
                    "sequence": sequence,
                    "operation": operation,
                    "payload": safe_payload,
                    "created_at": _utc_now(),
                }
                existing = self.spool_path.read_bytes()
                line = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                self._atomic_write(self.spool_path, existing + line.encode() + b"\n")
            else:
                sequence = int(event["sequence"])
        report = self.drain_spool() if self.settings.auto_flush else DrainReport(
            0, 0, self.pending_count(), False
        )
        with self._locked():
            state = self._load_state_unlocked()
            delivered = int(state.get("acked_through", 0)) >= sequence
            run = state.get("runs", {}).get(str(payload.get("local_run_id", "")), {})
        return TrackingResult(
            event_id=str(event["id"]),
            sequence=sequence,
            delivered=delivered,
            queued=not delivered,
            remote_id=str(run.get("remote_id")) if run.get("remote_id") else None,
            error=report.errors[0] if report.errors else None,
        )

    def validate_connection(self) -> dict[str, Any]:
        if not self.settings.configured:
            raise TrackingRequestError("W&B API key is not configured")
        data = self._graphql(
            "query SkynetViewer { viewer { id username entity } }",
            {},
        )
        viewer = data.get("viewer")
        if not isinstance(viewer, Mapping) or not viewer.get("id"):
            raise TrackingRequestError("W&B authentication succeeded without a viewer identity")
        entity = self.settings.entity or viewer.get("entity") or viewer.get("username")
        if not entity:
            raise TrackingRequestError("W&B account has no default entity; specify an entity explicitly")
        return {
            "viewer_id": str(viewer["id"]),
            "username": str(viewer.get("username") or ""),
            "entity": str(entity),
        }

    def validate_entity(self, entity: str) -> dict[str, Any]:
        data = self._graphql(
            "query SkynetEntity($name: String!) { "
            "entity(name: $name) { id name readOnly } }",
            {"name": entity},
        )
        value = data.get("entity")
        if not isinstance(value, Mapping) or not value.get("id"):
            raise TrackingRequestError(f"W&B entity {entity!r} does not exist")
        if value.get("readOnly") is True:
            raise TrackingRequestError(f"W&B entity {entity!r} is read-only for this account")
        return {"id": str(value["id"]), "name": str(value.get("name") or entity)}

    def ensure_experiment(self, entity: str, project: str) -> TrackingResult:
        return self._enqueue("ensure_project", {"entity": entity, "project": project})

    def ensure_run(
        self,
        *,
        entity: str,
        project: str,
        local_run_id: str,
        run_name: str,
        group: str,
        tags: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> TrackingResult:
        return self._enqueue("ensure_run", {
            "entity": entity,
            "project": project,
            "local_run_id": local_run_id,
            "run_name": run_name,
            "group": group,
            "tags": tags or {},
            "config": config or {},
        })

    def log_params(self, local_run_id: str, params: Mapping[str, Any]) -> TrackingResult:
        return self._enqueue("log_params", {"local_run_id": local_run_id, "params": params})

    def log_metrics(
        self,
        local_run_id: str,
        metrics: Mapping[str, float | int],
        *,
        step: int = 0,
        timestamp_ms: int | None = None,
        idempotency_key: str | None = None,
    ) -> TrackingResult:
        return self._enqueue("log_metrics", {
            "local_run_id": local_run_id,
            "metrics": metrics,
            "step": step,
            "timestamp_ms": (
                timestamp_ms if timestamp_ms is not None else _milliseconds_now()
            ),
            **({"idempotency_key": idempotency_key} if idempotency_key else {}),
        })

    def set_tags(self, local_run_id: str, tags: Mapping[str, Any]) -> TrackingResult:
        return self._enqueue("set_tags", {"local_run_id": local_run_id, "tags": tags})

    def log_system_metrics(
        self, local_run_id: str, metrics: Mapping[str, float | int], *,
        timestamp_ms: int, runtime_seconds: float, idempotency_key: str,
    ) -> TrackingResult:
        return self._enqueue("log_system_metrics", {
            "local_run_id": local_run_id, "metrics": metrics,
            "timestamp_ms": timestamp_ms, "runtime_seconds": runtime_seconds,
            "idempotency_key": idempotency_key,
        })

    def log_artifact_link(
        self,
        local_run_id: str,
        *,
        name: str,
        uri: str,
        artifact_type: str | None = None,
        sha256: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrackingResult:
        return self._enqueue("artifact_link", {
            "local_run_id": local_run_id,
            "name": name,
            "uri": uri,
            "artifact_type": artifact_type,
            "sha256": sha256,
            "metadata": metadata or {},
        })

    def finish_run(self, local_run_id: str, *, status: str = "FINISHED") -> TrackingResult:
        return self._enqueue("finish_run", {
            "local_run_id": local_run_id,
            "status": status.upper(),
        })

    def reopen_run(self, local_run_id: str, attempt_number: int) -> TrackingResult:
        attempt = _validated_attempt_number(attempt_number)
        return self._enqueue("reopen_run", {
            "local_run_id": local_run_id,
            "attempt_number": attempt,
            "idempotency_key": f"reopen:{local_run_id}:attempt:{attempt}",
        })

    def pending_count(self) -> int:
        with self._locked():
            state = self._load_state_unlocked()
            cursor = int(state.get("acked_through", 0))
            return sum(int(item["sequence"]) > cursor for item in self._events_unlocked())

    def metric_idempotency_keys(self) -> set[str]:
        with self._locked():
            return {
                str(payload["idempotency_key"])
                for event in self._events_unlocked()
                if event.get("operation") in {"log_metrics", "log_system_metrics"}
                and isinstance((payload := event.get("payload")), Mapping)
                and payload.get("idempotency_key")
            }

    def metric_names_by_idempotency_key(self) -> dict[str, set[str]]:
        """Include queued and delivered fields so enrichment cannot duplicate them."""
        with self._locked():
            return {
                str(payload["idempotency_key"]): set(payload.get("metrics") or {})
                for event in self._events_unlocked()
                if event.get("operation") == "log_metrics"
                and isinstance((payload := event.get("payload")), Mapping)
                and payload.get("idempotency_key")
            }

    def binding(self, local_run_id: str) -> dict[str, str] | None:
        with self._locked():
            value = self._load_state_unlocked().get("runs", {}).get(local_run_id)
            return dict(value) if isinstance(value, Mapping) else None

    def project_binding(self, entity: str, project: str) -> dict[str, str] | None:
        with self._locked():
            key = f"{entity}/{project}"
            value = self._load_state_unlocked().get("projects", {}).get(key)
            return dict(value) if isinstance(value, Mapping) else None

    def drain_spool(self, *, limit: int | None = None) -> DrainReport:
        if not self.settings.configured:
            return DrainReport(0, 0, self.pending_count(), False)
        attempted = delivered = 0
        errors: list[str] = []
        with self._locked():
            events = self._events_unlocked()
            state = self._load_state_unlocked()
            pending = [event for event in events
                       if int(event["sequence"]) > int(state.get("acked_through", 0))]
            if pending and time.time() < float(state.get("retry_not_before", 0)):
                return DrainReport(0, 0, len(pending), False,
                                   ("W&B is busy. Saved metrics will sync automatically.",))
            if limit is not None:
                pending = pending[:max(0, limit)]
            index = 0
            while index < len(pending):
                batch = [pending[index]]
                event = batch[0]
                if event.get("operation") in {"log_metrics", "log_system_metrics"}:
                    operation = event["operation"]
                    run_id = event["payload"].get("local_run_id")
                    for following in pending[index + 1:index + 100]:
                        if (following.get("operation") != operation
                                or following["payload"].get("local_run_id") != run_id):
                            break
                        batch.append(following)
                    if len(batch) > 1:
                        event = {"operation": operation + "_batch", "payload": {
                            "local_run_id": run_id, "samples": [item["payload"] for item in batch]}}
                attempted += len(batch)
                try:
                    delivery_state = copy.deepcopy(state)
                    self._deliver_event(event, delivery_state)
                except Exception as error:
                    message = str(sanitize(str(error), secrets=(self.settings.api_key,)))
                    if isinstance(error, TrackingRequestError) and error.status_code == 429:
                        retries = int(state.get("rate_limit_retries", 0)) + 1
                        delay = max(error.retry_after or 0, min(900, 60 * 2 ** min(retries - 1, 4)))
                        state.update(retry_not_before=time.time() + delay, rate_limit_retries=retries)
                        self._write_state_unlocked(state)
                        message = "W&B is busy. Saved metrics will sync automatically."
                    self._last_error = message
                    errors.append(message)
                    break
                state = delivery_state
                state["acked_through"] = int(batch[-1]["sequence"])
                state.pop("retry_not_before", None)
                state.pop("rate_limit_retries", None)
                self._last_error = None
                self._write_state_unlocked(state)
                delivered += len(batch)
                index += len(batch)
            remaining = sum(int(event["sequence"]) > int(state.get("acked_through", 0)) for event in events)
        return DrainReport(attempted, delivered, remaining, not errors, tuple(errors))

    def _deliver_event(self, event: Mapping[str, Any], state: dict[str, Any]) -> None:
        operation = str(event.get("operation"))
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("W&B event payload is invalid")
        if operation == "ensure_project":
            key = f"{payload['entity']}/{payload['project']}"
            state["projects"][key] = {
                "remote_id": key,
                "url": f"{wandb_web_base(self.settings.base_url)}/{urllib.parse.quote(str(payload['entity']), safe='')}/{urllib.parse.quote(str(payload['project']), safe='')}",
            }
            return

        local_run_id = str(payload.get("local_run_id") or "")
        run = state["runs"].get(local_run_id)
        if operation == "ensure_run":
            entity = str(payload["entity"])
            project = str(payload["project"])
            remote_name = local_run_id
            existing = run if isinstance(run, Mapping) else {}
            merged_config = dict(existing.get("config") or {})
            merged_config.update(dict(payload.get("config") or {}))
            merged_tags = dict(existing.get("tags") or {})
            merged_tags.update(dict(payload.get("tags") or {}))
            response = self._upsert_run(
                entity=entity,
                project=project,
                remote_name=remote_name,
                storage_id=(
                    str(existing["storage_id"]) if existing.get("storage_id") else None
                ),
                display_name=str(payload.get("run_name") or local_run_id),
                group=str(payload.get("group") or ""),
                tags=merged_tags,
                config=merged_config,
            )
            bucket = response.get("bucket")
            if not isinstance(bucket, Mapping) or not bucket.get("id"):
                raise TrackingRequestError("W&B upsert-run response did not contain a run id")
            actual_name = str(bucket.get("name") or remote_name)
            state["runs"][local_run_id] = {
                "remote_id": actual_name,
                "storage_id": str(bucket["id"]),
                "name": actual_name,
                "entity": entity,
                "project": project,
                "url": f"{wandb_web_base(self.settings.base_url)}/{urllib.parse.quote(entity, safe='')}/{urllib.parse.quote(project, safe='')}/runs/{urllib.parse.quote(actual_name, safe='')}",
                "config": merged_config,
                "summary": dict(existing.get("summary") or {}),
                "tags": merged_tags,
                "history_offset": int(existing.get("history_offset") or 0),
                "events_offset": int(existing.get("events_offset") or 0),
                **({"state": existing["state"]} if existing.get("state") else {}),
            }
            return
        if not isinstance(run, dict):
            raise TrackingRequestError(f"No remote W&B run exists for local run {local_run_id}")

        if operation == "log_params":
            run["config"].update(dict(payload.get("params") or {}))
        elif operation in {"log_metrics", "log_metrics_batch"}:
            samples = payload["samples"] if operation == "log_metrics_batch" else [payload]
            self._append_history_rows(run, [
                {**dict(sample.get("metrics") or {}),
                 "_step": int(sample.get("step") or 0),
                 "_timestamp": int(sample.get("timestamp_ms") or _milliseconds_now()) / 1000.0}
                for sample in samples
            ])
            for sample in samples:
                run["summary"].update(dict(sample.get("metrics") or {}))
        elif operation in {"log_system_metrics", "log_system_metrics_batch"}:
            samples = payload["samples"] if operation.endswith("_batch") else [payload]
            # W&B's system stream has its own cursor and clock, independent of
            # training steps. Native SDK runs retain ownership of their stream.
            self._append_system_rows(run, [
                {**{"system." + key: value for key, value in sample["metrics"].items()},
                 "_wandb": True, "_timestamp": sample["timestamp_ms"] / 1000.0,
                 "_runtime": max(0.0, float(sample["runtime_seconds"]))}
                for sample in samples
            ])
            return
        elif operation == "set_tags":
            run["tags"].update(dict(payload.get("tags") or {}))
        elif operation == "artifact_link":
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(payload.get("name") or "artifact"))
            run["summary"][f"skynet/artifact/{slug}"] = str(payload.get("uri") or "")
        elif operation == "finish_run":
            status = str(payload.get("status") or "FINISHED").upper()
            run["state"] = "finished" if status == "FINISHED" else "failed"
            # W&B's supported public update API has no running -> finished
            # transition. Persist the canonical terminal outcome as summary and
            # tags rather than relying on an undocumented GraphQL transition.
            run["summary"]["skynet/status"] = status
            run["tags"][RUN_STATUS_TAG] = status.lower()
        elif operation == "reopen_run":
            attempt = _validated_attempt_number(payload.get("attempt_number"))
            remote_id = str(run.get("remote_id") or "")
            storage_id = str(run.get("storage_id") or "")
            history_offset = int(run.get("history_offset") or 0)
            run["state"] = "running"
            run["summary"]["skynet/status"] = "RUNNING"
            run["tags"][RUN_STATUS_TAG] = "running"
            run["tags"][ATTEMPT_NUMBER_TAG] = attempt
            response = self._upsert_run(
                entity=str(run["entity"]),
                project=str(run["project"]),
                remote_name=str(run["name"]),
                storage_id=storage_id,
                display_name=None,
                group=None,
                tags=run.get("tags", {}),
                config=run.get("config", {}),
                summary=run.get("summary", {}),
                state_value="running",
            )
            bucket = response.get("bucket")
            if (
                not isinstance(bucket, Mapping)
                or str(bucket.get("id") or "") != storage_id
                or str(bucket.get("name") or "") != remote_id
            ):
                raise TrackingRequestError(
                    "W&B reopen changed the existing run identity"
                )
            if int(run.get("history_offset") or 0) != history_offset:
                raise TrackingRequestError(
                    "W&B reopen changed the existing history offset"
                )
            return
        else:
            raise ValueError(f"Unsupported W&B tracking operation: {operation}")
        self._upsert_run(
            entity=str(run["entity"]),
            project=str(run["project"]),
            remote_name=str(run["name"]),
            storage_id=str(run["storage_id"]),
            display_name=None,
            group=None,
            tags=run.get("tags", {}),
            config=run.get("config", {}),
            summary=run.get("summary", {}),
        )

    def _append_history_rows(self, run: dict[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
        """Upload queued samples together while retaining each step and timestamp."""
        self._append_stream_rows(run, rows, "wandb-history.jsonl", "history_offset")

    def _append_system_rows(self, run: dict[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
        self._append_stream_rows(run, rows, "wandb-events.jsonl", "events_offset")

    def _append_stream_rows(self, run, rows, filename, offset_key):
        offset = int(run.get(offset_key) or 0)
        lines = [json.dumps(sanitize(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
                 for row in rows]
        entity = urllib.parse.quote(str(run["entity"]), safe="")
        project = urllib.parse.quote(str(run["project"]), safe="")
        remote_name = urllib.parse.quote(str(run["name"]), safe="")
        uri = (
            f"{self.settings.base_url.rstrip('/')}/files/"
            f"{entity}/{project}/{remote_name}/file_stream"
        )
        token = base64.b64encode(f"api:{self.settings.api_key}".encode()).decode("ascii")
        body = json.dumps(
            {
                "files": {
                    filename: {
                        "offset": offset,
                        "content": lines,
                    }
                }
            },
            separators=(",", ":"),
        ).encode()
        request = urllib.request.Request(
            uri,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Basic {token}",
                "User-Agent": "skynet-slurm-console/0.2",
            },
            method="POST",
        )
        context = None
        if urllib.parse.urlsplit(uri).scheme == "https" and not self.settings.verify_tls:
            context = ssl._create_unverified_context()
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.timeout_seconds, context=context
            ) as response:
                response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:2000]
            raise TrackingRequestError(
                f"W&B {filename} returned HTTP {error.code}: {detail}",
                status_code=error.code,
                retry_after=_retry_after_seconds((error.headers or {}).get("Retry-After")),
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise TrackingRequestError(f"W&B {filename} transport unavailable: {error}") from error
        run[offset_key] = offset + len(lines)

    @staticmethod
    def _wandb_config(
        values: Mapping[str, Any],
        *,
        tag_metadata: Mapping[str, Any] | None = None,
    ) -> str:
        encoded = {
            str(key): {"value": sanitize(value)} for key, value in values.items()
        }
        if tag_metadata is not None:
            # W&B tags are short labels rather than key/value metadata. Keep
            # the complete, typed values in config so shortening a label never
            # discards provenance and legacy queued events migrate on replay.
            encoded[WANDB_TAG_METADATA_CONFIG_KEY] = {
                "value": sanitize({str(key): value for key, value in tag_metadata.items()})
            }
        return json.dumps(
            encoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def _upsert_run(
        self,
        *,
        entity: str,
        project: str,
        remote_name: str,
        storage_id: str | None = None,
        display_name: str | None,
        group: str | None,
        tags: Mapping[str, Any],
        config: Mapping[str, Any],
        summary: Mapping[str, Any] | None = None,
        state_value: str | None = None,
    ) -> dict[str, Any]:
        # This is the same public GraphQL shape used by wandb.Api._create_run.
        # Supplying the stable Skynet UUID as `name` makes retries idempotent and
        # lets W&B create the project on first use.
        if storage_id is None or state_value:
            data = self._graphql(
                """
                mutation SkynetCreateRun(
                    $project: String, $entity: String, $name: String!, $state: String
                ) {
                    upsertBucket(input: {
                        modelName: $project, entityName: $entity, name: $name, state: $state
                    }) { bucket { id name displayName } inserted }
                }
                """,
                {
                    "entity": entity,
                    "project": project,
                    "name": remote_name,
                    "state": state_value or "running",
                },
            )
            value = data.get("upsertBucket")
            if not isinstance(value, Mapping):
                raise TrackingRequestError("W&B create-run returned an unexpected response")
            bucket = value.get("bucket")
            if not isinstance(bucket, Mapping) or not bucket.get("id"):
                raise TrackingRequestError("W&B create-run response did not contain a run id")
            storage_id = str(bucket["id"])
        else:
            value = {"bucket": {"id": storage_id, "name": remote_name}, "inserted": False}

        # Public Run.update uses the storage id for metadata updates. Config is
        # encoded using W&B's {value, desc} representation.
        self._graphql(
            """
            mutation SkynetUpdateRun(
                $id: String!, $displayName: String, $tags: [String!],
                $config: JSONString!, $groupName: String
            ) {
                upsertBucket(input: {
                    id: $id, displayName: $displayName, tags: $tags,
                    config: $config, groupName: $groupName
                }) { bucket { id name displayName } }
            }
            """,
            {
                "id": storage_id,
                "displayName": display_name,
                "tags": [
                    wandb_tag_label(key, value)
                    for key, value in sorted(tags.items())
                ],
                "config": self._wandb_config(config, tag_metadata=tags),
                "groupName": group,
            },
        )
        if summary:
            # This is the mutation used by wandb.apis.public.summary.HTTPSummary.
            self._graphql(
                """
                mutation SkynetUpdateSummary($id: String, $summaryMetrics: JSONString) {
                    upsertBucket(input: {id: $id, summaryMetrics: $summaryMetrics}) {
                        bucket { id }
                    }
                }
                """,
                {
                    "id": storage_id,
                    "summaryMetrics": json.dumps(
                        sanitize(summary),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                },
            )
        if not isinstance(value, Mapping):
            raise TrackingRequestError("W&B upsert-run returned an unexpected response")
        return dict(value)

    def _graphql(self, query: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        if not self.settings.api_key:
            raise TrackingRequestError("W&B API key is not configured")
        uri = f"{self.settings.base_url.rstrip('/')}/graphql"
        token = base64.b64encode(f"api:{self.settings.api_key}".encode()).decode("ascii")
        body = json.dumps({"query": query, "variables": variables}, separators=(",", ":")).encode()
        request = urllib.request.Request(
            uri,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Basic {token}",
                "User-Agent": "skynet-slurm-console/0.2",
            },
            method="POST",
        )
        context = None
        if urllib.parse.urlsplit(uri).scheme == "https" and not self.settings.verify_tls:
            context = ssl._create_unverified_context()
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.timeout_seconds, context=context
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:2000]
            raise TrackingRequestError(
                f"W&B returned HTTP {error.code}: {detail}", status_code=error.code
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise TrackingRequestError(f"W&B transport unavailable: {error}") from error
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TrackingRequestError("W&B returned invalid JSON") from error
        if not isinstance(result, Mapping):
            raise TrackingRequestError("W&B returned an unexpected response")
        errors = result.get("errors")
        if isinstance(errors, list) and errors:
            messages = [str(item.get("message") if isinstance(item, Mapping) else item) for item in errors]
            raise TrackingRequestError(f"W&B GraphQL error: {'; '.join(messages)}")
        data = result.get("data")
        if not isinstance(data, Mapping):
            raise TrackingRequestError("W&B response did not contain data")
        return dict(data)


__all__ = [
    "DrainReport",
    "MLflowBridge",
    "TrackingRequestError",
    "TrackingResult",
    "TrackingSettings",
    "SESSION_CREDENTIALS",
    "SessionCredentialStore",
    "WandBBridge",
    "WandBSettings",
    "mlflow_experiment_url",
    "mlflow_run_url",
    "sanitize",
    "wandb_project_slug",
    "wandb_web_base",
]
