"""Bound native renderer hangs outside the simulator's Python interpreter."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

try:
    from observation_contracts import PREPARE_SCHEMA
except ImportError:
    from skynet_app.observation_contracts import PREPARE_SCHEMA

PROGRESS_SCHEMA = "skynet.observation-progress/v1"
INITIALIZATION_TIMEOUT = 300
FRAME_TIMEOUT = 120
PHASES = {"initializing", "episode", "capturing", "frame", "publishing", "closing"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path, limit):
    path = Path(path)
    if not path.exists():
        return None
    if path.is_symlink() or not 0 < path.stat().st_size <= limit:
        raise ValueError("Renderer control file is invalid or exceeds its size limit")
    return json.loads(path.read_text())


class ProgressReporter:
    """Report actual completed work, with bounded shared-filesystem writes."""
    def __init__(self, request, path, *, interval=1.0):
        self.request, self.path, self.interval = request, path, interval
        self.sequence, self.last_write, self.last_phase = 0, float("-inf"), None

    def __call__(self, phase, **fields):
        now = time.monotonic()
        if phase == self.last_phase == "frame" and now - self.last_write < self.interval:
            return
        self.sequence += 1
        write_json(self.path, dict(schema=PROGRESS_SCHEMA, request_id=self.request["request_id"],
            attempt_token=self.request["attempt_token"], sequence=self.sequence, phase=phase, **fields))
        self.last_write, self.last_phase = now, phase


class ProgressTracker:
    def __init__(self, request):
        self.request = request
        self.sequence, self.signature = 0, None
        self.frames = {}
        self.episodes = {source["episode_key"] for source in request.get("sources", [])}
        self.artifacts = {item["artifact_key"] for item in request.get("requests", [])}

    def accept(self, value):
        if value is None:
            return None
        if (value.get("schema") != PROGRESS_SCHEMA or value.get("request_id") != self.request["request_id"]
                or value.get("attempt_token") != self.request["attempt_token"]
                or value.get("phase") not in PHASES or type(value.get("sequence")) is not int):
            raise ValueError("Renderer progress belongs to a different attempt or has an invalid phase")
        if value["sequence"] <= self.sequence:
            return None
        phase = value["phase"]
        if phase in {"episode", "capturing", "frame"} and value.get("episode_key") not in self.episodes:
            raise ValueError("Renderer progress identifies an unexpected recording")
        if phase == "frame":
            key, frame, total = value["episode_key"], value.get("frame"), value.get("frames")
            if (type(frame) is not int or type(total) is not int or not 1 <= frame <= total <= 6000
                    or frame <= self.frames.get(key, 0)):
                raise ValueError("Renderer progress must advance completed recording frames")
            self.frames[key] = frame
        if phase == "publishing" and value.get("artifact_key") not in self.artifacts:
            raise ValueError("Renderer progress identifies an unexpected artifact")
        signature = (phase, value.get("episode_key"), value.get("frame"), value.get("artifact_key"))
        self.sequence = value["sequence"]
        if signature == self.signature:
            return None
        self.signature = signature
        return phase


def _receipt(path, request):
    value = _read_json(path, 10_000_000)
    if value is None:
        return None
    if (value.get("schema") != PREPARE_SCHEMA or value.get("request_id") != request["request_id"]
            or value.get("attempt_token") != request["attempt_token"] or value.get("state") not in {"READY", "FAILED"}
            or not isinstance(value.get("artifacts"), list)):
        raise ValueError("Renderer result belongs to a different attempt or has an invalid state")
    expected = {item["artifact_key"]: item["output_dir"] for item in request.get("requests", [])}
    found = set()
    for item in value["artifacts"]:
        key = item.get("artifact_key")
        if (key not in expected or key in found or item.get("path") != expected[key]
                or item.get("status") != "READY" or not re.fullmatch("[a-f0-9]{64}", item.get("manifest_sha256", ""))):
            raise ValueError("Renderer result contains an unexpected artifact")
        found.add(key)
    if value["state"] == "READY" and found != expected.keys():
        raise ValueError("Renderer result is missing requested artifacts")
    return value


def _stop_group(process, grace):
    # Only the process group created by this supervisor is ever signalled.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Also stop descendants which outlived the direct renderer child.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=grace)
    deadline = time.monotonic() + grace
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() >= deadline:
            # A surviving or unreaped descendant still owns this group. Leave
            # staging to producer deletion rather than racing a possible writer.
            return False
        time.sleep(min(0.05, grace))


def _cleanup_staging(request, output):
    """Remove only this producer's unpublished work after all its writers stop."""
    requested = request.get("staging_root")
    if not isinstance(requested, str):
        return False
    parent = output.parent.resolve(strict=True)
    staging = parent / "staging"
    if (output.parent.absolute() != parent or Path(requested) != staging
            or staging.is_symlink() or not staging.is_dir()):
        return False
    # Raw worker requests may place published outputs or inputs in staging.
    # Such requests never grant the supervisor permission to remove those bytes.
    protected = [source.get("recording") for source in request.get("sources", [])]
    for item in request.get("requests", []):
        protected.append(item.get("output_dir"))
        protected.extend(dependency.get("path") for dependency in item.get("dependencies", []))
    for value in protected:
        if value is None:
            continue
        if not isinstance(value, str):
            return False
        # Protect both a symlink's declared path and its resolved target.
        for path in (Path(value).absolute(), Path(value).resolve()):
            if path.is_relative_to(staging) or staging.is_relative_to(path):
                return False
    # rmtree removes internal symlinks themselves, never their external targets.
    # It also rejects a staging root replaced by a symlink before removal.
    shutil.rmtree(staging)
    return True


def supervise(request, request_path, result_path, *, worker_path, initialization_timeout=INITIALIZATION_TIMEOUT,
              frame_timeout=FRAME_TIMEOUT, termination_grace=5.0, poll_interval=0.2):
    """Run the same frozen request in a child and always persist its outcome.

    Child-only control files avoid accepting stale receipts from a previous
    invocation. A valid receipt survives native shutdown failures; absent or
    mismatched receipts become FAILED under the original request/attempt IDs.
    """
    if any(not math.isfinite(value) or value <= 0
           for value in (initialization_timeout, frame_timeout, termination_grace, poll_interval)):
        raise ValueError("Renderer supervision limits must be finite and positive")
    output = Path(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    started, process, outcome, problem = time.monotonic(), None, None, None
    interrupted, previous_handlers = [], {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda number, _frame: interrupted.append(number))
    try:
        with tempfile.TemporaryDirectory(prefix=".render-supervisor-", dir=output.parent) as directory:
            child_result, progress = Path(directory) / "result.json", Path(directory) / "progress.json"
            try:
                command = [sys.executable, str(Path(worker_path).resolve()), "--request", str(Path(request_path).resolve()),
                           "--result", str(child_result), "--render-child", "--progress", str(progress)]
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
                tracker, phase = ProgressTracker(request), "initializing"
                deadline = time.monotonic() + initialization_timeout
                while process.poll() is None:
                    if interrupted:
                        raise RuntimeError(f"Observation renderer interrupted by signal {interrupted[0]}")
                    changed = tracker.accept(_read_json(progress, 4096))
                    if changed:
                        phase = changed
                        limit = initialization_timeout if phase in {"initializing", "episode"} else frame_timeout
                        deadline = time.monotonic() + limit
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Observation renderer stalled during {phase}; no progress within "
                                           f"{initialization_timeout if phase in {'initializing', 'episode'} else frame_timeout:g}s")
                    time.sleep(poll_interval)
            except Exception as exc:
                problem = str(exc)
            finally:
                if process is not None:
                    try:
                        stopped = _stop_group(process, termination_grace)
                    except Exception as exc:
                        stopped = False
                        problem = f"{problem + '; ' if problem else ''}Could not stop renderer process group: {exc}"
                    if stopped:
                        try:
                            _cleanup_staging(request, output)
                        except Exception as exc:
                            # Cleanup failure must not discard a valid published receipt.
                            print(f"Could not clean renderer staging: {exc}", file=sys.stderr, flush=True)
            try:
                outcome = _receipt(child_result, request)
            except Exception as exc:
                problem = f"{problem + '; ' if problem else ''}{exc}"
            if outcome is None:
                code = process.returncode if process is not None else None
                outcome = dict(schema=PREPARE_SCHEMA, request_id=request.get("request_id"),
                    attempt_token=request.get("attempt_token"), state="FAILED", artifacts=[],
                    error=problem or f"Observation renderer exited with code {code} without a result receipt")
            outcome.setdefault("duration_seconds", round(time.monotonic() - started, 6))
            write_json(output, outcome)
            return outcome
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
