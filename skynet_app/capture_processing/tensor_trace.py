"""Verified, bounded, persistent tensor traces with no web-server ML dependency."""

from concurrent.futures import Future
import hashlib
import json
import math
from pathlib import Path
import shlex
import threading

from skynet_app.database import utc_now

WORKER = Path(__file__).with_name("tensor_trace_runner.py")
STEP_WIDTHS = [35, 35, 128, 128, 128, 128, 28, 28, 28]
STEP_IDS = [
    "input",
    "normalize",
    "layer-0",
    "layer-1",
    "layer-2",
    "layer-3",
    "layer-4",
    "denormalize",
    "clamp",
]


class TraceError(ValueError):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def validate_trace(data, frame, artifacts, frame_count):
    """Reject partial, mismatched, nonfinite or unsupported remote output."""

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite trace value")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        if isinstance(value, list):
            for item in value:
                finite(item)

    def require(condition):
        if not condition:
            raise ValueError("Incompatible trace data")

    try:
        finite(data)
        require(data["schema"] == "skynet.tensor-trace/v1")
        require(data["policy"] == "skynet-state-bc/v1" and data["device"] == "cpu")
        require(data["frame"] == frame and data["frame_count"] == frame_count)
        require(data["checkpoint_sha256"] == artifacts["state-bc.pt"]["sha256"])
        require(data["dataset_sha256"] == artifacts["dataset.hdf5"]["sha256"])
        require([s["id"] for s in data["steps"]] == STEP_IDS)
        for step, width in zip(data["steps"], STEP_WIDTHS, strict=True):
            require(step["shape"] == [1, width] and step["dtype"] == "float32")
            require(len(step["values"]) == width)
            require(all(type(v) in (float, int) for v in step["values"]))
            require(
                all(
                    isinstance(step[k], str)
                    for k in ("title", "explanation", "formula")
                )
            )
            require(
                all(
                    type(step["stats"][k]) in (float, int)
                    for k in ("min", "max", "mean")
                )
            )
            require(isinstance(step["parameters"], list))
            for parameter in step["parameters"]:
                require(isinstance(parameter["name"], str))
                require(isinstance(parameter["shape"], list))
                require(
                    all(type(v) is int and 0 < v <= 128 for v in parameter["shape"])
                )
                if "values" in parameter:
                    require(len(parameter["values"]) == math.prod(parameter["shape"]))
                    require(all(type(v) in (float, int) for v in parameter["values"]))
                else:
                    require(parameter["count"] == math.prod(parameter["shape"]))
        require(len(data["demonstrated_action"]) == 28)
        require(all(type(v) in (float, int) for v in data["demonstrated_action"]))
        require(
            all(type(v) is int and 0 <= v < 28 for v in data["changed_action_indices"])
        )
        require(type(data["source_index"]) is int and data["source_index"] >= 0)
        require(isinstance(data["provenance"], str))
        require(type(data["parameter_count"]) is int and data["parameter_count"] > 0)
        require(type(data["action_mae"]) in (int, float) and data["action_mae"] >= 0)
    except (KeyError, TypeError, ValueError) as error:
        raise TraceError(
            "The runtime returned an incomplete or incompatible tensor trace. No values are being substituted.",
            502,
        ) from error
    return data


class TensorTraceService:
    def __init__(self, processing):
        self.processing = processing
        self.lock = threading.Lock()
        self.inflight = {}
        self.slots = threading.BoundedSemaphore(2)
        with processing.database.transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS tensor_trace_cache (cache_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL, created_at TEXT NOT NULL)"
            )

    def get(self, identifier, frame=0):
        job = self.processing.get(identifier)
        if job["state"] != "SUCCEEDED":
            raise TraceError(
                "Tensor inspection becomes available after a completed cycle and its artifacts are verified.",
                409,
            )
        result = job.get("result") or {}
        if result.get("dataset", {}).get("training_policy") != "skynet-state-bc/v1":
            raise TraceError(
                "Unsupported model: layer inspection currently supports saved DexVerse state-BC v1 checkpoints. GR00T and pi0.5 need their own trace adapters."
            )
        count = result["dataset"]["frames"]
        if type(frame) is not int or not 0 <= frame < count:
            raise TraceError(f"Choose a dataset frame from 0 to {count - 1}.")
        artifacts = {
            name: result.get("artifacts", {}).get(name)
            for name in ("state-bc.pt", "dataset.hdf5")
        }
        if not all(artifacts.values()):
            raise TraceError(
                "This cycle is missing a verified checkpoint or dataset.", 409
            )
        worker = WORKER.read_text()
        key = hashlib.sha256(
            json.dumps(
                [artifacts, frame, hashlib.sha256(worker.encode()).hexdigest()],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        with self.lock:
            with self.processing.database.connection() as connection:
                row = connection.execute(
                    "SELECT payload_json FROM tensor_trace_cache WHERE cache_key=?",
                    (key,),
                ).fetchone()
            if row:
                try:
                    data = json.loads(row["payload_json"])
                    return validate_trace(data, frame, artifacts, count)
                except (ValueError, TypeError):
                    with self.processing.database.transaction() as connection:
                        connection.execute(
                            "DELETE FROM tensor_trace_cache WHERE cache_key=?", (key,)
                        )
            future = self.inflight.get(key)
            owner = future is None
            if owner:
                future = self.inflight[key] = Future()
        if not owner:
            return future.result(timeout=60)
        acquired = False
        try:
            acquired = self.slots.acquire(blocking=False)
            if not acquired:
                raise TraceError(
                    "Two traces are already loading. Wait for them to finish, then try again.",
                    503,
                )
            runtime = job["config"]["pipeline"]["runtime"] + "/bin/python"
            args = [
                runtime,
                "-",
                job["root"] + "/output",
                str(frame),
                json.dumps(artifacts),
            ]
            # The saved gateway is intentional; do not silently change runtime/host.
            command = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 50s " + shlex.join(
                args
            )
            output = self.processing.cluster.ssh(
                job["gateway"], command, stdin=worker, timeout=55
            )
            if len(output) > 150000:
                raise TraceError(
                    "Runtime trace exceeded the supported response size.", 502
                )
            try:
                data = json.loads(output)
            except ValueError as error:
                raise TraceError(
                    "Runtime did not return a readable tensor trace. Check the cluster connection and retry.",
                    502,
                ) from error
            validate_trace(data, frame, artifacts, count)
            with self.processing.database.transaction() as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO tensor_trace_cache VALUES (?, ?, ?)",
                    (key, json.dumps(data, allow_nan=False), utc_now()),
                )
                connection.execute(
                    "DELETE FROM tensor_trace_cache WHERE cache_key IN (SELECT cache_key FROM tensor_trace_cache ORDER BY created_at DESC, cache_key LIMIT -1 OFFSET 128)"
                )
            future.set_result(data)
            return data
        except Exception as error:
            future.set_exception(error)
            raise
        finally:
            if acquired:
                self.slots.release()
            with self.lock:
                self.inflight.pop(key, None)
