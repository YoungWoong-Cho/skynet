"""Download and inspect completed native captures without running a GPU pipeline."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import pickle
from pathlib import Path, PurePosixPath
import threading

import numpy as np

from .database import canonical_json, utc_now
from .live_xr_catalog import selection

MAX_BYTES = 100 * 1024 * 1024
MAX_PREVIEW_FRAMES = 2000
SCHEMA = "skynet.live-review/v1"


class ArrayUnpickler(pickle.Unpickler):
    """Only the NumPy array constructors emitted by DexVerse are accepted."""

    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        core = np._core if hasattr(np, "_core") else np.core
        for prefix in ("numpy.core", "numpy._core"):
            allowed[(prefix + ".multiarray", "_reconstruct")] = (
                core.multiarray._reconstruct
            )
            allowed[(prefix + ".multiarray", "scalar")] = core.multiarray.scalar
            allowed[(prefix + ".numeric", "_frombuffer")] = core.numeric._frombuffer
        if (module, name) not in allowed:
            raise ValueError(f"Unsupported recording object: {module}.{name}")
        return allowed[(module, name)]


def numeric_tree(value, depth=0):
    if depth > 12:
        raise ValueError("Recording state nesting is too deep")
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("Recording state keys must be text")
        return {k: numeric_tree(v, depth + 1) for k, v in value.items()}
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("Recording contains nonfinite or nonnumeric state values")
    return array.tolist()


def inspect(path, profile):
    """Read a bounded, application-produced capture; retain every raw byte separately."""
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Review supports native files up to 100 MB")
    raw = path.read_bytes()
    payload = ArrayUnpickler(io.BytesIO(raw)).load()
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "dexverse_trajectory"
        or payload.get("schema_version") != 3
    ):
        raise ValueError(
            "Unsupported native recording format; expected DexVerse trajectory v3"
        )
    task, hand = selection(profile["task"], profile["robot"])
    if payload.get("task") != task["key"] or payload.get("robot_type") != hand["key"]:
        raise ValueError(
            "The saved recording does not match this session's hand and task"
        )
    episodes = payload.get("episodes")
    if (
        not isinstance(episodes, list)
        or not episodes
        or payload.get("num_episodes") != len(episodes)
    ):
        raise ValueError("Recording contains no complete demonstration")
    reviewed = []
    for index, episode in enumerate(episodes):
        actions = np.asarray(episode.get("actions"))
        if (
            actions.ndim != 2
            or not all(actions.shape)
            or actions.dtype.kind not in "fiu"
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Recording contains invalid action values")
        states = episode.get("states")
        steps = len(actions)
        if (
            episode.get("num_steps") != steps
            or not isinstance(states, list)
            or len(states) != steps + 1
        ):
            raise ValueError(
                "Recording needs one initial scene state and one state per action"
            )
        if episode.get("success") is not True:
            raise ValueError("This recording did not meet the task success condition")
        # Validate every state, even when the browser preview is sampled.
        indices = set(
            np.linspace(
                0, steps, min(steps + 1, MAX_PREVIEW_FRAMES), dtype=int
            ).tolist()
        )
        frames = []
        for i, state in enumerate(states):
            if not isinstance(state, dict) or not state.get("articulation"):
                raise ValueError("Recording is missing robot scene states")
            checked = numeric_tree(state)
            if i in indices:
                frames.append(
                    {
                        "state_index": i,
                        "time_seconds": i / 60,
                        "action_index": i - 1 if i else None,
                        "action": actions[i - 1].tolist() if i else None,
                        "state": checked,
                    }
                )
        reviewed.append(
            {
                "index": index,
                "steps": steps,
                "success": True,
                "action_shape": list(actions.shape),
                "duration_seconds": steps / 60,
                "state_count": len(states),
                "sampled": len(frames) != len(states),
                "frames": frames,
                "video": episode.get("skynet_video"),
            }
        )
    return {
        "schema": SCHEMA,
        "task": task["key"],
        "task_name": task["name"],
        "robot": hand["key"],
        "hand_name": hand["name"],
        "episodes": reviewed,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "simulation_hz": 60,
        "time_note": "Simulation time at 60 Hz; pauses and headset setup are excluded.",
        "value_note": "Actions and scene states use DexVerse's saved simulator order. Joint names and action units are not stored in this format; no guessed mapping is applied.",
        "reviewed_at": utc_now(),
    }


class LiveReviewService:
    def __init__(self, live, root=None):
        self.live = live
        self.root = Path(root or live.root / "data/live-reviews")
        self.lock = threading.RLock()
        self.active = set()
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="live-review"
        )

    def source(self, identifier, index):
        job = self.live.get(identifier)
        if not job.get("recordings"):
            raise ValueError(
                "Review is available after a successful demonstration has been saved"
            )
        files = job.get("recordings", [])
        if not 0 <= index < len(files):
            raise KeyError("Recording not found")
        path = PurePosixPath(files[index])
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.is_relative_to("recordings")
            or path.suffix != ".pkl"
        ):
            raise ValueError("Invalid saved recording path")
        return job, job["root"] + "/output/" + str(path)

    def directory(self, identifier, index):
        # source() validates the identifier via an existing session, not a filesystem path.
        self.source(identifier, index)
        return self.root / identifier / str(index)

    def status(self, identifier, index=0):
        directory = self.directory(identifier, index)
        path = directory / "status.json"
        result = (
            json.loads(path.read_text())
            if path.exists()
            else {"state": "NOT_DOWNLOADED"}
        )
        if result["state"] == "DOWNLOADING" and (identifier, index) not in self.active:
            return {
                "state": "FAILED",
                "error": "Download was interrupted. Retry to resume review.",
            }
        if result["state"] == "READY" and not all(
            (directory / name).is_file()
            for name in ("review.json", "summary.json", "recording.pkl")
        ):
            return {
                "state": "FAILED",
                "error": "The local copy is missing a file. Retry to download it again.",
            }
        return result

    def publish(self, directory, **status):
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "status.tmp"
        tmp.write_text(canonical_json(dict(status, updated_at=utc_now())))
        tmp.replace(directory / "status.json")

    def create(self, identifier, index=0):
        with self.lock:
            current = self.status(identifier, index)
            key = (identifier, index)
            if current["state"] == "READY" or key in self.active:
                return current
            self.active.add(key)
            directory = self.directory(identifier, index)
            self.publish(directory, state="DOWNLOADING")
            self.executor.submit(self.prepare, identifier, index)
            return {"state": "DOWNLOADING"}

    def store(self, identifier, index, path):
        """Validate before atomically publishing the local review and original."""
        job, _ = self.source(identifier, index)
        result = inspect(path, job["profile"])
        expected = job.get("recording_checksums", {}).get(job["recordings"][index])
        if not expected and len(job["recordings"]) == 1:
            expected = (job.get("recording_summary") or {}).get("sha256")
        if expected and result["sha256"] != expected:
            raise ValueError(
                "The recording checksum differs from its saved validation report"
            )
        directory = self.directory(identifier, index)
        directory.mkdir(parents=True, exist_ok=True)
        text = canonical_json(result)
        if len(text.encode()) > 50 * 1024 * 1024:
            raise ValueError("Recording preview exceeds the 50 MB review limit")
        (directory / "review.tmp").write_text(text)
        (directory / "review.tmp").replace(directory / "review.json")
        if path != directory / "recording.pkl":
            (directory / "recording.tmp").write_bytes(path.read_bytes())
            (directory / "recording.tmp").replace(directory / "recording.pkl")
        summary = {k: v for k, v in result.items() if k != "episodes"}
        summary["episodes"] = [
            {k: v for k, v in ep.items() if k != "frames"} for ep in result["episodes"]
        ]
        (directory / "summary.json").write_text(canonical_json(summary))
        self.publish(directory, state="READY", summary=summary)
        return summary

    def prepare(self, identifier, index):
        directory = self.directory(identifier, index)
        temp = directory / "download.part"
        try:
            job, remote = self.source(identifier, index)
            transport = self.live.transport(job)
            host, size = transport.file_size(remote, job["gateway"])
            if not 0 < size <= MAX_BYTES:
                raise ValueError(
                    "Review supports native recordings from 1 byte to 100 MB"
                )
            received = 0
            with temp.open("wb") as stream:
                for block in transport.stream_file_range(
                    remote, host, start=0, end=size - 1
                ):
                    received += len(block)
                    if received > size:
                        raise ValueError("Recording size changed during download")
                    stream.write(block)
            if received != size:
                raise ValueError(
                    "Recording download was incomplete; retry when the host is reachable"
                )
            self.store(identifier, index, temp)
        except Exception as exc:
            self.publish(directory, state="FAILED", error=str(exc))
        finally:
            temp.unlink(missing_ok=True)
            with self.lock:
                self.active.discard((identifier, index))

    def artifact(self, identifier, index, name):
        if name not in {"review.json", "summary.json", "recording.pkl"}:
            raise KeyError("Review file not found")
        if self.status(identifier, index)["state"] != "READY":
            raise ValueError(
                "Download and validate the recording before opening its review"
            )
        path = self.directory(identifier, index) / name
        if not path.is_file():
            raise KeyError(
                "Local review file is missing; the original remains on the execution host"
            )
        return path
