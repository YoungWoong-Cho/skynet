"""Cache capture videos and render legacy scene-state recordings on their workstation."""

from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import threading
import time
import uuid

from .recording_guard import guarded_recording
from .database import canonical_json, utc_now
from .live_xr_review import ArrayUnpickler
from .live_xr_video_worker import control
from .live_xr_video_cluster import control_cluster
from .remote_artifacts import RemoteArtifact
from .cluster_runtime import WORK_ROOT
from .dexverse_versions import environment_profile
from .simulation_hands import upload as upload_hand

MAX_BYTES = 256 * 1024 * 1024
GPU_WAIT_SECONDS = 120
GPU_RETRY_SECONDS = 5
RENDER_TIMEOUT_SECONDS = 660  # Includes the remote 600-second limit and cleanup.


class VideoCancelled(Exception):
    """The caller cancelled this exact preparation generation."""


@dataclass
class VideoGeneration:
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    cancel: threading.Event = field(default_factory=threading.Event)
    finished: bool = False
    remote: tuple | None = None
    remote_stopped: bool = True
    stopping: bool = False
    cancel_error: str | None = None
    failure: str | None = None
    local_published: bool = False
    gpu_deadline: float | None = None
    gpu_waiting: bool = False
    busy_owner: dict | None = None
    render_deadline: float | None = None
    control_lock: object = field(default_factory=threading.RLock)
    cluster_job_id: str | None = None
    cluster_submission_started: bool = False
    persist_submission: object = None
    cluster_receipt_verified: bool = False


class LiveVideoService:
    def __init__(self, reviews):
        self.reviews = reviews
        self.live = reviews.live
        self.lock = threading.RLock()
        self.active = set()
        self.queue = []
        self.generations = {}
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="review-video"
        )

    def cluster_profile(self, session):
        """Resolve the archived recording's renderer without a conversion service."""
        profile = json.loads((self.live.root / "config/live_video.json").read_text())
        cluster = self.live.archive.cluster
        cluster.candidates(profile["gateway"])
        for name in ("repository", "runtime", "asset_bundle"):
            cluster._remote_path(profile[name])
        profile = dict(profile, execution="slurm", work_root=WORK_ROOT)
        original = session["profile"]
        profile = environment_profile(profile, original["task"], cluster_root=WORK_ROOT)
        if profile["source_revision"] != original["source_revision"]:
            raise ValueError(
                "The video renderer and recording use different DexVerse revisions"
            )
        for key in ("robot", "task", "hand", "hand_name", "task_name"):
            if key in original:
                profile[key] = original[key]
        if original.get("hand_bundle"):
            bundle = original["hand_bundle"]
            if not re.fullmatch(r"[a-f0-9]{64}", bundle["digest"]):
                raise ValueError("Recording has an invalid hand bundle identity")
            profile["hand_bundle"] = dict(
                bundle, root=f"{WORK_ROOT}/hands/{profile['robot']}/{bundle['digest']}"
            )
        return profile

    def ensure_hand(self, profile, transport, gateway):
        if profile.get("hand_bundle"):
            from .simulation_hands import find_bundle

            bundle = profile["hand_bundle"]
            local = find_bundle(profile["robot"], bundle["digest"], app_root=self.live.root)
            if upload_hand(local, WORK_ROOT, transport, gateway) != bundle["root"]:
                raise ValueError("Uploaded hand bundle path differs from the saved request")

    @cached_property
    def sources(self):
        sources = {
            name: (self.live.root / "ops/xr" / name).read_text()
            for name in ("render_recording.py", "video.py", "wrist.py")
        }
        sources["trajectory.py"] = Path(__file__).with_name("trajectory.py").read_text()
        sources["arrays.py"] = (
            "import pickle\nimport numpy as np\n" + inspect.getsource(ArrayUnpickler)
        )
        return sources

    @cached_property
    def version(self):
        return hashlib.sha256(canonical_json(self.sources).encode()).hexdigest()[:16]

    def source(self, identifier, index, episode):
        if getattr(self.live, "archive", None) is not None:
            status = self.reviews.status(identifier, index)
            if status["state"] != "READY":
                raise ValueError("Load and validate the recording before preparing video")
            review = status["summary"]
        else:
            path = self.reviews.artifact(identifier, index, "review.json")
            review = json.loads(path.read_text())
        if not 0 <= episode < len(review["episodes"]):
            raise KeyError("Demonstration not found")
        job, remote = self.reviews.source(identifier, index)
        directory = self.reviews.directory(identifier, index) / f"video-{episode}"
        return job, remote, review, directory

    def status(self, identifier, index, episode=0):
        with self.lock:
            return self._status(identifier, index, episode)

    def _status(self, identifier, index, episode=0):
        job, _, _, directory = self.source(identifier, index, episode)
        path = directory / "status.json"
        result = (
            json.loads(path.read_text()) if path.exists() else {"state": "NOT_PREPARED"}
        )
        if (result["state"] in {"QUEUED", "STARTING", "PREPARING", "WAITING_GPU", "CANCELLING"}
                and (job.get("archive") or {}).get("source_removed")
                and str(result.get("remote_root", "")).startswith(job["root"] + "/output/review-videos/")):
            # Archive cleanup proved these workstation generations had stopped.
            # Do not recreate the deleted session just to cancel an old receipt.
            return {"state": "NOT_PREPARED"}
        if (
            result["state"] in {"QUEUED", "STARTING", "PREPARING", "WAITING_GPU", "CANCELLING"}
            and (identifier, index, episode) not in self.active
        ):
            if result.get("generation"):
                return dict(result, state="INTERRUPTED", can_cancel=True,
                            detail="Preparation was interrupted. Cancel it to clean up before preparing again.")
            return {"state": "NOT_PREPARED"}
        # Old releases persisted temporary lock conflicts as permanent failures.
        # New failures carry a code and remain visible until explicitly retried.
        if (
            result["state"] == "FAILED"
            and not result.get("code")
            and any(
                message in result.get("error", "")
                for message in (
                    "End the live session before preparing video",
                    "The GPU is busy with another session or video job",
                    "Video preparation was interrupted",
                )
            )
        ):
            return {"state": "NOT_PREPARED"}
        if result["state"] == "QUEUED":
            with self.lock:
                key = (identifier, index, episode)
                if key in self.queue:
                    result = dict(
                        result,
                        detail=f"Waiting to prepare video… (queue position {self.queue.index(key) + 1})",
                    )
        if result["state"] == "READY" and not result.get("remote_artifact") and not (directory / "video.mp4").is_file():
            return {"state": "NOT_PREPARED"}
        if (
            result["state"] == "READY"
            and result.get("kind") == "replay"
            and result.get("renderer_version") != self.version
        ):
            return {"state": "NOT_PREPARED"}
        return result

    @staticmethod
    def publish(directory, **value):
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "status.tmp"
        temporary.write_text(canonical_json(dict(value, updated_at=utc_now())))
        temporary.replace(directory / "status.json")

    def _check(self, key, generation):
        if generation.cancel.is_set() or self.generations.get(key) is not generation:
            raise VideoCancelled()

    def _publish(self, key, generation, directory, **value):
        with self.lock:
            self._check(key, generation)
            self.publish(directory, generation=generation.token,
                         remote_root=generation.remote[2] if generation.remote else None,
                         cluster_submission_started=generation.cluster_submission_started,
                         cluster_job_id=generation.cluster_job_id,
                         can_cancel=value["state"] not in {"READY", "FAILED"}, **value)

    def _persist_submission(self, key, generation, directory):
        # Persist the ownership boundary before submit_script can accept a job.
        # Cancellation may already be displayed; keep that state intact.
        with self.lock:
            if self.generations.get(key) is not generation:
                raise ValueError("The video generation changed before submission")
            value = json.loads((directory / "status.json").read_text())
            self.publish(directory, **dict(value,
                cluster_submission_started=generation.cluster_submission_started,
                cluster_job_id=generation.cluster_job_id))

    @guarded_recording
    def create(self, identifier, index, episode=0):
        with self.lock:
            if (self.live.get(identifier).get("archive") or {}).get("state") == "COPYING":
                raise ValueError("Recordings are moving to sky2. Retry video after transfer completes.")
            state = self.status(identifier, index, episode)
            key = (identifier, index, episode)
            if state["state"] in {"READY", "INTERRUPTED"} or key in self.active:
                return state
            directory = self.source(*key)[3]
            self.active.add(key)
            self.queue.append(key)
            generation = self.generations[key] = VideoGeneration()
            self._publish(key, generation, directory, state="QUEUED", detail="Waiting to prepare video…")
            try:
                self.executor.submit(self.prepare, *key, generation)
            except Exception:
                self.active.discard(key)
                self.queue.remove(key)
                self.generations.pop(key, None)
                raise
            return self.status(*key)

    def _settle_cancel(self, key, generation, directory):
        """Called with the state lock; completion cannot overtake cancellation."""
        if self.generations.get(key) is not generation:
            return
        if generation.finished and generation.remote_stopped and not generation.stopping:
            (directory / f"download.{generation.token}.part").unlink(missing_ok=True)
            if generation.local_published:
                (directory / "video.mp4").unlink(missing_ok=True)
            self.publish(directory, state="CANCELLED" if not generation.failure else "FAILED",
                         generation=generation.token, can_cancel=False,
                         detail="Video preparation cancelled.",
                         error=generation.failure, code="PREPARATION_FAILED" if generation.failure else None)
            self.active.discard(key)
            self.generations.pop(key, None)
        else:
            self.publish(directory, state="CANCELLING", generation=generation.token,
                         remote_root=generation.remote[2] if generation.remote else None,
                         cluster_submission_started=generation.cluster_submission_started,
                         cluster_job_id=generation.cluster_job_id,
                         can_cancel=bool(generation.cancel_error) and not generation.stopping, error=generation.cancel_error,
                         detail=("Could not confirm cancellation: " + generation.cancel_error + ". Retry cancellation."
                                 if generation.cancel_error else "Cancelling video preparation…"))

    def _stop_remote(self, key, generation, directory):
        with self.lock:
            if self.generations.get(key) is not generation or generation.stopping:
                return
            if generation.remote_stopped:
                self._settle_cancel(key, generation, directory)
                return
            generation.stopping = True
            generation.cancel_error = None
            remote = generation.remote
        try:
            if remote:
                transport, job, root = remote
                if transport is None:
                    transport = self.live.transport(job)
                    with self.lock:
                        generation.remote = (transport, job, root)
                result = self._control(transport, job, generation, root, "cancel")
                if result.get("state") != "CANCELLED":
                    raise ValueError("The video service has not confirmed cancellation")
            with self.lock:
                generation.remote_stopped = True
        except Exception as exc:
            with self.lock:
                generation.remote_stopped = False
                generation.cancel_error = str(exc)
        finally:
            with self.lock:
                generation.stopping = False
                self._settle_cancel(key, generation, directory)

    def cancel(self, identifier, index, episode=0, expected_generation=None):
        key = (identifier, index, episode)
        job, _, review, directory = self.source(*key)
        with self.lock:
            current = self.status(*key)
            if expected_generation and expected_generation != current.get("generation"):
                return current
            if current["state"] in {"CANCELLED", "NOT_PREPARED", "READY", "FAILED"}:
                return current
            generation = self.generations.get(key)
            if generation is None:
                token = current.get("generation", "")
                if not re.fullmatch(r"[a-f0-9]{32}", token):
                    raise ValueError("This older preparation has no safely cancellable process identity")
                root = current.get("remote_root")
                recovered_remote = None
                if root:
                    prefix = job["root"] + f"/output/review-videos/{review['sha256']}/{episode}/"
                    suffix = f"/attempts/{token}"
                    cluster_root = f"{WORK_ROOT}/jobs/runs/{token}"
                    if root == cluster_root and getattr(self.live, "archive", None) is not None:
                        profile = self.cluster_profile(job)
                        recovered_remote = (self.live.archive.cluster, dict(job, profile=profile, gateway="sky2"), root)
                    elif not re.fullmatch(re.escape(prefix) + r"[a-f0-9]{16}" + re.escape(suffix), root):
                        raise ValueError("The saved video process identity does not match this recording")
                    else:
                        recovered_remote = (None, job, root)
                generation = self.generations[key] = VideoGeneration(
                    token=token, finished=True, remote=recovered_remote,
                    remote_stopped=recovered_remote is None,
                    cluster_submission_started=current.get("cluster_submission_started", bool(root and root == f"{WORK_ROOT}/jobs/runs/{token}")),
                    cluster_job_id=current.get("cluster_job_id"),
                )
                self.active.add(key)
            generation.cancel.set()
            if key in self.queue:
                self.queue.remove(key)
                generation.finished = True
            self._settle_cancel(key, generation, directory)
            if self.generations.get(key) is not generation:
                return self.status(*key)
        # Transport validation and SSH share the retryable cancellation path.
        self._stop_remote(key, generation, directory)
        return self.status(*key)

    @staticmethod
    def capture_source(job, index, metadata):
        if metadata.get("state") != "READY":
            return None
        path = PurePosixPath(metadata.get("path", ""))
        expected = PurePosixPath(job["recordings"][index]).with_suffix(".mp4")
        if path != expected or path.is_absolute() or ".." in path.parts:
            raise ValueError("The saved video path does not match its recording")
        return job["root"] + "/output/" + str(path)

    def _control(self, transport, job, generation, root, operation, **value):
        if job["profile"].get("execution") == "slurm":
            return control_cluster(transport, job, generation, root, operation, **value)
        program = inspect.getsource(control) + "\nimport json,sys\nprint(json.dumps(control(json.load(sys.stdin))))"
        return json.loads(transport.ssh(
            job["gateway"], "python3 -c " + shlex.quote(program),
            stdin=json.dumps(dict(value, operation=operation, generation=generation.token, root=root)),
            timeout=55 if operation == "cancel" else 40,
        ))

    def render(self, transport, job, remote, review, episode, generation=None, key=None):
        profile = job["profile"]
        archived = (job.get("archive") or {}).get("state") in {"VERIFIED", "CLEANUP_PENDING", "READY"}
        if archived:
            profile = self.cluster_profile(job)
            job = dict(job, profile=profile, gateway=job["archive"]["gateway"])
            transport = self.live.archive.cluster
            self.ensure_hand(profile, transport, job["gateway"])
        elif profile.get("execution") != "workstation":
            raise ValueError(
                "Video replay for older cluster recordings is unsupported. Use a workstation collection."
            )
        generation = generation or VideoGeneration()
        if generation.gpu_deadline is None:
            generation.gpu_deadline = time.monotonic() + GPU_WAIT_SECONDS
        root = (job["root"] + f"/output/review-videos/{review['sha256']}/{episode}/{self.version}"
                + f"/attempts/{generation.token}")
        if archived:
            root = f"{WORK_ROOT}/jobs/runs/{generation.token}"
        with self.lock:
            if key is not None:
                self._check(key, generation)
            generation.remote = (transport, job, root)
            generation.remote_stopped = False
            if archived and key is not None:
                directory = self.source(*key)[3]
                generation.persist_submission = lambda: self._persist_submission(key, generation, directory)
            if key is not None and not generation.gpu_waiting:
                self._publish(key, generation, self.source(*key)[3], state="STARTING", detail="Starting video renderer…")
        result = self._control(transport, job, generation, root, "start", sources=self.sources,
                               profile=profile, request=dict(recording=remote, profile=profile,
                                                            sha256=review["sha256"], episode=episode))
        while result.get("state") in {"STARTING", "PREPARING"}:
            now = time.monotonic()
            if generation.render_deadline is not None and now >= generation.render_deadline:
                raise ValueError("Video rendering exceeded its time limit. Retry video preparation.")
            if result.get("phase") == "starting" and result["state"] == "STARTING":
                if generation.render_deadline is None:
                    generation.render_deadline = now + RENDER_TIMEOUT_SECONDS
                if key is not None:
                    self._publish(key, generation, self.source(*key)[3], state="STARTING",
                                  detail="GPU allocated; starting video renderer…", job_id=result.get("job_id"))
            elif result["state"] == "PREPARING":
                if generation.render_deadline is None:
                    generation.render_deadline = now + RENDER_TIMEOUT_SECONDS
                if key is not None:
                    detail = (
                        "Finishing recorded video…" if result.get("phase") == "finalizing"
                        else "Rendering recorded scene…"
                    )
                    self._publish(key, generation, self.source(*key)[3], state="PREPARING", detail=detail)
            elif generation.render_deadline is None and now >= generation.gpu_deadline:
                raise ValueError(
                    self._gpu_timeout(generation) if generation.gpu_waiting
                    else "The video renderer did not start within 2 minutes. Retry video preparation."
                )
            elif result.get("phase") == "queued" and key is not None:
                self._publish(key, generation, self.source(*key)[3], state="WAITING_GPU",
                              detail="Waiting for a cluster GPU allocation…", job_id=result.get("job_id"))
            # A launch is only an attempt to obtain the lock. Retain WAITING_GPU
            # across retries until the remote renderer confirms it has started.
            if generation.cancel.wait(0.5):
                raise VideoCancelled()
            result = self._control(transport, job, generation, root, "status")
        if result.get("state") not in {"READY", "WAITING_GPU", "FAILED", "CANCELLED"}:
            raise ValueError("The video service returned an unknown state")
        with self.lock:
            if key is not None:
                self._check(key, generation)
            generation.remote_stopped = True
        if result.get("state") == "CANCELLED":
            raise VideoCancelled()
        return dict(result, renderer_version=self.version)

    @staticmethod
    def _gpu_owner_detail(generation):
        owner = generation.busy_owner or {}
        unit = str(owner.get("unit") or "")
        collection = re.fullmatch(r"skynet-live-([a-f0-9-]{36})\.service", unit)
        if collection:
            return f"Collection session {collection[1][:8]} is using the GPU."
        if owner.get("kind") == "video":
            return "Another video preparation is using the GPU."
        return "Another collection session or video preparation is reserving the GPU."

    @classmethod
    def _gpu_timeout(cls, generation):
        return (
            "The GPU is still busy. " + cls._gpu_owner_detail(generation)
            + " End collection or wait for the other video job, then retry."
        )

    def download(self, transport, host, remote, metadata, directory, generation=None, key=None):
        if not re.fullmatch(r"[a-f0-9]{64}", str(metadata.get("sha256", ""))):
            raise ValueError("Video is missing its saved checksum")
        host, size = transport.file_size(remote, host)
        if not 0 < size <= MAX_BYTES or size != metadata.get("size_bytes"):
            raise ValueError("Video is empty, oversized, or changed since saving")
        if generation and generation.cancel.is_set():
            raise VideoCancelled()
        temp = directory / (f"download.{generation.token}.part" if generation else "download.part")
        try:
            received = 0
            digest = hashlib.sha256()
            with temp.open("wb") as stream:
                for block in transport.stream_file_range(
                    remote, host, start=0, end=size - 1,
                    **({"cancel_event": generation.cancel} if generation else {}),
                ):
                    if generation and generation.cancel.is_set():
                        raise VideoCancelled()
                    received += len(block)
                    if received > size:
                        raise ValueError("Video size changed during download")
                    digest.update(block)
                    stream.write(block)
            if generation and generation.cancel.is_set():
                raise VideoCancelled()
            if received != size or digest.hexdigest() != metadata["sha256"]:
                raise ValueError(
                    "Video download is incomplete or its checksum differs. Retry to download again."
                )
            with temp.open("rb") as stream:
                if stream.read(12)[4:8] != b"ftyp":
                    raise ValueError("The saved video is not a supported MP4 file")
            with self.lock:
                if generation and key is not None:
                    self._check(key, generation)
                temp.replace(directory / "video.mp4")
                if generation:
                    generation.local_published = True
        finally:
            temp.unlink(missing_ok=True)

    @guarded_recording
    def prepare(self, identifier, index, episode, generation=None):
        directory = None
        key = (identifier, index, episode)
        with self.lock:
            if generation is not None and self.generations.get(key) is not generation:
                return
            generation = generation or self.generations.get(key) or VideoGeneration()
            self.generations[key] = generation
            self.active.add(key)
            if key in self.queue:
                self.queue.remove(key)
        try:
            job, remote, review, directory = self.source(*key)
            self._check(key, generation)
            transport = self.live.transport(job)
            metadata = review["episodes"][episode].get("video") or {}
            capture_error = metadata.get("error") if metadata.get("state") == "FAILED" else None
            capture = self.capture_source(job, index, metadata)
            if capture:
                metadata = dict(metadata, kind="capture")
                if getattr(self.live, "archive", None) is not None:
                    transport, host, capture = self.live.archive.resolve(job, metadata["path"])
            else:
                self._publish(key, generation, directory, state="STARTING", detail="Starting video renderer…")
                generation.gpu_deadline = time.monotonic() + GPU_WAIT_SECONDS
                while True:
                    if generation.gpu_waiting and time.monotonic() >= generation.gpu_deadline:
                        raise ValueError(self._gpu_timeout(generation))
                    metadata = self.render(transport, job, remote, review, episode, generation, key)
                    self._check(key, generation)
                    if metadata.get("state") != "WAITING_GPU":
                        break
                    generation.gpu_waiting = True
                    generation.busy_owner = metadata.get("busy_owner")
                    remaining = generation.gpu_deadline - time.monotonic()
                    if remaining <= 0:
                        raise ValueError(self._gpu_timeout(generation))
                    self._publish(key, generation, directory, state="WAITING_GPU",
                                  busy_owner=generation.busy_owner,
                                  retry_seconds_remaining=math.ceil(remaining),
                                  detail=("Waiting for the GPU… " + self._gpu_owner_detail(generation)
                                          + f" Retrying for up to {math.ceil(remaining)} seconds."))
                    if generation.cancel.wait(min(GPU_RETRY_SECONDS, remaining)):
                        raise VideoCancelled()
                if metadata.get("state") != "READY":
                    raise ValueError(metadata.get("error") or "Video rendering did not finish. Retry to continue.")
                capture = metadata["path"]
                if generation.remote:
                    transport = generation.remote[0]
            if getattr(self.live, "archive", None) is not None:
                host = (generation.remote[1]["gateway"] if generation.remote else
                        self.live.archive.resolve(job, metadata["path"])[1])
                self._publish(key, generation, directory, state="PREPARING", detail="Verifying video…")
                self.verify_remote_video(transport, host, capture, metadata)
                self._publish(key, generation, directory, state="READY", kind=metadata["kind"],
                              fps=metadata["fps"], frames=metadata["frames"], size_bytes=metadata["size_bytes"],
                              sha256=metadata["sha256"], renderer_version=metadata.get("renderer_version"),
                              remote_artifact={"gateway": host, "path": capture}, capture_error=capture_error)
                return
            self._publish(key, generation, directory, state="PREPARING", detail="Downloading video…")
            self.download(transport, job["gateway"], capture, metadata, directory, generation, key)
            self._publish(key, generation, directory, state="READY", kind=metadata["kind"],
                          fps=metadata["fps"], frames=metadata["frames"], size_bytes=metadata["size_bytes"],
                          sha256=metadata["sha256"], renderer_version=metadata.get("renderer_version"),
                          capture_error=capture_error)
        except VideoCancelled:
            generation.cancel.set()
        except Exception as exc:
            if not generation.cancel.is_set():
                generation.failure = str(exc)
            if not generation.remote_stopped:
                generation.cancel.set()
            elif directory is not None and not generation.cancel.is_set():
                self._publish(key, generation, directory, state="FAILED", code="PREPARATION_FAILED", error=str(exc))
        finally:
            with self.lock:
                generation.finished = True
                if self.generations.get(key) is generation and not generation.cancel.is_set():
                    self.active.discard(key)
                    self.generations.pop(key, None)
            if generation.cancel.is_set() and directory is not None:
                with self.lock:
                    failed_stop = generation.cancel_error is not None
                    if failed_stop:
                        self._settle_cancel(key, generation, directory)
                if not failed_stop:
                    self._stop_remote(key, generation, directory)

    def artifact(self, identifier, index, episode=0):
        result = self.status(identifier, index, episode)
        if result["state"] != "READY":
            raise ValueError("Video is not ready yet")
        if result.get("remote_artifact"):
            job = self.live.get(identifier)
            remote = result["remote_artifact"]
            path, gateway = remote["path"], remote["gateway"]
            original_prefix = job["root"] + "/output/"
            archive = getattr(self.live, "archive", None)
            archived_prefix = (job.get("archive") or {}).get("root", "")
            if path.startswith(original_prefix) and getattr(self.live, "archive", None) is not None:
                transport, gateway, path = self.live.archive.resolve(job, path[len(original_prefix):])
            elif archive is not None and archived_prefix and path.startswith(archived_prefix + "/"):
                transport, gateway, path = archive.resolve(job, path[len(archived_prefix) + 1:])
            elif re.fullmatch(r"[a-f0-9]{32}", result.get("generation", "")) and path.startswith(WORK_ROOT + "/jobs/runs/" + result["generation"] + "/"):
                transport = self.live.archive.cluster
            elif getattr(self.live, "archive", None) is not None and path.startswith(self.live.archive.derived_root(job) + "/"):
                transport = self.live.archive.cluster
            else:
                raise ValueError("The video location does not belong to this recording")
            return RemoteArtifact(transport, gateway, path, MAX_BYTES)
        return self.source(identifier, index, episode)[3] / "video.mp4"

    @staticmethod
    def verify_remote_video(transport, gateway, path, metadata):
        program = "import hashlib,json,sys; from pathlib import Path; p=Path(sys.argv[1]); s=p.stat().st_size; assert 0<s<=268435456; f=p.open('rb'); header=f.read(12); f.seek(0); h=hashlib.sha256(); [h.update(b) for b in iter(lambda:f.read(1048576),b'')]; print(json.dumps({'sha256':h.hexdigest(),'size_bytes':s,'mp4':header[4:8]==b'ftyp'}))"
        result = json.loads(transport.ssh(gateway, "python3 -c " + shlex.quote(program) + " " + shlex.quote(path), timeout=40))
        if not result.get("mp4") or any(result.get(key) != metadata.get(key) for key in ("sha256", "size_bytes")):
            raise ValueError("The remote video differs from its saved checksum or size")
