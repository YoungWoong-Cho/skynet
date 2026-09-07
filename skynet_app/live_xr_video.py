"""Cache capture videos and render legacy scene-state recordings on their workstation."""

from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import hashlib
import inspect
import json
from pathlib import PurePosixPath
import re
import shlex
import threading

from .database import canonical_json, utc_now
from .live_xr_review import ArrayUnpickler

MAX_BYTES = 256 * 1024 * 1024


class LiveVideoService:
    def __init__(self, reviews):
        self.reviews = reviews
        self.live = reviews.live
        self.lock = threading.RLock()
        self.active = set()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="review-video"
        )

    @cached_property
    def sources(self):
        sources = {
            name: (self.live.root / "ops/xr" / name).read_text()
            for name in ("render_recording.py", "video.py", "wrist.py")
        }
        sources["arrays.py"] = (
            "import pickle\nimport numpy as np\n" + inspect.getsource(ArrayUnpickler)
        )
        return sources

    @cached_property
    def version(self):
        return hashlib.sha256(canonical_json(self.sources).encode()).hexdigest()[:16]

    def source(self, identifier, index, episode):
        path = self.reviews.artifact(identifier, index, "review.json")
        review = json.loads(path.read_text())
        if not 0 <= episode < len(review["episodes"]):
            raise KeyError("Demonstration not found")
        job, remote = self.reviews.source(identifier, index)
        directory = path.parent / f"video-{episode}"
        return job, remote, review, directory

    def status(self, identifier, index, episode=0):
        _, _, _, directory = self.source(identifier, index, episode)
        path = directory / "status.json"
        result = (
            json.loads(path.read_text()) if path.exists() else {"state": "NOT_PREPARED"}
        )
        if (
            result["state"] == "PREPARING"
            and (identifier, index, episode) not in self.active
        ):
            return dict(
                state="FAILED",
                error="Video preparation was interrupted. Retry to continue.",
            )
        if result["state"] == "READY" and not (directory / "video.mp4").is_file():
            return dict(
                state="FAILED",
                error="The cached video is missing. Retry to download it again.",
            )
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

    def create(self, identifier, index, episode=0):
        with self.lock:
            state = self.status(identifier, index, episode)
            key = (identifier, index, episode)
            if state["state"] == "READY" or key in self.active:
                return state
            _, _, _, directory = self.source(*key)
            self.active.add(key)
            self.publish(directory, state="PREPARING", detail="Preparing video…")
            self.executor.submit(self.prepare, *key)
            return dict(state="PREPARING", detail="Preparing video…")

    @staticmethod
    def capture_source(job, index, metadata):
        if metadata.get("state") != "READY":
            return None
        path = PurePosixPath(metadata.get("path", ""))
        expected = PurePosixPath(job["recordings"][index]).with_suffix(".mp4")
        if path != expected or path.is_absolute() or ".." in path.parts:
            raise ValueError("The saved video path does not match its recording")
        return job["root"] + "/output/" + str(path)

    def render(self, transport, job, remote, review, episode):
        profile = job["profile"]
        if profile.get("execution") != "workstation":
            raise ValueError(
                "Video replay for older cluster recordings is unsupported. Use a workstation collection."
            )
        sources, version = self.sources, self.version
        directory = (
            job["root"]
            + f"/output/review-videos/{review['sha256']}/{episode}/{version}"
        )
        output = directory + "/video.mp4"
        request = dict(
            recording=remote,
            profile=profile,
            output=output,
            sha256=review["sha256"],
            episode=episode,
        )
        setup = """import json,sys
from pathlib import Path
value=json.load(sys.stdin); root=Path(value['root']); root.mkdir(parents=True,exist_ok=True)
for name,content in value['sources'].items():
 p=root/name
 if p.exists() and p.read_text()!=content: raise ValueError('Video worker differs from its immutable source')
 if not p.exists(): p.write_text(content)
(root/'request.json').write_text(json.dumps(value['request']))
p=root/'video.json'
print(p.read_text() if p.is_file() else '{}')
"""
        current = json.loads(
            transport.ssh(
                job["gateway"],
                "python3 -c " + shlex.quote(setup),
                stdin=json.dumps(
                    dict(root=directory, sources=sources, request=request)
                ),
            )
        )
        if current.get("state") == "READY":
            return dict(current, renderer_version=version)
        q = shlex.quote
        runtime = profile["runtime"]
        command = (
            "cd "
            + q(profile["repository"])
            + " && "
            + "OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 "
            + "LD_LIBRARY_PATH="
            + q(runtime + "/lib")
            + " "
            + "PYTHONPATH="
            + q(profile["repository"] + "/source/dexverse")
            + " "
            + "PATH="
            + q(runtime + "/bin")
            + ':"$PATH" '
            + "timeout -k 5 600 flock --nonblock --conflict-exit-code=75 "
            + q(profile["work_root"] + "/.gpu-session.lock")
            + " "
            + q(runtime + "/bin/python")
            + " "
            + q(directory + "/render_recording.py")
            + " "
            + q(directory + "/request.json")
            + " >"
            + q(directory + "/render.log")
            + " 2>&1\n"
            + "result=$?\n"
            + 'if [ "$result" = 75 ]; then echo "The GPU is busy with another session or video job. Retry when it finishes." >&2; exit 75; fi\n'
            + 'if [ "$result" != 0 ]; then tail -c 1500 '
            + q(directory + "/render.log")
            + ' >&2; exit "$result"; fi\n'
            + "cat "
            + q(directory + "/video.json")
        )
        return dict(
            json.loads(transport.ssh(job["gateway"], command, timeout=615)),
            renderer_version=version,
        )

    def download(self, transport, host, remote, metadata, directory):
        if not re.fullmatch(r"[a-f0-9]{64}", str(metadata.get("sha256", ""))):
            raise ValueError("Video is missing its saved checksum")
        host, size = transport.file_size(remote, host)
        if not 0 < size <= MAX_BYTES or size != metadata.get("size_bytes"):
            raise ValueError("Video is empty, oversized, or changed since saving")
        temp = directory / "download.part"
        try:
            received = 0
            digest = hashlib.sha256()
            with temp.open("wb") as stream:
                for block in transport.stream_file_range(
                    remote, host, start=0, end=size - 1
                ):
                    received += len(block)
                    if received > size:
                        raise ValueError("Video size changed during download")
                    digest.update(block)
                    stream.write(block)
            if received != size or digest.hexdigest() != metadata["sha256"]:
                raise ValueError(
                    "Video download is incomplete or its checksum differs. Retry to download again."
                )
            with temp.open("rb") as stream:
                if stream.read(12)[4:8] != b"ftyp":
                    raise ValueError("The saved video is not a supported MP4 file")
            temp.replace(directory / "video.mp4")
        finally:
            temp.unlink(missing_ok=True)

    def prepare(self, identifier, index, episode):
        directory = None
        try:
            job, remote, review, directory = self.source(identifier, index, episode)
            transport = self.live.transport(job)
            metadata = review["episodes"][episode].get("video") or {}
            capture_error = (
                metadata.get("error") if metadata.get("state") == "FAILED" else None
            )
            capture = self.capture_source(job, index, metadata)
            if capture:
                metadata = dict(metadata, kind="capture")
            else:
                self.publish(
                    directory, state="PREPARING", detail="Rendering recorded scene…"
                )
                metadata = self.render(transport, job, remote, review, episode)
                if metadata.get("state") != "READY":
                    raise ValueError(
                        metadata.get("error")
                        or "Video rendering did not finish. Retry to continue."
                    )
                capture = metadata["path"]
            self.publish(directory, state="PREPARING", detail="Downloading video…")
            self.download(transport, job["gateway"], capture, metadata, directory)
            self.publish(
                directory,
                state="READY",
                kind=metadata["kind"],
                fps=metadata["fps"],
                frames=metadata["frames"],
                size_bytes=metadata["size_bytes"],
                sha256=metadata["sha256"],
                renderer_version=metadata.get("renderer_version"),
                capture_error=capture_error,
            )
        except Exception as exc:
            if directory is not None:
                self.publish(directory, state="FAILED", error=str(exc))
        finally:
            with self.lock:
                self.active.discard((identifier, index, episode))

    def artifact(self, identifier, index, episode=0):
        if self.status(identifier, index, episode)["state"] != "READY":
            raise ValueError("Video is not ready yet")
        return self.source(identifier, index, episode)[3] / "video.mp4"
