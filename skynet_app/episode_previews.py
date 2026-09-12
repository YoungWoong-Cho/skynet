"""Cluster-backed viewer artifacts; local state contains only preparation status."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import shlex
import threading

from .adapters import episode_geometry
from . import episode_preview_worker
from .cluster_config import CLUSTER
from .live_xr_review import ArrayUnpickler
from .remote_artifacts import RemoteArtifact


class EpisodePreviews:
    def __init__(self, reviews):
        self.reviews = reviews
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="episode-preview")
        self.lock = threading.RLock()
        self.states = {}
        self.program = (Path(episode_geometry.__file__).read_text() + "\nimport pickle\n" +
                        inspect.getsource(ArrayUnpickler) + "\n" + Path(episode_preview_worker.__file__).read_text())
        self.version = hashlib.sha256(self.program.encode()).hexdigest()[:16]

    def location(self, identifier, index, episode):
        job, _ = self.reviews.source(identifier, index)
        if episode != 0:
            raise ValueError("Training camera sidecars contain one episode per recording")
        relative = job["recordings"][index]
        image = (job.get("recording_images") or {}).get(relative)
        if not image:
            raise ValueError("No training camera observations were saved for this recording")
        image_relative = PurePosixPath(image["path"])
        if image_relative.is_absolute() or ".." in image_relative.parts or not image_relative.is_relative_to("recordings"):
            raise ValueError("Invalid recording camera location")
        archive = self.reviews.live.archive
        transport, gateway, recording = archive.resolve(job, relative)
        image_transport, image_gateway, image_path = archive.resolve(job, str(image_relative))
        if image_gateway != gateway:
            raise ValueError("Recording and cameras are not on the same storage host")
        output = str(PurePosixPath(self.reviews.remote_location(identifier, index, "review.json").path).parent /
                     f"viewer-{self.version}" / str(episode))
        profile = job["profile"]
        runtime = next((p.environment_path for p in CLUSTER.runtime_profiles.values()
                        if "isaac_lab" in p.versions and p.environment_path), None)
        if not runtime:
            raise ValueError("No configured Python environment can read the training cameras")
        repository = str(PurePosixPath(CLUSTER.paths.repositories) / "skynet-dexverse" / profile["source_revision"])
        if gateway not in CLUSTER.gateways:
            runtime, repository = profile["runtime"], profile["repository"]
        source_sha = job.get("recording_checksums", {}).get(relative) or image.get("source_sha256")
        if not source_sha:
            raise ValueError("The original recording checksum is unavailable")
        request = dict(output=output, images={**image, "path": image_path}, recording=recording,
                       source_sha256=source_sha, episode=episode, repository=repository,
                       hand_bundle=(profile.get("hand_bundle") or {}).get("root"))
        return transport, gateway, runtime + "/bin/python", request

    def status(self, identifier, index, episode=0, *, start=False):
        key = identifier, index, episode
        with self.lock:
            value = self.states.get(key)
            if value and (value["state"] != "FAILED" or not start):
                return value
            try:
                location = self.location(*key)
            except ValueError as error:
                return {"state": "UNAVAILABLE", "detail": str(error)}
            if not start:
                return {"state": "NOT_PREPARED"}
            self.states[key] = {"state": "PREPARING", "detail": "Preparing saved camera views…"}
            self.executor.submit(self.prepare, key, location)
            return self.states[key]

    def prepare(self, key, location):
        transport, gateway, python, request = location
        code = self.program + "\nprint(json.dumps(prepare_preview(" + repr(request) + ")))\n"
        try:
            reply = transport.ssh(gateway, shlex.quote(python) + " -", stdin=code, timeout=240)
            result = json.loads(reply.strip().splitlines()[-1])
            if result != {"state": "READY"}:
                raise ValueError("Camera preview preparation did not complete")
        except Exception as error:
            result = {"state": "FAILED", "detail": str(error)}
        with self.lock:
            self.states[key] = result

    def artifact(self, identifier, index, episode, name):
        if name not in {"viewer.json", "views.mp4"}:
            raise KeyError("Unknown episode viewer artifact")
        transport, gateway, _, request = self.location(identifier, index, episode)
        return RemoteArtifact(transport, gateway, request["output"] + "/" + name,
                              episode_geometry.MAX_VIEWER_BYTES if name.endswith(".json") else None)
