"""CPU-only recording scenes, served beside immutable originals and hand assets."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import threading
from urllib.parse import quote

from .recording_guard import guarded_recording
from .adapters import episode_geometry
from . import episode_preview_worker
from .cluster_config import CLUSTER
from .cluster_runtime import WORK_ROOT
from .dexverse_versions import environment_profile
from .live_xr_review import ArrayUnpickler
from .remote_artifacts import RemoteArtifact


def verified_hand_file(request):
    """Read one manifest-listed file; this function also runs on the asset host."""
    import hashlib
    import json
    from pathlib import Path, PurePosixPath
    import xml.etree.ElementTree as ET
    from urllib.parse import quote
    root = Path(request["root"]).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size > 2_000_000:
        raise ValueError("Recorded hand manifest is unavailable")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("digest") != request["digest"] or manifest.get("robot") != request["robot"]:
        raise ValueError("Recorded hand manifest identity changed")
    name = request["name"]
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or (name != "simulation.urdf" and not name.startswith("assets/")):
        raise ValueError("Unknown recorded hand asset")
    expected = manifest.get("files", {}).get(name)
    path = (root / name).resolve()
    if not expected or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Recorded hand asset is unavailable")
    if not 0 < path.stat().st_size == expected["size_bytes"] <= 64_000_000:
        raise ValueError("Recorded hand asset size changed")
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected["sha256"]:
        raise ValueError("Recorded hand asset checksum changed")
    result = dict(path=str(path), size_bytes=expected["size_bytes"], sha256=expected["sha256"])
    if name == "simulation.urdf":
        if expected["size_bytes"] > 4_000_000:
            raise ValueError("Recorded hand description exceeds its size limit")
        xml = ET.fromstring(path.read_bytes())
        for item in [*xml.iter("mesh"), *xml.iter("texture")]:
            asset = item.get("filename", "")
            if asset not in manifest["files"] or not asset.startswith("assets/") or ".." in PurePosixPath(asset).parts:
                raise ValueError("Recorded hand description references an unlisted asset")
            item.set("filename", request["url_prefix"] + quote(asset, safe="/"))
        result["text"] = ET.tostring(xml, encoding="unicode")
    return result


class EpisodePreviews:
    def __init__(self, reviews):
        self.reviews = reviews
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="episode-preview")
        self.lock = threading.RLock()
        self.states = {}
        self.program = (Path(episode_geometry.__file__).read_text() + "\nimport pickle\n" +
                        inspect.getsource(ArrayUnpickler) + "\n" + Path(episode_preview_worker.__file__).read_text())
        self.version = hashlib.sha256(self.program.encode()).hexdigest()[:16]

    def hand_file(self, identifier, index, name):
        job, _ = self.reviews.source(identifier, index)
        profile = job["profile"]
        bundle = profile.get("hand_bundle") or {}
        digest = bundle.get("digest", "")
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("The recording has no verified hand visual bundle")
        request = dict(name=name, digest=digest, robot=profile["robot"],
                       url_prefix=f"/api/collection/live/sessions/{quote(identifier, safe='')}/recordings/{index}/hand/")
        from .simulation_hands import find_bundle
        try:
            directory = find_bundle(profile["robot"], digest, app_root=self.reviews.live.root)
        except ValueError:
            directory = None
        if directory is not None:
            return None, None, verified_hand_file(dict(request, root=str(directory)))
        # A verified archive need not have a Mac-side hand bundle. Inspect the same
        # immutable digest on storage, then the original collection host if needed.
        archive = self.reviews.live.archive
        transport, gateway, _ = archive.resolve(job, job["recordings"][index])
        candidates = [(transport, gateway, f"{WORK_ROOT}/hands/{profile['robot']}/{digest}")]
        if bundle.get("root") and job.get("gateway"):
            origin = self.reviews.live.transport(job)
            candidates.append((origin, origin.gateway_for(job["gateway"]), bundle["root"]))
        error = None
        seen = set()
        for transport, gateway, root in candidates:
            if (gateway, root) in seen:
                continue
            seen.add((gateway, root))
            code = inspect.getsource(verified_hand_file) + "\nimport json\nprint(json.dumps(verified_hand_file(" + repr(dict(request, root=root)) + ")))\n"
            try:
                result = json.loads(transport.run_with_fallback("python3 -", gateway, stdin=code, attempt_timeout=20)[1].strip().splitlines()[-1])
                return transport, gateway, result
            except (OSError, ValueError, RuntimeError) as exc:
                error = exc
        raise ValueError("The recording's frozen hand assets are unavailable") from error

    def location(self, identifier, index, episode):
        job, _ = self.reviews.source(identifier, index)
        if type(episode) is not int or episode < 0:
            raise ValueError("Invalid recording episode")
        relative = job["recordings"][index]
        archive = self.reviews.live.archive
        transport, gateway, recording = archive.resolve(job, relative)
        image = (job.get("recording_images") or {}).get(relative)
        images = None
        if image:
            path = PurePosixPath(image["path"])
            if path.is_absolute() or ".." in path.parts or not path.is_relative_to("recordings"):
                raise ValueError("Invalid legacy recording metadata location")
            _, image_gateway, image_path = archive.resolve(job, str(path))
            if image_gateway != gateway:
                raise ValueError("Recording and legacy metadata are not on the same storage host")
            images = dict(image, path=image_path)
        profile = job["profile"]
        runtime = next((p.environment_path for p in CLUSTER.runtime_profiles.values()
                        if "isaac_lab" in p.versions and p.environment_path), None)
        if not runtime:
            raise ValueError("No configured Python environment can read recorded arrays")
        if transport is archive.cluster:
            base = json.loads((self.reviews.live.root / "config/live_video.json").read_text())
            pinned = environment_profile(base, profile["task"], cluster_root=WORK_ROOT)
            if pinned["source_revision"] != profile["source_revision"]:
                raise ValueError("No pinned collection repository matches this recording's source revision")
            repository = pinned["repository"]
        else:
            runtime, repository = profile["runtime"], profile["repository"]
        source_sha = job.get("recording_checksums", {}).get(relative)
        if not isinstance(source_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", source_sha):
            raise ValueError("The original recording checksum is unavailable")
        output = str(PurePosixPath(self.reviews.remote_location(identifier, index, "review.json").path).parent /
                     f"scene-{self.version}-{source_sha}" / str(episode))
        request = dict(output=output, images=images, recording=recording, robot=profile["robot"], hand=profile.get("hand"),
                       source_sha256=source_sha, episode=episode, repository=repository, preview_version=self.version,
                       hand_bundle=(profile.get("hand_bundle") or {}).get("root"),
                       hand_digest=(profile.get("hand_bundle") or {}).get("digest"))
        return transport, gateway, runtime + "/bin/python", request

    @guarded_recording
    def status(self, identifier, index, episode=0, *, start=False):
        location = self.location(identifier, index, episode)
        key = identifier, index, episode
        identity = location[3]["source_sha256"]
        with self.lock:
            value = self.states.get(key)
            if value and value.get("source_sha256") == identity and (value["state"] != "FAILED" or not start):
                return value
            if not start:
                return {"state": "NOT_PREPARED"}
            self.states[key] = {"state": "PREPARING", "detail": "Loading recorded 3D scene…", "source_sha256": identity}
            self.executor.submit(self.prepare, key, location)
            return self.states[key]

    @guarded_recording
    def prepare(self, key, location):
        transport, gateway, python, request = location
        try:
            try:
                self.hand_file(key[0], key[1], "simulation.urdf")
                request["hand_visual_url"] = f"/api/collection/live/sessions/{quote(key[0], safe='')}/recordings/{key[1]}/hand/simulation.urdf"
            except (OSError, ValueError, RuntimeError) as exc:
                request["hand_visual_warning"] = str(exc)
            code = self.program + "\nprint(json.dumps(prepare_preview(" + repr(request) + ")))\n"
            _, reply = transport.run_with_fallback(shlex.quote(python) + " -", gateway, stdin=code, attempt_timeout=240)
            result = json.loads(reply.strip().splitlines()[-1])
            if result != {"state": "READY"}:
                raise ValueError("Recorded scene preparation did not complete")
        except Exception as error:
            result = {"state": "FAILED", "detail": str(error)}
        with self.lock:
            if self.states.get(key, {}).get("source_sha256") == request["source_sha256"]:
                self.states[key] = dict(result, source_sha256=request["source_sha256"])

    def artifact(self, identifier, index, episode, name):
        if name != "viewer.json":
            raise KeyError("Unknown recording scene artifact")
        transport, gateway, _, request = self.location(identifier, index, episode)
        return RemoteArtifact(transport, gateway, request["output"] + "/" + name, episode_geometry.MAX_VIEWER_BYTES)
