"""Prepare immutable camera/point-cloud artifacts only for an explicit Convert.

CLI: python observation_prepare.py --request request.json --result result.json
The CPU derive mode imports no Isaac or Torch modules. Render mode restores
recorded pre-action states in one simulator process for the requested episodes.
Database leases and consumer lifetimes belong to the caller. Request/attempt IDs
are echoed so that an expired worker cannot publish a dataset version in the DB.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import traceback

import h5py
import numpy as np

try:
    from observation_geometry import canonical_json, intrinsic_matrix, point_cloud, rigid_transform
except ImportError:
    from ops.datasets.observation_geometry import canonical_json, intrinsic_matrix, point_cloud, rigid_transform
try:
    from observation_contracts import ARTIFACT_SCHEMA, PREPARE_SCHEMA, CONTRACT_SCHEMA, content_digest, validate_requirements
except ImportError:
    from skynet_app.observation_contracts import ARTIFACT_SCHEMA, PREPARE_SCHEMA, CONTRACT_SCHEMA, content_digest, validate_requirements

try:
    from observation_render import render_identity
except ImportError:
    from ops.datasets.observation_render import render_identity

MAX_REQUEST_BYTES = 4_000_000
MAX_RECORDING_BYTES = 100_000_000
MAX_FRAMES = 6000
CALIBRATION = {"convention": "ros_optical", "pose": "world_from_camera", "units": "metres",
               "intrinsics": "calibration/intrinsics", "extrinsics": "calibration/world_from_camera",
               "depth_kind": "distance_to_image_plane"}


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a SHA-256 artifact identity")
    return value


def _json(path, limit=MAX_REQUEST_BYTES):
    path = Path(path)
    if not 0 < path.stat().st_size <= limit:
        raise ValueError("JSON document is empty or exceeds its size limit")
    return json.loads(path.read_text())


def _safe_child(root, name):
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts or "\\" in str(name):
        raise ValueError("Unsafe observation artifact path")
    target = root / candidate
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Observation artifact points outside its directory")
    return target


def verify_artifact(directory, expected_sha=None, *, request=None):
    """Verify all bytes before either reuse or derivation; never repair in place."""
    root = Path(directory)
    path = root / "manifest.json"
    if root.is_symlink() or path.is_symlink():
        raise ValueError("Observation artifact must not be a symlink")
    if expected_sha is not None and file_digest(path) != _sha(expected_sha):
        raise ValueError("Observation manifest checksum changed")
    manifest = _json(path)
    if (manifest.get("schema") != ARTIFACT_SCHEMA
            or manifest.get("validation", {}).get("status") != "PASSED"
            or not manifest.get("files") or set(manifest["files"]) != {"values.hdf5"}):
        raise ValueError("Observation artifact is incomplete or has an unsupported schema")
    if content_digest(manifest.get("spec")) != manifest.get("artifact_key") or any(manifest.get(k) != manifest.get("spec", {}).get(k) for k in ("source_sha256", "camera_id", "modality", "recipe")):
        raise ValueError("Observation artifact specification does not match its key")
    if request is not None and (manifest["artifact_key"] != request["artifact_key"]
                               or manifest["spec"] != request["spec"]):
        raise ValueError("Existing observation artifact has a different identity")
    for name, receipt in manifest["files"].items():
        entry = _safe_child(root, name)
        if (not entry.is_file() or entry.stat().st_size != receipt["size_bytes"]
                or file_digest(entry) != _sha(receipt["sha256"])):
            raise ValueError("Observation payload checksum changed: " + name)
    with h5py.File(root / "values.hdf5", "r") as data:
        frames = manifest["timing"]["frames"]
        if (list(data["values"].shape) != manifest["shape"]
                or str(data["values"].dtype) != manifest["dtype"]
                or data["values"].shape[0] != frames
                or data["frame_ids"].shape != (frames,) or data["timestamps"].shape != (frames,)
                or not bool(data.attrs.get("complete", False))):
            raise ValueError("Observation arrays differ from their manifest")
        if not np.array_equal(data["frame_ids"][:], np.arange(frames)):
            raise ValueError("Observation artifact does not preserve pre-action frame IDs")
        if not np.allclose(data["timestamps"][:], np.arange(frames) * manifest["timing"]["step_dt"], rtol=0, atol=1e-9):
            raise ValueError("Observation timestamps do not match recorded control timing")
    return manifest


def _receipt(directory, reused=False):
    manifest = verify_artifact(directory)
    return {"artifact_key": manifest["artifact_key"], "status": "READY", "path": str(Path(directory).resolve()),
            "manifest_sha256": file_digest(Path(directory) / "manifest.json"), "reused": reused,
            "frame_count": manifest["timing"]["frames"]}


class ArtifactWriter:
    """Bounded streaming writer; only a fully checked directory is published."""
    def __init__(self, request, source, *, frames, step_dt, staging_root=None):
        self.request, self.source = request, source
        self.destination = Path(request["output_dir"])
        if not self.destination.is_absolute() or self.destination.is_symlink():
            raise ValueError("Observation output must be an absolute nonsymlink directory")
        if type(frames) is not int or not 1 <= frames <= MAX_FRAMES or not np.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("Invalid recorded frame timing")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        stage_parent = Path(staging_root) if staging_root is not None else self.destination.parent
        if not stage_parent.is_absolute() or stage_parent.is_symlink():
            raise ValueError("Observation staging root must be an absolute nonsymlink directory")
        stage_parent.mkdir(parents=True, exist_ok=True)
        if stage_parent.stat().st_dev != self.destination.parent.stat().st_dev:
            raise ValueError("Observation staging and publication require the same filesystem")
        self.stage = Path(tempfile.mkdtemp(prefix="." + request["artifact_key"] + ".preparing-", dir=stage_parent))
        self.file = h5py.File(self.stage / "values.hdf5", "x")
        self.frames, self.step_dt, self.count = frames, float(step_dt), 0
        self.file.attrs["schema"] = ARTIFACT_SCHEMA
        self.metadata = {"artifact_key": request["artifact_key"], "source_sha256": source["sha256"],
                         "episode_key": source["episode_key"], "camera_id": request["camera_id"],
                         "modality": request["modality"], "recipe": request["recipe"],
                         "alignment": "pre_action_state", "step_dt": self.step_dt, "calibration": CALIBRATION}
        self.file.attrs["metadata"] = canonical_json(self.metadata)

    def append(self, values, *, frame_id, intrinsics=None, world_from_camera=None, valid_mask=None):
        if type(frame_id) is not int or frame_id != self.count or self.count >= self.frames:
            raise ValueError("Observation frames must be complete, ordered pre-action states")
        values = np.asarray(values)
        modality, recipe = self.request["modality"], self.request["recipe"]
        if modality == "rgb":
            expected = (recipe["height"], recipe["width"], 3)
            dtype = np.dtype("uint8")
        elif modality == "depth":
            expected = (recipe["height"], recipe["width"])
            dtype = np.dtype("float32")
        else:
            expected = (recipe["num_points"], 6 if recipe["channels"] == "XYZRGB" else 3)
            dtype = np.dtype("float32")
        if values.shape != expected or values.dtype != dtype or not np.isfinite(values).all():
            raise ValueError("Observation values do not satisfy the requested shape/dtype")
        arrays = {"values": values, "frame_ids": np.asarray(frame_id, dtype="int64"),
                  "timestamps": np.asarray(frame_id * self.step_dt, dtype="float64")}
        if modality in {"rgb", "depth"}:
            arrays["calibration/intrinsics"] = intrinsic_matrix(intrinsics).astype("float64")
            arrays["calibration/world_from_camera"] = rigid_transform(world_from_camera).astype("float64")
        if modality in {"depth", "point_cloud"}:
            valid_mask = np.asarray(valid_mask)
            if valid_mask.dtype != np.bool_ or valid_mask.shape != expected[:(2 if modality == "depth" else 1)]:
                raise ValueError("Observation validity mask differs from the requested values")
            if modality == "depth" and ((values[valid_mask] <= 0).any() or (values[~valid_mask] != 0).any()):
                raise ValueError("Depth must retain positive metric values and zero invalid pixels")
            arrays["valid_mask"] = valid_mask
        for name, value in arrays.items():
            if name not in self.file:
                self.file.create_dataset(name, shape=(self.frames, *value.shape), dtype=value.dtype,
                                         chunks=(1, *value.shape), compression="lzf")
            self.file[name][self.count] = value
        self.count += 1

    def finish(self):
        if self.count != self.frames:
            raise ValueError("Observation capture ended before all requested frames were written")
        self.file.attrs["complete"] = True
        shape, dtype = list(self.file["values"].shape), str(self.file["values"].dtype)
        self.file.close()
        payload = self.stage / "values.hdf5"
        with payload.open("rb") as stream:
            os.fsync(stream.fileno())
        manifest = {"schema": ARTIFACT_SCHEMA, "artifact_key": self.request["artifact_key"], "spec": self.request["spec"],
                    **{key: self.metadata[key] for key in ("source_sha256", "episode_key", "camera_id", "modality", "recipe")},
                    "shape": shape, "dtype": dtype, "timing": {"alignment": "pre_action_state", "frames": self.frames, "step_dt": self.step_dt},
                    "calibration": CALIBRATION if self.request["modality"] != "point_cloud" else {"coordinate_frame": self.request["recipe"]["coordinate_frame"]},
                    "capture": self.source.get("capture", {}),
                    "files": {"values.hdf5": {"sha256": file_digest(payload), "size_bytes": payload.stat().st_size}},
                    "validation": {"status": "PASSED", "checks": ["source_identity", "frame_alignment", "shape_dtype", "file_checksums"]}}
        with (self.stage / "manifest.json").open("x") as stream:
            stream.write(canonical_json(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        verify_artifact(self.stage, request=self.request)
        # The DB lease deduplicates computation; this short filesystem lock also
        # prevents concurrent/stale workers from replacing an immutable result.
        import fcntl
        with (self.destination.parent / ("." + self.request["artifact_key"] + ".publish.lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reused = self.destination.exists()
            if reused:
                verify_artifact(self.destination, request=self.request)
            else:
                os.rename(self.stage, self.destination)
                fd = os.open(self.destination.parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        self.discard()
        return _receipt(self.destination, reused)

    def discard(self):
        if self.file:
            self.file.close()
        if self.stage.exists():
            shutil.rmtree(self.stage)


def _load_source(source):
    """Use Skynet's restricted NumPy unpickler after checking the pinned bytes."""
    try:
        from arrays import ArrayUnpickler
        from trajectory import validate_identity
    except ImportError:
        from skynet_app.live_xr_review import ArrayUnpickler
        from skynet_app.trajectory import validate_identity
    path = Path(source["recording"])
    if not path.is_absolute() or path.is_symlink() or path.suffix != ".pkl" or not 0 < path.stat().st_size <= MAX_RECORDING_BYTES:
        raise ValueError("Invalid bounded original recording")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("Original recording checksum changed")
    payload = ArrayUnpickler(io.BytesIO(raw)).load()
    profile = source["profile"]
    validate_identity(payload, profile["task"], profile["robot"], profile.get("recording_schema_version"))
    episodes = payload.get("episodes", [])
    index = source.get("episode_index", 0)
    if type(index) is not int or not 0 <= index < len(episodes) or payload.get("num_episodes") != len(episodes):
        raise ValueError("Requested recording episode is missing")
    episode = episodes[index]
    actions = np.asarray(episode.get("actions"))
    if (actions.ndim != 2 or actions.dtype.kind not in "fiu" or not 0 < len(actions) <= MAX_FRAMES
            or not all(actions.shape) or not np.isfinite(actions).all()
            or episode.get("num_steps") != len(actions)
            or len(episode.get("states", [])) != len(actions) + 1):
        raise ValueError("Original recording lacks complete actions and saved states")
    metadata = payload.get("skynet_state_metadata", {})
    if not metadata and source.get("images"):
        legacy = Path(source["images"])
        if legacy.is_symlink() or file_digest(legacy) != _sha(source.get("image_sha256")):
            raise ValueError("Legacy image metadata checksum changed")
        with h5py.File(legacy, "r") as image_file:
            metadata = json.loads(image_file.attrs["metadata"])
            if metadata.get("source_sha256") != source["sha256"] or metadata.get("task") != profile["task"] or metadata.get("robot") != profile["robot"]:
                raise ValueError("Legacy image metadata belongs to a different recording")
    step_dt = payload.get("skynet_step_dt", metadata.get("step_dt"))
    if step_dt is not None and (not isinstance(step_dt, (int, float)) or not np.isfinite(step_dt) or step_dt <= 0):
        raise ValueError("Original recording has an invalid control timestep")
    recorded_revision = metadata.get("source_revision")
    if recorded_revision and recorded_revision != profile["source_revision"]:
        raise ValueError("Recording source revision differs from the requested renderer")
    bundle = profile.get("hand_bundle")
    if bundle and payload.get("skynet_hand", {}).get("digest") != bundle["digest"]:
        raise ValueError("Original recording hand differs from its frozen bundle")
    return payload, episode, float(step_dt) if step_dt is not None else None


def _validate_request(request):
    if not isinstance(request, dict) or request.get("schema") != PREPARE_SCHEMA or request.get("mode") not in {"render", "derive"}:
        raise ValueError("Unsupported observation preparation request")
    sources = request.get("sources", [])
    jobs = request.get("requests", [])
    if not 1 <= len(sources) <= 1000 or not 1 <= len(jobs) <= 16000:
        raise ValueError("Observation request has no bounded set of sources and artifacts")
    by_episode = {}
    for source in sources:
        _sha(source["sha256"])
        key = source["episode_key"]
        if not isinstance(key, str) or not key or key in by_episode:
            raise ValueError("Source episode keys must be distinct")
        by_episode[key] = source
    keys, outputs = set(), set()
    for job in jobs:
        key = _sha(job["artifact_key"])
        if key in keys or job["output_dir"] in outputs:
            raise ValueError("Observation request contains duplicate keys or output directories")
        keys.add(key)
        outputs.add(job["output_dir"])
        spec = job["spec"]
        if content_digest(spec) != key or spec.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError("Observation request specification does not match its content key")
        source = by_episode[job["episode_key"]]
        if (spec["source_sha256"] != source["sha256"] or spec["recipe"] != job["recipe"]
                or spec["modality"] != job["modality"] or spec["camera_id"] != job["camera_id"]
                or spec.get("timing") != {"alignment": "pre_action_state", "stride": 1}
                or spec.get("episode_index", 0) != source.get("episode_index", 0)):
            raise ValueError("Observation request differs from its immutable specification")
        deps = job.get("dependencies", [])
        if [item["artifact_key"] for item in deps] != spec.get("dependencies", []):
            raise ValueError("Observation dependencies differ from the content specification")
        expected = {"rgb", "depth"} if request["mode"] == "render" else {"point_cloud"}
        if job["modality"] not in expected:
            raise ValueError("Observation modality is incompatible with the worker mode")
        stream = dict(job["recipe"], name="prepared_observation", modality=job["modality"])
        stream.pop("camera", None)
        if job["modality"] != "point_cloud":
            stream["camera_ids"] = [job["camera_id"]]
        validate_requirements({"schema": CONTRACT_SCHEMA, "timing": spec["timing"], "streams": [stream]})
    return by_episode, jobs


def _derive(job, source, *, staging_root=None):
    deps = job.get("dependencies", [])
    if not deps:
        raise ValueError("Point-cloud preparation requires calibrated depth dependencies")
    views = {}
    timing, frame_ids, timestamps, render_identity, scene_receipt = None, None, None, None, None
    with ExitStack() as stack:
        for dependency in deps:
            manifest = verify_artifact(dependency["path"], dependency["manifest_sha256"])
            if manifest["artifact_key"] != dependency["artifact_key"] or manifest["source_sha256"] != source["sha256"]:
                raise ValueError("Point-cloud dependency belongs to a different original recording")
            if manifest["spec"].get("episode_index", 0) != source.get("episode_index", 0):
                raise ValueError("Point-cloud dependency belongs to a different episode")
            camera, modality = manifest["camera_id"], manifest["modality"]
            if camera not in job["recipe"]["camera_ids"] or modality not in {"rgb", "depth"}:
                raise ValueError("Unexpected point-cloud camera or modality dependency")
            data = stack.enter_context(h5py.File(Path(dependency["path"]) / "values.hdf5", "r"))
            if timing is None:
                timing, frame_ids, timestamps = manifest["timing"], data["frame_ids"][:], data["timestamps"][:]
                render_identity = manifest["spec"]["render_identity"]
                scene_receipt = {key: manifest.get("capture", {}).get(key) for key in ("rendered_scene_sha256", "rendered_assets_sha256")}
            if (manifest["timing"] != timing or not np.array_equal(data["frame_ids"][:], frame_ids)
                    or not np.array_equal(data["timestamps"][:], timestamps)
                    or manifest["spec"]["render_identity"] != render_identity
                    or render_identity != job["spec"]["render_identity"]
                    or {key: manifest.get("capture", {}).get(key) for key in ("rendered_scene_sha256", "rendered_assets_sha256")} != scene_receipt
                    or manifest.get("calibration") != CALIBRATION):
                raise ValueError("Point-cloud dependencies have incompatible capture provenance or timing")
            view = views.setdefault(camera, {})
            if modality in view:
                raise ValueError("Duplicate point-cloud dependency")
            view[modality] = data
        if set(views) != set(job["recipe"]["camera_ids"]):
            raise ValueError("Point-cloud request is missing a requested camera")
        for camera, view in views.items():
            required = {"depth", "rgb"} if job["recipe"]["channels"] == "XYZRGB" else {"depth"}
            if set(view) != required:
                raise ValueError("Point-cloud request is missing aligned depth or RGB")
            if "rgb" in view:
                if view["depth"]["values"].shape != view["rgb"]["values"].shape[:3]:
                    raise ValueError("RGB and depth resolutions are not aligned")
                for name in ("calibration/intrinsics", "calibration/world_from_camera"):
                    if not np.allclose(view["depth"][name][:], view["rgb"][name][:], rtol=0, atol=1e-7):
                        raise ValueError("RGB and depth camera calibration is not aligned")
        source = dict(source, capture=manifest.get("capture", {}))
        writer = ArtifactWriter(job, source, frames=timing["frames"], step_dt=timing["step_dt"], staging_root=staging_root)
        try:
            for frame in range(timing["frames"]):
                inputs = []
                for camera, view in views.items():
                    depth = view["depth"]
                    item = {"camera_id": camera, "depth": depth["values"][frame], "valid_mask": depth["valid_mask"][frame],
                            "intrinsics": depth["calibration/intrinsics"][frame],
                            "world_from_camera": depth["calibration/world_from_camera"][frame]}
                    if "rgb" in view:
                        item["rgb"] = view["rgb"]["values"][frame]
                    inputs.append(item)
                values, mask = point_cloud(inputs, job["recipe"], identity=job["artifact_key"], frame_id=int(frame_ids[frame]))
                writer.append(values, frame_id=frame, valid_mask=mask)
            return writer.finish()
        finally:
            writer.discard()


def run_request(request, *, renderer_factory=None, progress=None):
    started = time.monotonic()
    progress = progress or (lambda phase, **fields: None)
    result = {"schema": PREPARE_SCHEMA, "request_id": request.get("request_id"),
              "attempt_token": request.get("attempt_token"), "state": "FAILED", "artifacts": []}
    try:
        sources, jobs = _validate_request(request)
        pending = []
        for job in jobs:
            if Path(job["output_dir"]).exists():
                verify_artifact(job["output_dir"], request=job)
                result["artifacts"].append(_receipt(job["output_dir"], True))
            else:
                pending.append(job)
        if request["mode"] == "derive":
            for job in pending:
                result["artifacts"].append(_derive(job, sources[job["episode_key"]], staging_root=request.get("staging_root")))
        elif pending:
            if renderer_factory is None:
                try:
                    from observation_render import DexVerseRenderer
                except ImportError:
                    from ops.datasets.observation_render import DexVerseRenderer
                renderer_factory = DexVerseRenderer
            # Validate source bytes before starting the expensive simulator.
            episode_keys = list(dict.fromkeys(job["episode_key"] for job in pending))
            # Bound memory to one recording even for a large session.
            for key in episode_keys:
                source_path = Path(sources[key]["recording"])
                if not 0 < source_path.stat().st_size <= MAX_RECORDING_BYTES or file_digest(source_path) != sources[key]["sha256"]:
                    raise ValueError("Original recording checksum or size changed")
            progress("initializing")
            with renderer_factory(sources, pending) as renderer:
                for key in episode_keys:
                    progress("episode", episode_key=key)
                    payload, episode, step_dt = _load_source(sources[key])
                    selected = [job for job in pending if job["episode_key"] == key]
                    writers = []
                    try:
                        renderer.begin_episode(sources[key], payload, episode, step_dt)
                        capture = dict(renderer.capture_metadata)
                        capture["step_dt_source"] = "recorded" if step_dt is not None else "pinned_renderer"
                        if step_dt is None:
                            step_dt = capture["step_dt"]
                        source = dict(sources[key], capture=capture)
                        for job in selected:
                            writers.append(ArtifactWriter(job, source, frames=len(episode["actions"]), step_dt=step_dt, staging_root=request.get("staging_root")))
                        progress("capturing", episode_key=key)
                        for frame in range(len(episode["actions"])):
                            if frame % 60 == 0:
                                print(canonical_json({"event": "observation_progress", "episode_key": key, "frame": frame, "frames": len(episode["actions"])}), flush=True)
                            captured = renderer.capture(episode["states"][frame], selected)
                            for job, writer in zip(selected, writers):
                                sensor = captured[job["camera_id"]]
                                values = sensor[job["modality"]]
                                mask = None
                                if job["modality"] == "depth":
                                    values = np.asarray(values, dtype="float32")
                                    mask = np.isfinite(values) & (values > 0)
                                    values = np.where(mask, values, np.float32(0))
                                writer.append(values, frame_id=frame, intrinsics=sensor["intrinsics"],
                                              world_from_camera=sensor["world_from_camera"], valid_mask=mask)
                            progress("frame", episode_key=key, frame=frame + 1, frames=len(episode["actions"]))
                        for writer in writers:
                            progress("publishing", artifact_key=writer.request["artifact_key"])
                            result["artifacts"].append(writer.finish())
                    finally:
                        for writer in writers:
                            writer.discard()
                progress("closing")
        result["state"] = "READY"
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        result["error"] = str(exc)
    result["duration_seconds"] = round(time.monotonic() - started, 6)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--render-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--progress", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    request = _json(args.request)
    try:
        from observation_supervisor import ProgressReporter, supervise, write_json
    except ImportError:
        from ops.datasets.observation_supervisor import ProgressReporter, supervise, write_json
    if request.get("mode") == "render" and not args.render_child:
        result = supervise(request, args.request, args.result, worker_path=__file__)
    else:
        progress = ProgressReporter(request, args.progress) if args.render_child and args.progress else None
        result = run_request(request, progress=progress)
        write_json(args.result, result)
    return 0 if result["state"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
