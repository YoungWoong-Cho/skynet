"""Prepare a manifest referencing shared immutable recording streams.

No adapter payload export, video, image resize or ZIP is produced here. Existing
archived capture arrays are referenced in place after source verification.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np

try:
    from arrays import ArrayUnpickler
except ImportError:
    from skynet_app.live_xr_review import ArrayUnpickler
try:
    from recording_dataset import (FORMAT, canonical, digest, stream_reference, reject_symlinks,
                                   verify_reference, verify_dataset)
except ImportError:
    from skynet_app.adapters.recording_dataset import (FORMAT, canonical, digest, stream_reference, reject_symlinks,
                                                      verify_reference, verify_dataset)

SLOTS = {"cam_head": "scene_front", "cam_left_wrist": "scene_left", "cam_right_wrist": "scene_right"}

def camera_contract(camera):
    """Compare physical calibration independently of capture/provenance wording."""
    transform = camera.get("world_from_camera")
    if transform is None and camera.get("quaternion_world_ros") is not None:
        q = np.asarray(camera["quaternion_world_ros"], dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
            raise ValueError("Invalid recorded camera orientation")
        w, x, y, z = q / np.linalg.norm(q)
        transform = np.eye(4)
        transform[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        transform[:3, 3] = camera["position_world"]
        transform = transform.tolist()
    result = {k: camera.get(k) for k in ("sensor", "width", "height", "intrinsic_matrix")}
    result["world_from_camera"] = transform
    if transform is None or result["intrinsic_matrix"] is None:
        result["recipe"] = {k: camera[k] for k in ("offset", "projection", "mount") if k in camera}
    return result

def capture_contract(metadata):
    fields = ("robot", "task", "hand", "source_revision", "action_joint_names", "robot_joint_names",
              "groups", "action_scale", "action_offset", "action_semantics", "step_dt", "color_space",
              "hand_adapter_digest", "hand_asset", "units", "wrist_rotation_order", "hand_order", "hands")
    result = {key: metadata.get(key) for key in fields}
    result["cameras"] = {name: camera_contract(camera) for name, camera in metadata.get("cameras", {}).items()}
    return result

def same_capture_contract(first, second):
    def equal(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return bool(np.isclose(a, b, rtol=1e-6, atol=1e-8))
        return a == b
    return equal(capture_contract(first), capture_contract(second))

def raw_episode(path):
    path = Path(path)
    if not 0 < path.stat().st_size <= 100_000_000:
        raise ValueError("Recording exceeds its size limit")
    payload = ArrayUnpickler(io.BytesIO(path.read_bytes())).load()
    if (
        payload.get("format") != "dexverse_trajectory"
        or payload.get("schema_version") not in {3, 5}
        or payload.get("num_episodes") != 1
        or len(payload.get("episodes", [])) != 1
    ):
        raise ValueError(
            "Visual exports require one complete native episode per recording"
        )
    try:
        from trajectory import validate_identity
    except ImportError:
        from skynet_app.trajectory import validate_identity
    validate_identity(payload)
    episode = payload["episodes"][0]
    if episode.get("success") is not True:
        raise ValueError("Only successful demonstrations can be exported")
    return payload, episode

def validate_sidecar(source, h5):
    if h5.attrs.get("schema") != "skynet.rgb-trajectory/v1" or not h5.attrs.get(
        "complete"
    ):
        raise ValueError("Incomplete or unsupported image recording")
    meta = json.loads(h5.attrs["metadata"])
    payload, episode = raw_episode(source["recording"])
    if payload["robot_type"] != meta["robot"] or payload["task"] != meta["task"]:
        raise ValueError("Images belong to a different robot or task")
    native = payload.get("skynet_state_metadata", {})
    for key in ("action_joint_names", "robot_joint_names", "groups", "action_scale", "action_offset", "action_semantics", "step_dt"):
        if key in native and native[key] != meta.get(key):
            raise ValueError("Archived capture joint semantics differ from the original recording: " + key)
    image_receipt = episode.get("skynet_images", {})
    requested = payload.get("skynet_training_images")
    if requested is not None:
        recipe_hash = hashlib.sha256(
            json.dumps(
                requested.get("recipe"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if (
            requested.get("schema") != "skynet.training-images/v2"
            or requested.get("mode") != "saved_states"
            or requested.get("recipe_sha256") != recipe_hash
            or meta.get("image_recipe_sha256") != recipe_hash
            or meta.get("render_mode") != "saved_states"
            or meta.get("source_sha256") != source["sha256"]
        ):
            raise ValueError(
                "Rendered images do not match the original recording and its frozen camera recipe"
            )
        if payload.get("skynet_step_dt") != meta["step_dt"] or not np.array_equal(
            h5["wall_times"][:], episode.get("skynet_wall_times")
        ):
            raise ValueError("Rendered image timing differs from collection")
    elif image_receipt.get("sha256") != source["image_sha256"]:
        raise ValueError("Image sidecar does not match its native recording")
    actions = np.asarray(episode["actions"])
    n, dimension = h5["action"].shape
    if (
        not 0 < n <= 6000
        or actions.shape != (n, dimension)
        or len(episode["states"]) != n + 1
        or episode["num_steps"] != n
    ):
        raise ValueError("Frames and native actions are not aligned")
    if not np.array_equal(h5["action"][:], actions):
        raise ValueError("Image actions differ from the original commands")
    names = meta["action_joint_names"]
    if len(names) != dimension or len(set(names)) != dimension:
        raise ValueError("Invalid action joint names")
    ids = [meta["robot_joint_names"].index(name) for name in names]
    states = np.stack(
        [
            np.asarray(s["articulation"]["robot"]["joint_position"])[0, ids]
            for s in episode["states"][:-1]
        ]
    )
    if h5["state"].shape != actions.shape or not np.array_equal(h5["state"][:], states):
        raise ValueError("Image states differ from the original pre-action states")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Nonfinite robot data")
    dt = meta["step_dt"]
    if (
        not isinstance(dt, (int, float))
        or not np.isfinite(dt)
        or dt <= 0
        or h5["timestamps"].shape != (n,)
        or not np.allclose(h5["timestamps"][:], np.arange(n) * dt, rtol=0, atol=1e-9)
    ):
        raise ValueError("Image timestamps do not match control steps")
    if meta["color_space"] != "RGB":
        raise ValueError("Unsupported image color space")
    for name in SLOTS.values():
        ds = h5["images/" + name]
        if ds.dtype != np.uint8 or ds.shape != (n, 256, 256, 3):
            raise ValueError("Missing or misaligned 256 × 256 RGB camera: " + name)
    groups = meta["groups"]
    if [g["side"] for g in groups] != (
        ["left", "right"] if meta["hand"] == "both" else [meta["hand"]]
    ):
        raise ValueError("Unsupported hand grouping")
    order = [i for g in groups for i in g["wrist_indices"] + g["finger_indices"]]
    if any(
        len(g["wrist_indices"]) != 6 or not g["finger_indices"] for g in groups
    ) or sorted(order) != list(range(dimension)):
        raise ValueError("Joint mapping would lose or duplicate commands")
    return meta, n, order


def inspect_images(source, requirements, *, verify_files=True):
    """Validate a raw archive sidecar and expose eligible arrays without copying.

    This imports capture streams, not the removed adapter export formats. A
    point-cloud request must use its calibrated depth/RGB dependency pair.
    """
    if not source.get("images"):
        return {"streams": {}}
    if verify_files and (digest(source["recording"]) != source["sha256"] or digest(source["images"]) != source["image_sha256"]):
        raise ValueError("Recording or archived capture checksum failed")
    with h5py.File(source["images"], "r") as file:
        capture, steps, order = validate_sidecar(source, file)
    streams = {}
    cloud_cameras = {camera for stream in requirements.get("streams", [])
                     if stream["modality"] == "point_cloud" for camera in stream["camera_ids"]}
    for stream in requirements.get("streams", []):
        if (stream["modality"] != "rgb" or stream["camera_ids"][0] in cloud_cameras
                or (stream["width"], stream["height"]) != (256, 256)):
            continue
        camera = stream["camera_ids"][0]
        if camera not in capture.get("cameras", {}):
            continue
        streams[stream["name"]] = stream_reference(source["images"], "images/" + camera,
            source["image_sha256"], source_sha256=source["sha256"], modality="rgb",
            camera_id=camera, calibration=capture["cameras"][camera], owned=False)
    return {"streams": streams, "capture": capture, "steps": steps, "policy_to_source_indices": order}


def file_artifact(path, source_sha256, sha256, *, owned, codec):
    spec = dict(schema="skynet.recording-file/v1", source_sha256=source_sha256,
                sha256=sha256, path=str(Path(path).absolute()), owned=owned, codec=codec)
    return dict(artifact_key=hashlib.sha256(canonical(spec)).hexdigest(), spec=spec,
                path=spec["path"], manifest_sha256=sha256, source_sha256=source_sha256, owned=owned)


def write_streams(recording_root, source_sha256, codec_spec, arrays, metadata):
    """Publish one source/codec stream file. Concurrent converters reuse it."""
    import fcntl
    if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise ValueError("Shared stream publication requires a source SHA-256")
    lengths = {np.asarray(value).shape[0] for value in arrays.values() if np.asarray(value).ndim}
    if len(lengths) != 1 or not arrays:
        raise ValueError("Shared codec arrays must have the same frame count")
    spec = dict(schema="skynet.recording-stream/v1", source_sha256=source_sha256, codec=codec_spec)
    key = hashlib.sha256(canonical(spec)).hexdigest()
    parent = reject_symlinks(Path(recording_root) / source_sha256 / "state")
    folder = parent / key
    parent.mkdir(parents=True, exist_ok=True)
    reject_symlinks(folder)
    with reject_symlinks(parent / ("." + key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = folder / "values.hdf5"
        receipt_path = folder / "manifest.json"
        if not folder.exists():
            import shutil
            staging = Path(tempfile.mkdtemp(prefix="." + key + "-", dir=parent))
            try:
                with h5py.File(staging / "values.hdf5", "x") as file:
                    file.attrs["schema"] = spec["schema"]
                    file.attrs["metadata"] = canonical(metadata).decode()
                    for name, values in arrays.items():
                        values = np.asarray(values)
                        if not values.shape or not len(values) or not np.isfinite(values).all():
                            raise ValueError("Shared state streams must contain finite aligned frames")
                        file.create_dataset(name, data=values, chunks=True, compression="lzf")
                    file.attrs["complete"] = True
                    file.flush()
                with (staging / "values.hdf5").open("rb") as saved:
                    os.fsync(saved.fileno())
                receipt = dict(spec=spec, artifact_key=key, source_sha256=source_sha256,
                    sha256=digest(staging / "values.hdf5"), metadata=metadata,
                    streams={k: dict(shape=list(np.asarray(v).shape), dtype=str(np.asarray(v).dtype)) for k, v in arrays.items()})
                with (staging / "manifest.json").open("wb") as saved:
                    saved.write(canonical(receipt))
                    saved.flush()
                    os.fsync(saved.fileno())
                staging.rename(folder)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("spec") != spec or digest(path) != receipt["sha256"]:
            raise ValueError("Existing shared recording stream failed verification")
        if set(receipt["streams"]) != set(arrays):
            raise ValueError("Shared recording codec output keys changed without a new version")
        artifact = file_artifact(path, source_sha256, receipt["sha256"], owned=True, codec=codec_spec)
        refs = {name: stream_reference(path, name, receipt["sha256"], source_sha256=source_sha256,
                                      artifact_key=artifact["artifact_key"], owned=True) for name in arrays}
        for name, values in arrays.items():
            if refs[name]["shape"] != list(np.asarray(values).shape) or refs[name]["dtype"] != str(np.asarray(values).dtype):
                raise ValueError("Shared recording codec output shape changed without a new version")
        with h5py.File(path, "r") as file:
            if not file.attrs.get("complete") or any(not np.array_equal(file[name][:], value) for name, value in arrays.items()):
                raise ValueError("Shared recording codec values changed without a new identity")
        return refs, artifact


def source_values(source, capture):
    if digest(source["recording"]) != source["sha256"]:
        raise ValueError("Original recording checksum failed")
    payload, episode = raw_episode(source["recording"])
    supplied_capture = capture or {}
    capture = dict(payload.get("skynet_state_metadata") or supplied_capture)
    for key in ("cameras", "color_space", "observation_camera_recipes"):
        if key in supplied_capture:
            capture[key] = supplied_capture[key]
    names = capture.get("action_joint_names", [])
    actions = np.asarray(episode["actions"])
    if (actions.ndim != 2 or actions.shape[1] != len(names) or not names
            or len(set(names)) != len(names) or len(episode["states"]) != len(actions) + 1
            or episode["num_steps"] != len(actions) or not 0 < len(actions) <= 6000
            or payload["robot_type"] != capture.get("robot") or payload["task"] != capture.get("task")):
        raise ValueError("Original recording has no complete aligned joint metadata")
    indices = [capture["robot_joint_names"].index(name) for name in names]
    states = np.stack([np.asarray(value["articulation"]["robot"]["joint_position"])[0, indices]
                       for value in episode["states"][:-1]])
    if states.shape != actions.shape or not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Original joint data are nonfinite or misaligned")
    try:
        from recording_metadata import joint_layout
    except ImportError:
        from ops.xr.recording_metadata import joint_layout
    groups = capture["groups"]
    if groups != joint_layout(names, capture["hand"], [names[i] for g in groups for i in g["wrist_indices"]]):
        raise ValueError("Recorded joint grouping is inconsistent")
    dt = capture["step_dt"]
    if not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Original control timing is invalid")
    capture.pop("scene_geometry", None)
    capture.pop("kinematics_urdf", None)
    return payload, episode, capture, {"state": states, "action": actions,
                                     "timestamps": np.arange(len(actions), dtype=np.float64) * dt}


def observation_references(source):
    """Flatten verified observation artifacts into generic physical references."""
    try:
        from observation_artifacts import ArtifactArray
    except ImportError:
        from skynet_app.adapters.observation_artifacts import ArtifactArray
    streams, capture = {}, None
    for name, ref in source.get("observation_streams", {}).items():
        if "dataset" in ref:
            verify_reference(ref)
            streams[name] = ref
            continue
        array = ArtifactArray(ref, source_sha256=source["sha256"])
        info = array.manifest
        capture = capture or info.get("capture")
        sha = info["files"]["values.hdf5"]["sha256"]
        streams[name] = stream_reference(array.path, "values", sha,
            source_sha256=source["sha256"], artifact_key=ref["artifact_key"],
            modality=info["modality"], camera_id=info["camera_id"], recipe=info["recipe"], owned=True)
        with h5py.File(array.path, "r") as file:
            for key in ("valid_mask", "frame_ids", "timestamps"):
                if key in file:
                    streams[name + "_" + key] = stream_reference(array.path, key, sha, owned=True)
    # Calibrated render dependencies are shared even when the selected adapter
    # asks only for a point cloud. They are necessary to encode camera-frame actions.
    for camera, modalities in source.get("observation_artifacts", {}).items():
        ref = modalities.get("rgb") or modalities.get("depth")
        if not ref:
            continue
        array = ArtifactArray(ref, source_sha256=source["sha256"], camera_id=camera)
        info = array.manifest
        capture = capture or info.get("capture")
        sha = info["files"]["values.hdf5"]["sha256"]
        for key in ("intrinsics", "world_from_camera"):
            streams[camera + "_" + key] = stream_reference(array.path, "calibration/" + key, sha,
                source_sha256=source["sha256"], artifact_key=ref["artifact_key"], owned=True)
        recipe = info["recipe"]
        capture = dict(capture or {})
        capture.setdefault("cameras", {})[camera] = dict(recipe["camera"], width=recipe["width"], height=recipe["height"])
        # Fixed-camera policies need exact calibration at inference. Dynamic
        # calibration remains per-frame references for policies supporting it.
        with h5py.File(array.path, "r") as file:
            intrinsics = file["calibration/intrinsics"][:]
            poses = file["calibration/world_from_camera"][:]
        if np.allclose(intrinsics, intrinsics[0], rtol=0, atol=1e-6) and np.allclose(poses, poses[0], rtol=0, atol=1e-6):
            capture["cameras"][camera].update(intrinsic_matrix=intrinsics[0].tolist(), world_from_camera=poses[0].tolist())
        else:
            capture["cameras"][camera]["dynamic"] = True
        capture["color_space"] = "RGB"
    return streams, capture


def preflight_source(source, requirements, action_representation=None, *, verify_files=True):
    """CPU-only source/codec validation before scheduling any observation GPU work."""
    imported = inspect_images(source, requirements, verify_files=verify_files)
    payload, raw, capture, arrays = source_values(source, imported.get("capture"))
    representation = action_representation if isinstance(action_representation, dict) else {}
    if representation.get("id") == "skynet.unidex-faas/v1":
        from action_codecs.unidex import validate_source
        validate_source(raw, capture)
    return dict(imported, capture=capture, steps=len(arrays["action"]),
                policy_to_source_indices=[i for group in capture["groups"]
                                          for i in group["wrist_indices"] + group["finger_indices"]])


def recording_prompt(source, capture):
    profile = source.get("session_profile") or {}
    for value in (source.get("prompt"), profile.get("instructions"), profile.get("task_name"), capture.get("task")):
        if value is not None and not isinstance(value, str):
            raise ValueError("Recording language instructions must be text")
        if value and value.strip():
            return value.strip()
    raise ValueError("UniDex requires recorded task instructions or a task identifier")


def prepare(request):
    requirements = request.get("observation_requirements", {"streams": []})
    representation = request.get("action_representation")
    unidex = isinstance(representation, dict) and representation.get("id") == "skynet.unidex-faas/v1"
    episodes, artifacts, common = [], [], None
    for index, source in enumerate(request["sources"]):
        imported = inspect_images(source, requirements)
        streams = dict(imported["streams"])
        observations, observation_capture = observation_references(source)
        streams.update(observations)
        streams.update(source.get("available_streams", {}))
        streams.update(source.get("shared_image_streams", {}))
        streams.update(source.get("streams", {}))
        payload, raw, capture, arrays = source_values(source, imported.get("capture") or observation_capture)
        # Existing sidecars already contain exact native state/action/timestamps.
        # Refer to those physical arrays rather than generating another copy.
        if source.get("images"):
            artifact = file_artifact(source["images"], source["sha256"], source["image_sha256"],
                                     owned=False, codec={"id": "archived-rgb-trajectory/v1"})
            artifacts.append(artifact)
            for name in arrays:
                streams[name] = stream_reference(source["images"], name, source["image_sha256"],
                    source_sha256=source["sha256"], artifact_key=artifact["artifact_key"], owned=False)
            for reference in streams.values():
                if reference["path"] == source["images"]:
                    reference["artifact_key"] = artifact["artifact_key"]
        else:
            # Camera selection and adapter choice are not part of joint identity.
            joint_capture = {k: v for k, v in capture_contract(capture).items()
                             if k not in {"cameras", "color_space"}}
            refs, artifact = write_streams(request["recording_root"], source["sha256"],
                {"id": "native-joints/v1", "capture": joint_capture}, arrays, joint_capture)
            streams.update(refs)
            artifacts.append(artifact)
        codec_spec = None
        if unidex:
            from action_codecs.unidex import encode_recording
            calibration = streams.get("scene_front_world_from_camera")
            if calibration is None:
                # Archived fixed RGB captures already pin the measured camera
                # pose. Reuse it instead of requiring another render for FAAS.
                camera = streams.get("scene_front", {}).get("calibration", {})
                if camera.get("mount") != "fixed_scene" or camera.get("dynamic"):
                    raise ValueError("UniDex requires per-frame recorded scene_front calibration")
                try:
                    from observation_geometry import pose_from_ros, rigid_transform
                except ImportError:
                    from ops.datasets.observation_geometry import pose_from_ros, rigid_transform
                pose = (rigid_transform(camera["world_from_camera"]) if "world_from_camera" in camera
                        else pose_from_ros(camera.get("position_world"), camera.get("quaternion_world_ros")))
                poses = np.repeat(pose[None], len(arrays["action"]), axis=0)
            else:
                with h5py.File(calibration["path"], "r") as file:
                    poses = file[calibration["dataset"]][:]
            encoded, codec_spec = encode_recording(raw, capture, poses)
            # The action representation depends on poses, not on unrelated RGB
            # payload bytes in the same file or a render's camera resolution.
            codec_spec = dict(codec_spec, camera_calibration_sha256=hashlib.sha256(canonical(poses.tolist())).hexdigest())
            refs, artifact = write_streams(request["recording_root"], source["sha256"], codec_spec, encoded, codec_spec)
            streams.update(refs)
            artifacts.append(artifact)
        for name, reference in streams.items():
            verify_reference(reference, steps=len(arrays["action"]), verify_files=False)
            if reference.get("source_sha256", source["sha256"]) != source["sha256"]:
                raise ValueError("A shared stream belongs to another recording")
        for requirement in requirements.get("streams", []):
            if requirement["name"] not in streams:
                raise ValueError("Required observation has not been prepared: " + requirement["name"])
        order = [i for group in capture["groups"] for i in group["wrist_indices"] + group["finger_indices"]]
        current = dict(index=index, id=source.get("recording_id") or source["sha256"],
            recording_id=source.get("recording_id"), session_id=source.get("session_id"),
            source_index=source.get("index", index), steps=len(arrays["action"]),
            source={"path": source["recording"], "sha256": source["sha256"]},
            streams=streams, capture=capture, policy_to_source_indices=order)
        if codec_spec is not None:
            current.update(hand_id=capture["robot"], prompt=recording_prompt(source, capture),
                           action_representation=codec_spec)
        else:
            for key in ("prompt", "hand_id", "action_representation"):
                if key in source:
                    current[key] = source[key]
        episodes.append(current)
        if common is not None:
            if not unidex and not same_capture_contract(common, capture):
                raise ValueError("This joint policy requires the same embodiment and camera contract across episodes")
        else:
            common = capture
        print(json.dumps({"episodes_done": index + 1, "episodes_total": len(request["sources"])}), flush=True)
    if not episodes:
        raise ValueError("Select at least one complete recording")
    manifest = dict(format=FORMAT, contract=request["contract"], adapter=request.get("adapter", {}),
        adapter_data_preset=request.get("adapter_data_preset"), episodes=episodes,
        steps=sum(e["steps"] for e in episodes), split=request["split"],
        capture=common, policy_to_source_indices=episodes[0]["policy_to_source_indices"],
        camera_slots={slot: name for slot, name in SLOTS.items() if name in episodes[0]["streams"]},
        preprocessing=request.get("preprocessing", {}), observations=request.get("observations", []),
        observation_requirements=requirements, source_revision=request.get("source_revision"),
        shared_artifacts=artifacts, validation={"status": "PASSED", "checks": ["source_checksums", "aligned_shared_streams"]})
    for key in ("action_representation", "hand_profile", "hand_profile_digest"):
        if key in request:
            manifest[key] = request[key]
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=False)
    path = output / "manifest.json"
    path.write_bytes(canonical(manifest))
    verify_dataset(output)
    return manifest


if __name__ == "__main__":
    prepare(json.loads(Path(sys.argv[1]).read_text()))
