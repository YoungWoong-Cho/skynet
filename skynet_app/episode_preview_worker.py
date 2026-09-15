"""Bounded CPU scene previews from immutable recordings, without image generation."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def _legacy_metadata(image, source_sha, length):
    """Read verified legacy metadata only; never decode camera arrays."""
    import h5py
    path = Path(image["path"])
    if not 0 < path.stat().st_size == image["size_bytes"] <= 4_000_000_000:
        raise ValueError("Legacy camera file differs from its recording receipt")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != image["sha256"]:
            raise ValueError("Legacy camera checksum differs from its receipt")
    with h5py.File(path, "r") as source:
        metadata = json.loads(source.attrs["metadata"])
        if (source.attrs.get("schema") != "skynet.rgb-trajectory/v1" or not source.attrs.get("complete")
                or len(source["state"]) != length
                or metadata.get("source_sha256", source_sha) != source_sha):
            raise ValueError("Legacy metadata does not align with the original recording")
        if "scene_geometry" in source:
            saved = source["scene_geometry"]
            if saved.size > 12_000_000:
                raise ValueError("Saved scene geometry exceeds the preview limit")
            metadata["scene_geometry"] = json.loads(np.asarray(saved, dtype="u1").tobytes())
        elif metadata.get("scene_objects"):
            metadata["scene_geometry"] = dict(schema="skynet.scene-geometry/v1", objects=metadata["scene_objects"])
        return metadata


def prepare_preview(request):
    import fcntl
    root = Path(request["output"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        recording = Path(request["recording"])
        if not 0 < recording.stat().st_size <= 100_000_000:
            raise ValueError("Original recording exceeds the preview limit")
        raw = recording.read_bytes()
        if hashlib.sha256(raw).hexdigest() != request["source_sha256"]:
            raise ValueError("Original recording checksum changed")
        receipt = root / "viewer.json"
        if receipt.is_file() and receipt.stat().st_size <= MAX_VIEWER_BYTES:
            saved = json.loads(receipt.read_text())
            if saved.get("source_sha256") == request["source_sha256"] and saved.get("preview_version") == request.get("preview_version"):
                return {"state": "READY"}
        payload = ArrayUnpickler(io.BytesIO(raw)).load()
        if payload.get("format") != "dexverse_trajectory" or payload.get("schema_version") not in {3, 4, 5}:
            raise ValueError("Unsupported original recording schema")
        recorded_hand = payload.get("skynet_hand") or {}
        if request.get("hand_digest") and recorded_hand.get("digest") and recorded_hand["digest"] != request["hand_digest"]:
            raise ValueError("Recorded hand identity differs from its frozen bundle")
        episodes = payload.get("episodes", [])
        selected = request["episode"]
        if type(selected) is not int or not 0 <= selected < len(episodes):
            raise ValueError("Recording episode does not exist")
        episode = episodes[selected]
        states, actions = episode.get("states"), np.asarray(episode.get("actions"))
        if (actions.ndim != 2 or not all(actions.shape) or actions.dtype.kind not in "fiu"
                or not np.isfinite(actions).all() or len(actions) > 6000
                or episode.get("success") is not True or episode.get("num_steps") != len(actions)
                or not isinstance(states, list) or len(states) != len(actions) + 1):
            raise ValueError("Recording requires complete finite actions and T+1 saved states")
        metadata = payload.get("skynet_state_metadata") or {}
        warnings = []
        if not metadata and request.get("images"):
            metadata = _legacy_metadata(request["images"], request["source_sha256"], len(actions))
        metadata = dict(metadata)
        robot = metadata.get("robot") or payload.get("robot_type")
        if request.get("robot") and robot != request["robot"]:
            raise ValueError("Recording hand differs from the saved session")
        dt = float(metadata.get("step_dt", payload.get("skynet_step_dt", 1 / 60)))
        if not np.isfinite(dt) or not 0 < dt <= 1:
            raise ValueError("Invalid recorded control interval")
        scene = metadata.get("scene_geometry") or {}
        if scene and scene.get("schema") != "skynet.scene-geometry/v1":
            raise ValueError("Unsupported scene geometry format")
        # Retargeting debug markers are represented by this viewer's keypoint
        # layer; they are not physical scene objects that need mesh playback.
        debug_markers = {
            "Unsupported scene geometry: /Visuals/simple_relative_hand_keypoints",
            "Unsupported scene geometry: /Visuals/simple_relative_canonical_hand_keypoints",
            "Unsupported scene geometry: /Visuals/simple_relative_wrist_frames",
        }
        warnings.extend(warning for warning in scene.get("warnings", []) if warning not in debug_markers)
        if not scene.get("objects"):
            warnings.append("Scene shapes were not saved for this recording.")
        names, robot_names = metadata.get("action_joint_names", []), metadata.get("robot_joint_names", [])
        kinematics = None
        xml = None
        try:
            if not names or len(set(robot_names)) != len(robot_names) or not set(names).issubset(robot_names):
                raise ValueError("Recorded robot joint order is unavailable; hand poses cannot be inferred.")
            xml = recorded_urdf(metadata, request["repository"], request.get("hand_bundle"))
            wrists = [names[i] for group in metadata.get("groups", []) for i in group["wrist_indices"]]
            kinematics = HandKinematics(xml, names, wrists)
        except (ValueError, OSError, KeyError) as error:
            warnings.append(str(error))
        ids = [robot_names.index(name) for name in names] if kinematics else []
        frames = []
        for index, state in enumerate(states):
            if not isinstance(state, dict):
                raise ValueError("Invalid recorded scene state")
            frame = dict(time=index * dt, index=index)
            if kinematics:
                value = state.get("articulation", {}).get("robot", {})
                joints = np.asarray(value.get("joint_position"), dtype=float).reshape(-1)
                pose = np.asarray(value.get("root_pose"), dtype=float).reshape(-1)
                if joints.shape != (len(robot_names),) or not np.isfinite(joints).all():
                    raise ValueError("Recorded robot state differs from its named joint order")
                joints = joints[ids]
                frame["actual"] = kinematics.points(joints, pose)
                frame["hand_poses"] = {"actual": dict(joints=joints.tolist(), root=pose.tolist())}
            objects = {}
            for name, value in state.get("rigid_object", {}).items():
                if "root_pose" in value:
                    pose = np.asarray(value["root_pose"], dtype=float).reshape(-1)
                    pose_matrix(pose)
                    objects[name] = pose.tolist()
            frame["objects"] = objects
            frames.append(frame)
        visual_url = request.get("hand_visual_url")
        if not visual_url:
            warnings.append(request.get("hand_visual_warning") or "The recorded hand visual bundle is unavailable; saved keypoints remain available.")
        result = dict(schema="skynet.episode-viewer/v1", kind="collection", robot=robot,
                      hand=metadata.get("hand", request.get("hand")), views=[], frames=frames,
                      edges=kinematics.edges if kinematics else [], point_names=kinematics.links if kinematics else [],
                      kinematics_urdf=replay_urdf(xml) if kinematics else None, joint_names=names,
                      hand_visual_available=bool(visual_url), source_names=(payload.get("skynet_hand") or {}).get("source_names"),
                      scene_objects=scene.get("objects", []), scene_appearance=scene.get("appearance"),
                      duration=len(actions) * dt, layers={"actual": "Recorded hand"}, warnings=warnings,
                      source_sha256=request["source_sha256"], preview_version=request.get("preview_version"))
        encoded = json.dumps(result, allow_nan=False)
        if len(encoded.encode()) > MAX_VIEWER_BYTES:
            raise ValueError("Recorded scene exceeds the preview size limit")
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(encoded)
        temporary.replace(receipt)
    return {"state": "READY"}
