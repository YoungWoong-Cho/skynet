"""Bounded CPU preview conversion. Executed beside the cluster's original files."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np


def prepare_preview(request):
    import fcntl
    import h5py
    import imageio.v2 as imageio

    root = Path(request["output"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = root / "viewer.json"
        if receipt.exists() and (root / "views.mp4").is_file():
            return {"state": "READY"}
        image_path = Path(request["images"]["path"])
        if image_path.stat().st_size != request["images"]["size_bytes"] or image_path.stat().st_size > 4_000_000_000:
            raise ValueError("Training camera file differs from its recording receipt")
        with image_path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != request["images"]["sha256"]:
                raise ValueError("Training camera checksum differs from its receipt")
        recording = Path(request["recording"])
        if not 0 < recording.stat().st_size <= 100_000_000:
            raise ValueError("Original recording exceeds the preview limit")
        raw = recording.read_bytes()
        if hashlib.sha256(raw).hexdigest() != request["source_sha256"]:
            raise ValueError("Original recording checksum changed")
        payload = ArrayUnpickler(io.BytesIO(raw)).load()
        episode = payload["episodes"][request["episode"]]
        states = episode["states"]
        with h5py.File(image_path, "r") as source:
            metadata = json.loads(source.attrs["metadata"])
            length = len(source["state"])
            if (source.attrs.get("schema") != "skynet.rgb-trajectory/v1" or not source.attrs.get("complete")
                    or not 0 < length <= 6000 or len(states) != length + 1
                    or metadata.get("source_sha256", request["source_sha256"]) != request["source_sha256"]):
                raise ValueError("Images do not align with the original recording")
            cameras = {name: metadata["cameras"][name] for name in source["images"]}
            views = camera_layout(cameras)
            scene = {"objects": metadata.get("scene_objects", []), "warnings": []}
            if "scene_geometry" in source:
                saved_scene = source["scene_geometry"]
                if saved_scene.size > 12_000_000:
                    raise ValueError("Saved scene geometry exceeds the preview limit")
                scene = json.loads(np.asarray(saved_scene, dtype="u1").tobytes())
                if scene.get("schema") != "skynet.scene-geometry/v1":
                    raise ValueError("Unsupported scene geometry format")
            warnings = list(scene.get("warnings", []))
            if source.attrs.get("scene_geometry_error"):
                warnings.append(str(source.attrs["scene_geometry_error"]))
            kinematics = None
            try:
                xml = recorded_urdf(metadata, request["repository"], request.get("hand_bundle"))
                wrists = [metadata["action_joint_names"][i] for g in metadata.get("groups", []) for i in g["wrist_indices"]]
                kinematics = HandKinematics(xml, metadata["action_joint_names"], wrists)
            except (ValueError, OSError) as error:
                warnings.append(str(error))
            frames = []
            pending = root / "views.pending.mp4"
            try:
                with imageio.get_writer(pending, fps=1 / metadata["step_dt"], codec="libx264", macro_block_size=1,
                                        ffmpeg_params=["-threads", "2", "-preset", "fast"]) as writer:
                    for index in range(length):
                        images = [np.asarray(source["images"][view["id"]][index]) for view in views]
                        if any(image.shape != (v["height"], v["width"], 3) for image, v in zip(images, views)):
                            raise ValueError("Camera dimensions changed within the recording")
                        writer.append_data(np.concatenate(images, axis=1))
                        frame = {"time": float(source["timestamps"][index]), "index": index}
                        if kinematics:
                            pose = states[index]["articulation"]["robot"].get("root_pose")
                            if pose is None:
                                raise ValueError("Recorded root pose is unavailable; cannot position keypoints")
                            frame["actual"] = kinematics.points(source["state"][index], pose)
                            frame["hand_poses"] = {"actual": {"joints": np.asarray(source["state"][index]).reshape(-1).tolist(),
                                                               "root": np.asarray(pose).reshape(-1).tolist()}}
                        objects = states[index].get("rigid_object", {})
                        frame["objects"] = {name: np.asarray(value["root_pose"]).reshape(-1).tolist()
                                            for name, value in objects.items() if "root_pose" in value}
                        frames.append(frame)
                # A trajectory has one final state after its final camera/action pair.
                # Keep it available to frame stepping and the recorded-values table.
                final_state = states[-1]
                final_frame = {"time": length * float(metadata["step_dt"]), "index": length}
                if kinematics:
                    robot = final_state["articulation"]["robot"]
                    ids = [metadata["robot_joint_names"].index(name) for name in metadata["action_joint_names"]]
                    joints = np.asarray(robot["joint_position"]).reshape(-1)[ids]
                    pose = np.asarray(robot["root_pose"]).reshape(-1)
                    final_frame["actual"] = kinematics.points(joints, pose)
                    final_frame["hand_poses"] = {"actual": {"joints": joints.tolist(), "root": pose.tolist()}}
                final_frame["objects"] = {name: np.asarray(value["root_pose"]).reshape(-1).tolist()
                                           for name, value in final_state.get("rigid_object", {}).items() if "root_pose" in value}
                frames.append(final_frame)
                pending.replace(root / "views.mp4")
                result = dict(schema="skynet.episode-viewer/v1", kind="collection", robot=metadata["robot"],
                              hand=metadata["hand"], hand_key=metadata.get("hand_key"), views=views, frames=frames,
                              edges=kinematics.edges if kinematics else [], point_names=kinematics.links if kinematics else [],
                              kinematics_urdf=replay_urdf(xml) if kinematics else None, joint_names=metadata["action_joint_names"],
                              scene_objects=scene.get("objects", []), scene_appearance=scene.get("appearance"),
                              duration=length * metadata["step_dt"], layers={"actual": "Recorded hand"}, warnings=warnings,
                              source_sha256=request["source_sha256"], video_file="views.mp4")
                temporary = receipt.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, allow_nan=False))
                temporary.replace(receipt)
            finally:
                pending.unlink(missing_ok=True)
    return {"state": "READY"}
