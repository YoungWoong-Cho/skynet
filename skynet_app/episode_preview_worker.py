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
            warnings = []
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
                        frames.append(frame)
                pending.replace(root / "views.mp4")
                result = dict(schema="skynet.episode-viewer/v1", kind="collection", robot=metadata["robot"],
                              hand=metadata["hand"], hand_key=metadata.get("hand_key"), views=views, frames=frames,
                              edges=kinematics.edges if kinematics else [], point_names=kinematics.links if kinematics else [],
                              duration=length * metadata["step_dt"], layers={"actual": "Recorded hand"}, warnings=warnings,
                              source_sha256=request["source_sha256"], video_file="views.mp4")
                temporary = receipt.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, allow_nan=False))
                temporary.replace(receipt)
            finally:
                pending.unlink(missing_ok=True)
    return {"state": "READY"}
