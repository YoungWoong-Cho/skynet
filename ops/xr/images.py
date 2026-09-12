"""Synchronized RGB sidecars, frozen with each collection session.

No Isaac imports at module load: the file writer and layout validation are CPU-testable.
"""
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

CAMERAS = {"scene_front": "third_person_camera", "scene_left": "third_person_camera_left", "scene_right": "third_person_camera_right"}
SCHEMA = "skynet.rgb-trajectory/v1"
MAX_STEPS = 6000


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def configure_cameras(cfg):
    # Observations are read directly from sensors, without observation histories.
    cfg._apply_observation_preset("state")
    cfg._apply_multiview_cameras(True)
    for name in CAMERAS.values():
        camera = getattr(cfg.scene, name, None)
        if camera is None:
            raise ValueError("Training image camera is unavailable: " + name)
        camera.width = camera.height = 256
        camera.data_types = ["rgb"]
        camera.update_period = 0.0
    for name in vars(cfg.scene):
        if name not in CAMERAS.values() and "camera" in name:
            setattr(cfg.scene, name, None)
    cfg.observations.debug_vis = None


def joint_layout(names, hand, wrist_names=None):
    if len(names) != len(set(names)) or hand not in {"left", "right", "both"}:
        raise ValueError("Unsupported or duplicate action joint layout")
    sides = ["left", "right"] if hand == "both" else [hand]
    groups = []
    for side in sides:
        ids = [i for i, n in enumerate(names) if hand != "both" or n.startswith("lh_" if side == "left" else "rh_")]
        wrists = [i for i in ids if names[i] in (wrist_names or []) or names[i].endswith(("_translation_joint", "_rotation_joint"))]
        fingers = [i for i in ids if i not in wrists]
        if len(wrists) != 6 or not fingers:
            raise ValueError("Export requires six named virtual wrist joints and named finger joints per hand")
        groups.append(dict(side=side, wrist_indices=wrists, finger_indices=fingers))
    order = [i for g in groups for i in g["wrist_indices"] + g["finger_indices"]]
    if sorted(order) != list(range(len(names))):
        raise ValueError("Action joints cannot be mapped without losing commands")
    return groups


class ImageWriter:
    """Stream every control step to a temporary file; publish only complete episodes."""
    def __init__(self, root, metadata):
        import h5py
        self.root = Path(root)
        directory = self.root / "recordings/live"
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / ("images-" + uuid4().hex + ".hdf5")
        self.temp = self.path.with_suffix(".part")
        self.file = h5py.File(self.temp, "x")
        self.file.attrs["schema"] = SCHEMA
        self.file.attrs["metadata"] = json.dumps(metadata, allow_nan=False)
        self.steps = 0
        self.dimension = len(metadata["action_joint_names"])

    def append(self, state, action, images, timestamp, wall_time):
        state, action = array(state).reshape(-1), array(action).reshape(-1)
        if state.shape != (self.dimension,) or action.shape != state.shape or not np.isfinite([state, action]).all():
            raise ValueError("Invalid image-aligned state/action")
        if self.steps >= MAX_STEPS:
            raise ValueError("Image episode exceeded 6000 steps; restart with a shorter demonstration")
        values = dict(state=state.astype("f4"), action=action.astype("f4"), timestamps=np.asarray(timestamp, dtype="f8"), wall_times=np.asarray(wall_time, dtype="f8"))
        if set(images) != set(CAMERAS):
            raise ValueError("All three training cameras must be present")
        for name, image in images.items():
            image = array(image)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not all(image.shape):
                raise ValueError("Camera did not produce an RGB uint8 frame")
            values["images/" + name] = image
        if not np.isfinite([timestamp, wall_time]).all():
            raise ValueError("Invalid frame timestamp")
        if self.steps and timestamp <= self.file["timestamps"][-1]:
            raise ValueError("Frame timestamps must increase")
        for key, value in values.items():
            if key not in self.file:
                self.file.create_dataset(key, shape=(0, *value.shape), maxshape=(None, *value.shape), dtype=value.dtype, chunks=(1, *value.shape), compression="lzf")
            dataset = self.file[key]
            if dataset.shape[1:] != value.shape:
                raise ValueError("Camera or robot shape changed during the episode")
            dataset.resize(self.steps + 1, axis=0)
            dataset[self.steps] = value
        self.steps += 1
        if self.steps % 60 == 0:
            self.file.flush()
            if self.temp.stat().st_size > 4_000_000_000:
                raise ValueError("Image episode exceeds its 4 GB limit")

    def finish(self):
        if not self.steps:
            raise ValueError("Cannot save an empty image episode")
        self.file.attrs["complete"] = True
        self.file.close()
        with self.temp.open("rb") as stream:
            import os
            os.fsync(stream.fileno())
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if self.path.exists():
            raise FileExistsError("Refusing to replace saved images")
        self.temp.replace(self.path)
        return dict(path=str(self.path.relative_to(self.root)), sha256=digest, size_bytes=self.path.stat().st_size, steps=self.steps, schema=SCHEMA)

    def discard(self):
        self.file.close()
        self.temp.unlink(missing_ok=True)


def state_metadata(env, profile):
    robot = env.scene["robot"]
    names, terms, scales, offsets = [], {}, [], []
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        if type(term).__name__ != "JointPositionAction" or term.cfg.asset_name != "robot":
            raise ValueError("State capture supports named joint-position actions only: " + name)
        joint_names = list(term._joint_names)
        terms[name] = joint_names
        names.extend(joint_names)
        for key, target in [("_scale", scales), ("_offset", offsets)]:
            value = array(getattr(term, key))
            target.extend(np.broadcast_to(value, (1, len(joint_names)))[0].tolist())
    ids = [robot.joint_names.index(n) for n in names]
    metadata = dict(
        robot=profile["robot"], task=profile["task"], hand=profile["hand"],
        source_revision=profile["source_revision"], action_joint_names=names, robot_joint_names=list(robot.joint_names),
        action_terms=terms, action_scale=scales, action_offset=offsets,
        action_semantics="raw_joint_position_command; target = action * scale + offset",
        groups=joint_layout(names, profile["hand"], profile.get("hand_manifest", {}).get("wrist_joints")),
        step_dt=float(env.step_dt), alignment="image and state before action; simulation time excludes tracking pauses",
        color_space="RGB", cameras={k: dict(sensor=v, mount="fixed_scene", width=256, height=256) for k, v in CAMERAS.items()},
    )
    hand = profile.get("hand_manifest")
    if hand:
        metadata.update({k: hand[k] for k in ("hand_asset", "units", "wrist_rotation_order", "hand_order", "hands") if k in hand})
        metadata["hand_adapter_digest"] = hand["digest"]
    # Keep geometry portable when a collection host or simulator checkout disappears.
    # Only the kinematic tree is needed for replay, not meshes or physics settings.
    import xml.etree.ElementTree as ET
    if profile.get("hand_bundle", {}).get("root"):
        model_path = Path(profile["hand_bundle"]["root"]) / "simulation.urdf"
    else:
        model_path = Path(profile.get("repository", "")) / "source/dexverse/dexverse/robot_agents/shadow/retarget" / (profile["robot"] + ".urdf")
    if model_path.is_file() and model_path.stat().st_size <= 4_000_000:
        tree = ET.parse(model_path).getroot()
        for body in tree.findall("link"):
            for item in list(body):
                body.remove(item)
        metadata["kinematics_urdf"] = ET.tostring(tree, encoding="unicode")
        metadata["kinematics_sha256"] = hashlib.sha256(metadata["kinematics_urdf"].encode()).hexdigest()
    return ids, metadata


class ImageRecorder:
    def __init__(self, root, env, profile, retargeters):
        # Fail before recording if the runtime is missing HDF5 support.
        import h5py  # noqa: F401
        self.root, self.env, self.retargeters = root, env, retargeters
        self.writer = None
        self.ids, self.metadata = state_metadata(env, profile)

    def begin(self):
        self.discard()
        # Render several times after reset to warm the camera pipeline, without stepping physics.
        for _ in range(3):
            self.env.sim.render()
        for name, sensor_name in CAMERAS.items():
            sensor = self.env.scene[sensor_name]
            sensor.update(0.0, force_recompute=True)
            data = sensor.data
            self.metadata["cameras"][name].update(
                intrinsic_matrix=array(data.intrinsic_matrices)[0].tolist(),
                position_world=array(data.pos_w)[0].tolist(),
                quaternion_world_ros=array(data.quat_w_ros)[0].tolist(),
            )
        self.writer = ImageWriter(self.root, self.metadata)

    def append(self, action, wall_time=None):
        # Tracking overlays belong to the headset preview, not the training cameras.
        markers = [getattr(r, n, None) for r in self.retargeters for n in ("_markers", "_canonical_markers")]
        for marker in markers:
            if marker is not None:
                marker.set_visibility(False)
        try:
            self.env.sim.render()
            images = {}
            for name, sensor_name in CAMERAS.items():
                sensor = self.env.scene[sensor_name]
                sensor.update(0.0, force_recompute=True)
                images[name] = array(sensor.data.output["rgb"])[0, :, :, :3].copy()
            state = self.env.scene["robot"].data.joint_pos[0, self.ids]
            self.writer.append(state, action, images, self.writer.steps * self.metadata["step_dt"], time.time() if wall_time is None else wall_time)
        finally:
            for marker in markers:
                if marker is not None:
                    marker.set_visibility(True)

    def finish(self):
        result = self.writer.finish()
        self.writer = None
        return result

    def discard(self):
        if self.writer:
            self.writer.discard()
            self.writer = None


def camera_recipe(cfg):
    """Freeze camera geometry without creating sensors in the XR environment."""
    import copy
    prepared = copy.deepcopy(cfg)
    configure_cameras(prepared)
    for name in vars(prepared.scene):
        spawn = getattr(getattr(prepared.scene, name, None), "spawn", None)
        if spawn is not None and any(s in type(spawn).__name__ for s in ["MultiAsset", "MultiUsd"]):
            raise ValueError("Training images require a fixed scene asset layout")
    result = {}
    for name, sensor in CAMERAS.items():
        camera = getattr(prepared.scene, sensor)
        spawn = camera.spawn
        result[name] = dict(
            sensor=sensor, prim_path=camera.prim_path, width=256, height=256,
            offset=dict(pos=list(camera.offset.pos), rot=list(camera.offset.rot), convention=camera.offset.convention),
            projection={key: getattr(spawn, key, None) for key in ["focal_length", "focus_distance", "horizontal_aperture", "vertical_aperture", "horizontal_aperture_offset", "vertical_aperture_offset", "clipping_range", "projection_type"]},
        )
    return json.loads(json.dumps(result, allow_nan=False))


def training_image_request(recipe):
    raw = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return dict(schema="skynet.training-images/v2", mode="saved_states", recipe=recipe, recipe_sha256=hashlib.sha256(raw.encode()).hexdigest())
