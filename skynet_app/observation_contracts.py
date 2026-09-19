"""Versioned observation requirements and content-addressed preparation plans.

This module is deliberately dependency-free: the server and frozen preparation
worker use the same validation and identity rules. Adapter IDs, dataset names,
splits and storage paths are never part of an artifact's identity.
"""

from copy import deepcopy
import hashlib
import json
import math
import re

CONTRACT_SCHEMA = "skynet.observation-requirements/v1"
ARTIFACT_SCHEMA = "skynet.observation-artifact/v1"
PREPARE_SCHEMA = "skynet.observation-prepare/v1"
SCENE_CAMERAS = ("scene_front", "scene_left", "scene_right")


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("Observation names must be safe camera or stream identifiers")
    return value


def rgb_requirements(cameras=SCENE_CAMERAS, *, width=256, height=256):
    return validate_requirements({
        "schema": CONTRACT_SCHEMA,
        "timing": {"alignment": "pre_action_state", "stride": 1},
        "streams": [{"name": camera, "modality": "rgb", "camera_ids": [camera],
                     "width": width, "height": height, "dtype": "uint8", "color_space": "RGB"}
                    for camera in cameras],
    })


def validate_requirements(value):
    result = deepcopy(value)
    if not isinstance(result, dict) or result.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("Unsupported observation requirements schema")
    timing = result.get("timing", {})
    if timing != {"alignment": "pre_action_state", "stride": 1}:
        raise ValueError("Prepared observations must preserve every recorded pre-action frame")
    streams = result.get("streams")
    if not isinstance(streams, list) or len(streams) > 32:
        raise ValueError("Observation requirements must contain at most 32 streams")
    names = []
    for stream in streams:
        names.append(_name(stream.get("name")))
        cameras = stream.get("camera_ids")
        if not isinstance(cameras, list) or not cameras or len(cameras) > 16 or len(set(cameras)) != len(cameras):
            raise ValueError("Observation streams require distinct camera IDs")
        for camera in cameras:
            _name(camera)
        for axis in ("width", "height"):
            if type(stream.get(axis)) is not int or not 1 <= stream[axis] <= 4096:
                raise ValueError("Observation render dimensions must be between 1 and 4096")
        modality = stream.get("modality")
        if modality == "rgb":
            if len(cameras) != 1 or stream.get("dtype") != "uint8" or stream.get("color_space") != "RGB":
                raise ValueError("RGB streams require one camera and RGB uint8 values")
        elif modality == "depth":
            if len(cameras) != 1 or stream.get("dtype") != "float32" or stream.get("depth_kind") != "distance_to_image_plane" or stream.get("units") != "metres":
                raise ValueError("Depth streams require metric float32 image-plane depth")
        elif modality == "point_cloud":
            frame = stream.get("coordinate_frame")
            allowed_frames = {"world", *("camera:" + camera for camera in cameras)}
            if len(cameras) == 1:
                allowed_frames.add("camera")
            if stream.get("channels") not in ("XYZ", "XYZRGB") or frame not in allowed_frames:
                raise ValueError("Point clouds require XYZ/XYZRGB in world or an explicitly selected camera frame")
            if type(stream.get("num_points")) is not int or not 1 <= stream["num_points"] <= 100000:
                raise ValueError("Point-cloud point count is invalid")
            if stream.get("sampling") != "farthest_point" or type(stream.get("seed")) is not int:
                raise ValueError("Point clouds require deterministic farthest-point sampling")
            if stream.get("processing_order") != ["unproject", "world_transform", "merge", "crop", "sample"]:
                raise ValueError("Point-cloud cameras must be merged before final sampling")
            if stream.get("insufficient_points") != "repeat_with_mask":
                raise ValueError("Point clouds must explicitly mask repeated padding points")
            bounds = stream.get("crop")
            if bounds is not None:
                if set(bounds) != {"min", "max"} or any(len(bounds[k]) != 3 for k in bounds):
                    raise ValueError("Point-cloud crop requires XYZ min/max bounds in the selected coordinate frame")
                if any(not isinstance(v, (int, float)) or not math.isfinite(v) for k in bounds for v in bounds[k]) or any(a >= b for a, b in zip(bounds["min"], bounds["max"])):
                    raise ValueError("Point-cloud crop bounds must be finite and increasing")
            depth_range = stream.get("depth_range", [0.0, 2.5])
            if len(depth_range) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in depth_range) or not 0 <= depth_range[0] < depth_range[1]:
                raise ValueError("Point-cloud depth range must be finite positive metres")
            stream["depth_range"] = depth_range
        else:
            raise ValueError("Unsupported observation modality")
    if len(set(names)) != len(names):
        raise ValueError("Observation stream names must be unique")
    return result


def plan_artifacts(source_sha, source_info, contract, render_identity):
    """Return dependency-ordered nodes. Paths and job IDs are supplied by caller.

    ``render_identity`` must include the pinned scene/renderer identity and a
    ``cameras`` mapping of physical camera recipes. Only the selected camera's
    recipe participates in its key, so adding another view reuses existing work.
    """
    if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
        raise ValueError("Observation preparation requires a source SHA-256")
    contract = validate_requirements(contract)
    if not contract["streams"]:
        return []
    cameras = render_identity.get("cameras", {})
    scene = {k: v for k, v in render_identity.items() if k != "cameras"}
    if not scene:
        raise ValueError("Observation preparation requires a pinned render identity")
    nodes = {}

    def add(camera_id, modality, recipe, dependencies=()):
        spec = {"schema": ARTIFACT_SCHEMA, "source_sha256": source_sha,
                "episode_index": source_info.get("episode_index", 0),
                "modality": modality, "camera_id": camera_id, "recipe": recipe,
                "timing": contract["timing"], "render_identity": scene,
                "dependencies": list(dependencies)}
        key = content_digest(spec)
        nodes.setdefault(key, {"artifact_key": key, "episode_key": source_info.get("episode_key", source_sha),
                               "camera_id": camera_id, "modality": modality, "recipe": recipe,
                               "source_sha256": source_sha, "spec": spec,
                               "dependencies": list(dependencies)})
        return key

    def render(camera_id, modality, stream):
        if camera_id not in cameras:
            raise ValueError("Recording has no calibrated camera recipe for " + camera_id)
        recipe = {"camera": deepcopy(cameras[camera_id]), "width": stream["width"], "height": stream["height"],
                  "dtype": "uint8" if modality == "rgb" else "float32"}
        if modality == "rgb":
            recipe["color_space"] = "RGB"
        else:
            recipe.update(depth_kind="distance_to_image_plane", units="metres")
        return add(camera_id, modality, recipe)

    for stream in contract["streams"]:
        if stream["modality"] in {"rgb", "depth"}:
            key = render(stream["camera_ids"][0], stream["modality"], stream)
        else:
            dependencies = []
            for camera in stream["camera_ids"]:
                dependencies.append(render(camera, "depth", stream))
                if stream["channels"] == "XYZRGB":
                    dependencies.append(render(camera, "rgb", stream))
            recipe = {k: deepcopy(v) for k, v in stream.items() if k != "name"}
            camera_id = (stream["camera_ids"][0] if len(stream["camera_ids"]) == 1 else
                         "merged-" + content_digest(stream["camera_ids"])[:16])
            key = add(camera_id, "point_cloud", recipe, dependencies)
        nodes[key].setdefault("streams", []).append(stream["name"])
    return list(nodes.values())
