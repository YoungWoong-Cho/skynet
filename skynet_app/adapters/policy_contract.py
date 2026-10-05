"""Portable observation/action checks shared by planning and rollout workers."""

import math
from copy import deepcopy

try:
    from .recording_time import frequency_stride
except ImportError:  # Frozen worker capsules use standalone modules.
    from recording_time import frequency_stride

CAMERAS = ("scene_front", "scene_left", "scene_right")
JOINT_SEMANTICS = "raw_joint_position_command; target = action * scale + offset"


def observation_camera_recipe_matches(expected, actual):
    """Check explicitly frozen camera fields; source-pinned defaults may add keys."""
    def matches(wanted, value):
        if isinstance(wanted, dict):
            return isinstance(value, dict) and all(key in value and matches(item, value[key]) for key, item in wanted.items())
        if isinstance(wanted, (list, tuple)):
            return isinstance(value, (list, tuple)) and len(wanted) == len(value) and all(matches(a, b) for a, b in zip(wanted, value))
        if type(wanted) in (int, float) and type(value) in (int, float):
            return math.isfinite(wanted) and math.isfinite(value) and math.isclose(wanted, value, rel_tol=1e-6, abs_tol=1e-8)
        return wanted == value
    return bool(expected) and set(expected) == set(actual) and matches(expected, actual)


def recorded_contract(metadata, *, images=True, control_hz=None):
    capture = metadata.get("capture") or {}
    source_dt = capture.get("step_dt")
    if control_hz is not None:
        if type(source_dt) not in (int, float) or not math.isfinite(source_dt) or source_dt <= 0:
            raise ValueError("Recording control period is required to select a control frequency")
        frequency_stride(1 / source_dt, control_hz)
    return {
        "schema_version": "skynet.policy-io/v1",
        "representation": metadata.get("contract"),
        "robot": capture.get("robot"),
        "hand": capture.get("hand"),
        "joint_names": capture.get("action_joint_names"),
        "policy_to_source_indices": metadata.get("policy_to_source_indices"),
        "action_semantics": capture.get("action_semantics"),
        "action_scale": capture.get("action_scale"),
        "action_offset": capture.get("action_offset"),
        "source_step_dt": source_dt,
        "step_dt": source_dt if control_hz is None else 1 / control_hz,
        "cameras": {name: (capture.get("cameras") or {}).get(name) for name in CAMERAS} if images else {},
        "color_space": capture.get("color_space") if images else None,
        "source_revision": capture.get("source_revision"),
    }


def unidex_contract(metadata, *, control_hz=None):
    """Exact target-hand geometry and point-cloud contract for FAAS execution.

    FAAS82 is the upstream policy representation. Mapping it to these asset-bound
    simulator actuators and preparing this camera recipe are Skynet components.
    """
    try:
        from .unidex_input import dataset_pointcloud_recipe
    except ImportError:
        from unidex_input import dataset_pointcloud_recipe
    try:
        from ops.datasets.action_codecs.unidex import load_codec
        from ops.datasets.observation_render import camera_recipe_from_capture
    except ImportError:
        from action_codecs.unidex import load_codec
        from observation_render import camera_recipe_from_capture
    if metadata.get("contract") != "skynet.unidex-pointcloud-faas/v1":
        raise ValueError("UniDex evaluation requires its prepared point-cloud/FAAS dataset contract")
    episodes = metadata.get("episodes") or []
    hands = {episode.get("hand_id") for episode in episodes}
    if not episodes or len(hands) != 1 or None in hands:
        raise ValueError("Choose an evaluation dataset containing exactly one verified hand")
    capture = episodes[0].get("capture") or metadata.get("capture") or {}
    codec = load_codec(next(iter(hands)), capture)
    camera = (capture.get("cameras") or {}).get("scene_front") or {}
    keys = (("sensor", "prim_path", "width", "height", "offset", "projection") if "prim_path" in camera else
            ("sensor", "mount", "width", "height", "intrinsic_matrix", "position_world", "quaternion_world_ros"))
    if any(key not in camera for key in keys):
        raise ValueError("UniDex evaluation requires the frozen scene-front camera geometry")
    camera = {key: deepcopy(camera[key]) for key in keys}
    camera_recipe_from_capture(camera, capture.get("source_revision"))
    if (camera["width"], camera["height"]) != (256, 256):
        raise ValueError("UniDex evaluation requires the recorded 256 by 256 camera")
    pointcloud = dataset_pointcloud_recipe(metadata)
    for episode in episodes:
        episode_capture = episode.get("capture") or metadata.get("capture") or {}
        load_codec(codec.robot_id, episode_capture)
        recorded_camera = (episode_capture.get("cameras") or {}).get("scene_front") or {}
        if any(recorded_camera.get(key) != camera[key] for key in keys):
            raise ValueError("Evaluation dataset contains different scene-front camera recipes")
        representation = episode.get("action_representation") or {}
        if representation.get("codec_sha256") != codec.digest or representation.get("frame") != "camera_opengl":
            raise ValueError("UniDex evaluation requires the exact recorded FAAS codec and camera frame")
    result = recorded_contract({**metadata, "capture": capture,
                               "policy_to_source_indices": list(range(codec.action_dim))},
                              images=False, control_hz=control_hz)
    result.update(observation_mode="pointcloud", cameras={"scene_front": camera}, color_space="RGB",
                  pointcloud_recipe=deepcopy(pointcloud), codec_sha256=codec.digest,
                  state_representation="absolute_faas82_camera_opengl",
                  action_representation="chunk_relative_faas82_to_native_controller_commands")
    return result


def contract_issues(contract):
    """Missing evidence stays unknown; a contradictory contract is incompatible."""
    issues = []

    def issue(field, message, status="unknown"):
        issues.append({"field": field, "status": status, "message": message})

    names = contract.get("joint_names")
    if not isinstance(names, list) or not names:
        issue("joint_names", "The training dataset does not record its action joint names.")
        names = []
    elif any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        issue("joint_names", "Action joint names must be unique and nonempty.", "incompatible")
    order = contract.get("policy_to_source_indices")
    if not isinstance(order, list):
        issue("joint_order", "The checkpoint's policy-to-joint mapping is missing.")
    elif any(type(i) is not int for i in order) or sorted(order) != list(range(len(names))):
        issue("joint_order", "The policy-to-joint mapping is not a complete permutation.", "incompatible")
    for field in ("action_scale", "action_offset"):
        values = contract.get(field)
        if not isinstance(values, list):
            issue(field, f"The training dataset does not record {field}.")
        elif len(values) != len(names) or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            issue(field, f"Invalid {field} for the recorded joints.", "incompatible")
    semantics = contract.get("action_semantics")
    if not semantics:
        issue("action_semantics", "Action units and joint-position command semantics are not recorded.")
    elif semantics != JOINT_SEMANTICS:
        issue("action_semantics", "This simulator bridge requires raw joint-position commands; another action conversion is needed.", "mapping_required")
    dt = contract.get("step_dt")
    if dt is None:
        issue("step_dt", "The training control period is missing.")
    elif type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
        issue("step_dt", "The training control period is invalid.", "incompatible")
    source_dt = contract.get("source_step_dt", dt)
    if source_dt is None:
        issue("source_step_dt", "The recording control period is missing.")
    elif type(source_dt) not in (int, float) or not math.isfinite(source_dt) or source_dt <= 0:
        issue("source_step_dt", "The recording control period is invalid.", "incompatible")
    elif source_dt and type(dt) in (int, float) and math.isfinite(dt) and dt > 0:
        try:
            frequency_stride(1 / source_dt, 1 / dt)
        except ValueError as error:
            issue("step_dt", str(error), "incompatible")
    for field in ("robot", "hand", "source_revision"):
        if not contract.get(field):
            issue(field, f"The recorded {field.replace('_', ' ')} is missing.")
    for name, camera in (contract.get("cameras") or {}).items():
        if not isinstance(camera, dict):
            issue("cameras", f"Camera {name} is missing from the training dataset.")
        elif any(type(camera.get(key)) is not int or camera[key] <= 0 for key in ("width", "height")):
            issue("cameras", f"Camera {name} dimensions are missing or invalid.")
    if contract.get("cameras") and contract.get("color_space") != "RGB":
        issue("color_space", "RGB camera input is required.", "mapping_required")
    return issues


def simulation_stride(contract, step_dt):
    """Number of original simulator steps for each learned position command."""
    if type(step_dt) not in (int, float) or not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("Simulator control period must be finite and positive")
    source_dt = contract.get("source_step_dt", contract["step_dt"])
    if not math.isclose(step_dt, source_dt, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("Simulator control frequency differs from the recording")
    return frequency_stride(1 / step_dt, 1 / contract["step_dt"])


def environment_mapping(contract, names, scales, offsets, step_dt):
    """Resolve by joint identity; equal tensor sizes do not imply compatibility."""
    issues = contract_issues(contract)
    if issues:
        raise ValueError("; ".join(i["message"] for i in issues))
    expected = contract["joint_names"]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError("Simulator joint identities differ from the policy; an embodiment mapping is required")
    if len(scales) != len(names) or len(offsets) != len(names):
        raise ValueError("Simulator action scales and offsets do not cover every joint")
    for i, name in enumerate(expected):
        j = names.index(name)
        if not math.isclose(scales[j], contract["action_scale"][i], rel_tol=1e-6, abs_tol=1e-6) or not math.isclose(offsets[j], contract["action_offset"][i], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Simulator scale or offset differs for {name}; an action conversion is required")
    simulation_stride(contract, step_dt)
    return [names.index(expected[i]) for i in contract["policy_to_source_indices"]]


def validate_observation(contract, observation, *, require_pointcloud=True):
    import numpy as np
    state = np.asarray(observation["state"])
    if state.shape != (len(contract["joint_names"]),) or not np.isfinite(state).all():
        raise ValueError("Policy observation has invalid joint dimensions or non-finite values")
    for name, camera in contract["cameras"].items():
        image = np.asarray(observation.get("images", {}).get(name))
        if image.dtype != np.uint8 or image.shape != (camera["height"], camera["width"], 3):
            raise ValueError(f"Camera {name} does not match the policy's RGB uint8 image contract")
    if contract.get("action_representation") in {"hat128_to_wuji2_position_optimizer", "hat128_to_native_position_optimizer"}:
        try:
            from observation_geometry import rigid_transform
        except ImportError:
            from ops.datasets.observation_geometry import rigid_transform
        rigid_transform(observation.get("world_from_camera"))
        rigid_transform(observation.get("world_from_root"))
    if contract.get("observation_mode") == "pointcloud":
        if require_pointcloud:
            try:
                from .unidex_input import validate_pointcloud_recipe
            except ImportError:
                from unidex_input import validate_pointcloud_recipe
            recipe = validate_pointcloud_recipe(contract.get("pointcloud_recipe"))
            count = recipe["num_points"]
            cloud = np.asarray(observation.get("pointcloud"))
            if (cloud.shape != (count, 6) or cloud.dtype != np.float32 or not np.isfinite(cloud).all()
                    or np.any(cloud[:, 3:] < 0) or np.any(cloud[:, 3:] > 1)):
                raise ValueError(f"UniDex observation requires finite metric XYZRGB{count} with colors in [0, 1]")
        try:
            from observation_geometry import rigid_transform
        except ImportError:
            from ops.datasets.observation_geometry import rigid_transform
        rigid_transform(observation.get("world_from_camera"))
        rigid_transform(observation.get("world_from_root"))


def validate_actions(contract, actions):
    import numpy as np
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != len(contract["joint_names"]) or not np.isfinite(values).all():
        raise ValueError("Policy output has invalid action dimensions or non-finite values")
    return values
