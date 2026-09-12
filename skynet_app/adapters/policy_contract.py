"""Portable observation/action checks shared by planning and rollout workers."""

import math

CAMERAS = ("scene_front", "scene_left", "scene_right")
JOINT_SEMANTICS = "raw_joint_position_command; target = action * scale + offset"


def recorded_contract(metadata, *, images=True):
    capture = metadata.get("capture") or {}
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
        "step_dt": capture.get("step_dt"),
        "cameras": {name: (capture.get("cameras") or {}).get(name) for name in CAMERAS} if images else {},
        "color_space": capture.get("color_space") if images else None,
        "source_revision": capture.get("source_revision"),
    }


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
    if not math.isclose(step_dt, contract["step_dt"], rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("Simulator control frequency differs from the policy; temporal conversion is required")
    return [names.index(expected[i]) for i in contract["policy_to_source_indices"]]


def validate_observation(contract, observation):
    import numpy as np
    state = np.asarray(observation["state"])
    if state.shape != (len(contract["joint_names"]),) or not np.isfinite(state).all():
        raise ValueError("Policy observation has invalid joint dimensions or non-finite values")
    for name, camera in contract["cameras"].items():
        image = np.asarray(observation.get("images", {}).get(name))
        if image.dtype != np.uint8 or image.shape != (camera["height"], camera["width"], 3):
            raise ValueError(f"Camera {name} does not match the policy's RGB uint8 image contract")


def validate_actions(contract, actions):
    import numpy as np
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != len(contract["joint_names"]) or not np.isfinite(values).all():
        raise ValueError("Policy output has invalid action dimensions or non-finite values")
    return values
