"""Portable state, action and scene metadata; no camera capture dependency."""
import hashlib
from pathlib import Path
import numpy as np

def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


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
        step_dt=float(env.step_dt), alignment="state before action; simulation time excludes tracking pauses",
    )
    from scene_geometry import scene_snapshot
    metadata["scene_geometry"] = scene_snapshot(env.scene)
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
