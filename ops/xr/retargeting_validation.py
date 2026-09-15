"""Per-finger checks for the standalone simulator retargeting diagnostic."""

import xml.etree.ElementTree as ET

import numpy as np


def finger_joint_groups(urdf, hand):
    parents = {
        j.find("child").get("link"): (j.find("parent").get("link"), j.get("name"))
        for j in ET.fromstring(urdf).findall("joint")
    }
    fingers = set(hand["finger_joints"])
    groups = {}
    for tip in hand["tips"]:
        node, visited, joints = tip, set(), []
        while node != hand["control_frame"]:
            if node in visited or node not in parents:
                raise ValueError("Invalid finger chain: " + tip)
            visited.add(node)
            node, joint = parents[node]
            if joint in fingers:
                joints.append(joint)
        if not joints:
            raise ValueError("Finger has no controlled joints: " + tip)
        groups[tip] = joints
    return groups


def verify_finger_motion(groups, joint_names, action_names, samples):
    """A moving wrist or one moving finger must not pass for the whole hand."""
    report = {}
    for tip, names in groups.items():
        ids = [joint_names.index(n) for n in names]
        action_ids = [action_names.index(n) for n in names]
        poses = {
            label: np.asarray(sample["actual"])[ids]
            for label, sample in samples.items()
        }
        closed = float(np.linalg.norm(poses["closed"] - poses["open"]))
        reopened = float(np.linalg.norm(poses["reopened"] - poses["closed"]))
        error = max(
            float(
                np.max(np.abs(poses[label] - np.asarray(sample["target"])[action_ids]))
            )
            for label, sample in samples.items()
        )
        # The control frame removes wrist translation and rotation from the
        # fingertip motion. Its +Z points out of the back of the hand.
        z = {label: sample["tips_local"][tip][2] for label, sample in samples.items()}
        curl = float(z["open"] - z["closed"])
        uncurl = float(z["reopened"] - z["closed"])
        report[tip] = dict(
            closing_joint_motion_rad=closed,
            reopening_joint_motion_rad=reopened,
            max_command_error_rad=error,
            closing_tip_motion_m=curl,
            reopening_tip_motion_m=uncurl,
        )
        if not np.isfinite(list(report[tip].values())).all():
            raise AssertionError(f"{tip}: nonfinite finger motion")
        if closed < 0.25 or reopened < 0.25 or curl < 0.015 or uncurl < 0.015:
            raise AssertionError(
                f"{tip}: finger did not curl and reopen: {report[tip]}"
            )
        if error > 0.05:
            raise AssertionError(
                f"{tip}: finger did not track its joint commands: {report[tip]}"
            )
    return report
