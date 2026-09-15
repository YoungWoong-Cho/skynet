"""Prevent wrist motion or a slightly moving thumb from hiding frozen fingers."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "retargeting_validation",
    Path(__file__).resolve().parents[1] / "ops/xr/retargeting_validation.py",
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def motion():
    # Names intentionally differ between the command array and PhysX order.
    groups = {"index_tip": ["index_1", "index_2"], "thumb_tip": ["thumb_1"]}
    joint_names = ["wrist", "thumb_1", "index_2", "index_1"]
    action_names = ["wrist", "index_1", "index_2", "thumb_1"]
    samples = {}
    for label, angle, z in [("open", 0, 0), ("closed", 0.7, -0.06), ("reopened", 0, 0)]:
        samples[label] = dict(
            target=[0, angle, angle, angle],
            actual=[0, angle, angle, angle],
            tips_local={tip: [0.15, 0, z] for tip in groups},
        )
    return groups, joint_names, action_names, samples


def test_each_finger_must_curl_and_reopen_with_correct_commands():
    args = motion()
    result = validation.verify_finger_motion(*args)
    assert result["index_tip"]["closing_tip_motion_m"] == 0.06
    assert result["thumb_tip"]["max_command_error_rad"] == 0


def test_nearly_frozen_index_fails_even_when_wrist_and_thumb_move():
    groups, joint_names, action_names, samples = motion()
    # Reproduced failure: the index receives ~1.5 degree flexion commands,
    # while wrist/partial thumb motion passed the previous aggregate check.
    samples["closed"]["actual"] = [0.23, 0.7, 0.027, 0.025]
    samples["closed"]["target"] = [0.23, 0.025, 0.027, 0.7]
    with pytest.raises(AssertionError, match="index_tip: finger did not curl"):
        validation.verify_finger_motion(groups, joint_names, action_names, samples)


def test_simulator_must_follow_targets_and_hand_must_reopen():
    args = motion()
    args[-1]["closed"]["actual"][3] -= 0.2
    with pytest.raises(AssertionError, match="track its joint commands"):
        validation.verify_finger_motion(*args)
    args = motion()
    args[-1]["reopened"] = copy.deepcopy(args[-1]["closed"])
    with pytest.raises(AssertionError, match="curl and reopen"):
        validation.verify_finger_motion(*args)


def test_fingertip_motion_excludes_a_rigidly_moving_hand():
    args = motion()
    args[-1]["closed"]["tips_local"]["index_tip"] = [0.15, 0, 0]
    with pytest.raises(AssertionError, match="index_tip: finger did not curl"):
        validation.verify_finger_motion(*args)


def test_chain_groups_include_fingers_and_exclude_virtual_wrist():
    urdf = b"""<robot name="test">
      <joint name="wrist"><parent link="world"/><child link="palm"/></joint>
      <joint name="finger"><parent link="palm"/><child link="distal"/></joint>
      <joint name="tip_fixed"><parent link="distal"/><child link="tip"/></joint>
    </robot>"""
    hand = dict(control_frame="palm", finger_joints=["finger"], tips=["tip"])
    assert validation.finger_joint_groups(urdf, hand) == {"tip": ["finger"]}
    hand["tips"] = ["missing"]
    with pytest.raises(ValueError, match="Invalid finger chain"):
        validation.finger_joint_groups(urdf, hand)
