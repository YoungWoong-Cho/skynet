"""Selection and canonical model checks without a headset, GPU, or installed solver."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from skynet_app.retargeting import configuration, validate_hand, choices

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "retargeting_runtime", ROOT / "ops/xr/retargeting_runtime.py"
)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def wuji():
    xml = (ROOT / "tests/fixtures/recorded-wuji/kinematics.urdf").read_bytes()
    tree = ET.fromstring(xml)
    wrists = ["skynet_" + a for a in ("x", "y", "z", "roll", "pitch", "yaw")]
    fingers = [
        j.get("name")
        for j in tree.findall("joint")
        if j.get("type") != "fixed" and j.get("name") not in wrists
    ]
    hand = dict(
        control_frame="skynet_palm",
        wrist_joints=wrists,
        finger_joints=fingers,
        tips=[
            "h_r_thumb_tip",
            "h_r_index_finger_tip",
            "h_r_middle_finger_tip",
            "h_r_ring_finger_tip",
            "h_r_pinky_tip",
        ],
    )
    return xml, hand


def test_real_wuji_branch_preserves_all_26_joint_axes_and_limits(tmp_path):
    xml, hand = wuji()
    path = tmp_path / "kinematic.urdf"
    names, links, digest = runtime.kinematic_hand(xml, hand, path)
    assert len(names) == len(set(names)) == 26
    assert set(hand["tips"]) <= set(links) and "world" in links
    before, after = ET.fromstring(xml), ET.parse(path).getroot()
    for name in names:
        original = before.find(f"joint[@name='{name}']")
        converted = after.find(f"joint[@name='{name}']")
        assert ET.tostring(original) == ET.tostring(converted)
    assert not after.findall(".//visual") and not after.findall(".//collision")
    assert len(digest) == 64


def test_bimanual_model_extracts_only_selected_side(tmp_path):
    xml, hand = wuji()
    combined = ET.Element("robot", name="both")
    ET.SubElement(combined, "link", name="world")
    for prefix in ("lh_", "rh_"):
        tree = ET.fromstring(xml)
        for item in tree:
            if item.tag not in {"link", "joint"}:
                continue
            item.set("name", prefix + item.get("name"))
            for ref in item.findall("parent") + item.findall("child"):
                ref.set("link", prefix + ref.get("link"))
            combined.append(item)
        j = ET.SubElement(combined, "joint", name=prefix + "mount", type="fixed")
        ET.SubElement(j, "parent", link="world")
        ET.SubElement(j, "child", link=prefix + "skynet_base")
    for prefix in ("lh_", "rh_"):
        selected = {
            k: ([prefix + n for n in v] if isinstance(v, list) else prefix + v)
            for k, v in hand.items()
        }
        names, links, _ = runtime.kinematic_hand(
            ET.tostring(combined), selected, tmp_path / (prefix + ".urdf")
        )
        assert len(names) == 26
        assert all(n.startswith(prefix) for n in names)
        assert not any(n.startswith("rh_" if prefix == "lh_" else "lh_") for n in links)


def test_coupled_joints_and_incomplete_action_maps_rejected(tmp_path):
    xml, hand = wuji()
    tree = ET.fromstring(xml)
    j = tree.find(f"joint[@name='{hand['finger_joints'][1]}']")
    ET.SubElement(j, "mimic", joint=hand["finger_joints"][0])
    with pytest.raises(ValueError, match="coupled"):
        runtime.kinematic_hand(ET.tostring(tree), hand, tmp_path / "out.urdf")
    invalid = copy.deepcopy(hand)
    invalid["finger_joints"].pop()
    with pytest.raises(ValueError, match="every named"):
        runtime.kinematic_hand(xml, invalid, tmp_path / "out.urdf")
    cfg = configuration("vector-wrist-joint", "/test")
    with pytest.raises(ValueError, match="coupled"):
        validate_hand(cfg, {"mimic_joints": ["coupled"]})
    validate_hand(configuration("dexpilot", "/test"), {"mimic_joints": ["coupled"]})


def test_dexpilot_uses_exact_existing_commands_and_keeps_solver_untouched():
    calls = []
    continuity = SimpleNamespace(
        update=lambda action: (calls.append(action), action)[1],
        reset=lambda: calls.append("reset"),
    )
    teleop = SimpleNamespace(
        _retargeters=[SimpleNamespace(_retarget_hand_fingers="unchanged")]
    )
    commands = runtime.CollectionRetargeting({}, teleop, continuity, "/unused")
    source = np.arange(26, dtype=np.float32)
    assert commands.update(source, {}) is source
    assert teleop._retargeters[0]._retarget_hand_fingers == "unchanged"
    commands.reset()
    assert calls[-1] == "reset"
    assert commands.metadata["mode"] == "dexpilot"


def test_method_choices_are_isolated_and_unknowns_fail():
    first = choices()
    first[0]["key"] = "edited"
    assert choices()[0]["key"] == "dexpilot"
    with pytest.raises(ValueError, match="Unknown"):
        configuration("unknown", "/test")
    with pytest.raises(ValueError, match="not installed"):
        runtime.load_upstream("/nonexistent/skynet-retargeting-test")


def test_link_lookup_disambiguates_joint_and_link_frames(monkeypatch, tmp_path):
    import sys

    body = object()

    class Model:
        nframes = 3

        def getFrameId(self, name, frame_type=None):
            if name == "tip" and frame_type is None:
                raise ValueError("Several frames match the filter")
            return 2 if name == "tip" and frame_type is body else self.nframes

    class Robot:
        def __init__(self, *args):
            self.model = Model()

        def get_frame_pose(self, name):
            return self.get_frame_index(name)

        def get_frames_index(self, names):
            return [self.get_frame_index(name) for name in names]

    monkeypatch.setitem(
        sys.modules, "pinocchio", SimpleNamespace(FrameType=SimpleNamespace(BODY=body))
    )
    monkeypatch.setitem(
        sys.modules,
        "retargeting.core.kinematics.pinocchio_model",
        SimpleNamespace(RobotPinocchio=Robot),
    )
    model = runtime.link_kinematics(tmp_path / "hand.urdf")
    with pytest.raises(ValueError, match="Several frames"):
        model.model.getFrameId("tip")
    assert model.get_frame_pose("tip") == 2
    assert model.get_frames_index(["tip"]) == [2]
    with pytest.raises(ValueError, match="Unknown hand link"):
        model.get_frame_pose("missing")


def shadow_profile():
    import json

    return json.loads((ROOT / "tests/fixtures/vector-shadow-profile.json").read_text())


@pytest.mark.parametrize("side,prefix", [("right", "h_rh_"), ("left", "lh_h_lh_")])
def test_shadow_regularization_follows_source_names_not_joint_order(side, prefix):
    fixture = shadow_profile()
    order = ["THJ5", "FFJ2", "LFJ5", "FFJ4", "MFJ3", "LFJ1", "THJ4", "RFJ4"]
    joints = [prefix + name for name in order]
    manifest = dict(
        hand_key="shadow",
        source_names={
            f"{side}:{'rh_' if side == 'right' else 'lh_'}{name}": joint
            for name, joint in zip(order, joints)
        },
    )
    actual = runtime.finger_position_weights(
        manifest, {"finger_joints": joints}, fixture["robot"], fixture["profile"]
    )
    assert actual == [0.1, 0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.5]


def test_uncalibrated_hands_keep_flexion_free_of_a_neutral_pose_prior():
    _, hand = wuji()
    weights = runtime.finger_position_weights({"hand_key": "wuji-2"}, hand, {}, {})
    assert weights == [0.0] * len(hand["finger_joints"])
    config = configuration("vector-wrist-joint", "/test")
    assert config["adapter_version"] == 4
    assert "finger_position_weight" not in config
    assert config["finger_velocity_weight"] > 0


def test_shadow_regularization_rejects_incomplete_or_invalid_profiles():
    fixture = shadow_profile()
    manifest = dict(hand_key="shadow", source_names={"rh_FFJ3": "h_rh_FFJ3"})
    hand = {"finger_joints": ["h_rh_FFJ3"]}
    invalid = copy.deepcopy(fixture)
    invalid["profile"]["retargeting"]["joint_position_weights"].pop()
    with pytest.raises(ValueError, match="invalid joint map"):
        runtime.finger_position_weights(
            manifest, hand, invalid["robot"], invalid["profile"]
        )
    manifest["source_names"].clear()
    with pytest.raises(ValueError, match="Missing Shadow source joint"):
        runtime.finger_position_weights(
            manifest, hand, fixture["robot"], fixture["profile"]
        )


def test_temporal_tuning_only_overrides_the_measured_hand():
    config = configuration("vector-wrist-joint", "/test")
    assert runtime.finger_velocity_weight(config, {"hand_key": "wuji-2"}) == 0.001
    for hand in ("shadow", "wuji-1", "sharpa"):
        assert runtime.finger_velocity_weight(config, {"hand_key": hand}) == 0.01
    # Old sessions have no overrides and retain their frozen parameters.
    config.pop("hand_overrides")
    assert runtime.finger_velocity_weight(config, {"hand_key": "wuji-2"}) == 0.01
