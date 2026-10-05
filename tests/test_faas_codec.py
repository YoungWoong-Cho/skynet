"""Asset-bound FAAS codec shared by HAT recording preparation and rollout."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ops.datasets.action_codecs.geometry import inverse_transform, transform9
from ops.datasets.action_codecs.unidex import (
    anchor_faas_actions, encode_recording, load_codec, supported_robots, validate_source,
)
from skynet_app.adapters.recording_dataset import canonical


@pytest.mark.parametrize("robot", supported_robots())
def test_exact_hand_codec_roundtrip_fk_and_root_camera_frames(robot):
    codec = load_codec(robot)
    rng = np.random.default_rng(9)
    lower, upper = codec.spec.effective_lower.copy(), codec.spec.effective_upper.copy()
    lower[codec.wrist_indices[3:]], upper[codec.wrist_indices[3:]] = -0.5, 0.5
    q = lower + (upper - lower) * rng.uniform(0.25, 0.75, (4, codec.action_dim))
    action = (q - codec.action_offset) / codec.action_scale
    encoded, _ = codec.encode_actions(action, q[0])
    decoded = codec.decode_actions(encoded, q[0])
    np.testing.assert_allclose(decoded, action, atol=1e-7)
    poses = codec.spec.forward_kinematics(q[0])
    expected_wrist = poses[codec.spec.frame_link] @ np.asarray(codec.spec.data["link_to_frame"])
    np.testing.assert_allclose(codec.wrist_pose(q[0]), expected_wrist, atol=1e-9)
    capture = copy.deepcopy(codec.spec.capture)
    capture["robot_joint_names"] = list(codec.action_names)
    roots = np.tile([0.2, 0.3, 0.4, 1., 0., 0., 0.], (5, 1))
    episode = {"actions": action, "states": [{"articulation": {"robot": {
        "joint_position": q[min(i, 3)][None], "root_pose": roots[i][None]}}} for i in range(5)]}
    validate_source(episode, capture)
    camera = np.repeat(np.eye(4)[None], 4, axis=0)
    camera[:, 0, 3] = -0.5
    values, metadata = encode_recording(episode, capture, camera)
    assert metadata["frame"] == "camera_opengl"
    world_root = np.eye(4)
    world_root[:3, 3] = roots[0, :3]
    expected = np.diag([1., -1., -1., 1.]) @ inverse_transform(camera[0]) @ world_root @ expected_wrist
    np.testing.assert_allclose(transform9(values["faas_state_absolute"][0, :9]), expected, atol=1e-6)
    # Root translation cancels only when both current state and command share it.
    anchored = anchor_faas_actions(values["faas_state_absolute"][0], values["faas_action_absolute"])
    np.testing.assert_allclose(anchored, encoded, atol=2e-6)
    bad = copy.deepcopy(episode)
    bad["states"][0]["articulation"]["robot"]["root_pose"][0, 3] = 0
    with pytest.raises(ValueError, match="quaternion"):
        validate_source(bad, capture)


def test_unregistered_mapping_fails_before_render():
    with pytest.raises(ValueError, match="No exact asset-bound FAAS contract"):
        load_codec("unregistered_hand_right")


def test_unverified_mapping_still_fails_before_render(tmp_path):
    import hashlib
    import shutil

    from ops.datasets.action_codecs.hand_contract import DEFAULT_SPEC_ROOT
    spec_root = tmp_path / "specs"
    shutil.copytree(DEFAULT_SPEC_ROOT, spec_root)
    path = spec_root / "skynet_sharpa_right.json"
    spec = json.loads(path.read_text())
    spec["mapping_status"] = "BLOCKED"
    spec["blocked_reason"] = "Deliberately unverified test asset"
    spec["spec_hash"] = hashlib.sha256(canonical({k: v for k, v in spec.items() if k != "spec_hash"})).hexdigest()
    path.write_bytes(canonical(spec))
    with pytest.raises(ValueError, match="Deliberately unverified test asset"):
        load_codec("skynet_sharpa_right", spec_root=spec_root)


def test_codec_capsule_resolves_static_contracts_from_changed_working_directory(tmp_path):
    import shutil
    import subprocess
    import sys
    repo = Path(__file__).parents[1]
    capsule = tmp_path / "capsule"
    shutil.copytree(repo / "ops/datasets/action_codecs", capsule / "action_codecs", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(repo / "config/action_representations/unidex-faas-v1", capsule / "action_codecs/specs")
    script = capsule / "verify.py"
    script.write_text("from action_codecs.unidex import load_codec, supported_robots\n"
                      "assert all(load_codec(robot).action_dim > 0 for robot in supported_robots())\n")
    subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True)


def test_recorded_position_commands_are_not_clipped_to_physical_joint_limits():
    codec = load_codec("floating_shadow_right")
    q = (codec.spec.effective_lower + codec.spec.effective_upper) / 2
    q[codec.wrist_indices] = 0
    mapping = codec.independent[0]
    joint = codec.action_names.index(mapping["joint"])
    target = q.copy()
    target[joint] = codec.spec.lower[joint] - 0.001
    actions = ((target - codec.action_offset) / codec.action_scale)[None]
    capture = copy.deepcopy(codec.spec.capture)
    capture["robot_joint_names"] = list(codec.action_names)
    state = {"articulation": {"robot": {"joint_position": q[None],
             "root_pose": np.array([[0., 0., 0., 1., 0., 0., 0.]])}}}
    episode = {"actions": actions, "states": [state, copy.deepcopy(state)]}
    validate_source(episode, capture)
    values, _ = encode_recording(episode, capture, np.eye(4)[None])
    expected = target[joint] * mapping["scale"] + mapping["offset"]
    assert values["faas_action_absolute"][0,18+mapping["slot"]] == pytest.approx(expected)
    encoded, _ = codec.encode_actions(actions, q)
    np.testing.assert_allclose(codec.decode_actions(encoded, q), actions, atol=1e-9)
    bad = copy.deepcopy(episode)
    bad["actions"][0,joint] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_source(bad, capture)
