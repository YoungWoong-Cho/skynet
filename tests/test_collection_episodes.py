import importlib.util
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from enum import Enum

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/xr/hands"))
from anatomy import canonical_points, palm_frame  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "collection", ROOT / "ops/xr/collection.py"
)
collection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collection)


def human(side="right"):
    points = np.zeros((21, 3))
    for finger, y in enumerate([0.075, 0.03, 0, -0.025, -0.05]):
        for joint in range(4):
            points[1 + finger * 4 + joint] = [0.065 + joint * 0.03, y, 0]
    if side == "left":
        points[:, 1] *= -1
    return points


@pytest.mark.parametrize("side", ["right", "left"])
def test_finger_targets_do_not_inherit_wrist_axis_yaw_or_world_rotation(side):
    p = human(side)
    rotation, _ = np.linalg.qr(
        np.array([[1.0, 2.0, 4.0], [3.0, -2.0, 1.0], [2.0, 3.0, -1.0]])
    )
    rotation[:, 0] *= np.linalg.det(rotation)
    rotated = p @ rotation.T + [0.4, -0.3, 0.8]
    np.testing.assert_allclose(canonical_points(rotated, side), p, atol=1e-7)
    np.testing.assert_allclose(palm_frame(p, side), np.eye(3), atol=1e-7)
    # Real middle-finger bending must remain visible, rather than reorienting the frame.
    bent = p.copy()
    bent[10:13, 2] -= [0.01, 0.03, 0.06]
    np.testing.assert_allclose(canonical_points(bent, side), bent, atol=1e-7)


def test_missing_or_collapsed_tracking_cannot_be_aligned():
    with pytest.raises(ValueError):
        canonical_points(np.zeros((21, 3)), "right")
    p = human()
    assert collection.aligned_hand(p, "right", np.zeros(3))
    assert not collection.aligned_hand(p + [0.2, 0, 0], "right", np.zeros(3))
    c, s = np.cos(np.deg2rad(40)), np.sin(np.deg2rad(40))
    yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    assert not collection.aligned_hand(p @ yaw.T, "right", np.zeros(3))
    p[8] = p[5] + [0.005, 0, -0.07]
    assert not collection.aligned_hand(p, "right", np.zeros(3))


def test_alignment_must_remain_continuous_before_auto_start():
    gate = collection.AlignmentGate()
    assert gate.update(True, 0) == 0
    assert gate.update(True, 0.4) == 0.5
    assert gate.update(False, 0.5) == 0
    assert gate.update(True, 0.6) == 0
    assert gate.update(True, 1.5) == 1


def test_finger_calibration_removes_only_spread_bias_and_recalibrates():
    names = [
        "wrist",
        "index_flex",
        "index_mcp_abd",
        "middle_mcp_abd",
        "ring_mcp_abd",
        "pinky_mcp_abd",
        "thumb_flex",
    ]
    calibration = collection.FingerNeutralCalibration(
        dict(
            hand_key="wuji-2",
            wrist_joints=names[:1],
            finger_joints=names[1:],
            neutral={n: 0 for n in names},
            finger_limits={n: [-0.7, 0.7] for n in names},
        )
    )
    initial = np.array([0.0, 0.4, 0.27, 0.24, 0.10, 0.19, 0.6])
    receipt = calibration.capture(initial)
    corrected = initial - calibration.offset
    np.testing.assert_allclose(corrected, [0, 0.4, 0, 0, 0, 0, 0.6], atol=1e-7)
    assert len(receipt["joint_offsets"]) == 4
    moved = initial.copy()
    moved[2] += 0.1
    moved[1] += 0.2
    np.testing.assert_allclose(
        (moved - calibration.offset)[[1, 2]], [0.6, 0.1], atol=1e-7
    )
    calibration.capture(moved)
    np.testing.assert_allclose(
        (moved - calibration.offset)[calibration.indices], 0, atol=1e-7
    )
    with pytest.raises(ValueError, match="invalid"):
        calibration.capture([np.nan] * len(names))


def test_alignment_explains_the_unmatched_pose_and_allows_size_differences():
    p = human()
    matches, hint, details = collection.alignment_feedback(
        p + [0.25, 0, 0], "right", np.zeros(3)
    )
    assert not matches and "25 cm away" in hint and details["distance_cm"] == 25
    assert collection.aligned_hand(p + [0.10, 0, 0], "right", np.zeros(3))
    tilted = p @ np.diag([1, -1, -1])
    assert "palm down" in collection.alignment_feedback(tilted, "right", np.zeros(3))[1]
    sideways = p @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert (
        "same direction"
        in collection.alignment_feedback(sideways, "right", np.zeros(3))[1]
    )


def test_control_points_rotate_and_translate_in_the_robot_control_frame():
    p = human()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = collection.control_points(
        p, rotation, [1, 2, 3], [0.9, 2.2, 3.3], [0, 0, 0.8]
    )
    np.testing.assert_allclose(result[0], [0.1, -0.2, 0.5])
    np.testing.assert_allclose(result[12] - result[0], rotation @ p[12])


def test_blue_overlay_is_hidden_until_recording_and_uses_relative_wrist_rotation():
    class Rotation:
        def __init__(self, matrix):
            self.matrix = matrix

        def inv(self):
            return Rotation(self.matrix.T)

        def __mul__(self, other):
            return Rotation(self.matrix @ other.matrix)

        def as_matrix(self):
            return self.matrix

    class Hand(Enum):
        HAND_RIGHT = 1

    class Marker:
        def set_visibility(self, visible):
            self.visible = visible

        def visualize(self, translations):
            self.points = translations

    hand = Hand.HAND_RIGHT
    turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r = SimpleNamespace(
        _canonical_markers=Marker(),
        _wrist_markers=Marker(),
        latest_wrist_poses={hand: np.r_[[0.2, 0.3, 0.4], [0, 0, 0, 1]]},
        retarget_base_wrist_poses={hand: np.r_[[0.1, 0.2, 0.3], [1, 0, 0, 0]]},
        _convert_hand_to_canonical_joint_positions=lambda data, hand: human(),
        _get_normalized_wrist_rotation=lambda q: Rotation(
            base if q[0] == 1 else turn @ base
        ),
    )
    recording = False
    collection.install_control_point_display(
        r, {"right": np.array([0, 0, 0.8])}, lambda: recording
    )
    r._visualize_canonical_hand_keypoints({hand: {}})
    assert not r._canonical_markers.visible and not r._wrist_markers.visible
    recording = True
    r._visualize_canonical_hand_keypoints({hand: {}})
    assert r._canonical_markers.visible
    np.testing.assert_allclose(
        r._canonical_markers.points,
        human() @ turn.T + [0.1, 0.1, 0.9],
        atol=1e-6,
    )
    recording = False
    r._visualize_canonical_hand_keypoints({hand: {}})
    assert not r._canonical_markers.visible


def test_multiple_episodes_are_saved_independently_and_never_overwritten(tmp_path):
    metadata = dict(
        format="dexverse_trajectory", schema_version=3, task="task", robot_type="hand"
    )
    store = collection.EpisodeStore(tmp_path, metadata)
    episode = dict(
        actions=np.zeros((3, 2)),
        states=[{"robot": [0.0]}] * 4,
        success=True,
        num_steps=3,
    )
    first = store.save(episode)
    original = (tmp_path / first["path"]).read_bytes()
    second = store.save(dict(episode, actions=np.ones((3, 2))))
    assert first["path"] != second["path"]
    assert (tmp_path / first["path"]).read_bytes() == original
    assert len(json.loads((tmp_path / "episodes.json").read_text())) == 2
    assert pickle.loads(original)["num_episodes"] == 1
    with pytest.raises(ValueError):
        store.save(dict(episode, success=False))
    with pytest.raises(ValueError):
        store.save(dict(episode, states=[{"robot": [float("nan")]}] * 4))
    assert len(store.receipts) == 2
    restarted = collection.EpisodeStore(tmp_path, metadata)
    with pytest.raises(FileExistsError):
        restarted.save(episode)
    assert (tmp_path / first["path"]).read_bytes() == original
