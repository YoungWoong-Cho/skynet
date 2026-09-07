import importlib.util
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/xr/hands"))
sys.path.insert(0, str(ROOT / "ops/xr"))
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


def test_alignment_explains_the_unmatched_pose_and_allows_size_differences():
    p = human()
    matches, hint, details = collection.alignment_feedback(
        p + [0.25, 0, 0], "right", np.zeros(3)
    )
    assert not matches and "25 cm away" in hint and details["distance_cm"] == 25
    assert collection.aligned_hand(p + [0.02, 0, 0], "right", np.zeros(3))
    assert not collection.aligned_hand(p + [0.03, 0, 0], "right", np.zeros(3))
    tilted = p @ np.diag([1, -1, -1])
    assert "palm down" in collection.alignment_feedback(tilted, "right", np.zeros(3))[1]
    sideways = p @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert (
        "same direction"
        in collection.alignment_feedback(sideways, "right", np.zeros(3))[1]
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_alignment_targets_match_real_anchors_for_different_palm_sizes(side):
    from alignment import alignment_points

    target = np.array([0.4, -0.2, 0.85])
    for scale in (0.7, 1.5):
        p = human(side) * scale + target
        tracked, rings, errors = alignment_points(p, side, target)
        np.testing.assert_allclose(tracked, p[[0, 5, 17]])
        np.testing.assert_allclose(tracked, rings)
        np.testing.assert_allclose(errors, 0, atol=1e-12)
        assert collection.aligned_hand(p, side, target)
        # Turning/translation moves the dots, never drags the target with them.
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        moved = (p - target) @ rotation.T + target + [0.2, 0.1, 0]
        _, new_rings, errors = alignment_points(moved, side, target)
        np.testing.assert_allclose(new_rings, rings)
        assert np.max(errors) > 0.1


def test_alignment_guide_hides_untracked_dots_and_clears_when_recording():
    from alignment import AlignmentGuide

    class Marker:
        def set_visibility(self, visible):
            self.visible = visible

        def visualize(self, translations, marker_indices):
            self.positions, self.indices = translations, marker_indices

    guide = AlignmentGuide.__new__(AlignmentGuide)
    guide.markers = Marker()
    guide.sides, guide.last_targets, guide.visible = ["right"], {}, False
    target = {"right": np.zeros(3)}
    guide.update(None, target, True)
    assert guide.markers.indices == [1, 1, 1]
    guide.update({"right": human()}, target, True)
    assert guide.markers.indices == [0, 0, 0, 2, 2, 2]
    guide.update(None, target, True)
    assert guide.markers.indices == [1, 1, 1]
    guide.update({"right": np.zeros((21, 3))}, target, True)
    assert guide.markers.indices == [1, 1, 1]
    guide.update(None, target, False)
    assert not guide.markers.visible


def test_blue_points_show_actual_robot_joints_and_hide_outside_recording():
    class Marker:
        def set_visibility(self, visible):
            self.visible = visible

        def visualize(self, translations):
            self.points = translations.copy()

    r = SimpleNamespace(_canonical_markers=Marker(), _wrist_markers=Marker())
    positions = np.array([[0.2, 0.3, 0.8], [0.4, 0.2, 0.85]])
    recording = False
    collection.install_robot_point_display(r, lambda: positions, lambda: recording)
    r._visualize_canonical_hand_keypoints({})
    assert not r._canonical_markers.visible and not r._wrist_markers.visible
    recording = True
    r._visualize_canonical_hand_keypoints({})
    np.testing.assert_array_equal(r._canonical_markers.points, positions)
    positions[1] += [0.02, 0.03, -0.01]
    r._visualize_canonical_hand_keypoints({})
    np.testing.assert_array_equal(r._canonical_markers.points, positions)
    recording = False
    r._visualize_canonical_hand_keypoints({})
    assert not r._canonical_markers.visible


def test_segment_targets_preserve_direction_without_matching_human_bone_lengths():
    from anatomy import segment_targets

    p = human()
    indices = np.array([[5, 6, 7], [6, 7, 8]])
    lengths = np.array([0.04, 0.03, 0.02])
    a = segment_targets(p, indices, lengths)
    b = segment_targets(p * 1.7, indices, lengths)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), lengths)
    p[7] = p[6] + [0.02, 0, -0.02]
    p[8] = p[7] + [0, 0, -0.02]
    c = segment_targets(p, indices, lengths)
    np.testing.assert_allclose(c[2], [0, 0, -0.02])
    p[8] = p[7]
    with pytest.raises(ValueError, match="invalid"):
        segment_targets(p, indices, lengths)


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
