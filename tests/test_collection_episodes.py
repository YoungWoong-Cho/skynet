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


def test_manual_start_requires_a_click_and_does_not_reuse_stale_clicks():
    start = collection.ManualStart()
    assert not start.consume(True)
    start.request(start.attempt_id)
    assert start.consume(True)
    assert not start.consume(True)
    previous = start.attempt_id
    start.reset()
    start.request(previous)
    assert not start.consume(True)
    start.request(start.attempt_id)
    assert not start.consume(False)
    assert not start.consume(True)  # Tracking recovery cannot start it later.


def test_blue_points_show_actual_robot_joints_when_preview_is_visible():
    class Marker:
        def set_visibility(self, visible):
            self.visible = visible

        def visualize(self, translations):
            self.points = translations.copy()

    r = SimpleNamespace(_canonical_markers=Marker(), _wrist_markers=Marker())
    positions = np.array([[0.2, 0.3, 0.8], [0.4, 0.2, 0.85]])
    visible = False
    collection.install_robot_point_display(r, lambda: positions, lambda: visible)
    r._visualize_canonical_hand_keypoints({})
    assert not r._canonical_markers.visible and not r._wrist_markers.visible
    visible = True
    r._visualize_canonical_hand_keypoints({})
    np.testing.assert_array_equal(r._canonical_markers.points, positions)
    positions[1] += [0.02, 0.03, -0.01]
    r._visualize_canonical_hand_keypoints({})
    np.testing.assert_array_equal(r._canonical_markers.points, positions)
    visible = False
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
