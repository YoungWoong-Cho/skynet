import importlib.util
import json
from pathlib import Path
import pickle
import sys

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
