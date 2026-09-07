"""Rotation trajectories must stay continuous, not just reach the right endpoint."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops/xr"))
from wrist import ContinuousEulerXYZ  # noqa: E402


def rotation(angles):
    x, y, z = angles
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rx @ ry @ rz


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("direction", [-1, 1])
def test_two_turns_and_return_cross_all_euler_boundaries(axis, direction):
    tracker = ContinuousEulerXYZ()
    target = np.zeros(3)
    path = np.r_[
        np.linspace(0, direction * 4 * np.pi, 721),
        np.linspace(direction * 4 * np.pi, 0, 721),
    ]
    outputs = []
    for angle in path:
        target[axis] = angle
        result = tracker.update(rotation(target))
        np.testing.assert_allclose(rotation(result), rotation(target), atol=1e-7)
        np.testing.assert_allclose(result, target, atol=1e-7)
        outputs.append(result)
    assert np.max(np.abs(np.diff(outputs, axis=0))) < np.deg2rad(2)


@pytest.mark.parametrize("direction", [-1, 1])
def test_palm_roll_with_tilt_and_heading_does_not_spin_back(direction):
    tracker = ContinuousEulerXYZ()
    previous = None
    for t in np.linspace(0, 1, 1001):
        expected = [direction * 3 * np.pi * t, 0.35 * np.sin(2 * np.pi * t), 0.65 * t]
        actual = tracker.update(rotation(expected))
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        if previous is not None:
            assert np.max(np.abs(actual - previous)) < 0.02
        previous = actual


@pytest.mark.parametrize("pitch", [-np.pi / 2, np.pi / 2])
def test_gimbal_lock_preserves_roll_yaw_history(pitch):
    tracker = ContinuousEulerXYZ()
    for t in np.linspace(0, 1, 91):
        angles = [0.6 * t, pitch * t, -0.4 * t]
        result = tracker.update(rotation(angles))
        np.testing.assert_allclose(rotation(result), rotation(angles), atol=1e-7)
    before = result.copy()
    for _ in range(20):
        result = tracker.update(rotation(angles))
        np.testing.assert_allclose(result, before, atol=1e-7)
    assert np.max(np.abs(result - angles)) < 0.01


def test_episode_reset_drops_old_turn_count_and_invalid_tracking_does_not_change_state():
    tracker = ContinuousEulerXYZ()
    for x in np.linspace(0, 4 * np.pi, 100):
        tracker.update(rotation([x, 0, 0]))
    for invalid in (
        np.zeros((3, 3)),
        np.eye(2),
        np.full((3, 3), np.nan),
        np.diag([-1, 1, 1]),
    ):
        before = tracker.previous.copy()
        with pytest.raises(ValueError, match="invalid rotation"):
            tracker.update(invalid)
        np.testing.assert_array_equal(tracker.previous, before)
    tracker.reset()
    np.testing.assert_allclose(tracker.update(np.eye(3)), np.zeros(3))
