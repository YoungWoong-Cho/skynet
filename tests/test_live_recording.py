import importlib.util
from pathlib import Path
import pickle

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "native_session", Path(__file__).resolve().parents[1] / "ops/xr/native_session.py"
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.fixture
def recording(tmp_path):
    payload = {
        "format": "dexverse_trajectory",
        "schema_version": 3,
        "task": "Dexverse-PickUpStick-v0",
        "robot_type": "floating_shadow_right",
        "num_episodes": 1,
        "episodes": [
            {
                "actions": np.zeros((3, 2)),
                "num_steps": 3,
                "states": [{"robot": {}} for _ in range(4)],
                "success": True,
            }
        ],
    }

    def validate(edit=lambda p: None):
        edit(payload)
        path = tmp_path / "recording.pkl"
        path.write_bytes(pickle.dumps(payload))
        return worker.inspect_recording(path)

    return validate


def test_successful_native_recording_has_counted_actions(recording):
    assert recording() == {"episodes": 1, "steps": 3, "success": True}


def test_empty_pickle_is_not_a_completed_capture(recording):
    with pytest.raises(ValueError, match="no complete demonstration"):
        recording(lambda p: p.update(num_episodes=0, episodes=[]))


def test_native_actions_require_matching_scene_states(recording):
    with pytest.raises(ValueError, match="one initial state"):
        recording(lambda p: p["episodes"][0]["states"].pop())


def test_nonfinite_native_actions_are_rejected(recording):
    with pytest.raises(ValueError, match="invalid action"):
        recording(lambda p: p["episodes"][0]["actions"].fill(float("nan")))


def test_unsuccessful_native_episode_is_not_successful_capture(recording):
    with pytest.raises(ValueError, match="success condition"):
        recording(lambda p: p["episodes"][0].update(success=False))
