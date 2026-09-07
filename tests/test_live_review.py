import json
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
from skynet_app.live_xr_review import LiveReviewService, inspect
from skynet_app.live_xr_catalog import selection


@pytest.fixture
def payload():
    states = [
        {
            "articulation": {"robot": {"joint_position": np.ones((1, 28)) * i}},
            "rigid_object": {
                "object": {
                    "root_pose": np.array([[0.0, 0.0, i * 0.01, 1.0, 0.0, 0.0, 0.0]])
                }
            },
            "deformable_object": {},
        }
        for i in range(4)
    ]
    return {
        "format": "dexverse_trajectory",
        "schema_version": 3,
        "task": "Dexverse-PickUpStick-v0",
        "robot_type": "floating_shadow_right",
        "num_episodes": 1,
        "episodes": [
            {
                "num_steps": 3,
                "success": True,
                "actions": np.arange(84, dtype=float).reshape(3, 28),
                "states": states,
            }
        ],
    }


@pytest.fixture
def review(tmp_path, payload):
    job = {
        "id": "session",
        "state": "CAPTURED",
        "gateway": "test-host",
        "root": "/workspace/sessions/session",
        "profile": {"task": payload["task"], "robot": payload["robot_type"]},
        "recordings": ["recordings/live/demo.pkl"],
    }
    remote = pickle.dumps(payload)
    calls = []
    transport = SimpleNamespace(
        file_size=lambda *args: ("test-host", len(remote)),
        stream_file_range=lambda *args, **kw: iter([remote]),
    )
    live = SimpleNamespace(
        root=tmp_path,
        get=lambda _: job,
        transport=lambda _: calls.append("download") or transport,
    )
    service = LiveReviewService(live)
    return service, job, calls


def test_review_download_is_cached_and_has_correct_state_action_alignment(review):
    service, job, calls = review
    directory = service.directory("session", 0)
    directory.mkdir(parents=True)
    service.prepare("session", 0)
    assert service.status("session")["state"] == "READY"
    result = json.loads(service.artifact("session", 0, "review.json").read_text())
    episode = result["episodes"][0]
    assert episode["frames"][0]["action"] is None
    assert episode["frames"][1]["action_index"] == 0
    assert episode["frames"][1]["action"][0] == 0
    assert episode["frames"][3]["action"][0] == 56
    assert episode["duration_seconds"] == 3 / 60
    assert not episode["sampled"]
    assert service.create("session")["state"] == "READY"
    assert calls == [
        "download"
    ]  # No scheduler, simulator, training or evaluation calls.


def test_review_rejects_nonfinite_scene_state(tmp_path, payload):
    payload["episodes"][0]["states"][2]["rigid_object"]["object"]["root_pose"].fill(
        np.nan
    )
    p = tmp_path / "bad.pkl"
    p.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="nonfinite"):
        inspect(p, {"task": payload["task"], "robot": payload["robot_type"]})


def test_pickle_cannot_execute_code(tmp_path, payload):
    class Dangerous:
        def __reduce__(self):
            return (eval, ("1 + 1",))

    p = tmp_path / "bad.pkl"
    p.write_bytes(pickle.dumps(Dangerous()))
    with pytest.raises(ValueError, match="Unsupported recording object"):
        inspect(p, {"task": payload["task"], "robot": payload["robot_type"]})


def test_session_mismatch_and_path_escape_are_rejected(tmp_path, payload, review):
    p = tmp_path / "demo.pkl"
    p.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="does not match"):
        inspect(p, {"task": payload["task"], "robot": "floating_shadow_left"})
    service, job, _ = review
    job["recordings"] = ["recordings/../../secret.pkl"]
    with pytest.raises(ValueError, match="Invalid saved"):
        service.create("session")


def test_interrupted_and_failed_downloads_can_retry(review):
    service, job, calls = review
    directory = service.directory("session", 0)
    service.publish(directory, state="DOWNLOADING")
    assert service.status("session")["state"] == "FAILED"
    job["state"] = "AWAITING_HEADSET"
    job["recordings"] = []
    with pytest.raises(ValueError, match="after a successful"):
        service.create("session")


def test_saved_episode_can_be_reviewed_while_collection_continues(review):
    service, job, _ = review
    job["state"] = "COLLECTING"
    assert service.source("session", 0)[0]["state"] == "COLLECTING"


def test_checksum_mismatch_does_not_publish_download(review):
    service, job, _ = review
    job["recording_summary"] = {"sha256": "0" * 64}
    directory = service.directory("session", 0)
    directory.mkdir(parents=True)
    service.prepare("session", 0)
    assert service.status("session")["state"] == "FAILED"
    assert "checksum" in service.status("session")["error"]
    assert not (directory / "recording.pkl").exists()


def test_catalog_rejects_unknown_or_model_only_hands():
    for hand in ["wuji-1", "sharpa", "missing", ""]:
        with pytest.raises(ValueError, match="Unsupported live hand"):
            selection("Dexverse-PickUpStick-v0", hand)
    with pytest.raises(ValueError, match="Unsupported live task"):
        selection("Not-A-Task", "floating_shadow_right")
    assert selection("Dexverse-PickCube-v0", "floating_shadow_left")[1]["available"]
