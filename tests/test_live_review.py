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


def test_capture_video_download_is_verified_cached_and_never_uses_gpu(review):
    from skynet_app.live_xr_video import LiveVideoService
    import hashlib

    reviews, job, calls = review
    reviews.directory("session", 0).mkdir(parents=True)
    reviews.prepare("session", 0)
    path = reviews.artifact("session", 0, "review.json")
    document = json.loads(path.read_text())
    raw = b"\x00\x00\x00\x18ftypisom" + b"video payload"
    metadata = dict(
        state="READY",
        path="recordings/live/demo.mp4",
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        fps=30,
        frames=2,
    )
    document["episodes"][0]["video"] = metadata
    path.write_text(json.dumps(document))
    videos = LiveVideoService(reviews)
    reviews.live.transport = lambda _: SimpleNamespace(
        file_size=lambda *a: ("test-host", len(raw)),
        stream_file_range=lambda *a, **k: iter([raw[:5], raw[5:]]),
    )
    videos.render = lambda *a: pytest.fail(
        "A captured video must not launch a GPU render"
    )
    videos.prepare("session", 0, 0)
    assert videos.status("session", 0)["state"] == "READY"
    assert videos.status("session", 0)["kind"] == "capture"
    assert videos.artifact("session", 0).read_bytes() == raw
    reviews.live.transport = lambda _: pytest.fail("Cached video must work offline")
    assert videos.create("session", 0)["state"] == "READY"
    metadata["path"] = "recordings/../../secret.mp4"
    with pytest.raises(ValueError, match="does not match"):
        videos.capture_source(job, 0, metadata)
    with pytest.raises(KeyError, match="Demonstration"):
        videos.create("session", 0, 1)


def test_video_failure_and_interrupted_preparation_are_explicit_and_retryable(review):
    from skynet_app.live_xr_video import LiveVideoService

    reviews, job, calls = review
    reviews.directory("session", 0).mkdir(parents=True)
    reviews.prepare("session", 0)
    videos = LiveVideoService(reviews)
    directory = videos.source("session", 0, 0)[3]
    videos.publish(directory, state="PREPARING")
    assert videos.status("session", 0)["state"] == "NOT_PREPARED"
    videos.render = lambda *a: {
        "state": "FAILED",
        "error": "GPU is busy with live collection",
    }
    videos.prepare("session", 0, 0)
    assert "GPU is busy" in videos.status("session", 0)["error"]
    assert not (directory / "video.mp4").exists()
    queued = []
    videos.executor.submit = lambda *a: queued.append(a)
    assert videos.create("session", 0)["state"] == "QUEUED"
    assert videos.create("session", 0)["state"] == "QUEUED"
    assert "queue position 1" in videos.status("session", 0)["detail"]
    assert len(queued) == 1

    # A recording removed after queueing must not permanently block retries.
    reviews.source = lambda *a: (_ for _ in ()).throw(KeyError("Recording removed"))
    videos.prepare("session", 0, 0)
    assert not videos.active
    assert not videos.queue


def test_video_recovers_legacy_busy_errors_and_interrupted_jobs(review):
    from skynet_app.live_xr_video import LiveVideoService

    reviews, _, _ = review
    reviews.directory("session", 0).mkdir(parents=True)
    reviews.prepare("session", 0)
    videos = LiveVideoService(reviews)
    directory = videos.source("session", 0, 0)[3]
    for message in (
        "rl2-bonjour: End the live session before preparing video for this older recording.",
        "rl2-bonjour: The GPU is busy with another session or video job. Retry when it finishes.",
        "Video preparation was interrupted. Retry to continue.",
    ):
        videos.publish(directory, state="FAILED", error=message)
        assert videos.status("session", 0)["state"] == "NOT_PREPARED"
    for state in ("QUEUED", "PREPARING", "WAITING_GPU"):
        videos.publish(directory, state=state)
        assert videos.status("session", 0)["state"] == "NOT_PREPARED"
    videos.publish(directory, state="READY")
    assert videos.status("session", 0)["state"] == "NOT_PREPARED"


def test_video_waits_for_gpu_then_finishes_or_reports_bounded_failure(
    review, monkeypatch
):
    from skynet_app import live_xr_video

    reviews, _, _ = review
    reviews.directory("session", 0).mkdir(parents=True)
    reviews.prepare("session", 0)
    videos = live_xr_video.LiveVideoService(reviews)
    videos.version = "test-version"
    directory = videos.source("session", 0, 0)[3]
    key = ("session", 0, 0)
    ready = dict(
        state="READY",
        kind="replay",
        path="/video.mp4",
        fps=30,
        frames=2,
        size_bytes=24,
        sha256="a" * 64,
        renderer_version=videos.version,
    )
    renders = iter([{"state": "WAITING_GPU"}, ready])
    videos.render = lambda *a: next(renders)
    videos.download = lambda *a: (directory / "video.mp4").write_bytes(b"cached video")
    waits = []
    monkeypatch.setattr(
        live_xr_video.time, "sleep", lambda seconds: waits.append(videos.status(*key))
    )
    videos.active.add(key)
    videos.prepare(*key)
    assert waits[0]["state"] == "WAITING_GPU"
    assert videos.status(*key)["state"] == "READY"
    assert not videos.active

    (directory / "video.mp4").unlink()
    videos.render = lambda *a: {"state": "WAITING_GPU"}
    clock = iter([0, live_xr_video.GPU_WAIT_SECONDS + 1])
    monkeypatch.setattr(live_xr_video.time, "monotonic", lambda: next(clock))
    videos.prepare(*key)
    result = videos.status(*key)
    assert result["state"] == "FAILED"
    assert "GPU is still busy" in result["error"]
    assert not videos.active


def test_video_remote_lock_conflict_is_a_wait_state(review):
    from skynet_app.live_xr_video import LiveVideoService

    reviews, job, _ = review
    job["profile"].update(
        execution="workstation",
        runtime="/runtime",
        repository="/repo",
        work_root="/work",
    )
    videos = LiveVideoService(reviews)
    videos.sources = {}
    calls = []

    def ssh(*args, **kwargs):
        calls.append(args[1])
        return "{}" if len(calls) == 1 else '{"state":"WAITING_GPU"}'

    result = videos.render(
        SimpleNamespace(ssh=ssh), job, "/recording.pkl", {"sha256": "a" * 64}, 0
    )
    assert result["state"] == "WAITING_GPU"
    assert "--conflict-exit-code=75" in calls[1]
    assert '\'{"state":"WAITING_GPU"}\'; exit 0' in calls[1]


def test_video_checksum_or_incomplete_download_cannot_be_published(review):
    from skynet_app.live_xr_video import LiveVideoService
    import hashlib

    reviews, _, _ = review
    reviews.directory("session", 0).mkdir(parents=True)
    reviews.prepare("session", 0)
    videos = LiveVideoService(reviews)
    directory = videos.source("session", 0, 0)[3]
    directory.mkdir()
    raw = b"\x00\x00\x00\x18ftypisom"
    metadata = dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    for received in [raw[:-1], b"bad bytes!!!"]:
        transport = SimpleNamespace(
            file_size=lambda *a: ("host", len(raw)),
            stream_file_range=lambda *a, **k: iter([received]),
        )
        with pytest.raises(ValueError, match="incomplete|checksum"):
            videos.download(transport, "host", "/video.mp4", metadata, directory)
        assert not (directory / "video.mp4").exists()
        assert not (directory / "download.part").exists()


def test_browser_video_supports_byte_ranges(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from skynet_app import live_xr_api

    p = tmp_path / "video.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypisom" + bytes(range(128)))
    monkeypatch.setattr(live_xr_api, "videos", SimpleNamespace(artifact=lambda *a: p))
    app = FastAPI()
    app.include_router(live_xr_api.router)
    response = TestClient(app).get(
        "/api/collection/live/sessions/session/recordings/0/video.mp4",
        headers={"Range": "bytes=0-11"},
    )
    assert response.status_code == 206
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["content-range"] == f"bytes 0-11/{p.stat().st_size}"
    assert response.content == p.read_bytes()[:12]
