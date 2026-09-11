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


def test_cancelled_video_queue_task_cannot_run_after_a_retry(review):
    from skynet_app.live_xr_video import LiveVideoService

    reviews, _, _ = review
    reviews.directory("session", 0).mkdir(parents=True)
    reviews.prepare("session", 0)
    videos = LiveVideoService(reviews)
    callbacks = []
    videos.executor.submit = lambda *args: callbacks.append(args)
    videos.create("session", 0)
    assert videos.cancel("session", 0)["state"] == "CANCELLED"
    assert videos.cancel("session", 0)["state"] == "CANCELLED"
    assert not videos.active and not videos.queue
    assert videos.create("session", 0)["state"] == "QUEUED"
    # The cancelled callback can be dequeued after a retry of the same recording.
    # Its old generation must not consume the replacement task or use the GPU.
    videos.render = lambda *args: pytest.fail("Cancelled callback launched remote work")
    callbacks[0][0](*callbacks[0][1:])
    assert videos.status("session", 0)["state"] == "QUEUED"
    assert len(videos.active) == len(videos.queue) == 1
    videos.cancel("session", 0)
    videos.executor.shutdown()


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
    generation = live_xr_video.VideoGeneration()
    monkeypatch.setattr(generation.cancel, "wait", lambda seconds: waits.append(videos.status(*key)))
    videos.generations[key] = generation
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
        calls.append((args[1], json.loads(kwargs["stdin"])))
        return '{"state":"PREPARING"}' if len(calls) == 1 else '{"state":"WAITING_GPU"}'

    result = videos.render(
        SimpleNamespace(ssh=ssh), job, "/recording.pkl", {"sha256": "a" * 64}, 0
    )
    assert result["state"] == "WAITING_GPU"
    assert "--conflict-exit-code=75" in calls[0][0]
    assert [call[1]["operation"] for call in calls] == ["start", "status"]
    assert calls[0][1]["generation"] == calls[1][1]["generation"]


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


@pytest.fixture
def videos(review):
    from skynet_app.live_xr_video import LiveVideoService

    reviews, job, _ = review
    reviews.directory("session", 0).mkdir(parents=True)
    reviews.prepare("session", 0)
    job["profile"].update(execution="workstation", runtime="/runtime", repository="/repo", work_root="/work")
    service = LiveVideoService(reviews)
    service.sources = {}
    yield service
    service.executor.shutdown(wait=True)


def wait_video(videos, state):
    import time
    import threading

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = videos.status("session", 0)
        if result["state"] == state:
            return result
        threading.Event().wait(0.01)
    pytest.fail(f"Expected {state}, got {result}")


def test_active_gpu_wait_cancel_is_prompt_and_stale_cancel_cannot_stop_retry(videos):
    import time

    videos.render = lambda *args: {"state": "WAITING_GPU"}
    first = videos.create("session", 0)
    wait_video(videos, "WAITING_GPU")
    start = time.monotonic()
    videos.cancel("session", 0, expected_generation=first["generation"])
    wait_video(videos, "CANCELLED")
    assert time.monotonic() - start < 1
    assert not videos.active and not videos.queue
    videos.executor.submit = lambda *args: None
    second = videos.create("session", 0)
    assert second["generation"] != first["generation"]
    assert videos.cancel("session", 0, expected_generation=first["generation"])["state"] == "QUEUED"
    assert videos.cancel("session", 0, expected_generation=second["generation"])["state"] == "CANCELLED"


def test_active_download_cancel_closes_stream_and_removes_partial(videos):
    import hashlib
    import threading

    directory = videos.source("session", 0, 0)[3]
    review_path = directory.parent / "review.json"
    review = json.loads(review_path.read_text())
    raw = b"\x00\x00\x00\x18ftypisom" + b"test payload"
    review["episodes"][0]["video"] = dict(state="READY", path="recordings/live/demo.mp4",
        size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), fps=30, frames=1)
    review_path.write_text(json.dumps(review))
    entered, closed = threading.Event(), threading.Event()

    def stream(*args, cancel_event, **kwargs):
        try:
            yield raw[:12]
            entered.set()
            assert cancel_event.wait(2)
        finally:
            closed.set()

    videos.live.transport = lambda _: SimpleNamespace(
        file_size=lambda *a: ("test-host", len(raw)), stream_file_range=stream)
    videos.create("session", 0)
    assert entered.wait(2)
    assert list(directory.glob("download.*.part"))
    videos.cancel("session", 0)
    wait_video(videos, "CANCELLED")
    assert closed.is_set()
    assert not list(directory.glob("*.part"))
    assert not (directory / "video.mp4").exists()


def test_cancel_during_remote_launch_cannot_publish_late_ready(videos):
    import threading

    launched, release = threading.Event(), threading.Event()
    calls = []

    def control(transport, job, generation, root, operation, **kwargs):
        assert not videos.lock._is_owned(), "Network operations must not hold the state lock"
        calls.append((operation, generation.token, root))
        if operation == "start":
            launched.set()
            assert release.wait(2)
            return {"state": "READY", "path": root + "/video.mp4"}
        assert operation == "cancel"
        return {"state": "CANCELLED"}

    videos._control = control
    videos.create("session", 0)
    assert launched.wait(2)
    assert videos.cancel("session", 0)["state"] == "CANCELLING"
    assert videos.cancel("session", 0)["state"] == "CANCELLING"
    assert [call[0] for call in calls].count("cancel") == 1
    release.set()
    wait_video(videos, "CANCELLED")
    assert len({call[1] for call in calls}) == 1
    assert not (videos.source("session", 0, 0)[3] / "video.mp4").exists()


def test_failed_remote_stop_remains_cancelling_and_can_be_retried_after_restart(videos):
    from skynet_app.live_xr_video import VideoGeneration

    key = ("session", 0, 0)
    job, _, review, directory = videos.source(*key)
    generation = VideoGeneration(finished=True)
    root = job["root"] + f"/output/review-videos/{review['sha256']}/0/{videos.version}/attempts/{generation.token}"
    videos.publish(directory, state="PREPARING", generation=generation.token, remote_root=root)
    assert videos.status(*key)["state"] == "INTERRUPTED"
    assert videos.create(*key)["state"] == "INTERRUPTED"
    calls = []

    def control(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("SSH connection lost")
        return {"state": "CANCELLED"}

    videos._control = control
    failed = videos.cancel(*key)
    assert failed["state"] == "CANCELLING" and failed["can_cancel"]
    assert "SSH connection lost" in failed["error"]
    assert key in videos.active
    assert videos.cancel(*key)["state"] == "CANCELLED"
    assert videos.cancel(*key)["state"] == "CANCELLED"
    assert len(calls) == 2


def test_cancel_between_verified_download_and_ready_removes_own_local_copy(videos):
    import threading

    downloaded, release = threading.Event(), threading.Event()
    directory = videos.source("session", 0, 0)[3]
    videos.render = lambda *a: dict(state="READY", kind="replay", path="/video.mp4")

    def download(*args):
        generation = args[-2]
        (directory / "video.mp4").write_bytes(b"verified video")
        generation.local_published = True
        downloaded.set()
        assert release.wait(2)

    videos.download = download
    videos.create("session", 0)
    assert downloaded.wait(2)
    assert videos.cancel("session", 0)["state"] == "CANCELLING"
    release.set()
    wait_video(videos, "CANCELLED")
    assert not (directory / "video.mp4").exists()


@pytest.fixture
def video_supervisor(tmp_path, monkeypatch):
    import subprocess
    from skynet_app.live_xr_video_worker import control

    token = "a" * 32
    root = tmp_path / "attempts" / token
    state = dict(LoadState="not-found", ActiveState="inactive", SubState="dead", ExecMainStatus="0",
                 Environment=f"SKYNET_VIDEO_GENERATION={token}", MainPID="0", ControlGroup="")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["systemctl", "--user", "show"]:
            return SimpleNamespace(returncode=0, stdout="\n".join(f"{k}={v}" for k, v in state.items()), stderr="")
        if args[:3] == ["systemctl", "--user", "stop"]:
            state.update(ActiveState="inactive", SubState="dead")
        if args[0] == "systemd-run" or args[:3] == ["systemctl", "--user", "start"]:
            state.update(LoadState="loaded", ActiveState="active", SubState="running")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    value = dict(root=str(root), generation=token, sources={}, request={},
                 profile=dict(runtime="/runtime", repository="/repo", work_root="/work"))
    return control, value, state, calls, root


def test_remote_cancel_before_start_prevents_late_launch(video_supervisor):
    control, value, _, calls, _ = video_supervisor
    assert control(dict(value, operation="cancel"))["state"] == "CANCELLED"
    assert control(dict(value, operation="start"))["state"] == "CANCELLED"
    assert not any(call[0] == "systemd-run" for call in calls)


def test_remote_cancel_stops_only_owned_unit_and_cleans_only_owned_files(video_supervisor):
    control, value, state, calls, root = video_supervisor
    control(dict(value, operation="start"))
    sibling = root.parent / "another-generation.mp4"
    sibling.write_bytes(b"other video")
    (root / "video.uuid.partial.mp4").write_bytes(b"partial")
    (root / "video.mp4").write_bytes(b"own completed file")
    assert control(dict(value, operation="cancel"))["state"] == "CANCELLED"
    stops = [call for call in calls if call[:3] == ["systemctl", "--user", "stop"]]
    assert stops == [["systemctl", "--user", "stop", f"skynet-video-{value['generation']}.service"]]
    assert sibling.read_bytes() == b"other video"
    assert not list(root.glob("*.mp4"))
    launch = next(call for call in calls if call[0] == "systemd-run")
    assert "--property=KillMode=control-group" in launch
    assert "--conflict-exit-code=75" in launch
    assert "--property=RuntimeMaxSec=600" in launch


def test_remote_ownership_mismatch_cannot_stop_service(video_supervisor):
    control, value, state, calls, _ = video_supervisor
    state.update(LoadState="loaded", ActiveState="active", Environment="SKYNET_LIVE_SESSION_ID=other")
    with pytest.raises(RuntimeError, match="ownership does not match"):
        control(dict(value, operation="cancel"))
    assert not any(call[:3] == ["systemctl", "--user", "stop"] for call in calls)


def test_remote_ready_waits_for_process_exit_and_descendant_cleanup(video_supervisor):
    control, value, state, calls, root = video_supervisor
    control(dict(value, operation="start"))
    metadata = dict(state="READY", path=str(root / "video.mp4"))
    (root / "video.json").write_text(json.dumps(metadata))
    assert control(dict(value, operation="status"))["state"] == "PREPARING"
    state.update(SubState="exited")
    assert control(dict(value, operation="status"))["state"] == "READY"
    assert state["ActiveState"] == "inactive"
    assert any(call[:3] == ["systemctl", "--user", "stop"] for call in calls)


def test_remote_gpu_lock_exit_is_retryable_wait_state(video_supervisor):
    control, value, state, calls, _ = video_supervisor
    control(dict(value, operation="start"))
    state.update(ActiveState="failed", SubState="failed", ExecMainStatus="75")
    assert control(dict(value, operation="status"))["state"] == "WAITING_GPU"
    assert control(dict(value, operation="start"))["state"] == "STARTING"
    assert len([call for call in calls if call[0] == "systemd-run"]) == 1
    assert ["systemctl", "--user", "start", f"skynet-video-{value['generation']}.service"] in calls


def test_invalid_recovered_video_owner_cannot_poison_active_state(videos):
    directory = videos.source("session", 0, 0)[3]
    videos.publish(directory, state="PREPARING", generation="a" * 32,
                   remote_root="/another/recording/attempts/" + "a" * 32)
    for _ in range(2):
        with pytest.raises(ValueError, match="does not match this recording"):
            videos.cancel("session", 0)
        assert not videos.generations and not videos.active
        assert videos.status("session", 0)["state"] == "INTERRUPTED"


def test_recovered_video_transport_validation_failure_can_be_retried(videos):
    job, _, review, directory = videos.source("session", 0, 0)
    token = "a" * 32
    root = job["root"] + f"/output/review-videos/{review['sha256']}/0/{videos.version}/attempts/{token}"
    videos.publish(directory, state="PREPARING", generation=token, remote_root=root)
    original_transport = videos.live.transport
    videos.live.transport = lambda _: (_ for _ in ()).throw(ValueError("Invalid saved workstation runtime"))
    failed = videos.cancel("session", 0)
    assert failed["state"] == "CANCELLING" and failed["can_cancel"]
    assert "Invalid saved workstation runtime" in failed["error"]
    videos.live.transport = original_transport
    videos._control = lambda *args, **kwargs: {"state": "CANCELLED"}
    assert videos.cancel("session", 0)["state"] == "CANCELLED"


@pytest.mark.parametrize("legacy", [True, False])
def test_remote_completed_cache_is_reused_without_new_gpu_work(video_supervisor, legacy):
    control, value, _, calls, root = video_supervisor
    base = root.parent.parent
    cached_root = base if legacy else base / "attempts" / ("b" * 32)
    cached_root.mkdir(parents=True, exist_ok=True)
    video = cached_root / "video.mp4"
    video.write_bytes(b"cached video")
    (base / "video.json").write_text(json.dumps(dict(state="READY", path=str(video))))
    assert control(dict(value, operation="start"))["path"] == str(video)
    assert not any(call[0] == "systemd-run" for call in calls)


def test_remote_failed_unit_with_live_descendants_is_not_confirmed_cancelled(video_supervisor, monkeypatch):
    from pathlib import Path

    control, value, state, _, root = video_supervisor
    control(dict(value, operation="start"))
    state.update(ActiveState="failed", SubState="failed", ControlGroup="/skynet-test")
    (root / "video.uuid.partial.mp4").write_bytes(b"still owned by a process")
    original_exists, original_read = Path.exists, Path.read_text
    group = Path("/sys/fs/cgroup/skynet-test")
    populated = True

    def exists(path):
        return True if path in (group, group / "cgroup.events") else original_exists(path)

    def read(path, *args, **kwargs):
        if path == group / "cgroup.events":
            return f"populated {int(populated)}\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(RuntimeError, match="has not stopped yet"):
        control(dict(value, operation="cancel"))
    assert (root / "video.uuid.partial.mp4").exists()
    populated = False
    assert control(dict(value, operation="cancel"))["state"] == "CANCELLED"
    assert not (root / "video.uuid.partial.mp4").exists()


def test_late_worker_cleanup_does_not_repeat_a_failed_cancel_without_retry(videos):
    import threading

    launched, release = threading.Event(), threading.Event()
    cancel_calls = []

    def control(transport, job, generation, root, operation, **kwargs):
        if operation == "start":
            launched.set()
            assert release.wait(2)
            return {"state": "READY", "path": root + "/video.mp4"}
        cancel_calls.append(operation)
        if len(cancel_calls) == 1:
            raise RuntimeError("Synthetic cancellation unavailable")
        return {"state": "CANCELLED"}

    videos._control = control
    videos.create("session", 0)
    assert launched.wait(2)
    assert videos.cancel("session", 0)["can_cancel"]
    release.set()
    videos.executor.shutdown(wait=True)
    assert cancel_calls == ["cancel"]
    assert videos.status("session", 0)["state"] == "CANCELLING"
    assert videos.cancel("session", 0)["state"] == "CANCELLED"
    assert cancel_calls == ["cancel", "cancel"]
