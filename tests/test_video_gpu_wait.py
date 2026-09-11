"""GPU contention must not look like rendering or outlive its bounded request."""

import json
from types import SimpleNamespace

import pytest

from skynet_app import live_xr_video
from test_live_review import payload, review  # Shared local recording fixtures.


KEY = ("session", 0, 0)
BUSY_OWNER = {"pid": 12345, "unit": "skynet-live-c94e16e6-1111-4111-8111-111111111111.service", "kind": "collection"}


@pytest.fixture
def video_case(review, monkeypatch):
    reviews, job, _ = review
    reviews.directory("session", 0).mkdir(parents=True)
    reviews.prepare("session", 0)
    job["profile"].update(execution="workstation", runtime="/runtime", repository="/repo", work_root="/work")
    service = live_xr_video.LiveVideoService(reviews)
    service.sources = {}
    service.version = "a" * 16
    reviews.live.list = lambda: []
    callbacks = []
    monkeypatch.setattr(service.executor, "submit", lambda *args: callbacks.append(args))
    observations = []
    publish = service.publish

    def observed(directory, **value):
        observations.append(dict(value))
        return publish(directory, **value)

    monkeypatch.setattr(service, "publish", observed)
    first = service.create(*KEY)
    generation = service.generations[KEY]
    clock = {"now": 0.0}
    waits = []
    monkeypatch.setattr(live_xr_video.time, "monotonic", lambda: clock["now"])

    def wait(seconds):
        waits.append((seconds, service.status(*KEY)))
        clock["now"] += seconds
        return generation.cancel.is_set()

    monkeypatch.setattr(generation.cancel, "wait", wait)
    case = SimpleNamespace(service=service, job=job, first=first, generation=generation,
                           observations=observations, waits=waits, clock=clock,
                           directory=service.source(*KEY)[3], callbacks=callbacks)
    yield case
    if KEY in service.active:
        service.cancel(*KEY, expected_generation=generation.token)
    service.executor.shutdown(wait=True)
    reviews.executor.shutdown(wait=True)


def run_worker(case):
    callback = case.callbacks.pop(0)
    callback[0](*callback[1:])


def ready(case):
    return dict(state="READY", kind="replay", path="/synthetic/video.mp4", fps=30,
                frames=2, size_bytes=24, sha256="a" * 64, renderer_version=case.service.version)


def test_allocated_cluster_startup_uses_render_limit_instead_of_queue_limit(video_case, monkeypatch):
    case = video_case
    polls = []

    def control(transport, job, generation, root, operation, **kwargs):
        if operation == "start":
            return {"state": "STARTING", "phase": "queued", "job_id": "123"}
        polls.append(operation)
        if len(polls) == 1:
            case.clock["now"] = 119
            return {"state": "STARTING", "phase": "starting", "job_id": "123"}
        case.clock["now"] = 200
        return ready(case)

    monkeypatch.setattr(case.service, "_control", control)
    monkeypatch.setattr(case.service, "download", lambda *args: (case.directory / "video.mp4").write_bytes(b"cached"))
    run_worker(case)
    assert case.service.status(*KEY)["state"] == "READY"
    assert case.generation.render_deadline == 119 + live_xr_video.RENDER_TIMEOUT_SECONDS
    assert any("GPU allocated" in item.get("detail", "") for item in case.observations)


def test_gpu_busy_retries_keep_waiting_until_remote_confirms_rendering(video_case, monkeypatch):
    case = video_case
    calls = []
    rendered = {"confirmed": False}
    unsafe = []
    publish = case.service.publish

    def check_rendering(directory, **value):
        if value["state"] == "PREPARING" and not rendered["confirmed"]:
            unsafe.append(value)
        return publish(directory, **value)

    monkeypatch.setattr(case.service, "publish", check_rendering)
    starts = 0
    rendering_polls = 0

    def ssh(_host, _command, *, stdin, **_kwargs):
        nonlocal starts, rendering_polls
        request = json.loads(stdin)
        operation = request["operation"]
        calls.append((operation, request["generation"], case.service.status(*KEY)["state"]))
        if operation == "start":
            starts += 1
            return json.dumps({"state": "STARTING"})
        if operation == "cancel":
            return json.dumps({"state": "CANCELLED"})
        if starts <= 2:
            return json.dumps({"state": "WAITING_GPU", "busy_owner": BUSY_OWNER})
        rendering_polls += 1
        if rendering_polls <= 2:
            rendered["confirmed"] = True
            return json.dumps({"state": "PREPARING"})
        return json.dumps(ready(case))

    case.service.live.transport = lambda _job: SimpleNamespace(ssh=ssh)
    monkeypatch.setattr(case.service, "download", lambda *args: (case.directory / "video.mp4").write_bytes(b"cached"))
    run_worker(case)
    result = case.service.status(*KEY)
    assert result["state"] == "READY", result
    assert result["generation"] == case.generation.token
    assert not unsafe, "Rendering was announced before the remote worker acquired the GPU"
    starts_seen = [state for operation, _generation, state in calls if operation == "start"]
    assert starts_seen == ["STARTING", "WAITING_GPU", "WAITING_GPU"]
    first_wait = next(index for index, item in enumerate(case.observations) if item["state"] == "WAITING_GPU")
    first_render = next(index for index, item in enumerate(case.observations) if item["state"] == "PREPARING")
    assert {item["state"] for item in case.observations[first_wait:first_render]} == {"WAITING_GPU"}
    assert [seconds for seconds, status in case.waits if status["state"] == "WAITING_GPU" and seconds >= 1] == [5, 5]
    waits = [status for seconds, status in case.waits if status["state"] == "WAITING_GPU" and seconds >= 1]
    assert all(status["busy_owner"] == BUSY_OWNER and "c94e16e6" in status["detail"] for status in waits)
    assert waits[0]["retry_seconds_remaining"] > waits[1]["retry_seconds_remaining"] >= 0
    assert all(not item.get("busy_owner") and "c94e16e6" not in item.get("detail", "")
               for item in case.observations[first_render:])
    assert {generation for _operation, generation, _state in calls} == {case.generation.token}
    assert not case.service.active and not case.service.queue


def test_gpu_wait_has_a_finite_deadline_without_rendering_flicker(video_case, monkeypatch):
    case = video_case
    calls = []
    started_at = []

    def ssh(_host, _command, *, stdin, **_kwargs):
        request = json.loads(stdin)
        calls.append(request)
        assert request["operation"] == "start"
        started_at.append(case.clock["now"])
        return json.dumps({"state": "WAITING_GPU", "busy_owner": BUSY_OWNER})

    case.service.live.transport = lambda _job: SimpleNamespace(ssh=ssh)
    monkeypatch.setattr(case.service, "download", lambda *args: pytest.fail("Busy GPU must not download or publish a video"))
    run_worker(case)
    result = case.service.status(*KEY)
    assert result["state"] == "FAILED", result
    assert "GPU" in result["error"] and "busy" in result["error"]
    assert result["generation"] == case.generation.token and not result["can_cancel"]
    assert case.clock["now"] == live_xr_video.GPU_WAIT_SECONDS == 120
    assert started_at == list(range(0, 120, 5)), "No fresh launch is allowed at or after the wait deadline"
    assert all(seconds == live_xr_video.GPU_RETRY_SECONDS == 5 for seconds, _status in case.waits)
    assert all(status["state"] == "WAITING_GPU" for _seconds, status in case.waits)
    assert all(status["busy_owner"] == BUSY_OWNER and "c94e16e6" in status["detail"] for _seconds, status in case.waits)
    remaining = [status["retry_seconds_remaining"] for _seconds, status in case.waits]
    assert remaining == list(range(120, 0, -5))
    assert all(type(value) is int for value in remaining)
    assert not any(item["state"] == "PREPARING" for item in case.observations)
    assert {item["generation"] for item in calls} == {case.generation.token}
    assert not (case.directory / "video.mp4").exists()
    assert not case.service.active and not case.service.queue


def test_unconfirmed_starting_has_one_cumulative_deadline_and_owned_cleanup(video_case, monkeypatch):
    case = video_case
    calls = []

    def ssh(_host, _command, *, stdin, **_kwargs):
        request = json.loads(stdin)
        calls.append(request)
        return json.dumps({"state": "CANCELLED" if request["operation"] == "cancel" else "STARTING"})

    case.service.live.transport = lambda _job: SimpleNamespace(ssh=ssh)
    monkeypatch.setattr(case.service, "download", lambda *args: pytest.fail("Unconfirmed startup must not download"))
    run_worker(case)
    result = case.service.status(*KEY)
    assert result["state"] == "FAILED", result
    assert result["generation"] == case.generation.token and not result["can_cancel"]
    assert case.clock["now"] == live_xr_video.GPU_WAIT_SECONDS == 120
    assert [item["operation"] for item in calls].count("start") == 1
    assert [item["operation"] for item in calls].count("cancel") == 1
    assert {item["generation"] for item in calls} == {case.generation.token}
    assert not any(item["state"] == "PREPARING" for item in case.observations)
    assert not case.service.active and not (case.directory / "video.mp4").exists()


def test_render_deadline_survives_unexpected_remote_state_regression(video_case, monkeypatch):
    case = video_case
    monkeypatch.setattr(live_xr_video, "RENDER_TIMEOUT_SECONDS", 3)

    def ssh(_host, _command, *, stdin, **_kwargs):
        request = json.loads(stdin)
        if request["operation"] == "cancel":
            return json.dumps({"state": "CANCELLED"})
        if case.clock["now"] > 3:
            raise RuntimeError("Test guard: regressed state bypassed the rendering deadline")
        return json.dumps({"state": "PREPARING" if request["operation"] == "start" else "STARTING"})

    case.service.live.transport = lambda _job: SimpleNamespace(ssh=ssh)
    monkeypatch.setattr(case.service, "download", lambda *args: pytest.fail("Timed-out rendering must not download"))
    run_worker(case)
    result = case.service.status(*KEY)
    assert result["state"] == "FAILED", result
    assert case.clock["now"] == 3
    assert "time limit" in result["error"] and "Test guard" not in result["error"]
    assert result["generation"] == case.generation.token
    assert not case.service.active and not (case.directory / "video.mp4").exists()


@pytest.mark.parametrize("waiting_state", ["STARTING", "WAITING_GPU"])
def test_cancel_during_start_or_gpu_wait_stops_this_generation_without_another_retry(video_case, monkeypatch, waiting_state):
    case = video_case
    calls = []

    def ssh(_host, _command, *, stdin, **_kwargs):
        request = json.loads(stdin)
        calls.append(request)
        return json.dumps({"state": "CANCELLED" if request["operation"] == "cancel" else waiting_state})

    case.service.live.transport = lambda _job: SimpleNamespace(ssh=ssh)

    def cancel_in_wait(_seconds):
        status = case.service.status(*KEY)
        assert status["state"] == waiting_state
        case.service.cancel(*KEY, expected_generation=case.generation.token)
        return case.generation.cancel.is_set()

    monkeypatch.setattr(case.generation.cancel, "wait", cancel_in_wait)
    monkeypatch.setattr(case.service, "download", lambda *args: pytest.fail("Cancelled generation must not download"))
    run_worker(case)
    result = case.service.status(*KEY)
    assert result["state"] == "CANCELLED", result
    assert result["generation"] == case.generation.token
    assert [item["operation"] for item in calls].count("start") == 1
    assert {item["generation"] for item in calls} == {case.generation.token}
    assert not case.service.active and not case.service.queue
    assert not list(case.directory.glob("*.part")) and not (case.directory / "video.mp4").exists()
    replacement = case.service.create(*KEY)
    assert replacement["generation"] != case.generation.token
    assert case.service.cancel(*KEY, expected_generation=case.generation.token)["state"] == "QUEUED"
    case.service.cancel(*KEY, expected_generation=replacement["generation"])


def test_interrupted_starting_generation_is_recoverable_without_claiming_rendering(video_case):
    case = video_case
    case.service.active.clear()
    case.service.queue.clear()
    case.service.generations.clear()
    case.service.publish(case.directory, state="STARTING", generation=case.generation.token)
    result = case.service.status(*KEY)
    assert result["state"] == "INTERRUPTED" and result["can_cancel"]
    assert result["generation"] == case.generation.token
