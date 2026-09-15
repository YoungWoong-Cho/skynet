import json
from pathlib import Path
from uuid import uuid4

import pytest

from skynet_app.cluster_runtime import ClusterError, SubmissionOutcomeUnknown
from skynet_app.database import Database
from skynet_app.live_xr import LiveXRService
from skynet_app.live_xr_workstation import (
    LaunchRejected,
    WorkstationClient,
    validate_profile,
)


@pytest.fixture
def service(tmp_path, monkeypatch, prepared_hand_store):
    source = Path(__file__).resolve().parents[1]
    (tmp_path / "config").mkdir()
    (tmp_path / "ops/xr").mkdir(parents=True)
    for name in ("ops/xr/native_session.py",):
        (tmp_path / name).write_text((source / name).read_text())
    (tmp_path / "ops/xr/hands").mkdir()
    for name in (
        "collection.py",
        "hands/anatomy.py",
        "wrist.py",
        "retargeting_runtime.py",
        "images.py",
        "render_images.py",
    ):
        (tmp_path / "ops/xr" / name).write_text((source / "ops/xr" / name).read_text())
    (tmp_path / "config/live_xr.json").write_text(
        json.dumps(
            {
                **json.loads((source / "config/live_xr.json").read_text()),
                "execution": "workstation",
                "gateway": "test-workstation",
                "work_root": "/home/test/skynet-xr",
                "repository": "/home/test/skynet-xr/repos/dexverse",
                "runtime": "/home/test/skynet-xr/env",
                "cloudxr_runtime": "/home/test/skynet-xr/cloudxr",
                "duration_minutes": 30,
            }
        )
    )
    service = LiveXRService(Database(tmp_path / "db.store"), root=tmp_path)
    monkeypatch.setattr(service, "dispatch", lambda _: None)
    return service


def test_workstation_profile_has_no_cluster_fallback(service):
    profile = service.profile()
    assert profile["execution"] == "workstation"
    assert "account" not in profile and "partition" not in profile
    job = service.create(True)
    assert job["root"].startswith(profile["work_root"] + "/sessions/")
    script = service.compile(service.get(job["id"]))
    assert "#SBATCH" not in script
    assert job["root"] + "/runner.py" in script
    assert "\nexec python3 " in script, (
        "Stop must signal the worker, not a shell parent"
    )
    assert isinstance(service.transport(job), WorkstationClient)


def test_workstation_failure_never_submits_cluster_job(service, monkeypatch):
    class Unavailable(WorkstationClient):
        def ssh(self, *args, **kwargs):
            raise ClusterError("Runtime missing")

    monkeypatch.setattr(service, "transport", lambda job: Unavailable(job["profile"]))
    monkeypatch.setattr(
        service.cluster,
        "submit_script",
        lambda *a, **kw: pytest.fail("Unexpected Slurm fallback"),
    )
    job = service.create(True)
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "FAILED"
    assert "Runtime missing" in service.get(job["id"])["error"]


@pytest.mark.parametrize("value", ["/tmp/other", "/home/test/skynet-xr/../other"])
def test_workstation_runtime_must_stay_in_workspace(service, value):
    profile = service.profile()
    profile["runtime"] = value
    with pytest.raises(ValueError, match="inside"):
        validate_profile(profile)


def test_lost_workstation_reply_recovers_exact_service(service, monkeypatch):
    job = service.create(True)
    client = WorkstationClient(job["profile"])
    calls = []
    monkeypatch.setattr(client, "write_capsule_file", lambda *a: None)

    def lost(*args, **kwargs):
        calls.append(kwargs["stdin"])
        raise ClusterError("Reply lost")

    monkeypatch.setattr(client, "ssh", lost)
    with pytest.raises(SubmissionOutcomeUnknown):
        client.submit_script("echo test", job["id"], job["gateway"])
    launch = json.loads(calls[0])
    assert launch["unit"] == client.unit(job["id"])
    assert "--property=Restart=no" in launch["command"]
    assert "--conflict-exit-code=75" in launch["command"]
    assert job["profile"]["work_root"] + "/.gpu-session.lock" in launch["command"]
    monkeypatch.setattr(client, "service_status", lambda *a: {"LoadState": "loaded"})
    assert (
        client.recover_submission(job["id"], job["id"], job["gateway"]).job_id
        == launch["unit"]
    )
    assert len(calls) == 1


def test_clean_stop_survives_systemd_unloading_unit(service, monkeypatch):
    client = WorkstationClient(service.profile())
    unit = client.unit(str(uuid4()))
    monkeypatch.setattr(client, "service_status", lambda *a: {"LoadState": "not-found"})
    monkeypatch.setattr(
        client, "ssh", lambda *a, **k: json.dumps({"job_id": unit, "state": "STOPPED"})
    )
    assert client.job_statuses([unit], "rl2-bonjour")[1][unit]["State"] == "COMPLETED"
    monkeypatch.setattr(
        client,
        "ssh",
        lambda *a, **k: json.dumps({"job_id": unit, "state": "AWAITING_HEADSET"}),
    )
    assert client.job_statuses([unit], "rl2-bonjour")[1][unit]["State"] == "FAILED"


def test_rejected_launch_is_terminal(service, monkeypatch):
    job = service.create(True)
    service.update(job["id"], state="SUBMISSION_UNKNOWN")
    client = WorkstationClient(job["profile"])

    def reject(*a):
        raise LaunchRejected("Memory controller unavailable")

    monkeypatch.setattr(client, "recover_submission", reject)
    monkeypatch.setattr(service, "transport", lambda _: client)
    result = service.refresh(job["id"], force=True)
    assert result["state"] == "FAILED"
    assert "Memory controller" in result["error"]


def test_stop_is_idempotent_after_unit_unloads(service, monkeypatch):
    client = WorkstationClient(service.profile())
    unit = client.unit(str(uuid4()))
    monkeypatch.setattr(client, "service_status", lambda *a: {"LoadState": "not-found"})
    monkeypatch.setattr(
        client, "ssh", lambda *a, **k: pytest.fail("Unit already stopped")
    )
    client.cancel(unit, service.profile()["gateway"])


def test_stop_handles_unit_unloading_during_request(service, monkeypatch):
    client = WorkstationClient(service.profile())
    states = iter([{"LoadState": "loaded"}, {"LoadState": "not-found"}])
    monkeypatch.setattr(client, "service_status", lambda *a: next(states))

    def unloaded(*a, **k):
        raise ClusterError("Unit not loaded")

    monkeypatch.setattr(client, "ssh", unloaded)
    client.cancel(client.unit(str(uuid4())), service.profile()["gateway"])


def test_workstation_gpu_lock_conflict_is_explicit(service, monkeypatch):
    client = WorkstationClient(service.profile())
    unit = client.unit(str(uuid4()))
    monkeypatch.setattr(
        client,
        "service_status",
        lambda *a: {
            "LoadState": "loaded",
            "ActiveState": "failed",
            "ExecMainStatus": "75",
        },
    )
    result = client.job_statuses([unit], "test-workstation")[1][unit]
    assert result["State"] == "FAILED"
    assert "already running" in result["Result"]
    with pytest.raises(ValueError, match="inside the workstation"):
        client._remote_path("/home/test/skynet-xr/../../other.pkl")


def test_image_capture_stop_requests_graceful_shutdown_and_rendering(
    service, monkeypatch
):
    job = service.create(True, image_capture=True)
    service.update(
        job["id"], state="COLLECTING", job_id="skynet-live-" + job["id"] + ".service"
    )
    calls = []

    class Client:
        def ssh(self, *a, **k):
            calls.append((a, k))

        def cancel(self, *a, **k):
            pytest.fail("Cancelling systemd would kill image preparation")

    monkeypatch.setattr(service, "transport", lambda _: Client())
    service.stop(job["id"])
    assert calls and "stop.request" in str(calls)


@pytest.mark.parametrize("method", ["dexpilot", "vector-wrist-joint"])
def test_prepare_uses_real_workstation_file_validation(service, monkeypatch, method):
    from skynet_app.retargeting import configuration

    job = service.create(True)
    profile = dict(job["profile"])
    profile["retargeting"] = configuration(method, profile["work_root"])
    service.update(job["id"], profile=profile)
    client = WorkstationClient(profile)
    calls = []

    def ssh(gateway, command, **kwargs):
        calls.append((command, kwargs.get("stdin")))
        return client.unit(job["id"])

    # Keep write_capsule_file and submit_script intact so startup exercises
    # the workstation allowlist, not only a mock transport's write method.
    monkeypatch.setattr(client, "ssh", ssh)
    monkeypatch.setattr(service, "transport", lambda _: client)
    service.prepare(job["id"])

    result = service.get(job["id"])
    assert result["state"] == "STARTING_SERVER", result.get("error")
    assert result["job_id"] == client.unit(job["id"])
    writes = [command for command, body in calls if body is not None]
    for name in ("runner.py", "request.json", "run.sh"):
        assert any(job["root"] + "/" + name in command for command in writes)
    checks = [command for command, _ in calls if " --check " in command]
    assert bool(checks) == (method == "vector-wrist-joint")
    if checks:
        assert any(job["root"] + "/retargeting-check.py" in command for command in writes)
        assert profile["retargeting"]["runtime_root"] in checks[0]


@pytest.mark.parametrize("name", ["other.py", "../retargeting-check.py", "/retargeting-check.py", "output/retargeting-check.py"])
def test_unknown_session_files_are_rejected_before_ssh(service, monkeypatch, name):
    client = WorkstationClient(service.profile())
    monkeypatch.setattr(client, "ssh", lambda *a, **k: pytest.fail("Unexpected file upload"))
    with pytest.raises(ValueError, match="Unknown live session file"):
        client.write_capsule_file(str(uuid4()), name, "unapproved", "test-workstation")
