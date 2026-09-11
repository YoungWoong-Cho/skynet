"""Remote protocol distinguishes a launched flock from an acquired GPU."""

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from skynet_app.live_xr_video_worker import control


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    token = "a" * 32
    root = tmp_path / "attempts" / token
    profile = dict(runtime="/runtime", repository="/repo", work_root=str(tmp_path))
    value = dict(root=str(root), generation=token, sources={}, request=dict(profile=profile), profile=profile)
    fields = dict(LoadState="not-found", ActiveState="inactive", SubState="dead", ExecMainStatus="0",
                  Environment=f"SKYNET_VIDEO_GENERATION={token}", MainPID="0", ControlGroup="")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[:3] == ["systemctl", "--user", "show"]:
            return SimpleNamespace(returncode=0, stdout="\n".join(f"{k}={v}" for k, v in fields.items()), stderr="")
        if args[:3] == ["systemctl", "--user", "stop"]:
            fields.update(ActiveState="inactive", SubState="dead", MainPID="0")
        if args[0] == "systemd-run" or args[:3] == ["systemctl", "--user", "start"]:
            fields.update(LoadState="loaded", ActiveState="active", SubState="running", ExecMainStatus="0")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    return value, fields, calls, root


def test_launch_is_starting_until_renderer_proves_gpu_acquisition(supervisor):
    value, _, calls, root = supervisor
    assert control(dict(value, operation="start")) == {"state": "STARTING"}
    assert control(dict(value, operation="status")) == {"state": "STARTING"}
    assert control(dict(value, operation="start")) == {"state": "STARTING"}
    assert len([call for call in calls if call[0] == "systemd-run"]) == 1
    (root / "video.json").write_text(json.dumps(dict(state="RENDERING", frame=60, total=145)))
    assert control(dict(value, operation="status")) == {
        "state": "PREPARING", "phase": "rendering", "frame": 60, "total": 145,
    }
    launch = next(call for call in calls if call[0] == "systemd-run")
    assert "/usr/bin/flock" in launch and "--conflict-exit-code=75" in launch


@pytest.mark.parametrize("metadata_state", ["READY", "FAILED"])
def test_terminal_metadata_waits_for_owned_process_cleanup(supervisor, metadata_state):
    value, fields, calls, root = supervisor
    control(dict(value, operation="start"))
    (root / "video.json").write_text(json.dumps(dict(
        state=metadata_state, path=str(root / "video.mp4"), error="Renderer failed",
    )))
    assert control(dict(value, operation="status")) == {"state": "PREPARING", "phase": "finalizing"}
    assert not any(call[:3] == ["systemctl", "--user", "stop"] for call in calls)
    fields.update(SubState="exited")
    assert control(dict(value, operation="status"))["state"] == metadata_state
    assert ["systemctl", "--user", "stop", f"skynet-video-{value['generation']}.service"] in calls


@pytest.mark.parametrize("unit_loaded", [True, False])
def test_restart_does_not_reuse_stale_rendering_marker(supervisor, unit_loaded):
    value, fields, calls, root = supervisor
    control(dict(value, operation="start"))
    fields.update(LoadState="loaded" if unit_loaded else "not-found",
                  ActiveState="failed", SubState="failed", ExecMainStatus="75")
    (root / "video.json").write_text(json.dumps(dict(state="RENDERING", frame=60, total=145)))
    assert control(dict(value, operation="start")) == {"state": "STARTING"}
    assert not (root / "video.json").exists()
    assert control(dict(value, operation="status")) == {"state": "STARTING"}
    assert not any("skynet-live-" in str(call) for call in calls)


def test_busy_owner_is_matching_lock_holder_even_without_profile_in_status(supervisor, monkeypatch):
    value, fields, calls, root = supervisor
    control(dict(value, operation="start"))
    fields.update(ActiveState="failed", SubState="failed", ExecMainStatus="75")
    gpu_lock = Path(value["profile"]["work_root"]) / ".gpu-session.lock"
    gpu_lock.touch()
    info = gpu_lock.stat()
    device = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}"
    session = "c94e16e6-6db0-4e66-b351-184aec09e0d2"
    read_text = Path.read_text

    def read(path, *args, **kwargs):
        if path == Path("/proc/locks"):
            return ("1: FLOCK ADVISORY WRITE 99 00:00:1 0 EOF\n"
                    f"2: FLOCK ADVISORY WRITE 1234 {device} 0 EOF\n")
        if path == Path("/proc/1234/cgroup"):
            return f"0::/user.slice/user-1000.slice/user@1000.service/app.slice/skynet-live-{session}.service\n"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    result = control(dict(root=str(root), generation=value["generation"], operation="status"))
    assert result == {"state": "WAITING_GPU", "busy_owner": {
        "pid": 1234, "unit": f"skynet-live-{session}.service", "kind": "collection",
    }}
    assert not any(call[:3] == ["systemctl", "--user", "stop"] for call in calls)


def test_unavailable_owner_information_does_not_hide_lock_conflict(supervisor, monkeypatch):
    value, fields, _, _ = supervisor
    control(dict(value, operation="start"))
    fields.update(ActiveState="failed", SubState="failed", ExecMainStatus="75")
    (Path(value["profile"]["work_root"]) / ".gpu-session.lock").touch()
    read_text = Path.read_text

    def read(path, *args, **kwargs):
        if path == Path("/proc/locks"):
            raise PermissionError("Other processes are private")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert control(dict(value, operation="status")) == {"state": "WAITING_GPU"}


def test_renderer_failure_is_not_misreported_as_gpu_wait(supervisor):
    value, fields, _, root = supervisor
    control(dict(value, operation="start"))
    fields.update(ActiveState="failed", SubState="failed", ExecMainStatus="1")
    (root / "video.json").write_text(json.dumps(dict(state="FAILED", error="Camera initialization failed")))
    assert control(dict(value, operation="status")) == {"state": "FAILED", "error": "Camera initialization failed"}
