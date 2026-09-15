import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from skynet_app.cluster_runtime import (
    ClusterClient,
    SubmissionOutcomeUnknown,
    ClusterError,
)
from skynet_app.database import Database
from skynet_app.live_xr import LiveXRService


class Cluster(ClusterClient):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.control = {}
        self.state = "RUNNING"
        self.lost = False
        self.offline = False
        self.writes = []

    def ssh(self, gateway, command, **kwargs):
        if self.offline:
            raise ClusterError("Connection timed out")
        return json.dumps(self.control) if command.startswith("python3 - ") else ""

    def write_capsule_file(self, *args):
        self.writes.append(args)

    def submit_script(self, *args, **kwargs):
        self.calls += 1
        if self.lost:
            raise SubmissionOutcomeUnknown("Acknowledgement lost")
        return SimpleNamespace(job_id="123")

    def recover_submission(self, *args):
        return SimpleNamespace(job_id="123")

    def job_statuses(self, *args):
        return "sky2", {"123": {"State": self.state}}


@pytest.fixture
def service(tmp_path, monkeypatch, prepared_hand_store):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config").mkdir()
    (tmp_path / "ops/xr").mkdir(parents=True)
    profile = json.loads((root / "config/live_xr.json").read_text())
    profile.update(
        execution="slurm", gateway="sky2", account="overcap", partition="overcap",
        work_root="/coc/flash7/ycho420",
        repository="/coc/flash7/ycho420/repos/skynet-dexverse/" + profile["source_revision"],
        runtime="/coc/flash7/ycho420/envs/isaacsim-5.1.0_isaaclab-2.3.2_py311",
        cloudxr_runtime="/coc/flash7/ycho420/tools/skynet-xr/native-5.0.1",
        gpu_type="rtx_6000", duration_minutes=30,
    )
    (tmp_path / "config/live_xr.json").write_text(json.dumps(profile))
    (tmp_path / "ops/xr/native_session.py").write_text(
        (root / "ops/xr/native_session.py").read_text()
    )
    (tmp_path / "ops/xr/hands").mkdir()
    for name in (
        "collection.py",
        "hands/anatomy.py",
        "wrist.py",
        "retargeting_runtime.py",
        "images.py",
        "render_images.py",
    ):
        (tmp_path / "ops/xr" / name).write_text((root / "ops/xr" / name).read_text())
    service = LiveXRService(Database(tmp_path / "db.store"), Cluster(), root=tmp_path)
    monkeypatch.setattr(service, "dispatch", lambda _: None)
    return service


def test_requires_license_and_deduplicates_live_start(service):
    with pytest.raises(ValueError, match="license"):
        service.create()
    job = service.create(True)
    assert service.create()["id"] == job["id"]
    service.prepare(job["id"])
    assert service.cluster.calls == 1
    assert service.get(job["id"])["state"] == "PENDING"
    assert "runner.py" in service.cluster.writes[0]
    assert "--gres=gpu:rtx_6000:1" in service.get(job["id"])["script"]


def test_frozen_session_extracts_the_complete_collection_runtime(service, tmp_path):
    job = service.create(True)
    ns = {"__name__": "test_frozen_session"}
    exec(compile(service.get(job["id"])["worker"], "runner.py", "exec"), ns)
    destination = tmp_path / "frozen-runtime"
    destination.mkdir()
    entry = ns["write_collection_files"](destination)
    assert entry.name == "collection.py"
    assert {p.name for p in entry.parent.iterdir()} == {
        "collection.py",
        "anatomy.py",
        "wrist.py",
        "images.py",
        "render_images.py",
        "arrays.py",
        "trajectory.py",
        "scene_geometry.py",
        "retargeting_runtime.py",
    }
    assert (entry.parent / "anatomy.py").read_text() == (
        service.root / "ops/xr/hands/anatomy.py"
    ).read_text()
    ns["COLLECTION_FILES"]["../unexpected.py"] = ""
    with pytest.raises(ValueError, match="incomplete"):
        ns["write_collection_files"](destination)


def test_lost_submission_recovers_without_duplicate(service):
    job = service.create(True)
    service.cluster.lost = True
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "SUBMISSION_UNKNOWN"
    assert service.refresh(job["id"], force=True)["job_id"] == "123"
    assert service.cluster.calls == 1


def test_scheduler_end_overrides_stale_readiness(service):
    job = service.create(True)
    service.prepare(job["id"])
    service.cluster.control = {
        "job_id": "123",
        "state": "AWAITING_HEADSET",
        "server_ready": True,
        "address": "192.0.2.1",
    }
    assert service.refresh(job["id"], force=True)["server_ready"]
    service.cluster.state = "CANCELLED"
    ended = service.refresh(job["id"], force=True)
    assert ended["state"] == "STOPPED"
    assert not ended["server_ready"]


def test_network_failure_is_not_reported_as_ready_or_completed(service):
    job = service.create(True)
    service.prepare(job["id"])
    service.cluster.offline = True
    result = service.refresh(job["id"], force=True)
    assert result["state"] == "PENDING"
    assert not result["server_ready"]
    assert "timed out" in result["error"]


def test_persistent_sessions_and_consent_survive_restart(service):
    job = service.create(True)
    other = LiveXRService(service.database, service.cluster, root=service.root)
    assert other.consent()["accepted"]
    assert other.list()[0]["id"] == job["id"]
    assert "worker" not in other.list()[0]
    assert other.create()["id"] == job["id"]


def test_worker_finish_waits_for_scheduler_and_surfaces_cleanup_failure(service):
    job = service.create(True)
    service.prepare(job["id"])
    service.cluster.control = {
        "job_id": "123",
        "state": "STOPPED",
        "server_ready": False,
    }
    result = service.refresh(job["id"], force=True)
    assert result["state"] == "STOPPING"
    assert not result["scheduler_final"]
    service.cluster.state = "FAILED"
    result = service.refresh(job["id"], force=True)
    assert result["state"] == "FAILED"
    assert result["scheduler_final"]
    assert "FAILED" in result["error"]


def test_selected_hand_and_task_are_frozen_in_request(service):
    job = service.create(True, "Dexverse-PickCube-v0", "skynet_shadow_left")
    assert job["profile"]["task"] == "Dexverse-PickCube-v0"
    assert job["profile"]["robot"] == "skynet_shadow_left"
    assert job["profile"]["hand"] == "left"
    service.prepare(job["id"])
    request = next(w[2] for w in service.cluster.writes if w[1] == "request.json")
    assert json.loads(request)["robot"] == "skynet_shadow_left"
    with pytest.raises(ValueError, match="different live session"):
        service.create(True, "Dexverse-PickCube-v0", "skynet_shadow_right")
    assert service.cluster.calls == 1


def test_unsupported_selection_does_not_start_or_fall_back(service):
    with pytest.raises(ValueError, match="Unsupported live hand"):
        service.create(True, robot="wuji-1")
    with pytest.raises(ValueError, match="Unsupported live task"):
        service.create(True, task="")
    assert not service.list()
    assert service.cluster.calls == 0


@pytest.fixture
def imported_hand(monkeypatch, tmp_path):
    manifest = dict(
        digest="c" * 64,
        robot="skynet_wuji_1_right",
        source_revision="a" * 40,
        name="WUJI Hand 1 right",
        action_dimension=26,
        retargeting_scheme="vector",
        hand_asset={"schema": "skynet.hand-bundle/v1", "digest": "d" * 64},
    )
    directory = tmp_path / "immutable-hand"
    monkeypatch.setattr(
        "skynet_app.live_xr.build_hand", lambda robot: (directory, manifest)
    )
    return directory, manifest


def test_imported_hand_bundle_and_scheme_are_frozen_without_default_robot(
    service, imported_hand, monkeypatch
):
    directory, manifest = imported_hand
    uploaded = []
    remote = "/coc/flash7/ycho420/skynet-work/hands/wuji/digest"

    def upload(*args):
        uploaded.append(args)
        return remote

    monkeypatch.setattr("skynet_app.live_xr.upload_hand", upload)
    job = service.create(True, "Dexverse-PickCube-v0", manifest["robot"])
    assert job["profile"]["hand_bundle"] == manifest
    assert "hand_bundle_path" not in job
    assert service.get(job["id"])["hand_bundle_path"] == str(directory)
    # A repeated request returns the existing session even if library access later fails.
    monkeypatch.setattr(
        "skynet_app.live_xr.build_hand",
        lambda _: pytest.fail("Duplicate rebuilt its model"),
    )
    assert (
        service.create(robot=manifest["robot"], task="Dexverse-PickCube-v0")["id"]
        == job["id"]
    )
    service.prepare(job["id"])
    request = json.loads(
        next(w[2] for w in service.cluster.writes if w[1] == "request.json")
    )
    assert request["robot"] == manifest["robot"]
    assert request["hand_bundle"]["retargeting_scheme"] == "vector"
    assert request["hand_bundle"]["root"] == remote
    assert uploaded[0][0] == str(directory)
    assert service.cluster.calls == 1


def test_imported_hand_preparation_failure_does_not_launch_shadow(
    service, imported_hand, monkeypatch
):
    def fail(*args):
        raise ValueError("Hand asset checksum mismatch")

    monkeypatch.setattr("skynet_app.live_xr.upload_hand", fail)
    job = service.create(True, robot=imported_hand[1]["robot"])
    service.prepare(job["id"])
    result = service.get(job["id"])
    assert result["state"] == "FAILED"
    assert "checksum mismatch" in result["error"]
    assert result["failed_stage"] == "hand"
    assert result["server_connected_at"] and result["runtime_checked_at"]
    assert not result.get("hand_prepared_at")
    assert result["profile"]["robot"] == imported_hand[1]["robot"]
    assert service.cluster.calls == 0
    assert not service.cluster.writes


def test_connection_failure_retains_failed_step_without_claiming_readiness(service):
    job = service.create(True)
    service.cluster.offline = True
    service.prepare(job["id"])
    failed = service.get(job["id"])
    assert failed["state"] == "FAILED"
    assert failed["failed_stage"] == "server"
    assert not any(
        failed.get(k)
        for k in (
            "server_connected_at",
            "runtime_checked_at",
            "hand_prepared_at",
            "stream_ready_at",
            "scene_ready_at",
        )
    )
    restarted = LiveXRService(service.database, service.cluster, root=service.root)
    assert restarted.list()[0]["failed_stage"] == "server"
    assert service.cluster.calls == 0


def test_runtime_failure_keeps_successful_connection_milestone(service, monkeypatch):
    def ssh(gateway, command, **kwargs):
        if command == "true":
            return ""
        raise ClusterError("Required runtime file is missing")

    monkeypatch.setattr(service.cluster, "ssh", ssh)
    job = service.create(True)
    service.prepare(job["id"])
    failed = service.get(job["id"])
    assert failed["server_connected_at"]
    assert failed["failed_stage"] == "runtime"
    assert not failed.get("runtime_checked_at")
    assert not failed.get("hand_prepared_at")
    assert "Required runtime file is missing" in failed["error"]
    assert service.cluster.calls == 0


def test_worker_milestones_distinguish_stream_from_loaded_scene(service):
    job = service.create(True)
    service.prepare(job["id"])
    service.cluster.control = {
        "job_id": "123",
        "state": "STARTING_SIMULATION",
        "startup_stage": "simulation",
        "stream_ready_at": 100,
        "server_ready": True,
    }
    loading = service.refresh(job["id"], force=True)
    assert all(
        loading.get(k)
        for k in (
            "server_connected_at",
            "runtime_checked_at",
            "hand_prepared_at",
            "stream_ready_at",
        )
    )
    assert not loading.get("scene_ready_at")
    service.cluster.control.update(
        state="AWAITING_HEADSET", startup_stage="ready", scene_ready_at=102
    )
    ready = service.refresh(job["id"], force=True)
    assert ready["scene_ready_at"] == 102
    assert ready["startup_stage"] == "ready"


def test_worker_failure_keeps_specific_error_after_process_exits(service):
    job = service.create(True)
    service.prepare(job["id"])
    service.cluster.control = {
        "job_id": "123",
        "state": "FAILED",
        "startup_stage": "simulation",
        "failed_stage": "simulation",
        "stream_ready_at": 100,
        "error": "Selected hand has a missing palm body",
    }
    service.cluster.state = "FAILED"
    failed = service.refresh(job["id"], force=True)
    assert failed["failed_stage"] == "simulation"
    assert failed["error"] == "Selected hand has a missing palm body"
    assert failed["stream_ready_at"] == 100
    assert not failed.get("scene_ready_at")


def test_image_capture_is_frozen_and_cannot_change_an_active_session(service):
    job = service.create(True, image_capture=True)
    assert job["profile"]["image_capture"] is True
    ns = {"__name__": "test_image_capsule"}
    exec(compile(service.get(job["id"])["worker"], "runner.py", "exec"), ns)
    assert "class ImageRecorder" in ns["COLLECTION_FILES"]["images.py"]
    assert service.create(image_capture=True)["id"] == job["id"]
    with pytest.raises(ValueError, match="image capture"):
        service.create(image_capture=False)


def test_retargeter_selection_is_frozen_and_blocks_switching_active_session(
    service, monkeypatch
):
    from skynet_app import live_xr

    build = live_xr.build_hand

    def with_layout(robot):
        path, m = build(robot)
        m["hands"] = {"right": {"wrist_joints": list(range(6)), "tips": list(range(5))}}
        m["mimic_joints"] = []
        return path, m

    monkeypatch.setattr(live_xr, "build_hand", with_layout)
    with pytest.raises(ValueError, match="Unknown retargeting"):
        service.create(True, retargeter="unsupported")
    assert not service.list()
    job = service.create(True, retargeter="vector-wrist-joint")
    assert (
        job["profile"]["retargeting"]["revision"]
        == "3846d3fa207165bb0d498145aac8b885a28ea923"
    )
    assert service.create(True, retargeter="vector-wrist-joint")["id"] == job["id"]
    with pytest.raises(ValueError, match="before changing retargeting"):
        service.create(True, retargeter="dexpilot")
    assert len(service.list()) == 1
    service.prepare(job["id"])
    assert service.cluster.calls == 1
    assert any("retargeting-check.py" in args for args in service.cluster.writes)
    assert "class CollectionRetargeting" in service.get(job["id"])["worker"]


def test_retargeter_rejects_coupled_hand_before_creating_session(service, monkeypatch):
    from skynet_app import live_xr

    build = live_xr.build_hand

    def coupled(robot):
        path, manifest = build(robot)
        manifest["mimic_joints"] = ["coupled"]
        return path, manifest

    monkeypatch.setattr(live_xr, "build_hand", coupled)
    with pytest.raises(ValueError, match="coupled joints"):
        service.create(True, retargeter="vector-wrist-joint")
    assert not service.list()
    assert service.cluster.calls == 0


def test_live_progress_does_not_wait_for_unrelated_repository_writer(service):
    """Reproduce a training write blocking the old startup/status paths."""
    from concurrent.futures import ThreadPoolExecutor
    from skynet_app.db_backend import lock_key

    # Hold both the in-process repository mutex and PostgreSQL write lock.
    with service.database._write_lock, service.database.connection() as connection:
        connection.execute("BEGIN")
        connection.execute(
            "SELECT pg_advisory_xact_lock(?)", (lock_key("repository-write"),)
        )
        worker = ThreadPoolExecutor(max_workers=1)
        try:

            def startup():
                job = service.create(True)
                service.update(job["id"], startup_stage="runtime")
                return service.consent(), service.get(job["id"])

            future = worker.submit(startup)
            consent, job = future.result(timeout=3)
            assert consent["accepted"]
            assert job["startup_stage"] == "runtime"
        finally:
            connection.rollback()
    worker.shutdown(wait=True)


def test_status_is_cached_while_remote_check_runs_and_deduplicates(
    service, monkeypatch
):
    import threading

    job = service.create(True)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def slow_refresh(identifier):
        calls.append(identifier)
        entered.set()
        assert release.wait(5)
        service.update(identifier, startup_stage="runtime", error=None)

    monkeypatch.setattr(service, "refresh", slow_refresh)
    try:
        first = service.status(job["id"])
        assert entered.wait(2)
        assert first["startup_stage"] == "server"
        assert service.status(job["id"])["id"] == job["id"]
        assert calls == [job["id"]]
    finally:
        release.set()
        service.executor.shutdown(wait=True)
    assert service.status_pending == set()
    assert service.get(job["id"])["startup_stage"] == "runtime"


def test_status_reschedules_after_failed_check_and_skips_finished_sessions(
    service, monkeypatch
):
    job = service.create(True)
    monkeypatch.setattr(
        service, "refresh", lambda _: (_ for _ in ()).throw(ClusterError("offline"))
    )
    service.status_pending.add(job["id"])
    with pytest.raises(ClusterError):
        service._refresh_status(job["id"])
    assert not service.status_pending
    service.update(job["id"], state="FAILED", scheduler_final=True)
    assert service.status(job["id"])["state"] == "FAILED"
    assert not service.status_pending
