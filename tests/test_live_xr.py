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
def service(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config").mkdir()
    (tmp_path / "ops/xr").mkdir(parents=True)
    (tmp_path / "config/capture_pipelines.json").write_text(
        (root / "config/capture_pipelines.json").read_text()
    )
    (tmp_path / "config/live_xr.json").write_text(
        json.dumps(
            {
                "pipeline_key": "dexverse-shadow-right",
                "cloudxr_runtime": "/coc/flash7/ycho420/tools/skynet-xr/native-5.0.1",
                "gpu_type": "rtx_6000",
                "duration_minutes": 30,
            }
        )
    )
    (tmp_path / "ops/xr/native_session.py").write_text(
        (root / "ops/xr/native_session.py").read_text()
    )
    service = LiveXRService(Database(tmp_path / "db.sqlite"), Cluster(), root=tmp_path)
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
