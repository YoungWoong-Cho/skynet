from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from skynet_app.cluster_runtime import Submission
from skynet_app.collection import (
    CollectionAdapterManifest,
    CollectionCompleteRequest,
    CollectionService,
    CollectionSessionCreate,
    CollectionValidationError,
)
from skynet_app.database import Database


class FakeCluster:
    def __init__(self) -> None:
        self.script = ""
        self.state = "PENDING"
        self.submit_count = 0

    def test_script(self, script, gateway="auto"):
        self.script = script
        return "sky2", "Job can run"

    def submit_script(self, script, run_id, gateway="auto"):
        self.script = script
        self.submit_count += 1
        return Submission(
            "8123",
            "8123",
            "sky2",
            f"/coc/flash7/ycho420/jobs/runs/{run_id}/attempts/8123/job.sbatch",
            f"/coc/flash7/ycho420/jobs/runs/{run_id}",
        )

    def job_statuses(self, job_ids, gateway="auto"):
        return "sky2", {
            str(job_id): {
                "State": self.state,
                "ExitCode": "0:0",
                "Reason": "None",
                "NodeList": "bishop",
            }
            for job_id in job_ids
        }

    def read_log(self, path, gateway="auto", **kwargs):
        return "sky2", f"log:{path}"

    def cancel(self, job_id, gateway="auto"):
        return "sky2"


def adapter_manifest(*, display_name="Test collector", runnable=True):
    return CollectionAdapterManifest.model_validate(
        {
            "key": "test-collector",
            "display_name": display_name,
            "version": "1.0.0",
            "runnable": runnable,
            "streams": [
                {
                    "name": "action",
                    "kind": "action",
                    "shape": [7],
                    "dtype": "float32",
                    "units": "normalized",
                    "frame": "robot-base",
                    "rate_hz": 30,
                    "native_key": "trajectory/action",
                },
                {
                    "name": "proprio",
                    "kind": "proprio",
                    "shape": [14],
                    "dtype": "float32",
                    "units": "mixed",
                    "frame": "robot-base",
                    "rate_hz": 30,
                    "native_key": "trajectory/observation/state",
                },
                {
                    "name": "rgb",
                    "kind": "sensor",
                    "shape": [480, 640, 3],
                    "dtype": "uint8",
                    "units": None,
                    "frame": "wrist-camera",
                    "rate_hz": 30,
                    "native_key": "trajectory/observation/wrist_rgb",
                },
            ],
            "requirements": [
                {
                    "scope": "config",
                    "key": "task",
                    "description": "task is selected",
                }
            ],
            "launcher": {
                "environment": {"COLLECTION_TASK": "{{config.task}}"},
                "steps": [
                    {
                        "name": "capture",
                        "argv": [
                            "python3",
                            "capture.py",
                            "--output",
                            "{{storage.output_path}}",
                        ],
                        "cwd": "/coc/flash7/ycho420/workspace",
                    }
                ],
            },
        }
    )


def session_request(adapter_id):
    return CollectionSessionCreate.model_validate(
        {
            "adapter_id": adapter_id,
            "name": "pick-stick-50",
            "runtime_profile": "isaacsim-5.1.0_isaaclab-2.3.2_py311",
            "config": {"task": "PickStick-v0"},
            "software": {"repository_commit": "a" * 40},
            "calibration": {
                "identity": "cal-001",
                "sha256": "c" * 64,
                "metadata": {"robot": "test-arm"},
            },
            "storage": {
                "output_path": "/coc/flash7/ycho420/datasets/staging/test/pick-stick-50",
                "native_format": "test-native-pickle",
                "registration": {
                    "provider": "collection",
                    "namespace": "tests",
                    "name": "pick-stick",
                    "kind": "demonstrations",
                },
            },
            "capture": {
                "schema_name": "test_trajectory",
                "schema_version": "1",
                "clock_source": "simulation-step",
                "nominal_rate_hz": 30,
                "timestamp_unit": "step-index",
                "alignment": "action_then_post_state",
                "timestamps_recorded": False,
            },
            "resources": {
                "account": "rl2-lab",
                "partition": "rl2-lab",
                "gpu_count": 1,
                "gpu_type": "l40s",
                "cpu_count": 16,
                "memory_gb": 64,
                "time_limit": "01:00:00",
            },
        }
    )


def test_adapter_edit_archive_and_session_snapshot_are_independent():
    with tempfile.TemporaryDirectory() as directory:
        service = CollectionService(
            Database(Path(directory) / "skynet.db"), FakeCluster(), seed=False
        )
        adapter = service.store.create_adapter(adapter_manifest())
        session = service.store.create_session(session_request(adapter["id"]))
        original_hash = session["manifest_sha256"]
        original_adapter_hash = session["adapter_snapshot"]["manifest_sha256"]

        edited = service.store.update_adapter(
            adapter["id"], adapter_manifest(display_name="Edited collector")
        )
        archived = service.store.set_adapter_archived(adapter["id"], True)
        loaded = service.store.get_session(session["id"])

        assert edited["manifest_sha256"] != original_adapter_hash
        assert archived["archived_at"] is not None
        assert loaded["manifest_sha256"] == original_hash
        assert loaded["adapter_snapshot"]["manifest_sha256"] == original_adapter_hash
        assert loaded["adapter_snapshot"]["manifest"]["display_name"] == "Test collector"
        with pytest.raises(ValueError, match="archived"):
            service.store.create_session(session_request(adapter["id"]))


def test_prepare_submit_refresh_register_native_and_restart():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = CollectionService(database, cluster, seed=False)
        adapter = service.store.create_adapter(adapter_manifest())
        session = service.store.create_session(session_request(adapter["id"]))

        prepared = service.prepare(session["id"])
        script = prepared["sbatch"]["script"]
        assert prepared["session"]["status"] == "READY"
        assert "#SBATCH --account=rl2-lab" in script
        assert "#SBATCH --gres=gpu:l40s:1" in script
        assert "export LD_LIBRARY_PATH=/coc/flash7/ycho420/envs/isaacsim-5.1.0_isaaclab-2.3.2_py311/lib\"${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH:-}\"" in script
        assert "Raw capture stays in its adapter-native format" in script
        assert "trajectory/action" in prepared["session"]["adapter_snapshot"]["manifest"]["streams"][0]["native_key"]

        submitted = service.submit(session["id"])
        assert submitted["session"]["status"] == "SUBMITTED"
        assert cluster.submit_count == 1
        assert submitted["session"]["restart_of_session_id"] is None
        assert "8123" in submitted["session"]["stderr_path"]

        cluster.state = "RUNNING"
        assert service.refresh(session["id"])["status"] == "RUNNING"
        cluster.state = "COMPLETED"
        captured = service.refresh(session["id"])
        assert captured["status"] == "CAPTURED"

        result = service.complete(
            session["id"],
            CollectionCompleteRequest(manifest_sha256="b" * 64, size_bytes=1234),
        )
        assert result["session"]["status"] == "COMPLETED"
        assert result["version"]["path"] == captured["storage_snapshot"]["output_path"]
        assert result["version"]["format"] == "test-native-pickle"
        assert result["version"]["metadata"]["native_raw_preserved"] is True
        assert result["version"]["metadata"]["calibration"]["sha256"] == "c" * 64
        assert result["version"]["metadata"]["capture"]["schema_name"] == "test_trajectory"
        assert result["version"]["source_uri"] == f"collection:{session['id']}"

        restarted = service.store.restart_session(session["id"])
        assert restarted["restart_of_session_id"] == session["id"]
        assert restarted["adapter_snapshot"] == session["adapter_snapshot"]
        assert restarted["status"] == "DRAFT"


def test_preflight_reports_missing_declared_requirement_and_secret_values_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        service = CollectionService(
            Database(Path(directory) / "skynet.db"), FakeCluster(), seed=False
        )
        adapter = service.store.create_adapter(adapter_manifest())
        payload = session_request(adapter["id"]).model_dump(mode="python")
        payload["config"] = {}
        session = service.store.create_session(CollectionSessionCreate.model_validate(payload))

        report = service.preflight(session["id"])
        assert report["ok"] is False
        assert service.store.get_session(session["id"])["status"] == "PREFLIGHT_FAILED"
        with pytest.raises(CollectionValidationError):
            service.prepare(session["id"], remote_validate=False)

        payload["software"] = {"ngc_token": "plaintext"}
        with pytest.raises(ValueError, match="secret reference"):
            CollectionSessionCreate.model_validate(payload)

        payload = session_request(adapter["id"]).model_dump(mode="python")
        payload["calibration"]["sha256"] = "bad"
        with pytest.raises(ValueError, match="sha256"):
            CollectionSessionCreate.model_validate(payload)


def test_declarative_capabilities_report_unknown_and_admin_required_without_claiming_reachability():
    with tempfile.TemporaryDirectory() as directory:
        service = CollectionService(
            Database(Path(directory) / "skynet.db"), FakeCluster(), seed=False
        )
        manifest = adapter_manifest().model_dump(mode="python")
        manifest["key"] = "capability-test"
        manifest["capabilities"] = [
            {
                "id": "container-daemon",
                "kind": "container_daemon",
                "scope": "compute_node",
                "description": "Container daemon is usable in the allocation.",
                "default_status": "UNKNOWN",
            },
            {
                "id": "stream-ports",
                "kind": "network_ports",
                "scope": "network",
                "description": "Streaming ports are configured by the administrator.",
                "default_status": "ADMIN_REQUIRED",
                "tcp_ports": [48010],
                "udp_ports": [47998, 47999],
            },
        ]
        adapter = service.store.create_adapter(
            CollectionAdapterManifest.model_validate(manifest)
        )
        request = session_request(adapter["id"])
        session = service.store.create_session(request)
        report = service.preflight(session["id"])

        statuses = {item["name"]: item["status"] for item in report["checks"]}
        assert report["ok"] is False
        assert statuses["capability.container-daemon"] == "UNKNOWN"
        assert statuses["capability.stream-ports"] == "ADMIN_REQUIRED"

        payload = request.model_dump(mode="python")
        payload["capabilities"] = {
            "container-daemon": {
                "status": "verified",
                "scope": "compute_node",
                "verified_by": "cluster-admin",
                "checked_at": "2026-09-01T12:00:00Z",
            },
            "stream-ports": {
                "status": "verified",
                "scope": "network",
                "verified_by": "network-admin",
                "checked_at": "2026-09-01T12:00:00Z",
            },
        }
        verified = service.store.create_session(CollectionSessionCreate.model_validate(payload))
        verified_report = service.preflight(verified["id"])
        assert verified_report["ok"] is True
        port_check = next(
            item
            for item in verified_report["checks"]
            if item["name"] == "capability.stream-ports"
        )
        assert "reachability was not probed" in port_check["message"]

        payload["capabilities"]["typo-capability"] = {
            "status": "unknown",
            "scope": "operator",
        }
        with pytest.raises(ValueError, match="undeclared capability"):
            service.store.create_session(CollectionSessionCreate.model_validate(payload))


def test_dexverse_seed_has_pinned_native_layouts_and_manual_recorder_template():
    with tempfile.TemporaryDirectory() as directory:
        service = CollectionService(
            Database(Path(directory) / "skynet.db"), FakeCluster(), seed=True
        )
        adapter = service.store.adapter_by_key("dexverse-cloudxr")
        assert adapter is not None
        manifest = adapter["manifest"]
        assert manifest["runnable"] is False
        streams = {item["name"]: item for item in manifest["streams"]}
        assert streams["shadow_right_action"]["shape"] == [28]
        assert streams["shadow_left_action"]["shape"] == [28]
        assert streams["shadow_bimanual_action"]["shape"] == [56]
        assert streams["scene_initial_state"]["dynamic"] is True
        assert streams["scene_post_step_states"]["dynamic"] is True
        recorder = next(
            step["argv"]
            for step in manifest["launcher"]["operator_steps"]
            if "/workspace/dexverse/scripts/record_demos.py" in step["argv"]
        )
        assert "/workspace/dexverse/scripts/record_demos.py" in recorder
        assert "--enable_pinocchio" in recorder
        capability_ids = {item["id"] for item in manifest["capabilities"]}
        assert {
            "container-runtime",
            "container-daemon",
            "allocated-gpu",
            "cloudxr-port-configuration",
            "operator-gui",
            "stream-endpoint",
        } <= capability_ids
