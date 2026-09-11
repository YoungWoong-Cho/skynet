import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4
import pytest
from skynet_app.capture_processing.visionpro import (
    convert,
    JOINTS,
    calibrated_basis,
    quaternion,
)
from skynet_app.database import Database
from skynet_app.local_capture import LocalCaptureService
from capture_storage_fake import MemoryStorage
from skynet_app.capture_processing.service import ProcessingService


def pose(x=0, y=0, z=0):
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]


def recording(tmp_path, hand="right"):
    header = {
        "type": "header",
        "schema": "skynet.visionpro-tracking/v1",
        "session_id": str(uuid4()),
        "task": "Synthetic conversion regression",
        "units": "meters",
        "matrix_order": "column-major",
        "clock": "arkit-monotonic-seconds",
        "robot_actions": False,
        "images_recorded": False,
        "timestamp_clock": "CACurrentMediaTime",
        "joint_frame": "hand-anchor",
        "coordinate_frame": "ARKit session origin; right-handed; Y up",
    }
    positions = {name: (0, 0, -0.1) for name in JOINTS.values()}
    positions.update(
        wrist=(0, 0, 0),
        middleFingerKnuckle=(0, 0, -0.08),
        indexFingerKnuckle=(-0.03 if hand == "right" else 0.03, 0, -0.08),
        littleFingerKnuckle=(0.03 if hand == "right" else -0.03, 0, -0.08),
    )
    records = [header]
    for i in range(9):
        records.append(
            {
                "type": "frame",
                "index": i,
                "timestamp": 10 + i / 60,
                "source_timestamp": 100 + i / 60,
                "hand": hand,
                "tracked": True,
                "head_tracked": False,
                "origin_from_hand": pose(i / 100, 0, 0),
                "joints": [
                    {"name": name, "tracked": True, "anchor_from_joint": pose(*xyz)}
                    for name, xyz in positions.items()
                ],
            }
        )
    records.append({"type": "footer", "frames": 9, "completed": True})
    path = tmp_path / "capture.jsonl"

    def save():
        path.write_text("\n".join(json.dumps(v) for v in records) + "\n")
        return path

    return records, save


def test_real_transform_math_and_chirality(tmp_path):
    for side in ("left", "right"):
        records, save = recording(tmp_path, side)
        basis = calibrated_basis(records[1])
        assert basis == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = convert(save(), hand=side)
        first = result["samples"][0]["joints"]["wrist"]
        assert first[:3] == [0, 0, 0]
        # ARKit +X maps to simulator -Y; a later sample retains real displacement.
        assert result["samples"][2]["joints"]["wrist"][:3] == pytest.approx(
            [0, -0.02, 0]
        )
        assert math.sqrt(sum(v * v for v in first[3:])) == pytest.approx(1)


def test_resampling_uses_receive_clock_without_future_frames(tmp_path):
    _, save = recording(tmp_path)
    path = save()
    original = path.read_bytes()
    result = convert(path, fps=30)
    assert [s["source_index"] for s in result["samples"]] == [0, 2, 4, 6, 8]
    assert result["samples"][0]["source_timestamp"] == 100
    assert path.read_bytes() == original
    assert result["source_sha256"] == hashlib.sha256(original).hexdigest()


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda r: r[0].update(timestamp_clock="unknown"),
            "Unsupported timestamp_clock",
        ),
        (lambda r: r[2].update(tracked=False), "tracking lost"),
        (lambda r: r[2]["joints"].pop(), "missing/untracked"),
        (lambda r: r[2]["joints"][1].update(tracked=False), "missing/untracked"),
        (lambda r: r[2].update(origin_from_hand=[0] * 16), "homogeneous"),
        (lambda r: r[2]["origin_from_hand"].__setitem__(0, -1), "right-handed"),
        (lambda r: r[2]["origin_from_hand"].__setitem__(0, 2), "right-handed"),
    ],
)
def test_invalid_tracking_never_silently_falls_back(tmp_path, mutation, message):
    records, save = recording(tmp_path)
    mutation(records)
    with pytest.raises(ValueError, match=message):
        convert(save())


def test_gap_rejected_instead_of_stretched(tmp_path):
    records, save = recording(tmp_path)
    for record in records[5:-1]:
        record["timestamp"] += 0.3
    with pytest.raises(ValueError, match="gap"):
        convert(save())


@pytest.mark.parametrize(
    "m,expected",
    [
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 0, 0, 0]),
        ([[1, 0, 0], [0, -1, 0], [0, 0, -1]], [0, 1, 0, 0]),
        ([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [0, 0, 1, 0]),
        ([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [0, 0, 0, 1]),
    ],
)
def test_rotation_near_half_turn(m, expected):
    assert quaternion(m) == pytest.approx(expected)


def processor(tmp_path, monkeypatch):
    captures = LocalCaptureService(Database(tmp_path / "test.db"), storage=MemoryStorage())
    _, save = recording(tmp_path)
    capture = captures.import_file("visionpro-local", save())["capture"]
    config = Path(__file__).resolve().parents[1] / "config/capture_pipelines.json"
    service = ProcessingService(captures, config_path=config)
    monkeypatch.setattr(service, "dispatch", lambda identifier: None)
    return service, capture


def test_job_request_deduplicates_and_retains_snapshot(tmp_path, monkeypatch):
    service, capture = processor(tmp_path, monkeypatch)
    first = service.create(capture["sha256"])
    second = service.create(capture["sha256"])
    assert first["id"] == second["id"]
    assert len(service.list()) == 1
    assert "runner_source" not in first
    assert service.get(first["id"], private=True)["runner_source"]
    assert service.create(capture["sha256"], seed=1)["id"] != first["id"]


def test_bad_provider_and_parameters_are_explicit(tmp_path, monkeypatch):
    service, capture = processor(tmp_path, monkeypatch)
    for kwargs in (
        {"pipeline_key": "unknown"},
        {"epochs": 0},
        {"seed": True},
        {"eval_episodes": 21},
    ):
        with pytest.raises(ValueError):
            service.create(capture["sha256"], **kwargs)


def test_artifact_validation_rejects_path_escape_and_incomplete_result(
    tmp_path, monkeypatch
):
    service, capture = processor(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="missing"):
        service.verify_artifacts(
            {},
            {
                "artifacts": {},
                "dataset": {
                    "schema": "skynet.dexverse-state-actions/v1",
                    "task": "Dexverse-PickUpStick-v0",
                    "robot": "floating_shadow_right",
                },
                "stages": {
                    s: {"status": "SUCCEEDED"}
                    for s in ("simulation", "training", "evaluation")
                },
            },
        )
    artifacts = {
        name: {"sha256": "a" * 64, "size_bytes": 1}
        for name in (
            "dataset.hdf5",
            "dataset-manifest.json",
            "state-bc.pt",
            "replay.mp4",
            "evaluation-0.mp4",
            "../secret",
        )
    }
    with pytest.raises(ValueError, match="name"):
        service.verify_artifacts(
            {},
            {
                "artifacts": artifacts,
                "dataset": {
                    "schema": "skynet.dexverse-state-actions/v1",
                    "task": "Dexverse-PickUpStick-v0",
                    "robot": "floating_shadow_right",
                    "dataset_sha256": "a" * 64,
                },
                "stages": {
                    s: {"status": "SUCCEEDED"}
                    for s in ("simulation", "training", "evaluation")
                },
            },
        )


def test_byte_ranges():
    from skynet_app.capture_processing.api import byte_range

    assert byte_range(None, 100) == (0, 99, False)
    assert byte_range("bytes=20-49", 100) == (20, 49, True)
    assert byte_range("bytes=20-", 100) == (20, 99, True)
    assert byte_range("bytes=-10", 100) == (90, 99, True)
    for value in ("bytes=-0", "bytes=100-", "bytes=90-20", "bytes=", "bytes=0-1,3-4"):
        with pytest.raises(ValueError):
            byte_range(value, 100)


class StatusCluster:
    def __init__(self, state="COMPLETED", control=None, error=None):
        self.state = state
        self.control = control or {}
        self.error = error

    def job_statuses(self, ids, gateway):
        if self.error:
            raise self.error
        return gateway, {
            ids[0]: {"State": self.state, "Reason": "Priority", "ExitCode": "0:0"}
        }

    def ssh(self, *args, **kwargs):
        return json.dumps(self.control)


def submitted(service, capture):
    job = service.create(capture["sha256"])
    service.update(
        job["id"],
        state="PENDING",
        job_id="12345",
        root="/coc/flash7/ycho420/jobs/runs/" + job["id"],
    )
    return job


def test_zero_exit_without_result_is_failed(tmp_path, monkeypatch):
    service, capture = processor(tmp_path, monkeypatch)
    job = submitted(service, capture)
    service.cluster = StatusCluster()
    result = service.refresh(job["id"], force=True)
    assert result["state"] == "FAILED"
    assert "without a valid completed" in result["error"]
    assert service.database.list_data_resources(namespace="dexverse-offline") == []


def test_connection_failure_preserves_last_known_state(tmp_path, monkeypatch):
    from skynet_app.cluster_runtime import ClusterError

    service, capture = processor(tmp_path, monkeypatch)
    job = submitted(service, capture)
    service.cluster = StatusCluster(error=ClusterError("network disconnected"))
    result = service.refresh(job["id"], force=True)
    assert result["state"] == "PENDING"
    assert result["refresh_error"] == "network disconnected"
    assert not result["error"]


def test_gpu_failure_preserves_stage_error(tmp_path, monkeypatch):
    service, capture = processor(tmp_path, monkeypatch)
    job = submitted(service, capture)
    service.cluster = StatusCluster(
        "FAILED",
        {
            "error.json": {"error": "Required joint was not tracked"},
            "progress.json": {"stages": {"simulation": {"status": "FAILED"}}},
        },
    )
    result = service.refresh(job["id"], force=True)
    assert result["state"] == "FAILED"
    assert result["error"] == "Required joint was not tracked"
    assert result["stages"]["simulation"]["status"] == "FAILED"


def test_lost_submission_ack_is_not_a_failed_or_second_job(tmp_path, monkeypatch):
    from skynet_app.cluster_runtime import SubmissionOutcomeUnknown
    from skynet_app.capture_processing import service as module

    service, capture = processor(tmp_path, monkeypatch)

    class LostAck:
        _remote_path = staticmethod(lambda path: path)

        def ssh(self, *args, **kwargs):
            return ""

        def write_capsule_file(self, *args, **kwargs):
            return "sky2", "test"

        def submit_script(self, *args, **kwargs):
            raise SubmissionOutcomeUnknown("Acknowledgement lost")

    service.cluster = LostAck()
    monkeypatch.setattr(module, "upload_capture", lambda *args: None)
    job = service.create(capture["sha256"])
    service._prepare(job["id"])
    result = service.get(job["id"])
    assert result["state"] == "SUBMISSION_UNKNOWN"
    assert not result.get("job_id")
    assert service.get(job["id"], private=True)["script"]


def test_dataset_lineage_recovers_after_partial_registration(tmp_path, monkeypatch):
    service, capture = processor(tmp_path, monkeypatch)
    job = submitted(service, capture)
    job = service.get(job["id"])
    result = {
        "dataset": {
            "schema": "skynet.dexverse-state-actions/v1",
            "dataset_sha256": "a" * 64,
        },
        "artifacts": {
            "dataset-manifest.json": {"sha256": "b" * 64},
            "dataset.hdf5": {"size_bytes": 100},
        },
    }
    original = service.database.create_data_derivation
    monkeypatch.setattr(
        service.database,
        "create_data_derivation",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    with pytest.raises(RuntimeError):
        service.register_dataset(job, result)
    monkeypatch.setattr(service.database, "create_data_derivation", original)
    version = service.register_dataset(job, result)
    assert service.register_dataset(job, result)["id"] == version["id"]
    derivations = service.database.list_data_derivations()
    assert len(derivations) == 1
    assert derivations[0]["inputs"][0]["version_id"] == capture["version_id"]
