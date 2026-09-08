import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from skynet_app.cluster_runtime import ClusterError, SubmissionOutcomeUnknown, WORK_ROOT
from skynet_app.database import Database, canonical_json
from skynet_app.live_conversion import FORMAT, LiveConversionService
from skynet_app.live_xr_review import ArrayUnpickler, LiveReviewService

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "convert_dataset", ROOT / "ops/xr/convert_dataset.py"
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.fixture
def conversion(tmp_path, monkeypatch):
    profile = dict(
        execution="workstation",
        gateway="test-gpu",
        work_root="/home/test/skynet",
        repository="/home/test/skynet/repo",
        runtime="/home/test/skynet/env",
        cloudxr_runtime="/home/test/skynet/cloudxr",
        source_revision="a" * 40,
        task="Dexverse-PickCube-v0",
        robot="floating_shadow_right",
        hand="right",
    )
    session = dict(
        id="session",
        state="CAPTURED",
        gateway="test-gpu",
        profile=profile,
        root=profile["work_root"] + "/sessions/session",
        recordings=["recordings/one.pkl", "recordings/two.pkl"],
        recording_checksums={
            "recordings/one.pkl": "a" * 64,
            "recordings/two.pkl": "b" * 64,
        },
    )
    db = Database(tmp_path / "test.sqlite")
    live = SimpleNamespace(root=ROOT, database=db, get=lambda _: session)
    service = LiveConversionService(
        LiveReviewService(live, tmp_path / "reviews"), tmp_path / "conversions"
    )
    monkeypatch.setattr(
        service,
        "target",
        lambda: dict(
            profile,
            execution="slurm",
            gateway="sky2",
            work_root=WORK_ROOT,
            repository=WORK_ROOT + "/repos/dexverse",
            runtime=WORK_ROOT + "/envs/isaac",
            account="overcap",
            partition="overcap",
            local_source=str(tmp_path),
        ),
    )
    monkeypatch.setattr(
        "skynet_app.live_conversion.pinned_converter_digest", lambda _: "c" * 64
    )
    calls = []
    monkeypatch.setattr(service, "dispatch", calls.append)
    return service, session, calls


def test_selection_is_immutable_and_duplicate_clicks_recover_same_job(conversion):
    service, session, calls = conversion
    job = service.create("session", "Cube", [1, 0])
    assert job["indices"] == [0, 1]
    assert service.create("session", "Different name", [0, 1])["id"] == job["id"]
    assert len(service.list()) == 1
    assert service.create("session", "One recording", [0])["id"] != job["id"]
    assert job["sources"][0]["sha256"] == "a" * 64
    assert calls.count(job["id"]) == 2


@pytest.mark.parametrize("selection", [[], [0, 0], [-1], [2], [True], [0.5]])
def test_invalid_recording_selection_never_starts_work(conversion, selection):
    service, _, calls = conversion
    with pytest.raises(ValueError, match="valid recordings"):
        service.create("session", "Cube", selection)
    assert calls == []


def test_active_collection_and_unverified_source_cannot_convert(conversion):
    service, session, calls = conversion
    session["state"] = "COLLECTING"
    with pytest.raises(ValueError, match="End collection"):
        service.create("session", "Cube")
    session["state"] = "CAPTURED"
    session["recording_checksums"] = {}
    with pytest.raises(ValueError, match="checksum"):
        service.create("session", "Cube")
    assert calls == []


def test_changed_sources_produce_new_version_but_failed_attempts_can_retry(conversion):
    service, session, _ = conversion
    original = service.create("session", "Cube")
    service.update(original["id"], state="FAILED")
    retried = service.create("session", "Cube")
    assert retried["id"] != original["id"]
    assert retried["fingerprint"] == original["fingerprint"]
    session["recording_checksums"]["recordings/two.pkl"] = "c" * 64
    assert service.create("session", "Cube")["fingerprint"] != original["fingerprint"]


def test_ssh_loss_does_not_report_remote_work_failed(conversion):
    service, _, _ = conversion
    job = service.create("session", "Cube")
    service.update(job["id"], job_id="1234")

    def offline(*a, **kw):
        raise ClusterError("SSH timed out")

    service.cluster = SimpleNamespace(job_statuses=offline)
    service.work(job["id"])
    state = service.get(job["id"])
    assert state["state"] == "PREPARING"
    assert "timed out" in state["connection_error"]
    assert service.create("session", "Cube")["id"] == job["id"]


def test_launch_capsule_is_stable_after_an_uncertain_acknowledgement(conversion):
    service, session, _ = conversion
    job = service.create("session", "Cube")
    service.update(job["id"], staged=True)
    submissions = []

    def submit(script, identifier, gateway, **kwargs):
        submissions.append((script, identifier, gateway, kwargs))
        if len(submissions) == 1:
            raise SubmissionOutcomeUnknown("Lost acknowledgement")
        return SimpleNamespace(job_id="1234")

    service.cluster = SimpleNamespace(
        submit_script=submit,
        job_statuses=lambda *_: (
            "sky2",
            {"1234": {"State": "PENDING", "Reason": "Resources"}},
        ),
        ssh=lambda *a, **k: "{}",
    )
    service.work(job["id"])
    assert service.get(job["id"])["state"] == "SUBMISSION_UNKNOWN"
    service.work(job["id"])
    assert service.get(job["id"])["state"] == "QUEUED"
    assert submissions[0] == submissions[1]
    script = submissions[0][0]
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --partition=overcap" in script
    assert "convert_dataset.py" in script
    assert "flock" not in script and "systemd" not in script
    assert "training" not in script and "evaluation" not in script
    request = json.loads((service.root / job["id"] / "request.json").read_text())
    assert request["root"] == job["dataset_root"]
    assert request["root"].startswith(
        WORK_ROOT + "/datasets/derivatives/dexverse-live/"
    )
    assert all(
        s["path"].startswith(job["root"] + "/recordings/")
        for s in request["staged_sources"]
    )
    assert request["sources"][0]["path"].startswith(session["root"])
    assert request["profile"]["robot"] == session["profile"]["robot"]
    assert request["profile"]["task"] == session["profile"]["task"]
    assert request["profile"]["execution"] == "slurm"


def test_conversion_gpu_uses_existing_cluster_aliases(conversion):
    from skynet_app.capture_processing.slurm import compile_isaac_job

    service, _, _ = conversion
    profile = dict(service.target(), gpu_type="rtx_6000")
    script = compile_isaac_job(profile, WORK_ROOT + "/jobs/runs/test", "test", ["true"])
    assert "#SBATCH --gres=gpu:rtx_6000:1" in script
    profile["gpu_type"] = "made-up-gpu"
    with pytest.raises(ValueError, match="GPU type is not configured"):
        compile_isaac_job(profile, WORK_ROOT + "/jobs/runs/test", "test", ["true"])


def metadata(job, raw):
    return dict(
        format=FORMAT,
        task=job["profile"]["task"],
        robot=job["profile"]["robot"],
        converter_sha256=job["converter_sha256"],
        sources=job["sources"],
        episodes=2,
        steps=20,
        artifact=dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)),
    )


def test_verified_download_registers_dataset_bundle_and_lineage_once(conversion):
    service, _, _ = conversion
    job = service.create("session", "Shadow cube")
    raw = b"\x89HDF\r\n\x1a\n" + b"test download transport" * 10
    m = metadata(job, raw)
    manifest = canonical_json(m).encode()
    files = {
        job["dataset_root"] + "/dataset.hdf5": raw,
        job["dataset_root"] + "/manifest.json": manifest,
    }
    transport = SimpleNamespace(
        file_size=lambda path, host: (host, len(files[path])),
        stream_file_range=lambda path, host, **_: iter(
            [files[path][:10], files[path][10:]]
        ),
    )
    result = dict(
        metadata=m,
        manifest=dict(
            sha256=hashlib.sha256(manifest).hexdigest(), size_bytes=len(manifest)
        ),
    )
    service.finish(job, result, transport)
    first = service.get(job["id"])
    service.finish(job, result, transport)
    second = service.get(job["id"])
    assert first["state"] == "READY"
    assert first["bundle_id"] == second["bundle_id"]
    version = service.database.get_data_resource_version(first["version_id"])
    assert version["format"] == FORMAT
    assert version["metadata"]["storage_location"] == "cluster"
    assert version["path"] == job["dataset_root"] + "/dataset.hdf5"
    assert version["derivation_id"]
    assert len(service.database.list_data_bundles()) == 1
    assert service.artifact(job["id"], "dataset.hdf5").read_bytes() == raw
    with pytest.raises(KeyError):
        service.artifact(job["id"], "../worker-sources.json")


def test_truncated_download_is_never_published(conversion):
    service, _, _ = conversion
    job = service.create("session", "Cube")
    raw = b"0123456789"
    transport = SimpleNamespace(
        file_size=lambda path, host: (host, len(raw)),
        stream_file_range=lambda *a, **k: iter([raw[:4]]),
    )
    with pytest.raises(OSError, match="Incomplete"):
        service.download(
            job,
            "dataset.hdf5",
            dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=10),
            transport,
        )
    assert not (service.root / job["id"] / "dataset.hdf5").exists()
    assert not (service.root / job["id"] / "dataset.part").exists()


@pytest.mark.parametrize(
    "state", ["TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "FAILED", "PREEMPTED"]
)
def test_queue_exit_and_worker_errors_are_explicit(conversion, state):
    service, _, _ = conversion
    job = service.create("session", "Cube")
    service.update(job["id"], job_id="1234", launched=True)
    service.cluster = SimpleNamespace(
        job_statuses=lambda *_: ("sky2", {"1234": {"State": state}}),
        ssh=lambda *a, **k: "{}",
    )
    service.work(job["id"])
    assert service.get(job["id"])["state"] == "FAILED"
    assert state in service.get(job["id"])["error"]


def test_ready_worker_is_not_published_before_slurm_completion(conversion):
    service, _, _ = conversion
    job = service.create("session", "Cube")
    service.update(job["id"], job_id="1234")
    service.cluster = SimpleNamespace(
        job_statuses=lambda *_: ("sky2", {"1234": {"State": "RUNNING"}}),
        ssh=lambda *a, **k: '{"state":"READY"}',
    )
    service.work(job["id"])
    assert service.get(job["id"])["state"] == "RUNNING"
    assert service.database.list_data_resources() == []


def test_missing_result_is_not_a_successful_dataset(conversion, monkeypatch):
    service, _, _ = conversion
    job = service.create("session", "Cube")
    service.update(job["id"], job_id="1234")
    service.cluster = SimpleNamespace(
        job_statuses=lambda *_: ("sky2", {"1234": {"State": "COMPLETED"}}),
        ssh=lambda *a, **k: "{}",
    )
    monkeypatch.setattr("skynet_app.live_conversion.time.time", lambda: 1000)
    service.work(job["id"])
    assert service.get(job["id"])["state"] == "RUNNING"
    assert service.database.list_data_resources() == []
    monkeypatch.setattr("skynet_app.live_conversion.time.time", lambda: 1121)
    service.work(job["id"])
    assert service.get(job["id"])["state"] == "FAILED"
    assert "without a verified dataset" in service.get(job["id"])["error"]


def test_shared_storage_delay_recovers_same_completed_job(conversion, monkeypatch):
    service, _, _ = conversion
    job = service.create("session", "Cube")
    service.update(job["id"], job_id="1234")
    result = {}
    service.cluster = SimpleNamespace(
        job_statuses=lambda *_: ("sky2", {"1234": {"State": "COMPLETED"}}),
        ssh=lambda *a, **k: json.dumps(result),
    )
    service.work(job["id"])
    assert service.get(job["id"])["state"] == "RUNNING"
    result["state"] = "READY"
    completed = []
    monkeypatch.setattr(service, "finish", lambda j, *_: completed.append(j["job_id"]))
    service.work(job["id"])
    assert completed == ["1234"]
    assert len(service.list()) == 1


def test_corrupt_cached_input_cannot_be_submitted(conversion):
    service, _, _ = conversion
    job = service.create("session", "Cube")
    directory = service.reviews.directory("session", 0)
    directory.mkdir(parents=True)
    (directory / "recording.pkl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum changed"):
        service.local_recording(job, job["sources"][0])


def test_input_copy_failure_can_retry_without_submitting_a_gpu_job(
    conversion, monkeypatch
):
    service, _, _ = conversion
    job = service.create("session", "Cube")
    calls = []
    monkeypatch.setattr(
        service.reviews,
        "status",
        lambda *_: {"state": "FAILED", "error": "Source server unavailable"},
    )
    monkeypatch.setattr(service.reviews, "create", lambda *args: calls.append(args))
    assert service.local_recording(job, job["sources"][0]) is None
    job = service.get(job["id"])
    with pytest.raises(ValueError, match="Source server unavailable"):
        service.local_recording(job, job["sources"][0])
    assert calls == [("session", 0)]
    assert not job.get("job_id")


@pytest.fixture
def native(tmp_path):
    payload = dict(
        format="dexverse_trajectory",
        schema_version=3,
        task="Dexverse-PickCube-v0",
        robot_type="floating_shadow_right",
        num_episodes=1,
        episodes=[
            dict(
                num_steps=3,
                success=True,
                actions=np.arange(84, dtype=np.float32).reshape(3, 28),
                states=[
                    {
                        "articulation": {
                            "robot": {
                                "joint_position": np.full((1, 28), i, dtype=np.float32)
                            }
                        },
                        "deformable_object": {},
                    }
                    for i in range(4)
                ],
            )
        ],
    )
    p = tmp_path / "recording.pkl"

    def load(edit=lambda p: None):
        value = copy.deepcopy(payload)
        edit(value)
        p.write_bytes(pickle.dumps(value))
        return worker.load_recording(
            dict(path=str(p), **worker.file_info(p)),
            dict(task=payload["task"], robot=payload["robot_type"]),
            ArrayUnpickler,
        )

    return load


def test_worker_preserves_original_actions_and_rejects_incomplete_frames(native):
    assert np.array_equal(
        native()["episodes"][0]["actions"],
        np.arange(84, dtype=np.float32).reshape(3, 28),
    )
    with pytest.raises(ValueError, match="T actions"):
        native(lambda p: p["episodes"][0]["states"].pop())
    with pytest.raises(ValueError, match="Nonfinite"):
        native(
            lambda p: p["episodes"][0]["states"][1]["articulation"]["robot"][
                "joint_position"
            ].fill(np.nan)
        )
    with pytest.raises(ValueError, match="hand/task"):
        native(lambda p: p.update(robot_type="floating_shadow_left"))


def test_snapshot_copies_reused_simulator_buffers():
    buffer = np.ones(28)
    capture = SimpleNamespace(capture=lambda _: {"proprio/joint_pos": buffer})
    snapshot = worker.snapshot(capture)
    buffer.fill(9)
    assert np.all(snapshot["proprio/joint_pos"] == 1)
    buffer[0] = np.nan
    with pytest.raises(ValueError, match="invalid observations"):
        worker.snapshot(capture)


def test_hdf5_checks_episode_boundaries_and_observation_action_timing(tmp_path):
    import h5py

    path = tmp_path / "dataset.hdf5"
    episodes = [
        dict(num_steps=n, actions=np.arange(n * 2, dtype=np.float32).reshape(n, 2))
        for n in (1, 4, 2)
    ]
    with h5py.File(path, "w") as h5:
        for i, ep in enumerate(episodes):
            g = h5.create_group(f"data/demo_{i}")
            n = ep["num_steps"]
            frames = np.arange((n + 1) * 3, dtype=np.float32).reshape(n + 1, 3)
            g["actions"] = ep["actions"]
            g["obs/proprio/joint_pos"] = frames[:-1]
            g["next_obs/proprio/joint_pos"] = frames[1:]
            g["initial_obs/proprio/joint_pos"] = frames[0]
            g["final_obs/proprio/joint_pos"] = frames[-1]
    shapes = {"proprio/joint_pos": [3]}
    worker.validate_hdf5(path, episodes, shapes)
    with h5py.File(path, "a") as h5:
        h5["data/demo_1/next_obs/proprio/joint_pos"][0] = [99, 99, 99]
    with pytest.raises(ValueError, match="misaligned"):
        worker.validate_hdf5(path, episodes, shapes)
    with h5py.File(path, "a") as h5:
        del h5["data/demo_2"]
    with pytest.raises(ValueError, match="episode count"):
        worker.validate_hdf5(path, episodes, shapes)
