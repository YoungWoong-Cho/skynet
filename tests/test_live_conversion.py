import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
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


class LocalCluster:
    """Run the exact remote verification scripts against an isolated filesystem."""

    def __init__(self, root):
        self.root = root
        self.submissions = []

    def path(self, remote):
        return self.root / remote.lstrip("/")

    def write(self, remote, value):
        path = self.path(remote)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    def ssh(self, gateway, command, *, stdin, timeout):
        request = json.loads(stdin)
        if "repository" in request:
            return canonical_json(dict(runtime=True, converter=True, revision=request["source_revision"]))
        request["root"] = str(self.path(request["root"]))
        if "sources" in request:
            request["sources"] = [dict(s, source=str(self.path(s["source"]))) for s in request["sources"]]
        return subprocess.run(
            [sys.executable, "-c", shlex.split(command)[2]],
            input=canonical_json(request), capture_output=True, text=True,
            check=True, timeout=timeout,
        ).stdout

    def write_capsule_file(self, identifier, name, content, gateway):
        self.write(f"{WORK_ROOT}/jobs/runs/{identifier}/{name}", content.encode())

    def submit_script(self, script, identifier, gateway, **kwargs):
        self.submissions.append(identifier)
        return SimpleNamespace(job_id="1234")

    def file_size(self, remote, gateway):
        return gateway, self.path(remote).stat().st_size

    def stream_file_range(self, remote, gateway, *, start, end):
        yield self.path(remote).read_bytes()[start:end + 1]


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
    db = Database(tmp_path / "test.store")
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


def test_entire_session_is_immutable_and_duplicate_clicks_recover_same_job(conversion):
    service, session, calls = conversion
    job = service.create("session", "Cube")
    assert job["indices"] == [0, 1]
    assert service.create("session", "Different name")["id"] == job["id"]
    assert len(service.list()) == 1
    assert job["scope"] == "all_recordings"
    assert job["sources"][0]["sha256"] == "a" * 64
    assert calls.count(job["id"]) == 2


def test_all_51_recordings_are_frozen_in_the_conversion_request(conversion):
    service, session, _ = conversion
    session["recordings"] = [f"recordings/episode-{i}.pkl" for i in range(51)]
    session["recording_checksums"] = {
        path: str(i % 10) * 64 for i, path in enumerate(session["recordings"])
    }
    job = service.create("session", "All recordings")
    request = json.loads((service.root / job["id"] / "request.json").read_text())
    assert job["indices"] == list(range(51))
    assert len(request["sources"]) == len(request["staged_sources"]) == 51
    assert [s["index"] for s in request["sources"]] == list(range(51))


def test_oversized_session_is_rejected_without_omitting_recordings(conversion):
    service, session, calls = conversion
    session["recordings"] = [f"recordings/{i}.pkl" for i in range(1001)]
    with pytest.raises(ValueError, match="no recordings were omitted"):
        service.create("session", "Cube")
    assert calls == []


def test_api_rejects_selection_and_always_converts_the_session(
    conversion, monkeypatch, tmp_path
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from skynet_app import live_xr_api as api

    service, _, calls = conversion
    monkeypatch.setattr(api, "conversions", service)
    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    url = "/api/collection/live/sessions/session/conversions"
    rejected = client.post(url, json={"name": "Cube", "indices": [0]})
    assert rejected.status_code == 422
    assert "every recording" in rejected.text
    assert calls == []
    accepted = client.post(url, json={"name": "Cube"})
    assert accepted.status_code == 202
    assert accepted.json()["indices"] == [0, 1]


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


def test_cluster_verified_result_registers_dataset_and_lineage_once(conversion, tmp_path):
    service, _, _ = conversion
    job = service.create("session", "Shadow cube")
    raw = b"\x89HDF\r\n\x1a\n" + b"test download transport" * 10
    m = metadata(job, raw)
    manifest = canonical_json(m).encode()
    files = {
        job["dataset_root"] + "/dataset.hdf5": raw,
        job["dataset_root"] + "/manifest.json": manifest,
    }
    files[job["dataset_root"] + "/source-manifest.json"] = canonical_json(job["sources"]).encode()
    transport = LocalCluster(tmp_path / "cluster")
    service.cluster = transport
    for path, value in files.items():
        transport.write(path, value)
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
    assert first["bundle_id"] is second["bundle_id"] is None
    assert first["resource_id"] == second["resource_id"]
    assert first["version_id"] == second["version_id"]
    version = service.database.get_data_resource_version(first["version_id"])
    assert version["format"] == FORMAT
    assert version["metadata"]["storage_location"] == "cluster"
    assert version["path"] == job["dataset_root"] + "/dataset.hdf5"
    assert version["derivation_id"]
    assert [item["kind"] for item in version["locations"]] == ["cluster"]
    # Conversion publishes a dataset; experiment input snapshots are selected
    # later. Automatic user-managed bundles were removed from this workflow.
    assert service.database.list_data_bundles() == []
    artifact = service.artifact(job["id"], "dataset.hdf5")
    assert artifact.path == job["dataset_root"] + "/dataset.hdf5"
    assert artifact.transport is transport
    assert not (service.root / job["id"] / "dataset.hdf5").exists()
    assert not (service.root / job["id"] / "manifest.json").exists()
    with pytest.raises(KeyError):
        service.artifact(job["id"], "../worker-sources.json")


def test_truncated_cluster_result_is_never_published(conversion, tmp_path):
    service, _, _ = conversion
    job = service.create("session", "Cube")
    raw = b"0123456789"
    m = metadata(job, raw)
    manifest = canonical_json(m).encode()
    transport = LocalCluster(tmp_path / "cluster")
    transport.write(job["dataset_root"] + "/manifest.json", manifest)
    transport.write(job["dataset_root"] + "/dataset.hdf5", raw[:4])
    with pytest.raises(ValueError, match="artifact size changed"):
        service.finish(job, dict(metadata=m, manifest=worker.file_info(transport.path(job["dataset_root"] + "/manifest.json"))), transport)
    assert not service.database.list_data_resources()
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


@pytest.mark.parametrize("legacy", [False, True])
def test_conversion_uses_pinned_hand_from_cache_or_legacy_path(conversion, tmp_path, monkeypatch, legacy):
    from skynet_app import simulation_hands
    service, session, _ = conversion
    service.live.root = tmp_path / "app"
    cache = tmp_path / "cache"
    monkeypatch.setattr(simulation_hands, "cache_root", lambda: cache)
    local = (service.live.root / "data/simulation-hands" if legacy else cache) / "robot" / "digest"
    local.mkdir(parents=True)
    (local / "manifest.json").write_text("{}")
    calls=[]
    def upload(path, root, transport, gateway):
        calls.append(path)
        return "/cluster/hand/digest"
    monkeypatch.setattr("skynet_app.live_conversion.upload_hand", upload)
    service.ensure_hand({"robot":"robot", "hand_bundle":{"digest":"digest", "root":"/cluster/hand/digest"}}, object(), "sky2")
    assert calls == [local]
