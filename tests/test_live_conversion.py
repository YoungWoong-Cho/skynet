import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from skynet_app.cluster_runtime import ClusterError
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
    service = LiveConversionService(LiveReviewService(live), tmp_path / "conversions")
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

    def offline(*a, **kw):
        raise ClusterError("SSH timed out")

    service.live.transport = lambda _: SimpleNamespace(ssh=offline)
    service.work(job["id"])
    state = service.get(job["id"])
    assert state["state"] == "PREPARING"
    assert "unavailable" in state["connection_error"]
    assert service.create("session", "Cube")["id"] == job["id"]


def test_launch_capsule_is_stable_after_an_uncertain_acknowledgement(conversion):
    service, _, _ = conversion
    job = service.create("session", "Cube")
    capsules = []

    def send(*a, **kw):
        capsules.append(json.loads(kw["stdin"]))
        return '{"returncode":0}'

    transport = SimpleNamespace(ssh=send)
    service.launch(job, transport)
    service.update(job["id"], connection_error="Lost response")
    service.launch(service.get(job["id"]), transport)
    assert capsules[0] == capsules[1]
    script = capsules[0]["files"]["run.sh"]
    assert "flock --wait 1800" in script
    assert "convert_dataset.py" in script
    assert "training" not in script and "evaluation" not in script


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
        job["root"] + "/dataset.hdf5": raw,
        job["root"] + "/manifest.json": manifest,
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
    assert version["metadata"]["storage_location"] == "workstation"
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


def test_queue_exit_and_worker_errors_are_explicit(conversion):
    service, _, _ = conversion
    job = service.create("session", "Cube")
    service.update(job["id"], launched=True)
    result = dict(
        state="QUEUED", service={"ActiveState": "failed", "ExecMainStatus": "75"}
    )
    service.live.transport = lambda _: SimpleNamespace(
        ssh=lambda *a, **k: json.dumps(result)
    )
    service.work(job["id"])
    assert service.get(job["id"])["state"] == "FAILED"
    assert "GPU remained busy" in service.get(job["id"])["error"]


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
