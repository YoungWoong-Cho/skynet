import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import h5py
import numpy as np
import pytest
import zarr

from skynet_app.database import Database
from skynet_app.live_xr_review import LiveReviewService
from skynet_app.policy_exports import PolicyExportService, digest, conversion_failure
from policy_export_offline import prepare as offline_prepare, download as offline_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/xr"))
from images import ImageWriter, ImageRecorder, joint_layout


def test_worker_failure_distinguishes_progress_from_termination():
    log = '\n'.join(['{"episodes_done": 36, "episodes_total": 51}',
                     'resource_tracker.py: UserWarning: 1 leaked semaphore', 'warnings.warn("cleanup")'])
    message = conversion_failure(-15, log)
    assert "SIGTERM" in message and "36/51 episodes" in message
    assert "without an exception report" in message
    assert "leaked semaphore" not in message
    assert "ModuleNotFoundError: missing" in conversion_failure(1, log + '\nModuleNotFoundError: missing')


def test_preparation_retries_never_rename_existing_dataset(setup):
    service, session, _ = setup
    first = service.create(session["id"], "dp", "Original dataset")
    before = service.database.get_data_resource(first["resource_id"])
    service.update(first["id"], state="FAILED", error="Interrupted")
    second = service.create(session["id"], "act", "Accidental rename", first["resource_id"])
    after = service.database.get_data_resource(first["resource_id"])
    assert second["resource_id"] == first["resource_id"]
    assert second["name"] == "Original dataset"
    assert after["metadata"] == before["metadata"]
    assert after["description"] == before["description"]


def capture(root, index=0, both=False):
    wrists = [f"{axis}_translation_joint" for axis in "xyz"] + [f"{axis}_rotation_joint" for axis in "zyx"]
    names = wrists + ["finger_a", "finger_b"]
    if both:
        names = ["rh_" + n for n in wrists] + ["lh_" + n for n in wrists] + ["rh_finger_a", "rh_finger_b", "lh_finger_a", "lh_finger_b"]
    meta = dict(robot="floating_shadow_bimanual" if both else "floating_shadow_hand", task="test-task", hand="both" if both else "right", source_revision="abc", action_joint_names=names, robot_joint_names=list(reversed(names)), groups=joint_layout(names, "both" if both else "right"), step_dt=1 / 60, color_space="RGB", cameras={k: {"mount": "fixed_scene"} for k in ["scene_front", "scene_left", "scene_right"]})
    writer = ImageWriter(root, meta)
    actions = np.arange(3 * len(names), dtype="f4").reshape(3, -1) / 100 + index
    states = actions + 10
    rgb = np.zeros((256, 256, 3), dtype="u1")
    rgb[:, :, 0] = np.arange(256, dtype="u1")[:, None]
    rgb[:, :, 1] = 23
    rgb[:, :, 2] = 170
    for i in range(3):
        writer.append(states[i], actions[i], {"scene_front": rgb, "scene_left": rgb + np.uint8(2), "scene_right": rgb + np.uint8(3)}, i / 60, 1000 + i / 60)
    images = writer.finish()
    raw_states = [dict(articulation={"robot": {"joint_position": s[::-1][None]}}) for s in [*states, states[-1] + 1]]
    ep = dict(actions=actions, states=raw_states, num_steps=3, success=True, skynet_images=images)
    path = root / f"recordings/live/episode-{index:06d}.pkl"
    path.write_bytes(pickle.dumps(dict(format="dexverse_trajectory", schema_version=3, num_episodes=1, task=meta["task"], robot_type=meta["robot"], episodes=[ep])))
    return dict(path=str(path.relative_to(root)), sha256=digest(path), images=images), meta, actions, states


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "remote/output"
    source.mkdir(parents=True)
    receipts = [capture(source, i, both=True)[0] for i in range(2)]
    class Transport:
        def file_size(self, path, gateway):
            return gateway, Path(path).stat().st_size
        def stream_file_range(self, path, host, start, end):
            with Path(path).open("rb") as f:
                while block := f.read(8192):
                    yield block
    session = dict(id="session-1", state="STOPPED", root=str(source.parent), gateway="test-host", created_at="2026-09-08", profile=dict(display_name="Test hand", task="test-task", robot="floating_shadow_bimanual"), recordings=[r["path"] for r in receipts], recording_checksums={r["path"]: r["sha256"] for r in receipts}, recording_images={r["path"]: r["images"] for r in receipts})
    live = SimpleNamespace(root=ROOT, database=Database(tmp_path / "db.sqlite"), get=lambda identifier: session, list=lambda: [session], transport=lambda _: Transport())
    service = PolicyExportService(LiveReviewService(live, root=tmp_path / "reviews"), root=tmp_path / "exports")
    monkeypatch.setattr(service, "dispatch", lambda _: None)
    # Exercise the actual converter locally against tiny synthetic recordings.
    # Production prepare() exclusively dispatches the remote CPU lifecycle, which
    # has separate fake-cluster tests; no network belongs in these format tests.
    monkeypatch.setattr(service, "prepare", offline_prepare.__get__(service))
    monkeypatch.setattr(service, "download", offline_download.__get__(service), raising=False)
    service.processes = {}
    create = service.create
    def offline_create(*args, **kwargs):
        explicit_target = "target" in kwargs
        job = create(*args, **kwargs)
        return job if explicit_target else service.update(job["id"], target="local")
    monkeypatch.setattr(service, "create", offline_create)
    yield service, session, source
    service.stop()


@pytest.mark.parametrize("format", ["xpolicylab", "dp", "act"])
def test_real_export_preserves_commands_and_registers_immutable_lineage(setup, format):
    service, session, source = setup
    before = {str(p): digest(p) for p in source.rglob("*") if p.is_file()}
    job = service.create(session["id"], format, "My training data")
    assert service.create(session["id"], format, "Retry")["id"] == job["id"]
    service.prepare(job["id"])
    result = service.get(job["id"])
    assert result["state"] == "READY", result
    assert result["episodes"] == 2 and result["steps"] == 6
    manifest = json.loads(service.artifact(job["id"], "manifest.json").read_text())
    assert manifest["capture"]["hand"] == "both"
    version = service.database.get_data_resource_version(result["version_id"])
    assert version["status"] == "LOCAL" and version["derivation_id"]
    assert version["metadata"]["storage_location"] == "local"
    assert manifest["camera_slots"]["cam_left_wrist"] == "scene_left"
    with zipfile.ZipFile(service.artifact(job["id"], "dataset.zip")) as archive:
        assert "README.txt" in archive.namelist()
        assert "XPolicyLab/utils/robot/_robot_info.json" in archive.namelist()
        assert "env_cfg/robot/_robot_info.json" in archive.namelist()
        assert archive.testzip() is None
    assert {str(p): digest(p) for p in source.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("format", ["dp", "act"])
def test_streaming_output_matches_unmodified_upstream_converter(setup, tmp_path, format):
    service, session, _ = setup
    shared = service.create(session["id"], "xpolicylab", "Shared")
    service.prepare(shared["id"])
    assert service.get(shared["id"])["state"] == "READY"
    job = service.create(session["id"], format, "Policy")
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "READY", service.get(job["id"])
    sandbox = tmp_path / "upstream"
    shutil.copytree(ROOT / "ops/datasets/xpolicylab/XPolicyLab", sandbox / "XPolicyLab")
    shared_output = service.root / shared["id"] / "output"
    shutil.copytree(shared_output / "dataset", sandbox / "data/bench/run/skynet/data")
    shutil.copytree(shared_output / "env_cfg", sandbox / "env_cfg")
    folder = sandbox / "XPolicyLab/policy" / format.upper()
    script = folder / ("diffusion_policy/process_data.py" if format == "dp" else "detr/process_data.py")
    import os
    result = subprocess.run([sys.executable, str(script), "bench", "run", "skynet", "joint"], cwd=folder, env=dict(os.environ, PYTHONPATH=str(sandbox)), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = service.root / job["id"] / "output/dataset"
    if format == "dp":
        expected = zarr.open_group(str(folder / "data/bench-run-skynet-joint.zarr"), mode="r")
        got = zarr.open_group(str(actual / "demonstrations.zarr"), mode="r")
        for key in ["data/state", "data/action", "data/head_camera", "data/left_camera", "data/right_camera", "meta/episode_ends"]:
            np.testing.assert_array_equal(got[key][:], expected[key][:])
        np.testing.assert_array_equal(got["meta/episode_ends"][:], [3, 6])
        # A red gradient and constant blue channel must never be swapped.
        assert got["data/head_camera"][0, 2, 0, 0] == 170
    else:
        for i in range(2):
            with h5py.File(folder / f"processed_data/bench/run/skynet-joint/episode_{i}.hdf5") as expected, h5py.File(actual / f"episode_{i}.hdf5") as got:
                for key in ["action", "observations/qpos", "observations/images/cam_head", "observations/images/cam_left_wrist", "observations/images/cam_right_wrist"]:
                    np.testing.assert_array_equal(got[key][:], expected[key][:])


@pytest.mark.parametrize("mutation, message", [
    ("missing_images", "no completed training images"),
    ("active", "End the collection"),
    ("path", "Invalid saved image path"),
    ("hash", "checksums are missing"),
])
def test_rejects_ineligible_sources_before_dispatch(setup, mutation, message):
    service, session, _ = setup
    first = session["recordings"][0]
    if mutation == "missing_images": session["recording_images"].pop(first)
    if mutation == "active": session["state"] = "RUNNING"
    if mutation == "path": session["recording_images"][first]["path"] = "../../outside.hdf5"
    if mutation == "hash": session["recording_checksums"][first] = ""
    with pytest.raises(ValueError, match=message):
        service.create(session["id"], "dp", "Test")
    assert service.list() == []
    assert service.options()["sessions"][0]["eligible"] == (mutation == "missing_images")


def test_checksum_failure_does_not_register_a_version_and_can_retry(setup):
    service, session, source = setup
    job = service.create(session["id"], "dp", "Corrupt")
    raw = source / session["recordings"][0]
    raw.write_bytes(raw.read_bytes() + b"changed")
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "FAILED"
    assert all(v["format"] == "skynet.episodes/v1" for r in service.database.list_data_resources() for v in service.database.get_data_resource(r["id"])["versions"])
    assert service.create(session["id"], "dp", "Retry")["id"] != job["id"]


def test_restart_resumes_preparation_without_publishing_unverified_data(setup, monkeypatch):
    service, session, _ = setup
    job = service.create(session["id"], "dp", "Interrupted")
    service.update(job["id"], state="RUNNING")
    resumed = []
    monkeypatch.setattr(service, "dispatch", resumed.append)
    service.start()
    assert resumed == [job["id"]]
    assert service.get(job["id"])["state"] == "RUNNING"
    with pytest.raises(ValueError, match="not complete"):
        service.artifact(job["id"], "dataset.zip")
    with pytest.raises(KeyError):
        service.artifact(job["id"], "../source-0.pkl")


@pytest.mark.parametrize("kind", ["state", "action", "timestamps", "camera", "incomplete"])
def test_semantically_misaligned_images_are_rejected_even_with_valid_checksums(setup, kind):
    service, session, source = setup
    first = session["recordings"][0]
    receipt = session["recording_images"][first]
    image = source / receipt["path"]
    with h5py.File(image, "r+") as f:
        if kind in {"state", "action"}: f[kind][0, 0] += 1
        if kind == "timestamps": f["timestamps"][1] = 99
        if kind == "camera": del f["images/scene_left"]
        if kind == "incomplete": f.attrs["complete"] = False
    receipt.update(sha256=digest(image), size_bytes=image.stat().st_size)
    raw = source / first
    payload = pickle.loads(raw.read_bytes())
    payload["episodes"][0]["skynet_images"] = receipt
    raw.write_bytes(pickle.dumps(payload))
    session["recording_checksums"][first] = digest(raw)
    job = service.create(session["id"], "act", "Bad alignment")
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "FAILED"
    assert all(v["format"] == "skynet.episodes/v1" for r in service.database.list_data_resources() for v in service.database.get_data_resource(r["id"])["versions"])


def test_writer_discards_unsuccessful_partial_images_and_refuses_bad_frames(tmp_path):
    names = [f"{axis}_translation_joint" for axis in "xyz"] + [f"{axis}_rotation_joint" for axis in "zyx"] + ["finger"]
    writer = ImageWriter(tmp_path, {"action_joint_names": names})
    with pytest.raises(ValueError, match="three training cameras"):
        writer.append(np.zeros(7), np.zeros(7), {}, 0, 1)
    writer.discard()
    assert list((tmp_path / "recordings/live").iterdir()) == []
    with pytest.raises(ValueError):
        joint_layout(["unknown"] * 7, "right")


def test_capture_uses_current_pre_action_state_and_restores_tracking_markers(tmp_path):
    class Marker:
        def __init__(self): self.visible = True
        def set_visibility(self, value): self.visible = value
    markers = [Marker(), Marker()]
    recorder = ImageRecorder.__new__(ImageRecorder)
    recorder.retargeters = [SimpleNamespace(_markers=markers[0], _canonical_markers=markers[1])]
    recorder.ids = [2, 0]
    recorder.metadata = {"step_dt": 0.1}
    rgb = np.full((1, 256, 256, 4), 27, dtype="u1")
    def update(dt, force_recompute):
        assert dt == 0 and force_recompute and not any(m.visible for m in markers)
    scene = {name: SimpleNamespace(update=update, data=SimpleNamespace(output={"rgb": rgb})) for name in ["third_person_camera", "third_person_camera_left", "third_person_camera_right"]}
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(joint_pos=np.array([[10, 11, 12]])))
    calls = []
    recorder.env = SimpleNamespace(scene=scene, sim=SimpleNamespace(render=lambda: calls.append("render")))
    recorder.writer = SimpleNamespace(steps=2, append=lambda *args: calls.append(args))
    recorder.append(np.array([0.3, 0.4]))
    assert calls[0] == "render"
    np.testing.assert_array_equal(calls[1][0], [12, 10])
    assert calls[1][3] == 0.2 and all(m.visible for m in markers)
    assert calls[1][2]["scene_front"].shape == (256, 256, 3)


def test_export_api_validates_formats_and_exposes_downloads(setup, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from skynet_app import policy_exports_api as api
    service, session, _ = setup
    monkeypatch.setattr(api, "service", service)
    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    assert {f["id"] for f in client.get("/api/data/exports").json()["formats"]} == {"dp", "dp-state", "act", "xpolicylab", "egoverse"}
    body = dict(session_id=session["id"], format="dp", name="API export")
    assert client.post("/api/data/exports", json=dict(body, format="any-policy")).status_code == 409
    for removed in ({"target": "local"}, {"selections": [{"session_id": session["id"], "indices": [0]}]}):
        assert client.post("/api/data/exports", json=dict(body, **removed)).status_code == 422
    transfer_calls = []
    def transfer(job):
        transfer_calls.append(job)
        return service.database.record_data_location(job["version_id"], kind="cluster", host="skynet", path="/verified/dataset", manifest_sha256=job["manifest_sha256"])
    monkeypatch.setattr(service, "transfer", transfer)
    response = client.post("/api/data/exports", json=body)
    assert response.status_code == 202
    job = response.json()
    assert job["target"] == "cluster" and len(job["sources"]) == len(session["recordings"])
    assert client.get(f"/api/data/exports/{job['id']}/dataset.zip").status_code == 409
    service.prepare(job["id"])
    assert client.get(f"/api/data/exports/{job['id']}").json()["state"] == "READY"
    assert client.get(f"/api/data/exports/{job['id']}/dataset.zip").content[:2] == b"PK"
    assert len(transfer_calls) == 1
    assert client.post(f"/api/data/exports/{job['id']}/retry", json={"target": "local"}).status_code == 422
    assert client.get("/api/data/exports/missing").status_code == 404


def test_imported_hand_mapping_uses_declared_virtual_joints():
    wrist = ["skynet_x", "skynet_y", "skynet_z", "skynet_roll", "skynet_pitch", "skynet_yaw"]
    groups = joint_layout(wrist + ["J0", "J1", "J2"], "left", wrist)
    assert groups == [dict(side="left", wrist_indices=list(range(6)), finger_indices=[6, 7, 8])]


@pytest.mark.parametrize("format, mode", [("dp-state", "state"), ("dp", "rgb"), ("act", "rgb")])
def test_cluster_copy_checks_the_declared_observation_mode(setup, monkeypatch, format, mode):
    service, session, _ = setup
    job = service.create(session["id"], format, "Loader contract")
    service.prepare(job["id"])
    job = service.get(job["id"])
    assert job["state"] == "READY"
    receipt = dict(schema=f"skynet.{'act' if format == 'act' else 'dp'}-loader-validation/v1",
                   manifest_sha256=job["manifest_sha256"], observation_mode=mode)
    commands = []
    def ssh(gateway, command, timeout):
        commands.append(command)
        if "--verify-only" in command:
            return json.dumps(receipt)
        return json.dumps(dict(verified=True, manifest_sha256=job["manifest_sha256"]))
    capsules = {}
    service.cluster = SimpleNamespace(ssh=ssh, write_capsule_file=lambda job, name, content, gateway: capsules.update({name: content}))
    monkeypatch.setattr("skynet_app.policy_exports.upload_capture", lambda *args, **kwargs: "/upload/dataset.zip")
    location = service._transfer_host(job, "test-host")
    assert location["kind"] == "cluster"
    frozen = (service.root / job["id"] / "worker/training_parallel.py").read_text()
    assert capsules["adapter-support/training_parallel.py"] == frozen
    assert "class TrainingContext" in frozen
    if format != "act":
        assert f"--observation-mode {mode}" in commands[-1]
    receipt["observation_mode"] = "state" if mode == "rgb" else "rgb"
    with pytest.raises(ValueError, match="different dataset"):
        service._transfer_host(job, "test-host")


def test_missing_finished_artifact_can_be_regenerated(setup):
    service, session, _ = setup
    job = service.create(session["id"], "dp", "Missing")
    service.prepare(job["id"])
    service.artifact(job["id"], "dataset.zip").unlink()
    assert service.create(session["id"], "dp", "Retry")["id"] == job["id"]
    service.prepare(job["id"])
    assert service.artifact(job["id"], "dataset.zip").is_file()
    assert service.get(job["id"])["state"] == "READY"


def saved_state_sources(session, source):
    from images import training_image_request
    recipe = training_image_request({"front": {"position": [1, 2, 3]}})
    for name in session["recordings"]:
        raw = source / name
        payload = pickle.loads(raw.read_bytes())
        payload.update(skynet_training_images=recipe, skynet_step_dt=1 / 60)
        ep = payload["episodes"][0]
        ep.pop("skynet_images")
        ep["skynet_wall_times"] = [1000 + i / 60 for i in range(3)]
        raw.write_bytes(pickle.dumps(payload))
        checksum = digest(raw)
        receipt = session["recording_images"][name]
        image = source / receipt["path"]
        with h5py.File(image, "r+") as f:
            meta = json.loads(f.attrs["metadata"])
            meta.update(render_mode="saved_states", image_recipe_sha256=recipe["recipe_sha256"], source_sha256=checksum)
            f.attrs["metadata"] = json.dumps(meta)
        receipt.update(sha256=digest(image), size_bytes=image.stat().st_size, source_sha256=checksum)
        session["recording_checksums"][name] = checksum


@pytest.mark.parametrize("mutation", [None, "source_sha256", "image_recipe_sha256", "render_mode", "wall_times"])
def test_saved_state_images_bind_original_recipe_and_timing_without_modifying_recordings(setup, mutation):
    service, session, source = setup
    saved_state_sources(session, source)
    receipt = session["recording_images"][session["recordings"][0]]
    image = source / receipt["path"]
    if mutation:
        with h5py.File(image, "r+") as f:
            if mutation == "wall_times":
                f["wall_times"][1] += 1
            else:
                meta = json.loads(f.attrs["metadata"])
                meta[mutation] = "wrong"
                f.attrs["metadata"] = json.dumps(meta)
        receipt.update(sha256=digest(image), size_bytes=image.stat().st_size)
    before = {str(p): digest(p) for p in source.rglob("*.pkl")}
    job = service.create(session["id"], "dp", "Saved states")
    service.prepare(job["id"])
    result = service.get(job["id"])
    assert result["state"] == ("FAILED" if mutation else "READY"), result
    assert {str(p): digest(p) for p in source.rglob("*.pkl")} == before


def test_original_recording_locations_are_actual_directories(setup):
    service, session, source = setup
    locations = service.options()["sessions"][0]["locations"]
    assert locations == [{"kind": "remote", "host": "test-host", "path": str(source / "recordings/live")}]
    assert Path(locations[0]["path"]).is_dir()
    assert service.recording_locations({}) == []
    session["recordings"] = ["../../outside.pkl"]
    assert service.recording_locations(session) == []
