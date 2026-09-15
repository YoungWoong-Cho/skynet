import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from skynet_app.cluster_runtime import ClusterClient, ClusterError, SubmissionOutcomeUnknown, WORK_ROOT
from skynet_app.dataset_formats import RECIPES
from skynet_app.policy_exports import PolicyExportService
from skynet_app import policy_exports_api as api
from test_policy_exports import setup as offline_setup


class Cluster:
    def __init__(self):
        self.files, self.submissions = {}, []
        self.state = "PENDING"
        self.fail_submit = False
        self.fail_status = False

    def candidates(self, gateway):
        return ["sky2"]

    def write_capsule_file(self, identifier, path, content, gateway):
        assert gateway == "sky2"
        remote = f"{WORK_ROOT}/jobs/runs/{identifier}/{path}"
        self.files[remote] = content
        return gateway, remote

    def submit_script(self, script, identifier, gateway, *, submission_key):
        assert gateway == "sky2"
        # Keep the production identifier contract at this fake transport boundary.
        ClusterClient._run_id(identifier)
        ClusterClient._submission_token(submission_key)
        self.submissions.append((script, identifier, submission_key))
        if self.fail_submit:
            self.fail_submit = False
            raise SubmissionOutcomeUnknown("reply was lost after acceptance")
        return SimpleNamespace(job_id="42")

    def job_statuses(self, identifiers, gateway):
        assert identifiers == ["42"] and gateway == "sky2"
        if self.fail_status:
            raise ClusterError("temporary connection loss")
        return gateway, {"42": {"State": self.state}}

    def read_file(self, path, gateway, *, max_bytes):
        assert gateway == "sky2"
        if path not in self.files:
            raise ClusterError("missing receipt")
        value = self.files[path]
        assert len(value) <= max_bytes
        return gateway, value

    def file_size(self, path, gateway):
        return gateway, len(self.files[path].encode())

    def stream_file_range(self, path, gateway, *, start, end):
        assert gateway == "sky2"
        yield self.files[path].encode()[start:end + 1]


@pytest.fixture
def setup(offline_setup, monkeypatch):
    service, session, source = offline_setup
    monkeypatch.setattr(service, "prepare", PolicyExportService.prepare.__get__(service))
    monkeypatch.setattr(service, "create", PolicyExportService.create.__get__(service))
    session["archive"] = dict(state="READY", root=f"{WORK_ROOT}/datasets/raw/test/output")
    resolved, ensured = [], []
    def resolve(current, relative):
        assert current["id"] == session["id"]
        resolved.append(relative)
        return service.cluster, "sky2", session["archive"]["root"] + "/" + relative
    service.live.archive = SimpleNamespace(resolve=resolve, ensure=ensured.append)
    service.cluster = Cluster()
    monkeypatch.setattr(service, "download", lambda *a, **k: pytest.fail("raw data must never download to the Mac"))
    return service, session, source, resolved, ensured


def receipt(service, job, *, manifest=None):
    manifest = manifest or dict(format=RECIPES[job["format"]]["format"], episodes=[{}, {}], steps=6,
                                files={"dataset/example": {"sha256": "f" * 64, "size_bytes": 10}})
    raw = json.dumps(manifest, indent=2)
    sha = hashlib.sha256(raw.encode()).hexdigest()
    location = f"{WORK_ROOT}/datasets/prepared/{sha}"
    loader = service._loader_request(job, job["cluster_root"] + "/worker", job["cluster_root"])
    result = dict(schema="skynet.cluster-preparation/v1", job_id=job["id"], attempt_id=job["attempt_id"],
                  verified=True, manifest=manifest, manifest_sha256=sha, path=location,
                  archive_sha256="c" * 64, archive_path=job["cluster_root"] + "/dataset.zip",
                  loader_validation=dict(schema=loader["schemas"][0], manifest_sha256=sha, observation_mode=loader["mode"]) if loader else None)
    service.cluster.files[location + "/manifest.json"] = raw
    service.cluster.files[job["cluster_root"] + "/result.json"] = json.dumps(result)
    service.cluster.files[job["cluster_root"] + "/dataset.zip"] = "PK-synthetic-explicit-download"
    return result


@pytest.mark.parametrize("format", list(RECIPES))
def test_all_formats_use_resumable_cpu_job_and_only_remote_payloads(setup, format):
    service, session, _, resolved, _ = setup
    job = service.create(session["id"], format, "Archived demonstration")
    assert job["target"] == "cluster"
    assert service.options()["targets"] == [dict(id="cluster", name="Training cluster")]
    service.prepare(job["id"])
    pending = service.get(job["id"])
    assert pending["state"] == "PENDING"
    assert pending["cluster_partition"] == "rl2-lab"
    assert pending["cluster_account"] == "rl2-lab"
    assert pending["cluster_cpus"] == 4
    assert len(service.cluster.submissions) == 1
    script = pending["cluster_script"]
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --gres" not in script and "export CUDA_VISIBLE_DEVICES=" in script
    assert "#SBATCH --account=rl2-lab" in script
    request = json.loads(service.cluster.files[pending["cluster_root"] + "/request.json"])
    if format == "egoverse":
        assert pending["cluster_root"] + "/worker/egoverse_models.py" in service.cluster.files
        assert '"$UV_BIN" run --no-project' in script
        assert "--with zarr==3.1.5" in script and "--with simplejpeg==1.8.2" in script
    assert request["format"] == format and all(s["recording"].startswith(session["archive"]["root"]) for s in request["sources"])
    assert all(not s.get("images") or s["images"].startswith(session["archive"]["root"]) for s in request["sources"])
    service.cluster.state = "RUNNING"
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "RUNNING"
    receipt(service, pending)
    service.cluster.state = "COMPLETED"
    service.prepare(job["id"])
    ready = service.get(job["id"])
    assert ready["state"] == "READY", ready
    version = service.database.get_data_resource_version(ready["version_id"])
    assert version["status"] == "ON_CLUSTER"
    assert version["metadata"]["storage_location"] == "cluster"
    assert [v["kind"] for v in version["locations"]] == ["cluster"]
    assert version["derivation_id"]
    assert service.create(session["id"], format, "Repeated")["id"] == job["id"]
    assert service.retry(job["id"])["state"] == "READY"
    assert len(service.cluster.submissions) == 1
    assert resolved
    assert not (service.root / "sources").exists()
    assert not (service.root / job["id"] / "output").exists()
    assert not list(service.root.rglob("*.zip"))
    assert not list(service.root.rglob("*.pkl"))
    assert not list(service.root.rglob("*.hdf5"))


def test_waits_for_archive_without_any_payload_or_submission(setup):
    service, session, _, resolved, ensured = setup
    session["archive"]["state"] = "TRANSFERRING"
    job = service.create(session["id"], "dp", "Pending archive")
    service.prepare(job["id"])
    assert service.get(job["id"])["stage"] == "ARCHIVING"
    assert ensured == [session["id"]] and not resolved and not service.cluster.submissions
    assert not (service.root / "sources").exists()
    session["archive"]["state"] = "VERIFIED"
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "PENDING"


def test_lost_submission_reply_and_restart_reuse_exact_script_and_identity(setup, monkeypatch):
    service, session, _, _, _ = setup
    job = service.create(session["id"], "dp", "Recover")
    service.cluster.fail_submit = True
    service.prepare(job["id"])
    unknown = service.get(job["id"])
    assert unknown["state"] == "SUBMISSION_UNKNOWN"
    other = PolicyExportService(service.reviews, root=service.root, cluster=service.cluster)
    resumed = []
    monkeypatch.setattr(other, "dispatch", resumed.append)
    other.start()
    assert resumed == [job["id"]]
    other.prepare(job["id"])
    assert other.get(job["id"])["state"] == "PENDING"
    assert service.cluster.submissions[0] == service.cluster.submissions[1]
    service.cluster.fail_status = True
    other.prepare(job["id"])
    assert other.get(job["id"])["state"] == "PENDING"
    other.retry(job["id"])
    assert len(service.cluster.submissions) == 2
    other.stop()


@pytest.mark.parametrize("mutation", ["attempt_id", "manifest_sha256", "path", "format", "loader"])
def test_mismatched_receipts_never_publish(setup, mutation):
    service, session, _, _, _ = setup
    job = service.create(session["id"], "dp", "Mismatch")
    service.prepare(job["id"])
    job = service.get(job["id"])
    result = receipt(service, job)
    if mutation == "format":
        result["manifest"]["format"] = "wrong"
    elif mutation == "loader":
        result["loader_validation"]["manifest_sha256"] = "0" * 64
    else:
        result[mutation] = "wrong"
    service.cluster.files[job["cluster_root"] + "/result.json"] = json.dumps(result)
    service.cluster.state = "COMPLETED"
    service.prepare(job["id"])
    failed = service.get(job["id"])
    assert failed["state"] == "FAILED" and not failed.get("version_id")
    assert all(v["format"] == "skynet.episodes/v1" for v in service.database.get_data_resource(job["resource_id"])["versions"])


def test_failed_cpu_job_retry_creates_one_new_attempt(setup):
    service, session, _, _, _ = setup
    job = service.create(session["id"], "dp", "Retry")
    service.prepare(job["id"])
    first = service.get(job["id"])
    service.cluster.state = "OUT_OF_MEMORY"
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "FAILED"
    retried = service.retry(job["id"])
    assert retried["state"] == retried["stage"] == "QUEUED"
    service.cluster.state = "PENDING"
    service.prepare(job["id"])
    second = service.get(job["id"])
    assert second["attempt_id"] != first["attempt_id"]
    assert service.cluster.submissions[0][2] != service.cluster.submissions[1][2]


def test_bootstrap_failure_surfaces_the_actual_slurm_log(setup):
    service, session, _, _, _ = setup
    job = service.create(session["id"], "egoverse", "Bootstrap failure")
    service.prepare(job["id"])
    job = service.get(job["id"])
    service.cluster.files[job["cluster_root"] + "/export.log"] = "uv is required for the pinned EgoVerse converter\n"
    service.cluster.state = "FAILED"
    service.prepare(job["id"])
    assert "uv is required" in service.get(job["id"])["error"]


def test_converter_shell_uses_existing_configured_uv_bootstrap(setup, tmp_path, monkeypatch):
    from skynet_app import policy_exports_cluster as remote
    service, session, _, _, _ = setup
    config = remote.CLUSTER.model_copy(deep=True)
    config.paths.uv_cache = str(tmp_path / "cache with spaces")
    monkeypatch.setattr(remote, "CLUSTER", config)
    monkeypatch.setattr(remote, "WORK_ROOT", str(tmp_path / "cluster"))
    uv = Path(config.paths.uv_cache) / f"bootstrap-{config.defaults.uv_version}/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    uv.chmod(0o700)
    job = service.create(session["id"], "egoverse", "Existing bootstrap")
    service.prepare(job["id"])
    staged = service.get(job["id"])
    script = tmp_path / "prepare.sh"
    script.write_text(staged["cluster_script"])
    result = subprocess.run(["/bin/bash", str(script)], capture_output=True, text=True,
                            env={**os.environ, "PATH": "/usr/bin:/bin"}, timeout=10)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[:2] == ["run", "--no-project"]
    assert "zarr==3.1.5" in args and "numpy==2.2.6" in args
    assert args[-1] == staged["cluster_root"] + "/request.json"


def test_backend_rejects_local_target_and_explicit_download_streams_without_cache(setup, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    service, session, _, _, _ = setup
    with pytest.raises(ValueError, match="sky2"):
        service.create(session["id"], "dp", "No local", target="local")
    job = service.create(session["id"], "dp", "Download")
    service.prepare(job["id"])
    job = service.get(job["id"])
    receipt(service, job)
    service.cluster.state = "COMPLETED"
    service.prepare(job["id"])
    with pytest.raises(ValueError, match="sky2"):
        service.retry(job["id"], "local")
    before = set(service.root.rglob("*"))
    monkeypatch.setattr(api, "service", service)
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        response = client.get(f"/api/data/exports/{job['id']}/dataset.zip")
    assert response.status_code == 200 and response.content == b"PK-synthetic-explicit-download"
    assert response.headers["content-disposition"] == "attachment; filename*=UTF-8''dataset.zip"
    assert set(service.root.rglob("*")) == before


@pytest.mark.parametrize("failure", [None, "remote_failed", "local_changed", "unlisted_file", "wrong_manifest"])
def test_migration_retains_exact_local_payload_until_cluster_verification(setup, monkeypatch, failure):
    service, session, _, _, _ = setup
    job = service.create(session["id"], "xpolicylab", "Legacy local version")
    directory = service.root / job["id"]
    output = directory / "output"
    output.mkdir()
    payload = b"old exact prepared data"
    (output / "episode.hdf5").write_bytes(payload)
    manifest = dict(format=RECIPES["xpolicylab"]["format"], episodes=[{}, {}], steps=6,
                    files={"episode.hdf5": dict(size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())})
    raw = json.dumps(manifest, indent=2)
    sha = hashlib.sha256(raw.encode()).hexdigest()
    (output / "manifest.json").write_text(raw)
    (directory / "dataset.zip").write_bytes(b"old original zip")
    version = service.register(job, manifest, sha)
    local = service.database.record_data_location(version["id"], kind="local", host="local", path=str(output), manifest_sha256=sha)
    service.update(job["id"], state="READY", version_id=version["id"], manifest_sha256=sha)
    # User-authorized storage migration retains pinned version identities too.
    monkeypatch.setattr(service.database, "data_version_usage", lambda _: [dict(kind="experiment", id="pin")])
    uploaded = []
    def upload(cluster, path, identifier, checksum, gateway, *, relative_path, timeout):
        assert gateway == "sky2" and hashlib.sha256(path.read_bytes()).hexdigest() == checksum
        uploaded.append(relative_path)
        return f"{WORK_ROOT}/jobs/runs/{identifier}/{relative_path}"
    monkeypatch.setattr("skynet_app.capture_processing.service.upload_capture", upload)
    service.migrate_local_copy(job["id"])
    service.prepare(job["id"])
    pending = service.get(job["id"])
    assert pending["state"] == "PENDING"
    assert (output / "episode.hdf5").read_bytes() == payload
    assert (directory / "dataset.zip").read_bytes() == b"old original zip"
    assert len(uploaded) == 2
    result = receipt(service, pending, manifest=manifest)
    service.cluster.state = "COMPLETED"
    if failure == "remote_failed":
        service.cluster.state = "FAILED"
    elif failure == "local_changed":
        (output / "episode.hdf5").write_bytes(b"changed after upload")
    elif failure == "unlisted_file":
        (output / "user-note.bin").write_bytes(b"never uploaded")
    elif failure == "wrong_manifest":
        result["manifest_sha256"] = "f" * 64
        result["path"] = f"{WORK_ROOT}/datasets/prepared/" + "f" * 64
        service.cluster.files[pending["cluster_root"] + "/result.json"] = json.dumps(result)
    service.prepare(job["id"])
    final = service.get(job["id"])
    assert final["version_id"] == version["id"]
    current = service.database.get_data_resource_version(version["id"])
    assert len(service.database.get_data_resource(job["resource_id"])["versions"]) == 2
    if failure:
        assert final["state"] == "FAILED"
        assert output.is_dir() and (directory / "dataset.zip").is_file()
        if failure == "unlisted_file":
            assert (output / "user-note.bin").read_bytes() == b"never uploaded"
        assert next(v for v in current["locations"] if v["id"] == local["id"])["status"] == "AVAILABLE"
    else:
        assert final["state"] == "READY" and final["local_removed"]
        assert not output.exists() and not (directory / "dataset.zip").exists()
        assert next(v for v in current["locations"] if v["id"] == local["id"])["status"] == "REMOVED"
        assert any(v["kind"] == "cluster" and v["status"] == "AVAILABLE" for v in current["locations"])


@pytest.mark.parametrize("format", ["act", "dp", "dp-state", "xpolicylab", "corrupt-source"])
def test_cpu_capsule_executes_real_converter_and_publishes_verified_artifacts(setup, tmp_path, format):
    service, session, source, _, _ = setup
    corrupt = format == "corrupt-source"
    format = "dp" if corrupt else format
    job = service.create(session["id"], format, "Real remote worker")
    worker = service.root / job["id"] / "worker"
    worker_script = worker / "cluster_worker.py"
    worker_script.write_text((Path(__file__).parents[1] / "skynet_app/policy_export_worker.py").read_text())
    remote = tmp_path / "simulated-cluster-attempt"
    remote.mkdir()
    sources = [dict(item, recording=str(source / item["path"]), images=str(source / item["image_path"]) if item.get("image_path") else None) for item in job["sources"]]
    if corrupt:
        sources[0]["sha256"] = "0" * 64
    request = {key: job[key] for key in ("format", "split", "contract", "source_revision", "converter_sha256")}
    request.update(job_id=job["id"], attempt_id="synthetic-attempt", sources=sources, output=str(remote / "output"),
                   prepared_root=str(tmp_path / "simulated-published"), receipt_path=str(remote / "result.json"), loader=None)
    request_path = remote / "request.json"
    request_path.write_text(json.dumps(request))
    completed = subprocess.run([sys.executable, str(worker_script), str(request_path)], capture_output=True, text=True, timeout=30)
    if corrupt:
        result = json.loads((remote / "result.json").read_text())
        assert completed.returncode != 0 and result["verified"] is False
        assert "checksum" in result["error"]
        assert not (remote / "output").exists() and not (tmp_path / "simulated-published").exists()
        return
    assert completed.returncode == 0, completed.stderr
    result = json.loads((remote / "result.json").read_text())
    assert result["verified"] and result["job_id"] == job["id"] and result["attempt_id"] == "synthetic-attempt"
    from ops.datasets.artifacts import digest, verify
    manifest = verify(result["path"], result["manifest_sha256"])
    assert manifest["format"] == RECIPES[format]["format"] and len(manifest["episodes"]) == 2
    assert digest(result["archive_path"]) == result["archive_sha256"]
    assert not (service.root / "sources").exists() and not (service.root / job["id"] / "output").exists()
    if format == "act":
        retry = tmp_path / "retry-attempt"
        retry.mkdir()
        request.update(attempt_id="retry-attempt", output=str(retry / "output"),
                       reuse_output=str(remote / "output"), receipt_path=str(retry / "result.json"))
        request_path.write_text(json.dumps(request))
        again = subprocess.run([sys.executable, str(worker_script), str(request_path)], capture_output=True, text=True, timeout=30)
        assert again.returncode == 0, again.stderr
        reused = json.loads((retry / "result.json").read_text())
        assert reused["manifest_sha256"] == result["manifest_sha256"]
        assert reused["archive_path"] == result["archive_path"]
        assert not (retry / "output").exists(), "Retry must not convert or copy the dataset again"
        request["converter_sha256"] = "changed-converter"
        request_path.write_text(json.dumps(request))
        wrong = subprocess.run([sys.executable, str(worker_script), str(request_path)], capture_output=True, text=True, timeout=30)
        assert wrong.returncode != 0
        assert "pinned conversion" in json.loads((retry / "result.json").read_text())["error"]



@pytest.mark.parametrize("failure", [None, "cluster_verification", "zip_changed", "zip_record_mismatch"])
def test_existing_verified_cluster_copy_migrates_original_zip_without_cpu_job(offline_setup, monkeypatch, failure):
    service, session, _ = offline_setup
    job = service.create(session["id"], "xpolicylab", "Already copied")
    service.prepare(job["id"])
    job = service.get(job["id"])
    assert job["state"] == "READY", job
    archive = service.root / job["id"] / "dataset.zip"
    zip_bytes = archive.read_bytes()
    manifest_path = service.root / job["id"] / "output/manifest.json"
    raw_manifest = manifest_path.read_text()
    destination = f"{WORK_ROOT}/datasets/prepared/{job['manifest_sha256']}"
    service.database.record_data_location(job["version_id"], kind="cluster", host="skynet", path=destination, manifest_sha256=job["manifest_sha256"])
    service.cluster = Cluster()
    service.cluster.files[destination + "/manifest.json"] = raw_manifest
    def ssh(gateway, command, timeout):
        assert gateway == "sky2" and job["manifest_sha256"] in command
        return json.dumps(dict(verified=failure != "cluster_verification", manifest_sha256=job["manifest_sha256"]))
    service.cluster.ssh = ssh
    session["archive"] = dict(state="READY", root=f"{WORK_ROOT}/datasets/raw/test/output")
    service.live.archive = SimpleNamespace(resolve=lambda s, p: (service.cluster, "sky2", session["archive"]["root"] + "/" + p))
    monkeypatch.setattr(service, "prepare", PolicyExportService.prepare.__get__(service))
    uploads = []
    def upload(cluster, path, identifier, checksum, gateway, *, relative_path, timeout):
        assert path.read_bytes() == zip_bytes and checksum == job["archive_sha256"]
        uploads.append(relative_path)
        if failure == "zip_changed":
            archive.write_bytes(zip_bytes + b"new local bytes")
    monkeypatch.setattr("skynet_app.capture_processing.service.upload_capture", upload)
    if failure == "zip_record_mismatch":
        archive.write_bytes(zip_bytes + b"unrecorded bytes")
        with pytest.raises(ValueError, match="recorded checksum"):
            service.migrate_local_copy(job["id"])
    else:
        service.migrate_local_copy(job["id"])
        service.prepare(job["id"])
    final = service.get(job["id"])
    assert not service.cluster.submissions
    if failure:
        assert archive.is_file() and manifest_path.is_file()
    else:
        assert final["state"] == "READY", final
        assert len(uploads) == 1 and uploads[0].endswith("/dataset.zip")
        assert final["archive_sha256"] == job["archive_sha256"]
        assert final["version_id"] == job["version_id"] and final["remote_archive"]
        assert not archive.exists() and not manifest_path.exists()


def test_background_monitor_finishes_after_browser_closes_and_stops_cleanly(setup, monkeypatch):
    service, session, _, _, _ = setup
    job = service.create(session["id"], "dp", "Unattended completion")
    monkeypatch.setattr(service, "dispatch", PolicyExportService.dispatch.__get__(service))
    service.monitor_interval = 0.005
    completed = threading.Event()
    original_update = service.update
    def update(identifier, **changes):
        current = original_update(identifier, **changes)
        if changes.get("state") == "READY":
            completed.set()
        return current
    monkeypatch.setattr(service, "update", update)
    polls = []
    def statuses(identifiers, gateway):
        polls.append(identifiers)
        if len(polls) == 1:
            return gateway, {"42": {"State": "PENDING"}}
        receipt(service, service.get(job["id"]))
        return gateway, {"42": {"State": "COMPLETED"}}
    service.cluster.job_statuses = statuses
    service.start()
    assert completed.wait(3), service.get(job["id"])
    assert service.get(job["id"])["state"] == "READY"
    assert len(service.cluster.submissions) == 1 and len(polls) >= 2
    service.stop()
    assert not service.monitor_thread.is_alive()
    count = len(polls)
    service.dispatch(job["id"])
    assert len(polls) == count


def test_loader_timeout_keeps_diagnostics_and_terminates_child_group(tmp_path, monkeypatch):
    import importlib.util
    import time
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "ops/datasets"))
    spec = importlib.util.spec_from_file_location("loader_worker", root / "skynet_app/policy_export_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    marker = tmp_path / "should-not-exist"
    child = "import time,pathlib; time.sleep(1); pathlib.Path(" + repr(str(marker)) + ").touch()"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); print('loader-started',flush=True); time.sleep(60)"
    log = tmp_path / "loader.log"
    with pytest.raises(ValueError, match="timed out"):
        module.run_loader([sys.executable, "-c", parent], log, timeout=0.2)
    assert "loader-started" in log.read_text()
    time.sleep(1)
    assert not marker.exists()


def test_loader_uses_short_temporary_ipc_path_with_long_storage_root(tmp_path, monkeypatch):
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "ops/datasets"))
    spec = importlib.util.spec_from_file_location("short_ipc_worker", root / "skynet_app/policy_export_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    long_path = tmp_path / ("a" * 130)
    long_path.mkdir()
    monkeypatch.setenv("TMPDIR", str(long_path))
    script = "import json,multiprocessing.connection,os; listener=multiprocessing.connection.Listener(family='AF_UNIX'); listener.close(); print(json.dumps({'ipc':os.environ['TMPDIR']}))"
    receipt = module.run_loader([sys.executable, "-c", script], tmp_path / "loader.log", timeout=10)
    assert len(receipt["ipc"]) < 60
    assert not Path(receipt["ipc"]).exists()


@pytest.mark.parametrize("archive", ["previous", "unrelated"])
def test_retry_publishes_verified_archive_only_from_its_own_previous_attempt(setup, archive):
    service, session, _, _, _ = setup
    job = service.create(session["id"], "act-native", "Retry publication")
    service.prepare(job["id"])
    first = service.get(job["id"])
    service.cluster.state = "FAILED"
    service.prepare(job["id"])
    service.retry(job["id"])
    service.cluster.state = "PENDING"
    service.prepare(job["id"])
    second = service.get(job["id"])
    result = receipt(service, second)
    result["archive_path"] = first["cluster_root"] + "/dataset.zip" if archive == "previous" else f"{WORK_ROOT}/jobs/runs/another-job/dataset.zip"
    service.cluster.files[second["cluster_root"] + "/result.json"] = json.dumps(result)
    service.cluster.state = "COMPLETED"
    service.prepare(job["id"])
    final = service.get(job["id"])
    assert final["state"] == ("READY" if archive == "previous" else "FAILED"), final
    if archive == "previous":
        assert final["remote_archive"] == result["archive_path"]
