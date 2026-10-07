import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from skynet_app.cluster_config import CLUSTER
from skynet_app.cluster_runtime import DEFAULT_GATEWAY, ClusterClient, ClusterError, SubmissionOutcomeUnknown, WORK_ROOT
from skynet_app.training_contracts import RECORDING_DATASET_FORMAT
from skynet_app.policy_exports import PolicyExportService
from skynet_app import policy_exports_api as api
from test_policy_exports import setup as offline_setup, create


class Cluster:
    # The cluster, not its caller, chooses the login host that serves a request.
    host = "serving-host"

    def __init__(self):
        self.files, self.submissions = {}, []
        self.uploads = []
        self.state = "PENDING"
        self.fail_submit = False
        self.fail_status = False

    def candidates(self, gateway):
        return [self.host]

    def served(self, gateway):
        assert gateway == DEFAULT_GATEWAY, "Preparation routes with the configured default"
        return self.host

    def write_capsule_file(self, identifier, path, content, gateway):
        remote = f"{WORK_ROOT}/jobs/runs/{identifier}/{path}"
        self.files[remote] = content
        return self.served(gateway), remote

    def write_capsule_files(self, identifier, files, gateway):
        self.uploads.append(dict(files))
        return self.served(gateway), {
            path: self.write_capsule_file(identifier, path, content, gateway)[1]
            for path, content in files.items()
        }

    def submit_script(self, script, identifier, gateway, *, submission_key):
        host = self.served(gateway)
        # Keep the production identifier contract at this fake transport boundary.
        ClusterClient._run_id(identifier)
        ClusterClient._submission_token(submission_key)
        self.submissions.append((script, identifier, submission_key))
        if self.fail_submit:
            self.fail_submit = False
            raise SubmissionOutcomeUnknown("reply was lost after acceptance")
        return SimpleNamespace(job_id="42", gateway=host)

    def job_statuses(self, identifiers, gateway):
        assert identifiers == ["42"]
        host = self.served(gateway)
        if self.fail_status:
            raise ClusterError("temporary connection loss")
        return host, {"42": {"State": self.state}}

    def read_file(self, path, gateway, *, max_bytes):
        host = self.served(gateway)
        if path not in self.files:
            raise ClusterError("missing receipt")
        value = self.files[path]
        assert len(value) <= max_bytes
        return host, value

    def file_size(self, path, gateway):
        return self.served(gateway), len(self.files[path].encode())

    def stream_file_range(self, path, gateway, *, start, end):
        assert gateway == self.host, "A stream continues on the host that answered its size"
        yield self.files[path].encode()[start:end + 1]

@pytest.fixture
def setup(offline_setup, monkeypatch):
    service, session, source = offline_setup
    monkeypatch.setattr(service, "prepare", PolicyExportService.prepare.__get__(service))
    # These lifecycle tests isolate CPU source inspection and observation production,
    # which have independent suites; Slurm identity/retry/publication stays real.
    monkeypatch.setattr(PolicyExportService, "_preflight_sources", lambda self, job, sources: sources)
    monkeypatch.setattr(service.observations, "ensure", lambda job, sources: sources)
    session["archive"] = dict(state="READY", root=f"{WORK_ROOT}/datasets/raw/test/output")
    resolved, ensured = [], []
    service.cluster = Cluster()
    def resolve(current, relative):
        assert current["id"] == session["id"]
        resolved.append(relative)
        return service.cluster, DEFAULT_GATEWAY, session["archive"]["root"] + "/" + relative
    service.live.archive = SimpleNamespace(cluster=service.cluster, resolve=resolve, ensure=ensured.append)
    return service, session, source, resolved, ensured

def receipt(service, job, *, manifest=None):
    manifest = manifest or dict(format=RECORDING_DATASET_FORMAT, adapter=job["adapter"], contract=job["contract"], episodes=[{}, {}], steps=6,
                                files={"dataset/example": {"sha256": "f" * 64, "size_bytes": 10}})
    raw = json.dumps(manifest, indent=2)
    sha = hashlib.sha256(raw.encode()).hexdigest()
    location = f"{WORK_ROOT}/datasets/prepared/{sha}"
    loader = service._loader_request(job, job["cluster_root"] + "/worker", job["cluster_root"])
    result = dict(schema="skynet.cluster-preparation/v2", job_id=job["id"], attempt_id=job["attempt_id"],
                  verified=True, manifest=manifest, manifest_sha256=sha, path=location,
                  loader_validation=dict(schema=loader["schemas"][0], manifest_sha256=sha, observation_mode=loader["mode"]) if loader else None)
    service.cluster.files[location + "/manifest.json"] = raw
    service.cluster.files[job["cluster_root"] + "/result.json"] = json.dumps(result)
    return result

@pytest.mark.parametrize("format", ["fixture-state", "fixture-rgb", "act", "act-native", "egoverse"])
def test_all_formats_use_resumable_cpu_job_and_only_remote_payloads(setup, format):
    service, session, _, resolved, _ = setup
    job = create(service, session["id"], format, "Archived demonstration")
    assert job["target"] == "cluster"
    assert "gateway" not in job, "No login host has served this job yet"
    service.prepare(job["id"])
    pending = service.get(job["id"])
    assert pending["state"] == "PENDING"
    assert pending["gateway"] == service.cluster.host
    assert pending["cluster_partition"] == "rl2-lab"
    assert pending["cluster_account"] == "rl2-lab"
    assert pending["cluster_cpus"] == CLUSTER.defaults.background_jobs.recording_preparation.cpus_per_task
    assert len(service.cluster.submissions) == 1
    assert len(service.cluster.uploads) == 1
    assert any(path.endswith("/request.json") for path in service.cluster.uploads[0])
    assert any(path.endswith("/worker/cluster_worker.py") for path in service.cluster.uploads[0])
    script = pending["cluster_script"]
    assert f"#SBATCH --cpus-per-task={CLUSTER.defaults.background_jobs.recording_preparation.cpus_per_task}" in script
    assert "#SBATCH --gres" not in script and "export CUDA_VISIBLE_DEVICES=" in script
    assert "#SBATCH --account=rl2-lab" in script
    # The shared workspace prelude exports the roots the uv locator and caches resolve against.
    for key, value in (("WORK_ROOT", CLUSTER.paths.work_root), ("UV_CACHE_DIR", CLUSTER.paths.uv_cache), ("HF_HOME", CLUSTER.paths.huggingface_cache)):
        assert f"export {key}={value}\n" in script
    assert "UV_BIN=" not in script, "no conversion dependency, so the worker runs on the profile interpreter alone"
    request = json.loads(service.cluster.files[pending["cluster_root"] + "/request.json"])
    if format == "egoverse":
        assert pending["cluster_root"] + "/worker/egoverse_models.py" in service.cluster.files
    assert request["format"] == RECORDING_DATASET_FORMAT and all(s["recording"].startswith(session["archive"]["root"]) for s in request["sources"])
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
    assert [(v["kind"], v["host"]) for v in version["locations"]] == [("cluster", CLUSTER.id)]
    manifest = service.remote_artifact(job["id"], "manifest.json")
    host, size = manifest.location()
    assert host == service.cluster.host
    assert b"".join(service.cluster.stream_file_range(manifest.path, host, start=0, end=size - 1)).decode() \
        == service.cluster.files[manifest.path]
    assert version["derivation_id"]
    assert create(service, session["id"], format, "Repeated")["id"] == job["id"]
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
    job = create(service, session["id"], "fixture-rgb", "Pending archive")
    service.prepare(job["id"])
    assert service.get(job["id"])["stage"] == "ARCHIVING"
    assert ensured == [session["id"]] and not resolved and not service.cluster.submissions
    assert not (service.root / "sources").exists()
    session["archive"]["state"] = "VERIFIED"
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "PENDING"

def test_lost_submission_reply_and_restart_reuse_exact_script_and_identity(setup, monkeypatch):
    service, session, _, _, _ = setup
    job = create(service, session["id"], "fixture-rgb", "Recover")
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
    job = create(service, session["id"], "fixture-rgb", "Mismatch")
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
    job = create(service, session["id"], "fixture-rgb", "Retry")
    service.prepare(job["id"])
    first = service.get(job["id"])
    service.cluster.state = "OUT_OF_MEMORY"
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "FAILED"
    retried = service.retry(job["id"])
    assert retried["state"] == retried["stage"] == "QUEUED"
    assert retried["gateway"] is None, "a new attempt has not been accepted by any gateway yet"
    service.cluster.state = "PENDING"
    service.prepare(job["id"])
    second = service.get(job["id"])
    assert second["attempt_id"] != first["attempt_id"]
    assert service.cluster.submissions[0][2] != service.cluster.submissions[1][2]

def test_conversion_dependencies_run_the_worker_through_the_shared_uv_locator(setup):
    service, session, _, _, _ = setup
    adapter = service.test_adapters["egoverse-hpt"]
    manifest = adapter["latest_version"]["manifest"]
    preset = next(p for p in manifest["train"]["data_requirements"]["recording_conversion"]["presets"] if p["id"] == "hpt_joints")
    preset["conversion_dependencies"] = ["egoverse-converter==1.2.3"]
    service.test_adapters["egoverse-hpt"] = service.database.edit_adapter(adapter["id"], manifest=manifest)
    job = create(service, session["id"], "egoverse", "Pinned converter")
    service.prepare(job["id"])
    script = service.get(job["id"])["cluster_script"]
    locator = script.index("UV_BIN=")
    assert script.index(f"export UV_CACHE_DIR={CLUSTER.paths.uv_cache}\n") < locator
    assert f'UV_BOOTSTRAP_ROOT="$UV_CACHE_DIR"/bootstrap-{CLUSTER.defaults.uv_version}' in script
    assert '"$WORK_ROOT/.local/bin/uv"' in script and "command -v uv" in script and "python3 -m uv" in script
    assert "-m venv" not in script, "preparation jobs never bootstrap uv"
    launch = script.rstrip("\n").splitlines()[-1]
    assert launch.startswith("skynet_uv run --no-project --python ")
    assert " --with egoverse-converter==1.2.3 python " in launch and launch.endswith("/request.json")


def test_bootstrap_failure_surfaces_the_actual_slurm_log(setup):
    service, session, _, _, _ = setup
    job = create(service, session["id"], "egoverse", "Bootstrap failure")
    service.prepare(job["id"])
    job = service.get(job["id"])
    service.cluster.files[job["cluster_root"] + "/export.log"] = "uv is required for the pinned EgoVerse converter\n"
    service.cluster.state = "FAILED"
    service.prepare(job["id"])
    assert "uv is required" in service.get(job["id"])["error"]

def test_background_monitor_finishes_after_browser_closes_and_stops_cleanly(setup, monkeypatch):
    service, session, _, _, _ = setup
    job = create(service, session["id"], "fixture-rgb", "Unattended completion")
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
    monkeypatch.syspath_prepend(str(root / "skynet_app/adapters"))
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
    monkeypatch.syspath_prepend(str(root / "skynet_app/adapters"))
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

def test_archive_source_lookup_reads_each_selected_session_once(setup, monkeypatch):
    service, session, _, resolved, _ = setup
    session["profile"]["instructions"] = "Pick up the cube."
    job = create(service, session["id"], "fixture-rgb", "Archived demonstration")
    reads = []
    def get(identifier):
        reads.append(identifier)
        return session
    monkeypatch.setattr(service.live, "get", get)
    sources = service._archived_sources(job)
    assert len(sources) == len(session["recordings"])
    assert reads == [session["id"]]
    assert set(resolved) == {source["path"] for source in sources} | {source["image_path"] for source in sources}
    assert all(source["session_profile"]["instructions"] == "Pick up the cube." for source in sources)
    assert sources[0]["session_profile"] is not session["profile"]

@pytest.mark.parametrize("field", ["path", "image_path"])
def test_sources_outside_cluster_storage_are_rejected_whatever_host_is_recorded(setup, monkeypatch, field):
    service, session, _, _, _ = setup
    job = create(service, session["id"], "fixture-rgb", "Archived demonstration")
    resolve = service.live.archive.resolve
    # The transport places a source in cluster storage; a host name beside it does not.
    def recorded(current, relative):
        transport, _, path = resolve(current, relative)
        return transport, "retired-host", path
    monkeypatch.setattr(service.live.archive, "resolve", recorded)
    assert len(service._archived_sources(job)) == len(session["recordings"])
    workstation = object()
    paths = {item[field] for item in job["sources"]}
    def elsewhere(current, relative):
        transport, gateway, path = resolve(current, relative)
        return (workstation if relative in paths else transport), gateway, path
    monkeypatch.setattr(service.live.archive, "resolve", elsewhere)
    with pytest.raises(ValueError, match="must be stored on the training cluster"):
        service._archived_sources(job)
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "FAILED" and not service.cluster.submissions


def test_job_saved_with_a_login_host_is_prepared_through_the_default_route(setup):
    service, session, _, _, _ = setup
    job = create(service, session["id"], "fixture-rgb", "Saved with a login host")
    # Jobs created by earlier versions stored a login host before any submission.
    service.update(job["id"], gateway="sky2")
    service.prepare(job["id"])
    pending = service.get(job["id"])
    assert pending["state"] == "PENDING" and pending["gateway"] == service.cluster.host
    receipt(service, pending)
    service.cluster.state = "COMPLETED"
    service.prepare(job["id"])
    assert service.get(job["id"])["state"] == "READY"


def test_failed_capsule_upload_never_submits_job(setup, monkeypatch):
    service, session, *_ = setup
    job = create(service, session["id"], "fixture-rgb", "Archived demonstration")
    def broken(*args, **kwargs):
        raise ClusterError("Cluster capsule upload verification failed")
    monkeypatch.setattr(service.cluster, "write_capsule_files", broken)
    service.prepare(job["id"])
    assert service.cluster.submissions == []
    result = service.get(job["id"])
    assert not result.get("cluster_script")
    assert "upload verification failed" in result["error"]

@pytest.mark.parametrize('scheduler_state', ['REQUEUED', 'RESIZING', 'UNKNOWN', 'SPECIAL_EXIT', 'PREEMPTED', 'REVOKED'])
def test_scheduler_transition_and_unknown_state_preserve_cpu_attempt(setup, scheduler_state):
    service, session, _, _, _ = setup
    job = create(service, session['id'], 'fixture-rgb', 'Keep CPU attempt')
    service.prepare(job['id'])
    original = service.get(job['id'])
    service.cluster.state = scheduler_state
    service.prepare(job['id'])
    service.retry(job['id'])
    service.prepare(job['id'])
    current = service.get(job['id'])
    assert current['state'] == ('RUNNING' if scheduler_state == 'RESIZING' else 'PENDING')
    assert current['attempt_id'] == original['attempt_id']
    assert current['cluster_job_id'] == original['cluster_job_id']
    assert current['cluster_script'] == original['cluster_script']
    assert len(service.cluster.submissions) == 1
    receipt(service, current)
    service.cluster.state = 'COMPLETED'
    service.prepare(job['id'])
    assert service.get(job['id'])['state'] == 'READY'
    assert len(service.cluster.submissions) == 1
