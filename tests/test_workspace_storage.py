"""Exercise personal path routing through real SQLite, compilation and SSH commands."""

import json
from skynet_app.db_backend import INTEGRITY_ERRORS
import subprocess
from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_cluster_runtime import RecordingClusterClient
from test_pipeline import (
    FakeCluster,
    canonical_spec,
    make_pipeline_service,
    seed_repository_choices,
)
from test_slurm import make_spec

from skynet_app.adapters import resolve_adapter_plan
from skynet_app.cluster_config import CLUSTER
from skynet_app.cluster_runtime import ClusterClient
from skynet_app.database import Database
from skynet_app.pipeline_api import EvaluationRequest, PipelineService, router
from skynet_app.slurm import (
    compile_sbatch,
    resolve_slurm_log_path,
    resolve_slurm_log_paths_from_sbatch,
)
from skynet_app.workspace_storage import (
    WorkspaceStorage,
    paths_for_root,
    validate_work_root,
)
from skynet_app.workspaces import WorkspaceMiddleware, WorkspaceServices, session_router


@pytest.fixture
def services(tmp_path, monkeypatch):
    monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", tmp_path / "capsules")
    coordinator = WorkspaceServices(PipelineService(Database(tmp_path / "storage.db"), ClusterClient()))
    monkeypatch.setattr(ClusterClient, "initialize_personal_workspace", lambda self, root, owner, gateway="auto": "sky2")
    monkeypatch.setattr("skynet_app.pipeline_api.service", coordinator)
    return coordinator


def personal(services, email):
    workspace, _ = services.directory.open(email)
    return services.for_workspace(workspace["id"])


def configure(service, root, expected):
    return service.storage.configure(root, expected, Mock(initialize_personal_workspace=lambda *args: "sky2"))


def create_run(service, root, name="run"):
    db = service.database
    project = db.create_project(name)
    experiment = db.create_experiment(project_id=project["id"], name=name, requested_spec={})
    variant = db.create_variant(experiment["latest_revision"]["id"], name=name, parameters={}, resolved_spec={})
    run = db.create_run(variant["id"], seed=42, adapter_name="generic", adapter_version="1", run_directory=root + "/jobs/runs/pending")
    db.update_run(run["id"], run_directory=f"{root}/jobs/runs/{run['id']}")
    return run["id"]


@pytest.mark.parametrize("value", ["/", "relative", "~/data", "/a/../b", "/a/./b", "/a;touch-x", "/a/$(id)", "/a/`id`", "/a\nb", "/a\x00b", "/a b", "/a/%j"])
def test_reject_ambiguous_or_shell_paths(value):
    with pytest.raises(ValueError):
        validate_work_root(value)


def test_path_is_personal_persistent_and_does_not_rewrite_runs(services):
    alice = personal(services, "alice@example.com")
    bob = personal(services, "bob@example.com")
    old = alice.storage.work_root
    original_root = CLUSTER.paths.work_root
    run_id = create_run(alice, original_root)
    configure(alice, " /coc/flash7/alice//skynet/ ", old)
    assert alice.work_root == "/coc/flash7/alice/skynet"
    assert bob.storage.work_root is None
    assert services.system.cluster.storage is None
    assert alice.cluster is not bob.cluster
    assert alice.cluster.work_root == alice.work_root
    assert alice._run_directory(run_id) == f"{original_root}/jobs/runs/{run_id}"
    restarted = WorkspaceStorage(Database(alice.database.path, workspace_id=alice.database.workspace_id))
    assert restarted.work_root == alice.work_root
    assert restarted.root_for_run(run_id) == original_root
    with bob.database.transaction() as connection, pytest.raises(INTEGRITY_ERRORS):
        connection.execute("UPDATE workspace_storage SET work_root='/bad/path' WHERE owner_id=?", (alice.database.workspace_id,))
    with pytest.raises(ValueError, match="another tab"):
        configure(alice, "/new/path", old)
    assert alice.work_root == "/coc/flash7/alice/skynet"


def test_concurrent_saves_have_one_winner(services):
    alice = personal(services, "alice@example.com")
    old = alice.storage.work_root
    def save(root):
        try:
            configure(alice, root, old)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(save, ["/team/alice", "/team/alice2"])) == [False, True]


def test_authenticated_api_validation_and_workspace_isolation(services):
    app = FastAPI()
    app.add_middleware(WorkspaceMiddleware, services=services)
    app.include_router(session_router(services.directory))
    app.include_router(router)
    with TestClient(app) as client:
        payload = {"work_root": "/team/alice", "expected_work_root": None}
        assert client.put("/api/workspace/storage", json=payload).status_code == 401
        client.post("/api/workspace/session", json={"email": "alice@example.com"})
        assert client.put("/api/workspace/storage", json={**payload, "work_root": "/tmp/$(id)"}).status_code == 422
        assert client.put("/api/workspace/storage", json=payload).status_code == 200
        settings = client.get("/api/settings").json()
        assert settings["paths"]["work_root"] == "/team/alice"
        assert settings["cluster"]["paths"]["jobs"] == "/team/alice/jobs"
        assert settings["cluster"]["paths"]["datasets"] == CLUSTER.paths.datasets
        assert client.get("/api/capabilities").json()["cluster"]["paths"]["logs"] == "/team/alice/logs"
        assert client.put("/api/workspace/storage", json=payload).status_code == 409
        client.post("/api/workspace/session", json={"email": "bob@example.com"})
        assert client.get("/api/settings").json()["paths"]["work_root"] is None


def test_compiler_routes_every_job_path_but_preserves_runtime_and_data():
    root = "/team/alice"
    spec = make_spec(runtime={"backend": "existing", "environment_path": CLUSTER.paths.environments + "/installed", "bootstrap_uv": False})
    compiled = compile_sbatch(spec, resolve_adapter_plan(spec), run_id="run-1", work_root=root)
    for suffix in ("workspace", "logs/%x-%j.out", "logs/%x-%j.err", "jobs/runs/run-1", ".cache/uv", ".cache/huggingface", ".cache/torch", "eval"):
        assert root + "/" + suffix in compiled.script
    assert f'{root}/repos/' in compiled.script
    assert CLUSTER.paths.environments + "/installed" in compiled.script
    assert f"export HOME={CLUSTER.paths.home_root}" in compiled.script
    stdout, stderr = resolve_slurm_log_paths_from_sbatch(compiled.script, "123", logs_root=root + "/logs")
    assert stdout.startswith(root + "/logs/") and stderr.endswith("-123.err")
    assert resolve_slurm_log_path(CLUSTER.paths.logs + "/job-%j.out", "123", logs_root=root + "/logs") is None
    paths = paths_for_root(root)
    assert paths.datasets == CLUSTER.paths.datasets
    assert paths.environments == CLUSTER.paths.environments


def test_transport_keeps_receipts_secrets_and_log_reads_at_original_root(services):
    alice = personal(services, "alice@example.com")
    first = "/team/alice/first"
    configure(alice, first, alice.storage.work_root)
    run_id = create_run(alice, first)
    configure(alice, "/team/alice/second", first)
    original = RecordingClusterClient()
    transport = original.with_storage(alice.storage)
    assert original.storage is None
    submission = transport.submit_script("#!/bin/bash\ntrue\n", run_id, "sky2")
    assert submission.run_directory == f"{first}/jobs/runs/{run_id}"
    assert f"{first}/jobs/runs/{run_id}/submissions/" in original.commands[-1][1]
    assert "/team/alice/second" not in original.commands[-1][1]
    _, secret_path = transport.write_capsule_file(run_id, "state/key", "fixture", "sky2")
    assert secret_path == submission.run_directory + "/state/key"
    transport.remove_capsule_file(run_id, "state/key", "sky2")
    assert secret_path in original.commands[-1][1]
    transport.recover_submission(run_id, "attempt-1", "sky2")
    assert first in original.commands[-1][1]
    transport.read_log(first + "/logs/job.out", "sky2")
    assert first + "/logs/job.out" in original.commands[-1][1]

    with pytest.raises(ValueError, match="registered workspace root"):
        transport.read_log("/team/bob/logs/job.out", "sky2")


def test_materialized_runs_and_checkpoints_survive_preference_change(tmp_path, monkeypatch):
    monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", tmp_path / "capsules")

    class Transport(RecordingClusterClient):
        def __init__(self):
            super().__init__()
            self.simulation = FakeCluster()
            self.log_content = ""

        def ssh(self, host, command, *, stdin=None, timeout=30):
            self.commands.append((host, command, stdin))
            if "sbatch --parsable" in command:
                return "9001\n"
            if "SKYNET_LOG_NOT_READY" in command:
                return self.log_content
            return self.simulation.run_with_fallback(command, host, stdin=stdin, timeout=timeout)[1]

    global_service = make_pipeline_service(Database(tmp_path / "jobs.db"), Transport())
    services = WorkspaceServices(global_service)
    service = personal(services, "alice@example.com")
    manifest = next(item for item in global_service.database.list_adapter_registry() if item["seed_key"] == "pipeline_fixture")
    from skynet_app.adapters import AdapterManifest
    version = service._selected_version(service.database.get_adapter(manifest["id"]))
    seed_repository_choices(service, "pipeline_fixture", manifest=AdapterManifest.model_validate(version["manifest"]))
    root = "/team/alice/first"
    configure(service, root, service.storage.work_root)
    experiment = service.create_experiment(canonical_spec())
    run_id = experiment["runs"][0]["id"]
    assert service.database.get_run(run_id)["run_directory"] == f"{root}/jobs/runs/{run_id}"
    configure(service, "/team/alice/second", root)
    cluster = service.cluster
    submitted = service.submit_experiment(experiment["id"])
    assert submitted["submitted"][0]["job_id"] == "9001"
    job_commands = [item for item in cluster.commands if "sbatch --parsable" in item[1]]
    assert root + "/jobs/runs/" in job_commands[-1][1]
    assert f"export WORK_ROOT={root}" in job_commands[-1][2]
    run = service.database.get_run(run_id)
    assert run["attempts"][0]["stdout_path"].startswith(root + "/logs/")
    assert run["attempts"][0]["execution_snapshot_json"]["storage"]["work_root"] == root
    cluster.log_content = json.dumps({"path": f"{root}/jobs/runs/{run_id}/checkpoints/final.ckpt", "final": True, "size_bytes": 1, "file_count": 1, "is_directory": False, "sha256": "a" * 64})
    attempt = run["attempts"][0]
    checkpoint = service._capture_checkpoint(run_id, attempt["id"], required=True)
    assert checkpoint["path"].startswith(root + "/jobs/")
    service.database.update_job_attempt(attempt["id"], status="SUCCEEDED", slurm_state="COMPLETED", exit_code="0:0")
    service.database.update_stage(run["stages"][0]["id"], status="SUCCEEDED")
    service.database.update_run(run_id, status="SUCCEEDED")
    suite = next(s for s in service.database.list_evaluation_suites() if s["name"] == "libero_10")
    evaluation = service.create_evaluation(EvaluationRequest(
        run_id=run_id, checkpoint_path=checkpoint["path"], suite_id=suite["id"],
        tasks=[suite["config_json"]["tasks"][0]], seeds=[0], episodes_per_task=1,
        argv=["python3", "evaluate.py"],
    ))
    assert evaluation["result_path"].startswith(root + "/eval/runs/")
    job_commands = [item for item in cluster.commands if "sbatch --parsable" in item[1]]
    assert len(job_commands) == 2
    assert f"export WORK_ROOT={root}" in job_commands[-1][2]
    new_spec = canonical_spec()
    new_spec["identity"]["experiment"] = "new-root"
    new_experiment = service.create_experiment(new_spec)
    assert new_experiment["runs"][0]["run_directory"].startswith("/team/alice/second/jobs/runs/")


def test_first_use_has_no_fallback_and_failed_init_is_not_saved(services):
    alice = personal(services, "alice@example.com")
    assert alice.storage.settings()["work_root"] is None
    assert alice.storage.settings()["configured"] is False
    assert alice.storage.public_paths()["jobs"] is None
    with pytest.raises(ValueError, match="Set up Cluster storage"):
        _ = alice.work_root
    cluster = Mock()
    cluster.initialize_personal_workspace.side_effect = RuntimeError("Permission denied")
    with pytest.raises(RuntimeError, match="Permission denied"):
        alice.storage.configure("/team/alice", None, cluster)
    assert alice.storage.work_root is None
    configure(alice, "/team/alice", None)
    bob = personal(services, "bob@example.com")
    for path in ("/team/alice", "/team/alice/nested", "/team"):
        with pytest.raises(ValueError, match="another workspace"):
            configure(bob, path, None)


def test_required_storage_gate_blocks_jobs_but_keeps_reads_and_settings(services):
    app = FastAPI()
    app.add_middleware(WorkspaceMiddleware, services=services)
    app.include_router(session_router(services.directory))
    app.include_router(router)
    with TestClient(app) as client:
        client.post("/api/workspace/session", json={"email": "new@example.com"})
        assert client.get("/api/workspace/session").json()["workspace"]["storage_configured"] is False
        settings = client.get("/api/settings").json()
        assert settings["storage"]["work_root"] is None
        assert settings["cluster"]["paths"]["jobs"] is None
        assert client.get("/api/runs").status_code == 200
        for path in ("/api/experiments", "/api/runs/example/resume", "/api/evaluations", "/api/policy-exports"):
            response = client.post(path, json={})
            assert response.status_code == 409
            assert response.json()["code"] == "storage_required"
        assert client.post("/api/runs/example/cancel").status_code != 409
        response = client.put("/api/workspace/storage", json={"work_root": "/team/new"})
        assert response.status_code == 200, response.text
        assert client.get("/api/workspace/session").json()["workspace"]["storage_configured"] is True


def test_remote_initialization_empty_nonempty_files_symlinks_and_retries(tmp_path):
    from skynet_app.workspace_storage_remote import initialize, DIRECTORIES
    root = tmp_path / "fresh"
    result = initialize(root, "alice")
    assert result["ready"]
    assert all((root / d).is_dir() for d in DIRECTORIES)
    (root / "jobs" / "keep.txt").write_text("keep")
    assert initialize(root, "alice") == result
    assert (root / "jobs" / "keep.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="another workspace"):
        initialize(root, "bob")
    nonempty = tmp_path / "occupied"
    nonempty.mkdir()
    (nonempty / ".hidden").write_text("keep")
    with pytest.raises(ValueError, match="not empty"):
        initialize(nonempty, "alice")
    assert list(nonempty.iterdir()) == [nonempty / ".hidden"]
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        initialize(link, "alice")
    with pytest.raises(ValueError, match="outside"):
        initialize(root / "nested", "alice")
    file = tmp_path / "file"
    file.write_text("keep")
    with pytest.raises(FileExistsError):
        initialize(file, "alice")
    assert file.read_text() == "keep"


def test_real_initialization_script_transport_and_concurrent_claim(tmp_path):
    # Execute exactly the payload sent over SSH, against isolated temporary directories.
    class LocalCluster(ClusterClient):
        def ssh(self, host, command, *, stdin=None, timeout=30):
            result = subprocess.run(command, shell=True, input=stdin, text=True, capture_output=True, timeout=timeout)
            if result.returncode:
                from skynet_app.cluster_runtime import ClusterError
                raise ClusterError(result.stderr)
            return result.stdout
    root = tmp_path / "transport"
    client = LocalCluster()
    assert client.initialize_personal_workspace(str(root), "alice", "sky2") == "sky2"
    from skynet_app.cluster_runtime import ClusterError
    with pytest.raises(ClusterError, match="another workspace"):
        client.initialize_personal_workspace(str(root), "bob", "sky2")
    root = tmp_path / "concurrent"
    def claim(owner):
        try:
            client.initialize_personal_workspace(str(root), owner, "sky2")
            return True
        except ClusterError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(claim, ["alice", "bob"])) == [False, True]


def test_non_writable_path_does_not_get_registered(tmp_path, monkeypatch):
    from skynet_app.workspace_storage_remote import initialize
    root = tmp_path / "denied"
    root.mkdir()
    monkeypatch.setattr("skynet_app.workspace_storage_remote.os.access", lambda *args: False)
    with pytest.raises(ValueError, match="writable"):
        initialize(root, "alice")
    assert not list(root.iterdir())
