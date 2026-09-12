import ast
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import tempfile
import threading

import pytest
from pydantic import ValidationError

from skynet_app.cluster_runtime import ClusterError, Submission
from skynet_app.data_imports import _IMPORT_PROGRAM, build_huggingface_import_job
from skynet_app.database import Database
from skynet_app.pipeline_api import HuggingFaceImportRequest, PipelineService


RESOURCE = {
    "id": "resource-1",
    "provider": "huggingface",
    "namespace": "nvidia",
    "name": "PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
    "kind": "demonstrations",
}


def import_request() -> HuggingFaceImportRequest:
    return HuggingFaceImportRequest(
        revision="e" * 40,
        subset="gr1_arms_only.CanSort",
        format="groot-lerobot-v2.0",
        role="training_data",
        bundle_name="groot-gr1-cansort",
        bundle_version="n16-test",
        gateway="auto",
        queue="overcap",
    )


class ImportCluster:
    def __init__(self) -> None:
        self.script = ""
        self.result = None
        self.state = "COMPLETED"
        self.cancel_calls = []
        self.on_status = None
        self.on_cancel = None
        self.on_read = None

    def test_script(self, script: str, gateway: str = "auto"):
        self.script = script
        return "sky2", "valid"

    def submit_script(self, script: str, run_id: str, gateway: str = "auto"):
        self.script = script
        return Submission(
            "1234",
            "1234",
            "sky2",
            f"/coc/flash7/ycho420/jobs/runs/{run_id}/attempts/1234/job.sbatch",
            f"/coc/flash7/ycho420/jobs/runs/{run_id}",
        )

    def job_statuses(self, job_ids, gateway: str = "auto"):
        if self.on_status:
            self.on_status()
        return "sky2", {
            "1234": {
                "State": self.state,
                "ExitCode": "0:0",
                "NodeList": "node1",
            }
        }

    def cancel(self, job_id: str, gateway: str = "auto"):
        self.cancel_calls.append((job_id, gateway))
        if self.on_cancel:
            self.on_cancel()
        return "sky2"

    def read_file(self, path: str, gateway: str = "auto", max_bytes: int = 20_000_000):
        if self.on_read:
            self.on_read()
        return "sky2", json.dumps(self.result)


def test_generated_import_is_pinned_and_uses_overcap():
    job = build_huggingface_import_job("a" * 36, RESOURCE, import_request().model_dump())
    assert job.script.startswith("#!/usr/bin/env bash\n")
    assert "list_repo_tree(" in _IMPORT_PROGRAM
    assert "hf_hub_download(" in _IMPORT_PROGRAM
    assert "snapshot_download(" not in _IMPORT_PROGRAM
    assert "SKYNET_DATA_IMPORT_PROGRESS=" in _IMPORT_PROGRAM
    assert "#SBATCH --partition=overcap" in job.script
    assert "#SBATCH --account=overcap" in job.script
    assert "huggingface-hub==1.29.0" in job.script
    assert "refs/heads/main" not in job.script


def import_result(record):
    return {
        "schema_version": "skynet.data-import-result/v1",
        "import_id": record["id"],
        "resource_id": record["resource_id"],
        "provider": "huggingface",
        "namespace": RESOURCE["namespace"],
        "name": RESOURCE["name"],
        "kind": RESOURCE["kind"],
        "revision": "e" * 40,
        "version_revision": f"{'e' * 40}#subset=gr1_arms_only.CanSort",
        "subset": "gr1_arms_only.CanSort",
        "format": "groot-lerobot-v2.0",
        "role": "training_data",
        "bundle_name": "groot-gr1-cansort",
        "bundle_version": "n16-test",
        "manifest_sha256": "f" * 64,
        "inventory_sha256": "d" * 64,
        "file_count": 4,
        "size_bytes": 100,
        "path": "/coc/flash7/ycho420/datasets/resources/huggingface/nvidia/data",
        "manifest_path": "/coc/flash7/ycho420/datasets/resources/huggingface/nvidia/manifest.json",
        "source_uri": "https://huggingface.co/datasets/nvidia/example/tree/eeee/gr1",
        "snapshot_path": "/coc/flash7/ycho420/.cache/huggingface/snapshots/eeee",
    }


def test_submit_and_reconcile_publishes_directly_selectable_data():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        resource = database.create_data_resource(
            category="dataset",
            provider=RESOURCE["provider"],
            namespace=RESOURCE["namespace"],
            name=RESOURCE["name"],
            kind=RESOURCE["kind"],
        )
        cluster = ImportCluster()
        service = PipelineService(database, cluster)
        submitted = service.submit_data_import(resource["id"], import_request())
        assert submitted["state"] == "SUBMITTED"
        cluster.result = import_result(submitted)
        imports = service.reconcile_data_imports()
        completed = next(item for item in imports if item["id"] == submitted["id"])
        assert completed["state"] == "SUCCEEDED"
        assert completed["version_id"]
        assert completed["bundle_id"] is None
        assert database.get_data_resource_version(completed["version_id"])["status"] == "READY"


@pytest.fixture
def submitted_import(tmp_path):
    database = Database(tmp_path / "skynet.db")
    resource = database.create_data_resource(category="dataset", **{key: RESOURCE[key] for key in ("provider", "namespace", "name", "kind")})
    cluster = ImportCluster()
    cluster.state = "RUNNING"
    service = PipelineService(database, cluster)
    record = service.submit_data_import(resource["id"], import_request())
    return database, cluster, service, record


def test_cancellation_waits_for_scheduler_confirmation_and_is_idempotent(submitted_import):
    database, cluster, service, record = submitted_import
    first = service.cancel_data_import(record["id"])
    assert first["state"] == "CANCELLING"
    assert service.cancel_data_import(record["id"])["state"] == "CANCELLING"
    service.reconcile_data_imports()
    assert database.get_data_import(record["id"])["state"] == "CANCELLING"
    cluster.state = "CANCELLED by 1001"
    service.reconcile_data_imports()
    assert service.cancel_data_import(record["id"])["state"] == "CANCELLED"
    assert cluster.cancel_calls == [("1234", "sky2")]
    assert database.get_data_import(record["id"])["version_id"] is None


def test_stale_active_scheduler_response_does_not_undo_cancellation(submitted_import):
    database, cluster, service, record = submitted_import
    cluster.on_status = lambda: service.cancel_data_import(record["id"])
    service.reconcile_data_imports()
    current = database.get_data_import(record["id"])
    assert current["state"] == "CANCELLING"
    assert current["slurm_state"] == "RUNNING"


def test_completion_during_remote_cancel_remains_terminal(submitted_import):
    database, cluster, service, record = submitted_import

    def complete_while_cancel_is_pending():
        assert database.get_data_import(record["id"])["state"] == "CANCELLING"
        database.update_data_import(record["id"], state="SUCCEEDED", slurm_state="COMPLETED")

    cluster.on_cancel = complete_while_cancel_is_pending
    assert service.cancel_data_import(record["id"])["state"] == "SUCCEEDED"
    assert database.get_data_import(record["id"])["slurm_state"] == "COMPLETED"


@pytest.mark.parametrize("terminal_state", ["CANCELLED", "FAILED"])
def test_terminal_result_read_race_does_not_publish_registry_objects(submitted_import, terminal_state):
    database, cluster, service, record = submitted_import
    cluster.state = "COMPLETED"
    cluster.result = import_result(record)
    cluster.on_read = lambda: database.update_data_import(record["id"], state=terminal_state)

    service.reconcile_data_imports()

    current = database.get_data_import(record["id"])
    assert current["state"] == terminal_state
    assert current["version_id"] is None and current["bundle_id"] is None
    assert database.get_data_resource(record["resource_id"])["versions"] == []
    assert database.list_data_bundles() == []


def test_concurrent_completion_refresh_publishes_once_across_service_instances(submitted_import, monkeypatch):
    database, cluster, service, record = submitted_import
    cluster.state = "COMPLETED"
    cluster.result = import_result(record)
    simultaneous_reads = threading.Barrier(2)
    cluster.on_read = lambda: simultaneous_reads.wait(timeout=5)
    other_service = PipelineService(database, cluster)
    publish = PipelineService._publish_data_import_result
    published = []

    def observe_publication(self, claimed, result):
        published.append(claimed["id"])
        return publish(self, claimed, result)

    monkeypatch.setattr(PipelineService, "_publish_data_import_result", observe_publication)
    with ThreadPoolExecutor(max_workers=2) as executor:
        refreshes = [executor.submit(owner.reconcile_data_imports) for owner in (service, other_service)]
        for refresh in refreshes:
            assert refresh.result(timeout=5)[0]["state"] == "SUCCEEDED"

    current = database.get_data_import(record["id"])
    assert published == [record["id"]]
    assert current["error"] is None
    assert len(database.get_data_resource(record["resource_id"])["versions"]) == 1
    assert database.list_data_bundles() == []


def test_interrupted_finalization_can_resume_publication(submitted_import):
    database, cluster, service, record = submitted_import
    database.update_data_import(record["id"], state="FINALIZING")
    cluster.state = "COMPLETED"
    cluster.result = import_result(record)

    service.reconcile_data_imports()

    assert database.get_data_import(record["id"])["state"] == "SUCCEEDED"
    assert len(database.get_data_resource(record["resource_id"])["versions"]) == 1
    assert database.list_data_bundles() == []


def test_concurrent_cancellation_calls_issue_one_remote_request(submitted_import):
    database, cluster, service, record = submitted_import
    entered = threading.Event()
    release = threading.Event()

    def hold_cancel():
        entered.set()
        assert release.wait(timeout=5)

    cluster.on_cancel = hold_cancel
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.cancel_data_import, record["id"])
        try:
            assert entered.wait(timeout=5)
            second = executor.submit(service.cancel_data_import, record["id"])
            assert second.result(timeout=5)["state"] == "CANCELLING"
            assert database.get_data_import(record["id"])["state"] == "CANCELLING"
        finally:
            release.set()
        assert first.result(timeout=5)["state"] == "CANCELLING"
    assert len(cluster.cancel_calls) == 1


def test_failed_cancel_can_be_retried_without_overriding_completed_job(submitted_import):
    database, cluster, service, record = submitted_import

    def fail_cancel():
        raise ClusterError("gateway unavailable")

    cluster.on_cancel = fail_cancel
    with pytest.raises(ClusterError, match="gateway unavailable"):
        service.cancel_data_import(record["id"])
    current = database.get_data_import(record["id"])
    assert current["state"] == "SUBMITTED"
    assert "cancellation failed" in current["error"]
    cluster.on_cancel = None
    assert service.cancel_data_import(record["id"])["state"] == "CANCELLING"


def test_cancelling_import_still_claims_identity_despite_changed_budget(submitted_import):
    database, cluster, service, record = submitted_import
    service.cancel_data_import(record["id"])
    changed_budget = import_request().model_copy(update={"cpus": 2, "memory_gb": 4, "time_limit": "00:10:00", "gateway": "sky1"})
    with pytest.raises(ValueError, match="already active"):
        service.submit_data_import(record["resource_id"], changed_budget)
    assert len(database.list_data_imports()) == 1


def test_concurrent_import_identity_claim_is_atomic(tmp_path):
    database = Database(tmp_path / "skynet.db")
    resource = database.create_data_resource(category="dataset", **{key: RESOURCE[key] for key in ("provider", "namespace", "name", "kind")})
    barrier = threading.Barrier(2)

    def claim():
        barrier.wait(timeout=5)
        try:
            database.create_data_import(resource["id"], request=import_request().model_dump())
            return "created"
        except ValueError as error:
            assert "already active" in str(error)
            return "duplicate"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: claim(), range(2)))
    assert sorted(results) == ["created", "duplicate"]


@pytest.mark.parametrize(("field", "value"), [
    ("cpus", 0), ("cpus", 65), ("cpus", 1.5),
    ("memory_gb", 0), ("memory_gb", 513),
    ("time_limit", "00:00:59"), ("time_limit", "24:00:01"),
    ("time_limit", "00:60:00"), ("time_limit", "04:00:00\n#SBATCH --mem=999G"),
])
def test_import_resource_budget_rejects_invalid_limits(field, value):
    payload = {**import_request().model_dump(), field: value}
    with pytest.raises(ValidationError):
        HuggingFaceImportRequest.model_validate(payload)


def test_import_time_limit_normalizes_surrounding_whitespace_before_compilation():
    request = HuggingFaceImportRequest.model_validate({
        **import_request().model_dump(), "time_limit": "\n 04:00:00 \n",
    })
    assert request.time_limit == "04:00:00"
    job = build_huggingface_import_job("a" * 36, RESOURCE, request.model_dump())
    assert "#SBATCH --time=04:00:00\n" in job.script


@pytest.mark.parametrize(("cpus", "memory_gb", "time_limit"), [(1, 1, "00:01:00"), (64, 512, "1-00:00:00")])
def test_import_resource_budget_matches_generated_job(cpus, memory_gb, time_limit):
    request = HuggingFaceImportRequest.model_validate({**import_request().model_dump(), "cpus": cpus, "memory_gb": memory_gb, "time_limit": time_limit})
    job = build_huggingface_import_job("a" * 36, RESOURCE, request.model_dump())
    assert f"#SBATCH --cpus-per-task={cpus}\n" in job.script
    assert f"#SBATCH --mem={memory_gb}G\n" in job.script
    assert f"#SBATCH --time={time_limit}\n" in job.script
    payload = re.search(r"printf %s ([A-Za-z0-9+/=]+) \| base64 --decode > \"\$RUN_DIR/import-request.json\"", job.script)
    assert payload
    assert json.loads(base64.b64decode(payload.group(1)))["cpus"] == cpus


def test_download_worker_uses_allocated_cpu_budget(tmp_path):
    # Execute the actual worker function with a local cache and fake Hub so no
    # network or cluster job is involved in checking the concurrency budget.
    class RepoFile:
        path = "subset/file.parquet"
        size = 1

    snapshot = tmp_path / "snapshot"
    (snapshot / "subset").mkdir(parents=True)
    (snapshot / RepoFile.path).write_text("x")
    worker_counts = []

    def pool(*, max_workers):
        worker_counts.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    class Hub:
        def list_repo_tree(self, **kwargs):
            return [RepoFile()]

    function = next(node for node in ast.parse(_IMPORT_PROGRAM).body if isinstance(node, ast.FunctionDef) and node.name == "download_subset")
    namespace = {"Path": Path, "HfApi": Hub, "RepoFile": RepoFile, "hf_hub_download": lambda **kwargs: str(snapshot / kwargs["filename"]), "ThreadPoolExecutor": pool, "as_completed": as_completed}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "import-worker", "exec"), namespace)
    result = namespace["download_subset"]({"namespace": "test", "name": "dataset", "subset": "subset", "revision": "e" * 40, "cpus": 3}, tmp_path)
    assert result == snapshot
    assert worker_counts == [3]
