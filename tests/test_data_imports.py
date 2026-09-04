import json
from pathlib import Path
import tempfile

from skynet_app.cluster_runtime import Submission
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
        return "sky2", {
            "1234": {
                "State": "COMPLETED",
                "ExitCode": "0:0",
                "NodeList": "node1",
            }
        }

    def read_file(self, path: str, gateway: str = "auto", max_bytes: int = 20_000_000):
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


def test_submit_and_reconcile_publishes_version_and_bundle():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        resource = database.create_data_resource(
            provider=RESOURCE["provider"],
            namespace=RESOURCE["namespace"],
            name=RESOURCE["name"],
            kind=RESOURCE["kind"],
        )
        cluster = ImportCluster()
        service = PipelineService(database, cluster)
        submitted = service.submit_data_import(resource["id"], import_request())
        assert submitted["state"] == "SUBMITTED"
        cluster.result = {
            "schema_version": "skynet.data-import-result/v1",
            "import_id": submitted["id"],
            "resource_id": resource["id"],
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
        imports = service.reconcile_data_imports()
        completed = next(item for item in imports if item["id"] == submitted["id"])
        assert completed["state"] == "SUCCEEDED"
        assert completed["version_id"]
        assert completed["bundle_id"]
