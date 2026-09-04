from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import skynet_app.pipeline_api as pipeline_api
from skynet_app.cluster_runtime import ClusterError, Submission, SubmissionOutcomeUnknown
from skynet_app.adapters import (
    AdapterManifest,
    NativeTrackingIntegration,
    RepositoryArgumentValidation,
    builtin_adapter_manifests,
    resolve_adapter_plan,
)
from skynet_app.database import Database, content_sha256
from skynet_app.pipeline_api import EvaluationRequest, PipelineService, ResumeRequest, manual_run_actions
from skynet_app.source_validation import (
    CACHE_KIND as REPOSITORY_ARGUMENT_VALIDATION_CACHE_KIND,
    repository_argument_validation_cache_parameters,
)
from skynet_app.tracking import WandBSettings


COMMIT = "e17cf98fe4bc234c564b37abc9e155f25e76d566"


class FakeCluster:
    hosts = ("sky1", "sky2")

    def __init__(self) -> None:
        self.script = ""
        self.state = "RUNNING"
        self.submit_count = 0
        self.file_content = None
        self.log_content = None
        self.status_details = {}
        self.cancel_calls: list[tuple[str, str]] = []
        self.submit_environments: list[dict[str, str]] = []
        self.argument_validation_calls = 0
        self.run_commands: list[str] = []
        self.test_script_count = 0

    def test_script(self, script, gateway="auto"):
        self.test_script_count += 1
        self.script = script
        return "sky1", "Job 1 to start at 2026-01-01"

    def submit_script(
        self,
        script,
        run_id,
        gateway="auto",
        *,
        submission_key=None,
        forwarded_environment=None,
    ):
        self.script = script
        self.submit_count += 1
        self.submit_environments.append(dict(forwarded_environment or {}))
        return Submission("9001", "9001", "sky1", f"/coc/flash7/ycho420/jobs/runs/{run_id}/job.sbatch", f"/coc/flash7/ycho420/jobs/runs/{run_id}")

    def run_with_fallback(self, command, gateway="auto", **kwargs):
        self.run_commands.append(command)
        if "SKYNET_CHECKPOINT_PRESENT" in command:
            return "sky1", "SKYNET_CHECKPOINT_PRESENT"
        if "egomimic/hydra_configs/train_zarr_cartesian.yaml" in command:
            files = {
                "egomimic/hydra_configs/train_zarr_cartesian.yaml": "defaults:\n  - model: hpt_bc_flow_eva\n  - trainer: ddp\n  - data: eva\n",
                "egomimic/hydra_configs/model/hpt_bc_flow_eva.yaml": "optimizer:\n  lr: 1e-4\n",
                "egomimic/hydra_configs/data/eva.yaml": "train_dataloader_params:\n  eva_bimanual:\n    batch_size: 32\n    num_workers: 6\n",
                "egomimic/hydra_configs/trainer/ddp.yaml": "defaults:\n  - default\nstrategy: ddp\n",
                "egomimic/hydra_configs/trainer/default.yaml": "max_epochs: 2000\nprecision: bf16\nlimit_train_batches: 100\n",
            }
            lines = [f"COMMIT\t{COMMIT}"]
            for path, source in files.items():
                content = source.encode("utf-8")
                lines.extend(
                    [
                        f"FILE\t{path}\t{len(content)}\t{hashlib.sha256(content).hexdigest()}",
                        f"CONTENT\t{path}\t{base64.b64encode(content).decode('ascii')}",
                    ]
                )
            return "sky1", "\n".join(lines)
        return "sky1", "| rl2-lab | 6 / 8 | 15 / 16 | 0 / 0 | 40 / 424 | 15 / 16 |\n"

    def job_statuses(self, job_ids, gateway="auto"):
        return "sky1", {
            str(job_id): {
                "State": self.state,
                "ExitCode": "0:0",
                "Reason": "None",
                "NodeList": "node01",
                **self.status_details,
            }
            for job_id in job_ids
        }

    def cancel(self, job_id, gateway="auto"):
        self.cancel_calls.append((job_id, gateway))
        return gateway

    def read_log(self, *args, **kwargs):
        if self.log_content is None:
            raise ClusterError("not found")
        return "sky1", self.log_content

    def read_file(self, *args, **kwargs):
        if self.file_content is None:
            raise ClusterError("not found")
        return "sky1", self.file_content


class ToggleSubmissionFailureCluster(FakeCluster):
    def __init__(self) -> None:
        super().__init__()
        self.fail_quota = False
        self.fail_script_test = False
        self.quota_commands = []

    def run_with_fallback(self, command, gateway="auto", **kwargs):
        if "gpu_usage" in command:
            self.quota_commands.append(command)
        if self.fail_quota and "gpu_usage" in command:
            raise ClusterError(
                "sky2: ERROR:root:Error: The command 'sacctmgr' was not found"
            )
        return super().run_with_fallback(command, gateway, **kwargs)

    def test_script(self, script, gateway="auto"):
        if self.fail_script_test:
            raise ClusterError("sbatch validation unavailable")
        return super().test_script(script, gateway)


class ImmediateRecoveryCluster(FakeCluster):
    def submit_script(self, script, run_id, gateway="auto", *, submission_key=None):
        self.script = script
        self.submit_count += 1
        return Submission(
            "9002",
            "9002",
            "sky1",
            f"/coc/flash7/ycho420/jobs/runs/{run_id}/attempts/9002/job.sbatch",
            f"/coc/flash7/ycho420/jobs/runs/{run_id}",
            recovered=True,
        )


class DelayedRecoveryCluster(FakeCluster):
    def __init__(self) -> None:
        super().__init__()
        self.recovery_ready = False

    def submit_script(self, script, run_id, gateway="auto", *, submission_key=None):
        self.script = script
        self.submit_count += 1
        raise SubmissionOutcomeUnknown("SSH acknowledgement timed out")

    def recover_submission(self, run_id, submission_key, gateway="auto"):
        if not self.recovery_ready:
            return None
        return Submission(
            "9003",
            "9003",
            "sky1",
            f"/coc/flash7/ycho420/jobs/runs/{run_id}/attempts/9003/job.sbatch",
            f"/coc/flash7/ycho420/jobs/runs/{run_id}",
            recovered=True,
        )

    def read_file(self, path, gateway="auto", **kwargs):
        if path.endswith("/attempts/9003/job.sbatch"):
            return "sky1", self.script
        return super().read_file(path, gateway, **kwargs)


def canonical_spec():
    return {
        "apiVersion": "skynet.rl2/v1",
        "identity": {"project": "tests", "experiment": "pipeline"},
        "source": {
            "repository": "https://github.com/GaTech-RL2/EgoVerse",
            "revision": COMMIT,
            "adapter": "egoverse",
        },
        "runtime": {"backend": "existing", "bootstrap_uv": False},
        "train": {
            "max_steps": 1,
            "checkpoint": {
                "save_every_steps": 1000,
                "save_before_timeout_seconds": 60,
                "keep_last": 3,
                "auto_resume": True,
                "max_attempts": 2,
            },
        },
        "resources": {
            "queue_policy": "auto",
            "account": "rl2-lab",
            "partition": "rl2-lab",
            "gpu": {"mode": "explicit", "count": 2, "type": "a40"},
            "time_limit": "00:10:00",
        },
    }


def repository_validation_spec(database: Database, experiment: str) -> dict:
    manifest = next(
        item.model_dump(mode="json")
        for item in builtin_adapter_manifests()
        if item.slug == "generic"
    )
    manifest.update(
        {
            "slug": "repository_validation_fixture",
            "display_name": "Repository validation fixture",
            "default_repository": "https://example.test/repository-validation.git",
        }
    )
    manifest["capabilities"]["name"] = "repository_validation_fixture"
    manifest["train"] = {
        "argv": ["python", "train.py"],
        "parameter_flags": {
            "train.max_steps": {"flag": "--max-steps", "style": "separate"}
        },
        "argument_validation": {
            "argv": ["python", "validate.py", "{{generated_args}}"],
            "timeout_seconds": 5,
        },
    }
    database.create_adapter(name="Repository validation fixture", manifest=manifest)
    payload = canonical_spec()
    payload["identity"]["experiment"] = experiment
    payload["source"] = {
        "repository": "https://example.test/repository-validation.git",
        "revision": COMMIT,
        "adapter": "repository_validation_fixture",
    }
    payload["resources"]["queue_policy"] = "normal"
    return payload


def test_historical_executable_validation_contract_is_inert():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        payload = repository_validation_spec(database, "inert-validation-contract")

        preview = service.preview(payload)

        result = preview["argument_validations"][0]["result"]
        assert result["status"] == "passed"
        assert result["phase"] == "static_repository_metadata"
        assert result["heavy_execution"] is False
        assert "historical executable" in result["warnings"][0]
        assert cluster.argument_validation_calls == 0
        assert cluster.run_commands == []
        assert cluster.test_script_count == 0
        assert cluster.submit_count == 0


def test_preview_never_discovers_source_or_runtime_implicitly():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)

        fixture = repository_validation_spec(database, "no-implicit-discovery")
        symbolic = json.loads(json.dumps(fixture))
        symbolic["source"]["revision"] = "main"
        with pytest.raises(ValueError, match="pinned 40-character commit"):
            service.preview(symbolic)

        unresolved_runtime = json.loads(json.dumps(fixture))
        unresolved_runtime["runtime"] = {"backend": "auto"}
        with pytest.raises(ValueError, match="no exact pinned repository inspection"):
            service.preview(unresolved_runtime)

        assert cluster.run_commands == []
        assert cluster.test_script_count == 0
        assert cluster.submit_count == 0


def test_repository_argument_validation_success_is_cached_by_pinned_inputs():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        payload = repository_validation_spec(database, "native-validation-cache")

        first = service.preview(payload)
        second = service.preview(payload)

        assert first["blockers"] == []
        assert first["argument_validations"][0]["result"]["status"] == "passed"
        assert first["argument_validations"][0]["result"]["_cache"]["hit"] is False
        assert second["argument_validations"][0]["result"]["status"] == "passed"
        assert second["argument_validations"][0]["result"]["_cache"]["hit"] is True
        assert cluster.argument_validation_calls == 0
        with database.connection() as connection:
            row = connection.execute(
                "SELECT parameters_json FROM source_metadata_cache WHERE cache_kind = ?",
                (REPOSITORY_ARGUMENT_VALIDATION_CACHE_KIND,),
            ).fetchone()
        parameters = json.loads(row["parameters_json"])
        assert parameters["commit"] == COMMIT
        assert len(parameters["manifest_sha256"]) == 64
        assert len(parameters["argv_sha256"]) == 64

        payload["train"]["max_steps"] = 2
        changed = service.preview(payload)
        assert changed["argument_validations"][0]["result"]["status"] == "passed"
        assert changed["argument_validations"][0]["result"]["_cache"]["hit"] is False
        assert cluster.argument_validation_calls == 0


def test_repository_argument_validation_blocks_preview_create_and_submit():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        payload = repository_validation_spec(database, "native-validation-block")
        experiment = service.create_experiment(payload)
        assert cluster.argument_validation_calls == 0

        spec = service.normalize_spec(payload)
        manifest = AdapterManifest.model_validate(spec.source.adapter_manifest)
        plan = resolve_adapter_plan(spec)
        parameters = repository_argument_validation_cache_parameters(
            spec, manifest, plan, {}
        )
        service.source_metadata.put(
            REPOSITORY_ARGUMENT_VALIDATION_CACHE_KIND,
            spec.source.repository,
            parameters,
            {
                "schema_version": "skynet.static-repository-argument-validation/v1",
                "status": "failed",
                "phase": "static_repository_metadata",
                "validation_mode": "local_static_only",
                "valid": False,
                "heavy_execution": False,
                "errors": ["unknown native option --max-steps"],
                **parameters,
            },
        )

        with pytest.raises(ValueError, match="unknown native option"):
            service.submit_experiment(experiment["id"])
        run = database.get_run(experiment["runs"][0]["id"])
        assert cluster.submit_count == 0
        assert run["status"] == "DRAFT"
        assert run["attempts"] == []

        cached_preview = service.preview(payload)
        assert "unknown native option" in cached_preview["blockers"][0]["reasons"][0]
        known_invalid = json.loads(json.dumps(payload))
        known_invalid["identity"]["experiment"] = "known-invalid"
        with pytest.raises(ValueError, match="unknown native option"):
            service.create_experiment(known_invalid)
        assert len(database.list_experiments()) == 1


def test_changed_same_name_creates_revision_without_reconcile_lock_deadlock():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        service = PipelineService(database, FakeCluster())
        original = service.create_experiment(canonical_spec())
        changed = canonical_spec()
        changed["resources"]["time_limit"] = "00:11:00"

        revised = service.create_experiment(changed)

        assert revised["id"] == original["id"]
        assert revised["revision_count"] == 2
        assert revised["latest_revision"]["revision_number"] == 2


def test_native_wandb_first_sdk_attachment_allows_reserved_run_id(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        service = PipelineService(Database(Path(directory) / "skynet.db"), FakeCluster())
        payload = canonical_spec()
        payload["tracking"] = {
            "providers": [
                {
                    "provider": "wandb",
                    "enabled": True,
                    "entity": "team",
                    "project": "central-project",
                }
            ]
        }
        spec = service.normalize_spec(payload)
        plan = SimpleNamespace(
            native_tracking=[
                NativeTrackingIntegration(
                    provider="wandb",
                    parameter_paths={"enabled": "native.config.wandb_enabled"},
                )
            ]
        )
        monkeypatch.setattr(
            service,
            "_wandb_settings",
            lambda provider: WandBSettings(api_key="secret", entity="team"),
        )

        runtime = service._native_tracking_runtime(spec, plan, "run-id", "stage-id")

        assert runtime["environment"]["WANDB_RUN_ID"] == "run-id"
        assert runtime["environment"]["WANDB_RESUME"] == "allow"
        assert runtime["secret_contents"] == {
            "state/runtime-secrets/stage-id/wandb-api-key": "secret\n"
        }


def test_retroactive_tracking_attachment_is_terminal_idempotent_and_preserves_spec(
    monkeypatch,
):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        service = PipelineService(database, FakeCluster())
        experiment = service.create_experiment(canonical_spec())
        run_id = experiment["runs"][0]["id"]
        database.update_run(run_id, status="FAILED")
        before = database.get_run(run_id)
        original_spec = before["resolved_spec_json"]
        connections = {
            "wandb": {
                "provider": "wandb",
                "configured": True,
                "connected": True,
                "status": "connected",
                "base_url": "https://api.wandb.ai",
                "entity": "validated-team",
            },
            "mlflow": {
                "provider": "mlflow",
                "configured": False,
                "connected": False,
                "status": "not_configured",
            },
        }
        monkeypatch.setattr(
            service, "tracking_connections", lambda: {"connections": connections}
        )
        sync_calls = []

        def fake_sync(
            attached_run_id,
            status=None,
            *,
            providers=None,
            central_authoritative=False,
        ):
            sync_calls.append(
                {
                    "run_id": attached_run_id,
                    "status": status,
                    "providers": providers,
                    "central_authoritative": central_authoritative,
                }
            )
            database.upsert_tracking_binding(
                "wandb",
                "experiment",
                before["experiment_id"],
                remote_id="validated-team/pipeline",
                remote_url="https://wandb.ai/validated-team/pipeline",
                status="CONNECTED",
                metadata={"entity": "validated-team", "project": "pipeline"},
            )
            database.upsert_tracking_binding(
                "wandb",
                "run",
                attached_run_id,
                remote_id=attached_run_id,
                remote_url=(
                    "https://wandb.ai/validated-team/pipeline/runs/"
                    f"{attached_run_id}"
                ),
                status="FAILED",
                metadata={"entity": "validated-team", "project": "pipeline"},
            )

        monkeypatch.setattr(service, "_sync_tracking_outputs", fake_sync)

        actions = service.run_tracking_actions(before, connections)
        first = service.attach_run_tracking(run_id, "wandb")
        second = service.attach_run_tracking(run_id, "wandb")
        after = database.get_run(run_id)

        assert actions["wandb"]["enabled"] is True
        assert actions["wandb"]["label"] == "Connect W&B"
        assert actions["wandb"]["path"].endswith("/tracking/wandb/attach")
        assert first["remote_id"] == run_id
        assert second["remote_id"] == run_id
        assert len(sync_calls) == 1
        assert sync_calls[0]["status"] == "FAILED"
        assert sync_calls[0]["central_authoritative"] is True
        assert sync_calls[0]["providers"] == [{
            "provider": "wandb",
            "enabled": True,
            "base_url": "https://api.wandb.ai",
            "entity": "validated-team",
            "project": "pipeline",
            "run_name_template": None,
        }]
        assert after["resolved_spec_json"] == original_spec
        assert after["resolved_spec_json"]["tracking"]["providers"] == []
        assert service.run_tracking_actions(after, connections) == {}
        assert len(database.list_tracking_bindings("run", run_id)) == 1
        assert sum(
            event["event_type"] == "TRACKING_ATTACHED"
            for event in after["events"]
        ) == 1


def test_failed_wandb_binding_reopens_exact_same_run(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        database = Database(root / "skynet.db")
        service = PipelineService(database, FakeCluster())
        experiment = service.create_experiment(canonical_spec())
        run_id = experiment["runs"][0]["id"]
        run = database.get_run(run_id)
        spec = pipeline_api.ExperimentSpec.model_validate({
            **run["resolved_spec_json"],
            "tracking": {
                "providers": [{
                    "provider": "wandb",
                    "enabled": True,
                    "entity": "validated-team",
                    "project": "pipeline",
                }]
            },
        })
        remote_url = f"https://wandb.ai/validated-team/pipeline/runs/{run_id}"
        database.upsert_tracking_binding(
            "wandb",
            "run",
            run_id,
            remote_id=run_id,
            remote_url=remote_url,
            status="FAILED",
            metadata={
                "entity": "validated-team",
                "project": "pipeline",
                "endpoint": "https://api.wandb.ai",
            },
            last_error="prior attempt failed",
        )
        reopened = []

        class ExactBindingWandB:
            def __init__(self, _capsule, _settings):
                pass

            def ensure_experiment(self, entity, project):
                assert (entity, project) == ("validated-team", "pipeline")
                return SimpleNamespace(remote_id=f"{entity}/{project}", error=None)

            def binding(self, local_run_id):
                assert local_run_id == run_id
                return {"remote_id": run_id, "url": remote_url}

            def reopen_run(self, local_run_id, attempt_number):
                reopened.append((local_run_id, attempt_number))
                return SimpleNamespace(remote_id=local_run_id, error=None)

            def set_tags(self, local_run_id, tags):
                assert local_run_id == run_id
                assert tags["skynet.attempt_number"] == 2

        monkeypatch.setattr(pipeline_api, "WandBBridge", ExactBindingWandB)
        monkeypatch.setattr(
            service,
            "_wandb_settings",
            lambda _provider=None: WandBSettings(
                base_url="https://api.wandb.ai",
                api_key="secret",
                entity="validated-team",
            ),
        )

        service._start_tracking(
            spec,
            run,
            root / "capsules" / run_id,
            "3752691",
            providers=list(spec.tracking.providers),
            continuation_attempt_number=2,
        )

        binding = database.list_tracking_bindings("run", run_id)[0]
        assert reopened == [(run_id, 2)]
        assert binding["remote_id"] == run_id
        assert binding["remote_url"] == remote_url
        assert binding["status"] == "CONNECTED"
        assert binding["last_error"] is None


def test_create_submit_auto_overcap_and_reconcile(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")

        experiment = service.create_experiment(canonical_spec())
        submitted = service.submit_experiment(experiment["id"])

        assert experiment["variant_count"] == 1
        assert submitted["submitted"][0]["job_id"] == "9001"
        assert "#SBATCH --account=overcap" in cluster.script
        assert "#SBATCH --partition=overcap" in cluster.script
        run_id = experiment["runs"][0]["id"]
        submitted_run = database.get_run(run_id)
        assert submitted_run["status"] == "SUBMITTED"
        assert submitted_run["attempts"][0]["stdout_path"].endswith(
            "/pipeline-train-9001.out"
        )
        assert submitted_run["attempts"][0]["stderr_path"].endswith(
            "/pipeline-train-9001.err"
        )

        cluster.state = "PENDING"
        report = service.reconcile()
        pending = database.get_run(run_id)
        assert report["updated"] == 1
        assert pending["stages"][0]["status"] == "PENDING_SLURM"
        assert len(pending["attempts"]) == 1
        assert pending["attempts"][0]["started_at"] is None

        cluster.state = "RUNNING"
        service.reconcile()
        running = database.get_run(run_id)
        assert running["attempts"][0]["started_at"] is not None

        cluster.log_content = json.dumps({
            "path": f"/coc/flash7/ycho420/jobs/runs/{run_id}/checkpoints/step-1.ckpt",
            "final": True,
            "size_bytes": 12345,
            "file_count": 1,
            "is_directory": False,
            "sha256": "a" * 64,
        })
        cluster.state = "COMPLETED"
        report = service.reconcile()
        assert report["updated"] == 1
        completed = database.get_run(run_id)
        assert completed["status"] == "SUCCEEDED"
        assert completed["checkpoints"][0]["is_selected_for_inference"] == 1
        assert completed["checkpoints"][0]["size_bytes"] == 12345


def test_create_preflights_live_auto_queue_before_persisting_experiment():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = ToggleSubmissionFailureCluster()
        cluster.fail_quota = True
        service = PipelineService(database, cluster)

        with pytest.raises(ValueError, match="could not verify live GPU quota"):
            service.create_experiment(canonical_spec())

        assert cluster.quota_commands
        assert cluster.quota_commands[0].startswith("export PATH=")
        assert database.list_experiments() == []


def test_materialization_failure_rolls_back_new_experiment(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        service = PipelineService(database, FakeCluster())

        def fail_stage(*args, **kwargs):
            raise RuntimeError("stage persistence failed")

        monkeypatch.setattr(database, "create_stage", fail_stage)
        with pytest.raises(RuntimeError, match="stage persistence failed"):
            service.create_experiment(canonical_spec())

        assert database.list_experiments() == []


def test_locked_preflight_failure_is_idempotently_resubmitted(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = ToggleSubmissionFailureCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr(
            "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules"
        )
        experiment = service.create_experiment(canonical_spec())
        revision = experiment["latest_revision"]
        run = database.get_run(experiment["runs"][0]["id"])
        stage = next(
            item for item in run["stages"] if item["stage_type"] == "TRAIN"
        )
        assert database.claim_experiment_revision_submission(
            experiment["id"], revision["id"]
        )
        database.transition_workflow_state(
            stage_id=stage["id"],
            stage_updates={"status": "PENDING", "completed_at": None},
            run_id=run["id"],
            run_updates={"status": "PENDING", "completed_at": None},
        )
        database.update_experiment(experiment["id"], status="ACTIVE")

        replay = service.create_experiment(canonical_spec())
        repaired_run = database.get_run(run["id"])
        submitted = service.submit_experiment(replay["id"])

        assert replay["id"] == experiment["id"]
        assert replay["locked"] is False
        assert repaired_run["status"] == "DRAFT"
        assert repaired_run["stages"][0]["status"] == "DRAFT"
        assert submitted["idempotent_retry"] is False
        assert submitted["submitted"][0]["job_id"] == "9001"
        attempts = database.get_run(run["id"])["attempts"]
        assert len(attempts) == 1
        assert attempts[0]["slurm_job_id"] == "9001"


def test_changed_same_name_creates_revision_after_repairing_zero_attempt_orphan():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        service = PipelineService(database, FakeCluster())
        experiment = service.create_experiment(canonical_spec())
        revision = experiment["latest_revision"]
        run = database.get_run(experiment["runs"][0]["id"])
        stage = next(
            item for item in run["stages"] if item["stage_type"] == "TRAIN"
        )
        assert database.claim_experiment_revision_submission(
            experiment["id"], revision["id"]
        )
        database.transition_workflow_state(
            stage_id=stage["id"],
            stage_updates={"status": "PENDING", "completed_at": None},
            run_id=run["id"],
            run_updates={"status": "PENDING", "completed_at": None},
        )
        database.update_experiment(experiment["id"], status="ACTIVE")
        changed = canonical_spec()
        changed["resources"]["cpus_per_task"] += 1

        revised = service.create_experiment(changed)

        assert revised["id"] == experiment["id"]
        assert revised["revision_count"] == 2
        assert revised["latest_revision"]["revision_number"] == 2
        assert revised["latest_revision"]["submitted_at"] is None
        original = next(
            item for item in revised["revisions"] if item["revision_number"] == 1
        )
        assert original["submitted_at"] is None
        repaired_run = database.get_run(run["id"])
        assert repaired_run["status"] == "DRAFT"
        assert repaired_run["attempts"] == []


def test_failed_submit_preflight_never_persists_pending_without_attempt():
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = ToggleSubmissionFailureCluster()
        service = PipelineService(database, cluster)
        experiment = service.create_experiment(canonical_spec())
        run_id = experiment["runs"][0]["id"]
        cluster.fail_quota = True

        with pytest.raises(ValueError, match="could not verify live GPU quota"):
            service.submit_experiment(experiment["id"])

        detail = service.experiment_detail(experiment["id"])
        run = database.get_run(run_id)
        assert detail["locked"] is False
        assert run["status"] == "DRAFT"
        assert run["stages"][0]["status"] == "DRAFT"
        assert run["attempts"] == []


def test_changed_same_name_preserves_attempted_revision_and_creates_draft(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        service = PipelineService(database, FakeCluster())
        monkeypatch.setattr(
            "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules"
        )
        experiment = service.create_experiment(canonical_spec())
        run_id = experiment["runs"][0]["id"]
        submitted = service.submit_experiment(experiment["id"])
        assert submitted["submitted"][0]["job_id"] == "9001"
        changed = canonical_spec()
        changed["resources"]["cpus_per_task"] += 1

        revised = service.create_experiment(changed)

        assert revised["revision_count"] == 2
        assert revised["latest_revision"]["revision_number"] == 2
        assert revised["latest_revision"]["submitted_at"] is None
        attempted = database.get_run(run_id)["attempts"]
        assert len(attempted) == 1
        assert attempted[0]["slurm_job_id"] == "9001"


def test_known_pre_slurm_attempt_failure_can_retry_without_duplicate_job(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = ToggleSubmissionFailureCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr(
            "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules"
        )
        experiment = service.create_experiment(canonical_spec())
        run_id = experiment["runs"][0]["id"]
        cluster.fail_script_test = True

        failed = service.submit_experiment(experiment["id"])
        assert failed["submitted"][0]["status"] == "FAILED"
        assert database.get_run(run_id)["attempts"][0]["slurm_job_id"] is None

        cluster.fail_script_test = False
        retried = service.submit_experiment(experiment["id"])

        assert retried["idempotent_retry"] is True
        assert retried["submitted"][0]["job_id"] == "9001"
        attempts = database.get_run(run_id)["attempts"]
        assert len(attempts) == 2
        assert attempts[0]["status"] == "SUBMISSION_FAILED"
        assert attempts[1]["slurm_job_id"] == "9001"


def test_immediate_submission_recovery_uses_canonical_log_path_resolution(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = ImmediateRecoveryCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")

        experiment = service.create_experiment(canonical_spec())
        service.submit_experiment(experiment["id"])
        attempt = database.get_run(experiment["runs"][0]["id"])["attempts"][0]

        assert attempt["stdout_path"].endswith("/pipeline-train-9002.out")
        assert attempt["stderr_path"].endswith("/pipeline-train-9002.err")


def test_delayed_submission_recovery_resolves_pre_recorded_log_templates(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = DelayedRecoveryCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")

        experiment = service.create_experiment(canonical_spec())
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]
        pending_attempt = database.get_run(run_id)["attempts"][0]
        assert pending_attempt["stdout_path"].endswith("/pipeline-train-%j.out")
        assert pending_attempt["stderr_path"].endswith("/pipeline-train-%j.err")

        cluster.recovery_ready = True
        service.reconcile()
        recovered_attempt = database.get_run(run_id)["attempts"][0]
        assert recovered_attempt["slurm_job_id"] == "9003"
        assert recovered_attempt["stdout_path"].endswith("/pipeline-train-9003.out")
        assert recovered_attempt["stderr_path"].endswith("/pipeline-train-9003.err")


def test_reconcile_repairs_legacy_log_paths_from_archived_sbatch(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")

        experiment = service.create_experiment(canonical_spec())
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]
        attempt = database.get_run(run_id)["attempts"][0]
        database.update_job_attempt(attempt["id"], stdout_path=None, stderr_path=None)
        cluster.file_content = cluster.script

        service.reconcile()
        repaired = database.get_run(run_id)["attempts"][0]
        assert repaired["stdout_path"].endswith("/pipeline-train-9001.out")
        assert repaired["stderr_path"].endswith("/pipeline-train-9001.err")


def _declare_checkpoint_output(database, run_id):
    run = database.get_run(run_id)
    stage = run["stages"][0]
    resolved = stage["resolved_config_json"]
    resolved["plan"]["checkpoint_globs"] = ["checkpoints/*.ckpt"]
    database.update_stage(stage["id"], resolved_config_json=resolved)


def test_reconcile_fails_run_when_declared_checkpoint_marker_is_missing(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")
        experiment = service.create_experiment(canonical_spec())
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]
        _declare_checkpoint_output(database, run_id)

        cluster.state = "COMPLETED"
        service.reconcile()

        run = database.get_run(run_id)
        assert run["status"] == "FAILED"
        assert run["stages"][0]["status"] == "FAILED"
        assert run["attempts"][0]["status"] == "SUCCEEDED"
        assert run["attempts"][0]["exit_code"] == "0:0"
        assert run["attempts"][0]["started_at"] is not None
        assert run["checkpoints"] == []
        event = next(item for item in run["events"] if item["event_type"] == "CHECKPOINT_FINALIZATION_FAILED")
        assert event["new_status"] == "FAILED"
        assert event["details_json"]["process_status"] == "SUCCEEDED"
        assert "unavailable" in event["details_json"]["error"]


def test_reconcile_fails_run_when_required_checkpoint_registration_fails(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")
        experiment = service.create_experiment(canonical_spec())
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]
        _declare_checkpoint_output(database, run_id)
        cluster.log_content = json.dumps({
            "path": f"/coc/flash7/ycho420/jobs/runs/{run_id}/checkpoints/step-1.ckpt",
            "final": True,
            "size_bytes": 12345,
            "file_count": 1,
            "is_directory": False,
            "sha256": "a" * 64,
        })

        def fail_checkpoint_registration(*_args, **_kwargs):
            raise sqlite3.IntegrityError("forced checkpoint registration failure")

        monkeypatch.setattr(database, "create_checkpoint", fail_checkpoint_registration)
        cluster.state = "COMPLETED"
        service.reconcile()

        run = database.get_run(run_id)
        assert run["status"] == "FAILED"
        assert run["stages"][0]["status"] == "FAILED"
        assert run["attempts"][0]["status"] == "SUCCEEDED"
        assert run["attempts"][0]["exit_code"] == "0:0"
        event = next(item for item in run["events"] if item["event_type"] == "CHECKPOINT_FINALIZATION_FAILED")
        assert "could not be registered" in event["details_json"]["error"]


def test_failed_evaluation_preserves_successful_training_run_and_checkpoint(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr(
            "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules"
        )
        experiment = service.create_experiment(canonical_spec())
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]
        run = database.get_run(run_id)
        stage = run["stages"][0]
        attempt = run["attempts"][0]
        checkpoint = database.create_checkpoint(
            run_id,
            produced_by_attempt_id=attempt["id"],
            checkpoint_type="INFERENCE",
            path=f"/coc/flash7/ycho420/jobs/runs/{run_id}/checkpoints/step-1.ckpt",
            sha256="a" * 64,
            size_bytes=1,
            is_resumable=False,
            is_selected_for_inference=True,
        )
        suite = database.list_evaluation_suites()[0]
        with database.connection() as connection:
            connection.execute(
                "UPDATE workflow_stages SET stage_type = 'EVALUATE', status = 'SUBMITTED' WHERE id = ?",
                (stage["id"],),
            )
            connection.commit()
        database.update_run(
            run_id,
            status="SUCCEEDED",
            completed_at="2026-09-02T00:00:00.000Z",
        )
        evaluation = database.create_evaluation(
            run_id,
            stage_id=stage["id"],
            checkpoint_id=checkpoint["id"],
            evaluation_suite_id=suite["id"],
            evaluator_adapter=suite["evaluator_adapter"],
            evaluator_version=suite["evaluator_version"],
            suite_name=suite["name"],
            suite_version=suite["suite_version"],
            tasks=["fixture-task"],
            seeds=[0],
            episodes_per_task=1,
            status="SUBMITTED",
            result_path=f"/tmp/{stage['id']}/result.json",
        )

        cluster.state = "FAILED"
        report = service.reconcile()

        assert report["updated"] == 1
        loaded = database.get_run(run_id)
        assert loaded["status"] == "SUCCEEDED"
        assert loaded["completed_at"] == "2026-09-02T00:00:00.000Z"
        assert loaded["stages"][0]["status"] == "FAILED"
        assert loaded["checkpoints"][0]["status"] == "AVAILABLE"
        assert loaded["checkpoints"][0]["is_selected_for_inference"] == 1
        assert database.get_evaluation(evaluation["id"])["status"] == "FAILED"


def test_reconcile_resolves_decorated_cancelled_accounting_state(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")
        experiment = service.create_experiment(canonical_spec())
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]
        first_cancel = service.cancel_run(run_id)
        duplicate_cancel = service.cancel_run(run_id)

        cancelling = database.get_run(run_id)
        assert first_cancel["status"] == "CANCELLING"
        assert duplicate_cancel["status"] == "CANCELLING"
        assert cancelling["status"] == "CANCELLING"
        assert cancelling["stages"][0]["status"] == "CANCELLING"
        assert cancelling["attempts"][0]["status"] == "CANCELLING"
        assert cluster.cancel_calls == [("9001", "sky1")]
        assert sum(
            event["event_type"] == "CANCEL_REQUESTED"
            for event in cancelling["events"]
        ) == 1

        cluster.state = "CANCELLED by 3712043"
        cluster.status_details = {
            "StateRaw": "CANCELLED by 3712043",
            "CancelledBy": "3712043",
            "Start": "1788334844",
            "End": "1788336104",
        }

        report = service.reconcile()

        assert report["updated"] == 1
        run = database.get_run(run_id)
        attempt = run["attempts"][0]
        assert run["status"] == "CANCELLED"
        assert run["completed_at"] == "2026-09-02T08:01:44.000Z"
        assert run["stages"][0]["status"] == "CANCELLED"
        assert run["stages"][0]["completed_at"] == "2026-09-02T08:01:44.000Z"
        assert attempt["status"] == "CANCELLED"
        assert attempt["slurm_state"] == "CANCELLED"
        assert attempt["slurm_reason"] == "CANCELLED by 3712043"
        assert attempt["exit_code"] == "0:0"
        assert attempt["node_list"] == "node01"
        assert attempt["started_at"] == "2026-09-02T07:40:44.000Z"
        assert attempt["finished_at"] == "2026-09-02T08:01:44.000Z"
        event = next(item for item in run["events"] if item["event_type"] == "JOB_CANCELLED")
        assert event["details_json"]["state_raw"] == "CANCELLED by 3712043"
        assert event["details_json"]["accounting_start_raw"] == "1788334844"
        assert event["details_json"]["accounting_end_raw"] == "1788336104"
        assert len(run["attempts"]) == 1


def test_frontend_payload_resolves_to_canonical(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        service = PipelineService(Database(Path(directory) / "skynet.db"), FakeCluster())
        monkeypatch.setattr("skynet_app.pipeline_api._resolve_git_revision", lambda repo, revision: COMMIT)
        spec = service.normalize_spec({
            "name": "frontend",
            "adapter": "egoverse",
            "source": {"repository": "https://github.com/GaTech-RL2/EgoVerse", "revision": "main"},
            "runtime": {"type": "existing"},
            "hyperparameters": {
                "batch_size": 16,
                "gradient_accumulation": 2,
                "precision": "bf16",
            },
            "resources": {
                "queue_policy": "normal",
                "gpu_mode": "manual",
                "gpus_per_node": 1,
                "gpu_type": "a40",
                "nodes": 1,
                "time_limit": "04:00:00",
            },
            "sweep": {"definition": '{"model.encoder":["a","b"],"seed":[1,2]}'},
        })
        variants = service.preview(spec.model_dump(mode="json", by_alias=True))
        assert variants["variant_count"] == 4
        assert spec.source.revision == COMMIT
        assert spec.train.batch.gradient_accumulation_steps == 2
        assert spec.train.precision == "bf16"
        assert "precision" not in spec.native.overrides


def test_openpi_blank_frontend_parameters_preserve_repository_defaults(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        service = PipelineService(Database(Path(directory) / "skynet.db"), FakeCluster())
        monkeypatch.setattr(
            service, "_validate_repository_input_choices", lambda *args: {}
        )
        for batch_semantics in ("", "adapter-default"):
            spec = service.normalize_spec({
                "name": "openpi-defaults",
                "adapter": "openpi",
                "source": {
                    "repository": "https://github.com/Physical-Intelligence/openpi.git",
                    "revision": COMMIT,
                },
                "runtime": {"type": "existing"},
                "hyperparameters": {
                    "learning_rate": None,
                    "batch_semantics": batch_semantics,
                    "batch_size": None,
                    "gradient_accumulation": None,
                    "num_workers": None,
                    "max_steps": None,
                    "precision": "adapter-default",
                },
                "resources": {
                    "queue_policy": "normal",
                    "gpu_mode": "manual",
                    "gpus_per_node": 2,
                    "gpu_type": "l40s",
                    "time_limit": "04:00:00",
                },
                "native_overrides": ['config.config_name="pi05_libero"'],
            })

            plan = pipeline_api.resolve_adapter_plan(spec)
            assert spec.intent.explicit_parameters == []
            assert "batch_semantics" not in spec.train.hyperparameters
            assert plan.runnable is True
            assert plan.argv[plan.argv.index("--fsdp-devices") + 1] == "2"
            for flag in (
                "--batch-size",
                "--num-workers",
                "--num-train-steps",
                "--pytorch-training-precision",
                "--save-interval",
            ):
                assert flag not in plan.argv

            preview = service.preview(spec.model_dump(mode="json", by_alias=True))
            assert preview["blockers"] == []
            assert "--batch-size" not in preview["script"]


def test_openpi_frontend_global_batch_semantics_is_runnable_and_persisted(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        service = PipelineService(Database(Path(directory) / "skynet.db"), FakeCluster())
        monkeypatch.setattr(
            service, "_validate_repository_input_choices", lambda *args: {}
        )
        payload = {
            "name": "openpi-global-batch",
            "adapter": "openpi",
            "source": {
                "repository": "https://github.com/Physical-Intelligence/openpi.git",
                "revision": COMMIT,
            },
            "runtime": {"type": "existing"},
            "hyperparameters": {
                "batch_semantics": "global_before_accumulation",
                "batch_size": 32,
            },
            "resources": {
                "queue_policy": "normal",
                "gpu_mode": "manual",
                "gpus_per_node": 2,
                "gpu_type": "l40s",
                "time_limit": "04:00:00",
            },
            "native_overrides": ['config.config_name="pi05_libero"'],
        }

        spec = service.normalize_spec(payload)
        plan = pipeline_api.resolve_adapter_plan(spec)
        assert spec.train.batch.declared_semantics == "global_before_accumulation"
        assert spec.intent.explicit_parameters == [
            "train.batch.declared_semantics",
            "train.batch.value",
        ]
        assert plan.runnable is True
        assert plan.argv[plan.argv.index("--batch-size") + 1] == "32"

        experiment = service.create_experiment(payload)
        requested_spec = experiment["latest_revision"]["requested_spec_json"]
        assert requested_spec["train"]["batch"]["declared_semantics"] == (
            "global_before_accumulation"
        )
        assert "train.batch.declared_semantics" in requested_spec["intent"][
            "explicit_parameters"
        ]


def test_openpi_frontend_explicit_per_device_batch_blocks_multi_gpu(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        service = PipelineService(Database(Path(directory) / "skynet.db"), FakeCluster())
        monkeypatch.setattr(
            service, "_validate_repository_input_choices", lambda *args: {}
        )
        spec = service.normalize_spec({
            "name": "openpi-per-device-batch",
            "adapter": "openpi",
            "source": {
                "repository": "https://github.com/Physical-Intelligence/openpi.git",
                "revision": COMMIT,
            },
            "runtime": {"type": "existing"},
            "hyperparameters": {
                "batch_semantics": "per_device",
                "batch_size": 32,
            },
            "resources": {
                "queue_policy": "normal",
                "gpu_mode": "manual",
                "gpus_per_node": 2,
                "gpu_type": "l40s",
                "time_limit": "04:00:00",
            },
            "native_overrides": ['config.config_name="pi05_libero"'],
        })

        plan = pipeline_api.resolve_adapter_plan(spec)
        assert spec.train.batch.declared_semantics == "per_device"
        assert "train.batch.declared_semantics" in spec.intent.explicit_parameters
        assert plan.runnable is False
        assert any("--batch-size is global" in blocker for blocker in plan.blockers)
        assert not any("must be divisible" in blocker for blocker in plan.blockers)


def test_concurrent_dispatch_claims_stage_once(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")
        experiment = service.create_experiment(canonical_spec())
        run_id = experiment["runs"][0]["id"]
        stage = database.get_run(run_id)["stages"][0]
        database.update_stage(stage["id"], status="PENDING")
        database.update_run(run_id, status="PENDING")

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(
                lambda _: service._submit_stage(run_id, stage["id"], "auto"),
                range(2),
            ))

        assert cluster.submit_count == 1
        assert len(database.list_job_attempts(stage_id=stage["id"])) == 1
        assert any(result.get("skipped") for result in results)


def _mark_training_run_failed(database, run_id):
    run = database.get_run(run_id)
    stage = next(item for item in run["stages"] if item["stage_type"] == "TRAIN")
    attempt = next(item for item in run["attempts"] if item["stage_id"] == stage["id"])
    database.update_job_attempt(attempt["id"], status="FAILED")
    database.update_stage(stage["id"], status="FAILED")
    database.update_run(run_id, status="FAILED")
    return database.get_run(run_id), stage, attempt


def _mark_training_run_cancelled(database, run_id):
    run = database.get_run(run_id)
    stage = next(item for item in run["stages"] if item["stage_type"] == "TRAIN")
    attempt = next(item for item in run["attempts"] if item["stage_id"] == stage["id"])
    database.update_job_attempt(attempt["id"], status="CANCELLED")
    database.update_stage(stage["id"], status="CANCELLED")
    database.update_run(run_id, status="CANCELLED")
    return database.get_run(run_id), stage, attempt


def test_manual_run_actions_require_terminal_failure_checkpoint_and_attempt_budget(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        service = PipelineService(database, FakeCluster())
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")
        spec = canonical_spec()
        spec["train"]["checkpoint"]["save_every_steps"] = 1000
        experiment = service.create_experiment(spec)
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]

        active = manual_run_actions(database.get_run(run_id))
        assert not active["resume"]["enabled"]
        assert not active["rerun"]["enabled"]
        import skynet_app.pipeline_api as pipeline_api
        monkeypatch.setattr(pipeline_api, "service", service)
        assert pipeline_api.get_run(run_id)["run"]["manual_actions"] == active

        failed, stage, attempt = _mark_training_run_failed(database, run_id)
        actions = manual_run_actions(failed)
        assert not actions["resume"]["enabled"]
        assert "checkpoint" in actions["resume"]["reason"].lower()
        assert actions["rerun"]["enabled"]

        database.create_checkpoint(
            run_id,
            produced_by_attempt_id=attempt["id"],
            checkpoint_type="FULL_RESUME",
            path="/tmp/skynet-resumable-step-1.ckpt",
            training_step=1,
            is_resumable=True,
        )
        assert manual_run_actions(database.get_run(run_id))["resume"]["enabled"]

        database.update_stage(stage["id"], status="SUCCEEDED")
        database.update_run(run_id, status="SUCCEEDED")
        succeeded = manual_run_actions(database.get_run(run_id))
        assert not succeeded["resume"]["enabled"]
        assert succeeded["rerun"]["enabled"]

        database.update_stage(stage["id"], status="FAILED", max_attempts=1)
        database.update_run(run_id, status="FAILED")
        exhausted = manual_run_actions(database.get_run(run_id))
        assert exhausted["resume"]["enabled"]
        assert exhausted["rerun"]["enabled"]
        assert "checkpoint" in exhausted["resume"]["reason"].lower()


def test_rerun_is_checkpoint_free_new_run_with_pinned_variant(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")
        spec = canonical_spec()
        spec["train"]["checkpoint"]["save_every_steps"] = 1000
        experiment = service.create_experiment(spec)
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]
        failed, stage, attempt = _mark_training_run_failed(database, run_id)
        checkpoint_path = "/tmp/clean-retry-must-ignore-this.ckpt"
        database.create_checkpoint(
            run_id,
            produced_by_attempt_id=attempt["id"],
            checkpoint_type="FULL_RESUME",
            path=checkpoint_path,
            training_step=1,
            is_resumable=True,
        )

        result = service.rerun_run(run_id, "auto")

        assert result["status"] == "SUBMITTED"
        rerun_id = result["run_id"]
        assert rerun_id != run_id
        execution_path = next((Path(directory) / "capsules" / rerun_id).rglob("execution.json"))
        execution = json.loads(execution_path.read_text())
        assert execution["initial_checkpoint"] is None
        assert execution["auto_resume"] is True
        refreshed = database.get_run(rerun_id)
        refreshed_stage = refreshed["stages"][0]
        assert refreshed_stage["auto_resume"] is True
        assert refreshed["variant_id"] == failed["variant_id"]
        assert refreshed["experiment_revision_id"] == failed["experiment_revision_id"]
        assert refreshed["restarted_from_run_id"] == run_id
        assert refreshed["run_number"] == 2
        assert len(database.get_run(run_id)["attempts"]) == 1
        assert cluster.submit_count == 2


def test_manual_resume_submits_the_registered_resumable_checkpoint(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")
        spec = canonical_spec()
        spec["train"]["checkpoint"]["save_every_steps"] = 1000
        experiment = service.create_experiment(spec)
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]
        _, _, attempt = _mark_training_run_failed(database, run_id)
        checkpoint_path = "/tmp/manual-resume-step-1.ckpt"
        database.create_checkpoint(
            run_id,
            produced_by_attempt_id=attempt["id"],
            checkpoint_type="FULL_RESUME",
            path=checkpoint_path,
            training_step=1,
            is_resumable=True,
        )

        result = service.retry_run(run_id, "resume", "auto")

        assert result["status"] == "SUBMITTED"
        execution_path = next((Path(directory) / "capsules" / run_id).rglob("execution.json"))
        execution = json.loads(execution_path.read_text())
        assert execution["initial_checkpoint"] == checkpoint_path


def test_cancelled_run_resumes_from_valid_checkpoint_in_same_run(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        capsule_root = Path(directory) / "capsules"
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", capsule_root)
        spec = canonical_spec()
        spec["train"]["checkpoint"]["max_attempts"] = 3
        experiment = service.create_experiment(spec)
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]
        cancelled, _, first_attempt = _mark_training_run_cancelled(database, run_id)
        checkpoint = database.create_checkpoint(
            run_id,
            produced_by_attempt_id=first_attempt["id"],
            checkpoint_type="FULL_RESUME",
            path=(
                f"/coc/flash7/ycho420/jobs/runs/{run_id}/"
                "checkpoints/checkpoint-step-1.ckpt"
            ),
            sha256="a" * 64,
            training_step=1,
            is_resumable=True,
            status="AVAILABLE",
        )

        action = service.run_manual_actions(cancelled)["resume"]
        result = service.retry_run(run_id, "resume", "auto")

        assert action["enabled"]
        assert "checkpoint" in str(action["reason"]).lower()
        assert result["run_id"] == run_id
        assert result["status"] == "SUBMITTED"
        resumed = database.get_run(run_id)
        assert len(resumed["attempts"]) == 2
        second_attempt = resumed["attempts"][-1]
        assert second_attempt["attempt_number"] == 2
        assert second_attempt["resume_checkpoint_id"] == checkpoint["id"]
        assert second_attempt["execution_snapshot_json"]["migration_provenance"][
            "policy"
        ] == "strict_pinned_resume"
        execution_path = next((capsule_root / run_id).rglob("execution.json"))
        execution = json.loads(execution_path.read_text())
        assert execution["initial_checkpoint"] == checkpoint["path"]
        assert execution["auto_resume"] is True


def test_cancelled_run_without_checkpoint_replays_immutable_initial_execution(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        capsule_root = Path(directory) / "capsules"
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", capsule_root)

        manifest = next(
            item.model_dump(mode="json")
            for item in builtin_adapter_manifests()
            if item.slug == "generic"
        )
        manifest.update(
            {
                "slug": "cancel_replay_fixture",
                "display_name": "Cancellation replay fixture",
                "default_repository": "https://example.test/cancel-replay.git",
            }
        )
        manifest["capabilities"]["name"] = "cancel_replay_fixture"
        manifest["train"] = {
            "argv": ["python", "original-train.py"],
            "resume_argv": ["--resume", "{{tokens.resume_checkpoint}}"],
            "parameter_flags": {
                "train.max_steps": {
                    "flag": "--max-steps",
                    "style": "separate",
                }
            },
            "supported_canonical_fields": ["train.max_steps"],
        }
        adapter = database.create_adapter(
            name="Cancellation replay fixture", manifest=manifest
        )
        payload = canonical_spec()
        payload["identity"]["experiment"] = "cancel-replay"
        payload["source"] = {
            "repository": "https://example.test/cancel-replay.git",
            "revision": COMMIT,
            "adapter": "cancel_replay_fixture",
        }
        payload["train"]["checkpoint"]["max_attempts"] = 1
        experiment = service.create_experiment(payload)
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]
        submitted = database.get_run(run_id)
        initial_attempt = submitted["attempts"][0]
        initial_snapshot = initial_attempt["execution_snapshot_json"]

        drifted_manifest = json.loads(json.dumps(manifest))
        drifted_manifest["train"]["argv"] = ["python", "registry-drift.py"]
        database.edit_adapter(
            adapter["id"],
            manifest=drifted_manifest,
            expected_latest_version=1,
        )
        stage = submitted["stages"][0]
        drifted_stage_config = json.loads(json.dumps(stage["resolved_config_json"]))
        drifted_stage_config["plan"]["argv"] = ["python", "stage-drift.py"]
        database.update_stage(
            stage["id"], resolved_config_json=drifted_stage_config
        )
        cancelled, _, _ = _mark_training_run_cancelled(database, run_id)

        action = service.run_manual_actions(cancelled)["resume"]
        result = service.retry_run(run_id, "resume", "auto")

        assert action["enabled"]
        assert "initial" in str(action["reason"]).lower()
        assert result["run_id"] == run_id
        assert result["status"] == "SUBMITTED"
        replayed = database.get_run(run_id)
        assert len(replayed["attempts"]) == 2
        replay_attempt = replayed["attempts"][-1]
        replay_snapshot = replay_attempt["execution_snapshot_json"]
        assert replay_attempt["attempt_number"] == 2
        assert replay_attempt["resume_checkpoint_id"] is None
        assert replay_attempt["execution_snapshot_sha256"] == content_sha256(
            replay_snapshot
        )
        for key in (
            "resolved_spec",
            "adapter",
            "plan",
            "argv",
            "resume_argv",
            "repository_inputs",
            "common_hyperparameters",
        ):
            assert replay_snapshot[key] == initial_snapshot[key]
        assert "registry-drift.py" not in replay_snapshot["argv"]
        assert "stage-drift.py" not in replay_snapshot["argv"]
        execution_path = next((capsule_root / run_id).rglob("execution.json"))
        execution = json.loads(execution_path.read_text())
        assert execution["initial_checkpoint"] is None
        assert execution["auto_resume"] is False


@pytest.mark.parametrize("with_checkpoint", (False, True))
def test_terminal_timeout_manual_resume_ignores_automatic_attempt_budget(
    monkeypatch, with_checkpoint
):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        capsule_root = Path(directory) / "capsules"
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", capsule_root)
        spec = canonical_spec()
        spec["identity"]["experiment"] = (
            "timeout-checkpoint" if with_checkpoint else "timeout-initial-replay"
        )
        spec["train"]["checkpoint"]["max_attempts"] = 1
        experiment = service.create_experiment(spec)
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]
        run = database.get_run(run_id)
        stage = run["stages"][0]
        first_attempt = run["attempts"][0]
        initial_snapshot = first_attempt["execution_snapshot_json"]
        database.update_job_attempt(
            first_attempt["id"], status="TIMEOUT", slurm_state="TIMEOUT"
        )
        database.update_stage(stage["id"], status="TIMEOUT")
        database.update_run(run_id, status="TIMEOUT")
        checkpoint = None
        if with_checkpoint:
            checkpoint = database.create_checkpoint(
                run_id,
                produced_by_attempt_id=first_attempt["id"],
                checkpoint_type="FULL_RESUME",
                path=(
                    f"/coc/flash7/ycho420/jobs/runs/{run_id}/"
                    "checkpoints/checkpoint-step-1.ckpt"
                ),
                sha256="a" * 64,
                training_step=1,
                is_resumable=True,
                status="AVAILABLE",
            )
        terminal = database.get_run(run_id)
        action = service.run_manual_actions(terminal)["resume"]
        monkeypatch.setattr(pipeline_api, "service", service)
        app = FastAPI()
        app.include_router(pipeline_api.router)

        with TestClient(app) as client:
            response = client.post(
                f"/api/runs/{run_id}/resume",
                json={"mode": "resume", "gateway": "auto"},
            )

        assert action["enabled"]
        assert response.status_code == 200
        assert response.json()["run_id"] == run_id
        assert response.json()["status"] == "SUBMITTED"
        resumed = database.get_run(run_id)
        assert resumed["id"] == run_id
        assert len(resumed["attempts"]) == 2
        second_attempt = resumed["attempts"][-1]
        second_snapshot = second_attempt["execution_snapshot_json"]
        assert second_attempt["attempt_number"] == 2
        assert second_snapshot["adapter"] == initial_snapshot["adapter"]
        assert second_snapshot["argv"] == initial_snapshot["argv"]
        assert second_snapshot["resume_argv"] == initial_snapshot["resume_argv"]
        assert second_snapshot["common_hyperparameters"]["values"] == (
            initial_snapshot["common_hyperparameters"]["values"]
        )
        queued = next(
            event
            for event in resumed["events"]
            if event["event_type"] == "MANUAL_RESUME_QUEUED"
        )
        if checkpoint is not None:
            assert second_attempt["resume_checkpoint_id"] == checkpoint["id"]
            assert second_snapshot["migration_provenance"]["policy"] == (
                "strict_pinned_resume"
            )
            assert queued["details_json"]["strategy"] == "checkpoint_resume"
        else:
            assert second_attempt["resume_checkpoint_id"] is None
            assert second_snapshot["migration_provenance"]["policy"] == (
                "strict_pinned_initial_replay"
            )
            assert queued["details_json"]["strategy"] == "pinned_initial_replay"
            for key in ("resolved_spec", "plan", "repository_inputs"):
                assert second_snapshot[key] == initial_snapshot[key]


def test_active_attempt_blocks_manual_resume_despite_exhausted_automatic_budget(
    monkeypatch,
):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr(
            "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules"
        )
        spec = canonical_spec()
        spec["identity"]["experiment"] = "active-attempt-blocks-manual-resume"
        spec["train"]["checkpoint"]["max_attempts"] = 1
        experiment = service.create_experiment(spec)
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]
        run = database.get_run(run_id)
        stage = run["stages"][0]
        first_attempt = run["attempts"][0]
        database.update_job_attempt(
            first_attempt["id"], status="TIMEOUT", slurm_state="TIMEOUT"
        )
        active_attempt = database.create_job_attempt(
            stage["id"],
            status="RUNNING",
            slurm_job_id="9002",
            gateway="sky1",
            execution_snapshot_json=first_attempt["execution_snapshot_json"],
            execution_snapshot_sha256=first_attempt["execution_snapshot_sha256"],
        )
        # Model a stale parent status during reconciliation. Attempt state is the
        # authoritative guard against creating a concurrent third attempt.
        database.update_stage(stage["id"], status="TIMEOUT")
        database.update_run(run_id, status="TIMEOUT")
        stale_parent = database.get_run(run_id)

        action = service.run_manual_actions(stale_parent)["resume"]

        assert active_attempt["attempt_number"] == 2
        assert not action["enabled"]
        assert "active" in str(action["reason"]).lower()
        with pytest.raises(ValueError, match="[Aa]ctive"):
            service.retry_run(run_id, "resume", "auto")
        assert len(database.get_run(run_id)["attempts"]) == 2
        assert cluster.submit_count == 1


@pytest.mark.parametrize(
    ("evidence", "reason_fragment"),
    (("missing", "missing"), ("corrupt", "hash")),
)
def test_cancelled_run_without_checkpoint_blocks_invalid_pinned_evidence(
    monkeypatch, evidence, reason_fragment
):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr(
            "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules"
        )
        experiment = service.create_experiment(canonical_spec())
        run_id = experiment["runs"][0]["id"]
        run = database.get_run(run_id)
        stage = run["stages"][0]
        attempt_fields = {}
        if evidence == "corrupt":
            stored = stage["resolved_config_json"]
            spec = pipeline_api.ExperimentSpec.model_validate(stored["spec"])
            plan_payload = json.loads(json.dumps(stored["plan"]))
            plan_payload.pop("runnable", None)
            plan = pipeline_api.AdapterPlan.model_validate(plan_payload)
            snapshot = service._attempt_execution_snapshot(
                spec,
                plan,
                {
                    "policy": "pinned_variant",
                    "source": "variants.resolved_spec_json",
                    "source_spec_sha256": pipeline_api.canonical_sha256(
                        stored["spec"]
                    ),
                    "checkpoint": None,
                    "transformations": [],
                },
            )
            attempt_fields = {
                "execution_snapshot_json": snapshot,
                "execution_snapshot_sha256": "0" * 64,
            }
        database.create_job_attempt(
            stage["id"], status="CANCELLED", **attempt_fields
        )
        database.update_stage(stage["id"], status="CANCELLED")
        database.update_run(run_id, status="CANCELLED")
        cancelled = database.get_run(run_id)

        action = service.run_manual_actions(cancelled)["resume"]

        assert not action["enabled"]
        assert "snapshot" in str(action["reason"]).lower()
        assert reason_fragment in str(action["reason"]).lower()
        with pytest.raises(ValueError, match=reason_fragment):
            service.retry_run(run_id, "resume", "auto")
        assert len(database.get_run(run_id)["attempts"]) == 1
        assert cluster.submit_count == 0


def test_manual_retry_mode_is_rejected_in_favor_of_rerun(monkeypatch):
    try:
        ResumeRequest(mode="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid manual retry mode was accepted")

    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")
        experiment = service.create_experiment(canonical_spec())
        run_id = experiment["runs"][0]["id"]

        try:
            service.retry_run(run_id, "retry", "auto")
        except ValueError as error:
            assert "rerun" in str(error)
        else:
            raise AssertionError("deprecated retry mode was accepted")
        assert cluster.submit_count == 0


def test_new_revision_is_latest_scoped_and_can_submit_existing_draft(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")
        original_spec = canonical_spec()
        original_spec["train"]["checkpoint"]["save_every_steps"] = 1000
        experiment = service.create_experiment(original_spec)
        original_revision_id = experiment["latest_revision"]["id"]
        original_run_id = experiment["runs"][0]["id"]
        service.submit_experiment(experiment["id"])

        changed_spec = canonical_spec()
        changed_spec["train"]["checkpoint"]["save_every_steps"] = 1000
        changed_spec["train"]["learning_rate"] = 0.0002
        revised = service.create_experiment_revision(experiment["id"], changed_spec)

        assert revised["revision_number"] == 2
        assert revised["lifecycle"] == "DRAFT"
        assert not revised["locked"]
        assert revised["runs"][0]["id"] != original_run_id
        assert all(
            run["experiment_revision_id"] == revised["latest_revision"]["id"]
            for run in revised["runs"]
        )
        assert len(database.get_run(original_run_id)["attempts"]) == 1

        submitted = service.create_experiment_revision(
            experiment["id"], changed_spec, submit=True
        )

        assert submitted["lifecycle"] == "SUBMITTED"
        assert submitted["locked"]
        assert len(database.get_run(original_run_id)["attempts"]) == 1
        assert cluster.submit_count == 2
        assert original_revision_id != submitted["latest_revision"]["id"]
        try:
            service.create_experiment_revision(experiment["id"], changed_spec)
        except ValueError as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("duplicate experiment revision was accepted")


def _legacy_clean_retry_upgrade_scenario(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        capsule_root = Path(directory) / "capsules"
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", capsule_root)

        legacy_manifest = next(
            item.model_dump(mode="json")
            for item in builtin_adapter_manifests()
            if item.slug == "generic"
        )
        legacy_manifest.update(
            {
                "slug": "retry_fixture",
                "display_name": "Retry fixture",
                "default_repository": "https://example.test/retry-fixture.git",
            }
        )
        legacy_manifest["capabilities"]["name"] = "retry_fixture"
        legacy_manifest["train"] = {
            "argv": ["python", "train.py"],
            "resume_argv": ["--resume", "{{tokens.resume_checkpoint}}"],
            "parameter_flags": {
                "native.overrides.precision": {
                    "flag": "--precision",
                    "style": "separate",
                }
            },
        }
        adapter = database.create_adapter(
            name="Retry fixture", manifest=legacy_manifest
        )

        payload = canonical_spec()
        payload["identity"]["experiment"] = "clean-retry-upgrade"
        payload["source"] = {
            "repository": "https://example.test/retry-fixture.git",
            "revision": COMMIT,
            "adapter": "retry_fixture",
        }
        payload["native"] = {"overrides": {"precision": "bf16"}}
        payload["train"]["checkpoint"]["max_attempts"] = 4
        experiment = service.create_experiment(payload)
        run_id = experiment["runs"][0]["id"]
        original_run = database.get_run(run_id)
        original_stage = original_run["stages"][0]
        historical_config = original_stage["resolved_config_json"]
        assert "--precision" in historical_config["plan"]["argv"]
        service.submit_experiment(experiment["id"])
        _mark_training_run_failed(database, run_id)

        current_manifest = json.loads(json.dumps(legacy_manifest))
        current_manifest["train"]["parameter_flags"] = {
            "train.precision": {
                "flag": "--pytorch-training-precision",
                "style": "separate",
                "value_map": {"bf16": "bfloat16", "fp32": "float32"},
            }
        }
        current_manifest["train"]["retry_clean_argv"] = ["--fresh-output"]
        database.edit_adapter(
            adapter["id"],
            manifest=current_manifest,
            expected_latest_version=1,
        )
        action = service.run_manual_actions(database.get_run(run_id))["retry_clean"]
        assert action["enabled"]
        assert "pinned adapter v1" in action["reason"]
        assert "current adapter v2" in action["reason"]

        result = service.retry_run(run_id, "retry", "auto")
        assert result["status"] == "SUBMITTED"
        retried = database.get_run(run_id)
        assert retried["stages"][0]["resolved_config_json"] == historical_config
        clean_attempt = retried["attempts"][-1]
        clean_snapshot = clean_attempt["execution_snapshot_json"]
        assert clean_attempt["execution_snapshot_sha256"] == content_sha256(clean_snapshot)
        assert clean_snapshot["adapter"]["version"] == 2
        assert clean_snapshot["resolved_spec"]["train"]["precision"] == "bf16"
        assert "precision" not in clean_snapshot["resolved_spec"]["native"]["overrides"]
        assert "--precision" not in clean_snapshot["argv"]
        precision_index = clean_snapshot["argv"].index(
            "--pytorch-training-precision"
        )
        assert clean_snapshot["argv"][precision_index + 1] == "bfloat16"
        assert clean_snapshot["argv"][-1] == "--fresh-output"
        assert clean_snapshot["plan"]["retry_clean_argv"] == ["--fresh-output"]
        assert clean_snapshot["migration_provenance"]["transformations"] == [
            {
                "kind": "native_override_to_canonical_binding",
                "from": "native.overrides.precision",
                "to": "train.precision",
                "input_value": "bf16",
                "resolved_value": "bf16",
            },
            {
                "kind": "adapter_retry_clean_argv",
                "argv": ["--fresh-output"],
            },
        ]
        capsule_snapshot = json.loads(
            (capsule_root / run_id / "attempt-snapshot.json").read_text()
        )
        assert capsule_snapshot == clean_snapshot

        database.update_job_attempt(clean_attempt["id"], status="FAILED")
        database.update_stage(original_stage["id"], status="FAILED")
        database.update_run(run_id, status="FAILED")
        checkpoint = database.create_checkpoint(
            run_id,
            produced_by_attempt_id=clean_attempt["id"],
            checkpoint_type="FULL_RESUME",
            path="/tmp/retry-fixture.ckpt",
            training_step=1,
            is_resumable=True,
        )
        resumed = service.retry_run(run_id, "resume", "auto")
        assert resumed["status"] == "SUBMITTED"
        resume_attempt = database.get_run(run_id)["attempts"][-1]
        resume_snapshot = resume_attempt["execution_snapshot_json"]
        assert resume_attempt["resume_checkpoint_id"] == checkpoint["id"]
        assert resume_snapshot["adapter"]["version"] == 1
        assert "--precision" in resume_snapshot["argv"]
        assert "--pytorch-training-precision" not in resume_snapshot["argv"]
        assert "--fresh-output" not in resume_snapshot["argv"]
        assert resume_snapshot["migration_provenance"]["policy"] == "strict_pinned_resume"
        assert resume_snapshot["migration_provenance"]["checkpoint"] == {
            "id": checkpoint["id"],
            "path": checkpoint["path"],
        }


class AttemptLogCluster(FakeCluster):
    def __init__(self) -> None:
        super().__init__()
        self.logs: dict[str, str] = {}
        self.log_reads: list[tuple[str, str, int]] = []

    def read_log(self, path, gateway="auto", *, lines=500):
        self.log_reads.append((path, gateway, lines))
        if path not in self.logs:
            raise ClusterError("not found")
        return gateway, self.logs[path]


def _create_submitted_run(service: PipelineService, name: str) -> dict:
    spec = canonical_spec()
    spec["identity"]["experiment"] = name
    experiment = service.create_experiment(spec)
    service.submit_experiment(experiment["id"])
    return service.database.get_run(experiment["runs"][0]["id"])


def _create_active_evaluation(service: PipelineService, name: str) -> tuple[dict, dict]:
    database = service.database
    run = _create_submitted_run(service, name)
    train_stage = next(
        stage for stage in run["stages"] if stage["stage_type"] == "TRAIN"
    )
    train_attempt = next(
        attempt
        for attempt in run["attempts"]
        if attempt["stage_id"] == train_stage["id"]
    )
    database.update_job_attempt(
        train_attempt["id"],
        status="SUCCEEDED",
        slurm_state="COMPLETED",
        exit_code="0:0",
    )
    database.update_stage(train_stage["id"], status="SUCCEEDED")
    database.update_run(run["id"], status="SUCCEEDED")
    checkpoint_path = f"/tmp/{run['id']}/checkpoint.ckpt"
    database.create_checkpoint(
        run["id"],
        produced_by_attempt_id=train_attempt["id"],
        checkpoint_type="INFERENCE",
        path=checkpoint_path,
        sha256="a" * 64,
        size_bytes=1,
        is_selected_for_inference=True,
    )
    suite = database.list_evaluation_suites()[0]
    evaluation = service.create_evaluation(
        EvaluationRequest(
            run_id=run["id"],
            checkpoint_path=checkpoint_path,
            suite_id=suite["id"],
            tasks=[suite["config_json"]["tasks"][0]],
            episodes_per_task=1,
            seeds=[0],
            argv=["python3", "evaluate.py"],
        )
    )
    return database.get_run(run["id"]), evaluation


def test_evaluation_submission_forwards_only_explicit_operator_eula(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        cluster = FakeCluster()
        service = PipelineService(Database(Path(directory) / "skynet.db"), cluster)
        monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")

        _create_active_evaluation(service, "operator-eula-forwarding")

        assert cluster.submit_environments[-1] == {
            "OMNI_KIT_ACCEPT_EULA": "YES"
        }


def test_cancel_pending_training_stage_is_local_and_idempotent(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr(
            "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules"
        )
        experiment = service.create_experiment(canonical_spec())
        run_id = experiment["runs"][0]["id"]
        stage = database.get_run(run_id)["stages"][0]
        database.update_stage(stage["id"], status="PENDING")
        database.update_run(run_id, status="PENDING")

        first_cancel = service.cancel_run(run_id)
        duplicate_cancel = service.cancel_run(run_id)

        run = database.get_run(run_id)
        assert first_cancel["status"] == "CANCELLED"
        assert duplicate_cancel["status"] == "CANCELLED"
        assert run["status"] == "CANCELLED"
        assert run["stages"][0]["status"] == "CANCELLED"
        assert run["attempts"] == []
        assert cluster.cancel_calls == []
        event_types = [event["event_type"] for event in run["events"]]
        assert event_types.count("CANCEL_REQUESTED") == 1
        assert event_types.count("JOB_CANCELLED") == 1


def test_completed_training_wins_cancel_race(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr(
            "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules"
        )
        run = _create_submitted_run(service, "cancel-completion-race")
        service.cancel_run(run["id"])
        cluster.state = "COMPLETED"
        cluster.log_content = json.dumps(
            {
                "path": f"/coc/flash7/ycho420/jobs/runs/{run['id']}/checkpoints/checkpoint.ckpt",
                "final": True,
                "size_bytes": 1,
                "file_count": 1,
                "is_directory": False,
                "sha256": "a" * 64,
            }
        )

        service.reconcile()

        completed = database.get_run(run["id"])
        assert completed["status"] == "SUCCEEDED"
        assert completed["stages"][0]["status"] == "SUCCEEDED"
        assert completed["attempts"][0]["status"] == "SUCCEEDED"
        event_types = [event["event_type"] for event in completed["events"]]
        assert event_types.count("CANCEL_REQUESTED") == 1
        assert "JOB_CANCELLED" not in event_types


def test_run_and_evaluation_cancellation_are_stage_targeted(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr(
            "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules"
        )
        run, evaluation = _create_active_evaluation(service, "cancel-evaluation")

        with pytest.raises(ValueError, match="[Tt]raining"):
            service.cancel_run(run["id"])
        assert cluster.cancel_calls == []

        first_cancel = service.cancel_evaluation(evaluation["id"])
        duplicate_cancel = service.cancel_evaluation(evaluation["id"])

        loaded_run = database.get_run(run["id"])
        loaded_evaluation = database.get_evaluation(evaluation["id"])
        evaluation_stage = next(
            stage
            for stage in loaded_run["stages"]
            if stage["id"] == evaluation["stage_id"]
        )
        evaluation_attempt = next(
            attempt
            for attempt in loaded_run["attempts"]
            if attempt["stage_id"] == evaluation["stage_id"]
        )
        assert first_cancel["status"] == "CANCELLING"
        assert duplicate_cancel["status"] == "CANCELLING"
        assert loaded_run["status"] == "SUCCEEDED"
        assert loaded_evaluation["status"] == "CANCELLING"
        assert evaluation_stage["status"] == "CANCELLING"
        assert evaluation_attempt["status"] == "CANCELLING"
        assert cluster.cancel_calls == [(evaluation_attempt["slurm_job_id"], "sky1")]

        cluster.state = "CANCELLED"
        service.reconcile()

        cancelled_run = database.get_run(run["id"])
        assert cancelled_run["status"] == "SUCCEEDED"
        assert database.get_evaluation(evaluation["id"])["status"] == "CANCELLED"
        assert next(
            stage
            for stage in cancelled_run["stages"]
            if stage["id"] == evaluation["stage_id"]
        )["status"] == "CANCELLED"
        assert len(
            [
                attempt
                for attempt in cancelled_run["attempts"]
                if attempt["stage_id"] == evaluation["stage_id"]
            ]
        ) == 1


def test_cancel_routes_accept_empty_request_bodies(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr(
            "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules"
        )
        training_run = _create_submitted_run(service, "cancel-route-training")
        evaluation_run, evaluation = _create_active_evaluation(
            service, "cancel-route-evaluation"
        )
        monkeypatch.setattr(pipeline_api, "service", service)
        app = FastAPI()
        app.include_router(pipeline_api.router)

        with TestClient(app) as client:
            run_response = client.post(f"/api/runs/{training_run['id']}/cancel")
            evaluation_response = client.post(
                f"/api/evaluations/{evaluation['id']}/cancel"
            )
            wrong_stage_response = client.post(
                f"/api/runs/{evaluation_run['id']}/cancel"
            )

        assert run_response.status_code == 200
        assert run_response.json()["status"] == "CANCELLING"
        assert evaluation_response.status_code == 200
        assert evaluation_response.json()["status"] == "CANCELLING"
        assert wrong_stage_response.status_code == 422


def test_run_detail_attempts_expose_immutable_common_hyperparameters(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        service = PipelineService(database, FakeCluster())
        run = _create_submitted_run(service, "attempt-common-hyperparameters")
        stage = run["stages"][0]
        snapshotted = database.create_job_attempt(
            stage["id"],
            status="FAILED",
            execution_snapshot_json={
                "schema_version": "skynet.job-attempt/v1",
                "resolved_spec": {
                    "train": {
                        "learning_rate": None,
                        "batch": {
                            "declared_semantics": "global_effective",
                            "value": 48,
                            "gradient_accumulation_steps": 3,
                        },
                        "num_workers_per_rank": 7,
                        "max_steps": 321,
                        "precision": "fp16",
                    },
                    "intent": {
                        "explicit_parameters": [
                            "train.learning_rate",
                            "train.batch.declared_semantics",
                            "train.batch.value",
                            "train.batch.gradient_accumulation_steps",
                            "train.num_workers_per_rank",
                            "train.max_steps",
                            "train.precision",
                        ]
                    },
                },
            },
        )
        defaulted_snapshot = database.create_job_attempt(
            stage["id"],
            status="FAILED",
            execution_snapshot_json={
                "schema_version": "skynet.job-attempt/v1",
                "resolved_spec": {
                    "train": {
                        "learning_rate": None,
                        "batch": {
                            "declared_semantics": "per_device",
                            "value": 1,
                            "gradient_accumulation_steps": 1,
                        },
                        "num_workers_per_rank": 4,
                        "max_steps": 1000,
                        "precision": None,
                    },
                    "intent": {"explicit_parameters": []},
                },
            },
        )
        legacy = database.create_job_attempt(stage["id"], status="FAILED")
        monkeypatch.setattr(pipeline_api, "service", service)

        attempts = {
            attempt["id"]: attempt
            for attempt in pipeline_api.get_run(run["id"])["run"]["attempts"]
        }
        expected_keys = {
            "learning_rate",
            "batch_semantics",
            "batch_size",
            "gradient_accumulation",
            "num_workers",
            "max_steps",
            "precision",
        }
        assert attempts[snapshotted["id"]]["common_hyperparameters"] == {
            "learning_rate": None,
            "batch_semantics": "global_effective",
            "batch_size": 48,
            "gradient_accumulation": 3,
            "num_workers": 7,
            "max_steps": 321,
            "precision": "fp16",
        }
        assert attempts[defaulted_snapshot["id"]]["common_hyperparameters"] == {
            key: None for key in expected_keys
        }
        assert attempts[snapshotted["id"]]["common_hyperparameter_provenance"][
            "batch_size"
        ]["source"] == "explicit_spec"
        assert attempts[defaulted_snapshot["id"]]["common_hyperparameter_provenance"][
            "batch_size"
        ]["source"] == "not_applicable"
        original_attempt_id = run["attempts"][0]["id"]
        assert attempts[legacy["id"]]["common_hyperparameters"] == (
            attempts[original_attempt_id]["common_hyperparameters"]
        )
        assert all(
            set(attempt["common_hyperparameters"]) == expected_keys
            for attempt in attempts.values()
        )
        stored = attempts[original_attempt_id]["execution_snapshot_json"]
        assert stored["common_hyperparameters"]["schema_version"] == (
            "skynet.common-hyperparameters/v1"
        )


def test_effective_common_hyperparameters_use_pinned_manifest_defaults_not_schema_defaults():
    manifest = next(
        item for item in pipeline_api.builtin_adapter_manifests() if item.slug == "groot"
    ).model_dump(mode="json")
    spec = {
        "source": {"adapter_manifest": manifest},
        "intent": {"explicit_parameters": []},
        "train": {
            "batch": {"declared_semantics": "per_device", "value": 99},
            "num_workers_per_rank": 99,
            "max_steps": 1000,
        },
    }

    contract = pipeline_api._resolve_effective_common_hyperparameters(spec, manifest)

    assert contract["values"]["batch_size"] == 1
    assert contract["values"]["num_workers"] == 2
    assert contract["provenance"]["batch_size"]["source"] == (
        "adapter_manifest_default"
    )
    assert contract["values"]["max_steps"] is None


def test_effective_common_hyperparameters_prefer_explicit_then_pinned_repository_metadata():
    manifest = next(
        item for item in pipeline_api.builtin_adapter_manifests() if item.slug == "openpi"
    ).model_dump(mode="json")
    spec = {
        "source": {"repository": "https://example.test/openpi"},
        "intent": {"explicit_parameters": ["train.max_steps"]},
        "train": {
            "learning_rate": None,
            "batch": {"value": 1, "gradient_accumulation_steps": 1},
            "num_workers_per_rank": 4,
            "max_steps": 44,
            "precision": None,
        },
    }
    repository_inputs = {
        "native.config.config_name": {
            "value": "pi05_libero",
            "values": {
                "train.learning_rate": 5e-5,
                "train.batch.value": 256,
                "train.num_workers_per_rank": 2,
                "train.max_steps": 30_000,
                "train.precision": "bf16",
            },
            "evidence": {
                "train.batch.value": {
                    "file": "src/openpi/training/config.py",
                    "expression_sha256": "a" * 64,
                }
            },
            "source": {"commit": COMMIT, "rule_sha256": "b" * 64},
        }
    }

    contract = pipeline_api._resolve_effective_common_hyperparameters(
        spec, manifest, repository_inputs
    )

    assert contract["values"] == {
        "learning_rate": 5e-5,
        "batch_semantics": "global_before_accumulation",
        "batch_size": 256,
        "gradient_accumulation": 1,
        "num_workers": 2,
        "max_steps": 44,
        "precision": "bf16",
    }
    assert contract["provenance"]["batch_size"]["source"] == (
        "repository_inspection"
    )
    assert contract["provenance"]["batch_size"]["evidence"]["commit"] == COMMIT
    assert contract["provenance"]["max_steps"]["source"] == "explicit_spec"


def test_historical_attempt_enrichment_uses_exact_cache_and_write_once_receipt(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        service = PipelineService(database, FakeCluster())
        run = _create_submitted_run(service, "historical-common-hyperparameters")
        stage = run["stages"][0]
        manifest = next(
            item
            for item in pipeline_api.builtin_adapter_manifests()
            if item.slug == "openpi"
        ).model_dump(mode="json")
        config_field = next(
            field
            for field in manifest["train"]["input_fields"]
            if field["path"] == "native.config.config_name"
        )
        config_field["choice_source"].pop("metadata_fields", None)
        manifest["defaults"]["hyperparameters"].pop("batch_semantics", None)
        manifest["defaults"]["hyperparameters"][
            "gradient_accumulation_steps"
        ] = None
        manifest_sha256 = pipeline_api.canonical_sha256(manifest)
        resolved_spec = json.loads(json.dumps(run["resolved_spec_json"]))
        resolved_spec["source"].update(
            {
                "repository": "https://github.com/Physical-Intelligence/openpi",
                "revision": COMMIT,
                "adapter": "openpi",
                "adapter_manifest": manifest,
                "adapter_manifest_sha256": manifest_sha256,
                "project_subdirectory": ".",
            }
        )
        resolved_spec["intent"] = {"explicit_parameters": []}
        resolved_spec["native"]["config"] = {"config_name": "pi05_libero"}
        resolved_spec["train"].update(
            {
                "learning_rate": None,
                "num_workers_per_rank": 4,
                "max_steps": 1000,
                "precision": None,
            }
        )
        resolved_spec["train"]["batch"] = {
            "declared_semantics": "per_device",
            "value": 1,
            "gradient_accumulation_steps": 1,
        }
        snapshot = {
            "schema_version": "skynet.job-attempt/v1",
            "resolved_spec": resolved_spec,
            "resolved_spec_sha256": pipeline_api.canonical_sha256(resolved_spec),
            "adapter": {
                "slug": "openpi",
                "manifest": manifest,
                "manifest_sha256": manifest_sha256,
            },
        }
        historical = database.create_job_attempt(
            stage["id"],
            status="FAILED",
            execution_snapshot_json=snapshot,
            execution_snapshot_sha256=pipeline_api.content_sha256(snapshot),
        )
        run = database.get_run(run["id"])
        historical = next(
            item for item in run["attempts"] if item["id"] == historical["id"]
        )
        context = pipeline_api._attempt_pinned_enrichment_context(run, historical)
        choice_source = context["fields"][0]["choice_source"]
        resolved_values = {
            "train.learning_rate": 5e-5,
            "train.batch.value": 256,
            "train.num_workers_per_rank": 2,
            "train.max_steps": 30_000,
            "train.precision": "bf16",
        }
        source_paths = {
            item["canonical_path"]: item["source_path"]
            for item in choice_source["metadata_fields"]
        }
        entrypoint = choice_source["entrypoint"]
        files = [entrypoint, *choice_source["supporting_files"]]
        option = {
            "choices": ["pi05_libero"],
            "complete": True,
            "warnings": [],
            "metadata": {
                "pi05_libero": {
                    "values": resolved_values,
                    "evidence": {
                        path: {
                            "file": entrypoint,
                            "source_path": source_paths[path],
                            "origin": "constructor_argument",
                            "line": 1,
                            "column": 1,
                            "expression_sha256": "c" * 64,
                        }
                        for path in resolved_values
                    },
                }
            },
            "source": {
                "kind": "python_static_registry",
                "commit": COMMIT,
                "entrypoint": entrypoint,
                "registry": "_CONFIGS",
                "files": [
                    {"path": path, "size_bytes": 1, "sha256": "d" * 64}
                    for path in files
                ],
                "rule_sha256": pipeline_api.canonical_sha256(choice_source),
            },
        }
        monkeypatch.setattr(pipeline_api, "service", service)
        missing = pipeline_api._attempt_common_hyperparameter_resolution(
            service, run, historical, None
        )
        assert missing["status"] == "requires_explicit_fetch"
        with pytest.raises(pipeline_api.HTTPException) as error:
            pipeline_api.resolve_attempt_common_hyperparameters(
                run["id"],
                historical["id"],
                pipeline_api.ResolveAttemptCommonHyperparametersRequest(),
            )
        assert error.value.status_code == 409
        assert error.value.detail["code"] == "COMMON_HYPERPARAMETER_CACHE_MISS"
        service.source_metadata.put(
            "inspection",
            context["repository"],
            context["parameters"],
            {"input_options": {"native.config.config_name": option}},
        )

        before = pipeline_api.get_run(run["id"])["run"]
        before_attempt = next(
            item for item in before["attempts"] if item["id"] == historical["id"]
        )
        assert before_attempt["common_hyperparameters_resolution"]["status"] == (
            "available_from_cache"
        )

        result = pipeline_api.resolve_attempt_common_hyperparameters(
            run["id"],
            historical["id"],
            pipeline_api.ResolveAttemptCommonHyperparametersRequest(),
        )

        assert result["common_hyperparameters"] == {
            "learning_rate": 5e-5,
            "batch_semantics": "global_before_accumulation",
            "batch_size": 256,
            "gradient_accumulation": 1,
            "num_workers": 2,
            "max_steps": 30_000,
            "precision": "bf16",
        }
        assert result["common_hyperparameter_provenance"]["batch_size"]["source"] == (
            "repository_inspection"
        )
        assert result["common_hyperparameter_provenance"]["batch_semantics"][
            "source"
        ] == "historical_enrichment_rule"
        after = pipeline_api.get_run(run["id"])["run"]
        after_attempt = next(
            item for item in after["attempts"] if item["id"] == historical["id"]
        )
        assert after_attempt["common_hyperparameters_resolution"]["status"] == (
            "enriched_receipt"
        )
        assert after_attempt["execution_snapshot_json"] == snapshot
        pipeline_api.resolve_attempt_common_hyperparameters(
            run["id"],
            historical["id"],
            pipeline_api.ResolveAttemptCommonHyperparametersRequest(),
        )
        receipts = [
            event
            for event in database.list_events(
                entity_type="job_attempt", entity_id=historical["id"]
            )
            if event["event_type"] == "COMMON_HYPERPARAMETERS_ENRICHED_V1"
        ]
        assert len(receipts) == 1
        with pytest.raises(sqlite3.DatabaseError, match="receipts are immutable"):
            with database.transaction() as connection:
                connection.execute(
                    "UPDATE events SET details_json = '{}' WHERE id = ?",
                    (receipts[0]["id"],),
                )


def test_attempt_log_selects_requested_attempt_and_enforces_run_ownership(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = AttemptLogCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")

        first_run = _create_submitted_run(service, "attempt-log-first")
        first_attempt = first_run["attempts"][0]
        second_run = _create_submitted_run(service, "attempt-log-second")
        second_attempt = second_run["attempts"][0]
        newer_attempt = database.create_job_attempt(
            first_run["stages"][0]["id"],
            status="FAILED",
            gateway="sky2",
            stdout_path="/coc/flash7/ycho420/logs/newer.out",
            stderr_path="/coc/flash7/ycho420/logs/newer.err",
        )
        cluster.logs[first_attempt["stderr_path"]] = "older stderr"
        cluster.logs[newer_attempt["stderr_path"]] = "newer stderr"

        assert service.run_attempt_log(first_run["id"], first_attempt["id"], "stderr", 37) == "older stderr"
        assert cluster.log_reads[-1] == (first_attempt["stderr_path"], first_attempt["gateway"], 37)
        assert service.run_log(first_run["id"], "stderr", 500) == "newer stderr"

        for invalid_attempt_id in (second_attempt["id"], "missing-attempt"):
            try:
                service.run_attempt_log(first_run["id"], invalid_attempt_id, "stderr", 500)
            except KeyError as error:
                assert error.args == ("Attempt not found",)
            else:
                raise AssertionError("an attempt outside the selected run was accepted")


def test_attempt_log_endpoint_validates_stream_and_line_bounds(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = AttemptLogCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")
        run = _create_submitted_run(service, "attempt-log-api")
        attempt = run["attempts"][0]
        cluster.logs[attempt["stdout_path"]] = "selected stdout"
        monkeypatch.setattr(pipeline_api, "service", service)
        app = FastAPI()
        app.include_router(pipeline_api.router)

        with TestClient(app) as client:
            response = client.get(
                f"/api/runs/{run['id']}/attempts/{attempt['id']}/logs",
                params={"stream": "stdout", "lines": 19},
            )
            assert response.status_code == 200
            assert response.text == "selected stdout"
            assert cluster.log_reads[-1] == (attempt["stdout_path"], attempt["gateway"], 19)

            assert client.get(
                f"/api/runs/{run['id']}/attempts/{attempt['id']}/logs",
                params={"stream": "combined"},
            ).status_code == 422
            assert client.get(
                f"/api/runs/{run['id']}/attempts/{attempt['id']}/logs",
                params={"lines": 0},
            ).status_code == 422
            assert client.get(
                f"/api/runs/{run['id']}/attempts/{attempt['id']}/logs",
                params={"lines": 5001},
            ).status_code == 422
            assert client.get(
                f"/api/runs/{run['id']}/attempts/missing-attempt/logs"
            ).status_code == 404


def test_attempt_log_endpoint_distinguishes_content_from_retrieval_failure(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = AttemptLogCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")
        run = _create_submitted_run(service, "attempt-log-failure")
        attempt = run["attempts"][0]
        monkeypatch.setattr(pipeline_api, "service", service)
        app = FastAPI()
        app.include_router(pipeline_api.router)

        with TestClient(app) as client:
            content_that_looks_like_an_error = "not found\nLog path: emitted by the workload"
            cluster.logs[attempt["stderr_path"]] = content_that_looks_like_an_error
            response = client.get(
                f"/api/runs/{run['id']}/attempts/{attempt['id']}/logs"
            )
            assert response.status_code == 200
            assert response.text == content_that_looks_like_an_error

            del cluster.logs[attempt["stderr_path"]]
            response = client.get(
                f"/api/runs/{run['id']}/attempts/{attempt['id']}/logs"
            )
            assert response.status_code == 503
            assert response.json()["detail"] == "not found"


def test_evaluation_ingests_canonical_episode_ledger(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "skynet.db")
        cluster = FakeCluster()
        service = PipelineService(database, cluster)
        monkeypatch.setattr("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", Path(directory) / "capsules")
        experiment = service.create_experiment(canonical_spec())
        service.submit_experiment(experiment["id"])
        run_id = experiment["runs"][0]["id"]
        cluster.state = "COMPLETED"
        service.reconcile()
        suite = database.list_evaluation_suites()[0]
        task_id = suite["config_json"]["tasks"][0]
        database.create_checkpoint(
            run_id,
            checkpoint_type="INFERENCE",
            path="/coc/flash7/ycho420/artifacts/fake.ckpt",
            sha256="a" * 64,
            size_bytes=1,
            is_selected_for_inference=True,
        )
        cluster.file_content = json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "checkpoint": {"path": "/coc/flash7/ycho420/artifacts/fake.ckpt", "sha256": "a" * 64},
                "evaluator": {
                    "adapter": suite["evaluator_adapter"],
                    "version": suite["evaluator_version"],
                },
                "environment": {"suite": suite["name"], "version": suite["suite_version"]},
                "aggregate": [],
                "episodes": [
                    {
                            "task": task_id,
                        "seed": 7,
                        "episode_index": 0,
                        "success": True,
                        "reward": 1.0,
                        "episode_length": 3,
                        "status": "SUCCEEDED",
                    }
                ],
            }
        )
        evaluation = service.create_evaluation(
            EvaluationRequest(
                run_id=run_id,
                checkpoint_path="/coc/flash7/ycho420/artifacts/fake.ckpt",
                suite_id=suite["id"],
                    tasks=[task_id],
                episodes_per_task=1,
                seeds=[7],
                argv=["python3", "evaluate.py"],
            )
        )

        service.reconcile()
        loaded = database.get_evaluation(evaluation["id"])
        assert loaded["status"] == "SUCCEEDED"
        assert loaded["progress_completed"] == 1
        assert loaded["episodes"][0]["success"] is True
        assert loaded["episodes"][0]["reward"] == 1.0
        artifacts = database.get_run(run_id)["artifacts"]
        assert any(item["artifact_type"] == "EVALUATION_RESULT" for item in artifacts)
