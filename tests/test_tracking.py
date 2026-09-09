from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping
from types import SimpleNamespace
from unittest.mock import patch

from skynet_app.credential_store import (
    CredentialStoreUnavailable,
    KeyringCredentialStore,
)
from skynet_app.database import Database
from skynet_app.experiments import TrackingSpec
from skynet_app.pipeline_api import PipelineService, TrackingConnectionRequest, router

from skynet_app.tracking import (
    ATTEMPT_NUMBER_TAG,
    CURSOR_TAG,
    LOCAL_RUN_TAG,
    MLflowBridge,
    RUN_STATUS_TAG,
    TrackingRequestError,
    TrackingSettings,
    SESSION_CREDENTIALS,
    SessionCredentialStore,
    WANDB_TAG_METADATA_CONFIG_KEY,
    WandBBridge,
    WandBSettings,
    mlflow_experiment_url,
    mlflow_run_url,
    normalize_wandb_tag,
    sanitize,
    wandb_tag_label,
    wandb_web_base,
)


class FakePlatformKeyring:
    priority = 1

    def __init__(self, *, fail_writes: bool = False) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.fail_writes = fail_writes
        self.delete_calls = 0

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, password: str) -> None:
        if self.fail_writes:
            raise RuntimeError(f"simulated keychain failure containing {password}")
        self.values[(service, account)] = password

    def delete_password(self, service: str, account: str) -> None:
        self.delete_calls += 1
        self.values.pop((service, account), None)


class FakeMLflowBridge(MLflowBridge):
    def __init__(
        self, run_capsule: Path, settings: TrackingSettings | None = None
    ) -> None:
        self.experiments: dict[str, str] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.log_batch_calls = 0
        self.timeout_after_next_batch = False
        super().__init__(
            run_capsule,
            settings or TrackingSettings("http://mlflow.test:5000", auto_flush=True),
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = payload or {}
        if path.endswith("experiments/get-by-name"):
            name = str((query or {})["experiment_name"])
            if name not in self.experiments:
                raise TrackingRequestError("not found", status_code=404)
            return {"experiment": {"experiment_id": self.experiments[name], "name": name}}
        if path.endswith("experiments/create"):
            experiment_id = str(len(self.experiments) + 1)
            self.experiments[str(payload["name"])] = experiment_id
            return {"experiment_id": experiment_id}
        if path.endswith("runs/search"):
            local_run_id = str(payload["filter"]).rsplit("'", 2)[1]
            matching = [run for run in self.runs.values() if run["local_run_id"] == local_run_id]
            return {"runs": [{"info": {"run_id": run["run_id"]}} for run in matching]}
        if path.endswith("runs/create"):
            run_id = f"run-{len(self.runs) + 1}"
            tags = {tag["key"]: tag["value"] for tag in payload["tags"]}
            self.runs[run_id] = {
                "run_id": run_id,
                "local_run_id": tags[LOCAL_RUN_TAG],
                "tags": tags,
                "params": {},
                "metrics": [],
                "status": "RUNNING",
            }
            return {"run": {"info": {"run_id": run_id}}}
        if path.endswith("runs/get"):
            run = self.runs[str((query or {})["run_id"])]
            return {"run": {"info": {"run_id": run["run_id"]}, "data": {
                "tags": [{"key": key, "value": value} for key, value in run["tags"].items()]
            }}}
        if path.endswith("runs/log-batch"):
            self.log_batch_calls += 1
            run = self.runs[str(payload["run_id"])]
            run["params"].update({item["key"]: item["value"] for item in payload["params"]})
            run["metrics"].extend(payload["metrics"])
            run["tags"].update({item["key"]: item["value"] for item in payload["tags"]})
            if self.timeout_after_next_batch:
                self.timeout_after_next_batch = False
                raise TrackingRequestError("simulated response timeout")
            return {}
        if path.endswith("runs/update"):
            self.runs[str(payload["run_id"])]["status"] = payload["status"]
            return {}
        raise AssertionError(f"Unexpected fake request: {method} {path}")


class FakeWandBBridge(WandBBridge):
    def __init__(
        self,
        run_capsule: Path,
        settings: WandBSettings | None = None,
        *,
        api_key: str = "fake-wandb-key",
    ) -> None:
        self.remote_runs: dict[str, dict[str, Any]] = {}
        self.states: list[str] = []
        self.graphql_calls: list[tuple[str, dict[str, Any]]] = []
        self.history_rows: list[dict[str, Any]] = []
        super().__init__(
            run_capsule,
            settings or WandBSettings(api_key=api_key, entity="team", auto_flush=True),
        )

    def _append_history_rows(self, run, rows):
        self.history_rows.extend(dict(row) for row in rows)
        run["history_offset"] = int(run.get("history_offset") or 0) + len(rows)

    def _graphql(self, query: str, variables: Mapping[str, Any]) -> dict[str, Any]:
        self.graphql_calls.append((query, dict(variables)))
        if "SkynetViewer" in query:
            return {"viewer": {"id": "viewer-1", "username": "alice", "entity": "team"}}
        if "SkynetEntity" in query:
            return {"entity": {"id": "entity-1", "name": variables["name"], "readOnly": False}}
        if "SkynetCreateRun" in query:
            name = str(variables["name"])
            run = self.remote_runs.setdefault(name, {
                "id": f"storage-{name}",
                "name": name,
                "config": {},
                "summary": {},
                "tags": [],
            })
            if variables.get("state"):
                run["state"] = str(variables["state"])
                self.states.append(str(variables["state"]))
            return {"upsertBucket": {"bucket": {
                "id": run["id"], "name": name, "displayName": run.get("displayName")
            }, "inserted": len(self.remote_runs) == 1}}
        if "SkynetUpdateRun" in query:
            run = next(
                item for item in self.remote_runs.values() if item["id"] == variables["id"]
            )
            run["displayName"] = variables.get("displayName") or run.get("displayName")
            run["tags"] = list(variables.get("tags") or [])
            run["config"] = json.loads(str(variables["config"]))
            run["group"] = variables.get("groupName") or run.get("group")
            return {"upsertBucket": {"bucket": {
                "id": run["id"], "name": run["name"], "displayName": run.get("displayName")
            }}}
        if "SkynetUpdateSummary" in query:
            run = next(
                item for item in self.remote_runs.values() if item["id"] == variables["id"]
            )
            run["summary"] = json.loads(str(variables["summaryMetrics"]))
            return {"upsertBucket": {"bucket": {"id": run["id"]}}}
        raise AssertionError(f"Unexpected fake W&B query: {query}")


class TrackingSettingsTestCase(unittest.TestCase):
    def test_environment_and_secret_sanitization(self) -> None:
        settings = TrackingSettings.from_env({
            "MLFLOW_TRACKING_URI": "https://alice:hidden@example.test/mlflow?token=query-secret",
            "MLFLOW_TRACKING_TOKEN": "runtime-token",
            "MLFLOW_TRACKING_USERNAME": "alice",
            "MLFLOW_TRACKING_PASSWORD": "hidden",
            "MLFLOW_HTTP_REQUEST_TIMEOUT": "9",
        })

        public = settings.public_dict()
        representation = repr(settings)
        sanitized = sanitize(
            {"api_key": "key-value", "note": "runtime-token", "tokenizer": "paligemma"},
            secrets=(settings.token,),
        )

        self.assertNotIn("hidden", representation)
        self.assertNotIn("runtime-token", representation)
        self.assertNotIn("query-secret", public["tracking_uri"])
        self.assertEqual(public["timeout_seconds"], 9.0)
        self.assertEqual(sanitized["api_key"], "[REDACTED]")
        self.assertEqual(sanitized["note"], "[REDACTED]")
        self.assertEqual(sanitized["tokenizer"], "paligemma")


class OfflineTrackingTestCase(unittest.TestCase):
    def test_offline_operation_is_sanitized_and_queued(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capsule = Path(directory) / "run-capsule"
            bridge = MLflowBridge(
                capsule,
                TrackingSettings(None, token="runtime-token", auto_flush=True),
            )

            result = bridge.log_params("local-run", {
                "learning_rate": 1e-4,
                "hf_token": "runtime-token",
            })
            spool = capsule.joinpath("mlflow-spool.jsonl").read_text(encoding="utf-8")

            self.assertTrue(result.queued)
            self.assertFalse(result.delivered)
            self.assertEqual(bridge.pending_count(), 1)
            self.assertNotIn("runtime-token", spool)
            self.assertIn("[REDACTED]", spool)


class DeliveryTestCase(unittest.TestCase):
    def test_create_log_finish_and_retry_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = FakeMLflowBridge(Path(directory) / "run-capsule")

            experiment = bridge.ensure_experiment("manipulation")
            run = bridge.ensure_run(
                experiment_name="manipulation",
                local_run_id="local-42",
                run_name="encoder-resnet50-seed42",
                tags={"adapter": "groot"},
            )
            params = bridge.log_params("local-42", {"learning_rate": 1e-4, "batch_size": 64})
            bridge.timeout_after_next_batch = True
            interrupted = bridge.log_metrics("local-42", {"train/loss": 0.25}, step=10)
            calls_after_timeout = bridge.log_batch_calls
            recovered = bridge.drain_spool()
            artifact = bridge.log_artifact_link(
                "local-42",
                name="checkpoint",
                uri="/artifacts/model/checkpoint",
                artifact_type="checkpoint",
                metadata={"large": "metadata stays in the local manifest"},
            )
            finished = bridge.finish_run("local-42", status="FINISHED")

            remote = bridge.runs[run.remote_id]
            self.assertTrue(experiment.delivered)
            self.assertTrue(run.delivered)
            self.assertTrue(params.delivered)
            self.assertTrue(interrupted.queued)
            self.assertEqual(recovered.remaining, 0)
            self.assertEqual(bridge.log_batch_calls, calls_after_timeout + 2)
            self.assertEqual(len(remote["metrics"]), 1)
            self.assertGreaterEqual(int(remote["tags"][CURSOR_TAG]), interrupted.sequence)
            self.assertEqual(remote["status"], "FINISHED")
            self.assertTrue(finished.delivered)
            self.assertTrue(artifact.delivered)
            self.assertIn("skynet.artifact_manifest", remote["tags"])
            manifest = json.loads(
                bridge.run_capsule.joinpath("tracking-artifact-links.json").read_text()
            )
            self.assertEqual(len(manifest["artifacts"]), 1)
            self.assertEqual(bridge.pending_count(), 0)
            self.assertEqual(bridge.get_run("local-42")["info"]["run_id"], run.remote_id)

    def test_reopen_existing_mlflow_run_is_idempotent_per_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = FakeMLflowBridge(Path(directory) / "run-capsule")
            bridge.ensure_experiment("manipulation")
            created = bridge.ensure_run(
                experiment_name="manipulation",
                local_run_id="local-42",
                run_name="encoder-resnet50-seed42",
            )
            bridge.finish_run("local-42", status="KILLED")

            first = bridge.reopen_run("local-42", 2)
            duplicate = bridge.reopen_run("local-42", 2)

            remote = bridge.runs[created.remote_id]
            self.assertTrue(first.delivered)
            self.assertEqual(first.event_id, duplicate.event_id)
            self.assertEqual(first.sequence, duplicate.sequence)
            self.assertEqual(len(bridge.runs), 1)
            self.assertEqual(remote["status"], "RUNNING")
            self.assertEqual(remote["tags"][RUN_STATUS_TAG], "running")
            self.assertEqual(remote["tags"][ATTEMPT_NUMBER_TAG], "2")
            self.assertEqual(bridge.binding("local-42")["remote_id"], created.remote_id)


class WandBDeliveryTestCase(unittest.TestCase):
    def test_tag_normalization_boundaries_unicode_and_idempotence(self) -> None:
        one_character = "x"
        boundary = "b" * 64
        over_boundary = "c" * 65
        long_unicode = "robot-机器人-🤖-e\u0301/" * 12

        self.assertEqual(normalize_wandb_tag(one_character), one_character)
        self.assertEqual(normalize_wandb_tag(boundary), boundary)
        self.assertEqual(normalize_wandb_tag(""), "_")
        for original in (over_boundary, long_unicode):
            normalized = normalize_wandb_tag(original)
            self.assertEqual(len(normalized), 64)
            self.assertNotEqual(normalized, original)
            self.assertEqual(normalize_wandb_tag(original), normalized)
            self.assertEqual(normalize_wandb_tag(normalized), normalized)
        self.assertNotEqual(
            normalize_wandb_tag("shared-prefix-" + "a" * 80),
            normalize_wandb_tag("shared-prefix-" + "b" * 80),
        )
        self.assertEqual(
            wandb_tag_label("metadata", {"b": 2, "a": 1}),
            wandb_tag_label("metadata", {"a": 1, "b": 2}),
        )

    def test_graphql_payload_normalizes_every_tag_and_preserves_full_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = FakeWandBBridge(Path(directory) / "run-capsule")
            revision_id = "5455dc15-94e7-479c-bac6-6f6bf8563e18"
            tag_key = "skynet.experiment_revision_id"
            original_tag = f"{tag_key}:{revision_id}"

            bridge.ensure_run(
                entity="team",
                project="manipulation",
                local_run_id="local-run",
                run_name="run",
                group="revision-1",
                tags={tag_key: revision_id, "adapter": "openpi"},
                config={"source_commit": "full-metadata-remains-unchanged"},
            )

            update_payload = next(
                variables
                for query, variables in bridge.graphql_calls
                if "SkynetUpdateRun" in query
            )
            outbound_tags = update_payload["tags"]
            outbound_config = json.loads(update_payload["config"])
            self.assertEqual(len(original_tag), 66)
            self.assertNotIn(original_tag, outbound_tags)
            self.assertIn("adapter:openpi", outbound_tags)
            self.assertTrue(all(1 <= len(tag) <= 64 for tag in outbound_tags))
            self.assertEqual(
                bridge.binding("local-run")["tags"][tag_key],
                revision_id,
            )
            self.assertEqual(
                outbound_config["source_commit"]["value"],
                "full-metadata-remains-unchanged",
            )
            self.assertEqual(
                outbound_config[WANDB_TAG_METADATA_CONFIG_KEY]["value"],
                {tag_key: revision_id, "adapter": "openpi"},
            )
            self.assertNotIn(
                WANDB_TAG_METADATA_CONFIG_KEY,
                bridge.binding("local-run")["config"],
            )
            self.assertEqual(
                bridge.binding("local-run")["url"],
                "https://wandb.ai/team/manipulation/runs/local-run",
            )

    def test_queued_long_tag_is_normalized_when_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capsule = Path(directory) / "run-capsule"
            tag_key = "skynet.experiment_revision_id"
            revision_id = "5455dc15-94e7-479c-bac6-6f6bf8563e18"
            offline = WandBBridge(
                capsule,
                WandBSettings(api_key=None, entity="team", auto_flush=True),
            )
            queued = offline.ensure_run(
                entity="team",
                project="manipulation",
                local_run_id="queued-run",
                run_name="queued",
                group="revision-1",
                tags={tag_key: revision_id},
            )
            spooled = json.loads(
                capsule.joinpath("wandb-spool.jsonl").read_text(encoding="utf-8")
            )

            self.assertTrue(queued.queued)
            self.assertEqual(spooled["payload"]["tags"][tag_key], revision_id)

            replay = FakeWandBBridge(capsule)
            report = replay.drain_spool()
            outbound_tags = replay.remote_runs["queued-run"]["tags"]

            self.assertEqual(report.delivered, 1)
            self.assertEqual(report.remaining, 0)
            self.assertEqual(outbound_tags, [normalize_wandb_tag(f"{tag_key}:{revision_id}")])
            self.assertEqual(
                replay.remote_runs["queued-run"]["config"]
                [WANDB_TAG_METADATA_CONFIG_KEY]["value"][tag_key],
                revision_id,
            )

    def test_stable_run_id_metadata_summary_finish_and_retry_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capsule = Path(directory) / "run-capsule"
            bridge = FakeWandBBridge(capsule)
            local_id = "3f0bd131-29b1-4a8f-b277-b68bb91d72ec"

            self.assertEqual(bridge.validate_connection()["entity"], "team")
            self.assertEqual(bridge.validate_entity("team")["name"], "team")
            bridge.ensure_experiment("team", "manipulation")
            created = bridge.ensure_run(
                entity="team",
                project="manipulation",
                local_run_id=local_id,
                run_name="encoder/seed-42",
                group="encoder/revision-1",
                tags={"slurm.job_id": "100"},
                config={"learning_rate": 1e-4},
            )
            bridge.ensure_run(
                entity="team",
                project="manipulation",
                local_run_id=local_id,
                run_name="encoder/seed-42",
                group="encoder/revision-1",
                tags={"slurm.job_id": "101"},
                config={"batch_size": 32},
            )
            bridge.log_metrics(local_id, {"evaluation/success_rate": 0.75})
            bridge.log_artifact_link(
                local_id, name="checkpoint", uri="/artifacts/checkpoint", artifact_type="checkpoint"
            )
            finished = bridge.finish_run(local_id, status="FINISHED")

            binding = bridge.binding(local_id)
            remote = bridge.remote_runs[local_id]
            self.assertTrue(created.delivered)
            self.assertTrue(finished.delivered)
            self.assertEqual(created.remote_id, local_id)
            self.assertEqual(binding["remote_id"], local_id)
            self.assertEqual(len(bridge.remote_runs), 1)
            self.assertEqual(remote["state"], "running")
            self.assertEqual(remote["summary"]["skynet/status"], "FINISHED")
            self.assertIn("slurm.job_id:101", remote["tags"])
            self.assertEqual(remote["config"]["batch_size"]["value"], 32)
            self.assertEqual(remote["summary"]["evaluation/success_rate"], 0.75)
            self.assertEqual(
                remote["summary"]["skynet/artifact/checkpoint"], "/artifacts/checkpoint"
            )
            self.assertEqual(bridge.states, ["running"])

    def test_reopen_existing_wandb_run_preserves_identity_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = FakeWandBBridge(Path(directory) / "run-capsule")
            local_id = "local-resumed-run"
            created = bridge.ensure_run(
                entity="team",
                project="manipulation",
                local_run_id=local_id,
                run_name="resumed",
                group="revision-1",
            )
            bridge.log_metrics(local_id, {"loss": 0.5}, step=1)
            bridge.finish_run(local_id, status="FAILED")
            before = bridge.binding(local_id)

            first = bridge.reopen_run(local_id, 2)
            duplicate = bridge.reopen_run(local_id, 2)
            after = bridge.binding(local_id)

            self.assertTrue(first.delivered)
            self.assertEqual(first.event_id, duplicate.event_id)
            self.assertEqual(first.sequence, duplicate.sequence)
            self.assertEqual(len(bridge.remote_runs), 1)
            self.assertEqual(after["remote_id"], created.remote_id)
            self.assertEqual(after["storage_id"], before["storage_id"])
            self.assertEqual(after["history_offset"], before["history_offset"])
            self.assertEqual(after["state"], "running")
            self.assertEqual(after["summary"]["skynet/status"], "RUNNING")
            self.assertEqual(after["tags"][RUN_STATUS_TAG], "running")
            self.assertEqual(after["tags"][ATTEMPT_NUMBER_TAG], 2)
            self.assertEqual(bridge.remote_runs[local_id]["state"], "running")
            self.assertEqual(bridge.states, ["running", "running"])

            next_attempt = bridge.reopen_run(local_id, 3)
            self.assertNotEqual(next_attempt.event_id, first.event_id)
            self.assertEqual(bridge.binding(local_id)["history_offset"], before["history_offset"])

    def test_metric_history_preserves_step_time_and_deduplicates_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = FakeWandBBridge(Path(directory) / "run-capsule")
            local_id = "local-progress-run"
            bridge.ensure_run(
                entity="team",
                project="manipulation",
                local_run_id=local_id,
                run_name="progress",
                group="progress/revision-1",
            )

            first = bridge.log_metrics(
                local_id,
                {"training/completed": 42, "training/progress": 0.42},
                step=42,
                timestamp_ms=1_750_000_000_000,
                idempotency_key="training-progress:sample-42",
            )
            duplicate = bridge.log_metrics(
                local_id,
                {"training/completed": 42, "training/progress": 0.42},
                step=42,
                timestamp_ms=1_750_000_000_000,
                idempotency_key="training-progress:sample-42",
            )

            self.assertEqual(first.event_id, duplicate.event_id)
            self.assertEqual(len(bridge.history_rows), 1)
            self.assertEqual(bridge.history_rows[0]["_step"], 42)
            self.assertEqual(bridge.history_rows[0]["_timestamp"], 1_750_000_000.0)
            self.assertEqual(
                bridge.remote_runs[local_id]["summary"]["training/completed"], 42
            )


class TrackingConnectionSecurityTestCase(unittest.TestCase):
    def tearDown(self) -> None:
        SESSION_CREDENTIALS.clear("wandb")
        SESSION_CREDENTIALS.clear("mlflow")

    def test_wandb_key_can_be_session_only_and_is_never_echoed(self) -> None:
        secret = "wandb-secret-never-persist"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = Database(root / "skynet.db")
            service = PipelineService(database, object())
            with patch("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", root / "capsules"), patch.object(
                WandBBridge,
                "validate_connection",
                return_value={"viewer_id": "1", "username": "alice", "entity": "team"},
            ), patch.object(
                WandBBridge,
                "validate_entity",
                return_value={"id": "entity-1", "name": "team"},
            ):
                response = service.configure_tracking_connection(
                    "wandb", TrackingConnectionRequest(api_key=secret, remember=False)
                )
                listed = service.tracking_connections()

            serialized = json.dumps({"response": response, "listed": listed})
            self.assertNotIn(secret, serialized)
            self.assertEqual(response["connection"]["credential_source"], "session")
            self.assertNotIn(secret.encode(), database.path.read_bytes())
            for path in (root / "capsules").rglob("*"):
                if path.is_file():
                    self.assertNotIn(secret.encode(), path.read_bytes())

    def test_failed_wandb_validation_persists_nothing_and_redacts_error(self) -> None:
        secret = "bad-wandb-secret"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = Database(root / "skynet.db")
            service = PipelineService(database, object())
            with patch("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", root / "capsules"), patch.object(
                WandBBridge,
                "validate_connection",
                side_effect=TrackingRequestError(f"denied credential {secret}"),
            ):
                with self.assertRaises(TrackingRequestError) as raised:
                    service.configure_tracking_connection(
                        "wandb", TrackingConnectionRequest(api_key=secret)
                    )
            self.assertNotIn(secret, str(raised.exception))
            self.assertIsNone(database.get_tracking_connection("wandb"))
            self.assertNotIn(secret.encode(), database.path.read_bytes())

    def test_mlflow_token_is_session_only(self) -> None:
        secret = "mlflow-secret-never-persist"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = Database(root / "skynet.db")
            service = PipelineService(database, object())
            with patch("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", root / "capsules"), patch.object(
                MLflowBridge, "validate_connection", return_value={"reachable": True}
            ):
                response = service.configure_tracking_connection(
                    "mlflow",
                    TrackingConnectionRequest(
                        tracking_uri="https://mlflow.example.test",
                        token=secret,
                        remember=False,
                    ),
                )
            self.assertNotIn(secret, json.dumps(response))
            self.assertEqual(response["connection"]["credential_source"], "session")
            self.assertNotIn(secret.encode(), database.path.read_bytes())

    def test_remembered_wandb_key_restores_after_backend_restart(self) -> None:
        secret = "restart-safe-wandb-secret"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "skynet.db"
            platform_keyring = FakePlatformKeyring()
            credential_store = KeyringCredentialStore(keyring_backend=platform_keyring)
            first_session = SessionCredentialStore()
            first_service = PipelineService(
                Database(database_path),
                object(),
                credential_store=credential_store,
                session_credentials=first_session,
            )
            with patch("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", root / "capsules"), patch.object(
                WandBBridge,
                "validate_connection",
                return_value={"viewer_id": "1", "username": "alice", "entity": "team"},
            ), patch.object(
                WandBBridge,
                "validate_entity",
                return_value={"id": "entity-1", "name": "team"},
            ):
                connected = first_service.configure_tracking_connection(
                    "wandb", TrackingConnectionRequest(api_key=secret)
                )

            restarted_service = PipelineService(
                Database(database_path),
                object(),
                credential_store=KeyringCredentialStore(keyring_backend=platform_keyring),
                session_credentials=SessionCredentialStore(),
            )
            restored = restarted_service.tracking_connections()["connections"]["wandb"]

            self.assertEqual(connected["connection"]["credential_source"], "credential_store")
            self.assertTrue(restored["connected"])
            self.assertEqual(restored["credential_source"], "credential_store")
            self.assertEqual(restarted_service._wandb_settings().api_key, secret)
            self.assertNotIn(secret, json.dumps({"connected": connected, "restored": restored}))
            self.assertNotIn(secret, repr(credential_store.load("wandb")))
            self.assertNotIn(secret.encode(), database_path.read_bytes())

    def test_remembered_mlflow_basic_auth_restores_after_backend_restart(self) -> None:
        password = "restart-safe-mlflow-password"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "skynet.db"
            platform_keyring = FakePlatformKeyring()
            first_service = PipelineService(
                Database(database_path),
                object(),
                credential_store=KeyringCredentialStore(keyring_backend=platform_keyring),
                session_credentials=SessionCredentialStore(),
            )
            with patch("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", root / "capsules"), patch.object(
                MLflowBridge, "validate_connection", return_value={"reachable": True}
            ):
                first_service.configure_tracking_connection(
                    "mlflow",
                    TrackingConnectionRequest(
                        tracking_uri="https://mlflow.example.test",
                        username="alice",
                        password=password,
                    ),
                )

            restarted_service = PipelineService(
                Database(database_path),
                object(),
                credential_store=KeyringCredentialStore(keyring_backend=platform_keyring),
                session_credentials=SessionCredentialStore(),
            )
            restored = restarted_service.tracking_connections()["connections"]["mlflow"]
            settings = restarted_service._mlflow_settings()

            self.assertTrue(restored["connected"])
            self.assertEqual(restored["credential_source"], "credential_store")
            self.assertEqual(settings.username, "alice")
            self.assertEqual(settings.password, password)
            self.assertNotIn(password, json.dumps(restored))
            self.assertNotIn(password.encode(), database_path.read_bytes())

    def test_disconnect_deletes_remembered_credential(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            platform_keyring = FakePlatformKeyring()
            credential_store = KeyringCredentialStore(keyring_backend=platform_keyring)
            service = PipelineService(
                Database(root / "skynet.db"),
                object(),
                credential_store=credential_store,
                session_credentials=SessionCredentialStore(),
            )
            with patch("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", root / "capsules"), patch.object(
                WandBBridge,
                "validate_connection",
                return_value={"viewer_id": "1", "username": "alice", "entity": "team"},
            ), patch.object(
                WandBBridge,
                "validate_entity",
                return_value={"id": "entity-1", "name": "team"},
            ):
                service.configure_tracking_connection(
                    "wandb", TrackingConnectionRequest(api_key="delete-me")
                )
            self.assertIsNotNone(credential_store.load("wandb"))

            disconnected = service.disconnect_tracking_connection("wandb")

            self.assertIsNone(credential_store.load("wandb"))
            self.assertEqual(platform_keyring.delete_calls, 1)
            self.assertIsNone(service.database.get_tracking_connection("wandb"))
            self.assertIsNone(disconnected["connection"]["credential_source"])

    def test_environment_backed_disconnect_does_not_delete_external_credential(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"WANDB_API_KEY": "environment-only-secret"}, clear=False
        ):
            root = Path(directory)
            platform_keyring = FakePlatformKeyring()
            service = PipelineService(
                Database(root / "skynet.db"),
                object(),
                credential_store=KeyringCredentialStore(keyring_backend=platform_keyring),
                session_credentials=SessionCredentialStore(),
            )
            with patch("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", root / "capsules"), patch.object(
                WandBBridge,
                "validate_connection",
                return_value={"viewer_id": "1", "username": "alice", "entity": "team"},
            ), patch.object(
                WandBBridge,
                "validate_entity",
                return_value={"id": "entity-1", "name": "team"},
            ):
                connected = service.configure_tracking_connection(
                    "wandb", TrackingConnectionRequest()
                )
            disconnected = service.disconnect_tracking_connection("wandb")

            self.assertEqual(connected["connection"]["credential_source"], "environment")
            self.assertEqual(disconnected["connection"]["credential_source"], "environment")
            self.assertEqual(platform_keyring.delete_calls, 0)

    def test_unavailable_keychain_is_explicit_and_never_leaks_secret(self) -> None:
        secret = "unavailable-keychain-secret"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = PipelineService(
                Database(root / "skynet.db"),
                object(),
                credential_store=KeyringCredentialStore(
                    keyring_backend=FakePlatformKeyring(fail_writes=True)
                ),
                session_credentials=SessionCredentialStore(),
            )
            with patch("skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", root / "capsules"), patch.object(
                WandBBridge,
                "validate_connection",
                return_value={"viewer_id": "1", "username": "alice", "entity": "team"},
            ), patch.object(
                WandBBridge,
                "validate_entity",
                return_value={"id": "entity-1", "name": "team"},
            ):
                with self.assertRaises(CredentialStoreUnavailable) as raised:
                    service.configure_tracking_connection(
                        "wandb", TrackingConnectionRequest(api_key=secret)
                    )

            public = service.tracking_connections()["connections"]["wandb"]
            self.assertIn("OS credential store write failed", str(raised.exception))
            self.assertNotIn(secret, str(raised.exception))
            self.assertNotIn(secret, json.dumps(public))
            self.assertIsNone(service.database.get_tracking_connection("wandb"))

    def test_credentials_are_bound_to_validated_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "skynet.db")
            service = PipelineService(database, object())
            database.upsert_tracking_connection(
                "mlflow", endpoint="https://trusted.example/mlflow"
            )
            SESSION_CREDENTIALS.replace("mlflow", {"token": "session-secret"})
            with self.assertRaisesRegex(ValueError, "does not match"):
                service._mlflow_settings({
                    "provider": "mlflow",
                    "tracking_uri": "https://attacker.example/collect",
                })

    def test_remote_plaintext_and_nested_credentials_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = PipelineService(Database(Path(directory) / "skynet.db"), object())
            with self.assertRaisesRegex(ValueError, "must use HTTPS"):
                service._validate_tracking_endpoint("http://example.test/mlflow", "MLflow")
            self.assertEqual(
                service._validate_tracking_endpoint("http://127.0.0.1:5000", "MLflow"),
                "http://127.0.0.1:5000",
            )
            with self.assertRaisesRegex(ValueError, "credential-like"):
                service._reject_tracking_secrets({
                    "tags": {"safe": "ok", "nested": {"service_token": "secret"}}
                })

    def test_enabled_provider_requires_connection_but_disabled_never_uses_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"MLFLOW_TRACKING_URI": "https://environment.example"}
        ):
            service = PipelineService(Database(Path(directory) / "skynet.db"), object())
            enabled = SimpleNamespace(tracking=TrackingSpec.model_validate({
                "providers": [{"provider": "mlflow", "enabled": True}]
            }))
            disabled = SimpleNamespace(tracking=TrackingSpec.model_validate({
                "providers": [{"provider": "mlflow", "enabled": False}]
            }))
            with self.assertRaisesRegex(ValueError, "no validated connection"):
                service._validate_tracking_requirements(enabled)
            self.assertEqual(service._active_tracking_providers(disabled), [])
            service._validate_tracking_requirements(disabled)


class ProviderFlushBindingTestCase(unittest.TestCase):
    def test_wandb_successful_replay_rebinds_run_and_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capsule_root = root / "capsules"
            run_id = "local-run"
            experiment_id = "local-experiment"
            project = "openai-test-v0"
            endpoint = "https://api.wandb.ai"
            capsule = capsule_root / run_id
            offline = WandBBridge(
                capsule,
                WandBSettings(
                    base_url=endpoint,
                    api_key=None,
                    entity="team",
                    auto_flush=False,
                ),
            )
            offline.ensure_experiment("team", project)
            offline.ensure_run(
                entity="team",
                project=project,
                local_run_id=run_id,
                run_name="replayed run",
                group="revision-1",
            )
            offline.set_tags(run_id, {"adapter": "openpi"})

            database = Database(root / "skynet.db")
            database.upsert_tracking_binding(
                "wandb", "experiment", experiment_id,
                remote_id=f"team/{project}",
                remote_url=None,
                status="QUEUED",
                metadata={
                    "entity": "team",
                    "project": project,
                    "endpoint": endpoint,
                },
                last_error="W&B returned HTTP 400: stale tag error",
            )
            database.upsert_tracking_binding(
                "wandb", "run", run_id,
                remote_id=None,
                remote_url=None,
                status="QUEUED",
                metadata={
                    "entity": "team",
                    "project": project,
                    "endpoint": endpoint,
                },
                last_error="W&B returned HTTP 400: stale tag error",
            )
            service = PipelineService(database, object())
            settings = WandBSettings(
                base_url=endpoint,
                api_key="fake-wandb-key",
                entity="team",
            )
            with patch(
                "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", capsule_root
            ), patch(
                "skynet_app.pipeline_api.WandBBridge", FakeWandBBridge
            ), patch.object(
                service, "_wandb_settings", return_value=settings
            ), patch.object(
                database, "get_run", return_value={"experiment_id": experiment_id}
            ):
                report = service._flush_tracking_provider("wandb")

            run_binding = database.list_tracking_bindings("run", run_id)[0]
            experiment_binding = database.list_tracking_bindings(
                "experiment", experiment_id
            )[0]
            self.assertEqual(report["attempted"], 3)
            self.assertEqual(report["delivered"], 3)
            self.assertEqual(report["errors"], [])
            self.assertEqual(run_binding["status"], "CONNECTED")
            self.assertIsNone(run_binding["last_error"])
            self.assertTrue(run_binding["remote_url"])
            self.assertEqual(experiment_binding["status"], "CONNECTED")
            self.assertIsNone(experiment_binding["last_error"])
            self.assertEqual(
                experiment_binding["remote_url"],
                f"{wandb_web_base(endpoint)}/team/{project}",
            )

    def test_mlflow_successful_replay_rebinds_run_and_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capsule_root = root / "capsules"
            run_id = "local-run"
            experiment_id = "local-experiment"
            experiment_name = "openai-test-v0"
            endpoint = "http://mlflow.test:5000"
            capsule = capsule_root / run_id
            offline = MLflowBridge(
                capsule,
                TrackingSettings(None, auto_flush=False),
            )
            offline.ensure_experiment(experiment_name)
            offline.ensure_run(
                experiment_name=experiment_name,
                local_run_id=run_id,
                run_name="replayed run",
            )
            offline.log_params(run_id, {"adapter": "openpi"})

            database = Database(root / "skynet.db")
            database.upsert_tracking_binding(
                "mlflow", "experiment", experiment_id,
                remote_id=None,
                remote_url=None,
                status="QUEUED",
                metadata={"experiment": experiment_name, "endpoint": endpoint},
                last_error="MLflow stale connection error",
            )
            database.upsert_tracking_binding(
                "mlflow", "run", run_id,
                remote_id=None,
                remote_url=None,
                status="QUEUED",
                metadata={"experiment": experiment_name, "endpoint": endpoint},
                last_error="MLflow stale connection error",
            )
            service = PipelineService(database, object())
            settings = TrackingSettings(endpoint)
            with patch(
                "skynet_app.pipeline_api.LOCAL_CAPSULE_ROOT", capsule_root
            ), patch(
                "skynet_app.pipeline_api.MLflowBridge", FakeMLflowBridge
            ), patch.object(
                service, "_mlflow_settings", return_value=settings
            ), patch.object(
                database, "get_run", return_value={"experiment_id": experiment_id}
            ), patch.object(
                database, "update_run"
            ):
                report = service._flush_tracking_provider("mlflow")

            run_binding = database.list_tracking_bindings("run", run_id)[0]
            experiment_binding = database.list_tracking_bindings(
                "experiment", experiment_id
            )[0]
            self.assertEqual(report["attempted"], 3)
            self.assertEqual(report["delivered"], 3)
            self.assertEqual(report["errors"], [])
            self.assertEqual(run_binding["status"], "CONNECTED")
            self.assertIsNone(run_binding["last_error"])
            self.assertEqual(
                run_binding["remote_url"], mlflow_run_url(endpoint, "1", "run-1")
            )
            self.assertEqual(experiment_binding["status"], "CONNECTED")
            self.assertIsNone(experiment_binding["last_error"])
            self.assertEqual(experiment_binding["remote_id"], "1")
            self.assertEqual(
                experiment_binding["remote_url"],
                mlflow_experiment_url(endpoint, "1"),
            )

    def test_evaluation_aggregate_uses_semantic_metric_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = PipelineService(Database(Path(directory) / "skynet.db"), object())
            metrics, links = service._final_tracking_payload({
                "artifacts": [{
                    "id": "artifact-1",
                    "artifact_type": "EVALUATION_RESULT",
                    "evaluation_id": "evaluation-1",
                    "path": "/eval/result.json",
                    "metadata_json": {"aggregate": [{
                        "metric": "success_rate",
                        "task": "pick object",
                        "mean": 0.75,
                        "std": 0.1,
                        "sample_count": 20,
                    }]},
                }],
                "evaluations": [{
                    "id": "evaluation-1", "suite_name": "LIBERO 10"
                }],
            })
            self.assertEqual(
                metrics["evaluation/LIBERO-10/pick-object/success_rate/mean"], 0.75
            )
            self.assertEqual(
                metrics["evaluation/LIBERO-10/pick-object/success_rate/sample_count"],
                20.0,
            )
            self.assertEqual(links[0]["artifact_type"], "evaluation_result")

    def test_mlflow_artifact_failure_does_not_block_finish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bridge = FakeMLflowBridge(Path(directory) / "run-capsule")
            bridge.ensure_experiment("experiment")
            bridge.ensure_run(
                experiment_name="experiment",
                local_run_id="local-run",
                run_name="run",
            )
            with patch.object(
                bridge,
                "_deliver_artifact_link",
                side_effect=TrackingRequestError("artifact tag rejected"),
            ):
                artifact = bridge.log_artifact_link(
                    "local-run", name="artifact", uri="/artifact"
                )
            finished = bridge.finish_run("local-run", status="FINISHED")
            self.assertTrue(artifact.delivered)
            self.assertIn("artifact tag rejected", artifact.error)
            self.assertTrue(finished.delivered)
            self.assertEqual(bridge.pending_count(), 0)
            self.assertEqual(bridge.runs[finished.remote_id]["status"], "FINISHED")

    def test_tracking_failure_preserves_pinned_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "skynet.db")
            service = PipelineService(database, object())
            database.upsert_tracking_binding(
                "mlflow",
                "run",
                "local-run",
                remote_id="remote-run",
                remote_url="https://mlflow.example/run",
                status="QUEUED",
                metadata={"endpoint": "https://mlflow.example"},
            )
            service._tracking_failure("mlflow", {"id": "local-run"}, "offline")
            binding = database.list_tracking_bindings("run", "local-run")[0]
            self.assertEqual(binding["metadata_json"]["endpoint"], "https://mlflow.example")
            self.assertEqual(binding["remote_id"], "remote-run")

    def test_frontend_connect_route_and_boolean_native_tracking_contract(self) -> None:
        paths = {
            (route.path, method)
            for route in router.routes
            for method in getattr(route, "methods", set())
        }
        self.assertIn(("/api/tracking/connections/{provider}/connect", "POST"), paths)
        self.assertEqual(
            TrackingSpec.model_validate({"native_tracking": False}).native_tracking,
            "disable",
        )


if __name__ == "__main__":
    unittest.main()
