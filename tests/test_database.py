from __future__ import annotations

import re
import tempfile
import psycopg
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from skynet_app.db_backend import INTEGRITY_ERRORS, PostgresConnection
from skynet_app.database import Database
from skynet_app.tracking import TRACKING_PROVIDERS

from factories import make_run_chain

MIGRATIONS = Path(__file__).parents[1] / "skynet_app/migrations/postgresql"


def recorded_statements():
    """Patch PostgresConnection.execute to record each statement with its parameters."""
    statements = []
    original = PostgresConnection.execute

    def recorded(connection, statement, parameters=None):
        statements.append((statement, parameters))
        return original(connection, statement, parameters)

    return statements, mock.patch.object(PostgresConnection, "execute", recorded)


class DatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temporary_directory.name) / "skynet.db")
        chain = make_run_chain(
            self.database,
            project_name="robotics",
            project_description="Robot learning",
            experiment_name="encoder-comparison",
            requested_spec={"apiVersion": "skynet.rl2/v1", "train": {"learning_rate": 1e-4}},
            variant_name="resnet50",
            parameters={"model.encoder": "resnet50"},
            resolved_spec={"model": {"encoder": "resnet50"}, "seed": 42},
            seed=42,
            adapter_name="groot",
            run_directory="/coc/flash7/ycho420/jobs/run-1",
        )
        self.experiment, self.variant, self.run = chain.experiment, chain.variant, chain.run

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_experiment_revision_and_run_queries(self) -> None:
        second_revision = self.database.create_experiment_revision(
            self.experiment["id"],
            {"apiVersion": "skynet.rl2/v1", "train": {"learning_rate": 3e-5}},
        )

        loaded = self.database.get_experiment(self.experiment["id"])

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["project_name"], "robotics")
        self.assertEqual(loaded["latest_revision"]["id"], second_revision["id"])
        self.assertEqual(loaded["latest_revision"]["requested_spec_json"]["train"]["learning_rate"], 3e-5)
        self.assertEqual(self.database.list_runs(experiment_id=self.experiment["id"])[0]["id"], self.run["id"])

    def test_multiple_runs_preserve_variant_and_record_restart_lineage(self) -> None:
        rerun = self.database.create_run(
            self.variant["id"],
            seed=42,
            adapter_name="groot",
            adapter_version="1",
            run_directory="/coc/flash7/ycho420/jobs/run-2",
            restarted_from_run_id=self.run["id"],
        )

        self.assertEqual(self.run["run_number"], 1)
        self.assertEqual(rerun["run_number"], 2)
        self.assertEqual(rerun["restarted_from_run_id"], self.run["id"])
        self.assertEqual(rerun["variant_id"], self.run["variant_id"])

    def test_run_and_revision_scientific_identity_is_immutable(self) -> None:
        with self.assertRaises(ValueError):
            self.database.update_run(self.run["id"], source_commit="different")
        with self.database.transaction() as connection:
            with self.assertRaises(Exception):
                connection.execute(
                    "UPDATE variants SET name = 'changed' WHERE id = ?",
                    (self.variant["id"],),
                )

    def test_attempt_checkpoint_event_and_run_detail(self) -> None:
        stage = self.database.create_stage(
            self.run["id"], stage_type="TRAIN", name="train", auto_resume=True, max_attempts=5
        )
        attempt = self.database.create_job_attempt(
            stage["id"],
            slurm_job_id="72103",
            gateway="sky1",
            account="overcap",
            partition_name="overcap",
            gpu_count=4,
            stderr_path="/coc/flash7/ycho420/logs/train-72103.err",
        )
        self.database.update_job_attempt(
            attempt["id"], status="PREEMPTED", slurm_state="PREEMPTED", exit_code="0:0"
        )
        checkpoint = self.database.create_checkpoint(
            self.run["id"],
            produced_by_attempt_id=attempt["id"],
            checkpoint_type="FULL_RESUME",
            path="/coc/flash7/ycho420/artifacts/run-1/checkpoints/step-1000",
            training_step=1000,
            is_resumable=True,
        )
        self.database.record_event(
            entity_type="run",
            entity_id=self.run["id"],
            event_type="JOB_PREEMPTED",
            old_status="RUNNING",
            new_status="INTERRUPTED",
            details={"checkpoint_id": checkpoint["id"]},
        )

        statements, recording = recorded_statements()
        with recording:
            detail = self.database.get_run(self.run["id"])

        self.assertEqual(detail["stages"][0]["name"], "train")
        self.assertTrue(detail["stages"][0]["auto_resume"])
        self.assertEqual(detail["attempts"][0]["status"], "PREEMPTED")
        self.assertTrue(detail["checkpoints"][0]["is_resumable"])
        self.assertEqual(detail["events"][0]["details_json"]["checkpoint_id"], checkpoint["id"])
        # The run's events are read through the (entity_type, entity_id, created_at) index.
        events_query, parameters = next(item for item in statements if "FROM events" in item[0])
        with self.database.read_snapshot() as connection:
            connection.execute("SET LOCAL enable_seqscan = off")
            plan = "\n".join(row[0] for row in connection.execute(f"EXPLAIN {events_query}", parameters).fetchall())
        self.assertIn("idx_events_entity", plan)
        self.assertIn("entity_type = 'run'", plan)

    def test_run_experiment_ids_maps_only_visible_runs_in_one_query(self) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO workspaces(id,email) VALUES ('alice','alice@example.com'),('bob','bob@example.com')"
            )

        def owned_run(database):
            chain = make_run_chain(
                database, project_name="owned", experiment_name="owned", variant_name="owned",
                adapter_name="groot", run_directory="/runs/owned",
            )
            return chain.run["id"], chain.experiment["id"]

        alice = self.database.for_workspace("alice")
        alice_run, alice_experiment = owned_run(alice)
        bob_run, bob_experiment = owned_run(self.database.for_workspace("bob"))
        statements, recording = recorded_statements()
        with recording:
            visible = alice.run_experiment_ids([alice_run, bob_run, "missing"])
        self.assertEqual(visible, {alice_run: alice_experiment})
        self.assertEqual(len(statements), 1)
        self.assertEqual(
            self.database.run_experiment_ids([self.run["id"], alice_run, bob_run]),
            {self.run["id"]: self.experiment["id"], alice_run: alice_experiment, bob_run: bob_experiment},
        )
        self.assertEqual(alice.run_experiment_ids([]), {})

    def test_stage_submission_claim_is_atomic(self) -> None:
        stage = self.database.create_stage(
            self.run["id"], stage_type="TRAIN", name="train", status="PENDING"
        )

        claimed = self.database.claim_stage_for_submission(stage["id"])
        duplicate = self.database.claim_stage_for_submission(stage["id"])

        self.assertEqual(claimed["status"], "SUBMITTING")
        self.assertIsNone(duplicate)

    def test_checkpoint_registration_is_atomic_across_database_instances(self) -> None:
        databases = [self.database, Database(self.database.path)]
        barrier = threading.Barrier(2)
        receipt = {
            "checkpoint_type": "INFERENCE", "path": "/checkpoints/latest.ckpt",
            "sha256": "a" * 64, "size_bytes": 123,
            "is_selected_for_inference": True, "metadata": {"final": True},
        }

        def register(database):
            barrier.wait(timeout=10)
            return database.create_checkpoint(self.run["id"], **receipt)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = list(pool.map(register, databases))
        self.assertEqual(first, second)
        self.assertEqual(self.database.list_checkpoints(self.run["id"]), [first])

    def test_repeated_checkpoint_preserves_references_and_rejects_changed_receipt(self) -> None:
        receipt = {
            "checkpoint_type": "INFERENCE", "path": "/checkpoints/latest.ckpt",
            "sha256": "a" * 64, "size_bytes": 123,
            "is_selected_for_inference": True, "metadata": {"final": True},
        }
        first = self.database.create_checkpoint(self.run["id"], **receipt)
        evaluation = self.database.create_evaluation(
            self.run["id"], checkpoint_id=first["id"],
            evaluator_adapter="libero", evaluator_version="1",
            suite_name="libero_10", suite_version="1",
            tasks=["pick_up_cube"], seeds=[42], episodes_per_task=1,
        )
        second = self.database.create_checkpoint(
            self.run["id"], **{**receipt, "path": "/checkpoints/best.ckpt"}
        )
        with self.assertRaisesRegex(INTEGRITY_ERRORS, "different receipt"):
            self.database.create_checkpoint(
                self.run["id"], **{**receipt, "sha256": "b" * 64}
            )
        self.assertEqual(
            [c["id"] for c in self.database.list_checkpoints(self.run["id"])
             if c["is_selected_for_inference"]], [second["id"]]
        )
        repeated = self.database.create_checkpoint(self.run["id"], **receipt)
        self.assertEqual(repeated, first)
        self.assertEqual(
            self.database.get_evaluation(evaluation["id"])["checkpoint_id"], first["id"]
        )
        self.assertEqual(
            [c["id"] for c in self.database.list_checkpoints(self.run["id"])
             if c["is_selected_for_inference"]], [first["id"]]
        )

    def test_initialize_evaluation_ledger_preserves_completed_and_partial_rows(self):
        evaluation = self.database.create_evaluation(
            self.run["id"], evaluator_adapter="libero", evaluator_version="1",
            suite_name="test", suite_version="1", tasks=["pick"], seeds=[41, 42], episodes_per_task=50,
        )
        done = self.database.upsert_evaluation_episode(
            evaluation["id"], task="pick", seed=41, episode_index=0,
            status="SUCCEEDED", attempt_count=2, success=True, reward=1.0,
        )
        self.assertEqual(self.database.initialize_evaluation_episodes(evaluation["id"]), 99)
        self.assertEqual(self.database.initialize_evaluation_episodes(evaluation["id"]), 0)
        loaded = self.database.get_evaluation(evaluation["id"])
        self.assertEqual(len(loaded["episodes"]), 100)
        self.assertEqual(loaded["progress_completed"], 1)
        preserved = next(row for row in loaded["episodes"] if row["id"] == done["id"])
        self.assertEqual(preserved["status"], "SUCCEEDED")
        self.assertTrue(preserved["success"])
        self.assertEqual(preserved["attempt_count"], 2)
        self.assertEqual(sum(row["status"] == "PENDING" for row in loaded["episodes"]), 99)

    def test_native_reconciliation_repairs_only_incomplete_active_ledgers(self):
        active = self.database.create_evaluation(
            self.run["id"], evaluator_adapter="libero", evaluator_version="1",
            suite_name="test", suite_version="1", tasks=["pick"], seeds=[41, 42],
            episodes_per_task=50, status="PENDING",
        )
        completed = self.database.upsert_evaluation_episode(
            active["id"], task="pick", seed=41, episode_index=0,
            status="SUCCEEDED", attempt_count=2, success=True, reward=1.0,
        )
        cancelled = self.database.create_evaluation(
            self.run["id"], evaluator_adapter="libero", evaluator_version="1",
            suite_name="test", suite_version="1", tasks=["pick"], seeds=[42],
            episodes_per_task=20, status="CANCELLED",
        )
        self.assertEqual(self.database.repair_workflow_state_invariants()["evaluation_ledgers_repaired"], 1)
        self.assertEqual(self.database.repair_workflow_state_invariants()["evaluation_ledgers_repaired"], 0)
        loaded = self.database.get_evaluation(active["id"])
        self.assertEqual(len(loaded["episodes"]), 100)
        self.assertEqual(loaded["status"], "PENDING")
        existing = next(x for x in loaded["episodes"] if x["id"] == completed["id"])
        self.assertEqual(existing["attempt_count"], 2)
        self.assertTrue(existing["success"])
        self.assertEqual(self.database.get_evaluation(cancelled["id"])["episodes"], [])
        self.assertEqual(self.database.initialize_evaluation_episodes(
            cancelled["id"], expected_parent_states=("PENDING",)), 0)
        self.assertEqual(self.database.get_evaluation(cancelled["id"])["episodes"], [])

    def test_initialize_evaluation_ledger_rolls_back_all_rows_on_error(self):
        evaluation = self.database.create_evaluation(
            self.run["id"], evaluator_adapter="libero", evaluator_version="1",
            suite_name="test", suite_version="1", tasks=["pick", None], seeds=[41], episodes_per_task=2,
        )
        with self.assertRaises(INTEGRITY_ERRORS):
            self.database.initialize_evaluation_episodes(evaluation["id"])
        self.assertEqual(self.database.get_evaluation(evaluation["id"])["episodes"], [])

    def test_evaluation_suite_and_episode_resume_ledger(self) -> None:
        suite = self.database.register_evaluation_suite(
            evaluator_adapter="libero",
            evaluator_version="1",
            name="libero_10",
            suite_version="0.1.0",
            config={"tasks": ["pick_up_the_black_bowl"]},
        )
        evaluation = self.database.create_evaluation(
            self.run["id"],
            evaluator_adapter="libero",
            evaluator_version="1",
            suite_name="libero_10",
            suite_version="0.1.0",
            tasks=["pick_up_the_black_bowl"],
            seeds=[41, 42],
            episodes_per_task=2,
            evaluation_suite_id=suite["id"],
        )
        first = self.database.upsert_evaluation_episode(
            evaluation["id"],
            task="pick_up_the_black_bowl",
            seed=41,
            episode_index=0,
            status="RUNNING",
            attempt_count=1,
        )
        completed = self.database.upsert_evaluation_episode(
            evaluation["id"],
            task="pick_up_the_black_bowl",
            seed=41,
            episode_index=0,
            status="SUCCEEDED",
            attempt_count=2,
            success=True,
            reward=1.0,
            metrics={"success_rate": 1.0},
        )

        loaded = self.database.get_evaluation(evaluation["id"])

        self.assertEqual(first["id"], completed["id"])
        self.assertEqual(len(loaded["episodes"]), 1)
        self.assertTrue(loaded["episodes"][0]["success"])
        self.assertEqual(loaded["progress_completed"], 1)
        self.assertEqual(loaded["progress_total"], 4)
        self.assertEqual(self.database.list_evaluation_suites()[0]["config_json"]["tasks"][0], "pick_up_the_black_bowl")


def test_hot_run_and_status_filters_are_indexed(tmp_path):
    db = Database(tmp_path / "index.db")
    expected = {
        # Run list progress evidence, run detail and tracking sync read samples by run.
        "idx_training_progress_samples_run": ("training_progress_samples", "(run_id)"),
        # Workflow repair and the experiment dispatcher select stages by status.
        "idx_stages_status_run": ("workflow_stages", "(status, run_id)"),
        # The Slurm poll and submission recovery select attempts by status.
        "idx_attempts_status_stage": ("job_attempts", "(status, stage_id)"),
    }
    with db.connection() as c:
        definitions = {row[0]: (row[1], row[2]) for row in c.execute(
            "SELECT indexname, tablename, indexdef FROM pg_indexes WHERE indexname = ANY(?)", (list(expected),)
        ).fetchall()}
    for name, (table, columns) in expected.items():
        assert definitions[name][0] == table and definitions[name][1].endswith(columns), name


def test_tracking_provider_checks_list_the_tracking_providers():
    schema = next(MIGRATIONS.glob("001_*.sql")).read_text()
    checks = re.findall(r"CHECK\(provider IN \(([^)]*)\)\)", schema)
    assert len(checks) == 2, "tracking_bindings and tracking_connections"
    for check in checks:
        assert sorted(re.findall(r"'([^']*)'", check)) == sorted(TRACKING_PROVIDERS)


if __name__ == "__main__":
    unittest.main()
