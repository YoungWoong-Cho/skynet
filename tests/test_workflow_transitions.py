from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from skynet_app.database import Database


class WorkflowTransitionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temporary_directory.name) / "skynet.db")
        project = self.database.create_project("workflow-transitions")
        experiment = self.database.create_experiment(
            project_id=project["id"],
            name="atomic-state",
            requested_spec={"schema_version": "test"},
        )
        variant = self.database.create_variant(
            experiment["latest_revision"]["id"],
            name="variant",
            parameters={},
            resolved_spec={},
        )
        self.experiment_id = experiment["id"]
        self.run = self.database.create_run(
            variant["id"],
            seed=42,
            adapter_name="test",
            adapter_version="1",
            run_directory="/tmp/atomic-state",
            status="PENDING",
        )
        self.stage = self.database.create_stage(
            self.run["id"],
            stage_type="TRAIN",
            name="train",
            status="PENDING",
            auto_resume=True,
            max_attempts=3,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_claim_and_attempt_creation_are_atomic(self) -> None:
        attempt = self.database.claim_stage_and_create_job_attempt(
            self.stage["id"], status="SUBMITTING", gateway="auto"
        )

        self.assertIsNotNone(attempt)
        self.assertEqual(self.database.list_stages(self.run["id"])[0]["status"], "SUBMITTING")
        self.assertEqual(
            self.database.list_job_attempts(stage_id=self.stage["id"])[0]["status"],
            "SUBMITTING",
        )
        self.assertIsNone(
            self.database.claim_stage_and_create_job_attempt(self.stage["id"], gateway="auto")
        )

    def test_transition_rolls_back_every_record_when_one_update_fails(self) -> None:
        attempt = self.database.claim_stage_and_create_job_attempt(self.stage["id"])
        self.assertIsNotNone(attempt)

        with self.assertRaises(KeyError):
            self.database.transition_workflow_state(
                attempt_id=attempt["id"],
                attempt_updates={"status": "SUBMISSION_FAILED"},
                stage_id=self.stage["id"],
                stage_updates={"status": "FAILED"},
                run_id="missing-run",
                run_updates={"status": "FAILED"},
            )

        self.assertEqual(
            self.database.list_job_attempts(stage_id=self.stage["id"])[0]["status"],
            "SUBMITTING",
        )
        self.assertEqual(self.database.list_stages(self.run["id"])[0]["status"], "SUBMITTING")

    def test_submission_failure_transition_updates_all_records_and_event(self) -> None:
        attempt = self.database.claim_stage_and_create_job_attempt(self.stage["id"])
        self.assertIsNotNone(attempt)

        self.database.transition_workflow_state(
            attempt_id=attempt["id"],
            attempt_updates={"status": "SUBMISSION_FAILED", "slurm_reason": "gateway timeout"},
            stage_id=self.stage["id"],
            stage_updates={"status": "FAILED"},
            run_id=self.run["id"],
            run_updates={"status": "FAILED"},
            event={
                "entity_type": "run",
                "entity_id": self.run["id"],
                "event_type": "SUBMISSION_FAILED",
                "new_status": "FAILED",
                "details": {"error": "gateway timeout"},
            },
        )

        detail = self.database.get_run(self.run["id"])
        self.assertEqual(detail["status"], "FAILED")
        self.assertEqual(detail["stages"][0]["status"], "FAILED")
        self.assertEqual(detail["attempts"][0]["status"], "SUBMISSION_FAILED")
        self.assertEqual(detail["events"][0]["event_type"], "SUBMISSION_FAILED")

    def test_repair_fixes_interrupted_submission_failure_transition(self) -> None:
        attempt = self.database.claim_stage_and_create_job_attempt(self.stage["id"])
        self.assertIsNotNone(attempt)
        self.database.update_job_attempt(
            attempt["id"], status="SUBMISSION_FAILED", slurm_reason="process interrupted"
        )

        result = self.database.repair_workflow_state_invariants()
        detail = self.database.get_run(self.run["id"])

        self.assertEqual(result["repaired"], 1)
        self.assertEqual(result["experiment_ids"], [self.experiment_id])
        self.assertEqual(detail["status"], "FAILED")
        self.assertEqual(detail["stages"][0]["status"], "FAILED")
        self.assertEqual(detail["events"][0]["event_type"], "WORKFLOW_STATE_REPAIRED")


if __name__ == "__main__":
    unittest.main()
