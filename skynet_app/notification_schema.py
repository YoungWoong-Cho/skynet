"""Durable, workspace-owned job notifications, captured with state changes."""

from __future__ import annotations

import sqlite3

EVENT_STATUSES = {
    "RUNNING": "running",
    "CANCELLED": "cancelled",
    "SUCCEEDED": "completed",
    "FAILED": "failed",
    "SUBMISSION_FAILED": "failed",
    "BLOCKED": "failed",
    "RETRY_PENDING": "failed",
}


def migrate_notifications(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS slack_notifications (
            owner_id TEXT PRIMARY KEY REFERENCES workspaces(id),
            enabled INTEGER NOT NULL DEFAULT 0,
            events_json TEXT NOT NULL DEFAULT '["submitted","running","cancelled","failed","completed"]',
            app_url TEXT NOT NULL DEFAULT '',
            last_error TEXT, last_sent_at TEXT, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS notification_outbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id TEXT NOT NULL REFERENCES workspaces(id),
            stage_id TEXT NOT NULL REFERENCES workflow_stages(id) ON DELETE CASCADE,
            attempt_key TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL, job_status TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
            due_at REAL NOT NULL DEFAULT 0, lease_token TEXT,
            last_error TEXT, delivered_at TEXT,
            UNIQUE(owner_id, stage_id, attempt_key, category)
        );
        CREATE INDEX IF NOT EXISTS notification_outbox_delivery
        ON notification_outbox(owner_id, status, id);
    """)
    # These tables have explicit owner keys; upgrades do not alter the existing
    # workspace migration or its historical records.
    for table in ("slack_notifications", "notification_outbox"):
        for operation in ("INSERT", "UPDATE", "DELETE"):
            row = "NEW" if operation == "INSERT" else "OLD"
            connection.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_owner_{operation.lower()}
                BEFORE {operation} ON {table}
                WHEN current_workspace_id() IS NOT NULL
                    AND {row}.owner_id IS NOT current_workspace_id()
                BEGIN SELECT RAISE(ABORT, 'Record is outside this workspace'); END""")
        connection.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_owner_immutable
            BEFORE UPDATE OF owner_id ON {table} WHEN NEW.owner_id IS NOT OLD.owner_id
            BEGIN SELECT RAISE(ABORT, 'Notification ownership is immutable'); END""")
    cases = (
        "CASE NEW.status "
        + " ".join(
            f"WHEN '{status}' THEN '{category}'"
            for status, category in EVENT_STATUSES.items()
        )
        + " END"
    )
    states = ",".join(f"'{status}'" for status in EVENT_STATUSES)
    # Queue the workflow outcome, not just Slurm's exit state: a process that
    # exits successfully can still fail checkpoint or evaluation validation.
    connection.execute(f"""CREATE TRIGGER IF NOT EXISTS notify_workflow_status
        AFTER UPDATE OF status ON workflow_stages
        WHEN NEW.status IS NOT OLD.status AND NEW.status IN ({states})
        BEGIN
            INSERT OR IGNORE INTO notification_outbox(owner_id,stage_id,attempt_key,category,job_status)
            SELECT r.owner_id, NEW.id,
                COALESCE((SELECT id FROM job_attempts WHERE stage_id=NEW.id ORDER BY attempt_number DESC LIMIT 1),''),
                {cases}, NEW.status
            FROM runs r JOIN slack_notifications n ON n.owner_id=r.owner_id
            WHERE r.id=NEW.run_id AND n.enabled=1
                AND EXISTS (SELECT 1 FROM json_each(n.events_json) WHERE value={cases});
        END""")
    # A Slurm ID means submission was acknowledged, including recovery of an
    # ambiguous submission or a cancellation that raced with sbatch.
    for operation in ("INSERT", "UPDATE OF slurm_job_id"):
        suffix = "insert" if operation == "INSERT" else "update"
        changed = (
            ""
            if operation == "INSERT"
            else "AND NEW.slurm_job_id IS NOT OLD.slurm_job_id"
        )
        connection.execute(f"""CREATE TRIGGER IF NOT EXISTS notify_job_submitted_{suffix}
            AFTER {operation} ON job_attempts
            WHEN NEW.slurm_job_id IS NOT NULL AND NEW.slurm_job_id != '' {changed}
            BEGIN
                INSERT OR IGNORE INTO notification_outbox(owner_id,stage_id,attempt_key,category,job_status)
                SELECT r.owner_id, NEW.stage_id, NEW.id, 'submitted', 'SUBMITTED'
                FROM workflow_stages s JOIN runs r ON r.id=s.run_id
                JOIN slack_notifications n ON n.owner_id=r.owner_id
                WHERE s.id=NEW.stage_id AND n.enabled=1
                    AND EXISTS (SELECT 1 FROM json_each(n.events_json) WHERE value='submitted');
            END""")

    # Short jobs may start and finish between Slurm polls. Accounting confirms
    # execution even if the browser never observed a RUNNING state.
    connection.execute("""CREATE TRIGGER IF NOT EXISTS notify_accounted_start
        AFTER UPDATE OF started_at ON job_attempts
        WHEN NEW.started_at IS NOT NULL AND OLD.started_at IS NULL
            AND NEW.slurm_state IN ('RUNNING','COMPLETED','FAILED','OUT_OF_MEMORY','TIMEOUT')
        BEGIN
            INSERT OR IGNORE INTO notification_outbox(owner_id,stage_id,attempt_key,category,job_status)
            SELECT r.owner_id,NEW.stage_id,NEW.id,'running','RUNNING'
            FROM workflow_stages s JOIN runs r ON r.id=s.run_id
            JOIN slack_notifications n ON n.owner_id=r.owner_id
            WHERE s.id=NEW.stage_id AND n.enabled=1
                AND EXISTS (SELECT 1 FROM json_each(n.events_json) WHERE value='running');
        END""")
