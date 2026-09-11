"""Persistent, retryable delete intents. Completed intents are removed."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS maintenance_operations (
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(target_kind,target_id)
);
"""
