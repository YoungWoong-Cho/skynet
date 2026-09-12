"""Ownership and sessions for trusted-team email workspaces."""

from __future__ import annotations

import re


LEGACY_WORKSPACE = "legacy"
PRIVATE_TABLES = frozenset({
    "projects", "experiments", "experiment_revisions", "variants", "runs",
    "workflow_stages", "job_attempts", "checkpoints", "evaluations",
    "evaluation_episodes", "metrics", "training_progress_samples", "artifacts",
    "manifests", "events", "adapters", "adapter_validations", "runtime_profiles",
    "tracking_bindings",
})
SHARED_DEFAULTS = frozenset({"adapters", "runtime_profiles"})


def visible_sql(table: str, alias: str = "") -> str:
    if table not in PRIVATE_TABLES:
        return "1 = 1"
    column = f"{alias}.owner_id" if alias else "owner_id"
    shared = f" OR {column} IS NULL" if table in SHARED_DEFAULTS else ""
    if table == "adapters":
        seed = f"{alias}.seed_key" if alias else "seed_key"
        shared += f" OR ({column} = 'legacy' AND {seed} IS NOT NULL)"
    return f"(current_workspace_id() IS NULL OR {column} = current_workspace_id(){shared})"


def normalize_email(value: str) -> str:
    email = value.strip().casefold()
    if len(email) > 254 or not re.fullmatch(r"[^\s@<>\x00-\x1f]+@[^\s@<>.]+(?:\.[^\s@<>.]+)+", email):
        raise ValueError("Enter a valid email address")
    return email
