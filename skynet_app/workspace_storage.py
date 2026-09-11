"""Personal storage preferences and the immutable locations of existing runs."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from .cluster_config import CLUSTER, ClusterPaths
from .database import Database
from .workspace_schema import visible_sql


def validate_work_root(value: str) -> str:
    value = value.strip().rstrip("/")
    if (
        not re.fullmatch(r"/[A-Za-z0-9._/-]{1,510}", value)
        or any(part in {".", ".."} for part in value.split("/"))
    ):
        raise ValueError(
            "Use an absolute cluster directory, such as /coc/flash7/yourname. "
            "Use letters, numbers, slashes, dots, underscores or hyphens; no spaces or '..'."
        )
    return str(PurePosixPath(value))


def paths_for_root(work_root: str) -> ClusterPaths:
    """Rebase writable job paths; shared data and installed runtimes stay registered."""
    root = validate_work_root(work_root)
    original = CLUSTER.paths
    paths = original.model_dump()
    for key in (
        "work_root", "workspace", "repositories", "artifacts", "logs", "jobs",
        "evaluation", "uv_cache", "huggingface_cache", "torch_cache",
    ):
        path = PurePosixPath(paths[key])
        try:
            suffix = path.relative_to(original.work_root)
        except ValueError:
            continue
        paths[key] = str(PurePosixPath(root) / suffix)
    return ClusterPaths.model_validate(paths)


class WorkspaceStorage:
    def __init__(self, database: Database):
        self.database = database

    @property
    def work_root(self) -> str:
        if self.database.workspace_id is None:
            return CLUSTER.paths.work_root
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT work_root FROM workspace_storage WHERE owner_id=?",
                (self.database.workspace_id,),
            ).fetchone()
        return row[0] if row else CLUSTER.paths.work_root

    @property
    def paths(self) -> ClusterPaths:
        return paths_for_root(self.work_root)

    def settings(self) -> dict[str, str]:
        return {
            "work_root": self.work_root,
            "default_work_root": CLUSTER.paths.work_root,
            "shared_datasets": CLUSTER.paths.datasets,
            "shared_environments": CLUSTER.paths.environments,
        }

    def save(self, work_root: str, expected_work_root: str) -> dict[str, str]:
        if self.database.workspace_id is None:
            raise ValueError("Open an email workspace before saving a base path")
        root = validate_work_root(work_root)
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT work_root FROM workspace_storage WHERE owner_id=?",
                (self.database.workspace_id,),
            ).fetchone()
            current = row[0] if row else CLUSTER.paths.work_root
            if current != expected_work_root:
                raise ValueError("The base path changed in another tab. Refresh Settings and try again.")
            connection.execute(
                """INSERT INTO workspace_storage(owner_id,work_root) VALUES (?,?)
                ON CONFLICT(owner_id) DO UPDATE SET work_root=excluded.work_root""",
                (self.database.workspace_id, root),
            )
        return self.settings()

    def run_directory(self, run_id: str) -> str:
        with self.database.connection() as connection:
            row = connection.execute(
                f"SELECT run_directory FROM runs WHERE id=? AND {visible_sql('runs')}",
                (run_id,),
            ).fetchone()
        # Non-personal utility jobs keep using the shared deployment root.
        return row[0] if row and row[0] else f"{CLUSTER.paths.jobs}/runs/{run_id}"

    def root_for_run(self, run_id: str) -> str:
        directory = self.run_directory(run_id)
        suffix = f"/jobs/runs/{run_id}"
        if not directory.endswith(suffix):
            raise ValueError("The saved run directory does not match its run ID")
        return validate_work_root(directory.removesuffix(suffix))

    def allowed_roots(self) -> set[str]:
        roots = {self.work_root, CLUSTER.paths.work_root}
        with self.database.connection() as connection:
            rows = connection.execute(
                f"SELECT id,run_directory FROM runs WHERE {visible_sql('runs')}"
            ).fetchall()
        for row in rows:
            suffix = f"/jobs/runs/{row['id']}"
            if row["run_directory"] and row["run_directory"].endswith(suffix):
                roots.add(validate_work_root(row["run_directory"].removesuffix(suffix)))
        return roots
