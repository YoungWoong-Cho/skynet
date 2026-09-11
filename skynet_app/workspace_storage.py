"""Personal storage preferences and the immutable locations of existing runs."""

from __future__ import annotations

import re
import threading
from pathlib import PurePosixPath
from typing import Any

from .cluster_config import CLUSTER, ClusterPaths
from .database import Database
from .workspace_schema import visible_sql


STORAGE_REQUIRED = "Set up Cluster storage in Settings before using this workspace."
_SETUP_LOCK = threading.RLock()
PERSONAL_PATHS = (
    "work_root", "workspace", "repositories", "artifacts", "logs", "jobs",
    "evaluation", "uv_cache", "huggingface_cache", "torch_cache",
)


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
    return str(PurePosixPath("/" + value.lstrip("/")))


def paths_for_root(work_root: str) -> ClusterPaths:
    """Rebase writable job paths; shared data and installed runtimes stay registered."""
    root = validate_work_root(work_root)
    original = CLUSTER.paths
    paths = original.model_dump()
    for key in PERSONAL_PATHS:
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
    def work_root(self) -> str | None:
        if self.database.workspace_id is None:
            return CLUSTER.paths.work_root
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT work_root FROM workspace_storage WHERE owner_id=?",
                (self.database.workspace_id,),
            ).fetchone()
        return row[0] if row else None

    @property
    def paths(self) -> ClusterPaths:
        return paths_for_root(self.require_root())

    def require_root(self) -> str:
        root = self.work_root
        if root is None:
            raise ValueError(STORAGE_REQUIRED)
        return root

    def public_paths(self) -> dict[str, Any]:
        if self.work_root is not None:
            return self.paths.model_dump()
        paths = CLUSTER.paths.model_dump()
        for key in PERSONAL_PATHS:
            paths[key] = None
        return paths

    def settings(self) -> dict[str, Any]:
        return {
            "work_root": self.work_root,
            "configured": self.work_root is not None,
            "shared_datasets": CLUSTER.paths.datasets,
            "shared_environments": CLUSTER.paths.environments,
        }

    def _check_update(self, connection, root: str, expected: str | None) -> None:
        row = connection.execute(
            "SELECT work_root FROM workspace_storage WHERE owner_id=?",
            (self.database.workspace_id,),
        ).fetchone()
        current = row[0] if row else None
        if current != expected:
            raise ValueError("The base path changed in another tab. Refresh Settings and try again.")
        for other in connection.execute(
            "SELECT work_root FROM workspace_storage WHERE owner_id<>?",
            (self.database.workspace_id,),
        ):
            path = other[0]
            if root == path or root.startswith(path + "/") or path.startswith(root + "/"):
                raise ValueError("Choose a directory separate from another workspace's base path.")

    def configure(self, work_root: str, expected_work_root: str | None, cluster,
                  gateway: str = "auto") -> dict[str, Any]:
        if self.database.workspace_id is None:
            raise ValueError("Open an email workspace before setting up a base path")
        root = validate_work_root(work_root)
        # Serialize setup without keeping a SQLite transaction open during SSH.
        # A cluster-side ownership marker also protects retries and other servers.
        with _SETUP_LOCK:
            with self.database.connection() as connection:
                self._check_update(connection, root, expected_work_root)
            active_gateway = cluster.initialize_personal_workspace(
                root, self.database.workspace_id, gateway,
            )
            with self.database.transaction() as connection:
                self._check_update(connection, root, expected_work_root)
                connection.execute(
                    """INSERT INTO workspace_storage(owner_id,work_root) VALUES (?,?)
                    ON CONFLICT(owner_id) DO UPDATE SET work_root=excluded.work_root""",
                    (self.database.workspace_id, root),
                )
        return {**self.settings(), "gateway": active_gateway}

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
        roots = {CLUSTER.paths.work_root}
        if self.work_root is not None:
            roots.add(self.work_root)
        with self.database.connection() as connection:
            rows = connection.execute(
                f"SELECT id,run_directory FROM runs WHERE {visible_sql('runs')}"
            ).fetchall()
        for row in rows:
            suffix = f"/jobs/runs/{row['id']}"
            if row["run_directory"] and row["run_directory"].endswith(suffix):
                roots.add(validate_work_root(row["run_directory"].removesuffix(suffix)))
        return roots
