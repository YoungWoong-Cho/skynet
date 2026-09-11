from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping

from .database import Database, canonical_json, utc_now


FULL_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
MAX_CACHE_ENTRIES_PER_KIND = 512


class SourceMetadataStore:
    """Persistent repository metadata and source-choice storage."""

    def __init__(self, database: Database) -> None:
        self.database = database
        if database.is_postgres:
            return  # Versioned PostgreSQL migrations own the schema.
        with self.database.transaction() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS source_metadata_cache (
                    cache_key TEXT PRIMARY KEY,
                    cache_kind TEXT NOT NULL,
                    repository_url TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS source_metadata_cache_lookup
                ON source_metadata_cache(cache_kind, repository_url, updated_at)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS source_repository_selections (
                    owner_id TEXT NOT NULL DEFAULT 'legacy',
                    repository_url TEXT NOT NULL,
                    branch_name TEXT NOT NULL,
                    commits_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_id, repository_url)
                )
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(source_repository_selections)")}
            if "owner_id" not in columns:
                connection.execute("ALTER TABLE source_repository_selections RENAME TO source_selections_legacy")
                connection.execute("CREATE TABLE source_repository_selections(owner_id TEXT NOT NULL, repository_url TEXT NOT NULL, branch_name TEXT NOT NULL, commits_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(owner_id, repository_url))")
                connection.execute("INSERT INTO source_repository_selections SELECT 'legacy', * FROM source_selections_legacy")
                connection.execute("DROP TABLE source_selections_legacy")

    @staticmethod
    def _key(kind: str, repository_url: str, parameters: Mapping[str, Any]) -> str:
        identity = canonical_json(
            {
                "kind": kind,
                "repository_url": repository_url,
                "parameters": dict(parameters),
            }
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @staticmethod
    def _decode_object(raw: str, label: str) -> dict[str, Any]:
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"stored {label} is not valid JSON") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"stored {label} must be a JSON object")
        return value

    def get(
        self,
        kind: str,
        repository_url: str,
        parameters: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        cache_key = self._key(kind, repository_url, parameters)
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT cache_key, payload_json, created_at, updated_at
                FROM source_metadata_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "cache_key": row["cache_key"],
            "payload": self._decode_object(row["payload_json"], "source metadata payload"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def put(
        self,
        kind: str,
        repository_url: str,
        parameters: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        cache_key = self._key(kind, repository_url, parameters)
        now = utc_now()
        payload_value = copy.deepcopy(dict(payload))
        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO source_metadata_cache (
                    cache_key, cache_kind, repository_url, parameters_json,
                    payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    cache_key,
                    kind,
                    repository_url,
                    canonical_json(dict(parameters)),
                    canonical_json(payload_value),
                    now,
                    now,
                ),
            )
            connection.execute(
                f"""
                DELETE FROM source_metadata_cache
                WHERE cache_key IN (
                    SELECT cache_key
                    FROM source_metadata_cache
                    WHERE cache_kind = ?
                    ORDER BY updated_at DESC, cache_key DESC
                    {"OFFSET ?" if self.database.is_postgres else "LIMIT -1 OFFSET ?"}
                )
                """,
                (kind, MAX_CACHE_ENTRIES_PER_KIND),
            )
        return {
            "cache_key": cache_key,
            "payload": payload_value,
            "created_at": now,
            "updated_at": now,
        }

    @staticmethod
    def response(entry: Mapping[str, Any], *, cache_hit: bool) -> dict[str, Any]:
        payload = copy.deepcopy(dict(entry["payload"]))
        payload["_cache"] = {
            "backend": "database",
            "hit": cache_hit,
            "key": entry["cache_key"],
            "updated_at": entry["updated_at"],
        }
        return payload

    def get_selection(self, repository_url: str) -> dict[str, Any]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT branch_name, commits_json, updated_at
                FROM source_repository_selections
                WHERE owner_id = ? AND repository_url = ?
                """,
                (self.database.workspace_id or "legacy", repository_url),
            ).fetchone()
        if row is None:
            return {"branch": "", "commits": {}, "updated_at": None}
        commits = self._decode_object(row["commits_json"], "source commit selection")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in commits.items()):
            raise RuntimeError("stored source commit selection must map branch names to commit strings")
        return {
            "branch": row["branch_name"],
            "commits": commits,
            "updated_at": row["updated_at"],
        }

    def save_selection(self, repository_url: str, branch: str, commit: str = "") -> dict[str, Any]:
        if commit and not FULL_COMMIT_RE.fullmatch(commit):
            raise ValueError("source selection commit must be a full 40-character SHA")
        now = utc_now()
        with self.database.transaction() as connection:
            row = connection.execute(
                """
                SELECT commits_json, created_at
                FROM source_repository_selections
                WHERE owner_id = ? AND repository_url = ?
                """,
                (self.database.workspace_id or "legacy", repository_url),
            ).fetchone()
            commits = (
                self._decode_object(row["commits_json"], "source commit selection")
                if row is not None
                else {}
            )
            if commit:
                commits[branch] = commit.lower()
            connection.execute(
                """
                INSERT INTO source_repository_selections (
                    owner_id, repository_url, branch_name, commits_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, repository_url) DO UPDATE SET
                    branch_name = excluded.branch_name,
                    commits_json = excluded.commits_json,
                    updated_at = excluded.updated_at
                """,
                (
                    self.database.workspace_id or "legacy",
                    repository_url,
                    branch,
                    canonical_json(commits),
                    row["created_at"] if row is not None else now,
                    now,
                ),
            )
        return {"branch": branch, "commits": commits, "updated_at": now}


__all__ = ["SourceMetadataStore"]
