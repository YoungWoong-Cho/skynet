"""Migrate indexed app-host files and sanitized tracking state after DB import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .database import canonical_json
from .metadata_objects import MetadataObjects
from .tracking_journal import JOURNAL_FILES, TrackingJournal


def migrate_supporting_files(database, data_root: Path):
    """Caller must stop old app writers and retain the offline source snapshot.

    Uploads are immutable and verified before any DB path changes. Only location
    columns change; immutable execution snapshots are retained byte for byte.
    Credentials and unindexed caches are deliberately outside this migration.
    """
    objects = MetadataObjects(database)
    relocations = []
    with database.connection() as connection:
        for table, checksum in (
            ("artifacts", "sha256"),
            ("data_resource_versions", "manifest_sha256"),
        ):
            rows = connection.execute(f"SELECT * FROM {table}").fetchall()
            for row in rows:
                old = str(row["path"] or "")
                if not old.startswith(("/Users/", "/home/", "/private/var/folders/")):
                    continue
                path = Path(old)
                if not path.is_file():
                    raise FileNotFoundError(
                        "Indexed supporting file is missing: " + old
                    )
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                expected = row[checksum]
                # Manifests use a canonical source/split hash, equal to file SHA.
                if expected and expected != digest:
                    raise ValueError("Indexed file checksum mismatch: " + old)
                new = objects.put(content, name=path.name)
                if objects.read(new, digest) != content:
                    raise ValueError("Uploaded file verification failed")
                relocations.append(
                    {
                        "table": table,
                        "id": row["id"],
                        "old": old,
                        "new": new,
                        "sha256": digest,
                    }
                )
    with database.transaction() as connection:
        # Immutability guards normally reject path edits. This narrowly scoped,
        # offline migration disables only user triggers inside its transaction.
        connection.execute("ALTER TABLE data_resource_versions DISABLE TRIGGER USER")
        for entry in relocations:
            connection.execute(
                f"UPDATE {entry['table']} SET path=? WHERE id=? AND path=?",
                (entry["new"], entry["id"], entry["old"]),
            )
            if entry["table"] == "artifacts":
                row = connection.execute(
                    "SELECT metadata_json FROM artifacts WHERE id=?", (entry["id"],)
                ).fetchone()
                metadata = json.loads(row[0])
                metadata["location"] = "cluster_objects"
                connection.execute(
                    "UPDATE artifacts SET metadata_json=? WHERE id=?",
                    (canonical_json(metadata), entry["id"]),
                )
        connection.execute("ALTER TABLE data_resource_versions ENABLE TRIGGER USER")
    journals = []
    with database.connection() as connection:
        runs = connection.execute("SELECT id,owner_id FROM runs").fetchall()
    for run in runs:
        journal = TrackingJournal(database.for_workspace(run["owner_id"]), run["id"])
        for filename in sorted(JOURNAL_FILES):
            source = data_root / "capsules" / run["id"] / filename
            if not source.is_file():
                continue
            payload = source.read_bytes()
            destination = journal.file(filename)
            if destination.exists() and destination.read_bytes() != payload:
                raise ValueError("Destination tracking journal already differs")
            destination.write_bytes(payload)
            if destination.read_bytes() != payload:
                raise ValueError("Tracking journal verification failed")
            journals.append(
                {
                    "run_id": run["id"],
                    "filename": filename,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )
    return {"relocations": relocations, "journals": journals}
