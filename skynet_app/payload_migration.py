"""Verified, atomic relocation of existing large metadata to cluster objects.

Run during maintenance with application writers stopped. Back up the central
DB first. This never alters the source contents or immutable receipt hashes.
"""

from __future__ import annotations

import hashlib
import json

from .payload_store import FIELDS, reference
from .tracking_journal import chunk_document


def relocate(database):
    store = database.payload_store
    if store is None:
        raise ValueError("Configure PostgreSQL and its cluster object store first")
    report = {"documents": 0, "bytes_before": 0, "bytes_after": 0, "tables": {}}
    with database.operation_lock("pipeline"), database.transaction() as connection:
        # Raw SQL intentionally sees references, never the transparent hydration.
        raw = connection.raw
        documents = []
        for table, field in FIELDS.items():
            for row in raw.execute(
                f"SELECT id,{field} FROM {table} WHERE {field} IS NOT NULL"
            ).fetchall():
                if reference(row[1]) is None:
                    documents.append((table, row[0], field, row[1], None))
        for row in raw.execute(
            "SELECT owner_id,scope,filename,run_id,payload FROM tracking_journals"
        ).fetchall():
            body = bytes(row[4]).decode("utf-8")
            if reference(body) is None and chunk_document(bytes(row[4])) is None:
                identifier = row[1] if row[3] else row[0] + ":" + row[1]
                documents.append(
                    (
                        "tracking_journals",
                        identifier,
                        row[2],
                        body,
                        (row[0], row[1], row[2]),
                    )
                )
        content = {
            hashlib.sha256(body.encode()).hexdigest(): body.encode()
            for _, _, _, body, _ in documents
        }
        paths = store.objects.put_many(list(content.values()))
        # Read back from cluster storage independently of the in-memory cache.
        refs = [
            {"sha256": digest, "path": paths[digest], "size": len(body)}
            for digest, body in content.items()
        ]
        returned = store.objects.read_many(refs)
        if returned != list(content.values()):
            raise ValueError(
                "Cluster read-back does not match original database contents"
            )
        for digest, body in content.items():
            connection.execute(
                "INSERT INTO metadata_payloads(sha256,path,size_bytes) VALUES (?,?,?) ON CONFLICT DO NOTHING",
                (digest, paths[digest], len(body)),
            )
        connection.execute("SET LOCAL skynet.relocate_payloads='on'")
        for table, identifier, field, body, key in documents:
            pointer = store.put(connection, table, identifier, field, body)
            if table == "tracking_journals":
                raw.execute(
                    "UPDATE tracking_journals SET payload=%s WHERE owner_id=%s AND scope=%s AND filename=%s",
                    (pointer.encode(), *key),
                )
            else:
                raw.execute(
                    f"UPDATE {table} SET {field}=%s WHERE id=%s", (pointer, identifier)
                )
            report["documents"] += 1
            report["bytes_before"] += len(body.encode())
            report["bytes_after"] += len(pointer.encode())
            report["tables"][table] = report["tables"].get(table, 0) + 1
    report["verified_objects"] = len(content)
    return report


if __name__ == "__main__":
    from .database import Database

    print(json.dumps(relocate(Database()), indent=2))
