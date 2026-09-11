"""External bodies for large immutable receipts; searchable columns stay in SQL."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from threading import RLock

from .metadata_objects import MetadataObjects

FIELDS = {
    "adapters": "manifest_json",
    "job_attempts": "execution_snapshot_json",
    "workflow_stages": "resolved_config_json",
}
MARKER = "$skynet_object_v1"
_CACHE = OrderedDict()
_CACHE_LOCK = RLock()
_CACHE_BYTES = 128 * 1024 * 1024


def reference(value):
    if not isinstance(value, str) or MARKER not in value[:80]:
        return None
    try:
        result = json.loads(value)
    except ValueError:
        return None
    return (
        result.get(MARKER)
        if isinstance(result, dict) and set(result) == {MARKER}
        else None
    )


class PayloadStore:
    def __init__(self, database):
        self.objects = MetadataObjects(database)

    def _key(self, digest):
        return self.objects.host, str(self.objects.root), digest

    def _cache(self, digest, content):
        with _CACHE_LOCK:
            _CACHE[self._key(digest)] = content
            _CACHE.move_to_end(self._key(digest))
            while sum(len(v) for v in _CACHE.values()) > _CACHE_BYTES:
                _CACHE.popitem(last=False)

    def read(self, value):
        ref = reference(value)
        if ref is None:
            return value
        digest = ref["sha256"]
        with _CACHE_LOCK:
            content = _CACHE.get(self._key(digest))
        if content is None:
            content = self.objects.read(ref["path"], digest)
            self._cache(digest, content)
        if len(content) != ref["size"] or ref["path"] != self.objects.path(
            digest, "body"
        ):
            raise ValueError("Invalid stored metadata reference")
        return content.decode("utf-8")

    def prefetch(self, values):
        refs = {}
        with _CACHE_LOCK:
            for value in values:
                ref = reference(value)
                if ref and self._key(ref["sha256"]) not in _CACHE:
                    refs[ref["sha256"]] = ref
        # One SSH exchange per bounded batch, not one per adapter in a table.
        for content, ref in zip(
            self.objects.read_many(list(refs.values())), refs.values()
        ):
            self._cache(ref["sha256"], content)

    def put(self, connection, table, identifier, field, body):
        if body is None:
            return body
        if reference(body):
            raise ValueError("Object references cannot be supplied as document content")
        content = body.encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        row = connection.execute(
            "SELECT path FROM metadata_payloads WHERE sha256=?", (digest,)
        ).fetchone()
        if row:
            path = row[0]
        else:
            path = self.objects.put(content, name="body")
            connection.execute(
                "INSERT INTO metadata_payloads(sha256,path,size_bytes) VALUES (?,?,?) ON CONFLICT DO NOTHING",
                (digest, path, len(content)),
            )
        self._cache(digest, content)
        connection.execute(
            """INSERT INTO metadata_payload_refs(table_name,record_id,field_name,sha256)
               VALUES (?,?,?,?) ON CONFLICT(table_name,record_id,field_name)
               DO UPDATE SET sha256=excluded.sha256""",
            (table, identifier, field, digest),
        )
        return json.dumps(
            {MARKER: {"sha256": digest, "path": path, "size": len(content)}},
            separators=(",", ":"),
        )

    def encode_fields(self, connection, table, identifier, fields):
        field = FIELDS.get(table)
        if field not in fields:
            return fields
        fields = dict(fields)
        fields[field] = self.put(connection, table, identifier, field, fields[field])
        return fields
