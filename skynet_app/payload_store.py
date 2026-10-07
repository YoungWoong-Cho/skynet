"""External bodies for large immutable receipts; searchable columns stay in SQL."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from concurrent.futures import Future
from threading import RLock

from .metadata_objects import MetadataObjects
from .registry_reference_match import registry_reference_match

FIELDS = {
    "adapters": "manifest_json",
    "job_attempts": "execution_snapshot_json",
    "workflow_stages": "resolved_config_json",
}
# Offloaded documents whose projection (skynet_project_document) is written by the
# uploader, because the database trigger sees only the object reference. The paths
# are the subtrees that projection reads, so a repair can rebuild it remotely.
PROJECTED_PATHS = {
    "workflow_stages": ("context.target_dataset", "plan.native_config.canonical_evaluation.target_dataset"),
    "job_attempts": ("plan.native_config.initial_checkpoint", "migration_provenance.checkpoint",
                     "resolved_spec.native.config.initial_checkpoint"),
    # No paths: the projection reads the whole body, so a repair fetches it.
    "adapters": (),
}
MARKER = "$skynet_object_v1"
_CACHE = OrderedDict()
_CACHE_LOCK = RLock()
_CACHE_BYTES = 128 * 1024 * 1024

class ImmutableProjectionCache:
    """Byte-bounded LRU with shared in-flight reads of immutable content.

    Count limits made two large list views evict one another on every refresh.
    Cache encoded values so callers cannot mutate another request's projection.
    No cache lock is held while the remote loader runs.
    """

    def __init__(self, max_bytes=32 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.pending = {}
        self.size_bytes = 0
        self.lock = RLock()

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.size_bytes = 0

    def load(self, references, loader):
        values, owned = {}, {}
        with self.lock:
            for key, ref in references.items():
                if key in self.entries:
                    values[key] = self.entries[key][0]
                    self.entries.move_to_end(key)
                else:
                    if key not in self.pending:
                        self.pending[key] = Future()
                        owned[key] = ref
                    values[key] = self.pending[key]
        if owned:
            try:
                loaded = loader(list(owned.values()))
                if len(loaded) != len(owned):
                    raise ValueError("Incomplete immutable projection batch")
                encoded = [json.dumps(value, separators=(",", ":")).encode("utf-8")
                           for value in loaded]
                with self.lock:
                    for key, content in zip(owned, encoded):
                        size = len(content) + len(repr(key).encode("utf-8")) + 128
                        if size <= self.max_bytes:
                            self.entries[key] = (content, size)
                            self.size_bytes += size
                            while self.size_bytes > self.max_bytes:
                                _, (_, removed) = self.entries.popitem(last=False)
                                self.size_bytes -= removed
                        self.pending.pop(key).set_result(content)
            except BaseException as error:
                with self.lock:
                    for key in owned:
                        future = self.pending.pop(key, None)
                        if future is not None:
                            future.set_exception(error)
                raise
        return {key: json.loads(value.result() if isinstance(value, Future) else value)
                for key, value in values.items()}


_PROJECTIONS = ImmutableProjectionCache()
_REGISTRY_FLAGS = ImmutableProjectionCache()


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
        self.database = database
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
        return value if ref is None else self.read_bytes(ref).decode("utf-8")

    def read_bytes(self, ref):
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
        return content

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

    def project(self, documents, paths):
        paths = tuple(paths)
        references = {}
        for document in documents:
            if isinstance(document, dict) and MARKER in document:
                ref = document[MARKER]
                if ref["path"] != self.objects.path(ref["sha256"], "body"):
                    raise ValueError("Invalid metadata object path")
                key = (*self._key(ref["sha256"]), ref["size"], paths)
                references[key] = ref
        found = _PROJECTIONS.load(references, lambda refs: self.objects.project(refs, paths))
        result = []
        for document in documents:
            if isinstance(document, dict) and MARKER in document:
                ref = document[MARKER]
                value = found[(*self._key(ref["sha256"]), ref["size"], paths)]
            else:
                value = {}
                for path in paths:
                    item = document
                    for key in path.split("."):
                        item = item.get(key) if isinstance(item, dict) else None
                    value[path] = item
            # Callers can enrich a display without corrupting this immutable cache.
            result.append(json.loads(json.dumps(value)))
        return result

    def registry_matches(self, documents, identifiers):
        signature = tuple(sorted(identifiers))
        references = {}
        def key(ref):
            return (*self._key(ref["sha256"]), ref["size"], signature)
        for document in documents:
            if isinstance(document, dict) and MARKER in document:
                ref = document[MARKER]
                if ref["path"] != self.objects.path(ref["sha256"], "body"):
                    raise ValueError("Invalid metadata object path")
                references[key(ref)] = ref
        flags = _REGISTRY_FLAGS.load(references, lambda refs: self.objects.registry_matches(refs, signature))
        return [flags[key(document[MARKER])] if isinstance(document, dict) and MARKER in document
                else registry_reference_match(document, signature) for document in documents]

    def prepare(self, bodies):
        """Upload immutable bodies before taking the repository SQL write lock."""
        contents = {}
        for body in bodies:
            if body is None:
                continue
            if reference(body):
                raise ValueError("Object references cannot be supplied as document content")
            content = body if isinstance(body, bytes) else body.encode("utf-8")
            contents[hashlib.sha256(content).hexdigest()] = content
        with self.database.connection() as connection:
            known = {row['sha256']: row['path'] for row in connection.execute(
                'SELECT sha256,path FROM metadata_payloads WHERE sha256=ANY(?)', (list(contents),)
            ).fetchall()}
        uploaded = self.objects.put_many([content for digest, content in contents.items() if digest not in known])
        return contents, known, uploaded

    def register_prepared(self, connection, prepared):
        contents, known, uploaded = prepared
        current = {row['sha256']: row['path'] for row in connection.execute(
            'SELECT sha256,path FROM metadata_payloads WHERE sha256=ANY(?)', (list(known),)
        ).fetchall()}
        if current != known:
            raise RuntimeError('Metadata changed during preparation; retry required')
        from .maintenance_guard import guard_write
        for path in (known | uploaded).values():
            guard_write(connection, 'metadata_payload_refs', {'path': path})
        connection.executemany(
            'INSERT INTO metadata_payloads(sha256,path,size_bytes) VALUES (?,?,?) ON CONFLICT DO NOTHING',
            [(digest, (known | uploaded)[digest], len(content)) for digest, content in contents.items()],
        )
        for digest, content in contents.items():
            self._cache(digest, content)

    def put(self, connection, table, identifier, field, body):
        if body is None:
            return body
        if reference(body):
            raise ValueError("Object references cannot be supplied as document content")
        ref = self.put_bytes(connection, table, identifier, field, body.encode("utf-8"))
        if table in PROJECTED_PATHS:
            connection.execute(
                """INSERT INTO document_projections VALUES (?, ?, skynet_project_document(?, ?))
                   ON CONFLICT (table_name, record_id) DO UPDATE SET projection_json=excluded.projection_json""",
                (table, identifier, body, table),
            )
        return json.dumps({MARKER: ref}, separators=(",", ":"))

    def put_bytes(self, connection, table, identifier, field, content):
        """Store one body; its row reference keeps it from Storage cleanup."""
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
        from .maintenance_guard import guard_write
        guard_write(connection, 'metadata_payload_refs', {'path': path})
        self._cache(digest, content)
        connection.execute(
            """INSERT INTO metadata_payload_refs(table_name,record_id,field_name,sha256)
               VALUES (?,?,?,?) ON CONFLICT(table_name,record_id,field_name)
               DO UPDATE SET sha256=excluded.sha256""",
            (table, identifier, field, digest),
        )
        return {"sha256": digest, "path": path, "size": len(content)}

    def encode_fields(self, connection, table, identifier, fields):
        field = FIELDS.get(table)
        if field not in fields:
            return fields
        fields = dict(fields)
        fields[field] = self.put(connection, table, identifier, field, fields[field])
        return fields
