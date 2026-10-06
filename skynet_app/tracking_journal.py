"""Shared cursors and sanitized delivery queues for the tracking bridges."""

from __future__ import annotations

import hashlib
import json

from .payload_store import MARKER
from .workspace_schema import LEGACY_WORKSPACE

CHUNKS = "$skynet_chunks_v1"
CHUNK_BYTES = 64 * 1024
# Stored as one object so its path can be handed to a job; every other journal is chunked.
SINGLE_OBJECT_FILE = "tracking-artifact-links.json"

JOURNAL_FILES = frozenset(
    {
        "mlflow-state.json",
        "mlflow-spool.jsonl",
        "wandb-state.json",
        "wandb-spool.jsonl",
        "gpu-statistics.json",
        "tracking-artifact-links.json",
    }
)


class JournalFile:
    def __init__(self, journal, name):
        self.journal = journal
        self.name = name

    def __str__(self):
        row = self._row()
        if row and self.name == SINGLE_OBJECT_FILE:
            value = json.loads(bytes(row[0]))
            if MARKER in value:
                return value[MARKER]["path"]
        return f"skynet:journal/{self.journal.owner}/{self.journal.scope}/{self.name}"

    def _row(self):
        journal = self.journal
        with journal.database.connection() as connection:
            return connection.execute(
                "SELECT payload FROM tracking_journals WHERE owner_id=? AND scope=? AND filename=?",
                (journal.owner, journal.scope, self.name),
            ).fetchone()

    def exists(self):
        return self._row() is not None

    def read_bytes(self):
        row = self._row()
        if row is None:
            raise FileNotFoundError(self.name)
        payload = bytes(row[0])
        return decode_payload(self.journal.database.payload_store, payload)

    def read_text(self, encoding="utf-8"):
        return self.read_bytes().decode(encoding)

    def write_bytes(self, payload):
        journal = self.journal
        database = journal.database
        size = len(payload)
        # The bridge may already hold this reentrant lock. Serialize each journal,
        # while keeping remote object transfers outside the repository write lock.
        with journal.lock, database.metadata_lock:
            with database.connection() as connection:
                original_run = connection.execute(
                    "SELECT owner_id FROM runs WHERE id=?", (journal.scope,)
                ).fetchone()
                if original_run and original_run[0] != journal.owner:
                    raise ValueError("Tracking journal belongs to another workspace")
            before = self._row()
            original = bytes(before[0]) if before else None
            store = database.payload_store
            prepared = []
            if store:
                text = payload.decode("utf-8")
                chunks = ([text] if self.name == SINGLE_OBJECT_FILE else
                          [text[start:start + CHUNK_BYTES]
                           for start in range(0, len(text), CHUNK_BYTES)])
                contents = {hashlib.sha256(chunk.encode()).hexdigest(): chunk.encode()
                            for chunk in chunks}
                with database.connection() as connection:
                    known = {row["sha256"]: row["path"] for row in connection.execute(
                        "SELECT sha256,path FROM metadata_payloads WHERE sha256=ANY(?)",
                        (list(contents),),
                    ).fetchall()}
                uploaded = store.objects.put_many(
                    [content for digest, content in contents.items() if digest not in known]
                )
                prepared = [(digest, (known | uploaded)[digest], content)
                            for digest, content in contents.items()]
            with database.transaction() as connection:
                run = connection.execute(
                    "SELECT owner_id FROM runs WHERE id=?", (journal.scope,)
                ).fetchone()
                if run and run[0] != journal.owner:
                    raise ValueError("Tracking journal belongs to another workspace")
                if original_run and not run:
                    raise RuntimeError("Run was deleted during journal preparation")
                current = connection.execute(
                    "SELECT payload FROM tracking_journals WHERE owner_id=? AND scope=? AND filename=?",
                    (journal.owner, journal.scope, self.name),
                ).fetchone()
                if (bytes(current[0]) if current else None) != original:
                    raise RuntimeError("Tracking journal changed during object preparation; retry required")
                if store:
                    # A concurrent deletion of a previously indexed object must
                    # fail closed, rather than resurrecting a stale file pointer.
                    current_known = {row["sha256"]: row["path"] for row in connection.execute(
                        "SELECT sha256,path FROM metadata_payloads WHERE sha256=ANY(?)",
                        (list(known),),
                    ).fetchall()}
                    if current_known != known:
                        raise RuntimeError("Tracking objects changed during preparation; retry required")
                    connection.executemany(
                        "INSERT INTO metadata_payloads(sha256,path,size_bytes) VALUES (?,?,?) ON CONFLICT DO NOTHING",
                        [(digest, path, len(content)) for digest, path, content in prepared],
                    )
                    for digest, _, content in prepared:
                        store._cache(digest, content)
                    identifier = journal.scope if run else journal.owner + ":" + journal.scope
                    payload = encode_payload(connection, store, identifier, self.name,
                                             payload, original or b"")
                connection.execute(
                    """INSERT INTO tracking_journals(owner_id,scope,filename,run_id,payload)
                       VALUES (?,?,?,?,?) ON CONFLICT(owner_id,scope,filename)
                       DO UPDATE SET payload=excluded.payload""",
                    (journal.owner, journal.scope, self.name, journal.scope if run else None, payload),
                )
        return size


def chunk_document(payload):
    if not payload.startswith(b'{"' + CHUNKS.encode() + b'":'):
        return None
    return json.loads(payload)[CHUNKS]


def decode_payload(store, payload):
    if not store:
        return payload
    document = chunk_document(payload)
    if document is None:
        return store.read(payload.decode("utf-8")).encode("utf-8")
    pointers = [json.dumps({MARKER: ref}) for ref in document["chunks"]]
    store.prefetch(pointers)
    content = b"".join(store.read(pointer).encode("utf-8") for pointer in pointers)
    if hashlib.sha256(content).hexdigest() != document["sha256"]:
        raise ValueError("Tracking journal checksum mismatch")
    return content


def encode_payload(connection, store, identifier, name, payload, existing=b""):
    """Appending to a spool uploads only the changed tail, not every old event."""
    if name == SINGLE_OBJECT_FILE:
        return store.put(connection, "tracking_journals", identifier, name, payload.decode()).encode()
    old = chunk_document(existing) or {"chunks": []}
    # Split on Unicode boundaries; JSONL queues may include non-ASCII text.
    text = payload.decode("utf-8")
    chunks = [
        text[start : start + CHUNK_BYTES] for start in range(0, len(text), CHUNK_BYTES)
    ]
    refs = []
    for index, chunk in enumerate(chunks):
        digest = hashlib.sha256(chunk.encode()).hexdigest()
        previous = old["chunks"][index] if index < len(old["chunks"]) else None
        if previous and previous["sha256"] == digest:
            refs.append(previous)
        else:
            pointer = store.put(
                connection, "tracking_journals", identifier, f"{name}:{index}", chunk
            )
            refs.append(json.loads(pointer)[MARKER])
    for index in range(len(chunks), len(old["chunks"])):
        connection.execute(
            "DELETE FROM metadata_payload_refs WHERE table_name='tracking_journals' AND record_id=? AND field_name=?",
            (identifier, f"{name}:{index}"),
        )
    connection.execute(
        "DELETE FROM metadata_payload_refs WHERE table_name='tracking_journals' AND record_id=? AND field_name=?",
        (identifier, name),
    )
    return json.dumps(
        {CHUNKS: {"chunks": refs, "sha256": hashlib.sha256(payload).hexdigest()}},
        separators=(",", ":"),
    ).encode()


class TrackingJournal:
    def __init__(self, database, scope):
        self.database = database
        self.scope = scope
        self.owner = database.workspace_id or LEGACY_WORKSPACE
        self.lock = database.operation_lock(
            "tracking-journal:" + self.owner + ":" + scope
        )

    def file(self, name):
        if name not in JOURNAL_FILES:
            raise ValueError("Unknown tracking journal file")
        return JournalFile(self, name)
