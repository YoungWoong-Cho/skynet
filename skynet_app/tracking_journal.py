"""Shared cursors and sanitized delivery queues for the tracking bridges."""

from __future__ import annotations

import hashlib
import json

from .payload_store import MARKER

CHUNKS = "$skynet_chunks_v1"
CHUNK_BYTES = 64 * 1024

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
        if row and self.name == "tracking-artifact-links.json":
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
        size = len(payload)
        with journal.database.transaction() as connection:
            run = connection.execute(
                "SELECT owner_id FROM runs WHERE id=?", (journal.scope,)
            ).fetchone()
            if run and run[0] != journal.owner:
                raise ValueError("Tracking journal belongs to another workspace")
            if journal.database.payload_store:
                identifier = (
                    journal.scope if run else journal.owner + ":" + journal.scope
                )
                existing = connection.execute(
                    "SELECT payload FROM tracking_journals WHERE owner_id=? AND scope=? AND filename=?",
                    (journal.owner, journal.scope, self.name),
                ).fetchone()
                if self.name == "tracking-artifact-links.json":
                    payload = journal.database.payload_store.put(
                        connection,
                        "tracking_journals",
                        identifier,
                        self.name,
                        payload.decode(),
                    ).encode()
                else:
                    payload = encode_payload(
                        connection,
                        journal.database.payload_store,
                        identifier,
                        self.name,
                        payload,
                        bytes(existing[0]) if existing else b"",
                    )
            connection.execute(
                """INSERT INTO tracking_journals(owner_id,scope,filename,run_id,payload)
                   VALUES (?,?,?,?,?) ON CONFLICT(owner_id,scope,filename)
                   DO UPDATE SET payload=excluded.payload""",
                (
                    journal.owner,
                    journal.scope,
                    self.name,
                    journal.scope if run else None,
                    payload,
                ),
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
        self.owner = database.workspace_id or "legacy"
        self.lock = database.operation_lock(
            "tracking-journal:" + self.owner + ":" + scope
        )

    def file(self, name):
        if name not in JOURNAL_FILES:
            raise ValueError("Unknown tracking journal file")
        return JournalFile(self, name)
