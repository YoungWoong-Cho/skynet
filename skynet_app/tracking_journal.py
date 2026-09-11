"""Shared cursors and sanitized delivery queues for the tracking bridges."""

from __future__ import annotations

JOURNAL_FILES = frozenset(
    {
        "mlflow-state.json",
        "mlflow-spool.jsonl",
        "wandb-state.json",
        "wandb-spool.jsonl",
        "gpu-statistics.json",
    }
)


class JournalFile:
    def __init__(self, journal, name):
        self.journal = journal
        self.name = name

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
        return bytes(row[0])

    def read_text(self, encoding="utf-8"):
        return self.read_bytes().decode(encoding)

    def write_bytes(self, payload):
        journal = self.journal
        with journal.database.transaction() as connection:
            run = connection.execute(
                "SELECT owner_id FROM runs WHERE id=?", (journal.scope,)
            ).fetchone()
            if run and run[0] != journal.owner:
                raise ValueError("Tracking journal belongs to another workspace")
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
        return len(payload)


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
