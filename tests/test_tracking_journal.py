"""Journal spools record their digest, so unchanged queues are served without chunk reads."""
import hashlib

from test_payload_store import object_db as object_db, track_payload_reads
from test_postgres import pg as pg
from test_tracking import FakeWandBBridge

from skynet_app.payload_store import _CACHE, PayloadStore
from skynet_app.tracking import WandBSettings
from skynet_app.tracking_journal import TrackingJournal, chunk_document


def spool_chunk_digests(db, scope):
    with db.connection() as connection:
        row = connection.execute(
            "SELECT payload FROM tracking_journals WHERE scope=? AND filename='wandb-spool.jsonl'",
            (scope,),
        ).fetchone()
    return {ref["sha256"] for ref in chunk_document(bytes(row[0]))["chunks"]}


def test_unchanged_journal_spool_is_served_from_its_recorded_digest(object_db, tmp_path, monkeypatch):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    journal = TrackingJournal(db, "digest-run")
    settings = WandBSettings(api_key="fake", entity="team", auto_flush=False)
    bridge = FakeWandBBridge(tmp_path / "capsule", settings, journal=journal)
    bridge.ensure_run(local_run_id="digest-run", entity="team", project="test", run_name="test", group="test")
    for step in range(3):
        bridge.log_metrics("digest-run", {"loss": step}, step=step, idempotency_key=f"sample:{step}")
    spool = journal.file("wandb-spool.jsonl")
    content = spool.read_bytes()
    chunks = spool_chunk_digests(db, "digest-run")
    reads = track_payload_reads(db, monkeypatch)
    assert spool.sha256() == hashlib.sha256(content).hexdigest()
    assert reads == []

    # Another host's bridge re-reads cursors and queue identity without the chunk objects.
    restarted = FakeWandBBridge(tmp_path / "capsule", settings, journal=journal)
    assert restarted.pending_count() == 4
    assert restarted.metric_idempotency_keys() == {f"sample:{step}" for step in range(3)}
    assert restarted.metric_names_by_idempotency_key() == {f"sample:{step}": {"loss"} for step in range(3)}
    assert not chunks & set(reads)

    # A write from elsewhere changes the recorded digest, so the queue is fetched again.
    spool.write_bytes(content + b"\n")
    _CACHE.clear()
    assert restarted.pending_count() == 4
    assert spool_chunk_digests(db, "digest-run") & set(reads)
