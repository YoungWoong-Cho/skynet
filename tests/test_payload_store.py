"""Real SQL and disk receipts: migration is lossless, immutable and portable."""

import json
from pathlib import Path

import pytest
from test_postgres import pg as pg

from skynet_app.database import Database
from skynet_app.db_backend import INTEGRITY_ERRORS
from skynet_app.metadata_objects import _REMOTE, MetadataObjects
from skynet_app.payload_migration import relocate
from skynet_app.payload_store import _CACHE, PayloadStore
from skynet_app.tracking_journal import TrackingJournal


@pytest.fixture
def object_db(pg, tmp_path, monkeypatch):
    import subprocess

    db, url = pg
    db.endpoint_config = {
        "ssh_host": "test",
        "object_store_root": str(tmp_path / "objects"),
    }

    def exchange(self, request):
        result = subprocess.run(
            ["python3", "-c", _REMOTE],
            input=json.dumps(request),
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise OSError(result.stderr)
        return json.loads(result.stdout)

    monkeypatch.setattr(MetadataObjects, "_exchange", exchange)
    return db, url


def test_relocate_preserves_bytes_hashes_and_refuses_receipt_mutations(object_db):
    db, url = object_db
    adapter = db.upsert_seed_adapter(
        seed_key="qa", name="QA", manifest={"large": "x" * 200000}
    )
    project = db.create_project("p")
    experiment = db.create_experiment(
        project_id=project["id"], name="e", requested_spec={}
    )
    variant = db.create_variant(
        experiment["latest_revision"]["id"], name="v", parameters={}, resolved_spec={}
    )
    run = db.create_run(
        variant["id"],
        seed=1,
        adapter_name="qa",
        adapter_version="1",
        run_directory="/tmp/qa",
        status="SUCCEEDED",
    )
    stage = db.create_stage(
        run["id"],
        stage_type="TRAIN",
        name="train",
        resolved_config={"body": "s" * 100000},
        status="SUCCEEDED",
    )
    attempt = db.create_job_attempt(
        stage["id"],
        status="SUCCEEDED",
        execution_snapshot_json={"receipt": "r" * 300000},
    )
    journal = TrackingJournal(db, run["id"]).file("wandb-spool.jsonl")
    payload = b'{"metric":1}\n' * 100000
    journal.write_bytes(payload)
    original = db.get_run(run["id"])
    db.payload_store = PayloadStore(db)
    report = relocate(db)
    assert report["documents"] == 4
    assert report["bytes_after"] < report["bytes_before"] / 100
    _CACHE.clear()
    assert db.get_run(run["id"]) == original
    assert journal.read_bytes() == payload
    assert (
        db.get_adapter(adapter["id"])["versions"][0]["manifest"]["large"]
        == "x" * 200000
    )
    with pytest.raises(INTEGRITY_ERRORS), db.transaction() as c:
        c.execute(
            "UPDATE job_attempts SET execution_snapshot_json=? WHERE id=?",
            ("{}", attempt["id"]),
        )
    with pytest.raises(INTEGRITY_ERRORS), db.transaction() as c:
        c.execute("SET LOCAL skynet.relocate_payloads='on'")
        c.execute(
            "UPDATE job_attempts SET execution_snapshot_json=? WHERE id=?",
            ("{}", attempt["id"]),
        )
    assert relocate(db)["documents"] == 0
    # A second process/database object reads the same cluster bodies.
    second = Database(url=url)
    second.endpoint_config = db.endpoint_config
    second.payload_store = PayloadStore(second)
    _CACHE.clear()
    assert second.get_run(run["id"]) == original


def test_new_stage_writes_external_body_and_missing_or_changed_object_fails(object_db):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    project = db.create_project("p")
    experiment = db.create_experiment(
        project_id=project["id"], name="e", requested_spec={}
    )
    variant = db.create_variant(
        experiment["latest_revision"]["id"], name="v", parameters={}, resolved_spec={}
    )
    run = db.create_run(
        variant["id"],
        seed=1,
        adapter_name="qa",
        adapter_version="1",
        run_directory="/tmp/qa",
    )
    stage = db.create_stage(
        run["id"],
        stage_type="TRAIN",
        name="train",
        resolved_config={"value": "original"},
    )
    with db.backend.connect().raw as raw:
        pointer = json.loads(
            raw.execute(
                "SELECT resolved_config_json FROM workflow_stages WHERE id=%s",
                (stage["id"],),
            ).fetchone()[0]
        )["$skynet_object_v1"]
    assert Path(pointer["path"]).read_text() == '{"value":"original"}'
    Path(pointer["path"]).write_text("corrupt")
    _CACHE.clear()
    with pytest.raises((ValueError, OSError), match="checksum"):
        db.list_stages(run["id"])


def test_append_journal_reuses_old_chunks_and_stays_small_in_db(object_db, monkeypatch):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    journal = TrackingJournal(db, "qa-journal").file("wandb-spool.jsonl")
    content = b'{"epoch":1}\n' * 100000
    journal.write_bytes(content)
    puts = []
    original = db.payload_store.objects.put

    def record(content, **kwargs):
        puts.append(len(content))
        return original(content, **kwargs)

    monkeypatch.setattr(db.payload_store.objects, "put", record)
    journal.write_bytes(content + b'{"epoch":2}\n')
    assert len(puts) <= 2 and sum(puts) < 131072
    _CACHE.clear()
    assert journal.read_bytes() == content + b'{"epoch":2}\n'
    with db.connection() as c:
        assert (
            c.execute("SELECT octet_length(payload) FROM tracking_journals").fetchone()[
                0
            ]
            < len(content) / 50
        )


def test_deleted_evaluation_metrics_are_removed_without_losing_training_metrics():
    from skynet_app.history_journals import cleaned_payload

    deleted = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    payload = {
        "sequence": 5,
        "operation": "log_batch",
        "payload": {
            "local_run_id": "kept",
            "metrics": [
                {"key": "train/loss", "value": 0.2},
                {"key": f"evaluation/{deleted}/success", "value": 1},
            ],
        },
    }
    clean = json.loads(
        cleaned_payload(
            "wandb-spool.jsonl", json.dumps(payload).encode() + b"\n", [deleted]
        )
    )
    assert clean["sequence"] == 5
    assert clean["payload"]["metrics"] == [{"key": "train/loss", "value": 0.2}]
