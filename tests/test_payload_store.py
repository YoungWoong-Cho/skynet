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


def track_payload_reads(db, monkeypatch):
    reads = []
    original_read = db.payload_store.objects.read
    original_many = db.payload_store.objects.read_many

    def read(path, digest):
        reads.append(digest)
        return original_read(path, digest)

    def many(references):
        reads.extend(ref["sha256"] for ref in references)
        return original_many(references)

    monkeypatch.setattr(db.payload_store.objects, "read", read)
    monkeypatch.setattr(db.payload_store.objects, "read_many", many)
    _CACHE.clear()
    return reads


def test_seed_without_result_never_reads_payloads_and_default_keeps_full_history(object_db, monkeypatch):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    reads = track_payload_reads(db, monkeypatch)
    manifests = [{"slug": "qa", "revision": number, "body": "x" * 200000}
                 for number in range(4)]
    for manifest in manifests:
        assert db.upsert_seed_adapter(seed_key="qa", name="QA", manifest=manifest,
                                      materialize_result=False) is None
        _CACHE.clear()
    assert db.upsert_seed_adapter(seed_key="qa", name="QA", manifest=manifests[-1],
                                  materialize_result=False) is None
    assert reads == [], "Creating, advancing and comparing seeds must not read stored bodies"
    bundle = db.upsert_seed_adapter(seed_key="qa", name="QA", manifest=manifests[-1])
    assert bundle["version_count"] == bundle["latest_version_number"] == 4
    assert [version["manifest"] for version in bundle["versions"]] == list(reversed(manifests))
    assert len(reads) == 4, "The default public result materializes each version only once"


def test_seed_scalar_comparison_preserves_user_edits_and_archive_state(object_db, monkeypatch):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    owned = db.upsert_seed_adapter(seed_key="owned", name="Owned", manifest={"seed": 1})
    own_manifest = {"user": "custom configuration"}
    edited = db.edit_adapter(owned["id"], manifest=own_manifest, created_by="user")
    archived = db.upsert_seed_adapter(seed_key="archived", name="Archived", manifest={"seed": 1})
    archived = db.archive_adapter(archived["id"])
    reads = track_payload_reads(db, monkeypatch)
    db.upsert_seed_adapter(seed_key="owned", name="Owned", manifest={"seed": 2}, materialize_result=False)
    db.upsert_seed_adapter(seed_key="archived", name="Archived", manifest={"seed": 2}, materialize_result=False)
    assert reads == []
    with db.connection() as connection:
        rows = connection.execute(
            "SELECT adapter_key, max(version_number) AS version FROM adapters GROUP BY adapter_key"
        ).fetchall()
    assert {row["adapter_key"]: row["version"] for row in rows} == {owned["id"]: 2, archived["id"]: 2}
    assert db.get_adapter(owned["id"])["latest_version"] == edited["latest_version"]
    latest = db.get_adapter(archived["id"])
    assert latest["archived_at"] == archived["archived_at"] and not latest["enabled"]
    assert latest["latest_version"]["manifest"] == {"seed": 2}


def test_startup_seeding_reads_only_latest_catalog_body(object_db, monkeypatch):
    from skynet_app import pipeline_api
    from skynet_app.adapters import canonical_adapter_manifest
    from skynet_app.adapters.act_manifest import manifest as act_manifest

    db, _ = object_db
    db.payload_store = PayloadStore(db)
    manifest = act_manifest()
    canonical = canonical_adapter_manifest(manifest)
    for revision in range(3):
        db.upsert_seed_adapter(seed_key=manifest.slug, name=manifest.display_name,
                               manifest={**canonical, "description": str(revision)}, materialize_result=False)
    db.upsert_seed_adapter(seed_key=manifest.slug, name=manifest.display_name,
                           manifest=canonical, materialize_result=False)
    reads = track_payload_reads(db, monkeypatch)
    monkeypatch.setattr(pipeline_api, "builtin_adapter_manifests", lambda: [manifest])
    monkeypatch.setattr(pipeline_api, "get_evaluation_catalog", lambda: [])
    service = object.__new__(pipeline_api.PipelineService)
    service.database = db
    service._seed_registries()
    assert len(reads) == 1, "Startup's current catalog must not hydrate historical seed versions"


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
