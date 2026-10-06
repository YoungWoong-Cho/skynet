"""Real SQL and disk receipts: migration is lossless, immutable and portable."""

import json
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("external", [False, True])
def test_run_list_uses_compact_resume_evidence_and_preserves_details(object_db, monkeypatch, external):
    from skynet_app.pipeline_api import _attach_run_progress_summaries
    db, _ = object_db
    if external:
        db.payload_store = PayloadStore(db)
    project = db.create_project("compact-list")
    experiment = db.create_experiment(project_id=project["id"], name="compact", requested_spec={})
    spec = {"train": {"max_steps": 8000}, "resources": {"gpu": {"count": 1}},
            "source": {"adapter_manifest": {"train": {"capsule_files": {"code.py": "x" * 200000},
                "progress": {"unit": "step", "total_path": "train.max_steps", "starts_at_zero": True,
                             "source": {"kind": "jsonl", "path": "progress.jsonl", "completed_key": "step"}}}}},
            "data": {"bundle": {"name": "cube", "assignments": [{"role": "training_data",
                "version": {"format": "fixture", "metadata": {"display_name": "Cube",
                    "episodes": [{"body": "x" * 20000}] * 51, "shared_artifacts": ["x" * 200000]}}}]}}}
    variant = db.create_variant(experiment["latest_revision"]["id"], name="one", parameters={}, resolved_spec=spec)
    run = db.create_run(variant["id"], seed=1, adapter_name="generic", adapter_version="1",
                        run_directory="/fixture/run", status="RUNNING")
    stage = db.create_stage(run["id"], stage_type="TRAIN", name="train")
    snapshot = {"plan": {"native_config": {"initial_checkpoint": "/saved.ckpt"}},
                "code": "x" * 200000}
    attempt = db.create_job_attempt(stage["id"], status="RUNNING", execution_snapshot_json=snapshot)
    db.update_job_attempt(attempt["id"], started_at="2026-09-20T10:00:00Z")
    reads = track_payload_reads(db, monkeypatch) if external else []
    rows = db.list_runs()
    _attach_run_progress_summaries(db, rows)
    assert reads == [], "Listing must never download execution bodies"
    assert len(json.dumps(rows)) < 15000
    assert rows[0]["latest_attempt"]["has_initial_checkpoint"] is True
    assert "execution_snapshot_json" not in rows[0]["latest_attempt"]
    assert rows[0]["progress_summary"]["completed"] is None, "Pinned resume must not assume a zero baseline"
    assert rows[0]["progress_summary"]["total"] == 8000
    assert rows[0]["training_data"][0]["episodes"] == 51
    assert rows[0]["resources"] == spec["resources"]
    detail = db.get_run(run["id"])
    assert detail["attempts"][0]["execution_snapshot_json"] == snapshot
    assert detail["resolved_spec_json"] == spec


def test_reference_projection_keeps_nested_paths_without_loading_bodies(object_db, monkeypatch):
    from skynet_app.maintenance import Maintenance
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    project = db.create_project("references")
    spec = {"nested": [{"path": "/shared/cube"}, ["/shared/other", "relative", "/"]],
            "body": "x" * 200000}
    experiment = db.create_experiment(project_id=project["id"], name="refs", requested_spec=spec)
    variant = db.create_variant(experiment["latest_revision"]["id"], name="one", parameters={}, resolved_spec=spec)
    run = db.create_run(variant["id"], seed=1, adapter_name="generic", adapter_version="1",
                       run_directory="/fixture/run", status="SUCCEEDED")
    stage = db.create_stage(run["id"], stage_type="TRAIN", name="train")
    attempt = db.create_job_attempt(stage["id"], execution_snapshot_json={"large": "x" * 200000},
                                   stdout_path="/shared/log.out")
    reads = track_payload_reads(db, monkeypatch)
    manager = Maintenance(db, object())
    with db.connection() as c:
        refs = manager._references(c)
        retained = manager._references(c, {"job_attempts": [attempt]})
    assert reads == []
    for table, identifier in (("experiment_revisions", experiment["latest_revision"]["id"]),
                              ("variants", variant["id"])):
        assert {path for path, kind, owner in refs if (kind, owner) == (table, identifier)} == {
            "/shared/cube", "/shared/other", "/"}
    assert ("/shared/log.out", "job_attempts", attempt["id"]) in refs
    assert ("/shared/log.out", "job_attempts", attempt["id"]) not in retained
    assert manager._referenced("/shared/cube/checkpoint", refs, {})


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
    assert len(reads) <= 4 and len(reads) == len(set(reads)), "No version may be fetched twice; prepared bodies are already cached"


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
    assert len(reads) <= 1, "Startup must only fetch the latest body, unless preparation already cached it"


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


def test_binary_body_round_trips_and_keeps_its_row_reference(object_db):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    content = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 64
    with db.transaction(payloads=(content,)) as connection:
        ref = db.payload_store.put_bytes(connection, "note_attachments", "fixture", "content", content)
    assert Path(ref["path"]).read_bytes() == content and ref["size"] == len(content)
    _CACHE.clear()
    assert db.payload_store.read_bytes(ref) == content
    with db.connection() as c:
        assert c.execute(
            "SELECT sha256 FROM metadata_payload_refs WHERE table_name='note_attachments' AND record_id='fixture'"
        ).fetchone()[0] == ref["sha256"]
    with pytest.raises(ValueError, match="Invalid stored metadata reference"):
        db.payload_store.read_bytes({**ref, "size": len(content) + 1})


def test_append_journal_reuses_old_chunks_and_stays_small_in_db(object_db, monkeypatch):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    journal = TrackingJournal(db, "qa-journal").file("wandb-spool.jsonl")
    content = b'{"epoch":1}\n' * 100000
    journal.write_bytes(content)
    puts = []
    original = db.payload_store.objects.put_many

    def record(contents):
        puts.extend(len(content) for content in contents)
        return original(contents)

    monkeypatch.setattr(db.payload_store.objects, "put_many", record)
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


def test_journal_upload_does_not_hold_repository_write_lock(object_db, monkeypatch):
    from skynet_app.db_backend import lock_key
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    journal = TrackingJournal(db, "unblocked-journal").file("wandb-state.json")
    original = db.payload_store.objects.put_many
    def upload(contents):
        with db.connection() as connection:
            key = lock_key("repository-write")
            assert connection.execute("SELECT pg_try_advisory_lock(?)", (key,)).fetchone()[0]
            connection.execute("SELECT pg_advisory_unlock(?)", (key,))
        return original(contents)
    monkeypatch.setattr(db.payload_store.objects, "put_many", upload)
    body = json.dumps({"state": "界" * 100000}).encode()
    journal.write_bytes(body)
    assert journal.read_bytes() == body


def test_failed_journal_upload_preserves_previous_committed_state(object_db, monkeypatch):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    journal = TrackingJournal(db, "retry-journal").file("wandb-state.json")
    original = b'{"acked_through":1}'
    journal.write_bytes(original)
    def unavailable(_):
        raise ConnectionError("object storage unavailable")
    monkeypatch.setattr(db.payload_store.objects, "put_many", unavailable)
    with pytest.raises(ConnectionError, match="object storage unavailable"):
        journal.write_bytes(b'{"acked_through":2}')
    assert journal.read_bytes() == original


def test_prepared_journal_cannot_overwrite_a_concurrent_change(object_db, monkeypatch):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    journal = TrackingJournal(db, "changed-journal").file("wandb-state.json")
    journal.write_bytes(b'{"acked_through":1}')
    original = db.payload_store.objects.put_many
    def upload(contents):
        result = original(contents)
        with db.transaction() as connection:
            connection.execute("UPDATE tracking_journals SET payload=? WHERE scope=?",
                               (b'{"acked_through":3}', "changed-journal"))
        return result
    monkeypatch.setattr(db.payload_store.objects, "put_many", upload)
    with pytest.raises(RuntimeError, match="changed during object preparation"):
        journal.write_bytes(b'{"acked_through":2}')
    assert journal.read_bytes() == b'{"acked_through":3}'


def test_compressed_transport_preserves_disk_bytes_and_reduces_upload(object_db, monkeypatch):
    import base64, hashlib
    db, _ = object_db
    objects = MetadataObjects(db)
    body = json.dumps({"capsule": "print('frozen code')\n" * 300000}).encode()
    wire = []
    exchange = objects._exchange
    def measure(request):
        wire.append(len(json.dumps(request)))
        return exchange(request)
    monkeypatch.setattr(objects, "_exchange", measure)
    digest = hashlib.sha256(body).hexdigest()
    path = objects.put_many([body])[digest]
    assert Path(path).read_bytes() == body
    assert wire[0] < len(body) // 20
    assert objects.read(path, digest) == body
    assert objects.read_many([{"sha256": digest, "path": path, "size": len(body)}]) == [body]
    # Legacy identity transport remains readable without changing stored objects.
    response = exchange({"operation": "put", "root": str(objects.root), "sha256": digest,
        "name": "body", "content": base64.b64encode(body).decode()})
    assert response["size"] == len(body)


@pytest.mark.parametrize("kind", ["oversized", "truncated", "trailing", "unknown"])
def test_compressed_metadata_rejects_invalid_or_unbounded_content(object_db, kind):
    import base64, hashlib, zlib
    from skynet_app.metadata_objects import MAX_BYTES, _decode_content
    db, _ = object_db
    objects = MetadataObjects(db)
    body = b"x" * (MAX_BYTES + 1 if kind == "oversized" else 100)
    packed = zlib.compress(body, 1)
    if kind == "truncated": packed = packed[:-2]
    if kind == "trailing": packed += b"unexpected"
    value = {"encoding": "unknown" if kind == "unknown" else "zlib", "content": base64.b64encode(packed).decode()}
    with pytest.raises(ValueError): _decode_content(value)
    with pytest.raises(OSError):
        objects._exchange({"operation": "put", "root": str(objects.root),
            "sha256": hashlib.sha256(body).hexdigest(), "name": "body", **value})


def test_offloaded_evaluation_targets_are_found_and_repaired_without_body_reads(object_db, monkeypatch, tmp_path):
    from skynet_app import prepared_deletion
    from skynet_app.payload_store import PROJECTED_PATHS
    from test_evaluation_dataset_lifecycle import dataset, evaluation_reference
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    reads = track_payload_reads(db, monkeypatch)
    parent, target = dataset(db)
    experiment, run, stage = evaluation_reference(db, target, kind='context')
    with db.connection() as c:
        # Nested inside a JSON document the column keeps its object reference instead of being hydrated.
        marker = c.execute("SELECT to_jsonb(s) AS document FROM workflow_stages s WHERE id=?", (stage['id'],)).fetchone()[0]['resolved_config_json']
        assert marker.startswith('{"$skynet_object_v1"'), 'the stage body is offloaded'
        projected = c.execute("SELECT projection_json FROM document_projections WHERE table_name='workflow_stages' AND record_id=?", (stage['id'],)).fetchone()[0]
    assert projected['assignments'][0]['digest'] == target['manifest_sha256']
    assert db.data_version_usage(target['manifest_sha256']) == [dict(experiment_id=experiment['id'], name=experiment['name'], revision_number=1, run_id=run['id'], run_status='COMPLETED')]
    service = SimpleNamespace(database=db, root=tmp_path)
    assert prepared_deletion.preview(service, SimpleNamespace(owns=lambda *_: True), 'dataset', target['id'])['blockers'][0]['id'] == experiment['id']
    with pytest.raises(ValueError, match='used by an experiment'):
        db.delete_prepared_dataset(parent['id'], lambda *_: None, version_id=target['id'])
    assert reads == [], 'usage and deletion checks never download stage bodies'
    # Rows written before migration 020 have no projection: deletion fails closed, the repair rebuilds them remotely.
    with db.transaction() as c:
        c.execute("DELETE FROM document_projections WHERE table_name='workflow_stages' AND record_id=?", (stage['id'],))
    with pytest.raises(ValueError, match='incomplete'):
        db.delete_prepared_dataset(parent['id'], lambda *_: None, version_id=target['id'])
    projected_paths = []
    original_project = db.payload_store.objects.project
    def project(references, paths):
        projected_paths.append(tuple(paths))
        return original_project(references, paths)
    monkeypatch.setattr(db.payload_store.objects, 'project', project)
    assert db.repair_document_projections() == 1
    assert projected_paths == [PROJECTED_PATHS['workflow_stages']] and reads == []
    assert db.repair_document_projections() == 0
    assert db.data_version_usage(target['manifest_sha256'])[0]['experiment_id'] == experiment['id']
    # An offloaded execution snapshot projects its resume pin; the run list never inspects the body.
    train = db.create_stage(run['id'], stage_type='TRAIN', name='train')
    attempt = db.create_job_attempt(train['id'], status='RUNNING',
                                    execution_snapshot_json={'plan': {'native_config': {'initial_checkpoint': '/saved.ckpt'}}, 'code': 'x' * 200000})
    monkeypatch.setattr(db.payload_store.objects, 'truthy_paths', lambda *_: pytest.fail('the list must not inspect snapshots'))
    assert db.run_progress_evidence([run['id']])[run['id']]['attempts'][0]['has_initial_checkpoint'] is True
    with db.transaction() as c:
        c.execute("DELETE FROM document_projections WHERE table_name='job_attempts' AND record_id=?", (attempt['id'],))
    with pytest.raises(ValueError, match='incomplete'):
        db.run_progress_evidence([run['id']])
    projected_paths.clear()
    assert db.repair_document_projections() == 1
    assert projected_paths == [PROJECTED_PATHS['job_attempts']] and reads == []
    assert db.run_progress_evidence([run['id']])[run['id']]['attempts'][0]['has_initial_checkpoint'] is True
    with db.transaction() as c:
        c.execute('DELETE FROM workflow_stages WHERE id=?', (stage['id'],))
    assert db.data_version_usage(target['manifest_sha256']) == []
    assert db.delete_prepared_dataset(parent['id'], lambda *_: None, version_id=target['id'])['deleted']


def test_subtree_projection_equals_full_body_projection(object_db):
    from skynet_app.payload_store import PROJECTED_PATHS
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    target = {'version_id': 'v', 'manifest_sha256': 'd' * 64, 'path': '/t', 'metadata': {'x': 1}}
    full = {'context': {'target_dataset': target, 'other': 'x'}, 'plan': {'argv': ['a'], 'native_config': {'canonical_evaluation': {'target_dataset': target}}}}
    subtree = {'context': {'target_dataset': target}, 'plan': {'native_config': {'canonical_evaluation': {'target_dataset': target}}}}
    with db.connection() as c:
        values = db.payload_store.project([full], PROJECTED_PATHS['workflow_stages'])[0]
        assert set(values) == set(PROJECTED_PATHS['workflow_stages'])
        rows = [c.execute("SELECT skynet_project_document(?, 'workflow_stages')", (json.dumps(body),)).fetchone()[0] for body in (full, subtree, {'plan': subtree['plan']}, {})]
    assert rows[0] == rows[1] == rows[2] == {'assignments': [{'role': 'evaluation_target', 'version_id': 'v', 'digest': 'd' * 64, 'path': '/t'}]}
    assert rows[3] == {'assignments': []}


def test_offloaded_adapter_manifests_are_listed_from_projections_without_body_reads(object_db, monkeypatch):
    from skynet_app.payload_store import PROJECTED_PATHS
    from test_document_projections import ADAPTER_MANIFEST, compact
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    adapter = db.create_adapter(name='Compact', manifest=ADAPTER_MANIFEST)
    version = adapter['latest_version']['id']
    with db.connection() as c:
        marker = c.execute("SELECT to_jsonb(a) AS document FROM adapters a WHERE id=?", (version,)).fetchone()[0]['manifest_json']
        assert marker.startswith('{"$skynet_object_v1"'), 'the manifest is offloaded'
        projected = c.execute("SELECT projection_json FROM document_projections WHERE table_name='adapters' AND record_id=?", (version,)).fetchone()[0]
    assert projected == {'manifest': compact(ADAPTER_MANIFEST)}
    reads = track_payload_reads(db, monkeypatch)
    original_read_many = db.payload_store.objects.read_many
    monkeypatch.setattr(db.payload_store.objects, 'read_many', lambda references: pytest.fail('the list must not read manifest bodies') if references else [])
    listed = next(item for item in db.list_adapter_registry(manifests='projected') if item['id'] == adapter['id'])
    assert listed['latest_version']['manifest'] == compact(ADAPTER_MANIFEST) and reads == []
    assert next(item for item in db.list_adapter_registry(manifests='none') if item['id'] == adapter['id'])['latest_version']['manifest'] == {}
    # A version written before migration 023 has no projection: the list fails closed, the repair reads the body once.
    with db.transaction() as c:
        c.execute("DELETE FROM document_projections WHERE table_name='adapters' AND record_id=?", (version,))
    with pytest.raises(ValueError, match='incomplete'):
        db.list_adapter_registry(manifests='projected')
    fetched = []
    def read_many(references):
        fetched.append([ref['sha256'] for ref in references])
        return original_read_many(references)
    monkeypatch.setattr(db.payload_store.objects, 'read_many', read_many)
    assert PROJECTED_PATHS['adapters'] == () and db.repair_document_projections() == 1
    assert [refs for refs in fetched if refs] == [[json.loads(marker)['$skynet_object_v1']['sha256']]], 'one body read, for the one missing projection'
    assert next(item for item in db.list_adapter_registry(manifests='projected') if item['id'] == adapter['id'])['latest_version']['manifest'] == compact(ADAPTER_MANIFEST)
    assert db.get_adapter(adapter['id'])['latest_version']['manifest'] == ADAPTER_MANIFEST
