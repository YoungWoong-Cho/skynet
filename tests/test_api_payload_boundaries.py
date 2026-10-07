"""Large immutable training receipts must not leak into list/metric hot paths."""
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from factories import make_run_chain
from test_payload_store import object_db
from test_postgres import pg
from skynet_app.db_backend import lock_key
from skynet_app.payload_store import PayloadStore, _CACHE, _PROJECTIONS
from skynet_app import pipeline_api


HAT_RUN = dict(project_name='payload boundaries', experiment_name='large', adapter_name='hat',
               adapter_version='fixed', run_directory='/run')


def test_evaluation_lists_never_read_execution_bodies(object_db, monkeypatch):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    run = make_run_chain(db, **HAT_RUN).run
    stage = db.create_stage(run['id'], stage_type='EVALUATE', name='eval', resolved_config={
        'resources': {'gpu': {'count': 1}}, 'unused': 'x' * 2_000_000})
    attempt = db.create_job_attempt(stage['id'], execution_snapshot_json={'code': 'y' * 2_000_000})
    evaluation = db.create_evaluation(run['id'], stage_id=stage['id'], evaluator_adapter='hat',
        evaluator_version='fixed', suite_name='cube', suite_version='1', tasks=['cube'], seeds=[1], episodes_per_task=1)
    _CACHE.clear(); _PROJECTIONS.clear()
    def forbidden(*args, **kwargs):
        raise AssertionError('List tried to fetch a full immutable body')
    monkeypatch.setattr(db.payload_store.objects, 'read', forbidden)
    monkeypatch.setattr(db.payload_store.objects, 'read_many', lambda refs: [] if not refs else forbidden())
    rows = db.list_evaluations()
    pipeline_api._attach_evaluation_progress_summaries(db, rows)
    assert rows[0]['resources'] == {'gpu': {'count': 1}}
    assert rows[0]['latest_attempt']['id'] == attempt['id']
    assert len(json.dumps(rows)) < 10_000
    assert rows[0]['id'] == evaluation['id']
    service = pipeline_api.PipelineService.__new__(pipeline_api.PipelineService)
    service.database = db
    assert 'cancel' in service.evaluation_manual_actions(evaluation)


@pytest.mark.parametrize('operation', ['attempt', 'stage', 'transition', 'mismatch'])
def test_status_writes_never_read_unchanged_execution_bodies(object_db, monkeypatch, operation):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    run = make_run_chain(db, **HAT_RUN).run
    stage = db.create_stage(run['id'], stage_type='TRAIN', name='train', resolved_config={'body': 'x' * 200_000})
    attempt = db.create_job_attempt(stage['id'], execution_snapshot_json={'body': 'y' * 200_000})
    _CACHE.clear()
    original = db.payload_store.objects.read
    calls = []
    def read(path, digest):
        with db.connection() as c:
            assert c.execute('SELECT pg_try_advisory_xact_lock(?)', (lock_key('repository-write'),)).fetchone()[0]
        calls.append(digest)
        return original(path, digest)
    monkeypatch.setattr(db.payload_store.objects, 'read', read)
    if operation == 'attempt':
        result = db.update_job_attempt(attempt['id'], status='RUNNING')
        assert 'execution_snapshot_json' not in result
        assert result['status'] == 'RUNNING'
    elif operation == 'stage':
        result = db.update_stage(stage['id'], status='RUNNING')
        assert 'resolved_config_json' not in result
        assert result['status'] == 'RUNNING'
    else:
        result = db.transition_workflow_state(stage_id=stage['id'], stage_updates={'status': 'RUNNING'},
            attempt_id=attempt['id'], attempt_updates={'status': 'RUNNING'},
            expected_stage_statuses=['FAILED' if operation == 'mismatch' else 'PENDING'])
        assert result['applied'] is (operation != 'mismatch')
        assert result['stage']['id'] == stage['id']
        assert 'resolved_config_json' not in result['stage']
        if 'attempt' in result:
            assert 'execution_snapshot_json' not in result['attempt']
    assert not calls


def test_experiments_summary_does_not_load_run_lists(object_db, monkeypatch):
    db, _ = object_db
    experiment = make_run_chain(db, **HAT_RUN).experiment
    monkeypatch.setattr(pipeline_api, 'service', SimpleNamespace(database=db))
    monkeypatch.setattr(db, 'list_runs', lambda **kw: pytest.fail('N+1 run listing'))
    row = pipeline_api.list_experiments()['experiments'][0]
    assert row['run_count'] == row['variant_count'] == 1
    assert row['adapter'] == 'hat'
    db.create_experiment_revision(experiment['id'], requested_spec={})
    row = pipeline_api.list_experiments()['experiments'][0]
    assert row['run_count'] == row['variant_count'] == 0
    assert row['adapter'] is None


def test_display_projection_keeps_identity_schema_and_original_receipt():
    source = {'manifest_sha256': 'fixed', 'metadata': {'episodes': [{'x': 'x' * 200_000}],
              'shared_artifacts': ['big'], 'action_dim': 21},
              'train': {'capsule_files': {'train.py': 'y' * 200_000}, 'input_fields': [{'id': 'batch_size'}]}}
    before = json.dumps(source)
    result = pipeline_api._display_payload(source)
    assert result['manifest_sha256'] == 'fixed'
    assert result['metadata'] == {'episodes': 1, 'action_dim': 21}
    assert result['train']['input_fields'] == [{'id': 'batch_size'}]
    assert len(json.dumps(result)) < 500
    assert json.dumps(source) == before


def test_capsule_upload_does_not_hold_repository_write_lock(object_db, monkeypatch):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    run = make_run_chain(db, **HAT_RUN).run
    original = db.payload_store.objects.put_many
    uploads = []
    def upload(contents):
        if contents:
            with db.connection() as c:
                assert c.execute('SELECT pg_try_advisory_xact_lock(?)', (lock_key('repository-write'),)).fetchone()[0]
            uploads.append(len(contents))
        return original(contents)
    monkeypatch.setattr(db.payload_store.objects, 'put_many', upload)
    stage = db.create_stage(run['id'], stage_type='TRAIN', name='train', resolved_config={'large': 'x' * 1_000_000})
    attempt = db.claim_stage_and_create_job_attempt(stage['id'], execution_snapshot_json={'large': 'y' * 1_000_000})
    assert attempt['execution_snapshot_json']['large'].startswith('y')
    assert len(uploads) == 2


def test_failed_preupload_cannot_claim_submission(object_db, monkeypatch):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    run = make_run_chain(db, **HAT_RUN).run
    stage = db.create_stage(run['id'], stage_type='TRAIN', name='train')
    monkeypatch.setattr(db.payload_store.objects, 'put_many', lambda *_: (_ for _ in ()).throw(ConnectionError('offline')))
    with pytest.raises(ConnectionError):
        db.claim_stage_and_create_job_attempt(stage['id'], execution_snapshot_json={'large': 'x' * 1_000_000})
    assert db.list_job_attempts(stage_id=stage['id']) == []
    assert db.list_stages(run['id'])[0]['status'] == 'PENDING'


@pytest.mark.parametrize('operation', ['edit', 'archive', 'restore', 'clone'])
def test_registry_body_reads_are_outside_write_lock(object_db, monkeypatch, operation):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    adapter = db.create_adapter(name='large', manifest={'body': 'x' * 200_000})
    _CACHE.clear()
    original = db.payload_store.objects.read
    def read(path, digest):
        with db.connection() as c:
            assert c.execute('SELECT pg_try_advisory_xact_lock(?)', (lock_key('repository-write'),)).fetchone()[0]
        return original(path, digest)
    monkeypatch.setattr(db.payload_store.objects, 'read', read)
    if operation == 'edit': result = db.edit_adapter(adapter['id'], manifest={'body': 'y' * 200_000})
    elif operation == 'archive': result = db.archive_adapter(adapter['id'])
    elif operation == 'restore': result = db.restore_adapter(adapter['id'])
    else: result = db.clone_adapter(adapter['id'], name='copy')
    assert result['latest_version']['manifest']['body']


def test_evaluation_planning_and_dispatch_read_only_required_execution_bodies(object_db, monkeypatch):
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    run = make_run_chain(db, **HAT_RUN).run
    old = db.create_stage(run['id'], stage_type='EVALUATE', name='old', status='SUBMITTED', stage_id='old-evaluation',
        resolved_config={'context': {'execution_key': 'old-evaluation'}, 'large': 'x' * 300_000})
    db.create_job_attempt(old['id'], execution_snapshot_json={'large': 'y' * 300_000})
    selected = db.create_stage(run['id'], stage_type='EVALUATE', name='selected', stage_id='selected-evaluation',
        resolved_config={'context': {'execution_key': 'selected-evaluation'}, 'large': 'z' * 300_000})
    attempt = db.create_job_attempt(selected['id'], execution_snapshot_json={'large': 'a' * 300_000})
    _CACHE.clear(); _PROJECTIONS.clear()
    calls=[]; original=db.payload_store.objects.read
    def read(path, digest):
        calls.append(digest)
        return original(path,digest)
    monkeypatch.setattr(db.payload_store.objects,'read',read)
    original_many=db.payload_store.objects.read_many
    def read_many(refs):
        calls.extend(ref['sha256'] for ref in refs)
        return original_many(refs)
    monkeypatch.setattr(db.payload_store.objects,'read_many',read_many)
    planned=db.get_run(run['id'],execution_stage_ids=[])
    assert not calls, 'Planning must never download any execution body'
    assert pipeline_api._evaluation_busy_reason(planned) is None
    assert planned['resolved_spec_json']=={}
    assert all('execution_snapshot_json' not in row for row in planned['attempts'])
    dispatched=db.get_run(run['id'],execution_stage_ids=[selected['id']])
    assert len(calls)==2, 'Only the selected stage and its attempt may be read'
    assert next(s for s in dispatched['stages'] if s['id']==selected['id'])['resolved_config_json']['large'].startswith('z')
    assert next(a for a in dispatched['attempts'] if a['id']==attempt['id'])['execution_snapshot_json']['large'].startswith('a')
    assert 'large' not in next(s for s in dispatched['stages'] if s['id']==old['id'])['resolved_config_json']


@pytest.mark.parametrize("selection", ["latest", "historical_id", "historical_number"])
def test_selected_adapter_reads_only_one_manifest(object_db, monkeypatch, selection):
    from test_payload_store import track_payload_reads
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    record = db.create_adapter(name="history", manifest={"body": "a" * 200_000})
    old = record["latest_version"]
    for letter in "bcde":
        record = db.edit_adapter(record["id"], manifest={"body": letter * 200_000})
    expected = record["latest_version"] if selection == "latest" else old
    kwargs = {"historical_id": {"version_id": old["id"]},
              "historical_number": {"version_number": 1}}.get(selection, {})
    reads = track_payload_reads(db, monkeypatch)
    actual = db.get_adapter_version(record["id"], **kwargs)
    assert actual["id"] == expected["id"]
    assert actual["manifest"] == expected["manifest"]
    assert reads == [expected["manifest_sha256"]]


def test_selected_adapter_rejects_foreign_version_and_mismatched_number(object_db, monkeypatch):
    from test_payload_store import track_payload_reads
    db, _ = object_db
    db.payload_store = PayloadStore(db)
    record = db.create_adapter(name="selected", manifest={"body": "a" * 200_000})
    other = db.create_adapter(name="other", manifest={"body": "b" * 200_000})
    reads = track_payload_reads(db, monkeypatch)
    assert db.get_adapter_version(record["id"], version_id=other["latest_version"]["id"]) is None
    assert db.get_adapter_version(record["id"], version_id=record["latest_version"]["id"], version_number=2) is None
    assert reads == []
