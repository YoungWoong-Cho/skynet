import json

import pytest

from skynet_app import pipeline_api
from skynet_app.database import Database
from test_pipeline import GatewayRecoveryCluster, canonical_spec, make_pipeline_service
from test_tracking_reconciliation import completed_tracking, submitted


@pytest.mark.parametrize('tracking_outage', [False, True])
def test_recovered_receipt_also_repairs_tracking_without_resubmission(tmp_path, monkeypatch, tracking_outage):
    db = Database(tmp_path / 'recover.db')
    cluster = GatewayRecoveryCluster()
    service = make_pipeline_service(db, cluster)
    experiment = service.create_experiment(canonical_spec())
    service.submit_experiment(experiment['id'])
    run_id = experiment['runs'][0]['id']
    calls = []
    def start(spec, run, capsule, job_id, **kwargs):
        calls.append(job_id)
        if tracking_outage:
            raise ConnectionError('temporary tracking outage')
        for scope, identifier in [('run', run_id), ('experiment', run['experiment_id'])]:
            db.upsert_tracking_binding('wandb', scope, identifier, remote_id=None, remote_url=None, status='QUEUED')
    monkeypatch.setattr(service, '_active_tracking_providers', lambda _: [{'provider': 'wandb'}])
    monkeypatch.setattr(service, '_start_tracking', start)
    result = service.recover_run_submission(run_id, 'sky2')
    assert result['slurm_job_id'] == '9004'
    assert calls == ['9004']
    assert db.list_tracking_bindings('run', run_id)[0]['status'] == ('ERROR' if tracking_outage else 'QUEUED')
    assert db.get_run(run_id)['status'] == 'SUBMITTED'
    if not tracking_outage:
        service.recover_run_submission(run_id, 'sky2')
        assert calls == ['9004']
        assert cluster.submit_count == 2
        assert len(db.get_run(run_id)['attempts']) == 1


def test_terminal_run_with_zero_bindings_is_recovered_and_finalized_once(tmp_path, monkeypatch):
    import test_tracking_reconciliation as fixtures
    monkeypatch.setattr(pipeline_api.PipelineService, '_validate_tracking_requirements', lambda *_: None)
    original = fixtures.canonical_spec
    def tracked_spec():
        spec = original()
        spec['tracking'] = {'providers': [{'provider': 'wandb', 'enabled': True, 'entity': 'team', 'project': 'tracking-test'}]}
        return spec
    monkeypatch.setattr(fixtures, 'canonical_spec', tracked_spec)
    f = completed_tracking(tmp_path, monkeypatch, initialize=False)
    run = f.db.get_run(f.run_id)
    with f.db.transaction() as c:
        c.execute("DELETE FROM tracking_bindings WHERE scope_type='run' AND scope_id=?", (f.run_id,))
    assert f.db.list_tracking_bindings('run', f.run_id) == []
    f.service.reconcile_tracking()
    f.service.reconcile_tracking()
    f.service.reconcile_tracking()
    binding = f.db.list_tracking_bindings('run', f.run_id)[0]
    assert binding['status'] == 'FINISHED'
    assert binding['remote_url']
    assert len(f.remote) == 1
    assert [r['_step'] for r in f.history if 'train_loss' in r] == [1000]
    assert len([e for e in f.bridge()._events_unlocked() if e['operation'] == 'finish_run']) == 1


def test_batch_progress_is_atomic_idempotent_and_attempt_scoped(tmp_path, monkeypatch):
    db, cluster, service, run_id = submitted(tmp_path, monkeypatch)
    run = db.get_run(run_id)
    attempt_id = run['attempts'][0]['id']
    rows = [dict(restart_count=0, completed=i, total=8000, source_kind='jsonl', evidence={'metrics': {'loss': i / 8000}})
            for i in range(20, 8001, 20)]
    db.record_training_progress_samples(run_id, attempt_id, rows)
    originals = db.list_training_progress_samples(run_id)
    db.record_training_progress_samples(run_id, attempt_id, rows)
    assert db.list_training_progress_samples(run_id) == originals
    assert len(originals) == 400 and max(r['completed'] for r in originals) == 8000
    with pytest.raises(ValueError):
        db.record_training_progress_samples(run_id, attempt_id, [dict(rows[0], restart_count=1), dict(rows[1], completed=-1)])
    assert db.list_training_progress_samples(run_id) == originals
    other = db.create_run(run['variant_id'], seed=99, adapter_name='generic', adapter_version='1', run_directory='/other')
    with pytest.raises(KeyError):
        db.record_training_progress_samples(other['id'], attempt_id, rows)


def test_successful_empty_tail_is_retried_and_never_invents_final_progress(tmp_path):
    from types import SimpleNamespace
    from test_progress_refresh import create_run
    from skynet_app.adapters.act_manifest import manifest
    db = Database(tmp_path / 'empty-final.db')
    run = create_run(db)
    contract = manifest().train.progress.model_dump(mode='json')
    stage = db.create_stage(run['id'], stage_type='TRAIN', name='train',
                           resolved_config={'plan': {'progress': contract}})
    attempt = db.create_job_attempt(stage['id'], status='SUCCEEDED', slurm_job_id='123')
    db.update_run(run['id'], status='SUCCEEDED')
    service = pipeline_api.PipelineService.__new__(pipeline_api.PipelineService)
    service.database = db
    tail = {'text': ''}
    service.cluster = SimpleNamespace(read_log=lambda *a, **kw: ('test', tail['text']))
    value = db.get_run(run['id'])
    value['resolved_spec_json'] = {'native': {'config': {'epochs': 10}}}
    assert service._ingest_training_progress(value) == 0
    assert not getattr(service, '_training_progress_final_reads', set())
    assert db.list_training_progress_samples(run['id']) == []
    tail['text'] = json.dumps({'epoch': 9, 'train_loss': 0.2}) + '\n'
    service._training_progress_final_failures = {}
    assert service._ingest_training_progress(value) == 1
    assert db.list_training_progress_samples(run['id'])[0]['completed'] == 10


def test_progress_cache_reuses_only_immutable_spec_and_reads_new_samples(tmp_path, monkeypatch):
    from skynet_app.db_backend import PostgresConnection
    db, cluster, service, run_id = submitted(tmp_path, monkeypatch)
    run = db.get_run(run_id)
    first = db.run_progress_evidence([run_id])
    original = PostgresConnection.execute
    def no_reparse(connection, sql, parameters=None):
        assert 'selected_specs AS MATERIALIZED' not in sql
        return original(connection, sql, parameters)
    monkeypatch.setattr(PostgresConnection, 'execute', no_reparse)
    db.record_training_progress_sample(run_id, run['attempts'][0]['id'], restart_count=0,
                                      completed=1, total=1, source_kind='jsonl')
    db.update_job_attempt(run['attempts'][0]['id'], status='SUCCEEDED')
    second = db.run_progress_evidence([run_id])
    assert second[run_id]['resolved_spec_json'] == first[run_id]['resolved_spec_json']
    assert second[run_id]['attempts'][0]['status'] == 'SUCCEEDED'
    assert second[run_id]['progress_samples'][0]['completed'] == 1
    assert first[run_id]['progress_samples'] == []


def test_provider_flush_does_not_touch_unselected_runs(tmp_path, monkeypatch):
    f = completed_tracking(tmp_path, monkeypatch)
    existing = f.db.list_tracking_bindings('run', f.run_id)[0]
    f.db.upsert_tracking_binding('wandb', 'run', f.run_id,
        remote_id=existing['remote_id'], remote_url=existing['remote_url'],
        metadata=existing['metadata_json'], status='QUEUED')
    monkeypatch.setattr(f.service, '_wandb_bridge', lambda *_: pytest.fail('Excluded run was contacted'))
    result = f.service._flush_tracking_provider('wandb', run_ids=set())
    assert result['attempted'] == result['delivered'] == 0
    assert f.db.list_tracking_bindings('run', f.run_id)[0]['status'] == 'QUEUED'
