"""Separate evaluation targets are retained by their stage's frozen context or plan."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from skynet_app import data_selection, prepared_deletion
from skynet_app.database import Database, canonical_json
from skynet_app.workspaces import WorkspaceDirectory
from test_data_version_retirement import migration


def dataset(db):
    parent = db.create_data_resource(category='dataset', provider='filesystem', namespace='test', source_key='eval-only', kind='dataset')
    version = db.create_data_resource_version(parent['id'], revision='1', format='skynet.recording-dataset/v1',
        path='/target-dataset', manifest_sha256='d'*64)
    db.record_data_location(version['id'], kind='cluster', host='skynet', path=version['path'],
                            manifest_sha256=version['manifest_sha256'])
    return parent, version


def evaluation_run(db, name='Evaluate held-out hand'):
    experiment = db.create_experiment(name=name, requested_spec={})
    variant = db.create_variant(experiment['latest_revision']['id'], name='run', parameters={}, resolved_spec={})
    run = db.create_run(variant['id'], seed=1, adapter_name='human-policy-hat', adapter_version='1', run_directory='/run', status='COMPLETED')
    return experiment, run


def evaluation_reference(db, target, *, kind='context', name='Evaluate held-out hand'):
    experiment, run = evaluation_run(db, name)
    receipt = dict(version_id=target['id'], manifest_sha256=target['manifest_sha256'], path=target['path'], metadata={})
    config = {'context': {'target_dataset': receipt}} if kind == 'context' else {}
    plan = {'native_config': {'canonical_evaluation': {'target_dataset': receipt}}}
    if kind == 'plan':
        config['plan'] = plan
    stage = db.create_stage(run['id'], stage_type='EVALUATE', name='eval', status='COMPLETED', resolved_config=config)
    if kind == 'attempt':
        db.create_job_attempt(stage['id'], status='COMPLETED', execution_snapshot_json={'plan': plan})
    return experiment, run, stage


@pytest.mark.parametrize('kind', ['context', 'plan'])
def test_eval_only_target_usage_prevents_all_cleanup_until_history_is_removed(tmp_path, kind):
    db = Database(tmp_path/'targets')
    parent, target = dataset(db)
    experiment, run, stage = evaluation_reference(db, target, kind=kind)
    usages = db.data_version_usage(target['manifest_sha256'])
    assert usages == [dict(experiment_id=experiment['id'], name=experiment['name'], revision_number=1,
                           run_id=run['id'], run_status='COMPLETED')]
    service = SimpleNamespace(database=db, root=tmp_path)
    preview = prepared_deletion.preview(service, SimpleNamespace(owns=lambda *_: True), 'dataset', target['id'])
    assert preview['blockers'][0]['id'] == experiment['id']
    calls = []
    with pytest.raises(ValueError, match='used by an experiment'):
        db.delete_prepared_dataset(parent['id'], lambda *args: calls.append(args), version_id=target['id'])
    assert calls == [] and db.get_dataset(target['id'])
    # Dataset presence is independent of stage success/failure or training inputs.
    db.update_stage(stage['id'], status='CANCELLED')
    with pytest.raises(ValueError, match='used by an experiment'):
        db.delete_prepared_dataset(parent['id'], lambda *args: calls.append(args), version_id=target['id'])
    with db.transaction() as connection:
        connection.execute('DELETE FROM workflow_stages WHERE id=?', (stage['id'],))
    assert db.data_version_usage(target['manifest_sha256']) == []
    assert db.delete_prepared_dataset(parent['id'], lambda *args: calls.append(args), version_id=target['id'])['deleted']
    assert len(calls) == 1


def test_evaluation_usage_redacts_other_workspaces_and_deduplicates_snapshot_copies(tmp_path):
    db = Database(tmp_path/'privacy')
    _, target = dataset(db)
    directory = WorkspaceDirectory(db)
    alice = db.for_workspace(directory.open('alice@example.com')[0]['id'])
    bob = db.for_workspace(directory.open('bob@example.com')[0]['id'])
    own, run, stage = evaluation_reference(alice, target)
    target_receipt = stage['resolved_config_json']['context']['target_dataset']
    plan = {'native_config': {'canonical_evaluation': {'target_dataset': target_receipt}}}
    alice.update_stage(stage['id'], resolved_config_json={**stage['resolved_config_json'], 'plan': plan})
    alice.create_job_attempt(stage['id'], status='COMPLETED', execution_snapshot_json={'plan': plan})
    evaluation_reference(bob, target, name='Private evaluation name')
    rows = db.data_version_usage_many([target['manifest_sha256']], workspace_id=alice.workspace_id)[target['manifest_sha256']]
    assert rows == [dict(experiment_id=own['id'], name=own['name'], revision_number=1, run_id=run['id'], run_status='COMPLETED'),
                    {'other_workspace': True}]
    assert 'Private' not in canonical_json(rows)


@pytest.mark.parametrize('kind', ['context', 'plan'])
def test_retirement_cannot_delete_an_evaluation_target_even_after_training_head_rebound(migration, kind):
    m = migration
    evaluation_reference(m.db, m.old, kind=kind)
    with pytest.raises(ValueError, match='evaluation.*references'):
        m.retirement.preview(m.old['id'], m.new['id'])
    assert not m.calls


def test_resume_availability_checks_frozen_target_not_just_training_data(migration):
    m = migration
    plan = m.retirement.preview(m.old['id'], m.new['id'])
    m.retirement.retire(m.old['id'], m.new['id'], plan['token'])
    config = {'context': {'target_dataset': {'version_id': m.old['id'], 'manifest_sha256': m.old['manifest_sha256']}}}
    with pytest.raises(ValueError, match='retired converted data'):
        data_selection.assert_available(m.db, {'data': {}}, config)


@pytest.mark.parametrize('change', ['archived', 'removed', 'deleted', 'hash', 'path'])
def test_initial_evaluation_reference_rechecks_target_before_creating_stage(tmp_path, change):
    db = Database(tmp_path/'atomic')
    parent, target = dataset(db)
    _, run = evaluation_run(db)
    receipt = dict(version_id=target['id'], manifest_sha256=target['manifest_sha256'], path=target['path'])
    if change == 'archived':
        db.update_dataset(target['id'], archived=True)
    elif change == 'removed':
        db.record_data_location(target['id'], kind='cluster', host='skynet', path=target['path'],
                                manifest_sha256=target['manifest_sha256'], status='REMOVED')
    elif change == 'deleted':
        db.delete_prepared_dataset(parent['id'], lambda *_: None, version_id=target['id'])
    elif change == 'hash':
        receipt['manifest_sha256'] = 'e'*64
    elif change == 'path':
        receipt['path'] = '/unverified-copy'
    with pytest.raises(ValueError, match='Evaluation target dataset changed'):
        db.create_stage(run['id'], stage_type='EVALUATE', name='eval', status='BLOCKED',
                        resolved_config={'context': {'target_dataset': receipt}})
    assert db.list_stages(run['id']) == []


def test_initial_evaluation_reference_cannot_race_dataset_cleanup(tmp_path):
    db = Database(tmp_path/'cleanup-race')
    parent, target = dataset(db)
    _, run = evaluation_run(db)
    # An independent repository instance has a different in-process lock.
    # PostgreSQL must serialize the delete and reference creation themselves.
    other = Database(url=db.url, data_root=db.data_root)
    cleanup_started, release_cleanup, creation_started = Event(), Event(), Event()
    receipt = dict(version_id=target['id'], manifest_sha256=target['manifest_sha256'], path=target['path'])

    def cleanup(*_):
        cleanup_started.set()
        assert release_cleanup.wait(5), 'test did not release cleanup'

    def create():
        creation_started.set()
        return other.create_stage(run['id'], stage_type='EVALUATE', name='eval', status='BLOCKED',
                                  resolved_config={'context': {'target_dataset': receipt}})

    with ThreadPoolExecutor(max_workers=2) as pool:
        deleting = pool.submit(db.delete_prepared_dataset, parent['id'], cleanup, version_id=target['id'])
        assert cleanup_started.wait(5), 'cleanup never reached the locked transaction'
        creating = pool.submit(create)
        assert creation_started.wait(5)
        release_cleanup.set()
        assert deleting.result(timeout=5)['deleted']
        with pytest.raises(ValueError, match='Evaluation target dataset changed'):
            creating.result(timeout=5)
    assert db.list_stages(run['id']) == []
