"""Real SQLite/session isolation; cluster side effects are forbidden in these tests."""
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from skynet_app.db_backend import INTEGRITY_ERRORS
from skynet_app.database import Database
from skynet_app.pipeline_api import PipelineService, router
from skynet_app.source_metadata_cache import SourceMetadataStore
from skynet_app.tracking import SessionCredentialStore
from skynet_app.workspaces import (
    COOKIE, CURRENT_WORKSPACE, WorkspaceDirectory, WorkspaceMiddleware,
    WorkspaceServices, session_router,
)


@pytest.fixture
def services(tmp_path, monkeypatch):
    database = Database(tmp_path / 'workspace.db')
    coordinator = WorkspaceServices(PipelineService(
        database, cluster=Mock(), credential_store=Mock(load=lambda _: None),
        session_credentials=SessionCredentialStore(),
    ))
    coordinator.directory.claim_legacy('ycho420@gatech.edu')
    monkeypatch.setattr('skynet_app.pipeline_api.service', coordinator)
    return coordinator


def open_db(services, email):
    workspace, _ = services.directory.open(email)
    return services.for_workspace(workspace['id']).database


def graph(database):
    project = database.create_project('same name')
    experiment = database.create_experiment(project_id=project['id'], name='same experiment', requested_spec={})
    revision = experiment['latest_revision']
    variant = database.create_variant(revision['id'], name='v1', parameters={}, resolved_spec={})
    run = database.create_run(variant['id'], seed=42, adapter_name='hpt', adapter_version='2', run_directory='/test/' + variant['id'])
    stage = database.create_stage(run['id'], stage_type='TRAIN', name='train')
    attempt = database.create_job_attempt(stage['id'], slurm_job_id='123', status='RUNNING')
    checkpoint = database.create_checkpoint(run['id'], checkpoint_type='INFERENCE', path='/test/checkpoint', sha256='a'*64, size_bytes=123)
    evaluation = database.create_evaluation(run['id'], checkpoint_id=checkpoint['id'], evaluator_adapter='hpt', evaluator_version='2', suite_name='cube', suite_version='1', tasks=['cube'], seeds=[42], episodes_per_task=1)
    episode = database.upsert_evaluation_episode(evaluation['id'], task='cube', seed=42, episode_index=0, status='SUCCEEDED', success=True)
    database.record_metric(run['id'], name='loss', value=1.0, scope='train', step=1)
    database.upsert_tracking_binding('wandb', 'run', run['id'], remote_id='remote-run', remote_url='https://wandb.ai/test/run', status='connected')
    return dict(project=project, experiment=experiment, revision=revision, variant=variant, run=run, stage=stage, attempt=attempt, checkpoint=checkpoint, evaluation=evaluation, episode=episode)


def clients(services):
    app = FastAPI()
    app.add_middleware(WorkspaceMiddleware, services=services)
    app.include_router(session_router(services.directory))
    app.include_router(router)
    return TestClient(app), TestClient(app)


def test_sessions_normalize_resume_revoke_and_expire(services):
    directory = services.directory
    first, token = directory.open('  YCHO420@GATECH.EDU  ')
    assert first == {'id': 'legacy', 'email': 'ycho420@gatech.edu'}
    assert directory.resolve(token) == first
    again, second = directory.open('ycho420@gatech.edu')
    assert again == first and second != token
    with directory.database.connection() as connection:
        assert not connection.execute('SELECT 1 FROM workspace_sessions WHERE token_hash=?', (token,)).fetchone()
    directory.close(token)
    assert directory.resolve(token) is None
    with directory.database.transaction() as connection:
        connection.execute('UPDATE workspace_sessions SET expires_at=?', (int(time.time())-1,))
    assert directory.resolve(second) is None
    with pytest.raises(ValueError):
        directory.open('invalid')
    with pytest.raises(ValueError):
        directory.claim_legacy('other@example.com')


def test_same_email_concurrent_logins_share_one_workspace(services):
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(services.directory.open, ['alice@example.com']*8))
    assert len({record[0]['id'] for record in records}) == 1
    assert len({record[1] for record in records}) == 8


def test_personal_graphs_and_writes_are_isolated(services):
    alice, bob = (open_db(services, email) for email in ('alice@example.com', 'bob@example.com'))
    a, b = graph(alice), graph(bob)
    assert [r['id'] for r in alice.list_runs(limit=1)] == [a['run']['id']]
    assert [r['id'] for r in bob.list_runs(limit=1)] == [b['run']['id']]
    assert len(alice.list_experiments()) == len(bob.list_experiments()) == 1
    assert len(alice.list_evaluations()) == len(bob.list_evaluations()) == 1
    assert bob.get_run(a['run']['id']) is None
    assert bob.get_evaluation(a['evaluation']['id']) is None
    assert bob.list_metrics(a['run']['id']) == []
    assert bob.list_tracking_bindings('run', a['run']['id']) == []
    with pytest.raises(KeyError):
        bob.update_run(a['run']['id'], status='CANCELLED')
    with pytest.raises(KeyError):
        bob.upsert_tracking_binding('wandb', 'run', a['run']['id'], remote_id='bad', remote_url=None, status='connected')
    with bob.transaction() as connection, pytest.raises(INTEGRITY_ERRORS):
        connection.execute("UPDATE runs SET status='CANCELLED' WHERE id=?", (a['run']['id'],))
    assert alice.get_run(a['run']['id'])['status'] != 'CANCELLED'
    assert len(alice.get_evaluation(a['evaluation']['id'])['episodes']) == 1
    with services.system.database.connection() as connection:
        if getattr(connection, "dialect", "sqlite") == "postgresql":
            assert connection.execute("SELECT conname FROM pg_constraint WHERE contype='f' AND NOT convalidated").fetchall() == []
        else:
            assert connection.execute('PRAGMA foreign_key_check').fetchall() == []


def test_saved_settings_and_shared_adapters(services, monkeypatch):
    alice, bob = (open_db(services, email) for email in ('alice@example.com', 'bob@example.com'))
    alice.upsert_tracking_connection('wandb', endpoint='https://api.wandb.ai', workspace='alice')
    assert bob.get_tracking_connection('wandb') is None
    bob.upsert_tracking_connection('wandb', endpoint='https://api.wandb.ai', workspace='bob')
    assert alice.get_tracking_connection('wandb')['workspace'] == 'alice'
    bob.delete_tracking_connection('wandb')
    assert alice.get_tracking_connection('wandb')['workspace'] == 'alice'
    sa, sb = SourceMetadataStore(alice), SourceMetadataStore(bob)
    sa.save_selection('https://github.com/example/repo', branch='main', commit='a'*40)
    assert sb.get_selection('https://github.com/example/repo')['branch'] == ''
    sb.save_selection('https://github.com/example/repo', branch='other', commit='b'*40)
    assert sa.get_selection('https://github.com/example/repo')['branch'] == 'main'
    builtin = alice.list_adapter_registry()[0]
    assert bob.get_adapter(builtin['id']) is not None
    with pytest.raises(KeyError):
        alice.archive_adapter(builtin['id'])
    alice.create_adapter(name='My adapter', manifest={})
    bob.create_adapter(name='My adapter', manifest={})
    assert len([a for a in alice.list_adapter_registry() if a['name'] == 'My adapter']) == 1
    alice.record_adapter_validation(builtin['id'], status='PASSED', evidence={'private': 'alice'})
    assert bob.list_adapter_validations(builtin['id']) == []
    service_a, service_b = (services.for_workspace(db.workspace_id) for db in (alice, bob))
    assert service_a.credentials is not service_b.credentials
    assert service_a.credential_store.service_name != service_b.credential_store.service_name
    monkeypatch.setenv('WANDB_API_KEY', 'legacy-only')
    assert service_a._environment_credentials('wandb') == {}
    assert services.for_workspace('legacy')._environment_credentials('wandb') == {'api_key': 'legacy-only'}


def test_api_requires_email_and_rejects_cross_workspace_actions(services):
    a, b = clients(services)
    assert a.get('/api/experiments').status_code == 401
    assert a.get('/api/workspace/session').json() == {'workspace': None}
    assert a.post('/api/workspace/session', json={'email': 'invalid'}).status_code == 422
    login = a.post('/api/workspace/session', json={'email': 'alice@example.com'})
    assert login.status_code == 200
    assert 'HttpOnly' in login.headers['set-cookie'] and 'SameSite=lax' in login.headers['set-cookie']
    a_id = login.json()['workspace']['id']
    b.post('/api/workspace/session', json={'email': 'bob@example.com'})
    # This test exercises ownership after both users completed storage setup.
    with services.system.database.transaction() as connection:
        for row in connection.execute("SELECT id FROM workspaces WHERE email IN ('alice@example.com','bob@example.com')").fetchall():
            connection.execute("INSERT INTO workspace_storage VALUES (?,?)", (row[0], '/team/' + row[0]))
    owned = graph(services.for_workspace(a_id).database)
    listing = a.get('/api/experiments')
    assert listing.headers['cache-control'] == 'private, no-store'
    assert len(listing.json()['experiments']) == 1
    assert b.get('/api/experiments').json()['experiments'] == []
    paths = [f"/api/experiments/{owned['experiment']['id']}", f"/api/runs/{owned['run']['id']}", f"/api/runs/{owned['run']['id']}/logs", f"/api/evaluations/{owned['evaluation']['id']}"]
    for path in paths:
        assert b.get(path).status_code == 404, path
    for path in [f"/api/runs/{owned['run']['id']}/cancel", f"/api/evaluations/{owned['evaluation']['id']}/cancel"]:
        assert b.post(path).status_code == 404, path
    assert b.post('/api/evaluations', json={'run_id': owned['run']['id']}).status_code == 404
    assert b.get('/api/runs', params={'experiment_revision_id': owned['revision']['id']}).status_code == 404
    assert services.system.cluster.mock_calls == []
    assert CURRENT_WORKSPACE.get() is None
    # Another tab now has Bob's cookie, but Alice's page still sends her ID.
    switched = a.post('/api/workspace/session', json={'email': 'bob@example.com'})
    assert switched.status_code == 200
    assert a.get('/api/experiments', headers={'X-Skynet-Workspace': a_id}).status_code == 409
    assert a.delete('/api/workspace/session', headers={'X-Skynet-Workspace': a_id}).status_code == 409
    old_token = a.cookies.get(COOKIE)
    assert a.delete('/api/workspace/session').status_code == 200
    assert services.directory.resolve(old_token) is None
    assert a.get('/api/experiments').status_code == 401
    assert a.post('/api/workspace/session', json={'email': 'eve@example.com'}, headers={'Origin': 'https://another-site.example'}).status_code == 403


def test_legacy_schema_migration_preserves_records_and_claims_configured_owner(tmp_path):
    from skynet_app.database import SCHEMA
    path = tmp_path / 'legacy.db'
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA)
        connection.execute("INSERT INTO projects(id,name,description,created_at) VALUES ('saved','Existing project','Keep me','2026-01-01')")
        connection.execute("INSERT INTO tracking_connections(provider,endpoint,workspace,config_json,updated_at) VALUES ('wandb','https://api.wandb.ai','existing','{}','2026-01-01')")
    path.with_name('workspace-owner.json').write_text('{"email":"ycho420@gatech.edu"}')
    system = PipelineService(Database(path), cluster=Mock(), credential_store=Mock(load=lambda _: None))
    manager = WorkspaceServices(system)
    owner = open_db(manager, 'YCHO420@gatech.edu')
    other = open_db(manager, 'teammate@example.com')
    assert owner.list_projects()[0]['id'] == 'saved'
    assert owner.get_tracking_connection('wandb')['workspace'] == 'existing'
    assert other.list_projects() == []
    assert other.get_tracking_connection('wandb') is None
    reopened = Database(path)
    assert reopened.list_projects()[0]['description'] == 'Keep me'
    assert WorkspaceDirectory(reopened).open('ycho420@gatech.edu')[0]['id'] == 'legacy'


def test_background_reconciliation_selects_only_its_own_jobs(services, monkeypatch):
    alice, bob = (open_db(services, email) for email in ('alice@example.com', 'bob@example.com'))
    a, b = graph(alice), graph(bob)
    alice.update_job_attempt(a['attempt']['id'], slurm_job_id='111')
    bob.update_job_attempt(b['attempt']['id'], slurm_job_id='222')
    service = services.for_workspace(alice.workspace_id)
    monkeypatch.setattr(service, '_flush_tracking_provider', lambda *a, **k: None)
    monkeypatch.setattr(service, '_repair_missing_active_tracking_bindings', lambda rows: 0)
    # Stop at the actual external boundary, after real owner-filtered SQL runs.
    services.system.cluster.job_statuses.side_effect = RuntimeError('external boundary')
    token = CURRENT_WORKSPACE.set(bob.workspace_id)
    try:
        with pytest.raises(RuntimeError, match='external boundary'):
            service.reconcile()
    finally:
        CURRENT_WORKSPACE.reset(token)
    services.system.cluster.job_statuses.assert_called_once_with(['111'])
    assert bob.get_run(b['run']['id'])['status'] == b['run']['status']
    assert bob.run_progress_evidence([a['run']['id']]) == {}
    assert bob.evaluation_progress_evidence([a['evaluation']['id']]) == {}
