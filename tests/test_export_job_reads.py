"""Progress reads use current policies without scanning recordings or scheduling work."""

import copy

from fastapi import FastAPI
from fastapi.testclient import TestClient

from skynet_app.database import canonical_json
from skynet_app.db_backend import PostgresConnection
from test_policy_exports import setup, create


def test_job_overview_avoids_recording_catalog_and_remote_work_and_reads_current_state(setup, monkeypatch):
    service, session, _ = setup
    job = create(service, session['id'], 'fixture-rgb', 'Fresh dataset')
    for name in ('dispatch', 'sources'):
        monkeypatch.setattr(service, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError('Progress read did extra work')))
    monkeypatch.setattr(service.live, 'list', lambda **kw: (_ for _ in ()).throw(AssertionError('Read recording catalog')))
    monkeypatch.setattr(service.database, 'list_data_resources', lambda **kw: (_ for _ in ()).throw(AssertionError('Read resource catalog')))
    assert service.job_overview()['exports'][0]['state'] == 'QUEUED'
    service.update(job['id'], state='RUNNING', stage='CONVERTING', detail='Changed after first read')
    result = service.job_overview()
    assert set(result) == {'output_format', 'exports'}
    assert result['exports'][0]['state'] == 'RUNNING'
    assert result['exports'][0]['detail'] == 'Changed after first read'


def test_job_queries_do_not_grow_with_completed_jobs(setup, monkeypatch):
    service, session, _ = setup
    job = create(service, session['id'], 'fixture-rgb', 'Dataset')
    service.prepare(job['id'])
    complete = service.get(job['id'])
    assert complete['state'] == 'READY'
    # Distinct results ensure batching scales with versions as well as job rows.
    first = service.database.get_data_resource_version(complete['version_id'])
    versions = [first]
    for index in range(1, 20):
        version = service.database.create_data_resource_version(
            first['resource_id'], revision=f"{first['revision']}-{index}",
            **{key: first[key] for key in (
                'format', 'path', 'manifest_sha256', 'status', 'size_bytes', 'source_uri', 'metadata',
            )},
        )
        for location in first['locations']:
            service.database.record_data_location(version['id'], **{
                key: location[key] for key in ('kind', 'host', 'path', 'manifest_sha256', 'status')
            })
        versions.append(version)
    statements = []
    execute = PostgresConnection.execute
    def traced(connection, statement, parameters=None):
        statements.append(statement)
        return execute(connection, statement, parameters)
    monkeypatch.setattr(PostgresConnection, 'execute', traced)
    counts = []
    for count in (1, 20):
        with service.database.transaction() as connection:
            connection.execute('DELETE FROM policy_exports')
            for i in range(count):
                item = dict(copy.deepcopy(complete), id=f'job-{i}', version_id=versions[i]['id'])
                connection.execute('INSERT INTO policy_exports VALUES (?,?)', (item['id'], canonical_json(item)))
        statements.clear()
        result = service.job_overview()
        assert len(result['exports']) == count
        assert all(j['locations'] for j in result['exports'])
        reads = [statement for statement in statements
                 if statement.lstrip().upper().startswith(('SELECT', 'WITH'))]
        setup_statements = [statement for statement in statements if statement not in reads]
        # Snapshot configuration is not a data read; keep it explicit so this
        # test also rejects any accidental write or scheduling statement.
        assert setup_statements == ['SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY']
        counts.append(len(reads))
    # Job/version/link/usage reads plus one narrow current-adapter status query.
    assert counts == [9, 9], statements


def test_jobs_endpoint_precedes_dynamic_job_id_route(setup, monkeypatch):
    from skynet_app import policy_exports_api
    service, session, _ = setup
    create(service, session['id'], 'fixture-rgb', 'Dataset')
    monkeypatch.setattr(policy_exports_api, 'service', service)
    app = FastAPI()
    app.include_router(policy_exports_api.router)
    with TestClient(app) as client:
        result = client.get('/api/data/exports/jobs')
        assert result.status_code == 200
        assert len(result.json()['exports']) == 1
        assert 'policies' not in result.json()


def test_job_read_uses_current_policy_availability(setup):
    service, session, _ = setup
    adapter = service.test_adapters['test-recording-inputs']
    job = create(service, session['id'], 'fixture-rgb', 'Dataset')
    service.update(job['id'], state='READY', training_ready=True)
    first = service.job_overview()['exports'][0]
    assert first['training_ready'] is True
    assert first['training_setup']['adapter'] == 'test-recording-inputs'
    service.database.archive_adapter(adapter['id'])
    second = service.job_overview()['exports'][0]
    assert second['training_ready'] is False
    assert second['training_setup'] is None
    # Never rewrite immutable/execution records merely to serve a GET.
    assert service.get(job['id'])['training_ready'] is True


def test_empty_job_overview_does_not_load_policy_or_version_catalogs(setup, monkeypatch):
    service, _, _ = setup
    def unexpected(*args, **kwargs):
        raise AssertionError("Empty progress list loaded a catalog")
    monkeypatch.setattr(service.database, "list_adapter_registry", unexpected)
    statements = []
    execute = PostgresConnection.execute
    def traced(connection, statement, parameters=None):
        statements.append(statement)
        return execute(connection, statement, parameters)
    monkeypatch.setattr(PostgresConnection, "execute", traced)
    assert service.job_overview()["exports"] == []
    assert len(statements) == 1 and "FROM policy_exports" in statements[0]
