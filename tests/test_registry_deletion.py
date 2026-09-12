"""Exercise registry removal through the same dependency/token flow as run history."""
import json
import subprocess

import psycopg
from types import SimpleNamespace

import pytest

from skynet_app.database import Database, canonical_json, utc_now
from skynet_app.maintenance import Maintenance
from skynet_app.pipeline_api import PipelineService


class NoFiles:
    def resolve_gateway(self, gateway):
        raise AssertionError('Registry removal must not touch installed source or runtimes')


@pytest.fixture
def registry(tmp_path):
    root = Database(tmp_path / 'registry')
    db = root.for_workspace('legacy')
    return db, root, Maintenance(db, NoFiles())


def adapter(db):
    return db.upsert_seed_adapter(seed_key='example', name='Example', manifest={'slug': 'example'})


def suite(db, version='1'):
    return db.register_evaluation_suite(evaluator_adapter='isaac_lab', evaluator_version='1',
        name='cube', suite_version=version, config={'tasks': ['cube']})


def erase(service, kind, identifier):
    preview = service.preview(kind, identifier)
    assert not preview['blockers'], preview['blockers']
    assert preview['files'] == []
    service.delete(kind, identifier, preview['token'])
    return preview


def experiment(db, spec):
    return db.create_experiment(project_id=db.create_project('Project')['id'], name='Consumer', requested_spec=spec)


def pending(db, service, kind, identifier):
    plan = service.preview(kind, identifier)
    with db.transaction() as c:
        c.execute('INSERT INTO maintenance_operations(target_kind,target_id,owner_id,plan_json,created_at) VALUES (?,?,?,?,?)',
            (kind, identifier, 'legacy', canonical_json(plan), utc_now()))
    return plan


def test_delete_adapter_versions_validations_events_and_preserve_user_edits(registry):
    db, root, service = registry
    original = adapter(root)
    edited = db.edit_adapter(original['id'], manifest={'slug': 'example', 'description': 'mine'})
    assert edited['latest_version_number'] == 2
    assert db.get_adapter(original['id'], version_number=1)['selected_version']['manifest'] == {'slug': 'example'}
    assert adapter(root)['latest_version_number'] == 2, 'seeding must not overwrite user edits'
    db.record_adapter_validation(original['id'], status='PASSED')
    db.record_event(entity_type='adapter', entity_id=original['id'], event_type='EDITED', details={})
    plan = erase(service, 'adapter', original['id'])
    assert plan['counts'] == {'adapters': 2, 'adapter_validations': 1}
    with db.connection() as c:
        for table in ('adapters', 'adapter_validations', 'events', 'maintenance_operations'):
            assert c.execute(f'SELECT count(*) FROM {table}').fetchone()[0] == 0
        assert c.execute('SELECT seed_key FROM registry_exclusions WHERE kind=?', ('adapter',)).fetchone()[0] == 'example'
    assert service.delete('adapter', original['id'], '0'*64)['already_deleted']


@pytest.mark.parametrize('pinned', [True, False])
def test_adapter_consumers_block_removal(registry, pinned):
    db, root, service = registry
    a = adapter(root)
    source = {'adapter_version_id': a['selected_version']['id']} if pinned else {'adapter': 'example'}
    e = experiment(db, {'source': source})
    plan = service.preview('adapter', a['id'])
    assert any(b['kind'] == 'experiment' and b['id'] == e['id'] for b in plan['blockers'])
    with pytest.raises(ValueError, match='dependencies'):
        service.delete('adapter', a['id'], plan['token'])
    assert db.get_adapter(a['id']) is not None


def test_suite_removes_all_versions_and_blocks_selected_experiment(registry):
    db, root, service = registry
    first, latest = suite(root), suite(root, '2')
    e = experiment(db, {'evaluation': {'suites': ['cube']}})
    plan = service.preview('suite', latest['id'])
    assert any(b['id'] == e['id'] for b in plan['blockers'])
    erase(service, 'experiment', e['id'])
    assert erase(service, 'suite', latest['id'])['counts'] == {'evaluation_suites': 2}
    assert root.list_evaluation_suites(enabled_only=False) == []


@pytest.mark.parametrize('use_fk', [True, False])
def test_suite_pinned_and_legacy_evaluation_dependencies(registry, use_fk):
    db, root, service = registry
    s = suite(root)
    e = experiment(db, {})
    v = db.create_variant(e['latest_revision']['id'], name='v', parameters={}, resolved_spec={})
    run = db.create_run(v['id'], seed=1, adapter_name='test', adapter_version='1', run_directory='unset', status='SUCCEEDED')
    evaluation = db.create_evaluation(run['id'], evaluator_adapter='isaac_lab', evaluator_version='1',
        suite_name='cube', suite_version='1', tasks=['cube'], seeds=[1], episodes_per_task=1,
        **({'evaluation_suite_id': s['id']} if use_fk else {}))
    assert any(b['kind'] == 'evaluation' and b['id'] == evaluation['id'] for b in service.preview('suite', s['id'])['blockers'])


def test_stale_plan_and_pending_references_are_rejected(registry):
    db, root, service = registry
    a = adapter(root)
    plan = service.preview('adapter', a['id'])
    db.edit_adapter(a['id'], manifest={'slug': 'example', 'description': 'changed'})
    with pytest.raises(ValueError, match='changed'):
        service.delete('adapter', a['id'], plan['token'])
    pending(db, service, 'adapter', a['id'])
    with pytest.raises(ValueError, match='being deleted'):
        db.edit_adapter(a['id'], manifest={})
    with pytest.raises(ValueError, match='being deleted'):
        experiment(db, {'source': {'adapter': 'example'}})
    with pytest.raises(ValueError, match='being deleted'):
        db.clone_adapter(a['id'], name='Clone')
    s = suite(root)
    pending(db, service, 'suite', s['id'])
    with pytest.raises(ValueError, match='being deleted'):
        suite(root, '2')


def test_defaults_stay_removed_after_startup(registry, monkeypatch):
    import skynet_app.pipeline_api as api
    db, root, service = registry
    a, s = adapter(root), suite(root)
    erase(service, 'adapter', a['id'])
    erase(service, 'suite', s['id'])
    monkeypatch.setattr(api, 'builtin_adapter_manifests', lambda: [SimpleNamespace(slug='example')])
    monkeypatch.setattr(api, 'get_evaluation_catalog', lambda: [SimpleNamespace(evaluator='isaac_lab', suite='cube')])
    pipeline = PipelineService.__new__(PipelineService)
    pipeline.database = root
    pipeline._seed_registries()
    assert root.list_adapter_registry(include_archived=True) == []
    assert root.list_evaluation_suites(enabled_only=False) == []


def test_other_workspace_cannot_edit_delete_or_leak_dependency(registry):
    db, root, service = registry
    a, s = adapter(root), suite(root)
    with root.transaction() as c:
        c.execute("INSERT INTO workspaces(id,email) VALUES ('other','other@example.com')")
    other = root.for_workspace('other')
    assert other.get_adapter(a['id'])
    assert not other.owns('adapters', a['id'], writable=True)
    for kind, identifier in [('adapter', a['id']), ('suite', s['id'])]:
        with pytest.raises(KeyError):
            Maintenance(other, NoFiles()).preview(kind, identifier)
    e = experiment(other, {'source': {'adapter_version_id': a['selected_version']['id']}})
    plan = service.preview('adapter', a['id'])
    assert any(b['id'] is None and b['label'] == "Another workspace's item" for b in plan['blockers'])
    assert e['id'] not in json.dumps(plan['blockers'])


def test_suite_adapter_defaults_and_clone_dependencies(registry):
    db, root, service = registry
    s = suite(root)
    a = db.create_adapter(name='Consumer', manifest={'defaults': {'evaluation': {'suites': ['cube']}}})
    assert any(b['kind'] == 'adapter' and b['id'] == a['id'] for b in service.preview('suite', s['id'])['blockers'])
    clone = db.clone_adapter(a['id'], name='Derived')
    assert any(b['kind'] == 'adapter' and b['id'] == clone['id'] for b in service.preview('adapter', a['id'])['blockers'])
    erase(service, 'adapter', clone['id'])
    erase(service, 'adapter', a['id'])
    erase(service, 'suite', s['id'])


def test_suite_supported_capability_is_a_notice_not_a_dependency(registry):
    db, root, service = registry
    s = suite(root)
    manifest = {'slug': 'hpt', 'evaluations': [{
        'environment': 'isaac_lab', 'suites': ['cube'], 'command': {'argv': ['evaluate']},
    }]}
    a = db.create_adapter(name='HPT', manifest=manifest)
    spec = {'source': {'adapter_manifest': manifest}, 'evaluation': []}
    e = experiment(db, spec)
    v = db.create_variant(e['latest_revision']['id'], name='v', parameters={}, resolved_spec=spec)
    run = db.create_run(v['id'], seed=1, adapter_name='hpt', adapter_version='1',
                        run_directory='unset', status='SUCCEEDED')
    plan = service.preview('suite', s['id'])
    assert not plan['blockers']
    assert len(plan['notices']) == 1 and 'HPT' in plan['notices'][0]
    assert 'checkpoints will be retained' in plan['notices'][0]
    service.delete('suite', s['id'], plan['token'])
    assert db.get_run(run['id']) is not None
    assert db.get_adapter(a['id']) is not None


def test_suite_notice_uses_current_supported_environment_and_hides_private_names(registry):
    db, root, service = registry
    s = suite(root)
    manifest = {'evaluations': [{'environment': 'isaac_lab', 'suites': ['cube'], 'command': {'argv': ['evaluate']}}]}
    old = db.create_adapter(name='Old capability', manifest=manifest)
    db.edit_adapter(old['id'], manifest={'evaluations': []})
    db.create_adapter(name='Other environment', manifest={'evaluations': [{**manifest['evaluations'][0], 'environment': 'mujoco'}]})
    db.create_adapter(name='No evaluator', manifest={'evaluations': [{**manifest['evaluations'][0], 'command': None}]})
    assert service.preview('suite', s['id'])['notices'] == []
    with root.transaction() as c:
        c.execute("INSERT INTO workspaces(id,email) VALUES ('other','other@example.com')")
    root.for_workspace('other').create_adapter(name='Private adapter', manifest=manifest)
    notices = service.preview('suite', s['id'])['notices']
    assert len(notices) == 1 and "Another workspace's adapter" in notices[0]
    assert 'Private adapter' not in notices[0]


@pytest.mark.parametrize('error', [
    psycopg.OperationalError('private connection details'),
    subprocess.TimeoutExpired('private command', 120),
])
def test_connection_failures_have_retry_guidance(error):
    from fastapi import HTTPException
    from skynet_app.maintenance_api import invoke
    def broken():
        raise error
    with pytest.raises(HTTPException) as caught:
        invoke(broken)
    assert caught.value.status_code == 503
    assert 'Reconnect' in caught.value.detail
    assert 'private' not in caught.value.detail


def test_suite_updated_time_tracks_changes_not_repeated_registration(registry, monkeypatch):
    _, root, _ = registry
    monkeypatch.setattr("skynet_app.database.utc_now", lambda: "2026-09-01T00:00:00Z")
    original = suite(root)
    monkeypatch.setattr("skynet_app.database.utc_now", lambda: "2026-09-02T00:00:00Z")
    assert suite(root)["updated_at"] == original["updated_at"]
    disabled = root.register_evaluation_suite(evaluator_adapter="isaac_lab", evaluator_version="1",
        name="cube", suite_version="1", config={"tasks": ["cube"]}, enabled=False)
    assert disabled["updated_at"] == "2026-09-02T00:00:00Z"
    assert not disabled["enabled"]
